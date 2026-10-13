"""Smart charging, the daily-target planner and day controllers with an AC-coupled battery (ADR 0004 C7, C11, M7)."""

import math

import numpy as np
import pandas as pd
import pytest

from breos._controller import _InstructionBuffer
from breos._daily_targets import _DayEvaluator, daily_target_instructions, solve_daily_targets, target_grid
from breos.app_config import resolve_app_config
from breos.battery import _pv_inverter_conversion, _step_energy_cap
from breos.dispatch_instructions import DispatchInstructions
from breos.smart_charging import SmartChargingSpec
from tests.test_daily_targets import OFF_PEAK, _pack, _priced_run, _problem

AC_PACK = {"coupling": "ac", "battery_inverter_efficiency": 0.9, "battery_inverter_ac_capacity_w": 3000.0}
TOU = {
    "schedule": "pt_mainland_2026_daily_bi",
    "currency": "EUR",
    "import_prices": {"peak": 0.28, "off_peak": 0.11},
    "export_prices": {"all": 0.05},
    "fixed_charge_per_day": 0.25,
}
FIXED = {
    "mode": "fixed_target",
    "target_usable_fraction": 0.5,
    "charge_periods": ["off_peak"],
    "discharge_periods": ["peak"],
    "grid_import_limit_w": 5000,
}
PERSISTENCE = {
    "mode": "daily_persistence",
    "charge_periods": ["off_peak"],
    "discharge_periods": ["peak"],
    "target_levels": 4,
    "soc_states": 5,
}
APP_AC = {
    "battery_coupling": "ac",
    "battery_inverter_ac_rating_kw": 3.0,
    "battery_inverter_efficiency": 0.95,
    "costs": {"battery_inverter_cost_per_kw": 120.0},
}
BASE = {
    "location": "porto",
    "n_modules": 8,
    "annual_consumption_kwh": 4000,
    "battery_kwh": 5.0,
    "projection_years": 1,
    "tariff": TOU,
    "start_date": "2025-01-01",
    "period": {"start": "2025-03-01", "end": "2025-03-05"},
}


def _ac_problem(**kwargs):
    return _problem(config=_pack(**AC_PACK), efficiency=None, **kwargs)


# --- configuration -------------------------------------------------------------


@pytest.mark.parametrize("table", [FIXED, PERSISTENCE], ids=["fixed_target", "daily_persistence"])
def test_ac_refuses_a_grid_charge_efficiency_and_dc_still_needs_one(table):
    with pytest.raises(ValueError, match="grid_charge_efficiency' is for a DC-coupled battery"):
        resolve_app_config({**BASE, **APP_AC, "smart_charging": {**table, "grid_charge_efficiency": 0.95}})
    spec = resolve_app_config({**BASE, **APP_AC, "smart_charging": table}).smart_charging
    assert spec is not None
    assert (spec.grid_charge_efficiency, spec.battery_coupling) == (None, "ac")
    with pytest.raises(ValueError, match="needs smart_charging.grid_charge_efficiency"):
        resolve_app_config({**BASE, "smart_charging": table})


def test_discharge_only_is_the_same_for_both_couplings():
    table = {"mode": "discharge_only", "discharge_periods": ["peak"]}
    assert resolve_app_config({**BASE, **APP_AC, "smart_charging": table}).smart_charging is not None


def test_the_spec_checks_its_coupling():
    with pytest.raises(ValueError, match="grid_charge_efficiency' is for a DC-coupled battery"):
        SmartChargingSpec(**{**FIXED, "grid_charge_efficiency": 0.95}, battery_coupling="ac")
    with pytest.raises(ValueError, match="'battery_coupling' must be one of: dc, ac"):
        SmartChargingSpec(**FIXED, battery_coupling="hybrid")
    with pytest.raises(ValueError, match="needs smart_charging.grid_charge_efficiency"):
        SmartChargingSpec(**FIXED)


# --- the App -----------------------------------------------------------------


@pytest.mark.usefixtures("_patch_weather")
@pytest.mark.filterwarnings("ignore:'projection_years'")
@pytest.mark.parametrize("table", [FIXED, PERSISTENCE], ids=["fixed_target", "daily_persistence"])
def test_an_ac_coupled_app_run_grid_charges_through_its_battery_inverter(table):
    from breos import App

    app = App({**BASE, **APP_AC, "smart_charging": table})
    app.simulate()
    result = app.result()
    assert result["provenance"]["smart_charging"]["grid_charge_efficiency"] is None
    frame = app._artifacts.first_year_results_df
    assert frame["Grid_AC_To_Battery"].sum() > 0.0
    assert frame["Grid_Charge_Conversion_Loss"].sum() > 0.0
    assert frame["PV_DC_To_Battery"].sum() == 0.0
    # The controller keeps what it executed although the efficiency is None.
    executed = app._artifacts.projection.controller_instructions
    assert executed is not None and executed.grid_charge_efficiency is None
    assert len(executed) == len(frame)


# --- the daily-target planner (M7) ----------------------------------------------


def test_the_evaluator_converts_the_forecast_pv_once_on_the_scalar_row():
    problem = _ac_problem(n_days=2)
    evaluator = _DayEvaluator(problem, target_grid(2), "python")
    config = problem.battery_config
    hours = problem.hours_per_step
    expected = _pv_inverter_conversion(
        np.maximum(0.0, np.asarray(problem.pv_dc_w) * hours),
        battery_config=config,
        hours_per_step=hours,
        cap_wh=_step_energy_cap(config.inverter_ac_capacity_w, hours),
    )
    chain = evaluator.state["pv_chain"]
    for got, want in zip(chain[:3], expected[:3]):
        np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(chain[3], np.zeros(len(problem.pv_dc_w)))
    dc = _DayEvaluator(_problem(n_days=2), target_grid(2), "python")
    assert dc.state["pv_chain"] is None


def test_an_ac_plan_is_the_dispatch_production_gives():
    problem = _ac_problem(n_days=1)
    plan = solve_daily_targets(problem, initial_energy_wh=1200.0, target_levels=[0.6])
    results, cost = _priced_run(
        problem, daily_target_instructions(problem.instructions, problem.day_starts, [0.6]), initial_energy_wh=1200.0
    )
    assert results["Grid_AC_To_Battery"].sum() > 0.0
    assert plan.stage_cost == pytest.approx(cost, rel=1e-12)
    assert plan.end_energy_wh == results["Battery_Energy_End"].iloc[-1]


def test_an_ac_refill_is_priced_through_the_battery_inverter():
    # C11: eta_b x eff_charge in place of grid_charge_efficiency x eff_charge.
    problem = _ac_problem(n_steps=68)
    plan = solve_daily_targets(problem, initial_energy_wh=1200.0)
    full_wh = 5000.0 * 0.9
    assert plan.end_energy_wh < full_wh
    expected = (full_wh - plan.end_energy_wh) / 1000 * OFF_PEAK / (0.9 * 0.95)
    assert plan.terminal_cost == pytest.approx(expected, rel=1e-12)


# --- day controllers (C11) --------------------------------------------------------


def _decision(n, efficiency, target=0.5):
    return DispatchInstructions(
        discharge_allowed=np.zeros(n, dtype=bool),
        reserve_fraction=np.zeros(n),
        grid_target_fraction=np.full(n, target),
        grid_charge_efficiency=efficiency,
        grid_import_limit_w=math.inf,
    )


def test_the_controller_binds_an_ac_none_and_reports_what_it_executed():
    buffer = _InstructionBuffer(4, ac_coupled=True)
    assert buffer.executed() is None
    buffer.bind_scalars(_decision(2, None))
    buffer.write(0, _decision(2, None), 0, 2)
    executed = buffer.executed()
    assert executed is not None and executed.grid_charge_efficiency is None
    # None still compares equal with None on the next decision, and a float differs.
    buffer.bind_scalars(_decision(2, None))
    with pytest.raises(ValueError, match="take grid_charge_efficiency = None"):
        buffer.bind_scalars(_decision(2, 0.95))


def test_an_ac_controller_accepts_a_no_op_efficiency_without_grid_targets():
    buffer = _InstructionBuffer(2, ac_coupled=True)
    buffer.bind_scalars(_decision(2, 1.0, target=np.nan))
    dc = _InstructionBuffer(2)
    dc.bind_scalars(_decision(2, 0.95))
    assert dc.executed() is not None


def test_ac_run_frames_match_between_backends():
    pytest.importorskip("numba")
    problem = _ac_problem(n_days=2)
    instructions = daily_target_instructions(problem.instructions, problem.day_starts, [0.6, 0.4])
    python, _ = _priced_run(problem, instructions, initial_energy_wh=1200.0, execution_backend="python")
    numba, _ = _priced_run(problem, instructions, initial_energy_wh=1200.0, execution_backend="numba")
    pd.testing.assert_frame_equal(python, numba, check_exact=False, rtol=1e-12, atol=1e-9)
