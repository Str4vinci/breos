"""Smart charging, the daily-target planner and day controllers with an AC-coupled battery (ADR 0004 C7, C11, M7)."""

import math

import numpy as np
import pandas as pd
import pytest

from breos._controller import _InstructionBuffer
from breos._daily_targets import _DayEvaluator, daily_target_instructions, solve_daily_targets, target_grid
from breos.app_config import resolve_app_config
from breos.battery import BatteryConfig, simulate_energy_balance
from breos.dispatch_instructions import DispatchInstructions
from breos.inverter import InverterConversion
from breos.optimization_config import resolve_optimization_config
from breos.smart_charging import SmartChargingSpec
from tests.offline_app import TOU
from tests.test_daily_targets import OFF_PEAK, _pack, _priced_run, _problem
from tests.test_inverter_curve import CURVE

AC_PACK = {"coupling": "ac", "battery_inverter_efficiency": 0.9, "battery_inverter_ac_capacity_w": 3000.0}
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


def test_a_dc_table_reports_its_missing_efficiency_first_as_before():
    # The requirement stays in the DC table check, ahead of the overlap check.
    table = {**FIXED, "discharge_periods": ["peak", "off_peak"]}
    with pytest.raises(ValueError, match="needs smart_charging.grid_charge_efficiency"):
        resolve_app_config({**BASE, "smart_charging": table})
    with pytest.raises(ValueError, match="needs smart_charging.grid_charge_efficiency"):
        resolve_optimization_config({"location": {"latitude": 41.15, "longitude": -8.61}, "smart_charging": FIXED})


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


def _conversion(night_w):
    # A curve on a non-default row, a cable and, when asked, a night draw.
    return InverterConversion(
        np.asarray(CURVE.output_fraction), np.asarray(CURVE.efficiency), 1.4, 2e-6, night_consumption_w=night_w
    )


def _rich_problem(steps_per_day, night_w, pv_rating_w=2500.0):
    """An AC problem whose PV inverter has its own finite rating, curve, cable, output scale and night draw."""
    config = _pack(
        **{**AC_PACK, "battery_inverter_ac_capacity_w": 1800.0},
        inverter_ac_capacity_w=pv_rating_w,
        ac_output_scale=0.97,
        inverter_conversion=_conversion(night_w),
    )
    return _problem(n_days=1, config=config, efficiency=None, steps_per_day=steps_per_day)


def _run(problem, instructions, battery_config=None, **kwargs):
    """Production physics over ``problem`` at its own step, priced at its tariff."""
    freq = "h" if problem.hours_per_step == 1.0 else "15min"
    index = pd.date_range("2026-03-02", periods=len(problem.pv_dc_w), freq=freq, tz="UTC")
    results, *_ = simulate_energy_balance(
        pv_dc=pd.Series(problem.pv_dc_w, index=index),
        houseload=pd.DataFrame({"Load": problem.load_w}, index=index),
        battery_config=problem.battery_config if battery_config is None else battery_config,
        freq=freq,
        temperature_series=pd.Series(problem.temperature_c, index=index),
        dispatch_instructions=instructions,
        **kwargs,
    )
    cost = (
        (
            results["Import_From_Grid"].to_numpy() @ problem.import_price_per_kwh
            - results["PV_AC_Export"].to_numpy() @ problem.export_price_per_kwh
        )
        * problem.hours_per_step
        / 1000
    )
    return results, cost


@pytest.mark.parametrize("steps_per_day", [24, 96], ids=["h", "15min"])
def test_the_evaluators_forecast_chain_is_the_pv_only_conversion_without_night_draw(steps_per_day):
    # M7: the PV inverter converts the forecast as the PV-only system does, on
    # the scalar row at its full rating, and the planner leaves out its night draw.
    problem = _rich_problem(steps_per_day, night_w=7.5, pv_rating_w=450.0)
    chain = _DayEvaluator(problem, target_grid(2), "python").state["pv_chain"]
    config = problem.battery_config
    pv_only = BatteryConfig(
        nominal_energy_wh=0.0,
        inverter_efficiency=config.inverter_efficiency,
        inverter_ac_capacity_w=config.inverter_ac_capacity_w,
        ac_output_scale=config.ac_output_scale,
        inverter_conversion=_conversion(0.0),
    )
    frame, _ = _run(problem, None, battery_config=pv_only)
    hours = problem.hours_per_step
    for got, column in zip(chain[:3], ("PV_Production", "PV_Direct_Inverter_Loss", "PV_DC_Curtailed")):
        np.testing.assert_allclose(got, frame[column].to_numpy() * hours, rtol=1e-12, atol=1e-9, err_msg=column)
    assert chain[2].sum() > 0.0, "the PV inverter's rating must clip"
    assert not chain[3].any()


@pytest.mark.parametrize("steps_per_day", [24, 96], ids=["h", "15min"])
def test_an_ac_plan_is_the_dispatch_production_gives(steps_per_day):
    problem = _rich_problem(steps_per_day, night_w=0.0)
    plan = solve_daily_targets(problem, initial_energy_wh=1200.0, target_levels=[0.6])
    results, cost = _run(
        problem, daily_target_instructions(problem.instructions, problem.day_starts, [0.6]), initial_energy_wh=1200.0
    )
    assert results["Grid_AC_To_Battery"].sum() > 0.0
    assert results["PV_AC_To_Battery"].sum() > 0.0
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


def _decision(n, efficiency, target=0.5, limit_w=math.inf):
    return DispatchInstructions(
        discharge_allowed=np.zeros(n, dtype=bool),
        reserve_fraction=np.zeros(n),
        grid_target_fraction=np.full(n, target),
        grid_charge_efficiency=efficiency,
        grid_import_limit_w=limit_w,
    )


def test_the_controller_binds_an_ac_none_and_reports_what_it_executed():
    buffer = _InstructionBuffer(4, ac_coupled=True)
    assert buffer.executed() is None
    buffer.bind_scalars(_decision(2, None))
    buffer.write(0, _decision(2, None), 0, 2)
    executed = buffer.executed()
    assert executed is not None and executed.grid_charge_efficiency is None
    # None still compares equal with None on the next decision; a changed
    # import limit does not, and a float efficiency is refused outright.
    buffer.bind_scalars(_decision(2, None))
    with pytest.raises(ValueError, match="must keep grid_charge_efficiency and grid_import_limit_w"):
        buffer.bind_scalars(_decision(2, None, limit_w=3000.0))
    with pytest.raises(ValueError, match="take grid_charge_efficiency = None"):
        buffer.bind_scalars(_decision(2, 0.95))


def test_a_dc_controller_still_requires_equal_scalars():
    buffer = _InstructionBuffer(2)
    buffer.bind_scalars(_decision(2, 0.95))
    with pytest.raises(ValueError, match="must keep grid_charge_efficiency"):
        buffer.bind_scalars(_decision(2, 0.96))


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
