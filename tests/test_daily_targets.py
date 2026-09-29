"""The daily charge-target planner behind daily-persistence smart charging (plan step 7)."""

import importlib.util
import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from breos._daily_targets import (
    DEFAULT_HORIZON_DAYS,
    DailyTargetProblem,
    daily_target_instructions,
    solve_daily_targets,
    target_grid,
)
from breos.app_config import resolve_tariff_spec
from breos.battery import BatteryConfig, simulate_energy_balance
from breos.dispatch_instructions import DispatchInstructions
from breos.smart_charging import SmartChargingSpec, resolve_instructions

_BACKENDS = [
    "python",
    pytest.param(
        "numba", marks=pytest.mark.skipif(not importlib.util.find_spec("numba"), reason="numba not installed")
    ),
]
OFF_PEAK, PEAK, EXPORT = 0.10, 0.30, 0.05


def _days(n_days, steps_per_day=24, pv_peaks=(600.0, 3500.0, 1500.0, 300.0)):
    """PV, load, off-peak mask and import price for ``n_days`` synthetic days."""
    hours_per_step = 24 / steps_per_day
    hour = (np.arange(n_days * steps_per_day) % steps_per_day) * hours_per_step
    day = np.arange(n_days * steps_per_day) // steps_per_day
    shape = np.where((hour >= 8) & (hour < 18), np.sin(np.pi * (hour - 8) / 10), 0.0)
    pv = shape * np.resize(np.asarray(pv_peaks), n_days)[day]
    load = np.where((hour >= 18) & (hour < 23), 1400.0, 350.0)
    cheap = (hour < 8) | (hour >= 22)
    return pv, load, cheap, np.where(cheap, OFF_PEAK, PEAK)


def _layout(cheap, *, efficiency=1.0, limit_w=math.inf):
    """Fixed-target instructions: grid charge off-peak, discharge on peak."""
    return DispatchInstructions(
        discharge_allowed=~cheap,
        reserve_fraction=np.zeros(len(cheap)),
        grid_target_fraction=np.where(cheap, 0.0, np.nan),
        grid_charge_efficiency=efficiency,
        grid_import_limit_w=limit_w,
    )


def _pack(**overrides):
    settings = {
        "nominal_energy_wh": 5000.0,
        "min_soc": 0.1,
        "max_soc": 0.9,
        "charge_efficiency": 0.95,
        "discharge_efficiency": 0.95,
        "standby_loss_wh": 2.0,
        "thermal_resistance_kw": 0.0,
        "inverter_efficiency": 0.96,
        "enable_replacement": False,
    }
    return BatteryConfig(**{**settings, **overrides})


def _problem(n_days=4, *, config=None, steps_per_day=24, temperature_c=25.0, efficiency=1.0, n_steps=None, **kwargs):
    pv, load, cheap, price = _days(n_days, steps_per_day)
    n = len(pv) if n_steps is None else n_steps
    starts = (*range(0, n, steps_per_day), n)
    return DailyTargetProblem(
        pv_dc_w=pv[:n],
        load_w=load[:n],
        temperature_c=np.broadcast_to(temperature_c, (len(pv),))[:n],
        import_price_per_kwh=price[:n],
        export_price_per_kwh=np.full(n, EXPORT),
        day_starts=starts,
        instructions=_layout(cheap[:n], efficiency=efficiency),
        battery_config=_pack() if config is None else config,
        hours_per_step=24 / steps_per_day,
        **kwargs,
    )


def _priced_run(problem, instructions, **kwargs):
    """Production physics for ``instructions`` over ``problem``, priced at its tariff."""
    index = pd.date_range("2026-03-02", periods=len(problem.pv_dc_w), freq="h", tz="UTC")
    results, *_ = simulate_energy_balance(
        pv_dc=pd.Series(problem.pv_dc_w, index=index),
        houseload=pd.DataFrame({"Load": problem.load_w}, index=index),
        battery_config=problem.battery_config,
        freq="h",
        temperature_series=pd.Series(problem.temperature_c, index=index),
        dispatch_instructions=instructions,
        **kwargs,
    )
    cost = (
        results["Import_From_Grid"].to_numpy() @ problem.import_price_per_kwh
        - results["PV_AC_Export"].to_numpy() @ problem.export_price_per_kwh
    ) / 1000
    return results, cost


# --- the legacy dynamic program ------------------------------------------


@pytest.mark.parametrize(
    ("n_steps", "targets", "objective", "terminal_cost", "end_energy_wh"),
    [
        (96, [0.9, 0.1, 0.0, 1.0], 5.393975446817737, 0.0, 4500.0),
        (68, [0.9, 0.1, 0.0], 2.5598835185905546, 0.3235974145891044, 1425.8245614035086),
    ],
)
@pytest.mark.parametrize("backend", _BACKENDS)
def test_the_plan_matches_the_legacy_dynamic_program(
    n_steps, targets, objective, terminal_cost, end_energy_wh, backend
):
    # Expected values are the legacy solve_daily_sc_schedule at 07dc0e40 on
    # the same inputs: 11 levels, 21 SOC states, a flat inverter and no power
    # limits, where the legacy kernel and the BREOS step agree.
    plan = solve_daily_targets(_problem(n_steps=n_steps), initial_energy_wh=1200.0, execution_backend=backend)
    np.testing.assert_array_equal(plan.targets, targets)
    assert plan.objective == pytest.approx(objective, rel=1e-12)
    assert plan.terminal_cost == pytest.approx(terminal_cost, rel=1e-12, abs=1e-15)
    assert plan.end_energy_wh == pytest.approx(end_energy_wh, rel=1e-12)
    assert plan.objective == plan.stage_cost + plan.terminal_cost


@pytest.mark.parametrize("steps_per_day", [24, 96])
def test_python_and_numba_plans_agree_bit_for_bit(steps_per_day):
    pytest.importorskip("numba")
    config = _pack(inverter_ac_capacity_w=2500.0, max_charge_power_w=1500.0, max_discharge_power_w=2000.0)
    temperature = 8.0 + 10.0 * np.sin(np.arange(3 * steps_per_day) / steps_per_day * 2 * np.pi)
    problem = _problem(3, config=config, steps_per_day=steps_per_day, temperature_c=temperature, efficiency=0.93)
    python = solve_daily_targets(problem, initial_energy_wh=900.0, execution_backend="python")
    numba = solve_daily_targets(problem, initial_energy_wh=900.0, execution_backend="numba")
    np.testing.assert_array_equal(python.targets, numba.targets)
    assert (python.objective, python.end_energy_wh) == (numba.objective, numba.end_energy_wh)


# --- degenerate problems ---------------------------------------------------


def test_one_level_is_the_forced_chain_production_physics_gives():
    # One day: the production run holds health fixed until the day closes,
    # as the planner does, so the two are the same dispatch.
    problem = _problem(1)
    plan = solve_daily_targets(problem, initial_energy_wh=1200.0, target_levels=[0.6])
    results, cost = _priced_run(
        problem, daily_target_instructions(problem.instructions, problem.day_starts, [0.6]), initial_energy_wh=1200.0
    )
    assert plan.targets.tolist() == [0.6]
    assert plan.stage_cost == pytest.approx(cost, rel=1e-12)
    assert plan.end_energy_wh == results["Battery_Energy_End"].iloc[-1]


def test_one_level_chains_the_days_from_each_end_energy():
    problem = _problem(3)
    plan = solve_daily_targets(problem, initial_energy_wh=1200.0, target_levels=1)
    energy, stage = 1200.0, 0.0
    for day in range(3):
        step = solve_daily_targets(problem.horizon(day, 1), initial_energy_wh=energy, target_levels=1)
        energy, stage = step.end_energy_wh, stage + step.stage_cost
    assert plan.targets.tolist() == [0.0, 0.0, 0.0]
    assert (plan.end_energy_wh, plan.stage_cost) == (energy, pytest.approx(stage, rel=1e-12))


def test_no_battery_has_no_choice_and_prices_the_pv_only_balance():
    problem = _problem(2, config=_pack(nominal_energy_wh=0.0))
    plan = solve_daily_targets(problem)
    _results, cost = _priced_run(problem, None)
    assert plan.targets.tolist() == [0.0, 0.0]
    assert plan.stage_cost == pytest.approx(cost, rel=1e-12)
    assert (plan.terminal_cost, plan.start_energy_wh, plan.end_energy_wh) == (0.0, 0.0, 0.0)


def test_ties_go_to_the_lowest_target():
    # A full battery that may never discharge: every target leaves the same
    # dispatch, so every level costs the same.
    problem = _problem(2, config=_pack(standby_loss_wh=0.0))
    problem = replace(problem, instructions=replace(problem.instructions, discharge_allowed=np.zeros(48, dtype=bool)))
    plan = solve_daily_targets(problem, target_levels=[0.3, 0.6, 1.0])
    assert plan.targets.tolist() == [0.3, 0.3]


# --- the end of the horizon --------------------------------------------------


def test_a_terminal_shortfall_is_bought_back_at_the_cheapest_price_through_both_efficiencies():
    problem = _problem(n_steps=68, efficiency=0.9)
    plan = solve_daily_targets(problem, initial_energy_wh=1200.0)
    full_wh = 5000.0 * 0.9
    assert plan.end_energy_wh < full_wh
    expected = (full_wh - plan.end_energy_wh) / 1000 * OFF_PEAK / (0.9 * 0.95)
    assert plan.terminal_cost == pytest.approx(expected, rel=1e-12)


def test_the_refill_cost_stops_the_last_day_draining_the_battery():
    problem = _problem(efficiency=0.9)
    refill = solve_daily_targets(problem, initial_energy_wh=1200.0)
    free = solve_daily_targets(problem, initial_energy_wh=1200.0, free_terminal=True)
    # Refilling ends full at no terminal cost; free to end empty, the plan
    # skips the last night's charge and leaves the battery at its floor.
    assert (refill.targets[-1], refill.end_energy_wh, refill.terminal_cost) == (1.0, 4500.0, 0.0)
    assert (free.targets[-1], free.end_energy_wh, free.terminal_cost) == (0.0, 500.0, 0.0)
    assert free.stage_cost < refill.stage_cost


# --- instructions, civil days and horizons ----------------------------------------


def test_daily_targets_replace_only_the_charge_steps_of_their_day():
    cheap = np.array([True, False, True, True, False, True])
    base = _layout(cheap, efficiency=0.9, limit_w=4000.0)
    placed = daily_target_instructions(base, (0, 2, 5, 6), [0.4, np.nan, 1.0])
    np.testing.assert_array_equal(placed.grid_target_fraction, [0.4, np.nan, np.nan, np.nan, np.nan, 1.0])
    np.testing.assert_array_equal(placed.discharge_allowed, base.discharge_allowed)
    assert (placed.grid_charge_efficiency, placed.grid_import_limit_w) == (0.9, 4000.0)
    with pytest.raises(ValueError, match="one target per day"):
        daily_target_instructions(base, (0, 2, 5, 6), [0.4, 0.5])
    with pytest.raises(ValueError, match="rise strictly from 0"):
        daily_target_instructions(base, (0, 2, 5), [0.4, 0.5])


def test_a_problem_on_a_tariff_plans_its_civil_days():
    # Lisbon summer on a UTC index: civil days start at 23:00 UTC.
    index = pd.date_range("2026-07-01", periods=72, freq="h", tz="UTC")
    tariff = resolve_tariff_spec(
        {
            "tariff": {
                "schedule": "pt_mainland_2026_daily_bi",
                "currency": "EUR",
                "import_prices": {"peak": PEAK, "off_peak": OFF_PEAK},
                "export_prices": {"all": EXPORT},
            },
            "costs": None,
            "resolution": "h",
        },
        "Europe/Lisbon",
    ).resolve(index, "Europe/Lisbon")
    spec = SmartChargingSpec(
        mode="fixed_target",
        target_usable_fraction=0.5,
        charge_periods=("off_peak",),
        discharge_periods=("peak",),
        grid_charge_efficiency=0.95,
    )
    pv, load, _cheap, _price = _days(3)
    problem = DailyTargetProblem.from_tariff(
        tariff,
        resolve_instructions(spec, tariff),
        _pack(),
        pv_dc_w=pv,
        load_w=load,
        temperature_c=np.full(72, 25.0),
        freq="h",
    )
    assert problem.day_starts == tariff.day_starts == (0, 23, 47, 71, 72)
    plan = solve_daily_targets(problem, target_levels=3, soc_states=5)
    assert len(plan.targets) == tariff.n_days == 4


def test_a_horizon_is_cut_at_the_last_day():
    problem = _problem(4, soh_fraction=0.9, eff_charge=0.9)
    window = problem.horizon(1)
    assert DEFAULT_HORIZON_DAYS == 2
    assert window.day_starts == (0, 24, 48)
    np.testing.assert_array_equal(window.load_w, problem.load_w[24:72])
    assert window.health() == problem.health() == (0.9, 0.9, 0.95)
    assert problem.horizon(3, 5).day_starts == (0, 24)
    with pytest.raises(ValueError, match="'first_day'"):
        problem.horizon(4)


def test_a_planner_holds_the_given_health_for_its_window():
    faded = solve_daily_targets(_problem(2, soh_fraction=0.8), initial_energy_wh=1000.0)
    fresh = solve_daily_targets(_problem(2), initial_energy_wh=1000.0)
    assert faded.soc_grid_wh[-1] == pytest.approx(0.8 * fresh.soc_grid_wh[-1])
    assert faded.objective != fresh.objective


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target_levels": 0}, "at least 1"),
        ({"target_levels": [0.5, 0.2]}, "strictly increasing"),
        ({"target_levels": [0.5, 1.2]}, "between 0 and 1"),
        ({"soc_states": 1}, "at least 2"),
        ({"initial_energy_wh": math.nan}, "must be finite"),
    ],
)
def test_bad_planner_settings_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        solve_daily_targets(_problem(1), **kwargs)


def test_target_grid_defaults_to_tenths():
    np.testing.assert_allclose(target_grid(11), np.arange(11) / 10)
    assert target_grid(1).tolist() == [0.0]
