"""The perfect-foresight LP bound on one year's bill (plan step 7)."""

import dataclasses
import json
import math

import numpy as np
import pandas as pd
import pytest

from breos._dispatch import _dc_ac, _dc_for_ac
from breos.dispatch_instructions import DispatchInstructions
from tools.oracles.daily_target_dp import run_daily_target_oracle
from tools.oracles.lp_bound import (
    LP_BOUND_SCHEMA,
    LP_SCHEDULE_SCHEMA,
    LpBoundProblem,
    build_linear_program,
    inverter_hull_cuts,
    ledger_point,
    load_secants,
    lp_instructions,
    main,
    report,
    run_lp_bound,
    solve_lp_bound,
)
from tools.oracles.replay import prepare_replay, replay_instructions

BASE = {"location": "porto", "n_modules": 8, "annual_consumption_kwh": 4000, "battery_kwh": 5.0, "projection_years": 1}
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
    "grid_charge_efficiency": 0.95,
    "grid_import_limit_w": 5000,
}
# Clipping at a 2 kW inverter, both battery power limits, and the raw
# weather temperature so the usable window moves step to step.
LIMITED = {
    "inverter_ac_rating_kw": 2.0,
    "battery_max_charge_power_w": 1200.0,
    "battery_max_discharge_power_w": 900.0,
    "battery_temperature": "weather",
    "battery_indoor_model": {"enabled": False},
}


def _config(days=4, **overrides):
    """A [period] window of ``days`` civil days from 1 March 2025 in Lisbon."""
    end = (pd.Timestamp("2025-03-01") + pd.Timedelta(days=days)).date().isoformat()
    return {
        **BASE,
        "tariff": TOU,
        "smart_charging": FIXED,
        "start_date": "2025-01-01",
        "period": {"start": "2025-03-01", "end": end},
        **overrides,
    }


pytestmark = [
    pytest.mark.usefixtures("_patch_weather"),
    pytest.mark.filterwarnings("ignore:'projection_years'"),
]


# --- the inverter relaxation ------------------------------------------------


@pytest.mark.parametrize(("efficiency", "scale"), [(0.96, 1.0), (0.96, 0.9), (0.995, 1.0)])
def test_the_hull_cuts_lie_on_or_above_the_inverter_curve(efficiency, scale):
    ac_rating = 3520.0
    intercepts, slopes = inverter_hull_cuts(efficiency, ac_rating, scale)
    dc = np.linspace(0.0, 1.5 * ac_rating / efficiency, 20001)
    delivered = np.array([_dc_ac(x, ac_rating, efficiency, scale, 2.0)[0] for x in dc])

    def bound(x):
        return np.minimum((intercepts[:, None] + slopes[:, None] * np.atleast_1d(x)).min(axis=0), scale * ac_rating)

    assert (bound(dc) >= delivered - 1e-9).all()
    # The bound is tight at the rated point, where the curve meets the rating.
    rated = ac_rating / efficiency
    assert bound(rated)[0] == pytest.approx(_dc_ac(rated, ac_rating, efficiency, scale, 2.0)[0], rel=1e-9)
    # And no looser than the peak ratio anywhere: about 1.003 of the nominal efficiency.
    assert (bound(dc[1:]) / dc[1:]).max() <= scale * min(1.0, efficiency * 1.0027)


def test_the_load_secants_hold_for_every_output_up_to_and_past_the_load():
    ac_rating, efficiency = 3520.0, 0.96
    loads = np.array([0.0, 50.0, 180.0, 400.0, 900.0, 1800.0, 2500.0, 4000.0])
    dc_per_ac, dc_per_export = load_secants(loads, efficiency, ac_rating)
    peak = dc_per_ac[-1]
    for load, per_ac, per_export in zip(loads, dc_per_ac, dc_per_export):
        # Any AC up to the load, and any export past it with the load served.
        for ac in np.linspace(0.0, min(load, ac_rating), 41)[1:]:
            assert _dc_for_ac(ac, ac_rating, efficiency, 1.0) >= per_ac * ac
        for export in np.linspace(0.0, ac_rating - load, 41) if load < ac_rating else []:
            assert _dc_for_ac(load + export, ac_rating, efficiency, 1.0) >= per_ac * load + per_export * export
    # A light load converts worse than the peak, so its secant is the tighter cut.
    assert dc_per_ac[1] > dc_per_ac[4] > peak
    assert (dc_per_export[[0, -2, -1]] == 0.0).all() and (dc_per_ac[[0, -2, -1]] == peak).all()


# --- the program is a relaxation of the dispatch ------------------------------------


def _random_instructions(case, seed):
    rng = np.random.default_rng(seed)
    n = len(case.index)
    charge = rng.random(n) < 0.3
    return DispatchInstructions(
        discharge_allowed=~charge & (rng.random(n) < 0.7),
        reserve_fraction=np.where(rng.random(n) < 0.5, rng.random(n), 0.0),
        grid_target_fraction=np.where(charge, rng.random(n), np.nan),
        grid_charge_efficiency=FIXED["grid_charge_efficiency"],
        grid_import_limit_w=FIXED["grid_import_limit_w"],
    )


@pytest.mark.parametrize(
    "overrides",
    [{}, LIMITED, {"battery_power_limit_c_rate": 0.3, "resolution": "15min"}],
    ids=["plain", "limited", "c_rate_15min"],
)
def test_every_dispatch_production_delivers_is_a_point_of_the_program(overrides):
    # The bound claim itself: whatever instructions drive it, the dispatch's
    # ledger satisfies every constraint and bound of the program, and the
    # program prices it at the replayed cost.
    case = prepare_replay(_config(3, **overrides))
    problem = LpBoundProblem.from_case(case)
    candidates = [case.configured_instructions(), DispatchInstructions.noop(len(case.index))]
    candidates += [_random_instructions(case, seed) for seed in (1, 2)]
    for instructions in candidates:
        replay = replay_instructions(case, instructions)
        frame = replay.artifacts.first_year_results_df
        # Fixed health is the one relaxation taken at a value: the floor must
        # be at or below the lowest health the replay reached.
        at_health = dataclasses.replace(problem, floor_soh_fraction=frame["Battery_SOH"].min() / 100.0)
        program = build_linear_program(at_health)
        x = ledger_point(program, frame, problem.hours_per_step)
        violation = program.violation(x)
        assert max(violation.values()) < 1e-6, violation
        assert program.cost @ x == pytest.approx(replay.first_year_step_cost.sum(), rel=1e-9)
    assert frame["Battery_SOH"].min() < 100.0


def test_the_bound_is_below_every_replayed_schedule():
    case = prepare_replay(_config(4))
    result = run_lp_bound(case)
    bound = result.bound.objective
    dp = run_daily_target_oracle(case, target_levels=6, soc_states=9)
    greedy = replay_instructions(case, DispatchInstructions.noop(len(case.index)))
    costs = {
        "fixed_target": result.reference.first_year_step_cost.sum(),
        "daily_target_dp": dp.replay.first_year_step_cost.sum(),
        "greedy": greedy.first_year_step_cost.sum(),
        "lp_schedule": result.lp_replay.first_year_step_cost.sum(),
    }
    assert all(bound < cost for cost in costs.values()), (bound, costs)
    # The LP's own schedule, replayed, comes closest of these.
    assert costs["lp_schedule"] == min(costs.values())
    assert bound == pytest.approx(result.bound.import_cost - result.bound.export_revenue, rel=1e-9)

    summary = report(result, case)
    assert summary["schema"] == LP_BOUND_SCHEMA
    assert summary["reference"]["dispatch"] == "fixed_target"
    assert summary["reference"]["minus_bound"] == pytest.approx(costs["fixed_target"] - bound)
    assert summary["lp_replay"]["minus_bound"] > 0.0
    json.dumps(summary, allow_nan=False)


def test_a_lower_floor_health_can_only_lower_the_bound():
    case = prepare_replay(_config(2))
    fixed = solve_lp_bound(LpBoundProblem.from_case(case)).objective
    faded = solve_lp_bound(LpBoundProblem.from_case(case, floor_soh_fraction=0.9)).objective
    assert faded <= fixed + 1e-12


def test_without_a_grid_charge_converter_the_program_bounds_self_consumption():
    config = {key: value for key, value in _config(2).items() if key != "smart_charging"}
    case = prepare_replay(config)
    result = run_lp_bound(case)
    assert result.problem.grid_charge_efficiency is None
    assert result.bound.schedule["grid_ac_to_battery_wh"].max() == 0.0
    assert report(result, case)["reference"]["dispatch"] == "greedy"
    assert result.bound.objective <= result.reference.first_year_step_cost.sum()
    # Allowing the converter can only help.
    allowed = solve_lp_bound(LpBoundProblem.from_case(case, grid_charge_efficiency=0.95))
    assert allowed.objective <= result.bound.objective + 1e-12


def test_the_schedule_becomes_instructions_that_charge_or_discharge_as_planned():
    case = prepare_replay(_config(2))
    problem = LpBoundProblem.from_case(case)
    bound = solve_lp_bound(problem)
    instructions = lp_instructions(bound, problem)
    charging = bound.schedule["grid_ac_to_battery_wh"] > 1e-6
    emin, emax = problem.window()
    energy = bound.schedule["battery_energy_end_wh"]
    assert charging.any()
    np.testing.assert_allclose(
        instructions.grid_target_fraction[charging], ((energy - emin) / (emax - emin))[charging], atol=1e-9
    )
    assert not instructions.discharge_allowed[charging].any()
    discharging = instructions.discharge_allowed
    assert (bound.schedule["battery_discharge_stored_wh"][discharging] > 1e-6).all()
    assert (instructions.grid_charge_efficiency, instructions.grid_import_limit_w) == (0.95, 5000.0)


def test_bad_problems_are_rejected():
    case = prepare_replay(_config(1))
    with pytest.raises(ValueError, match="floor_soh_fraction <= soh_fraction"):
        LpBoundProblem.from_case(case, floor_soh_fraction=1.1)
    with pytest.raises(ValueError, match="'grid_charge_efficiency'"):
        LpBoundProblem.from_case(case, grid_charge_efficiency=0.0)
    with pytest.raises(ValueError, match="'grid_import_limit_w'"):
        LpBoundProblem.from_case(case, grid_import_limit_w=math.nan)
    with pytest.raises(ValueError, match="'tangents'"):
        solve_lp_bound(LpBoundProblem.from_case(case, tangents=0))


def test_the_command_line_writes_a_tagged_summary_and_schedule(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(_config(2)), encoding="utf-8")
    output, schedule = tmp_path / "bound.json", tmp_path / "schedule.csv"
    assert main(["--config", str(config), "--output", str(output), "--csv", str(schedule)]) == 0

    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["schema"] == LP_BOUND_SCHEMA and summary["n_steps"] == 48
    assert summary["lp"]["bound"] <= summary["lp_replay"]["first_year_cost"]
    assert schedule.read_text(encoding="utf-8").splitlines()[0] == f"# schema: {LP_SCHEDULE_SCHEMA}"
    frame = pd.read_csv(schedule, comment="#")
    assert len(frame) == 48
    assert frame["delivered_step_cost"].sum() == pytest.approx(summary["lp_replay"]["first_year_cost"], rel=1e-9)
