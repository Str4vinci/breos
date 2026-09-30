"""The perfect-foresight LP bound on one year's bill (plan step 7)."""

import dataclasses
import json
import math

import numpy as np
import pandas as pd
import pytest

from breos._dispatch import _dc_ac, _dc_for_ac, _dispatch_day_python
from breos.battery import BatteryConfig, _ResultBuffers, _step_energy_cap
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
    [
        {},
        LIMITED,
        {"battery_power_limit_c_rate": 0.3, "resolution": "15min"},
        {"enable_resistance_fade": True},
    ],
    ids=["plain", "limited", "c_rate_15min", "resistance_fade"],
)
def test_every_dispatch_production_delivers_is_a_point_of_the_program(overrides):
    # The bound claim itself: whatever instructions drive it, the dispatch's
    # ledger satisfies every constraint and bound of the program, and the
    # program prices it at the replayed cost. With resistance fade the
    # dispatch converts worse than the program assumes, so the point draws
    # less energy than the ledger did and costs no more.
    fade = overrides.get("enable_resistance_fade", False)
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
        replayed = replay.first_year_step_cost.sum()
        if fade:
            assert program.cost @ x <= replayed + 1e-12
        else:
            assert program.cost @ x == pytest.approx(replayed, rel=1e-9)
    assert frame["Battery_SOH"].min() < 100.0


def test_the_default_bound_is_strict_for_every_run_it_reports():
    case = prepare_replay(_config(4))
    dp = run_daily_target_oracle(case, target_levels=6, soc_states=9)
    greedy = replay_instructions(case, DispatchInstructions.noop(len(case.index)))
    result = run_lp_bound(case, covering={"daily_target_dp": dp.replay, "greedy": greedy})
    bound = result.bound.objective
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
    # The floor is the lowest health of the four runs, unrounded, so each is covered.
    lowest = min(run.min_soh_fraction for run in result.runs)
    assert result.problem.health() == (1.0, lowest) and lowest < 1.0
    assert result.bound_is_strict and {run.name for run in result.runs} == {
        "reference",
        "lp_replay",
        "daily_target_dp",
        "greedy",
    }
    # A lower floor can only lower the optimum.
    assert bound <= result.fixed_health_estimate.objective

    summary = report(result, case)
    assert summary["schema"] == LP_BOUND_SCHEMA and summary["bound_is_strict"] is True
    assert summary["lp"]["floor_mode"] == "auto" and summary["lp"]["floor_soh_fraction"] == lowest
    assert summary["reference"]["dispatch"] == "fixed_target"
    assert summary["reference"]["minus_bound"] == pytest.approx(costs["fixed_target"] - bound)
    assert summary["covering"]["greedy"]["covered_by_bound"] is True
    assert summary["lp_replay"]["minus_bound"] > 0.0
    json.dumps(summary, allow_nan=False)


def test_a_floor_at_the_opening_health_is_only_an_estimate():
    # The program's own schedule drains the battery to its floor, and health
    # has faded by then, so its ledger sits below a floor at the opening
    # health. Coverage is decided by that ledger, not assumed.
    case = prepare_replay(_config(2))
    with pytest.warns(UserWarning, match="does not provably cover the lp_replay run: .*its health fell"):
        result = run_lp_bound(case, floor_soh="opening")
    assert not result.bound_is_strict
    assert result.bound.objective == result.fixed_health_estimate.objective
    summary = report(result, case)
    assert summary["bound_is_strict"] is False and summary["lp_replay"]["covered_by_bound"] is False
    # The fixed-target run never reaches its floor here, so its ledger is a point of the program anyway.
    assert summary["reference"]["covered_by_bound"] is True


def test_a_replacement_in_the_first_year_is_never_covered():
    # At a 99.9 % end of life the pack is swapped within days: the swap adds
    # stored energy and health the program does not model.
    case = prepare_replay(_config(3, battery_eol_percentage=0.999))
    with pytest.warns(UserWarning, match="replaced the battery"):
        result = run_lp_bound(case, replay=False)
    run = result.runs[0]
    assert (run.name, run.covered, run.replacements > 0) == ("reference", False, True)
    assert report(result, case)["bound_is_strict"] is False


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
    assert summary["bound_is_strict"] is True and summary["lp"]["floor_mode"] == "auto"
    assert summary["lp"]["bound"] <= summary["lp_replay"]["first_year_cost"]
    assert schedule.read_text(encoding="utf-8").splitlines()[0] == f"# schema: {LP_SCHEDULE_SCHEMA}"
    frame = pd.read_csv(schedule, comment="#")
    assert len(frame) == 48
    assert frame["delivered_step_cost"].sum() == pytest.approx(summary["lp_replay"]["first_year_cost"], rel=1e-9)


# --- the dispatch step itself, fuzzed ------------------------------------------------


def _fuzzed_ledger(seed, efficiency, scale, hours, *, ac_w=3000.0, standby=5.0, low_pv=False, **limits):
    """``_dispatch_day`` on random inputs and instructions at fixed health, as a frame and its problem."""
    rng = np.random.default_rng(seed)
    n = 200
    config = BatteryConfig(
        nominal_energy_wh=5000.0,
        inverter_efficiency=efficiency,
        inverter_ac_capacity_w=ac_w,
        ac_output_scale=scale,
        standby_loss_wh=standby,
        max_charge_power_w=limits.get("charge_w"),
        max_discharge_power_w=limits.get("discharge_w"),
        power_limit_c_rate=limits.get("c_rate"),
    )
    pv = rng.choice([0.0, 1.0], n, p=[0.3, 0.7]) * rng.random(n) ** 2 * (400.0 if low_pv else 5000.0)
    load = rng.choice([0.0, 1.0], n, p=[0.1, 0.9]) * rng.random(n) ** 2 * 4000.0
    temperature = rng.uniform(-15.0, 35.0, n)
    charge = rng.random(n) < 0.35
    instructions = DispatchInstructions(
        discharge_allowed=~charge & (rng.random(n) < 0.7),
        reserve_fraction=np.where(rng.random(n) < 0.5, rng.random(n), 0.0),
        grid_target_fraction=np.where(charge, rng.random(n), np.nan),
        grid_charge_efficiency=0.95,
        grid_import_limit_w=5000.0,
    )
    buffers = _ResultBuffers(n)
    _dispatch_day_python(
        buffers,
        pv,
        load,
        temperature,
        0,
        n,
        battery_config=config,
        battery_soh_decimal=1.0,
        Battery_Energy_Wh=config.nominal_energy_wh * config.max_soc,
        Battery_PV_Origin_Energy_Wh=0.0,
        Battery_Grid_Origin_Energy_Wh=0.0,
        eff_charge=config.charge_efficiency,
        eff_discharge=config.discharge_efficiency,
        hours_per_step=hours,
        standby_loss_per_step_wh=standby * hours,
        cap_wh=_step_energy_cap(ac_w, hours),
        cap_charge_wh=_step_energy_cap(config.max_charge_power_w, hours),
        cap_discharge_wh=_step_energy_cap(config.max_discharge_power_w, hours),
        cap_stored_wh=_step_energy_cap(config.stored_power_limit_w, hours),
        instructions=instructions,
    )
    frame = pd.DataFrame({name: np.array(values) for name, values in buffers.columns.items()})
    problem = LpBoundProblem(
        pv_dc_w=pv,
        load_w=load,
        temperature_c=temperature,
        import_price_per_kwh=np.full(n, 0.2),
        export_price_per_kwh=np.full(n, 0.05),
        battery_config=config,
        hours_per_step=hours,
        grid_charge_efficiency=0.95,
        grid_import_limit_w=5000.0,
    )
    return frame, problem


FUZZ_LIMITS = {
    "none": {},
    "charge_discharge": {"charge_w": 1200.0, "discharge_w": 900.0},
    "c_rate": {"c_rate": 0.3},
    "small_inverter": {"ac_w": 1500.0},
    "low_pv": {"low_pv": True},
    "large_standby": {"standby": 50.0},
}


@pytest.mark.parametrize("hours", [1.0, 0.25])
@pytest.mark.parametrize(("efficiency", "scale"), [(0.9, 1.0), (0.96, 1.0), (0.96, 0.9), (0.995, 1.0), (1.0, 0.9)])
def test_fuzzed_dispatch_ledgers_are_points_of_the_program(efficiency, scale, hours):
    # The dispatch step on random inputs and instructions, across inverter
    # efficiencies (the AC <= DC piece is active near 1), output scales,
    # limits, clipping, cold windows and both resolutions. No App run.
    shared_rating = secant_tight = 0
    for seed, limits in enumerate(FUZZ_LIMITS.values()):
        frame, problem = _fuzzed_ledger(seed, efficiency, scale, hours, **limits)
        program = build_linear_program(problem)
        x = ledger_point(program, frame, hours)
        violation = program.violation(x)
        assert max(violation.values()) < 1e-6, (limits, violation)

        wh = {name: frame[name].to_numpy() * hours for name in frame}
        cap = problem.battery_config.inverter_ac_capacity_w * hours
        pv_ac = wh["PV_AC_To_Load"] + wh["PV_AC_Export"]
        # Grid charge taking the AC rating PV left, with PV present.
        shared_rating += int(
            (
                (wh["Grid_AC_To_Battery"] > 1e-6)
                & (wh["PV_DC"] > 1e-6)
                & (pv_ac + wh["Grid_AC_To_Battery"] >= cap - 1e-6)
            ).sum()
        )
        # The load secant is met with equality where the inverter serves the whole load, below the peak.
        load = wh["Houseload"]
        dc_per_ac, _per_export = load_secants(load, efficiency, cap, scale)
        dc = wh["PV_DC_To_Inverter"] + wh["Battery_Discharge_DC"] - wh["Battery_Discharge_Loss"]
        served = (np.abs(wh["PV_AC_To_Load"] + wh["Battery_AC_To_Load"] - load) < 1e-9) & (wh["PV_AC_Export"] == 0.0)
        slack = dc - dc_per_ac * load
        secant_tight += int((served & (load >= 1.0) & (np.abs(slack) <= 1e-6 * load + 1e-9)).sum())
    assert shared_rating > 0 and secant_tight > 0, (shared_rating, secant_tight)


def test_a_run_that_grid_charges_where_the_program_cannot_is_not_covered():
    # Without a [smart_charging] table the program has no grid-charge
    # converter, so it bounds self-consumption only. A controller that grid
    # charges at 0.95 is outside it, whatever its health: the ledger check
    # must refuse it rather than count it covered.
    case = prepare_replay({key: value for key, value in _config(3).items() if key != "smart_charging"})
    charging = replay_instructions(case, _random_instructions(case, 3))
    assert charging.artifacts.first_year_results_df["Grid_AC_To_Battery"].sum() > 0.0
    with pytest.warns(UserWarning, match="charged from the grid, which the program does not allow"):
        result = run_lp_bound(case, replay=False, covering={"controller": charging})
    runs = {run.name: run for run in result.runs}
    assert runs["reference"].covered and not runs["controller"].covered
    assert not result.bound_is_strict
    assert "not a point of the program" in runs["controller"].reason
    # With the converter the same run is covered.
    allowed = run_lp_bound(case, replay=False, covering={"controller": charging}, grid_charge_efficiency=0.95)
    assert allowed.bound_is_strict
    assert allowed.bound.objective <= charging.first_year_step_cost.sum()


def test_an_invalid_floor_health_is_a_usage_error(tmp_path, capsys):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(_config(1)), encoding="utf-8")
    for value in ("sometimes", "1.5"):
        with pytest.raises(SystemExit) as exit_info:
            main(["--config", str(config), "--floor-soh", value])
        assert exit_info.value.code == 2
        assert "--floor-soh" in capsys.readouterr().err
