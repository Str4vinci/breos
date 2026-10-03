"""The daily-target oracle: perfect-information daily targets on a grid.

For an App configuration with fixed-target smart charging, the oracle plans
a whole project year at once with the private daily-target dynamic program
(:mod:`breos._daily_targets`). It sees the true PV, load and battery
temperature of every day. The policy class is the configured instruction
layout, with one target per day in place of the configured one: every
charge step of a day takes that day's target, from a grid of usable
fractions. The discharge gate, the reserve, the grid-charge efficiency and
the site limit stay as configured. Under ``overlap_policy = "hold_target"``
a period that both charges and discharges holds each day's target as its
discharge floor (ADR 0002 A18), so Always dispatch with off-peak charging
plans and replays as one policy. The planned schedule is then replayed
through the production App run (:mod:`tools.oracles.replay`) and priced,
next to App's own fixed-target run of the same configuration.

This is the BREOS counterpart of the offline oracle of the legacy
``tools/compute_a2_daily_sc_oracle.py`` (``--forecast-mode perfect``, at
``07dc0e40``). It is a validation tool, not public API. It is an optimum
over one policy class under the planner's model, not a bound: the linear
program of :mod:`tools.oracles.lp_bound` bounds every policy whose health
stays at or above its floor health.

The planner holds state of health and both efficiencies at the year's
opening values, and its value function lives on a grid of stored energy.
The replay is the accounting truth: it runs the instructions under the
health production physics gives. ``planned`` is what the same instructions
deliver at the planner's fixed health, one dispatch call over the window,
so the planned-against-delivered comparison isolates what health moving
during the year changed. With perfect foresight and one plan for the whole
window, the replayed cost and the plan's stage cost differ only by that.

Days are the tariff's civil days (ADR 0002 A1), or with
``decision_boundary = "charge_window_start"`` (``--decision-boundary``) the
charge windows of the layout (A19): each window's target is chosen at its
start and holds until the next window starts, as ``daily_persistence``
decides under the same boundary. The steps before the first window form a
planning day of their own, and under ``yearly`` planning a window that
crosses the year seam takes a new target at the year start. By default, energy that
ends the year below full is bought back at the cheapest price a charge
step may pay, so the plan does not drain the battery on the last day
(``free_terminal`` drops the refill).

``wear_cost_per_kwh`` passes the planner's battery-wear weight (ADR 0002
A17) to every solve: a price per kWh of DC energy the battery discharges,
added to the cost the plan minimises. It is a planning weight only. The
plan reports it as ``wear_cost``, apart from ``stage_cost``; the planned and
replayed costs are import cost less export revenue alone. At 0, the
default, the plan is the one the planner makes without it.

Two planning modes:

- ``first_year`` (the default) plans the first project year from a new
  battery. Every project year replays those instructions (A2); the costs
  compared are the first year's.
- ``yearly`` plans each project year as it begins (A16). Year ``y`` is
  planned on that year's PV, with its degradation applied, its load and
  temperature, from the battery state the production replay of years
  ``1..y-1`` reached: stored energy, state of health and efficiencies. The
  replay then runs year ``y`` and carries its state into year ``y+1``.
  Energy that ends a year below what it began with is bought back, capped at
  the max-SOC energy at the year's last temperature; in year one that cap
  is the ``first_year`` target, so the two modes plan year one alike.

Neither mode is a bound on lifetime NPV. Each year is optimal only within
the policy class, on the planner's grid and at that year's opening health;
the year-by-year choice ignores what one year's cycling costs the later
years in health.

Legacy behaviour not ported: the causal persistence-forecast controller
(BREOS has it as the ``daily_persistence`` smart-charging mode), the matched
sweep over every fixed target (``--compare-fixed``) and the residual-value
sensitivity.

Usage:
    python tools/oracles/daily_target_dp.py --config my.toml --output dp.json --csv days.csv
    python tools/oracles/daily_target_dp.py --config my.toml --planning yearly --soc-states 41 --target-levels 21
    python tools/oracles/daily_target_dp.py --config my.toml --wear-cost-per-kwh 0.05
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from breos._daily_targets import (
    DEFAULT_SOC_STATES,
    DEFAULT_TARGET_LEVELS,
    DailyTargetPlan,
    DailyTargetProblem,
    check_wear_cost,
    daily_target_instructions,
    full_terminal_energy_wh,
    solve_daily_targets,
    target_grid,
)
from breos.app_inputs import reuse_prepared_inputs
from breos.battery import _resolve_dispatch_day, _ResultBuffers, _step_energy_cap, align_simulation_inputs
from breos.dispatch_instructions import DispatchInstructions
from breos.projection import YearStart
from breos.smart_charging import DECISION_BOUNDARIES
from breos.tariffs import result_currency
from tools.oracles._output import load_config, write_csv, write_json
from tools.oracles.replay import (
    PLANNED_FLOWS,
    ReplayCase,
    ReplayResult,
    Tolerance,
    compare_with_plan,
    plan_comparison,
    prepare_replay,
    replay_instructions,
    replay_summary,
)

DP_ORACLE_SCHEMA = "breos_daily_target_dp_oracle_v1"
DP_DAYS_SCHEMA = "breos_daily_target_dp_days_v1"
DP_YEAR_DAYS_SCHEMA = "breos_daily_target_dp_year_days_v1"
PLANNING_MODES = ("first_year", "yearly")
# Planned and delivered energy must agree to within this before a step counts as a mismatch.
DEFAULT_REPLAY_TOLERANCE = Tolerance(atol_wh=1.0)


@dataclass(frozen=True)
class YearPlan:
    """One project year's plan under yearly replanning, and the inputs it was planned on.

    ``year`` counts from 1. ``start_energy_wh`` and the problem's health are
    the state the replay opened the year in; ``terminal_energy_wh`` is the
    stored energy below which the year's end is bought back.
    """

    year: int
    problem: DailyTargetProblem
    plan: DailyTargetPlan
    instructions: DispatchInstructions
    start_energy_wh: float
    terminal_energy_wh: float
    resistance_growth: float
    pv_degradation_factor: float
    solve_seconds: float

    def inputs_sha256(self) -> str:
        """A sha256 of the PV, load and temperature the year was planned on."""
        digest = hashlib.sha256(b"breos-daily-target-year-inputs-v1")
        for series in (self.problem.pv_dc_w, self.problem.load_w, self.problem.temperature_c):
            digest.update(np.ascontiguousarray(series, dtype="<f8").tobytes())
        return digest.hexdigest()


class YearlyDailyTargetPlanner:
    """A year planner (ADR 0002 A16) that solves the daily-target program as each project year begins.

    The production replay calls it with the state it has reached, so year
    ``y`` is planned at the battery state years ``1..y-1`` left. ``plans``
    holds each year's plan in order.
    """

    def __init__(
        self,
        case: ReplayCase,
        *,
        target_levels: int | Sequence[float],
        soc_states: int,
        free_terminal: bool,
        execution_backend: str,
        decision_boundary: str = "civil_day",
        wear_cost_per_kwh: float = 0.0,
    ) -> None:
        self.case = case
        self.decision_boundary = decision_boundary
        self.layout = case.configured_instructions()
        self.target_levels = target_levels
        self.soc_states = soc_states
        self.free_terminal = free_terminal
        self.execution_backend = execution_backend
        self.wear_cost_per_kwh = wear_cost_per_kwh
        self.plans: list[YearPlan] = []

    def __call__(self, start: YearStart) -> DispatchInstructions:
        case, year, state = self.case, start.year, start.battery_state
        freq = case.resolved.cfg["resolution"]
        if year.pv_dc is None or year.houseload is None:
            raise ValueError("yearly replanning needs per-step project years")
        aligned = align_simulation_inputs(year.pv_dc, year.houseload, year.temperature_series, freq=freq)
        if not aligned.index.equals(case.index):
            raise ValueError("The aligned simulation calendar differs from the tariff's calendar")
        problem = DailyTargetProblem.from_tariff(
            case.tariff,
            self.layout,
            start.battery_config,
            pv_dc_w=aligned.pv_dc_w,
            load_w=aligned.load_w,
            temperature_c=aligned.temperature_c,
            freq=freq,
            decision_boundary=self.decision_boundary,
            soh_fraction=state.soh_fraction,
            eff_charge=state.charge_efficiency,
            eff_discharge=state.discharge_efficiency,
        )
        # Buy back what the year ends without, up to what the battery can hold then.
        terminal = min(state.energy_wh, full_terminal_energy_wh(problem))
        started = time.perf_counter()
        plan = solve_daily_targets(
            problem,
            initial_energy_wh=state.energy_wh,
            target_levels=self.target_levels,
            soc_states=self.soc_states,
            terminal_energy_wh=terminal,
            free_terminal=self.free_terminal,
            execution_backend=self.execution_backend,
            wear_cost_per_kwh=self.wear_cost_per_kwh,
        )
        elapsed = time.perf_counter() - started
        instructions = daily_target_instructions(problem.instructions, problem.day_starts, plan.targets)
        self.plans.append(
            YearPlan(
                year=start.year_idx + 1,
                problem=problem,
                plan=plan,
                instructions=instructions,
                start_energy_wh=state.energy_wh,
                terminal_energy_wh=terminal,
                resistance_growth=state.resistance_growth,
                pv_degradation_factor=year.pv_degradation_factor,
                solve_seconds=elapsed,
            )
        )
        return instructions


@dataclass(frozen=True)
class DailyTargetOracleResult:
    """The oracle's plan, its replay, and App's own fixed-target run.

    ``problem``, ``plan`` and ``instructions`` are the first project year's.
    ``planned`` holds the flows the plan's instructions deliver at the
    planner's fixed health, and ``planned_step_cost`` their per-step import
    cost less export revenue. ``replay`` compares them with production.
    ``fixed_target`` replays the configured instructions, which App
    dispatches with. Under ``yearly`` planning, ``year_plans`` holds every
    year's plan and ``solve_seconds`` their sum. ``wear_cost_per_kwh`` is
    the wear weight every solve was given.
    """

    problem: DailyTargetProblem
    plan: DailyTargetPlan
    instructions: DispatchInstructions
    planned: dict[str, np.ndarray]
    planned_step_cost: np.ndarray
    replay: ReplayResult
    fixed_target: ReplayResult
    solve_seconds: float
    target_levels: np.ndarray
    soc_states: int
    free_terminal: bool
    planning: str = "first_year"
    wear_cost_per_kwh: float = 0.0
    year_plans: tuple[YearPlan, ...] = ()
    schema: str = DP_ORACLE_SCHEMA
    decision_boundary: str = "civil_day"


def daily_target_problem(case: ReplayCase, decision_boundary: str = "civil_day") -> DailyTargetProblem:
    """``case``'s first project year as a daily-target problem on its fixed-target layout.

    ``decision_boundary`` partitions it into civil days or charge windows.

    Raises:
        ValueError: If the configuration has no fixed-target ``[smart_charging]`` table.
    """
    spec = case.resolved.smart_charging
    if spec is None or spec.mode != "fixed_target":
        raise ValueError("The daily-target oracle needs a [smart_charging] table with mode = 'fixed_target'")
    aligned = case.aligned_inputs()
    return DailyTargetProblem.from_tariff(
        case.tariff,
        case.configured_instructions(),
        case.battery_config(),
        pv_dc_w=aligned.pv_dc_w,
        load_w=aligned.load_w,
        temperature_c=aligned.temperature_c,
        freq=case.resolved.cfg["resolution"],
        decision_boundary=decision_boundary,
    )


def fixed_health_state(problem: DailyTargetProblem) -> dict[str, Any]:
    """The day state the planner dispatches every transition with, less the stored energy and instructions.

    The same keys and values as the planner's own (``_DayEvaluator.state``);
    ``tests/test_daily_target_oracle.py`` pins them, so a field the planner
    gains cannot leave the planned flows on another model.
    """
    soh, eff_charge, eff_discharge = problem.health()
    config = problem.battery_config
    hours = problem.hours_per_step
    return {
        "battery_config": config,
        "battery_soh_decimal": soh,
        "Battery_PV_Origin_Energy_Wh": 0.0,
        "Battery_Grid_Origin_Energy_Wh": 0.0,
        "eff_charge": eff_charge,
        "eff_discharge": eff_discharge,
        "hours_per_step": hours,
        "standby_loss_per_step_wh": config.standby_loss_wh * hours,
        "cap_wh": _step_energy_cap(config.inverter_ac_capacity_w, hours),
        "cap_charge_wh": _step_energy_cap(config.max_charge_power_w, hours),
        "cap_discharge_wh": _step_energy_cap(config.max_discharge_power_w, hours),
        "cap_stored_wh": _step_energy_cap(config.stored_power_limit_w, hours),
    }


def fixed_health_flows(
    problem: DailyTargetProblem, instructions: DispatchInstructions, *, execution_backend: str = "python"
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """What ``instructions`` deliver over ``problem`` at its fixed health, and each step's cost.

    One call of the production dispatch step from a full battery, as the
    planner's forward pass chains its days. Returns the flows named in
    :data:`tools.oracles.replay.PLANNED_FLOWS` and the per-step import cost
    less export revenue.
    """
    config = problem.battery_config
    buffers = _ResultBuffers(len(instructions))
    # Writable copies of the series, as the planner dispatches on.
    _resolve_dispatch_day(execution_backend)(
        buffers,
        np.array(problem.pv_dc_w),
        np.array(problem.load_w),
        np.array(problem.temperature_c),
        0,
        len(instructions),
        Battery_Energy_Wh=config.nominal_energy_wh * problem.health()[0] * config.max_soc,
        instructions=instructions,
        **fixed_health_state(problem),
    )
    columns = buffers.columns
    planned = {
        name: np.array(columns[column], dtype=np.float64) for name, column in PLANNED_FLOWS.items() if column in columns
    }
    planned["battery_energy_wh"] = np.array(columns["Battery_Energy"], dtype=np.float64)
    step_cost = (
        np.asarray(columns["Import_From_Grid"]) * problem.import_price_per_kwh
        - np.asarray(columns["PV_AC_Export"]) * problem.export_price_per_kwh
    ) * (problem.hours_per_step / 1000.0)
    return planned, step_cost


def run_daily_target_oracle(
    case: ReplayCase,
    *,
    target_levels: int | Sequence[float] = DEFAULT_TARGET_LEVELS,
    soc_states: int = DEFAULT_SOC_STATES,
    free_terminal: bool = False,
    tolerance: Tolerance = DEFAULT_REPLAY_TOLERANCE,
    execution_backend: str | None = None,
    planning: str = "first_year",
    decision_boundary: str = "civil_day",
    wear_cost_per_kwh: float = 0.0,
) -> DailyTargetOracleResult:
    """Plan ``case`` one target per day, replay the plan and App's fixed-target run.

    ``planning`` is ``"first_year"`` or ``"yearly"`` (see the module
    docstring). ``soc_states`` and ``target_levels`` set the program's grid
    of stored energy and of targets. ``wear_cost_per_kwh`` is the planner's
    wear weight (ADR 0002 A17), given to every solve; it must be finite and
    at least 0. ``decision_boundary`` plans one target per civil day (the
    default) or per charge window (ADR 0002 A19). The planner and the
    replays run on the configuration's execution backend unless
    ``execution_backend`` names one.
    """
    if planning not in PLANNING_MODES:
        raise ValueError(f"'planning' must be one of {', '.join(PLANNING_MODES)}")
    if decision_boundary not in DECISION_BOUNDARIES:
        raise ValueError(f"'decision_boundary' must be one of {', '.join(DECISION_BOUNDARIES)}")
    wear = check_wear_cost(wear_cost_per_kwh)
    backend = execution_backend or case.resolved.cfg.get("execution_backend", "python")
    if planning == "yearly":
        return _run_yearly(
            case,
            target_levels=target_levels,
            soc_states=soc_states,
            free_terminal=free_terminal,
            tolerance=tolerance,
            backend=backend,
            decision_boundary=decision_boundary,
            wear_cost_per_kwh=wear,
        )
    problem = daily_target_problem(case, decision_boundary)
    started = time.perf_counter()
    plan = solve_daily_targets(
        problem,
        target_levels=target_levels,
        soc_states=soc_states,
        free_terminal=free_terminal,
        execution_backend=backend,
        wear_cost_per_kwh=wear,
    )
    elapsed = time.perf_counter() - started
    instructions = daily_target_instructions(problem.instructions, problem.day_starts, plan.targets)
    planned, planned_step_cost = fixed_health_flows(problem, instructions, execution_backend=backend)
    replay = replay_instructions(case, instructions, planned=planned, tolerance=tolerance, execution_backend=backend)
    fixed_target = replay_instructions(case, problem.instructions, execution_backend=backend)
    return DailyTargetOracleResult(
        problem=problem,
        plan=plan,
        instructions=instructions,
        planned=planned,
        planned_step_cost=planned_step_cost,
        replay=replay,
        fixed_target=fixed_target,
        solve_seconds=elapsed,
        target_levels=target_grid(target_levels),
        soc_states=int(soc_states),
        free_terminal=free_terminal,
        decision_boundary=decision_boundary,
        wear_cost_per_kwh=wear,
    )


def _run_yearly(
    case: ReplayCase,
    *,
    target_levels: int | Sequence[float],
    soc_states: int,
    free_terminal: bool,
    tolerance: Tolerance,
    backend: str,
    wear_cost_per_kwh: float,
    decision_boundary: str = "civil_day",
) -> DailyTargetOracleResult:
    daily_target_problem(case)  # the same table check as the first-year mode
    planner = YearlyDailyTargetPlanner(
        case,
        target_levels=target_levels,
        soc_states=soc_states,
        free_terminal=free_terminal,
        execution_backend=backend,
        decision_boundary=decision_boundary,
        wear_cost_per_kwh=wear_cost_per_kwh,
    )
    replay = replay_instructions(case, planner, execution_backend=backend)
    first = planner.plans[0]
    planned, planned_step_cost = fixed_health_flows(first.problem, first.instructions, execution_backend=backend)
    replay = compare_with_plan(case, replay, planned, tolerance)
    fixed_target = replay_instructions(case, first.problem.instructions, execution_backend=backend)
    return DailyTargetOracleResult(
        problem=first.problem,
        plan=first.plan,
        instructions=first.instructions,
        planned=planned,
        planned_step_cost=planned_step_cost,
        replay=replay,
        fixed_target=fixed_target,
        solve_seconds=sum(plan.solve_seconds for plan in planner.plans),
        target_levels=target_grid(target_levels),
        soc_states=int(soc_states),
        free_terminal=free_terminal,
        planning="yearly",
        wear_cost_per_kwh=wear_cost_per_kwh,
        year_plans=tuple(planner.plans),
        decision_boundary=decision_boundary,
    )


def _year_costs(replay: ReplayResult) -> list[float]:
    """Each project year's import cost less export revenue at year-1 prices, from the replay's year rows."""
    yearly = replay.artifacts.yearly_df
    return [float(value) for value in (yearly["Import_Cost"] - yearly["Export_Revenue"])]


def _lifetime(replay: ReplayResult) -> dict[str, Any]:
    """The replay's per-year costs, replacements and health, and its NPV of savings."""
    artifacts = replay.artifacts
    yearly = artifacts.yearly_df
    projection = artifacts.cost_projection
    return {
        "year_costs": _year_costs(replay),
        "replacements": [int(value) for value in yearly["Replacements"]],
        "end_soh_fraction": [float(value) / 100.0 for value in yearly["Battery_SOH_%"]],
        "npv_savings": float(projection.attrs["final_npv_savings"]) if projection is not None else None,
    }


def _year_record(year_plan: YearPlan, replayed_cost: float, levels: np.ndarray) -> dict[str, Any]:
    problem, plan = year_plan.problem, year_plan.plan
    soh, eff_charge, eff_discharge = problem.health()
    hours = problem.hours_per_step
    return {
        "year": year_plan.year,
        "inputs": {
            "pv_degradation_factor": year_plan.pv_degradation_factor,
            "pv_dc_kwh": float(problem.pv_dc_w.sum() * hours / 1000.0),
            "load_kwh": float(problem.load_w.sum() * hours / 1000.0),
            "sha256": year_plan.inputs_sha256(),
        },
        "opening_state": {
            "soh_fraction": soh,
            "resistance_growth": year_plan.resistance_growth,
            "eff_charge": eff_charge,
            "eff_discharge": eff_discharge,
            "start_energy_wh": year_plan.start_energy_wh,
        },
        "terminal_energy_wh": year_plan.terminal_energy_wh,
        "objective": plan.objective,
        "stage_cost": plan.stage_cost,
        "wear_cost": plan.wear_cost,
        "terminal_cost": plan.terminal_cost,
        "end_energy_wh": plan.end_energy_wh,
        "replayed_cost": replayed_cost,
        "mean_target": float(plan.targets.mean()) if len(plan.targets) else None,
        "days_by_target": {f"{level:g}": int((plan.targets == level).sum()) for level in levels},
        "instruction_hash": year_plan.instructions.instruction_hash(),
        "solve_seconds": year_plan.solve_seconds,
    }


def report(result: DailyTargetOracleResult, case: ReplayCase) -> dict[str, Any]:
    """A JSON-safe summary of ``result``."""
    plan, problem = result.plan, result.problem
    hours = problem.hours_per_step
    replayed = replay_summary(result.replay, hours)
    fixed = replay_summary(result.fixed_target, hours)
    spec = case.resolved.smart_charging
    targets = plan.targets
    replayed_costs = _year_costs(result.replay)
    return {
        "schema": result.schema,
        "currency": result_currency(case.tariff),
        "n_steps": len(problem.instructions),
        "n_days": problem.n_days,
        "resolution": case.resolved.cfg["resolution"],
        "start": str(case.index[0]),
        "end": str(case.index[-1]),
        "battery_kwh": problem.battery_config.nominal_energy_wh / 1000.0,
        "cost_basis": "first project year import cost less export revenue; standing charge excluded",
        "planner": {
            "planning": result.planning,
            "decision_boundary": result.decision_boundary,
            "target_levels": result.target_levels.tolist(),
            "soc_states": result.soc_states,
            "free_terminal": result.free_terminal,
            "wear_cost_per_kwh": result.wear_cost_per_kwh,
            "soh_fraction": problem.health()[0],
            "objective": plan.objective,
            "stage_cost": plan.stage_cost,
            "wear_cost": plan.wear_cost,
            "terminal_cost": plan.terminal_cost,
            "planned_cost": float(result.planned_step_cost.sum()),
            "start_energy_wh": plan.start_energy_wh,
            "end_energy_wh": plan.end_energy_wh,
            "mean_target": float(targets.mean()) if len(targets) else None,
            "days_by_target": {f"{level:g}": int((targets == level).sum()) for level in result.target_levels},
            "solve_seconds": result.solve_seconds,
        },
        "replay": {
            **replayed,
            "minus_planned": replayed["first_year_cost"] - float(result.planned_step_cost.sum()),
            **plan_comparison(result.replay),
        },
        "fixed_target": {
            "target_usable_fraction": spec.target_usable_fraction if spec is not None else None,
            **fixed,
        },
        "replay_minus_fixed_target": replayed["first_year_cost"] - fixed["first_year_cost"],
        # Every project year of both replays, at year-1 prices; an NPV of a
        # replayed schedule, not a bound on any other schedule's.
        "lifetime": {
            "basis": "replayed project years; perfect-information daily targets on a grid, not an NPV bound",
            "year_instruction_hashes": list(result.replay.year_instruction_hashes),
            "replay": _lifetime(result.replay),
            "fixed_target": _lifetime(result.fixed_target),
        },
        "years": [
            _year_record(year_plan, replayed_costs[year_plan.year - 1], result.target_levels)
            for year_plan in result.year_plans
        ],
    }


def days_frame(result: DailyTargetOracleResult, case: ReplayCase) -> pd.DataFrame:
    """One row per planning day (civil day or charge window): its target, and its planned and replayed cost and end energy."""
    starts = np.asarray(result.problem.day_starts)
    first, last = starts[:-1], starts[1:] - 1
    delivered = result.replay.artifacts.first_year_results_df["Battery_Energy_End"].to_numpy()
    return pd.DataFrame(
        {
            "day_start": case.index[first].astype(str),
            "target_usable_fraction": result.plan.targets,
            "planned_cost": np.add.reduceat(result.planned_step_cost, first),
            "replayed_cost": np.add.reduceat(result.replay.first_year_step_cost, first),
            "fixed_target_cost": np.add.reduceat(result.fixed_target.first_year_step_cost, first),
            "planned_end_energy_wh": result.planned["battery_energy_wh"][last],
            "replayed_end_energy_wh": delivered[last],
        }
    )


def year_days_frame(result: DailyTargetOracleResult, case: ReplayCase) -> pd.DataFrame:
    """One row per project year and planning day of a yearly plan: the target it chose."""
    first = np.asarray(result.problem.day_starts)[:-1]
    day_start = case.index[first].astype(str)
    return pd.concat(
        [
            pd.DataFrame(
                {"year": year_plan.year, "day_start": day_start, "target_usable_fraction": year_plan.plan.targets}
            )
            for year_plan in result.year_plans
        ],
        ignore_index=True,
    )


def _wear_cost(text: str) -> float:
    try:
        return check_wear_cost(float(text))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"must be a finite number of at least 0, not {text!r}") from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config", required=True, help="App configuration (TOML or JSON) with fixed-target smart charging"
    )
    parser.add_argument("--output", default="-", help="JSON summary path; '-' (default) writes to standard output")
    parser.add_argument(
        "--csv", help="Write one row per day to this CSV; under yearly planning, one per project year and day"
    )
    parser.add_argument(
        "--planning", choices=PLANNING_MODES, default="first_year", help="Plan the first year, or replan every year"
    )
    parser.add_argument(
        "--decision-boundary",
        choices=DECISION_BOUNDARIES,
        default="civil_day",
        help="One target per civil day (default), or per charge window from its start",
    )
    parser.add_argument("--target-levels", type=int, default=DEFAULT_TARGET_LEVELS, help="Targets from 0 to 1")
    parser.add_argument("--soc-states", type=int, default=DEFAULT_SOC_STATES, help="Stored-energy grid points")
    parser.add_argument("--free-terminal", action="store_true", help="Do not buy back energy the year ends without")
    parser.add_argument(
        "--wear-cost-per-kwh",
        type=_wear_cost,
        default=0.0,
        help="Planner wear weight per kWh of battery DC discharge, in the tariff's currency (default 0)",
    )
    parser.add_argument("--atol-wh", type=float, default=DEFAULT_REPLAY_TOLERANCE.atol_wh, help="Replay tolerance")
    parser.add_argument("--execution-backend", choices=("python", "numba"), help="Planner and replay backend")
    args = parser.parse_args(argv)

    with reuse_prepared_inputs():
        case = prepare_replay(load_config(args.config))
        result = run_daily_target_oracle(
            case,
            target_levels=args.target_levels,
            soc_states=args.soc_states,
            free_terminal=args.free_terminal,
            tolerance=Tolerance(atol_wh=args.atol_wh),
            execution_backend=args.execution_backend,
            planning=args.planning,
            decision_boundary=args.decision_boundary,
            wear_cost_per_kwh=args.wear_cost_per_kwh,
        )
    write_json(report(result, case), args.output)
    if args.csv:
        if result.planning == "yearly":
            write_csv(year_days_frame(result, case), DP_YEAR_DAYS_SCHEMA, args.csv)
        else:
            write_csv(days_frame(result, case), DP_DAYS_SCHEMA, args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
