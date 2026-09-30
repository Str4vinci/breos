"""The daily-target oracle: the best grid-charge target for each day, with perfect foresight.

For an App configuration with fixed-target smart charging, the oracle plans
the whole first project year at once with the private daily-target dynamic
program (:mod:`breos._daily_targets`). It sees the true PV, load and battery
temperature of every day. The policy class is the configured instruction
layout, with one target per day in place of the configured one: every
charge step of a day takes that day's target, from a grid of usable
fractions. The discharge gate, the reserve, the grid-charge efficiency and
the site limit stay as configured. The planned schedule is then replayed
through the production App run (:mod:`tools.oracles.replay`) and priced,
next to App's own fixed-target run of the same configuration.

This is the BREOS counterpart of the offline oracle of the legacy
``tools/compute_a2_daily_sc_oracle.py`` (``--forecast-mode perfect``, at
``07dc0e40``). It is a validation tool, not public API. It is an optimum
over one policy class under the planner's model, not a bound: the linear
program of :mod:`tools.oracles.lp_bound` bounds every policy.

The planner holds state of health and both efficiencies at the year's
opening values, and its value function lives on a grid of stored energy.
The replay is the accounting truth: it runs the instructions under the
health production physics gives. ``planned`` is what the same instructions
deliver at the planner's fixed health, one dispatch call over the window,
so the planned-against-delivered comparison isolates what health moving
during the year changed. With perfect foresight and one plan for the whole
window, the replayed cost and the plan's stage cost differ only by that.

Days are the tariff's civil days (ADR 0002 A1). By default, energy that
ends the year below full is bought back at the cheapest price a charge
step may pay, so the plan does not drain the battery on the last day
(``free_terminal`` drops the refill). Every project year replays the
first year's instructions (A2); the costs compared are the first year's.

Legacy behaviour not ported: the causal persistence-forecast controller
(that is the ``daily_persistence`` controller of plan step 7), re-planning
at each project year's opening health, the matched sweep over every fixed
target (``--compare-fixed``), the terminal-health sensitivity and the
lifetime NPV sums.

Usage:
    python tools/oracles/daily_target_dp.py --config my.toml --output dp.json --csv days.csv
"""

from __future__ import annotations

import argparse
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
    _DayEvaluator,
    daily_target_instructions,
    solve_daily_targets,
    target_grid,
)
from breos.app_inputs import reuse_prepared_inputs
from breos.battery import _ResultBuffers
from breos.dispatch_instructions import DispatchInstructions
from breos.tariffs import result_currency
from tools.oracles._output import load_config, write_csv, write_json
from tools.oracles.replay import (
    PLANNED_FLOWS,
    ReplayCase,
    ReplayResult,
    Tolerance,
    plan_comparison,
    prepare_replay,
    replay_instructions,
    replay_summary,
)

DP_ORACLE_SCHEMA = "breos_daily_target_dp_oracle_v1"
DP_DAYS_SCHEMA = "breos_daily_target_dp_days_v1"
# Planned and delivered energy must agree to within this before a step counts as a mismatch.
DEFAULT_REPLAY_TOLERANCE = Tolerance(atol_wh=1.0)


@dataclass(frozen=True)
class DailyTargetOracleResult:
    """The oracle's plan, its replay, and App's own fixed-target run.

    ``planned`` holds the flows the plan's instructions deliver at the
    planner's fixed health, and ``planned_step_cost`` their per-step import
    cost less export revenue. ``replay`` compares them with production.
    ``fixed_target`` replays the configured instructions, which App
    dispatches with.
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
    schema: str = DP_ORACLE_SCHEMA


def daily_target_problem(case: ReplayCase) -> DailyTargetProblem:
    """``case``'s first project year as a daily-target problem on its fixed-target layout.

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
    )


def fixed_health_flows(
    problem: DailyTargetProblem, instructions: DispatchInstructions, *, execution_backend: str = "python"
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """What ``instructions`` deliver over ``problem`` at its fixed health, and each step's cost.

    One call of the production dispatch step from a full battery, as the
    planner's forward pass chains its days. Returns the flows named in
    :data:`tools.oracles.replay.PLANNED_FLOWS` and the per-step import cost
    less export revenue.
    """
    evaluator = _DayEvaluator(problem, target_grid(1), execution_backend)
    soh, _eff_charge, _eff_discharge = problem.health()
    config = problem.battery_config
    buffers = _ResultBuffers(len(instructions))
    evaluator.dispatch_day(
        buffers,
        *evaluator.series,
        0,
        len(instructions),
        Battery_Energy_Wh=config.nominal_energy_wh * soh * config.max_soc,
        instructions=instructions,
        **evaluator.state,
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
) -> DailyTargetOracleResult:
    """Plan ``case``'s first year one target per day, replay the plan and App's fixed-target run.

    The planner and the replays run on the configuration's execution
    backend unless ``execution_backend`` names one.
    """
    backend = execution_backend or case.resolved.cfg.get("execution_backend", "python")
    problem = daily_target_problem(case)
    started = time.perf_counter()
    plan = solve_daily_targets(
        problem,
        target_levels=target_levels,
        soc_states=soc_states,
        free_terminal=free_terminal,
        execution_backend=backend,
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
    )


def report(result: DailyTargetOracleResult, case: ReplayCase) -> dict[str, Any]:
    """A JSON-safe summary of ``result``."""
    plan, problem = result.plan, result.problem
    hours = problem.hours_per_step
    replayed = replay_summary(result.replay, hours)
    fixed = replay_summary(result.fixed_target, hours)
    spec = case.resolved.smart_charging
    targets = plan.targets
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
            "target_levels": result.target_levels.tolist(),
            "soc_states": result.soc_states,
            "free_terminal": result.free_terminal,
            "soh_fraction": problem.health()[0],
            "objective": plan.objective,
            "stage_cost": plan.stage_cost,
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
    }


def days_frame(result: DailyTargetOracleResult, case: ReplayCase) -> pd.DataFrame:
    """One row per civil day: its target, and its planned and replayed cost and end energy."""
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config", required=True, help="App configuration (TOML or JSON) with fixed-target smart charging"
    )
    parser.add_argument("--output", default="-", help="JSON summary path; '-' (default) writes to standard output")
    parser.add_argument("--csv", help="Write one row per day to this CSV")
    parser.add_argument("--target-levels", type=int, default=DEFAULT_TARGET_LEVELS, help="Targets from 0 to 1")
    parser.add_argument("--soc-states", type=int, default=DEFAULT_SOC_STATES, help="Stored-energy grid points")
    parser.add_argument("--free-terminal", action="store_true", help="Do not buy back energy the year ends without")
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
        )
    write_json(report(result, case), args.output)
    if args.csv:
        write_csv(days_frame(result, case), DP_DAYS_SCHEMA, args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
