"""Replay dispatch instructions through the production App run, and price them.

An oracle or controller hands over a :class:`~breos.dispatch_instructions.DispatchInstructions`
and, optionally, the per-step flows it expects them to produce. The replay
runs the instructions through the App runner itself
(:func:`breos.runners.app.run_app_simulation`): the same inputs, the same
year loop and ``[period]`` handling, and stored energy, origins and
degradation carried from one project year into the next. The tariff prices
the result. Each planned flow is then compared with what the first project
year delivered.

This is the BREOS counterpart of the legacy ``pvbat/dispatch_replay.py``
(``07dc0e40``). The legacy replay took explicit power requests. Here the
control is the instruction arrays, which the dispatch step applies under
every physical limit, and a planned flow is only what the planner expected:
it never drives the dispatch. A mismatch therefore means production physics
did not do what the planner's model assumed, for example because a power
limit bound or health moved during the year.

It evaluates one schedule and makes no optimisation claim. It is a
validation tool, not public API. To replay several schedules on one
configuration without preparing its inputs each time, run
:func:`prepare_replay` and the replays inside
:func:`breos.app_inputs.reuse_prepared_inputs`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping

import numpy as np
import pandas as pd

from breos.app import App
from breos.app_config import ResolvedAppConfig, resolve_app_config
from breos.app_inputs import AppRuntimeDependencies, PreparedSimulationInputs, prepare_simulation_inputs_cached
from breos.battery import AlignedSimulationInputs, BatteryConfig, align_simulation_inputs
from breos.dispatch_instructions import DispatchInstructions
from breos.execution import config_has_battery
from breos.projection import CarryState, build_battery_config
from breos.runners import app as app_runner
from breos.runners.app import SimulationArtifacts
from breos.smart_charging import resolve_instructions
from breos.tariffs import ResolvedTariff
from breos.utils import get_hours_per_step

REPLAY_SCHEMA = "breos_dispatch_replay_v2"

# A planned flow, by name, and the results column it is compared with. The
# ``_w`` flows are average power over the step, as the ledger reports them;
# ``battery_energy_wh`` is the stored energy at the end of the step.
PLANNED_FLOWS: dict[str, str] = {
    "pv_charge_dc_w": "PV_DC_To_Battery",
    "grid_charge_ac_w": "Grid_AC_To_Battery",
    "discharge_ac_w": "Battery_AC_To_Load",
    "battery_energy_wh": "Battery_Energy_End",
}


@dataclass(frozen=True)
class Tolerance:
    """How far a delivered flow may miss its plan on one step and still match.

    A step matches when ``|planned - delivered| <= atol_wh + rtol * |delivered|``,
    with both sides in Wh over the step.
    """

    atol_wh: float = 1e-7
    rtol: float = 0.0

    def __post_init__(self) -> None:
        for name in ("atol_wh", "rtol"):
            value = getattr(self, name)
            if not (np.isfinite(value) and value >= 0.0):
                raise ValueError(f"'{name}' must be finite and non-negative")


DEFAULT_TOLERANCE = Tolerance()


@dataclass(frozen=True)
class ReplayCase:
    """One App configuration, prepared once and replayable with any instructions.

    ``inputs`` and ``tariff`` are what a planner needs to build its problem:
    the per-step PV, load and battery temperature, and the tariff resolved on
    the simulation index.
    """

    resolved: ResolvedAppConfig
    deps: AppRuntimeDependencies
    inputs: PreparedSimulationInputs
    tariff: ResolvedTariff

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.tariff.index

    def aligned_inputs(self) -> AlignedSimulationInputs:
        """The first project year's PV, load and battery temperature, as the dispatch reads them.

        They are aligned onto the simulation calendar the way the runner's
        projection aligns them, so a planner sees the per-step values the
        replay dispatches on.
        """
        aligned = align_simulation_inputs(
            self.inputs.dc_system_base,
            self.inputs.load_data,
            self.inputs.temperature_series,
            freq=self.resolved.cfg["resolution"],
        )
        if not aligned.index.equals(self.index):
            raise ValueError("The aligned simulation calendar differs from the tariff's calendar")
        return aligned

    def battery_config(self) -> BatteryConfig:
        """The battery the first project year runs, at the state of health it starts with."""
        return build_battery_config(self.resolved.cfg, self.resolved, initial_soh=CarryState().soh_pct)

    def configured_instructions(self) -> DispatchInstructions:
        """What App dispatches with: the ``[smart_charging]`` instructions, or greedy no-ops without them."""
        spec = self.resolved.smart_charging
        instructions = resolve_instructions(spec, self.tariff) if spec is not None else None
        return instructions if instructions is not None else DispatchInstructions.noop(len(self.index))


def prepare_replay(config: dict[str, Any], *, deps: AppRuntimeDependencies | None = None) -> ReplayCase:
    """Resolve ``config`` and prepare its inputs and tariff as App does.

    Raises:
        ValueError: If the configuration has no tariff to price with, or no
            battery for the instructions to act on.
    """
    resolved = resolve_app_config(config)
    cfg = resolved.cfg
    if resolved.tariff is None:
        raise ValueError("A replay prices its result with a tariff; the configuration has no [tariff] table")
    if not config_has_battery(cfg):
        raise ValueError("A replay needs a battery: a PV-only run ignores dispatch instructions")
    deps = deps or App._runtime_dependencies()
    # The runner's own preparation, so a reuse_prepared_inputs block shares it.
    inputs = prepare_simulation_inputs_cached(cfg, resolved, deps, prepare=app_runner.prepare_simulation_inputs)
    tariff = resolved.tariff.resolve(pd.DatetimeIndex(inputs.dc_system_base.index), resolved.timezone)
    return ReplayCase(resolved=resolved, deps=deps, inputs=inputs, tariff=tariff)


@dataclass(frozen=True)
class ReplayResult:
    """A replayed schedule: the App run, and its planned against delivered flows.

    ``artifacts`` is what the App runner produced: the priced year rows, the
    cost projection (None for a ``[period]`` window), the first year's
    per-step frame and the projection. ``planned``, ``delivered`` and
    ``planned_minus_delivered_wh`` hold the flows the caller planned, for the
    first project year only; a power difference is converted to energy over
    the step. ``mismatched_steps`` are the steps where any planned flow missed
    its tolerance, and ``plan_matched`` says there were none. With no planned
    flows nothing is compared and the plan matches. ``first_year_step_cost``
    is each first-year step's import cost less export revenue.
    """

    artifacts: SimulationArtifacts
    instruction_hash: str
    first_year_step_cost: np.ndarray
    planned: dict[str, np.ndarray]
    delivered: dict[str, np.ndarray]
    planned_minus_delivered_wh: dict[str, np.ndarray]
    mismatched_steps: tuple[int, ...]
    plan_matched: bool
    tolerances: dict[str, Tolerance]
    schema: str = REPLAY_SCHEMA


def _planned_arrays(planned: Mapping[str, Any], n_steps: int) -> dict[str, np.ndarray]:
    unknown = sorted(set(planned) - set(PLANNED_FLOWS))
    if unknown:
        raise ValueError(f"Unknown planned flow(s) {', '.join(unknown)}; known: {', '.join(PLANNED_FLOWS)}")
    arrays = {}
    for name, values in planned.items():
        array = np.array(values, dtype=np.float64)
        if array.shape != (n_steps,) or not np.isfinite(array).all() or (array < 0.0).any():
            raise ValueError(f"Planned '{name}' must hold {n_steps} finite, non-negative values")
        arrays[name] = array
    return arrays


def _tolerances(tolerance: Tolerance | Mapping[str, Tolerance], flows: Mapping[str, Any]) -> dict[str, Tolerance]:
    if isinstance(tolerance, Tolerance):
        return {name: tolerance for name in flows}
    unknown = sorted(set(tolerance) - set(PLANNED_FLOWS))
    if unknown:
        raise ValueError(f"Tolerance for unknown flow(s) {', '.join(unknown)}; known: {', '.join(PLANNED_FLOWS)}")
    return {name: tolerance.get(name, DEFAULT_TOLERANCE) for name in flows}


def replay_instructions(
    case: ReplayCase,
    instructions: DispatchInstructions,
    *,
    planned: Mapping[str, Any] | None = None,
    tolerance: Tolerance | Mapping[str, Tolerance] = DEFAULT_TOLERANCE,
    execution_backend: str | None = None,
) -> ReplayResult:
    """Run ``instructions`` through ``case``'s App run and compare the first year with ``planned``.

    The instructions cover one year on the tariff's calendar and are replayed
    every project year, as App replays its own (ADR 0002 A2). ``planned``
    maps names in :data:`PLANNED_FLOWS` to one value per step of the first
    project year, the only year whose per-step frame the run keeps; later
    years are simulated and priced but not compared. ``tolerance`` is one
    :class:`Tolerance` for every flow, or one per flow name, with
    :data:`DEFAULT_TOLERANCE` for a flow it leaves out. A planned power is
    compared as energy over the step, so its tolerance is in Wh too. The
    execution backend defaults to the configuration's.
    """
    n_steps = len(case.index)
    if len(instructions) != n_steps:
        raise ValueError(f"The instructions cover {len(instructions)} steps; the tariff's calendar has {n_steps}")
    requested = _planned_arrays(planned or {}, n_steps)
    tolerances = _tolerances(tolerance, requested)
    resolved = (
        case.resolved
        if execution_backend is None
        else replace(case.resolved, cfg={**case.resolved.cfg, "execution_backend": execution_backend})
    )

    artifacts = app_runner.run_app_simulation(resolved, case.deps, instructions=instructions)

    frame = artifacts.first_year_results_df
    hours_per_step = get_hours_per_step(resolved.cfg["resolution"])
    step_cost = (
        frame["Import_From_Grid"].to_numpy() * np.asarray(case.tariff.import_price_per_kwh)
        - frame["PV_AC_Export"].to_numpy() * np.asarray(case.tariff.export_price_per_kwh)
    ) * (hours_per_step / 1000)

    delivered = {name: frame[PLANNED_FLOWS[name]].to_numpy(dtype=np.float64) for name in requested}
    mismatched = np.zeros(n_steps, dtype=bool)
    differences = {}
    for name, values in requested.items():
        to_wh = 1.0 if name.endswith("_wh") else hours_per_step
        differences[name] = (values - delivered[name]) * to_wh
        allowed = tolerances[name].atol_wh + tolerances[name].rtol * np.abs(delivered[name] * to_wh)
        mismatched |= np.abs(differences[name]) > allowed
    return ReplayResult(
        artifacts=artifacts,
        instruction_hash=instructions.instruction_hash(),
        first_year_step_cost=step_cost,
        planned=requested,
        delivered=delivered,
        planned_minus_delivered_wh=differences,
        mismatched_steps=tuple(int(step) for step in np.flatnonzero(mismatched)),
        plan_matched=not bool(mismatched.any()),
        tolerances=tolerances,
    )


def replay_summary(replay: ReplayResult, hours_per_step: float) -> dict[str, Any]:
    """A replay's first-year cost, its instructions' hash, its lowest health and its grid charge."""
    frame = replay.artifacts.first_year_results_df
    return {
        "first_year_cost": float(replay.first_year_step_cost.sum()),
        "instruction_hash": replay.instruction_hash,
        "min_soh_pct": float(frame["Battery_SOH"].min()),
        "grid_charge_kwh": float(frame["Grid_AC_To_Battery"].sum() * hours_per_step / 1000.0),
        "battery_ac_to_load_kwh": float(frame["Battery_AC_To_Load"].sum() * hours_per_step / 1000.0),
    }


def reference_dispatch(case: ReplayCase) -> str:
    """``"fixed_target"`` when App dispatches with a fixed-target table, else ``"greedy"``."""
    spec = case.resolved.smart_charging
    return "fixed_target" if spec is not None and spec.mode == "fixed_target" else "greedy"


def plan_comparison(replay: ReplayResult) -> dict[str, Any]:
    """How far a replay's delivered flows missed the plan: counts and totals, in kWh."""
    return {
        "plan_matched": replay.plan_matched,
        "mismatched_steps": len(replay.mismatched_steps),
        "tolerances_wh": {name: asdict(tol) for name, tol in replay.tolerances.items()},
        "planned_minus_delivered_kwh": {
            name: {
                "sum": float(difference.sum() / 1000.0),
                "max_abs": float(np.abs(difference).max(initial=0.0) / 1000.0),
            }
            for name, difference in replay.planned_minus_delivered_wh.items()
        },
    }
