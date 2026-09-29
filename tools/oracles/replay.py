"""Replay dispatch instructions through the production projection, and price them.

An oracle or controller hands over a :class:`~breos.dispatch_instructions.DispatchInstructions`
and, optionally, the per-step flows it expects them to produce. The replay
runs the instructions through the App's own projection: the same inputs, the
same year loop, and stored energy, origins and degradation carried from one
project year into the next. The tariff prices the result. Each planned flow
is then compared with what the first project year delivered, and every step
where the two differ by more than the tolerance is reported as corrected.

This is the BREOS counterpart of the legacy ``pvbat/dispatch_replay.py``
(``07dc0e40``). The legacy replay took explicit power requests. Here the
control is the instruction arrays, which the dispatch step applies under
every physical limit, and a planned flow is what the planner expected. A
correction means production physics did not do what the plan assumed, for
example because a power limit bound or health moved during the year.

It evaluates one schedule and makes no optimisation claim. It is a
validation tool, not public API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from breos.app import App
from breos.app_config import ResolvedAppConfig, resolve_app_config
from breos.app_inputs import AppRuntimeDependencies, PreparedSimulationInputs, prepare_simulation_inputs
from breos.dispatch_instructions import DispatchInstructions
from breos.execution import is_pv_only_dispatch
from breos.projection import ProjectionRun, ProjectionValue, ProjectionYear, run_projection, value_projection
from breos.tariffs import ResolvedTariff
from breos.utils import get_hours_per_step

REPLAY_SCHEMA = "breos_dispatch_replay_v1"
DEFAULT_TOLERANCE_WH = 1e-7

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
class ReplayCase:
    """One App configuration, prepared once and replayable with any instructions.

    ``inputs`` and ``tariff`` are what a planner needs to build its problem:
    the per-step PV, load and battery temperature, and the tariff resolved on
    the simulation index.
    """

    cfg: dict[str, Any]
    resolved: ResolvedAppConfig
    inputs: PreparedSimulationInputs
    tariff: ResolvedTariff

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.tariff.index


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
    if is_pv_only_dispatch(cfg["battery_kwh"] * 1000, cfg["battery_max_soc"], cfg["battery_min_soc"]):
        raise ValueError("A replay needs a battery: a PV-only run ignores dispatch instructions")
    inputs = prepare_simulation_inputs(cfg, resolved, deps or App._runtime_dependencies())
    tariff = resolved.tariff.resolve(pd.DatetimeIndex(inputs.dc_system_base.index), resolved.timezone)
    return ReplayCase(cfg=cfg, resolved=resolved, inputs=inputs, tariff=tariff)


@dataclass(frozen=True)
class ReplayResult:
    """A replayed schedule: the projection, its price, and planned against delivered.

    ``planned``, ``delivered`` and ``planned_minus_delivered_wh`` hold the
    flows the caller planned, for the first project year. A power difference
    is converted to energy over the step. ``corrected_steps`` are the steps
    where any planned flow missed by more than ``tolerance_wh``; with no
    planned flows there are none, and ``requests_feasible`` is True.
    ``first_year_step_cost`` is each step's import cost less export revenue.
    """

    run: ProjectionRun
    value: ProjectionValue
    instruction_hash: str
    first_year_step_cost: np.ndarray
    planned: dict[str, np.ndarray]
    delivered: dict[str, np.ndarray]
    planned_minus_delivered_wh: dict[str, np.ndarray]
    corrected_steps: tuple[int, ...]
    requests_feasible: bool
    tolerance_wh: float
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


def replay_instructions(
    case: ReplayCase,
    instructions: DispatchInstructions,
    *,
    planned: Mapping[str, Any] | None = None,
    tolerance_wh: float = DEFAULT_TOLERANCE_WH,
    execution_backend: str | None = None,
) -> ReplayResult:
    """Run ``instructions`` through ``case``'s projection and compare them with ``planned``.

    The instructions cover one year on the tariff's calendar and are replayed
    every project year, as App replays its own (ADR 0002 A2). The execution
    backend defaults to the configuration's.
    """
    if not (np.isfinite(tolerance_wh) and tolerance_wh >= 0.0):
        raise ValueError("'tolerance_wh' must be finite and non-negative")
    n_steps = len(case.index)
    if len(instructions) != n_steps:
        raise ValueError(f"The instructions cover {len(instructions)} steps; the tariff's calendar has {n_steps}")
    requested = _planned_arrays(planned or {}, n_steps)
    cfg, inputs = case.cfg, case.inputs

    # The year loop is App's (breos.runners.app.run_app_simulation): a
    # [period] window runs once and bills its civil days.
    period = case.resolved.period
    years = 1 if period is not None else cfg["projection_years"]
    extra = {"Billed_Days": float(period.days)} if period is not None else {}

    def year_inputs(year_idx: int) -> ProjectionYear:
        factor = (1 - cfg["pv_degradation_rate"]) ** year_idx
        return ProjectionYear(
            pv_degradation_factor=factor,
            pv_dc=inputs.dc_system_base * factor,
            houseload=inputs.load_data,
            temperature_series=inputs.temperature_series,
            extra=extra,
        )

    run = run_projection(
        cfg,
        case.resolved,
        years,
        year_inputs,
        has_battery=True,
        execution_backend=execution_backend or cfg["execution_backend"],
        tariff=case.tariff,
        instructions=instructions,
    )
    value = value_projection(cfg, case.resolved, run)

    frame = run.first_year_results_df
    assert frame is not None
    hours_per_step = get_hours_per_step(cfg["resolution"])
    step_cost = (
        frame["Import_From_Grid"].to_numpy() * np.asarray(case.tariff.import_price_per_kwh)
        - frame["PV_AC_Export"].to_numpy() * np.asarray(case.tariff.export_price_per_kwh)
    ) * (hours_per_step / 1000)

    delivered = {name: frame[PLANNED_FLOWS[name]].to_numpy(dtype=np.float64) for name in requested}
    differences = {
        name: (values - delivered[name]) * (1.0 if name.endswith("_wh") else hours_per_step)
        for name, values in requested.items()
    }
    corrected = np.zeros(n_steps, dtype=bool)
    for difference in differences.values():
        corrected |= np.abs(difference) > tolerance_wh
    return ReplayResult(
        run=run,
        value=value,
        instruction_hash=instructions.instruction_hash(),
        first_year_step_cost=step_cost,
        planned=requested,
        delivered=delivered,
        planned_minus_delivered_wh=differences,
        corrected_steps=tuple(int(step) for step in np.flatnonzero(corrected)),
        requests_feasible=not bool(corrected.any()),
        tolerance_wh=float(tolerance_wh),
    )
