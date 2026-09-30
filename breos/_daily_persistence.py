"""Experimental daily-persistence smart charging (ADR 0002 A12).

``[smart_charging] mode = "daily_persistence"`` keeps the fixed-target
instruction layout, a discharge gate on the discharge periods and a grid
target on the charge periods, but chooses the target once per configured-zone
civil day while the App runs. Each day it:

1. forecasts the planner's window by repeating the last complete observed
   civil day (PV DC, load and input temperature), slot by local wall-clock
   time and DST fold;
2. builds a fresh :class:`~breos._daily_targets.DailyTargetProblem` on that
   forecast, the known tariff, and the battery's live state: measured stored
   energy, state of health and resistance-adjusted efficiencies;
3. solves it and executes the first day's target only.

Until one complete local day has been observed there is nothing to repeat,
so the day runs with no grid target: PV charging and the discharge gate
still apply. The planner prices energy that ends its window below the day's
starting energy (``preserve_start_energy``); that is a planning penalty, not
a dispatch instruction, and the live battery carries its physical state from
year to year as every App run does.

The controller runs through the private civil-day seam in
:mod:`breos._controller`: it sees no current or future PV, load or
temperature, and the canonical dispatch step, run by the App's projection,
remains the only simulation of the battery.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Any, Sequence, cast

import numpy as np

from breos._controller import ControllerDayDecision, ControllerDayInput, ObservedCivilDay, SlotKey
from breos._daily_targets import DailyTargetProblem, daily_target_instructions, solve_daily_targets
from breos._dispatch import lfp_capacity_factor
from breos.battery import BatteryConfig
from breos.dispatch_instructions import DispatchInstructions
from breos.smart_charging import TERMINAL_CONVENTION, SmartChargingSpec, check_tariff_periods, period_layout
from breos.tariffs import ResolvedTariff
from breos.utils import get_hours_per_step

# Version identifiers of the policy and of the planner it solves, recorded in
# provenance. A change to either's decisions bumps its version.
CONTROLLER_VERSION = "1"
PLANNER_VERSION = "1"
FORECAST_POLICY = "repeat_previous_complete_local_day"
WARM_START_POLICY = "no_grid_until_one_complete_local_day"
PLANNER_TERMINAL_POLICY = "preserve_start_energy"
# The candidate targets' placeholder on the layout: any finite value marks a
# charge step; the planner replaces it with each day's target.
_CHARGE_STEP = 1.0


def persistence_forecast(
    observed: ObservedCivilDay, slot_keys: Sequence[SlotKey]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PV DC, load and input temperature for ``slot_keys``, repeated from ``observed``.

    Slots match by configured-zone wall-clock time and DST fold, never by
    position, so a 23- or 25-hour day on either side keeps every other slot
    on its own wall time and the tariff's periods stay aligned. A target slot
    in the repeated hour of a fall-back day (fold 1) that the observed day
    does not repeat reuses the observed sample at the same wall time. A
    target slot the observed day lacks entirely, such as the hour a
    spring-forward day skips, is interpolated linearly in wall time between
    the observed slots on either side (the nearest one when only one side
    has a slot).
    """
    position = {key: index for index, key in enumerate(observed.slot_keys)}
    # Each wall time's first occurrence, in wall-time order, for interpolation.
    first: dict[int, int] = {}
    for index, (wall, _fold) in enumerate(observed.slot_keys):
        first.setdefault(wall, index)
    walls = sorted(first.items())
    wall_times = [wall for wall, _ in walls]
    series = (
        np.asarray(observed.pv_dc_w, dtype=np.float64),
        np.asarray(observed.load_w, dtype=np.float64),
        np.asarray(observed.temperature_c, dtype=np.float64),
    )
    out = tuple(np.empty(len(slot_keys), dtype=np.float64) for _ in series)
    for target, (wall, fold) in enumerate(slot_keys):
        source = position.get((wall, fold))
        if source is None and fold:
            source = position.get((wall, 0))
        if source is not None:
            for values, result in zip(series, out, strict=True):
                result[target] = values[source]
            continue
        right = bisect.bisect_left(wall_times, wall)
        if right == 0 or right == len(walls):
            nearest = walls[min(right, len(walls) - 1)][1]
            for values, result in zip(series, out, strict=True):
                result[target] = values[nearest]
            continue
        (left_wall, left), (right_wall, right_index) = walls[right - 1], walls[right]
        weight = (wall - left_wall) / (right_wall - left_wall)
        for values, result in zip(series, out, strict=True):
            result[target] = values[left] + weight * (values[right_index] - values[left])
    pv, load, temperature = out
    return pv, load, temperature


def _slice(instructions: DispatchInstructions, count: int) -> DispatchInstructions:
    return DispatchInstructions(
        discharge_allowed=instructions.discharge_allowed[:count],
        reserve_fraction=instructions.reserve_fraction[:count],
        grid_target_fraction=instructions.grid_target_fraction[:count],
        grid_charge_efficiency=instructions.grid_charge_efficiency,
        grid_import_limit_w=instructions.grid_import_limit_w,
    )


@dataclass(frozen=True)
class DailyPersistenceController:
    """The ``daily_persistence`` policy as a private daily controller (ADR 0002 A11, A12).

    ``spec`` is a resolved ``daily_persistence`` spec: its periods, grid-charge
    efficiency and import limit fix the instruction layout, and its planner
    settings the solve. ``hours_per_step`` is the simulation's step and
    ``execution_backend`` the backend the planner's day transitions run on,
    the simulation's own. It keeps no policy state: every decision follows
    from the day's input.
    """

    spec: SmartChargingSpec
    hours_per_step: float
    execution_backend: str

    def __post_init__(self) -> None:
        if self.spec.mode != "daily_persistence":
            raise ValueError("a daily-persistence controller needs a spec with mode = 'daily_persistence'")

    @classmethod
    def for_run(
        cls, spec: SmartChargingSpec, tariff: ResolvedTariff, *, freq: str, execution_backend: str
    ) -> DailyPersistenceController:
        """The controller for a run on ``tariff``'s calendar at resolution ``freq``."""
        check_tariff_periods(spec, tariff)
        return cls(spec, get_hours_per_step(freq), execution_backend)

    @property
    def tariff_horizon_days(self) -> int:
        horizon = self.spec.forecast_horizon_days
        assert horizon is not None
        return horizon

    def decide_day(self, day: ControllerDayInput, policy_state: object | None) -> ControllerDayDecision:
        horizon = day.tariff
        count = day.decision_step_count
        layout = period_layout(self.spec, horizon.period_labels, _CHARGE_STEP)
        observed = day.last_complete_observed_day
        if observed is None:
            # Warm start: no complete day to repeat yet, so no grid target.
            today = _slice(layout, count)
            return ControllerDayDecision(
                DispatchInstructions(
                    discharge_allowed=today.discharge_allowed,
                    reserve_fraction=today.reserve_fraction,
                    grid_target_fraction=np.full(count, np.nan),
                    grid_charge_efficiency=today.grid_charge_efficiency,
                    grid_import_limit_w=today.grid_import_limit_w,
                ),
                policy_state,
            )

        pv, load, temperature = persistence_forecast(observed, horizon.slot_keys)
        battery = BatteryConfig(**cast("dict[str, Any]", dict(day.battery_config)))
        state = day.battery_state
        problem = DailyTargetProblem(
            pv_dc_w=pv,
            load_w=load,
            temperature_c=temperature,
            import_price_per_kwh=np.asarray(horizon.import_price_per_kwh, dtype=np.float64),
            export_price_per_kwh=np.asarray(horizon.export_price_per_kwh, dtype=np.float64),
            day_starts=horizon.civil_day_offsets,
            instructions=layout,
            battery_config=battery,
            hours_per_step=self.hours_per_step,
            soh_fraction=state.soh_fraction,
            eff_charge=state.charge_efficiency,
            eff_discharge=state.discharge_efficiency,
        )
        plan = solve_daily_targets(
            problem,
            initial_energy_wh=state.energy_wh,
            target_levels=self._setting("target_levels"),
            soc_states=self._setting("soc_states"),
            terminal_energy_wh=preserve_start_energy(battery, state.energy_wh, state.soh_fraction, temperature),
            free_terminal=False,
            execution_backend=self.execution_backend,
        )
        # Only today's target is executed; tomorrow is planned again.
        today = daily_target_instructions(_slice(layout, count), (0, count), plan.targets[:1])
        return ControllerDayDecision(today, policy_state)

    def _setting(self, name: str) -> int:
        value = getattr(self.spec, name)
        assert isinstance(value, int)
        return value


def preserve_start_energy(
    battery: BatteryConfig, energy_wh: float, soh_fraction: float, temperature_c: Sequence[float] | np.ndarray
) -> float:
    """The ``preserve_start_energy`` terminal target of one rolling solve, in Wh.

    The day's measured starting energy, capped at the most the battery can
    hold at the forecast's final temperature: the max-SOC energy of the
    pack at its current health.
    """
    final = float(temperature_c[-1])
    capacity = battery.nominal_energy_wh * soh_fraction * battery.max_soc * lfp_capacity_factor(final)
    return min(float(energy_wh), capacity)


def daily_persistence_provenance(
    spec: SmartChargingSpec,
    tariff: ResolvedTariff,
    executed: DispatchInstructions,
    initial_stored_energy: dict[str, float],
    final_stored_energy: dict[str, float],
) -> dict[str, Any]:
    """A JSON-safe record of a daily-persistence run for provenance.

    ``executed`` is every instruction the run dispatched, in project order
    across all its years; its hash identifies what the policy did without
    exposing its forecasts or per-day targets.
    """
    return {
        "mode": spec.mode,
        "experimental": True,
        "controller_version": CONTROLLER_VERSION,
        "planner_version": PLANNER_VERSION,
        "charge_periods": list(spec.charge_periods),
        "discharge_periods": list(spec.discharge_periods),
        "grid_charge_efficiency": spec.grid_charge_efficiency,
        # None, not math.inf: strict JSON has no infinity.
        "grid_import_limit_w": spec.grid_import_limit_w,
        "forecast_horizon_days": spec.forecast_horizon_days,
        "target_levels": spec.target_levels,
        "soc_states": spec.soc_states,
        "forecast_policy": FORECAST_POLICY,
        "warm_start_policy": WARM_START_POLICY,
        "planner_terminal_policy": PLANNER_TERMINAL_POLICY,
        "terminal_convention": TERMINAL_CONVENTION,
        "schedule_hash": tariff.schedule_hash,
        "instruction_hash": executed.instruction_hash(),
        "initial_stored_energy": dict(initial_stored_energy),
        "final_stored_energy": dict(final_stored_energy),
    }
