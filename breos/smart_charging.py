"""Smart-charging controllers: tariff periods to dispatch instructions (ADR 0002).

A controller decides when the battery may discharge and how full the grid
may charge it. It does not simulate the battery: the canonical dispatch step
applies its :class:`~breos.dispatch_instructions.DispatchInstructions` under
every physical limit.

``fixed_target`` is the supported first strategy. On a step whose tariff
period is a charge period the grid may charge the battery toward
``target_usable_fraction`` of that step's usable window (A7). On a step whose
period is a discharge period the battery may discharge. A step in neither
set does neither; PV may charge the battery on every step.
With ``overlap_policy = "hold_target"`` a period in both sets keeps the grid
target as its discharge floor: above it discharge is allowed, below it grid
charging is allowed. The default ``reject`` requires disjoint periods.

``discharge_only`` lets the battery discharge on a step whose period is a
discharge period and hold its charge on every other step. The grid never
charges it, so it takes no grid-charging settings; PV may still charge the
battery on every step. Discharging in every period is greedy dispatch.

App runs ``fixed_target`` and ``discharge_only`` through the private
civil-day controller seam (ADR 0002 A11) with
:class:`FixedTargetDayController`, which hands each day its slice of the
resolved instructions.

``daily_persistence`` is experimental and App-only (ADR 0002 A12). It keeps
the fixed-target layout but chooses each day's target at runtime: a private
controller (``breos._daily_persistence``) re-plans every civil day from a
forecast that repeats the last complete observed day. It has no static
instructions, so :func:`resolve_instructions` refuses it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np

from breos._controller import ControllerDayDecision, ControllerDayInput
from breos._daily_targets import DEFAULT_HORIZON_DAYS, DEFAULT_SOC_STATES, DEFAULT_TARGET_LEVELS
from breos.dispatch_instructions import DispatchInstructions
from breos.tariffs import ResolvedTariff

SMART_CHARGING_MODES = ("disabled", "fixed_target", "daily_persistence", "discharge_only")
OVERLAP_POLICIES = ("reject", "hold_target")
# The modes that plan targets at runtime; they take the planner settings.
PLANNER_MODES = ("daily_persistence",)
# The modes that never charge from the grid; they take discharge periods only.
DISCHARGE_ONLY_MODES = ("discharge_only",)
# The planner settings, their defaults (the planner's own) and their minima.
PLANNER_SETTINGS: dict[str, tuple[int, int]] = {
    "forecast_horizon_days": (DEFAULT_HORIZON_DAYS, 1),
    "target_levels": (DEFAULT_TARGET_LEVELS, 1),
    "soc_states": (DEFAULT_SOC_STATES, 2),
}
# Normal App runs carry stored energy, origin shares and degradation from one
# project year into the next (ADR 0002, boundary and terminal conventions).
TERMINAL_CONVENTION = "physical_carry"


@dataclass(frozen=True)
class SmartChargingSpec:
    """A configured ``[smart_charging]`` table before it meets a tariff.

    ``grid_import_limit_w`` is a site import limit that the load's own import
    counts against: grid charging never takes total import above it, but
    load import is never cut, so a load above the limit still imports in
    full. None is no limit. With ``mode = "disabled"`` the other fields keep their defaults.

    ``fixed_target`` needs ``target_usable_fraction``. ``daily_persistence``
    refuses it, since the planner picks each day's target, and takes the
    three planner settings instead. ``discharge_only`` takes
    ``discharge_periods`` alone: it never charges from the grid, so every
    grid-charging setting stays unset. A planner setting left None is filled
    with the planner's default, so a resolved spec always holds the values
    the run used. The App checks the table
    (``breos.app_config.SMART_CHARGING_TABLE``); this value only keeps a
    direct construction coherent.
    """

    mode: str
    target_usable_fraction: float | None = None
    charge_periods: tuple[str, ...] = ()
    discharge_periods: tuple[str, ...] = ()
    grid_charge_efficiency: float | None = None
    grid_import_limit_w: float | None = None
    forecast_horizon_days: int | None = None
    target_levels: int | None = None
    soc_states: int | None = None
    overlap_policy: str = "reject"

    def __post_init__(self) -> None:
        if self.mode not in SMART_CHARGING_MODES:
            raise ValueError(f"'smart_charging.mode' must be one of: {', '.join(SMART_CHARGING_MODES)}")
        check_overlap_policy(self.mode, self.overlap_policy)
        object.__setattr__(self, "charge_periods", tuple(self.charge_periods))
        object.__setattr__(self, "discharge_periods", tuple(self.discharge_periods))
        planner = {name: getattr(self, name) for name in PLANNER_SETTINGS}
        if self.mode == "disabled":
            settings = (
                self.target_usable_fraction,
                self.charge_periods,
                self.discharge_periods,
                self.grid_charge_efficiency,
                self.grid_import_limit_w,
                *planner.values(),
            )
            if any(value not in (None, ()) for value in settings):
                raise ValueError("'smart_charging' with mode = 'disabled' takes no other settings")
            return
        if self.mode in DISCHARGE_ONLY_MODES:
            if not self.discharge_periods:
                raise ValueError(f"'smart_charging' needs smart_charging.discharge_periods for mode = '{self.mode}'")
            given = [
                name
                for name in (
                    "target_usable_fraction",
                    "charge_periods",
                    "grid_charge_efficiency",
                    "grid_import_limit_w",
                    *PLANNER_SETTINGS,
                )
                if getattr(self, name) not in (None, ())
            ]
            if given:
                raise ValueError(
                    f"'smart_charging' with mode = '{self.mode}' never charges from the grid and takes only "
                    f"discharge_periods; remove {', '.join(f'smart_charging.{name}' for name in given)}"
                )
            return
        required = ["charge_periods", "discharge_periods", "grid_charge_efficiency"]
        if self.mode == "fixed_target":
            required.insert(0, "target_usable_fraction")
        missing = [name for name in required if getattr(self, name) in (None, ())]
        if missing:
            raise ValueError(
                f"'smart_charging' needs {', '.join(f'smart_charging.{name}' for name in missing)} "
                f"for mode = '{self.mode}'"
            )
        if self.mode in PLANNER_MODES:
            if self.target_usable_fraction is not None:
                raise ValueError(
                    f"'smart_charging.target_usable_fraction' is not a setting of mode = '{self.mode}': "
                    "the planner chooses each day's target"
                )
            for name, (default, minimum) in PLANNER_SETTINGS.items():
                value = default if planner[name] is None else planner[name]
                if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                    raise ValueError(f"'smart_charging.{name}' must be an integer of at least {minimum}")
                object.__setattr__(self, name, int(value))
        else:
            given = [name for name, value in planner.items() if value is not None]
            if given:
                raise ValueError(
                    f"'smart_charging' with mode = '{self.mode}' takes no planner settings; "
                    f"remove {', '.join(f'smart_charging.{name}' for name in given)}"
                )
        overlap = sorted(set(self.charge_periods) & set(self.discharge_periods))
        if overlap and self.overlap_policy == "reject":
            raise ValueError(
                f"'smart_charging.charge_periods' and 'smart_charging.discharge_periods' share {', '.join(overlap)}; "
                "every step either charges or discharges (ADR 0002 A8); "
                "set overlap_policy = 'hold_target' with mode = 'fixed_target' to allow overlap"
            )


def check_overlap_policy(mode: str, policy: str) -> None:
    """Validate the overlap policy for a table or a directly constructed spec."""
    if policy not in OVERLAP_POLICIES:
        raise ValueError(f"'smart_charging.overlap_policy' must be one of: {', '.join(OVERLAP_POLICIES)}")
    if policy == "hold_target" and mode != "fixed_target":
        reason = (
            "the daily-persistence planner replaces grid targets but keeps reserves fixed, "
            "so planning and replay cannot hold the same target"
            if mode == "daily_persistence"
            else "this mode has no grid-charge target to hold"
        )
        raise ValueError(f"'smart_charging.overlap_policy' = 'hold_target' requires mode = 'fixed_target': {reason}")


def resolve_instructions(spec: SmartChargingSpec, tariff: ResolvedTariff | None) -> DispatchInstructions | None:
    """The dispatch instructions ``spec`` gives on ``tariff``'s index; None when disabled.

    Raises:
        ValueError: If fixed-target mode has no tariff, or names a period the
            tariff's schedule does not have.
    """
    if spec.mode == "disabled":
        return None
    if spec.mode in PLANNER_MODES:
        raise ValueError(
            f"smart_charging mode = '{spec.mode}' decides its instructions day by day while it runs; "
            "it has no static instructions"
        )
    if tariff is None:
        raise ValueError(f"smart_charging mode = '{spec.mode}' needs a resolved tariff")
    check_tariff_periods(spec, tariff)
    if spec.mode in DISCHARGE_ONLY_MODES:
        return discharge_layout(spec, tariff.period_labels)
    assert spec.target_usable_fraction is not None
    return period_layout(spec, tariff.period_labels, spec.target_usable_fraction)


def check_tariff_periods(spec: SmartChargingSpec, tariff: ResolvedTariff) -> None:
    """Raise ValueError if ``spec`` names a period that ``tariff``'s schedule does not have."""
    periods = set(tariff.schedule.periods)
    for name in ("charge_periods", "discharge_periods"):
        unknown = sorted(set(getattr(spec, name)) - periods)
        if unknown:
            raise ValueError(
                f"'smart_charging.{name}' has period(s) {', '.join(unknown)} that schedule "
                f"{tariff.schedule.identifier!r} does not have. Its periods: {', '.join(sorted(periods))}."
            )


def period_layout(spec: SmartChargingSpec, period_labels: Any, target_usable_fraction: float) -> DispatchInstructions:
    """The fixed-target instruction layout on ``period_labels``, charge steps targeting ``target_usable_fraction``.

    A discharge period may discharge; a charge period may grid-charge; the
    reserve is zero except on overlapping steps under ``hold_target``, where
    it equals the target. ``daily_persistence`` plans on the disjoint layout.
    """
    assert spec.grid_charge_efficiency is not None
    labels = np.asarray(period_labels, dtype=object)
    charge = np.isin(labels, spec.charge_periods)
    discharge = np.isin(labels, spec.discharge_periods)
    return DispatchInstructions(
        discharge_allowed=discharge,
        reserve_fraction=np.where(charge & discharge, target_usable_fraction, 0.0),
        grid_target_fraction=np.where(charge, target_usable_fraction, np.nan),
        grid_charge_efficiency=spec.grid_charge_efficiency,
        grid_import_limit_w=math.inf if spec.grid_import_limit_w is None else spec.grid_import_limit_w,
    )


def discharge_layout(spec: SmartChargingSpec, period_labels: Any) -> DispatchInstructions:
    """The ``discharge_only`` instructions on ``period_labels``: discharge in the discharge periods, never grid-charge.

    The reserve is zero and no step has a grid target, so the grid-charge
    efficiency and import limit take their no-op values and never act.
    Discharging in every period gives exactly :meth:`DispatchInstructions.noop`.
    """
    labels = np.asarray(period_labels, dtype=object)
    return DispatchInstructions(
        discharge_allowed=np.isin(labels, spec.discharge_periods),
        reserve_fraction=np.zeros(len(labels)),
        grid_target_fraction=np.full(len(labels), np.nan),
        grid_charge_efficiency=1.0,
        grid_import_limit_w=math.inf,
    )


@dataclass(frozen=True)
class FixedTargetDayController:
    """Static smart-charging instructions as a private daily controller (ADR 0002 A11).

    It runs ``fixed_target`` and ``discharge_only``, whose instructions are
    fixed by the tariff calendar.

    ``instructions`` are :func:`resolve_instructions`' output on the replayed
    tariff calendar. Each day's decision is their slice at the day's
    calendar positions, so a run through this controller dispatches exactly
    the static instructions. It reads the known tariff calendar only, and
    keeps no policy state.
    """

    instructions: DispatchInstructions
    tariff_horizon_days: int = 1

    def decide_day(self, day: ControllerDayInput, policy_state: object | None) -> ControllerDayDecision:
        positions = np.asarray(day.tariff.calendar_positions[: day.decision_step_count], dtype=np.intp)
        source = self.instructions
        return ControllerDayDecision(
            DispatchInstructions(
                discharge_allowed=source.discharge_allowed[positions],
                reserve_fraction=source.reserve_fraction[positions],
                grid_target_fraction=source.grid_target_fraction[positions],
                grid_charge_efficiency=source.grid_charge_efficiency,
                grid_import_limit_w=source.grid_import_limit_w,
            ),
            policy_state,
        )


def stored_energy_by_origin(energy_wh: float, pv_origin_wh: float, grid_origin_wh: float) -> dict[str, float]:
    """Split stored energy into its PV, grid and unattributed origins (ADR 0002 A8)."""
    return {
        "total_wh": float(energy_wh),
        "pv_origin_wh": float(pv_origin_wh),
        "grid_origin_wh": float(grid_origin_wh),
        "unattributed_wh": float(energy_wh - pv_origin_wh - grid_origin_wh),
    }


def smart_charging_provenance(
    spec: SmartChargingSpec, instructions: DispatchInstructions, tariff: ResolvedTariff
) -> dict[str, Any]:
    """A JSON-safe record of a smart-charging run for provenance.

    It pairs the configured parameters with the hash of the instructions they
    gave and the hash of the schedule they were resolved on.
    """
    if len(instructions) != len(tariff.period_labels):
        raise ValueError(
            f"The instructions have {len(instructions)} steps but the tariff has {len(tariff.period_labels)}"
        )
    return {
        "mode": spec.mode,
        "overlap_policy": spec.overlap_policy,
        "target_usable_fraction": spec.target_usable_fraction,
        "charge_periods": list(spec.charge_periods),
        "discharge_periods": list(spec.discharge_periods),
        "grid_charge_efficiency": spec.grid_charge_efficiency,
        # None, not math.inf: strict JSON has no infinity.
        "grid_import_limit_w": spec.grid_import_limit_w,
        "instruction_hash": instructions.instruction_hash(),
        "schedule_hash": tariff.schedule_hash,
        "terminal_convention": TERMINAL_CONVENTION,
    }
