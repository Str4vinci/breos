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
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from breos.dispatch_instructions import DispatchInstructions
from breos.tariffs import ResolvedTariff

SMART_CHARGING_MODES = ("disabled", "fixed_target")
# Normal App runs carry stored energy, origin shares and degradation from one
# project year into the next (ADR 0002, boundary and terminal conventions).
TERMINAL_CONVENTION = "physical_carry"


@dataclass(frozen=True)
class SmartChargingSpec:
    """A configured ``[smart_charging]`` table before it meets a tariff.

    ``grid_import_limit_w`` caps total site import, load included; None is no
    limit. With ``mode = "disabled"`` the other fields keep their defaults.
    The App checks the table (``breos.app_config.SMART_CHARGING_TABLE``);
    this value only keeps a direct construction coherent.
    """

    mode: str
    target_usable_fraction: float | None = None
    charge_periods: tuple[str, ...] = ()
    discharge_periods: tuple[str, ...] = ()
    grid_charge_efficiency: float | None = None
    grid_import_limit_w: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in SMART_CHARGING_MODES:
            raise ValueError(f"'smart_charging.mode' must be one of: {', '.join(SMART_CHARGING_MODES)}")
        object.__setattr__(self, "charge_periods", tuple(self.charge_periods))
        object.__setattr__(self, "discharge_periods", tuple(self.discharge_periods))
        if self.mode == "disabled":
            settings = (
                self.target_usable_fraction,
                self.charge_periods,
                self.discharge_periods,
                self.grid_charge_efficiency,
                self.grid_import_limit_w,
            )
            if any(value not in (None, ()) for value in settings):
                raise ValueError("'smart_charging' with mode = 'disabled' takes no other settings")
            return
        missing = [
            name
            for name in ("target_usable_fraction", "charge_periods", "discharge_periods", "grid_charge_efficiency")
            if getattr(self, name) in (None, ())
        ]
        if missing:
            raise ValueError(
                f"'smart_charging' needs {', '.join(f'smart_charging.{name}' for name in missing)} "
                "for mode = 'fixed_target'"
            )
        overlap = sorted(set(self.charge_periods) & set(self.discharge_periods))
        if overlap:
            raise ValueError(
                f"'smart_charging.charge_periods' and 'smart_charging.discharge_periods' share {', '.join(overlap)}; "
                "every step either charges or discharges (ADR 0002 A8)"
            )


def resolve_instructions(spec: SmartChargingSpec, tariff: ResolvedTariff | None) -> DispatchInstructions | None:
    """The dispatch instructions ``spec`` gives on ``tariff``'s index; None when disabled.

    Raises:
        ValueError: If fixed-target mode has no tariff, or names a period the
            tariff's schedule does not have.
    """
    if spec.mode == "disabled":
        return None
    if tariff is None:
        raise ValueError("smart_charging mode = 'fixed_target' needs a resolved tariff")
    periods = set(tariff.schedule.periods)
    for name in ("charge_periods", "discharge_periods"):
        unknown = sorted(set(getattr(spec, name)) - periods)
        if unknown:
            raise ValueError(
                f"'smart_charging.{name}' has period(s) {', '.join(unknown)} that schedule "
                f"{tariff.schedule.identifier!r} does not have. Its periods: {', '.join(sorted(periods))}."
            )
    assert spec.target_usable_fraction is not None and spec.grid_charge_efficiency is not None
    labels = np.asarray(tariff.period_labels, dtype=object)
    charge = np.isin(labels, spec.charge_periods)
    return DispatchInstructions(
        discharge_allowed=np.isin(labels, spec.discharge_periods),
        reserve_fraction=np.zeros(len(labels)),
        grid_target_fraction=np.where(charge, spec.target_usable_fraction, np.nan),
        grid_charge_efficiency=spec.grid_charge_efficiency,
        grid_import_limit_w=math.inf if spec.grid_import_limit_w is None else spec.grid_import_limit_w,
    )


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
