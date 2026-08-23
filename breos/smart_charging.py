"""Tariff-aligned battery dispatch instructions.

Controllers in this module decide when the battery may discharge and when it
may charge from the grid. The battery simulator remains responsible for every
physical flow, limit, loss, and state transition.
"""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Real
from typing import Sequence

from breos.dispatch import DispatchInstructions
from breos.tariffs import ResolvedTariff


def _fraction(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"'{where}' must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"'{where}' must be a finite number")
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"'{where}' must be between 0 and 1")
    return result


def _positive(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"'{where}' must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"'{where}' must be a finite number")
    if result <= 0.0:
        raise ValueError(f"'{where}' must be greater than 0")
    return result


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _period_set(values: Sequence[str], known: set[str], where: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"'{where}' must be a list of tariff period names")
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise TypeError(f"'{where}' must contain non-empty tariff period names")
        period = value.strip()
        if period not in known:
            available = ", ".join(sorted(known))
            raise ValueError(f"Unknown {where} period {period!r}. Available: {available}")
        if period not in result:
            result.append(period)
    if not result:
        raise ValueError(f"'{where}' must contain at least one tariff period")
    return tuple(result)


def resolve_fixed_target_instructions(
    tariff: ResolvedTariff,
    *,
    target_usable_fraction: float,
    charge_periods: Sequence[str],
    discharge_periods: Sequence[str],
    grid_charge_efficiency: float = 0.95,
    grid_import_limit_w: float | None = None,
) -> DispatchInstructions:
    """Resolve fixed-target smart charging against tariff period labels."""
    target = _fraction(target_usable_fraction, "target_usable_fraction")
    efficiency = _fraction(grid_charge_efficiency, "grid_charge_efficiency")
    if efficiency == 0.0:
        raise ValueError("'grid_charge_efficiency' must be greater than 0")
    limit = _positive(grid_import_limit_w, "grid_import_limit_w") if grid_import_limit_w is not None else None

    known = set(tariff.schedule.periods)
    charge = _period_set(charge_periods, known, "charge_periods")
    discharge = _period_set(discharge_periods, known, "discharge_periods")
    overlap = set(charge) & set(discharge)
    if overlap:
        raise ValueError(f"Smart-charging charge and discharge periods overlap: {', '.join(sorted(overlap))}")

    discharge_allowed = tuple(label in discharge for label in tariff.period_labels)
    minimum_usable_fraction = (0.0,) * len(tariff.index)
    grid_targets = tuple(target if label in charge else None for label in tariff.period_labels)
    instruction_hash = _canonical_hash(
        {
            "mode": "fixed_target",
            "schedule_hash": tariff.schedule_hash,
            "discharge_allowed": discharge_allowed,
            "minimum_usable_fraction": minimum_usable_fraction,
            "grid_charge_target_usable_fraction": grid_targets,
            "grid_charge_efficiency": efficiency,
            "grid_import_limit_w": limit,
        }
    )
    return DispatchInstructions(
        index=tariff.index.copy(),
        discharge_allowed=discharge_allowed,
        minimum_usable_fraction=minimum_usable_fraction,
        grid_charge_target_usable_fraction=grid_targets,
        grid_charge_efficiency=efficiency,
        grid_import_limit_w=limit,
        mode="fixed_target",
        instruction_hash=instruction_hash,
    )
