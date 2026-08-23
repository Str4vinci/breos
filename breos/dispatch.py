"""Generic, index-aligned battery dispatch instructions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import pandas as pd


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


@dataclass(frozen=True)
class DispatchInstructions:
    """Immutable controller instructions aligned to a simulation index."""

    index: pd.DatetimeIndex
    discharge_allowed: tuple[bool, ...]
    minimum_usable_fraction: tuple[float, ...]
    grid_charge_target_usable_fraction: tuple[float | None, ...]
    grid_charge_efficiency: float
    grid_import_limit_w: float | None
    mode: str
    instruction_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.index, pd.DatetimeIndex):
            raise TypeError("'instructions.index' must be a pandas DatetimeIndex")
        if self.index.tz is None:
            raise ValueError("'instructions.index' must be timezone-aware")
        object.__setattr__(self, "index", self.index.copy())
        object.__setattr__(self, "discharge_allowed", tuple(self.discharge_allowed))
        object.__setattr__(self, "minimum_usable_fraction", tuple(self.minimum_usable_fraction))
        object.__setattr__(
            self,
            "grid_charge_target_usable_fraction",
            tuple(self.grid_charge_target_usable_fraction),
        )
        lengths = {
            len(self.index),
            len(self.discharge_allowed),
            len(self.minimum_usable_fraction),
            len(self.grid_charge_target_usable_fraction),
        }
        if len(lengths) != 1:
            raise ValueError("Dispatch instruction arrays must have the same length as their index")
        if any(not isinstance(value, bool) for value in self.discharge_allowed):
            raise TypeError("'discharge_allowed' values must be booleans")
        for value in self.minimum_usable_fraction:
            _fraction(value, "minimum_usable_fraction")
        for value in self.grid_charge_target_usable_fraction:
            if value is not None:
                _fraction(value, "grid_charge_target_usable_fraction")
        if any(
            allowed and target is not None
            for allowed, target in zip(self.discharge_allowed, self.grid_charge_target_usable_fraction)
        ):
            raise ValueError("Dispatch instructions cannot allow discharge and grid charging in the same step")
        _fraction(self.grid_charge_efficiency, "grid_charge_efficiency")
        if self.grid_charge_efficiency == 0.0:
            raise ValueError("'grid_charge_efficiency' must be greater than 0")
        if self.grid_import_limit_w is not None:
            _positive(self.grid_import_limit_w, "grid_import_limit_w")

    def validate_index(self, index: pd.DatetimeIndex) -> None:
        """Reject a simulation index that differs from the resolved instructions."""
        if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
            raise ValueError("Dispatch instructions require a timezone-aware simulation index")
        expected = self.index.tz_convert("UTC").as_unit("ns")
        actual = index.tz_convert("UTC").as_unit("ns")
        if not expected.equals(actual):
            raise ValueError("Dispatch instruction index does not match the simulation index")
