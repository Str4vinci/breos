"""Per-step instructions for the canonical dispatch step (ADR 0002).

A controller, such as fixed-target smart charging in
:mod:`breos.smart_charging`, turns a resolved tariff into aligned arrays: may
the battery discharge, how much usable energy it keeps before it does, and
how full the grid may charge it. The arrays are inputs to the dispatch step,
not energy flows: the step still decides every flow from the current
capacity window, the power limits, load, PV and the inverter.

The module does not import pandas, so the compiled dispatch path can take the
arrays as they are.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral, Real
from typing import Any

import numpy as np

_HASH_SCHEMA = b"breos-dispatch-instructions-v1"


def _frozen_array(value: Any, dtype: type, name: str) -> np.ndarray:
    try:
        array: np.ndarray = np.array(value, dtype=dtype, order="C", copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"'{name}' must be a 1-D array of {np.dtype(dtype).name}") from exc
    if array.ndim != 1:
        raise ValueError(f"'{name}' must be 1-D, got shape {array.shape}")
    return array


def _scalar(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"'{name}' must be a number")
    return float(value)


@dataclass(frozen=True)
class DispatchInstructions:
    """Per-step instructions for the canonical dispatch step (ADR 0002)."""

    discharge_allowed: np.ndarray  # bool, shape (n,)
    reserve_fraction: np.ndarray  # float64 in [0, 1]; usable fraction kept before discharge
    grid_target_fraction: np.ndarray  # float64 in [0, 1]; NaN = no grid charge on that step
    grid_charge_efficiency: float  # in (0, 1]; AC-to-DC, applied before the existing charge efficiency (A6)
    grid_import_limit_w: float  # > 0; math.inf = no site limit

    def __post_init__(self) -> None:
        raw = np.asarray(self.discharge_allowed)
        if raw.dtype != np.bool_:
            raise TypeError(f"'discharge_allowed' must be a boolean array, got dtype {raw.dtype}")
        discharge = _frozen_array(raw, np.bool_, "discharge_allowed")
        # Adding 0.0 turns -0.0 into 0.0, so equal instructions hash equally.
        reserve = _frozen_array(self.reserve_fraction, np.float64, "reserve_fraction") + 0.0
        target = _frozen_array(self.grid_target_fraction, np.float64, "grid_target_fraction") + 0.0

        n = len(discharge)
        for name, array in (("reserve_fraction", reserve), ("grid_target_fraction", target)):
            if len(array) != n:
                raise ValueError(f"'{name}' has {len(array)} steps but 'discharge_allowed' has {n}")
        if not np.isfinite(reserve).all():
            raise ValueError("'reserve_fraction' must be finite")
        if ((reserve < 0.0) | (reserve > 1.0)).any():
            raise ValueError("'reserve_fraction' must be between 0 and 1")
        idle = np.isnan(target)
        # One NaN bit pattern, so the hash does not see how a NaN was made.
        target[idle] = np.nan
        active = target[~idle]
        if not np.isfinite(active).all():
            raise ValueError("'grid_target_fraction' must be finite, or NaN for no grid charge")
        if ((active < 0.0) | (active > 1.0)).any():
            raise ValueError("'grid_target_fraction' must be between 0 and 1, or NaN for no grid charge")
        # ADR 0002 A8: a step either charges from the grid or may discharge,
        # which keeps the step's single origin fraction exact.
        both = np.flatnonzero(discharge & ~idle)
        if both.size:
            raise ValueError(
                f"Step {int(both[0])} both allows discharge and has a grid-charge target; "
                "every step either charges or discharges (ADR 0002 A8)"
            )

        efficiency = _scalar(self.grid_charge_efficiency, "grid_charge_efficiency")
        if not (math.isfinite(efficiency) and 0.0 < efficiency <= 1.0):
            raise ValueError("'grid_charge_efficiency' must be between 0 (exclusive) and 1 (inclusive)")
        limit = _scalar(self.grid_import_limit_w, "grid_import_limit_w")
        if math.isnan(limit) or limit <= 0.0:
            raise ValueError("'grid_import_limit_w' must be greater than 0, or math.inf for no site limit")

        for array in (discharge, reserve, target):
            array.setflags(write=False)
        object.__setattr__(self, "discharge_allowed", discharge)
        object.__setattr__(self, "reserve_fraction", reserve)
        object.__setattr__(self, "grid_target_fraction", target)
        object.__setattr__(self, "grid_charge_efficiency", efficiency)
        object.__setattr__(self, "grid_import_limit_w", limit)

    @classmethod
    def noop(cls, n: int) -> DispatchInstructions:
        """Instructions that change nothing: greedy self-consumption dispatch for ``n`` steps.

        The set is immutable, so one per length is built and then shared:
        every greedy run asks for one, and Monte Carlo and the optimizer make
        many runs of the same length.
        """
        if isinstance(n, bool) or not isinstance(n, Integral) or n < 0:
            raise ValueError("'n' must be a non-negative integer")
        if cls is DispatchInstructions:
            return _shared_noop(int(n))
        return cls._build_noop(int(n))

    @classmethod
    def _build_noop(cls, n: int) -> DispatchInstructions:
        return cls(
            discharge_allowed=np.ones(n, dtype=np.bool_),
            reserve_fraction=np.zeros(n),
            grid_target_fraction=np.full(n, np.nan),
            grid_charge_efficiency=1.0,
            grid_import_limit_w=math.inf,
        )

    def __len__(self) -> int:
        return len(self.discharge_allowed)

    def instruction_hash(self) -> str:
        """A sha256 of every array and scalar, the same on every platform and run."""
        digest = hashlib.sha256(_HASH_SCHEMA)
        digest.update(struct.pack("<q", len(self)))
        digest.update(self.discharge_allowed.astype(np.uint8).tobytes())
        digest.update(self.reserve_fraction.astype("<f8").tobytes())
        digest.update(self.grid_target_fraction.astype("<f8").tobytes())
        digest.update(struct.pack("<dd", self.grid_charge_efficiency, self.grid_import_limit_w))
        return digest.hexdigest()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DispatchInstructions):
            return NotImplemented
        return self.instruction_hash() == other.instruction_hash()

    def __hash__(self) -> int:
        return hash(self.instruction_hash())

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Rebuild through ``__init__``: a pickled array comes back writeable."""
        return type(self), (
            self.discharge_allowed,
            self.reserve_fraction,
            self.grid_target_fraction,
            self.grid_charge_efficiency,
            self.grid_import_limit_w,
        )


@lru_cache(maxsize=8)
def _shared_noop(n: int) -> DispatchInstructions:
    return DispatchInstructions._build_noop(n)
