"""Near shading of an array's plane-of-array irradiance from a shading table.

A layout tool that holds the 3D scene computes, for each array, the fraction
of the beam that nearby objects shade at each sun position, and the share of
the sky dome the array still sees. BREOS reads the table at each step's sun
position and reduces the array's irradiance before the incidence-angle
modifier (design record 0008).

The beam shaded fraction depends on the sun's direction and the geometry
only, so one table serves every weather year and resolution. Angles follow
pvlib: azimuth in degrees clockwise from north, elevation above the horizon.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

#: Transposition models whose sky diffuse has a circumsolar part pvlib can
#: separate. That part comes from around the sun's disc, so an object that
#: shades the beam shades it too.
CIRCUMSOLAR_MODELS = ("haydavies", "perez", "perez-driesse")

#: The keys of a ``pv_arrays[i].shading`` table.
SHADING_TABLE_KEYS = ("azimuth_deg", "elevation_deg", "beam_shaded_fraction", "sky_view_factor")


def _axis(values: Any, name: str, low: float, high: float) -> tuple[float, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise ValueError(f"'{name}' must be a non-empty list of numbers")
    axis: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
            raise ValueError(f"'{name}' must hold finite numbers")
        if not low <= float(value) <= high:
            raise ValueError(f"'{name}' values must be between {low:g} and {high:g}")
        axis.append(float(value))
    if any(b <= a for a, b in zip(axis, axis[1:], strict=False)):
        raise ValueError(f"'{name}' must increase")
    return tuple(axis)


@dataclass(frozen=True)
class ShadingTable:
    """One array's beam shaded fraction on a grid of sun positions, and its sky view factor.

    ``beam_shaded_fraction`` holds one row per ``elevation_deg`` value and one
    value per ``azimuth_deg`` value, each from 0 (fully lit) to 1 (fully
    shaded). Both axes take any increasing values. The table is read
    bilinearly. The azimuth axis is periodic when it closes the circle: when
    the gap from its last value round through north to its first is no wider
    than its widest step, the last column interpolates to the first across
    north. Otherwise, and on the elevation axis, the nearest edge holds
    outside the grid. An axis that runs from 0 to 360 must give the same
    column at both ends.

    ``sky_view_factor`` is the share of the isotropic sky diffuse the array
    still sees, from 0 to 1; ``sky_view_factor_stated`` records whether the
    caller gave it or it took the default 1.
    """

    azimuth_deg: tuple[float, ...]
    elevation_deg: tuple[float, ...]
    beam_shaded_fraction: tuple[tuple[float, ...], ...]
    sky_view_factor: float = 1.0
    sky_view_factor_stated: bool = False

    def __post_init__(self) -> None:
        azimuth = _axis(self.azimuth_deg, "azimuth_deg", 0.0, 360.0)
        elevation = _axis(self.elevation_deg, "elevation_deg", 0.0, 90.0)
        if azimuth[-1] - azimuth[0] > 360.0:
            raise ValueError("'azimuth_deg' must span at most 360 degrees")
        rows = self.beam_shaded_fraction
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or len(rows) != len(elevation):
            raise ValueError("'beam_shaded_fraction' must have one row per elevation_deg value")
        table: list[tuple[float, ...]] = []
        for i, row in enumerate(rows):
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != len(azimuth):
                raise ValueError(f"'beam_shaded_fraction[{i}]' must have one value per azimuth_deg value")
            values: list[float] = []
            for value in row:
                if isinstance(value, bool) or not isinstance(value, Real) or not 0.0 <= float(value) <= 1.0:
                    raise ValueError("'beam_shaded_fraction' values must be numbers from 0 to 1")
                values.append(float(value))
            table.append(tuple(values))
        if azimuth[-1] - azimuth[0] == 360.0 and any(row[0] != row[-1] for row in table):
            raise ValueError("'azimuth_deg' runs the full circle, so its first and last columns must match")
        view = self.sky_view_factor
        if isinstance(view, bool) or not isinstance(view, Real) or not 0.0 <= float(view) <= 1.0:
            raise ValueError("'sky_view_factor' must be a number from 0 to 1")
        object.__setattr__(self, "azimuth_deg", azimuth)
        object.__setattr__(self, "elevation_deg", elevation)
        object.__setattr__(self, "beam_shaded_fraction", tuple(table))
        object.__setattr__(self, "sky_view_factor", float(view))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ShadingTable:
        """Build a table from a ``pv_arrays[i].shading`` mapping."""
        unknown = sorted(set(value) - set(SHADING_TABLE_KEYS))
        if unknown:
            raise ValueError(f"unknown shading key(s) {unknown}; valid keys: {', '.join(SHADING_TABLE_KEYS)}")
        missing = [key for key in SHADING_TABLE_KEYS[:3] if key not in value]
        if missing:
            raise ValueError(f"a shading table needs {', '.join(missing)}")
        stated = "sky_view_factor" in value
        return cls(
            azimuth_deg=value["azimuth_deg"],
            elevation_deg=value["elevation_deg"],
            beam_shaded_fraction=value["beam_shaded_fraction"],
            sky_view_factor=value["sky_view_factor"] if stated else 1.0,
            sky_view_factor_stated=stated,
        )

    @property
    def periodic(self) -> bool:
        """Whether the azimuth axis closes the circle, so it interpolates across north."""
        azimuth = self.azimuth_deg
        if len(azimuth) < 2:
            return False
        wrap = azimuth[0] + 360.0 - azimuth[-1]
        return wrap <= max(b - a for a, b in zip(azimuth, azimuth[1:], strict=False))

    def beam_fraction(self, sun_azimuth: Any, sun_elevation: Any) -> np.ndarray:
        """The beam shaded fraction at each sun position, bilinearly from the table.

        The sun on or below the horizon reads the lowest row; it carries no
        beam there, so the value does not matter.
        """
        table = np.asarray(self.beam_shaded_fraction, dtype=float)
        azimuth = np.asarray(self.azimuth_deg, dtype=float)
        sun_az = np.mod(np.nan_to_num(np.asarray(sun_azimuth, dtype=float), nan=0.0), 360.0)
        sun_el = np.nan_to_num(np.asarray(sun_elevation, dtype=float), nan=0.0)
        axis = azimuth
        if self.periodic:
            if azimuth[-1] - azimuth[0] < 360.0:
                # Extend the axis one step round each side of north.
                axis = np.concatenate(([azimuth[-1] - 360.0], azimuth, [azimuth[0] + 360.0]))
                table = np.concatenate((table[:, -1:], table, table[:, :1]), axis=1)
            # Read the sun's azimuth in the axis's own window.
            sun_az = np.where(sun_az < axis[0], sun_az + 360.0, sun_az)
            sun_az = np.where(sun_az > axis[-1], sun_az - 360.0, sun_az)
        columns = _weights(axis, sun_az)
        rows = _weights(np.asarray(self.elevation_deg, dtype=float), sun_el)
        (a0, a1, wa), (e0, e1, we) = columns, rows
        low = table[e0, a0] * (1.0 - wa) + table[e0, a1] * wa
        high = table[e1, a0] * (1.0 - wa) + table[e1, a1] * wa
        return np.clip(low * (1.0 - we) + high * we, 0.0, 1.0)

    def as_dict(self) -> dict[str, Any]:
        """A JSON-ready summary of the table for the result."""
        return {
            "form": "table",
            "azimuth_points": len(self.azimuth_deg),
            "elevation_points": len(self.elevation_deg),
            "azimuth_periodic": self.periodic,
            "sky_view_factor": self.sky_view_factor,
            "sky_view_factor_stated": self.sky_view_factor_stated,
        }


def _weights(axis: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lower and upper grid indices and the upper weight of each ``x``; edges hold outside."""
    if axis.size == 1:
        zeros = np.zeros(x.shape, dtype=int)
        return zeros, zeros, np.zeros(x.shape)
    clipped = np.clip(x, axis[0], axis[-1])
    upper = np.clip(np.searchsorted(axis, clipped, side="right"), 1, axis.size - 1)
    lower = upper - 1
    weight = (clipped - axis[lower]) / (axis[upper] - axis[lower])
    return lower, upper, weight


def as_shading_table(value: Any) -> ShadingTable | None:
    """A :class:`ShadingTable` from a table, a mapping, or None."""
    if value is None or isinstance(value, ShadingTable):
        return value
    if isinstance(value, Mapping):
        return ShadingTable.from_mapping(value)
    raise TypeError("'shading' must be a table of azimuth_deg, elevation_deg and beam_shaded_fraction")
