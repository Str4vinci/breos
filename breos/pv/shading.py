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
        weights = self.grid_weights(sun_azimuth, sun_elevation)
        return read_grid(np.asarray(self.beam_shaded_fraction, dtype=float), weights)

    def grid_weights(self, sun_azimuth: Any, sun_elevation: Any) -> GridWeights:
        """The table cells and bilinear weights of each sun position, for any table on these axes."""
        azimuth = np.asarray(self.azimuth_deg, dtype=float)
        sun_az = np.mod(np.nan_to_num(np.asarray(sun_azimuth, dtype=float), nan=0.0), 360.0)
        sun_el = np.nan_to_num(np.asarray(sun_elevation, dtype=float), nan=0.0)
        axis = azimuth
        columns = np.arange(azimuth.size)
        if self.periodic:
            if azimuth[-1] - azimuth[0] < 360.0:
                # Extend the axis one step round each side of north.
                axis = np.concatenate(([azimuth[-1] - 360.0], azimuth, [azimuth[0] + 360.0]))
                columns = np.concatenate(([azimuth.size - 1], columns, [0]))
            # Read the sun's azimuth in the axis's own window.
            sun_az = np.where(sun_az < axis[0], sun_az + 360.0, sun_az)
            sun_az = np.where(sun_az > axis[-1], sun_az - 360.0, sun_az)
        a0, a1, wa = _weights(axis, sun_az)
        e0, e1, we = _weights(np.asarray(self.elevation_deg, dtype=float), sun_el)
        return GridWeights(e0, e1, we, columns[a0], columns[a1], wa)

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


@dataclass(frozen=True)
class GridWeights:
    """Each sun position's two rows and two columns of a table, and the weight of the second of each."""

    row0: np.ndarray
    row1: np.ndarray
    row_weight: np.ndarray
    column0: np.ndarray
    column1: np.ndarray
    column_weight: np.ndarray


def read_grid(tables: np.ndarray, weights: GridWeights) -> np.ndarray:
    """Bilinear values of one table, or a stack of tables on the same axes, at each sun position.

    ``tables`` has the elevation rows and azimuth columns as its last two
    axes; the result has its leading axes followed by the sun positions.
    """
    w = weights
    low = tables[..., w.row0, w.column0] * (1.0 - w.column_weight) + tables[..., w.row0, w.column1] * w.column_weight
    high = tables[..., w.row1, w.column0] * (1.0 - w.column_weight) + tables[..., w.row1, w.column1] * w.column_weight
    return np.clip(low * (1.0 - w.row_weight) + high * w.row_weight, 0.0, 1.0)


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


#: Bypass diodes per module when ``module_shading`` does not state them.
DEFAULT_BYPASS_DIODES = 3

#: The keys of a ``pv_strings[i].module_shading`` table, of each of its
#: ``modules`` and of each module's ``substrings``.
MODULE_SHADING_KEYS = ("azimuth_deg", "elevation_deg", "bypass_diodes", "modules")
MODULE_ENTRY_KEYS = ("beam_shaded_fraction", "substrings", "sky_view_factor")
SUBSTRING_KEYS = ("beam_shaded_fraction", "limiting_shaded_fraction")


def _keys(value: Any, allowed: Sequence[str], where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"'{where}' must be a table")
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise ValueError(f"unknown key(s) {unknown} in '{where}'; valid keys: {', '.join(allowed)}")
    return value


@dataclass(frozen=True)
class ModuleShade:
    """One module's shading: per bypass-diode substring, its mean and limiting tables, and its sky view factor.

    ``mean`` holds each substring's area-mean beam shaded fraction table and
    ``limiting`` the shaded fraction of its most-shaded series cell, or None
    where the shade is even over the substring. ``whole`` records that the
    caller gave one table for the whole module.
    """

    mean: tuple[ShadingTable, ...]
    limiting: tuple[ShadingTable | None, ...]
    sky_view_factor: float = 1.0
    whole: bool = False
    sky_view_factor_stated: bool = False


@dataclass(frozen=True)
class ModuleShading:
    """The shading of each module of one string, in series order, on one grid of sun positions.

    Each module's tables follow :class:`ShadingTable`'s rules on the shared
    axes. ``bypass_diodes`` is the number of bypass-diode substrings per
    module; ``bypass_diodes_stated`` whether the caller gave it.
    """

    modules: tuple[ModuleShade, ...]
    bypass_diodes: int = DEFAULT_BYPASS_DIODES
    bypass_diodes_stated: bool = False

    @classmethod
    def from_mapping(cls, value: Any) -> ModuleShading:
        """Build the shading from a ``pv_strings[i].module_shading`` mapping."""
        table = _keys(value, MODULE_SHADING_KEYS, "module_shading")
        missing = [key for key in ("azimuth_deg", "elevation_deg", "modules") if key not in table]
        if missing:
            raise ValueError(f"module_shading needs {', '.join(missing)}")
        stated = "bypass_diodes" in table
        diodes = table.get("bypass_diodes", DEFAULT_BYPASS_DIODES)
        if isinstance(diodes, bool) or not isinstance(diodes, int) or diodes < 1:
            raise ValueError("'module_shading.bypass_diodes' must be an integer of at least 1")
        entries = table["modules"]
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)) or not entries:
            raise ValueError("'module_shading.modules' must be a non-empty list, one entry per module")

        def grid(values: Any, where: str) -> ShadingTable:
            try:
                return ShadingTable(table["azimuth_deg"], table["elevation_deg"], values)
            except ValueError as error:
                raise ValueError(f"'{where}': {error}") from None

        modules = []
        for m, raw in enumerate(entries):
            where = f"module_shading.modules[{m}]"
            entry = _keys(raw, MODULE_ENTRY_KEYS, where)
            if ("beam_shaded_fraction" in entry) == ("substrings" in entry):
                raise ValueError(f"'{where}' needs either beam_shaded_fraction or substrings")
            view = entry.get("sky_view_factor", 1.0)
            if isinstance(view, bool) or not isinstance(view, Real) or not 0.0 <= float(view) <= 1.0:
                raise ValueError(f"'{where}.sky_view_factor' must be a number from 0 to 1")
            if "beam_shaded_fraction" in entry:
                whole = grid(entry["beam_shaded_fraction"], f"{where}.beam_shaded_fraction")
                modules.append(
                    ModuleShade((whole,) * diodes, (None,) * diodes, float(view), True, "sky_view_factor" in entry)
                )
                continue
            substrings = entry["substrings"]
            if (
                not isinstance(substrings, Sequence)
                or isinstance(substrings, (str, bytes))
                or len(substrings) != diodes
            ):
                raise ValueError(f"'{where}.substrings' must hold one table per bypass diode ({diodes})")
            means, limits = [], []
            for k, raw_sub in enumerate(substrings):
                sub_where = f"{where}.substrings[{k}]"
                sub = _keys(raw_sub, SUBSTRING_KEYS, sub_where)
                if "beam_shaded_fraction" not in sub:
                    raise ValueError(f"'{sub_where}' needs beam_shaded_fraction")
                means.append(grid(sub["beam_shaded_fraction"], f"{sub_where}.beam_shaded_fraction"))
                limit = sub.get("limiting_shaded_fraction")
                limits.append(None if limit is None else grid(limit, f"{sub_where}.limiting_shaded_fraction"))
            modules.append(ModuleShade(tuple(means), tuple(limits), float(view), False, "sky_view_factor" in entry))
        return cls(tuple(modules), int(diodes), stated)

    def inconsistency(self, cells: int) -> str | None:
        """Why these tables cannot describe a module of ``cells`` series cells, or None when they can.

        A substring holds ``cells / bypass_diodes`` cells, at least one, and
        at least two when a limiting fraction is given; its most-shaded cell alone shades
        that share of its area: its mean shaded fraction is at least its
        limiting fraction over its cell count, at every node.
        """
        per_substring = cells / self.bypass_diodes
        if per_substring < 1:
            return f"bypass_diodes ({self.bypass_diodes}) is more than the module's {cells} cells"
        for m, module in enumerate(self.modules):
            for k, (mean, limit) in enumerate(zip(module.mean, module.limiting, strict=True)):
                if limit is None:
                    continue
                where = f"modules[{m}].substrings[{k}]"
                if per_substring < 2:
                    return (
                        f"{where} gives a limiting_shaded_fraction, but {cells} cells over "
                        f"{self.bypass_diodes} bypass diodes leave fewer than 2 cells per substring"
                    )
                low = np.asarray(limit.beam_shaded_fraction) / per_substring - np.asarray(mean.beam_shaded_fraction)
                if (low > 1e-9).any():
                    return (
                        f"{where}: its beam_shaded_fraction is below limiting_shaded_fraction / "
                        f"{per_substring:g} at some node, so its most-shaded cell would shade more than its area "
                        "mean allows; take the mean over the same cells as the limiting value"
                    )
        return None

    @property
    def axes(self) -> ShadingTable:
        """A table on the string's axes, for its grid weights."""
        return self.modules[0].mean[0]

    def substring_fractions(self, weights: GridWeights) -> tuple[np.ndarray, np.ndarray]:
        """Each substring's mean and limiting beam shaded fraction at each sun position.

        Both have shape ``(modules, bypass_diodes, steps)``; the limiting
        fraction is never below the mean.
        """
        mean = read_grid(
            np.asarray([[t.beam_shaded_fraction for t in module.mean] for module in self.modules], dtype=float),
            weights,
        )
        limiting = mean.copy()
        for m, module in enumerate(self.modules):
            for k, table in enumerate(module.limiting):
                if table is not None:
                    limiting[m, k] = np.maximum(read_grid(np.asarray(table.beam_shaded_fraction), weights), mean[m, k])
        return mean, limiting

    @property
    def sky_view_factors(self) -> np.ndarray:
        """Each module's sky view factor."""
        return np.asarray([module.sky_view_factor for module in self.modules], dtype=float)

    def as_config(self) -> dict[str, Any]:
        """The shading as a plain ``module_shading`` mapping of floats, as the result echoes it."""
        axes = self.axes
        config: dict[str, Any] = {"azimuth_deg": list(axes.azimuth_deg), "elevation_deg": list(axes.elevation_deg)}
        if self.bypass_diodes_stated:
            config["bypass_diodes"] = self.bypass_diodes
        modules = []
        for module in self.modules:
            if module.whole:
                entry: dict[str, Any] = {
                    "beam_shaded_fraction": [list(row) for row in module.mean[0].beam_shaded_fraction]
                }
            else:
                substrings = []
                for mean, limit in zip(module.mean, module.limiting, strict=True):
                    sub: dict[str, Any] = {"beam_shaded_fraction": [list(row) for row in mean.beam_shaded_fraction]}
                    if limit is not None:
                        sub["limiting_shaded_fraction"] = [list(row) for row in limit.beam_shaded_fraction]
                    substrings.append(sub)
                entry = {"substrings": substrings}
            if module.sky_view_factor_stated:
                entry["sky_view_factor"] = module.sky_view_factor
            modules.append(entry)
        config["modules"] = modules
        return config

    def as_dict(self) -> dict[str, Any]:
        """A JSON-ready summary for the result."""
        axes = self.axes
        return {
            "azimuth_points": len(axes.azimuth_deg),
            "elevation_points": len(axes.elevation_deg),
            "azimuth_periodic": axes.periodic,
            "bypass_diodes": self.bypass_diodes,
            "bypass_diodes_stated": self.bypass_diodes_stated,
            "modules_by_substring": sum(not module.whole for module in self.modules),
            "modules_with_limiting_cells": sum(
                any(table is not None for table in module.limiting) for module in self.modules
            ),
        }


@dataclass(frozen=True)
class ModuleMeanShading:
    """An array's shading as the mean of its strings' module shading, for the array's irradiance stage.

    The irradiance stage is linear in the shaded fraction and the sky view
    factor, so the array's irradiance at their means is the mean of its
    modules' irradiance. Each string counts once per module.
    """

    strings: tuple[ModuleShading, ...]

    @property
    def sky_view_factor(self) -> float:
        factors = np.concatenate([shading.sky_view_factors for shading in self.strings])
        return float(factors.mean())

    @property
    def sky_view_factor_stated(self) -> bool:
        return any(module.sky_view_factor_stated for shading in self.strings for module in shading.modules)

    def beam_fraction(self, sun_azimuth: Any, sun_elevation: Any) -> np.ndarray:
        total = None
        count = 0
        for shading in self.strings:
            mean, _ = shading.substring_fractions(shading.axes.grid_weights(sun_azimuth, sun_elevation))
            part = mean.sum(axis=(0, 1)) / shading.bypass_diodes
            total = part if total is None else total + part
            count += len(shading.modules)
        assert total is not None
        return total / count

    def as_dict(self) -> dict[str, Any]:
        """A JSON-ready summary for the result."""
        return {
            "form": "modules",
            "strings": len(self.strings),
            "modules": sum(len(shading.modules) for shading in self.strings),
            "sky_view_factor": self.sky_view_factor,
            "sky_view_factor_stated": self.sky_view_factor_stated,
        }


#: An array's shading: its own table, or the mean of its strings' modules.
ArrayShading = ShadingTable | ModuleMeanShading


def as_shading_table(value: Any) -> ArrayShading | None:
    """An array's shading from a table, a mapping, a module mean, or None."""
    if value is None or isinstance(value, (ShadingTable, ModuleMeanShading)):
        return value
    if isinstance(value, Mapping):
        return ShadingTable.from_mapping(value)
    raise TypeError("'shading' must be a table of azimuth_deg, elevation_deg and beam_shaded_fraction")
