"""An independent cell-level model of a 144 half-cell module under shade, for the module-shading tests.

The module has 6 columns and 24 rows. Each of its 3 substrings is two
columns: a top string of the upper 12 rows' 24 half-cells in parallel with a
bottom string of the lower 12 rows', under one bypass diode. A half-cell
takes the module's single-diode fit at 72 positions of 2 half-cells: half
the photocurrent and saturation current, twice the resistances over 72 and
the thermal voltage over 72, which reproduces the module's curve under even
light. A cell carries at most its own short-circuit current or, given a
breakdown voltage, holds at it beyond. Each string's voltage is the sum of
its cells' over a current grid; the two strings' currents add at each
voltage; the diode clamps the substring; the substrings and modules add in
series.

It shares nothing with ``breos.pv.mismatch`` but the module's single-diode
fit.
"""

from __future__ import annotations

import numpy as np
import pvlib

from breos.pv.mismatch import ShadedStringCurve
from breos.pv.shading import ModuleShading
from breos.pv_modules import PVModuleParams
from breos.solar import IrradianceParts, _get_cec_params

BEAM, DIFFUSE, TEMP = 800.0, 200.0, 25.0
BYPASS_V = 0.5
_VOLTAGE = np.linspace(-BYPASS_V, 30.0, 61001)


def table(value: float) -> list[list[float]]:
    return [[value]]


def half(mean: float, limiting: float | None = None) -> dict:
    """One string's shading on a single-node grid."""
    entry = {"beam_shaded_fraction": table(mean)}
    if limiting is not None:
        entry["limiting_shaded_fraction"] = table(limiting)
    return entry


def parts(module: PVModuleParams) -> IrradianceParts:
    one = np.ones(1)
    return IrradianceParts(
        beam=BEAM * one, circumsolar=0.0 * one, sky=DIFFUSE * one, other=0.0 * one, temp_cell=TEMP * one,
        sun_azimuth=180.0 * one, sun_elevation=50.0 * one, pv_params=module,
    )  # fmt: skip


def breos_power(module: PVModuleParams, entries: list[dict]) -> float:
    """BREOS's maximum power of a string of modules with these ``module_shading`` entries."""
    shading = ModuleShading.from_mapping({"azimuth_deg": [0], "elevation_deg": [0], "modules": entries})
    bound = shading.for_module(module.bypass_diodes, module.cell_strings_in_parallel)
    return float(ShadedStringCurve(bound, parts(module), np.ones(1), len(entries)).mpp_bare[0])


def breos_entry(shaded: np.ndarray) -> dict:
    """A module's ``module_shading`` entry: each string's mean and most-shaded cell."""
    substrings = []
    for s in range(3):
        fraction = shaded[:, 2 * s : 2 * s + 2].astype(float)
        halves = [half(float(p.mean()), float(p.max())) for p in (fraction[:12], fraction[12:])]
        substrings.append({"halves": halves})
    return {"substrings": substrings}


def portrait(rows: int, columns: list[int]) -> np.ndarray:
    shaded = np.zeros((24, 6), dtype=bool)
    shaded[24 - rows :, columns] = True
    return shaded


def landscape(rows: int) -> np.ndarray:
    # On its side, the module's long edge is its first columns.
    shaded = np.zeros((24, 6), dtype=bool)
    shaded[:, :rows] = True
    return shaded


class Reference:
    """The cell-level model of one module type; a shaded cell's beam is fully blocked."""

    def __init__(self, module: PVModuleParams, breakdown_v: float | None = None) -> None:
        self.module = module
        self.breakdown_v = breakdown_v
        self.current = np.linspace(0.0, 1.02 * module.Isc, 20001)
        self._fit = _get_cec_params(module)

    def _half_cell(self, irradiance: float) -> tuple[np.ndarray, float]:
        il_ref, io_ref, rs, rsh_ref, a_ref, adjust = self._fit
        il, io, rs_t, rsh, nv = (
            float(np.asarray(x))
            for x in pvlib.pvsystem.calcparams_cec(
                irradiance, TEMP, self.module.alpha_sc, a_ref, il_ref, io_ref, rsh_ref, rs, adjust
            )
        )
        params = (il / 2, io / 2, rs_t * 2 / 72, rsh * 2 / 72, nv / 72)
        isc = float(pvlib.pvsystem.i_from_v(0.0, *params))
        current = self.current / 2
        voltage = np.asarray(pvlib.pvsystem.v_from_i(np.minimum(current, isc), *params), dtype=float)
        beyond = -np.inf if self.breakdown_v is None else self.breakdown_v
        return np.where(current > isc, beyond, voltage), isc

    def _string_current(self, irradiances: np.ndarray) -> np.ndarray:
        voltage = np.zeros(self.current.size)
        cap = np.inf
        for value in np.unique(irradiances):
            cell, isc = self._half_cell(float(value))
            voltage = voltage + np.count_nonzero(irradiances == value) * cell
            cap = min(cap, isc)
        half_current = self.current / 2
        if self.breakdown_v is not None:
            return np.interp(_VOLTAGE, voltage[::-1], half_current[::-1])
        finite = np.isfinite(voltage)
        current = np.interp(_VOLTAGE, voltage[finite][::-1], half_current[finite][::-1], left=cap, right=0.0)
        return np.minimum(current, cap)

    def module_voltage(self, shaded: np.ndarray) -> np.ndarray:
        """The module's voltage over :attr:`current`, -inf where it cannot carry it."""
        irradiance = np.where(shaded, DIFFUSE, BEAM + DIFFUSE)
        total = np.zeros(self.current.size)
        for s in range(3):
            columns = irradiance[:, 2 * s : 2 * s + 2]
            current = self._string_current(columns[:12].ravel()) + self._string_current(columns[12:].ravel())
            voltage = np.interp(self.current, current[::-1], _VOLTAGE[::-1], right=-np.inf)
            total += np.maximum(np.where(self.current > current[0], -np.inf, voltage), -BYPASS_V)
        return total

    def power(self, voltage: np.ndarray) -> float:
        return float(np.max(self.current * voltage))
