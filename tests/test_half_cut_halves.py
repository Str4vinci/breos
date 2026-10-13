"""Half-cut modules: each bypass-diode substring is two cell strings in parallel, each with its own shading."""

from dataclasses import replace

import numpy as np
import pvlib
import pytest

from breos.pv.mismatch import ShadedStringCurve
from breos.pv.shading import ModuleShading
from breos.pv_modules import get_module
from breos.solar import IrradianceParts, _get_cec_params

BEAM, DIFFUSE, TEMP = 800.0, 200.0, 25.0
MODULE = get_module("Generic_400W")
ONE = np.ones(1)
PARTS = IrradianceParts(
    beam=BEAM * ONE, circumsolar=0.0 * ONE, sky=DIFFUSE * ONE, other=0.0 * ONE, temp_cell=TEMP * ONE,
    sun_azimuth=180.0 * ONE, sun_elevation=50.0 * ONE, pv_params=MODULE,
)  # fmt: skip


def _table(value: float) -> list[list[float]]:
    return [[value]]


def _raw(modules: list[dict]) -> ModuleShading:
    return ModuleShading.from_mapping({"azimuth_deg": [0], "elevation_deg": [0], "modules": modules})


def _half(mean: float, limiting: float | None = None) -> dict:
    half = {"beam_shaded_fraction": _table(mean)}
    if limiting is not None:
        half["limiting_shaded_fraction"] = _table(limiting)
    return half


LIT = {"beam_shaded_fraction": _table(0.0)}


# -- the configuration ----------------------------------------------------------------------------


def test_halves_are_validated():
    with pytest.raises(ValueError, match="give its fractions in them only"):
        _raw([{"substrings": [{**_half(0.1), "halves": [_half(0.1), _half(0.1)]}, LIT, LIT]}])
    with pytest.raises(ValueError, match="one table per parallel cell string"):
        _raw([{"substrings": [{"halves": [_half(0.1)]}, LIT, LIT]}])
    with pytest.raises(ValueError, match="needs beam_shaded_fraction"):
        _raw([{"substrings": [{"halves": [_half(0.1), {"limiting_shaded_fraction": _table(0.5)}]}, LIT, LIT]}])
    with pytest.raises(ValueError, match="unknown key"):
        _raw([{"substrings": [{"halves": [_half(0.1), {**_half(0.1), "sky_view_factor": 1.0}]}, LIT, LIT]}])
    halved = _raw([{"substrings": [{"halves": [_half(0.1), _half(0.0)]}, LIT, LIT]}])
    with pytest.raises(ValueError, match=r"cell_strings_in_parallel = 1"):
        halved.for_module(3, 1)
    with pytest.raises(ValueError, match="holds 2 entries; the module has 3 cell strings"):
        halved.for_module(3, 3)


def test_a_half_whose_worst_cell_shades_more_than_its_mean_is_refused():
    # 144 cells over 2 strings and 3 diodes: 24 cells per string, so a fully
    # shaded cell alone shades 1/24 of the string's area.
    shading = _raw([{"substrings": [{"halves": [_half(0.01, 1.0), _half(0.0)]}, LIT, LIT]}]).for_module(3, 2)
    problem = shading.inconsistency(MODULE.N_Cells)
    assert problem is not None and "substrings[0].halves[0]" in problem and "24" in problem
    assert (
        _raw([{"substrings": [{"halves": [_half(1 / 24, 1.0), _half(0.0)]}, LIT, LIT]}])
        .for_module(3, 2)
        .inconsistency(MODULE.N_Cells)
        is None
    )


def test_the_substrings_mean_is_its_halves_mean_and_the_tables_round_trip():
    raw = {"substrings": [{"halves": [_half(0.2, 0.6), _half(0.0)]}, LIT, {**_half(0.1, 0.4)}]}
    shading = _raw([raw]).for_module(3, 2)
    assert shading.modules[0].mean[0].beam_shaded_fraction[0][0] == pytest.approx(0.1)
    echoed = shading.as_config()
    assert echoed["modules"][0] == raw
    assert _raw(echoed["modules"]).for_module(3, 2) == shading
    summary = shading.as_dict()
    assert summary["cell_strings_in_parallel"] == 2 and summary["substrings_by_halves"] == 1
    assert summary["modules_with_limiting_cells"] == 1
    only_halves = {"substrings": [{"halves": [_half(0.2, 0.6), _half(0.0)]}, LIT, LIT]}
    assert _raw([only_halves]).for_module(3, 2).as_dict()["modules_with_limiting_cells"] == 1


def test_cell_strings_must_divide_the_cells():
    with pytest.raises(ValueError, match="cell_strings_in_parallel"):
        replace(MODULE, cell_strings_in_parallel=5)
    assert MODULE.cell_strings_in_parallel == 2 and MODULE.cells_in_series == 72


# -- the curve ------------------------------------------------------------------------------------


def _power(modules: list[dict]) -> float:
    shading = _raw(modules).for_module(3, 2)
    return float(ShadedStringCurve(shading, PARTS, ONE, len(modules)).mpp_bare[0])


def test_halves_shaded_alike_are_the_substring_without_halves():
    alike = {"substrings": [{"halves": [_half(0.1, 0.5), _half(0.1, 0.5)]}, LIT, LIT]}
    plain = {"substrings": [_half(0.1, 0.5), LIT, LIT]}
    assert _power([alike, LIT]) == pytest.approx(_power([plain, LIT]), rel=1e-12)


def test_shade_on_one_half_costs_less_than_on_both():
    # A cell a fifth shaded in one string of the substring caps that string
    # only; the other still carries its full current. In both strings it caps
    # the substring.
    one = _power([{"substrings": [{"halves": [_half(0.2 / 24, 0.2), _half(0.0)]}, LIT, LIT]}] + [LIT] * 3)
    both = _power([{"substrings": [_half(0.2 / 24, 0.2), LIT, LIT]}] + [LIT] * 3)
    lit = _power([LIT] * 4)
    assert both < one < lit


# -- an independent cell-level reference ----------------------------------------------------------
#
# A 144 half-cell module of 6 columns and 24 rows. Each of its 3 substrings
# is two columns: a top string of the upper 12 rows' 24 half-cells in
# parallel with a bottom string of the lower 12 rows', under one bypass
# diode. A half-cell takes the module's single-diode fit at 72 positions of 2
# half-cells: half the photocurrent and saturation current, twice the
# resistances over 72 and the thermal voltage over 72, which reproduces the
# module's curve under even light. A cell carries at most its own
# short-circuit current. Each string's voltage is the sum of its cells' over
# a current grid; the two strings' currents add at each voltage; the diode
# clamps the substring; the substrings and modules add in series.

_CURRENT = np.linspace(0.0, 1.02 * MODULE.Isc, 20001)
_VOLTAGE = np.linspace(-0.5, 30.0, 61001)
_FIT = _get_cec_params(MODULE)


def _half_cell_voltage(irradiance: float) -> tuple[np.ndarray, float]:
    il_ref, io_ref, rs, rsh_ref, a_ref, adjust = _FIT
    il, io, rs_t, rsh, nv = (
        float(np.asarray(x))
        for x in pvlib.pvsystem.calcparams_cec(
            irradiance, TEMP, MODULE.alpha_sc, a_ref, il_ref, io_ref, rsh_ref, rs, adjust
        )
    )
    params = (il / 2, io / 2, rs_t * 2 / 72, rsh * 2 / 72, nv / 72)
    isc = float(pvlib.pvsystem.i_from_v(0.0, *params))
    current = _CURRENT / 2
    voltage = np.asarray(pvlib.pvsystem.v_from_i(np.minimum(current, isc), *params), dtype=float)
    return np.where(current > isc, -np.inf, voltage), isc


def _string_current(irradiances: np.ndarray) -> np.ndarray:
    voltage = np.zeros(_CURRENT.size)
    cap = np.inf
    for value in np.unique(irradiances):
        cell, isc = _half_cell_voltage(float(value))
        voltage = voltage + np.count_nonzero(irradiances == value) * cell
        cap = min(cap, isc)
    finite = np.isfinite(voltage)
    current = np.interp(_VOLTAGE, voltage[finite][::-1], (_CURRENT / 2)[finite][::-1], left=cap, right=0.0)
    return np.minimum(current, cap)


def _reference_module_voltage(shaded: np.ndarray) -> np.ndarray:
    irradiance = np.where(shaded, DIFFUSE, BEAM + DIFFUSE)
    total = np.zeros(_CURRENT.size)
    for s in range(3):
        columns = irradiance[:, 2 * s : 2 * s + 2]
        current = _string_current(columns[:12].ravel()) + _string_current(columns[12:].ravel())
        voltage = np.interp(_CURRENT, current[::-1], _VOLTAGE[::-1], right=-np.inf)
        total += np.maximum(np.where(_CURRENT > current[0], -np.inf, voltage), -0.5)
    return total


def _reference_power(voltage: np.ndarray) -> float:
    return float(np.max(_CURRENT * voltage))


def _breos_entry(shaded: np.ndarray) -> dict:
    substrings = []
    for s in range(3):
        fraction = shaded[:, 2 * s : 2 * s + 2].astype(float)
        halves = [_half(float(part.mean()), float(part.max())) for part in (fraction[:12], fraction[12:])]
        substrings.append({"halves": halves})
    return {"substrings": substrings}


def _portrait(rows: int, columns: list[int]) -> np.ndarray:
    shaded = np.zeros((24, 6), dtype=bool)
    shaded[24 - rows :, columns] = True
    return shaded


def _landscape(rows: int) -> np.ndarray:
    # On its side, the module's long edge is its first columns.
    shaded = np.zeros((24, 6), dtype=bool)
    shaded[:, :rows] = True
    return shaded


@pytest.mark.parametrize(
    "shaded",
    [
        pytest.param(_portrait(1, list(range(6))), id="portrait-bottom-row"),
        pytest.param(_portrait(6, list(range(6))), id="portrait-bottom-6-rows"),
        pytest.param(_portrait(12, list(range(6))), id="portrait-bottom-half"),
        pytest.param(_portrait(4, [0, 1]), id="portrait-corner"),
        pytest.param(_landscape(2), id="landscape-2-rows"),
    ],
)
def test_one_module_matches_the_cell_level_reference(shaded):
    reference = _reference_power(_reference_module_voltage(shaded))
    assert _power([_breos_entry(shaded)]) == pytest.approx(reference, rel=2e-3)


def test_a_string_matches_the_cell_level_reference():
    shaded, clear = _portrait(4, list(range(6))), np.zeros((24, 6), dtype=bool)
    voltage = 3 * _reference_module_voltage(shaded) + 7 * _reference_module_voltage(clear)
    power = _power([_breos_entry(shaded)] * 3 + [_breos_entry(clear)] * 7)
    assert power == pytest.approx(_reference_power(voltage), rel=1e-3)


def test_one_landscape_row_is_overstated_by_the_worst_cell_approximation():
    # One shaded column puts one fully shaded and one lit cell side by side in
    # each of a string's rows. A half is described by its mean and its single
    # worst cell, so its other shaded cells are spread evenly over the rest;
    # here that overstates the module's power by under 1 %.
    shaded = _landscape(1)
    reference = _reference_power(_reference_module_voltage(shaded))
    error = _power([_breos_entry(shaded)]) / reference - 1.0
    assert 0.0 < error < 0.01
