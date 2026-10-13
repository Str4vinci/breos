"""Half-cut modules: each bypass-diode substring is two cell strings in parallel, each with its own shading."""

from dataclasses import replace

import numpy as np
import pvlib
import pytest

from breos.pv.mismatch import ShadedStringCurve
from breos.pv.shading import ModuleShading
from breos.pv_modules import get_module
from breos.solar import IrradianceParts, _get_cec_params
from tests.cell_reference import Reference, breos_entry, breos_power, landscape, portrait

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


# -- an independent cell-level reference (tests/cell_reference.py) --------------------------------

_REFERENCE = Reference(MODULE)
_FIT = _get_cec_params(MODULE)


@pytest.mark.parametrize("rows", range(1, 13))
@pytest.mark.parametrize("columns", [list(range(6)), [0, 1]], ids=["every-section", "one-section"])
def test_portrait_row_shade_matches_the_cell_level_reference(rows, columns):
    shaded = portrait(rows, columns)
    error = breos_power(MODULE, [breos_entry(shaded)]) / _REFERENCE.power(_REFERENCE.module_voltage(shaded)) - 1.0
    if rows <= 7 or rows == 12:
        assert abs(error) < 2.5e-3
    else:
        # A chain is its mean and its single most-shaded cell, so with most of
        # its cells dark but a few lit, its other dark cells are spread over
        # the lit ones: 1.1 % high at 11 of 12 rows.
        assert 0.0 < error < 0.012


def test_two_landscape_rows_match_the_cell_level_reference():
    shaded = landscape(2)
    reference = _REFERENCE.power(_REFERENCE.module_voltage(shaded))
    assert breos_power(MODULE, [breos_entry(shaded)]) == pytest.approx(reference, rel=1e-3)


def test_a_string_matches_the_cell_level_reference():
    shaded, clear = portrait(4, list(range(6))), np.zeros((24, 6), dtype=bool)
    voltage = 3 * _REFERENCE.module_voltage(shaded) + 7 * _REFERENCE.module_voltage(clear)
    power = breos_power(MODULE, [breos_entry(shaded)] * 3 + [breos_entry(clear)] * 7)
    assert power == pytest.approx(_REFERENCE.power(voltage), rel=1e-3)


def test_one_landscape_row_is_overstated_by_the_worst_cell_approximation():
    # One shaded column puts one fully shaded and one lit cell side by side in
    # each of a string's rows. A half is described by its mean and its single
    # worst cell, so its other shaded cells are spread evenly over the rest;
    # here that overstates the module's power by under 1 %.
    shaded = landscape(1)
    reference = _REFERENCE.power(_REFERENCE.module_voltage(shaded))
    error = breos_power(MODULE, [breos_entry(shaded)]) / reference - 1.0
    assert 0.0 < error < 0.01


# -- second review regressions --------------------------------------------------------------------


def _chain_current(voltage: float, irradiance: float, strings: int = 2) -> float:
    """A chain of 24 half-cells at one irradiance, its current at ``voltage`` from the diode equation alone."""
    il_ref, io_ref, rs, rsh_ref, a_ref, adjust = _FIT
    il, io, rs_t, rsh, nv = (
        float(np.asarray(x))
        for x in pvlib.pvsystem.calcparams_cec(
            irradiance, TEMP, MODULE.alpha_sc, a_ref, il_ref, io_ref, rsh_ref, rs, adjust
        )
    )
    return float(pvlib.pvsystem.i_from_v(voltage / 24, il / strings, io / strings, rs_t * strings / 72,
                                         rsh * strings / 72, nv / 72))  # fmt: skip


def test_a_shaded_chain_takes_the_lit_chains_current_back_at_open_circuit():
    from scipy.optimize import brentq

    sky = 20.0
    parts = replace(PARTS, sky=sky * ONE)
    halves = {"substrings": [{"halves": [_half(0.0), _half(1.0)]}] * 3}
    shading = _raw([halves]).for_module(3, 2)
    curve = ShadedStringCurve(shading, parts, ONE, 1)
    section = brentq(lambda v: _chain_current(v, BEAM + sky) + _chain_current(v, sky), 0.0, 25.0, xtol=1e-9)
    assert curve.v_oc[0] == pytest.approx(3 * section, rel=1e-3)


def test_a_dark_section_reads_alike_on_its_own_and_beside_steps_with_differing_halves():
    # With no diffuse light, a section whose halves are both fully shaded
    # carries no current; at open circuit it holds 0 V, not the diode's drop.
    two = np.ones(2)
    dark = replace(PARTS, sky=0.0 * two, beam=BEAM * two, circumsolar=0.0 * two, other=0.0 * two,
                   temp_cell=TEMP * two, sun_azimuth=np.array([90.0, 180.0]), sun_elevation=50.0 * two)  # fmt: skip
    grid = {"azimuth_deg": [90, 180], "elevation_deg": [0]}
    first = {"substrings": [{"halves": [{"beam_shaded_fraction": [[0.0, 1.0]]}, {"beam_shaded_fraction": [[1.0, 1.0]]}]},
                            {"beam_shaded_fraction": [[0.0, 0.0]]}, {"beam_shaded_fraction": [[0.0, 0.0]]}]}  # fmt: skip
    batched = ShadedStringCurve(ModuleShading.from_mapping({**grid, "modules": [first]}).for_module(3, 2), dark, two, 1)
    alone_parts = replace(PARTS, sky=0.0 * ONE, other=0.0 * ONE)
    alone = ShadedStringCurve(_raw([{"substrings": [_half(1.0), LIT, LIT]}]).for_module(3, 2), alone_parts, ONE, 1)
    assert batched.v_oc[1] == pytest.approx(alone.v_oc[0], rel=1e-6)


def test_a_cell_string_count_that_does_not_divide_the_cells_is_refused_on_edit():
    module = replace(MODULE)
    with pytest.raises(ValueError, match="must divide"):
        module.cell_strings_in_parallel = 5
    with pytest.raises(ValueError, match="must divide"):
        module.N_Cells = 145
    with pytest.raises(ValueError, match="must divide"):
        module.bypass_diodes = 5
    assert module.cell_strings_in_parallel == 2 and module.N_Cells == 144 and module.bypass_diodes == 3


def test_every_section_holds_strings_of_whole_cells():
    # 88 half-cells under the default 3 diodes would put 14.7 cells in each chain.
    with pytest.raises(ValueError, match=r"bypass_diodes \(3\) sections"):
        replace(MODULE, N_Cells=88)
    four = replace(MODULE, N_Cells=88, bypass_diodes=4)
    assert four.cells_in_series == 44
    module = replace(MODULE, bypass_diodes=4)
    module.bypass_diodes = 6
    assert module.bypass_diodes == 6
