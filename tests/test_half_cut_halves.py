"""Half-cut modules: each bypass-diode substring is two cell strings in parallel, each with its own shading."""

from dataclasses import replace

import numpy as np
import pvlib
import pytest

from breos.pv.mismatch import ShadedStringCurve
from breos.pv.shading import ModuleShading
from breos.pv_modules import get_module
from breos.solar import _get_cec_params
from tests.cell_reference import (
    BEAM,
    TEMP,
    Reference,
    bottom_rows,
    breos_entry,
    breos_power,
    half,
    landscape,
    parts,
    portrait,
    table,
)

MODULE = get_module("Generic_400W")
ONE = np.ones(1)
PARTS = parts(MODULE)


def _raw(modules: list[dict]) -> ModuleShading:
    return ModuleShading.from_mapping({"azimuth_deg": [0], "elevation_deg": [0], "modules": modules})


LIT = {"beam_shaded_fraction": table(0.0)}


# -- the configuration ----------------------------------------------------------------------------


def test_halves_are_validated():
    with pytest.raises(ValueError, match="give its fractions in them only"):
        _raw([{"substrings": [{**half(0.1), "halves": [half(0.1), half(0.1)]}, LIT, LIT]}])
    with pytest.raises(ValueError, match="one table per parallel cell string"):
        _raw([{"substrings": [{"halves": [half(0.1)]}, LIT, LIT]}])
    with pytest.raises(ValueError, match="needs beam_shaded_fraction"):
        _raw([{"substrings": [{"halves": [half(0.1), {"limiting_shaded_fraction": table(0.5)}]}, LIT, LIT]}])
    with pytest.raises(ValueError, match="unknown key"):
        _raw([{"substrings": [{"halves": [half(0.1), {**half(0.1), "sky_view_factor": 1.0}]}, LIT, LIT]}])
    halved = _raw([{"substrings": [{"halves": [half(0.1), half(0.0)]}, LIT, LIT]}])
    with pytest.raises(ValueError, match=r"cell_strings_in_parallel = 1"):
        halved.for_module(3, 1)
    with pytest.raises(ValueError, match="holds 2 entries; the module has 3 cell strings"):
        halved.for_module(3, 3)


def test_a_half_whose_worst_cell_shades_more_than_its_mean_is_refused():
    # 144 cells over 2 strings and 3 diodes: 24 cells per string, so a fully
    # shaded cell alone shades 1/24 of the string's area.
    shading = _raw([{"substrings": [{"halves": [half(0.01, 1.0), half(0.0)]}, LIT, LIT]}]).for_module(3, 2)
    problem = shading.inconsistency(MODULE.N_Cells)
    assert problem is not None and "substrings[0].halves[0]" in problem and "24" in problem
    assert (
        _raw([{"substrings": [{"halves": [half(1 / 24, 1.0), half(0.0)]}, LIT, LIT]}])
        .for_module(3, 2)
        .inconsistency(MODULE.N_Cells)
        is None
    )


def test_the_substrings_mean_is_its_halves_mean_and_the_tables_round_trip():
    raw = {"substrings": [{"halves": [half(0.2, 0.6), half(0.0)]}, LIT, {**half(0.1, 0.4)}]}
    shading = _raw([raw]).for_module(3, 2)
    assert shading.modules[0].mean[0].beam_shaded_fraction[0][0] == pytest.approx(0.1)
    echoed = shading.as_config()
    assert echoed["modules"][0] == raw
    assert _raw(echoed["modules"]).for_module(3, 2) == shading
    summary = shading.as_dict()
    assert summary["cell_strings_in_parallel"] == 2 and summary["substrings_by_halves"] == 1
    assert summary["modules_with_limiting_cells"] == 1
    only_halves = {"substrings": [{"halves": [half(0.2, 0.6), half(0.0)]}, LIT, LIT]}
    assert _raw([only_halves]).for_module(3, 2).as_dict()["modules_with_limiting_cells"] == 1


# -- the curve ------------------------------------------------------------------------------------


def test_halves_shaded_alike_are_the_substring_without_halves():
    alike = {"substrings": [{"halves": [half(0.1, 0.5), half(0.1, 0.5)]}, LIT, LIT]}
    plain = {"substrings": [half(0.1, 0.5), LIT, LIT]}
    assert breos_power(MODULE, [alike, LIT]) == pytest.approx(breos_power(MODULE, [plain, LIT]), rel=1e-12)


def test_shade_on_one_half_costs_less_than_on_both():
    # A cell a fifth shaded in one string of the substring caps that string
    # only; the other still carries its full current. In both strings it caps
    # the substring.
    one = breos_power(MODULE, [{"substrings": [{"halves": [half(0.2 / 24, 0.2), half(0.0)]}, LIT, LIT]}] + [LIT] * 3)
    both = breos_power(MODULE, [{"substrings": [half(0.2 / 24, 0.2), LIT, LIT]}] + [LIT] * 3)
    lit = breos_power(MODULE, [LIT] * 4)
    assert both < one < lit


# -- an independent cell-level reference (tests/cell_reference.py) --------------------------------

_REFERENCE = Reference(MODULE)
_FIT = _get_cec_params(MODULE)


@pytest.mark.parametrize("rows", range(1, 13))
@pytest.mark.parametrize("columns", [list(range(6)), [0, 1]], ids=["every-section", "one-section"])
def test_portrait_row_shade_matches_the_cell_level_reference(rows, columns):
    # A chain's shade is concentrated in as many cells as its mean allows, so
    # row shade, every cell lit or dark, is described exactly, however many cells it darkens.
    shaded = portrait(rows, columns)
    reference = _REFERENCE.power(_REFERENCE.module_voltage(shaded))
    assert breos_power(MODULE, [breos_entry(shaded)]) == pytest.approx(reference, rel=1e-3)


@pytest.mark.parametrize(
    "fractions",
    [[1.0, 0.5], [1.0, 0.66, 0.33], [1.0, 1.0, 0.5], [0.8, 0.6, 0.4, 0.2]],
    ids=["penumbra", "wide-penumbra", "two-rows-and-penumbra", "gradient"],
)
def test_soft_shade_matches_the_cell_level_reference(fractions):
    # Concentrating a chain's partly shaded cells into fewer darker ones
    # slightly understates it: 0.1 % for a gradient over four rows.
    shaded = bottom_rows(fractions)
    reference = _REFERENCE.power(_REFERENCE.module_voltage(shaded))
    assert breos_power(MODULE, [breos_entry(shaded)]) == pytest.approx(reference, rel=1.5e-3)


def test_a_string_matches_the_cell_level_reference():
    shaded, clear = portrait(4, list(range(6))), np.zeros((24, 6), dtype=bool)
    voltage = 3 * _REFERENCE.module_voltage(shaded) + 7 * _REFERENCE.module_voltage(clear)
    power = breos_power(MODULE, [breos_entry(shaded)] * 3 + [breos_entry(clear)] * 7)
    assert power == pytest.approx(_REFERENCE.power(voltage), rel=1e-3)


@pytest.mark.parametrize("rows", [1, 2])
def test_landscape_rows_match_the_cell_level_reference(rows):
    # One shaded column puts one fully shaded and one lit cell side by side in
    # each of a string's rows: twelve dark cells per chain.
    shaded = landscape(rows)
    reference = _REFERENCE.power(_REFERENCE.module_voltage(shaded))
    assert breos_power(MODULE, [breos_entry(shaded)]) == pytest.approx(reference, rel=1e-3)


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
    dim_sky = replace(PARTS, sky=sky * ONE)
    halves = {"substrings": [{"halves": [half(0.0), half(1.0)]}] * 3}
    shading = _raw([halves]).for_module(3, 2)
    curve = ShadedStringCurve(shading, dim_sky, ONE, 1)
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
    alone = ShadedStringCurve(_raw([{"substrings": [half(1.0), LIT, LIT]}]).for_module(3, 2), alone_parts, ONE, 1)
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
    with pytest.raises(ValueError, match="cell_strings_in_parallel"):
        replace(MODULE, cell_strings_in_parallel=5)
    assert MODULE.cell_strings_in_parallel == 2 and MODULE.cells_in_series == 72
    # 88 half-cells under the default 3 diodes would put 14.7 cells in each chain.
    with pytest.raises(ValueError, match=r"bypass_diodes \(3\) sections"):
        replace(MODULE, N_Cells=88)
    four = replace(MODULE, N_Cells=88, bypass_diodes=4)
    assert four.cells_in_series == 44
    module = replace(MODULE, bypass_diodes=4)
    module.bypass_diodes = 6
    assert module.bypass_diodes == 6
    # 64 * 3 wraps to -64 in int8, which divides 128.
    with pytest.raises(ValueError, match="must divide"):
        replace(MODULE, N_Cells=np.int16(128), cell_strings_in_parallel=np.int8(64))


def test_a_chains_shade_is_concentrated_without_an_empty_part():
    from breos.pv.mismatch import _concentrated

    mean = np.array([3 / 24, 23.5 / 24, 0.2, 0.0])
    limiting = np.array([1.0, 1.0, 0.2, 0.0])
    counts, fractions = _concentrated(24.0, mean, limiting, limiting > mean)
    np.testing.assert_allclose(sum(counts), 24.0)
    np.testing.assert_allclose(sum(c * f for c, f in zip(counts, fractions, strict=True)), 24.0 * mean)
    assert (np.asarray(counts) > 0).all()
    # Three dark cells, none at the remainder and 20 clear.
    assert [c[0] for c in counts] == [3.0, 1.0, 20.0] and [f[0] for f in fractions] == [1.0, 0.0, 0.0]
    # 23 dark cells and the half-shaded one split between the last two parts.
    assert [c[1] for c in counts] == [23.0, 0.5, 0.5] and [f[1] for f in fractions] == [1.0, 0.5, 0.5]
    # Unlimited, every cell at the mean.
    assert all(f[2] == 0.2 and f[3] == 0.0 for f in fractions)
