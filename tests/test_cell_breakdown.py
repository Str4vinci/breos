"""A cell's reverse breakdown voltage: a shaded cell driven past its short-circuit current holds there."""

from dataclasses import replace

import numpy as np
import pytest

from breos.pv_modules import PVModuleParams, get_module
from tests.cell_reference import Reference, bottom_rows, breos_entry, breos_power, portrait

BASE = get_module("Generic_400W")
CLEAR = np.zeros((24, 6), dtype=bool)


def _with(breakdown: float | None) -> PVModuleParams:
    return replace(BASE, breakdown_voltage_v=breakdown)


def _cells(*positions: tuple[int, int]) -> np.ndarray:
    shaded = CLEAR.copy()
    for row, column in positions:
        shaded[row, column] = True
    return shaded


# One dark cell in a chain of the first substring, in both chains alike, and
# in one chain of every substring.
ONE_CELL = {
    "one-chain": _cells((23, 0)),
    "both-chains": _cells((0, 0), (23, 0)),
    "every-substring": _cells((23, 0), (23, 2), (23, 4)),
}


def test_the_breakdown_voltage_is_validated():
    for value in (0.0, 1.0, float("nan"), float("-inf"), True):
        with pytest.raises(ValueError, match="breakdown_voltage_v"):
            _with(value)
    module = _with(-3.0)
    with pytest.raises(ValueError, match="breakdown_voltage_v"):
        module.breakdown_voltage_v = 2.0
    assert module.breakdown_voltage_v == -3.0 and BASE.breakdown_voltage_v is None


@pytest.mark.parametrize("shaded", ONE_CELL.values(), ids=ONE_CELL.keys())
def test_a_breakdown_beyond_the_bypass_diodes_clamp_changes_nothing(shaded):
    # Twenty-three lit cells hold the substring above -15 V, so a cell that
    # breaks down at -1000 V never does before its bypass diode conducts.
    entry = [breos_entry(shaded)]
    assert breos_power(_with(-1000.0), entry) == pytest.approx(breos_power(BASE, entry), rel=1e-7)


@pytest.mark.parametrize("breakdown", [-3.0, -5.5, -15.0])
@pytest.mark.parametrize("shaded", ONE_CELL.values(), ids=ONE_CELL.keys())
def test_one_dark_cell_matches_the_cell_level_reference(shaded, breakdown):
    module = _with(breakdown)
    reference = Reference(module, breakdown)
    expected = reference.power(reference.module_voltage(shaded))
    assert breos_power(module, [breos_entry(shaded)]) == pytest.approx(expected, rel=5e-4)


def test_a_low_breakdown_keeps_a_shaded_substring_working_in_a_string():
    module = _with(-3.0)
    reference = Reference(module, -3.0)
    shaded = ONE_CELL["every-substring"]
    voltage = 3 * reference.module_voltage(shaded) + 7 * reference.module_voltage(CLEAR)
    power = breos_power(module, [breos_entry(shaded)] * 3 + [breos_entry(CLEAR)] * 7)
    assert power == pytest.approx(reference.power(voltage), rel=5e-4)
    # Without breakdown the three shaded modules lose their substrings.
    bypassed = breos_power(BASE, [breos_entry(shaded)] * 3 + [breos_entry(CLEAR)] * 7)
    assert power > 1.3 * bypassed


@pytest.mark.parametrize("rows", [1, 2, 4])
def test_several_dark_cells_in_a_chain_break_down_together(rows):
    # One portrait row darkens two cells of each bottom chain, which at -3 V
    # hold it at -6 V and so bypass it. A chain's shade is concentrated in as
    # many dark cells as its mean allows, and they break down together.
    module = _with(-3.0)
    reference = Reference(module, -3.0)
    shaded = portrait(rows, list(range(6)))
    expected = reference.power(reference.module_voltage(shaded))
    assert breos_power(module, [breos_entry(shaded)]) == pytest.approx(expected, rel=1e-3)


def test_soft_shade_around_a_dark_cell_breaks_down_too_soon():
    # A chain gives only its mean and its most-shaded cell. Three half-shaded
    # rows and one dark cell are concentrated into three dark cells and a
    # half-shaded one, which at -3 V break down as three: the module is
    # understated by 15 %. The same shade without breakdown is exact.
    shaded = bottom_rows([0.5, 0.5, 0.5])
    shaded[23, 0] = 1.0
    module = _with(-3.0)
    reference = Reference(module, -3.0)
    error = breos_power(module, [breos_entry(shaded)]) / reference.power(reference.module_voltage(shaded)) - 1.0
    assert -0.16 < error < -0.13
    reference = Reference(BASE)
    expected = reference.power(reference.module_voltage(shaded))
    assert breos_power(BASE, [breos_entry(shaded)]) == pytest.approx(expected, rel=1e-3)


# -- review regressions ---------------------------------------------------------------------------


def _split_curve(module: str, beam: float, sky: float, temp: float, breakdown: float | None = -3.0):
    """One module whose three substrings each have one dark cell in one half and two in the other."""
    from breos.pv.mismatch import ShadedStringCurve
    from breos.pv.shading import ModuleShading
    from breos.solar import IrradianceParts

    one = np.ones(1)
    pv = replace(get_module(module), breakdown_voltage_v=breakdown)
    parts = IrradianceParts(
        beam=beam * one, circumsolar=0.0 * one, sky=sky * one, other=0.0 * one, temp_cell=temp * one,
        sun_azimuth=180.0 * one, sun_elevation=50.0 * one, pv_params=pv,
    )  # fmt: skip
    halves = [
        {"beam_shaded_fraction": [[1 / 24]], "limiting_shaded_fraction": [[1.0]]},
        {"beam_shaded_fraction": [[2 / 24]], "limiting_shaded_fraction": [[1.0]]},
    ]
    entry = {"substrings": [{"halves": halves}] * 3}
    shading = ModuleShading.from_mapping({"azimuth_deg": [0], "elevation_deg": [0], "modules": [entry]})
    return ShadedStringCurve(shading.for_module(3, 2), parts, one, 1)


def test_the_tabled_current_follows_the_curve_across_a_chains_breakdown():
    curve = _split_curve("Generic_400W", 800.0, 200.0, 25.0)
    step = np.array([True])

    def exact(voltage: float) -> float:
        low, high = 0.0, float(curve._table_i.max())
        for _ in range(80):
            middle = 0.5 * (low + high)
            if curve._terminal_voltage(np.array([middle]), np.array([0]), False)[0] > voltage:
                low = middle
            else:
                high = middle
        return 0.5 * (low + high)

    for voltage in np.linspace(30.0, 46.0, 33):
        tabled = curve.current(np.array([[voltage]]), step)[0, 0]
        assert tabled == pytest.approx(exact(voltage), rel=2e-3, abs=1e-3), voltage


def test_a_dark_cell_has_no_power_above_the_curves_reach():
    # No diffuse light: a fully shaded cell's short-circuit current is 0, and
    # past it, at any current, it breaks down.
    curve = _split_curve("Suntech_STP550S_STC", 1000.0, 0.0, 25.0)
    step = np.array([True])
    # Any current breaks the dark cells down, so the module falls to 38.8 V,
    # well below its 47.7 V open circuit, and nothing lies between.
    assert curve._terminal_voltage(np.array([1e-6]), np.array([0]), False)[0] < 39.0
    power, _ = curve.constrained_mpp(step, 43.0, 44.0, np.inf)
    assert power[0] < 1e-6


def test_a_dark_cell_does_not_break_down_at_open_circuit():
    breaking = _split_curve("Generic_400W", 1000.0, 0.0, -20.0)
    plain = _split_curve("Generic_400W", 1000.0, 0.0, -20.0, breakdown=None)
    assert breaking.v_oc[0] == pytest.approx(plain.v_oc[0], rel=1e-6)
