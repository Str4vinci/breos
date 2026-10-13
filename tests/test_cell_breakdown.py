"""A cell's reverse breakdown voltage: a shaded cell driven past its short-circuit current holds there."""

from dataclasses import replace

import numpy as np
import pytest

from breos.pv_modules import PVModuleParams, get_module
from tests.cell_reference import Reference, breos_entry, breos_power, portrait

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


def test_several_dark_cells_in_a_chain_break_down_as_one():
    # A chain is described by its mean and its most-shaded cell, so only that
    # cell breaks down; its other dark cells are spread over the lit ones. One
    # portrait row darkens two cells of each bottom chain, which at -3 V hold
    # it at -6 V and so bypass it; BREOS holds it at -3 V and keeps it working.
    module = _with(-3.0)
    reference = Reference(module, -3.0)
    shaded = portrait(1, list(range(6)))
    expected = reference.power(reference.module_voltage(shaded))
    assert breos_power(module, [breos_entry(shaded)]) > 1.2 * expected
