"""Reverse-branch accuracy and cached current queries on represented curves."""

from dataclasses import replace

import numpy as np
import pytest

from breos.pv.mismatch import ShadedStringCurve
from breos.solar import IrradianceParts
from tests.test_module_shading import LIT, _parts, _shading
from tests.test_solver_convergence import LAYOUTS, SEED, represented


@pytest.mark.parametrize("layout", LAYOUTS)
def test_reverse_table_matches_the_exact_represented_curve_without_query_solves(monkeypatch, layout):
    curve = represented(SEED, *layout, cases=3)
    rows = np.ones(3, dtype=bool)
    amps = -float(curve.parts.pv_params.Isc) * np.geomspace(1e-9, 1.0, 101)[None, :].repeat(3, 0)
    voltage = curve._voltage(amps) - (curve.scale * curve.cable_ohm)[:, None] * amps
    expected = curve.scale[:, None] * amps
    curve.current(voltage, rows, signed=True)

    def unexpected(*args, **kwargs):
        raise AssertionError("An in-range current query must use the cached curve")

    monkeypatch.setattr(curve, "_voltage", unexpected)
    monkeypatch.setattr(curve, "_reverse_current", unexpected)
    monkeypatch.setattr(IrradianceParts, "diode_parameters", unexpected)
    measured = curve.current(voltage, rows, signed=True)
    # The search tolerance is 1e-4 relative power. Bound the branch-current
    # error by that fraction of a module's reference current, including near Voc.
    error = np.max(np.abs(measured - expected)) / float(curve.parts.pv_params.Isc)
    print(f"layout={layout}: largest reverse-current error / Isc={error:.9g}")
    assert error <= 1e-4
    assert np.all(measured <= 0.0)


def test_dark_step_parameters_are_cached_on_the_first_dark_query(monkeypatch):
    parts = replace(_parts(beam=0.0, sky=0.0, other=0.0), temp_cell=np.full(3, 25.0))
    curve = ShadedStringCurve(_shading([LIT] * 7), parts, np.ones(3), 7, cable_ohm=0.2)
    rows = np.ones(3, dtype=bool)
    voltage = np.full((3, 7), 280.0)
    assert "_dark_params" not in vars(curve)
    assert "_reverse_curve" not in vars(curve)
    expected = curve.current(voltage, rows, signed=True)
    assert "_dark_params" in vars(curve)
    assert "_reverse_curve" not in vars(curve)

    def unexpected(*args, **kwargs):
        raise AssertionError("Dark-step parameters must be generated once")

    monkeypatch.setattr(IrradianceParts, "diode_parameters", unexpected)
    np.testing.assert_array_equal(curve.current(voltage, rows, signed=True), expected)
    assert np.all(expected < 0.0)


def test_forward_queries_leave_parallel_caches_unbuilt_then_reuse_the_first_reverse_table(monkeypatch):
    curve = represented(SEED, 3, 1, -3.0, cases=3)
    rows = np.ones(3, dtype=bool)
    assert "_reverse_curve" not in vars(curve)
    assert "_dark_params" not in vars(curve)
    for signed in (False, True):
        curve.current(curve.v_mp, rows, signed=signed)
        assert "_reverse_curve" not in vars(curve)
        assert "_dark_params" not in vars(curve)
    voltage = curve.v_oc + 1.0
    expected = curve.current(voltage, rows, signed=True)
    assert "_reverse_curve" in vars(curve)
    assert "_dark_params" not in vars(curve)
    cached = curve._reverse_curve

    def unexpected(*args, **kwargs):
        raise AssertionError("The reverse table must be reused")

    monkeypatch.setattr(curve, "_voltage", unexpected)
    monkeypatch.setattr(IrradianceParts, "diode_parameters", unexpected)
    np.testing.assert_array_equal(curve.current(voltage[:, None], rows, signed=True)[:, 0], expected)
    assert curve._reverse_curve is cached


def test_a_reverse_query_beyond_the_table_uses_checked_refinement(monkeypatch):
    curve = represented(SEED, 3, 1, -3.0, cases=3)
    amps = np.full(3, -2.0 * float(curve.parts.pv_params.Isc))
    voltage = curve._terminal_voltage(amps, np.arange(3), True)
    original = curve._reverse_current
    calls = []

    def checked(*args):
        calls.append(len(args[0]))
        return original(*args)

    monkeypatch.setattr(curve, "_reverse_current", checked)
    measured = curve.current(voltage, np.ones(3, dtype=bool), signed=True)
    assert calls == [3]
    np.testing.assert_allclose(measured, curve.scale * amps, rtol=1e-7)
