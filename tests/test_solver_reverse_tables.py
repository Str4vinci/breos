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
    curve._reverse_curve
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


def test_half_cut_reverse_current_is_continuous_across_the_cached_edge():
    curve = represented(SEED, 3, 2, -3.0, cases=3, cable_ohm=0.0)
    edge = curve._reverse_curve[0][2, -1]
    rows = np.array([False, False, True])
    currents = curve.current(np.array([[edge - 1e-7, edge + 1e-7]]), rows, signed=True)
    np.testing.assert_allclose(currents[0, 1], currents[0, 0], rtol=1e-6)
    actual = curve._voltage(currents / curve.scale[2], np.array([2]))
    np.testing.assert_allclose(actual, [[edge - 1e-7, edge + 1e-7]], rtol=1e-10, atol=1e-8)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_reverse_currents_are_monotonic_and_have_checked_residuals_to_the_feasible_edge(layout):
    curve = represented(SEED, *layout, cases=3, cable_ohm=0.0)
    select = np.arange(3)
    # Three other STC strings can drive at most this much reverse current.
    cap = -3.0 * float(curve.parts.pv_params.Isc) / curve.scale
    curve._reverse_curve
    curve._extend_reverse(cap, select)
    maximum = curve._voltage(cap, select)
    voltage = curve.v_oc[:, None] + (maximum - curve.v_oc)[:, None] * np.linspace(0.0, 1.0, 65)
    current = curve.current(voltage, np.ones(3, dtype=bool), signed=True)
    assert np.all(np.diff(current, axis=1) <= 1e-7)
    measured = curve._voltage(current / curve.scale[:, None], select)
    # In-range interpolation uses the 1e-4 represented-curve tolerance.
    np.testing.assert_allclose(measured, voltage, rtol=1e-4, atol=1e-7)
    exact = curve._reverse_current(voltage.ravel(), np.repeat(select, 65), (current / curve.scale[:, None]).ravel())
    found = curve._voltage(exact.reshape(3, 65), select)
    assert np.all(np.abs(found - voltage) <= 1e-8 + 1e-10 * np.abs(voltage))
    assert np.all(np.diff(exact.reshape(3, 65), axis=1) <= 1e-7)


def test_parallel_power_above_the_half_cut_reverse_edge_matches_an_extended_table_reference(monkeypatch):
    import breos.pv.mismatch as mismatch
    from breos.electrical import _curves_power, _CurveSource, _GroupSource, _StringGroup
    from breos.pv_modules import get_module
    from tests.test_string_energy import _breakdown

    curve = represented(SEED, 3, 2, -3.0, cases=3, step=2, cable_ohm=0.0)
    edge = curve._reverse_curve[0][0, -1]
    group = _GroupSource(_StringGroup(_breakdown(get_module("Generic_400W"), 7, [1089.0], [25.0], 1.0), 4, modules=7))
    limits = (edge + 1.0, edge + 10.0, np.inf, np.inf)
    production = _curves_power([group, _CurveSource(curve)], *limits)[1]
    monkeypatch.setattr(mismatch, "_REVERSE_EXTENSION_POINTS", 2048)
    reference = represented(SEED, 3, 2, -3.0, cases=3, step=2, cable_ohm=0.0)
    expected = _curves_power([group, _CurveSource(reference)], *limits)[1]
    print(f"reverse-edge parallel: production={production}, extended reference={expected}")
    assert expected[0] > 6000.0
    np.testing.assert_allclose(production, expected, rtol=1e-4)


@pytest.mark.parametrize("failure", ["bracket", "residual"])
def test_an_unresolved_reverse_current_raises_with_solver_context(monkeypatch, failure):
    curve = represented(SEED, 3, 1, -3.0, cases=3, cable_ohm=0.0)
    curve._solver_location = {"inverter": 2, "array": 4, "string": 8}
    monkeypatch.setattr(
        curve,
        "_voltage",
        lambda current, select: (
            np.zeros_like(current) if failure == "bracket" else np.where(current < -1.0, 200.0, 0.0)
        ),
    )
    with pytest.raises(
        ValueError, match=f"reverse-current {failure} failed at string_reverse_current, step 2"
    ) as error:
        curve._reverse_current(np.array([100.0]), np.array([2]), np.array([-1.0]))
    for detail in (
        "inverter=2",
        "array=4",
        "string=8",
        "photocurrent=",
        "saturation_current=",
        "Rs=",
        "Rsh=",
        "nNsVth=",
    ):
        assert detail in str(error.value)


def test_further_half_chain_extension_preserves_existing_reverse_values():
    curve = represented(SEED, 3, 2, -3.0, cases=3, cable_ohm=0.0)
    select = np.arange(3)
    rows = np.ones(3, dtype=bool)
    voltage = curve._reverse_curve[0][:, -1] - 1.0
    before = curve.current(voltage, rows, signed=True)
    old_curve = curve._voltage(before / curve.scale, select)
    cap = np.full(3, -12.0 * float(curve.parts.pv_params.Isc))
    curve._extend_reverse(cap, select)
    target = curve._voltage(cap, select)
    found = curve.current(target, rows, signed=True)
    np.testing.assert_allclose(found, cap * curve.scale, rtol=1e-7)
    np.testing.assert_array_equal(curve.current(voltage, rows, signed=True), before)
    np.testing.assert_array_equal(curve._voltage(before / curve.scale, select), old_curve)


def test_binary_current_lookup_keeps_both_sides_of_a_repeated_voltage():
    from breos.pv.mismatch import _table_current

    voltage = np.tile([10.0, 9.0, 9.0, 5.0, 0.0, -1.0], (6, 1))
    current = np.tile([-1.0, 0.0, 3.0, 4.0, 5.0, 6.0], (6, 1))
    query = np.array([9.0, np.nextafter(9.0, 0.0), 5.0, 0.0, 11.0, -2.0])
    np.testing.assert_allclose(_table_current(voltage, current, query), [0.0, 3.0, 4.0, 5.0, 0.0, 6.0], atol=1e-15)


def _two_alike_half_sections(breakdown: float = -3.0) -> ShadedStringCurve:
    # Two substrings each have one cell at half beam in one half and the other
    # half clear: merging solves them as one section of weight 2 and leaves the
    # second section with no step of its own.
    from tests.test_module_shading import _table

    shaded_half = {"beam_shaded_fraction": _table(0.5 / 24), "limiting_shaded_fraction": _table(0.5)}
    alike = {"halves": [LIT, shaded_half]}
    parts = replace(_parts(), pv_params=replace(_parts().pv_params, breakdown_voltage_v=breakdown))
    return ShadedStringCurve(_shading([{"substrings": [alike, alike, LIT]}]), parts, np.ones(3), 1)


def test_signed_current_above_open_circuit_with_merged_half_sections():
    curve = _two_alike_half_sections()
    for offset in (0.01, 1.0, 5.0):
        current = curve.current(curve.v_oc[:, None] + offset, np.ones(3, dtype=bool), signed=True)
        assert np.isfinite(current).all() and (current < 0.0).all()


def test_a_parallel_partner_drives_merged_half_sections_in_reverse():
    from breos.electrical import _curves_power, _CurveSource

    clear = ShadedStringCurve(_shading([LIT]), _parts(), np.ones(3), 1)
    power = _curves_power([_CurveSource(_two_alike_half_sections()), _CurveSource(clear)], 0, np.inf, np.inf, np.inf)
    assert np.isfinite(np.asarray(power, dtype=float)).all()
