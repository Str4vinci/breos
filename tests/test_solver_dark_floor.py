"""A shared module photocurrent floor keeps dark IV curves physical."""

from dataclasses import replace

import numpy as np
import pytest

from breos.electrical import (
    PVString,
    _curves_power,
    _CurveSource,
    _parallel_arrays_power,
    _single_array_power,
    _StringGroup,
    string_limited_dc,
)
from breos.equipment import MpptSpec
from breos.pv import _solver
from breos.pv_modules import get_module
from breos.solar import _get_cec_params
from tests.test_module_shading import LIT, _parts, _shading
from tests.test_solver_convergence import LAYOUTS, SEED, represented
from tests.test_solver_regressions import TWILIGHT, reverse_reference
from tests.test_string_energy import _breakdown, _inverter, _setup

SCALES = 10.0 ** np.arange(-30, 1)


def test_floor_is_strict_and_leaves_values_at_or_above_it_unchanged():
    reference = _get_cec_params(get_module("Generic_400W"))[0]
    boundary = _solver.DARK_FRACTION * reference
    photo = np.array([np.nextafter(boundary, 0.0), boundary, np.nextafter(boundary, np.inf), reference, np.nan])
    result = _solver.floor_photocurrent(photo, reference)
    assert result[0] == 0.0
    np.testing.assert_array_equal(result[1:], photo[1:])
    assert _solver.floor_photocurrent(photo[0], reference) == 0.0
    assert photo[0] != 0.0, "flooring must not mutate caller inputs"


def test_both_cec_generation_sites_apply_the_module_floor():
    irradiance = np.array([0.999, 1.001, 1.0e6]) * (1000 * _solver.DARK_FRACTION)
    parts = replace(_parts(), temp_cell=np.full(3, 25.0))
    params = parts.diode_parameters(irradiance)
    operating = _breakdown(parts.pv_params, 1, irradiance, [25.0] * 3, 1.0).module_operating
    reference = _get_cec_params(parts.pv_params)[0]
    assert params[0][0] == operating.photocurrent[0] == 0.0
    expected = reference * irradiance[1:] / 1000
    np.testing.assert_allclose(params[0][1:], expected, rtol=1e-15)
    np.testing.assert_array_equal(operating.photocurrent, params[0])


def test_dark_parts_keep_their_iv_curve_without_recovery_counts():
    params = [np.array([value]) for value in (0.0, *TWILIGHT[1:])]
    currents = np.array([1.0, 0.0, -0.01, -0.1, -1.0])
    diagnostics = _solver.SolverDiagnostics()
    with _solver.solver_scope(diagnostics=diagnostics):
        mpp = _solver.mpp(*params)
        voltage = _solver.voltage(currents, *params)
        measured_current = _solver.current(voltage, *params)
        direct = _solver._reverse_voltage(currents[1:], *[np.broadcast_to(p, (4,)) for p in params])
        assert _solver.voltage(0.0, *params)[0] == 0.0
    for column in mpp.values():
        np.testing.assert_array_equal(column, 0.0)
    expected = np.array(
        [
            (TWILIGHT[1] - 1.0) * TWILIGHT[3] - TWILIGHT[2],
            0.0,
            *[reverse_reference(i, (0.0, *TWILIGHT[1:])) for i in currents[2:]],
        ]
    )
    np.testing.assert_allclose(voltage, expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(measured_current, currents, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(direct, expected[1:], rtol=1e-10, atol=1e-12)
    assert diagnostics.result() == {"recovered_steps": 0, "by_stage": {}}


def test_half_cells_inherit_the_module_floor_without_a_second_threshold(monkeypatch):
    import breos.pv.mismatch as mismatch
    from tests.test_module_shading import _shading

    reference = _get_cec_params(get_module("Generic_400W"))[0]
    parts = replace(_parts(beam=1.1e-6, sky=0.0, other=0.0), temp_cell=np.full(3, 25.0))
    original = mismatch._HalfSection
    photocurrents = []

    def capture(chains, *args, **kwargs):
        photocurrents.extend(part[0] for chain in chains for part in chain)
        return original(chains, *args, **kwargs)

    monkeypatch.setattr(mismatch, "_HalfSection", capture)
    halves = [{"beam_shaded_fraction": [[fraction]]} for fraction in (0.0, 0.05)]
    shading = _shading([{"substrings": [{"halves": halves}] * 3}])
    mismatch.ShadedStringCurve(shading, parts, np.ones(3), 1)
    assert photocurrents
    for photo in photocurrents:
        assert np.all((photo > 0.0) & (photo < _solver.DARK_FRACTION * reference))


@pytest.mark.parametrize("layout", LAYOUTS)
def test_represented_curves_are_robust_from_darkness_to_full_light(monkeypatch, layout):
    import tests.test_solver_convergence as convergence

    original = convergence.IrradianceParts
    active_scale = 1.0

    def scaled(**parts):
        for key in ("beam", "circumsolar", "sky", "other"):
            parts[key] = np.asarray(parts[key]) * active_scale
        return original(**parts)

    monkeypatch.setattr(convergence, "IrradianceParts", scaled)
    previous = np.zeros(2)
    worst_decrease = 0.0
    recovered = 0
    for active_scale in SCALES:
        diagnostics = _solver.SolverDiagnostics()
        with _solver.solver_scope(diagnostics=diagnostics):
            curve = represented(SEED, *layout, cases=2)
        power = curve.mpp
        assert np.all(np.isfinite(power)) and np.all(power >= 0.0)
        decrease = float(np.max(previous - power))
        worst_decrease = max(worst_decrease, decrease)
        assert np.all(power >= previous - 1e-12 - 1e-10 * np.abs(previous)), (layout, active_scale, previous, power)
        if active_scale <= 1e-10:
            np.testing.assert_array_equal(power, 0.0)
            assert diagnostics.result() == {"recovered_steps": 0, "by_stage": {}}
        recovered += diagnostics.result()["recovered_steps"]
        previous = power.copy()
    print(f"layout={layout}: 62 steps, worst decrease={worst_decrease:.9g} W, recovered lit steps={recovered}")


@pytest.mark.parametrize("path", ["single", "parallel_arrays", "parallel_curves"])
def test_array_and_parallel_inputs_are_robust_across_the_dark_floor(monkeypatch, path):
    import tests.test_solver_convergence as convergence

    original = convergence.IrradianceParts
    active_scale = 1.0

    def scaled(**parts):
        for key in ("beam", "circumsolar", "sky", "other"):
            parts[key] = np.asarray(parts[key]) * active_scale
        return original(**parts)

    monkeypatch.setattr(convergence, "IrradianceParts", scaled)
    previous = np.zeros(2)
    recovered = 0
    for active_scale in SCALES:
        diagnostics = _solver.SolverDiagnostics()
        with _solver.solver_scope(diagnostics=diagnostics):
            if path == "parallel_curves":
                sources = [_CurveSource(represented(SEED + j, 3, 2, -3.0, cases=2)) for j in range(2)]
                power = _curves_power(sources, 0.0, np.inf, np.inf, np.inf)[1]
            else:
                arrays = [
                    _breakdown(
                        get_module("Generic_400W"), 1, np.array([800.0, 1000.0]) * active_scale, [25.0, 40.0], 1.0
                    )
                    for _ in range(2 if path == "parallel_arrays" else 1)
                ]
                if path == "single":
                    power = _single_array_power(arrays[0].module_operating, np.ones(2), 1, 0.0, np.inf, np.inf)[1]
                else:
                    power = _parallel_arrays_power(
                        [_StringGroup(a, 1) for a in arrays], 1, 0.0, np.inf, np.inf, np.inf
                    )[1]
        assert np.all(np.isfinite(power)) and np.all(power >= 0.0)
        assert np.all(power >= previous - 1e-12 - 1e-10 * np.abs(previous)), (path, active_scale, previous, power)
        if active_scale <= 1e-10:
            np.testing.assert_array_equal(power, 0.0)
            assert diagnostics.result() == {"recovered_steps": 0, "by_stage": {}}
        recovered += diagnostics.result()["recovered_steps"]
        previous = power.copy()
    print(f"path={path}: 62 steps, recovered lit steps={recovered}")


def test_dark_iv_bracketing_with_finite_shunt_is_physical_and_not_a_recovery(monkeypatch):
    import pvlib

    params = (0.0, 1e-9, 0.2, 500.0, 1.5)
    monkeypatch.setattr(pvlib.pvsystem, "v_from_i", lambda *a, **kw: np.array([np.nan]))
    diagnostics = _solver.SolverDiagnostics()
    with _solver.solver_scope(diagnostics=diagnostics):
        voltage = _solver.voltage(np.array([-0.01]), *params)
    assert voltage[0] == pytest.approx(reverse_reference(-0.01, params), rel=1e-10)
    assert voltage[0] > 0.0
    assert diagnostics.result() == {"recovered_steps": 0, "by_stage": {}}


def _mixed_light_input(irradiance, path, startup=0.0):
    module = get_module("Generic_400W")
    irradiance = np.asarray(irradiance, dtype=float)
    arrays = []
    strings = []
    for array, light in enumerate((np.full(len(irradiance), 800.0), irradiance)):
        breakdown = _breakdown(module, 7, light, np.full(len(light), 25.0), 1.0)
        parts = replace(
            _parts(),
            beam=light,
            circumsolar=np.zeros(len(light)),
            sky=np.zeros(len(light)),
            other=np.zeros(len(light)),
            temp_cell=np.full(len(light), 25.0),
            sun_azimuth=np.full(len(light), 180.0),
            sun_elevation=np.full(len(light), 50.0),
        )
        arrays.append(replace(breakdown, irradiance_parts=parts))
        shaded = path == "curves" or (path in ("mixed", "mismatched") and array == 1)
        modules = [{"beam_shaded_fraction": [[0.1 * j]]} for j in range(7)] if path == "mismatched" else [LIT] * 7
        strings.append(PVString(0, 0, array, 7, module_shading=_shading(modules) if shaded else None))
    inverter = _inverter(MpptSpec(25.0), startup_voltage_v=startup)
    return string_limited_dc(_setup(inverter, strings, [(module, 7)] * 2), arrays)


@pytest.mark.parametrize("path", ["groups", "curves", "mixed", "mismatched"])
@pytest.mark.parametrize("startup", [0.0, 80.0])
def test_a_dark_parallel_string_loads_the_lit_string_without_blocking_startup(path, startup):
    energy = _mixed_light_input([0.0, 0.999e-6, 1.001e-6], path, startup)
    power = energy.dc.to_numpy()
    print(f"path={path}, startup={startup:g}: {power}")
    assert np.all(power > 2200.0)
    np.testing.assert_allclose(power, power[-1], rtol=1e-6)
    # Both branches carry current at the shared MPP, including the dark one.
    np.testing.assert_allclose(power, 2218.37, rtol=1e-5)


@pytest.mark.parametrize("path", ["groups", "curves", "mixed", "mismatched"])
def test_one_parallel_string_can_sweep_from_darkness_while_the_other_stays_lit(path):
    irradiance = np.unique(np.r_[0.0, 800.0 * SCALES, [0.999e-6, 1.001e-6]])
    diagnostics = _solver.SolverDiagnostics()
    with _solver.solver_scope(diagnostics=diagnostics):
        energy = _mixed_light_input(irradiance, path, 80.0)
    power = energy.dc.to_numpy()
    assert np.all(np.isfinite(power)) and np.all(power > 0.0)
    # Rsh still changes below the photocurrent floor: its extra loading costs
    # at most 2e-8 W here. Allow 1e-10 relative for that and search roundoff.
    assert np.all(np.diff(power) >= -1e-10 * power[:-1]), (irradiance, power)
    np.testing.assert_allclose(power[irradiance <= 1.001e-6], power[0], rtol=1e-6)
    print(f"path={path}: {len(power)} steps, largest decrease={max(0.0, -np.diff(power).min()):.9g} W")


def test_a_dark_string_keeps_its_static_current_losses_and_cable_resistance():
    module = get_module("Generic_400W")
    factor = 0.98 * 0.96 * 0.95
    breakdown = replace(
        _breakdown(module, 7, [0.0, 0.999e-6, 1.001e-6, 800.0], [25.0] * 4, factor * 0.97),
        pvwatts_components_pct={"soiling": 2.0, "mismatch": 4.0, "availability": 3.0},
        age_degradation_pct=5.0,
    )
    group = _StringGroup(breakdown, 1, cable_ohm=0.2, modules=7)
    np.testing.assert_allclose(group.string_scale, factor, rtol=1e-15)
    np.testing.assert_allclose(
        group.operating.resistance_series,
        breakdown.module_operating.resistance_series + factor * 0.2 / 7,
        rtol=1e-15,
    )
