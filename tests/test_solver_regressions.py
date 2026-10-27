"""Regressions for infeasible windows and ill-conditioned single-diode roots."""

import numpy as np
import pytest
from scipy.optimize import brentq

from breos.electrical import _curves_power, _CurveSource, _parallel_arrays_power, _single_array_power, _StringGroup
from breos.pv import _solver
from breos.pv_modules import get_module
from tests.test_module_shading import _curve
from tests.test_string_energy import _breakdown

TWILIGHT = (7.969103562317852e-18, 2.4728117058742096e-11, 0.1509924700380003, 2.2033579794118368e20, 1.856751155059707)


def reverse_reference(current, params=TWILIGHT):
    photo, saturation, series, shunt, diode = params
    junction = brentq(lambda v: photo - saturation * np.expm1(v / diode) - v / shunt - current, 0.0, 100.0)
    return junction - current * series


@pytest.mark.parametrize("current", [-0.1, -0.01, -1.0])
@pytest.mark.parametrize("scalar", [False, True], ids=["array", "scalar"])
def test_twilight_reverse_voltage_is_on_the_diode_curve(current, scalar):
    expected = reverse_reference(current)
    diagnostics = _solver.SolverDiagnostics()
    with _solver.solver_scope(diagnostics=diagnostics):
        result = (
            _solver.voltage(current, *TWILIGHT)
            if scalar
            else _solver.voltage(np.array([current]), *[np.array([p]) for p in TWILIGHT])
        )
    assert np.asarray(result).item() == pytest.approx(expected, rel=1e-10)
    assert diagnostics.result() == {"recovered_steps": 1, "by_stage": {"voltage": 1}}


def test_reverse_bracket_survives_endpoint_roundoff():
    actual = _solver._reverse_voltage(np.array([-0.1]), *[np.array([p]) for p in TWILIGHT])
    assert actual[0] == pytest.approx(reverse_reference(-0.1), rel=1e-10)


@pytest.mark.parametrize(
    "path", ["single", "uniform", "uniform_direct", "shaded", "parallel_arrays", "parallel_curves", "mixed_curves"]
)
@pytest.mark.parametrize("current_limit", [np.inf, 1.0])
def test_an_mppt_window_above_voc_is_infeasible_without_solver_recoveries(path, current_limit):
    diagnostics = _solver.SolverDiagnostics()
    with _solver.solver_scope(diagnostics=diagnostics):
        breakdown = _breakdown(get_module("Generic_400W"), 1, [800.0], [25.0], 1.0)
        clear = {"beam_shaded_fraction": [[0.0]]}
        shade = {"substrings": [{"beam_shaded_fraction": [[0.5]]}, clear, clear]}
        if path == "single":
            _, power, voltage = _single_array_power(
                breakdown.module_operating, np.ones(1), 1, 1500.0, 1600.0, current_limit
            )
        elif path == "parallel_arrays":
            groups = [_StringGroup(breakdown, 1), _StringGroup(breakdown, 1)]
            _, power, voltage = _parallel_arrays_power(groups, 1, 1500.0, 1600.0, current_limit, np.inf)
        else:
            curves = [_curve([clear if path.startswith("uniform") else shade])]
            if path in ("parallel_curves", "mixed_curves"):
                curves.append(_curve([clear if path == "mixed_curves" else shade]))
            if path == "uniform_direct":
                power, voltage = curves[0]._uniform_constrained(np.arange(3), 1500.0, 1600.0, current_limit)
            elif len(curves) == 1:
                power, voltage = curves[0].constrained_mpp(np.ones(3, dtype=bool), 1500.0, 1600.0, current_limit)
            else:
                _, power, voltage = _curves_power(
                    [_CurveSource(c) for c in curves], 1500.0, 1600.0, current_limit, np.inf
                )
    np.testing.assert_array_equal(power, 0.0)
    np.testing.assert_array_equal(voltage, 0.0)
    assert diagnostics.result() == {"recovered_steps": 0, "by_stage": {}}


@pytest.mark.parametrize("kind", ["current", "voltage"])
@pytest.mark.parametrize("bad_fallback", [False, True], ids=["recover", "reject_fallback"])
def test_finite_iv_values_inside_bounds_still_need_to_satisfy_the_diode_equation(monkeypatch, kind, bad_fallback):
    import pvlib

    params = [np.array([p]) for p in (8.0, 1e-9, 0.2, 500.0, 1.5)]
    name = "i_from_v" if kind == "current" else "v_from_i"
    original = getattr(pvlib.pvsystem, name)
    expected = original(np.array([1.0]), *params)

    def wrong(value, *args, method="lambertw"):
        if method == "lambertw" or bad_fallback:
            return np.array([1.0])
        return original(value, *args, method=method)

    monkeypatch.setattr(pvlib.pvsystem, name, wrong)
    diagnostics = _solver.SolverDiagnostics()
    with _solver.solver_scope(diagnostics=diagnostics):
        if bad_fallback:
            with pytest.raises(ValueError, match=f"{kind}, step 0"):
                getattr(_solver, kind)(np.array([1.0]), *params)
            assert diagnostics.result()["recovered_steps"] == 0
        else:
            actual = getattr(_solver, kind)(np.array([1.0]), *params)
            np.testing.assert_allclose(actual, expected, rtol=1e-10)
            assert diagnostics.result() == {"recovered_steps": 1, "by_stage": {kind: 1}}


@pytest.mark.parametrize("amps", [0.0, -0.01, -1.0, -1e6])
def test_scaled_iv_residual_accepts_valid_primary_values_across_current_scales(monkeypatch, amps):
    import pvlib

    diode = [np.array([p]) for p in TWILIGHT]
    voltage = 0.0 if amps == 0.0 else reverse_reference(amps)
    diagnostics = _solver.SolverDiagnostics()

    def valid(*args, method="lambertw"):
        assert method == "lambertw", "valid primary result was sent to fallback"
        return np.array([voltage])

    monkeypatch.setattr(pvlib.pvsystem, "v_from_i", valid)
    with _solver.solver_scope(diagnostics=diagnostics):
        actual = _solver.voltage(np.array([amps]), *diode)
    assert actual[0] == voltage
    assert diagnostics.result()["recovered_steps"] == 0


@pytest.mark.parametrize("layout", ["chimney", "module", "cell", "halves"])
def test_annual_designer_shading_keeps_valid_primary_solves(layout, _patch_weather, synthetic_weather):
    from breos import App
    from tests.test_module_shading import CHIMNEY, _module_shaded_config

    # The sine fixture leaves 1e-16 irradiance at 18:00. Use exact night
    # zeros here; genuine ill-conditioned twilight is tested above.
    night = (synthetic_weather.index.hour <= 6) | (synthetic_weather.index.hour >= 18)
    synthetic_weather.loc[night, ["ghi", "dni", "dhi"]] = 0.0
    config = _module_shaded_config("Generic_400W", CHIMNEY)
    config.update(projection_years=1, resolution="h", start_date="2023-01-01", battery_kwh=0.0)

    def table(value):
        return [[value] * 4] * 3

    def chain(mean):
        return {"beam_shaded_fraction": table(mean), "limiting_shaded_fraction": table(1.0)}

    if layout == "module":
        entry = {"beam_shaded_fraction": table(0.5)}
    elif layout == "cell":
        entry = {"substrings": [chain(1 / 24)] * 3}
    elif layout == "halves":
        entry = {"substrings": [{"halves": [chain(1 / 24), chain(2 / 24)]}] * 3}
    else:
        entry = {"beam_shaded_fraction": CHIMNEY}
    config["pv_strings"][0]["module_shading"]["modules"][0] = entry
    app = App(config)
    app.simulate()
    assert app.result()["electrical"]["solver"] == {"recovered_steps": 0, "by_stage": {}}


@pytest.mark.parametrize("path", ["single", "parallel_arrays", "parallel_curves", "mixed_curves"])
def test_an_infeasible_hot_step_does_not_change_feasible_steps_in_the_same_batch(path):
    from dataclasses import replace

    from breos.electrical import _diode_params
    from breos.pv.mismatch import ShadedStringCurve
    from tests.test_module_shading import _parts, _shading

    temperatures = np.array([25.0, 75.0, 25.0])
    clear = {"beam_shaded_fraction": [[0.0]]}
    shade = {"substrings": [{"beam_shaded_fraction": [[0.5]]}, clear, clear]}
    diagnostics = _solver.SolverDiagnostics()
    with _solver.solver_scope(diagnostics=diagnostics):
        if path in ("single", "parallel_arrays"):
            array = _breakdown(get_module("Generic_400W"), 1, [800.0] * 3, temperatures, 1.0)
            voc = _solver.voltage(0.0, *_diode_params(array.module_operating))
            minimum = 0.5 * (voc[0] + voc[1])
            if path == "single":
                _, power, voltage = _single_array_power(
                    array.module_operating, np.ones(3), 1, minimum, minimum + 2.0, np.inf
                )
            else:
                _, power, voltage = _parallel_arrays_power(
                    [_StringGroup(array, 1)] * 2, 1, minimum, minimum + 2.0, np.inf, np.inf
                )
        else:
            parts = replace(_parts(), temp_cell=temperatures)
            curves = [
                ShadedStringCurve(_shading([shade]), parts, np.ones(3), 1),
                ShadedStringCurve(_shading([clear if path == "mixed_curves" else shade]), parts, np.ones(3), 1),
            ]
            voc = np.max([curve.v_oc for curve in curves], axis=0)
            minimum = 0.5 * (voc[0] + voc[1])
            _, power, voltage = _curves_power([_CurveSource(c) for c in curves], minimum, minimum + 2.0, np.inf, np.inf)
    assert power[1] == voltage[1] == 0.0
    assert power[0] > 0.0
    assert power[0] == power[2]
    assert diagnostics.result() == {"recovered_steps": 0, "by_stage": {}}


def test_a_current_limit_between_isc_and_photocurrent_does_not_need_a_voltage_solve(monkeypatch):
    import pvlib

    from breos.electrical import _diode_params

    operating = _breakdown(get_module("Generic_400W"), 1, [800.0], [25.0], 1.0).module_operating
    isc = _solver.current(0.0, *_diode_params(operating))[0]
    limit = 0.5 * (isc + operating.photocurrent[0])
    assert isc < limit < operating.photocurrent[0]
    original = pvlib.pvsystem.v_from_i
    calls = []

    def observed(current, *args, **kwargs):
        calls.append(current)
        assert np.all(np.asarray(current) == 0.0), "a nonbinding current limit was inverted"
        return original(current, *args, **kwargs)

    monkeypatch.setattr(pvlib.pvsystem, "v_from_i", observed)
    minimum = operating.v_mp[0] + 1.0
    _, power, _ = _single_array_power(operating, np.ones(1), 1, minimum, minimum + 1.0, limit)
    assert power[0] > 0.0
    assert len(calls) == 1


def test_a_dark_string_has_no_feasible_power_even_when_zero_volts_is_allowed():
    from breos.pv.mismatch import ShadedStringCurve
    from tests.test_module_shading import _parts, _shading

    curve = ShadedStringCurve(
        _shading([{"beam_shaded_fraction": [[0.0]]}]), _parts(beam=0.0, sky=0.0, other=0.0), np.ones(3), 1
    )
    power, voltage = curve.constrained_mpp(np.ones(3, dtype=bool), 0.0, 1600.0, np.inf)
    np.testing.assert_array_equal(power, 0.0)
    np.testing.assert_array_equal(voltage, 0.0)


@pytest.mark.parametrize("scale", [1e-16, 1e-18])
def test_vanishing_light_open_circuit_voltage_is_bracketed(monkeypatch, scale):
    # At about 1e-13 W/m2 Lambert W puts a part's Voc slightly below zero and
    # pvlib's own bracket misses the root; the retry must still find it. At
    # 1e-15 W/m2 a rounded Isc above the photocurrent once sent a bypass-region
    # current to the forward solver.
    import tests.test_solver_convergence as convergence

    original = convergence.IrradianceParts

    def faint(**parts):
        for key in ("beam", "circumsolar", "sky", "other"):
            parts[key] = np.asarray(parts[key]) * scale
        return original(**parts)

    monkeypatch.setattr(convergence, "IrradianceParts", faint)
    diagnostics = _solver.SolverDiagnostics()
    with _solver.solver_scope(diagnostics=diagnostics):
        curve = convergence.represented(80621, 3, 1, None, cases=3)
    assert np.all(np.isfinite(curve.mpp)) and np.all(curve.mpp >= 0.0)
    assert diagnostics.result()["recovered_steps"] >= 1
