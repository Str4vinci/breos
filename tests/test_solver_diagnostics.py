"""Darkness, infeasible operating points and failed solves stay distinguishable."""

import numpy as np
import pvlib
import pytest

from breos import App
from breos.app_inputs import reuse_prepared_inputs
from breos.electrical import PVString, string_limited_dc
from breos.equipment import MpptSpec
from breos.pv import _solver
from breos.pv.mismatch import _part_voltage
from breos.pv_modules import get_module
from breos.solar import _get_cec_params
from tests.test_module_shading import CHIMNEY, _module_shaded_config
from tests.test_solver_convergence import represented
from tests.test_string_energy import _breakdown, _inverter, _setup


def params():
    return (
        np.array([0.0, 5.0, 8.0]),
        np.full(3, 1e-9),
        np.full(3, 0.2),
        np.array([np.inf, 500.0, 500.0]),
        np.full(3, 1.5),
    )


def test_darkness_has_zero_power_and_no_recoveries():
    diagnostics = _solver.SolverDiagnostics()
    with _solver.solver_scope(diagnostics=diagnostics):
        found = _solver.mpp(*[p[:1] for p in params()])
    assert found["p_mp"][0] == 0
    assert diagnostics.result() == {"recovered_steps": 0, "by_stage": {}}


def test_startup_infeasibility_is_not_a_numerical_failure():
    diagnostics = _solver.SolverDiagnostics()
    module = get_module("Generic_400W")
    with _solver.solver_scope(diagnostics=diagnostics):
        array = _breakdown(module, 1, [800.0, 0.0], [25.0, 25.0], 1.0)
        inverter = _inverter(MpptSpec(max_input_current_a=25.0), startup_voltage_v=1000.0)
        setup = _setup(inverter, [PVString(0, 0, 0, 1)], [(module, 1)])
        energy = string_limited_dc(setup, [array])
    np.testing.assert_array_equal(energy.dc, 0.0)
    assert diagnostics.result() == {"recovered_steps": 0, "by_stage": {}}


@pytest.mark.parametrize("fault", ["nan", "inf", "negative_power", "voltage", "current", "zero"])
def test_only_failed_lit_rows_take_the_mpp_fallback(monkeypatch, fault):
    original = pvlib.pvsystem.max_power_point
    expected = _solver.mpp(*params())
    calls = []

    def forced(*args, method="newton", **kwargs):
        result = original(*args, method=method, **kwargs)
        if method == "newton":
            key, value = {
                "nan": ("p_mp", np.nan),
                "inf": ("v_mp", np.inf),
                "negative_power": ("p_mp", -1.0),
                "voltage": ("v_mp", 100.0),
                "current": ("i_mp", 50.0),
                "zero": ("p_mp", 0.0),
            }[fault]
            result[key][1] = value
            # A non-finite dark result is safe to replace with zero.
            result["p_mp"][0] = np.nan
        else:
            calls.append(np.asarray(args[0]).copy())
        return result

    monkeypatch.setattr(pvlib.pvsystem, "max_power_point", forced)
    diagnostics = _solver.SolverDiagnostics()
    with _solver.solver_scope(diagnostics=diagnostics):
        recovered = _solver.mpp(*params(), stage="test_mpp", steps=[10, 20, 30])
    for key in expected:
        np.testing.assert_allclose(recovered[key], expected[key], rtol=1e-7, atol=1e-8)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], [5.0])
    assert diagnostics.result() == {"recovered_steps": 1, "by_stage": {"test_mpp": 1}}


@pytest.mark.parametrize("kind", ["current", "voltage"])
def test_iv_fallback_only_solves_failed_grid_points(monkeypatch, kind):
    name = "i_from_v" if kind == "current" else "v_from_i"
    original = getattr(pvlib.pvsystem, name)
    diode = [p[:, None] for p in params()]
    # A positive current: zero and reverse currents retry on BREOS's own bracket, not pvlib's.
    query = np.zeros((3, 4)) if kind == "current" else np.full((3, 4), 0.1)
    expected = getattr(_solver, kind)(query, *diode)
    fallback_sizes = []

    def forced(value, *args, method="lambertw"):
        result = np.asarray(original(value, *args, method=method)).copy()
        if method == "lambertw":
            result[1, 2] = np.nan
        else:
            fallback_sizes.append(np.size(value))
        return result

    monkeypatch.setattr(pvlib.pvsystem, name, forced)
    diagnostics = _solver.SolverDiagnostics()
    with _solver.solver_scope(diagnostics=diagnostics):
        recovered = getattr(_solver, kind)(query, *diode, stage=kind, steps=[10, 20, 30])
    np.testing.assert_allclose(recovered, expected, rtol=1e-7, atol=1e-8)
    assert fallback_sizes == [1]
    assert diagnostics.result() == {"recovered_steps": 1, "by_stage": {kind: 1}}


def test_bypass_region_does_not_require_a_forward_solve(monkeypatch):
    diode = [p[1:2] for p in params()]
    isc = _solver.current(0.0, *diode)
    monkeypatch.setattr(pvlib.pvsystem, "v_from_i", lambda *a, **kw: np.array([np.nan]))
    diagnostics = _solver.SolverDiagnostics()
    with _solver.solver_scope(diagnostics=diagnostics):
        assert _part_voltage(np.array([8.0]), [*diode, isc, np.array([-3.0])])[0] == -3.0
    assert diagnostics.result()["recovered_steps"] == 0


def test_double_failure_names_stage_step_equipment_and_parameters(monkeypatch):
    def forced(*args, **kwargs):
        return {k: np.full(np.shape(args[0]), np.nan) for k in ("p_mp", "v_mp", "i_mp")}

    monkeypatch.setattr(pvlib.pvsystem, "max_power_point", forced)
    with _solver.solver_scope(array=2, inverter=1, string=4):
        with pytest.raises(ValueError) as error:
            _solver.mpp(*params(), stage="cable_mpp", steps=["night", "noon", "afternoon"])
    for text in [
        "cable_mpp",
        "step noon",
        "array=2",
        "inverter=1",
        "string=4",
        "photocurrent=",
        "saturation_current=",
        "Rs=",
        "Rsh=",
        "nNsVth=",
    ]:
        assert text in str(error.value)


def test_app_records_recovery_and_retains_it_when_inputs_are_reused(monkeypatch, _patch_weather):
    cfg = {**_module_shaded_config("Generic_400W", CHIMNEY), "projection_years": 1, "resolution": "h"}
    expected = App(cfg)
    expected.simulate()
    original = pvlib.pvsystem.max_power_point
    injected = False

    def forced(*args, method="newton", **kwargs):
        nonlocal injected
        result = original(*args, method=method, **kwargs)
        if method == "newton" and not injected:
            row = np.flatnonzero(np.asarray(args[0]) > 0.1)[0]
            result["p_mp"][row] = np.nan
            injected = True
        return result

    monkeypatch.setattr(pvlib.pvsystem, "max_power_point", forced)
    with reuse_prepared_inputs():
        recovered = App(cfg)
        recovered.simulate()
        repeated = App(cfg)
        repeated.simulate()
    assert injected
    assert recovered.result()["electrical"]["solver"] == {"recovered_steps": 1, "by_stage": {"module_mpp": 1}}
    assert repeated.result()["electrical"]["solver"] == recovered.result()["electrical"]["solver"]
    assert recovered.result()["pv_dc_generation_kwh"] == expected.result()["pv_dc_generation_kwh"]
    # Recovery changes only the failed row, with 1e-7 relative agreement.
    np.testing.assert_allclose(
        recovered.timeseries()["PV_DC"],
        expected.timeseries()["PV_DC"],
        rtol=1e-7,
        atol=1e-8,
    )
    revalued = recovered.revalue({"discount_rate": 0.07})
    assert revalued["electrical"]["solver"] == recovered.result()["electrical"]["solver"]
    # Editing a returned block cannot change diagnostics retained for revaluation.
    revalued["electrical"]["solver"]["by_stage"].clear()
    assert recovered.revalue({"discount_rate": 0.08})["electrical"]["solver"]["by_stage"] == {"module_mpp": 1}
    fresh = App(cfg)
    fresh.simulate()
    assert fresh.result()["electrical"]["solver"] == {"recovered_steps": 0, "by_stage": {}}


def test_shaded_curve_recovery_names_the_original_step(monkeypatch):
    original = pvlib.pvsystem.v_from_i
    injected = False

    def forced(value, *args, method="lambertw"):
        nonlocal injected
        result = np.asarray(original(value, *args, method=method)).copy()
        if method == "lambertw" and not injected and result.ndim == 2:
            result[2, 0] = np.nan
            injected = True
        return result

    expected = represented(23, 3, 1, -3.0)
    monkeypatch.setattr(pvlib.pvsystem, "v_from_i", forced)
    diagnostics = _solver.SolverDiagnostics()
    with _solver.solver_scope(diagnostics=diagnostics):
        recovered = represented(23, 3, 1, -3.0)
    np.testing.assert_allclose(recovered.mpp, expected.mpp, rtol=1e-7, atol=1e-8)
    assert injected
    assert diagnostics.stages["part_voltage"] == {"2"}


def test_deferred_curve_failure_retains_equipment_context(monkeypatch):
    with _solver.solver_scope(inverter=1, string=4, array=2):
        curve = represented(23, 3, 1, -3.0)

    def failed(value, *args, **kwargs):
        return np.full(np.broadcast_shapes(np.shape(value), np.shape(args[0])), np.nan)

    monkeypatch.setattr(pvlib.pvsystem, "v_from_i", failed)
    with pytest.raises(ValueError) as error:
        curve._voltage(np.full(len(curve.mismatched), 0.1))
    for text in ["part_voltage", "step 0", "inverter=1", "string=4", "array=2"]:
        assert text in str(error.value)


def test_vanishing_light_does_not_trigger_an_endpoint_cancellation_failure():
    diode = [
        7.969103562317852e-18,
        2.4728117058742096e-11,
        0.1509924700380003,
        2.2033579794118368e20,
        1.856751155059707,
    ]
    reference = _get_cec_params(get_module("Generic_400W"))[0]
    diode[0] = _solver.floor_photocurrent(diode[0], reference)
    diagnostics = _solver.SolverDiagnostics()
    with _solver.solver_scope(diagnostics=diagnostics):
        result = _solver.mpp(*diode)
    assert all(value == 0.0 for value in result.values())
    assert diagnostics.result() == {"recovered_steps": 0, "by_stage": {}}


def test_an_already_infeasible_window_does_not_require_its_unused_current(monkeypatch):
    from breos.electrical import _single_array_power

    operating = _breakdown(get_module("Generic_400W"), 1, [800.0], [25.0], 1.0).module_operating
    original = pvlib.pvsystem.i_from_v
    calls = 0

    def forced(*args, **kwargs):
        nonlocal calls
        calls += 1
        # Isc establishes whether the current limit binds. The following
        # current evaluation is discarded once that limit excludes the window.
        return np.array([np.nan]) if calls == 2 else original(*args, **kwargs)

    monkeypatch.setattr(pvlib.pvsystem, "i_from_v", forced)
    diagnostics = _solver.SolverDiagnostics()
    with _solver.solver_scope(diagnostics=diagnostics):
        _, limited, voltage = _single_array_power(operating, np.ones(1), 1, 0.0, 1.0, 1.0)
    assert limited[0] == voltage[0] == 0.0
    assert diagnostics.result() == {"recovered_steps": 0, "by_stage": {}}


@pytest.mark.parametrize("kind", ["mpp", "current", "voltage"])
@pytest.mark.parametrize("column", range(5), ids=["photo", "saturation", "Rs", "Rsh", "nNsVth"])
@pytest.mark.parametrize("value", [np.nan, np.inf], ids=["nan", "inf"])
def test_nonfinite_lit_inputs_raise_input_error_without_solving(monkeypatch, kind, column, value):
    diode = [p.copy() for p in params()]
    diode[column][1] = value

    def unexpected(*args, **kwargs):
        pytest.fail("non-finite lit inputs reached the solver")

    name = {"mpp": "max_power_point", "current": "i_from_v", "voltage": "v_from_i"}[kind]
    monkeypatch.setattr(pvlib.pvsystem, name, unexpected)
    diagnostics = _solver.SolverDiagnostics()
    with _solver.solver_scope(diagnostics=diagnostics, array=2):
        with pytest.raises(ValueError) as error:
            args = diode if kind == "mpp" else [0.0, *diode]
            getattr(_solver, kind)(*args, stage="input_check", steps=["night", "noon", "afternoon"])
    for text in [
        "non-finite diode parameters",
        "input_check",
        "step noon",
        "array=2",
        "photocurrent=",
        "saturation_current=",
        "Rs=",
        "Rsh=",
        "nNsVth=",
    ]:
        assert text in str(error.value)
    assert diagnostics.result() == {"recovered_steps": 0, "by_stage": {}}


def test_nan_parameters_at_a_known_dark_step_still_give_zero_without_fallback(monkeypatch):
    diode = [p[:1].copy() for p in params()]
    diode[1][0] = np.nan
    original = pvlib.pvsystem.max_power_point
    methods = []

    def observed(*args, method="newton", **kwargs):
        methods.append(method)
        return original(*args, method=method, **kwargs)

    monkeypatch.setattr(pvlib.pvsystem, "max_power_point", observed)
    diagnostics = _solver.SolverDiagnostics()
    with _solver.solver_scope(diagnostics=diagnostics):
        result = _solver.mpp(*diode)
    assert result["p_mp"][0] == 0.0
    assert methods == ["newton"]
    assert diagnostics.result() == {"recovered_steps": 0, "by_stage": {}}
