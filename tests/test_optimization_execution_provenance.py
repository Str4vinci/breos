"""Optimizer provenance records the backend that ran and the one asked for (#463)."""

import pytest

pytest.importorskip("pymoo")

from breos.execution import PV_ONLY_DISPATCH_PATH, resolve_execution_backend  # noqa: E402
from breos.load_profiles import load_profile  # noqa: E402
from breos.optimization import evaluate_projected_design, optimize_system_multi_objective  # noqa: E402

_CONFIG = {
    "location": {"latitude": 41.15, "longitude": -8.61, "timezone": "UTC"},
    "constraints": {"budget": 100000.0, "max_area_m2": 100.0, "max_battery_kwh": 10.0},
    "simulation": {"resolution": "h", "years_projection": 2},
    "mode": {"fixed_azimuth": 180},
    "optimization": {"early_stop": {"enabled": False}},
}


@pytest.fixture(scope="module")
def _inputs():
    from tests.conftest import _build_synthetic_weather

    return _build_synthetic_weather(2023), load_profile("demandlib_h0", 3500, start_date="2023-01-01", timezone="UTC")


def _design_execution(inputs, *, battery_kwh, **kwargs):
    weather, load = inputs
    result = evaluate_projected_design(
        weather, load, dict(_CONFIG), n_modules=6, battery_kwh=battery_kwh, tilt=30.0, azimuth=180.0, **kwargs
    )
    return result.provenance["execution"]


def _search_execution(inputs, **kwargs):
    weather, load = inputs
    result = optimize_system_multi_objective(weather, load, dict(_CONFIG), pop_size=4, n_gen=1, seed=1, **kwargs)
    return result.details["provenance"]["execution"]


def test_projected_design_records_auto_request_and_resolved_backend(_inputs):
    execution = _design_execution(_inputs, battery_kwh=5.0)

    assert execution["requested_execution_backend"] == "auto"
    assert execution["execution_backend"] == resolve_execution_backend("auto")
    assert "dispatch_path" not in execution


def test_projected_design_records_explicit_python(_inputs):
    execution = _design_execution(_inputs, battery_kwh=5.0, execution_backend="python")

    assert execution["execution_backend"] == "python"
    assert execution["requested_execution_backend"] == "python"
    assert "jit_cache" not in execution


def test_battery_free_projected_design_records_the_pv_only_path(_inputs):
    execution = _design_execution(_inputs, battery_kwh=0.0, execution_backend="python")

    assert execution["dispatch_path"] == PV_ONLY_DISPATCH_PATH


def test_search_records_auto_request_and_resolved_backend(_inputs):
    execution = _search_execution(_inputs)

    assert execution["requested_execution_backend"] == "auto"
    assert execution["execution_backend"] == resolve_execution_backend("auto")
    # A search mixes designs with and without a battery, so it claims no single path.
    assert "dispatch_path" not in execution
    if execution["execution_backend"] == "numba":
        assert execution["jit_cache"] == "unknown"


def test_search_records_explicit_python(_inputs):
    execution = _search_execution(_inputs, execution_backend="python")

    assert execution["execution_backend"] == "python"
    assert execution["requested_execution_backend"] == "python"
    assert "dispatch_path" not in execution
