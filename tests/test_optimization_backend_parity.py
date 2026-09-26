"""Projected optimization agrees across backends and with the fixed-design evaluator (#184)."""

import numpy as np
import pytest

pytest.importorskip("pymoo")
pytest.importorskip("numba", reason="the compiled backend needs the breos[fast] extra")

from breos.load_profiles import load_profile  # noqa: E402
from breos.optimization import SolarDesignProblem, evaluate_projected_design  # noqa: E402

_CONFIG = {
    "location": {"latitude": 41.15, "longitude": -8.61, "timezone": "UTC"},
    "optimization": {"objective_basis": "projected"},
    "constraints": {"budget_eur": 100000.0, "max_area_m2": 100.0},
    "simulation": {"resolution": "h", "years_projection": 3},
    "mode": {"fixed_azimuth": 180},
    "battery": {"eol_percentage": 0.93, "temperature": "weather"},
}
# (modules, battery kWh, tilt): PV only, a pack that is replaced, and a large pack.
_DESIGNS = [(6, 0.0, 30.0), (8, 5.0, 35.0), (12, 10.0, 25.0)]


@pytest.fixture(scope="module")
def _inputs():
    from tests.conftest import _build_synthetic_weather

    return _build_synthetic_weather(2023), load_profile("1", 3500, start_date="2023-01-01", timezone="UTC")


def _problem_metrics(weather, load, backend, design):
    problem = SolarDesignProblem(weather, load, _CONFIG, "results/_test_run/backend_parity", execution_backend=backend)
    out: dict = {}
    problem._evaluate(np.array(design, dtype=float), out)
    return out


@pytest.mark.parametrize("design", _DESIGNS)
def test_projected_scoring_is_identical_on_both_backends(_inputs, design):
    weather, load = _inputs

    python = _problem_metrics(weather, load, "python", design)
    compiled = _problem_metrics(weather, load, "numba", design)

    assert python.keys() == compiled.keys()
    for key in python:
        np.testing.assert_array_equal(np.asarray(python[key]), np.asarray(compiled[key]), err_msg=key)


@pytest.mark.parametrize("design", _DESIGNS)
def test_evaluate_projected_design_is_identical_on_both_backends(_inputs, design):
    weather, load = _inputs
    n_modules, battery_kwh, tilt = design
    kwargs = dict(n_modules=n_modules, battery_kwh=battery_kwh, tilt=tilt, azimuth=180.0)

    python = evaluate_projected_design(weather, load, _CONFIG, execution_backend="python", **kwargs)
    compiled = evaluate_projected_design(weather, load, _CONFIG, execution_backend="numba", **kwargs)

    assert python.metrics == compiled.metrics
    assert python.yearly.equals(compiled.yearly)
    assert python.financial.equals(compiled.financial)


@pytest.mark.parametrize("design", _DESIGNS)
def test_problem_scores_match_the_fixed_design_evaluator(_inputs, design):
    weather, load = _inputs
    n_modules, battery_kwh, tilt = design

    scored = _problem_metrics(weather, load, "python", design)
    evaluated = evaluate_projected_design(
        weather, load, _CONFIG, n_modules=n_modules, battery_kwh=battery_kwh, tilt=tilt, azimuth=180.0
    ).metrics

    shared = sorted(key for key in scored if key.startswith("Projected_") and key in evaluated)
    assert "Projected_NPV_Eur" in shared and "Projected_Grid_Independence_%" in shared
    for key in shared:
        # A design that never breaks even reports NaN on both sides.
        np.testing.assert_array_equal(scored[key], evaluated[key], err_msg=key)
