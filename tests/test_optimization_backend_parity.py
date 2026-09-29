"""Projected optimization agrees across backends and with the fixed-design evaluator (#184)."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pymoo")
pytest.importorskip("numba", reason="the compiled backend needs the breos[fast] extra")

from breos.load_profiles import load_profile  # noqa: E402
from breos.optimization import SolarDesignProblem, evaluate_projected_design  # noqa: E402

_CONFIG = {
    "location": {"latitude": 41.15, "longitude": -8.61, "timezone": "UTC"},
    "optimization": {"objective_basis": "projected"},
    "constraints": {"budget": 100000.0, "max_area_m2": 100.0},
    "simulation": {"resolution": "h", "years_projection": 3},
    "mode": {"fixed_azimuth": 180},
    "battery": {"eol_percentage": 0.93, "temperature": "weather"},
}
# (modules, battery kWh, tilt): PV only, a pack that is replaced, and a large pack.
_DESIGNS = [(6, 0.0, 30.0), (8, 5.0, 35.0), (12, 10.0, 25.0)]


@pytest.fixture(scope="module")
def _inputs():
    from tests.conftest import _build_synthetic_weather

    return _build_synthetic_weather(2023), load_profile("demandlib_h0", 3500, start_date="2023-01-01", timezone="UTC")


def _problem_metrics(weather, load, backend, design, config=_CONFIG):
    problem = SolarDesignProblem(weather, load, config, "results/_test_run/backend_parity", execution_backend=backend)
    out: dict = {}
    problem._evaluate(np.array(design, dtype=float), out)
    return out


def _assert_metrics_identical(python, compiled):
    # Per key, so a NaN (a design that never breaks even) must be NaN on both
    # sides; dict == would pass only while both hold the np.nan singleton.
    assert python.keys() == compiled.keys()
    for key in python:
        np.testing.assert_array_equal(np.asarray(python[key]), np.asarray(compiled[key]), err_msg=key)


@pytest.mark.parametrize("design", _DESIGNS)
def test_projected_scoring_is_identical_on_both_backends(_inputs, design):
    weather, load = _inputs

    python = _problem_metrics(weather, load, "python", design)
    compiled = _problem_metrics(weather, load, "numba", design)

    _assert_metrics_identical(python, compiled)


@pytest.mark.parametrize("design", _DESIGNS)
def test_evaluate_projected_design_is_identical_on_both_backends(_inputs, design):
    weather, load = _inputs
    n_modules, battery_kwh, tilt = design
    kwargs = dict(n_modules=n_modules, battery_kwh=battery_kwh, tilt=tilt, azimuth=180.0)

    python = evaluate_projected_design(weather, load, _CONFIG, execution_backend="python", **kwargs)
    compiled = evaluate_projected_design(weather, load, _CONFIG, execution_backend="numba", **kwargs)

    _assert_metrics_identical(python.metrics, compiled.metrics)
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
    assert "Projected_NPV" in shared and "Projected_Grid_Independence_%" in shared
    for key in shared:
        # A design that never breaks even reports NaN on both sides.
        np.testing.assert_array_equal(scored[key], evaluated[key], err_msg=key)


# Fixed-target smart charging under a time-of-use tariff: the battery charges
# from the grid off-peak up to half its usable capacity and discharges only at
# peak. The PT schedule is in civil time, so the run needs the Lisbon timezone.
_TARIFF = {
    "schedule": "pt_mainland_2026_daily_bi",
    "currency": "EUR",
    "import_prices": {"peak": 0.50, "off_peak": 0.10},
    "export_prices": {"all": 0.03},
    "fixed_charge_per_day": 0.40,
}
_FIXED_TARGET = {
    "mode": "fixed_target",
    "target_usable_fraction": 0.5,
    "charge_periods": ["off_peak"],
    "discharge_periods": ["peak"],
    "grid_charge_efficiency": 0.95,
}
_BLAST = {"degradation_engine": "blast", "blast_model": "nmc_gr_50ah_b1", "enable_resistance_fade": False}
# The 5 kWh pack charges from the grid and is replaced within three years.
_FIXED_TARGET_DESIGN = (8, 5.0, 35.0)


def _fixed_target_config(resolution, engine):
    config = deepcopy(_CONFIG)
    config["location"]["timezone"] = "Europe/Lisbon"
    config["simulation"]["resolution"] = resolution
    if engine == "blast":
        config["battery"].update(_BLAST)
    config["tariff"] = deepcopy(_TARIFF)
    config["smart_charging"] = dict(_FIXED_TARGET)
    return config


@pytest.fixture(scope="module", params=["h", "15min"])
def _lisbon_inputs(request):
    from tests.conftest import _build_open_meteo_weather

    resolution = request.param
    index = pd.date_range("2026-01-01", "2027-01-01", inclusive="left", freq=resolution, tz="Europe/Lisbon")
    hour = index.hour.to_numpy() + index.minute.to_numpy() / 60.0
    # A morning and an evening peak, so the peak window has load to serve.
    load_w = 250.0 + 300.0 * np.exp(-(((hour - 7.5) / 1.5) ** 2)) + 700.0 * np.exp(-(((hour - 20.0) / 2.0) ** 2))
    return resolution, _build_open_meteo_weather(index), pd.DataFrame({"Load": load_w}, index=index)


@pytest.mark.parametrize("engine", ["native", "blast"])
@pytest.mark.filterwarnings("ignore::breos.degradation.validation.BlastExperimentalRangeWarning")
def test_fixed_target_scoring_is_identical_on_both_backends(_lisbon_inputs, engine):
    resolution, weather, load = _lisbon_inputs
    config = _fixed_target_config(resolution, engine)

    python = _problem_metrics(weather, load, "python", _FIXED_TARGET_DESIGN, config)
    compiled = _problem_metrics(weather, load, "numba", _FIXED_TARGET_DESIGN, config)

    # The problem metrics carry no grid-charge total; the evaluation test below
    # checks grid charging for the same config and design.
    assert python["Projected_Total_Replacements"] > 0
    _assert_metrics_identical(python, compiled)


@pytest.mark.parametrize("engine", ["native", "blast"])
@pytest.mark.filterwarnings("ignore::breos.degradation.validation.BlastExperimentalRangeWarning")
def test_fixed_target_evaluation_is_identical_on_both_backends(_lisbon_inputs, engine):
    resolution, weather, load = _lisbon_inputs
    config = _fixed_target_config(resolution, engine)
    n_modules, battery_kwh, tilt = _FIXED_TARGET_DESIGN
    kwargs = dict(n_modules=n_modules, battery_kwh=battery_kwh, tilt=tilt, azimuth=180.0)

    python = evaluate_projected_design(weather, load, config, execution_backend="python", **kwargs)
    compiled = evaluate_projected_design(weather, load, config, execution_backend="numba", **kwargs)

    # The case must exercise grid charging and a replacement, or parity is vacuous.
    assert python.yearly["Grid_AC_To_Battery_kWh"].sum() > 0
    assert python.metrics["Projected_Total_Replacements"] > 0
    _assert_metrics_identical(python.metrics, compiled.metrics)
    assert python.yearly.equals(compiled.yearly)
    assert python.financial.equals(compiled.financial)
