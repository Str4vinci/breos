"""Tests for optimization guardrails."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pymoo")

from breos.optimization import SolarDesignProblem


def test_solar_design_problem_area_constraint_uses_pv_dimensions(monkeypatch):
    idx = pd.date_range("2025-01-01 00:00", periods=2, freq="h", tz="UTC")
    tmy_data = pd.DataFrame({"temp_air": [15.0, 16.0], "ghi": [0.0, 0.0]}, index=idx)
    houseload = pd.DataFrame({"Load": [500.0, 500.0]}, index=idx)
    dc = pd.Series([0.0, 0.0], index=idx)
    summary = pd.DataFrame({"Import [kWh]": [1.0], "Sell [kWh]": [0.0]})
    config = {
        "location": {"latitude": 41.15, "longitude": -8.61, "timezone": "UTC"},
        "simulation": {"resolution": "h"},
        "constraints": {"budget_eur": 100000, "max_area_m2": 10.0, "max_modules": 5},
        "mode": {"fixed_azimuth": 180},
        "pv": {
            "module": "Suntech_STP550S_STC",
            "dimensions": {"width": 2.0, "length": 3.0},
        },
        "battery": {"max_soc": 0.9, "min_soc": 0.1},
    }

    monkeypatch.setattr("breos.optimization.calculate_pv_production_dc", lambda **kwargs: dc)
    monkeypatch.setattr(
        "breos.optimization.simulate_energy_balance",
        lambda **kwargs: (pd.DataFrame(), 0.0, summary, 0.0, 0, pd.DataFrame()),
    )
    monkeypatch.setattr("breos.optimization.calculate_financials", lambda *args, **kwargs: (0.0, 0.0))

    problem = SolarDesignProblem(tmy_data, houseload, config, "results/_test_run/problem_area")
    out: dict = {}
    problem._evaluate(np.array([2.0, 0.0, 10.0], dtype=float), out)

    assert out["G"][1] == pytest.approx(2.0)


def test_solar_design_problem_honors_module_and_tilt_bounds():
    idx = pd.date_range("2025-01-01 00:00", periods=2, freq="h", tz="UTC")
    tmy_data = pd.DataFrame({"temp_air": [15.0, 16.0], "ghi": [0.0, 0.0]}, index=idx)
    houseload = pd.DataFrame({"Load": [500.0, 500.0]}, index=idx)
    config = {
        "location": {"latitude": 41.15, "longitude": -8.61, "timezone": "UTC"},
        "constraints": {"max_modules": 5, "max_battery_kwh": 7.0, "max_tilt_deg": 45.0},
        "mode": {"fixed_azimuth": 180},
    }

    problem = SolarDesignProblem(tmy_data, houseload, config, "results/_test_run/problem_bounds")

    assert problem.xu[0] == pytest.approx(5.0)
    assert problem.xu[1] == pytest.approx(7.0)
    assert problem.xu[2] == pytest.approx(45.0)


def test_solar_design_problem_uses_configured_resolution(monkeypatch):
    idx = pd.date_range("2025-01-01 00:00", periods=2, freq="15min", tz="UTC")
    tmy_data = pd.DataFrame({"temp_air": [15.0, 16.0], "ghi": [0.0, 0.0]}, index=idx)
    houseload = pd.DataFrame({"Load": [1000.0, 1000.0]}, index=idx)
    dc = pd.Series([0.0, 0.0], index=idx)
    summary = pd.DataFrame({"Import [kWh]": [0.5], "Sell [kWh]": [0.0]})
    captured: dict = {}
    config = {
        "location": {"latitude": 41.15, "longitude": -8.61, "timezone": "UTC"},
        "simulation": {"resolution": "15min"},
        "constraints": {"budget_eur": 100000, "max_area_m2": 100.0},
        "mode": {"fixed_azimuth": 180},
        "battery": {"temperature": 20.0, "indoor_model": {"enabled": False}},
    }

    def fake_pv(**kwargs):
        captured["pv_freq"] = kwargs["freq"]
        return dc

    def fake_balance(**kwargs):
        captured["balance_freq"] = kwargs["freq"]
        captured["temperature_series"] = kwargs["temperature_series"]
        return pd.DataFrame(), 0.0, summary, 0.0, 0, pd.DataFrame()

    def fake_financials(*args, **kwargs):
        captured["annual_load_kwh"] = args[4]
        return 0.0, 0.0

    monkeypatch.setattr("breos.optimization.calculate_pv_production_dc", fake_pv)
    monkeypatch.setattr("breos.optimization.simulate_energy_balance", fake_balance)
    monkeypatch.setattr("breos.optimization.calculate_financials", fake_financials)

    problem = SolarDesignProblem(tmy_data, houseload, config, "results/_test_run/problem_resolution")
    out: dict = {}
    problem._evaluate(np.array([2.0, 0.0, 10.0], dtype=float), out)

    assert captured["pv_freq"] == "15min"
    assert captured["balance_freq"] == "15min"
    assert captured["annual_load_kwh"] == pytest.approx(0.5)
    assert list(captured["temperature_series"]) == [20.0, 20.0]
