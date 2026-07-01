"""Tests for the Monte Carlo runner (weather + demand resampling)."""

import numpy as np
import pandas as pd
import pytest

from breos.montecarlo import MonteCarloSettings, _sample_load_scale, run_montecarlo


def _write_multiyear_weather(path, years=(2021, 2022)):
    """Write a small synthetic multi-year hourly weather CSV.

    Columns use Open-Meteo names that breos.solar recognizes
    (shortwave_radiation/direct_normal_irradiance/diffuse_radiation +
    temperature_2m/wind_speed_10m).
    """
    frames = []
    for year in years:
        idx = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
        idx = idx[~((idx.month == 2) & (idx.day == 29))]  # keep 8760 rows/year
        hour = idx.hour.to_numpy()
        # Simple daytime bell centered at noon.
        daylight = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None)
        ghi = 700.0 * daylight
        frames.append(
            pd.DataFrame(
                {
                    "date": idx,
                    "temperature_2m": 15.0 + 8.0 * daylight,
                    "wind_speed_10m": 2.0,
                    "shortwave_radiation": ghi,
                    "direct_normal_irradiance": 0.8 * ghi,
                    "diffuse_radiation": 0.2 * ghi,
                }
            )
        )
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    return path


def _base_config():
    return {
        "location": "porto",
        "n_modules": 8,
        "annual_consumption_kwh": 4000,
        "battery_kwh": 5.0,
        "cost_preset": "residential_pt",
        "emissions_country": "PT",
        "resolution": "h",
        "projection_years": 3,
    }


def test_sample_load_scale_respects_bounds():
    rng = np.random.default_rng(0)
    scales = [_sample_load_scale(rng, 0.5, min_scale=0.2, max_scale=1.5) for _ in range(500)]
    assert all(0.2 <= s <= 1.5 for s in scales)


def test_run_montecarlo_shapes_and_years(tmp_path):
    weather = _write_multiyear_weather(tmp_path / "multi.csv")
    settings = MonteCarloSettings(weather_file=str(weather), n_runs=3, years_per_run=2, seed=1)
    result = run_montecarlo(_base_config(), settings)

    assert len(result.runs) == 3
    assert result.available_years == [2021, 2022]
    for col in ("npv_savings_eur", "lcoe_eur_kwh", "final_soh_pct", "mean_grid_independence_pct"):
        assert col in result.runs.columns
    assert "npv_savings_eur" in result.summary
    assert set(result.summary["npv_savings_eur"]) >= {"mean", "p5", "p50", "p95"}


def test_run_montecarlo_is_reproducible_with_seed(tmp_path):
    weather = _write_multiyear_weather(tmp_path / "multi.csv")
    settings = MonteCarloSettings(weather_file=str(weather), n_runs=4, years_per_run=3, seed=42)
    a = run_montecarlo(_base_config(), settings).runs["npv_savings_eur"].to_numpy()
    b = run_montecarlo(_base_config(), settings).runs["npv_savings_eur"].to_numpy()
    np.testing.assert_allclose(a, b)


def test_run_montecarlo_defaults_years_to_projection_years(tmp_path):
    weather = _write_multiyear_weather(tmp_path / "multi.csv")
    settings = MonteCarloSettings(weather_file=str(weather), n_runs=1, seed=0)
    result = run_montecarlo(_base_config(), settings)
    # projection_years=3 in the base config -> 3 weather years sampled per run.
    assert len(result.runs) == 1
    assert result.summary["npv_savings_eur"]["std"] == 0.0


def test_run_montecarlo_threads_sell_price_inflation(tmp_path, monkeypatch):
    import breos.montecarlo as mc_module

    weather = _write_multiyear_weather(tmp_path / "multi.csv")
    seen = {}
    original = mc_module.cost_analysis_projection

    def _capture(*args, **kwargs):
        seen["sell_price_inflation"] = kwargs.get("sell_price_inflation")
        return original(*args, **kwargs)

    monkeypatch.setattr(mc_module, "cost_analysis_projection", _capture)
    settings = MonteCarloSettings(weather_file=str(weather), n_runs=1, years_per_run=2, seed=0)
    run_montecarlo({**_base_config(), "sell_price_inflation": 0.04}, settings)

    assert seen["sell_price_inflation"] == 0.04


def test_run_montecarlo_rejects_empty_weather(tmp_path):
    empty = tmp_path / "empty.csv"
    pd.DataFrame({"date": [], "shortwave_radiation": []}).to_csv(empty, index=False)
    settings = MonteCarloSettings(weather_file=str(empty), n_runs=1)
    with pytest.raises(ValueError):
        run_montecarlo(_base_config(), settings)
