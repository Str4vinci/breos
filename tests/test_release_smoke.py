"""Release-smoke coverage for documented public workflows."""

import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import breos
from breos.montecarlo import MonteCarloSettings, run_montecarlo

REPO_ROOT = Path(__file__).resolve().parents[1]


def _weather_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    hour = index.hour.to_numpy()
    daylight = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None)
    ghi = 700.0 * daylight
    return pd.DataFrame(
        {
            "temperature_2m": 15.0 + 8.0 * daylight,
            "wind_speed_10m": 2.0,
            "shortwave_radiation": ghi,
            "direct_normal_irradiance": 0.8 * ghi,
            "diffuse_radiation": 0.2 * ghi,
        },
        index=index,
    )


def _write_multiyear_weather(path: Path, years=(2021,)) -> Path:
    frames = []
    for year in years:
        idx = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
        weather = _weather_frame(idx)
        weather.insert(0, "date", idx)
        frames.append(weather)
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    return path


def test_readme_quickstart_smoke(_patch_weather):
    app = breos.App(
        {
            "location": "porto",
            "n_modules": 10,
            "annual_consumption_kwh": 4000,
            "battery_kwh": 5.0,
            "cost_preset": "residential_pt",
            "emissions_country": "PT",
        }
    )

    app.simulate()
    result = app.result()

    assert result["grid_independence_pct"] > 0
    assert result["payback_year"] is None or result["payback_year"] >= 1
    assert "npv_savings_eur" in result
    assert result["co2_avoided_total_kg"] > 0


def test_montecarlo_example_config_smoke(tmp_path):
    with (REPO_ROOT / "configs" / "examples" / "montecarlo.toml").open("rb") as f:
        config = tomllib.load(f)

    weather_file = _write_multiyear_weather(tmp_path / "historical.csv")
    config["projection_years"] = 1
    config["montecarlo"]["weather_file"] = str(weather_file)
    config["montecarlo"]["n_runs"] = 1
    config["montecarlo"]["years_per_run"] = 1

    settings = MonteCarloSettings(
        weather_file=str(weather_file),
        n_runs=1,
        years_per_run=1,
        seed=42,
    )

    result = run_montecarlo(config, settings)

    assert len(result.runs) == 1
    assert result.available_years == [2021]
    assert "npv_savings_eur" in result.summary


def test_multi_objective_optimization_smoke():
    pytest.importorskip("pymoo")

    from breos.optimization import optimize_system_multi_objective

    idx = pd.date_range("2025-01-01 00:00", periods=24, freq="h", tz="UTC")
    tmy_data = _weather_frame(idx)
    houseload = pd.DataFrame({"Load": [500.0] * len(idx)}, index=idx)
    config = {
        "location": {"latitude": 41.15, "longitude": -8.61, "timezone": "UTC"},
        "simulation": {"resolution": "h"},
        "constraints": {
            "budget_eur": 100000.0,
            "max_area_m2": 100.0,
            "max_modules": 4,
            "max_battery_kwh": 2.0,
            "max_tilt_deg": 30.0,
        },
        "mode": {"fixed_azimuth": 180},
        "battery": {"temperature": 20.0, "min_soc": 0.1, "max_soc": 0.9},
    }

    result = optimize_system_multi_objective(
        tmy_data,
        houseload,
        config,
        pop_size=4,
        n_gen=1,
        seed=1,
        verbose=False,
    )

    pareto = result.details["pareto"]
    assert not pareto.empty
    assert {"Modules", "Battery_kWh", "Grid_Independence_%", "NPV_Eur", "ZEB_Ratio"}.issubset(pareto.columns)
