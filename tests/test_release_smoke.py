"""Release-smoke coverage for documented public workflows."""

import hashlib
import json
import shutil
import tomllib
from pathlib import Path

import pandas as pd
import pytest

import breos
from breos import cli

REPO_ROOT = Path(__file__).resolve().parents[1]


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
    assert "npv_savings" in result
    assert result["co2_avoided_total_kg"] > 0


def test_montecarlo_example_config_smoke(tmp_path, write_multiyear_weather):
    """The documented ``breos montecarlo --config`` run reads the example's [montecarlo] table.

    Flags replace the file's run count and horizon, and the weather file its
    missing one; every other setting must come from the example.
    """
    config_path = tmp_path / "montecarlo.toml"
    shutil.copy(REPO_ROOT / "configs" / "examples" / "montecarlo.toml", config_path)
    with config_path.open("rb") as f:
        section = tomllib.load(f)["montecarlo"]
    weather_file = write_multiyear_weather(tmp_path / "historical.csv", years=(2021,))
    output = tmp_path / "runs.csv"

    code = cli.main(
        [
            "montecarlo",
            "--config",
            str(config_path),
            "--weather-file",
            str(weather_file),
            "--runs",
            "1",
            "--years",
            "1",
            "--output",
            str(output),
        ]
    )

    assert code == 0
    runs = pd.read_csv(output)
    assert runs["run"].tolist() == [1]
    assert runs["npv_savings"].notna().all()
    provenance = json.loads(output.with_name("runs.provenance.json").read_text())
    settings = provenance["settings"]
    assert settings["n_runs"] == 1
    assert settings["years_per_run"] == 1
    assert settings["weather_file"] == str(weather_file)
    for key in ("load_uncertainty", "load_distribution", "target_year", "seed", "collect_yearly", "n_procs"):
        assert settings[key] == section[key], key
    assert provenance["available_weather_years"] == [2021]
    assert provenance["resolved_config"]["pv_module"] == "Suntech_STP550S_STC"
    assert provenance["weather_file_sha256"] == hashlib.sha256(weather_file.read_bytes()).hexdigest()


def test_multi_objective_optimization_smoke(open_meteo_weather):
    pytest.importorskip("pymoo")

    from breos.optimization import optimize_system_multi_objective

    idx = pd.date_range("2025-01-01 00:00", periods=24, freq="h", tz="UTC")
    tmy_data = open_meteo_weather(idx)
    houseload = pd.DataFrame({"Load": [500.0] * len(idx)}, index=idx)
    config = {
        "location": {"latitude": 41.15, "longitude": -8.61, "timezone": "UTC"},
        "simulation": {"resolution": "h"},
        "constraints": {
            "budget": 100000.0,
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
    assert {"Modules", "Battery_kWh", "Grid_Independence_%", "NPV", "ZEB_Ratio"}.issubset(pareto.columns)
