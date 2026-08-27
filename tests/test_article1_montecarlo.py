"""Static coverage for the forthcoming publication study Monte Carlo workflow."""

import argparse
import tomllib

from tools.reproduce_article1_montecarlo import DEFAULT_CONFIG, _selected_cases, _settings


def test_article1_montecarlo_config_pins_manuscript_method():
    config = tomllib.loads(DEFAULT_CONFIG.read_text())

    assert config["montecarlo"] == {
        "n_runs": 10000,
        "years_per_run": 20,
        "load_distribution": "uniform",
        "load_uncertainty": 0.05,
        "target_year": 2025,
        "weather_start_year": 2005,
        "weather_end_year": 2023,
        "seed": 1,
        "min_load_scale": 0.95,
        "max_load_scale": 1.05,
        "preserve_irradiance_energy": True,
        "collect_yearly": True,
        "n_procs": 1,
    }
    assert set(config["cases"]) == {"C1", "C2", "C3", "C4", "C5"}
    assert config["cases"]["C2"] == {
        "label": "Balanced",
        "n_modules": 9,
        "battery_kwh": 5.0,
        "tilt": 25.0,
        "azimuth": 185.0,
    }


def test_article1_montecarlo_cli_overrides_only_runtime_size_and_workers():
    config = tomllib.loads(DEFAULT_CONFIG.read_text())
    args = argparse.Namespace(weather_file="historical.csv", runs=25, n_procs=4)

    settings = _settings(config, args)

    assert settings.weather_file == "historical.csv"
    assert settings.n_runs == 25
    assert settings.n_procs == 4
    assert settings.load_distribution == "uniform"
    assert settings.collect_yearly is True
    assert settings.weather_start_year == 2005
    assert settings.weather_end_year == 2023


def test_article1_montecarlo_case_selection_is_explicit():
    cases = {"C1": {}, "C2": {}, "C3": {}}

    assert _selected_cases(cases, ["C2"]) == ["C2"]
    assert _selected_cases(cases, ["all"]) == ["C1", "C2", "C3"]
