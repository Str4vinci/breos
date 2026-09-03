"""Static coverage for the forthcoming publication Monte Carlo workflow."""

import argparse
import tomllib

import tools.reproduce_article1_montecarlo as article1_montecarlo
from breos.app_config import resolve_app_config
from tools.reproduce_article1_montecarlo import DEFAULT_CONFIG, _pv_module_provenance, _selected_cases, _settings


def test_article1_montecarlo_config_pins_publication_method():
    config = tomllib.loads(DEFAULT_CONFIG.read_text())

    assert config["montecarlo"] == {
        "n_runs": 10000,
        "years_per_run": 20,
        "load_distribution": "uniform",
        "load_uncertainty": 0.05,
        "target_year": 2025,
        "weather_start_year": 2005,
        "weather_end_year": 2023,
        "seed": 42,
        "min_load_scale": 0.95,
        "max_load_scale": 1.05,
        "preserve_irradiance_energy": True,
        "collect_yearly": True,
        "n_procs": 1,
    }
    assert set(config["cases"]) == {"C1", "C2", "C3", "C4", "C5"}
    # C2 is the Gate 2 replacement for the manuscript's 9 modules and 5 kWh.
    assert config["cases"]["C2"] == {
        "label": "Best-value storage",
        "n_modules": 9,
        "battery_kwh": 7.0,
        "tilt": 35.0,
        "azimuth": 200.0,
    }
    # The Monte Carlo limit must match the optimization config's 1 C rating.
    assert config["battery_power_limit_c_rate"] == 1.0
    assert "battery_max_charge_power_w" not in config
    assert "battery_max_discharge_power_w" not in config
    assert config["battery_temperature"] == "weather"
    assert config["battery_indoor_model"] == {"enabled": True}
    assert config["solar_position"] == "weather"
    assert config["model_perez"] == "allsitescomposite1990"
    assert config["iam_model"] == "ashrae"
    assert config["diffuse_iam"] == "marion"
    assert config["albedo"] == 0.25
    assert config["temperature_model"] == "faiman"
    assert config["bifacial_model"] == "none"
    module = _pv_module_provenance(config)
    assert module["parameters"]["T_Pmax_pct"] == -0.34
    assert module["parameters"]["T_Voc_pct"] == -0.26
    assert module["area_m2"] == 1.134 * 2.278


def test_article1_montecarlo_cli_overrides_only_runtime_size_workers_and_backend():
    config = tomllib.loads(DEFAULT_CONFIG.read_text())
    args = argparse.Namespace(weather_file="historical.csv", runs=25, n_procs=4, execution_backend=None)

    settings = _settings(config, args)

    assert settings.weather_file == "historical.csv"
    assert settings.n_runs == 25
    assert settings.n_procs == 4
    assert settings.load_distribution == "uniform"
    assert settings.collect_yearly is True
    assert settings.weather_start_year == 2005
    assert settings.weather_end_year == 2023
    # The publication configuration pins no backend, so the reference path is
    # what runs unless a run explicitly asks for the accelerator.
    assert settings.execution_backend == "python"


def test_article1_montecarlo_backend_is_overridable_without_touching_the_config():
    config = tomllib.loads(DEFAULT_CONFIG.read_text())
    assert "execution_backend" not in config["montecarlo"]
    args = argparse.Namespace(weather_file="historical.csv", runs=None, n_procs=None, execution_backend="numba")

    settings = _settings(config, args)

    assert settings.execution_backend == "numba"
    # Every other input still comes from the pinned configuration.
    assert settings.n_runs == config["montecarlo"]["n_runs"]
    assert settings.seed == config["montecarlo"]["seed"]


def test_article1_montecarlo_case_selection_is_explicit():
    cases = {"C1": {}, "C2": {}, "C3": {}}

    assert _selected_cases(cases, ["C2"]) == ["C2"]
    assert _selected_cases(cases, ["all"]) == ["C1", "C2", "C3"]


def test_article1_metadata_is_removed_before_app_config_validation():
    config = tomllib.loads(DEFAULT_CONFIG.read_text())

    simulation_config, cases, module = article1_montecarlo._split_article_config(config)
    case = cases["C2"]
    simulation_config.update(
        {
            "n_modules": case["n_modules"],
            "battery_kwh": case["battery_kwh"],
            "tilt": case["tilt"],
            "azimuth": case["azimuth"],
        }
    )

    resolved = resolve_app_config(simulation_config)

    assert resolved.cfg["n_modules"] == 9
    assert "pv_module_width_m" not in simulation_config
    assert "pv_module_length_m" not in simulation_config
    assert module["width_m"] == 1.134
    assert module["length_m"] == 2.278
