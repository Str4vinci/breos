"""Opt-in regression against the licensed Article 1 household profile."""

import os
import tomllib
from pathlib import Path

import pandas as pd
import pytest

from tools.reproduce_article1 import (
    DEFAULT_CONFIG,
    _battery_cost_scenarios,
    _battery_cost_slug,
    _fixed_candidates,
    _load_inputs,
    _pv_module_provenance,
    _resolved_rlp_path,
    _select_pareto_representatives,
    _sha256,
)

EXPECTED_RLP_SHA256 = "23becc5a7bfc927b1f7604156e0e4953dcc6bb65268ca947b38db3dc4f2b28bc"
EXPECTED_CONFIG_SHA256 = "674dec1987b4c4b668a88a9af68b53f737d9e4ecec9da4ccab8aac7d1ebc0d37"
FIXED_REGRESSION_LABELS = {"C1", "C2", "C3", "C4"}


def test_article1_config_pins_projected_run_controls():
    config = tomllib.loads(DEFAULT_CONFIG.read_text())

    assert _sha256(DEFAULT_CONFIG) == EXPECTED_CONFIG_SHA256
    assert config["simulation"] == {
        "resolution": "15min",
        "years_projection": 20,
        "weather_file": "validation/data/weather/porto_tmy_2005_2023_pvgis-sarah3.csv.gz",
        "irradiance_resampling": "clear_sky_energy_conserving",
    }
    assert config["optimization"] == {
        "algorithm": "nsga2",
        "objective_basis": "projected",
        "pop_size": 100,
        "n_offsprings": 50,
        "n_gen": 40,
        "seed": 1,
        "early_stop": {"ftol": 0.0025, "period": 10, "min_gen": 20, "n_skip": 0},
    }
    assert config["constraints"]["enforce_zeb"] is False
    assert config["pv"]["module_width_m"] == 1.134
    assert config["pv"]["module_length_m"] == 2.278
    assert config["battery"]["temperature"] == 25.0
    assert config["battery"]["indoor_model"] == {"enabled": False}
    assert config["emissions"]["average_grid_carbon_intensity_gco2_kwh"] == 127.91
    assert len(config["reference_candidates"]) == 5
    assert config["reference_candidates"][-1] == {
        "label": "C5",
        "modules": 4,
        "battery_kwh": 0.0,
        "tilt": 35.0,
        "azimuth": 180.0,
    }

    module = _pv_module_provenance(config)
    assert module["parameters"]["T_Pmax_pct"] == -0.34
    assert module["parameters"]["T_Voc_pct"] == -0.26
    assert module["area_m2"] == pytest.approx(1.134 * 2.278)


def test_article1_battery_cost_scenarios_are_explicit_and_stable():
    assert _battery_cost_scenarios(None) == [None]
    assert _battery_cost_scenarios([350.0, 500.0, 711.0, 500.0]) == [350.0, 500.0, 711.0]
    assert _battery_cost_slug(350.0) == "350"
    assert _battery_cost_slug(711.5) == "711p5"

    with pytest.raises(ValueError, match="non-negative"):
        _battery_cost_scenarios([-1.0])


def test_article1_rlp_provenance_tracks_the_resolution_specific_file(tmp_path):
    config = tomllib.loads(DEFAULT_CONFIG.read_text())

    assert _resolved_rlp_path(config, tmp_path).name == "EREDES_2025_BTN_1000kwh_15min.csv"

    config["simulation"]["resolution"] = "h"
    assert _resolved_rlp_path(config, tmp_path).name == "EREDES_2025_BTN_1000kwh_hourly.csv"

    config["load"]["profile_type"] = "1"
    assert _resolved_rlp_path(config, tmp_path) is None


def test_article1_pareto_representatives_use_projected_objectives():
    pareto = pd.DataFrame(
        {
            "Modules": [6, 9, 9],
            "Battery_kWh": [0.0, 9.0, 20.0],
            "Tilt": [30.0, 35.0, 45.0],
            "Azimuth": [190.0, 190.0, 175.0],
            "Projected_Grid_Independence_%": [40.0, 75.0, 90.0],
            "Projected_NPV_Eur": [5500.0, 3000.0, -5000.0],
        }
    )

    selected = _select_pareto_representatives(pareto).set_index("Representative")

    assert selected.loc["max_npv", "Battery_kWh"] == 0.0
    assert selected.loc["knee", "Battery_kWh"] == 9.0
    assert selected.loc["max_gi", "Battery_kWh"] == 20.0


def test_article1_fixed_candidates_with_licensed_rlp(tmp_path):
    """Pin deterministic v0.6 projected values when the external RLP is available."""
    rlp_value = os.environ.get("BREOS_ARTICLE1_RLP_DIRECTORY")
    if not rlp_value:
        pytest.skip("set BREOS_ARTICLE1_RLP_DIRECTORY to run the licensed Article 1 regression")
    pytest.skip(
        "pin the final corrected C1-C4 values after the study author runs the clean 0.6 release-candidate workflow"
    )

    config = tomllib.loads(DEFAULT_CONFIG.read_text())
    weather, load, _weather_path, rlp_path = _load_inputs(config, Path(rlp_value))
    assert _sha256(rlp_path) == EXPECTED_RLP_SHA256

    _fixed_candidates(config, weather, load, FIXED_REGRESSION_LABELS, tmp_path)
    for label in FIXED_REGRESSION_LABELS:
        candidate_dir = tmp_path / "candidates" / label.lower()
        assert (candidate_dir / "yearly_summary.csv").is_file()
        assert (candidate_dir / "cost_projection.csv").is_file()
        assert (candidate_dir / "metrics.json").is_file()
