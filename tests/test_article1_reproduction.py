"""Opt-in regression against the licensed profile used by the forthcoming publication."""

import os
import tomllib
from pathlib import Path

import pandas as pd
import pytest

from breos.weather import read_weather_csv, weather_representative_time_offset
from tools import reproduce_article1 as article1_reproduction
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
EXPECTED_CONFIG_SHA256 = "d5795adf6a8741d4b927a1e677ade9ae59249c89ccb7140862243f30ca094a49"
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
    assert config["solar_position"] == "weather"
    assert config["model_perez"] == "allsitescomposite1990"
    assert config["iam_model"] == "ashrae"
    assert config["diffuse_iam"] == "marion"
    assert config["albedo"] == 0.25
    assert config["temperature_model"] == "faiman"
    assert config["bifacial_model"] == "none"
    assert config["pv"]["module_width_m"] == 1.134
    assert config["pv"]["module_length_m"] == 2.278
    assert config["battery"]["temperature"] == "weather"
    assert config["battery"]["indoor_model"] == {"enabled": True}
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


def test_article1_tmy_preserves_the_exact_pvgis_irradiance_offset():
    config = tomllib.loads(DEFAULT_CONFIG.read_text())
    weather_path = DEFAULT_CONFIG.parents[2] / config["simulation"]["weather_file"]

    weather = read_weather_csv(weather_path)

    metadata = weather.attrs["breos_weather_metadata"]
    assert metadata["irradiance_time_offset_hours"] == 0.1714
    assert weather_representative_time_offset(weather, "h") == pd.Timedelta(hours=0.1714)


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
            "Modules": [6, 6, 9, 9, 9],
            "Battery_kWh": [0.0, 1.0, 9.0, 13.0, 20.0],
            "Tilt": [30.0, 30.0, 35.0, 45.0, 50.0],
            "Azimuth": [200.0, 200.0, 195.0, 190.0, 185.0],
            "Projected_Grid_Independence_%": [40.0, 45.0, 75.0, 85.0, 90.0],
            "Projected_NPV_Eur": [5500.0, 5000.0, 3000.0, 300.0, -5000.0],
        }
    )

    selected = _select_pareto_representatives(pareto).set_index("Representative")

    assert list(selected.index) == [
        "max_npv",
        "max_npv_battery",
        "knee",
        "max_gi_positive_npv",
        "max_gi",
    ]
    assert selected.loc["max_npv", "Battery_kWh"] == 0.0
    assert selected.loc["max_npv_battery", "Battery_kWh"] == 1.0
    assert selected.loc["knee", "Battery_kWh"] == 9.0
    assert selected.loc["max_gi_positive_npv", "Battery_kWh"] == 13.0
    assert selected.loc["max_gi", "Battery_kWh"] == 20.0


@pytest.mark.parametrize(
    ("battery", "npv", "error"),
    [
        ([0.0, 0.0], [100.0, -100.0], "PV-plus-battery"),
        ([0.0, 1.0], [0.0, -100.0], "positive-NPV"),
    ],
)
def test_article1_pareto_representatives_require_each_constrained_category(battery, npv, error):
    pareto = pd.DataFrame(
        {
            "Battery_kWh": battery,
            "Projected_Grid_Independence_%": [40.0, 80.0],
            "Projected_NPV_Eur": npv,
        }
    )

    with pytest.raises(ValueError, match=error):
        _select_pareto_representatives(pareto)


def test_article1_battery_max_npv_can_match_the_overall_maximum():
    pareto = pd.DataFrame(
        {
            "Battery_kWh": [6.0, 10.0, 20.0],
            "Projected_Grid_Independence_%": [60.0, 80.0, 90.0],
            "Projected_NPV_Eur": [5500.0, 3000.0, -5000.0],
        }
    )

    selected = _select_pareto_representatives(pareto).set_index("Representative")

    assert selected.loc["max_npv", "Battery_kWh"] == 6.0
    assert selected.loc["max_npv_battery", "Battery_kWh"] == 6.0


def test_article1_replays_only_requested_pareto_representatives(monkeypatch, tmp_path):
    pareto = pd.DataFrame(
        {
            "Modules": [6, 6, 9, 9, 9],
            "Battery_kWh": [0.0, 1.0, 9.0, 13.0, 20.0],
            "Tilt": [30.0, 30.0, 35.0, 45.0, 50.0],
            "Azimuth": [200.0, 200.0, 195.0, 190.0, 185.0],
            "Projected_Grid_Independence_%": [40.0, 45.0, 75.0, 85.0, 90.0],
            "Projected_NPV_Eur": [5500.0, 5000.0, 3000.0, 300.0, -5000.0],
        }
    )
    calls = []

    def fake_evaluate(_weather, _load, _config, *, n_modules, battery_kwh, tilt, azimuth, execution_backend):
        calls.append((n_modules, battery_kwh, tilt, azimuth, execution_backend))
        metrics = {
            "Modules": n_modules,
            "Battery_kWh": battery_kwh,
            "Tilt": tilt,
            "Azimuth": azimuth,
            "Projected_Grid_Independence_%": 80.0,
            "Projected_NPV_Eur": 100.0,
        }
        return type(
            "ProjectedResult",
            (),
            {
                "metrics": metrics,
                "yearly": pd.DataFrame({"Year": [1]}),
                "financial": pd.DataFrame({"Year": [1]}),
            },
        )()

    monkeypatch.setattr(article1_reproduction, "evaluate_projected_design", fake_evaluate)
    representatives, artifacts = article1_reproduction._export_pareto_representatives(
        {},
        pd.DataFrame(),
        pd.DataFrame(),
        pareto,
        tmp_path,
        execution_backend="numba",
        representative_names={"max_npv_battery", "max_gi_positive_npv"},
    )

    assert list(representatives["Representative"]) == ["max_npv_battery", "max_gi_positive_npv"]
    assert calls == [
        (6, 1.0, 30.0, 200.0, "numba"),
        (9, 13.0, 45.0, 190.0, "numba"),
    ]
    assert set(artifacts) == {"max_npv_battery", "max_gi_positive_npv"}
    assert not (tmp_path / "representatives/max_npv").exists()
    assert not (tmp_path / "representatives/knee").exists()
    assert not (tmp_path / "representatives/max_gi").exists()


def test_article1_fixed_candidates_with_licensed_rlp(tmp_path):
    """Pin deterministic v0.6 projected values when the external RLP is available."""
    rlp_value = os.environ.get("BREOS_ARTICLE1_RLP_DIRECTORY")
    if not rlp_value:
        pytest.skip("set BREOS_ARTICLE1_RLP_DIRECTORY to run the forthcoming publication regression")
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
