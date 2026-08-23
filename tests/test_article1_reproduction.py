"""Opt-in regression against the licensed Article 1 household profile."""

import os
import tomllib
from pathlib import Path

import pytest

from tools.reproduce_article1 import DEFAULT_CONFIG, _fixed_candidates, _load_inputs, _sha256

EXPECTED_RLP_SHA256 = "23becc5a7bfc927b1f7604156e0e4953dcc6bb65268ca947b38db3dc4f2b28bc"
EXPECTED_CONFIG_SHA256 = "675244c823f4f6dd4947efa359266ad7b1508b1a12d85218954b1c09fc3ce32e"
EXPECTED = {
    "C1": (40.68817313425559, 5374.158425855234),
    "C2": (63.885472875836804, 3621.655066348969),
    "C3": (77.60057480030382, 2456.262367231171),
    "C4": (88.74974235783394, -5263.658994360769),
}


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
    assert len(config["reference_candidates"]) == 4


def test_article1_fixed_candidates_with_licensed_rlp():
    """Pin deterministic v0.5.3-line values when the external RLP is available."""
    rlp_value = os.environ.get("BREOS_ARTICLE1_RLP_DIRECTORY")
    if not rlp_value:
        pytest.skip("set BREOS_ARTICLE1_RLP_DIRECTORY to run the licensed Article 1 regression")

    config = tomllib.loads(DEFAULT_CONFIG.read_text())
    weather, load, _weather_path, rlp_path = _load_inputs(config, Path(rlp_value))
    assert _sha256(rlp_path) == EXPECTED_RLP_SHA256

    reproduced = _fixed_candidates(config, weather, load, set()).set_index("Label")
    for label, (expected_gi, expected_npv) in EXPECTED.items():
        # The deterministic series are stable to much tighter precision on the
        # reference platform. These tolerances allow only floating-point solver
        # noise across supported Python/NumPy combinations, not model drift.
        assert reproduced.loc[label, "BREOS_Projected_GI_%"] == pytest.approx(expected_gi, abs=5e-5)
        assert reproduced.loc[label, "BREOS_Projected_NPV_Eur"] == pytest.approx(expected_npv, abs=0.05)
