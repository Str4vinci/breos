"""Checks for the Article 1 validation-recovery verifier."""

import json

import pandas as pd
import pytest

from tools.validation.recovery.verify_recovery import verify


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _valid_recovery_tree(root):
    _write_csv(
        root / "validation_sandia_task13_recovered_20260902/thermal_metrics.csv",
        [
            {
                "model": "breos_faiman_default",
                "gpoa_threshold_W_m2": 200,
                "n": 26_023,
                "bias_C": -0.038,
                "rmse_C": 2.993,
                "r": 0.970,
            }
        ],
    )
    _write_csv(
        root / "validation_pcoe_recovered_20260902/thermal_metrics.csv",
        [
            {
                "model": "breos_faiman_default",
                "gpoa_threshold_W_m2": 200,
                "n": 12_708,
                "bias_C": 2.694,
                "rmse_C": 3.821,
            }
        ],
    )
    reunion = root / "validation_reunion_microgrid_recovered_20260902"
    _write_csv(
        reunion / "thermal_metrics.csv",
        [
            {
                "model": "breos_faiman_default",
                "poa_threshold_W_m2": 200,
                "n": 108_986,
                "bias_C": 8.882,
                "rmse_C": 11.162,
            }
        ],
    )
    _write_csv(reunion / "battery_thermal_metrics.csv", [{"n": 5_655, "bias_C": -3.292}])
    _write_csv(
        root / "validation_orientation_diversity_recovered_20260902/orientation_screen_metrics.csv",
        [
            {"panel": "Total", "model": "common_radiation", "rmse_W": 159.126, "r2": 0.8},
            {"panel": "Total", "model": "radiation_plus_aoi", "rmse_W": 131.821, "r2": 0.883},
        ],
    )
    dkasc = root / "validation_dkasc_recovered_20260902/results"
    _write_csv(
        dkasc / "transposition_leg_a.csv",
        [
            {
                "transposition": "perez",
                "perez_set": "allsitescomposite1990",
                "albedo": "default",
                "bias_%": -0.4787011936222662,
                "r": 0.9931215534958022,
            }
        ],
    )
    _write_csv(
        dkasc / "transposition_leg_b.csv",
        [{"transposition": "perez", "ratio_err_%": 0.5949768499216912}],
    )
    hkust = root / "validation_hkust_timing-corrected-exploratory-v4_recovered_20260902"
    hkust.mkdir(parents=True)
    (hkust / "aggregate_metrics.json").write_text(
        json.dumps(
            {
                "sites_scored": 56,
                "test_raw_energy_bias_pct": 27.122,
                "all_scored_sites": {
                    "pooled_daylight_raw_metrics": {
                        "rmse_W": 10_635.397,
                        "r": 0.911280,
                    }
                },
            }
        )
    )


def test_recovery_verifier_accepts_every_recorded_checkpoint(tmp_path, capsys):
    _valid_recovery_tree(tmp_path)

    verify(tmp_path)

    output = capsys.readouterr().out
    assert output.count("PASS ") == 17
    assert output.endswith("All recovered local-data validations reproduce their recorded checkpoints.\n")


def test_recovery_verifier_rejects_numerical_drift(tmp_path):
    _valid_recovery_tree(tmp_path)
    sandia = tmp_path / "validation_sandia_task13_recovered_20260902/thermal_metrics.csv"
    frame = pd.read_csv(sandia)
    frame.loc[0, "bias_C"] = -0.050
    frame.to_csv(sandia, index=False)

    with pytest.raises(AssertionError, match="Sandia bias C"):
        verify(tmp_path)


def test_recovery_verifier_rejects_ambiguous_checkpoint_rows(tmp_path):
    _valid_recovery_tree(tmp_path)
    sandia = tmp_path / "validation_sandia_task13_recovered_20260902/thermal_metrics.csv"
    frame = pd.read_csv(sandia)
    pd.concat([frame, frame], ignore_index=True).to_csv(sandia, index=False)

    with pytest.raises(AssertionError, match="Expected one row.*found 2"):
        verify(tmp_path)
