"""Tests for the public degradation-result schema builder."""

from breos.degradation.profiles import BLAST_STATE_SCHEMA_VERSION, get_battery_model_profile
from breos.degradation.results import build_degradation_summary_from_state


def test_native_summary_preserves_legacy_schema_and_precision():
    replacements = [{"year": 2, "count": 1}]

    summary = build_degradation_summary_from_state(
        engine="native",
        model_key="naumann_lam_field_calibrated",
        final_soh_pct=93.456,
        replacement_events=replacements,
        state={"degradation_engine": "native"},
    )

    assert summary == {
        "engine": "native",
        "model_key": "naumann_lam_field_calibrated",
        "initial_soh_pct": 100.0,
        "final_soh_pct": 93.46,
        "replacement_events": replacements,
    }


def test_blast_summary_preserves_schema_precision_and_warning_categories():
    warnings = [
        {"category": "experimental_range", "code": "temperature"},
        {"category": "aging_horizon", "code": "horizon"},
        {"category": "other", "code": "ignored"},
    ]

    summary = build_degradation_summary_from_state(
        engine="blast",
        model_key="lfp_gr_250ah_prismatic",
        final_soh_pct=93.456,
        replacement_events=[{"year": 3, "count": 2}],
        state={"blast_engine": {"warnings": warnings}},
    )

    assert list(summary) == [
        "engine",
        "model_key",
        "model_profile",
        "initial_soh_pct",
        "final_soh_pct",
        "replacement_events",
        "calibration_basis",
        "pack_calibrated",
        "experimental_range_warnings",
        "aging_horizon_extrapolation_warnings",
        "state_schema_version",
    ]
    assert summary == {
        "engine": "blast",
        "model_key": "lfp_gr_250ah_prismatic",
        "model_profile": get_battery_model_profile("lfp_gr_250ah_prismatic").as_dict(),
        "initial_soh_pct": 100.0,
        "final_soh_pct": 93.5,
        "replacement_events": [{"year": 3, "count": 2}],
        "calibration_basis": "cell-model",
        "pack_calibrated": False,
        "experimental_range_warnings": [warnings[0]],
        "aging_horizon_extrapolation_warnings": [warnings[1]],
        "state_schema_version": BLAST_STATE_SCHEMA_VERSION,
    }
    assert BLAST_STATE_SCHEMA_VERSION == "1.0"
