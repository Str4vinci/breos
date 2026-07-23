"""Tests for the public degradation-result schema builder."""

from breos.degradation.results import build_degradation_summary


def test_native_summary_preserves_legacy_schema_and_precision():
    replacements = [{"year": 2, "count": 1}]

    summary = build_degradation_summary(
        engine="native",
        model_key="naumann_lam_field_calibrated",
        final_soh_pct=93.456,
        replacement_events=replacements,
    )

    assert summary == {
        "engine": "native",
        "model_key": "naumann_lam_field_calibrated",
        "initial_soh_pct": 100.0,
        "final_soh_pct": 93.46,
        "replacement_events": replacements,
    }


def test_blast_summary_preserves_schema_precision_and_warning_categories():
    profile = {"key": "lfp_gr_250ah_prismatic", "upstream": {"commit": "abc"}}
    warnings = [
        {"category": "experimental_range", "code": "temperature"},
        {"category": "aging_horizon", "code": "horizon"},
        {"category": "other", "code": "ignored"},
    ]

    summary = build_degradation_summary(
        engine="blast",
        model_key="lfp_gr_250ah_prismatic",
        model_profile=profile,
        final_soh_pct=93.456,
        replacement_events=[{"year": 3, "count": 2}],
        warning_records=warnings,
        state_schema_version="1.0",
    )

    assert summary == {
        "engine": "blast",
        "model_key": "lfp_gr_250ah_prismatic",
        "model_profile": profile,
        "initial_soh_pct": 100.0,
        "final_soh_pct": 93.5,
        "replacement_events": [{"year": 3, "count": 2}],
        "calibration_basis": "cell-model",
        "pack_calibrated": False,
        "experimental_range_warnings": [warnings[0]],
        "aging_horizon_extrapolation_warnings": [warnings[1]],
        "state_schema_version": "1.0",
    }
