"""Tests for the public degradation-result schema builder."""

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
