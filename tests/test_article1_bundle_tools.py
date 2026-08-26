"""Static checks for Article 1 input and result-bundle verification tools."""

import hashlib

import pytest

from tools.preflight_article1_inputs import EXPECTED_SHA256, _checked_input
from tools.verify_article1_bundle import (
    EXPECTED_EXTERNAL_VALIDATION_HASHES,
    EXPECTED_HISTORICAL_WEATHER_SHA256,
    EXPECTED_MONTE_CARLO_YEARLY_COLUMNS,
    BundleAudit,
    _report_source_paths,
)

CORRECTED_VALIDATION_HASHES = {
    "monthly_results.csv": "84cce17d0d51f745895bad1c98adaa3ef3d6043f076ce69d3a1dff5ecad1a526",
    "weekly_results.csv": "07c7b938da100ca5c5301a0ef9a6988b24616f7e5eb5166408056afd1ae79375",
    "daily_results.csv": "2d12468d982f0b59af875841bf8b9e228a532c6f4060070245907bbb243b0bfb",
}


def test_article1_tools_pin_corrected_validation_inputs():
    assert EXPECTED_EXTERNAL_VALIDATION_HASHES == CORRECTED_VALIDATION_HASHES
    assert {
        filename: EXPECTED_SHA256[f"validation_{filename.removesuffix('_results.csv')}"]
        for filename in CORRECTED_VALIDATION_HASHES
    } == CORRECTED_VALIDATION_HASHES


ARTICLE1_INSTANT_WEATHER_SHA256 = "0b2d42e6f3e2309aed3c0f65de461cab11b885173f6e45f0cd93adee29417650"


def test_article1_tools_pin_instant_openmeteo_weather():
    assert EXPECTED_SHA256["historical_weather"] == ARTICLE1_INSTANT_WEATHER_SHA256
    assert EXPECTED_HISTORICAL_WEATHER_SHA256 == ARTICLE1_INSTANT_WEATHER_SHA256


def test_article1_input_preflight_accepts_only_pinned_hash(tmp_path, monkeypatch):
    path = tmp_path / "input.csv"
    path.write_text("value\n1\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setitem(EXPECTED_SHA256, "test_input", digest)

    record = _checked_input("test_input", path)

    assert record["sha256"] == digest
    assert record["size_bytes"] == path.stat().st_size


def test_article1_input_preflight_rejects_hash_drift(tmp_path, monkeypatch):
    path = tmp_path / "input.csv"
    path.write_text("changed\n")
    monkeypatch.setitem(EXPECTED_SHA256, "test_input", "0" * 64)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _checked_input("test_input", path)


def test_article1_bundle_audit_reports_missing_files(tmp_path):
    audit = BundleAudit(tmp_path)

    audit.require_file("missing.csv")

    assert audit.errors == ["missing file: missing.csv"]


def test_article1_bundle_accepts_exact_monte_carlo_yearly_schema(tmp_path):
    path = tmp_path / "monte-carlo-v1/c1/yearly.csv"
    path.parent.mkdir(parents=True)
    path.write_text(",".join(EXPECTED_MONTE_CARLO_YEARLY_COLUMNS) + "\n")
    audit = BundleAudit(tmp_path)

    audit.verify_monte_carlo_yearly_schema("c1")

    assert audit.errors == []


@pytest.mark.parametrize(
    ("columns", "expected_error"),
    [
        (EXPECTED_MONTE_CARLO_YEARLY_COLUMNS[:-1], "missing ['Marginal_Grid_CI_gCO2_kWh']"),
        ((*EXPECTED_MONTE_CARLO_YEARLY_COLUMNS, "Extra"), "unexpected ['Extra']"),
        (
            (
                EXPECTED_MONTE_CARLO_YEARLY_COLUMNS[1],
                EXPECTED_MONTE_CARLO_YEARLY_COLUMNS[0],
                *EXPECTED_MONTE_CARLO_YEARLY_COLUMNS[2:],
            ),
            "column order differs",
        ),
    ],
)
def test_article1_bundle_rejects_monte_carlo_yearly_schema_drift(tmp_path, columns, expected_error):
    path = tmp_path / "monte-carlo-v1/c1/yearly.csv"
    path.parent.mkdir(parents=True)
    path.write_text(",".join(columns) + "\n")
    audit = BundleAudit(tmp_path)

    audit.verify_monte_carlo_yearly_schema("c1")

    assert any(expected_error in error for error in audit.errors)


def test_article1_source_compatibility_uses_the_generator_for_each_report():
    assert "tools/reproduce_article1.py" in _report_source_paths("base-v1/reproduction.json")
    assert "tools/reproduce_article1_context.py" in _report_source_paths("orientation/provenance.json")
    assert "tools/reproduce_article1_montecarlo.py" in _report_source_paths("monte-carlo-v1/c2/provenance.json")


def test_article1_bundle_accepts_unaffected_ancestor_outputs(tmp_path, monkeypatch):
    audit = BundleAudit(tmp_path)
    audit.reports = [
        (
            tmp_path / "base-v1/reproduction.json",
            {
                "breos_source": {"commit": "old"},
                "dependency_versions": {"numpy": "1"},
            },
        ),
        (
            tmp_path / "monte-carlo-v1/c2/provenance.json",
            {
                "breos_source": {"commit": "new"},
                "dependency_versions": {"numpy": "1"},
            },
        ),
    ]
    monkeypatch.setattr(audit, "_is_ancestor", lambda commit, target: commit in {"old", "new"} and target == "new")
    monkeypatch.setattr(audit, "_numerical_source_changed", lambda commit, target: False)
    monkeypatch.setattr(audit, "_source_object", lambda commit, path: f"unchanged:{path}")

    audit._verify_source_compatibility({"breos_source": {"commit": "new"}})

    assert audit.errors == []
    assert audit.source_commits == {"old", "new"}


def test_article1_bundle_accepts_reports_with_compatible_dependency_subsets(tmp_path, monkeypatch):
    audit = BundleAudit(tmp_path)
    audit.reports = [
        (
            tmp_path / "base-v1/reproduction.json",
            {
                "breos_source": {"commit": "old"},
                "dependency_versions": {"numpy": "1", "pymoo": "2"},
            },
        ),
        (
            tmp_path / "orientation/provenance.json",
            {
                "breos_source": {"commit": "new"},
                "dependency_versions": {"numpy": "1"},
            },
        ),
    ]
    monkeypatch.setattr(audit, "_is_ancestor", lambda commit, target: True)
    monkeypatch.setattr(audit, "_numerical_source_changed", lambda commit, target: False)
    monkeypatch.setattr(audit, "_source_object", lambda commit, path: f"same:{path}")

    audit._verify_source_compatibility({"breos_source": {"commit": "new"}})

    assert audit.errors == []


def test_article1_bundle_rejects_conflicting_dependency_versions(tmp_path, monkeypatch):
    audit = BundleAudit(tmp_path)
    audit.reports = [
        (
            tmp_path / "base-v1/reproduction.json",
            {
                "breos_source": {"commit": "old"},
                "dependency_versions": {"numpy": "1"},
            },
        ),
        (
            tmp_path / "orientation/provenance.json",
            {
                "breos_source": {"commit": "new"},
                "dependency_versions": {"numpy": "2"},
            },
        ),
    ]
    monkeypatch.setattr(audit, "_is_ancestor", lambda commit, target: True)
    monkeypatch.setattr(audit, "_numerical_source_changed", lambda commit, target: False)
    monkeypatch.setattr(audit, "_source_object", lambda commit, path: f"same:{path}")

    audit._verify_source_compatibility({"breos_source": {"commit": "new"}})

    assert any("conflicting dependency versions" in error for error in audit.errors)


def test_article1_bundle_rejects_changed_generator_for_ancestor_output(tmp_path, monkeypatch):
    audit = BundleAudit(tmp_path)
    audit.reports = [
        (
            tmp_path / "base-v1/reproduction.json",
            {"breos_source": {"commit": "old"}},
        )
    ]
    monkeypatch.setattr(audit, "_is_ancestor", lambda commit, target: True)
    monkeypatch.setattr(audit, "_numerical_source_changed", lambda commit, target: False)
    monkeypatch.setattr(
        audit,
        "_source_object",
        lambda commit, path: f"{commit}:{path}" if path == "tools/reproduce_article1.py" else f"same:{path}",
    )

    audit._verify_source_compatibility({"breos_source": {"commit": "new"}})

    assert any("tools/reproduce_article1.py differs" in error for error in audit.errors)


def test_article1_bundle_rejects_changed_numerical_source(tmp_path, monkeypatch):
    audit = BundleAudit(tmp_path)
    audit.reports = [
        (
            tmp_path / "base-v1/reproduction.json",
            {"breos_source": {"commit": "old"}},
        )
    ]
    monkeypatch.setattr(audit, "_is_ancestor", lambda commit, target: True)
    monkeypatch.setattr(audit, "_numerical_source_changed", lambda commit, target: True)
    monkeypatch.setattr(audit, "_source_object", lambda commit, path: f"same:{path}")

    audit._verify_source_compatibility({"breos_source": {"commit": "new"}})

    assert any("numerical source differs" in error for error in audit.errors)
