"""Static checks for Article 1 input and result-bundle verification tools."""

import hashlib

import pytest

from tools.preflight_article1_inputs import EXPECTED_SHA256, _checked_input
from tools.verify_article1_bundle import BundleAudit, _report_source_paths


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
