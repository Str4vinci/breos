"""Static checks for Article 1 input and result-bundle verification tools."""

import hashlib

import pytest

from tools.preflight_article1_inputs import EXPECTED_SHA256, _checked_input
from tools.verify_article1_bundle import BundleAudit


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
