"""Tests for the maintained BLAST parity-fixture generation process."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from breos.degradation.engine import BLAST_MODEL_CLASSES
from breos.degradation.profiles import BLAST_UPSTREAM_COMMIT, BLAST_UPSTREAM_VERSION
from tools.generate_blast_parity_fixture import (
    FIXTURE_PATH,
    MANIFEST_PATH,
    MODEL_CLASS_NAMES,
    PINNED_BLAST_COMMIT,
    PINNED_BLAST_VERSION,
    ParityGenerationError,
    build_profiles,
    main,
    validate_checkout,
    validate_manifest,
)


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _minimal_blast_checkout(tmp_path: Path) -> tuple[Path, str]:
    checkout = tmp_path / "BLAST-Lite"
    (checkout / "blast" / "models").mkdir(parents=True)
    (checkout / "blast" / "models" / "__init__.py").write_text("# test checkout\n", encoding="utf-8")
    _git(checkout, "init", "--quiet")
    _git(checkout, "config", "user.name", "BREOS tests")
    _git(checkout, "config", "user.email", "tests@example.invalid")
    _git(checkout, "add", "blast/models/__init__.py")
    _git(checkout, "commit", "--quiet", "-m", "test checkout")
    return checkout, _git(checkout, "rev-parse", "HEAD")


def test_committed_manifest_binds_fixture_and_generator_definitions():
    fixture_bytes = FIXTURE_PATH.read_bytes()
    fixture = json.loads(fixture_bytes)
    manifest = json.loads(MANIFEST_PATH.read_bytes())

    validate_manifest(fixture, fixture_bytes, manifest)
    assert manifest["source"]["commit"] == BLAST_UPSTREAM_COMMIT
    assert manifest["source"]["version"] == BLAST_UPSTREAM_VERSION
    assert build_profiles() == {name: condition["profile"] for name, condition in fixture["conditions"].items()}
    assert set(MODEL_CLASS_NAMES) == set(BLAST_MODEL_CLASSES)
    assert PINNED_BLAST_COMMIT == BLAST_UPSTREAM_COMMIT
    assert PINNED_BLAST_VERSION == BLAST_UPSTREAM_VERSION


def test_manifest_validation_detects_fixture_drift():
    fixture_bytes = FIXTURE_PATH.read_bytes()
    fixture = json.loads(fixture_bytes)
    manifest = json.loads(MANIFEST_PATH.read_bytes())
    fixture["conditions"]["hot_storage"]["profile"]["soc"][0] = 0.8
    drifted_bytes = (json.dumps(fixture) + "\n").encode()

    with pytest.raises(ParityGenerationError, match="does not match fixture"):
        validate_manifest(fixture, drifted_bytes, manifest)


def test_checkout_validation_requires_clean_expected_commit(tmp_path):
    checkout, commit = _minimal_blast_checkout(tmp_path)

    assert validate_checkout(checkout, expected_commit=commit) == (checkout.resolve(), commit)

    with pytest.raises(ParityGenerationError, match="unexpected commit"):
        validate_checkout(checkout, expected_commit="0" * 40)
    assert validate_checkout(
        checkout,
        expected_commit="0" * 40,
        allow_unexpected_commit=True,
    ) == (checkout.resolve(), commit)

    (checkout / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ParityGenerationError, match="must be clean"):
        validate_checkout(checkout, expected_commit=commit)


def test_checkout_validation_requires_repository_root(tmp_path):
    checkout, commit = _minimal_blast_checkout(tmp_path)

    with pytest.raises(ParityGenerationError, match="repository root"):
        validate_checkout(checkout / "blast", expected_commit=commit)


def test_cli_requires_source_version_for_commit_override(tmp_path, capsys):
    checkout, _ = _minimal_blast_checkout(tmp_path)

    assert main(["--blast-checkout", str(checkout), "--allow-unexpected-commit"]) == 2
    assert "--source-version is required" in capsys.readouterr().err
