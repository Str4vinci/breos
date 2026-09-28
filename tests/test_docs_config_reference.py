"""The generated configuration key reference must match the config registry."""

import subprocess
import sys
from pathlib import Path

import pytest

from breos.app_config import APP_CONFIG_FIELDS, NESTED_TABLE_SPECS, PV_ARRAY_TABLE

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_config_reference_matches_registry():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "generate_config_docs.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout


@pytest.mark.parametrize("name", sorted(APP_CONFIG_FIELDS))
def test_every_top_level_key_has_doc_text(name):
    assert APP_CONFIG_FIELDS[name].doc.strip()


@pytest.mark.parametrize("spec", [*NESTED_TABLE_SPECS.values(), PV_ARRAY_TABLE], ids=lambda spec: spec.name)
def test_every_nested_table_key_has_doc_text(spec):
    assert set(spec.docs) == set(spec.keys)
    assert all(text.strip() for text in spec.docs.values())
