"""The example gallery's stored results: present, intact, loadable, and reproducible.

The docs build only loads what ``tools/regenerate_gallery_results.py`` stored
under ``docs/examples/_results``. These checks need no simulation, except the
``--check`` rerun of one cheap case.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from importlib import metadata
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "docs" / "examples" / "_results"
PAGES = sorted((PROJECT_ROOT / "docs" / "examples").glob("*/plot_*.py"))
MANIFEST_KEYS = {
    "case",
    "title",
    "breos_version",
    "git_describe",
    "generated_utc",
    "runtime_s",
    "cheap",
    "configs",
    "weather",
    "dependencies",
    "files",
}


def _tool():
    path = PROJECT_ROOT / "tools" / "regenerate_gallery_results.py"
    spec = importlib.util.spec_from_file_location("regenerate_gallery_results", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOL = _tool()


@pytest.mark.parametrize("name", list(TOOL.CASES))
def test_every_case_has_an_intact_manifest(name):
    directory = RESULTS / name
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert MANIFEST_KEYS <= manifest.keys()
    assert manifest["case"] == name
    assert manifest["cheap"] == TOOL.CASES[name].cheap
    for config in manifest["configs"]:
        assert (PROJECT_ROOT / config).is_file(), config
    for weather, entry in manifest["weather"].items():
        assert entry["sha256"] and entry["attribution"]
        if weather.startswith("validation/"):
            assert hashlib.sha256((PROJECT_ROOT / weather).read_bytes()).hexdigest() == entry["sha256"]
    stored = {path.name for path in directory.iterdir()} - {"manifest.json"}
    assert stored == set(manifest["files"])
    for filename, digest in manifest["files"].items():
        data = (directory / filename).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, f"{name}/{filename} changed after it was stored"
        if filename.endswith(".json"):
            json.loads(data)
        else:
            assert len(pd.read_csv(directory / filename)) > 0


def test_stored_results_hold_no_local_paths():
    for path in RESULTS.rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            assert not re.search(r"(/home/|/Users/|/tmp/|[A-Za-z]:\\\\)", text), path


def test_stored_results_stay_small():
    total = sum(path.stat().st_size for path in RESULTS.rglob("*") if path.is_file())
    assert total < 400 * 1024, f"{total / 1024:.0f} KB of stored gallery results"


def test_every_page_loads_a_stored_case():
    assert PAGES
    loaded = set()
    for page in PAGES:
        names = re.findall(r'load_case\("([a-z0-9_]+)"\)', page.read_text(encoding="utf-8"))
        assert names, f"{page.name} loads no stored case"
        loaded.update(names)
    assert loaded <= set(TOOL.CASES)
    assert set(TOOL.CASES) <= loaded, f"stored cases no page shows: {set(TOOL.CASES) - loaded}"


def test_every_page_loads_its_results():
    sys.path.insert(0, str(PROJECT_ROOT / "docs" / "_ext"))
    try:
        from gallery_results import load_case
    finally:
        sys.path.pop(0)
    for name in TOOL.CASES:
        case = load_case(name)
        for filename in case.manifest["files"]:
            if filename.endswith(".json"):
                assert case.json(filename) is not None
            else:
                assert not case.csv(filename).empty
        assert "Stored results" in case.stamp()._repr_html_()


def _same_stack(manifest) -> bool:
    recorded = manifest["dependencies"]
    for name in ("numpy", "pandas", "pvlib", "scipy"):
        try:
            if metadata.version(name) != recorded.get(name):
                return False
        except metadata.PackageNotFoundError:
            return False
    return True


# The stored results come from another machine. BREOS does not promise bit
# identity across machines: over 20 projected years a rounded value can move
# by its last digit (0.01). These tolerances accept that and nothing more a
# page would show; the maintainer's own --check runs at 1e-9.
CROSS_MACHINE = ["--rtol", "1e-3", "--atol", "0.05"]


def test_check_reproduces_the_quickstart_case(capsys):
    manifest = json.loads((RESULTS / "first_home" / "manifest.json").read_text(encoding="utf-8"))
    if not _same_stack(manifest):
        pytest.skip("the stored results were made with other numpy/pandas/pvlib/scipy versions")
    assert TOOL.main(["--check", "first_home", *CROSS_MACHINE]) == 0, capsys.readouterr().out


@pytest.mark.slow
def test_check_reproduces_every_cheap_case(capsys):
    manifest = json.loads((RESULTS / "first_home" / "manifest.json").read_text(encoding="utf-8"))
    if not _same_stack(manifest):
        pytest.skip("the stored results were made with other numpy/pandas/pvlib/scipy versions")
    assert TOOL.main(["--check", *CROSS_MACHINE]) == 0, capsys.readouterr().out
