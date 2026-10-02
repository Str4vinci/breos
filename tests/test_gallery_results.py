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
# identity across machines: a rounded value can move by one unit in the last
# decimal place it is written with (CI runners without AVX-512 move 24 of
# first_home's values by 0.01). --cross-machine accepts that and nothing more;
# the maintainer's own --check runs at 1e-9.
def test_check_reproduces_the_quickstart_case(capsys):
    manifest = json.loads((RESULTS / "first_home" / "manifest.json").read_text(encoding="utf-8"))
    if not _same_stack(manifest):
        pytest.skip("the stored results were made with other numpy/pandas/pvlib/scipy versions")
    assert TOOL.main(["--check", "first_home", "--cross-machine"]) == 0, capsys.readouterr().out


@pytest.mark.slow
def test_check_reproduces_every_cheap_case(capsys):
    manifest = json.loads((RESULTS / "first_home" / "manifest.json").read_text(encoding="utf-8"))
    if not _same_stack(manifest):
        pytest.skip("the stored results were made with other numpy/pandas/pvlib/scipy versions")
    assert TOOL.main(["--check", "--cross-machine"]) == 0, capsys.readouterr().out


def _compare(tmp_path, stored, fresh, suffix, cross_machine=True):
    paths = []
    for name, content in (("stored", stored), ("fresh", fresh)):
        path = tmp_path / name / f"case{suffix}"
        path.parent.mkdir(exist_ok=True)
        if suffix == ".json":
            path.write_text(json.dumps(content, indent=1), encoding="utf-8")
        else:
            pd.DataFrame(content).to_csv(path, index=False)
        paths.append(path)
    return TOOL.compare_file(*paths, cross_machine=cross_machine)


@pytest.mark.parametrize(
    ("stored", "fresh", "accepted"),
    [
        ({"npv_savings": 4735.25}, {"npv_savings": 4735.26}, True),
        ({"npv_savings": 4735.25}, {"npv_savings": 4735.27}, False),
        ({"npv_savings": 5000.0}, {"npv_savings": 5004.0}, False),
        # The field's precision is its most decimals in either file, not one value's trailing zeros.
        ({"rows": [{"kwh": 4000.0}, {"kwh": 812.34}]}, {"rows": [{"kwh": 4000.1}, {"kwh": 812.34}]}, False),
        ({"rows": [{"kwh": 4000.0}, {"kwh": 812.34}]}, {"rows": [{"kwh": 4000.01}, {"kwh": 812.34}]}, True),
        ({"lcoe_per_kwh": 0.1015}, {"lcoe_per_kwh": 0.1016}, True),
        ({"lcoe_per_kwh": 0.1015}, {"lcoe_per_kwh": 0.1025}, False),
        ({"payback_year": 13}, {"payback_year": 14}, False),
        ({"battery_replacements": 1}, {"battery_replacements": 2}, False),
        ({"bill_year1": 512.40}, {"bill_year1": 512.43}, True),
        ({"bill_year1": 512.40}, {"bill_year1": 512.44}, False),
        ({"residual_kwh": -0.0}, {"residual_kwh": 0.0}, True),
        ({"method": "repriced"}, {"method": "resimulated"}, False),
        ({"credit": None}, {"credit": 0.0}, False),
    ],
)
def test_cross_machine_json_allows_one_unit_in_the_last_written_place(tmp_path, stored, fresh, accepted):
    assert (_compare(tmp_path, stored, fresh, ".json") == []) is accepted


@pytest.mark.parametrize(
    ("column", "stored", "fresh", "accepted"),
    [
        ("Battery_SOC_Normalized", [0.5, 0.5123], [0.5001, 0.5123], True),
        ("Battery_SOC_Normalized", [0.5, 0.5123], [0.54, 0.5123], False),
        ("PV_Production", [1234.5, 0.0], [1234.6, 0.0], True),
        ("PV_Production", [1234.5, 0.0], [1234.7, 0.0], False),
        # Floats only because a missing value forces the column to float: still a count.
        ("payback_year", [13.0, None], [14.0, None], False),
        ("payback_year", [13.0, None], [13.0, 12.0], False),
        ("storage_cost_per_kwh", [150, 175], [151, 175], False),
    ],
)
def test_cross_machine_csv_allows_one_unit_in_the_last_written_place(tmp_path, column, stored, fresh, accepted):
    assert (_compare(tmp_path, {column: stored}, {column: fresh}, ".csv") == []) is accepted


def test_same_machine_check_allows_no_last_place_change(tmp_path):
    assert _compare(tmp_path, {"npv_savings": 4735.25}, {"npv_savings": 4735.26}, ".json", cross_machine=False)
    assert not _compare(tmp_path, {"npv_savings": 4735.25}, {"npv_savings": 4735.25}, ".json", cross_machine=False)


def _gallery_helpers():
    path = PROJECT_ROOT / "docs" / "_ext" / "gallery_results.py"
    spec = importlib.util.spec_from_file_location("gallery_results", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("value", "decimals", "text"),
    [(5000.4, 0, "5000"), (12345.4, 0, "12 345"), (-123456.78, 1, "−123 456.8"), (2026, 0, "2026")],
)
def test_gallery_numbers_group_digits_from_five_up(value, decimals, text):
    assert _gallery_helpers().number(value, decimals) == text


def test_gallery_money_and_tables_never_use_commas():
    helpers = _gallery_helpers()
    assert helpers.money(-6077.4) == "−6077 EUR"
    assert helpers.money(-0.3) == "0 EUR"
    markup = helpers.table(pd.DataFrame({"NPV": [12345.6, 999.0]}), NPV=",.0f").markup
    assert "12 346" in markup and "999" in markup and "," not in markup
