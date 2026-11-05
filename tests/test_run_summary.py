"""Tests for the summary that ``breos run`` prints on a terminal."""

import json
import sys
import warnings
from contextlib import contextmanager
from unittest import mock

import pytest

from breos import cli
from breos._run_summary import _number, format_run_summary
from tests.conftest import INSTANT_AT_LABELS
from tools.generate_app_golden import synthetic_weather

CONFIG = """\
location = "porto"
n_modules = 8
annual_consumption_kwh = 4000
battery_kwh = 5
start_date = "2025-01-01"
projection_years = 2
cost_preset = "residential_pt"
execution_backend = "python"
"""


@contextmanager
def _offline_weather():
    """Serve the golden synthetic year as the PVGIS fetch, offline."""

    def fetch(*_args, **kwargs):
        weather = synthetic_weather(int(kwargs.get("sample_year") or 2025)).copy()
        weather.attrs["breos_weather_metadata"] = {
            "source": "synthetic",
            **INSTANT_AT_LABELS,
            "horizon": {"status": "not_applied", "provider": "pvgis", "profile": None},
        }
        return weather, {"inputs": {"location": {"latitude": 41.1579, "longitude": -8.6291, "elevation": 0}}}

    with (
        mock.patch("breos.app.fetch_tmy_weather_data", fetch),
        mock.patch("breos.app.load_weather", lambda **_kwargs: None),
    ):
        yield


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "home.toml"
    path.write_text(CONFIG, encoding="utf-8")
    return path


def _run_cli(args, capsys, monkeypatch, *, terminal):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: terminal)
    with _offline_weather():
        assert cli.main(["run", *args]) == 0
    return capsys.readouterr()


def test_a_terminal_gets_the_summary_and_a_pipe_gets_the_json(config_path, capsys, monkeypatch):
    piped = _run_cli(["--config", str(config_path)], capsys, monkeypatch, terminal=False).out
    shown = _run_cli(["--config", str(config_path)], capsys, monkeypatch, terminal=True).out
    result = json.loads(piped)

    assert shown.splitlines()[0].startswith(f"BREOS {result['provenance']['breos_version']}, home.toml, Porto")
    assert f"  PV generation (DC)  {_number(result['pv_dc_generation_kwh']):>14} kWh" in shown
    assert f"{_number(result['grid_independence_pct'], 1)} % grid independence" in shown
    assert f"Net present value  {_number(result['npv_savings']):>15} EUR" in shown
    assert "Battery over 2 years" in shown
    assert shown.rstrip().endswith("or pipe the command to get JSON.")
    assert max(len(line) for line in shown.splitlines()) <= 80
    assert "," not in shown.replace(", ", "")


def test_json_flag_prints_the_same_json_on_a_terminal(config_path, capsys, monkeypatch):
    piped = _run_cli(["--config", str(config_path)], capsys, monkeypatch, terminal=False).out
    forced = _run_cli(["--config", str(config_path), "--json"], capsys, monkeypatch, terminal=True).out

    assert forced == piped


def test_output_file_is_unchanged_and_the_terminal_names_it(config_path, tmp_path, capsys, monkeypatch):
    quiet, shown = tmp_path / "quiet.json", tmp_path / "shown.json"
    assert (
        _run_cli(["--config", str(config_path), "--output", str(quiet)], capsys, monkeypatch, terminal=False).out == ""
    )
    out = _run_cli(["--config", str(config_path), "--output", str(shown)], capsys, monkeypatch, terminal=True).out

    assert shown.read_bytes() == quiet.read_bytes()
    assert out.rstrip().endswith(f"Full results written to {shown} (JSON, result schema 1).")


def test_a_period_run_has_no_lifetime_blocks(tmp_path, capsys, monkeypatch):
    path = tmp_path / "week.toml"
    path.write_text(CONFIG + '\n[period]\nstart = "2025-06-01"\nend = "2025-06-08"\n', encoding="utf-8")
    shown = _run_cli(["--config", str(path)], capsys, monkeypatch, terminal=True).out

    assert "Energy, 2025-06-01 to 2025-06-08 (end exclusive, 7 days)" in shown
    assert "Year 1" not in shown
    assert "Battery at the end of the period" in shown
    assert "Economics: skipped, a period shorter than a year has no project lifetime." in shown
    assert "Net present value" not in shown


@pytest.mark.parametrize(
    ("value", "decimals", "text"),
    [(5000, 0, "5000"), (12345, 0, "12 345"), (9999.6, 0, "10 000"), (-0.4, 0, "0"), (-12345.678, 2, "-12 345.68")],
)
def test_numbers_have_no_commas(value, decimals, text):
    assert _number(value, decimals) == text


def test_vehicle_rows_replace_grid_independence():
    result = {
        "consumption_kwh": 4000.0,
        "grid_import_kwh": 4202.4,
        "grid_independence_pct": -5.06,
        "npv_savings": -6497.6,
        "payback_year": None,
        "npv_savings_household_adjusted": -6498.9,
        "household_adjusted_comparable": True,
        "vehicle_benefit_npv": -2496.6,
        "vehicle_benefit_comparable": False,
        "yearly": [{}, {}, {}],
        "provenance": {"currency": "EUR"},
        "vehicle": {
            "mode": "v2h",
            "reference_mode": "v1g",
            "households": {"R2": {"yearly": [{"pv_ac_charge_kwh": 758.2, "grid_ac_charge_kwh": 3485.9}]}},
        },
    }
    shown = format_run_summary(result)

    assert "Vehicle charging at home      4244 kWh       758 kWh from PV" in shown
    assert "house and vehicle" in shown
    assert "grid independence" not in shown
    assert "Discounted payback        not within 3 years" in shown
    assert "V2H benefit vs V1G           -2497 EUR       not comparable" in shown


def test_missing_values_leave_their_lines_out():
    assert format_run_summary({"grid_independence_pct": None, "pv_kwp": "x"}).splitlines() == [
        "BREOS",
        "",
        "Full results: add --output result.json, or pipe the command to get JSON.",
    ]


def test_breos_warnings_print_as_one_line_and_others_keep_their_source():
    def shown(filename):
        return cli._format_breos_warning(warnings.formatwarning, "check this", UserWarning, filename, 7, "code()")

    assert shown(cli.__file__) == "breos: warning: check this\n"
    assert shown("/somewhere/else.py") == "/somewhere/else.py:7: UserWarning: check this\n  code()\n"


def test_the_cli_restores_the_warning_format_when_it_returns(config_path, capsys, monkeypatch):
    before = warnings.formatwarning
    _run_cli(["--config", str(config_path), "--dry-run"], capsys, monkeypatch, terminal=True)

    assert warnings.formatwarning is before
