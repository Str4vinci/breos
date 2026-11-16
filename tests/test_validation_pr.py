"""Checks for the public AC path, shared independent POA, and PR report columns."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from validation import common, compare, run_breos
from validation.common import load_spec


def test_runner_uses_public_ac_and_computes_poa_once(monkeypatch, synthetic_weather):
    system, locations = load_spec()
    poa = Mock(return_value=2000.0)
    ac = Mock(return_value=synthetic_weather["ghi"])
    monkeypatch.setattr(run_breos, "load_validation_weather", lambda path: synthetic_weather)
    monkeypatch.setattr(run_breos, "compute_annual_poa", poa)
    monkeypatch.setattr(common, "calculate_pv_production_ac", ac)
    monkeypatch.setattr(run_breos, "MODEL_CONFIGS", {k: common.MODEL_CONFIGS[k] for k in ("isotropic", "perez")})
    result = run_breos.run_location("porto", locations["porto"], system, Path("weather.csv"))
    poa.assert_called_once()
    assert ac.call_count == 2
    for call, model in zip(ac.call_args_list, result["models"].values()):
        assert call.kwargs["inverter_loading_ratio"] == system["dc_ac_ratio"]
        assert call.kwargs["inverter_efficiency"] == system["inverter_efficiency"]
        assert model["annual_poa_kwh_m2"] == 2000.0
        assert model["dc_nameplate_kw"] == 4.0


@pytest.fixture
def report_results(monkeypatch):
    monkeypatch.setattr(
        compare,
        "load_reference",
        lambda key: {
            "pvwatts_v8": {
                "dataset": "nsrdb",
                "annual_kwh": 5000,
                "solrad_annual_kwh_m2_day": 5,
                "params": {"system_capacity": 4.0},
                "trusted": True,
            }
        },
    )
    model = {
        "annual_kwh": 5600,
        "monthly_kwh": [100.0] * 12,
        "annual_poa_kwh_m2": 2000,
        "dc_nameplate_kw": 4.0,
    }
    return {
        "breos_version": "test",
        "locations": {"porto": {"models": {"isotropic": dict(model), "perez": dict(model)}}},
    }


def test_pr_columns_and_writer(report_results, monkeypatch, tmp_path, capsys):
    rows = compare.build_rows(report_results)
    assert rows[0]["breos_pr"] == pytest.approx(0.7)
    assert rows[0]["pvwatts_pr"] == pytest.approx(5000 / (5 * 365 * 4))
    assert rows[0]["pvwatts_pr_dev"] == pytest.approx(2.2)
    assert rows[0]["pvwatts_dev"] == pytest.approx(12.0)
    compare.print_table(rows, report_results)
    output = capsys.readouterr().out
    assert "PR BREOS" in output and "PR PVWatts" in output and "Δ PR" in output
    assert "0.700" in output and "0.685" in output and "+2.2%" in output

    report_path = tmp_path / "report.md"
    monkeypatch.setattr(compare, "REPORT_PATH", report_path)
    monkeypatch.setattr(compare, "PROJECT_ROOT", tmp_path)
    compare.write_report(rows, report_results)
    report = report_path.read_text()
    assert "| PR BREOS | PR PVWatts | Δ PR |" in report
    assert "| 0.700 | 0.685 | +2.2% |" in report
    assert "±8%" in report and "±10%" in report
    assert "not the banded quantity" in report
    annual_section = report.split("## Annual AC yield (kWh)")[1].split("## Equal-nameplate")[0]
    table = [
        line for line in annual_section.splitlines() if line.startswith("| Location") or line.startswith("| porto")
    ]
    assert len(table) == 2
    assert table[0].count("|") == table[1].count("|") == 21


def test_older_results_without_poa_show_missing_pr(report_results):
    del report_results["locations"]["porto"]["models"]["perez"]["annual_poa_kwh_m2"]
    row = compare.build_rows(report_results)[0]
    assert row["breos_pr"] is None
    assert row["pvwatts_pr_dev"] is None
    assert row["pvwatts_pr"] is not None
    assert "run validation/run_breos.py" in row["notes"]


def test_report_uses_each_models_own_nameplate(report_results):
    report_results["locations"]["porto"]["models"]["perez"]["dc_nameplate_kw"] = 2.0
    row = compare.build_rows(report_results)[0]
    assert row["breos_pr"] == pytest.approx(1.4)
    assert row["pvwatts_pr"] == pytest.approx(5000 / (5 * 365 * 4))
    assert "BREOS 2.0 kW, PVWatts 4.0 kW" in row["notes"]
