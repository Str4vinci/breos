"""Load-profile keys, files and provenance through App, Monte Carlo and the CLI (#182)."""

import json

import numpy as np
import pandas as pd
import pytest

from breos import cli
from breos.app import App
from breos.load_profiles import _btn_day_type
from breos.montecarlo import MonteCarloSettings, run_montecarlo
from breos.runners.app import run_app_simulation

LISBON = "Europe/Lisbon"
BASE = {"location": "porto", "n_modules": 6, "annual_consumption_kwh": 3000, "projection_years": 1}


def _write_measured(path, n_rows=8760, freq="h"):
    stamps = pd.date_range("2025-01-01", periods=n_rows, freq=freq)
    hours = stamps.hour.to_numpy()
    kw = 0.2 + 0.3 * ((hours >= 18) & (hours < 22))
    pd.DataFrame({"meter_kw": kw, "flag": 1}, index=stamps).to_csv(path)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"load_profile": "1"}, r"'1' was removed in BREOS 0\.7\.0; use 'demandlib_h0'"),
        ({"load_profile": "custom", "load_profile_unit": "kW"}, "needs load_profile_file"),
        ({"load_profile": "custom", "load_profile_file": "x.csv"}, "needs load_profile_unit"),
        ({"load_profile": "eredes_btn_c", "load_profile_column": "BTN A - Wh"}, "apply only to load_profile 'custom'"),
        (
            {"load_profile": "custom", "load_profile_file": "x.csv", "load_profile_unit": "MW"},
            "Unknown load_profile_unit",
        ),
    ],
)
def test_app_rejects_profile_config_at_construction(config, message):
    with pytest.raises(ValueError, match=message):
        App({**BASE, **config})


@pytest.mark.usefixtures("_patch_weather")
def test_app_records_the_bundled_profile_in_provenance():
    app = App(BASE)
    app.simulate()
    provenance = app.result()["provenance"]

    assert provenance["resolved_config"]["load_profile"] == "demandlib_h0"
    assert provenance["load_profile"]["key"] == "demandlib_h0"
    assert provenance["load_profile"]["file"] == "h0SLP_demandlib_1000kwh_hourly.csv"
    assert provenance["load_profile"]["packaged"] is True


@pytest.mark.usefixtures("_patch_weather")
def test_app_runs_a_custom_profile_and_records_its_file(tmp_path):
    _write_measured(tmp_path / "meter.csv")
    config = {
        **BASE,
        "load_profile": "custom",
        "rlp_directory": str(tmp_path),
        "load_profile_file": "meter.csv",
        "load_profile_column": "meter_kw",
        "load_profile_unit": "kW",
    }

    app = App(config)
    app.simulate()
    result = app.result()

    assert result["consumption_kwh"] == pytest.approx(3000, abs=0.01)
    record = result["provenance"]["load_profile"]
    assert record["file"] == str((tmp_path / "meter.csv").resolve())
    assert (record["key"], record["column"], record["unit"], record["packaged"]) == ("custom", "meter_kw", "kW", False)
    json.dumps(record, allow_nan=False)


def test_validate_config_prints_the_resolved_profile_file(tmp_path, capsys):
    (tmp_path / "rlp").mkdir()
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        'location = "porto"\nn_modules = 6\nannual_consumption_kwh = 3000\n'
        f'load_profile = "ree_2.0td"\nrlp_directory = "{tmp_path / "rlp"}"\n'
    )

    assert cli.main(["validate-config", str(config_path)]) == 0
    assert "Load profile: ree_2.0td at h, from file not found yet: " in capsys.readouterr().out

    _write_measured(tmp_path / "rlp" / "REE_2026_2.0TD_1000kwh_hourly.csv")
    assert cli.main(["validate-config", str(config_path), "--json"]) == 0
    load = json.loads(capsys.readouterr().out)["load"]
    assert load["load_profile_file"] == str((tmp_path / "rlp" / "REE_2026_2.0TD_1000kwh_hourly.csv").resolve())
    assert load["load_profile_file_error"] is None


def test_validate_config_rejects_several_matching_files(tmp_path, capsys):
    for year in (2025, 2026):
        _write_measured(tmp_path / f"REE_{year}_2.0TD_1000kwh_hourly.csv")
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        'location = "porto"\nn_modules = 6\nannual_consumption_kwh = 3000\n'
        f'load_profile = "ree_2.0td"\nrlp_directory = "{tmp_path}"\n'
    )

    assert cli.main(["validate-config", str(config_path)]) == 1
    assert "2 files" in capsys.readouterr().err


def test_list_load_profiles_shows_canonical_keys_only(capsys):
    assert cli.main(["list", "load-profiles", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)

    assert [row["key"] for row in rows] == [
        "demandlib_h0",
        "eredes_btn_a",
        "eredes_btn_b",
        "eredes_btn_c",
        "bdew_h0",
        "ree_2.0td",
        "custom",
    ]
    assert [row["key"] for row in rows if row["bundled"]] == ["demandlib_h0"]
    assert next(row for row in rows if row["key"] == "bdew_h0")["files"] == ["bdew_h0_*_15min.csv"]


def test_montecarlo_records_the_profile_file(tmp_path, write_multiyear_weather):
    weather = write_multiyear_weather(tmp_path / "multi.csv")
    settings = MonteCarloSettings(weather_file=str(weather), n_runs=2, years_per_run=2, seed=1)

    result = run_montecarlo({**BASE, "battery_kwh": 0}, settings)

    assert result.provenance["load_profile"]["key"] == "demandlib_h0"
    assert result.provenance["load_profile"]["file"] == "h0SLP_demandlib_1000kwh_hourly.csv"
    assert np.isfinite(result.runs["npv_savings"]).all()


def _write_day_class_eredes(path, year=2025, dated=True):
    """A 15-minute E-REDES file whose level on each date is its BTN class in ``year``.

    Working days are 1 Wh per quarter-hour, Saturdays 2 and Sundays and
    national holidays 3, so the class of the source day reaches the App run.
    """
    stamps = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="15min", inclusive="left")
    level = np.array([1.0 + _btn_day_type(day) for day in stamps.normalize()])
    frame = pd.DataFrame({"BTN A - Wh": level, "BTN B - Wh": level, "BTN C - Wh": level})
    if dated:
        frame.insert(0, "DateTime", stamps.strftime("%Y-%m-%d %H:%M"))
    frame.to_csv(path, index=False)


@pytest.mark.usefixtures("_patch_weather")
def test_a_dated_eredes_profile_reaches_a_tou_run_with_the_study_years_day_classes(tmp_path):
    tariff = {
        "schedule": "pt_mainland_2026_weekly_bi",
        "currency": "EUR",
        "import_prices": {"peak": 0.30, "off_peak": 0.10},
        "export_prices": {"all": 0.05},
        "fixed_charge_per_day": 0.25,
    }
    config = {**BASE, "start_date": "2026-01-01", "resolution": "15min", "tariff": tariff}

    def run(directory):
        app = App({**config, "load_profile": "eredes_btn_c", "rlp_directory": str(directory)})
        return run_app_simulation(app._resolved, app._runtime_dependencies())

    classes = {}
    for dated in (True, False):
        directory = tmp_path / ("dated" if dated else "undated")
        directory.mkdir()
        _write_day_class_eredes(directory / "EREDES_2025_BTN_1000kwh_15min.csv", dated=dated)
        artifacts = run(directory)
        frame = artifacts.first_year_results_df
        local = pd.Series(frame["Houseload"].to_numpy(), index=pd.DatetimeIndex(frame["Datetime"]).tz_convert(LISBON))
        daily = local.groupby(local.index.date).mean()
        classes[dated] = (np.rint(daily / daily.min()) - 1).astype(int)
        classes[dated, "bill"] = artifacts.yearly_df["Baseline_Import_Cost"].iloc[0]

    expected = pd.Series([_btn_day_type(pd.Timestamp(day)) for day in classes[True].index], index=classes[True].index)
    # Every 2026 date carries a 2025 source day of its own class, holidays included.
    pd.testing.assert_series_equal(classes[True], expected, check_names=False)
    # Placed by position, the 2025 rows start on a Wednesday and land one weekday late.
    assert (classes[False] != expected).sum() > 100
    assert classes[True, "bill"] != pytest.approx(classes[False, "bill"], rel=1e-4)
