"""A [period] window shorter than a year in App (#242).

The window is a pair of civil dates in the location's timezone, end
exclusive. The full-year load is built and scaled to annual_consumption_kwh,
then weather, PV, load and battery temperature are cut to the window, which
runs once. Lifetime economics are None for it; energy covers the window.
"""

import json
import warnings
from contextlib import contextmanager
from datetime import date, datetime
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from breos import App, cli
from breos.app_config import resolve_app_config
from breos.app_inputs import input_configuration_key
from breos.load_profiles import load_profile
from breos.montecarlo import MonteCarloSettings, build_year_cache, run_montecarlo
from breos.optimization_config import resolve_optimization_config
from tools.generate_app_golden import synthetic_weather

BASE = {
    "location": "porto",
    "n_modules": 8,
    "annual_consumption_kwh": 4000,
    "start_date": "2025-01-01",
    "cost_preset": "residential_pt",
    "emissions_country": "PT",
    "execution_backend": "python",
}
JUNE_WEEK = {"start": "2025-06-01", "end": "2025-06-08"}
LISBON = "Europe/Lisbon"
TOU = {
    "schedule": "pt_mainland_2026_daily_bi",
    "currency": "EUR",
    "import_prices": {"peak": 0.28, "off_peak": 0.11},
    "export_prices": {"all": 0.05},
    "fixed_charge_per_day": 0.25,
}
# Every per-step flow of a PV-only run: with no battery state, each step
# depends on its own inputs only.
FLOWS = ("PV_DC", "PV_Production", "PV_AC_To_Load", "PV_AC_Export", "PV_DC_Curtailed", "Houseload", "Import_From_Grid")


@contextmanager
def _weather(frame=None):
    """Serve ``frame`` (default the golden synthetic year) as the PVGIS fetch, offline."""

    def fetch(*_args, **kwargs):
        weather = (synthetic_weather(int(kwargs.get("sample_year") or 2025)) if frame is None else frame).copy()
        weather.attrs["breos_weather_metadata"] = {
            "source": "synthetic_period",
            "horizon": {"status": "not_applied", "provider": "pvgis", "profile": None},
        }
        return weather, {"inputs": {"location": {"latitude": 41.1579, "longitude": -8.6291, "elevation": 0}}}

    with (
        mock.patch("breos.app.fetch_tmy_weather_data", fetch),
        mock.patch("breos.app.load_weather", lambda **_kwargs: None),
    ):
        yield


def _run(config, weather=None):
    with _weather(weather):
        app = App(config)
        app.simulate()
    return app


def _window(frame, start, end, tz=LISBON):
    stamps = pd.DatetimeIndex(frame["Datetime"])
    return frame[(stamps >= pd.Timestamp(start, tz=tz)) & (stamps < pd.Timestamp(end, tz=tz))]


@pytest.fixture(scope="module")
def pv_only_runs():
    config = {**BASE, "battery_kwh": 0}
    return _run({**config, "projection_years": 1}), _run({**config, "period": JUNE_WEEK})


@pytest.fixture(scope="module")
def battery_runs():
    config = {**BASE, "battery_kwh": 5}
    return _run({**config, "projection_years": 1}), _run({**config, "period": JUNE_WEEK})


def test_a_june_week_pv_only_equals_the_full_year_slice_exactly(pv_only_runs):
    full, week = pv_only_runs
    sliced = _window(full._artifacts.first_year_results_df, "2025-06-01", "2025-06-08")
    frame = week._artifacts.first_year_results_df

    # Lisbon is on UTC+1 in June, so the window starts at 23:00 UTC on 31 May.
    assert len(frame) == 168
    assert pd.Timestamp(frame["Datetime"].iloc[0]) == pd.Timestamp("2025-05-31 23:00", tz="UTC")
    assert pd.Timestamp(frame["Datetime"].iloc[-1]) == pd.Timestamp("2025-06-07 22:00", tz="UTC")
    for column in FLOWS:
        np.testing.assert_array_equal(frame[column].to_numpy(), sliced[column].to_numpy(), err_msg=column)

    result = week.result()
    kwh = {column: float(sliced[column].sum()) / 1000 for column in FLOWS}
    assert result["pv_dc_generation_kwh"] == round(kwh["PV_DC"], 2)
    assert result["consumption_kwh"] == round(kwh["Houseload"], 2)
    assert result["grid_import_kwh"] == round(kwh["Import_From_Grid"], 2)
    assert result["grid_export_kwh"] == round(kwh["PV_AC_Export"], 2)


def test_a_june_week_with_a_battery_starts_from_the_battery_initial_state(battery_runs):
    full, week = battery_runs
    sliced = _window(full._artifacts.first_year_results_df, "2025-06-01", "2025-06-08")
    frame = week._artifacts.first_year_results_df

    # The inputs are the full year's, step for step.
    for column in ("PV_DC", "Houseload"):
        np.testing.assert_array_equal(frame[column].to_numpy(), sliced[column].to_numpy(), err_msg=column)
    # The dispatch is not: the window starts at the initial state of charge
    # and a new pack, not at the state the full year reaches on 31 May.
    assert frame["Battery_Energy_Beginning"].iloc[0] != sliced["Battery_Energy_Beginning"].iloc[0]

    result = week.result()
    balance = result["pv_loss_waterfall"]["energy_balance"]
    assert balance["pv_dc"]["residual_kwh"] == pytest.approx(0.0, abs=1e-6)
    assert balance["battery_stored_energy"]["residual_kwh"] == pytest.approx(0.0, abs=1e-6)
    assert result["pv_loss_waterfall"]["basis"] == "period"
    assert result["pv_loss_waterfall"]["flow_unit"] == "kWh over the period"
    assert 99.0 < result["battery_soh_end_pct"] < 100.0
    assert len(result["yearly"]) == 1


def test_the_load_is_the_window_share_of_the_scaled_year(pv_only_runs):
    full, week = pv_only_runs
    assert full.result()["consumption_kwh"] == pytest.approx(4000, abs=0.01)
    june = _window(full._artifacts.first_year_results_df, "2025-06-01", "2025-06-08")
    assert week.result()["consumption_kwh"] == round(float(june["Houseload"].sum()) / 1000, 2)
    assert week.result()["consumption_kwh"] < 4000 * 8 / 365

    # The window's load is the day-type-aligned H0 year sliced, not a
    # positional copy of the 2023 source: Sunday 1 June 2025 carries the
    # source Sunday 4 June 2023 (#298).
    load = load_profile("demandlib_h0", 4000, start_date="2025-01-01", freq="h", timezone=LISBON).iloc[:, 0]
    houseload = week._artifacts.first_year_results_df["Houseload"].to_numpy()
    window = (load.index >= pd.Timestamp("2025-06-01", tz=LISBON)) & (
        load.index < pd.Timestamp("2025-06-08", tz=LISBON)
    )
    np.testing.assert_array_equal(houseload, load[window].to_numpy())
    source = load_profile("demandlib_h0", 4000, start_date="2023-01-01", freq="h", timezone=LISBON).iloc[:, 0]
    ratio = houseload[:24] / source.loc["2023-06-04"].to_numpy()
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-12)


@pytest.mark.parametrize(
    ("resolution", "day", "steps"),
    [
        ("h", "2025-06-15", 24),
        ("15min", "2025-06-15", 96),
        # Lisbon springs forward on 30 March: a 23-hour civil day.
        ("h", "2025-03-30", 23),
        # And falls back on 26 October: a 25-hour one.
        ("15min", "2025-10-26", 100),
    ],
)
def test_a_one_day_window_covers_the_civil_day(resolution, day, steps):
    end = (pd.Timestamp(day) + pd.Timedelta(days=1)).date().isoformat()
    app = _run({**BASE, "battery_kwh": 5, "resolution": resolution, "period": {"start": day, "end": end}})
    frame = app._artifacts.first_year_results_df
    stamps = pd.DatetimeIndex(frame["Datetime"])

    assert len(frame) == steps
    assert stamps[0] == pd.Timestamp(day, tz=LISBON)
    result = app.result()
    assert result["period"]["simulated_hours"] == steps * (1 if resolution == "h" else 0.25)
    assert result["period"]["days"] == 1
    assert [row["month"] for row in result["monthly"]] == [pd.Timestamp(day).strftime("%b")]


def test_lifetime_economics_are_none_and_say_why(battery_runs):
    _full, week = battery_runs
    result = week.result()

    for key in (
        "npv_savings",
        "payback_year",
        "lcoe_per_kwh",
        "financial",
        "battery_replacement_cost_t0_prices",
        "battery_replacement_cost_npv",
        "co2_avoided_self_consumption_lifetime_kg",
        "co2_avoided_export_lifetime_kg",
        "co2_avoided_total_lifetime_kg",
        "co2_avoided_total_kg",
    ):
        assert result[key] is None, key
        assert key in result["period"]["skipped_fields"]
    assert result["period"]["lifetime_economics"] == "skipped"
    assert "runs once" in result["period"]["lifetime_economics_reason"]
    # The window keeps its energy money at year-1 prices, its CO2 and the investment.
    assert result["total_investment"] > 0
    assert result["grid_import_cost_year1_prices"] > 0
    # The fixed charge is billed on the window's 7 days, not a year's.
    assert result["fixed_charge_year1_prices"] == round(7 * week._artifacts.costs["daily_power_cost"], 2)
    assert result["co2_avoided_total_year1_kg"] > 0
    assert result["co2_avoided_year1_kg"] == result["co2_avoided_total_year1_kg"]
    # Strict JSON, as the CLI writes it.
    json.dumps(result, allow_nan=False)


def test_the_yearly_row_and_provenance_record_the_window(battery_runs):
    _full, week = battery_runs
    result = week.result()

    assert result["yearly"][0]["period_start"] == "2025-06-01"
    assert result["yearly"][0]["period_end"] == "2025-06-08"
    window = {
        "start": "2025-06-01",
        "end": "2025-06-08",
        "end_exclusive": True,
        "timezone": LISBON,
        "days": 7,
        "start_time": "2025-06-01T00:00:00+01:00",
        "end_time": "2025-06-08T00:00:00+01:00",
        "projection_years_used": 1,
    }
    assert result["provenance"]["period"] == window
    assert {key: result["period"][key] for key in window} == window
    assert result["provenance"]["resolved_config"]["period"] == JUNE_WEEK


def test_a_full_year_result_has_no_period_keys(battery_runs):
    full, _week = battery_runs
    result = full.result()

    assert "period" not in result
    assert "period" not in result["provenance"]
    assert "period" not in result["provenance"]["resolved_config"]
    assert "period_start" not in result["yearly"][0]
    assert result["pv_loss_waterfall"]["basis"] == "year_1"
    assert isinstance(result["npv_savings"], float)


def test_toml_dates_are_accepted_and_stored_as_iso_strings():
    resolved = resolve_app_config({**BASE, "period": {"start": date(2025, 6, 1), "end": date(2025, 6, 8)}})
    assert resolved.cfg["period"] == JUNE_WEEK
    assert resolved.period is not None
    assert resolved.period.start_time == pd.Timestamp("2025-06-01", tz=LISBON)
    assert resolve_app_config(BASE).period is None


@pytest.mark.parametrize(
    ("period", "error", "match"),
    [
        ({"start": "2025-06-08", "end": "2025-06-08"}, ValueError, "must be before"),
        ({"start": "2025-06-08", "end": "2025-06-01"}, ValueError, "must be before"),
        ({"start": "2024-12-25", "end": "2025-01-08"}, ValueError, "must lie in 2025"),
        ({"start": "2025-12-25", "end": "2026-01-02"}, ValueError, "must lie in 2025"),
        ({"start": "2025-01-01", "end": "2026-01-01"}, ValueError, "covers the whole of 2025"),
        ({"start": "2025-06-01"}, ValueError, "needs period.end"),
        ({"start": "2025-06-01", "end": "2025-06-08", "annualize": True}, ValueError, "Unknown key 'period.annualize'"),
        ({"start": "1 June", "end": "2025-06-08"}, ValueError, "ISO date"),
        ({"start": datetime(2025, 6, 1, 12), "end": "2025-06-08"}, TypeError, "not a date and time"),
        ("2025-06", TypeError, "'period' must be a table"),
    ],
)
def test_invalid_periods_are_refused(period, error, match):
    with pytest.raises(error, match=match):
        App({**BASE, "period": period})


def test_the_last_week_of_the_year_may_end_on_1_january():
    app = _run({**BASE, "battery_kwh": 0, "period": {"start": "2025-12-25", "end": "2026-01-01"}})
    assert len(app._artifacts.first_year_results_df) == 7 * 24


def _truncated(days_missing_at_end):
    year = synthetic_weather(2025)
    return year.iloc[: len(year) - days_missing_at_end * 24]


def test_weather_missing_its_trailing_week_raises_for_a_window_that_needs_it():
    with pytest.raises(ValueError, match=r"trailing 7 days 00:00:00 \(168 steps\) of the window is missing"):
        _run({**BASE, "period": {"start": "2025-12-20", "end": "2026-01-01"}}, weather=_truncated(7))
    # So does a full-year run, as before (#259).
    with pytest.raises(ValueError, match="does not cover the simulated year 2025"):
        _run(BASE, weather=_truncated(7))


def test_weather_missing_its_trailing_week_serves_a_window_it_covers():
    short = _run({**BASE, "period": JUNE_WEEK}, weather=_truncated(7)).result()
    full = _run({**BASE, "period": JUNE_WEEK}).result()
    assert short["grid_import_kwh"] == full["grid_import_kwh"]


def test_weather_covering_only_the_window_is_enough():
    year = synthetic_weather(2025)
    week = year[
        (year.index >= pd.Timestamp("2025-06-01", tz=LISBON)) & (year.index < pd.Timestamp("2025-06-08", tz=LISBON))
    ]
    assert len(week) == 168
    short = _run({**BASE, "battery_kwh": 5, "period": JUNE_WEEK}, weather=week).result()
    full = _run({**BASE, "battery_kwh": 5, "period": JUNE_WEEK}).result()
    assert short["grid_import_kwh"] == full["grid_import_kwh"]
    assert short["battery_soh_end_pct"] == full["battery_soh_end_pct"]


def test_weather_missing_the_window_start_raises():
    year = synthetic_weather(2025)
    late = year[year.index >= pd.Timestamp("2025-06-02", tz="UTC")]
    with pytest.raises(ValueError, match=r"the leading 1 days 01:00:00 \(25 steps\) of the window is missing"):
        _run({**BASE, "period": JUNE_WEEK}, weather=late)


def test_a_window_off_the_weather_step_raises():
    # Weather stamped at half past the hour cannot place a window that starts
    # at local midnight.
    year = synthetic_weather(2025)
    year.index = year.index + pd.Timedelta(minutes=30)
    with pytest.raises(ValueError, match="does not start on a step of the weather"):
        _run({**BASE, "period": JUNE_WEEK}, weather=year)


def test_monte_carlo_rejects_a_period(tmp_path):
    with pytest.raises(ValueError, match="'period' is not supported with Monte Carlo"):
        run_montecarlo({**BASE, "period": JUNE_WEEK}, MonteCarloSettings(weather_file=str(tmp_path / "none.csv")))
    # Also before a year cache loads any weather: the file does not exist.
    with pytest.raises(ValueError, match="'period' is not supported with Monte Carlo"):
        build_year_cache({**BASE, "period": JUNE_WEEK}, MonteCarloSettings(weather_file=str(tmp_path / "none.csv")))


@pytest.mark.parametrize(
    "config",
    [
        {"location": {"latitude": 41.15, "longitude": -8.63}, "period": JUNE_WEEK},
        {"location": {"latitude": 41.15, "longitude": -8.63}, "simulation": {"period": JUNE_WEEK}},
    ],
)
def test_the_optimizer_rejects_a_period(config):
    with pytest.raises(ValueError, match="'period' is not supported by the optimizer"):
        resolve_optimization_config(config)


def test_revalue_reprices_the_window_and_keeps_lifetime_economics_none(battery_runs):
    _full, week = battery_runs
    with _weather():
        revalued = week.revalue({"costs": {"electricity_cost": 0.40}})
    fresh = _run({**BASE, "battery_kwh": 5, "period": JUNE_WEEK, "costs": {"electricity_cost": 0.40}}).result()

    assert revalued["provenance"]["revaluation"]["method"] == "repriced"
    assert revalued["npv_savings"] is None
    assert revalued["grid_import_cost_year1_prices"] == fresh["grid_import_cost_year1_prices"]
    assert revalued["grid_import_cost_year1_prices"] != week.result()["grid_import_cost_year1_prices"]
    with pytest.raises(ValueError, match="period is not a price key"):
        week.revalue({"period": {"start": "2025-07-01", "end": "2025-07-08"}})


def test_a_tariff_with_smart_charging_prices_the_window():
    tariff = TOU
    smart = {
        "mode": "fixed_target",
        "target_usable_fraction": 0.6,
        "charge_periods": ["off_peak"],
        "discharge_periods": ["peak"],
        "grid_charge_efficiency": 0.95,
    }
    config = {**BASE, "battery_kwh": 5, "tariff": tariff, "smart_charging": smart, "period": JUNE_WEEK}
    config.pop("cost_preset")
    result = _run(config).result()

    assert result["fixed_charge_year1_prices"] == pytest.approx(7 * 0.25)
    assert result["smart_charging"]["yearly"][0]["grid_charge_ac_kwh"] > 0
    assert result["npv_savings"] is None


def test_the_window_is_part_of_the_input_cache_key():
    week = input_configuration_key({**BASE, "period": JUNE_WEEK})
    july = input_configuration_key({**BASE, "period": {"start": "2025-07-01", "end": "2025-07-08"}})
    assert week != july
    assert input_configuration_key(BASE) != week


def test_a_sweep_varies_the_window(tmp_path, capsys):
    config = tmp_path / "sweep.toml"
    config.write_text(
        'location = "porto"\nn_modules = 8\nannual_consumption_kwh = 4000\nstart_date = "2025-01-01"\n'
        "[period]\nstart = 2025-06-01\nend = 2025-06-08\n"
        '[sweep]\n"period.end" = ["2025-06-02", "2025-06-08"]\n'
    )
    output = tmp_path / "sweep.csv"
    with _weather():
        assert cli.main(["sweep", "--config", str(config), "--output", str(output)]) == 0
    rows = pd.read_csv(output)
    assert list(rows["param_period.end"]) == ["2025-06-02", "2025-06-08"]
    assert rows["consumption_kwh"].iloc[0] < rows["consumption_kwh"].iloc[1]
    assert rows["npv_savings"].isna().all()


def test_the_cli_runs_a_toml_period_and_reports_it(tmp_path, capsys):
    config = tmp_path / "week.toml"
    config.write_text(
        'location = "porto"\nn_modules = 8\nannual_consumption_kwh = 4000\nstart_date = "2025-01-01"\n'
        "[period]\nstart = 2025-06-01\nend = 2025-06-08\n"
    )
    assert cli.main(["validate-config", str(config)]) == 0
    assert "Period: 2025-06-01 to 2025-06-08 (end exclusive)" in capsys.readouterr().out

    output = tmp_path / "result.json"
    with _weather():
        assert cli.main(["run", "--config", str(config), "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    assert result["period"]["days"] == 7
    assert result["npv_savings"] is None
    assert result["provenance"]["resolved_config"]["period"] == JUNE_WEEK


# (config changes, window start, window end, weather year served). A leap
# year is served a common-year TMY, which fill_leap_day completes.
SLICE_CASES = {
    "15min_june_week": ({"resolution": "15min"}, "2025-06-01", "2025-06-08", 2025),
    "spring_forward_day": ({}, "2025-03-30", "2025-03-31", 2025),
    "fall_back_day_15min": ({"resolution": "15min"}, "2025-10-26", "2025-10-27", 2025),
    "leap_day": ({"start_date": "2024-01-01"}, "2024-02-29", "2024-03-01", 2023),
    "last_week_to_1_january": ({}, "2025-12-25", "2026-01-01", 2025),
}


@pytest.mark.parametrize("case", list(SLICE_CASES))
def test_a_pv_only_window_equals_the_full_year_slice(case):
    changes, start, end, weather_year = SLICE_CASES[case]
    config = {**BASE, "battery_kwh": 0, **changes}
    weather = synthetic_weather(weather_year)
    full = _run({**config, "projection_years": 1}, weather=weather)._artifacts.first_year_results_df
    window = _run({**config, "period": {"start": start, "end": end}}, weather=weather)._artifacts
    frame = window.first_year_results_df
    sliced = _window(full, start, end)

    expected_hours = (pd.Timestamp(end, tz=LISBON) - pd.Timestamp(start, tz=LISBON)) / pd.Timedelta(hours=1)
    steps_per_hour = 4 if changes.get("resolution") == "15min" else 1
    assert len(frame) == len(sliced) == expected_hours * steps_per_hour
    np.testing.assert_array_equal(pd.DatetimeIndex(frame["Datetime"]).asi8, pd.DatetimeIndex(sliced["Datetime"]).asi8)
    for column in FLOWS:
        np.testing.assert_array_equal(frame[column].to_numpy(), sliced[column].to_numpy(), err_msg=column)


@pytest.mark.parametrize(
    ("start", "end", "days", "hours"),
    [
        # Lisbon springs forward on 30 March, so 1 March to 1 April is 31
        # civil days but 743 hours.
        ("2025-03-30", "2025-03-31", 1, 23),
        ("2025-10-26", "2025-10-27", 1, 25),
        ("2025-03-01", "2025-04-01", 31, 743),
    ],
)
def test_the_fixed_charge_is_billed_on_the_window_civil_days(start, end, days, hours):
    app = _run({**BASE, "period": {"start": start, "end": end}})
    result = app.result()

    assert result["period"]["simulated_hours"] == hours
    assert result["fixed_charge_year1_prices"] == round(days * app._artifacts.costs["daily_power_cost"], 2)
    assert app._artifacts.yearly_df["Billed_Days"].iloc[0] == days


def test_a_tariff_fixed_charge_is_billed_on_the_window_civil_days():
    config = {**BASE, "tariff": TOU, "period": {"start": "2025-03-01", "end": "2025-04-01"}}
    app = _run(config)
    assert app.result()["fixed_charge_year1_prices"] == round(31 * TOU["fixed_charge_per_day"], 2)

    # A price change re-prices the stored window, billing the same days.
    with _weather():
        revalued = app.revalue({"tariff": {"fixed_charge_per_day": 0.5}})
    assert revalued["provenance"]["revaluation"]["method"] == "repriced"
    assert revalued["fixed_charge_year1_prices"] == 15.5


def test_revalue_that_adds_a_tariff_simulates_the_window_again(battery_runs):
    _full, week = battery_runs
    with _weather():
        revalued = week.revalue({"tariff": TOU})
    fresh = _run({**BASE, "battery_kwh": 5, "period": JUNE_WEEK, "tariff": TOU}).result()

    assert revalued["provenance"]["revaluation"]["method"] == "resimulated"
    del revalued["provenance"]["revaluation"]
    for result in (revalued, fresh):
        del result["provenance"]["execution"]
    assert revalued == fresh
    assert revalued["npv_savings"] is None
    assert revalued["period"]["days"] == 7


def test_an_explicit_projection_years_is_ignored_with_a_warning():
    with pytest.warns(UserWarning, match=r"'projection_years' \(25\) is ignored"):
        app = _run({**BASE, "projection_years": 25, "period": JUNE_WEEK})
    assert app.result()["period"]["projection_years_used"] == 1
    assert len(app.result()["yearly"]) == 1

    # Left at its default, it is ignored silently.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        App({**BASE, "period": JUNE_WEEK})
        App({**BASE, "projection_years": 25})


def test_the_config_summary_lists_the_period_only_when_set():
    assert "period" not in cli._resolved_config_summary(BASE)["simulation"]
    assert cli._resolved_config_summary({**BASE, "period": JUNE_WEEK})["simulation"]["period"] == JUNE_WEEK


@pytest.mark.parametrize(
    ("location", "period", "missing"),
    [
        (
            {"latitude": 52.52, "longitude": 13.405, "timezone": "Europe/Berlin"},
            {"start": "2025-01-01", "end": "2025-01-08"},
            r"the leading 0 days 01:00:00 \(1 steps\)",
        ),
        (
            {"latitude": 40.71, "longitude": -74.01, "timezone": "America/New_York"},
            {"start": "2025-12-25", "end": "2026-01-01"},
            r"the trailing 0 days 05:00:00 \(5 steps\)",
        ),
    ],
    ids=["berlin_1_january", "new_york_31_december"],
)
def test_utc_year_weather_cannot_serve_a_window_at_the_civil_year_edge(location, period, missing):
    # The synthetic year covers one UTC year, as an Open-Meteo file does.
    with pytest.raises(ValueError, match=missing):
        _run({**BASE, "location": location, "period": period})
