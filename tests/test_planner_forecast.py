"""Planner forecasts other than persistence (ADR 0002 A20).

``smart_charging.forecast`` lets the experimental ``daily_persistence``
planner forecast PV from day-ahead forecast weather (``"weather"``), one file
per lead day, or plan on the run's own future (``"perfect"``). These tests
cover its configuration, the rule that picks a published lead for every
planned step, the resampling reach that rule relies on, App runs with each
forecast and their provenance, and the fetchers of forecast and satellite
weather.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from breos import App, fetch_forecast_archive, fetch_satellite_weather
from breos._planner_forecast import (
    PERSISTENCE_LEAD,
    RESAMPLING_REACH_HOURS,
    PlannerForecast,
    calendar_epoch_seconds,
    epoch_seconds,
    source_times,
)
from breos.app_config import resolve_app_config
from breos.app_inputs import remap_tmy_year
from breos.smart_charging import DEFAULT_PUBLICATION_DELAY_HOURS, SmartChargingSpec
from breos.weather import IRRADIANCE_RESAMPLING_POLICIES, WEATHER_METADATA_KEY, save_weather_csv
from tests.conftest import _build_synthetic_weather
from tests.test_weather import _install_fake_openmeteo

TOU = {
    "schedule": "pt_mainland_2026_daily_bi",
    "currency": "EUR",
    "import_prices": {"peak": 0.28, "off_peak": 0.11},
    "export_prices": {"all": 0.05},
    "fixed_charge_per_day": 0.25,
}
DAILY = {
    "mode": "daily_persistence",
    "charge_periods": ["off_peak"],
    "discharge_periods": ["peak"],
    "grid_charge_efficiency": 0.95,
    "target_levels": 3,
    "soc_states": 3,
}
BASE = {
    "location": "porto",
    "n_modules": 8,
    "annual_consumption_kwh": 4000,
    "start_date": "2023-01-01",
    "battery_kwh": 5.0,
    "execution_backend": "python",
    "tariff": TOU,
    "weather_file": "actual.csv",
}
# Winter days on which the planner buys off-peak energy, and summer days on which PV fills the battery.
WINDOW = {"start": "2023-01-10", "end": "2023-01-15"}
SUMMER = {"start": "2023-06-10", "end": "2023-06-15"}


def _interval_mean(frame: pd.DataFrame) -> pd.DataFrame:
    """``frame`` declared as right-labelled hourly means, as Open-Meteo files are."""
    frame = frame.copy()
    frame.attrs[WEATHER_METADATA_KEY] = {
        "source": "OpenMeteo_historical",
        "radiation_time_basis": "interval_mean",
        "timestamp_label_basis": "right",
        "timestamp_timezone": "GMT",
    }
    return frame


@pytest.fixture
def site(tmp_path, monkeypatch):
    """A working directory with actual weather and seven lead files equal to it."""
    weather = _build_synthetic_weather(2023)
    weather.index = weather.index + pd.Timedelta(hours=1)
    weather.index.name = "date"
    weather = _interval_mean(weather)
    save_weather_csv(weather, tmp_path / "actual.csv")
    for lead in range(1, 8):
        save_weather_csv(weather, tmp_path / f"lead{lead}.csv")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "breos.app.fetch_tmy_weather_data", lambda *a, **k: pytest.fail("App fetched weather in a forecast test")
    )
    return SimpleNamespace(path=tmp_path, weather=weather)


def _leads(n: int) -> list[str]:
    return [f"lead{k}.csv" for k in range(1, n + 1)]


def _run(table: dict, window: dict = WINDOW, **extra) -> dict:
    app = App({**BASE, "period": window, "smart_charging": {**DAILY, **table}, **extra})
    app.simulate()
    return app.result()


# -- configuration ------------------------------------------------------------


def test_persistence_is_the_default_and_weather_takes_the_publication_delay(site):
    spec = resolve_app_config({**BASE, "smart_charging": DAILY}).smart_charging
    assert spec.forecast == "persistence" and spec.forecast_weather_files == ()
    assert spec.forecast_publication_delay_hours is None
    table = {**DAILY, "forecast": "weather", "forecast_weather_files": _leads(2)}
    spec = resolve_app_config({**BASE, "smart_charging": table}).smart_charging
    assert spec.forecast_weather_files == ("lead1.csv", "lead2.csv")
    assert spec.forecast_publication_delay_hours == DEFAULT_PUBLICATION_DELAY_HOURS == 6.0
    spec = resolve_app_config(
        {**BASE, "smart_charging": {**table, "forecast_publication_delay_hours": 0}}
    ).smart_charging
    assert spec.forecast_publication_delay_hours == 0.0


@pytest.mark.parametrize(
    ("table", "error", "message"),
    [
        ({"forecast": "weather"}, ValueError, "needs smart_charging.forecast_weather_files"),
        ({"forecast_weather_files": ["lead1.csv"]}, ValueError, "needs smart_charging.forecast = 'weather'"),
        ({"forecast": "perfect", "forecast_publication_delay_hours": 3}, ValueError, "not 'perfect'"),
        ({"forecast": "oracle"}, ValueError, "forecast"),
        ({"forecast": "weather", "forecast_weather_files": ["missing.csv"]}, FileNotFoundError, "missing.csv"),
        (
            {"forecast": "weather", "forecast_weather_files": ["lead1.csv"], "forecast_publication_delay_hours": -1},
            ValueError,
            "forecast_publication_delay_hours",
        ),
    ],
)
def test_bad_forecast_settings_are_refused(site, table, error, message):
    with pytest.raises(error, match=message):
        resolve_app_config({**BASE, "smart_charging": {**DAILY, **table}})


def test_only_the_planner_takes_a_forecast(site):
    fixed = {**DAILY, "mode": "fixed_target", "target_usable_fraction": 0.5}
    for key in ("target_levels", "soc_states"):
        fixed.pop(key)
    with pytest.raises(ValueError, match="does not take smart_charging.forecast"):
        resolve_app_config({**BASE, "smart_charging": {**fixed, "forecast": "perfect"}})


def test_a_spec_built_directly_checks_its_forecast():
    common = {
        "mode": "daily_persistence",
        "charge_periods": ("off_peak",),
        "discharge_periods": ("peak",),
        "grid_charge_efficiency": 0.95,
    }
    with pytest.raises(ValueError, match="needs smart_charging.forecast_weather_files"):
        SmartChargingSpec(**common, forecast="weather")
    with pytest.raises(ValueError, match="needs smart_charging.forecast = 'weather'"):
        SmartChargingSpec(**common, forecast_publication_delay_hours=2.0)
    with pytest.raises(ValueError, match="takes no planner settings"):
        SmartChargingSpec(
            mode="fixed_target",
            target_usable_fraction=0.5,
            charge_periods=("off_peak",),
            discharge_periods=("peak",),
            grid_charge_efficiency=0.95,
            forecast="perfect",
        )


# -- the lead rule -------------------------------------------------------------


def _forecast(freq="15min", days=10, leads=7, delay_h=6.0, support=None, decision=None) -> PlannerForecast:
    index = pd.date_range("2026-01-01", periods=days * (96 if freq == "15min" else 24), freq=freq, tz="UTC")
    n = len(index)
    step_s = 900 if freq == "15min" else 3600
    return PlannerForecast(
        "weather",
        calendar_epoch_seconds(index),
        step_s,
        pv_degradation_rate=0.0,
        support_source_s=support,
        decision_source_s=decision,
        pv_dc_by_lead_w=np.vstack([np.arange(n, dtype=float) + 1000 * k for k in range(1, leads + 1)]),
        publication_delay_s=int(delay_h * 3600),
        resampling_reach_s=int(RESAMPLING_REACH_HOURS[freq] * 3600),
    )


def test_a_step_takes_the_youngest_lead_published_before_the_decision():
    # Decision at 22:00 on 15-minute steps: a step's forecast covers the raw hours up to 4 h after its
    # end and is published 6 h after issue, so lead 1 serves until noon the next day.
    leads = _forecast().leads(88, 96 * 8)
    assert (leads[:56] == 1).all() and (leads[56:152] == 2).all() and leads[152] == 3
    # Beyond the last lead file the planner keeps persistence, never a younger forecast.
    assert (leads[:632] >= 1).all() and (leads[632:] == PERSISTENCE_LEAD).all()
    assert (_forecast(leads=2).leads(88, 300)[152:] == PERSISTENCE_LEAD).all()
    # Hourly steps read their own hour: lead 1 for the 18 h after a 22:00 decision.
    hourly = _forecast("h").leads(22, 48)
    assert (hourly[:18] == 1).all() and (hourly[18:42] == 2).all()
    # No publication delay: lead 1 covers the steps whose shaping hours end within a day.
    assert (_forecast(delay_h=0).leads(88, 80)[:80] == 1).all()


def _restamped(raw: pd.DatetimeIndex, year: int) -> tuple[np.ndarray, np.ndarray]:
    """Raw right-labelled hours restamped onto ``year`` by the App: their calendar and source seconds."""
    frame = remap_tmy_year(pd.DataFrame({"source_s": epoch_seconds(raw)}, index=raw), year)
    return epoch_seconds(pd.DatetimeIndex(frame.index)), frame["source_s"].to_numpy(dtype=np.int64)


def _leads_on(raw, year, steps, start, freq="15min", delay_h=6.0):
    step_s = 900 if freq == "15min" else 3600
    reach = int(RESAMPLING_REACH_HOURS[freq] * 3600)
    calendar, source = _restamped(raw, year)
    support, decision = source_times(calendar, source, epoch_seconds(steps), step_s, reach)
    n = len(steps)
    forecast = PlannerForecast(
        "weather",
        epoch_seconds(steps),
        step_s,
        pv_degradation_rate=0.0,
        support_source_s=support,
        decision_source_s=decision,
        pv_dc_by_lead_w=np.zeros((7, n)),
        publication_delay_s=int(delay_h * 3600),
        resampling_reach_s=reach,
    )
    return forecast.leads(start, n - start)


def test_ages_count_a_day_the_calendar_drops_even_at_a_period_edge():
    """2024 weather on the 2026 calendar: 29 February is gone, so 1 March is a day further away in source time."""
    raw = pd.date_range("2024-01-01 01:00", "2025-01-01 00:00", freq="h", tz="UTC")
    steps = pd.date_range("2026-02-27", "2026-03-01 23:45", freq="15min", tz="UTC")
    decided = steps.get_loc(pd.Timestamp("2026-02-28", tz="UTC"))
    leads = _leads_on(raw, 2026, steps, decided)
    times = steps[decided:]
    # Evening steps of 28 February are shaped by 1 March hours, issued a source day later.
    evening = (times >= pd.Timestamp("2026-02-28 20:00", tz="UTC")) & (times < pd.Timestamp("2026-03-01", tz="UTC"))
    assert (leads[evening] >= 3).all()
    plain = _forecast(days=3).leads(0, 96)
    morning = times < pd.Timestamp("2026-02-28 12:00", tz="UTC")
    np.testing.assert_array_equal(leads[morning], plain[: int(morning.sum())])


def test_a_filled_day_puts_the_decision_back_in_source_time():
    """2025 weather on the leap 2028 calendar: 29 February repeats 28 February, so a decision there is earlier."""
    raw = pd.date_range("2025-01-01 01:00", "2026-01-01 00:00", freq="h", tz="UTC")
    steps = pd.date_range("2028-02-28", "2028-03-01 23:45", freq="15min", tz="UTC")
    decided = steps.get_loc(pd.Timestamp("2028-02-29", tz="UTC"))
    leads = _leads_on(raw, 2028, steps, decided)
    # The first steps after the decision are shaped by the evening of 28 February, the source day after it.
    assert (leads[:8] >= 2).all()


def test_ages_follow_the_weathers_own_clock():
    """Weather stamped at +10:00 drops its local 29 February, whose end is 14:00 UTC on 28 February 2026."""
    raw = pd.date_range("2024-01-01 01:00", "2025-01-01 00:00", freq="h", tz="+10:00")
    steps = pd.date_range("2026-02-27", "2026-03-02 23:00", freq="h", tz="UTC")
    decided = steps.get_loc(pd.Timestamp("2026-02-28", tz="UTC"))
    leads = _leads_on(raw, 2026, steps, decided, freq="h")
    times = steps[decided:]
    # The hourly step from 13:00 reads the raw hour labelled 14:00 UTC, local 1 March 00:00: a source day on.
    jump = times >= pd.Timestamp("2026-02-28 13:00", tz="UTC")
    # Before it, a day's lead; from it, one more for the same calendar distance.
    assert leads[(times >= pd.Timestamp("2026-02-28 10:00", tz="UTC")) & ~jump].max() == 1
    assert leads[jump & (times < pd.Timestamp("2026-02-28 18:00", tz="UTC"))].min() == 2


def test_the_contiguous_case_is_the_calendar_rule():
    steps = pd.date_range("2026-01-01", periods=96 * 4, freq="15min", tz="UTC")
    raw = pd.date_range("2026-01-01 01:00", "2027-01-01 00:00", freq="h", tz="UTC")
    np.testing.assert_array_equal(_leads_on(raw, 2026, steps, 88), _forecast(days=4).leads(88, 96 * 4 - 88))


def test_the_forecast_replaces_only_pv_and_degrades_it_by_project_year():
    forecast = _forecast(days=10, leads=2)
    forecast.pv_degradation_rate = 0.1
    day = SimpleNamespace(project_step_ordinal=88 + 960, projection_year=1)
    pv, load, temperature = np.full(300, -1.0), np.full(300, -2.0), np.full(300, -3.0)
    out_pv, out_load, out_temperature = forecast.series(day, 300, pv, load, temperature)
    assert out_load is load and out_temperature is temperature
    leads = forecast.leads(88, 300)
    steps = (88 + np.arange(300)) % 960
    published = leads > 0
    np.testing.assert_allclose(out_pv[published], (steps[published] + 1000 * leads[published]) * 0.9)
    assert (out_pv[~published] == -1.0).all() and (~published).any()
    record = forecast.record()
    assert record["decisions"] == 1 and sum(record["planned_steps_by_lead"].values()) == 300

    # A planned step past the calendar's end is in the next project year, and degraded as such.
    seam = forecast.series(SimpleNamespace(project_step_ordinal=958, projection_year=0), 4, pv[:4], load, temperature)
    np.testing.assert_allclose(seam[0] / (np.array([958, 959, 0, 1]) + 1000), [1, 1, 0.9, 0.9])

    index = pd.date_range("2026-01-01", periods=96, freq="15min", tz="UTC")
    truth = np.arange(96, dtype=float)
    perfect = PlannerForecast(
        "perfect",
        calendar_epoch_seconds(index),
        900,
        pv_degradation_rate=0.5,
        true_pv_dc_w=truth,
        true_load_w=truth + 1,
        true_temperature_c=truth + 2,
    )
    out = perfect.series(SimpleNamespace(project_step_ordinal=96 + 90, projection_year=1), 10, pv, load, temperature)
    steps = (90 + np.arange(10)) % 96
    np.testing.assert_allclose(out[0], steps * 0.5 ** (1 + (90 + np.arange(10)) // 96))
    np.testing.assert_allclose(out[1], steps + 1.0)
    np.testing.assert_allclose(out[2], steps + 2.0)


@pytest.mark.parametrize("policy", IRRADIANCE_RESAMPLING_POLICIES)
def test_a_raw_hour_shapes_no_step_more_than_the_reach_before_it(site, policy):
    """The lead rule's reach: a raw hour moves no 15-minute PV step earlier than 4 h before its label."""
    from tools.oracles.replay import prepare_replay

    config = {**BASE, "resolution": "15min", "irradiance_resampling": policy}
    reference = np.asarray(prepare_replay(config).aligned_inputs().pv_dc_w)
    index = prepare_replay(config).index
    reach = pd.Timedelta(hours=RESAMPLING_REACH_HOURS["15min"])
    for label in ("2023-06-12 13:00", "2023-03-20 09:00", "2023-12-01 16:00"):
        changed = site.weather.copy()
        changed.loc[pd.Timestamp(label, tz="UTC"), ["ghi", "dni", "dhi"]] *= 0.3
        changed.loc[pd.Timestamp(label, tz="UTC"), "temp_air"] += 8
        save_weather_csv(_interval_mean(changed), site.path / "changed.csv")
        pv = np.asarray(prepare_replay({**config, "weather_file": "changed.csv"}).aligned_inputs().pv_dc_w)
        moved = np.flatnonzero(np.abs(pv - reference) > 1e-9)
        assert len(moved), label
        restamped = pd.Timestamp(label, tz="UTC").replace(year=index[0].year)
        assert index[moved[0]] >= restamped - reach, (label, index[moved[0]])


# -- App runs --------------------------------------------------------------------


def test_persistence_runs_record_no_planner_forecast(site):
    record = _run({})["provenance"]["smart_charging"]
    assert "planner_forecast" not in record
    assert record["forecast_policy"] == "repeat_previous_complete_local_day"


def test_a_perfect_forecast_plans_on_the_run_itself(site):
    record = _run({"forecast": "perfect"})["provenance"]["smart_charging"]
    assert record["forecast_policy"] == "perfect_information_reference"
    assert record["planner_forecast"]["forecast"] == "perfect" and record["planner_forecast"]["decisions"] > 0


def test_a_weather_forecast_records_its_files_and_leads(site):
    result = _run({"forecast": "weather", "forecast_weather_files": _leads(3)})
    record = result["provenance"]["smart_charging"]["planner_forecast"]
    assert record["forecast"] == "weather" and record["leads"] == 3
    assert [f["lead_day"] for f in record["files"]] == [1, 2, 3]
    assert all(len(f["sha256"]) == 64 for f in record["files"])
    assert record["publication_delay_hours"] == 6.0 and record["resampling_reach_hours"] == 0.0
    assert set(record["planned_steps_by_lead"]) <= {"0", "1", "2", "3"} and record["decisions"] > 0


def test_the_planner_never_reads_a_lead_it_could_not_have(site):
    """Leads beyond what a one-day horizon reaches do not change the run; lead 1 does."""
    table = {"forecast": "weather", "forecast_weather_files": _leads(7), "forecast_horizon_days": 1}
    base = _run(table, SUMMER)
    used = {int(k) for k in base["provenance"]["smart_charging"]["planner_forecast"]["planned_steps_by_lead"]}
    unused = [lead for lead in range(1, 8) if lead not in used]
    assert unused
    dark = site.weather.copy()
    dark[["ghi", "dni", "dhi"]] = 0.0
    for lead in unused:
        save_weather_csv(_interval_mean(dark), site.path / f"lead{lead}.csv")
    again = _run(table, SUMMER)
    assert (
        again["provenance"]["smart_charging"]["instruction_hash"]
        == (base["provenance"]["smart_charging"]["instruction_hash"])
    )
    save_weather_csv(_interval_mean(dark), site.path / "lead1.csv")
    darker = _run(table, SUMMER)
    assert (
        darker["provenance"]["smart_charging"]["instruction_hash"]
        != (base["provenance"]["smart_charging"]["instruction_hash"])
    )


def test_lead_files_of_different_years_are_refused(site):
    other = _build_synthetic_weather(2022)
    other.index = other.index + pd.Timedelta(hours=1)
    other.index.name = "date"
    save_weather_csv(_interval_mean(other), site.path / "lead2.csv")
    with pytest.raises(ValueError, match="different years"):
        _run({"forecast": "weather", "forecast_weather_files": _leads(2)})


# -- fetchers ----------------------------------------------------------------------


def test_the_forecast_archive_writes_one_file_per_lead(monkeypatch, tmp_path):
    captured: dict = {}
    _install_fake_openmeteo(monkeypatch, captured)
    leads = fetch_forecast_archive(41.15, -8.63, 2025, leads=2, location_name="porto", output_dir=str(tmp_path))
    assert captured["url"] == "https://previous-runs-api.open-meteo.com/v1/forecast"
    assert captured["params"]["models"] == "best_match" and captured["params"]["timezone"] == "GMT"
    assert captured["params"]["hourly"][:2] == ["temperature_2m_previous_day1", "wind_speed_10m_previous_day1"]
    assert sorted(leads) == [1, 2]
    frame = leads[2]
    assert list(frame.columns)[:3] == ["temperature_2m", "wind_speed_10m", "shortwave_radiation"]
    assert frame.index[0] == pd.Timestamp("2025-01-01 01:00") and frame.index[-1] == pd.Timestamp("2026-01-01")
    assert len(frame) == 8760
    metadata = frame.attrs[WEATHER_METADATA_KEY]
    assert metadata["forecast_lead_days"] == 2 and metadata["timestamp_label_basis"] == "right"
    assert (tmp_path / "porto_forecast_lead2_2025_openmeteo-best_match.csv").is_file()
    assert (tmp_path / "porto_forecast_lead2_2025_openmeteo-best_match.csv.metadata.json").is_file()
    with pytest.raises(ValueError, match="from 1 to 7"):
        fetch_forecast_archive(41.15, -8.63, 2025, leads=8, save_to_file=False)


def test_satellite_weather_joins_irradiance_and_reanalysis(monkeypatch, tmp_path):
    captured: dict = {}
    _install_fake_openmeteo(monkeypatch, captured)
    frame = fetch_satellite_weather(41.15, -8.63, 2025, save_to_file=False)
    assert list(frame.columns) == [
        "temperature_2m",
        "wind_speed_10m",
        "shortwave_radiation",
        "direct_radiation",
        "diffuse_radiation",
        "direct_normal_irradiance",
    ]
    assert len(frame) == 8760 and frame.attrs[WEATHER_METADATA_KEY]["source"] == "OpenMeteo_satellite"


def test_a_year_the_archive_does_not_cover_is_refused(monkeypatch):
    captured: dict = {}
    _install_fake_openmeteo(monkeypatch, captured)

    def gappy(url, params, fields, year):
        index = pd.date_range(f"{year}-01-01 01:00", f"{year + 1}-01-01", freq="h")
        frame = pd.DataFrame({name: 1.0 for name in fields}, index=index)
        frame.iloc[:20] = np.nan
        return frame

    monkeypatch.setattr("breos.weather._openmeteo_year", gappy)
    with pytest.raises(ValueError, match="20 hour"):
        fetch_forecast_archive(41.15, -8.63, 2024, leads=1, save_to_file=False)


def test_a_truncated_year_is_refused(monkeypatch):
    captured: dict = {}
    _install_fake_openmeteo(monkeypatch, captured)
    real = __import__("breos.weather", fromlist=["openmeteo_requests"]).openmeteo_requests.Client

    class Short(real):  # type: ignore[misc, valid-type]
        def weather_api(self, url, *, params):
            return super().weather_api(url, params={**params, "start_date": "2025-01-02"})

    monkeypatch.setattr("breos.weather.openmeteo_requests.Client", Short)
    with pytest.raises(ValueError, match="returned 8737 hours for 2025, not the 8760"):
        fetch_forecast_archive(41.15, -8.63, 2025, leads=1, save_to_file=False)
