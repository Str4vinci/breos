"""Vectorised tariff classification gives the labels of the step-by-step loop (#392).

``_reference_classify`` is the 0.7.0 loop, kept here only as the reference:
every step's season, rule and period found one timestamp at a time. The
vectorised classifier must give the same period and season labels for every
bundled schedule over a full year, and across clock changes, holidays and the
first and last UTC hours of a year that are already another local year.
A classification is kept, so App.revalue at new prices does not classify again.
"""

from datetime import date
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from breos import App, tariffs
from breos.tariffs import (
    HolidayCalendar,
    MonthSeasons,
    ScheduleDefinition,
    ScheduleRule,
    TariffSchedule,
    _as_definition,
    _classify,
    _day_type,
    _holiday_dates,
    available_tariff_schedules,
    get_schedule_definition,
    schedule_resolution_minutes,
)
from tools.generate_app_golden import SCENARIOS, _fake_fetch


def _reference_classify(index, schedule, *, timezone):
    """The 0.7.0 step loop, for an index the classifier has checked."""
    definition = _as_definition(schedule)
    local_index = index.tz_convert(timezone)
    holidays = definition.holidays
    years, counts = np.unique(local_index.year, return_counts=True)
    step_hours = (local_index[1] - local_index[0]).total_seconds() / 3600 if len(local_index) > 1 else 24.0
    covered = {int(year) for year, count in zip(years, counts, strict=True) if count * step_hours >= 24}
    holiday_dates = _holiday_dates(definition.schedule.identifier, holidays, covered)
    holiday_day_type = holidays.day_type if holidays is not None else "sunday"
    labels: list[str] = []
    seasons: list[str] = []
    for timestamp in local_index:
        season = definition.season_of(timestamp)
        rule = definition.rule_for(_day_type(timestamp, holiday_dates, holiday_day_type), season)
        minute = timestamp.hour * 60 + timestamp.minute
        labels.append(next(period for start, end, period in rule.intervals if start <= minute < end))
        seasons.append(season)
    return tuple(labels), tuple(seasons)


def _assert_same_labels(index, schedule, *, timezone, study_date=None):
    actual = _classify(index, schedule, timezone=timezone, study_date=study_date, boundary_policy="strict")
    assert actual == _reference_classify(index, schedule, timezone=timezone)


def _local_year(timezone, year, freq):
    """A civil year in ``timezone``, on UTC instants."""
    return pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq=freq, tz=timezone, inclusive="left").tz_convert(
        "UTC"
    )


def _utc_year(year, freq):
    """A UTC calendar year: east of UTC its last steps are already the next local year."""
    return pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq=freq, tz="UTC", inclusive="left")


def _bundled_year(identifier):
    """A year each bundled schedule is effective in, and has a holiday calendar for."""
    definition = get_schedule_definition(identifier)
    effective_from = definition.schedule.effective_from
    return 2028 if effective_from is not None and effective_from.year == 2027 else 2026


def _freqs(identifier):
    year = _bundled_year(identifier)
    hourly = schedule_resolution_minutes(identifier, (year - 1, year, year + 1)) % 60 == 0
    return ("15min", "h") if hourly else ("15min",)


BUNDLED_CASES = [
    (identifier, freq, clock)
    for identifier in available_tariff_schedules()
    for freq in _freqs(identifier)
    for clock in ("local", "utc")
]


@pytest.mark.parametrize(("identifier", "freq", "clock"), BUNDLED_CASES)
def test_every_bundled_schedule_classifies_a_full_year_as_the_step_loop_did(identifier, freq, clock):
    definition = get_schedule_definition(identifier)
    zone = definition.schedule.timezone
    year = _bundled_year(identifier)
    index = _local_year(zone, year, freq) if clock == "local" else _utc_year(year, freq)
    _assert_same_labels(index, identifier, timezone=zone)


def test_a_study_date_classifies_a_reference_year_as_the_step_loop_did():
    index = _local_year("Europe/Lisbon", 2026, "15min")
    _assert_same_labels(index, "pt_mainland_2027_weekly_tri", timezone="Europe/Lisbon", study_date=date(2027, 7, 1))


# Weekday windows that differ between standard time and DST, boundaries next
# to the clock changes, and holidays that take the Sunday rule. ``unit`` is
# the finest boundary: half an hour, or an hour for hourly steps.
_PERIODS = ("off_peak", "mid", "peak")


def _dst_schedule(timezone, holidays, unit):
    def intervals(*bounds_and_periods):
        bounds = [0, *(bound * unit for bound in bounds_and_periods[1::2]), 1440]
        return tuple(zip(bounds[:-1], bounds[1:], bounds_and_periods[::2], strict=True))

    rules = (
        ScheduleRule("weekday", "standard", intervals("off_peak", 3, "mid", 5, "peak", 18 * 60 // unit, "mid")),
        ScheduleRule("weekday", "dst", intervals("mid", 2, "off_peak", 3, "peak", 21 * 60 // unit, "mid")),
        ScheduleRule("saturday", "all", intervals("peak", 1, "mid", 1440 // unit - 1, "off_peak")),
        ScheduleRule("sunday", "all", ((0, 1440, "off_peak"),)),
    )
    return ScheduleDefinition(
        schedule=TariffSchedule(
            identifier="dst_edges", version="1", timezone=timezone, cycle="weekly", periods=_PERIODS
        ),
        rules=rules,
        holidays=HolidayCalendar(day_type="sunday", dates=frozenset(holidays)),
    )


@pytest.mark.parametrize(
    ("timezone", "year", "holidays"),
    [
        ("Europe/Lisbon", 2026, (date(2026, 3, 29), date(2026, 10, 25), date(2026, 12, 25))),
        ("Europe/Madrid", 2026, (date(2026, 1, 1), date(2026, 3, 30), date(2026, 10, 26))),
        # On 31 March 1996 Lisbon moved from CET to WEST summer time without
        # changing its UTC offset: DST starts inside a run of one offset.
        ("Europe/Lisbon", 1996, (date(1996, 3, 31),)),
        # Santiago turns its clocks back at midnight and repeats the day's last hour.
        ("America/Santiago", 2026, (date(2026, 4, 4), date(2026, 9, 6))),
        # Lord Howe moves its clocks by half an hour.
        ("Australia/Lord_Howe", 2026, (date(2026, 4, 5), date(2026, 10, 4))),
    ],
)
@pytest.mark.parametrize("freq", ["15min", "30min", "h"])
@pytest.mark.parametrize("clock", ["local", "utc"])
def test_clock_changes_and_holidays_classify_as_the_step_loop_did(timezone, year, holidays, freq, clock):
    if timezone == "Australia/Lord_Howe" and freq == "h":
        pytest.skip("a half-hour clock change needs 30-minute steps")
    definition = _dst_schedule(timezone, holidays, 60 if freq == "h" else 30)
    index = _local_year(timezone, year, freq) if clock == "local" else _utc_year(year, freq)
    _assert_same_labels(index, definition, timezone=timezone)


def test_month_seasons_and_holidays_classify_as_the_step_loop_did():
    windowed = ((0, 360, "low"), (360, 1020, "standard"), (1020, 1260, "high"), (1260, 1440, "standard"))
    definition = ScheduleDefinition(
        schedule=TariffSchedule(
            identifier="quarters",
            version="1",
            timezone="Europe/Berlin",
            cycle="custom",
            periods=("low", "standard", "high"),
        ),
        rules=(
            ScheduleRule("weekday", "q1", windowed),
            ScheduleRule("weekday", "q4", windowed),
            ScheduleRule("weekday", "summer", ((0, 1440, "standard"),)),
            ScheduleRule("saturday", "all", ((0, 600, "low"), (600, 1440, "standard"))),
            ScheduleRule("sunday", "all", ((0, 1440, "low"),)),
        ),
        holidays=HolidayCalendar(
            day_type="saturday", dates=frozenset({date(2026, 1, 1), date(2026, 4, 3), date(2026, 12, 25)})
        ),
        seasons=MonthSeasons((("q1", (1, 2, 3)), ("summer", (4, 5, 6, 7, 8, 9)), ("q4", (10, 11, 12)))),
    )
    for index in (_local_year("Europe/Berlin", 2026, "15min"), _utc_year(2026, "15min")):
        _assert_same_labels(index, definition, timezone="Europe/Berlin")


def test_short_and_single_step_indexes_classify_as_the_step_loop_did():
    zone = "Europe/Lisbon"
    spring = pd.date_range("2026-03-29 00:00", "2026-03-29 03:00", freq="15min", tz="UTC")
    for index in (spring, spring[:1], spring[5:7]):
        _assert_same_labels(index, "pt_mainland_2026_weekly_tri", timezone=zone)
    assert _classify(
        spring[:0], "pt_mainland_2026_weekly_tri", timezone=zone, study_date=None, boundary_policy="strict"
    ) == (
        (),
        (),
    )


# --- reuse ------------------------------------------------------------------


@pytest.fixture
def classified(monkeypatch):
    """The schedules each uncached classification ran for."""
    calls = []
    classify_steps = tariffs._classify_steps

    def counted(index, definition, zone, study_date):
        calls.append(definition.schedule.identifier)
        return classify_steps(index, definition, zone, study_date)

    monkeypatch.setattr(tariffs, "_classify_steps", counted)
    return calls


def test_the_same_instants_and_schedule_are_classified_once(classified):
    zone = "Europe/Lisbon"
    index = _local_year(zone, 2026, "15min")
    first = _classify(index, "pt_mainland_2026_weekly_tri", timezone=zone, study_date=None, boundary_policy="strict")
    # The index's own timezone does not matter, only its instants.
    again = _classify(
        index.tz_convert("Asia/Tokyo"),
        "pt_mainland_2026_weekly_tri",
        timezone=zone,
        study_date=None,
        boundary_policy="strict",
    )
    assert again == first
    assert classified == ["pt_mainland_2026_weekly_tri"]

    _classify(index, "pt_mainland_2026_daily_bi", timezone=zone, study_date=None, boundary_policy="strict")
    _classify(index[1:], "pt_mainland_2026_weekly_tri", timezone=zone, study_date=None, boundary_policy="strict")
    _classify(
        index, "pt_mainland_2027_weekly_tri", timezone=zone, study_date=date(2027, 7, 1), boundary_policy="strict"
    )
    _classify(
        index, "pt_mainland_2027_weekly_tri", timezone=zone, study_date=date(2027, 8, 1), boundary_policy="strict"
    )
    assert classified == [
        "pt_mainland_2026_weekly_tri",
        "pt_mainland_2026_daily_bi",
        "pt_mainland_2026_weekly_tri",
        "pt_mainland_2027_weekly_tri",
        "pt_mainland_2027_weekly_tri",
    ]


def test_a_failed_classification_is_not_kept_and_fails_again():
    zone = "Europe/Lisbon"
    index = _local_year(zone, 2026, "15min")
    for _ in range(2):
        with pytest.raises(ValueError, match="not effective across the index date range"):
            _classify(index, "pt_mainland_2027_weekly_tri", timezone=zone, study_date=None, boundary_policy="strict")
    with pytest.raises(TypeError, match="'study_date' must be a date when configured"):
        _classify(
            index, "pt_mainland_2026_weekly_tri", timezone=zone, study_date="2026-07-01", boundary_policy="strict"
        )
    assert tariffs._CLASSIFICATIONS == {}


def test_only_the_latest_classifications_are_kept():
    zone = "Europe/Lisbon"
    index = _local_year(zone, 2026, "h")
    for offset in range(tariffs._CLASSIFICATIONS_KEPT + 3):
        _classify(index[offset:], "pt_mainland_2026_daily_bi", timezone=zone, study_date=None, boundary_policy="strict")
    assert len(tariffs._CLASSIFICATIONS) == tariffs._CLASSIFICATIONS_KEPT


def test_revalue_at_new_prices_does_not_classify_again(classified):
    schedule = "pt_mainland_2026_daily_bi"
    tou = {
        "schedule": schedule,
        "currency": "EUR",
        "import_prices": {"peak": 0.28, "off_peak": 0.11},
        "export_prices": {"all": 0.05},
        "fixed_charge_per_day": 0.25,
    }
    reference = {"schedule": schedule, "currency": "EUR", "import_prices": {"all": 0.20}, "fixed_charge_per_day": 0.30}
    config = {**SCENARIOS["native_h_replacement"], "execution_backend": "python", "tariff": tou}
    with (
        mock.patch("breos.app.fetch_tmy_weather_data", _fake_fetch),
        mock.patch("breos.app.load_weather", lambda **_kwargs: None),
    ):
        app = App({**config, "reference_tariff": reference})
        app.simulate()
        # The reference is on the tariff's schedule and index: one classification serves both.
        assert classified == [schedule]
        revalued = app.revalue(
            {
                "tariff": {"import_prices": {"peak": 0.33, "off_peak": 0.09}},
                "reference_tariff": {"import_prices": {"all": 0.24}},
            }
        )
    assert revalued["provenance"]["revaluation"]["method"] == "repriced"
    assert revalued["provenance"]["tariff"]["import_prices"] == {"peak": 0.33, "off_peak": 0.09}
    assert classified == [schedule]
