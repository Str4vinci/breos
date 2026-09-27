"""Bundled tariff schedules match their primary sources (ADR 0002, plan acceptance criteria).

Each case is a local civil time and the period the source assigns to it.
Sources:
- PT 2026: Diretiva ERSE n.º 1/2026, Art. 45.º (Diário da República, 2.ª série, n.º 4, 2026-01-07).
- PT 2027: Diretiva ERSE n.º 3/2026, de 19 de agosto, Art. 2.º n.º 4 and n.º 7.
- ES 2.0TD: CNMC Circular 3/2020, Art. 7.3; 2026 holidays from BOE-A-2025-21667.
2026-01-14 is a Wednesday (winter), 2026-01-17 a Saturday, 2026-01-18 a Sunday;
2026-07-15 is a Wednesday (summer), 2026-07-18 a Saturday.
"""

from datetime import date

import pandas as pd
import pytest

from breos.tariffs import available_tariff_schedules, classify_tariff_periods, get_tariff_schedule

LISBON, MADRID = "Europe/Lisbon", "Europe/Madrid"
WINTER_WED, SAT_W, SUN_W = "2026-01-14", "2026-01-17", "2026-01-18"
SUMMER_WED, SAT_S = "2026-07-15", "2026-07-18"
# The 2027 schedules are checked on the same weekdays in 2027.
W27_WED, S27_WED, SAT27, SUN27 = "2027-01-13", "2027-07-14", "2027-01-16", "2027-01-17"

CASES = {
    "pt_mainland_2026_daily_bi": [
        (WINTER_WED, "07:45", "off_peak"),
        (WINTER_WED, "08:00", "peak"),
        (WINTER_WED, "21:45", "peak"),
        (WINTER_WED, "22:00", "off_peak"),
        (SUMMER_WED, "12:00", "peak"),
        (SUN_W, "12:00", "peak"),
    ],
    "pt_mainland_2026_daily_tri": [
        (WINTER_WED, "07:30", "off_peak"),
        (WINTER_WED, "08:30", "mid_peak"),
        (WINTER_WED, "09:00", "peak"),
        (WINTER_WED, "10:15", "peak"),
        (WINTER_WED, "10:30", "mid_peak"),
        (WINTER_WED, "18:00", "peak"),
        (WINTER_WED, "20:30", "mid_peak"),
        (WINTER_WED, "22:00", "off_peak"),
        (SUMMER_WED, "10:00", "mid_peak"),
        (SUMMER_WED, "10:30", "peak"),
        (SUMMER_WED, "13:00", "mid_peak"),
        (SUMMER_WED, "19:30", "peak"),
        (SUMMER_WED, "21:00", "mid_peak"),
        (SUMMER_WED, "22:00", "off_peak"),
    ],
    "pt_mainland_2026_weekly_bi": [
        (WINTER_WED, "06:30", "off_peak"),
        (WINTER_WED, "07:00", "peak"),
        (WINTER_WED, "23:30", "peak"),
        (SAT_W, "09:00", "off_peak"),
        (SAT_W, "09:30", "peak"),
        (SAT_W, "13:00", "off_peak"),
        (SAT_W, "18:30", "peak"),
        (SAT_W, "22:00", "off_peak"),
        (SAT_S, "09:00", "peak"),
        (SAT_S, "14:00", "off_peak"),
        (SAT_S, "20:00", "peak"),
        (SAT_S, "22:00", "off_peak"),
        (SUN_W, "12:00", "off_peak"),
    ],
    "pt_mainland_2026_weekly_tri": [
        (WINTER_WED, "06:45", "off_peak"),
        (WINTER_WED, "07:00", "mid_peak"),
        (WINTER_WED, "09:30", "peak"),
        (WINTER_WED, "12:00", "mid_peak"),
        (WINTER_WED, "18:30", "peak"),
        (WINTER_WED, "21:00", "mid_peak"),
        (SUMMER_WED, "09:00", "mid_peak"),
        (SUMMER_WED, "09:15", "peak"),
        (SUMMER_WED, "12:00", "peak"),
        (SUMMER_WED, "12:15", "mid_peak"),
        (SUMMER_WED, "23:45", "mid_peak"),
        (SAT_W, "09:30", "mid_peak"),
        (SAT_W, "13:00", "off_peak"),
        (SAT_S, "09:00", "mid_peak"),
        (SUN_W, "19:00", "off_peak"),
    ],
    "pt_mainland_2027_daily_bi": [
        (W27_WED, "08:15", "off_peak"),
        (W27_WED, "08:30", "peak"),
        (W27_WED, "22:15", "peak"),
        (W27_WED, "22:30", "off_peak"),
        (S27_WED, "08:30", "peak"),
    ],
    "pt_mainland_2027_daily_tri": [
        (W27_WED, "08:30", "mid_peak"),
        (W27_WED, "17:15", "mid_peak"),
        (W27_WED, "17:30", "peak"),
        (W27_WED, "21:30", "mid_peak"),
        (W27_WED, "22:30", "off_peak"),
        (S27_WED, "18:00", "peak"),
    ],
    "pt_mainland_2027_weekly_bi": [
        (W27_WED, "00:15", "peak"),
        (W27_WED, "00:30", "off_peak"),
        (W27_WED, "07:15", "off_peak"),
        (W27_WED, "07:30", "peak"),
        (SAT27, "16:45", "off_peak"),
        (SAT27, "17:00", "peak"),
        (SUN27, "20:00", "off_peak"),
    ],
    "pt_mainland_2027_weekly_tri": [
        (W27_WED, "00:15", "mid_peak"),
        (W27_WED, "00:30", "off_peak"),
        (W27_WED, "07:30", "mid_peak"),
        (W27_WED, "17:00", "peak"),
        (W27_WED, "22:00", "mid_peak"),
        (S27_WED, "17:00", "mid_peak"),
        (S27_WED, "19:30", "peak"),
        (S27_WED, "22:30", "mid_peak"),
        (SAT27, "17:00", "mid_peak"),
        (SUN27, "20:00", "off_peak"),
    ],
    "es_2_0td": [
        (WINTER_WED, "07:00", "off_peak"),
        (WINTER_WED, "08:00", "mid_peak"),
        (WINTER_WED, "10:00", "peak"),
        (WINTER_WED, "14:00", "mid_peak"),
        (WINTER_WED, "18:00", "peak"),
        (WINTER_WED, "22:00", "mid_peak"),
        (SUMMER_WED, "12:00", "peak"),
        (SAT_W, "12:00", "off_peak"),
        (SUN_W, "19:00", "off_peak"),
        # Wednesday 6 January and Monday 12 October 2026 are national holidays;
        # Monday 7 December replaces a Sunday holiday and is not one.
        ("2026-01-06", "12:00", "off_peak"),
        ("2026-10-12", "19:00", "off_peak"),
        ("2026-12-07", "12:00", "peak"),
    ],
}
TIMEZONES = {name: MADRID if name.startswith("es_") else LISBON for name in CASES}


def test_every_bundled_schedule_has_source_cases():
    assert set(CASES) == set(available_tariff_schedules())


def _classify(schedule, day, clock, freq):
    zone = TIMEZONES[schedule]
    local_day = pd.Timestamp(day, tz=zone)
    index = pd.date_range(local_day, local_day + pd.Timedelta(days=1), freq=freq, inclusive="left")
    study_date = date(2027, 7, 1) if schedule.startswith("pt_mainland_2027") else None
    labels = classify_tariff_periods(index.tz_convert("UTC"), schedule, timezone=zone, study_date=study_date)
    return dict(zip(index.strftime("%H:%M"), labels, strict=True))[clock]


@pytest.mark.parametrize(
    ("schedule", "day", "clock", "expected"),
    [(name, day, clock, expected) for name, cases in CASES.items() for day, clock, expected in cases],
)
def test_schedule_matches_its_source(schedule, day, clock, expected):
    assert _classify(schedule, day, clock, "15min") == expected


BTN_PERIODS = {"off_peak", "mid_peak", "peak"}


@pytest.mark.parametrize("schedule", sorted(CASES))
def test_schedule_periods_and_provenance(schedule):
    spec = get_tariff_schedule(schedule)
    assert set(spec.periods) <= BTN_PERIODS  # BTN tri-hourly has three periods: no super vazio
    assert spec.source and spec.source_url and spec.source_url.startswith("https://")
    if schedule.startswith("pt_mainland_2027"):
        assert spec.effective_from == date(2027, 7, 1)
        assert "de 19 de agosto" in spec.source


@pytest.mark.parametrize(
    ("schedule", "exact_hourly"),
    [
        ("pt_mainland_2026_daily_bi", True),
        ("es_2_0td", True),
        ("pt_mainland_2026_daily_tri", False),
        ("pt_mainland_2026_weekly_tri", False),
        ("pt_mainland_2027_daily_bi", False),
    ],
)
def test_half_hour_boundaries_need_15_minute_input(schedule, exact_hourly):
    zone = TIMEZONES[schedule]
    hourly = pd.date_range("2026-01-12", periods=24 * 7, freq="h", tz=zone)
    study_date = date(2027, 7, 1) if schedule.startswith("pt_mainland_2027") else None
    if exact_hourly:
        assert len(classify_tariff_periods(hourly, schedule, timezone=zone, study_date=study_date)) == len(hourly)
    else:
        with pytest.raises(ValueError, match="minute resolution or finer"):
            classify_tariff_periods(hourly, schedule, timezone=zone, study_date=study_date)


def test_spanish_holidays_need_a_calendar_for_the_year():
    index = pd.date_range("2025-03-01", periods=48, freq="h", tz="UTC")
    with pytest.raises(ValueError, match="no holiday calendar for 2025"):
        classify_tariff_periods(index, "es_2_0td", timezone=MADRID)


def test_a_utc_year_grazing_the_next_local_year_needs_no_second_calendar():
    # The last UTC hour of 2026 is already 1 January 2027 in Madrid.
    index = pd.date_range("2026-01-01", "2027-01-01", freq="h", tz="UTC", inclusive="left")
    labels = classify_tariff_periods(index, "es_2_0td", timezone=MADRID)
    assert len(labels) == len(index)


@pytest.mark.parametrize(
    ("schedule", "day", "expected_steps"),
    [
        ("pt_mainland_2026_weekly_tri", "2026-03-29", 92),
        ("pt_mainland_2026_weekly_tri", "2026-10-25", 100),
        ("es_2_0td", "2026-03-29", 92),
        ("es_2_0td", "2026-10-25", 100),
    ],
)
def test_dst_days_keep_every_instant(schedule, day, expected_steps):
    zone = TIMEZONES[schedule]
    start = pd.Timestamp(day, tz=zone)
    # A calendar day in the zone: 23 or 25 hours at the transitions.
    index = pd.date_range(start, start + pd.offsets.Day(1), freq="15min", inclusive="left")
    labels = classify_tariff_periods(index, schedule, timezone=zone)
    assert len(labels) == expected_steps
