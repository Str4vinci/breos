"""Tariff schedules as ScheduleDefinition values: parsing, resolution, identity and pickling."""

import hashlib
import pickle
from dataclasses import replace
from datetime import date

import pandas as pd
import pytest

from breos.resources import load_config_json
from breos.tariffs import (
    HolidayCalendar,
    ScheduleDefinition,
    ScheduleRule,
    TariffPrices,
    TariffSpec,
    available_tariff_schedules,
    classify_tariff_periods,
    get_schedule_definition,
    get_tariff_schedule,
    parse_schedule_definition,
    resolve_named_tariff,
    schedule_resolution_minutes,
)

LISBON, MADRID = "Europe/Lisbon", "Europe/Madrid"
PRICES = TariffPrices(currency="EUR", import_prices={"all": 0.2}, export_prices={"all": 0.05})

# Every bundled schedule over a leap and a non-leap local year: step count, the
# first 16 hex digits of the SHA-256 of the comma-joined labels, and the
# schedule hash. Recorded before schedules became ScheduleDefinition values.
PINS = {
    ("es_2_0td", 2026, "h"): (
        8760,
        "22625570f71b53bf",
        "a73ea0211aedf0d86878b22f7b6ab11c6dfde699fb66548609eb2500b3fa697f",
    ),
    ("es_2_0td", 2026, "15min"): (
        35040,
        "ec25064b65ba1d94",
        "5666d6d4c6771457627034954528a9965e7c366d3486d32c211ad35d66475eae",
    ),
    ("pt_mainland_2026_daily_bi", 2024, "h"): (
        8784,
        "bcad9857f21784b6",
        "76931de22f221f0f64b37e88c60e21e0fe4b1449c69d308c51d2107ba08dc548",
    ),
    ("pt_mainland_2026_daily_bi", 2024, "15min"): (
        35136,
        "4cb0111b5de3a260",
        "e332d561154f1254a081c0e8f5cb5dca3a263a55263f52caef51bf75fe163035",
    ),
    ("pt_mainland_2026_daily_bi", 2026, "h"): (
        8760,
        "fd781eac8a96679c",
        "5168b98eb15b56ad87fe219a386dbf945d5acc5acf507bd8548e5803023e2406",
    ),
    ("pt_mainland_2026_daily_bi", 2026, "15min"): (
        35040,
        "30e84ae97ed3bdd8",
        "092e4f5b4ce3c540dea65ad63b3ace17d22db6a51e9cc87298a4f6e7fbd01f3f",
    ),
    ("pt_mainland_2026_daily_tri", 2024, "15min"): (
        35136,
        "843c33d4b7f13796",
        "53dd81dcaf95899e6640707228e1e0142ee59ae6ca3cbfb6aa12337f669b243c",
    ),
    ("pt_mainland_2026_daily_tri", 2026, "15min"): (
        35040,
        "c6612ef8dfe46922",
        "95a300813f412f36d7f5b396698eb5604c7e24f7c024c8800fd91164f70ade56",
    ),
    ("pt_mainland_2026_weekly_bi", 2024, "15min"): (
        35136,
        "60c444b5b55bf24d",
        "d916026e1ea9a8d918f37d172c0f08cff696d5d99271ff93a0d1c33f0ba636b1",
    ),
    ("pt_mainland_2026_weekly_bi", 2026, "15min"): (
        35040,
        "2c9dac353f3da4a5",
        "37dccbfa3a04954447ed1907eb207b342fc8de8e7cb0c59b6a33156928e5718f",
    ),
    ("pt_mainland_2026_weekly_tri", 2024, "15min"): (
        35136,
        "8dd5cf1c00318198",
        "e2d29a86fea355ba43a4f46349d521b7779e6705de3b5d302c9ed365cd13d390",
    ),
    ("pt_mainland_2026_weekly_tri", 2026, "15min"): (
        35040,
        "50da0ac58113be55",
        "cb98ecdddb1ada98bea8d3dc3ae12977cbeb65526b56b551a9ec7025ede0f427",
    ),
    ("pt_mainland_2027_daily_bi", 2024, "15min"): (
        35136,
        "0f0c9fd98083a743",
        "22f8f31bd447c32d8f8d8f8657ccd84c61190f5e8d6acfa5817d448cb7247c54",
    ),
    ("pt_mainland_2027_daily_bi", 2026, "15min"): (
        35040,
        "0bc4396597de8775",
        "5c895284cc4ce36e65dd5e3e27485e9c8d9a699a18c55282c5b0ee1ad95edd58",
    ),
    ("pt_mainland_2027_daily_tri", 2024, "15min"): (
        35136,
        "a4a5515dfec99334",
        "11e38b5ea5f2494458e12a522c1c51d4f0b3c7b5790b9ede4911545345cbd154",
    ),
    ("pt_mainland_2027_daily_tri", 2026, "15min"): (
        35040,
        "259c824719b61a67",
        "ce4601589e4d1370911a6571e9578e09fdbea8843d34bf7924815dfc13146a97",
    ),
    ("pt_mainland_2027_weekly_bi", 2024, "15min"): (
        35136,
        "9aad42d3596ac594",
        "5a7c47cc8076aa0e59afbf9f7e1f72799eb65a75952eefb4ac832cf3b106aaa7",
    ),
    ("pt_mainland_2027_weekly_bi", 2026, "15min"): (
        35040,
        "4f1316e013699122",
        "5d62eed71b16a75f6e66a5a2ddf94ec165e5b6230d4d9e774e163b36b19e0bc4",
    ),
    ("pt_mainland_2027_weekly_tri", 2024, "15min"): (
        35136,
        "c007dbece41388f5",
        "fce030ee8f02bd09025ab42f1ddbde79f3b1809cea1df9010c00390630c191f9",
    ),
    ("pt_mainland_2027_weekly_tri", 2026, "15min"): (
        35040,
        "bd5ce39ba0e38e7e",
        "b7213035f9be2b13b3868e166f098145829e650ae6032f734ba051e729f73dbe",
    ),
}
# Cases the same years and steps cannot resolve: half-hour boundaries on hourly
# steps, and a Spanish year without a holiday calendar.
REJECTED = {
    ("es_2_0td", 2024, "h"): "no holiday calendar for 2024",
    ("es_2_0td", 2024, "15min"): "no holiday calendar for 2024",
    **{
        (name, year, "h"): "minute resolution or finer"
        for name in (
            "pt_mainland_2026_daily_tri",
            "pt_mainland_2026_weekly_bi",
            "pt_mainland_2026_weekly_tri",
            "pt_mainland_2027_daily_bi",
            "pt_mainland_2027_daily_tri",
            "pt_mainland_2027_weekly_bi",
            "pt_mainland_2027_weekly_tri",
        )
        for year in (2024, 2026)
    },
}


def _zone(name):
    return MADRID if name.startswith("es_") else LISBON


def _study_date(name):
    return date(2027, 7, 1) if name.startswith("pt_mainland_2027") else None


def _same_resolution(first, second):
    assert first.index.equals(second.index)
    for name in (
        "timezone",
        "schedule",
        "prices",
        "period_labels",
        "period_codes",
        "import_price_per_kwh",
        "export_price_per_kwh",
        "day_starts",
        "boundary_policy",
        "schedule_hash",
        "price_hash",
    ):
        assert getattr(first, name) == getattr(second, name), name


def _local_year(year, freq, zone):
    start, end = pd.Timestamp(f"{year}-01-01", tz=zone), pd.Timestamp(f"{year + 1}-01-01", tz=zone)
    return pd.date_range(start, end, freq=freq, inclusive="left").tz_convert("UTC")


def test_pins_cover_every_bundled_schedule_year_and_step():
    cases = set(PINS) | set(REJECTED)
    names = available_tariff_schedules()
    assert cases == {(name, year, freq) for name in names for year in (2024, 2026) for freq in ("h", "15min")}


@pytest.mark.parametrize(("name", "year", "freq"), sorted(PINS))
def test_bundled_schedule_labels_and_hash_are_unchanged(name, year, freq):
    steps, labels_digest, schedule_hash = PINS[name, year, freq]
    index = _local_year(year, freq, _zone(name))
    resolved = resolve_named_tariff(index, name, PRICES, timezone=_zone(name), study_date=_study_date(name))

    assert len(resolved.period_labels) == steps
    assert hashlib.sha256(",".join(resolved.period_labels).encode()).hexdigest()[:16] == labels_digest
    assert resolved.schedule_hash == schedule_hash


@pytest.mark.parametrize(("name", "year", "freq"), sorted(REJECTED))
def test_bundled_schedule_rejections_are_unchanged(name, year, freq):
    index = _local_year(year, freq, _zone(name))
    with pytest.raises(ValueError, match=REJECTED[name, year, freq]):
        resolve_named_tariff(index, name, PRICES, timezone=_zone(name), study_date=_study_date(name))


@pytest.mark.parametrize("name", available_tariff_schedules())
def test_a_definition_resolves_exactly_like_its_identifier(name):
    definition = get_schedule_definition(name)
    index = _local_year(2026, "15min", _zone(name))
    by_name = resolve_named_tariff(index, name, PRICES, timezone=_zone(name), study_date=_study_date(name))
    by_definition = resolve_named_tariff(index, definition, PRICES, timezone=_zone(name), study_date=_study_date(name))

    assert definition.schedule == get_tariff_schedule(name)
    _same_resolution(by_definition, by_name)
    assert parse_schedule_definition(name, load_config_json("tariffs.json")["schedules"][name]) == definition


def test_required_resolution_follows_the_interval_boundaries():
    # 2026 bi-hourly and 2.0TD change period only on even hours; the 2026
    # weekly tri-hourly schedule has quarter-hour boundaries.
    assert {name: schedule_resolution_minutes(name) for name in available_tariff_schedules()} == {
        "es_2_0td": 120,
        "pt_mainland_2026_daily_bi": 120,
        "pt_mainland_2026_daily_tri": 30,
        "pt_mainland_2026_weekly_bi": 30,
        "pt_mainland_2026_weekly_tri": 15,
        "pt_mainland_2027_daily_bi": 30,
        "pt_mainland_2027_daily_tri": 30,
        "pt_mainland_2027_weekly_bi": 30,
        "pt_mainland_2027_weekly_tri": 30,
    }
    two_hourly = pd.date_range("2026-01-12", periods=12, freq="2h", tz=LISBON)
    assert classify_tariff_periods(two_hourly, "pt_mainland_2026_daily_bi", timezone=LISBON) == (
        ("off_peak",) * 4 + ("peak",) * 7 + ("off_peak",)
    )


def _custom(**overrides):
    raw = {
        "version": "1",
        "timezone": LISBON,
        "cycle": "custom",
        "periods": ["off_peak", "peak"],
        "rules": [
            {
                "days": "weekday",
                "season": "all",
                "intervals": {"off_peak": [["00:00", "07:10"]], "peak": [["07:10", "24:00"]]},
            },
            {"days": "saturday", "season": "all", "intervals": {"off_peak": [["00:00", "24:00"]]}},
            {"days": "sunday", "season": "all", "intervals": {"off_peak": [["00:00", "24:00"]]}},
        ],
        # TOML gives dates, not strings.
        "holidays": {"day_type": "sunday", "dates": {"2026": [date(2026, 1, 1)]}},
        **overrides,
    }
    return parse_schedule_definition("my_supplier", raw, where="tariff.custom_schedule")


def test_a_custom_definition_classifies_resolves_and_pickles():
    definition = _custom()
    assert definition.resolution_minutes == 10
    assert definition.holidays == HolidayCalendar(
        day_type="sunday", dates=frozenset({date(2026, 1, 1)}), years=frozenset({2026})
    )

    # 2026-01-01 is a Thursday holiday, 2026-01-02 a Friday.
    index = pd.date_range("2026-01-01", "2026-01-03", freq="10min", tz=LISBON, inclusive="left")
    labels = classify_tariff_periods(index.tz_convert("UTC"), definition, timezone=LISBON)
    assert set(labels[:144]) == {"off_peak"}
    assert labels[144 + 42] == "off_peak" and labels[144 + 43] == "peak"
    with pytest.raises(ValueError, match="10-minute resolution or finer"):
        classify_tariff_periods(
            pd.date_range("2026-01-01", periods=4, freq="15min", tz=LISBON), definition, timezone=LISBON
        )

    spec = TariffSpec(schedule=definition, prices=PRICES)
    restored = pickle.loads(pickle.dumps(spec))
    assert restored == spec and hash(restored.definition) == hash(definition)
    _same_resolution(restored.resolve(index, LISBON), spec.resolve(index, LISBON))
    assert spec.resolve(index, LISBON).schedule.identifier == "my_supplier"


def test_holidays_without_years_are_complete_for_every_year():
    definition = _custom()
    open_ended = replace(definition, holidays=replace(definition.holidays, years=None))
    index = pd.date_range("2025-12-31", periods=2 * 144, freq="10min", tz=LISBON)

    with pytest.raises(ValueError, match="no holiday calendar for 2025"):
        classify_tariff_periods(index, definition, timezone=LISBON)
    labels = classify_tariff_periods(index, open_ended, timezone=LISBON)
    # 2025-12-31 is a Wednesday; 2026-01-01 is a holiday.
    assert labels[42] == "off_peak" and labels[43] == "peak" and set(labels[144:]) == {"off_peak"}


@pytest.mark.parametrize(
    ("overrides", "error", "match"),
    [
        (
            {"rules": [{"days": "all", "season": "all", "intervals": {"peak": [["00:00", "23:00"]]}}]},
            ValueError,
            r"tariff\.custom_schedule\.rules\[0\]': Intervals must cover",
        ),
        (
            {
                "rules": [
                    {"days": "all", "season": "all", "intervals": {"peak": [["00:00", "12:00"], ["13:00", "24:00"]]}}
                ]
            },
            ValueError,
            "gap at minute 720",
        ),
        (
            {
                "rules": [
                    {"days": "all", "season": "all", "intervals": {"peak": [["00:00", "13:00"], ["12:00", "24:00"]]}}
                ]
            },
            ValueError,
            "overlap at minute 720",
        ),
        (
            {"rules": [{"days": "all", "season": "all", "intervals": {"super_peak": [["00:00", "24:00"]]}}]},
            ValueError,
            r"rules\[0\] uses unknown period.*super_peak",
        ),
        (
            {"rules": [{"days": "weekday", "season": "all", "intervals": {"peak": [["00:00", "24:00"]]}}]},
            ValueError,
            "exactly one rule for saturday/standard",
        ),
        (
            {"rules": [{"days": "monday", "season": "all", "intervals": {"peak": [["00:00", "24:00"]]}}]},
            ValueError,
            "Unknown day selector",
        ),
        (
            {"rules": [{"days": "all", "season": "all", "intervals": {"peak": [["00:00", "24:30"]]}}]},
            ValueError,
            r"intervals\.peak\[0\]\[1\]' must be a clock value",
        ),
        (
            {"holidays": {"day_type": "sunday", "dates": {"2026": ["2027-01-01"]}}},
            ValueError,
            "must hold dates in 2026",
        ),
        ({"holidays": {"day_type": "monday", "dates": {"2026": ["2026-01-01"]}}}, ValueError, r"holidays\.day_type"),
        ({"effective_from": "July 2027"}, ValueError, r"tariff\.custom_schedule\.effective_from"),
        ({"cycle": "hourly"}, ValueError, "schedule.cycle"),
    ],
)
def test_parse_rejects_malformed_definitions(overrides, error, match):
    with pytest.raises(error, match=match):
        _custom(**overrides)


def test_definitions_validate_when_built_directly():
    schedule = get_tariff_schedule("pt_mainland_2026_daily_bi")
    rule = ScheduleRule(days="all", season="all", intervals=((480, 1440, "peak"), (0, 480, "off_peak")))

    assert rule.intervals == ((0, 480, "off_peak"), (480, 1440, "peak"))
    assert ScheduleDefinition(schedule=schedule, rules=[rule]).rules == (rule,)
    with pytest.raises(ValueError, match="exactly one rule for weekday/standard; found 2"):
        ScheduleDefinition(schedule=schedule, rules=(rule, rule))
    with pytest.raises(ValueError, match="whole minutes"):
        ScheduleRule(days="all", season="all", intervals=((0, 1500, "peak"),))
    with pytest.raises(ValueError, match="outside the calendar's years"):
        HolidayCalendar(day_type="sunday", dates=frozenset({date(2027, 1, 1)}), years=frozenset({2026}))
    with pytest.raises(TypeError, match="bundled schedule identifier or a ScheduleDefinition"):
        classify_tariff_periods(pd.date_range("2026-01-01", periods=2, freq="h", tz=LISBON), schedule, timezone=LISBON)
