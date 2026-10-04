from dataclasses import FrozenInstanceError
from datetime import date

import pandas as pd
import pytest

from breos.tariffs import (
    TariffPrices,
    TariffSchedule,
    available_tariff_schedules,
    classify_tariff_periods,
    get_tariff_schedule,
    resolve_flat_tariff,
    resolve_named_tariff,
    resolve_tariff,
)

LISBON = "Europe/Lisbon"


def _daily_schedule() -> TariffSchedule:
    return TariffSchedule(
        identifier="test_daily",
        version="2026-01",
        timezone="Europe/Lisbon",
        cycle="daily",
        periods=("off_peak", "peak"),
        source_url="https://example.test/schedule",
    )


def test_tariff_values_are_frozen_and_currency_qualified():
    schedule = _daily_schedule()
    prices = TariffPrices(
        currency="eur",
        import_prices={"off_peak": 0.12, "peak": 0.30},
        export_prices={"all": 0.04},
    )

    assert prices.currency == "EUR"
    with pytest.raises(TypeError):
        prices.import_prices["peak"] = 1.0
    with pytest.raises(FrozenInstanceError):
        schedule.cycle = "weekly"


@pytest.mark.parametrize("currency", ["XYZ", "EURO", "US$"])
def test_unknown_currency_fails_before_resolution(currency):
    with pytest.raises(ValueError, match="must be an ISO 4217 currency code"):
        TariffPrices(currency=currency, import_prices={"all": 0.2}, export_prices={"all": 0.04})


def test_currency_codes_are_upper_cased():
    prices = TariffPrices(currency=" usd ", import_prices={"all": 20.0}, export_prices={"all": 4.0})
    assert prices.currency == "USD"


@pytest.mark.parametrize("value", [-0.01, float("nan"), float("inf"), True])
def test_invalid_prices_are_rejected(value):
    error = TypeError if value is True else ValueError
    with pytest.raises(error):
        TariffPrices(currency="EUR", import_prices={"all": value}, export_prices={"all": 0.04})


def test_resolve_tariff_uses_explicit_all_fallback_and_stable_codes():
    index = pd.date_range("2026-01-01", periods=4, freq="h", tz="Europe/Lisbon")
    schedule = _daily_schedule()
    prices = TariffPrices(
        currency="EUR",
        import_prices={"off_peak": 0.12, "peak": 0.30},
        export_prices={"all": 0.04},
        fixed_charge_per_day=0.25,
    )

    resolved = resolve_tariff(index, ("off_peak", "off_peak", "peak", "peak"), schedule, prices, timezone=LISBON)

    assert resolved.period_codes == (0, 0, 1, 1)
    assert resolved.import_price_per_kwh == (0.12, 0.12, 0.30, 0.30)
    assert resolved.export_price_per_kwh == (0.04, 0.04, 0.04, 0.04)
    assert resolved.prices.fixed_charge_per_day == 0.25
    assert len(resolved.schedule_hash) == 64
    assert len(resolved.price_hash) == 64


def test_missing_used_period_price_fails_without_country_or_period_fallback():
    index = pd.date_range("2026-01-01", periods=2, freq="h", tz="Europe/Lisbon")
    prices = TariffPrices(currency="EUR", import_prices={"off_peak": 0.12}, export_prices={"all": 0.04})

    with pytest.raises(ValueError, match="Missing import price.*peak"):
        resolve_tariff(index, ("off_peak", "peak"), _daily_schedule(), prices, timezone=LISBON)


def test_unknown_price_period_fails_even_when_unused():
    index = pd.date_range("2026-01-01", periods=1, freq="h", tz="Europe/Lisbon")
    prices = TariffPrices(
        currency="EUR",
        import_prices={"all": 0.12, "typo_peak": 0.30},
        export_prices={"all": 0.04},
    )

    with pytest.raises(ValueError, match="Unknown import_prices period.*typo_peak"):
        resolve_tariff(index, ("off_peak",), _daily_schedule(), prices, timezone=LISBON)


def test_same_instants_in_different_timezones_have_same_schedule_hash():
    utc_index = pd.date_range("2026-10-25 00:00", periods=5, freq="h", tz="UTC")
    lisbon_index = utc_index.tz_convert("Europe/Lisbon")
    prices = TariffPrices(currency="EUR", import_prices={"all": 0.2}, export_prices={"all": 0.04})
    labels = ("off_peak",) * len(utc_index)

    utc_resolved = resolve_tariff(utc_index, labels, _daily_schedule(), prices, timezone=LISBON)
    lisbon_resolved = resolve_tariff(lisbon_index, labels, _daily_schedule(), prices, timezone=LISBON)

    assert utc_resolved.schedule_hash == lisbon_resolved.schedule_hash
    assert lisbon_index[0].hour == lisbon_index[1].hour == 1
    assert lisbon_index[0].utcoffset() != lisbon_index[1].utcoffset()


def test_schedule_hash_is_independent_of_pandas_datetime_storage_unit():
    microseconds = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC").as_unit("us")
    nanoseconds = microseconds.as_unit("ns")

    first = resolve_flat_tariff(
        timezone=LISBON, index=microseconds, import_price_per_kwh=0.20, export_price_per_kwh=0.04
    )
    second = resolve_flat_tariff(
        timezone=LISBON, index=nanoseconds, import_price_per_kwh=0.20, export_price_per_kwh=0.04
    )

    assert first.schedule_hash == second.schedule_hash


def test_schedule_and_price_hashes_change_independently():
    index = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")
    first = resolve_flat_tariff(timezone=LISBON, index=index, import_price_per_kwh=0.20, export_price_per_kwh=0.04)
    repriced = resolve_flat_tariff(timezone=LISBON, index=index, import_price_per_kwh=0.25, export_price_per_kwh=0.04)

    assert first.schedule_hash == repriced.schedule_hash
    assert first.price_hash != repriced.price_hash


def test_resolution_requires_unique_monotonic_timezone_aware_instants():
    naive = pd.date_range("2026-01-01", periods=2, freq="h")
    with pytest.raises(ValueError, match="timezone-aware"):
        resolve_flat_tariff(timezone=LISBON, index=naive, import_price_per_kwh=0.2, export_price_per_kwh=0.04)

    duplicate = pd.DatetimeIndex(["2026-01-01T00:00Z", "2026-01-01T00:00Z"])
    with pytest.raises(ValueError, match="unique instants"):
        resolve_flat_tariff(timezone=LISBON, index=duplicate, import_price_per_kwh=0.2, export_price_per_kwh=0.04)


def test_bundled_schedule_catalog_contains_price_independent_portuguese_cycles():
    identifiers = available_tariff_schedules()

    assert identifiers == tuple(sorted(identifiers))
    assert {
        "pt_mainland_2026_daily_bi",
        "pt_mainland_2026_daily_tri",
        "pt_mainland_2026_weekly_bi",
        "pt_mainland_2026_weekly_tri",
        "pt_mainland_2027_daily_bi",
        "pt_mainland_2027_daily_tri",
        "pt_mainland_2027_weekly_bi",
        "pt_mainland_2027_weekly_tri",
    } <= set(identifiers)

    schedule = get_tariff_schedule("pt_mainland_2027_daily_tri")
    assert schedule.timezone == "Europe/Lisbon"
    assert schedule.periods == ("off_peak", "mid_peak", "peak")
    assert schedule.effective_from == date(2027, 7, 1)


def test_unknown_bundled_schedule_fails_without_country_fallback():
    with pytest.raises(KeyError, match="Unknown tariff schedule.*pt"):
        get_tariff_schedule("pt")


@pytest.mark.parametrize(
    ("timestamp", "expected"),
    [
        ("2026-01-12 08:59", "mid_peak"),
        ("2026-01-12 09:00", "peak"),
        ("2026-01-12 10:30", "mid_peak"),
        ("2026-07-13 10:29", "mid_peak"),
        ("2026-07-13 10:30", "peak"),
        ("2026-07-13 13:00", "mid_peak"),
    ],
)
def test_portuguese_2026_daily_tri_classifies_standard_and_dst_boundaries(timestamp, expected):
    instant = pd.DatetimeIndex([timestamp]).tz_localize("Europe/Lisbon")

    assert classify_tariff_periods(instant, "pt_mainland_2026_daily_tri", timezone=LISBON) == (expected,)


def test_portuguese_2026_weekly_tri_requires_quarter_hour_alignment():
    hourly = pd.date_range("2026-07-13", periods=24, freq="h", tz="Europe/Lisbon")
    with pytest.raises(ValueError, match="requires 15-minute resolution"):
        classify_tariff_periods(hourly, "pt_mainland_2026_weekly_tri", timezone=LISBON)

    offset = pd.date_range("2026-07-13 00:05", periods=4, freq="15min", tz="Europe/Lisbon")
    with pytest.raises(ValueError, match="boundaries do not align"):
        classify_tariff_periods(offset, "pt_mainland_2026_weekly_tri", timezone=LISBON)

    quarter_hour = pd.date_range("2026-07-13 09:00", periods=15, freq="15min", tz="Europe/Lisbon")
    labels = classify_tariff_periods(quarter_hour, "pt_mainland_2026_weekly_tri", timezone=LISBON)
    assert labels[:1] == ("mid_peak",)
    assert labels[1:13] == ("peak",) * 12
    assert labels[13:] == ("mid_peak",) * 2


@pytest.mark.parametrize(
    ("day", "expected_steps"),
    [("2026-03-29", 23), ("2026-10-25", 25)],
)
def test_tariff_classification_preserves_portuguese_dst_day_instants(day, expected_steps):
    index = pd.date_range(day, pd.Timestamp(day) + pd.Timedelta(days=1), freq="h", inclusive="left", tz="Europe/Lisbon")

    labels = classify_tariff_periods(index, "pt_mainland_2026_daily_bi", timezone=LISBON)

    assert len(index) == expected_steps
    assert len(labels) == expected_steps


def test_same_instants_classify_identically_from_different_timezone_views():
    utc_index = pd.date_range("2026-10-25 00:00", periods=5, freq="h", tz="UTC")
    lisbon_index = utc_index.tz_convert("Europe/Lisbon")

    assert classify_tariff_periods(utc_index, "pt_mainland_2026_daily_bi", timezone=LISBON) == classify_tariff_periods(
        lisbon_index, "pt_mainland_2026_daily_bi", timezone=LISBON
    )


def test_future_schedule_requires_an_explicit_effective_study_date_for_reference_year_data():
    reference_year = pd.date_range("2005-01-01", periods=2, freq="30min", tz="Europe/Lisbon")

    with pytest.raises(ValueError, match="not effective across the index date range.*study_date"):
        classify_tariff_periods(reference_year, "pt_mainland_2027_daily_tri", timezone=LISBON)

    assert classify_tariff_periods(
        reference_year,
        "pt_mainland_2027_daily_tri",
        timezone=LISBON,
        study_date=date(2027, 7, 1),
    ) == ("off_peak", "off_peak")


def test_named_tariff_combines_packaged_schedule_with_user_prices():
    index = pd.date_range("2027-07-05 17:00", periods=3, freq="30min", tz="Europe/Lisbon")
    prices = TariffPrices(
        currency="EUR",
        import_prices={"off_peak": 0.12, "mid_peak": 0.20, "peak": 0.30},
        export_prices={"all": 0.04},
    )

    resolved = resolve_named_tariff(index, "pt_mainland_2027_daily_tri", prices, timezone=LISBON)

    assert resolved.period_labels == ("mid_peak", "peak", "peak")
    assert resolved.import_price_per_kwh == (0.20, 0.30, 0.30)
    assert resolved.schedule.identifier == "pt_mainland_2027_daily_tri"


# --- ADR 0002 A1: configured civil time and civil-day boundaries -------------


@pytest.mark.parametrize(
    ("clock", "freq"),
    [("UTC", "h"), ("Etc/GMT-1", "h"), ("Europe/Lisbon", "h"), ("UTC", "15min"), ("Etc/GMT+5", "15min")],
)
def test_index_clock_does_not_change_the_periods(clock, freq):
    # The same instants on a UTC, fixed-offset, or IANA index classify alike:
    # the resolver converts to the configured zone and never reads index.tz.
    instants = pd.date_range("2026-03-28", "2026-04-01", freq=freq, tz="UTC", inclusive="left")
    reference = classify_tariff_periods(instants, "pt_mainland_2026_daily_bi", timezone=LISBON)

    assert (
        classify_tariff_periods(instants.tz_convert(clock), "pt_mainland_2026_daily_bi", timezone=LISBON) == reference
    )


@pytest.mark.parametrize(("freq", "per_hour"), [("h", 1), ("15min", 4)])
def test_day_starts_follow_civil_days_across_dst(freq, per_hour):
    index = pd.date_range("2026-03-28", "2026-11-01", freq=freq, tz="UTC", inclusive="left")
    resolved = resolve_flat_tariff(timezone=LISBON, index=index, import_price_per_kwh=0.2, export_price_per_kwh=0.05)

    starts = resolved.day_starts
    lengths = {
        index[start].tz_convert(LISBON).date(): (end - start) // per_hour for start, end in zip(starts, starts[1:])
    }
    assert starts[0] == 0 and starts[-1] == len(index)
    assert lengths[date(2026, 3, 29)] == 23
    assert lengths[date(2026, 10, 25)] == 25
    assert lengths[date(2026, 6, 1)] == 24
    # Every step belongs to exactly one civil day, which starts at local midnight.
    assert sum(end - start for start, end in zip(starts, starts[1:])) == len(index)
    for start in starts[1:-1]:
        local = index[start].tz_convert(LISBON)
        assert (local.hour, local.minute) == (0, 0)
    assert resolved.timezone == LISBON


def test_a_schedule_is_not_moved_to_another_zone():
    index = pd.date_range("2026-01-01", periods=24, freq="h", tz="UTC")

    with pytest.raises(ValueError, match="defined in Europe/Lisbon civil time.*Europe/Berlin"):
        classify_tariff_periods(index, "pt_mainland_2026_daily_bi", timezone="Europe/Berlin")
    with pytest.raises(ValueError, match="Unknown IANA timezone"):
        classify_tariff_periods(index, "pt_mainland_2026_daily_bi", timezone="Mars/Olympus")


def test_flat_tariff_has_a_neutral_identity():
    index = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
    resolved = resolve_flat_tariff(timezone="UTC", index=index, import_price_per_kwh=0.2, export_price_per_kwh=0.05)

    assert (resolved.prices.identifier, resolved.prices.version) == ("flat", "1")
    assert resolved.import_price_per_kwh == (0.2, 0.2, 0.2)
    assert resolved.n_days == 1
