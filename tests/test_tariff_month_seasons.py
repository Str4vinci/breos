"""Calendar-month seasons in custom tariff schedules (#337).

A schedule may name seasons by calendar month, a quarter being three months.
Its rules select a season by name, and its prices may be given per season and
period. These tests pin the partition and coverage rules, classification at
the year, quarter and clock-change boundaries, and every valuation path: App,
Monte Carlo, projected optimization and ``App.revalue``.
"""

import json
import pickle
from copy import deepcopy
from datetime import timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from breos import _daily_persistence, optimization
from breos.app import App
from breos.montecarlo import MonteCarloSettings, run_montecarlo
from breos.runners.app import run_app_simulation
from breos.tariffs import (
    MonthSeasons,
    ScheduleDefinition,
    ScheduleRule,
    TariffPrices,
    TariffSchedule,
    classify_tariff_periods,
    classify_tariff_seasons,
    parse_schedule_definition,
    resolve_named_tariff,
    resolve_tariff,
    tariff_provenance,
)

QUARTERS = {"q1": [1, 2, 3], "q2": [4, 5, 6], "q3": [7, 8, 9], "q4": [10, 11, 12]}
# A §14a Module 3-style day: low, standard and high windows. The windows and
# prices are illustrative, not a network operator's published values.
WINDOWED = {
    "low": [["00:00", "06:00"]],
    "standard": [["06:00", "17:00"], ["21:00", "24:00"]],
    "high": [["17:00", "21:00"]],
}
STANDARD_ALL_DAY = {"standard": [["00:00", "24:00"]]}


def _quarterly(timezone="Europe/Berlin", identifier="illustrative_module3"):
    """Windows in Q1 and Q4, standard all day in Q2 and Q3."""
    return {
        "identifier": identifier,
        "version": "1",
        "timezone": timezone,
        "cycle": "custom",
        "periods": ["low", "standard", "high"],
        "seasons": deepcopy(QUARTERS),
        "rules": [
            {"days": "all", "season": "q1", "intervals": deepcopy(WINDOWED)},
            {"days": "all", "season": "q2", "intervals": deepcopy(STANDARD_ALL_DAY)},
            {"days": "all", "season": "q3", "intervals": deepcopy(STANDARD_ALL_DAY)},
            {"days": "all", "season": "q4", "intervals": deepcopy(WINDOWED)},
        ],
    }


def _definition(raw=None, timezone="Europe/Berlin"):
    raw = deepcopy(raw) if raw is not None else _quarterly(timezone)
    identifier = raw.pop("identifier")
    return parse_schedule_definition(identifier, raw, where="tariff.custom_schedule")


WINDOW_PRICES = {"low": 0.25, "standard": 0.33, "high": 0.42}
SEASON_IMPORT = {
    "q1": dict(WINDOW_PRICES),
    "q2": {"standard": 0.31},
    "q3": {"standard": 0.30},
    "q4": dict(WINDOW_PRICES),
}
SEASON_PRICES = TariffPrices(currency="EUR", import_prices=SEASON_IMPORT, export_prices={"all": 0.08})


# --- the partition --------------------------------------------------------


@pytest.mark.parametrize(
    ("seasons", "message"),
    [
        ({"winter": [1, 2, 3, 10, 11, 12], "summer": [4, 5, 6, 7, 8]}, r"Month\(s\) 9 are in no season"),
        (
            {"winter": [1, 2, 3, 4, 10, 11, 12], "summer": [4, 5, 6, 7, 8, 9]},
            "Month 4 is in both 'winter' and 'summer'",
        ),
        ({"year": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12]}, "Month 12 is twice"),
        ({"year": [0, *range(1, 13)]}, "months must be whole numbers from 1 through 12, not 0"),
        ({"year": [*range(1, 12), 13]}, "not 13"),
        ({"year": [True, *range(2, 13)]}, "not True"),
        ({"year": [1.0, *range(2, 13)]}, "not 1.0"),
        ({"empty": [], "year": list(range(1, 13))}, "Season 'empty' must have at least one month"),
        ({"dst": list(range(1, 13))}, "Season name 'dst' is reserved"),
        ({"all": list(range(1, 13))}, "Season name 'all' is reserved"),
        ({"Winter": list(range(1, 13))}, "lowercase letters"),
    ],
)
def test_month_seasons_must_partition_the_year(seasons, message):
    with pytest.raises((TypeError, ValueError), match=message):
        MonthSeasons(tuple((name, tuple(months)) for name, months in seasons.items()))


def test_month_seasons_are_immutable_and_pickle():
    seasons = MonthSeasons(tuple((name, tuple(months)) for name, months in QUARTERS.items()))
    assert seasons.names == ("q1", "q2", "q3", "q4")
    assert [seasons.season_of_month(month) for month in (1, 3, 4, 9, 10, 12)] == ["q1", "q1", "q2", "q3", "q4", "q4"]
    assert seasons.as_dict() == QUARTERS
    definition = _definition()
    assert pickle.loads(pickle.dumps(definition)) == definition
    assert pickle.loads(pickle.dumps(SEASON_PRICES)) == SEASON_PRICES


# --- the rules ------------------------------------------------------------


def _mutated(mutate):
    raw = _quarterly()
    mutate(raw)
    return raw


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda raw: raw["rules"][0].update(season="standard"),
            r"rules\[0\] selects season 'standard', but its month seasons are q1, q2, q3, q4",
        ),
        (
            lambda raw: raw["rules"].pop(),
            "must define exactly one rule for weekday/q4; found 0",
        ),
        (
            lambda raw: raw["rules"].append({"days": "sunday", "season": "q4", "intervals": STANDARD_ALL_DAY}),
            "must define exactly one rule for sunday/q4; found 2",
        ),
        (
            lambda raw: raw["rules"].append({"days": "all", "season": "all", "intervals": STANDARD_ALL_DAY}),
            "must define exactly one rule for weekday/q1; found 2",
        ),
        (
            lambda raw: (
                raw.update(seasons={"high": list(range(1, 13))})
                or raw.update(rules=[{"days": "all", "season": "all", "intervals": STANDARD_ALL_DAY}])
            ),
            "uses 'high' as both a season and a period",
        ),
        (
            lambda raw: raw.update(seasons={"q1": [1, 2, 3]}),
            r"Month\(s\) 4, 5, 6, 7, 8, 9, 10, 11, 12 are in no season",
        ),
        (lambda raw: raw.update(seasons=[1, 2]), "must map season names to lists of months"),
        (lambda raw: raw["seasons"].update(q1=3), r"seasons\.q1' must be a list of months"),
    ],
)
def test_rules_select_month_seasons_with_exactly_one_rule_each(mutate, message):
    with pytest.raises((TypeError, ValueError), match=message):
        _definition(_mutated(mutate))


def test_without_month_seasons_a_rule_cannot_select_one():
    raw = _quarterly()
    del raw["seasons"]
    with pytest.raises(ValueError, match=r"selects season 'q1', but without month seasons a rule selects standard"):
        _definition(raw)
    with pytest.raises(ValueError, match="Unknown season selector"):
        ScheduleRule(days="all", season="Q1", intervals=((0, 1440, "standard"),))


def test_a_definition_built_directly_checks_its_seasons():
    schedule = TariffSchedule("direct", "1", "Europe/Berlin", "custom", ("standard",))
    rule = ScheduleRule(days="all", season="all", intervals=((0, 1440, "standard"),))
    with pytest.raises(TypeError, match="'seasons' must be MonthSeasons"):
        ScheduleDefinition(schedule, (rule,), seasons={"year": list(range(1, 13))})  # type: ignore[arg-type]
    year = MonthSeasons((("year", tuple(range(1, 13))),))
    assert ScheduleDefinition(schedule, (rule,), seasons=year).season_names == ("year",)


def test_season_periods_are_the_periods_its_rules_use():
    definition = _definition()
    assert definition.season_periods("q1") == {"low", "standard", "high"}
    assert definition.season_periods("q2") == {"standard"}


# --- classification at the boundaries --------------------------------------


def _resolve(index, prices=SEASON_PRICES, definition=None, zone="Europe/Berlin"):
    return resolve_named_tariff(index, definition or _definition(timezone=zone), prices, timezone=zone)


@pytest.mark.parametrize("freq", ["h", "15min"])
def test_the_year_boundary_moves_from_q4_to_q1_on_the_local_date(freq):
    # In UTC the last local hour of 2025 is 22:00 on 31 December: the local
    # date, not the UTC one, selects the season.
    index = pd.date_range("2025-12-31 20:00", "2026-01-01 04:00", freq=freq, tz="UTC", inclusive="left")
    resolved = _resolve(index)
    local = index.tz_convert("Europe/Berlin")
    seasons = np.asarray(resolved.season_labels)
    assert set(seasons[local.year == 2025]) == {"q4"}
    assert set(seasons[local.year == 2026]) == {"q1"}
    # 21:00-24:00 is standard, 00:00-06:00 low, in both quarters.
    labels = np.asarray(resolved.period_labels)
    assert set(labels[local.year == 2025]) == {"standard"}
    assert set(labels[local.year == 2026]) == {"low"}
    assert len(resolved.import_price_per_kwh) == len(index)


@pytest.mark.parametrize("freq", ["h", "15min"])
def test_the_quarter_boundary_follows_the_march_clock_change(freq):
    # Clocks go forward at 02:00 on Sunday 29 March 2026; Q2 starts on 1 April.
    index = pd.date_range("2026-03-29", "2026-04-02", freq=freq, tz="Europe/Berlin", inclusive="left")
    resolved = _resolve(index.tz_convert("UTC"))
    local = index
    seasons = pd.Series(resolved.season_labels, index=local)
    labels = pd.Series(resolved.period_labels, index=local)
    prices = pd.Series(resolved.import_price_per_kwh, index=local)
    steps_per_hour = 4 if freq == "15min" else 1
    # The 23-hour day keeps every step it has, all in Q1, with the window.
    assert (local.date == pd.Timestamp("2026-03-29").date()).sum() == 23 * steps_per_hour
    assert set(seasons[:"2026-03-31 23:59"]) == {"q1"}
    assert set(seasons["2026-04-01":]) == {"q2"}
    evening, night = (pd.Timestamp(f"2026-03-31 {clock}", tz="Europe/Berlin") for clock in ("18:00", "03:00"))
    assert labels[evening] == "high" and prices[evening] == 0.42
    assert labels[night] == "low" and prices[night] == 0.25
    assert set(labels["2026-04-01":]) == {"standard"} and set(prices["2026-04-01":]) == {0.31}


@pytest.mark.parametrize("freq", ["h", "15min"])
def test_the_october_clock_change_stays_inside_q4(freq):
    # Clocks go back at 03:00 on Sunday 25 October 2026: 02:00-03:00 occurs twice.
    index = pd.date_range("2026-10-25", "2026-10-26", freq=freq, tz="Europe/Berlin", inclusive="left")
    resolved = _resolve(index.tz_convert("UTC"))
    steps_per_hour = 4 if freq == "15min" else 1
    assert len(index) == 25 * steps_per_hour
    assert set(resolved.season_labels) == {"q4"}
    labels = np.asarray(resolved.period_labels)
    repeated = np.flatnonzero(index.hour == 2)
    assert len(repeated) == 2 * steps_per_hour
    assert set(labels[repeated]) == {"low"}
    assert resolved.day_starts == (0, len(index))


def test_the_index_timezone_does_not_move_a_season():
    index = pd.date_range("2026-03-30", "2026-04-03", freq="h", tz="Europe/Berlin", inclusive="left")
    reference = _resolve(index)
    for other in (
        index.tz_convert("UTC"),
        index.tz_convert("Asia/Tokyo"),
        index.tz_convert(timezone(timedelta(hours=5))),
    ):
        resolved = _resolve(other)
        assert resolved.season_labels == reference.season_labels
        assert resolved.period_labels == reference.period_labels
        assert resolved.schedule_hash == reference.schedule_hash


def test_holidays_and_day_types_combine_with_month_seasons():
    raw = _quarterly()
    raw["rules"] = [
        {"days": "weekday", "season": "q1", "intervals": WINDOWED},
        {"days": "saturday", "season": "q1", "intervals": STANDARD_ALL_DAY},
        {"days": "sunday", "season": "q1", "intervals": {"low": [["00:00", "24:00"]]}},
        {"days": "all", "season": "q2", "intervals": STANDARD_ALL_DAY},
        {"days": "all", "season": "q3", "intervals": STANDARD_ALL_DAY},
        {"days": "all", "season": "q4", "intervals": WINDOWED},
    ]
    raw["holidays"] = {"day_type": "sunday", "dates": {"2026": ["2026-01-01"]}}
    definition = _definition(raw)
    # Thursday 1 January is a holiday, taking the Sunday rule; Friday 2 January is a weekday.
    index = pd.date_range("2026-01-01", "2026-01-04", freq="h", tz="Europe/Berlin", inclusive="left")
    labels = pd.Series(classify_tariff_periods(index, definition, timezone="Europe/Berlin"), index=index)
    assert set(labels["2026-01-01"]) == {"low"}
    assert labels["2026-01-02 18:00"] == "high"
    assert set(labels["2026-01-03"]) == {"standard"}


# --- prices and hashes ----------------------------------------------------


def test_seasonal_prices_price_each_step_by_its_season_and_period():
    index = pd.date_range("2026-01-01", "2027-01-01", freq="h", tz="Europe/Berlin", inclusive="left")
    resolved = _resolve(index)
    local = index
    prices = np.asarray(resolved.import_price_per_kwh)
    expected = np.select(
        [
            local.quarter.isin([1, 4]) & (local.hour < 6),
            local.quarter.isin([1, 4]) & (local.hour >= 17) & (local.hour < 21),
        ],
        [0.25, 0.42],
        default=np.select([local.quarter == 2, local.quarter == 3], [0.31, 0.30], default=0.33),
    )
    np.testing.assert_array_equal(prices, expected)
    assert set(resolved.export_price_per_kwh) == {0.08}


@pytest.mark.parametrize(
    ("import_prices", "error", "message"),
    [
        ({"q1": {"all": 0.3}, "peak": 0.2}, ValueError, "mixes prices and season tables"),
        ({"q1": {}}, ValueError, r"'prices\.import_prices\.q1' must define at least one price"),
        ({"q1": {"low": -1.0}}, ValueError, r"import_prices\.q1\.low' must be >= 0"),
    ],
)
def test_seasonal_price_lists_are_frozen_strictly(import_prices, error, message):
    with pytest.raises(error, match=message):
        TariffPrices(currency="EUR", import_prices=import_prices, export_prices={"all": 0.0})


@pytest.mark.parametrize(
    ("import_prices", "definition", "message"),
    [
        ({"q1": {"all": 0.3}}, None, r"no prices for season\(s\) q2, q3, q4"),
        ({**SEASON_IMPORT, "q5": {"all": 0.3}}, None, r"has season\(s\) q5 that schedule"),
        ({**SEASON_IMPORT, "q2": {"peak": 0.3}}, None, r"q2' has period\(s\) peak that schedule"),
        ({**SEASON_IMPORT, "q1": {"low": 0.3}}, None, "q1' has no price for high, standard"),
        (SEASON_IMPORT, "pt_mainland_2026_daily_bi", "has no month seasons"),
    ],
)
def test_resolution_rejects_prices_its_seasons_cannot_use(import_prices, definition, message):
    index = pd.date_range("2026-01-01", periods=24 * 7, freq="h", tz="Europe/Lisbon")
    prices = TariffPrices(currency="EUR", import_prices=import_prices, export_prices={"all": 0.0})
    zone = "Europe/Lisbon" if definition else "Europe/Berlin"
    with pytest.raises(ValueError, match=message):
        resolve_named_tariff(index, definition or _definition(), prices, timezone=zone)


@pytest.mark.parametrize("name", ["import_prices", "export_prices"])
@pytest.mark.parametrize(
    ("season", "season_prices", "start", "end", "message"),
    [
        (
            "q2",
            {"standard": 0.31, "high": 0.99},
            "2026-04-01",
            "2026-07-01",
            r"q2' prices high, which season 'q2' never uses",
        ),
        (
            "q4",
            {"low": 0.25, "standard": 0.33},
            "2026-01-01",
            "2026-02-01",
            r"q4' has no price for high",
        ),
    ],
)
def test_named_resolution_validates_every_seasons_used_periods(name, season, season_prices, start, end, message):
    index = pd.date_range(start, end, freq="h", tz="Europe/Berlin", inclusive="left")
    values = {**SEASON_IMPORT, season: season_prices}
    price_lists = {"import_prices": SEASON_IMPORT, "export_prices": SEASON_IMPORT}
    price_lists[name] = values
    prices = TariffPrices(currency="EUR", **price_lists)
    with pytest.raises(ValueError, match=message):
        resolve_named_tariff(index, _definition(), prices, timezone="Europe/Berlin")


def test_the_two_step_path_resolves_a_month_season_schedule_as_one_step_does():
    index = pd.date_range("2026-03-25", periods=24 * 14, freq="h", tz="Europe/Berlin")
    definition = _definition()
    one_step = resolve_named_tariff(index, definition, SEASON_PRICES, timezone="Europe/Berlin")
    labels = classify_tariff_periods(index, definition, timezone="Europe/Berlin")
    season_labels = classify_tariff_seasons(index, definition, timezone="Europe/Berlin")
    two_step = resolve_tariff(
        index,
        labels,
        definition.schedule,
        SEASON_PRICES,
        timezone="Europe/Berlin",
        season_labels=season_labels,
        seasons=definition.seasons,
    )
    assert two_step.schedule_hash == one_step.schedule_hash
    assert two_step.import_price_per_kwh == one_step.import_price_per_kwh
    assert set(season_labels) == {"q1", "q2"}
    # Without month seasons there are no season labels to pass.
    assert classify_tariff_seasons(index[:24], "pt_mainland_2026_daily_bi", timezone="Europe/Lisbon") is None
    with pytest.raises(ValueError, match="was resolved without month seasons"):
        resolve_tariff(index, labels, definition.schedule, SEASON_PRICES, timezone="Europe/Berlin")


def test_bundled_schedule_and_flat_price_hashes_are_unchanged():
    # Pinned against origin/develop 8bb75b8, before seasons existed.
    index = pd.date_range("2026-01-01", periods=48 * 7, freq="h", tz="Europe/Lisbon")
    prices = TariffPrices(
        currency="EUR",
        import_prices={"peak": 0.28, "off_peak": 0.11},
        export_prices={"all": 0.05},
        fixed_charge_per_day=0.25,
    )
    resolved = resolve_named_tariff(index, "pt_mainland_2026_daily_bi", prices, timezone="Europe/Lisbon")
    assert resolved.schedule_hash == "3c782b4d4ac33ea71e009e24ba997b418687c3fc1f251de99cde0a863aaddeaf"
    assert resolved.price_hash == "f1320b850176039384e386a76bb094f88aae5b68c331f05ba498fe39bd2e5597"
    assert resolved.season_labels is None and resolved.seasons is None
    assert "seasons" not in tariff_provenance(resolved, calendar_year=2026)


def test_the_month_partition_joins_the_schedule_hash():
    # Two partitions with identical rules give identical periods, but each
    # step's season, and so its price bucket, differs.
    index = pd.date_range("2026-01-01", "2027-01-01", freq="h", tz="Europe/Berlin", inclusive="left")
    one_rule = {**_quarterly(), "rules": [{"days": "all", "season": "all", "intervals": WINDOWED}]}
    halves = {**one_rule, "seasons": {"winter": [1, 2, 3, 10, 11, 12], "summer": [4, 5, 6, 7, 8, 9]}}
    no_seasons = {key: value for key, value in one_rule.items() if key != "seasons"}
    prices = TariffPrices(currency="EUR", import_prices={"all": 0.3}, export_prices={"all": 0.0})
    quarterly, halved, plain = (_resolve(index, prices, _definition(raw)) for raw in (one_rule, halves, no_seasons))
    assert quarterly.period_labels == halved.period_labels == plain.period_labels
    assert len({quarterly.schedule_hash, halved.schedule_hash, plain.schedule_hash}) == 3
    assert quarterly.price_hash == halved.price_hash == plain.price_hash


def test_provenance_records_the_partition_and_the_seasonal_prices():
    index = pd.date_range("2026-01-01", "2026-01-02", freq="h", tz="Europe/Berlin", inclusive="left")
    record = tariff_provenance(_resolve(index), calendar_year=2026)
    assert record["seasons"] == QUARTERS
    assert record["import_prices"] == SEASON_IMPORT
    assert record["export_prices"] == {"all": 0.08}
    assert json.loads(json.dumps(record)) == record


# --- App ------------------------------------------------------------------

BASE = {"location": "porto", "n_modules": 8, "annual_consumption_kwh": 4000, "battery_kwh": 5.0, "projection_years": 2}
LISBON_QUARTERLY = _quarterly("Europe/Lisbon", identifier="illustrative_quarterly_lisbon")
SEASONAL_TARIFF = {
    "custom_schedule": LISBON_QUARTERLY,
    "currency": "EUR",
    "import_prices": deepcopy(SEASON_IMPORT),
    "export_prices": {"q1": {"all": 0.06}, "q2": {"all": 0.04}, "q3": {"all": 0.04}, "q4": {"all": 0.06}},
    "fixed_charge_per_day": 0.2,
}
PERIOD_TARIFF = {**SEASONAL_TARIFF, "import_prices": dict(WINDOW_PRICES), "export_prices": {"all": 0.05}}


def _artifacts(config):
    app = App(config)
    return app, run_app_simulation(app._resolved, app._runtime_dependencies())


def _season_prices(local, table):
    quarter = local.quarter.to_numpy()
    hour = local.hour.to_numpy()
    windowed = np.isin(quarter, [1, 4])
    period = np.where(windowed & (hour < 6), "low", np.where(windowed & (hour >= 17) & (hour < 21), "high", "standard"))
    return np.array(
        [table[f"q{q}"].get(p, table[f"q{q}"].get("all")) for q, p in zip(quarter, period, strict=True)], dtype=float
    )


@pytest.mark.usefixtures("_patch_weather")
@pytest.mark.parametrize(("resolution", "hours"), [("h", 1.0), ("15min", 0.25)])
def test_app_money_reconciles_with_the_step_ledger_by_season(resolution, hours):
    app, artifacts = _artifacts({**BASE, "resolution": resolution, "tariff": SEASONAL_TARIFF})
    frame = artifacts.first_year_results_df
    local = pd.DatetimeIndex(frame["Datetime"]).tz_convert("Europe/Lisbon")
    import_price = _season_prices(local, SEASON_IMPORT)
    export_price = _season_prices(local, SEASONAL_TARIFF["export_prices"])
    year1 = artifacts.yearly_df.iloc[0]

    def priced(column, price):
        return float((frame[column] * price).sum() * hours / 1000)

    assert year1["Import_Cost"] == pytest.approx(priced("Import_From_Grid", import_price), rel=1e-12)
    assert year1["Baseline_Import_Cost"] == pytest.approx(priced("Houseload", import_price), rel=1e-12)
    assert year1["Export_Revenue"] == pytest.approx(priced("PV_AC_Export", export_price), rel=1e-12)
    # The year boundary and both clock changes are in the replayed calendar.
    assert len(frame) == 8760 / hours
    assert pd.Series(local.date).value_counts()[pd.Timestamp("2023-03-26").date()] == 23 / hours
    # Seasons price energy; they do not change it.
    flat = _artifacts({**BASE, "resolution": resolution})[1]
    pd.testing.assert_series_equal(artifacts.yearly_df["Import_kWh"], flat.yearly_df["Import_kWh"])


@pytest.mark.usefixtures("_patch_weather")
def test_app_records_the_seasons_and_replays_them_from_its_resolved_config():
    app = App({**BASE, "projection_years": 1, "tariff": SEASONAL_TARIFF})
    app.simulate()
    result = app.result()

    assert result["result_schema_version"] == "2.6"
    tariff = result["provenance"]["tariff"]
    assert tariff["seasons"] == QUARTERS
    assert tariff["import_prices"] == SEASON_IMPORT
    recorded = result["provenance"]["resolved_config"]["tariff"]
    assert recorded["custom_schedule"]["seasons"] == QUARTERS
    assert recorded["import_prices"] == SEASON_IMPORT

    replay = App(json.loads(json.dumps(result["provenance"]["resolved_config"])))
    replay.simulate()
    again = replay.result()
    assert again["provenance"]["tariff"]["schedule_hash"] == tariff["schedule_hash"]
    assert again["provenance"]["tariff"]["price_hash"] == tariff["price_hash"]
    assert again["npv_savings"] == result["npv_savings"]


def _with_tariff(**changes):
    tariff = deepcopy(SEASONAL_TARIFF)
    for key, value in changes.items():
        if callable(value):
            value(tariff)
        else:
            tariff[key] = value
    return {**BASE, "tariff": tariff}


@pytest.mark.parametrize(
    ("config", "error", "message"),
    [
        (
            _with_tariff(import_prices={**SEASON_IMPORT, "low": 0.2}),
            ValueError,
            r"'tariff\.import_prices' mixes prices and season tables",
        ),
        (
            _with_tariff(import_prices={key: SEASON_IMPORT[key] for key in ("q1", "q2", "q3")}),
            ValueError,
            r"'tariff\.import_prices' has no prices for season\(s\) q4",
        ),
        (
            _with_tariff(import_prices={**SEASON_IMPORT, "winter": {"all": 0.3}}),
            ValueError,
            r"'tariff\.import_prices' has season\(s\) winter that schedule",
        ),
        (
            _with_tariff(import_prices={**SEASON_IMPORT, "q2": {"standard": 0.3, "high": 0.4}}),
            ValueError,
            r"'tariff\.import_prices\.q2' prices high, which season 'q2' never uses",
        ),
        (
            _with_tariff(import_prices={**SEASON_IMPORT, "q1": {"low": 0.2, "standard": 0.3}}),
            ValueError,
            r"'tariff\.import_prices\.q1' has no price for high",
        ),
        (
            _with_tariff(import_prices={**SEASON_IMPORT, "q1": {"all": 0.3, "peak": 0.5}}),
            ValueError,
            r"'tariff\.import_prices\.q1' has period\(s\) peak that schedule",
        ),
        (
            _with_tariff(import_prices={**SEASON_IMPORT, "q1": {"all": "cheap"}}),
            TypeError,
            r"'tariff\.import_prices\.q1\.all' must be a finite number",
        ),
        (
            {
                **BASE,
                "tariff": {
                    "schedule": "pt_mainland_2026_daily_bi",
                    "currency": "EUR",
                    "import_prices": {"q1": {"all": 0.2}},
                    "export_prices": {"all": 0.0},
                },
            },
            ValueError,
            r"gives prices by season, but schedule 'pt_mainland_2026_daily_bi' has no month seasons",
        ),
        (
            _with_tariff(custom_schedule=lambda t: t["custom_schedule"]["rules"][0].update(season="dst")),
            ValueError,
            r"rules\[0\]\.season' must be one of: all, q1, q2, q3, q4; got 'dst'",
        ),
        (
            _with_tariff(custom_schedule=lambda t: t["custom_schedule"].pop("seasons")),
            ValueError,
            r"rules\[0\]\.season' must be one of: all, dst, standard; got 'q1'",
        ),
        (
            _with_tariff(custom_schedule=lambda t: t["custom_schedule"]["seasons"].update(q4=[10, 11, 12, 13])),
            ValueError,
            r"'tariff\.custom_schedule\.seasons\.q4\[3\]' must be a month from 1 through 12",
        ),
        (
            _with_tariff(custom_schedule=lambda t: t["custom_schedule"]["seasons"].update(q4=["10", 11, 12])),
            TypeError,
            r"'tariff\.custom_schedule\.seasons\.q4\[0\]' must be an integer",
        ),
        (
            _with_tariff(custom_schedule=lambda t: t["custom_schedule"]["seasons"].update(q4=[11, 12])),
            ValueError,
            r"Month\(s\) 10 are in no season",
        ),
        (
            _with_tariff(custom_schedule=lambda t: t["custom_schedule"].update(seasons=[[1, 2, 3]])),
            TypeError,
            r"'tariff\.custom_schedule\.seasons' must be a table",
        ),
        (
            _with_tariff(custom_schedule=lambda t: t["custom_schedule"]["seasons"].update(q4=[])),
            ValueError,
            r"'tariff\.custom_schedule\.seasons\.q4' needs at least 1 entry",
        ),
    ],
)
def test_app_rejects_uncovered_or_ambiguous_seasons_and_prices(config, error, message):
    with pytest.raises(error, match=message):
        App(config)


@pytest.mark.usefixtures("_patch_weather")
def test_smart_charging_periods_follow_the_seasons():
    smart = {
        "mode": "fixed_target",
        "target_usable_fraction": 0.8,
        "charge_periods": ["low"],
        "discharge_periods": ["high", "standard"],
        "grid_charge_efficiency": 0.95,
    }
    _app, artifacts = _artifacts({**BASE, "projection_years": 1, "tariff": SEASONAL_TARIFF, "smart_charging": smart})
    frame = artifacts.first_year_results_df
    local = pd.DatetimeIndex(frame["Datetime"]).tz_convert("Europe/Lisbon")
    grid_charge = frame["Grid_AC_To_Battery"].to_numpy()
    windowed_night = np.isin(local.quarter, [1, 4]) & (local.hour < 6)
    # Only Q1 and Q4 nights have a low period to charge in.
    assert grid_charge[windowed_night].sum() > 0
    assert grid_charge[~windowed_night].sum() == 0
    assert artifacts.yearly_df.iloc[0]["Grid_Charge_Cost"] == pytest.approx(
        float((grid_charge * _season_prices(local, SEASON_IMPORT)).sum() / 1000), rel=1e-12
    )


@pytest.mark.usefixtures("_patch_weather")
def test_daily_persistence_plans_on_the_seasonal_prices(monkeypatch):
    planned = []
    solve = _daily_persistence.solve_daily_targets

    def spy(problem, **kwargs):
        planned.append(
            (
                sorted(set(problem.import_price_per_kwh.tolist())),
                sorted(set(problem.export_price_per_kwh.tolist())),
            )
        )
        return solve(problem, **kwargs)

    monkeypatch.setattr(_daily_persistence, "solve_daily_targets", spy)
    smart = {
        "mode": "daily_persistence",
        "charge_periods": ["low"],
        "discharge_periods": ["high", "standard"],
        "grid_charge_efficiency": 0.95,
        "forecast_horizon_days": 1,
        "target_levels": 3,
        "soc_states": 5,
    }
    # A window across the Q1/Q2 boundary keeps the planner run short.
    window = {"start": "2023-03-27", "end": "2023-04-04"}
    config = {key: value for key, value in BASE.items() if key != "projection_years"}
    app = App({**config, "tariff": SEASONAL_TARIFF, "smart_charging": smart, "period": window})
    app.simulate()
    result = app.result()
    assert result["provenance"]["smart_charging"]["mode"] == "daily_persistence"
    assert result["provenance"]["tariff"]["seasons"] == QUARTERS
    # Each day is planned on its own season's prices: Q1's windows through
    # 31 March, Q2's single price from 1 April. The first day has no
    # observation to forecast from.
    q1 = (sorted(WINDOW_PRICES.values()), [0.06])
    q2 = ([0.31], [0.04])
    assert planned == [q1] * 4 + [q2] * 3

    # The planner reads the prices, so a seasonal price change re-simulates.
    q2_price = {**SEASON_IMPORT, "q2": {"standard": 0.29}}
    revalued = app.revalue({"tariff": {"import_prices": q2_price, "export_prices": SEASONAL_TARIFF["export_prices"]}})
    assert revalued["provenance"]["revaluation"]["method"] == "resimulated"
    assert planned[7:] == [q1] * 4 + [([0.29], [0.04])] * 3


def _revalue(app, changes):
    return app.revalue(changes)


@pytest.mark.usefixtures("_patch_weather")
def test_revalue_reprices_seasonal_prices_without_simulating_again():
    app = App({**BASE, "tariff": PERIOD_TARIFF})
    app.simulate()
    changes = {"tariff": {"import_prices": SEASON_IMPORT, "export_prices": SEASONAL_TARIFF["export_prices"]}}
    revalued = _revalue(app, changes)
    fresh_app = App({**BASE, "tariff": SEASONAL_TARIFF})
    fresh_app.simulate()
    fresh = fresh_app.result()

    assert revalued["provenance"]["revaluation"]["method"] == "repriced"
    assert revalued["provenance"]["tariff"]["schedule_hash"] == fresh["provenance"]["tariff"]["schedule_hash"]
    assert revalued["provenance"]["tariff"]["price_hash"] == fresh["provenance"]["tariff"]["price_hash"]
    # Energy by season and period against energy times price per step.
    for key in ("npv_savings", "grid_import_cost_year1_prices", "grid_export_revenue_year1_prices"):
        assert revalued[key] == pytest.approx(fresh[key], abs=0.011), key
    assert revalued["grid_import_cost_year1_prices"] != app.result()["grid_import_cost_year1_prices"]

    # A different partition is a different schedule: the run is simulated again.
    halves = deepcopy(SEASONAL_TARIFF["custom_schedule"])
    halves["seasons"] = {"q1": [1, 2, 3], "q2": [4, 5, 6, 7], "q3": [8, 9], "q4": [10, 11, 12]}
    moved = _revalue(app, {"tariff": {**changes["tariff"], "custom_schedule": halves}})
    assert moved["provenance"]["revaluation"]["method"] == "resimulated"


# --- Monte Carlo and projected optimization --------------------------------


def test_montecarlo_prices_every_trajectory_by_season(tmp_path, write_multiyear_weather):
    weather = write_multiyear_weather(tmp_path / "multi.csv")
    settings = MonteCarloSettings(
        weather_file=str(weather), n_runs=2, years_per_run=2, target_year=2026, seed=5, collect_yearly=True
    )
    # Every season priced like the period tariff gives the same per-step prices.
    same_as_periods = {
        **SEASONAL_TARIFF,
        "import_prices": {
            "q1": dict(WINDOW_PRICES),
            "q2": {"standard": 0.33},
            "q3": {"standard": 0.33},
            "q4": dict(WINDOW_PRICES),
        },
        "export_prices": {season: {"all": 0.05} for season in QUARTERS},
    }
    periods = run_montecarlo({**BASE, "tariff": PERIOD_TARIFF}, settings)
    seasonal_equal = run_montecarlo({**BASE, "tariff": same_as_periods}, settings)
    seasonal = run_montecarlo({**BASE, "tariff": SEASONAL_TARIFF}, settings)

    assert seasonal.provenance["tariff"]["seasons"] == QUARTERS
    assert seasonal.provenance["resolved_config"]["tariff"]["custom_schedule"]["seasons"] == QUARTERS
    assert seasonal.provenance["result_schema_version"] == "2.6"
    money = ["Import_Cost", "Export_Revenue", "Baseline_Import_Cost", "Fixed_Charge"]
    pd.testing.assert_frame_equal(seasonal_equal.yearly[money], periods.yearly[money], check_exact=True)
    pd.testing.assert_series_equal(seasonal.yearly["Import_kWh"], periods.yearly["Import_kWh"])
    assert not np.allclose(seasonal.yearly["Import_Cost"], periods.yearly["Import_Cost"])


@pytest.fixture
def quarter_boundary_case(monkeypatch):
    # 48 hours across the Q1/Q2 boundary, in Lisbon.
    index = pd.date_range("2026-03-31", periods=48, freq="h", tz="Europe/Lisbon")
    weather = pd.DataFrame({"temp_air": 20.0}, index=index)
    load = pd.DataFrame({"Load": 1000.0}, index=index)
    pv = pd.Series(np.where((index.hour >= 10) & (index.hour < 16), 1800.0, 0.0), index=index)
    monkeypatch.setattr(optimization, "calculate_pv_production_dc", lambda **kwargs: pv.copy())
    config = {
        "location": {"latitude": 41.15, "longitude": -8.61, "timezone": "Europe/Lisbon", "altitude": 100},
        "simulation": {"resolution": "h", "years_projection": 2},
        "battery": {"temperature": 20.0},
        "mode": {"fixed_azimuth": 180.0},
        "constraints": {"budget": 100000, "max_area_m2": 100, "max_tilt_deg": 60},
        "tariff": deepcopy(SEASONAL_TARIFF),
    }
    return weather, load, config, index


def test_optimizer_prices_a_design_by_season_across_a_quarter_boundary(quarter_boundary_case):
    weather, load, config, _index = quarter_boundary_case
    design = optimization.evaluate_projected_design(
        weather, load, config, n_modules=4, battery_kwh=5.0, tilt=30.0, azimuth=180.0
    )
    assert design.provenance["tariff"]["seasons"] == QUARTERS
    assert design.provenance["result_schema_version"] == "2.6"
    # 1 kW of load each hour: 24 Q1 hours (windowed) and 24 Q2 hours (standard all day).
    expected = 6 * 0.25 + 4 * 0.42 + 14 * 0.33 + 24 * 0.31
    assert design.yearly["Baseline_Import_Cost"].iloc[0] == pytest.approx(expected, rel=1e-12)

    flat = deepcopy(config)
    flat["tariff"] = {**PERIOD_TARIFF, "custom_schedule": deepcopy(LISBON_QUARTERLY)}
    flat_design = optimization.evaluate_projected_design(
        weather, load, flat, n_modules=4, battery_kwh=5.0, tilt=30.0, azimuth=180.0
    )
    pd.testing.assert_series_equal(design.yearly["Import_kWh"], flat_design.yearly["Import_kWh"])
    assert design.metrics["Projected_NPV"] != flat_design.metrics["Projected_NPV"]


def test_optimizer_search_pickles_the_seasonal_tariff(quarter_boundary_case):
    pytest.importorskip("pymoo")
    weather, load, config, _index = quarter_boundary_case
    problem = optimization.SolarDesignProblem(weather, load, config)
    assert problem.tariff.season_labels is not None
    restored = pickle.loads(pickle.dumps(problem.tariff))
    assert restored.import_price_per_kwh == problem.tariff.import_price_per_kwh
    assert restored.prices == problem.tariff.prices
