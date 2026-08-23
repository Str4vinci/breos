"""Tariff schedule and price values.

The tariff domain is intentionally independent from battery dispatch. Schedule
classifiers prepare period labels; this module validates and resolves those
labels to immutable, index-aligned prices with reproducibility metadata.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from numbers import Real
from types import MappingProxyType
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

from breos.resources import load_config_json

SUPPORTED_CURRENCIES = frozenset({"EUR"})
BOUNDARY_POLICIES = frozenset({"strict"})
SCHEDULE_CYCLES = frozenset({"flat", "daily", "weekly", "custom"})

_PERIOD_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_CLOCK_PATTERN = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$|^24:00$")
_DAY_TYPES = ("weekday", "saturday", "sunday")
_SEASONS = ("standard", "dst")


def _nonempty_text(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"'{where}' must be a non-empty string")
    return value.strip()


def _period_name(value: object, where: str) -> str:
    result = _nonempty_text(value, where)
    if not _PERIOD_PATTERN.fullmatch(result):
        raise ValueError(f"'{where}' must use lowercase letters, digits, and underscores")
    return result


def _nonnegative_price(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"'{where}' must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"'{where}' must be a finite number")
    if result < 0:
        raise ValueError(f"'{where}' must be >= 0")
    return result


def _date_range(effective_from: date | None, effective_to: date | None, where: str) -> None:
    for name, value in (("effective_from", effective_from), ("effective_to", effective_to)):
        if value is not None and not isinstance(value, date):
            raise TypeError(f"'{where}.{name}' must be a date when configured")
    if effective_from is not None and effective_to is not None and effective_from > effective_to:
        raise ValueError(f"'{where}.effective_from' must be on or before '{where}.effective_to'")


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TariffSchedule:
    """Metadata for one versioned tariff schedule.

    Period-classification rules live in schedule resolvers and packaged data;
    this value records the stable identity and provenance used by a resolved
    tariff.
    """

    identifier: str
    version: str
    timezone: str
    cycle: str
    periods: tuple[str, ...]
    source_url: str | None = None
    effective_from: date | None = None
    effective_to: date | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "identifier", _nonempty_text(self.identifier, "schedule.identifier"))
        object.__setattr__(self, "version", _nonempty_text(self.version, "schedule.version"))

        timezone = _nonempty_text(self.timezone, "schedule.timezone")
        try:
            ZoneInfo(timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Unknown IANA tariff timezone: {timezone!r}") from exc
        object.__setattr__(self, "timezone", timezone)

        cycle = _nonempty_text(self.cycle, "schedule.cycle")
        if cycle not in SCHEDULE_CYCLES:
            allowed = ", ".join(sorted(SCHEDULE_CYCLES))
            raise ValueError(f"'schedule.cycle' must be one of: {allowed}")
        object.__setattr__(self, "cycle", cycle)

        periods = tuple(_period_name(period, f"schedule.periods[{index}]") for index, period in enumerate(self.periods))
        if not periods:
            raise ValueError("'schedule.periods' must define at least one period")
        if len(periods) != len(set(periods)):
            raise ValueError("'schedule.periods' must not contain duplicates")
        object.__setattr__(self, "periods", periods)

        if self.source_url is not None:
            object.__setattr__(self, "source_url", _nonempty_text(self.source_url, "schedule.source_url"))
        _date_range(self.effective_from, self.effective_to, "schedule")


@dataclass(frozen=True)
class TariffPrices:
    """Currency-qualified import/export prices and their provenance."""

    currency: str
    import_prices: Mapping[str, float]
    export_prices: Mapping[str, float]
    fixed_charge_per_day: float = 0.0
    identifier: str = "user"
    version: str = "1"
    source_url: str | None = None
    effective_from: date | None = None
    effective_to: date | None = None

    def __post_init__(self) -> None:
        currency = _nonempty_text(self.currency, "prices.currency").upper()
        if currency not in SUPPORTED_CURRENCIES:
            allowed = ", ".join(sorted(SUPPORTED_CURRENCIES))
            raise ValueError(f"Unsupported tariff currency {currency!r}. Available: {allowed}")
        object.__setattr__(self, "currency", currency)

        object.__setattr__(self, "import_prices", self._freeze_prices(self.import_prices, "import_prices"))
        object.__setattr__(self, "export_prices", self._freeze_prices(self.export_prices, "export_prices"))
        object.__setattr__(
            self,
            "fixed_charge_per_day",
            _nonnegative_price(self.fixed_charge_per_day, "prices.fixed_charge_per_day"),
        )
        object.__setattr__(self, "identifier", _nonempty_text(self.identifier, "prices.identifier"))
        object.__setattr__(self, "version", _nonempty_text(self.version, "prices.version"))
        if self.source_url is not None:
            object.__setattr__(self, "source_url", _nonempty_text(self.source_url, "prices.source_url"))
        _date_range(self.effective_from, self.effective_to, "prices")

    @staticmethod
    def _freeze_prices(values: Mapping[str, float], name: str) -> Mapping[str, float]:
        if not isinstance(values, Mapping):
            raise TypeError(f"'prices.{name}' must be a mapping of period names to prices")
        if not values:
            raise ValueError(f"'prices.{name}' must define at least one price")
        resolved: dict[str, float] = {}
        for raw_period, raw_price in values.items():
            period = _period_name(raw_period, f"prices.{name} period")
            resolved[period] = _nonnegative_price(raw_price, f"prices.{name}.{period}")
        return MappingProxyType(resolved)


@dataclass(frozen=True)
class ResolvedTariff:
    """Immutable tariff values aligned to a timezone-aware simulation index."""

    index: pd.DatetimeIndex
    schedule: TariffSchedule
    prices: TariffPrices
    period_labels: tuple[str, ...]
    period_codes: tuple[int, ...]
    import_price_per_kwh: tuple[float, ...]
    export_price_per_kwh: tuple[float, ...]
    boundary_policy: str
    schedule_hash: str
    price_hash: str


FLAT_SCHEDULE = TariffSchedule(
    identifier="flat",
    version="1",
    timezone="UTC",
    cycle="flat",
    periods=("all",),
)


def _parse_date(value: object, where: str) -> date | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"'{where}' must be an ISO date string")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"'{where}' must be an ISO date string") from exc


def _clock_minutes(value: object, where: str) -> int:
    if not isinstance(value, str) or not _CLOCK_PATTERN.fullmatch(value):
        raise ValueError(f"'{where}' must be a clock value from 00:00 through 24:00")
    if value == "24:00":
        return 24 * 60
    hour, minute = value.split(":")
    return int(hour) * 60 + int(minute)


def _validate_rule_intervals(
    schedule_id: str,
    periods: tuple[str, ...],
    rule_index: int,
    intervals: object,
) -> tuple[tuple[int, int, str], ...]:
    where = f"tariffs.json schedules.{schedule_id}.rules[{rule_index}].intervals"
    if not isinstance(intervals, Mapping) or not intervals:
        raise TypeError(f"'{where}' must be a non-empty mapping")

    unknown = set(intervals) - set(periods)
    if unknown:
        raise ValueError(f"'{where}' contains unknown period(s): {', '.join(sorted(unknown))}")

    flattened: list[tuple[int, int, str]] = []
    for raw_period, raw_ranges in intervals.items():
        period = str(raw_period)
        if not isinstance(raw_ranges, list) or not raw_ranges:
            raise TypeError(f"'{where}.{period}' must be a non-empty list of [start, end] ranges")
        for range_index, raw_range in enumerate(raw_ranges):
            range_where = f"{where}.{period}[{range_index}]"
            if not isinstance(raw_range, list) or len(raw_range) != 2:
                raise TypeError(f"'{range_where}' must contain exactly [start, end]")
            start = _clock_minutes(raw_range[0], f"{range_where}[0]")
            end = _clock_minutes(raw_range[1], f"{range_where}[1]")
            if start >= end:
                raise ValueError(f"'{range_where}' must have start before end")
            flattened.append((start, end, period))

    flattened.sort()
    cursor = 0
    for start, end, _period in flattened:
        if start != cursor:
            problem = "overlap" if start < cursor else "gap"
            raise ValueError(f"'{where}' has a {problem} at minute {min(start, cursor)}")
        cursor = end
    if cursor != 24 * 60:
        raise ValueError(f"'{where}' must cover the complete civil day")
    return tuple(flattened)


@lru_cache(maxsize=1)
def _schedule_catalog() -> Mapping[str, Mapping[str, Any]]:
    payload = load_config_json("tariffs.json")
    raw_schedules = payload.get("schedules")
    if not isinstance(raw_schedules, dict) or not raw_schedules:
        raise ValueError("'tariffs.json schedules' must be a non-empty mapping")

    catalog: dict[str, Mapping[str, Any]] = {}
    for raw_identifier, raw_definition in raw_schedules.items():
        identifier = _nonempty_text(raw_identifier, "tariffs.json schedule identifier")
        if not isinstance(raw_definition, dict):
            raise TypeError(f"'tariffs.json schedules.{identifier}' must be a mapping")

        periods = tuple(
            _period_name(period, f"tariffs.json schedules.{identifier}.periods[{position}]")
            for position, period in enumerate(raw_definition.get("periods", ()))
        )
        schedule = TariffSchedule(
            identifier=identifier,
            version=raw_definition.get("version"),
            timezone=raw_definition.get("timezone"),
            cycle=raw_definition.get("cycle"),
            periods=periods,
            source_url=raw_definition.get("source_url"),
            effective_from=_parse_date(
                raw_definition.get("effective_from"), f"tariffs.json schedules.{identifier}.effective_from"
            ),
            effective_to=_parse_date(
                raw_definition.get("effective_to"), f"tariffs.json schedules.{identifier}.effective_to"
            ),
        )

        resolution = raw_definition.get("required_resolution_minutes")
        if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution <= 0:
            raise ValueError(
                f"'tariffs.json schedules.{identifier}.required_resolution_minutes' must be a positive integer"
            )

        raw_rules = raw_definition.get("rules")
        if not isinstance(raw_rules, list) or not raw_rules:
            raise TypeError(f"'tariffs.json schedules.{identifier}.rules' must be a non-empty list")
        rules: list[Mapping[str, Any]] = []
        for rule_index, raw_rule in enumerate(raw_rules):
            if not isinstance(raw_rule, dict):
                raise TypeError(f"'tariffs.json schedules.{identifier}.rules[{rule_index}]' must be a mapping")
            days = raw_rule.get("days")
            season = raw_rule.get("season")
            if days not in {*_DAY_TYPES, "all"}:
                raise ValueError(f"Unknown day selector in schedule {identifier!r}: {days!r}")
            if season not in {*_SEASONS, "all"}:
                raise ValueError(f"Unknown season selector in schedule {identifier!r}: {season!r}")
            rules.append(
                MappingProxyType(
                    {
                        "days": days,
                        "season": season,
                        "intervals": _validate_rule_intervals(
                            identifier, schedule.periods, rule_index, raw_rule.get("intervals")
                        ),
                    }
                )
            )

        for day_type in _DAY_TYPES:
            for season in _SEASONS:
                matches = [
                    rule for rule in rules if rule["days"] in {day_type, "all"} and rule["season"] in {season, "all"}
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"Schedule {identifier!r} must define exactly one rule for {day_type}/{season}; "
                        f"found {len(matches)}"
                    )

        catalog[identifier] = MappingProxyType(
            {
                "schedule": schedule,
                "required_resolution_minutes": resolution,
                "rules": tuple(rules),
            }
        )
    return MappingProxyType(catalog)


def available_tariff_schedules() -> tuple[str, ...]:
    """Return the identifiers of bundled, price-independent schedules."""
    return tuple(sorted(_schedule_catalog()))


def get_tariff_schedule(identifier: str) -> TariffSchedule:
    """Return immutable metadata for one bundled schedule identifier."""
    name = _nonempty_text(identifier, "schedule identifier")
    try:
        return _schedule_catalog()[name]["schedule"]
    except KeyError as exc:
        available = ", ".join(available_tariff_schedules())
        raise ValueError(f"Unknown tariff schedule {name!r}. Available: {available}") from exc


def _validate_schedule_resolution(index: pd.DatetimeIndex, identifier: str, required_minutes: int) -> None:
    if len(index) < 2:
        return
    utc_nanoseconds = index.tz_convert("UTC").as_unit("ns").asi8
    deltas = utc_nanoseconds[1:] - utc_nanoseconds[:-1]
    unique_deltas = set(int(delta) for delta in deltas)
    if len(unique_deltas) != 1:
        raise ValueError("Tariff classification requires a regular simulation index")
    cadence_ns = unique_deltas.pop()
    minute_ns = 60 * 1_000_000_000
    if cadence_ns % minute_ns:
        raise ValueError("Tariff classification requires a whole-minute simulation resolution")
    cadence_minutes = cadence_ns // minute_ns
    if required_minutes % cadence_minutes:
        raise ValueError(
            f"Schedule {identifier!r} requires {required_minutes}-minute resolution or finer; "
            f"the index uses {cadence_minutes}-minute steps"
        )
    first_local = index[0].tz_convert(get_tariff_schedule(identifier).timezone)
    first_day_offset_ns = (
        (first_local.hour * 60 + first_local.minute) * minute_ns
        + first_local.second * 1_000_000_000
        + first_local.microsecond * 1_000
        + first_local.nanosecond
    )
    if first_day_offset_ns % cadence_ns:
        raise ValueError(
            f"Schedule {identifier!r} boundaries do not align with an index starting at "
            f"{first_local.strftime('%H:%M:%S')}"
        )


def _day_type(timestamp: pd.Timestamp) -> str:
    if timestamp.weekday() < 5:
        return "weekday"
    return "saturday" if timestamp.weekday() == 5 else "sunday"


def _season(timestamp: pd.Timestamp) -> str:
    offset = timestamp.dst()
    return "dst" if offset is not None and offset.total_seconds() > 0 else "standard"


def _validate_study_date(schedule: TariffSchedule, index: pd.DatetimeIndex, study_date: date | None) -> None:
    local_dates = index.tz_convert(schedule.timezone).date
    if study_date is not None and not isinstance(study_date, date):
        raise TypeError("'study_date' must be a date when configured")
    references = (study_date,) if study_date is not None else tuple(local_dates)
    if not references:
        return

    outside_window = any(
        (schedule.effective_from is not None and reference < schedule.effective_from)
        or (schedule.effective_to is not None and reference > schedule.effective_to)
        for reference in references
    )
    if outside_window and study_date is None:
        first_date = min(references)
        last_date = max(references)
        index_range = (
            first_date.isoformat() if first_date == last_date else f"{first_date.isoformat()}..{last_date.isoformat()}"
        )
        raise ValueError(
            f"Schedule {schedule.identifier!r} is not effective across the index date range {index_range}; "
            "provide an explicit in-range 'study_date' when the index is a weather reference year"
        )
    if outside_window:
        raise ValueError(f"Schedule {schedule.identifier!r} is not effective on study date {study_date.isoformat()}")


def classify_tariff_periods(
    index: pd.DatetimeIndex,
    identifier: str,
    *,
    study_date: date | None = None,
    boundary_policy: str = "strict",
) -> tuple[str, ...]:
    """Classify instants with a bundled schedule in its local civil time."""
    resolved_index = _validate_index(index)
    if boundary_policy not in BOUNDARY_POLICIES:
        allowed = ", ".join(sorted(BOUNDARY_POLICIES))
        raise ValueError(f"'boundary_policy' must be one of: {allowed}")

    schedule = get_tariff_schedule(identifier)
    definition = _schedule_catalog()[schedule.identifier]
    _validate_schedule_resolution(resolved_index, schedule.identifier, definition["required_resolution_minutes"])
    _validate_study_date(schedule, resolved_index, study_date)

    labels: list[str] = []
    for timestamp in resolved_index.tz_convert(schedule.timezone):
        day_type = _day_type(timestamp)
        season = _season(timestamp)
        rule = next(
            rule
            for rule in definition["rules"]
            if rule["days"] in {day_type, "all"} and rule["season"] in {season, "all"}
        )
        minute = timestamp.hour * 60 + timestamp.minute
        label = next(period for start, end, period in rule["intervals"] if start <= minute < end)
        labels.append(label)
    return tuple(labels)


def resolve_named_tariff(
    index: pd.DatetimeIndex,
    schedule_identifier: str,
    prices: TariffPrices,
    *,
    study_date: date | None = None,
    boundary_policy: str = "strict",
) -> ResolvedTariff:
    """Classify and price one bundled tariff schedule."""
    schedule = get_tariff_schedule(schedule_identifier)
    labels = classify_tariff_periods(
        index,
        schedule_identifier,
        study_date=study_date,
        boundary_policy=boundary_policy,
    )
    return resolve_tariff(index, labels, schedule, prices, boundary_policy=boundary_policy)


def _validate_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("'index' must be a pandas DatetimeIndex")
    if index.tz is None:
        raise ValueError("'index' must be timezone-aware before tariff resolution")
    if not index.is_unique:
        raise ValueError("'index' must contain unique instants")
    if not index.is_monotonic_increasing:
        raise ValueError("'index' must be monotonic increasing")
    return index.copy()


def _validate_price_periods(schedule: TariffSchedule, prices: TariffPrices) -> None:
    allowed = {*schedule.periods, "all"}
    for name, values in (("import_prices", prices.import_prices), ("export_prices", prices.export_prices)):
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(
                f"Unknown {name} period(s) for schedule {schedule.identifier!r}: {', '.join(sorted(unknown))}"
            )


def validate_tariff_prices(schedule: TariffSchedule, prices: TariffPrices) -> None:
    """Validate that price keys cover every period in a schedule."""
    _validate_price_periods(schedule, prices)
    _prices_for_labels(schedule.periods, prices.import_prices, "import price")
    _prices_for_labels(schedule.periods, prices.export_prices, "export price")


def _prices_for_labels(labels: tuple[str, ...], values: Mapping[str, float], name: str) -> tuple[float, ...]:
    fallback = values.get("all")
    missing = sorted({label for label in labels if label not in values and fallback is None})
    if missing:
        raise ValueError(f"Missing {name} for used tariff period(s): {', '.join(missing)}")
    return tuple(values.get(label, fallback) for label in labels)  # type: ignore[arg-type]


def resolve_tariff(
    index: pd.DatetimeIndex,
    period_labels: Sequence[str],
    schedule: TariffSchedule,
    prices: TariffPrices,
    *,
    boundary_policy: str = "strict",
) -> ResolvedTariff:
    """Resolve classified schedule periods to aligned import/export prices."""
    resolved_index = _validate_index(index)
    labels = tuple(_period_name(label, f"period_labels[{position}]") for position, label in enumerate(period_labels))
    if len(labels) != len(resolved_index):
        raise ValueError("'period_labels' must have the same length as 'index'")

    unknown_labels = set(labels) - set(schedule.periods)
    if unknown_labels:
        raise ValueError(
            f"Unknown resolved period(s) for schedule {schedule.identifier!r}: {', '.join(sorted(unknown_labels))}"
        )
    if boundary_policy not in BOUNDARY_POLICIES:
        allowed = ", ".join(sorted(BOUNDARY_POLICIES))
        raise ValueError(f"'boundary_policy' must be one of: {allowed}")

    _validate_price_periods(schedule, prices)
    import_values = _prices_for_labels(labels, prices.import_prices, "import price")
    export_values = _prices_for_labels(labels, prices.export_prices, "export price")
    code_by_period = {period: code for code, period in enumerate(schedule.periods)}
    codes = tuple(code_by_period[label] for label in labels)

    utc_nanoseconds = tuple(int(value) for value in resolved_index.tz_convert("UTC").as_unit("ns").asi8)
    schedule_hash = _canonical_hash(
        {
            "identifier": schedule.identifier,
            "version": schedule.version,
            "timezone": schedule.timezone,
            "instants_utc_ns": utc_nanoseconds,
            "period_labels": labels,
        }
    )
    price_hash = _canonical_hash(
        {
            "currency": prices.currency,
            "identifier": prices.identifier,
            "version": prices.version,
            "source_url": prices.source_url,
            "import_prices": dict(prices.import_prices),
            "export_prices": dict(prices.export_prices),
            "fixed_charge_per_day": prices.fixed_charge_per_day,
            "effective_from": prices.effective_from.isoformat() if prices.effective_from else None,
            "effective_to": prices.effective_to.isoformat() if prices.effective_to else None,
        }
    )

    return ResolvedTariff(
        index=resolved_index,
        schedule=schedule,
        prices=prices,
        period_labels=labels,
        period_codes=codes,
        import_price_per_kwh=import_values,
        export_price_per_kwh=export_values,
        boundary_policy=boundary_policy,
        schedule_hash=schedule_hash,
        price_hash=price_hash,
    )


def resolve_flat_tariff(
    index: pd.DatetimeIndex,
    *,
    import_price_per_kwh: float,
    export_price_per_kwh: float,
    fixed_charge_per_day: float = 0.0,
    currency: str = "EUR",
) -> ResolvedTariff:
    """Build an auditable flat tariff without changing legacy App behavior."""
    prices = TariffPrices(
        currency=currency,
        import_prices={"all": import_price_per_kwh},
        export_prices={"all": export_price_per_kwh},
        fixed_charge_per_day=fixed_charge_per_day,
        identifier="legacy_flat",
        version="0.5.x",
    )
    return resolve_tariff(index, ("all",) * len(index), FLAT_SCHEDULE, prices)
