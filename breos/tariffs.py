"""Tariff schedules, prices, and their resolution onto a simulation index.

The tariff domain is independent of battery dispatch (ADR 0002). A schedule
maps civil-time instants to named periods and carries regulatory provenance;
prices map periods to per-kWh values in one currency; a resolved tariff is the
pair aligned to one simulation index, with period labels and codes, price
arrays, the civil-day boundaries, and hashes for provenance.

Periods are classified in the configured timezone. The simulation index is
often UTC or a fixed offset, so it is converted explicitly and its own
timezone is never read (ADR 0002 A1).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from functools import lru_cache
from numbers import Real
from types import MappingProxyType
from typing import Any, Mapping, Sequence, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd

from breos.resources import load_config_json

SUPPORTED_CURRENCIES = frozenset({"EUR"})
# The bundled cost catalogue's currency, and the currency of a run without a tariff.
DEFAULT_CURRENCY = "EUR"
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


def _utc_nanoseconds(index: pd.DatetimeIndex) -> Any:
    """The index's instants as UTC nanoseconds since the epoch.

    pandas 3 builds microsecond indexes, so the unit is set explicitly: the
    hashed integers are nanoseconds whatever the storage unit (ADR 0002 A1).
    """
    return cast(Any, index.tz_convert("UTC").as_unit("ns")).asi8


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TariffSchedule:
    """Metadata for one versioned tariff schedule.

    The period rules live in a :class:`ScheduleDefinition`; this value
    records the stable identity and provenance used by a resolved tariff.
    """

    identifier: str
    version: str
    timezone: str
    cycle: str
    periods: tuple[str, ...]
    source_url: str | None = None
    # The regulatory citation behind the periods, and caveats on applying them.
    source: str | None = None
    note: str | None = None
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

        for name in ("source_url", "source", "note"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _nonempty_text(value, f"schedule.{name}"))
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

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Rebuild immutable price maps in optimizer worker processes."""
        return type(self), (
            self.currency,
            dict(self.import_prices),
            dict(self.export_prices),
            self.fixed_charge_per_day,
            self.identifier,
            self.version,
            self.source_url,
            self.effective_from,
            self.effective_to,
        )


@dataclass(frozen=True)
class ResolvedTariff:
    """Immutable tariff values aligned to a timezone-aware simulation index.

    ``timezone`` is the civil time the periods were classified in.
    ``day_starts`` holds the position of the first step of every civil day in
    that zone, followed by ``len(index)``, so day ``i`` is
    ``index[day_starts[i]:day_starts[i + 1]]``. A civil day has 23, 24 or 25
    hours; everything with per-day meaning (the fixed charge, daily charge
    windows) uses these boundaries, never a steps-per-day count.
    """

    index: pd.DatetimeIndex
    timezone: str
    schedule: TariffSchedule
    prices: TariffPrices
    period_labels: tuple[str, ...]
    period_codes: tuple[int, ...]
    import_price_per_kwh: tuple[float, ...]
    export_price_per_kwh: tuple[float, ...]
    day_starts: tuple[int, ...]
    boundary_policy: str
    schedule_hash: str
    price_hash: str

    @property
    def n_days(self) -> int:
        """The number of civil days the index touches."""
        return len(self.day_starts) - 1


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
    # JSON gives ISO strings; TOML and Python give dates.
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        raise TypeError(f"'{where}' must be a date or an ISO date string")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"'{where}' must be a date or an ISO date string") from exc


def _clock_minutes(value: object, where: str) -> int:
    if not isinstance(value, str) or not _CLOCK_PATTERN.fullmatch(value):
        raise ValueError(f"'{where}' must be a clock value from 00:00 through 24:00")
    if value == "24:00":
        return 24 * 60
    hour, minute = value.split(":")
    return int(hour) * 60 + int(minute)


@dataclass(frozen=True)
class ScheduleRule:
    """The periods of one day selector and season, tiling the civil day.

    ``days`` is ``weekday``, ``saturday``, ``sunday`` or ``all``; ``season``
    is ``standard``, ``dst`` or ``all``. Each interval is ``(start, end,
    period)`` in minutes after local midnight, ``end`` exclusive; together
    they cover 0 through 1440 with no gap or overlap.
    """

    days: str
    season: str
    intervals: tuple[tuple[int, int, str], ...]

    def __post_init__(self) -> None:
        if self.days not in {*_DAY_TYPES, "all"}:
            raise ValueError(f"Unknown day selector: {self.days!r}")
        if self.season not in {*_SEASONS, "all"}:
            raise ValueError(f"Unknown season selector: {self.season!r}")
        intervals = []
        for start, end, period in self.intervals:
            for bound in (start, end):
                if isinstance(bound, bool) or not isinstance(bound, int) or not 0 <= bound <= 24 * 60:
                    raise ValueError(f"Interval bounds must be whole minutes from 0 through 1440, not {bound!r}")
            if start >= end:
                raise ValueError(f"Interval {period} [{start}, {end}) must have start before end")
            intervals.append((start, end, _period_name(period, "interval period")))
        intervals.sort()
        cursor = 0
        for start, end, _period in intervals:
            if start != cursor:
                problem = "overlap" if start < cursor else "gap"
                raise ValueError(f"Intervals have a {problem} at minute {min(start, cursor)}")
            cursor = end
        if cursor != 24 * 60:
            raise ValueError("Intervals must cover the complete civil day")
        object.__setattr__(self, "intervals", tuple(intervals))

    def applies_to(self, day_type: str, season: str) -> bool:
        return self.days in {day_type, "all"} and self.season in {season, "all"}


@dataclass(frozen=True)
class HolidayCalendar:
    """Dates a schedule treats as another day type.

    ``years`` are the years ``dates`` is complete for: an index covering a
    day of any other year raises rather than miss a holiday. ``None`` means
    the dates are complete for every year.
    """

    day_type: str
    dates: frozenset[date]
    years: frozenset[int] | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        if self.day_type not in _DAY_TYPES:
            raise ValueError(f"'holidays.day_type' must be one of: {', '.join(_DAY_TYPES)}")
        dates = frozenset(self.dates)
        if any(not isinstance(day, date) or isinstance(day, datetime) for day in dates):
            raise TypeError("'holidays.dates' must hold dates")
        object.__setattr__(self, "dates", dates)
        if self.years is not None:
            years = frozenset(self.years)
            if any(isinstance(year, bool) or not isinstance(year, int) for year in years):
                raise TypeError("'holidays.years' must hold integer years")
            outside = sorted(day for day in dates if day.year not in years)
            if outside:
                raise ValueError(f"'holidays.dates' has {outside[0].isoformat()}, outside the calendar's years")
            object.__setattr__(self, "years", years)
        if self.source is not None:
            object.__setattr__(self, "source", _nonempty_text(self.source, "holidays.source"))


@dataclass(frozen=True)
class ScheduleDefinition:
    """A complete tariff schedule: its metadata, period rules and holidays.

    Every day type and season is matched by exactly one rule. A holiday takes
    the rule of ``holidays.day_type``. The values are immutable and pickle,
    so a definition can travel to optimizer worker processes. The bundled
    catalogue is built with :func:`parse_schedule_definition`.

    A definition is hashable, but its ``hash()`` varies between processes:
    never store it or use it as a persistent key. A resolved tariff's
    ``schedule_hash`` is the stable identity.
    """

    schedule: TariffSchedule
    rules: tuple[ScheduleRule, ...]
    holidays: HolidayCalendar | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.schedule, TariffSchedule):
            raise TypeError("'schedule' must be a TariffSchedule")
        rules = tuple(self.rules)
        if not rules or any(not isinstance(rule, ScheduleRule) for rule in rules):
            raise TypeError("'rules' must be a non-empty sequence of ScheduleRule")
        identifier = self.schedule.identifier
        for position, rule in enumerate(rules):
            unknown = {period for _start, _end, period in rule.intervals} - set(self.schedule.periods)
            if unknown:
                raise ValueError(
                    f"Schedule {identifier!r} rules[{position}] uses unknown period(s): {', '.join(sorted(unknown))}"
                )
        for day_type in _DAY_TYPES:
            for season in _SEASONS:
                matches = sum(rule.applies_to(day_type, season) for rule in rules)
                if matches != 1:
                    raise ValueError(
                        f"Schedule {identifier!r} must define exactly one rule for {day_type}/{season}; found {matches}"
                    )
        object.__setattr__(self, "rules", rules)
        if self.holidays is not None and not isinstance(self.holidays, HolidayCalendar):
            raise TypeError("'holidays' must be a HolidayCalendar when configured")

    @property
    def resolution_minutes(self) -> int:
        """The coarsest step, in minutes, that lands on every boundary: input steps must divide it.

        A regular index moves on the local clock when the zone changes its UTC
        offset, so the changes count as boundaries too: a Lisbon or Madrid
        schedule needs 60 minutes or finer whatever its periods.
        """
        bounds = (bound for rule in self.rules for start, end, _ in rule.intervals for bound in (start, end))
        return math.gcd(24 * 60, _offset_change_minutes(self.schedule.timezone), *bounds)

    def rule_for(self, day_type: str, season: str) -> ScheduleRule:
        return next(rule for rule in self.rules if rule.applies_to(day_type, season))


@lru_cache(maxsize=None)
def _offset_change_minutes(timezone: str) -> int:
    """The greatest common divisor, in minutes, of the changes between the UTC offsets a zone uses.

    The offsets are read at every UTC midnight from 1970 through 2100. A
    change that is not a whole number of minutes counts as one minute; ``0``
    means the zone keeps one offset.
    """
    instants = pd.date_range("1970-01-01", "2101-01-01", freq="D", tz="UTC")
    offsets = (instants.tz_convert(timezone).tz_localize(None) - instants.tz_localize(None)).unique()
    seconds = sorted(int(offset.total_seconds()) for offset in offsets)
    result = 0
    for change in (value - seconds[0] for value in seconds[1:]):
        result = math.gcd(result, change // 60 if change % 60 == 0 else 1)
    return result


def _parse_rule(raw: object, where: str) -> ScheduleRule:
    if not isinstance(raw, Mapping):
        raise TypeError(f"'{where}' must be a mapping")
    intervals = raw.get("intervals")
    if not isinstance(intervals, Mapping) or not intervals:
        raise TypeError(f"'{where}.intervals' must be a non-empty mapping")
    flattened: list[tuple[int, int, str]] = []
    for raw_period, raw_ranges in intervals.items():
        period = str(raw_period)
        if not isinstance(raw_ranges, (list, tuple)) or not raw_ranges:
            raise TypeError(f"'{where}.intervals.{period}' must be a non-empty list of [start, end] ranges")
        for range_index, raw_range in enumerate(raw_ranges):
            range_where = f"{where}.intervals.{period}[{range_index}]"
            if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
                raise TypeError(f"'{range_where}' must contain exactly [start, end]")
            start = _clock_minutes(raw_range[0], f"{range_where}[0]")
            end = _clock_minutes(raw_range[1], f"{range_where}[1]")
            flattened.append((start, end, period))
    try:
        return ScheduleRule(
            days=cast(str, raw.get("days")), season=cast(str, raw.get("season")), intervals=tuple(flattened)
        )
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"'{where}': {exc}") from exc


def _parse_holidays(raw: object, where: str) -> HolidayCalendar | None:
    """Read a holiday calendar: the day type holidays take, and dates by year."""
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise TypeError(f"'{where}' must be a mapping")
    day_type = raw.get("day_type")
    if day_type not in _DAY_TYPES:
        raise ValueError(f"'{where}.day_type' must be one of: {', '.join(_DAY_TYPES)}")
    raw_dates = raw.get("dates")
    if not isinstance(raw_dates, Mapping) or not raw_dates:
        raise TypeError(f"'{where}.dates' must map years to lists of ISO dates")
    years: set[int] = set()
    dates: set[date] = set()
    for raw_year, values in raw_dates.items():
        try:
            year = int(raw_year)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"'{where}.dates' keys must be years, not {raw_year!r}") from exc
        if not isinstance(values, (list, tuple)):
            raise TypeError(f"'{where}.dates.{year}' must be a list of ISO dates")
        parsed = {_parse_date(value, f"{where}.dates.{year}") for value in values}
        if any(day is None or day.year != year for day in parsed):
            raise ValueError(f"'{where}.dates.{year}' must hold dates in {year}")
        years.add(year)
        dates.update(cast(set[date], parsed))
    return HolidayCalendar(day_type=day_type, dates=frozenset(dates), years=frozenset(years), source=raw.get("source"))


def parse_schedule_definition(
    identifier: str, raw: Mapping[str, Any], *, where: str | None = None
) -> ScheduleDefinition:
    """Build a :class:`ScheduleDefinition` from its mapping form, as in the bundled ``tariffs.json``.

    ``raw`` holds ``version``, ``timezone``, ``cycle``, ``periods`` and
    ``rules``, and optionally ``source_url``, ``source``, ``note``,
    ``effective_from``, ``effective_to`` and ``holidays``. Each rule has a
    ``days`` selector, a ``season`` selector and ``intervals`` mapping periods
    to ``["HH:MM", "HH:MM"]`` ranges; ``"24:00"`` closes the day. Holidays
    have a ``day_type`` and ``dates`` mapping each year to its dates. The
    time resolution the schedule needs follows from the boundaries and the
    zone's UTC-offset changes.
    ``where`` names the source in error messages; it defaults to the
    identifier.
    """
    name = _nonempty_text(identifier, "schedule identifier")
    where = where or name
    if not isinstance(raw, Mapping):
        raise TypeError(f"'{where}' must be a mapping")

    raw_periods = raw.get("periods", ())
    if not isinstance(raw_periods, (list, tuple)):
        raise TypeError(f"'{where}.periods' must be a list of period names")
    periods = tuple(_period_name(period, f"{where}.periods[{position}]") for position, period in enumerate(raw_periods))
    schedule = TariffSchedule(
        identifier=name,
        # TariffSchedule validates these; a missing one fails there.
        version=cast(str, raw.get("version")),
        timezone=cast(str, raw.get("timezone")),
        cycle=cast(str, raw.get("cycle")),
        periods=periods,
        source_url=raw.get("source_url"),
        source=raw.get("source"),
        note=raw.get("note"),
        effective_from=_parse_date(raw.get("effective_from"), f"{where}.effective_from"),
        effective_to=_parse_date(raw.get("effective_to"), f"{where}.effective_to"),
    )

    raw_rules = raw.get("rules")
    if not isinstance(raw_rules, (list, tuple)) or not raw_rules:
        raise TypeError(f"'{where}.rules' must be a non-empty list")
    rules = tuple(_parse_rule(raw_rule, f"{where}.rules[{position}]") for position, raw_rule in enumerate(raw_rules))
    return ScheduleDefinition(
        schedule=schedule, rules=rules, holidays=_parse_holidays(raw.get("holidays"), f"{where}.holidays")
    )


@lru_cache(maxsize=1)
def _schedule_catalog() -> Mapping[str, ScheduleDefinition]:
    payload = load_config_json("tariffs.json")
    raw_schedules = payload.get("schedules")
    if not isinstance(raw_schedules, dict) or not raw_schedules:
        raise ValueError("'tariffs.json schedules' must be a non-empty mapping")

    catalog: dict[str, ScheduleDefinition] = {}
    for raw_identifier, raw_definition in raw_schedules.items():
        identifier = _nonempty_text(raw_identifier, "tariffs.json schedule identifier")
        catalog[identifier] = parse_schedule_definition(
            identifier, raw_definition, where=f"tariffs.json schedules.{identifier}"
        )
    return MappingProxyType(catalog)


def available_tariff_schedules() -> tuple[str, ...]:
    """Return the identifiers of bundled, price-independent schedules."""
    return tuple(sorted(_schedule_catalog()))


def get_schedule_definition(identifier: str) -> ScheduleDefinition:
    """Return the full definition of one bundled schedule identifier."""
    name = _nonempty_text(identifier, "schedule identifier")
    try:
        return _schedule_catalog()[name]
    except KeyError as exc:
        available = ", ".join(available_tariff_schedules())
        raise KeyError(f"Unknown tariff schedule {name!r}. Available: {available}") from exc


def get_tariff_schedule(identifier: str) -> TariffSchedule:
    """Return immutable metadata for one bundled schedule identifier."""
    return get_schedule_definition(identifier).schedule


def _as_definition(schedule: str | ScheduleDefinition) -> ScheduleDefinition:
    if isinstance(schedule, ScheduleDefinition):
        return schedule
    if not isinstance(schedule, str):
        raise TypeError("'schedule' must be a bundled schedule identifier or a ScheduleDefinition")
    return get_schedule_definition(schedule)


def _validate_schedule_resolution(
    index: pd.DatetimeIndex, identifier: str, required_minutes: int, timezone: str
) -> None:
    if len(index) < 2:
        return
    utc_nanoseconds = _utc_nanoseconds(index)
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
    # Every step must start on the local step grid, not just the first: a
    # regular index moves on the local clock when the UTC offset changes.
    # The cadence divides a day, so the wall-clock epoch is on the grid.
    local_index = index.tz_convert(timezone)
    wall_ns = cast(Any, local_index.tz_localize(None).as_unit("ns")).asi8
    misaligned = np.flatnonzero(wall_ns % cadence_ns)
    if misaligned.size:
        step = local_index[int(misaligned[0])]
        raise ValueError(
            f"Schedule {identifier!r} boundaries do not align with the index step at "
            f"{step.strftime('%Y-%m-%d %H:%M:%S')} local time"
        )


def _holiday_dates(identifier: str, holidays: HolidayCalendar | None, years: set[int]) -> frozenset[date]:
    if holidays is None:
        return frozenset()
    if holidays.years is not None:
        missing = sorted(years - holidays.years)
        if missing:
            known = ", ".join(str(year) for year in sorted(holidays.years))
            raise ValueError(
                f"Schedule {identifier!r} treats national holidays as {holidays.day_type}s, but BREOS has no "
                f"holiday calendar for {', '.join(map(str, missing))} (it has {known}). The calendar is published "
                "every year; simulate a year it covers."
            )
    return frozenset(day for day in holidays.dates if day.year in years)


def _day_type(
    timestamp: pd.Timestamp, holidays: frozenset[date] = frozenset(), holiday_day_type: str = "sunday"
) -> str:
    if timestamp.date() in holidays:
        return holiday_day_type
    if timestamp.weekday() < 5:
        return "weekday"
    return "saturday" if timestamp.weekday() == 5 else "sunday"


def _season(timestamp: pd.Timestamp) -> str:
    offset = timestamp.dst()
    return "dst" if offset is not None and offset.total_seconds() > 0 else "standard"


def _validate_study_date(
    schedule: TariffSchedule, index: pd.DatetimeIndex, study_date: date | None, timezone: str
) -> None:
    local_dates = index.tz_convert(timezone).date
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
    if outside_window and study_date is not None:
        raise ValueError(f"Schedule {schedule.identifier!r} is not effective on study date {study_date.isoformat()}")


def _check_timezone(schedule: TariffSchedule, timezone: str) -> str:
    name = _nonempty_text(timezone, "timezone")
    try:
        ZoneInfo(name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unknown IANA timezone: {name!r}") from exc
    if name != schedule.timezone:
        raise ValueError(
            f"Schedule {schedule.identifier!r} is defined in {schedule.timezone} civil time, but the "
            f"location's timezone is {name}. BREOS does not move a schedule to another zone."
        )
    return name


def classify_tariff_periods(
    index: pd.DatetimeIndex,
    schedule: str | ScheduleDefinition,
    *,
    timezone: str,
    study_date: date | None = None,
    boundary_policy: str = "strict",
) -> tuple[str, ...]:
    """Classify instants with a schedule in the configured civil time.

    ``schedule`` is a bundled schedule identifier or a
    :class:`ScheduleDefinition`. ``timezone`` is the location's IANA zone
    (the App's ``ResolvedAppConfig.timezone``). The index is converted to it
    explicitly; its own timezone only has to identify the instants. The zone
    must be the one the schedule is defined in.

    Raises:
        ValueError: If the zone differs from the schedule's, the index cannot
            represent the schedule's boundaries (a half-hour boundary on
            hourly steps, under ``boundary_policy="strict"``), or the
            schedule is not effective on the index dates or ``study_date``.
    """
    resolved_index = _validate_index(index)
    if boundary_policy not in BOUNDARY_POLICIES:
        allowed = ", ".join(sorted(BOUNDARY_POLICIES))
        raise ValueError(f"'boundary_policy' must be one of: {allowed}")

    definition = _as_definition(schedule)
    metadata = definition.schedule
    zone = _check_timezone(metadata, timezone)
    _validate_schedule_resolution(resolved_index, metadata.identifier, definition.resolution_minutes, zone)
    _validate_study_date(metadata, resolved_index, study_date, zone)

    local_index = resolved_index.tz_convert(zone)
    holidays = definition.holidays
    # A year the index only grazes (the last UTC hour of a year is already the
    # next local year east of UTC) needs no calendar: its steps are fewer than
    # one civil day. Every year the index covers for a day or more does.
    years, counts = np.unique(local_index.year, return_counts=True)
    step_hours = (local_index[1] - local_index[0]).total_seconds() / 3600 if len(local_index) > 1 else 24.0
    covered = {int(year) for year, count in zip(years, counts, strict=True) if count * step_hours >= 24}
    holiday_dates = _holiday_dates(metadata.identifier, holidays, covered)
    holiday_day_type = holidays.day_type if holidays is not None else "sunday"
    labels: list[str] = []
    for timestamp in local_index:
        rule = definition.rule_for(_day_type(timestamp, holiday_dates, holiday_day_type), _season(timestamp))
        minute = timestamp.hour * 60 + timestamp.minute
        label = next(period for start, end, period in rule.intervals if start <= minute < end)
        labels.append(label)
    return tuple(labels)


def resolve_named_tariff(
    index: pd.DatetimeIndex,
    schedule: str | ScheduleDefinition,
    prices: TariffPrices,
    *,
    timezone: str,
    study_date: date | None = None,
    boundary_policy: str = "strict",
) -> ResolvedTariff:
    """Classify and price one tariff schedule, bundled or defined, in the configured civil time."""
    definition = _as_definition(schedule)
    labels = classify_tariff_periods(
        index,
        definition,
        timezone=timezone,
        study_date=study_date,
        boundary_policy=boundary_policy,
    )
    return resolve_tariff(
        index, labels, definition.schedule, prices, timezone=timezone, boundary_policy=boundary_policy
    )


def civil_day_starts(index: pd.DatetimeIndex, timezone: str) -> tuple[int, ...]:
    """Positions where the civil date changes in ``timezone``, plus ``len(index)``."""
    if len(index) == 0:
        return (0,)
    local_days = cast(Any, index.tz_convert(timezone).normalize().tz_localize(None).as_unit("ns")).asi8
    changes = [
        0,
        *(position for position in range(1, len(local_days)) if local_days[position] != local_days[position - 1]),
    ]
    return (*changes, len(index))


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
    timezone: str,
    boundary_policy: str = "strict",
) -> ResolvedTariff:
    """Resolve classified schedule periods to aligned import/export prices.

    ``timezone`` is the civil time the labels were classified in; it sets the
    civil-day boundaries.
    """
    resolved_index = _validate_index(index)
    zone = _nonempty_text(timezone, "timezone")
    try:
        ZoneInfo(zone)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unknown IANA timezone: {zone!r}") from exc
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

    utc_nanoseconds = tuple(int(value) for value in _utc_nanoseconds(resolved_index))
    schedule_hash = _canonical_hash(
        {
            "identifier": schedule.identifier,
            "version": schedule.version,
            "timezone": zone,
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
        timezone=zone,
        schedule=schedule,
        prices=prices,
        period_labels=labels,
        period_codes=codes,
        import_price_per_kwh=import_values,
        export_price_per_kwh=export_values,
        day_starts=civil_day_starts(resolved_index, zone),
        boundary_policy=boundary_policy,
        schedule_hash=schedule_hash,
        price_hash=price_hash,
    )


def resolve_flat_tariff(
    index: pd.DatetimeIndex,
    *,
    timezone: str,
    import_price_per_kwh: float,
    export_price_per_kwh: float,
    fixed_charge_per_day: float = 0.0,
    currency: str = DEFAULT_CURRENCY,
) -> ResolvedTariff:
    """Resolve one import and one export price for every step, as the flat path prices energy."""
    prices = TariffPrices(
        currency=currency,
        import_prices={"all": import_price_per_kwh},
        export_prices={"all": export_price_per_kwh},
        fixed_charge_per_day=fixed_charge_per_day,
        identifier="flat",
        version="1",
    )
    return resolve_tariff(index, ("all",) * len(index), FLAT_SCHEDULE, prices, timezone=timezone)


@dataclass(frozen=True)
class TariffSpec:
    """A configured tariff before it meets a simulation index: the App's ``[tariff]`` table.

    ``schedule`` is a bundled schedule identifier or a
    :class:`ScheduleDefinition`. ``resolve`` classifies and prices one index. Every project year replays
    the start-year calendar (ADR 0002 A2), so a run resolves once and reuses
    the result for every year.
    """

    schedule: str | ScheduleDefinition
    prices: TariffPrices
    boundary_policy: str = "strict"
    study_date: date | None = None

    @property
    def definition(self) -> ScheduleDefinition:
        return _as_definition(self.schedule)

    def resolve(self, index: pd.DatetimeIndex, timezone: str) -> ResolvedTariff:
        return resolve_named_tariff(
            index,
            self.schedule,
            self.prices,
            timezone=timezone,
            study_date=self.study_date,
            boundary_policy=self.boundary_policy,
        )


def result_currency(tariff: TariffSpec | ResolvedTariff | None) -> str:
    """The currency a run's money is in: its tariff's, or the cost catalogue's without one."""
    return tariff.prices.currency if tariff is not None else DEFAULT_CURRENCY


def tariff_provenance(resolved: ResolvedTariff, *, calendar_year: int) -> dict[str, Any]:
    """A JSON-safe record of a resolved tariff for run provenance."""
    schedule, prices = resolved.schedule, resolved.prices
    return {
        "schedule": schedule.identifier,
        "schedule_version": schedule.version,
        "schedule_source": schedule.source,
        "schedule_source_url": schedule.source_url,
        "timezone": resolved.timezone,
        "currency": prices.currency,
        "import_prices": dict(prices.import_prices),
        "export_prices": dict(prices.export_prices),
        "fixed_charge_per_day": prices.fixed_charge_per_day,
        "boundary_policy": resolved.boundary_policy,
        "schedule_hash": resolved.schedule_hash,
        "price_hash": resolved.price_hash,
        # ADR 0002 A2: every project year replays this calendar; weekdays,
        # holidays and effective dates do not advance.
        "calendar_policy": "replay_start_year",
        "calendar_year": int(calendar_year),
    }


def schedule_resolution_minutes(schedule: str | ScheduleDefinition) -> int:
    """The step, in minutes, that lands on every boundary of a schedule: input steps must divide it."""
    return _as_definition(schedule).resolution_minutes
