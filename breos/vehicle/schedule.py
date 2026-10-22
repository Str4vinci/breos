"""Resolve civil-time absences to immutable arrays, without dispatch or weather.

The template dedicates its first and last away steps to driving. Trip files
retain actual parked hours, even in a partially-away step. Repeated civil
times in templates use the first departure and last arrival; nonexistent
times move to the transition. Ambiguous/nonexistent naive CSV timestamps
are refused: use explicit offsets to describe those instants.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Mapping, cast
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from breos.vehicle.config import VehicleConfig, VehicleDaySchedule, VehicleScheduleSpec

_HOUR_NS = 3_600_000_000_000


@dataclass(frozen=True, eq=False)
class VehicleSchedule:
    """Resolved schedule in Wh and elapsed parked hours, one entry per step.

    ``steps_to_departure`` includes the current plugged step, and is zero
    away or when a file has no departures. Driving runs first, then away
    charging for ``park_hours``, then the second driving leg. More than one
    parked interval within a step is collapsed into that order (ADR V3).
    """

    index: pd.DatetimeIndex
    hours_per_step: float
    plugged: np.ndarray
    steps_to_departure: np.ndarray
    drive_first_wh: np.ndarray
    park_hours: np.ndarray
    drive_second_wh: np.ndarray

    def __post_init__(self) -> None:
        for name, dtype in (
            ("plugged", np.bool_),
            ("steps_to_departure", np.int64),
            ("drive_first_wh", np.float64),
            ("park_hours", np.float64),
            ("drive_second_wh", np.float64),
        ):
            array: np.ndarray = np.array(getattr(self, name), dtype=dtype, copy=True, order="C")
            if array.ndim != 1 or len(array) != len(self.index):
                raise ValueError(f"'{name}' must have one entry per simulation step")
            if not np.isfinite(array).all() or (array < 0).any():
                raise ValueError(f"'{name}' must be finite and non-negative")
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    def schedule_hash(self) -> str:
        """SHA-256 of the resolved timeline and arrays, with explicit byte order."""
        digest = hashlib.sha256(b"breos-vehicle-schedule-v1")
        digest.update(struct.pack("<qd", len(self.index), self.hours_per_step))
        digest.update(cast(Any, self.index.as_unit("ns")).asi8.astype("<i8").tobytes())
        digest.update(self.plugged.astype(np.uint8).tobytes())
        digest.update(self.steps_to_departure.astype("<i8").tobytes())
        for array in (self.drive_first_wh, self.park_hours, self.drive_second_wh):
            digest.update((array + 0.0).astype("<f8").tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class _Segment:
    start: pd.Timestamp
    end: pd.Timestamp
    kind: str
    energy_wh: float = 0.0
    charging: bool = False


def _civil(day: pd.Timestamp, clock: str, timezone: str, *, departure: bool) -> pd.Timestamp:
    naive = pd.Timestamp(f"{day.date()} {clock}")
    # Earliest departure, latest arrival: never invent extra home time in
    # the repeated hour. Both nonexistent times meet at the DST transition.
    return naive.tz_localize(timezone, ambiguous=departure, nonexistent="shift_forward")


def _timestamp(value: Any, timezone: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            raise ValueError("missing timestamp")
        if stamp.tzinfo is None:
            return stamp.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
        return stamp.tz_convert(timezone)
    except Exception as exc:
        raise ValueError(
            f"Invalid vehicle trip timestamp {value!r}; DST transition times need an explicit offset"
        ) from exc


def _read_segments(path: str, timezone: str) -> list[_Segment]:
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"start", "end", "kind"}
    allowed = required | {"energy_kwh", "away_charging"}
    if not required <= set(frame) or set(frame) - allowed:
        raise ValueError("Vehicle trip CSV needs start, end, kind, and drive energy_kwh / park away_charging columns")
    segments: list[_Segment] = []
    for row in frame.to_dict("records"):
        start, end = (_timestamp(row[key], timezone) for key in ("start", "end"))
        if end <= start:
            raise ValueError("Vehicle trip segments need end > start")
        if segments and start < segments[-1].end:
            raise ValueError("Vehicle trip segments must be in time order and must not overlap")
        kind = row["kind"]
        if kind not in ("drive", "park"):
            raise ValueError("Vehicle trip kind must be 'drive' or 'park'")
        energy = 0.0
        charging = False
        if kind == "drive":
            try:
                if isinstance(row.get("energy_kwh"), bool):
                    raise ValueError("a boolean is not driving energy")
                energy = float(row.get("energy_kwh", "")) * 1000
            except (TypeError, ValueError) as exc:
                raise ValueError("Vehicle drive segment needs finite non-negative energy_kwh") from exc
            if not np.isfinite(energy) or energy < 0:
                raise ValueError("Vehicle drive segment needs finite non-negative energy_kwh")
            if row.get("away_charging", "") not in ("", False, "false", "False"):
                raise ValueError("Vehicle drive segment cannot have away_charging")
        else:
            flag = str(row.get("away_charging", "")).lower()
            if flag not in ("true", "false"):
                raise ValueError("Vehicle park segment needs away_charging = true or false")
            charging = flag == "true"
            if row.get("energy_kwh", "") not in ("", 0, "0"):
                raise ValueError("Vehicle park segment cannot have driving energy_kwh")
        segments.append(_Segment(start, end, kind, energy, charging))
    return segments


def _file_absences(path: str, timezone: str, year: int) -> list[list[_Segment]]:
    segments = _read_segments(path, timezone)
    # A file is an annual calendar. Its halo carries trips across the seam
    # and supplies the next departure beyond a truncated simulation window.
    # Replay the resolved reference year, including its DST hours, exactly
    # as the per-step arrays replay in project_years. Relocalizing each row
    # in an adjacent year could collapse a drive within the repeated hour
    # to zero duration when that date is no longer a DST transition.
    year_start = pd.Timestamp(f"{year}-01-01", tz=timezone)
    year_end = pd.Timestamp(f"{year + 1}-01-01", tz=timezone)
    if segments and not any(s.start < year_end and s.end > year_start for s in segments):
        raise ValueError(f"Vehicle trip CSV timestamps must use the simulation's reference calendar year ({year})")
    period = year_end - year_start
    replayed = [
        _Segment(s.start + period * repeat, s.end + period * repeat, s.kind, s.energy_wh, s.charging)
        for repeat in (-1, 0, 1)
        for s in segments
    ]
    replayed.sort(key=lambda segment: segment.start)
    absences: list[list[_Segment]] = []
    for segment in replayed:
        if absences and segment.start < absences[-1][-1].end:
            raise ValueError("Vehicle trip calendar overlaps when replayed across the year seam")
        if absences and segment.start == absences[-1][-1].end:
            absences[-1].append(segment)
        else:
            absences.append([segment])
    return absences


def _template_absences(spec: VehicleScheduleSpec, local: pd.DatetimeIndex, timezone: str) -> list[list[_Segment]]:
    first_year, last_year = local[0].year, local[-1].year
    absences: list[list[_Segment]] = []
    for day in pd.date_range(f"{first_year - 1}-12-30", f"{last_year + 1}-01-02", freq="D"):
        # Outside the simulated calendar, repeat its own December/January
        # day types rather than next year's different weekdays.
        source = day
        if day.year < first_year:
            source = day + pd.DateOffset(years=1)
        elif day.year > last_year:
            source = day - pd.DateOffset(years=1)
        rule = spec.weekday if source.weekday() < 5 else spec.weekend
        assert isinstance(rule, VehicleDaySchedule)
        start = _civil(day, rule.depart, timezone, departure=True)
        arrival_day = day + timedelta(days=rule.arrive <= rule.depart)
        end = _civil(arrival_day, rule.arrive, timezone, departure=False)
        if end <= start or (absences and start < absences[-1][-1].end):
            raise ValueError("Vehicle day rules produce overlapping or non-positive absences")
        # The actual driving legs are placed on the rounded away steps below.
        absences.append([_Segment(start, end, "template", rule.trip_kwh * 1000, rule.away_charging)])
    return absences


def resolve_vehicle_schedule(
    config: VehicleConfig | VehicleScheduleSpec | Mapping[str, Any],
    index: pd.DatetimeIndex,
    *,
    timezone: str,
) -> VehicleSchedule:
    """Resolve a schedule on an aware, regular interval-start simulation index.

    ``config`` may be a VehicleConfig, a schedule spec, or the schedule table
    itself. The configured zone determines day types regardless of the
    index's display zone. A file uses the simulation's reference calendar
    year; its resolved timeline replays at the seam. An empty CSV means
    always home with no departure target.
    """
    ZoneInfo(timezone)
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None or index.empty or index.hasnans:
        raise ValueError("Vehicle schedule needs a non-empty timezone-aware simulation index")
    times = cast(Any, index.as_unit("ns")).asi8
    if len(index) == 1:
        if index.freq is None:
            raise ValueError("A one-step vehicle schedule index needs a fixed frequency")
        step = index.freq.nanos
    else:
        step = int(times[1] - times[0])
    if step <= 0 or (np.diff(times) != step).any():
        raise ValueError("Vehicle schedule needs a regular increasing simulation index")
    if isinstance(config, VehicleConfig):
        config = config.schedule
    spec = config if isinstance(config, VehicleScheduleSpec) else VehicleScheduleSpec(**config)
    local = index.tz_convert(timezone)
    absences = (
        _file_absences(spec.file, timezone, local[0].year) if spec.file else _template_absences(spec, local, timezone)
    )
    n = len(index)
    plugged = np.ones(n, dtype=np.bool_)
    first = np.zeros(n)
    second = np.zeros(n)
    parked = np.zeros(n)
    first_park = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    drives: list[_Segment] = []
    departures: list[int] = []
    origin = int(times[0])
    for segments in absences:
        start, end = segments[0].start.value, segments[-1].end.value
        left = (start - origin) // step
        right = -((origin - end) // step)
        departures.append(left)
        lo, hi = max(0, left), min(n, right)
        if lo >= hi:
            continue
        plugged[lo:hi] = False
        if segments[0].kind == "template":
            segment = segments[0]
            if 0 <= left < n:
                first[left] += segment.energy_wh / 2
            if 0 <= right - 1 < n:
                second[right - 1] += segment.energy_wh / 2
            if segment.charging:
                parked[max(0, left + 1) : min(n, right - 1)] += step / _HOUR_NS
            continue
        for segment in segments:
            if segment.kind == "park":
                a_pos = max(lo, (segment.start.value - origin) // step)
                b_pos = min(hi, -((origin - segment.end.value) // step))
                for pos in range(a_pos, b_pos):
                    a, b = max(int(times[pos]), segment.start.value), min(int(times[pos]) + step, segment.end.value)
                    if b > a:
                        first_park[pos] = min(first_park[pos], a)
                        if segment.charging:
                            parked[pos] += (b - a) / _HOUR_NS
            else:
                drives.append(segment)
    # First parked time, even without charging permission, separates the
    # withdrawals. Use all segments in the step, including separate short
    # absences with no complete home step between them.
    for segment in drives:
        duration = segment.end.value - segment.start.value
        lo = max(0, (segment.start.value - origin) // step)
        hi = min(n, -((origin - segment.end.value) // step))
        for pos in range(lo, hi):
            a, b = max(int(times[pos]), segment.start.value), min(int(times[pos]) + step, segment.end.value)
            if b > a:
                before = max(0, min(b, int(first_park[pos])) - a)
                first[pos] += segment.energy_wh * before / duration
                second[pos] += segment.energy_wh * (b - a - before) / duration
    counts = np.zeros(n, dtype=np.int64)
    if departures:
        positions = np.arange(n)
        starts = np.array(sorted(departures), dtype=np.int64)
        next_pos = np.searchsorted(starts, positions, side="right")
        valid = plugged & (next_pos < len(starts))
        counts[valid] = starts[next_pos[valid]] - positions[valid]
    return VehicleSchedule(index, step / _HOUR_NS, plugged, counts, first, parked, second)
