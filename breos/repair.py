"""Explicit, reported repair of measured load and PV power series.

The simulation boundary rejects PV and load input with a gap, a non-finite
value, or negative load (#151): a missing reading is not a real zero. Measured
data often has such readings, so :func:`repair_series` is the opt-in step a
caller runs first. It fixes only what it is asked to fix, refuses what it
cannot fix safely, and returns an :class:`InputRepairReport` of every change.
Pass the report to :class:`breos.App` as ``input_repairs`` so the repaired
input stays visible in the run's provenance.

Two repairs exist in this version:

- Small negative readings (sensor noise around zero) are clipped to zero.
  Larger or sustained negative readings are refused: for load they usually
  mean the meter recorded net flow (load minus on-site PV), and gross demand
  cannot be recovered without the PV data.
- Gaps (missing timestamps on an otherwise regular index, NaN, and ±inf) are
  refused by default. ``gap_fill="nearby_days"`` fills each missing step with
  the mean of the same time of day on the nearest days that have a valid
  reading, within a bounded window.

Duplicate timestamps and an irregular index are refused, not repaired.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

GAP_FILL_METHODS = ("raise", "nearby_days")
SERIES_KINDS = ("load", "pv")

# Readings down to -10 W are treated as sensor noise and clipped to zero. A
# building never draws negative power, and its standby baseload is normally
# well above this. One register count of a 1 Wh meter is 4 W at 15-minute
# resolution, so 10 W covers two counts of interval rounding, while a net-meter
# reading during any useful PV output is far more negative.
DEFAULT_NEGATIVE_CLIP_W = 10.0
# A negative stretch longer than this is refused even when every reading is
# within the clip threshold. Noise flickers around zero; hours of negative
# readings look like net flow on a day when PV roughly matches load.
DEFAULT_MAX_NEGATIVE_RUN = "1h"
DEFAULT_WINDOW_DAYS = 7
DEFAULT_NEIGHBOUR_DAYS = 2

_NS_PER_DAY = 86_400 * 10**9
# 1970-01-01, day 0 of the epoch day count, was a Thursday (dayofweek 3).
_EPOCH_DAYOFWEEK = 3


@dataclass(frozen=True)
class RepairEvent:
    """One contiguous run of repaired steps.

    Attributes:
        issue: ``"gap"`` (missing timestamps, NaN or ±inf) or
            ``"negative"`` (small negative readings).
        start: ISO timestamp of the first repaired step.
        end: ISO timestamp of the last repaired step (inclusive).
        steps: Number of repaired steps.
        method: ``"nearby_days"`` or ``"clip_to_zero"``.
        missing_timestamps: Steps whose timestamp was absent from the input.
        non_finite_values: Steps present with NaN or ±inf.
        original_min_w: Most negative original reading, for a negative
            event; None for a gap, which has no original value.
        replaced_min_w: Smallest value written.
        replaced_mean_w: Mean value written.
        replaced_max_w: Largest value written.
        energy_wh: Energy the repair adds to the series (negative if it
            removes energy). A gap counts as zero before the repair.
    """

    issue: str
    start: str
    end: str
    steps: int
    method: str
    missing_timestamps: int
    non_finite_values: int
    original_min_w: Optional[float]
    replaced_min_w: float
    replaced_mean_w: float
    replaced_max_w: float
    energy_wh: float

    def to_dict(self) -> dict[str, Any]:
        """Return the event as a strict-JSON-serialisable dict."""
        return asdict(self)


@dataclass(frozen=True)
class InputRepairReport:
    """What :func:`repair_series` changed in one series.

    Attributes:
        series: ``"load"`` or ``"pv"``.
        name: The input series' name, as a string, or None.
        unit: Power unit of the values, always ``"W"``.
        freq: Step of the regular index, as a pandas offset alias.
        start: ISO timestamp of the first step of the repaired series.
        end: ISO timestamp of the last step of the repaired series.
        steps: Number of steps in the repaired series.
        gap_fill: Gap strategy requested, ``"raise"`` or ``"nearby_days"``.
        negative_clip_w: Negative readings down to minus this many watts
            were clipped to zero.
        max_negative_run_minutes: Longest negative stretch allowed to be
            clipped, in minutes.
        window_days: Search window for ``nearby_days``, in days either side.
        neighbour_days: Number of nearest valid days averaged per step.
        same_day_type: Whether ``nearby_days`` preferred days of the same
            type (weekday or weekend).
        clock: Time-of-day basis for ``nearby_days``: ``"local"`` is the
            index's own clock, ``"utc"`` is UTC.
        events: Every repaired run, in time order.
        steps_repaired: Total repaired steps.
        energy_before_wh: Energy of the finite input readings.
        energy_after_wh: Energy of the repaired series.
        energy_added_wh: Energy added by the repairs.
        energy_removed_wh: Energy removed by the repairs, as a
            non-negative number. Neither repair in this version removes
            energy, so it is 0.0.
    """

    series: str
    name: Optional[str]
    unit: str
    freq: str
    start: str
    end: str
    steps: int
    gap_fill: str
    negative_clip_w: float
    max_negative_run_minutes: float
    window_days: int
    neighbour_days: int
    same_day_type: bool
    clock: str
    events: tuple[RepairEvent, ...]
    steps_repaired: int
    energy_before_wh: float
    energy_after_wh: float
    energy_added_wh: float
    energy_removed_wh: float

    @property
    def repaired(self) -> bool:
        """True when at least one step was changed."""
        return bool(self.events)

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a strict-JSON-serialisable dict.

        ``json.dumps(report.to_dict(), allow_nan=False)`` always succeeds:
        every value is a str, int, finite float, bool, None, list or dict.
        """
        record = asdict(self)
        record["events"] = [event.to_dict() for event in self.events]
        return record


def repair_series(
    series: pd.Series,
    *,
    kind: str = "load",
    gap_fill: str = "raise",
    freq: Optional[str] = None,
    index: Optional[pd.DatetimeIndex] = None,
    negative_clip_w: float = DEFAULT_NEGATIVE_CLIP_W,
    max_negative_run: Union[str, pd.Timedelta] = DEFAULT_MAX_NEGATIVE_RUN,
    window_days: int = DEFAULT_WINDOW_DAYS,
    neighbour_days: int = DEFAULT_NEIGHBOUR_DAYS,
    same_day_type: Optional[bool] = None,
) -> tuple[pd.Series, InputRepairReport]:
    """Repair a measured power series and report every change.

    Run this before simulating, only when repair is intended. The simulation
    itself keeps rejecting gaps, non-finite values and negative load.

    Checks, in order:

    1. The index must be a :class:`pandas.DatetimeIndex` without NaT or
       duplicate timestamps, sorted, and regular at ``freq``: every
       timestamp lies on the grid of the first one. Timestamps missing from
       that grid are gaps. A tz-aware index is stepped in absolute time, so a
       local index across a DST change is regular.
    2. Negative readings. Readings from ``-negative_clip_w`` up to zero are
       clipped to zero, unless a negative stretch lasts longer than
       ``max_negative_run``. Anything more negative, or a longer stretch,
       raises ``ValueError``: for load it usually means net flow (load minus
       on-site PV), which cannot be turned into gross demand without the PV
       data.
    3. Gaps: missing timestamps, NaN and ±inf. ``gap_fill="raise"`` (the
       default) raises ``ValueError``. ``gap_fill="nearby_days"`` fills each
       gap step with the mean of the same time of day on the
       ``neighbour_days`` nearest days within ``window_days`` either side that
       have a valid reading there. Only original readings (after clipping)
       are used, never filled ones. With ``same_day_type``, days of the same
       type as the gap (weekday or weekend) are used when any is available,
       and other days only when none is. Nearer days come first, and the
       earlier of two equally near days. A gap step without any valid day in
       the window raises ``ValueError``. Public holidays are not treated
       separately.

    Args:
        series: Power in W on a DatetimeIndex, for example a measured load
            column or measured PV output.
        kind: ``"load"`` or ``"pv"``. It sets the error messages and the
            defaults for ``same_day_type`` and the time-of-day clock: load
            follows the index's own (legal) clock and prefers the same day
            type; PV follows UTC, closer to solar time, and ignores the day
            type.
        gap_fill: ``"raise"`` or ``"nearby_days"``.
        freq: Step of the index, such as ``"h"`` or ``"15min"``. Inferred as
            the most common step when omitted. It must divide one day.
        index: Optional target index, regular at the same step, that the
            repaired series must cover (for example the simulation year).
            Its timestamps missing from ``series`` are gaps. Every timestamp
            of ``series`` must be on it.
        negative_clip_w: Clip threshold in W, non-negative. 0 refuses every
            negative reading.
        max_negative_run: Longest negative stretch that is clipped, as a
            :class:`pandas.Timedelta` or a string such as ``"1h"``.
        window_days: Days either side searched by ``nearby_days``.
        neighbour_days: Number of nearest valid days averaged per step.
        same_day_type: Prefer days of the same type. Defaults to True for
            load and False for PV.

    Returns:
        ``(repaired, report)``: the repaired float series on the full regular
        index, with the input's name, and an :class:`InputRepairReport`. With
        nothing to repair, the series is returned as float and the report has
        no events.

    Raises:
        TypeError: If ``series`` is not a numeric Series on a DatetimeIndex.
        ValueError: For an invalid option, duplicate or unsorted timestamps,
            an irregular index, negative readings beyond the clip rules, gaps
            with ``gap_fill="raise"``, or a gap step without a valid day in
            the window.

    Example:
        >>> repaired, report = repair_series(load_w, gap_fill="nearby_days")  # doctest: +SKIP
        >>> app = App(config, input_repairs=[report])  # doctest: +SKIP
    """
    kind = _choice(kind, SERIES_KINDS, "kind")
    gap_fill = _choice(gap_fill, GAP_FILL_METHODS, "gap_fill")
    clip_w = _non_negative_float(negative_clip_w, "negative_clip_w")
    max_run = _non_negative_timedelta(max_negative_run, "max_negative_run")
    window_days = _positive_int(window_days, "window_days")
    neighbour_days = _positive_int(neighbour_days, "neighbour_days")
    if same_day_type is None:
        same_day_type = kind == "load"
    elif not isinstance(same_day_type, bool):
        raise TypeError(f"same_day_type must be a bool or None, got {type(same_day_type).__name__}")

    if not isinstance(series, pd.Series):
        raise TypeError(f"repair_series needs a pandas Series, got {type(series).__name__}")
    if not pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype):
        raise TypeError(f"{kind} series must be numeric power in W, got dtype {series.dtype}")

    step = _step(series.index, freq, kind)
    full = _regular_index(series.index, step, index, kind)
    hours = step / pd.Timedelta(hours=1)

    present = full.isin(series.index)
    values = series.reindex(full).to_numpy(dtype=float, copy=True)
    finite = np.isfinite(values)
    energy_before_wh = float(values[finite].sum() * hours)

    events: list[tuple[int, RepairEvent]] = []

    negative = finite & (values < 0.0)
    if negative.any():
        _refuse_large_negatives(values, negative, clip_w, max_run, step, full, kind)
        for first, last in _runs(negative):
            original = values[first : last + 1].copy()
            values[first : last + 1] = 0.0
            events.append(
                (
                    first,
                    _event(
                        "negative",
                        "clip_to_zero",
                        full,
                        first,
                        last,
                        values[first : last + 1],
                        energy_wh=float(-original.sum() * hours),
                        original_min_w=float(original.min()),
                    ),
                )
            )

    gaps = ~finite
    if gaps.any():
        gap_runs = _runs(gaps)
        if gap_fill == "raise":
            first, last = gap_runs[0]
            raise ValueError(
                f"{kind} series has {int(gaps.sum())} missing or non-finite steps in {len(gap_runs)} "
                f"gap(s), the first from {full[first].isoformat()} to {full[last].isoformat()} "
                f"({last - first + 1} steps). A missing reading is not a zero. Pass "
                "gap_fill='nearby_days' to fill gaps from the same time of day on nearby days, "
                "or fix the data."
            )
        filled = _fill_from_nearby_days(
            values, finite, full, kind=kind, window=window_days, neighbours=neighbour_days, same_type=same_day_type
        )
        for first, last in gap_runs:
            run = slice(first, last + 1)
            values[run] = filled[run]
            missing = int((~present[run]).sum())
            events.append(
                (
                    first,
                    _event(
                        "gap",
                        "nearby_days",
                        full,
                        first,
                        last,
                        values[run],
                        energy_wh=float(values[run].sum() * hours),
                        missing_timestamps=missing,
                        non_finite_values=last - first + 1 - missing,
                    ),
                )
            )

    ordered = tuple(event for _, event in sorted(events, key=lambda item: item[0]))
    energies = [event.energy_wh for event in ordered]
    repaired = pd.Series(values, index=full, name=series.name)
    report = InputRepairReport(
        series=kind,
        name=None if series.name is None else str(series.name),
        unit="W",
        freq=_freq_alias(step),
        start=full[0].isoformat(),
        end=full[-1].isoformat(),
        steps=len(full),
        gap_fill=gap_fill,
        negative_clip_w=clip_w,
        max_negative_run_minutes=max_run / pd.Timedelta(minutes=1),
        window_days=window_days,
        neighbour_days=neighbour_days,
        same_day_type=same_day_type,
        clock="utc" if kind == "pv" and full.tz is not None else "local",
        events=ordered,
        steps_repaired=sum(event.steps for event in ordered),
        energy_before_wh=energy_before_wh,
        energy_after_wh=float(values.sum() * hours),
        energy_added_wh=float(sum(e for e in energies if e > 0)),
        energy_removed_wh=float(-sum(e for e in energies if e < 0)),
    )
    return repaired, report


def input_repair_records(reports: Any) -> Optional[list[dict[str, Any]]]:
    """Normalise ``App(input_repairs=...)`` into strict-JSON provenance records.

    Accepts None, one :class:`InputRepairReport` or mapping, or an iterable of
    them. Mappings are kept as given (for example a report saved with
    ``to_dict()`` and loaded back from JSON) but must be strict JSON.
    """
    if reports is None:
        return None
    if isinstance(reports, (InputRepairReport, Mapping)):
        reports = [reports]
    if isinstance(reports, (str, bytes)) or not isinstance(reports, Iterable):
        raise TypeError("input_repairs must be an InputRepairReport, a dict, or a list of them")
    records = []
    for report in reports:
        if isinstance(report, InputRepairReport):
            record = report.to_dict()
        elif isinstance(report, Mapping):
            record = dict(report)
        else:
            raise TypeError(f"input_repairs entries must be InputRepairReport or dict, got {type(report).__name__}")
        try:
            records.append(json.loads(json.dumps(record, allow_nan=False)))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"input_repairs entry is not strict JSON: {exc}") from exc
    return records


def _choice(value: Any, allowed: tuple[str, ...], label: str) -> str:
    normalized = str(value).strip().lower() if isinstance(value, str) else value
    if normalized not in allowed:
        raise ValueError(f"{label} must be one of {list(allowed)}, got {value!r}")
    return normalized


def _non_negative_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{label} must be a number, got {type(value).__name__}")
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be finite and non-negative, got {value}")
    return value


def _non_negative_timedelta(value: Any, label: str) -> pd.Timedelta:
    try:
        delta = pd.Timedelta(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a duration such as '1h', got {value!r}") from exc
    if pd.isna(delta) or delta < pd.Timedelta(0):
        raise ValueError(f"{label} must be a non-negative duration, got {value!r}")
    return delta


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{label} must be a positive integer, got {value!r}")
    return int(value)


def _step(idx: pd.Index, freq: Optional[str], kind: str) -> pd.Timedelta:
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError(f"{kind} series needs a DatetimeIndex, got {type(idx).__name__}")
    if len(idx) < 2:
        raise ValueError(f"{kind} series needs at least two timestamps to define its step")
    if idx.hasnans:
        raise ValueError(f"{kind} series has {int(idx.isna().sum())} NaT timestamps")
    duplicated = idx.duplicated(keep=False)
    if duplicated.any():
        raise ValueError(
            f"{kind} series has {int(duplicated.sum())} rows on duplicate timestamps (first at "
            f"{idx[duplicated][0].isoformat()}). repair_series does not choose between duplicate "
            "readings; drop or aggregate them first."
        )
    if not idx.is_monotonic_increasing:
        raise ValueError(f"{kind} series index is not sorted; call sort_index() first")
    if freq is not None:
        try:
            step = pd.Timedelta(pd.tseries.frequencies.to_offset(freq))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"freq must be a fixed step such as 'h' or '15min', got {freq!r}") from exc
    else:
        step = pd.Series(idx[1:] - idx[:-1]).mode().iloc[0]
    if step <= pd.Timedelta(0) or pd.Timedelta(days=1) % step != pd.Timedelta(0):
        raise ValueError(f"the step of a {kind} series must divide one day evenly, got {step}")
    return step


def _regular_index(
    idx: pd.DatetimeIndex, step: pd.Timedelta, target: Optional[pd.DatetimeIndex], kind: str
) -> pd.DatetimeIndex:
    """Return the full regular index, refusing an index off the step grid."""
    ns = idx.as_unit("ns").asi8
    step_ns = step.as_unit("ns").value
    off_grid = np.flatnonzero(np.diff(ns) % step_ns != 0)
    if off_grid.size:
        row = int(off_grid[0]) + 1
        raise ValueError(
            f"{kind} series index is irregular: {idx[row].isoformat()} follows {idx[row - 1].isoformat()}, "
            f"which is not a whole number of {_freq_alias(step)} steps. repair_series fills missing "
            "timestamps on a regular grid only; resample the data first."
        )
    if target is None:
        return pd.date_range(idx[0], idx[-1], freq=step, name=idx.name)

    if not isinstance(target, pd.DatetimeIndex) or len(target) == 0:
        raise TypeError("index must be a non-empty DatetimeIndex")
    if (target.tz is None) != (idx.tz is None):
        raise ValueError("index and the series must both be timezone-aware or both naive")
    if target.hasnans or target.has_duplicates or not target.is_monotonic_increasing:
        raise ValueError("index must be sorted, without NaT or duplicate timestamps")
    if len(target) > 1 and (np.diff(target.as_unit("ns").asi8) != step_ns).any():
        raise ValueError(f"index must be regular at the series step ({_freq_alias(step)})")
    outside = ~idx.isin(target)
    if outside.any():
        raise ValueError(
            f"{kind} series has {int(outside.sum())} timestamps that are not on index (first "
            f"{idx[outside][0].isoformat()})."
        )
    return target


def _refuse_large_negatives(
    values: np.ndarray,
    negative: np.ndarray,
    clip_w: float,
    max_run: pd.Timedelta,
    step: pd.Timedelta,
    full: pd.DatetimeIndex,
    kind: str,
) -> None:
    if kind == "load":
        cause = (
            "Load must be gross building demand. Readings like these usually mean the meter recorded "
            "net flow (load minus on-site PV), and gross demand cannot be recovered without the PV "
            "data, so repair_series does not repair them."
        )
    else:
        cause = (
            "PV output is not negative beyond inverter standby draw. Check the sign convention and "
            "whether the series is a net flow; repair_series does not repair it."
        )
    large = negative & (values < -clip_w)
    if large.any():
        first = int(np.flatnonzero(large)[0])
        raise ValueError(
            f"{kind} series is below -{clip_w:g} W at {int(large.sum())} steps (minimum "
            f"{float(values[large].min()):.6g} W, first at {full[first].isoformat()}). {cause}"
        )
    for first, last in _runs(negative):
        duration = (last - first + 1) * step
        if duration > max_run:
            raise ValueError(
                f"{kind} series is negative for {last - first + 1} consecutive steps ({duration}) from "
                f"{full[first].isoformat()}, longer than max_negative_run ({max_run}). Sensor noise "
                f"flickers around zero; a sustained negative reading does not. {cause}"
            )


def _fill_from_nearby_days(
    values: np.ndarray,
    valid: np.ndarray,
    full: pd.DatetimeIndex,
    *,
    kind: str,
    window: int,
    neighbours: int,
    same_type: bool,
) -> np.ndarray:
    """Return fill values for every invalid step, or raise if one has no donor."""
    clock = full.tz_convert("UTC") if kind == "pv" and full.tz is not None else full
    wall = clock.tz_localize(None) if clock.tz is not None else clock
    wall_ns = wall.as_unit("ns").asi8
    day = wall_ns // _NS_PER_DAY
    time_of_day = wall_ns - day * _NS_PER_DAY

    # Mean valid reading per (day, time of day). A repeated fall-back hour has
    # two steps on one key; both readings count.
    donors = pd.Series(values[valid]).groupby([day[valid], time_of_day[valid]]).mean().to_dict() if valid.any() else {}

    filled = values.copy()
    unfilled: list[int] = []
    for pos in np.flatnonzero(~valid):
        gap_day, tod = int(day[pos]), int(time_of_day[pos])
        gap_weekend = _is_weekend(gap_day)
        candidates = []
        for distance in range(1, window + 1):
            for side in (-1, 1):
                reading = donors.get((gap_day + side * distance, tod))
                if reading is not None:
                    candidates.append((distance, side, _is_weekend(gap_day + side * distance), reading))
        if same_type:
            matching = [c for c in candidates if c[2] == gap_weekend]
            candidates = matching or candidates
        if not candidates:
            unfilled.append(int(pos))
            continue
        chosen = sorted(candidates, key=lambda c: (c[0], c[1]))[:neighbours]
        filled[pos] = float(np.mean([c[3] for c in chosen]))

    if unfilled:
        raise ValueError(
            f"{kind} series has {len(unfilled)} gap steps with no valid reading at the same time of day "
            f"within {window} days either side (first at {full[unfilled[0]].isoformat()}). Widen "
            "window_days or fix the data."
        )
    return filled


def _is_weekend(day_number: int) -> bool:
    return (day_number + _EPOCH_DAYOFWEEK) % 7 >= 5


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return (first, last) positions of each run of True values."""
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def _event(
    issue: str,
    method: str,
    full: pd.DatetimeIndex,
    first: int,
    last: int,
    written: np.ndarray,
    *,
    energy_wh: float,
    original_min_w: Optional[float] = None,
    missing_timestamps: int = 0,
    non_finite_values: int = 0,
) -> RepairEvent:
    return RepairEvent(
        issue=issue,
        start=full[first].isoformat(),
        end=full[last].isoformat(),
        steps=last - first + 1,
        method=method,
        missing_timestamps=missing_timestamps,
        non_finite_values=non_finite_values,
        original_min_w=original_min_w,
        replaced_min_w=float(written.min()),
        replaced_mean_w=float(written.mean()),
        replaced_max_w=float(written.max()),
        energy_wh=energy_wh,
    )


def _freq_alias(step: pd.Timedelta) -> str:
    return pd.tseries.frequencies.to_offset(step).freqstr
