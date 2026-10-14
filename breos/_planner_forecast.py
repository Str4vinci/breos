"""Planner forecasts other than persistence (ADR 0002 A20).

``daily_persistence`` forecasts the planning window by repeating the day
before the decision. With ``smart_charging.forecast = "weather"`` its PV
forecast comes from day-ahead forecast weather instead: one weather file per
lead day, the first holding for each hour the value forecast at least one day
before it, the second at least two days before, and so on (the layout of the
Open-Meteo Previous Runs archive, :func:`breos.weather.fetch_forecast_archive`).
Each file runs through the run's own PV model, so the forecast PV differs
from the true PV by the weather forecast's error alone.

A decision plans each step on the shortest lead whose forecasts of every raw
hour that shapes the step were published before the decision. A forecast of
the hour labelled ``h`` at lead ``k`` is issued at ``h - k days`` and
published ``forecast_publication_delay_hours`` later. The raw hours that
shape a step are those labelled within ``resampling_reach`` of it, on either
side (the hourly-to-15-minute resampling interpolates across neighbouring
hours). Times are the weather's own: :func:`source_times` maps every raw
hour through the App's own restamping onto the simulated year, so a day the
calendar drops (29 February of a leap year on a common calendar) or repeats
(a filled 29 February) is counted where it really falls. A step takes lead
``ceil(max(source age, calendar age) / 1 day)``; one that would need a lead
beyond the last file keeps the persistence value. Load and temperature stay
persistence.

``forecast = "perfect"`` plans on the run's own future PV, load and
temperature: a perfect-information reference, not a controller.

The forecast changes only what the planner believes. The canonical dispatch
step remains the only simulation of the battery.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from breos._controller import ControllerDayInput

# How far from a step the raw hours that shape its PV can be labelled, by
# simulation resolution. At 15 minutes the resampler reads neighbouring hours
# (a raw hour moves the steps from 4 h before its label to 3 h after it;
# pinned by a test that perturbs single hours); an hourly run reads its own
# hour.
RESAMPLING_REACH_HOURS = {"15min": 4.0, "h": 0.0}
# A step needing a lead beyond the last file keeps persistence: lead 0.
PERSISTENCE_LEAD = 0
_DAY_S = 86400
_HOUR_S = 3600


def resampling_reach_seconds(freq: str) -> int:
    if freq not in RESAMPLING_REACH_HOURS:
        raise ValueError(f"smart_charging.forecast = 'weather' supports resolutions {sorted(RESAMPLING_REACH_HOURS)}")
    return int(RESAMPLING_REACH_HOURS[freq] * _HOUR_S)


def epoch_seconds(index: pd.DatetimeIndex) -> np.ndarray:
    """Each timestamp in seconds since the epoch, UTC (a naive index is read as UTC)."""
    utc = index.tz_convert("UTC") if index.tz is not None else index.tz_localize("UTC")
    return np.asarray((utc - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1), dtype=np.int64)


# Kept under its earlier name for the controller's calendar.
calendar_epoch_seconds = epoch_seconds


def source_times(
    raw_calendar_s: np.ndarray, raw_source_s: np.ndarray, steps_s: np.ndarray, step_s: int, reach_s: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per step, the latest source time of the raw hours shaping it, and the source time of a decision there.

    ``raw_calendar_s`` and ``raw_source_s`` give every raw weather row's
    label on the simulated calendar (after restamping) and in the weather's
    own time; ``steps_s`` each step's start on the calendar. A step is
    shaped by the rows labelled from ``reach_s`` before its start to
    ``reach_s`` after its end. A decision at a step's start is at the
    earliest source time that the rows labelled within the hour either side
    of it map it to. A step with no such row gets the calendar time.
    """
    order = np.argsort(raw_calendar_s, kind="stable")
    calendar = raw_calendar_s[order]
    source = raw_source_s[order]
    offset = source - calendar

    def window(lo_s: np.ndarray, hi_s: np.ndarray, values: np.ndarray, pick: Any, empty: np.ndarray) -> np.ndarray:
        lo = np.searchsorted(calendar, lo_s, side="left")
        hi = np.searchsorted(calendar, hi_s, side="right")
        width = int(np.max(hi - lo, initial=0))
        out = empty.astype(np.float64).copy()
        for k in range(width):
            position = lo + k
            inside = position < hi
            value = values[np.minimum(position, len(values) - 1)]
            out = np.where(inside, pick(out, value), out)
        return out

    latest = window(steps_s - reach_s, steps_s + step_s + reach_s, source, np.fmax, np.full(len(steps_s), np.nan))
    latest = np.where(np.isnan(latest), steps_s + step_s + reach_s, latest).astype(np.int64)
    shift = window(steps_s - _HOUR_S, steps_s + _HOUR_S, offset, np.fmin, np.full(len(steps_s), np.nan))
    decision = (steps_s + np.where(np.isnan(shift), 0, shift)).astype(np.int64)
    return latest, decision


@dataclass
class PlannerForecast:
    """What the planner forecasts with in place of persistence, on the run's calendar.

    ``kind`` is ``weather`` (PV from ``pv_dc_by_lead_w``, one row per lead
    day, undegraded) or ``perfect`` (the run's PV, load and temperature).
    ``calendar_epoch_s`` holds each step's start in seconds since the epoch,
    UTC. Under ``weather``, ``support_source_s`` and ``decision_source_s``
    come from :func:`source_times`. PV of project year ``y`` (from 0) is
    degraded by ``(1 - rate) ** y``, as the runner degrades the true PV; a
    planned step past the calendar's end is in the next project year.
    """

    kind: str
    calendar_epoch_s: np.ndarray
    step_s: int
    pv_degradation_rate: float
    support_source_s: np.ndarray | None = None
    decision_source_s: np.ndarray | None = None
    pv_dc_by_lead_w: np.ndarray | None = None
    true_pv_dc_w: np.ndarray | None = None
    true_load_w: np.ndarray | None = None
    true_temperature_c: np.ndarray | None = None
    publication_delay_s: int = 0
    resampling_reach_s: int = 0
    files: tuple[dict[str, Any], ...] = ()
    decisions: int = 0
    lead_steps: Counter[int] = field(default_factory=Counter)

    def __post_init__(self) -> None:
        n = len(self.calendar_epoch_s)
        if self.kind == "weather":
            if self.pv_dc_by_lead_w is None or self.pv_dc_by_lead_w.ndim != 2 or self.pv_dc_by_lead_w.shape[1] != n:
                raise ValueError("a weather forecast needs PV of shape (leads, steps)")
            if self.support_source_s is None or self.decision_source_s is None:
                # The weather's own time is the calendar's.
                self.support_source_s, self.decision_source_s = source_times(
                    self.calendar_epoch_s, self.calendar_epoch_s, self.calendar_epoch_s, self.step_s, 0
                )
            if len(self.support_source_s) != n or len(self.decision_source_s) != n:
                raise ValueError("support_source_s and decision_source_s need one value per step")
        elif self.kind == "perfect":
            for name in ("true_pv_dc_w", "true_load_w", "true_temperature_c"):
                value = getattr(self, name)
                if value is None or len(value) != n:
                    raise ValueError(f"a perfect forecast needs {name} with one value per step")
        else:
            raise ValueError(f"unknown planner forecast {self.kind!r}")

    @property
    def steps(self) -> int:
        return len(self.calendar_epoch_s)

    def leads(self, start: int, count: int) -> np.ndarray:
        """The lead day of each of ``count`` steps planned at calendar step ``start``; 0 is persistence.

        A step past the calendar's end replays the calendar's start (A2): its
        weather is older in source time, so its calendar age decides.
        """
        assert self.pv_dc_by_lead_w is not None
        assert self.support_source_s is not None and self.decision_source_s is not None
        n = self.steps
        j = np.arange(count, dtype=np.int64)
        steps = start + j
        delay = self.publication_delay_s
        calendar_age = (j + 1) * self.step_s + self.resampling_reach_s + delay
        source_age = self.support_source_s[steps % n] + delay - self.decision_source_s[start]
        age = np.where(steps < n, np.maximum(calendar_age, source_age), calendar_age)
        lead = np.maximum(-((-age) // _DAY_S), 1)
        return np.where(lead > len(self.pv_dc_by_lead_w), PERSISTENCE_LEAD, lead).astype(np.int64)

    def series(
        self, day: ControllerDayInput, count: int, pv: np.ndarray, load: np.ndarray, temperature: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The planner's PV, load and temperature for ``count`` steps from ``day``'s decision.

        ``pv``, ``load`` and ``temperature`` are the persistence forecast of
        the same steps; what this forecast does not replace is kept.
        """
        n = self.steps
        start = day.project_step_ordinal % n
        absolute = start + np.arange(count, dtype=np.int64)
        steps = absolute % n
        # A step past the calendar's end is in the next project year.
        factor = (1.0 - self.pv_degradation_rate) ** (day.projection_year + absolute // n)
        self.decisions += 1
        if self.kind == "perfect":
            assert self.true_pv_dc_w is not None and self.true_load_w is not None
            assert self.true_temperature_c is not None
            return self.true_pv_dc_w[steps] * factor, self.true_load_w[steps], self.true_temperature_c[steps]
        assert self.pv_dc_by_lead_w is not None
        lead = self.leads(start, count)
        self.lead_steps.update(dict(zip(*(a.tolist() for a in np.unique(lead, return_counts=True)), strict=True)))
        forecast = self.pv_dc_by_lead_w[np.maximum(lead, 1) - 1, steps] * factor
        return np.where(lead == PERSISTENCE_LEAD, np.asarray(pv, dtype=np.float64), forecast), load, temperature

    def record(self) -> dict[str, Any]:
        """A JSON-safe record of the forecast for provenance."""
        record: dict[str, Any] = {"forecast": self.kind, "decisions": self.decisions}
        if self.kind == "weather":
            assert self.pv_dc_by_lead_w is not None
            record.update(
                leads=len(self.pv_dc_by_lead_w),
                files=[dict(f) for f in self.files],
                publication_delay_hours=self.publication_delay_s / _HOUR_S,
                resampling_reach_hours=self.resampling_reach_s / _HOUR_S,
                # Planned steps by lead day; 0 kept persistence.
                planned_steps_by_lead={str(k): int(v) for k, v in sorted(self.lead_steps.items())},
            )
        else:
            record["perfect_information"] = True
        return record
