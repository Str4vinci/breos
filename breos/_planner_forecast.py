"""Planner forecasts other than persistence (ADR 0002 A20).

``daily_persistence`` forecasts the planning window by repeating the day
before the decision. With ``smart_charging.forecast = "weather"`` its PV
forecast comes from day-ahead forecast weather instead: one weather file per
lead day, the first holding for each hour the value forecast at least one day
before it, the second at least two days before, and so on (the layout of the
Open-Meteo Previous Runs archive, :func:`breos.weather.fetch_forecast_archive`).
Each file runs through the run's own PV model, so the forecast PV differs
from the true PV by the weather forecast's error alone.

A decision plans each step on the shortest lead whose forecasts were
published before the decision. A step's PV is shaped by the raw hours up to
``resampling_reach`` after the step's end (the hourly-to-15-minute
resampling interpolates across neighbouring hours), and a forecast of hour
``h`` at lead ``k`` is issued at ``h - k days`` and published
``forecast_publication_delay_hours`` later. So a step takes lead
``ceil((end + reach + delay - decision) / 1 day)``, counted in the weather's
own time: a day the simulated calendar drops from the source year, such as
29 February of a leap-year file restamped onto a common year, still counts.
A step that would need a lead beyond the last file keeps the persistence
value. Load and temperature stay persistence.

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

# How far after a step's end the latest raw hour that shapes its PV can lie,
# by simulation resolution. At 15 minutes the resampler reads neighbouring
# hours (a raw hour moves the steps from 4 h before its label; pinned by a
# test that perturbs single hours); an hourly run reads its own hour.
RESAMPLING_REACH_HOURS = {"15min": 4.0, "h": 0.0}
# A step needing a lead beyond the last file keeps persistence: lead 0.
PERSISTENCE_LEAD = 0
_DAY_S = 86400


def resampling_reach_seconds(freq: str) -> int:
    if freq not in RESAMPLING_REACH_HOURS:
        raise ValueError(f"smart_charging.forecast = 'weather' supports resolutions {sorted(RESAMPLING_REACH_HOURS)}")
    return int(RESAMPLING_REACH_HOURS[freq] * 3600)


def source_offsets(index: pd.DatetimeIndex, source_year: int) -> np.ndarray:
    """Per step of the simulated calendar, the source weather's time less the step's, in seconds.

    The App restamps one year of weather onto the simulated year by whole
    years, dropping 29 February of a leap source year on a common calendar
    (and filling it from 28 February on a leap calendar). A step's source
    time is its date in ``source_year``; a filled 29 February is 28 February.
    """
    utc = index.tz_convert("UTC") if index.tz is not None else index.tz_localize("UTC")
    dates = pd.DatetimeIndex(utc.normalize())
    out = np.empty(len(index), dtype=np.int64)
    for day in dates.unique():
        month, dom = day.month, day.day
        if month == 2 and dom == 29 and not pd.Timestamp(year=source_year, month=1, day=1).is_leap_year:
            dom = 28
        # A filled 29 February repeats the source's 28 February: one day further back.
        source = pd.Timestamp(year=source_year, month=month, day=dom, tz="UTC")
        out[dates == day] = int((source - day).total_seconds())
    return out


@dataclass
class PlannerForecast:
    """What the planner forecasts with in place of persistence, on the run's calendar.

    ``kind`` is ``weather`` (PV from ``pv_dc_by_lead_w``, one row per lead
    day, undegraded) or ``perfect`` (the run's PV, load and temperature).
    ``calendar_epoch_s`` holds each step's start in seconds since the epoch,
    UTC; ``source_offset_s`` its source-time offset (:func:`source_offsets`).
    PV of project year ``y`` (from 0) is degraded by ``(1 - rate) ** y``, as
    the runner degrades the true PV.
    """

    kind: str
    calendar_epoch_s: np.ndarray
    step_s: int
    pv_degradation_rate: float
    source_offset_s: np.ndarray | None = None
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
            if self.source_offset_s is None:
                self.source_offset_s = np.zeros(n, dtype=np.int64)
            if len(self.source_offset_s) != n:
                raise ValueError("source_offset_s needs one value per step")
        elif self.kind == "perfect":
            for name in ("true_pv_dc_w", "true_load_w", "true_temperature_c"):
                value = getattr(self, name)
                if value is None or len(value) != n:
                    raise ValueError(f"a perfect forecast needs {name} with one value per step")
        else:
            raise ValueError(f"unknown planner forecast {self.kind!r}")
        self._year_s = int(self.calendar_epoch_s[-1] - self.calendar_epoch_s[0]) + self.step_s

    @property
    def steps(self) -> int:
        return len(self.calendar_epoch_s)

    def _offset(self, steps: np.ndarray) -> np.ndarray:
        """Source offsets of calendar ``steps``; a step past the calendar's end is its replay, a year on."""
        assert self.source_offset_s is not None
        n = self.steps
        return self.source_offset_s[steps % n] - (steps // n) * self._year_s

    def leads(self, start: int, count: int) -> np.ndarray:
        """The lead day of each of ``count`` steps planned at calendar step ``start``; 0 is persistence.

        ``start`` is the decision's step on the calendar. A backward jump in
        source time (spliced weather) counts calendar time.
        """
        assert self.pv_dc_by_lead_w is not None
        j = np.arange(count, dtype=np.int64)
        steps = start + j
        reach = self.resampling_reach_s // self.step_s + 1
        offset = np.maximum(self._offset(steps), self._offset(steps + reach))
        offset_t = self._offset(np.array([start]))[0]
        age = (j + 1) * self.step_s + self.resampling_reach_s + self.publication_delay_s
        age = age + np.maximum(offset - offset_t, 0)
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
        steps = (start + np.arange(count, dtype=np.int64)) % n
        factor = (1.0 - self.pv_degradation_rate) ** day.projection_year
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
                publication_delay_hours=self.publication_delay_s / 3600,
                resampling_reach_hours=self.resampling_reach_s / 3600,
                # Planned steps by lead day; 0 kept persistence.
                planned_steps_by_lead={str(k): int(v) for k, v in sorted(self.lead_steps.items())},
            )
        else:
            record["perfect_information"] = True
        return record


def calendar_epoch_seconds(index: pd.DatetimeIndex) -> np.ndarray:
    """Each step's start in seconds since the epoch, UTC (a naive index is read as UTC)."""
    utc = index.tz_convert("UTC") if index.tz is not None else index.tz_localize("UTC")
    return np.asarray((utc - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1), dtype=np.int64)
