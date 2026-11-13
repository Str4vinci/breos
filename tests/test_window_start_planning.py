"""Daily-target planning at the start of each charge window (ADR 0002 A19).

``decision_boundary = "charge_window_start"`` makes ``daily_persistence``
decide one target per charge window, at the window's start, held until the
next window starts. These tests cover the window definition, the decision
the controller makes inside a civil day, the weekly cycle, DST, the A2 year
seam, held floors, and provenance. The civil-day default is covered in
``tests/test_daily_persistence.py``.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

import breos._daily_persistence as persistence_module
from breos._controller import ObservedCivilDay, PendingObservedCivilDay
from breos._daily_persistence import (
    WINDOW_POLICY,
    DailyPersistenceController,
    observed_day_before,
    persistence_forecast,
    window_plan_span,
)
from breos._daily_targets import charge_window_day_starts, charge_window_starts
from breos.app_config import resolve_app_config
from breos.dispatch_instructions import DispatchInstructions
from breos.smart_charging import SmartChargingSpec
from breos.tariffs import resolve_tariff
from breos.utils import get_hours_per_step
from tests.test_controller_seam import FREQS, _core, _project, _require, _scenario, _steps_per_day
from tests.test_daily_persistence import ALWAYS, BASE, COARSE, DAILY, TOU, WINDOW, _app, _planner_calls

WINDOWED = {"decision_boundary": "charge_window_start"}
# Off-peak 22:00-07:00 local: every charge window crosses midnight.
SPEC = SmartChargingSpec(
    mode="daily_persistence",
    charge_periods=("off_peak",),
    discharge_periods=("peak", "mid"),
    grid_charge_efficiency=0.95,
    grid_import_limit_w=6000.0,
    target_levels=3,
    soc_states=3,
    decision_boundary="charge_window_start",
)


def _night_scenario(start: str, periods: int, freq: str, zone: str, **kwargs):
    """The seam scenario, relabelled so off-peak runs 22:00-07:00 local."""
    scenario = _scenario(start, periods, freq, zone, **kwargs)
    hour = scenario.pv.index.tz_convert(zone).hour
    labels = np.where((hour < 7) | (hour >= 22), "off_peak", np.where(hour >= 17, "peak", "mid"))
    tariff = scenario.tariff
    resolved = resolve_tariff(scenario.pv.index, list(labels), tariff.schedule, tariff.prices, timezone=zone)
    # A winter load large enough that the planner buys off-peak energy.
    return dataclasses.replace(scenario, tariff=resolved, load=scenario.load + 1500.0)


class _Recording:
    """Wraps a controller, keeping every input and decision, and forwarding its decision starts."""

    def __init__(self, inner: DailyPersistenceController) -> None:
        self.inner = inner
        self.tariff_horizon_days = inner.tariff_horizon_days
        self.days: list = []
        self.decisions: list = []

    def decision_starts(self, tariff):
        return self.inner.decision_starts(tariff)

    def decide_day(self, day, policy_state):
        decision = self.inner.decide_day(day, policy_state)
        self.days.append(day)
        self.decisions.append(decision)
        return decision


def _controller(freq: str, backend: str = "python", spec: SmartChargingSpec = SPEC) -> DailyPersistenceController:
    return DailyPersistenceController(spec, get_hours_per_step(freq), backend)


def _windows(charge: np.ndarray) -> list[slice]:
    """Every maximal run of charge steps."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], charge.astype(np.int8), [0]))))
    return [slice(lo, hi) for lo, hi in zip(edges[::2], edges[1::2], strict=True)]


def _one_target_per_window(executed: DispatchInstructions, charge: np.ndarray) -> list[float]:
    """Each window's single target (NaN for none), asserting it holds for the whole window."""
    targets = []
    for window in _windows(charge):
        values = executed.grid_target_fraction[window]
        assert np.isnan(values).all() or len(set(values.tolist())) == 1, window
        targets.append(float(values[0]))
    return targets


# -- what a window is -------------------------------------------------------------


def test_a_window_is_a_run_of_charge_steps():
    charge = np.array([1, 1, 0, 0, 1, 1, 1, 0, 1], dtype=bool)
    np.testing.assert_array_equal(charge_window_starts(charge), [1, 0, 0, 0, 1, 0, 0, 0, 1])
    # A run that continues from the step before does not start there.
    np.testing.assert_array_equal(charge_window_starts(charge, previous_charge=True), [0, 0, 0, 0, 1, 0, 0, 0, 1])
    assert not charge_window_starts(np.zeros(0, dtype=bool)).any()


def test_adjacent_charge_periods_form_one_window_and_a_day_may_hold_several():
    # Tri-hourly: off-peak 00:00-08:00 and 22:00-24:00, mid-peak 08:00-09:00
    # among others. Charging in off-peak and mid-peak joins 00:00-09:00 into
    # one window; 10:30-18:00 and 20:30-24:00 are two more windows that day.
    index = pd.date_range("2025-01-06", periods=48, freq="30min", tz="Europe/Lisbon")
    wall = index.hour + index.minute / 60.0
    labels = np.select(
        [(wall < 8) | (wall >= 22), ((wall >= 9) & (wall < 10.5)) | ((wall >= 18) & (wall < 20.5))],
        ["off_peak", "peak"],
        "mid_peak",
    )
    charge = np.isin(labels, ("off_peak", "mid_peak"))
    layout = DispatchInstructions(
        discharge_allowed=~charge,
        reserve_fraction=np.zeros(len(index)),
        grid_target_fraction=np.where(charge, 0.5, np.nan),
        grid_charge_efficiency=0.95,
        grid_import_limit_w=np.inf,
    )
    starts = charge_window_day_starts(layout)
    assert [str(index[start].time()) for start in starts[:-1]] == ["00:00:00", "10:30:00", "20:30:00"]
    assert starts[-1] == len(index)


# -- configuration ------------------------------------------------------------------


def test_the_boundary_is_a_daily_persistence_setting():
    assert resolve_app_config({**BASE, "smart_charging": DAILY}).smart_charging.decision_boundary == "civil_day"
    windowed = resolve_app_config({**BASE, "smart_charging": {**DAILY, **WINDOWED}}).smart_charging
    assert windowed.decision_boundary == "charge_window_start"
    fixed = {**DAILY, "mode": "fixed_target", "target_usable_fraction": 0.5, **WINDOWED}
    with pytest.raises(ValueError, match="does not take smart_charging.decision_boundary"):
        resolve_app_config({**BASE, "smart_charging": fixed})
    with pytest.raises(ValueError, match="decision_boundary"):
        resolve_app_config({**BASE, "smart_charging": {**DAILY, "decision_boundary": "hourly"}})
    with pytest.raises(ValueError, match="must be one of: civil_day, charge_window_start"):
        dataclasses.replace(SPEC, decision_boundary="hourly")
    with pytest.raises(ValueError, match="takes no planner settings"):
        SmartChargingSpec(
            mode="fixed_target",
            target_usable_fraction=0.5,
            charge_periods=("off_peak",),
            discharge_periods=("peak",),
            grid_charge_efficiency=0.95,
            decision_boundary="civil_day",
        )


def test_a_window_must_end_somewhere():
    every = {**ALWAYS, "charge_periods": ["off_peak", "peak"], **WINDOWED}
    with pytest.raises(ValueError, match="a charge window never ends"):
        resolve_app_config({**BASE, "smart_charging": every})
    # Under civil days the same table is a valid, if unusual, policy.
    resolve_app_config({**BASE, "smart_charging": {**every, "decision_boundary": "civil_day"}})
    scenario = _night_scenario("2025-01-06T00:00Z", 24, "h", "Europe/Lisbon")
    spec = dataclasses.replace(
        SPEC, charge_periods=("off_peak", "mid", "peak"), discharge_periods=("peak",), overlap_policy="hold_target"
    )
    with pytest.raises(ValueError, match="a charge window never ends"):
        DailyPersistenceController.for_run(spec, scenario.tariff, freq="h", execution_backend="python")


# -- the controller -----------------------------------------------------------------


@pytest.mark.parametrize("freq", FREQS)
def test_each_window_takes_one_target_decided_at_its_start(freq):
    steps_per_day = _steps_per_day(freq)
    scenario = _night_scenario("2025-01-06T00:00Z", 6 * steps_per_day, freq, "Europe/Lisbon")
    recording = _Recording(_controller(freq))
    run = _core(scenario, "python", controller=recording)
    executed = run.controller_instructions
    labels = np.asarray(scenario.tariff.period_labels)
    charge = labels == "off_peak"
    # The calendar ends off-peak, so its first step continues a window.
    window_starts = np.flatnonzero(charge_window_starts(charge, previous_charge=True))

    # A decision at every civil midnight and at every window start; only the
    # window starts decide a target.
    decided = [day.project_step_ordinal for day in recording.days if day.decision_start]
    assert decided == window_starts.tolist()
    midnights = [day.project_step_ordinal for day in recording.days if not day.decision_start]
    assert midnights == list(scenario.tariff.day_starts[:-1])
    targets = _one_target_per_window(executed, charge)
    # The run opens inside a window it did not see start (no target), and the
    # 22:00 window of the first, incomplete-history day is a warm start.
    assert np.isnan(targets[:2]).all()
    assert np.isfinite(targets[2:]).all() and max(targets[2:]) > 0.0

    energy = run.buffers.columns["Battery_Energy"]
    for day in recording.days:
        position = day.project_step_ordinal
        if not day.decision_start or position == 0:
            continue
        # The decision sees the stored energy the step before left, and the
        # day's observations up to it, not after.
        assert day.battery_state.energy_wh == energy[position - 1]
        today = day.observed_today
        midnight = max(start for start in scenario.tariff.day_starts if start <= position)
        assert today.captured_slots == position - midnight
        assert today.pv_dc_w[: position - midnight] == tuple(scenario.pv.iloc[midnight:position])
        assert set(today.pv_dc_w[position - midnight :]) <= {None}


def test_a_window_decision_never_sees_its_own_window():
    scenario = _night_scenario("2025-01-06T00:00Z", 5 * 24, "h", "Europe/Lisbon")
    start = scenario.tariff.day_starts[2] + 22
    pv, load = scenario.pv.copy(), scenario.load.copy()
    pv.iloc[start:] = 0.0
    load.iloc[start:] = load.iloc[start:] + 2000.0
    changed = dataclasses.replace(scenario, pv=pv, load=load)
    seen, seen_changed = _Recording(_controller("h")), _Recording(_controller("h"))
    _core(scenario, "python", controller=seen)
    _core(changed, "python", controller=seen_changed)
    # Every decision up to and including the one at 22:00 on day 2 is the same.
    count = next(n for n, day in enumerate(seen.days) if day.project_step_ordinal == start) + 1
    assert seen.days[:count] == seen_changed.days[:count]
    assert seen.decisions[:count] == seen_changed.decisions[:count]


def test_the_forecast_repeats_the_local_day_before_the_decision(monkeypatch):
    scenario = _night_scenario("2025-01-06T00:00Z", 5 * 24, "h", "Europe/Lisbon")
    recording = _Recording(_controller("h"))
    calls = _planner_calls(monkeypatch)
    _core(scenario, "python", controller=recording)
    planned = [day for day in recording.days if day.decision_start and day.last_complete_observed_day]
    assert len(planned) == len(calls) > 0
    pv = scenario.pv.to_numpy()
    for day, (problem, _kwargs, _plan) in zip(planned, calls, strict=True):
        position = day.project_step_ordinal
        if position + 48 > len(pv):
            # The known horizon ends with the run.
            assert problem.day_starts[-1] == len(pv) - position
            continue
        # Lisbon in January: every local day has 24 hours, so the forecast is
        # the 24 observed hours before the decision, slot for slot.
        np.testing.assert_array_equal(problem.pv_dc_w[:24], pv[position - 24 : position])
        # Two windows of 24 hours, each from one 22:00 to the next.
        assert problem.day_starts == (0, 24, 48)


def test_the_day_before_merges_todays_observations_over_the_last_complete_day():
    keys = tuple((hour * 3600, 0) for hour in range(4))
    last = ObservedCivilDay(0, "2025-01-06", "UTC", keys, (1.0, 2.0, 3.0, 4.0), (1.0,) * 4, (5.0,) * 4)
    today = PendingObservedCivilDay.empty(1, "2025-01-07", "UTC", keys + ((3600, 1),))
    assert observed_day_before(None, today) is None
    assert observed_day_before(last, today) is last
    today = today.with_segment(0, keys[:2], [10.0, 20.0], [7.0, 7.0], [6.0, 6.0])
    merged = observed_day_before(last, today)
    assert merged.pv_dc_w == (10.0, 20.0, 3.0, 4.0) and merged.load_w == (7.0, 7.0, 1.0, 1.0)
    # A slot the last complete day lacks, such as a repeated fall-back hour, is added.
    whole = PendingObservedCivilDay.empty(1, "2025-01-07", "UTC", ((0, 0), (3600, 0), (3600, 1)))
    whole = whole.with_segment(0, ((0, 0), (3600, 0), (3600, 1)), [9.0, 8.0, 7.0], [0.0] * 3, [0.0] * 3)
    merged = observed_day_before(last, whole)
    assert merged.slot_keys == (*keys, (3600, 1)) and merged.pv_dc_w == (9.0, 8.0, 3.0, 4.0, 7.0)
    assert persistence_forecast(merged, ((3600, 1),))[0].tolist() == [7.0]


@pytest.mark.parametrize(("start", "short"), [("2025-03-27T00:00Z", 23), ("2025-10-22T23:00Z", 25)])
def test_windows_keep_their_wall_clock_start_across_dst(monkeypatch, start, short):
    zone = "Europe/Lisbon"
    scenario = _night_scenario(start, 7 * 24, "h", zone)
    recording = _Recording(_controller("h"))
    calls = _planner_calls(monkeypatch)
    run = _core(scenario, "python", controller=recording)
    hours = scenario.pv.index.tz_convert(zone).hour
    decided = [day.project_step_ordinal for day in recording.days if day.decision_start]
    assert set(hours[decided[1:]]) == {22}
    charge = np.asarray(scenario.tariff.period_labels) == "off_peak"
    targets = _one_target_per_window(run.controller_instructions, charge)
    assert np.isfinite(targets[2:]).all()
    lengths = set()
    full = [problem for problem, _kwargs, _plan in calls if problem.day_starts[-1] >= 47]
    assert len(full) >= 4
    for problem in full:
        # Each plan covers two windows, each to the next 22:00 start. Near
        # the end the known horizon ends with the run instead.
        assert problem.n_days == 2
        lengths.update(np.diff(problem.day_starts).tolist())
    assert lengths == {24, short}


def test_the_plan_span_counts_civil_days_from_the_decision_time():
    scenario = _night_scenario("2025-01-06T00:00Z", 6 * 24, "h", "Europe/Lisbon")
    recording = _Recording(_controller("h"))
    _core(scenario, "python", controller=recording)
    day = next(day for day in recording.days if day.decision_start and day.project_step_ordinal > 24)
    layout = persistence_module.period_layout(SPEC, day.tariff.period_labels, 1.0)
    # 22:00 to 22:00 two days on, in two windows.
    assert window_plan_span(day.tariff, layout, 2) == (48, (0, 24, 48))
    assert window_plan_span(day.tariff, layout, 1) == (24, (0, 24))


@pytest.mark.parametrize("zone", ["Europe/Lisbon", "Europe/Berlin"])
def test_a_window_across_the_year_seam_keeps_one_target(zone):
    _require("numba")
    n_steps = 365 * 24
    scenario = _night_scenario("2023-01-01T00:00Z", n_steps, "h", zone)
    recording = _Recording(_controller("h", "numba"))
    run, _years = _project(scenario, "numba", 2, controller=recording)
    executed = run.controller_instructions
    assert len(executed) == 2 * n_steps
    charge = np.tile(np.asarray(scenario.tariff.period_labels) == "off_peak", 2)
    targets = _one_target_per_window(executed, charge)
    # Berlin's first day is partial, so its warm start lasts one window longer.
    assert np.isfinite(targets[2 if zone == "Europe/Lisbon" else 3 :]).all()
    # The window that runs from 31 December 22:00 into the replayed 1 January
    # is one window with one target.
    seam = next(window for window in _windows(charge) if window.start < n_steps < window.stop)
    assert np.isfinite(executed.grid_target_fraction[seam]).all()
    # The calendar's head, which the replay continues, is not a new window.
    assert not recording.inner.decision_starts(scenario.tariff)[0]


# -- App runs --------------------------------------------------------------------------

BIHOURLY_WEEK = {"start": "2025-01-10", "end": "2025-01-16"}


def _executed(app):
    artifacts = app._artifacts
    charge = np.asarray(artifacts.resolved_tariff.period_labels) == "off_peak"
    return artifacts.projection.controller_instructions, charge


def test_the_window_boundary_gives_bihourly_windows_one_target_where_civil_days_split_them():
    config = {**WINDOW, "resolution": "h", "period": BIHOURLY_WEEK}
    civil = _app({**config, "smart_charging": {**DAILY, **COARSE}})
    windowed = _app({**config, "smart_charging": {**DAILY, **COARSE, **WINDOWED}})

    executed, charge = _executed(civil)
    split = [window for window in _windows(charge) if len(set(executed.grid_target_fraction[window].tolist())) > 1]
    # Civil days split a 22:00-08:00 window between two targets.
    assert split
    executed, charge = _executed(windowed)
    targets = _one_target_per_window(executed, charge)
    assert np.isfinite(targets[2:]).all()


@pytest.mark.filterwarnings("ignore:.*execution_backend = 'numba'")
def test_a_held_window_has_one_floor_across_midnight():
    config = {**WINDOW, "resolution": "15min", "period": BIHOURLY_WEEK}
    app = _app({**config, "smart_charging": {**ALWAYS, **COARSE, **WINDOWED}})
    executed, charge = _executed(app)
    _one_target_per_window(executed, charge)
    planned = ~np.isnan(executed.grid_target_fraction)
    assert planned.any()
    # The floor is the window's target, the whole window long, across midnight.
    np.testing.assert_array_equal(executed.reserve_fraction[planned], executed.grid_target_fraction[planned])
    np.testing.assert_array_equal(executed.reserve_fraction[~planned], 0.0)
    assert executed.discharge_allowed.all()


@pytest.mark.filterwarnings("ignore:.*execution_backend = 'numba'")
def test_a_weekly_cycle_weekend_is_one_window():
    # 2026 weekly cycle, winter: Saturday off-peak 00:00-09:30, 13:00-18:30
    # and from 22:00 through Sunday to Monday 07:00; weekdays 00:00-07:00.
    tariff = {**TOU, "schedule": "pt_mainland_2026_weekly_bi"}
    config = {
        **WINDOW,
        "tariff": tariff,
        "resolution": "15min",
        "period": {"start": "2025-01-08", "end": "2025-01-15"},
        "smart_charging": {**DAILY, **COARSE, **WINDOWED},
    }
    app = _app(config)
    executed, charge = _executed(app)
    index = app._artifacts.resolved_tariff.index.tz_convert("Europe/Lisbon")
    windows = _windows(charge)
    starts = [str(index[window.start])[:16] for window in windows]
    assert starts == [
        "2025-01-08 00:00",
        "2025-01-09 00:00",
        "2025-01-10 00:00",
        "2025-01-11 00:00",
        "2025-01-11 13:00",
        "2025-01-11 22:00",
        "2025-01-14 00:00",
    ]
    weekend = windows[5]
    assert str(index[weekend.stop - 1])[:16] == "2025-01-13 06:45"
    targets = _one_target_per_window(executed, charge)
    assert np.isfinite(targets[2:]).all()
    record = app.result()["provenance"]["smart_charging"]
    assert record["decision_boundary"] == "charge_window_start"


def test_provenance_records_the_window_boundary_and_its_versions():
    config = {**WINDOW, "resolution": "h", "period": {"start": "2025-01-10", "end": "2025-01-12"}}
    record = _app({**config, "smart_charging": {**DAILY, **COARSE, **WINDOWED}}).result()["provenance"]
    windowed = record["smart_charging"]
    assert windowed["decision_boundary"] == "charge_window_start"
    assert {key: windowed[key] for key in WINDOW_POLICY} == WINDOW_POLICY
    civil = _app({**config, "smart_charging": {**DAILY, **COARSE}}).result()["provenance"]["smart_charging"]
    explicit = _app({**config, "smart_charging": {**DAILY, **COARSE, "decision_boundary": "civil_day"}})
    explicit_record = explicit.result()["provenance"]["smart_charging"]
    # Naming the default changes nothing.
    assert explicit_record == civil
    assert civil["decision_boundary"] == "civil_day"
    assert (civil["controller_version"], civil["planner_version"]) == ("1", "1")


def test_the_session_refuses_a_decision_at_an_unmarked_step():
    from breos._controller import ControllerBatteryState, _ControllerSession

    scenario = _night_scenario("2025-01-06T00:00Z", 2 * 24, "h", "Europe/Lisbon")
    session = _ControllerSession(
        _controller("h"),
        scenario.tariff,
        scenario.pv.index,
        hours_per_step=1.0,
        carry=None,
        projection_year=0,
        replay_seam=False,
        battery_config={},
    )
    state = ControllerBatteryState(0.0, 0.0, 0.0, 1.0, 1.0, 0.95, 0.95)
    with pytest.raises(RuntimeError, match="civil day 0 begins at step 0, not 5"):
        session.decide(5, state)
