"""Experimental daily-persistence smart charging (ADR 0002 A12).

The controller re-plans every configured-zone civil day through the private
seam of ADR 0002 A11. These tests cover what the mode adds on top of that
seam: its configuration, its causal forecast, the rolling solve it executes,
its refusal outside App, and its provenance. The seam's own dispatch, aging
and year-carry rules are covered in ``tests/test_controller_seam.py``, whose
scenarios these tests reuse.
"""

from __future__ import annotations

import dataclasses
import json
import math
import warnings
from copy import deepcopy
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import breos._daily_persistence as persistence_module
import breos.projection as projection_module
from breos import App, optimization
from breos._controller import ObservedCivilDay, _slot_arrays
from breos._daily_persistence import (
    CONTROLLER_VERSION,
    FORECAST_POLICY,
    PLANNER_TERMINAL_POLICY,
    PLANNER_VERSION,
    WARM_START_POLICY,
    DailyPersistenceController,
    persistence_forecast,
    preserve_start_energy,
)
from breos._daily_targets import DEFAULT_HORIZON_DAYS, DEFAULT_SOC_STATES, DEFAULT_TARGET_LEVELS
from breos._dispatch import _ROW_COLUMNS, lfp_capacity_factor
from breos.app_config import resolve_app_config
from breos.dispatch_instructions import DispatchInstructions
from breos.montecarlo import MonteCarloSettings, build_year_cache, run_montecarlo
from breos.optimization_config import resolve_optimization_config
from breos.result_schema import RESULT_SCHEMA_VERSION
from breos.smart_charging import SmartChargingSpec, resolve_instructions, stored_energy_by_origin
from breos.utils import get_hours_per_step
from tests.test_controller_seam import (
    BACKENDS,
    FREQS,
    _battery,
    _core,
    _project,
    _Recording,
    _require,
    _scenario,
    _steps_per_day,
)
from tools.generate_app_golden import _fake_fetch

# A coarse planner grid keeps the rolling solves cheap; the policy is the same.
SPEC = SmartChargingSpec(
    mode="daily_persistence",
    charge_periods=("off_peak",),
    discharge_periods=("peak", "mid"),
    grid_charge_efficiency=0.95,
    grid_import_limit_w=6000.0,
    target_levels=5,
    soc_states=7,
)
TOU = {
    "schedule": "pt_mainland_2026_daily_bi",
    "currency": "EUR",
    "import_prices": {"peak": 0.28, "off_peak": 0.11},
    "export_prices": {"all": 0.05},
    "fixed_charge_per_day": 0.25,
}
DAILY = {
    "mode": "daily_persistence",
    "charge_periods": ["off_peak"],
    "discharge_periods": ["peak"],
    "grid_charge_efficiency": 0.95,
}
FIXED = {**DAILY, "mode": "fixed_target", "target_usable_fraction": 0.5}
BASE = {
    "location": "porto",
    "n_modules": 8,
    "annual_consumption_kwh": 4000,
    "start_date": "2025-01-01",
    "battery_kwh": 5.0,
    "projection_years": 2,
    "execution_backend": "python",
    "tariff": TOU,
}
# The App tests' coarse grid.
COARSE = {"target_levels": 3, "soc_states": 3}
JUNE = {"start": "2025-06-01", "end": "2025-06-05"}
# Winter days on which the planner buys off-peak energy.
JANUARY = {"start": "2025-01-10", "end": "2025-01-14"}
# A [period] window runs once, so its runs set no projection_years.
WINDOW = {key: value for key, value in BASE.items() if key != "projection_years"}


def _controller(freq: str, backend: str, spec: SmartChargingSpec = SPEC) -> DailyPersistenceController:
    return DailyPersistenceController(spec, get_hours_per_step(freq), backend)


def _without(table, *keys):
    return {key: value for key, value in table.items() if key not in keys}


def _offline():
    return (
        mock.patch("breos.app.fetch_tmy_weather_data", _fake_fetch),
        mock.patch("breos.app.load_weather", lambda **_kwargs: None),
    )


def _app(config) -> App:
    fetch, weather = _offline()
    with fetch, weather:
        app = App(config)
        app.simulate()
    return app


def _planner_calls(monkeypatch):
    """Record every rolling solve: the problem, its keyword arguments, and the plan."""
    calls: list[tuple[object, dict, object]] = []
    solve = persistence_module.solve_daily_targets

    def recording(problem, **kwargs):
        plan = solve(problem, **kwargs)
        calls.append((problem, kwargs, plan))
        return plan

    monkeypatch.setattr(persistence_module, "solve_daily_targets", recording)
    return calls


# -- configuration ------------------------------------------------------------


def test_the_daily_mode_resolves_with_the_planner_defaults():
    spec = resolve_app_config({**BASE, "smart_charging": DAILY}).smart_charging

    assert spec.mode == "daily_persistence" and spec.target_usable_fraction is None
    assert (spec.forecast_horizon_days, spec.target_levels, spec.soc_states) == (
        DEFAULT_HORIZON_DAYS,
        DEFAULT_TARGET_LEVELS,
        DEFAULT_SOC_STATES,
    )
    assert (DEFAULT_HORIZON_DAYS, DEFAULT_TARGET_LEVELS, DEFAULT_SOC_STATES) == (2, 11, 21)
    assert spec.grid_import_limit_w is None
    given = {**DAILY, "forecast_horizon_days": 3, "target_levels": 1, "soc_states": 2, "grid_import_limit_w": 4000}
    spec = resolve_app_config({**BASE, "smart_charging": given}).smart_charging
    assert (spec.forecast_horizon_days, spec.target_levels, spec.soc_states) == (3, 1, 2)
    assert spec.grid_import_limit_w == 4000.0
    # Only the planner mode takes the settings; the others keep None.
    assert resolve_app_config({**BASE, "smart_charging": FIXED}).smart_charging.target_levels is None


@pytest.mark.parametrize(
    ("table", "extra", "error", "message"),
    [
        (
            _without(DAILY, "charge_periods"),
            {},
            ValueError,
            r"'smart_charging' needs smart_charging\.charge_periods for mode = 'daily_persistence'",
        ),
        (
            _without(DAILY, "discharge_periods"),
            {},
            ValueError,
            r"'smart_charging' needs smart_charging\.discharge_periods for mode = 'daily_persistence'",
        ),
        (
            _without(DAILY, "grid_charge_efficiency"),
            {},
            ValueError,
            r"'smart_charging' needs smart_charging\.grid_charge_efficiency for mode = 'daily_persistence'",
        ),
        ({**DAILY, "charge_periods": []}, {}, ValueError, r"'smart_charging\.charge_periods' needs at least 1"),
        (
            {**DAILY, "discharge_periods": ["peak", "off_peak"]},
            {},
            ValueError,
            r"'smart_charging\.charge_periods' and 'smart_charging\.discharge_periods' share off_peak",
        ),
        (
            {**DAILY, "charge_periods": ["night"]},
            {},
            ValueError,
            r"'smart_charging\.charge_periods' has period\(s\) night that schedule 'pt_mainland_2026_daily_bi'",
        ),
        (DAILY, {"tariff": None}, ValueError, r"'smart_charging\.mode' = 'daily_persistence' needs a \[tariff\]"),
        (
            DAILY,
            {"battery_kwh": 0.0},
            ValueError,
            r"'smart_charging\.mode' = 'daily_persistence' needs a battery; set battery_kwh > 0",
        ),
        (
            {**DAILY, "target_usable_fraction": 0.5},
            {},
            ValueError,
            r"'smart_charging\.mode' = 'daily_persistence' does not take smart_charging\.target_usable_fraction: "
            r"the planner chooses each day's target",
        ),
        (
            {**FIXED, "forecast_horizon_days": 2},
            {},
            ValueError,
            r"'smart_charging\.mode' = 'fixed_target' does not take smart_charging\.forecast_horizon_days",
        ),
        (
            {**FIXED, "target_levels": 11, "soc_states": 21},
            {},
            ValueError,
            r"does not take smart_charging\.target_levels, smart_charging\.soc_states",
        ),
        (
            {"mode": "disabled", "soc_states": 21},
            {},
            ValueError,
            r"'smart_charging\.mode' = 'disabled' takes no other keys; remove smart_charging\.soc_states",
        ),
        ({**DAILY, "forecast_horizon_days": True}, {}, TypeError, r"'smart_charging\.forecast_horizon_days' must be"),
        ({**DAILY, "target_levels": 11.0}, {}, TypeError, r"'smart_charging\.target_levels' must be an integer"),
        ({**DAILY, "soc_states": "21"}, {}, TypeError, r"'smart_charging\.soc_states' must be an integer"),
        (
            {**DAILY, "forecast_horizon_days": 0},
            {},
            ValueError,
            r"'smart_charging\.forecast_horizon_days' must be >= 1",
        ),
        ({**DAILY, "target_levels": 0}, {}, ValueError, r"'smart_charging\.target_levels' must be >= 1"),
        ({**DAILY, "soc_states": 1}, {}, ValueError, r"'smart_charging\.soc_states' must be >= 2"),
    ],
)
def test_the_daily_mode_is_checked_before_any_input_is_prepared(table, extra, error, message):
    def no_inputs(*_args, **_kwargs):
        raise AssertionError("the configuration was not checked before inputs were prepared")

    with (
        mock.patch("breos.app.fetch_tmy_weather_data", no_inputs),
        mock.patch("breos.app.load_weather", no_inputs),
        pytest.raises(error, match=message),
    ):
        App({**BASE, **extra, "smart_charging": table})


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"target_usable_fraction": 0.5}, r"not a setting of mode = 'daily_persistence'"),
        ({"soc_states": 1}, r"'smart_charging\.soc_states' must be an integer of at least 2"),
        ({"target_levels": False}, r"'smart_charging\.target_levels' must be an integer of at least 1"),
        ({"forecast_horizon_days": 2.0}, r"'smart_charging\.forecast_horizon_days' must be an integer of at least 1"),
    ],
)
def test_a_daily_spec_built_directly_stays_coherent(fields, message):
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(SPEC, **fields)
    with pytest.raises(ValueError, match=r"takes no planner settings; remove smart_charging\.soc_states"):
        SmartChargingSpec(
            mode="fixed_target",
            target_usable_fraction=0.5,
            charge_periods=("off_peak",),
            discharge_periods=("peak",),
            grid_charge_efficiency=0.9,
            soc_states=3,
        )


def test_the_daily_mode_has_no_static_instructions():
    scenario = _scenario("2024-06-01T00:00Z", 48, "h", "Europe/Berlin")
    with pytest.raises(ValueError, match=r"decides its instructions day by day"):
        resolve_instructions(SPEC, scenario.tariff)
    with pytest.raises(ValueError, match=r"has period\(s\) night"):
        DailyPersistenceController.for_run(
            dataclasses.replace(SPEC, charge_periods=("night",)), scenario.tariff, freq="h", execution_backend="python"
        )


# -- causality and warm start ---------------------------------------------------


@pytest.mark.parametrize("truth", ["pv", "load", "temperature"])
def test_a_day_changes_only_decisions_made_after_it_completes(truth):
    # Berlin summer on a UTC index: civil days start at 22:00 UTC.
    scenario = _scenario("2024-06-01T00:00Z", 6 * 24, "h", "Europe/Berlin")
    starts = scenario.tariff.day_starts
    day = slice(starts[3], starts[4])
    pv, load, temperature = scenario.pv.copy(), scenario.load.copy(), scenario.temperature.copy()
    if truth == "pv":
        pv.iloc[day] = 0.0
    elif truth == "load":
        load.iloc[day] = load.iloc[day] + 1500.0
    else:
        temperature.iloc[day] = temperature.iloc[day] - 35.0
    changed = dataclasses.replace(scenario, pv=pv, load=load, temperature=temperature)

    seen, seen_changed = _Recording(_controller("h", "python")), _Recording(_controller("h", "python"))
    _core(scenario, "python", controller=seen)
    _core(changed, "python", controller=seen_changed)

    # The same completed history, whatever day 3 turns out to be: days 0-3
    # are decided alike, day 3 included.
    assert seen.days[:4] == seen_changed.days[:4]
    assert seen.decisions[:4] == seen_changed.decisions[:4]
    # Day 4 is decided on day 3's completed observations.
    assert seen.days[4].last_complete_observed_day.logical_day_ordinal == 3
    observed, observed_changed = (
        seen.days[4].last_complete_observed_day,
        seen_changed.days[4].last_complete_observed_day,
    )
    assert observed != observed_changed
    assert observed_changed == ObservedCivilDay(
        3,
        observed.local_date,
        observed.timezone,
        observed.slot_keys,
        tuple(pv.iloc[day]),
        tuple(load["load"].iloc[day]),
        tuple(temperature.iloc[day]),
    )


@pytest.mark.parametrize("freq", FREQS)
def test_warm_start_sends_no_grid_target_until_one_complete_local_day(freq):
    steps_per_day = _steps_per_day(freq)
    # Local 12:00 in Berlin: a partial first day, then complete days.
    scenario = _scenario("2024-06-10T10:00Z", 4 * steps_per_day, freq, "Europe/Berlin")
    recording = _Recording(_controller(freq, "python"))
    run = _core(scenario, "python", controller=recording, initial_energy_wh=600.0)
    starts = scenario.tariff.day_starts
    layout = scenario.instructions
    grid = run.buffers.columns["Grid_AC_To_Battery"]

    for number in (0, 1):
        # The partial day counts for nothing, and the first complete day is
        # only observed once it ends.
        assert recording.days[number].complete_days_observed == 0
        planned = recording.decisions[number].instructions
        span = slice(starts[number], starts[number + 1])
        assert np.isnan(planned.grid_target_fraction).all()
        # PV charging and the discharge gate still run as configured.
        assert np.array_equal(planned.discharge_allowed, layout.discharge_allowed[span])
        assert np.array_equal(planned.reserve_fraction, layout.reserve_fraction[span])
    assert grid[: starts[2]].sum() == 0
    assert run.buffers.columns["PV_DC_To_Battery"][: starts[2]].sum() > 0

    first_planned = recording.decisions[2].instructions
    charge = ~np.isnan(layout.grid_target_fraction[starts[2] : starts[3]])
    assert recording.days[2].complete_days_observed == 1
    assert np.isfinite(first_planned.grid_target_fraction[charge]).all()
    assert np.isnan(first_planned.grid_target_fraction[~charge]).all()
    assert grid[starts[2] :].sum() > 0


@pytest.mark.parametrize("freq", FREQS)
def test_a_period_edge_is_never_forecast_history(freq):
    steps_per_day = _steps_per_day(freq)
    half = steps_per_day // 2
    # Local 12:00 on 10 June to 11:00 on 13 June in Berlin, as a standalone
    # span: two complete days between two halves that never join.
    scenario = _scenario("2024-06-10T10:00Z", 3 * steps_per_day, freq, "Europe/Berlin")
    recording = _Recording(_controller(freq, "python"))
    with pytest.MonkeyPatch.context() as patch:
        calls = _planner_calls(patch)
        run = _core(scenario, "python", controller=recording, initial_energy_wh=600.0)

    days = recording.days
    assert [day.complete_days_observed for day in days] == [0, 0, 1, 2]
    assert days[2].last_complete_observed_day.logical_day_ordinal == 1
    assert days[3].last_complete_observed_day.logical_day_ordinal == 2
    # Two solves: the two days decided on complete history.
    assert len(calls) == 2
    last_problem = calls[-1][0]
    # The clipped last day plans on its own half, with no head joined after it.
    assert days[3].clipped_end and days[3].decision_step_count == half
    assert last_problem.day_starts == (0, half) and len(last_problem.instructions) == half
    assert run.controller_carry.pending_day.captured_slots == half
    assert run.controller_carry.complete_days_observed == 2


# -- the rolling solve ----------------------------------------------------------


def _check_rolling_solves(scenario, recording, calls, run, spec=SPEC):
    """Every planned day solved a fresh problem from measured state and executed its first target only."""
    columns = run.buffers.columns
    planned_days = [
        (day, decision)
        for day, decision in zip(recording.days, recording.decisions, strict=True)
        if day.last_complete_observed_day is not None
    ]
    assert len(planned_days) == len(calls) > 0
    for (day, decision), (problem, kwargs, plan) in zip(planned_days, calls, strict=True):
        state = day.battery_state
        position = day.project_step_ordinal
        if position:
            # The measured energy the previous step ended with, after any
            # aging close or replacement on that step.
            assert state.energy_wh == columns["Battery_Energy_End"][position - 1]
        assert kwargs["initial_energy_wh"] == state.energy_wh
        assert (problem.soh_fraction, problem.eff_charge, problem.eff_discharge) == (
            state.soh_fraction,
            state.charge_efficiency,
            state.discharge_efficiency,
        )
        assert kwargs["free_terminal"] is False
        assert (kwargs["target_levels"], kwargs["soc_states"]) == (spec.target_levels, spec.soc_states)
        cap = (
            problem.battery_config.nominal_energy_wh
            * state.soh_fraction
            * problem.battery_config.max_soc
            * lfp_capacity_factor(float(problem.temperature_c[-1]))
        )
        assert kwargs["terminal_energy_wh"] == min(state.energy_wh, cap)
        # The forecast repeats the last complete day on every horizon day.
        forecast = persistence_forecast(day.last_complete_observed_day, day.tariff.slot_keys)
        for believed, expected in zip((problem.pv_dc_w, problem.load_w, problem.temperature_c), forecast, strict=True):
            assert np.array_equal(believed, expected)
        assert problem.day_starts == day.tariff.civil_day_offsets
        assert np.array_equal(problem.import_price_per_kwh, day.tariff.import_price_per_kwh)
        # Only today's target is executed.
        targets = decision.instructions.grid_target_fraction
        charge = ~np.isnan(problem.instructions.grid_target_fraction[: day.decision_step_count])
        assert np.array_equal(targets[charge], np.full(charge.sum(), plan.targets[0]))
        assert np.isnan(targets[~charge]).all()


@pytest.mark.parametrize("freq", FREQS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_an_interior_boundary_replans_from_the_carried_energy(monkeypatch, freq, backend):
    _require(backend)
    steps_per_day = _steps_per_day(freq)
    # Berlin summer on a UTC index: every civil day starts two hours before
    # a positional window closes.
    scenario = _scenario("2024-06-01T00:00Z", 5 * steps_per_day, freq, "Europe/Berlin")
    recording = _Recording(_controller(freq, backend))
    calls = _planner_calls(monkeypatch)
    run = _core(scenario, backend, controller=recording, initial_energy_wh=600.0)

    _check_rolling_solves(scenario, recording, calls, run)
    closes = pd.DataFrame(run.degradation_tracking)
    for day in recording.days[1:]:
        closed = day.project_step_ordinal // steps_per_day
        expected = 1.0 if closed == 0 else closes["SOH"].iloc[closed - 1] / 100.0
        assert day.battery_state.soh_fraction == pytest.approx(expected, rel=1e-15, abs=0.0)
    assert run.buffers.columns["Grid_AC_To_Battery"].sum() > 0


@pytest.mark.parametrize("freq", FREQS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_a_shared_boundary_replans_on_the_replaced_pack(monkeypatch, freq, backend):
    _require(backend)
    steps_per_day = _steps_per_day(freq)
    # UTC days coincide with the positional windows, and the first close
    # replaces a pack this near its end of life.
    scenario = _scenario("2024-06-01T00:00Z", 4 * steps_per_day, freq, "UTC")
    battery = _battery(initial_soh=70.01)
    recording = _Recording(_controller(freq, backend))
    calls = _planner_calls(monkeypatch)
    run = _core(scenario, backend, controller=recording, battery=battery)

    assert np.flatnonzero(run.buffers.replaced).tolist() == [steps_per_day - 1]
    _check_rolling_solves(scenario, recording, calls, run)
    # Day 1 is the first planned day, on the fresh pack after the close.
    problem, kwargs, _plan = calls[0]
    assert recording.days[1].project_step_ordinal == steps_per_day
    assert kwargs["initial_energy_wh"] == battery.nominal_energy_wh * battery.max_soc
    assert problem.soh_fraction == 1.0
    assert (problem.eff_charge, problem.eff_discharge) == (battery.charge_efficiency, battery.discharge_efficiency)


@pytest.mark.parametrize("freq", FREQS)
def test_python_and_numba_plan_and_dispatch_alike(freq):
    _require("numba")
    steps_per_day = _steps_per_day(freq)
    # Berlin across its spring change, a UTC span whose first close replaces
    # the pack, and New York across its fall change; each starts drained.
    for zone, start, soh in (
        ("Europe/Berlin", "2024-03-28T10:00Z", 100.0),
        ("UTC", "2024-06-01T00:00Z", 70.01),
        ("America/New_York", "2024-11-01T15:00Z", 100.0),
    ):
        scenario = _scenario(start, 5 * steps_per_day + 7 * steps_per_day // 24, freq, zone)
        runs = {}
        for backend in BACKENDS:
            recording = _Recording(_controller(freq, backend))
            battery = _battery(initial_soh=soh)
            run = _core(scenario, backend, controller=recording, battery=battery, initial_energy_wh=600.0)
            runs[backend] = (run, recording)
        (python, python_days), (numba, numba_days) = runs["python"], runs["numba"]
        for row, name in enumerate(_ROW_COLUMNS):
            assert np.array_equal(python.buffers.matrix[row], numba.buffers.matrix[row]), (zone, name)
        assert python_days.decisions == numba_days.decisions
        assert python.controller_instructions == numba.controller_instructions
        assert python.aging == numba.aging
        assert python.buffers.columns["Grid_AC_To_Battery"].sum() > 0, zone


def _small_grid(freq: str, backend: str) -> DailyPersistenceController:
    return _controller(freq, backend, dataclasses.replace(SPEC, target_levels=3, soc_states=3))


@pytest.mark.parametrize("zone", ["Europe/Lisbon", "Europe/Berlin"])
@pytest.mark.parametrize("freq", FREQS)
def test_full_utc_years_carry_the_policy_across_the_year_seam(zone, freq):
    backend = "numba"
    _require(backend)
    steps_per_day = _steps_per_day(freq)
    n_steps = 365 * steps_per_day
    scenario = _scenario("2023-01-01T00:00Z", n_steps, freq, zone)
    recording = _Recording(_small_grid(freq, backend))
    run, years = _project(scenario, backend, 2, controller=recording)

    days = recording.days
    assert [day.logical_day_ordinal for day in days] == list(range(len(days)))
    # One warm-start day (two in Berlin, whose first day is partial), then a
    # solve every day, on history that crosses the seam unbroken.
    warm = [day for day in days if day.last_complete_observed_day is None]
    assert len(warm) == (1 if zone == "Europe/Lisbon" else 2)
    for earlier, later in zip(days, days[1:]):
        if later.last_complete_observed_day is not None:
            assert later.last_complete_observed_day.logical_day_ordinal == earlier.logical_day_ordinal
    executed = run.controller_instructions
    assert len(executed) == 2 * n_steps
    # The executed trace is what each year dispatched, in project order.
    year_one, year_two = (carry for _frame, carry in years)
    assert year_two.complete_days_observed == year_one.complete_days_observed + 365
    if zone == "Europe/Berlin":
        tail = steps_per_day // 24
        seam = days[365]
        assert seam.projection_year == 0 and seam.segment_step_count == tail
        assert seam.decision_step_count == steps_per_day
        decided = recording.decisions[365].instructions
        # The seam day's decision runs its last hour's worth at year one's
        # end and the rest at year two's head, from the carried decision.
        assert np.array_equal(
            executed.grid_target_fraction[n_steps - tail : n_steps], decided.grid_target_fraction[:tail], equal_nan=True
        )
        assert np.array_equal(
            executed.grid_target_fraction[n_steps : n_steps + steps_per_day - tail],
            decided.grid_target_fraction[tail:],
            equal_nan=True,
        )
        assert days[366].last_complete_observed_day.logical_day_ordinal == 365


@pytest.mark.parametrize("freq", FREQS)
def test_the_year_seam_plans_alike_on_both_backends(freq):
    _require("numba")
    steps_per_day = _steps_per_day(freq)
    # Berlin, the seam that splits a day: the final 40 days of the year,
    # replayed twice, keep the case small.
    scenario = _scenario("2023-11-22T00:00Z", 40 * steps_per_day, freq, "Europe/Berlin")
    # A winter load large enough that the planner buys off-peak energy.
    scenario = dataclasses.replace(scenario, load=scenario.load + 1500.0)
    runs = {}
    for backend in BACKENDS:
        recording = _Recording(_small_grid(freq, backend))
        run, years = _project(scenario, backend, 2, controller=recording)
        runs[backend] = (run, years, recording)
    (python, python_years, python_days), (numba, numba_years, numba_days) = runs["python"], runs["numba"]
    for (python_frame, _), (numba_frame, _) in zip(python_years, numba_years, strict=True):
        for column in python_frame.columns.drop("Datetime"):
            assert np.array_equal(python_frame[column].to_numpy(), numba_frame[column].to_numpy()), column
    pd.testing.assert_frame_equal(python.yearly_df, numba.yearly_df, check_exact=True)
    assert python_days.decisions == numba_days.decisions
    assert python.controller_instructions == numba.controller_instructions
    assert python.controller_instructions.instruction_hash() == numba.controller_instructions.instruction_hash()
    assert np.nansum(python.controller_instructions.grid_target_fraction) > 0


# -- forecast mapping -----------------------------------------------------------


def _slot_keys(start: str, end: str, freq: str, zone: str):
    """The configured-zone wall slots of UTC steps from ``start`` to ``end``."""
    index = pd.date_range(start, end, freq=freq, inclusive="left", tz="UTC")
    wall, fold = _slot_arrays(index.as_unit("ns").asi8, zone)
    return tuple(zip(wall.tolist(), fold.tolist(), strict=True))


def _observed(slot_keys) -> ObservedCivilDay:
    """A day whose PV is its slot position, load ten times it, and temperature its wall hour."""
    n = len(slot_keys)
    return ObservedCivilDay(
        0,
        "2024-01-01",
        "Europe/Berlin",
        tuple(slot_keys),
        tuple(float(position) for position in range(n)),
        tuple(10.0 * position for position in range(n)),
        tuple(wall / 3600.0 for wall, _fold in slot_keys),
    )


# Berlin local days as UTC spans: 30 March 2024 (24 h), 31 March (23 h,
# spring forward at 02:00), 27 October (25 h, fall back at 03:00).
NORMAL = ("2024-03-29T23:00Z", "2024-03-30T23:00Z")
SPRING = ("2024-03-30T23:00Z", "2024-03-31T22:00Z")
FALL = ("2024-10-26T22:00Z", "2024-10-27T23:00Z")


@pytest.mark.parametrize("freq", FREQS)
def test_a_forecast_maps_slots_by_wall_time_and_fold(freq):
    zone = "Europe/Berlin"
    steps_per_hour = _steps_per_day(freq) // 24
    normal, spring, fall = (_slot_keys(*span, freq, zone) for span in (NORMAL, SPRING, FALL))
    assert (len(normal), len(spring), len(fall)) == tuple(hours * steps_per_hour for hours in (24, 23, 25))

    # Same shape: the observed day comes back as it was.
    observed = _observed(normal)
    for values, expected in zip(
        persistence_forecast(observed, normal), (observed.pv_dc_w, observed.load_w, observed.temperature_c)
    ):
        assert values.tolist() == list(expected)

    # A 24-hour day onto a 25-hour target: the repeated 02:00 hour (fold 1)
    # reuses the observed 02:00 samples; every other slot keeps its wall time.
    pv, load, temperature = persistence_forecast(observed, fall)
    by_wall = {wall: position for position, (wall, _fold) in enumerate(normal)}
    assert pv.tolist() == [float(by_wall[wall]) for wall, _fold in fall]
    assert load.tolist() == [10.0 * by_wall[wall] for wall, _fold in fall]
    assert temperature.tolist() == [wall / 3600.0 for wall, _fold in fall]
    repeated = [position for position, (_wall, fold) in enumerate(fall) if fold]
    assert (
        len(repeated) == steps_per_hour and pv[repeated].tolist() == pv[[p - steps_per_hour for p in repeated]].tolist()
    )

    # A 23-hour day onto a 24-hour target: only the skipped 02:00 hour is
    # interpolated, in wall time, between 01:xx's last slot and 03:00.
    observed = _observed(spring)
    pv, _load, temperature = persistence_forecast(observed, normal)
    spring_by_wall = {wall: position for position, (wall, _fold) in enumerate(spring)}
    step = 3600 // steps_per_hour
    for position, (wall, _fold) in enumerate(normal):
        if wall in spring_by_wall:
            assert pv[position] == spring_by_wall[wall]
            continue
        assert 7200 <= wall < 10800
        left, right = 7200 - step, 10800
        weight = (wall - left) / (right - left)
        expected = spring_by_wall[left] + weight * (spring_by_wall[right] - spring_by_wall[left])
        assert pv[position] == pytest.approx(expected, rel=1e-15)
        assert temperature[position] == pytest.approx(wall / 3600.0, rel=1e-15)

    # A 25-hour day onto a 24-hour target: the first 02:00 hour is kept, its
    # repeat is not used.
    observed = _observed(fall)
    pv, _load, _temperature = persistence_forecast(observed, normal)
    first = {}
    for position, (wall, _fold) in enumerate(fall):
        first.setdefault(wall, position)
    assert pv.tolist() == [float(first[wall]) for wall, _fold in normal]
    # And onto another 25-hour day, both folds match their own samples.
    assert persistence_forecast(observed, fall)[0].tolist() == [float(p) for p in range(len(fall))]


@pytest.mark.parametrize(
    ("start", "zone"),
    [
        ("2024-03-28T10:00Z", "Europe/Berlin"),
        ("2024-10-25T13:00Z", "Europe/Berlin"),
        ("2024-11-01T15:00Z", "America/New_York"),
    ],
)
def test_the_planner_sees_the_resolved_tariff_across_dst_days(monkeypatch, start, zone):
    scenario = _scenario(start, 6 * 24, "h", zone)
    recording = _Recording(_controller("h", "python", dataclasses.replace(SPEC, forecast_horizon_days=3)))
    calls = _planner_calls(monkeypatch)
    _core(scenario, "python", controller=recording, replay_seam=True)

    lengths = set()
    for (day, _decision), (problem, _kwargs, _plan) in zip(
        [(d, x) for d, x in zip(recording.days, recording.decisions, strict=True) if d.last_complete_observed_day],
        calls,
        strict=True,
    ):
        positions = np.asarray(day.tariff.calendar_positions)
        tariff = scenario.tariff
        assert np.array_equal(problem.import_price_per_kwh, np.asarray(tariff.import_price_per_kwh)[positions])
        assert np.array_equal(problem.export_price_per_kwh, np.asarray(tariff.export_price_per_kwh)[positions])
        labels = np.asarray(tariff.period_labels, dtype=object)[positions]
        assert np.array_equal(problem.instructions.discharge_allowed, np.isin(labels, SPEC.discharge_periods))
        assert np.array_equal(~np.isnan(problem.instructions.grid_target_fraction), labels == "off_peak")
        assert problem.day_starts == day.tariff.civil_day_offsets
        lengths.update(np.diff(problem.day_starts).tolist())
        observed = day.last_complete_observed_day
        # Every horizon slot repeats the observed day at its own wall time.
        for key, pv in zip(day.tariff.slot_keys, problem.pv_dc_w, strict=True):
            if key in observed.slot_keys:
                assert pv == observed.pv_dc_w[observed.slot_keys.index(key)]
    # The horizons crossed a 23- or 25-hour day.
    assert lengths & {23, 25}


# -- App only -------------------------------------------------------------------


def test_monte_carlo_refuses_the_daily_mode_before_any_work(tmp_path, write_multiyear_weather):
    weather = write_multiyear_weather(tmp_path / "multi.csv")
    settings = MonteCarloSettings(weather_file=str(weather), n_runs=2, years_per_run=2, seed=3)
    config = {**BASE, "smart_charging": DAILY}

    def no_work(*_args, **_kwargs):
        raise AssertionError("Monte Carlo started work before refusing the mode")

    with (
        mock.patch("breos.montecarlo._load_weather_years", no_work),
        mock.patch("breos.montecarlo.resolve_instructions", no_work),
    ):
        for entry in (run_montecarlo, build_year_cache):
            with pytest.raises(ValueError, match=r"'daily_persistence' is experimental and runs in breos\.App only"):
                entry(config, settings)


def test_the_optimizer_refuses_the_daily_mode_before_any_candidate(monkeypatch):
    index = pd.date_range("2026-01-05", periods=48, freq="h", tz="Europe/Lisbon")
    weather = pd.DataFrame({"temp_air": 20.0}, index=index)
    load = pd.DataFrame({"Load": 1000.0}, index=index)

    def no_work(**_kwargs):
        raise AssertionError("the optimizer simulated a candidate before refusing the mode")

    monkeypatch.setattr(optimization, "calculate_pv_production_dc", no_work)
    config = {
        "location": {"latitude": 41.15, "longitude": -8.61, "timezone": "Europe/Lisbon", "altitude": 100},
        "simulation": {"resolution": "h", "years_projection": 2},
        "battery": {"temperature": 20.0},
        "mode": {"fixed_azimuth": 180.0},
        "constraints": {"budget": 100000, "max_area_m2": 100, "max_tilt_deg": 60, "max_battery_kwh": 10},
        "tariff": {**TOU, "import_prices": {"peak": 0.5, "off_peak": 0.1}},
        "smart_charging": deepcopy(DAILY),
    }
    message = r"'smart_charging\.mode' = 'daily_persistence' is experimental and runs in breos\.App only"
    with pytest.raises(ValueError, match=message):
        resolve_optimization_config(config)
    with pytest.raises(ValueError, match=message):
        optimization.evaluate_projected_design(
            weather, load, config, n_modules=4, battery_kwh=5.0, tilt=30.0, azimuth=180.0
        )
    with pytest.raises(ValueError, match=message):
        optimization.SolarDesignProblem(weather, load, config)


def test_revalue_simulates_again_when_the_planner_prices_change():
    app = _app({**WINDOW, "period": JANUARY, "smart_charging": {**DAILY, **COARSE}})
    fetch, weather = _offline()
    with fetch, weather:
        priced = app.revalue({"tariff": {"import_prices": {"peak": 0.45, "off_peak": 0.06}}})
        exported = app.revalue({"tariff": {"export_prices": {"all": 0.2}}})
        fixed = app.revalue({"tariff": {"fixed_charge_per_day": 0.5}})

    assert priced["provenance"]["revaluation"]["method"] == "resimulated"
    assert exported["provenance"]["revaluation"]["method"] == "resimulated"
    # A fixed charge alone reaches no planner input, so it is re-priced.
    assert fixed["provenance"]["revaluation"]["method"] == "repriced"
    assert fixed["provenance"]["smart_charging"] == app.result()["provenance"]["smart_charging"]
    # Cheaper nights and dearer peaks move the policy's own decisions.
    before = app.result()["provenance"]["smart_charging"]["instruction_hash"]
    assert priced["provenance"]["smart_charging"]["instruction_hash"] != before
    fresh = _app(
        {
            **WINDOW,
            "period": JANUARY,
            "smart_charging": {**DAILY, **COARSE},
            "tariff": {**TOU, "import_prices": {"peak": 0.45, "off_peak": 0.06}},
        }
    )
    assert priced["provenance"]["smart_charging"] == fresh.result()["provenance"]["smart_charging"]


@pytest.mark.parametrize(
    ("backend", "resolution", "warned"),
    [("python", "15min", True), ("python", "h", False), ("numba", "15min", False)],
)
def test_python_at_15_minutes_recommends_numba_once(backend, resolution, warned):
    if backend == "numba":
        _require(backend)
    config = {
        **WINDOW,
        "execution_backend": backend,
        "resolution": resolution,
        "period": {"start": "2025-06-01", "end": "2025-06-02"},
        "smart_charging": {**DAILY, **COARSE},
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _app(config)
    recommended = [item for item in caught if "execution_backend = 'numba'" in str(item.message)]
    assert len(recommended) == (1 if warned else 0)
    if warned:
        assert recommended[0].category is UserWarning


# -- provenance -----------------------------------------------------------------


def test_provenance_identifies_the_policy_and_its_executed_trace(monkeypatch):
    traces: list[DispatchInstructions] = []
    detailed = projection_module._simulate_detailed_run

    def recording(**kwargs):
        run = detailed(**kwargs)
        traces.append(run.controller_instructions)
        return run

    monkeypatch.setattr(projection_module, "_simulate_detailed_run", recording)
    config = {**BASE, "smart_charging": {**DAILY, "grid_import_limit_w": 5000, **COARSE}}
    app = _app(config)
    artifacts = app._artifacts
    result = app.result()
    record = result["provenance"]["smart_charging"]

    assert result["result_schema_version"] == RESULT_SCHEMA_VERSION == "2.7"
    assert set(record) == {
        "mode",
        "overlap_policy",
        "experimental",
        "controller_version",
        "planner_version",
        "charge_periods",
        "discharge_periods",
        "grid_charge_efficiency",
        "grid_import_limit_w",
        "forecast_horizon_days",
        "target_levels",
        "soc_states",
        "forecast_policy",
        "warm_start_policy",
        "planner_terminal_policy",
        "terminal_convention",
        "schedule_hash",
        "instruction_hash",
        "initial_stored_energy",
        "final_stored_energy",
    }
    assert {key: record[key] for key in record if not key.endswith(("_hash", "_stored_energy"))} == {
        "mode": "daily_persistence",
        "overlap_policy": "reject",
        "experimental": True,
        "controller_version": CONTROLLER_VERSION,
        "planner_version": PLANNER_VERSION,
        "charge_periods": ["off_peak"],
        "discharge_periods": ["peak"],
        "grid_charge_efficiency": 0.95,
        "grid_import_limit_w": 5000.0,
        "forecast_horizon_days": 2,
        "target_levels": 3,
        "soc_states": 3,
        "forecast_policy": FORECAST_POLICY,
        "warm_start_policy": WARM_START_POLICY,
        "planner_terminal_policy": PLANNER_TERMINAL_POLICY,
        "terminal_convention": "physical_carry",
    }
    assert (FORECAST_POLICY, WARM_START_POLICY, PLANNER_TERMINAL_POLICY) == (
        "repeat_previous_complete_local_day",
        "no_grid_until_one_complete_local_day",
        "preserve_start_energy",
    )
    assert record["schedule_hash"] == result["provenance"]["tariff"]["schedule_hash"]

    # The hash covers every step both years executed, in project order.
    n_steps = len(artifacts.first_year_results_df)
    assert len(traces) == 2 and all(len(trace) == n_steps for trace in traces)
    executed = artifacts.projection.controller_instructions
    assert len(executed) == 2 * n_steps
    joined = DispatchInstructions(
        discharge_allowed=np.concatenate([trace.discharge_allowed for trace in traces]),
        reserve_fraction=np.concatenate([trace.reserve_fraction for trace in traces]),
        grid_target_fraction=np.concatenate([trace.grid_target_fraction for trace in traces]),
        grid_charge_efficiency=0.95,
        grid_import_limit_w=5000.0,
    )
    assert executed == joined
    assert record["instruction_hash"] == executed.instruction_hash() == joined.instruction_hash()
    first_year = DispatchInstructions(
        discharge_allowed=joined.discharge_allowed[:n_steps],
        reserve_fraction=joined.reserve_fraction[:n_steps],
        grid_target_fraction=joined.grid_target_fraction[:n_steps],
        grid_charge_efficiency=0.95,
        grid_import_limit_w=5000.0,
    )
    assert record["instruction_hash"] != first_year.instruction_hash()
    # Warm start: the first local day has no grid target.
    assert np.isnan(executed.grid_target_fraction[:24]).all()
    assert np.isfinite(executed.grid_target_fraction).any()

    # Stored energy by origin at the project's start and end, unrounded here
    # and rounded in the result's smart_charging block.
    first = artifacts.first_year_results_df.iloc[0]
    carry = artifacts.projection.carry
    initial = stored_energy_by_origin(
        first["Battery_Energy_Beginning"],
        first["Battery_PV_Origin_Energy_Beginning"],
        first["Battery_Grid_Origin_Energy_Beginning"],
    )
    final = stored_energy_by_origin(carry.energy_wh, carry.pv_origin_energy_wh, carry.grid_origin_energy_wh)
    assert record["initial_stored_energy"] == initial and record["final_stored_energy"] == final
    assert set(final) == {"total_wh", "pv_origin_wh", "grid_origin_wh", "unattributed_wh"}
    block = result["smart_charging"]
    assert block["mode"] == "daily_persistence" and block["terminal_convention"] == "physical_carry"
    for key in ("initial_stored_energy", "final_stored_energy"):
        assert block[key] == {name: round(value, 2) + 0.0 for name, value in record[key].items()}
    assert block["yearly"][0]["grid_charge_ac_kwh"] > 0
    # Strict JSON, with no forecast or per-day target in it.
    text = json.dumps(result, allow_nan=False)
    assert "forecast_pv" not in text and "targets" not in json.dumps(record)
    assert math.isfinite(record["final_stored_energy"]["total_wh"])
    # The resolved configuration echoes the table as given.
    assert result["provenance"]["resolved_config"]["smart_charging"] == config["smart_charging"]
    assert artifacts.instructions is None


def test_fixed_target_provenance_is_unchanged_by_the_new_mode():
    app = _app({**WINDOW, "period": JUNE, "smart_charging": FIXED})
    record = app.result()["provenance"]["smart_charging"]
    assert set(record) == {
        "mode",
        "overlap_policy",
        "target_usable_fraction",
        "charge_periods",
        "discharge_periods",
        "grid_charge_efficiency",
        "grid_import_limit_w",
        "instruction_hash",
        "schedule_hash",
        "terminal_convention",
    }


def test_the_terminal_target_is_the_start_energy_capped_at_the_final_forecast_capacity():
    battery = _battery()
    cold, hot = -10.0, 25.0
    full = battery.nominal_energy_wh * 0.9 * battery.max_soc
    assert preserve_start_energy(battery, 1000.0, 0.9, [5.0, hot]) == 1000.0
    assert preserve_start_energy(battery, 10_000.0, 0.9, [hot, cold]) == full * lfp_capacity_factor(cold)
    assert preserve_start_energy(battery, 10_000.0, 0.9, np.array([cold, hot])) == full * lfp_capacity_factor(hot)
