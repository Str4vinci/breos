"""The private civil-day controller seam (ADR 0002 A11).

Fixed-target smart charging run as a daily controller must be
indistinguishable from the same resolved instructions passed statically,
ledger row for ledger row, while the seam splits dispatch at civil-day
boundaries, keeps degradation on its positional windows, and carries its
decisions and observations across an A2 year seam.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import timedelta
from datetime import timezone as fixed_timezone

import numpy as np
import pandas as pd
import pytest

import breos.projection as projection_module
from breos._controller import ControllerDayDecision, ControllerDayInput
from breos._dispatch import _ROW_COLUMNS
from breos.battery import BatteryConfig, _simulate_core
from breos.dispatch_instructions import DispatchInstructions
from breos.projection import ProjectionYear, project_years
from breos.smart_charging import FixedTargetDayController, SmartChargingSpec, resolve_instructions
from breos.tariffs import ResolvedTariff, TariffPrices, TariffSchedule, resolve_tariff

BACKENDS = ["python", "numba"]
FREQS = ["h", "15min"]
SPEC = SmartChargingSpec(
    mode="fixed_target",
    target_usable_fraction=0.8,
    charge_periods=("off_peak",),
    discharge_periods=("peak", "mid"),
    grid_charge_efficiency=0.95,
    grid_import_limit_w=6000.0,
)


def _require(backend: str) -> None:
    if backend == "numba":
        pytest.importorskip("numba", reason="the compiled backend needs the breos[fast] extra")


def _steps_per_day(freq: str) -> int:
    return 24 if freq == "h" else 96


@dataclass(frozen=True)
class _Scenario:
    freq: str
    pv: pd.Series
    load: pd.DataFrame
    temperature: pd.Series
    tariff: ResolvedTariff
    instructions: DispatchInstructions

    @property
    def n_steps(self) -> int:
        return len(self.pv)


def _scenario(start: str, periods: int, freq: str, zone: str, *, index_tz=None, seed: int = 7) -> _Scenario:
    index = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    if index_tz is not None:
        index = index.tz_convert(index_tz)
    rng = np.random.default_rng(seed)
    utc = index.tz_convert("UTC")
    hour = utc.hour.to_numpy() + utc.minute.to_numpy() / 60.0
    season = 0.6 + 0.4 * np.sin(utc.dayofyear.to_numpy() / 366.0 * np.pi)
    pv = np.maximum(0.0, 4200.0 * np.sin((hour - 6.0) / 12.0 * np.pi)) * season * rng.uniform(0.3, 1.0, periods)
    load = 450.0 + 250.0 * np.cos((hour - 19.0) / 24.0 * 2.0 * np.pi) + rng.uniform(0.0, 300.0, periods)
    temperature = 12.0 + 8.0 * np.sin(utc.dayofyear.to_numpy() / 366.0 * 2.0 * np.pi) + rng.uniform(-3.0, 3.0, periods)
    local_hour = index.tz_convert(zone).hour
    labels = np.where(local_hour < 7, "off_peak", np.where((local_hour >= 17) & (local_hour < 22), "peak", "mid"))
    schedule = TariffSchedule(
        identifier="controller-test", version="1", timezone=zone, cycle="daily", periods=("off_peak", "mid", "peak")
    )
    prices = TariffPrices(
        currency="EUR", import_prices={"off_peak": 0.1, "mid": 0.2, "peak": 0.3}, export_prices={"all": 0.05}
    )
    tariff = resolve_tariff(index, list(labels), schedule, prices, timezone=zone)
    instructions = resolve_instructions(SPEC, tariff)
    assert instructions is not None
    return _Scenario(
        freq=freq,
        pv=pd.Series(pv, index=index),
        load=pd.DataFrame({"load": load}, index=index),
        temperature=pd.Series(temperature, index=index),
        tariff=tariff,
        instructions=instructions,
    )


def _battery(**overrides) -> BatteryConfig:
    values = {
        "nominal_energy_wh": 5000.0,
        "max_charge_power_w": 2500.0,
        "max_discharge_power_w": 2500.0,
        "inverter_ac_capacity_w": 4000.0,
        "enable_resistance_fade": True,
        **overrides,
    }
    return BatteryConfig(**values)


class _Recording:
    """Wraps a controller, keeping every day input and decision it saw."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.tariff_horizon_days = inner.tariff_horizon_days
        self.days: list[ControllerDayInput] = []
        self.decisions: list[ControllerDayDecision] = []

    def decide_day(self, day, policy_state):
        decision = self.inner.decide_day(day, policy_state)
        self.days.append(day)
        self.decisions.append(decision)
        return decision


@dataclass(frozen=True)
class _NoGridUntilObserved:
    """Fixed target, but no grid charging until one complete local day is observed."""

    inner: FixedTargetDayController
    tariff_horizon_days: int = 1

    def decide_day(self, day, policy_state):
        decision = self.inner.decide_day(day, policy_state)
        if day.complete_days_observed:
            return decision
        planned = decision.instructions
        return ControllerDayDecision(
            DispatchInstructions(
                discharge_allowed=planned.discharge_allowed,
                reserve_fraction=planned.reserve_fraction,
                grid_target_fraction=np.full(len(planned), np.nan),
                grid_charge_efficiency=planned.grid_charge_efficiency,
                grid_import_limit_w=planned.grid_import_limit_w,
            ),
            policy_state,
        )


def _core(scenario: _Scenario, backend: str, *, controller=None, battery=None, replay_seam=False, **kwargs):
    return _simulate_core(
        pv_dc=scenario.pv,
        houseload=scenario.load,
        temperature_series=scenario.temperature,
        battery_config=battery or _battery(),
        freq=scenario.freq,
        execution_backend=backend,
        dispatch_instructions=None if controller is not None else scenario.instructions,
        day_controller=controller,
        controller_tariff=scenario.tariff if controller is not None else None,
        replay_seam=replay_seam,
        **kwargs,
    )


def _assert_same_run(static, adapted) -> None:
    for row, name in enumerate(_ROW_COLUMNS):
        assert np.array_equal(static.buffers.matrix[row], adapted.buffers.matrix[row]), name
    assert np.array_equal(static.buffers.replaced, adapted.buffers.replaced)
    assert np.array_equal(static.buffers.replaced_capacity, adapted.buffers.replaced_capacity)
    pd.testing.assert_frame_equal(
        pd.DataFrame(static.degradation_tracking), pd.DataFrame(adapted.degradation_tracking), check_exact=True
    )
    assert static.aging == adapted.aging


# -- fixed target as a controller -------------------------------------------------

# (configured zone, index clock, UTC start): a UTC zone whose civil days
# coincide with the positional windows; Berlin on a UTC index across the
# spring change, on a fixed +01:00 index (the PVGIS clock) across the fall
# change; New York across its fall change. Every span starts and ends mid-day.
CASES = {
    "utc-coincident": ("UTC", None, "2024-03-28T00:00Z"),
    "berlin-spring": ("Europe/Berlin", None, "2024-03-28T10:00Z"),
    "berlin-fall-fixed-offset": ("Europe/Berlin", fixed_timezone(timedelta(hours=1)), "2024-10-25T13:00Z"),
    "new-york-fall": ("America/New_York", None, "2024-11-01T15:00Z"),
}


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("freq", FREQS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_fixed_target_as_a_controller_reproduces_the_static_ledger(case, freq, backend):
    _require(backend)
    zone, index_tz, start = CASES[case]
    steps_per_day = _steps_per_day(freq)
    scenario = _scenario(start, 5 * steps_per_day + 7 * steps_per_day // 24, freq, zone, index_tz=index_tz)
    recording = _Recording(FixedTargetDayController(scenario.instructions))

    static = _core(scenario, backend)
    adapted = _core(scenario, backend, controller=recording)

    _assert_same_run(static, adapted)
    starts = [day.project_step_ordinal for day in recording.days]
    assert starts == list(scenario.tariff.day_starts[:-1])
    assert [day.logical_day_ordinal for day in recording.days] == list(range(len(starts)))
    inside = [start % steps_per_day != 0 for start in starts[1:]]
    if case == "utc-coincident":
        assert not any(inside)
    else:
        # Civil days start inside positional aging windows.
        assert all(inside)
    lengths = {len(day.expected_slot_keys) for day in recording.days}
    if case == "berlin-spring":
        assert {23 * steps_per_day // 24, 24 * steps_per_day // 24} <= lengths
    if case in ("berlin-fall-fixed-offset", "new-york-fall"):
        assert 25 * steps_per_day // 24 in lengths
    if case != "utc-coincident":
        # Partial first and last spans: clipped, never stitched, never complete.
        first, last = recording.days[0], recording.days[-1]
        assert first.civil_slot_offset > 0 and first.clipped_start and not first.initial_partial_day
        assert last.clipped_end and last.decision_step_count == scenario.n_steps - last.project_step_ordinal
        assert adapted.controller_carry.pending_day is not None
        assert adapted.controller_carry.complete_days_observed == len(recording.days) - 2


@pytest.mark.parametrize("freq", FREQS)
def test_the_daily_decisions_concatenate_to_the_resolved_instructions(freq):
    steps_per_day = _steps_per_day(freq)
    scenario = _scenario("2024-03-28T10:00Z", 9 * steps_per_day, freq, "Europe/Berlin")
    recording = _Recording(FixedTargetDayController(scenario.instructions))
    _core(scenario, "python", controller=recording)

    decisions = [decision.instructions for decision in recording.decisions]
    for day, decision in zip(recording.days, decisions, strict=True):
        positions = np.arange(day.project_step_ordinal, day.project_step_ordinal + len(decision))
        assert np.array_equal(decision.discharge_allowed, scenario.instructions.discharge_allowed[positions])
    joined = DispatchInstructions(
        discharge_allowed=np.concatenate([d.discharge_allowed for d in decisions]),
        reserve_fraction=np.concatenate([d.reserve_fraction for d in decisions]),
        grid_target_fraction=np.concatenate([d.grid_target_fraction for d in decisions]),
        grid_charge_efficiency=decisions[0].grid_charge_efficiency,
        grid_import_limit_w=decisions[0].grid_import_limit_w,
    )
    assert all(
        (d.grid_charge_efficiency, d.grid_import_limit_w) == (joined.grid_charge_efficiency, joined.grid_import_limit_w)
        for d in decisions
    )
    assert joined.instruction_hash() == scenario.instructions.instruction_hash()


# -- dispatch and aging order -----------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_civil_boundary_inside_an_aging_window_sees_carried_state_and_unchanged_health(backend):
    _require(backend)
    # Berlin summer on a UTC index: local midnight is 22:00 UTC, two steps
    # before each positional window closes.
    scenario = _scenario("2024-06-01T00:00Z", 4 * 24, "h", "Europe/Berlin")
    recording = _Recording(FixedTargetDayController(scenario.instructions))
    run = _core(scenario, backend, controller=recording)
    columns = run.buffers.columns
    closes = pd.DataFrame(run.degradation_tracking)

    assert [day.project_step_ordinal for day in recording.days] == [0, 22, 46, 70, 94]
    for day in recording.days[1:]:
        before = day.project_step_ordinal - 1
        state = day.battery_state
        assert state.energy_wh == columns["Battery_Energy_End"][before]
        assert state.pv_origin_energy_wh == columns["Battery_PV_Origin_Energy_End"][before]
        assert state.grid_origin_energy_wh == columns["Battery_Grid_Origin_Energy_End"][before]
        # Health is the last close's, not the close due two steps later
        # (the frame keeps SOH in percent, hence the one-ulp tolerance).
        closed = day.project_step_ordinal // 24
        expected_soh = 1.0 if closed == 0 else closes["SOH"].iloc[closed - 1] / 100.0
        assert state.soh_fraction == pytest.approx(expected_soh, rel=1e-15, abs=0.0)
        assert state.soh_fraction != pytest.approx(closes["SOH"].iloc[closed] / 100.0, rel=1e-12, abs=0.0)
    # One degradation close per positional window, at its last step.
    assert list(closes["Datetime"]) == list(scenario.pv.index[[23, 47, 71, 95]])


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_shared_boundary_decides_after_aging_and_replacement(backend):
    _require(backend)
    # UTC days coincide with the positional windows, and a pack this close to
    # end of life is replaced at the first window's closing row.
    scenario = _scenario("2024-06-01T00:00Z", 3 * 24, "h", "UTC")
    battery = _battery(initial_soh=70.01)
    recording = _Recording(FixedTargetDayController(scenario.instructions))
    static = _core(scenario, backend, battery=_battery(initial_soh=70.01))
    adapted = _core(scenario, backend, controller=recording, battery=battery)

    _assert_same_run(static, adapted)
    replaced = np.flatnonzero(adapted.buffers.replaced)
    assert replaced.tolist() == [23]
    day = recording.days[1]
    assert day.project_step_ordinal == 24
    state = day.battery_state
    fresh = battery.nominal_energy_wh * battery.max_soc
    assert state.energy_wh == fresh == adapted.buffers.columns["Battery_Energy_End"][23]
    assert state.pv_origin_energy_wh == 0.0 and state.grid_origin_energy_wh == 0.0
    assert state.soh_fraction == 1.0 and state.resistance_growth == 0.0
    assert (state.charge_efficiency, state.discharge_efficiency) == (
        battery.charge_efficiency,
        battery.discharge_efficiency,
    )
    # The retired pack's day is still observed, complete, after the swap.
    assert day.complete_days_observed == 1
    assert day.last_complete_observed_day.pv_dc_w == tuple(scenario.pv.iloc[:24])


# -- the A2 year seam -------------------------------------------------------------


def _project(scenario: _Scenario, backend: str, years: int, *, controller=None, replay_seam=True):
    """Run ``years`` replayed years, returning the run and every year's frame and controller carry."""
    per_year: list[tuple[pd.DataFrame, object]] = []
    detailed, balance = projection_module._simulate_detailed_run, projection_module.simulate_energy_balance

    def recording_detailed(**kwargs):
        run = detailed(**kwargs)
        per_year.append((run.results_df, run.controller_carry))
        return run

    def recording_balance(**kwargs):
        result = balance(**kwargs)
        per_year.append((result[0], None))
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(projection_module, "_simulate_detailed_run", recording_detailed)
        patch.setattr(projection_module, "simulate_energy_balance", recording_balance)
        run = project_years(
            years,
            lambda _year: ProjectionYear(
                1.0, pv_dc=scenario.pv, houseload=scenario.load, temperature_series=scenario.temperature
            ),
            battery_config=lambda soh: _battery(initial_soh=soh),
            freq=scenario.freq,
            has_battery=True,
            execution_backend=backend,
            tariff=scenario.tariff,
            instructions=None if controller is not None else scenario.instructions,
            day_controller=controller,
            replay_seam=replay_seam,
        )
    return run, per_year


@pytest.mark.parametrize("zone", ["Europe/Lisbon", "Europe/Berlin"])
@pytest.mark.parametrize("freq", FREQS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_full_utc_years_carry_the_controller_across_the_year_seam(zone, freq, backend):
    _require(backend)
    steps_per_day = _steps_per_day(freq)
    n_steps = 365 * steps_per_day
    scenario = _scenario("2023-01-01T00:00Z", n_steps, freq, zone)
    recording = _Recording(FixedTargetDayController(scenario.instructions))

    static, static_years = _project(scenario, backend, 2)
    adapted, adapted_years = _project(scenario, backend, 2, controller=recording)

    for (static_frame, _), (adapted_frame, _) in zip(static_years, adapted_years, strict=True):
        for column in static_frame.columns.drop("Datetime"):
            assert np.array_equal(static_frame[column].to_numpy(), adapted_frame[column].to_numpy()), column
    pd.testing.assert_frame_equal(static.yearly_df, adapted.yearly_df, check_exact=True)
    assert dataclasses.replace(adapted.carry, controller_carry=None) == static.carry

    days = recording.days
    ordinals = [day.logical_day_ordinal for day in days]
    steps = [day.project_step_ordinal for day in days]
    # One invocation per logical day, on a clock that never restarts.
    assert ordinals == list(range(len(days)))
    assert all(left < right for left, right in zip(steps, steps[1:]))
    first_year, second_year = adapted_years[0][1], adapted_years[1][1]
    year_two = [day for day in days if day.projection_year == 1]
    assert all(day.project_step_ordinal >= n_steps for day in year_two)

    if zone == "Europe/Lisbon":
        # UTC is Lisbon's winter clock: the seam is a local midnight.
        assert len(days) == 730 and not days[0].initial_partial_day
        assert first_year.complete_days_observed == 365 and second_year.complete_days_observed == 730
        assert first_year.pending_day is None and first_year.active_day_decision is None
        assert year_two[0].logical_day_ordinal == 365 and year_two[0].project_step_ordinal == n_steps
        return

    # Berlin: the year ends at local 1 January 00:00 and the replay resumes at
    # 01:00, so 1 + 23 hours (4 + 92 quarter-hours) are one logical day.
    tail = steps_per_day // 24
    assert days[0].initial_partial_day and days[0].civil_slot_offset == tail
    assert days[0].decision_step_count == steps_per_day - tail
    assert len(days) == 731
    seam = days[365]
    assert (seam.projection_year, seam.project_step_ordinal) == (0, n_steps - tail)
    assert seam.decision_step_count == steps_per_day and seam.segment_step_count == tail
    assert seam.civil_slot_offset == 0 and not (seam.clipped_start or seam.clipped_end)
    assert seam.tariff.calendar_positions == (
        *range(n_steps - tail, n_steps),
        *range(steps_per_day - tail),
    )
    assert first_year.complete_days_observed == 364
    assert first_year.pending_day.logical_day_ordinal == 365
    assert first_year.pending_day.captured_slots == tail
    carried = first_year.active_day_decision
    assert carried.logical_day_ordinal == 365 and carried.next_slot_offset == tail
    assert len(carried.instructions) == steps_per_day
    # The carried decision is consumed without asking again: the next
    # invocation is 2 January, after the head's 23 hours.
    assert year_two[0].logical_day_ordinal == 366
    assert year_two[0].project_step_ordinal == n_steps + steps_per_day - tail
    observed = year_two[0].last_complete_observed_day
    assert observed.logical_day_ordinal == 365 and len(observed.slot_keys) == steps_per_day
    pv = scenario.pv.to_numpy()
    assert observed.pv_dc_w == (*pv[n_steps - tail :], *pv[: steps_per_day - tail])
    assert year_two[0].complete_days_observed == 365
    assert second_year.complete_days_observed == 729


@pytest.mark.parametrize("freq", FREQS)
def test_a_standalone_period_is_never_stitched(freq):
    steps_per_day = _steps_per_day(freq)
    half = steps_per_day // 2
    # Local 12:00 on 10 June to 11:00 on 13 June in Berlin: the start and the
    # end are the two halves of one wall-clock day, but not of one day.
    scenario = _scenario("2024-06-10T10:00Z", 3 * steps_per_day, freq, "Europe/Berlin")
    standalone = _Recording(FixedTargetDayController(scenario.instructions))
    replayed = _Recording(FixedTargetDayController(scenario.instructions))
    static = _core(scenario, "python")
    period = _core(scenario, "python", controller=standalone)
    replay = _core(scenario, "python", controller=replayed, replay_seam=True)

    _assert_same_run(static, period)
    _assert_same_run(static, replay)
    first, last = standalone.days[0], standalone.days[-1]
    assert first.clipped_start and not first.initial_partial_day and first.decision_step_count == half
    assert last.clipped_end and last.decision_step_count == half
    assert last.tariff.calendar_positions == tuple(range(scenario.n_steps - half, scenario.n_steps))
    carry = period.controller_carry
    assert carry.complete_days_observed == 2 and carry.last_complete_observed_day.logical_day_ordinal == 2
    assert carry.pending_day.logical_day_ordinal == 3 and carry.pending_day.captured_slots == half
    assert carry.active_day_decision is None
    # The same span as a replayed year would stitch its end to its head.
    assert replayed.days[-1].decision_step_count == steps_per_day
    assert replay.controller_carry.active_day_decision.next_slot_offset == half


@pytest.mark.parametrize("periods_in_days", [1, 3])
def test_warm_start_waits_for_one_complete_local_day(periods_in_days):
    scenario = _scenario("2024-06-10T10:00Z", periods_in_days * 24, "h", "Europe/Berlin")
    drained = {"initial_energy_wh": 600.0}
    plain = _core(scenario, "python", controller=FixedTargetDayController(scenario.instructions), **drained)
    warm = _core(
        scenario, "python", controller=_NoGridUntilObserved(FixedTargetDayController(scenario.instructions)), **drained
    )

    plain_grid = plain.buffers.columns["Grid_AC_To_Battery"]
    warm_grid = warm.buffers.columns["Grid_AC_To_Battery"]
    first_complete_end = scenario.tariff.day_starts[2]
    assert plain_grid[:first_complete_end].sum() > 0
    assert warm_grid[:first_complete_end].sum() == 0
    if periods_in_days == 1:
        # Two partial days and no complete one: never any grid charging.
        assert warm.controller_carry.complete_days_observed == 0
        assert warm_grid.sum() == 0
    else:
        assert warm_grid[first_complete_end:].sum() > 0


# -- causality --------------------------------------------------------------------


def _arrays_in(value, path="day"):
    if isinstance(value, np.ndarray):
        yield path
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for item in dataclasses.fields(value):
            yield from _arrays_in(getattr(value, item.name), f"{path}.{item.name}")
    elif isinstance(value, (tuple, list)):
        for position, item in enumerate(value):
            yield from _arrays_in(item, f"{path}[{position}]")
    elif hasattr(value, "items"):
        for key, item in value.items():
            yield from _arrays_in(item, f"{path}[{key!r}]")


def test_the_controller_sees_no_current_or_future_truth():
    fields = {item.name for item in dataclasses.fields(ControllerDayInput)}
    assert not {name for name in fields if any(word in name for word in ("pv", "load", "temperature"))}

    steps_per_day = 24
    scenario = _scenario("2024-06-01T00:00Z", 6 * steps_per_day, "h", "Europe/Berlin")
    changed_from = scenario.tariff.day_starts[3]
    future = slice(changed_from, None)
    pv, load, temperature = scenario.pv.copy(), scenario.load.copy(), scenario.temperature.copy()
    pv.iloc[future] = pv.iloc[future] * 0.2
    load.iloc[future] = load.iloc[future] + 900.0
    temperature.iloc[future] = temperature.iloc[future] - 15.0
    changed = dataclasses.replace(scenario, pv=pv, load=load, temperature=temperature)

    seen, seen_changed = (
        _Recording(FixedTargetDayController(scenario.instructions)),
        _Recording(FixedTargetDayController(scenario.instructions)),
    )
    _core(scenario, "python", controller=seen)
    _core(changed, "python", controller=seen_changed)

    for day in seen.days:
        assert list(_arrays_in(day)) == []
        observed = day.last_complete_observed_day
        if observed is not None:
            assert observed.logical_day_ordinal < day.logical_day_ordinal
            lo = scenario.tariff.day_starts[observed.logical_day_ordinal]
            assert lo + len(observed.pv_dc_w) <= day.project_step_ordinal
            assert observed.pv_dc_w == tuple(scenario.pv.iloc[lo : lo + len(observed.pv_dc_w)])
            assert observed.load_w == tuple(scenario.load["load"].iloc[lo : lo + len(observed.load_w)])
            assert observed.temperature_c == tuple(scenario.temperature.iloc[lo : lo + len(observed.temperature_c)])
    # Changing truth from day 3 on leaves every decision up to day 3 alone;
    # day 4 is the first to observe it.
    assert seen.days[:4] == seen_changed.days[:4]
    assert seen.days[4] != seen_changed.days[4]


# -- guards -----------------------------------------------------------------------


def test_the_seam_refuses_what_it_cannot_run():
    scenario = _scenario("2024-06-01T00:00Z", 48, "h", "Europe/Berlin")
    controller = FixedTargetDayController(scenario.instructions)
    common = {
        "pv_dc": scenario.pv,
        "houseload": scenario.load,
        "temperature_series": scenario.temperature,
        "day_controller": controller,
        "controller_tariff": scenario.tariff,
    }
    with pytest.raises(ValueError, match="either dispatch_instructions or a daily controller"):
        _simulate_core(battery_config=_battery(), dispatch_instructions=scenario.instructions, **common)
    with pytest.raises(ValueError, match="needs a battery"):
        _simulate_core(battery_config=BatteryConfig(nominal_energy_wh=0), **common)
    with pytest.raises(ValueError, match="needs the resolved tariff"):
        _simulate_core(battery_config=_battery(), **{**common, "controller_tariff": None})
    other = _scenario("2024-07-01T00:00Z", 48, "h", "Europe/Berlin")
    with pytest.raises(ValueError, match="resolved on the simulated index"):
        _simulate_core(battery_config=_battery(), **{**common, "controller_tariff": other.tariff})

    @dataclass(frozen=True)
    class _Short:
        tariff_horizon_days: int = 1

        def decide_day(self, day, policy_state):
            return ControllerDayDecision(DispatchInstructions.noop(day.decision_step_count - 1))

    with pytest.raises(ValueError, match="decision span has"):
        _simulate_core(battery_config=_battery(), **{**common, "day_controller": _Short()})

    @dataclass(frozen=True)
    class _Drifting:
        tariff_horizon_days: int = 1

        def decide_day(self, day, policy_state):
            efficiency = 0.9 if day.logical_day_ordinal else 0.95
            planned = DispatchInstructions.noop(day.decision_step_count)
            return ControllerDayDecision(dataclasses.replace(planned, grid_charge_efficiency=efficiency))

    with pytest.raises(ValueError, match="must keep grid_charge_efficiency"):
        _simulate_core(battery_config=_battery(), **{**common, "day_controller": _Drifting()})
