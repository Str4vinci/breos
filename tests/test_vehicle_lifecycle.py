"""ADR 0005 V9: independent ageing, same-SOC swaps and physical carry."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from breos.battery import BatteryConfig, _build_simulation_summary, _simulate_core, align_simulation_inputs
from breos.degradation.protocol import DegradationStep
from breos.dispatch_instructions import DispatchInstructions
from breos.projection import CarryState, ProjectionYear, project_years
from breos.vehicle import VehicleCarry, VehicleSchedule, resolve_vehicle_schedule
from tests.backend_parity import BACKENDS, assert_frames_bit_equal
from tests.energy_conservation import assert_energy_conservation, assert_vehicle_conservation
from tests.test_ac_several_inverters import _inputs as multi_inputs
from tests.test_vehicle_away import AWAY, timeline
from tests.test_vehicle_dispatch import VEHICLE_COLUMNS, stationary, vehicle


@pytest.fixture
def scripted(monkeypatch):
    """Control day-close health for exact policy/reference cases, through the real protocol."""
    from breos.vehicle import lifecycle

    instances = []
    health = [0.7]

    class Engine:
        def __init__(self, carry):
            self.health = 1.0 if carry is None else carry.soh_fraction
            self.calendar = 0.0
            self.days = []
            self.settled = 0
            instances.append(self)

        def step(self, day):
            self.days.append(day)
            old = self.health
            self.health = health[min(len(self.days) - 1, len(health) - 1)]
            self.calendar += len(day.soc) * day.step_seconds
            return DegradationStep(self.health, 0.0, self.calendar, 0.0, max(0.0, old - self.health))

        def finalize_cycles(self):
            self.settled += 1
            return DegradationStep(self.health, 0.0, self.calendar, 0.0, 0.0)

        def soh(self):
            return self.health

        def reset(self):
            self.health, self.calendar = 1.0, 0.0

        def snapshot(self, **kwargs):
            return dict(
                degradation_engine="native",
                soh_fraction=self.health,
                fec_cum=0.0,
                cumulative_calendar_seconds=self.calendar,
                day_start_soc_absolute=kwargs["day_start_soc"],
                day_start_temperature_c=kwargs["day_start_temperature_c"],
            )

        def tracking_fields(self, step):
            return {}

    monkeypatch.setattr(lifecycle, "_build_vehicle_lifecycle", lambda cfg, carry, rainflow_kernel=None: Engine(carry))
    return health, instances


def inputs(freq, days=3):
    steps = 24 if freq == "h" else 96
    index = pd.date_range("2025-12-31", periods=days * steps, freq=freq, tz="UTC")
    hour = index.hour.to_numpy() + index.minute.to_numpy() / 60.0
    pv = pd.Series(5000.0 * np.clip(np.sin((hour - 6) * np.pi / 12), 0.0, None), index=index)
    load = pd.DataFrame({"Load": 700.0}, index=index)
    temp = pd.Series(22.0 + np.sin(hour * np.pi / 12), index=index)
    return pv, load, temp


def run(cfg, resolved, backend, *, carry=None, battery=None, pv=None, load=None, temp=None, **kwargs):
    idx, hours = resolved.index, resolved.hours_per_step
    battery = battery or BatteryConfig(nominal_energy_wh=0.0)
    core = _simulate_core(
        pv_dc=pd.Series(0.0, index=idx) if pv is None else pv,
        houseload=pd.DataFrame({"Load": 0.0}, index=idx) if load is None else load,
        temperature_series=pd.Series(25.0, index=idx) if temp is None else temp,
        freq="h" if hours == 1.0 else "15min",
        battery_config=battery,
        vehicle_config=cfg,
        vehicle_schedule=resolved,
        vehicle_carry=carry,
        execution_backend=backend,
        **kwargs,
    )
    frame = core.buffers.to_frame(core.rng)
    assert_energy_conservation(frame, battery)
    assert_vehicle_conservation(frame, cfg, hours, schedule=resolved, initial_carry=carry)
    return core, frame


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("freq", ["h", "15min"])
@pytest.mark.parametrize("plugged", [False, True])
def test_vh6_same_soc_swap_at_home_or_away_and_q7(scripted, backend, freq, plugged):
    idx = pd.date_range("2025-01-01", periods=1, freq=freq, tz="UTC")
    resolved = timeline(idx, [plugged])
    cfg = vehicle(enable_replacement=True)
    carry = VehicleCarry(20000.0, 10000.0, 5000.0, plugged=plugged)
    core, frame = run(cfg, resolved, backend, carry=carry, finalize_degradation=False)
    row = frame.iloc[0]
    hours = resolved.hours_per_step
    assert row["Vehicle_Replacement_Energy_Removed"] * hours == 20000.0
    assert row["Vehicle_Replacement_Energy_Added"] * hours == pytest.approx(28571.428571428572, rel=1e-9)
    assert row["PV_Origin_Vehicle_Replacement_Energy_Removed"] * hours == 10000.0
    assert row["Grid_Origin_Vehicle_Replacement_Energy_Removed"] * hours == 5000.0
    assert row["Vehicle_Capacity_Window_Loss"] == 0.0
    assert row["Vehicle_Energy_Onboard"] == row["Vehicle_Replacement_Energy_Added"] * hours
    assert row["Vehicle_SOC_Absolute"] == pytest.approx(20000.0 / 28000.0)
    assert row["Vehicle_Energy"] == (row["Vehicle_Energy_Onboard"] if plugged else 0.0)
    assert row["Vehicle_SOH"] == 1.0
    assert row["Vehicle_PV_Origin_Energy_End"] == row["Vehicle_Grid_Origin_Energy_End"] == 0.0
    assert core.vehicle_carry.n_replacements == 1
    assert core.vehicle_carry.end_of_life_events[0].soh_pct == 70.0
    assert core.vehicle_end_of_life_events[0].action == "replaced"
    assert frame.attrs["vehicle_end_of_life_events"][0]["step"] == 0
    summary = _build_simulation_summary(core, return_degradation_state=True)
    assert summary.vehicle_carry == core.vehicle_carry
    assert summary.vehicle_end_of_life_events == core.vehicle_end_of_life_events
    assert scripted[1][0].settled == 1
    # Omitting either inventory row must break on-board closure (Q7).
    for name in ("Vehicle_Replacement_Energy_Removed", "Vehicle_Replacement_Energy_Added"):
        corrupt = frame.copy()
        corrupt[name] = 0.0
        with pytest.raises(AssertionError):
            assert_vehicle_conservation(corrupt, cfg, hours, schedule=resolved, initial_carry=carry)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("freq", ["h", "15min"])
@pytest.mark.parametrize("plugged", [False, True])
def test_capacity_window_is_booked_at_day_close_with_proportional_origins(scripted, backend, freq, plugged):
    idx = pd.date_range("2025-01-01", periods=1, freq=freq, tz="UTC")
    cfg = vehicle(enable_replacement=False)
    carry = VehicleCarry(40000.0, 20000.0, 10000.0, plugged=plugged)
    core, frame = run(cfg, timeline(idx, [plugged]), backend, carry=carry)
    hours = 1.0 if freq == "h" else 0.25
    row = frame.iloc[0]
    assert row["Vehicle_Capacity_Window_Loss"] * hours == 12000.0
    assert row["PV_Origin_Vehicle_Capacity_Window_Loss"] * hours == 6000.0
    assert row["Grid_Origin_Vehicle_Capacity_Window_Loss"] * hours == 3000.0
    assert row["Vehicle_Energy_Onboard"] == 28000.0
    assert core.vehicle_carry.soh_fraction == 0.7
    assert row["Vehicle_SOC_Absolute"] == 1.0


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("freq", ["h", "15min"])
@pytest.mark.parametrize("action", ["keep", "retire"])
def test_skipped_swap_keeps_or_retires_and_retired_trips_are_unserved(scripted, backend, freq, action):
    steps = 24 if freq == "h" else 96
    idx = pd.date_range("2025-01-01", periods=steps * 2, freq=freq, tz="UTC")
    first, second, park = np.zeros(len(idx)), np.zeros(len(idx)), np.zeros(len(idx))
    first[steps] = second[-1] = 5000.0
    park[steps + 1 : -1] = 24 / steps
    resolved = timeline(idx, [True] * steps + [False] * steps, first, park, second)
    cfg = vehicle(enable_replacement=False, skipped_replacement_action=action, away_charging=AWAY)
    core, frame = run(cfg, resolved, backend, carry=VehicleCarry(20000.0))
    assert core.vehicle_end_of_life_events[0].action == ("kept" if action == "keep" else "retired")
    assert core.vehicle_end_of_life_events[0].reason == "replacement_disabled"
    assert len(core.vehicle_end_of_life_events) == 1
    assert core.vehicle_carry.retired == (action == "retire")
    if action == "retire":
        assert (frame["Vehicle_Energy_Onboard"].iloc[steps:] == 0.0).all()
        assert frame["Vehicle_Drive_Unserved"].sum() * resolved.hours_per_step == 10000.0
        assert frame["Vehicle_Away_Charge_Purchased"].sum() == 0.0
    else:
        assert frame["Vehicle_Drive_Unserved"].sum() == 0.0


def test_kept_pack_retires_at_zero_health_and_cannot_revive(scripted):
    scripted[0][:] = [0.7, 0.0, 0.0]
    pv, load, temp = inputs("h")
    cfg = vehicle(enable_replacement=False, away_charging=AWAY)
    resolved = resolve_vehicle_schedule(cfg, pv.index, timezone="UTC", reference_year=2025)
    core, frame = run(cfg, resolved, "python", pv=pv, load=load, temp=temp)
    assert [event.reason for event in core.vehicle_end_of_life_events] == ["replacement_disabled", "zero_health"]
    assert core.vehicle_carry.retired
    assert frame["Vehicle_Drive_Unserved"].iloc[48:].sum() > 0.0
    continued, _ = run(replace(cfg, enable_replacement=True), resolved, "python", carry=core.vehicle_carry)
    assert continued.vehicle_carry.retired
    assert continued.vehicle_end_of_life_events == ()
    assert continued.vehicle_carry.n_replacements == 0


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "minimum,terminal,after,actions",
    [
        (0.0, True, 0, ["replaced", "replaced"]),
        (0.0, False, 0, ["replaced", "kept"]),
        (0.75, True, 0, ["kept"]),
        (0.75, True, 2, ["replaced", "replaced"]),
    ],
)
def test_vehicle_reuses_the_horizon_guards(scripted, backend, minimum, terminal, after, actions):
    pv, _, _ = inputs("h", days=2)
    cfg = vehicle(enable_replacement=True, replacement_min_remaining_years=minimum)
    core, _ = run(
        cfg,
        timeline(pv.index, np.ones(len(pv), dtype=bool)),
        backend,
        vehicle_allow_terminal_replacement=terminal,
        vehicle_replacement_years_after_span=after,
    )
    assert [event.action for event in core.vehicle_end_of_life_events] == actions
    if minimum > 0.0 and after == 0:
        assert core.vehicle_end_of_life_events[0].reason == "min_remaining_years"


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("freq", ["h", "15min"])
@pytest.mark.parametrize("engine", ["native", "blast"])
@pytest.mark.filterwarnings("ignore::breos.degradation.validation.BlastExperimentalRangeWarning")
def test_real_onboard_lifecycle_and_backend_parity(backend, freq, engine):
    pv, load, temp = inputs(freq)
    cfg = vehicle(
        enable_replacement=False,
        degradation_engine=engine,
        blast_model="nmc_gr_50ah_b1" if engine == "blast" else None,
        away_charging=AWAY,
    )
    resolved = resolve_vehicle_schedule(cfg, pv.index, timezone="UTC", reference_year=2025)
    core, frame = run(cfg, resolved, backend, pv=pv, load=load, temp=temp)
    assert 0.0 < core.vehicle_carry.soh_fraction < 1.0
    assert core.vehicle_carry.fec_all_packs > 0.0
    assert len(core.vehicle_degradation_tracking) == 3
    assert core.vehicle_carry.degradation_state["degradation_engine"] == engine
    if backend == "numba":
        python, py_frame = run(cfg, resolved, "python", pv=pv, load=load, temp=temp)
        assert_frames_bit_equal(py_frame, frame)
        np.testing.assert_equal(python.vehicle_carry.degradation_state, core.vehicle_carry.degradation_state)
        assert python.vehicle_degradation_tracking == core.vehicle_degradation_tracking


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("freq", ["h", "15min"])
@pytest.mark.parametrize("engine", ["native", "blast"])
@pytest.mark.filterwarnings("ignore::breos.degradation.validation.BlastExperimentalRangeWarning")
def test_year_seam_carries_lifecycle_endpoints_and_rainflow(backend, freq, engine):
    pv, load, temp = inputs(freq, days=2)
    cfg = vehicle(
        enable_replacement=False,
        degradation_engine=engine,
        blast_model="nmc_gr_50ah_b1" if engine == "blast" else None,
        away_charging=AWAY,
    )
    resolved = resolve_vehicle_schedule(cfg, pv.index, timezone="UTC", reference_year=2025)
    continuous, frame = run(cfg, resolved, backend, pv=pv, load=load, temp=temp)
    n = len(pv) // 2

    def slice_schedule(take):
        return VehicleSchedule(
            pv.index[take],
            resolved.hours_per_step,
            *(
                getattr(resolved, name)[take]
                for name in ("plugged", "steps_to_departure", "drive_first_wh", "park_hours", "drive_second_wh")
            ),
        )

    first, left = run(
        cfg,
        slice_schedule(slice(0, n)),
        backend,
        pv=pv.iloc[:n],
        load=load.iloc[:n],
        temp=temp.iloc[:n],
        finalize_degradation=False,
    )
    state = first.vehicle_carry.degradation_state
    assert state["day_start_temperature_c"] == temp.iloc[n - 1]
    if engine == "native":
        assert state["native_rainflow_state"]["current"] is not None
    second, right = run(
        cfg,
        slice_schedule(slice(n, None)),
        backend,
        carry=first.vehicle_carry,
        pv=pv.iloc[n:],
        load=load.iloc[n:],
        temp=temp.iloc[n:],
    )
    assert_frames_bit_equal(frame, pd.concat([left, right], ignore_index=True))
    assert continuous.vehicle_carry.soh_fraction == second.vehicle_carry.soh_fraction
    assert continuous.vehicle_carry.fec_all_packs == second.vehicle_carry.fec_all_packs
    assert (
        continuous.vehicle_degradation_tracking
        == first.vehicle_degradation_tracking + second.vehicle_degradation_tracking
    )


@pytest.mark.parametrize("summary", [False, True])
def test_project_carries_replacement_history_and_allows_earlier_year_end_swap(scripted, summary):
    pv, load, temp = inputs("h", days=1)
    cfg = vehicle(enable_replacement=True)
    resolved = timeline(pv.index, np.ones(len(pv), dtype=bool))
    if summary:
        year = ProjectionYear(
            1.0, aligned=align_simulation_inputs(pv * 0.0, load * 0.0, temp, freq="h"), vehicle_schedule=resolved
        )
    else:
        year = ProjectionYear(
            1.0, pv_dc=pv * 0.0, houseload=load * 0.0, temperature_series=temp, vehicle_schedule=resolved
        )
    projected = project_years(
        2,
        lambda _: year,
        battery_config=lambda _: BatteryConfig(nominal_energy_wh=0.0),
        freq="h",
        has_battery=False,
        execution_backend="python",
        vehicle_config=cfg,
        vehicle_allow_terminal_replacement=False,
        initial_carry=CarryState(vehicle=VehicleCarry(20000.0)),
    )
    assert projected.carry.vehicle.n_replacements == 1
    history = projected.carry.vehicle.end_of_life_events
    assert [event.action for event in history] == ["replaced", "kept"]
    assert [event.step for event in history] == [23, 47]
    assert projected.carry.vehicle.elapsed_steps == 48


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("freq", ["h", "15min"])
@pytest.mark.parametrize("kind", ["dc", "ac", "retired", "none", "several", "several-ac"])
def test_q2_actual_vehicle_ageing_leaves_stationary_state_and_instructions_identical(backend, freq, kind):
    pv, load, temp = inputs(freq)
    units = None
    if kind.startswith("several"):
        pv, load, temp, units = multi_inputs(freq=freq, rich=True)
    battery = stationary(kind)
    instructions = DispatchInstructions.noop(len(pv))
    signature = instructions.instruction_hash()
    kwargs = dict(
        pv_dc=pv,
        houseload=load,
        temperature_series=temp,
        battery_config=battery,
        freq=freq,
        execution_backend=backend,
        dispatch_instructions=instructions,
    )
    if kind == "retired":
        kwargs.update(battery_retired=True, initial_energy_wh=0.0)
    if units is not None:
        kwargs["inverter_units"] = units
    baseline = _simulate_core(**kwargs)
    cfg = vehicle(enable_replacement=True, eol_percentage=0.99999)
    resolved = resolve_vehicle_schedule(cfg, pv.index, timezone="UTC", reference_year=2025)
    with_vehicle = _simulate_core(**kwargs, vehicle_config=cfg, vehicle_schedule=resolved)
    left, right = baseline.buffers.to_frame(baseline.rng), with_vehicle.buffers.to_frame(with_vehicle.rng)
    columns = [
        name for name in left if name not in VEHICLE_COLUMNS and name not in {"Import_From_Grid", "PV_AC_Export"}
    ]
    assert_frames_bit_equal(left[columns], right[columns])
    assert baseline.degradation_tracking == with_vehicle.degradation_tracking
    assert instructions.instruction_hash() == signature
    assert with_vehicle.vehicle_carry.n_replacements > 0
    assert_vehicle_conservation(right, cfg, resolved.hours_per_step, schedule=resolved)


@pytest.mark.parametrize("source", ["weather", "fixed", "csv", "independent"])
@pytest.mark.parametrize("projection_path", [None, "frames", "summary"])
def test_vehicle_temperature_uses_weather_boundary_without_indoor_model(tmp_path, source, projection_path):
    pv, load, temp = inputs("h", days=1)
    temp[:] = -5.0
    cfg_value = "weather"
    extra = {}
    expected = -5.0
    if source == "fixed":
        cfg_value, expected = 35.0, 35.0
    elif source == "csv":
        path = tmp_path / "vehicle_temp.csv"
        pd.DataFrame({"Datetime": pv.index, "Temperature": 15.0}).to_csv(path, index=False)
        cfg_value, expected = str(path), 15.0
    elif source == "independent":
        extra["vehicle_temperature_series"] = pd.Series(30.0, index=pv.index)
        expected = 30.0
    cfg = vehicle(battery_temperature=cfg_value)
    core, _ = run(cfg, timeline(pv.index, np.zeros(len(pv), dtype=bool)), "python", temp=temp, **extra)
    assert core.vehicle_degradation_tracking[0]["Mean_Temperature_C"] == expected
    assert core.vehicle_carry.degradation_state["day_start_temperature_c"] == expected
    if projection_path is not None:
        resolved = timeline(pv.index, np.zeros(len(pv), dtype=bool))
        year_kwargs = dict(
            vehicle_schedule=resolved, vehicle_temperature_series=extra.get("vehicle_temperature_series")
        )
        if projection_path == "summary":
            year_kwargs["aligned"] = align_simulation_inputs(pv * 0.0, load * 0.0, temp, freq="h")
        else:
            year_kwargs.update(pv_dc=pv * 0.0, houseload=load * 0.0, temperature_series=temp)
        projected = project_years(
            1,
            lambda _: ProjectionYear(1.0, **year_kwargs),
            battery_config=lambda _: BatteryConfig(nominal_energy_wh=0.0),
            freq="h",
            has_battery=False,
            execution_backend="python",
            vehicle_config=cfg,
        )
        np.testing.assert_equal(projected.carry.vehicle.degradation_state, core.vehicle_carry.degradation_state)


@pytest.mark.parametrize("control", ["controller", "planner"])
def test_direct_core_refuses_daily_targets_with_vehicle(control):
    pv, load, _ = inputs("h", days=1)
    kwargs = {"day_controller": object()} if control == "controller" else {"instruction_planner": lambda _: None}
    with pytest.raises(ValueError, match="daily_persistence"):
        _simulate_core(
            pv,
            load,
            vehicle_config=vehicle(),
            vehicle_schedule=timeline(pv.index, np.ones(len(pv), dtype=bool)),
            **kwargs,
        )


@pytest.mark.parametrize("energy,expected", [(20000.0, 20000.0), (0.0, 4000.0)])
def test_zero_capacity_swap_uses_last_measured_soc_without_unbooked_energy(scripted, energy, expected):
    scripted[0][:] = [0.0]
    idx = pd.date_range("2025-01-01", periods=1, freq="h", tz="UTC")
    cfg = vehicle(enable_replacement=True)
    seed = VehicleCarry(energy, soh_fraction=1.0 if energy > 0.0 else 0.0)
    core, frame = run(cfg, timeline(idx, [True]), "python", carry=seed)
    assert core.vehicle_carry.energy_wh == expected
    assert frame["Vehicle_Replacement_Energy_Added"].iloc[0] == expected
    assert frame["Vehicle_Capacity_Window_Loss"].iloc[0] == energy
    assert core.vehicle_carry.n_replacements == 1


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("freq", ["h", "15min"])
@pytest.mark.parametrize("summary", [False, True])
@pytest.mark.parametrize("engine", ["native", "blast"])
@pytest.mark.filterwarnings("ignore::breos.degradation.validation.BlastExperimentalRangeWarning")
def test_project_years_preserves_real_lifecycle_in_frame_and_summary_paths(backend, freq, summary, engine):
    pv, load, temp = inputs(freq, days=2)
    cfg = vehicle(
        enable_replacement=False,
        degradation_engine=engine,
        blast_model="nmc_gr_50ah_b1" if engine == "blast" else None,
        away_charging=AWAY,
    )
    resolved = resolve_vehicle_schedule(cfg, pv.index, timezone="UTC", reference_year=2025)
    continuous, _ = run(cfg, resolved, backend, pv=pv, load=load, temp=temp)
    count = len(pv) // 2

    def year_inputs(year):
        take = slice(year * count, (year + 1) * count)
        schedule = VehicleSchedule(
            pv.index[take],
            resolved.hours_per_step,
            *(
                getattr(resolved, name)[take]
                for name in ("plugged", "steps_to_departure", "drive_first_wh", "park_hours", "drive_second_wh")
            ),
        )
        if summary:
            return ProjectionYear(
                1.0,
                aligned=align_simulation_inputs(pv.iloc[take], load.iloc[take], temp.iloc[take], freq=freq),
                vehicle_schedule=schedule,
            )
        return ProjectionYear(
            1.0,
            pv_dc=pv.iloc[take],
            houseload=load.iloc[take],
            temperature_series=temp.iloc[take],
            vehicle_schedule=schedule,
        )

    projected = project_years(
        2,
        year_inputs,
        battery_config=lambda _: BatteryConfig(nominal_energy_wh=0.0),
        freq=freq,
        has_battery=False,
        execution_backend=backend,
        vehicle_config=cfg,
    )
    expected = continuous.vehicle_carry
    actual = projected.carry.vehicle
    assert actual.soh_fraction == expected.soh_fraction
    assert actual.energy_wh == expected.energy_wh
    assert actual.fec_all_packs == expected.fec_all_packs
    assert actual.end_of_life_events == expected.end_of_life_events
    np.testing.assert_equal(actual.degradation_state, expected.degradation_state)
