"""ADR 0005 V6: driving, away charging, boundaries and physical carry."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from breos.battery import BatteryConfig, _simulate_core, align_simulation_inputs
from breos.projection import CarryState, ProjectionYear, project_years
from breos.vehicle import VehicleCarry, VehicleSchedule, resolve_vehicle_schedule
from tests.backend_parity import BACKENDS, assert_frames_bit_equal
from tests.energy_conservation import assert_energy_conservation, assert_vehicle_conservation
from tests.test_vehicle_dispatch import vehicle

pytestmark = pytest.mark.usefixtures("vehicle_full_health")

AWAY = dict(target_soc=0.7, power_kw=3.7, efficiency=0.9, price=0.0)


def timeline(index, plugged, first=None, park=None, second=None):
    n = len(index)
    countdown = np.zeros(n, dtype=int)
    for i in range(n - 2, -1, -1):
        if plugged[i]:
            countdown[i] = 1 if not plugged[i + 1] else countdown[i + 1] + (countdown[i + 1] > 0)
    return VehicleSchedule(
        index,
        index.freq.nanos / 3.6e12,
        plugged,
        countdown,
        np.zeros(n) if first is None else first,
        np.zeros(n) if park is None else park,
        np.zeros(n) if second is None else second,
    )


def run(resolved, cfg, backend, *, carry=None, battery=None, pv_w=0.0, load_w=0.0, **kwargs):
    index = resolved.index
    battery = battery or BatteryConfig(nominal_energy_wh=0.0)
    core = _simulate_core(
        pv_dc=pd.Series(pv_w, index=index),
        houseload=pd.DataFrame({"Load": load_w}, index=index),
        temperature_series=pd.Series(25.0, index=index),
        freq="h" if resolved.hours_per_step == 1.0 else "15min",
        battery_config=battery,
        vehicle_config=cfg,
        vehicle_schedule=resolved,
        vehicle_carry=carry,
        execution_backend=backend,
        **kwargs,
    )
    frame = core.buffers.to_frame(core.rng)
    assert_energy_conservation(frame, battery)
    assert_vehicle_conservation(frame, cfg, resolved.hours_per_step, schedule=resolved, initial_carry=carry)
    return core, frame


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("freq", ["h", "15min"])
def test_vh4_driving_legs_away_charging_and_origins(backend, freq):
    index = pd.date_range("2025-06-01", periods=6, freq=freq, tz="UTC")
    hours = index.freq.nanos / 3.6e12
    resolved = timeline(
        index,
        [True, False, False, False, False, True],
        [0, 4500, 0, 0, 0, 0],
        [0, 0, hours, hours, 0, 0],
        [0, 0, 0, 0, 4500, 0],
    )
    core, frame = run(resolved, vehicle(away_charging=AWAY), backend, carry=VehicleCarry(32000.0, 16000.0, 8000.0))
    expected = {
        "Vehicle_Drive_Energy": [4500.0, 0.0, 0.0, 4500.0],
        "Vehicle_Away_Charge_Purchased": [0.0, 500 / 0.9, 0.0, 0.0],
        "Vehicle_Away_Charge_Loss": [0.0, 500 / 0.9 - 500, 0.0, 0.0],
        "Vehicle_Energy_Onboard": [27500.0, 28000.0, 28000.0, 23500.0],
        "Vehicle_PV_Origin_Energy_End": [13750.0, 13750.0, 13750.0, 11540.17857142857],
        "Vehicle_Grid_Origin_Energy_End": [6875.0, 7375.0, 7375.0, 6189.732142857143],
    }
    for column, expected_values in expected.items():
        values = frame[column].iloc[1:5].to_numpy()
        if not column.endswith(("Onboard", "End")):
            values = values * hours
        np.testing.assert_allclose(values, expected_values, rtol=1e-9, atol=1e-9, err_msg=column)
    assert frame["Vehicle_Departure_Energy"].iloc[1] * hours == 32000.0
    assert frame["Vehicle_Arrival_Energy"].iloc[5] * hours == 23500.0
    assert frame["PV_Origin_Vehicle_Drive_Energy"].sum() * hours == pytest.approx(4459.821428571429, rel=1e-9)
    assert (frame["Import_From_Grid"] == 0.0).all()
    assert core.vehicle_carry.energy_wh == 23500.0


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("freq", ["h", "15min"])
@pytest.mark.parametrize("away", [False, True])
def test_vh5_short_departure_is_not_repaired_by_away_charging(backend, freq, away):
    steps = 1 if freq == "h" else 4
    index = pd.date_range("2025-06-01", periods=steps * 2 + 3, freq=freq, tz="UTC")
    n = len(index)
    first, second, park = np.zeros(n), np.zeros(n), np.zeros(n)
    first[steps] = second[-2] = 7500.0
    park[steps + 1 : -2] = 1 / steps
    resolved = timeline(index, [True] * steps + [False] * (steps + 2) + [True], first, park, second)
    cfg = vehicle(away_charging=AWAY if away else None)
    _, frame = run(resolved, cfg, backend, carry=VehicleCarry(10000.0))
    hours = 1 / steps
    assert frame["Vehicle_Energy_Onboard"].iloc[steps - 1] == 16678.5
    assert frame["Vehicle_Energy_Onboard"].iloc[steps] == 9178.5
    assert frame["Vehicle_Departure_Shortfall"].sum() * hours == 15321.5
    expected = {
        "Vehicle_Drive_Energy": 15000.0 if away else 12678.5,
        "Vehicle_Drive_Unserved": 0.0 if away else 2321.5,
        "Vehicle_Away_Charge_Purchased": 3700.0 if away else 0.0,
        "Vehicle_Arrival_Energy": 5008.5 if away else 4000.0,
    }
    for column, wh in expected.items():
        assert frame[column].sum() * hours == pytest.approx(wh, rel=1e-9), column


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("freq", ["h", "15min"])
def test_driving_cannot_borrow_from_charging_later_in_the_step(backend, freq):
    index = pd.date_range("2025-06-01", periods=1, freq=freq, tz="UTC")
    hours = index.freq.nanos / 3.6e12
    resolved = timeline(index, [False], [7000.0], [hours / 2], [600.0])
    _, frame = run(resolved, vehicle(away_charging=AWAY), backend, carry=VehicleCarry(4000.0, 2000.0, 1000.0))
    row = frame.iloc[0]
    second_served = min(600.0, 3700.0 * hours / 2 * 0.9)
    assert row["Vehicle_Drive_Energy"] * hours == second_served
    assert row["Vehicle_Drive_Unserved"] * hours == 7600.0 - second_served
    assert row["Vehicle_Energy_Onboard"] >= 4000.0


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("freq", ["h", "15min"])
@pytest.mark.parametrize(
    "initial,target,park_fraction,stored_wh_per_hour",
    [
        (10000.0, 0.7, 1.0, 3330.0),
        (10000.0, 0.7, 0.25, 832.5),
        (10000.0, 0.7, 0.0, 0.0),
        (27500.0, 0.7, 1.0, None),
        (32000.0, 0.7, 1.0, 0.0),
        (39500.0, 1.0, 1.0, None),
    ],
)
def test_away_stopping_target_power_and_park_hours(backend, freq, initial, target, park_fraction, stored_wh_per_hour):
    index = pd.date_range("2025-06-01", periods=1, freq=freq, tz="UTC")
    hours = index.freq.nanos / 3.6e12
    resolved = timeline(index, [False], park=[hours * park_fraction])
    _, frame = run(resolved, vehicle(away_charging=AWAY | {"target_soc": target}), backend, carry=VehicleCarry(initial))
    expected = 500.0 if stored_wh_per_hour is None else stored_wh_per_hour * hours
    assert frame["Vehicle_Energy_Onboard"].iloc[0] == initial + expected
    assert frame["Vehicle_Away_Charge_Purchased"].iloc[0] * hours == pytest.approx(expected / 0.9)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("freq", ["h", "15min"])
@pytest.mark.parametrize("fraction,reported", [(0.5, False), (2.0, True)])
def test_departure_shortfall_tolerance_before_away_charging(backend, freq, fraction, reported):
    index = pd.date_range("2025-06-01", periods=1, freq=freq, tz="UTC")
    hours = index.freq.nanos / 3.6e12
    cfg = vehicle(away_charging=AWAY | {"target_soc": 0.8})
    _, frame = run(
        timeline(index, [False], park=[hours]), cfg, backend, carry=VehicleCarry(32000.0 - fraction * 1e-9 * 40000.0)
    )
    assert (frame["Vehicle_Departure_Shortfall"].iloc[0] > 0.0) == reported
    assert frame["Vehicle_Energy_Onboard"].iloc[0] == 32000.0


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("freq", ["h", "15min"])
@pytest.mark.parametrize("mode", ["uncontrolled", "v1g", "v2h"])
def test_q4_q5_complete_trips_and_backend_parity(backend, freq, mode):
    index = pd.date_range("2025-06-01", periods=120 if freq == "h" else 480, freq=freq, tz="UTC")
    cfg = vehicle(mode, away_charging=AWAY)
    resolved = resolve_vehicle_schedule(cfg, index, timezone="UTC")
    carry = VehicleCarry(32000.0, 8000.0, 4000.0)
    pv = 5000.0 * np.clip(np.sin((index.hour.to_numpy() - 6) * np.pi / 12), 0.0, None)
    _, frame = run(resolved, cfg, backend, carry=carry, pv_w=pv, load_w=1500.0)
    assert frame["Vehicle_Departure_Energy"].sum() > 0.0
    assert frame["Vehicle_Drive_Energy"].sum() > 0.0
    assert frame["Vehicle_Departure_Shortfall"].sum() == 0.0
    assert frame["Vehicle_Drive_Unserved"].sum() == 0.0
    if backend == "numba":
        _, python = run(resolved, cfg, "python", carry=carry, pv_w=pv, load_w=1500.0)
        assert_frames_bit_equal(python, frame)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("freq", ["h", "15min"])
@pytest.mark.parametrize("stationary", [False, True])
def test_multiday_absence_crosses_day_windows_once(tmp_path, backend, freq, stationary):
    index = pd.date_range("2025-12-29 00:00", "2026-01-03 23:00", freq=freq, tz="UTC")
    path = tmp_path / "trip.csv"
    path.write_text(
        "start,end,kind,energy_kwh,away_charging\n"
        "2025-12-29T23:00Z,2025-12-30T01:00Z,drive,4.5,\n"
        "2025-12-30T01:00Z,2026-01-02T01:00Z,park,,true\n"
        "2026-01-02T01:00Z,2026-01-02T03:00Z,drive,4.5,\n"
    )
    cfg = vehicle(schedule={"file": str(path)}, away_charging=AWAY)
    resolved = resolve_vehicle_schedule(cfg, index, timezone="UTC")
    battery = BatteryConfig(nominal_energy_wh=5000.0 if stationary else 0.0, enable_replacement=False)
    _, frame = run(resolved, cfg, backend, battery=battery, carry=VehicleCarry(32000.0, 16000.0, 8000.0))
    hours = resolved.hours_per_step
    assert np.count_nonzero(frame["Vehicle_Departure_Energy"]) == 1
    assert np.count_nonzero(frame["Vehicle_Arrival_Energy"]) == 1
    assert frame["Vehicle_Drive_Energy"].sum() * hours == pytest.approx(9000.0)
    assert frame["Vehicle_Arrival_Energy"].sum() * hours == pytest.approx(23500.0)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("freq", ["h", "15min"])
@pytest.mark.parametrize("stationary", [False, True])
@pytest.mark.parametrize("summary", [False, True])
def test_project_years_carries_absence_across_the_year_end(tmp_path, backend, freq, stationary, summary):
    # Replay one calendar as project_years does: its tail departs, its head
    # continues that absence. The first project head starts already away.
    index = pd.date_range(
        "2025-01-01", "2025-12-31 23:45" if freq == "15min" else "2025-12-31 23:00", freq=freq, tz="UTC"
    )
    path = tmp_path / "year_trip.csv"
    path.write_text(
        "start,end,kind,energy_kwh,away_charging\n"
        "2025-12-31T23:00Z,2026-01-01T01:00Z,drive,4.5,\n"
        "2026-01-01T01:00Z,2026-01-02T01:00Z,park,,true\n"
        "2026-01-02T01:00Z,2026-01-02T03:00Z,drive,4.5,\n"
    )
    cfg = vehicle("v1g", schedule={"file": str(path)}, away_charging=AWAY)
    resolved = resolve_vehicle_schedule(cfg, index, timezone="UTC")
    pv, load, temp = pd.Series(0.0, index=index), pd.DataFrame({"Load": 0.0}, index=index), pd.Series(25.0, index=index)
    aligned = align_simulation_inputs(pv, load, freq=freq, temperature_series=temp)
    year = (
        ProjectionYear(1.0, aligned=aligned, vehicle_schedule=resolved)
        if summary
        else ProjectionYear(1.0, pv_dc=pv, houseload=load, temperature_series=temp, vehicle_schedule=resolved)
    )
    battery = BatteryConfig(nominal_energy_wh=5000.0 if stationary else 0.0, enable_replacement=False)
    kwargs = dict(
        battery_config=lambda _: battery,
        freq=freq,
        has_battery=stationary,
        execution_backend=backend,
        vehicle_config=cfg,
    )
    combined = project_years(2, lambda _: year, **kwargs)
    first = project_years(1, lambda _: year, **kwargs)
    assert first.carry.vehicle.plugged is False
    # A direct resumed core must continue the first leg, not depart again.
    continued, frame = run(resolved, cfg, backend, carry=first.carry.vehicle)
    assert frame["Vehicle_Departure_Energy"].iloc[0] == 0.0
    arrival = (
        frame.loc[frame["Vehicle_Arrival_Energy"] > 0.0, "Vehicle_Arrival_Energy"].iloc[0] * resolved.hours_per_step
    )
    assert arrival == pytest.approx(23500.0)
    assert combined.carry.vehicle == continued.vehicle_carry
    resumed = project_years(1, lambda _: year, initial_carry=first.carry, **kwargs)
    assert resumed.carry.vehicle == combined.carry.vehicle
    if not summary:
        assert first.first_year_results_df["Vehicle_Departure_Energy"].iloc[0] == 0.0


@pytest.mark.parametrize(
    "column",
    [
        "Vehicle_Departure_Energy",
        "Vehicle_Arrival_Energy",
        "Vehicle_Drive_Unserved",
        "Vehicle_Away_Charge_Purchased",
        "PV_Origin_Vehicle_Drive_Energy",
    ],
)
def test_closure_helper_catches_away_or_boundary_corruption(column):
    index = pd.date_range("2025-06-01", periods=3, freq="h", tz="UTC")
    resolved = timeline(index, [True, False, True], [0.0, 5000.0, 0.0], [0.0, 0.5, 0.0])
    cfg, carry = vehicle(away_charging=AWAY), VehicleCarry(32000.0, 16000.0, 8000.0)
    _, frame = run(resolved, cfg, "python", carry=carry)
    frame.loc[1, column] += 1.0
    with pytest.raises(AssertionError):
        assert_vehicle_conservation(frame, cfg, 1.0, schedule=resolved, initial_carry=carry)


def test_vehicle_carry_validation():
    index = pd.date_range("2025-06-01", periods=1, freq="h", tz="UTC")
    resolved = timeline(index, [False])
    with pytest.raises(ValueError, match="not both"):
        run(resolved, vehicle(), "python", carry=VehicleCarry(32000.0), initial_vehicle_energy_wh=20000.0)
    with pytest.raises(ValueError, match="must be a bool"):
        run(resolved, vehicle(), "python", carry=replace(VehicleCarry(32000.0), plugged="false"))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("freq", ["h", "15min"])
@pytest.mark.parametrize("stationary", [False, True])
@pytest.mark.parametrize("summary", [False, True])
def test_project_years_matches_unbroken_absence_at_the_seam(backend, freq, stationary, summary):
    start = "2025-12-31 20:00" if freq == "h" else "2025-12-31 23:00"
    index = pd.date_range(start, periods=8, freq=freq, tz="UTC")
    hours = index.freq.nanos / 3.6e12
    resolved = timeline(
        index,
        [True, False, False, False, False, False, True, True],
        [0, 4500, 0, 0, 0, 0, 0, 0],
        [0, 0, hours, hours, hours, 0, 0, 0],
        [0, 0, 0, 0, 0, 4500, 0, 0],
    )
    cfg, carry = vehicle(away_charging=AWAY), VehicleCarry(32000.0, 16000.0, 8000.0)
    battery = BatteryConfig(nominal_energy_wh=5000.0 if stationary else 0.0, enable_replacement=False)
    continuous, _ = run(resolved, cfg, backend, battery=battery, carry=carry)

    def year_inputs(year):
        take = slice(year * 4, (year + 1) * 4)
        subindex = index[take]
        schedule = VehicleSchedule(
            subindex,
            hours,
            *(
                getattr(resolved, name)[take]
                for name in ("plugged", "steps_to_departure", "drive_first_wh", "park_hours", "drive_second_wh")
            ),
        )
        pv, load, temp = (
            pd.Series(0.0, index=subindex),
            pd.DataFrame({"Load": 0.0}, index=subindex),
            pd.Series(25.0, index=subindex),
        )
        if summary:
            return ProjectionYear(
                1.0,
                aligned=align_simulation_inputs(pv, load, freq=freq, temperature_series=temp),
                vehicle_schedule=schedule,
            )
        return ProjectionYear(1.0, pv_dc=pv, houseload=load, temperature_series=temp, vehicle_schedule=schedule)

    projected = project_years(
        2,
        year_inputs,
        battery_config=lambda _: battery,
        freq=freq,
        has_battery=stationary,
        execution_backend=backend,
        vehicle_config=cfg,
        initial_carry=CarryState(vehicle=carry),
    )
    assert projected.carry.vehicle == continuous.vehicle_carry
    assert projected.carry.vehicle.energy_wh == 23500.0


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("freq", ["h", "15min"])
def test_arrival_on_first_resumed_step_is_booked_before_home_dispatch(backend, freq):
    index = pd.date_range("2026-01-01", periods=1, freq=freq, tz="UTC")
    carry = VehicleCarry(23500.0, 11540.17857142857, 6189.732142857143, plugged=False)
    resolved = timeline(index, [True])
    cfg = vehicle("uncontrolled")
    core, frame = run(resolved, cfg, backend, carry=carry)
    assert frame["Vehicle_Arrival_Energy"].iloc[0] * resolved.hours_per_step == 23500.0
    assert frame["Grid_AC_To_Vehicle"].iloc[0] == 7400.0
    assert core.vehicle_carry.plugged is True
