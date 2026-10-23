"""Vehicle trips replay the explicit reference calendar, on either side of UTC."""

import numpy as np
import pandas as pd
import pytest

from breos.montecarlo import MonteCarloSettings, run_montecarlo
from tests.backend_parity import BACKENDS
from tests.offline_app import BASE, simulate_offline
from tests.test_vehicle_config import VEHICLE


def calendar_config(year, zone, freq, **extra):
    return {
        **BASE,
        "start_date": f"{year}-01-01",
        "location": {"latitude": 41.15, "longitude": -8.61, "altitude": 100, "timezone": zone},
        "resolution": freq,
        "battery_kwh": 0,
        "projection_years": 2,
        "vehicle": {**VEHICLE, "mode": "uncontrolled", "reference_mode": "none"},
        **extra,
    }


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("freq", ["h", "15min"])
@pytest.mark.parametrize("zone,year,depart", [("America/New_York", 2022, "17:00"), ("Europe/Berlin", 2023, "00:00")])
def test_template_seam_conserves_each_departures_trip_and_annual_driving(backend, freq, zone, year, depart):
    cfg = calendar_config(year, zone, freq, execution_backend=backend)
    cfg["vehicle"]["schedule"] = {
        "weekday": {"depart": depart, "arrive": "06:00", "trip_kwh": 2},
        "weekend": {"depart": depart, "arrive": "06:00", "trip_kwh": 20},
    }
    app = simulate_offline(cfg)
    schedule = app._artifacts.vehicle_households.schedule
    local = schedule.index.tz_convert(zone)
    departures = np.flatnonzero(schedule.drive_first_wh)
    returns = np.flatnonzero(schedule.drive_second_wh)
    # Pair every departure with its return, wrapping at the replay boundary.
    # In New York this includes the reported Saturday 31 December trip.
    assert len(departures) == len(returns) == 365
    for pos in departures:
        civil = local[pos].replace(year=year)
        expected_wh = 1000 * (2 if civil.weekday() < 5 else 20)
        arrival = returns[np.searchsorted(returns, pos) % len(returns)]
        assert schedule.drive_first_wh[pos] == schedule.drive_second_wh[arrival] == expected_wh / 2
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31")
    expected_kwh = np.where(days.weekday < 5, 2, 20).sum()
    assert (schedule.drive_first_wh + schedule.drive_second_wh).sum() == expected_kwh * 1000
    for run in app._artifacts.vehicle_households.runs.values():
        frame = run.first_year_results_df
        np.testing.assert_allclose(
            (frame["Vehicle_Drive_Energy"] + frame["Vehicle_Drive_Unserved"]) * schedule.hours_per_step,
            schedule.drive_first_wh + schedule.drive_second_wh,
        )
    for household in app.result()["vehicle"]["households"].values():
        for row in household["yearly"]:
            assert row["drive_served_kwh"] + row["drive_unserved_kwh"] == pytest.approx(expected_kwh)
            assert row["drive_unserved_kwh"] == 0


@pytest.mark.parametrize("entry", ["app", "montecarlo"])
@pytest.mark.parametrize("zone", ["America/New_York", "Europe/Berlin"])
@pytest.mark.parametrize("freq", ["h", "15min"])
def test_reference_calendar_csv_uses_app_year_or_montecarlo_target(
    tmp_path, write_multiyear_weather, entry, zone, freq
):
    path = tmp_path / "trips.csv"
    path.write_text("start,end,kind,energy_kwh,away_charging\n2025-06-02 08:00,2025-06-02 09:00,drive,3,\n")
    # Monte Carlo's target calendar intentionally differs from start_date.
    cfg = calendar_config(2025 if entry == "app" else 2024, zone, freq, projection_years=1)
    cfg["vehicle"]["schedule"] = {"file": str(path)}
    if entry == "app":
        result = simulate_offline(cfg).result()
        for household in result["vehicle"]["households"].values():
            assert household["totals"]["drive_served_kwh"] == pytest.approx(3)
            assert household["totals"]["drive_unserved_kwh"] == 0
    else:
        weather = write_multiyear_weather(tmp_path / "weather.csv", years=(2021,))
        result = run_montecarlo(
            cfg, MonteCarloSettings(weather_file=str(weather), n_runs=1, target_year=2025, load_uncertainty=0)
        )
        for key in ("r0", "r2"):
            assert result.runs.iloc[0][f"vehicle_{key}_drive_served_kwh"] == pytest.approx(3)
            assert result.runs.iloc[0][f"vehicle_{key}_drive_unserved_kwh"] == 0
