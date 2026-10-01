"""App.timeseries(): the first simulated year step by step."""

import pandas as pd
import pytest

from breos import App

BASE = {
    "location": "porto",
    "n_modules": 8,
    "annual_consumption_kwh": 4000,
    "battery_kwh": 5.0,
    "projection_years": 2,
    "execution_backend": "python",
}
# Step columns in W whose first-year energy result() reports in kWh.
ENERGY_TOTALS = {
    "Import_From_Grid": "grid_import_kwh",
    "PV_AC_Export": "grid_export_kwh",
    "Houseload": "consumption_kwh",
    "PV_DC": "pv_dc_generation_kwh",
}
STABLE_COLUMNS = {
    "Datetime",
    "PV_DC",
    "PV_Production",
    "Houseload",
    "Import_From_Grid",
    "PV_AC_To_Load",
    "PV_AC_Export",
    "PV_DC_Curtailed",
    "Battery_AC_To_Load",
    "Battery_Energy",
    "Battery_Energy_Beginning",
    "Battery_Energy_End",
    "Battery_SOC_Normalized",
    "Battery_SOH",
}


def _simulated(config):
    app = App(config)
    app.simulate()
    return app


def test_timeseries_before_simulate_raises():
    with pytest.raises(RuntimeError, match=r"Call simulate\(\) before timeseries\(\)"):
        App(BASE).timeseries()


@pytest.mark.parametrize(("resolution", "steps", "hours_per_step"), [("h", 8760, 1.0), ("15min", 35040, 0.25)])
def test_timeseries_has_one_row_per_step_of_the_first_year(_patch_weather, resolution, steps, hours_per_step):
    app = _simulated({**BASE, "resolution": resolution})
    frame = app.timeseries()
    result = app.result()

    assert isinstance(frame.index, pd.RangeIndex) and len(frame) == steps
    assert STABLE_COLUMNS <= set(frame.columns)
    stamps = pd.DatetimeIndex(frame["Datetime"])
    assert stamps.tz is not None
    assert stamps[0] == pd.Timestamp("2023-01-01", tz=stamps.tz)
    assert (stamps[1] - stamps[0]) == pd.Timedelta(hours=hours_per_step)
    for column, key in ENERGY_TOTALS.items():
        assert frame[column].sum() * hours_per_step / 1000 == pytest.approx(result[key], abs=0.01)
    # Only the first of the two project years is kept step by step.
    assert len(result["yearly"]) == 2
    assert frame["Battery_SOH"].iloc[-1] == pytest.approx(result["yearly"][0]["soh_pct"], abs=0.01)


def test_timeseries_returns_a_copy(_patch_weather):
    app = _simulated(BASE)
    frame = app.timeseries()
    before = app.result()["grid_import_kwh"]

    frame["Houseload"] = 0.0
    frame.drop(columns="Import_From_Grid", inplace=True)

    again = app.timeseries()
    assert again is not frame
    assert "Import_From_Grid" in again.columns and again["Houseload"].sum() > 0
    # Revaluation reads the stored per-step load, so it must be untouched too.
    assert app.revalue({"discount_rate": 0.05})["grid_import_kwh"] == before


def test_period_run_timeseries_covers_its_window_only(_patch_weather):
    config = {key: value for key, value in BASE.items() if key != "projection_years"}
    app = _simulated({**config, "period": {"start": "2023-06-01", "end": "2023-06-08"}})
    frame = app.timeseries()
    result = app.result()

    assert len(frame) == 7 * 24
    local = pd.DatetimeIndex(frame["Datetime"]).tz_convert("Europe/Lisbon")
    assert local[0] == pd.Timestamp("2023-06-01", tz="Europe/Lisbon")
    assert local[-1] == pd.Timestamp("2023-06-07 23:00", tz="Europe/Lisbon")
    assert frame["Import_From_Grid"].sum() / 1000 == pytest.approx(result["grid_import_kwh"], abs=0.01)
    assert result["financial"] is None
