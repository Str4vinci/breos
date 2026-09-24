"""A leap-year start_date runs on a TMY remapped onto the leap calendar (#170)."""

import numpy as np
import pandas as pd
import pytest

from breos.app import App
from breos.app_inputs import remap_tmy_year
from breos.weather import fetch_tmy_weather_data, fill_leap_day


def _tmy(year: int, tz: str = "UTC") -> pd.DataFrame:
    index = pd.date_range(f"{year}-01-01", periods=8760, freq="h", tz=tz)
    frame = pd.DataFrame(
        {"ghi": np.arange(8760, dtype=float), "temp_air": np.arange(8760, dtype=float) / 100.0}, index=index
    )
    frame.attrs["breos_weather_metadata"] = {"source": "PVGIS_TMY"}
    return frame


def test_fill_leap_day_copies_28_february_without_shifting_march():
    weather = _tmy(2027, tz="Etc/GMT-1")
    weather.index = weather.index + pd.DateOffset(years=1)

    filled = fill_leap_day(weather)

    assert len(filled) == 8784
    leap_day = filled.loc["2028-02-29"]
    pd.testing.assert_frame_equal(leap_day.reset_index(drop=True), filled.loc["2028-02-28"].reset_index(drop=True))
    assert filled.loc["2028-03-01 00:00", "ghi"].item() == weather.loc["2028-03-01 00:00", "ghi"].item()
    assert filled.index.is_monotonic_increasing and filled.index.is_unique
    assert filled.attrs["breos_weather_metadata"]["leap_day"] == {"year": 2028, "filled_from": "2028-02-28"}


def test_fill_leap_day_leaves_other_years_alone():
    weather = _tmy(2027)
    assert fill_leap_day(weather) is weather


def test_fetch_tmy_accepts_a_leap_sample_year(monkeypatch):
    captured = {}

    def fake_get_pvgis_tmy(*args, **kwargs):
        captured.update(kwargs)
        return _tmy(kwargs["coerce_year"]), {}

    monkeypatch.setattr("breos.weather.pvlib.iotools.get_pvgis_tmy", fake_get_pvgis_tmy)

    # On develop this raised "Sample year 2028 is a leap year".
    weather, _ = fetch_tmy_weather_data(41.0, -8.0, sample_year=2028, timezone="UTC")

    assert captured["coerce_year"] == 2027
    assert len(weather) == 8784
    assert weather.index[0] == pd.Timestamp("2028-01-01", tz="UTC")
    assert weather.index[-1] == pd.Timestamp("2028-12-31 23:00", tz="UTC")
    assert weather.attrs["breos_weather_metadata"]["leap_day"]["year"] == 2028


def test_remap_tmy_year_onto_a_leap_year_fills_29_february():
    remapped = remap_tmy_year(_tmy(2025), 2028)

    assert len(remapped) == 8784
    assert (remapped.index.year == 2028).all()
    np.testing.assert_array_equal(
        remapped.loc["2028-02-29", "ghi"].to_numpy(), remapped.loc["2028-02-28", "ghi"].to_numpy()
    )


@pytest.mark.parametrize("resolution", ["h", "15min"])
def test_app_runs_a_leap_year_with_storage(_patch_weather, resolution):
    """On develop this passed validation and then failed in simulate()."""
    app = App(
        {
            "location": "porto",
            "n_modules": 8,
            "annual_consumption_kwh": 3500,
            "battery_kwh": 5.0,
            "projection_years": 2,
            "resolution": resolution,
            "start_date": "2028-01-01",
        }
    )
    app.simulate()
    result = app.result()

    assert result["consumption_kwh"] == pytest.approx(3500.0, rel=1e-6)
    assert len(result["monthly"]) == 12
    assert result["pv_production_kwh"] > 0
    assert result["provenance"]["weather"]["leap_day"]["year"] == 2028
