"""EPW radiation is a left-labelled hourly interval mean (#213)."""

import numpy as np
import pandas as pd
import pytest
from pvlib.location import Location

from breos.pv_modules import get_module
from breos.solar import calculate_pv_production_dc
from breos.weather import read_epw_file

_HEADER = [
    "LOCATION,Porto,-,PRT,SYNTHETIC,085450,41.23,-8.68,0.0,73.0",
    "DESIGN CONDITIONS,0",
    "TYPICAL/EXTREME PERIODS,0",
    "GROUND TEMPERATURES,0",
    "HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0",
    "COMMENTS 1,synthetic test file",
    "COMMENTS 2,",
    "DATA PERIODS,1,1,Data,Sunday, 1/ 1,12/31",
]


def _write_epw(path, days=3):
    """A clear-sky-shaped EPW: each record is the mean over the hour it ends."""
    site = Location(41.23, -8.68, tz="UTC", altitude=73.0)
    fine = pd.date_range("2025-06-01", periods=days * 24 * 60, freq="min", tz="UTC")
    clear = site.get_clearsky(fine).resample("h", label="left", closed="left").mean()
    rows = []
    for start, sky in clear.iterrows():
        end = start + pd.Timedelta(hours=1)
        hour = 24 if end.hour == 0 else end.hour
        day = start if end.hour == 0 else end
        fields = [2025, day.month, day.day, hour, 60, "?", 18.0, 10.0, 60, 101325, 9999, 9999, 350]
        fields += [round(sky["ghi"]), round(sky["dni"]), round(sky["dhi"])]
        fields += [999999, 999999, 999999, 9999, 180, 2.0, 0, 0, 9999, 99999, 9, 999999999, 999, 0.999, 999, 99]
        fields += [0.2, 0, 0]
        rows.append(",".join(str(v) for v in fields))
    path.write_text("\n".join(_HEADER + rows) + "\n")
    return clear


def test_epw_records_left_labelled_interval_means(tmp_path):
    path = tmp_path / "porto.epw"
    _write_epw(path)

    weather = read_epw_file(str(path))
    metadata = weather.attrs["breos_weather_metadata"]

    assert metadata["radiation_time_basis"] == "interval_mean"
    assert metadata["timestamp_label_basis"] == "left"
    # pvlib labels the hour ending at 01:00 as 00:00.
    assert weather.index[0] == pd.Timestamp("2025-06-01 00:00", tz="UTC")


def test_15min_epw_keeps_the_basis_and_resampling_provenance(tmp_path):
    path = tmp_path / "porto.epw"
    _write_epw(path)

    weather = read_epw_file(str(path), freq="15min")
    metadata = weather.attrs["breos_weather_metadata"]

    assert metadata["radiation_time_basis"] == "interval_mean"
    assert metadata["timestamp_label_basis"] == "left"
    assert metadata["input_resolution"] == "h"
    assert metadata["output_resolution"] == "15min"
    # On a clear day, clear-sky scaling at the interval midpoints reproduces
    # the true 15-minute means. Evaluated at the labels, every hour was skewed
    # toward its start: a mean error of 27 W/m², and up to 77 W/m².
    site = Location(41.23, -8.68, tz="UTC", altitude=73.0)
    fine = pd.date_range("2025-06-01", periods=3 * 24 * 60, freq="min", tz="UTC")
    truth = site.get_clearsky(fine)["ghi"].resample("15min").mean().reindex(weather.index)
    error = (weather["ghi"] - truth).abs()
    assert error.mean() < 3.0
    assert error.max() < 20.0


def test_solar_position_weather_runs_on_epw(tmp_path):
    path = tmp_path / "porto.epw"
    _write_epw(path)
    weather = read_epw_file(str(path))
    location = Location(41.23, -8.68, tz="UTC")
    kwargs = dict(
        weather_data=weather,
        location=location,
        tilt=35,
        surface_azimuth=180,
        n_modules=1,
        pv_params=get_module("Suntech_STP550S_STC"),
        freq="h",
    )

    by_weather = calculate_pv_production_dc(**kwargs, solar_position="weather")
    mid_interval = calculate_pv_production_dc(**kwargs, solar_position="mid-interval")

    # "weather" resolves a left-labelled hourly mean to the interval midpoint.
    pd.testing.assert_series_equal(by_weather, mid_interval)
    assert by_weather.sum() > 0
