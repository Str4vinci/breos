"""PV-side weather input is refused instead of repaired (#172)."""

import logging

import numpy as np
import pandas as pd
import pytest

from breos.solar import calculate_pv_production_dc, calculate_pv_production_tracking_breakdown
from breos.weather import _complete_weather_years, build_battery_temperature_series


def _dc(weather, location, pv_params, freq="h"):
    return calculate_pv_production_dc(
        weather_data=weather,
        location=location,
        tilt=35,
        surface_azimuth=180,
        n_modules=1,
        pv_params=pv_params,
        freq=freq,
    )


def test_weather_with_a_missing_month_raises(synthetic_weather, porto_location, pv_params):
    # Nearest-row reindexing used to give June zero PV (the nearest rows are
    # night-time 31 May and 1 July) without a message.
    no_june = synthetic_weather[synthetic_weather.index.month != 6]

    with pytest.raises(ValueError, match=r"must step evenly at freq='h': row 3624 is 2023-07-01 00:00"):
        _dc(no_june, porto_location, pv_params)


def test_weather_at_another_resolution_raises(synthetic_weather_15min, porto_location, pv_params):
    with pytest.raises(ValueError, match=r"must step evenly at freq='h': row 1 .* 0 days 00:15:00 after"):
        _dc(synthetic_weather_15min, porto_location, pv_params, freq="h")


def test_tracking_path_checks_the_weather_grid(synthetic_weather, porto_location, pv_params):
    with pytest.raises(ValueError, match="must step evenly"):
        calculate_pv_production_tracking_breakdown(
            weather_data=synthetic_weather.drop(synthetic_weather.index[100]),
            location=porto_location,
            n_modules=1,
            pv_params=pv_params,
            freq="h",
        )


def test_regular_weather_is_unchanged(synthetic_weather, porto_location, pv_params):
    dc = _dc(synthetic_weather, porto_location, pv_params)

    assert dc.index.equals(synthetic_weather.index)
    assert np.isfinite(dc.to_numpy()).all()


@pytest.mark.parametrize(
    ("column", "match"),
    [("temp_air", "no air temperature column"), ("wind_speed", "no wind speed column")],
)
def test_weather_without_a_met_column_raises(synthetic_weather, porto_location, pv_params, column, match):
    # These used to become 25 °C and 1 m/s without a message.
    with pytest.raises(ValueError, match=match):
        _dc(synthetic_weather.drop(columns=column), porto_location, pv_params)


def test_explicit_constant_met_columns_run(synthetic_weather, porto_location, pv_params):
    weather = synthetic_weather.drop(columns=["temp_air", "wind_speed"])
    weather["temp_air"] = 25.0
    weather["wind_speed"] = 1.0

    assert _dc(weather, porto_location, pv_params).sum() > 0


def test_non_finite_air_temperature_raises(synthetic_weather, porto_location, pv_params):
    # A NaN air temperature used to give a NaN cell temperature, replaced by 25 °C.
    weather = synthetic_weather.copy()
    weather.iloc[4000, weather.columns.get_loc("temp_air")] = np.nan

    with pytest.raises(ValueError, match=r"'temp_air' has 1 values that are not finite .* 2023-06-16 16:00"):
        _dc(weather, porto_location, pv_params)


def test_battery_weather_temperature_without_a_column_raises(synthetic_weather):
    weather = synthetic_weather.drop(columns="temp_air")

    with pytest.raises(ValueError, match="needs an air temperature column"):
        build_battery_temperature_series("weather", synthetic_weather.index, weather_df=weather)
    # The explicit fixed temperature is the documented alternative.
    fixed = build_battery_temperature_series(25.0, synthetic_weather.index, indoor_model={"enabled": False})
    assert (fixed == 25.0).all()


def _write_year(path, stamps):
    frame = pd.DataFrame({"date": stamps, "ghi": 0.0, "dni": 0.0, "dhi": 0.0, "temperature_2m": 10.0})
    frame.to_csv(path, index=False)


def test_naive_weather_file_timestamps_are_warned_about(tmp_path, caplog):
    path = tmp_path / "naive.csv"
    _write_year(path, pd.date_range("2023-01-01", periods=8760, freq="h").strftime("%Y-%m-%d %H:%M"))

    with caplog.at_level(logging.WARNING, logger="breos.weather"):
        years, _ = _complete_weather_years(str(path))

    assert list(years) == [2023]
    assert "timestamps without a timezone; BREOS reads them as UTC" in caplog.text


def test_weather_file_with_utc_offsets_is_not_warned_about(tmp_path, caplog):
    path = tmp_path / "aware.csv"
    _write_year(path, pd.date_range("2023-01-01", periods=8760, freq="h", tz="UTC").strftime("%Y-%m-%dT%H:%M%z"))

    with caplog.at_level(logging.WARNING, logger="breos.weather"):
        _complete_weather_years(str(path))

    assert "without a timezone" not in caplog.text
