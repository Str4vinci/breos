"""Tests for weather and weather-derived helpers."""

import pandas as pd
import pytest

from breos.weather import (
    build_battery_temperature_series,
    fetch_tmy_weather_data,
    parse_weather_filename,
    preload_weather_by_year,
    read_epw_file,
    resample_to_15min,
    select_random_year_and_replace_datetime,
)


def _write_leap_year_15min_weather(tmp_path):
    idx = pd.date_range("2024-01-01 00:00", "2024-12-31 23:45", freq="15min")
    df = pd.DataFrame({"date": idx, "temp_air": range(len(idx))})
    path = tmp_path / "weather.csv"
    df.to_csv(path, index=False)
    return path, df


def test_battery_temperature_helper_applies_indoor_default():
    idx = pd.date_range("2025-01-01 00:00", periods=2, freq="h", tz="UTC")
    weather = pd.DataFrame({"temp_air": [10.0, 20.0]}, index=idx)

    temps = build_battery_temperature_series("weather", index=idx, weather_df=weather)

    assert float(temps.iloc[0]) == pytest.approx(18.4)
    assert float(temps.iloc[1]) == pytest.approx(21.4)


def test_battery_temperature_helper_can_disable_indoor_model():
    idx = pd.date_range("2025-01-01 00:00", periods=2, freq="h", tz="UTC")
    weather = pd.DataFrame({"temp_air": [10.0, 20.0]}, index=idx)

    temps = build_battery_temperature_series("weather", index=idx, weather_df=weather, indoor_model={"enabled": False})

    assert list(temps) == [10.0, 20.0]


def test_weather_filename_parser_accepts_locations_with_underscores():
    parsed = parse_weather_filename("new_york_city_historical_2020_2024_openmeteo.csv")

    assert parsed == {
        "location": "new_york_city",
        "type": "historical",
        "year_start": "2020",
        "year_end": "2024",
        "source": "openmeteo",
    }


def test_resample_to_15min_keeps_all_slots_in_last_hour():
    idx = pd.date_range("2025-01-01 00:00", periods=3, freq="h")
    weather = pd.DataFrame({"temp_air": [0.0, 4.0, 8.0]}, index=idx)

    resampled = resample_to_15min(weather, method="linear")

    assert len(resampled) == 12
    assert resampled.index[-1] == pd.Timestamp("2025-01-01 02:45")


def test_fetch_tmy_weather_accepts_hourly_frequency_alias(monkeypatch):
    tmy = pd.DataFrame({"ghi": [0.0]}, index=pd.date_range("2020-01-01 00:00", periods=1, freq="h"))

    def fake_get_pvgis_tmy(*args, **kwargs):
        return tmy.copy(), {}

    monkeypatch.setattr("breos.weather.pvlib.iotools.get_pvgis_tmy", fake_get_pvgis_tmy)

    weather, _metadata = fetch_tmy_weather_data(41.0, -8.0, sample_year=None, freq="H")

    assert len(weather) == 1


def test_read_epw_accepts_15t_frequency_alias(monkeypatch):
    epw = pd.DataFrame(
        {
            "ghi": [0.0, 10.0],
            "dni": [0.0, 5.0],
            "dhi": [0.0, 5.0],
            "temp_air": [12.0, 13.0],
            "wind_speed": [1.0, 1.5],
        },
        index=pd.date_range("2025-01-01 00:00", periods=2, freq="h"),
    )
    calls = {}

    def fake_read_epw(_filepath):
        return epw.copy(), {"latitude": 41.0, "longitude": -8.0}

    def fake_resample(df, method="makima", latitude=None, longitude=None, **_kwargs):
        calls["method"] = method
        calls["latitude"] = latitude
        calls["longitude"] = longitude
        return df

    monkeypatch.setattr("breos.weather.pvlib.iotools.read_epw", fake_read_epw)
    monkeypatch.setattr("breos.weather.resample_to_15min", fake_resample)

    read_epw_file("dummy.epw", freq="15T")

    assert calls == {"method": "makima", "latitude": 41.0, "longitude": -8.0}


def test_select_random_year_accepts_15min_leap_year_after_dropping_feb_29(tmp_path):
    weather_path, source = _write_leap_year_15min_weather(tmp_path)

    selected, selected_year = select_random_year_and_replace_datetime(str(weather_path), target_year=2025)

    dates = pd.to_datetime(selected["date"])
    source_march_1 = source.loc[source["date"] == pd.Timestamp("2024-03-01 00:00"), "temp_air"].item()
    mapped_march_1 = selected.loc[dates == pd.Timestamp("2025-03-01 00:00"), "temp_air"].item()

    assert selected_year == 2024
    assert len(selected) == 35040
    assert not ((dates.dt.month == 2) & (dates.dt.day == 29)).any()
    assert dates.iloc[0] == pd.Timestamp("2025-01-01 00:00")
    assert dates.iloc[-1] == pd.Timestamp("2025-12-31 23:45")
    assert mapped_march_1 == source_march_1


def test_preload_weather_by_year_accepts_15min_leap_year_after_dropping_feb_29(tmp_path):
    weather_path, source = _write_leap_year_15min_weather(tmp_path)

    by_year = preload_weather_by_year(str(weather_path), target_year=2025)
    selected = by_year[2024]

    dates = pd.to_datetime(selected["date"])
    source_march_1 = source.loc[source["date"] == pd.Timestamp("2024-03-01 00:00"), "temp_air"].item()
    mapped_march_1 = selected.loc[dates == pd.Timestamp("2025-03-01 00:00"), "temp_air"].item()

    assert len(selected) == 35040
    assert not ((dates.dt.month == 2) & (dates.dt.day == 29)).any()
    assert dates.iloc[0] == pd.Timestamp("2025-01-01 00:00")
    assert dates.iloc[-1] == pd.Timestamp("2025-12-31 23:45")
    assert mapped_march_1 == source_march_1
