"""A leap-year start_date runs on a TMY remapped onto the leap calendar (#170)."""

from datetime import timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pvlib.location import Location

from breos.app import App
from breos.app_inputs import AppRuntimeDependencies, load_weather_for_simulation, remap_tmy_year
from breos.weather import fetch_tmy_weather_data, fill_leap_day, load_weather, resample_to_15min


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
    assert result["usable_ac_system_production_kwh"] > 0
    assert result["provenance"]["weather"]["leap_day"]["year"] == 2028


def _offset_tmy(hours: int, freq: str, year: int = 2021) -> pd.DataFrame:
    """A TMY stamped at a fixed UTC offset whose every day has distinct values."""
    steps = 8760 if freq == "h" else 35040
    index = pd.date_range(f"{year}-01-01", periods=steps, freq=freq, tz=timezone(timedelta(hours=hours)))
    frame = pd.DataFrame({"ghi": np.arange(steps, dtype=float)}, index=index)
    frame.attrs["breos_weather_metadata"] = {"source": "PVGIS_TMY"}
    return frame


@pytest.mark.parametrize("freq", ["h", "15min"])
@pytest.mark.parametrize("hours", [1, 11])
def test_remap_tmy_year_fills_29_february_on_a_fixed_offset_clock(hours, freq):
    """#313: shifted in UTC, local 1 March 00:00 landed on 29 February."""
    source = _offset_tmy(hours, freq)
    per_day = 24 if freq == "h" else 96

    remapped = remap_tmy_year(source, 2028)

    index = remapped.index
    assert index.tz == source.index.tz
    assert len(remapped) == 366 * per_day
    assert index.is_monotonic_increasing and index.is_unique
    assert (index.year == 2028).all()
    feb28 = remapped.loc[(index.month == 2) & (index.day == 28), "ghi"].to_numpy()
    feb29 = remapped.loc[(index.month == 2) & (index.day == 29), "ghi"].to_numpy()
    np.testing.assert_array_equal(feb29, feb28)
    # 1 March keeps its own values on its own clock.
    march_1 = remapped.loc[(index.month == 3) & (index.day == 1), "ghi"].to_numpy()
    np.testing.assert_array_equal(march_1, source.loc["2021-03-01", "ghi"].to_numpy())
    assert remapped.attrs["breos_weather_metadata"]["leap_day"] == {"year": 2028, "filled_from": "2028-02-28"}


def _clear_sky_tmy(latitude: float, longitude: float, tz) -> pd.DataFrame:
    index = pd.date_range("2021-01-01", periods=8760, freq="h", tz=tz)
    clear_sky = Location(latitude, longitude).get_clearsky(index)
    # A different cloudiness each day, so 29 February is told apart from its neighbours.
    scale = np.repeat(np.random.default_rng(0).uniform(0.3, 1.0, 365), 24)
    frame = clear_sky[["ghi", "dni", "dhi"]].mul(scale, axis=0)
    frame["temp_air"] = 10.0
    frame["wind_speed"] = 2.0
    return frame


def _load_cached_tmy(tmp_path, frame, freq, *, name, latitude, longitude, timezone_name):
    frame.to_csv(tmp_path / f"{name}_tmy_2005_2020_pvgis-sarah3.csv")
    deps = AppRuntimeDependencies(
        load_profile=lambda **kwargs: None,
        load_weather=load_weather,
        fetch_tmy_weather_data=lambda **kwargs: pytest.fail("the cached TMY should be used"),
        resample_to_15min=resample_to_15min,
        build_battery_temperature_series=lambda **kwargs: None,
    )
    resolved = SimpleNamespace(loc_key=name, lat=latitude, lon=longitude, timezone=timezone_name)
    return load_weather_for_simulation(resolved, freq, 2028, deps, weather_dir=tmp_path)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("freq", ["h", "15min"])
@pytest.mark.parametrize(
    ("name", "latitude", "longitude", "hours", "timezone_name"),
    [
        ("berlin", 52.52, 13.40, 1, "Europe/Berlin"),
        ("melbourne", -37.81, 144.96, 11, "Australia/Melbourne"),
    ],
)
def test_cached_fixed_offset_tmy_gets_29_february(tmp_path, freq, name, latitude, longitude, hours, timezone_name):
    """#313: the App's cached-TMY path, read by the real load_weather."""
    tz = timezone(timedelta(hours=hours))
    source = _clear_sky_tmy(latitude, longitude, tz)

    weather = _load_cached_tmy(
        tmp_path, source, freq, name=name, latitude=latitude, longitude=longitude, timezone_name=timezone_name
    )

    per_day = 24 if freq == "h" else 96
    index = weather.index
    assert index.tz.utcoffset(None) == timedelta(hours=hours)
    assert len(weather) == 366 * per_day
    feb28 = weather.loc[(index.month == 2) & (index.day == 28), "ghi"]
    feb29 = weather.loc[(index.month == 2) & (index.day == 29), "ghi"]
    assert len(feb29) == per_day
    if freq == "h":
        np.testing.assert_array_equal(feb29.to_numpy(), feb28.to_numpy())
    else:
        # The resampler reads one neighbouring hour across each midnight.
        assert feb29.sum() == pytest.approx(feb28.sum(), rel=1e-3)
    assert weather.attrs["breos_weather_metadata"]["leap_day"] == {"year": 2028, "filled_from": "2028-02-28"}


@pytest.mark.filterwarnings("ignore")
def test_cached_utc_tmy_is_shifted_as_before(tmp_path):
    """A UTC-stamped TMY is still shifted in UTC and filled there, unchanged by #313."""
    source = _clear_sky_tmy(41.15, -8.61, "UTC")

    weather = _load_cached_tmy(
        tmp_path, source, "h", name="porto", latitude=41.15, longitude=-8.61, timezone_name="Europe/Lisbon"
    )

    expected = source.copy()
    expected.index = (expected.index + pd.DateOffset(years=7)).as_unit(weather.index.unit)
    expected = fill_leap_day(expected)
    assert str(weather.index.tz) == "UTC"
    pd.testing.assert_frame_equal(weather, expected, check_freq=False)
    assert weather.attrs["breos_weather_metadata"]["leap_day"] == {"year": 2028, "filled_from": "2028-02-28"}


def test_remap_tmy_year_of_a_naive_index_matches_its_utc_remap():
    naive = _tmy(2025)
    naive.index = naive.index.tz_localize(None)

    remapped = remap_tmy_year(naive, 2028)

    utc = remap_tmy_year(_tmy(2025), 2028)
    assert remapped.index.tz is None
    np.testing.assert_array_equal(remapped.index, utc.index.tz_localize(None))
    np.testing.assert_array_equal(remapped["ghi"].to_numpy(), utc["ghi"].to_numpy())
