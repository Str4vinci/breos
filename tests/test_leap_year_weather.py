"""A leap-year start_date runs on a TMY remapped onto the leap calendar (#170)."""

import calendar
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


def _zone_tmy(zone: str, freq: str, year: int) -> pd.DataFrame:
    """One civil year in a named zone whose every row has a distinct value."""
    index = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq=freq, tz=zone, inclusive="left")
    frame = pd.DataFrame({"ghi": np.arange(len(index), dtype=float)}, index=index)
    frame.attrs["breos_weather_metadata"] = {"source": "PVGIS_TMY"}
    return frame


def _civil_year(zone: str, freq: str, year: int) -> pd.DatetimeIndex:
    """The App's simulation calendar for ``year`` in ``zone``: steps of fixed length."""
    return pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq=freq, tz=zone, inclusive="left")


def _is_29_february(index: pd.DatetimeIndex) -> np.ndarray:
    return (index.month == 2) & (index.day == 29)


def _on_day(frame: pd.DataFrame, month: int, day: int) -> np.ndarray:
    index = frame.index
    return frame.loc[(index.month == month) & (index.day == day), "ghi"].to_numpy()


@pytest.mark.parametrize("freq", ["h", "15min"])
@pytest.mark.parametrize("zone", ["Europe/Berlin", "Australia/Sydney"])
def test_remap_tmy_year_fills_29_february_in_a_named_zone(zone, freq):
    """#329: shifted in UTC, local 1 March 00:00 landed on 29 February."""
    source = _zone_tmy(zone, freq, 2021)
    per_day = 24 if freq == "h" else 96

    remapped = remap_tmy_year(source, 2028)

    # The study year's own transitions: its spring hour has no row and its
    # autumn hour two, as on the App's calendar.
    assert str(remapped.index.tz) == zone
    pd.testing.assert_index_equal(remapped.index, _civil_year(zone, freq, 2028).as_unit(remapped.index.unit))
    np.testing.assert_array_equal(_on_day(remapped, 2, 29), _on_day(remapped, 2, 28))
    assert len(_on_day(remapped, 2, 29)) == per_day
    np.testing.assert_array_equal(_on_day(remapped, 3, 1), _on_day(source, 3, 1))
    # Every other row is the source's, in its order.
    others = remapped[~_is_29_february(remapped.index)]
    np.testing.assert_array_equal(others["ghi"].to_numpy(), source["ghi"].to_numpy())
    assert remapped.attrs["breos_weather_metadata"]["leap_day"] == {"year": 2028, "filled_from": "2028-02-28"}


@pytest.mark.parametrize("freq", ["h", "15min"])
def test_remap_tmy_year_fills_29_february_before_almatys_offset_change(freq):
    """#329: Almaty rolls back an hour at 1 March 2024, after the fill's fixed-clock day."""
    source = _zone_tmy("Asia/Almaty", freq, 2023)
    per_day = 24 if freq == "h" else 96
    clock = timezone(timedelta(hours=6))

    remapped = remap_tmy_year(source, 2024)

    assert str(remapped.index.tz) == "Asia/Almaty"
    assert len(remapped) == 366 * per_day
    assert remapped.index.is_monotonic_increasing and remapped.index.is_unique
    fixed = remapped.tz_convert(clock)
    assert len(_on_day(fixed, 2, 29)) == per_day
    np.testing.assert_array_equal(_on_day(fixed, 2, 29), _on_day(fixed, 2, 28))
    np.testing.assert_array_equal(_on_day(fixed, 2, 28), _on_day(source, 2, 28))
    np.testing.assert_array_equal(_on_day(fixed, 3, 1), _on_day(source, 3, 1))
    np.testing.assert_array_equal(fixed.loc[~_is_29_february(fixed.index), "ghi"], source["ghi"])
    # After conversion, local 29 February has a repeated final hour from
    # fixed-clock 1 March; the copied day itself still has 24 fixed-clock hours.
    assert len(_on_day(remapped, 2, 29)) == per_day + per_day // 24
    assert remapped.attrs["breos_weather_metadata"]["leap_day"] == {"year": 2024, "filled_from": "2024-02-28"}
    assert source.attrs == {"breos_weather_metadata": {"source": "PVGIS_TMY"}}


def test_remap_tmy_year_uses_the_local_year_for_partial_new_year_input():
    """#329: these local 2021 rows belong to UTC 2020, but should move to local 2025."""
    source = _zone_tmy("Europe/Berlin", "15min", 2021).iloc[:3].copy()
    assert (source.index.tz_convert("UTC").year == 2020).all()

    remapped = remap_tmy_year(source, 2025)

    expected = source.copy()
    expected.index = pd.date_range("2025-01-01", periods=3, freq="15min", tz="Europe/Berlin")
    pd.testing.assert_frame_equal(remapped, expected, check_freq=False)
    assert remapped.attrs == source.attrs


@pytest.mark.parametrize(("zone", "year"), [("Africa/Casablanca", 2025), ("Pacific/Fiji", 2019)])
def test_remap_tmy_year_reads_the_local_year_when_february_has_another_offset(zone, year):
    """#329: the year comes from the zone's clock, not from its late-February offset."""
    source = _zone_tmy(zone, "15min", year).iloc[:3].copy()
    january = source.index[0].utcoffset()
    february = pd.Timestamp(year=year, month=2, day=28, hour=12).tz_localize(zone).utcoffset()
    assert january != february

    # Read on the February clock these rows are the year before, and would move one year too far.
    assert remap_tmy_year(source, year).index.equals(source.index)
    clock = timezone(february)
    moved = remap_tmy_year(source, year + 2)
    expected = pd.DatetimeIndex([t.replace(year=t.year + 2) for t in source.index.tz_convert(clock)]).tz_convert(zone)
    assert moved.index.equals(expected)


def test_remap_tmy_year_keeps_named_zone_rows_with_the_sun():
    """Between the two years' transition dates a row keeps its instant, not its wall time."""
    source = _zone_tmy("Europe/Berlin", "h", 2021)

    remapped = remap_tmy_year(source, 2028)

    # 2028 springs forward on 26 March, 2021 on 28 March: local noon on 27
    # March 2028 (10:00 UTC) carries 27 March 2021 at 11:00 (also 10:00 UTC).
    noon = pd.Timestamp("2028-03-27 12:00", tz="Europe/Berlin")
    assert remapped.loc[noon, "ghi"] == source.loc[pd.Timestamp("2021-03-27 11:00", tz="Europe/Berlin"), "ghi"]
    wall = remapped.index.tz_localize(None)
    assert not (wall == pd.Timestamp("2028-03-26 02:00")).any()
    assert (wall == pd.Timestamp("2028-10-29 02:00")).sum() == 2
    assert (wall == pd.Timestamp("2028-10-31 02:00")).sum() == 1


@pytest.mark.parametrize("freq", ["h", "15min"])
@pytest.mark.parametrize("zone", ["Europe/Berlin", "Australia/Sydney"])
def test_remap_tmy_year_drops_the_local_29_february_of_a_named_zone(zone, freq):
    """#329: a leap TMY moved to a common year drops its own local 29 February."""
    source = _zone_tmy(zone, freq, 2024)

    remapped = remap_tmy_year(source, 2025)

    pd.testing.assert_index_equal(remapped.index, _civil_year(zone, freq, 2025).as_unit(remapped.index.unit))
    np.testing.assert_array_equal(_on_day(remapped, 2, 28), _on_day(source, 2, 28))
    np.testing.assert_array_equal(_on_day(remapped, 3, 1), _on_day(source, 3, 1))
    kept = source[~_is_29_february(source.index)]
    np.testing.assert_array_equal(remapped["ghi"].to_numpy(), kept["ghi"].to_numpy())
    assert "leap_day" not in remapped.attrs["breos_weather_metadata"]


@pytest.mark.parametrize("freq", ["h", "15min"])
@pytest.mark.parametrize("zone", ["Europe/Berlin", "Australia/Sydney", "America/New_York", "Asia/Tokyo"])
@pytest.mark.parametrize(
    ("source_year", "target_year"),
    [(2021, 2023), (2020, 2024), (2021, 2025), (2025, 2021), (2024, 2028), (2028, 2020)],
)
def test_remap_tmy_year_of_a_named_zone_between_like_years_is_its_utc_shift(zone, freq, source_year, target_year):
    """Common to common and leap to leap, a named-zone TMY is shifted as before #329."""
    source = _zone_tmy(zone, freq, source_year)

    remapped = remap_tmy_year(source, target_year)

    expected = source.copy()
    expected.index = (expected.index.tz_convert("UTC") + pd.DateOffset(years=target_year - source_year)).tz_convert(
        zone
    )
    pd.testing.assert_frame_equal(remapped, expected, check_freq=False)
    assert remapped.attrs == source.attrs


@pytest.mark.parametrize("freq", ["h", "15min"])
@pytest.mark.parametrize("tz", ["UTC", "Etc/GMT-1", timezone(timedelta(hours=1)), timezone(timedelta(hours=-5)), None])
@pytest.mark.parametrize(("source_year", "target_year"), [(2021, 2025), (2024, 2028), (2021, 2028), (2024, 2025)])
def test_remap_tmy_year_of_a_fixed_clock_is_a_shift_on_that_clock(tz, freq, source_year, target_year):
    """UTC, fixed-offset and naive indices are shifted on their own clock, unchanged by #329."""
    source = _zone_tmy("UTC", freq, source_year)
    source.index = source.index.tz_convert(tz) if tz is not None else source.index.tz_localize(None)

    remapped = remap_tmy_year(source, target_year)

    expected = source.copy() if calendar.isleap(target_year) else source[~_is_29_february(source.index)].copy()
    expected.index = expected.index + pd.DateOffset(years=target_year - source_year)
    expected = fill_leap_day(expected)
    pd.testing.assert_frame_equal(remapped, expected, check_freq=False)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("freq", ["h", "15min"])
@pytest.mark.parametrize(
    ("latitude", "longitude", "zone"), [(52.52, 13.40, "Europe/Berlin"), (-33.87, 151.21, "Australia/Sydney")]
)
def test_named_zone_weather_gets_29_february_in_the_app_input_stage(tmp_path, freq, latitude, longitude, zone):
    """#329: injected weather in a named zone, through load_weather_for_simulation."""
    source = _clear_sky_tmy(latitude, longitude, zone)
    source.attrs["breos_weather_metadata"] = {"source": "injected"}
    deps = AppRuntimeDependencies(
        load_profile=lambda **kwargs: None,
        load_weather=lambda **kwargs: source.copy(),
        fetch_tmy_weather_data=lambda **kwargs: pytest.fail("the injected weather should be used"),
        resample_to_15min=resample_to_15min,
        build_battery_temperature_series=lambda **kwargs: None,
    )
    resolved = SimpleNamespace(loc_key="here", lat=latitude, lon=longitude, timezone=zone)

    weather = load_weather_for_simulation(resolved, freq, 2028, deps, weather_dir=tmp_path)

    per_day = 24 if freq == "h" else 96
    pd.testing.assert_index_equal(weather.index, _civil_year(zone, freq, 2028).as_unit(weather.index.unit))
    feb28 = _on_day(weather, 2, 28)
    feb29 = _on_day(weather, 2, 29)
    assert len(feb29) == per_day
    if freq == "h":
        np.testing.assert_array_equal(feb29, feb28)
    else:
        # The resampler reads one neighbouring hour across each midnight.
        assert feb29.sum() == pytest.approx(feb28.sum(), rel=1e-3)
    assert weather.attrs["breos_weather_metadata"]["leap_day"] == {"year": 2028, "filled_from": "2028-02-28"}
