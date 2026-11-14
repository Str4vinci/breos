"""Terrain-horizon profile validation and shading tests."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pvlib.location import Location

from breos.app_config import resolve_app_config
from breos.app_inputs import AppRuntimeDependencies, load_weather_for_simulation
from breos.pv.horizon import (
    apply_terrain_horizon_profile,
    interpolate_horizon_elevation,
    normalise_horizon_profile,
)
from breos.solar import _prepare_solarpos_and_weather


class _FakeLocation:
    def __init__(self, solar_position: pd.DataFrame):
        self.solar_position = solar_position
        self.requested_times = None

    def get_solarposition(self, times):
        self.requested_times = times
        result = self.solar_position.copy()
        result.index = times
        return result


def _weather(status: str = "not_applied") -> pd.DataFrame:
    index = pd.date_range("2025-01-01 10:00", periods=3, freq="h", tz="UTC")
    weather = pd.DataFrame(
        {
            "ghi": [600.0, 600.0, 600.0],
            "dni": [500.0, 500.0, 500.0],
            "dhi": [100.0, 100.0, 100.0],
            "temp_air": [15.0, 15.0, 15.0],
            "wind_speed": [3.0, 3.0, 3.0],
        },
        index=index,
    )
    weather.attrs["breos_weather_metadata"] = {
        "source": "test",
        "horizon": {"status": status, "provider": "test", "profile": None},
    }
    return weather


def _solar_position(index: pd.DatetimeIndex) -> pd.DataFrame:
    elevations = np.array([5.0, 15.0, 5.0])
    return pd.DataFrame(
        {
            "azimuth": [350.0, 10.0, 90.0],
            "apparent_elevation": elevations,
            "apparent_zenith": 90.0 - elevations,
        },
        index=index,
    )


def test_horizon_profile_normalises_360_sorts_and_converts_to_floats():
    assert normalise_horizon_profile([[180, 2], [360, 8], [90, 4]]) == [
        [0.0, 8.0],
        [90.0, 4.0],
        [180.0, 2.0],
    ]


@pytest.mark.parametrize(
    ("profile", "error", "message"),
    [
        ("0,5", TypeError, "list"),
        ([[0, 5]], ValueError, "at least two"),
        ([[0, 5, 2], [180, 0]], TypeError, "pair"),
        ([[0, 5], [361, 0]], ValueError, "between 0 and 360"),
        ([[0, 5], [180, 91]], ValueError, "between -90 and 90"),
        ([[0, 5], [360, 0]], ValueError, "duplicate azimuth"),
    ],
)
def test_horizon_profile_rejects_invalid_shapes_and_values(profile, error, message):
    with pytest.raises(error, match=message):
        normalise_horizon_profile(profile)


def test_horizon_interpolation_wraps_smoothly_across_north():
    profile = [[0.0, 10.0], [90.0, 0.0], [270.0, 0.0]]

    elevations = interpolate_horizon_elevation(profile, [350.0, 0.0, 10.0])

    np.testing.assert_allclose(elevations, [8.8888888889, 10.0, 8.8888888889])


def test_apply_horizon_zeros_beam_and_removes_direct_horizontal_from_ghi():
    weather = _weather()
    location = _FakeLocation(_solar_position(weather.index))

    shaded = apply_terrain_horizon_profile(
        weather,
        location,
        [[0, 10], [180, 0]],
        freq="h",
        solar_position="interval-start",
    )

    assert shaded["dni"].tolist() == [0.0, 500.0, 0.0]
    expected_ghi = [
        600.0 - 500.0 * np.cos(np.radians(85.0)),
        600.0,
        600.0 - 500.0 * np.cos(np.radians(85.0)),
    ]
    np.testing.assert_allclose(shaded["ghi"], expected_ghi)
    assert weather["dni"].tolist() == [500.0, 500.0, 500.0]
    horizon = shaded.attrs["breos_weather_metadata"]["horizon"]
    assert horizon["status"] == "applied"
    assert horizon["provider"] == "breos"
    assert horizon["profile"]["points"] == [[0.0, 10.0], [180.0, 0.0]]
    assert horizon["profile"]["shaded_timesteps"] == 2
    assert horizon["profile"]["diffuse_shading"] == "not_modeled"


@pytest.mark.parametrize("status", ["applied", "unknown"])
def test_apply_horizon_refuses_double_counting_or_unknown_weather(status):
    weather = _weather(status)
    location = _FakeLocation(_solar_position(weather.index))

    with pytest.raises(ValueError, match="double-count|Cannot safely"):
        apply_terrain_horizon_profile(weather, location, [[0, 10], [180, 0]], freq="h")


def test_app_config_accepts_and_normalises_inline_horizon_pairs():
    resolved = resolve_app_config(
        {
            "location": "porto",
            "n_modules": 2,
            "annual_consumption_kwh": 3000,
            "horizon_profile": [[180, 2], [0, 8], [90, 4]],
        }
    )

    assert resolved.cfg["horizon_profile"] == [[0.0, 8.0], [90.0, 4.0], [180.0, 2.0]]


def test_active_profile_requests_unshaded_pvgis_weather(monkeypatch, tmp_path):
    # App weather must cover the whole simulated year, so the three test rows
    # are repeated across 2025.
    index = pd.date_range("2025-01-01", periods=8760, freq="h", tz="UTC")
    weather = _weather().iloc[np.arange(len(index)) % 3].set_axis(index)
    captured = {}

    def fetch(**kwargs):
        captured.update(kwargs)
        return weather, {}

    deps = AppRuntimeDependencies(
        load_profile=lambda **kwargs: None,
        load_weather=lambda **kwargs: None,
        fetch_tmy_weather_data=fetch,
        resample_to_15min=lambda frame, **kwargs: frame,
        build_battery_temperature_series=lambda **kwargs: None,
    )
    resolved = SimpleNamespace(loc_key="porto", lat=41.0, lon=-8.0, timezone="UTC")
    solar_position = _solar_position(weather.index[:3]).iloc[np.arange(len(index)) % 3]
    monkeypatch.setattr(
        "breos.pv.horizon.Location.get_solarposition", lambda self, times: solar_position.set_axis(times)
    )

    loaded = load_weather_for_simulation(
        resolved,
        "h",
        2025,
        deps,
        weather_dir=tmp_path,
        horizon_profile=[[0, 10], [180, 0]],
        solar_position="interval-start",
    )

    assert captured["use_horizon"] is False
    assert loaded.attrs["breos_weather_metadata"]["horizon"]["provider"] == "breos"


def _clear_sky_weather(names=("ghi", "dni", "dhi")) -> pd.DataFrame:
    """Two clear June days in Porto, with an unapplied horizon."""
    idx = pd.date_range("2025-06-01", periods=48, freq="h", tz="UTC")
    weather = Location(41.15, -8.61).get_clearsky(idx)[["ghi", "dni", "dhi"]]
    weather.columns = list(names)
    weather.attrs["breos_weather_metadata"] = {"horizon": {"status": "not_applied"}}
    return weather


def _interval_weather(radiation: dict) -> pd.DataFrame:
    weather = _weather()
    weather.attrs["breos_weather_metadata"].update(radiation)
    return weather


@pytest.mark.parametrize(
    ("radiation", "offset"),
    [
        pytest.param(
            {"radiation_time_basis": "instant", "irradiance_time_offset_hours": 0.1714},
            pd.Timedelta(hours=0.1714),
            id="instant-provider-offset",
        ),
        pytest.param(
            {"radiation_time_basis": "interval_mean", "timestamp_label_basis": "left"},
            pd.Timedelta(minutes=30),
            id="left-labelled-mean",
        ),
        pytest.param(
            {"radiation_time_basis": "interval_mean", "timestamp_label_basis": "right"},
            pd.Timedelta(minutes=-30),
            id="right-labelled-mean",
        ),
    ],
)
def test_shading_and_transposition_evaluate_the_weather_sun_at_the_same_times(radiation, offset):
    # The offset table itself is tested in test_weather.py; these cases show
    # shading and transposition both read it, at the same absolute times.
    weather = _interval_weather(radiation)
    shading_location = _FakeLocation(_solar_position(weather.index))
    transposition_location = _FakeLocation(_solar_position(weather.index))

    shaded = apply_terrain_horizon_profile(
        weather, shading_location, [[0, 10], [180, 0]], freq="h", solar_position="weather"
    )
    labels, solar_position, _weather = _prepare_solarpos_and_weather(
        weather, transposition_location, "h", solar_position="weather"
    )

    assert shading_location.requested_times.equals(weather.index + offset)
    assert transposition_location.requested_times.equals(weather.index + offset)
    assert solar_position.index.equals(labels)
    assert shaded.attrs["breos_weather_metadata"]["horizon"]["profile"]["solar_position"] == "weather"


def test_apply_horizon_weather_timing_requires_radiation_metadata():
    weather = _weather()
    location = _FakeLocation(_solar_position(weather.index))

    with pytest.raises(ValueError, match="radiation_time_basis"):
        apply_terrain_horizon_profile(weather, location, [[0, 10], [180, 0]], freq="h", solar_position="weather")


def test_interval_mean_beam_is_kept_while_the_mid_interval_sun_clears_the_horizon():
    # At 06:00 UTC on 21 June in Porto the sun is at 8.9 degrees; at 06:30,
    # the middle of the left-labelled hour, it is at 14.1 degrees. A flat 12
    # degree horizon therefore blocks the label but not the interval mean.
    index = pd.DatetimeIndex(["2025-06-21 06:00"], tz="UTC")
    weather = pd.DataFrame(
        {"ghi": [300.0], "dni": [600.0], "dhi": [100.0], "temp_air": [15.0], "wind_speed": [3.0]}, index=index
    )
    weather.attrs["breos_weather_metadata"] = {
        "source": "test",
        "radiation_time_basis": "interval_mean",
        "timestamp_label_basis": "left",
        "horizon": {"status": "not_applied", "provider": None, "profile": None},
    }
    location = Location(41.15, -8.63, tz="UTC")
    profile = [[0, 12], [180, 12]]

    by_label = apply_terrain_horizon_profile(weather, location, profile, freq="h", solar_position="interval-start")
    by_midpoint = apply_terrain_horizon_profile(weather, location, profile, freq="h", solar_position="mid-interval")
    by_metadata = apply_terrain_horizon_profile(weather, location, profile, freq="h", solar_position="weather")

    assert by_label["dni"].iloc[0] == 0.0
    assert by_midpoint["dni"].iloc[0] == 600.0
    assert by_metadata["dni"].iloc[0] == 600.0


@pytest.mark.parametrize(
    "names",
    [
        ("ghi", "dni", "dhi"),
        ("GHI", "DNI", "DHI"),
        ("shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation"),
        ("global_horizontal_irradiance", "direct_normal_irradiance", "diffuse_horizontal_irradiance"),
    ],
)
def test_apply_horizon_finds_irradiance_under_every_recognised_name(names):
    # BREOS's own Open-Meteo fetch returns the long names; they used to raise (#211).
    weather = _clear_sky_weather(names)

    shaded = apply_terrain_horizon_profile(
        weather,
        Location(41.15, -8.61),
        [[0, 30], [90, 30], [180, 30], [270, 30]],
        freq="h",
        solar_position="interval-start",
    )

    assert list(shaded.columns) == list(names)
    assert shaded.attrs["breos_weather_metadata"]["horizon"]["profile"]["shaded_timesteps"] == 9
    assert (shaded[names[1]] == 0).sum() >= 9


def test_apply_horizon_names_every_accepted_column_when_one_is_missing():
    idx = pd.date_range("2025-06-01", periods=4, freq="h", tz="UTC")
    weather = pd.DataFrame({"ghi": 0.0, "dni": 0.0}, index=idx)
    weather.attrs["breos_weather_metadata"] = {"horizon": {"status": "not_applied"}}

    with pytest.raises(ValueError, match=r"DHI column \('dhi', 'diffuse_radiation', 'diffuse_horizontal_irradiance'"):
        apply_terrain_horizon_profile(weather, Location(41.15, -8.61), [[0, 30], [180, 30]], freq="h")


@pytest.mark.parametrize(
    ("dtype", "shaded_ghi_dtype", "missing_row"),
    [
        ("float64", "float64", False),
        ("float32", "float32", False),
        ("int64", "float64", False),
        ("Float64", "Float64", False),
        ("Float32", "Float32", False),
        ("Int64", "Float64", False),
        # Explicitly fill a missing nullable row before shading.
        pytest.param("Float64", "Float64", True, id="Float64-missing-row"),
    ],
)
def test_apply_horizon_accepts_any_numeric_irradiance_dtype(dtype, shaded_ghi_dtype, missing_row):
    # Open-Meteo returns float32 columns; pandas 3 refused to write the
    # float64 shaded GHI into them (#210). Nullable pandas dtypes must not
    # reach NumPy's astype, which cannot interpret them.
    location = Location(41.15, -8.61)
    reference = _clear_sky_weather()
    weather = reference.round().astype(dtype)
    weather.attrs = reference.attrs
    if missing_row:
        weather.iloc[12, :] = pd.NA
        weather = weather.fillna(0.0)
    profile = [[0, 30], [90, 30], [180, 30], [270, 30]]

    shaded = apply_terrain_horizon_profile(weather, location, profile, freq="h", solar_position="interval-start")
    expected = apply_terrain_horizon_profile(
        weather.astype("float64"), location, profile, freq="h", solar_position="interval-start"
    )

    assert shaded.attrs["breos_weather_metadata"]["horizon"]["profile"]["shaded_timesteps"] == 9
    assert shaded["dhi"].dtype == weather["dhi"].dtype
    assert shaded["ghi"].dtype == shaded_ghi_dtype
    if missing_row:
        assert (shaded.iloc[12] == 0).all()
    np.testing.assert_allclose(
        shaded[["ghi", "dni", "dhi"]].to_numpy(dtype=float, na_value=np.nan),
        expected[["ghi", "dni", "dhi"]],
        rtol=1e-6,
    )


@pytest.mark.parametrize("column,bad", [("ghi", 0.0), ("ghi", np.nan), ("dni", np.inf), ("dhi", -np.inf)])
def test_horizon_cannot_conceal_invalid_source_irradiance(column, bad):
    weather = _weather()
    weather.loc[weather.index[0], column] = bad
    original = weather.copy(deep=True)
    with pytest.raises(ValueError, match="inconsistent irradiance|not finite numbers") as exc:
        apply_terrain_horizon_profile(
            weather, _FakeLocation(_solar_position(weather.index)), [[0, 90], [180, 90]], freq="h"
        )
    assert str(weather.index[0]) in str(exc.value)
    pd.testing.assert_frame_equal(weather, original)
