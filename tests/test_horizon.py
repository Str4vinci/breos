"""Terrain-horizon profile validation and shading tests."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from breos.app_config import resolve_app_config
from breos.app_inputs import AppRuntimeDependencies, load_weather_for_simulation
from breos.pv.horizon import (
    apply_terrain_horizon_profile,
    interpolate_horizon_elevation,
    normalise_horizon_profile,
)


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


def test_apply_horizon_uses_mid_interval_solar_position():
    weather = _weather()
    location = _FakeLocation(_solar_position(weather.index))

    apply_terrain_horizon_profile(
        weather,
        location,
        [[0, 10], [180, 0]],
        freq="h",
        solar_position="mid-interval",
    )

    assert location.requested_times.equals(weather.index + pd.Timedelta(minutes=30))


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
    weather = _weather()
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
    solar_position = _solar_position(weather.index)
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
    )

    assert captured["use_horizon"] is False
    assert loaded.attrs["breos_weather_metadata"]["horizon"]["provider"] == "breos"
