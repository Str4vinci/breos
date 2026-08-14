"""Focused tests for App input normalization."""

from types import SimpleNamespace

import pandas as pd

from breos.app_inputs import AppRuntimeDependencies, load_weather_for_simulation, remap_tmy_year


def test_remap_tmy_year_preserves_weather_metadata():
    weather = pd.DataFrame(
        {"ghi": [0.0, 1.0]},
        index=pd.date_range("2020-01-01", periods=2, freq="h", tz="UTC"),
    )
    weather.attrs["breos_weather_metadata"] = {
        "source": "test",
        "horizon": {"status": "not_applied", "provider": "test", "profile": None},
    }

    remapped = remap_tmy_year(weather, 2025)

    assert remapped.index[0].year == 2025
    assert remapped.attrs["breos_weather_metadata"] == weather.attrs["breos_weather_metadata"]
    assert remapped.attrs["breos_weather_metadata"] is not weather.attrs["breos_weather_metadata"]


def test_load_weather_for_simulation_marks_injected_weather_horizon_unknown(tmp_path):
    weather = pd.DataFrame(
        {"ghi": [0.0, 1.0]},
        index=pd.date_range("2020-01-01", periods=2, freq="h", tz="UTC"),
    )
    deps = AppRuntimeDependencies(
        load_profile=lambda **kwargs: None,
        load_weather=lambda **kwargs: None,
        fetch_tmy_weather_data=lambda **kwargs: (weather, {}),
        resample_to_15min=lambda frame, **kwargs: frame,
        build_battery_temperature_series=lambda **kwargs: None,
    )
    resolved = SimpleNamespace(loc_key="porto", lat=41.0, lon=-8.0, timezone="UTC")

    loaded = load_weather_for_simulation(resolved, "h", 2025, deps, weather_dir=tmp_path)

    assert loaded.attrs["breos_weather_metadata"] == {
        "source": "runtime_dependency_or_unknown",
        "note": "The injected weather provider did not expose source metadata.",
        "horizon": {"status": "unknown", "provider": None, "profile": None},
    }
