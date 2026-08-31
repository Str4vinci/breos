"""Focused tests for App input normalization."""

from types import SimpleNamespace

import pandas as pd

from breos.app_inputs import (
    AppRuntimeDependencies,
    load_weather_for_simulation,
    prepare_simulation_inputs,
    remap_tmy_year,
)


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


def test_prepare_inputs_threads_explicit_battery_temperature(monkeypatch):
    import breos.app_inputs as app_inputs

    index = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
    weather = pd.DataFrame({"temp_air": [10.0, 11.0]}, index=index)
    dc = pd.Series([0.0, 1.0], index=index)
    captured = {}

    monkeypatch.setattr(app_inputs, "load_weather_for_simulation", lambda *args, **kwargs: weather)
    monkeypatch.setattr(
        app_inputs,
        "build_pv_production_breakdown",
        lambda *args, **kwargs: SimpleNamespace(dc_after_losses=dc),
    )
    monkeypatch.setattr(app_inputs, "load_consumption_profile", lambda *args, **kwargs: pd.Series(index=index))

    def temperature_builder(temp_config, **kwargs):
        captured["temp_config"] = temp_config
        captured["indoor_model"] = kwargs["indoor_model"]
        return pd.Series(25.0, index=index)

    deps = AppRuntimeDependencies(
        load_profile=lambda **kwargs: None,
        load_weather=lambda **kwargs: None,
        fetch_tmy_weather_data=lambda **kwargs: None,
        resample_to_15min=lambda frame, **kwargs: frame,
        build_battery_temperature_series=temperature_builder,
    )
    cfg = {
        "resolution": "h",
        "start_date": "2025-01-01",
        "horizon_profile": None,
        "solar_position": "interval-start",
        "battery_temperature": 25.0,
        "battery_indoor_model": {"enabled": False},
    }
    resolved = SimpleNamespace(timezone="UTC")

    prepared = prepare_simulation_inputs(cfg, resolved, deps)

    assert prepared.temperature_series.tolist() == [25.0, 25.0]
    assert captured == {"temp_config": 25.0, "indoor_model": {"enabled": False}}
