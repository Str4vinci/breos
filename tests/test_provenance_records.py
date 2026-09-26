"""Provenance describes what the model did, and the recorded config replays (#215)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

import breos.montecarlo as montecarlo_module
from breos import App
from breos.app_config import default_module_key
from breos.cli import _resolved_config_summary
from breos.montecarlo import MonteCarloSettings, _precompute_year_caches

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "parity"))

from app_parity import flatten  # noqa: E402

_SINGLE = {
    "location": "porto",
    "n_modules": 8,
    "annual_consumption_kwh": 4000,
    "battery_kwh": 5,
    "resolution": "15min",
    "projection_years": 2,
}
_ARRAYS = {
    "location": "porto",
    "annual_consumption_kwh": 4000,
    "battery_kwh": 5,
    "projection_years": 2,
    "pv_arrays": [{"modules": 4, "tilt": 30, "azimuth": 90}, {"modules": 4, "tilt": 30, "azimuth": 270}],
}


def _run(config):
    app = App(config)
    app.simulate()
    return app.result()


@pytest.mark.usefixtures("_patch_weather")
def test_15min_app_run_keeps_the_resampling_provenance():
    # App used to write the pre-resampling metadata back over the resampler's.
    weather = _run(_SINGLE)["provenance"]["weather"]

    assert weather["source"] == "PVGIS_TMY"
    assert weather["input_resolution"] == "h"
    assert weather["output_resolution"] == "15min"
    assert weather["irradiance_resampling_method"] == "makima"
    assert weather["preserve_irradiance_energy"] is False


@pytest.mark.usefixtures("_patch_weather")
@pytest.mark.parametrize("config", [_SINGLE, _ARRAYS], ids=["single", "pv_arrays"])
def test_recorded_config_names_a_module_app_accepts_and_replays(config):
    result = _run(config)
    recorded = result["provenance"]["resolved_config"]

    # The default module used to be recorded by its display name,
    # "Suntech_STP550S-C72/Vmh", which App rejects.
    assert recorded["pv_module"] == default_module_key()
    assert _resolved_config_summary(config)["pv"]["module"] == default_module_key()

    replayed = flatten(_run(recorded))
    original = flatten(result)
    # A replayed location is a coordinate dict, so it carries no preset name.
    ignored = ("provenance.execution", "provenance.resolved_config.location.preset")
    differing = [key for key in original if not key.startswith(ignored) and original[key] != replayed.get(key)]
    assert not differing, differing[:10]


@pytest.mark.parametrize(("spelling", "minutes"), [("Mid-Interval", 30.0), (" INTERVAL-START ", 0.0)])
def test_montecarlo_records_the_offset_the_pv_model_applies(monkeypatch, spelling, minutes):
    # A non-canonical spelling used to be recorded as a 0-minute offset while
    # the PV model applied half a step.
    frame = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=2, freq="h"), "temperature_2m": [10.0, 11.0]})
    index = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
    monkeypatch.setattr(montecarlo_module, "preload_weather_by_year", lambda *args, **kwargs: {2021: frame})
    monkeypatch.setattr(montecarlo_module, "build_dc_system_base", lambda *args, **kwargs: pd.Series(0.0, index=index))
    monkeypatch.setattr(
        montecarlo_module, "build_battery_temperature_series", lambda *args, **kwargs: pd.Series(25.0, index=index)
    )
    runtime_weather = {}

    _precompute_year_caches(
        {
            "resolution": "h",
            "solar_position": spelling,
            "battery_temperature": 25.0,
            "battery_indoor_model": {"enabled": False},
        },
        type("Resolved", (), {"lat": 41.0, "lon": -8.0})(),
        MonteCarloSettings(weather_file="unused.csv"),
        runtime_weather=runtime_weather,
    )

    assert runtime_weather["solar_position_method"] == spelling.strip().lower()
    assert runtime_weather["solar_position_offset_minutes"] == minutes
