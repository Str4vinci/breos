"""Static coverage for Article 1 contextual source-data tooling."""

import hashlib
import json
import tomllib

import pandas as pd
import pytest

import tools.reproduce_article1_context as context
from tools.reproduce_article1_context import (
    DEFAULT_CONFIG,
    _inclusive_values,
    _pv_module_provenance,
    _resolved_config_sha256,
)


def test_article1_context_grid_includes_both_bounds():
    assert _inclusive_values(10.0, 20.0, 5.0).tolist() == [10.0, 15.0, 20.0]


def test_article1_context_grid_rejects_non_positive_step():
    with pytest.raises(ValueError, match="positive"):
        _inclusive_values(10.0, 20.0, 0.0)


def test_article1_context_records_resolved_pv_module():
    config = tomllib.loads(DEFAULT_CONFIG.read_text())

    module = _pv_module_provenance(config)

    assert module["parameters"]["T_Pmax_pct"] == -0.34
    assert module["width_m"] == 1.134
    assert module["length_m"] == 2.278


def _weather() -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
    return pd.DataFrame(
        {"ghi": 0.0, "dni": 0.0, "dhi": 0.0, "temp_air": 20.0, "wind_speed": 1.0},
        index=index,
    )


def _assert_configured_chain(kwargs: dict) -> None:
    assert kwargs["transposition_model"] == "perez"
    assert kwargs["model_perez"] == "allsitescomposite1990"
    assert kwargs["iam_model"] == "ashrae"
    assert kwargs["diffuse_iam"] == "marion"
    assert kwargs["albedo"] == 0.25
    assert kwargs["temperature_model"] == "faiman"
    assert kwargs["bifacial_model"] == "none"
    assert kwargs["solar_position"] == "weather"


def test_orientation_forwards_the_configured_pv_chain(monkeypatch):
    seen = {}

    def spy(weather_data, location, **kwargs):
        seen.update(kwargs)
        return pd.Series(0.0, index=weather_data.index)

    monkeypatch.setattr(context, "calculate_pv_production_dc", spy)
    config = tomllib.loads(DEFAULT_CONFIG.read_text())

    context._annual_module_production(_weather(), config, tilt=35.0, azimuth=180.0)

    _assert_configured_chain(seen)


def test_weather_comparison_forwards_the_configured_pv_chain(monkeypatch):
    seen = {}

    def spy(weather_data, location, **kwargs):
        seen.update(kwargs)
        return pd.Series(0.0, index=weather_data.index)

    monkeypatch.setattr(context, "calculate_pv_production_dc", spy)
    config = tomllib.loads(DEFAULT_CONFIG.read_text())

    context._monthly_weather_rows({2025: _weather()}, config, tilt=35.0, azimuth=180.0)

    _assert_configured_chain(seen)


def test_resolved_config_hash_is_independently_recomputable():
    config = tomllib.loads(DEFAULT_CONFIG.read_text())
    independently_serialized = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    expected = hashlib.sha256(independently_serialized.encode("utf-8")).hexdigest()

    assert _resolved_config_sha256(config) == expected
