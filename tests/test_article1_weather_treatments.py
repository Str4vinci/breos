"""Regression coverage for the Article 1 radiation-treatment comparison."""

import tomllib

import pandas as pd

import tools.compare_article1_radiation_treatments as comparison


def test_annual_comparison_forwards_the_actual_configured_pv_chain(monkeypatch):
    index = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
    weather = pd.DataFrame(
        {
            "ghi": [0.0, 1.0],
            "dni": [0.0, 1.0],
            "dhi": [0.0, 1.0],
            "temp_air": 20.0,
            "wind_speed": 1.0,
        },
        index=index,
    )
    config = tomllib.loads(comparison.DEFAULT_CONFIG.read_text())
    seen = {}

    def spy(weather_data, location, **kwargs):
        seen.update(kwargs)
        return pd.Series(0.0, index=weather_data.index)

    monkeypatch.setattr(comparison, "calculate_pv_production_dc", spy)

    row = comparison._annual_row(weather, config, scenario="instant_at_label", year=2025, tilt=35.0, azimuth=180.0)

    assert seen["transposition_model"] == "perez"
    assert seen["model_perez"] == "allsitescomposite1990"
    assert seen["iam_model"] == "ashrae"
    assert seen["diffuse_iam"] == "marion"
    assert seen["albedo"] == 0.25
    assert seen["temperature_model"] == "faiman"
    assert seen["bifacial_model"] == "none"
    assert seen["solar_position"] == "weather"
    assert row["fixed_design_dc_kwh_per_module"] == 0.0
