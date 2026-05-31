"""Tests for inverter conversion helpers."""

import pytest

import breos
from breos.inverter import calculate_dc_ac_efficiency, calculate_dc_ac_power


def test_dc_ac_power_exposes_dc_side_clipping_losses():
    result = calculate_dc_ac_power(
        pv_dc_power=1500.0,
        inverter_ac_power=1000.0,
        inverter_efficiency=0.8,
    )

    assert result.ac_power_w == pytest.approx(1000.0)
    assert result.conversion_loss_w == pytest.approx(250.0)
    assert result.clipping_loss_dc_w == pytest.approx(250.0)
    assert result.clipping_loss_ac_equivalent_w == pytest.approx(200.0)
    assert result.total_dc_input_w == pytest.approx(1500.0)
    assert calculate_dc_ac_efficiency(1500.0, 1000.0, 0.8) == pytest.approx(1000.0)


def test_dc_ac_power_clips_negative_inputs_to_zero():
    result = calculate_dc_ac_power(
        pv_dc_power=-100.0,
        inverter_ac_power=-1000.0,
        inverter_efficiency=0.96,
    )

    assert result.ac_power_w == 0.0
    assert result.total_dc_input_w == 0.0


def test_package_all_exports_new_public_helpers():
    expected = {
        "calculate_dc_ac_power",
        "InverterConversionResult",
        "build_battery_temperature_series",
        "preload_weather_by_year",
        "remap_datetime_index_years",
    }

    assert expected.issubset(set(breos.__all__))
