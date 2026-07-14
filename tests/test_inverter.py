"""Tests for inverter conversion helpers."""

import pytest

import breos
from breos.inverter import calculate_dc_ac_power, dc_power_for_ac_output


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


def test_dc_ac_power_clips_negative_inputs_to_zero():
    result = calculate_dc_ac_power(
        pv_dc_power=-100.0,
        inverter_ac_power=-1000.0,
        inverter_efficiency=0.96,
    )

    assert result.ac_power_w == 0.0
    assert result.total_dc_input_w == 0.0


@pytest.mark.parametrize("ac_fraction", [0.01, 0.1, 0.5, 0.9, 1.0])
def test_dc_ac_inverse_round_trip(ac_fraction):
    ac_rating = 5000.0
    target = ac_rating * ac_fraction
    dc_input = dc_power_for_ac_output(target, ac_rating, inverter_efficiency=0.96)
    result = calculate_dc_ac_power(dc_input, ac_rating, inverter_efficiency=0.96)

    assert result.ac_power_w == pytest.approx(target, rel=1e-12, abs=1e-9)


def test_package_all_exports_stable_inverter_helpers():
    expected = {
        "calculate_dc_ac_power",
        "InverterConversionResult",
    }

    assert expected.issubset(set(breos.__all__))
