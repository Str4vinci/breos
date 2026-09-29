"""Tests for inverter conversion helpers."""

import pytest

from breos.inverter import InverterConfig, calculate_dc_ac_power, dc_power_for_ac_output


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("nominal_power_w", -1.0),
        ("nominal_power_w", float("nan")),
        ("dc_ac_ratio", 0.0),
        ("dc_ac_ratio", float("inf")),
        ("inverter_efficiency", 0.0),
        ("inverter_efficiency", 1.01),
        ("inverter_efficiency", float("nan")),
    ],
)
def test_inverter_rejects_invalid_numeric_fields(field, value):
    with pytest.raises(ValueError, match=field):
        InverterConfig(**{field: value})


@pytest.mark.parametrize("is_hybrid", [0, 1, "true", None])
def test_inverter_requires_boolean_hybrid_flag(is_hybrid):
    with pytest.raises(ValueError, match="is_hybrid must be a bool"):
        InverterConfig(is_hybrid=is_hybrid)


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_inverter_rejects_non_positive_or_non_integer_mppt_channels(value):
    with pytest.raises(ValueError, match="mppt_channels"):
        InverterConfig(mppt_channels=value)


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


def test_unity_nominal_efficiency_cannot_create_energy():
    dc_input = dc_power_for_ac_output(600.0, 1000.0, inverter_efficiency=1.0)
    result = calculate_dc_ac_power(dc_input, 1000.0, inverter_efficiency=1.0)

    assert dc_input == pytest.approx(600.0)
    assert result.ac_power_w == pytest.approx(600.0)
    assert result.total_dc_input_w == pytest.approx(dc_input)
