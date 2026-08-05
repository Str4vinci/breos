"""Tests for inverter conversion helpers."""

import pytest

import breos
from breos.inverter import InverterConfig, calculate_dc_ac_power, dc_power_for_ac_output


def test_inverter_datasheet_limits_are_optional_for_legacy_callers():
    config = InverterConfig()

    assert config.max_dc_voltage_v is None
    assert config.max_dc_power_w is None
    assert config.min_mppt_voltage_v is None
    assert config.max_mppt_voltage_v is None
    assert config.startup_voltage_v is None
    assert config.max_input_current_per_mppt_a is None
    assert config.max_short_circuit_current_per_mppt_a is None
    assert config.max_strings_per_mppt is None


def test_inverter_accepts_complete_datasheet_limits():
    config = InverterConfig(
        nominal_power_w=5000.0,
        max_dc_voltage_v=600.0,
        max_dc_power_w=7500.0,
        min_mppt_voltage_v=120.0,
        max_mppt_voltage_v=560.0,
        startup_voltage_v=150.0,
        max_input_current_per_mppt_a=16.0,
        max_short_circuit_current_per_mppt_a=24.0,
        max_strings_per_mppt=2,
    )

    assert config.max_dc_voltage_v == 600.0
    assert config.max_dc_power_w == 7500.0
    assert config.min_mppt_voltage_v == 120.0
    assert config.max_mppt_voltage_v == 560.0
    assert config.startup_voltage_v == 150.0
    assert config.max_input_current_per_mppt_a == 16.0
    assert config.max_short_circuit_current_per_mppt_a == 24.0
    assert config.max_strings_per_mppt == 2


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
        ("cost_per_kw_simple", -0.01),
        ("cost_per_kw_hybrid", float("inf")),
    ],
)
def test_inverter_rejects_invalid_existing_numeric_fields(field, value):
    with pytest.raises(ValueError, match=field):
        InverterConfig(**{field: value})


@pytest.mark.parametrize("is_hybrid", [0, 1, "true", None])
def test_inverter_requires_boolean_hybrid_flag(is_hybrid):
    with pytest.raises(ValueError, match="is_hybrid must be a bool"):
        InverterConfig(is_hybrid=is_hybrid)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_dc_voltage_v", -1.0),
        ("max_dc_power_w", float("nan")),
        ("min_mppt_voltage_v", float("nan")),
        ("max_mppt_voltage_v", float("inf")),
        ("startup_voltage_v", True),
        ("max_input_current_per_mppt_a", -0.1),
        ("max_short_circuit_current_per_mppt_a", float("-inf")),
    ],
)
def test_inverter_rejects_invalid_supplied_datasheet_quantities(field, value):
    with pytest.raises(ValueError, match=field):
        InverterConfig(**{field: value})


def test_inverter_rejects_inverted_mppt_voltage_window():
    with pytest.raises(ValueError, match="min_mppt_voltage_v must not exceed max_mppt_voltage_v"):
        InverterConfig(min_mppt_voltage_v=500.0, max_mppt_voltage_v=120.0)


def test_inverter_accepts_mppt_voltage_ceiling_equal_to_dc_voltage_ceiling():
    config = InverterConfig(max_dc_voltage_v=600.0, max_mppt_voltage_v=600.0)

    assert config.max_mppt_voltage_v == config.max_dc_voltage_v


def test_inverter_rejects_mppt_voltage_ceiling_above_dc_voltage_ceiling():
    with pytest.raises(ValueError, match="max_mppt_voltage_v must not exceed max_dc_voltage_v"):
        InverterConfig(max_dc_voltage_v=600.0, max_mppt_voltage_v=600.1)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("min_mppt_voltage_v", 600.1),
        ("startup_voltage_v", 600.1),
    ],
)
def test_inverter_rejects_mppt_limits_above_dc_voltage_ceiling_without_mppt_maximum(field, value):
    with pytest.raises(ValueError, match=f"{field} must not exceed max_dc_voltage_v"):
        InverterConfig(max_dc_voltage_v=600.0, **{field: value})


def test_inverter_accepts_startup_voltage_below_the_mppt_window():
    """Fronius Primo-style datasheet: startup well under the MPP range minimum."""
    config = InverterConfig(
        max_dc_voltage_v=1000.0,
        min_mppt_voltage_v=240.0,
        max_mppt_voltage_v=800.0,
        startup_voltage_v=80.0,
    )

    assert config.startup_voltage_v == 80.0


def test_inverter_accepts_startup_voltage_above_the_mppt_window():
    config = InverterConfig(
        max_dc_voltage_v=1000.0,
        min_mppt_voltage_v=240.0,
        max_mppt_voltage_v=800.0,
        startup_voltage_v=850.0,
    )

    assert config.startup_voltage_v == 850.0


@pytest.mark.parametrize("field", ["mppt_channels", "max_strings_per_mppt"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_inverter_rejects_non_positive_or_non_integer_channel_counts(field, value):
    with pytest.raises(ValueError, match=field):
        InverterConfig(**{field: value})


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


def test_package_all_exports_stable_inverter_helpers():
    expected = {
        "calculate_dc_ac_power",
        "InverterConversionResult",
    }

    assert expected.issubset(set(breos.__all__))
