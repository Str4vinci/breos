"""Correctness-release tests for the public App configuration boundary."""

import math

import pytest

from breos.app import App
from breos.app_config import merge_defaults, validate_config


def _config(**overrides):
    return {
        "location": "porto",
        "n_modules": 10,
        "annual_consumption_kwh": 4000,
        **overrides,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("annual_consumption_kwh", math.nan),
        ("battery_kwh", math.inf),
        ("inverter_efficiency", "0.96"),
        ("inverter_loading_ratio", True),
        ("inflation_rate", -1.0),
        ("discount_rate", -1.0),
        ("projection_years", 2.5),
        ("battery_max_charge_power_w", -1.0),
        ("battery_max_discharge_power_w", math.inf),
        ("export_emissions_factor_gco2_kwh", -1.0),
    ],
)
def test_invalid_numeric_config_fails_at_app_boundary(field, value):
    with pytest.raises((TypeError, ValueError), match=field):
        App(_config(**{field: value}))


@pytest.mark.parametrize(
    "location",
    [
        {"latitude": 91, "longitude": 0, "timezone": "UTC"},
        {"latitude": 0, "longitude": -181, "timezone": "UTC"},
        {"latitude": 0, "longitude": 0, "timezone": "Mars/Olympus"},
    ],
)
def test_invalid_custom_location_fails_early(location):
    with pytest.raises((TypeError, ValueError)):
        App(_config(location=location))


def test_invalid_start_date_and_calendar_model_fail_early():
    with pytest.raises(ValueError, match="start_date"):
        App(_config(start_date="2024-02-30"))
    with pytest.raises(ValueError, match="calendar_model"):
        App(_config(calendar_model="invented"))


def test_unknown_top_level_and_array_modules_fail_early():
    with pytest.raises(ValueError, match="Unknown PV module"):
        App(_config(pv_module="Imaginary_900W"))
    with pytest.raises(ValueError, match=r"pv_arrays\[0\]"):
        App(
            _config(
                pv_arrays=[{"modules": 4, "module": "Imaginary_900W"}],
            )
        )


def test_ac_coupling_is_an_explicitly_unsupported_public_configuration():
    with pytest.raises(NotImplementedError, match="DC-coupled"):
        App(_config(dc_coupled=False))
    with pytest.raises(TypeError, match="dc_coupled"):
        App(_config(dc_coupled="true"))


@pytest.mark.parametrize(
    ("overrides", "error_type", "message"),
    [
        (
            {"location": {"latitude": 91, "longitude": 0, "timezone": "UTC"}},
            ValueError,
            "'location.latitude' must be between -90 and 90",
        ),
        ({"pv_arrays": [{"modules": 0}]}, ValueError, "'pv_arrays[0].modules' must be >= 1"),
        (
            {"inverter_efficiency": 0},
            ValueError,
            "'inverter_efficiency' must be between 0 (exclusive) and 1 (inclusive)",
        ),
        ({"resolution": "30min"}, ValueError, "'resolution' must be 'h' or '15min'"),
        ({"discount_rate": -1}, ValueError, "'discount_rate' must be greater than -1"),
        (
            {"battery_min_soc": 0.9, "battery_max_soc": 0.2},
            ValueError,
            "'battery_min_soc' and 'battery_max_soc' must satisfy 0 <= min < max <= 1",
        ),
    ],
)
def test_validation_subsystems_preserve_public_error_contract(overrides, error_type, message):
    """Characterize exact public errors at each validation subsystem boundary."""
    cfg = merge_defaults(_config(**overrides))

    with pytest.raises(error_type) as exc_info:
        validate_config(cfg)

    assert str(exc_info.value) == message


def test_validation_preserves_blast_normalization_and_conflict_errors():
    cfg = merge_defaults(_config(degradation_engine=" BLAST ", blast_model="lfp_gr_250ah_prismatic", battery_kwh=5.0))

    validate_config(cfg)

    assert cfg["degradation_engine"] == "blast"

    invalid = merge_defaults(
        _config(degradation_engine="blast", blast_model="lfp_gr_250ah_prismatic", battery_kwh=5.0, montecarlo={})
    )
    with pytest.raises(ValueError) as exc_info:
        validate_config(invalid)
    assert str(exc_info.value) == "'degradation_engine=blast' is not supported with Monte Carlo yet"
