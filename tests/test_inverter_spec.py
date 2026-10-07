"""A product's nameplate is shared by its catalog and aggregate simulation."""

from dataclasses import FrozenInstanceError, asdict

import pytest

from breos.app_config import resolve_app_config
from breos.equipment import InverterSpec


def test_the_spec_sets_the_ac_rating_used_by_simulation_and_economics():
    spec = InverterSpec(nominal_power_w=4000, inverter_efficiency=0.975)
    resolved = resolve_app_config(
        {"location": "porto", "n_modules": 12, "annual_consumption_kwh": 4000, **spec.as_app_config()}
    )
    assert resolved.inverter_ac_capacity_w == 4000
    assert resolved.cfg["inverter_loading_ratio"] is None
    assert resolved.cfg["inverter_efficiency"] == 0.975
    assert resolved.cost_params.dc_ac_ratio == 12 * resolved.avg_module_power_w / 4000


def test_unknown_datasheet_values_stay_unknown_and_do_not_override_engine_defaults():
    spec = InverterSpec(nominal_power_w=4000)
    assert spec.mppt_channels is None
    assert spec.max_dc_voltage_v is None
    assert spec.inverter_efficiency is None
    assert spec.as_app_config() == {"inverter_ac_rating_kw": 4.0, "inverter_loading_ratio": None}


@pytest.mark.parametrize(
    "field,value",
    [
        (field, value)
        for field in (
            "nominal_power_w",
            "max_dc_voltage_v",
            "max_dc_power_w",
            "startup_voltage_v",
            "min_mppt_voltage_v",
            "max_mppt_voltage_v",
            "max_input_current_per_mppt_a",
            "max_short_circuit_current_per_mppt_a",
        )
        for value in (True, 0, -1, float("nan"), float("inf"), "600")
    ]
    + [
        ("inverter_efficiency", 1.01),
        ("inverter_efficiency", 0),
        ("inverter_efficiency", True),
        ("mppt_channels", 1.5),
        ("mppt_channels", False),
        ("max_strings_per_mppt", 0),
        ("is_hybrid", 1),
    ],
)
def test_invalid_datasheet_values_are_refused(field, value):
    with pytest.raises(ValueError, match=field):
        InverterSpec(**{"nominal_power_w": 4000, field: value})


@pytest.mark.parametrize(
    "limits",
    [
        {"min_mppt_voltage_v": 500, "max_mppt_voltage_v": 400},
        {"max_dc_voltage_v": 400, "max_mppt_voltage_v": 500},
        {"max_dc_voltage_v": 400, "min_mppt_voltage_v": 500},
        {"max_dc_voltage_v": 400, "startup_voltage_v": 500},
        {"max_input_current_per_mppt_a": 16, "max_short_circuit_current_per_mppt_a": 12},
    ],
)
def test_inconsistent_limits_are_refused(limits):
    with pytest.raises(ValueError):
        InverterSpec(nominal_power_w=4000, **limits)


def test_startup_can_be_below_the_mppt_window_and_specs_round_trip():
    spec = InverterSpec(
        nominal_power_w=4000,
        startup_voltage_v=80,
        min_mppt_voltage_v=240,
        max_mppt_voltage_v=600,
        max_dc_voltage_v=1000,
    )
    assert InverterSpec(**asdict(spec)) == spec
    with pytest.raises(FrozenInstanceError):
        spec.nominal_power_w = 5000
