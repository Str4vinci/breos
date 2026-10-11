"""A product's nameplate is shared by its catalog and aggregate simulation."""

from dataclasses import FrozenInstanceError, asdict

import pytest

from breos.app_config import resolve_app_config
from breos.equipment import InverterSpec, MpptSpec


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
    assert spec.mppts is None
    assert spec.max_dc_voltage_v is None
    assert spec.inverter_efficiency is None
    assert spec.isc_comparison_factor is None
    assert spec.as_app_config() == {"inverter_ac_rating_kw": 4.0, "inverter_loading_ratio": None}


def test_asymmetric_inputs_keep_their_own_limits_in_datasheet_order():
    spec = InverterSpec(
        nominal_power_w=8000,
        mppts=[
            MpptSpec(max_input_current_a=27, max_short_circuit_current_a=40.5, max_strings=2),
            MpptSpec(max_input_current_a=16.5, max_short_circuit_current_a=24.8, max_strings=1),
        ],
    )
    assert isinstance(spec.mppts, tuple)
    assert [m.max_strings for m in spec.mppts] == [2, 1]
    assert [m.max_input_current_a for m in spec.mppts] == [27, 16.5]
    assert hash(spec) == hash(InverterSpec(**{**asdict(spec), "mppts": spec.mppts}))


def test_an_input_count_can_be_known_before_its_limits():
    spec = InverterSpec(nominal_power_w=4000, mppts=(MpptSpec(),) * 2)
    assert spec.mppts is not None and len(spec.mppts) == 2


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
            "isc_comparison_factor",
        )
        for value in (True, 0, -1, float("nan"), float("inf"), "600")
    ]
    + [
        ("inverter_efficiency", 1.01),
        ("inverter_efficiency", 0),
        ("inverter_efficiency", True),
        ("isc_comparison_factor", 0.9),
        ("is_hybrid", 1),
        ("mppts", ()),
        ("mppts", [{"max_strings": 2}]),
        ("mppts", MpptSpec()),
    ],
)
def test_invalid_datasheet_values_are_refused(field, value):
    with pytest.raises(ValueError, match=field):
        InverterSpec(**{"nominal_power_w": 4000, field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        (field, value)
        for field in ("max_input_current_a", "max_short_circuit_current_a")
        for value in (True, 0, -1, float("nan"), float("inf"), "16")
    ]
    + [("max_strings", 0), ("max_strings", 1.5), ("max_strings", False)],
)
def test_invalid_input_limits_are_refused(field, value):
    with pytest.raises(ValueError, match=field):
        MpptSpec(**{field: value})


@pytest.mark.parametrize(
    "limits",
    [
        {"min_mppt_voltage_v": 500, "max_mppt_voltage_v": 400},
        {"max_dc_voltage_v": 400, "max_mppt_voltage_v": 500},
        {"max_dc_voltage_v": 400, "min_mppt_voltage_v": 500},
        {"max_dc_voltage_v": 400, "startup_voltage_v": 500},
    ],
)
def test_inconsistent_limits_are_refused(limits):
    with pytest.raises(ValueError):
        InverterSpec(nominal_power_w=4000, **limits)


def test_an_input_operating_current_above_its_short_circuit_rating_is_refused():
    with pytest.raises(ValueError, match="max_input_current_a"):
        MpptSpec(max_input_current_a=16, max_short_circuit_current_a=12)


def test_startup_can_be_below_the_mppt_window_and_specs_round_trip():
    spec = InverterSpec(
        nominal_power_w=4000,
        startup_voltage_v=80,
        min_mppt_voltage_v=240,
        max_mppt_voltage_v=600,
        max_dc_voltage_v=1000,
        mppts=(MpptSpec(max_input_current_a=13, max_short_circuit_current_a=20, max_strings=1),),
        isc_comparison_factor=1.25,
    )
    fields = asdict(spec)
    fields["mppts"] = tuple(MpptSpec(**m) for m in fields["mppts"])
    assert InverterSpec(**fields) == spec
    with pytest.raises(FrozenInstanceError):
        spec.nominal_power_w = 5000
