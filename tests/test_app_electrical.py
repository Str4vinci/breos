"""App runs a string-topology check when it is given specified equipment."""

import math
import re
import tomllib

import pytest

from breos import App
from breos.app_config import resolve_app_config
from breos.pv_modules import MODULES, PVModuleParams

# Trina Solar TSM-450NEG9R.28, datasheet TSM_EN_2024_C, PDF page 2:
# https://static.trinasolar.com/sites/default/files/Datasheet_Vertex%20S%2B_NEG9R.28_EN_2024_C_web.pdf
TRINA_450 = PVModuleParams(
    Mpp=450,
    Vmp=44.6,
    Imp=10.09,
    Voc=52.9,
    Isc=10.74,
    T_Pmax_pct=-0.29,
    T_Voc_pct=-0.24,
    T_Isc_pct=0.04,
    N_Cells=144,
    Max_System_Voltage=1500,
)

# Fronius Symo GEN24 6.0, technical data, retrieved 2026-10-11:
# https://www.fronius.com/en-gb/uk/solar-energy/installers-partners/technical-data/all-products/inverters/fronius-symo-gen24/fronius-symo-gen24-6-0
FRONIUS_GEN24_6 = {
    "nominal_power_w": 6000,
    "inverter_efficiency": 0.982,
    "max_dc_voltage_v": 1000,
    "max_dc_power_w": 9000,
    "min_mppt_voltage_v": 80,
    "max_mppt_voltage_v": 800,
    "startup_voltage_v": 80,
    "isc_comparison_factor": 1.0,
    "mppts": [
        {"max_input_current_a": 25, "max_short_circuit_current_a": 40, "max_strings": 2},
        {"max_input_current_a": 12.5, "max_short_circuit_current_a": 20, "max_strings": 1},
    ],
}

# Huawei SUN2000-4KTL-L1, European datasheet, PDF page 2:
# https://solar.huawei.com/admin/asset/v1/pro/view/fc2bdc8911d8462dbc1a88dae66314cc.pdf
HUAWEI_4KTL = {
    "nominal_power_w": 4000,
    "inverter_efficiency": 0.984,
    "max_dc_voltage_v": 600,
    "recommended_dc_power_w": 6000,
    "min_mppt_voltage_v": 90,
    "max_mppt_voltage_v": 560,
    "startup_voltage_v": 100,
    "isc_comparison_factor": 1.0,
    "mppts": [{"max_input_current_a": 12.5, "max_short_circuit_current_a": 18, "max_strings": 1}] * 2,
}

# The Isc comparison basis was not transcribed; 1 states a direct comparison for these tests.
PORTO = {"min_design_temp_c": -5, "max_ambient_temp_c": 40}


@pytest.fixture
def trina(monkeypatch):
    monkeypatch.setitem(MODULES, "Trina_TSM_450NEG9R28", TRINA_450)
    return "Trina_TSM_450NEG9R28"


@pytest.fixture
def east_west(trina):
    """Seven Trina modules east and seven west, one face per input of the Fronius inverter."""
    return {
        "location": "porto",
        "annual_consumption_kwh": 4000,
        "pv_arrays": [
            {"modules": 7, "module": trina, "tilt": 20, "azimuth": 90},
            {"modules": 7, "module": trina, "tilt": 20, "azimuth": 270},
        ],
        "inverters": [FRONIUS_GEN24_6],
        "pv_strings": [
            {"inverter": 0, "mppt": 0, "array": 0, "modules": 7},
            {"inverter": 0, "mppt": 1, "array": 1, "modules": 7},
        ],
        "electrical_environment": PORTO,
    }


def test_specified_inverters_size_the_ac_side(east_west):
    resolved = resolve_app_config(east_west)
    assert resolved.inverter_ac_capacity_w == 6000
    assert resolved.cfg["inverter_loading_ratio"] is None
    assert resolved.cfg["inverter_efficiency"] == 0.982
    assert resolved.cost_params.dc_ac_ratio == pytest.approx(14 * 450 / 6000)
    assert resolved.electrical is not None
    assert resolved.electrical.check.status == "pass"
    assert resolved.electrical.inverters[0].mppts[1].max_input_current_a == 12.5


def test_several_inverters_add_up_and_weight_their_efficiency(trina):
    resolved = resolve_app_config(
        {
            "location": "porto",
            "annual_consumption_kwh": 4000,
            "pv_arrays": [{"modules": 7, "module": trina}, {"modules": 10, "module": trina}],
            "inverters": [FRONIUS_GEN24_6, HUAWEI_4KTL],
            "pv_strings": [
                {"inverter": 0, "mppt": 0, "array": 0, "modules": 7},
                {"inverter": 1, "mppt": 0, "array": 1, "modules": 10},
            ],
            "electrical_environment": PORTO,
        }
    )
    assert resolved.inverter_ac_capacity_w == 10000
    assert resolved.cfg["inverter_efficiency"] == pytest.approx((6000 * 0.982 + 4000 * 0.984) / 10000)
    assert resolved.electrical.check.status == "pass"


def test_without_pv_arrays_the_single_array_is_array_0(trina):
    resolved = resolve_app_config(
        {
            "location": "porto",
            "annual_consumption_kwh": 4000,
            "n_modules": 10,
            "pv_module": trina,
            "inverters": [HUAWEI_4KTL],
            "pv_strings": [{"inverter": 0, "mppt": 0, "array": 0, "modules": 10}],
            "electrical_environment": PORTO,
        }
    )
    assert resolved.electrical.check.status == "pass"


def test_the_check_runs_without_changing_aggregate_energy(east_west, _patch_weather):
    specified = App(east_west)
    specified.simulate()
    nameplate = {
        key: value
        for key, value in east_west.items()
        if key not in ("inverters", "pv_strings", "electrical_environment")
    }
    aggregate = App({**nameplate, "inverter_ac_rating_kw": 6.0, "inverter_efficiency": 0.982})
    aggregate.simulate()

    result, reference = specified.result(), aggregate.result()
    assert result["electrical"] == {"fidelity": "aggregate", "status": "pass", "findings": []}
    for key in ("usable_ac_system_production_kwh", "curtailment_dc_kwh", "total_investment", "npv_savings"):
        assert result[key] == reference[key]
    assert reference["electrical"] == {"fidelity": "aggregate", "status": "not_evaluated", "findings": []}


def test_a_failing_topology_still_simulates_as_a_diagnostic(east_west, _patch_weather):
    east_west["pv_strings"] = [{"inverter": 0, "mppt": 1, "array": 0, "modules": 7, "parallel": 2}]
    east_west["pv_arrays"] = [{"modules": 14, "module": east_west["pv_arrays"][0]["module"]}]
    app = App(east_west)
    app.simulate()
    block = app.result()["electrical"]
    assert block["status"] == "fail"
    assert {f["code"] for f in block["findings"]} == {
        "too_many_strings",
        "isc_above_mppt_limit",
        "imp_above_mppt_limit",
    }
    finding = next(f for f in block["findings"] if f["code"] == "too_many_strings")
    assert finding == {
        "code": "too_many_strings",
        "severity": "error",
        "message": "MPPT 1 of inverter 0 has 2 strings, above its limit of 1.",
        "value": 2,
        "limit": 1,
        "inverter": 0,
        "mppt": 1,
        "string": None,
        "array": None,
        "field": None,
        "unit": "count",
        "limit_field": "max_strings",
    }


def test_toml_can_state_an_unbounded_limit(east_west):
    loaded = tomllib.loads("max_dc_voltage_v = inf\nmin_mppt_voltage_v = 0\n")
    east_west["inverters"] = [{**FRONIUS_GEN24_6, **loaded}]
    resolved = resolve_app_config(east_west)
    assert resolved.electrical.inverters[0].max_dc_voltage_v == math.inf
    assert resolved.electrical.check.status == "pass"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"pv_strings": None}, "go together; set 'pv_strings'"),
        ({"electrical_environment": None}, "set 'electrical_environment'"),
        ({"inverter_loading_ratio": 1.2}, "remove 'inverter_loading_ratio'"),
        ({"inverter_ac_rating_kw": 6.0}, "remove 'inverter_ac_rating_kw'"),
        ({"inverter_efficiency": 0.97}, "remove 'inverter_efficiency'"),
        ({"inverters": []}, "at least one inverter"),
        ({"inverters": [{**FRONIUS_GEN24_6, "colour": "red"}]}, "inverters[0].colour"),
        ({"inverters": [{"mppts": []}]}, "inverters[0].nominal_power_w"),
        ({"inverters": [{**FRONIUS_GEN24_6, "mppts": [{"max_strings": 0}]}]}, "inverters[0].mppts[0]"),
        ({"inverters": [{**FRONIUS_GEN24_6, "max_dc_voltage_v": -1}]}, "'inverters[0]': max_dc_voltage_v"),
        ({"inverters": [FRONIUS_GEN24_6, {"nominal_power_w": 4000}]}, "an inverter_efficiency, or none"),
        ({"pv_strings": [{"inverter": 0, "mppt": 0, "array": 0}]}, "pv_strings[0].modules"),
        ({"pv_strings": [{"inverter": 0, "mppt": 0, "array": 0, "modules": 0}]}, "'pv_strings[0]': modules"),
        ({"electrical_environment": {**PORTO, "mounting": "roof"}}, "'electrical_environment': mounting"),
    ],
)
def test_malformed_equipment_and_topology_are_refused(east_west, change, message):
    with pytest.raises((TypeError, ValueError), match=re.escape(message)):
        resolve_app_config({**east_west, **change})


def test_json_states_an_unbounded_limit_as_text_and_results_stay_strict_json(east_west, _patch_weather):
    import json

    east_west["inverters"] = [{**FRONIUS_GEN24_6, "max_dc_voltage_v": "inf"}]
    resolved = resolve_app_config(east_west)
    assert resolved.electrical.inverters[0].max_dc_voltage_v == math.inf
    app = App(east_west)
    app.simulate()
    result = app.result()
    json.dumps(result, allow_nan=False)
    assert result["provenance"]["resolved_config"]["inverters"][0]["max_dc_voltage_v"] == "inf"


def test_an_array_may_set_its_own_mounting_for_the_check(east_west):
    east_west["pv_arrays"][1]["mounting"] = "flush"
    east_west["pv_arrays"][1]["hot_cell_temp_c"] = 75
    resolved = resolve_app_config(east_west)
    assert resolved.pv_arrays[1]["mounting"] == "flush"
    assert resolved.electrical.check.status == "pass"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"mounting": "roof"}, "'pv_arrays[0]': mounting"),
        ({"temperature_model": "sapm"}, "'pv_arrays[0].temperature_model' must be one of"),
    ],
)
def test_malformed_array_thermal_settings_are_refused(east_west, change, message):
    east_west["pv_arrays"][0].update(change)
    with pytest.raises(ValueError, match=re.escape(message)):
        resolve_app_config(east_west)


def test_array_mounting_needs_a_string_topology(east_west):
    plain = {k: v for k, v in east_west.items() if k not in ("inverters", "pv_strings", "electrical_environment")}
    plain["pv_arrays"][0]["mounting"] = "flush"
    with pytest.raises(ValueError, match="applies to the string check"):
        resolve_app_config(plain)
