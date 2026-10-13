"""String-level energy with several inverters, for systems without a battery (ADR 0007 D10)."""

import numpy as np
import pandas as pd
import pytest

from breos import App
from breos.app_config import resolve_app_config
from breos.inverter import InverterConversion, InverterUnits, _calculate_dc_ac_power_arrays, convert_units
from breos.montecarlo import MonteCarloSettings, run_montecarlo
from breos.pv_modules import MODULES
from tests.test_string_energy import FRONIUS_GEN24_6, PORTO, TRINA_450

# A small inverter, so the east face clips at the synthetic year's 800 W/m2
# peak: 7 modules give about 2.1 kW of DC there, above its 1.8 kW nameplate.
SMALL = {**FRONIUS_GEN24_6, "nominal_power_w": 1800, "inverter_efficiency": 0.97, "night_consumption_w": 2.0}
EAST = {"modules": 7, "module": "Trina_TSM_450NEG9R28", "tilt": 20, "azimuth": 90}
WEST = {"modules": 7, "module": "Trina_TSM_450NEG9R28", "tilt": 20, "azimuth": 270}


@pytest.fixture
def trina(monkeypatch):
    monkeypatch.setitem(MODULES, "Trina_TSM_450NEG9R28", TRINA_450)


def config(arrays, inverters, strings, **extra):
    return {
        "location": "porto",
        "annual_consumption_kwh": 4000,
        "battery_kwh": 0.0,
        "pv_arrays": arrays,
        "inverters": inverters,
        "pv_strings": strings,
        "electrical_environment": PORTO,
        "electrical_fidelity": "string",
        **extra,
    }


TWO_INVERTERS = config(
    [EAST, WEST],
    [SMALL, FRONIUS_GEN24_6],
    [
        {"inverter": 0, "mppt": 0, "array": 0, "modules": 7},
        {"inverter": 1, "mppt": 0, "array": 1, "modules": 7},
    ],
)


def timeseries(cfg):
    app = App(cfg)
    app.simulate()
    return app.timeseries(), app.result()


def test_each_inverter_converts_its_own_strings_as_it_would_alone(trina, _patch_weather):
    both, result = timeseries(TWO_INVERTERS)
    east, _ = timeseries(config([EAST], [SMALL], [{"inverter": 0, "mppt": 0, "array": 0, "modules": 7}]))
    west, _ = timeseries(config([WEST], [FRONIUS_GEN24_6], [{"inverter": 0, "mppt": 0, "array": 0, "modules": 7}]))
    for column in (
        "PV_DC",
        "PV_DC_Curtailed",
        "PV_Production",
        "PV_Direct_Inverter_Loss",
        "Inverter_Night_Consumption",
    ):
        np.testing.assert_allclose(both[column], east[column] + west[column], rtol=1e-9, atol=1e-9, err_msg=column)
    # The small inverter clips; the large one does not.
    assert east["PV_DC_Curtailed"].sum() > 0
    assert west["PV_DC_Curtailed"].sum() == 0
    rows = result["pv_loss_waterfall"]["inverter"]["inverters"]
    assert [row["inverter"] for row in rows] == [0, 1]
    assert rows[0]["clipping_kwh"] == pytest.approx(east["PV_DC_Curtailed"].sum() / 1000, abs=0.005)
    assert rows[1]["clipping_kwh"] == 0
    assert rows[0]["ac_output_kwh"] == pytest.approx(east["PV_Production"].sum() / 1000, abs=0.005)
    assert rows[0]["night_consumption_kwh"] > 0 and rows[1]["night_consumption_kwh"] == 0
    assert result["pv_loss_waterfall"]["inverter"]["night_consumption"]["power_w"] == 2.0


def test_one_inverter_of_the_combined_rating_would_not_clip(trina, _patch_weather):
    both, _ = timeseries(TWO_INVERTERS)
    combined = {**FRONIUS_GEN24_6, "nominal_power_w": 7800, "night_consumption_w": 2.0}
    one, _ = timeseries(
        config(
            [EAST, WEST],
            [combined],
            [
                {"inverter": 0, "mppt": 0, "array": 0, "modules": 7},
                {"inverter": 0, "mppt": 1, "array": 1, "modules": 7},
            ],
        )
    )
    np.testing.assert_allclose(both["PV_DC"], one["PV_DC"], rtol=1e-9)
    assert one["PV_DC_Curtailed"].sum() == 0
    assert both["PV_DC_Curtailed"].sum() > 0


def test_the_shared_ac_cable_loses_on_the_summed_output():
    index = pd.date_range("2025-06-01", periods=4, freq="h", tz="UTC")
    units = InverterUnits(
        ac_capacity_w=(3000.0, 3000.0),
        efficiency=(0.96, 0.96),
        conversions=(InverterConversion(), InverterConversion()),
        shares=pd.DataFrame({0: [0.5, 1.0, 0.0, 0.0], 1: [0.5, 0.0, 1.0, 0.0]}, index=index),
        ac_derate=pd.DataFrame({0: [1.0] * 4, 1: [1.0] * 4}, index=index),
        curve_rows=pd.DataFrame({0: [0.0] * 4, 1: [0.0] * 4}, index=index),
        feeder_cable_k_per_w=2e-6,
    )
    dc = np.array([4000.0, 2000.0, 2000.0, 0.0])
    converted = convert_units(dc, units)
    alone = [_calculate_dc_ac_power_arrays(dc * units.shares[i].to_numpy(), 3000.0, 0.96)[0] for i in (0, 1)]
    gross = alone[0] + alone[1]
    np.testing.assert_allclose(converted.feeder_cable_loss, 2e-6 * gross * gross)
    np.testing.assert_allclose(converted.ac, gross - 2e-6 * gross * gross)
    np.testing.assert_allclose(converted.conversion_loss, dc - converted.ac)


def test_each_inverter_draws_at_night_only_without_its_own_dc():
    index = pd.date_range("2025-06-01", periods=3, freq="h", tz="UTC")
    units = InverterUnits(
        ac_capacity_w=(3000.0, 3000.0),
        efficiency=(0.96, 0.96),
        conversions=(InverterConversion(night_consumption_w=5.0), InverterConversion(night_consumption_w=1.0)),
        shares=pd.DataFrame({0: [1.0, 0.0, 0.0], 1: [0.0, 1.0, 0.0]}, index=index),
        ac_derate=pd.DataFrame({0: [1.0] * 3, 1: [1.0] * 3}, index=index),
        curve_rows=pd.DataFrame({0: [0.0] * 3, 1: [0.0] * 3}, index=index),
    )
    converted = convert_units(np.array([1000.0, 1000.0, 0.0]), units)
    np.testing.assert_array_equal(converted.night, [1.0, 5.0, 6.0])


def test_own_cables_and_curves_belong_to_each_inverter(trina, _patch_weather):
    cable = {"length_m": 30, "cross_section_mm2": 4, "phases": 1, "voltage_v": 230}
    curve = {"output_fraction": [0.1, 0.5, 1.0], "efficiency": [[0.93, 0.965, 0.96]]}
    cfg = {
        **TWO_INVERTERS,
        "inverters": [{**SMALL, "ac_cable": cable, "efficiency_curve": curve}, FRONIUS_GEN24_6],
        "ac_cable": {**cable, "length_m": 5},
    }
    resolved = resolve_app_config(cfg)
    assert resolved.inverter_conversion is None
    _, result = timeseries(cfg)
    block = result["pv_loss_waterfall"]["inverter"]
    rows = block["inverters"]
    assert rows[0]["ac_cable_loss_kwh"] > 0 and rows[1]["ac_cable_loss_kwh"] == 0
    assert block["ac_cable"]["loss_kwh"] > 0


def test_a_battery_with_several_inverters_is_refused(trina):
    with pytest.raises(ValueError, match="2 inverters need a system without a battery"):
        resolve_app_config({**TWO_INVERTERS, "battery_kwh": 5.0})


def test_monte_carlo_refuses_several_string_inverters(trina):
    with pytest.raises(ValueError, match="runs in breos.App only"):
        run_montecarlo(TWO_INVERTERS, MonteCarloSettings(weather_file="unread.csv", n_runs=1))


def test_own_cables_with_several_inverters_need_string_level_energy(trina):
    cable = {"length_m": 30, "cross_section_mm2": 4, "phases": 1, "voltage_v": 230}
    cfg = {**TWO_INVERTERS, "electrical_fidelity": "aggregate", "inverters": [{**SMALL, "ac_cable": cable}, SMALL]}
    with pytest.raises(ValueError, match="needs electrical_fidelity = 'string'"):
        resolve_app_config(cfg)


def test_one_inverter_takes_its_cable_in_either_place(trina):
    cable = {"length_m": 30, "cross_section_mm2": 4, "phases": 1, "voltage_v": 230}
    single = config([EAST], [{**SMALL, "ac_cable": cable}], [{"inverter": 0, "mppt": 0, "array": 0, "modules": 7}])
    top = {**single, "inverters": [SMALL], "ac_cable": cable}
    assert resolve_app_config(single).ac_cable == resolve_app_config(top).ac_cable
    with pytest.raises(ValueError, match="not both"):
        resolve_app_config({**single, "ac_cable": cable})


def test_parallel_mismatch_inputs_name_their_inverter(trina, _patch_weather):
    cfg = config(
        [EAST, WEST, {**EAST, "azimuth": 180}],
        [SMALL, FRONIUS_GEN24_6],
        [
            {"inverter": 0, "mppt": 0, "array": 0, "modules": 7},
            {"inverter": 1, "mppt": 0, "array": 1, "modules": 7},
            {"inverter": 1, "mppt": 0, "array": 2, "modules": 7},
        ],
    )
    _, result = timeseries(cfg)
    inputs = result["pv_loss_waterfall"]["parallel_mismatch"]["inputs"]
    assert [(i["inverter"], i["mppt"], i["strings"]) for i in inputs] == [(1, 0, [1, 2])]


def test_temperature_derating_lowers_only_its_own_inverters_rating(trina, _patch_weather):
    derating = {"ambient_temp_c": [-40.0, 60.0], "max_output_fraction": [0.8, 0.8]}
    derated = {**SMALL, "temperature_derating": derating}
    both, result = timeseries({**TWO_INVERTERS, "inverters": [derated, FRONIUS_GEN24_6]})
    east, _ = timeseries(config([EAST], [derated], [{"inverter": 0, "mppt": 0, "array": 0, "modules": 7}]))
    west, _ = timeseries(config([WEST], [FRONIUS_GEN24_6], [{"inverter": 0, "mppt": 0, "array": 0, "modules": 7}]))
    for column in ("PV_DC_Curtailed", "PV_Production"):
        np.testing.assert_allclose(both[column], east[column] + west[column], rtol=1e-9, atol=1e-9, err_msg=column)
    rows = result["pv_loss_waterfall"]["inverter"]["inverters"]
    assert rows[0]["temperature_derating"]["min_output_fraction"] == 0.8
    assert "temperature_derating" not in rows[1]
