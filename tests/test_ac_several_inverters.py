"""An AC-coupled battery beside several PV inverters (ADR 0004 M8, step 5)."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from breos import App
from breos.app_config import resolve_app_config
from breos.battery import BatteryConfig, simulate_energy_balance
from breos.inverter import InverterConversion, InverterUnits
from tests.energy_conservation import assert_energy_conservation
from tests.test_multi_inverter import TWO_INVERTERS, timeseries, trina  # noqa: F401

_BACKENDS = [
    "python",
    pytest.param(
        "numba", marks=pytest.mark.skipif(not importlib.util.find_spec("numba"), reason="numba not installed")
    ),
]
# The PV inverters' side of the ledger, which the battery must not change (P1).
PV_SIDE = (
    "PV_DC",
    "PV_DC_To_Inverter",
    "PV_DC_Curtailed",
    "PV_Direct_Inverter_Loss",
    "PV_Production",
    "Inverter_Night_Consumption",
)
APP_AC = {
    "battery_kwh": 5.0,
    "battery_coupling": "ac",
    "battery_inverter_ac_rating_kw": 2.5,
    "battery_inverter_efficiency": 0.95,
    "costs": {"battery_inverter_cost_per_kw": 120.0},
}
TOU = {
    "schedule": "pt_mainland_2026_daily_bi",
    "currency": "EUR",
    "import_prices": {"peak": 0.28, "off_peak": 0.11},
    "export_prices": {"all": 0.05},
}


# --- the dispatch -------------------------------------------------------------


def _inputs(days=3):
    index = pd.date_range("2025-06-01", periods=24 * days, freq="h", tz="UTC")
    hours = np.arange(len(index)) % 24
    pv = pd.Series(np.clip(np.sin((hours - 6) / 12 * np.pi), 0.0, None) * 5000.0, index=index)
    load = pd.DataFrame({"Load": 500.0 + 900.0 * (hours >= 18)}, index=index)
    east = np.where(hours < 12, 0.7, 0.3)
    units = InverterUnits(
        ac_capacity_w=(1500.0, 3000.0),
        efficiency=(0.97, None),
        conversions=(InverterConversion(night_consumption_w=2.0), InverterConversion(night_consumption_w=1.0)),
        shares=pd.DataFrame({0: east, 1: 1.0 - east}, index=index),
        ac_derate=pd.DataFrame({0: 1.0, 1: 1.0}, index=index),
        curve_rows=pd.DataFrame({0: 0.0, 1: 0.0}, index=index),
        feeder_cable_k_per_w=1e-6,
    )
    return pv, load, pd.Series(25.0, index=index), units


def _run(battery, backend="python", **kwargs):
    pv, load, temperature, units = _inputs()
    return simulate_energy_balance(
        pv_dc=pv,
        houseload=load,
        battery_config=battery,
        freq="h",
        temperature_series=temperature,
        inverter_units=units,
        execution_backend=backend,
        **kwargs,
    )[0]


def _ac(**overrides):
    settings = {
        "nominal_energy_wh": 8000.0,
        "coupling": "ac",
        "battery_inverter_ac_capacity_w": 2500.0,
        "battery_inverter_efficiency": 0.95,
        **overrides,
    }
    return BatteryConfig(**settings)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_each_pv_inverter_converts_as_without_the_battery(backend):
    pv_only = _run(BatteryConfig(nominal_energy_wh=0.0), backend)
    battery = _ac()
    frame = _run(battery, backend)
    assert pv_only["PV_DC_Curtailed"].sum() > 0.0, "the small inverter must clip"
    assert pv_only["Inverter_Night_Consumption"].sum() > 0.0
    for column in PV_SIDE:
        np.testing.assert_array_equal(frame[column].to_numpy(), pv_only[column].to_numpy(), err_msg=column)
    assert frame["PV_AC_To_Battery"].sum() > 0.0
    assert frame["Battery_AC_To_Load"].sum() > 0.0
    assert (frame["PV_DC_To_Battery"] == 0.0).all()
    assert_energy_conservation(frame, battery)


def test_both_backends_agree_beside_several_inverters():
    pytest.importorskip("numba")
    python, compiled = _run(_ac(), "python"), _run(_ac(), "numba")
    pd.testing.assert_frame_equal(python, compiled, check_exact=False, rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_a_retired_pack_is_the_pv_only_run_with_several_inverters(backend):
    pv_only = _run(BatteryConfig(nominal_energy_wh=0.0), backend)
    retired = _run(_ac(), backend, battery_retired=True, initial_energy_wh=0.0)
    for column in (*PV_SIDE, "PV_AC_To_Load", "PV_AC_Export", "Import_From_Grid"):
        np.testing.assert_array_equal(retired[column].to_numpy(), pv_only[column].to_numpy(), err_msg=column)


def test_a_dc_coupled_battery_beside_several_inverters_is_refused():
    with pytest.raises(ValueError, match="a battery beside several inverters must be AC-coupled"):
        _run(BatteryConfig(nominal_energy_wh=8000.0))


# --- the App ------------------------------------------------------------------


def test_an_app_run_keeps_each_inverters_pv_and_charges_the_battery(trina, _patch_weather):  # noqa: F811
    pv_only, _ = timeseries(TWO_INVERTERS)
    frame, result = timeseries({**TWO_INVERTERS, **APP_AC})
    for column in PV_SIDE:
        np.testing.assert_array_equal(frame[column].to_numpy(), pv_only[column].to_numpy(), err_msg=column)
    assert frame["PV_AC_To_Battery"].sum() > 0.0
    waterfall = result["pv_loss_waterfall"]
    assert len(waterfall["inverter"]["inverters"]) == 2
    assert waterfall["battery_inverter"]["ac_capacity_kw"] == 2.5


def test_fixed_target_runs_and_the_planner_is_refused_beside_several_inverters(trina, _patch_weather):  # noqa: F811
    fixed = {
        "mode": "fixed_target",
        "target_usable_fraction": 0.5,
        "charge_periods": ["off_peak"],
        "discharge_periods": ["peak"],
    }
    frame, _ = timeseries({**TWO_INVERTERS, **APP_AC, "tariff": TOU, "smart_charging": fixed})
    assert frame["Grid_AC_To_Battery"].sum() > 0.0
    planner = {"mode": "daily_persistence", "charge_periods": ["off_peak"], "discharge_periods": ["peak"]}
    with pytest.raises(ValueError, match="plans with one PV inverter, not 2"):
        resolve_app_config({**TWO_INVERTERS, **APP_AC, "tariff": TOU, "smart_charging": planner})


def test_app_refuses_a_dc_battery_beside_several_inverters(trina):  # noqa: F811
    with pytest.raises(ValueError, match="a battery beside 2 inverters must be AC-coupled"):
        resolve_app_config({**TWO_INVERTERS, "battery_kwh": 5.0})
    App({**TWO_INVERTERS, **APP_AC})
