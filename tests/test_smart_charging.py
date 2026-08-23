"""Fixed-target smart-charging controller and battery integration tests."""

import json

import numpy as np
import pandas as pd
import pytest

from breos.app import App
from breos.battery import BatteryConfig, simulate_energy_balance
from breos.runners.app import _tariff_year_values
from breos.smart_charging import resolve_fixed_target_instructions
from breos.tariffs import TariffPrices, resolve_named_tariff


def _resolved_daily_tariff(index: pd.DatetimeIndex):
    return resolve_named_tariff(
        index,
        "pt_mainland_2026_daily_bi",
        TariffPrices(
            currency="EUR",
            import_prices={"off_peak": 0.10, "peak": 0.40},
            export_prices={"all": 0.05},
        ),
    )


def _instructions(index: pd.DatetimeIndex, *, target: float = 0.5, import_limit_w: float | None = 600.0):
    return resolve_fixed_target_instructions(
        _resolved_daily_tariff(index),
        target_usable_fraction=target,
        charge_periods=["off_peak"],
        discharge_periods=["peak"],
        grid_charge_efficiency=0.8,
        grid_import_limit_w=import_limit_w,
    )


def test_fixed_target_controller_resolves_period_aligned_immutable_instructions():
    index = pd.date_range("2026-01-05", periods=12, freq="h", tz="Europe/Lisbon")

    first = _instructions(index)
    second = _instructions(index)

    assert first.discharge_allowed[:8] == (False,) * 8
    assert first.discharge_allowed[8:] == (True,) * 4
    assert first.grid_charge_target_usable_fraction[:8] == (0.5,) * 8
    assert first.grid_charge_target_usable_fraction[8:] == (None,) * 4
    assert first.minimum_usable_fraction == (0.0,) * 12
    assert first.instruction_hash == second.instruction_hash
    assert len(first.instruction_hash) == 64


def test_fixed_target_controller_rejects_unknown_and_overlapping_periods():
    index = pd.date_range("2026-01-05", periods=2, freq="h", tz="Europe/Lisbon")
    tariff = _resolved_daily_tariff(index)

    with pytest.raises(ValueError, match="Unknown charge_periods period"):
        resolve_fixed_target_instructions(
            tariff,
            target_usable_fraction=0.5,
            charge_periods=["cheap"],
            discharge_periods=["peak"],
        )

    with pytest.raises(ValueError, match="periods overlap"):
        resolve_fixed_target_instructions(
            tariff,
            target_usable_fraction=0.5,
            charge_periods=["off_peak"],
            discharge_periods=["off_peak"],
        )


def test_grid_charging_obeys_target_import_limit_and_origin_accounting():
    index = pd.date_range("2026-01-05", periods=9, freq="h", tz="Europe/Lisbon")
    load = [200.0, 0.0, 300.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000.0]
    config = BatteryConfig(
        nominal_energy_wh=1000.0,
        min_soc=0.1,
        max_soc=0.9,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
        standby_loss_wh=0.0,
        enable_replacement=False,
        inverter_efficiency=1.0,
        inverter_ac_capacity_w=1000.0,
        max_charge_power_w=1000.0,
        max_discharge_power_w=1000.0,
    )

    results, *_ = simulate_energy_balance(
        pv_dc=pd.Series(0.0, index=index),
        houseload=pd.DataFrame({"Load": load}, index=index),
        battery_config=config,
        temperature_series=pd.Series(25.0, index=index),
        initial_energy_wh=100.0,
        dispatch_instructions=_instructions(index),
    )
    greedy, *_ = simulate_energy_balance(
        pv_dc=pd.Series(0.0, index=index),
        houseload=pd.DataFrame({"Load": load}, index=index),
        battery_config=config,
        temperature_series=pd.Series(25.0, index=index),
        initial_energy_wh=100.0,
    )

    assert results["Grid_AC_To_Battery"].iloc[0] == pytest.approx(400.0)
    assert results["Grid_AC_To_Battery"].iloc[1] == pytest.approx(100.0)
    assert results["Import_From_Grid"].iloc[0] == pytest.approx(600.0)
    assert results["Battery_Energy_End"].iloc[1] == pytest.approx(500.0)
    assert results["Battery_AC_To_Load"].iloc[2] == 0.0
    assert results["Grid_AC_To_Load"].iloc[2] == pytest.approx(300.0)
    assert results["Battery_AC_To_Load"].iloc[8] == pytest.approx(400.0)
    assert results["Grid_Origin_Battery_AC_To_Load"].iloc[8] == pytest.approx(400.0)
    assert results["PV_Origin_Battery_AC_To_Load"].iloc[8] == 0.0

    np.testing.assert_allclose(
        results["Import_From_Grid"],
        results["Grid_AC_To_Load"] + results["Grid_AC_To_Battery"],
    )
    np.testing.assert_allclose(
        results["Grid_Battery_Charge_Stored"],
        results["Grid_AC_To_Battery"] * 0.8,
    )
    np.testing.assert_allclose(
        results["Battery_Energy_Delta"],
        results["PV_Battery_Charge_Stored"]
        + results["Grid_Battery_Charge_Stored"]
        - results["Battery_Discharge_DC"]
        - results["Standby_Loss"]
        - results["Capacity_Window_Loss"]
        - results["Battery_Replacement_Energy_Removed"]
        + results["Battery_Replacement_Energy_Added"],
        atol=1e-8,
    )
    np.testing.assert_allclose(
        results["Battery_Energy_End"],
        results["Battery_PV_Origin_Energy_End"] + results["Battery_Grid_Origin_Energy_End"],
    )
    tariff = _resolved_daily_tariff(index)
    assert (
        _tariff_year_values(results, tariff, "h")["Tariff_Import_Cost_Base"]
        < _tariff_year_values(greedy, tariff, "h")["Tariff_Import_Cost_Base"]
    )


def test_grid_charging_does_not_run_while_pv_is_exported():
    index = pd.date_range("2026-01-05", periods=1, freq="h", tz="Europe/Lisbon")
    config = BatteryConfig(
        nominal_energy_wh=1000.0,
        min_soc=0.1,
        max_soc=0.9,
        standby_loss_wh=0.0,
        enable_replacement=False,
        inverter_efficiency=1.0,
        inverter_ac_capacity_w=1000.0,
        max_charge_power_w=100.0,
    )

    results, *_ = simulate_energy_balance(
        pv_dc=pd.Series([2000.0], index=index),
        houseload=pd.DataFrame({"Load": [0.0]}, index=index),
        battery_config=config,
        temperature_series=pd.Series(25.0, index=index),
        initial_energy_wh=100.0,
        dispatch_instructions=_instructions(index, target=1.0, import_limit_w=None),
    )

    assert results["PV_AC_Export"].iloc[0] > 0.0
    assert results["Grid_AC_To_Battery"].iloc[0] == 0.0


def test_grid_charging_uses_only_shared_inverter_headroom():
    index = pd.date_range("2026-01-05", periods=1, freq="h", tz="Europe/Lisbon")
    config = BatteryConfig(
        nominal_energy_wh=2000.0,
        min_soc=0.1,
        max_soc=0.9,
        standby_loss_wh=0.0,
        enable_replacement=False,
        charge_efficiency=1.0,
        inverter_efficiency=1.0,
        inverter_ac_capacity_w=1000.0,
        max_charge_power_w=1000.0,
    )

    results, *_ = simulate_energy_balance(
        pv_dc=pd.Series([800.0], index=index),
        houseload=pd.DataFrame({"Load": [1000.0]}, index=index),
        battery_config=config,
        temperature_series=pd.Series(25.0, index=index),
        initial_energy_wh=200.0,
        dispatch_instructions=_instructions(index, target=1.0, import_limit_w=None),
    )

    assert results["PV_AC_To_Load"].iloc[0] == pytest.approx(800.0)
    assert results["Grid_AC_To_Battery"].iloc[0] == pytest.approx(200.0)
    assert results["Grid_AC_To_Load"].iloc[0] == pytest.approx(200.0)


BASE_APP_CONFIG = {
    "location": "porto",
    "n_modules": 6,
    "annual_consumption_kwh": 3000,
    "battery_kwh": 5.0,
    "cost_preset": "residential_pt",
    "projection_years": 1,
}

APP_TARIFF = {
    "schedule": "pt_mainland_2026_daily_bi",
    "currency": "EUR",
    "import_prices": {"off_peak": 0.10, "peak": 0.40},
    "export_prices": {"all": 0.05},
}

FIXED_TARGET = {
    "mode": "fixed_target",
    "target_usable_fraction": 0.5,
    "charge_periods": ["off_peak"],
    "discharge_periods": ["peak"],
    "grid_charge_efficiency": 0.95,
    "grid_import_limit_w": 5000.0,
}


def test_app_rejects_fixed_target_without_tariff_or_battery():
    with pytest.raises(ValueError, match="requires a tariff"):
        App({**BASE_APP_CONFIG, "smart_charging": FIXED_TARGET})

    with pytest.raises(ValueError, match="requires battery_kwh > 0"):
        App(
            {
                **BASE_APP_CONFIG,
                "battery_kwh": 0.0,
                "tariff": APP_TARIFF,
                "smart_charging": FIXED_TARGET,
            }
        )


def test_app_normalizes_fixed_target_defaults_and_rejects_period_overlap():
    app = App(
        {
            **BASE_APP_CONFIG,
            "tariff": APP_TARIFF,
            "smart_charging": {
                "mode": "fixed-target",
                "target_usable_fraction": 0.25,
                "charge_periods": ["off_peak"],
                "discharge_periods": ["peak"],
            },
        }
    )
    assert app._cfg["smart_charging"] == {
        "mode": "fixed_target",
        "target_usable_fraction": 0.25,
        "charge_periods": ["off_peak"],
        "discharge_periods": ["peak"],
        "grid_charge_efficiency": 0.95,
        "grid_import_limit_w": None,
    }

    with pytest.raises(ValueError, match="periods overlap"):
        App(
            {
                **BASE_APP_CONFIG,
                "tariff": APP_TARIFF,
                "smart_charging": {
                    **FIXED_TARGET,
                    "discharge_periods": ["off_peak"],
                },
            }
        )


def test_disabled_smart_charging_preserves_greedy_physical_results(_patch_weather):
    greedy = App(BASE_APP_CONFIG)
    disabled = App({**BASE_APP_CONFIG, "smart_charging": {"mode": "disabled"}})
    greedy.simulate()
    disabled.simulate()

    greedy_result = greedy.result()
    disabled_result = disabled.result()
    for key in (
        "pv_dc_generation_kwh",
        "usable_ac_system_production_kwh",
        "self_consumption_kwh",
        "grid_import_kwh",
        "grid_export_kwh",
        "battery_soh_end_pct",
    ):
        assert disabled_result[key] == greedy_result[key]
    assert "smart_charging" not in disabled_result


def test_app_fixed_target_reports_grid_flows_provenance_and_tariff_costs(_patch_weather):
    app = App(
        {
            **BASE_APP_CONFIG,
            "tariff": APP_TARIFF,
            "smart_charging": FIXED_TARGET,
        }
    )
    app.simulate()
    result = app.result()

    smart = result["smart_charging"]
    assert smart["mode"] == "fixed_target"
    assert smart["terminal_convention"] == "physical_carry"
    assert smart["resolved_steps"] == 8760
    assert len(smart["instruction_hash"]) == 64
    assert smart["yearly"][0]["grid_charge_kwh"] > 0.0
    assert result["grid_charge_kwh"] == pytest.approx(smart["yearly"][0]["grid_charge_kwh"], abs=0.01)
    assert result["grid_origin_battery_ac_load_kwh"] > 0.0
    assert result["tariff"]["yearly"][0]["import_cost"] > 0.0
    assert result["provenance"]["ledger_schema_version"] == "2.0"
    assert result["provenance"]["smart_charging"]["instruction_hash"] == smart["instruction_hash"]
    assert result["yearly"][0]["grid_charge_kwh"] > 0.0
    assert any(month["grid_charge_kwh"] > 0.0 for month in result["monthly"])
    json.dumps(result)
