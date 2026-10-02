"""Tests for the App runner's projection-year loop."""

from types import SimpleNamespace

import pandas as pd
import pytest

import breos.projection as projection_module
from breos.app_config import resolve_app_config
from breos.app_inputs import PreparedSimulationInputs
from breos.battery import BatteryConfig, simulate_energy_balance
from breos.runners import app as app_runner
from breos.solar import PVProductionBreakdown


def _empty_projection() -> pd.DataFrame:
    """A stand-in cost projection carrying the attrs the runners read."""
    projection = pd.DataFrame()
    projection.attrs.update({"lcoe_per_kwh": 0.0, "total_replacement_cost": 0.0})
    return projection


def _pv_breakdown(pv: pd.Series) -> PVProductionBreakdown:
    zeros = pd.Series(0.0, index=pv.index)
    return PVProductionBreakdown(
        horizontal_reference_dc=pv,
        poa_global_dc=pv,
        front_effective_irradiance_dc=pv,
        rear_gain_dc=zeros,
        effective_irradiance_dc=pv,
        module_dc=pv,
        dc_after_static_losses=pv,
        dc_after_losses=pv,
        pvwatts_component_losses={},
        pvwatts_components_pct={},
        pvwatts_combined_pct=0.0,
        age_degradation_pct=0.0,
        age_degradation_loss=zeros,
    )


def test_app_runner_threads_blast_state_across_projection_years(monkeypatch):
    idx = pd.date_range("2025-01-01 00:00", periods=24, freq="h", tz="UTC")
    one_day_pv = pd.Series(2000.0, index=idx)
    one_day_load = pd.DataFrame({"Load": 0.0}, index=idx)
    one_day_temperature = pd.Series(25.0, index=idx)
    inputs = PreparedSimulationInputs(
        weather=pd.DataFrame(index=idx),
        dc_system_base=one_day_pv,
        load_data=one_day_load,
        temperature_series=one_day_temperature,
        pv_breakdown=_pv_breakdown(one_day_pv),
    )

    monkeypatch.setattr(app_runner, "prepare_simulation_inputs", lambda cfg, resolved, deps: inputs)
    monkeypatch.setattr(
        projection_module,
        "build_costs_dict",
        lambda cfg, resolved: {
            "total_initial_cost": 0.0,
            "electricity_cost": 0.0,
            "electricity_sold_cost": 0.0,
            "daily_power_cost": 0.0,
        },
    )
    monkeypatch.setattr(projection_module, "cost_analysis_projection", lambda **kwargs: _empty_projection())
    monkeypatch.setattr(app_runner, "find_payback_year", lambda projection: None)
    real_battery_config = projection_module.BatteryConfig

    def _battery_config_without_standby(**kwargs):
        kwargs.setdefault("standby_loss_wh", 0.0)
        return real_battery_config(**kwargs)

    monkeypatch.setattr(projection_module, "BatteryConfig", _battery_config_without_standby)

    resolved = resolve_app_config(
        {
            "location": "porto",
            "n_modules": 1,
            "annual_consumption_kwh": 4000,
            "battery_kwh": 5.0,
            "projection_years": 20,
            "pv_degradation_rate": 0.0,
            "inflation_rate": 0.0,
            "sell_price_inflation": 0.0,
            "discount_rate": 0.0,
            "degradation_engine": "blast",
            "blast_model": "lfp_gr_250ah_prismatic",
        }
    )

    artifacts = app_runner.run_app_simulation(resolved, deps=SimpleNamespace())

    continuous_idx = pd.date_range("2025-01-01 00:00", periods=24 * 20, freq="h", tz="UTC")
    continuous_pv = pd.Series(2000.0, index=continuous_idx)
    continuous_load = pd.DataFrame({"Load": 0.0}, index=continuous_idx)
    continuous_temperature = pd.Series(25.0, index=continuous_idx)
    continuous_config = BatteryConfig(nominal_energy_wh=5000, standby_loss_wh=0.0, enable_replacement=True)
    *_, continuous_degradation = simulate_energy_balance(
        pv_dc=continuous_pv,
        houseload=continuous_load,
        battery_config=continuous_config,
        freq="h",
        temperature_series=continuous_temperature,
        degradation_engine="blast",
        blast_model="lfp_gr_250ah_prismatic",
    )

    assert artifacts.yearly_df["Battery_SOH_%"].iloc[-1] == pytest.approx(
        continuous_degradation["SOH"].iloc[-1],
        abs=1e-12,
    )
    range_warnings = artifacts.degradation_summary["experimental_range_warnings"]
    assert range_warnings == []
