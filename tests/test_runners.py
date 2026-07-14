"""Tests for workflow runner module boundaries."""

from types import SimpleNamespace

import pandas as pd
import pytest

from breos.app_inputs import PreparedSimulationInputs
from breos.battery import BatteryConfig, simulate_energy_balance
from breos.runners import SimulationArtifacts, run_app_simulation
from breos.runners import app as app_runner
from breos.runners.app import SimulationArtifacts as AppSimulationArtifacts
from breos.runners.app import run_app_simulation as run_app_runner
from breos.solar import PVProductionBreakdown


def _pv_breakdown(pv: pd.Series) -> PVProductionBreakdown:
    zeros = pd.Series(0.0, index=pv.index)
    return PVProductionBreakdown(
        horizontal_reference_dc=pv,
        poa_global_dc=pv,
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


def test_app_runner_exports_are_available_from_runner_package():
    assert run_app_simulation is run_app_runner
    assert SimulationArtifacts is AppSimulationArtifacts


def test_app_runner_native_default_matches_explicit_native(monkeypatch):
    idx = pd.date_range("2025-01-01 00:00", periods=48, freq="h", tz="UTC")
    pv = pd.Series(([0.0] * 8 + [1800.0] * 8 + [0.0] * 8) * 2, index=idx)
    load = pd.DataFrame({"Load": ([600.0] * 12 + [1000.0] * 12) * 2}, index=idx)
    temperature = pd.Series(22.0, index=idx)
    inputs = PreparedSimulationInputs(
        weather=pd.DataFrame(index=idx),
        dc_system_base=pv,
        load_data=load,
        temperature_series=temperature,
        pv_breakdown=_pv_breakdown(pv),
    )

    monkeypatch.setattr(app_runner, "prepare_simulation_inputs", lambda cfg, resolved, deps: inputs)
    monkeypatch.setattr(app_runner, "build_costs_dict", lambda cfg, resolved: {"total_initial_cost": 0.0})
    monkeypatch.setattr(app_runner, "cost_analysis_projection", lambda **kwargs: pd.DataFrame())
    monkeypatch.setattr(app_runner, "calculate_lcoe_from_projection", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(app_runner, "find_payback_year", lambda projection: None)

    cfg = {
        "resolution": "h",
        "battery_kwh": 5.0,
        "projection_years": 2,
        "pv_degradation_rate": 0.0,
        "n_modules": 1,
        "inverter_loading_ratio": 1.25,
        "battery_rte": None,
        "battery_max_charge_power_w": None,
        "battery_max_discharge_power_w": None,
        "enable_resistance_fade": False,
        "battery_eol_percentage": 0.70,
        "battery_max_soc": 0.90,
        "battery_min_soc": 0.10,
        "dc_coupled": True,
        "inverter_efficiency": 0.96,
        "calendar_model": "naumann_lam_field_calibrated",
        "inflation_rate": 0.0,
        "sell_price_inflation": 0.0,
        "discount_rate": 0.0,
    }
    resolved = SimpleNamespace(
        cost_params=SimpleNamespace(battery_cost_per_kwh=500.0),
        avg_module_power_w=400.0,
        emissions_params=None,
    )

    default_artifacts = run_app_runner(cfg, resolved, deps=SimpleNamespace())
    native_artifacts = run_app_runner({**cfg, "degradation_engine": "native"}, resolved, deps=SimpleNamespace())

    pd.testing.assert_frame_equal(default_artifacts.yearly_df, native_artifacts.yearly_df, check_exact=True)
    pd.testing.assert_frame_equal(
        default_artifacts.first_year_results_df,
        native_artifacts.first_year_results_df,
        check_exact=True,
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
    monkeypatch.setattr(app_runner, "build_costs_dict", lambda cfg, resolved: {"total_initial_cost": 0.0})
    monkeypatch.setattr(app_runner, "cost_analysis_projection", lambda **kwargs: pd.DataFrame())
    monkeypatch.setattr(app_runner, "calculate_lcoe_from_projection", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(app_runner, "find_payback_year", lambda projection: None)
    real_battery_config = app_runner.BatteryConfig

    def _battery_config_without_standby(**kwargs):
        kwargs.setdefault("standby_loss_wh", 0.0)
        return real_battery_config(**kwargs)

    monkeypatch.setattr(app_runner, "BatteryConfig", _battery_config_without_standby)

    cfg = {
        "resolution": "h",
        "battery_kwh": 5.0,
        "projection_years": 20,
        "pv_degradation_rate": 0.0,
        "n_modules": 1,
        "inverter_loading_ratio": 1.25,
        "battery_rte": None,
        "battery_max_charge_power_w": None,
        "battery_max_discharge_power_w": None,
        "enable_resistance_fade": False,
        "battery_eol_percentage": 0.70,
        "battery_max_soc": 0.90,
        "battery_min_soc": 0.10,
        "dc_coupled": True,
        "inverter_efficiency": 0.96,
        "calendar_model": "naumann_lam_field_calibrated",
        "inflation_rate": 0.0,
        "sell_price_inflation": 0.0,
        "discount_rate": 0.0,
        "degradation_engine": "blast",
        "blast_model": "lfp_gr_250ah_prismatic",
    }
    resolved = SimpleNamespace(
        cost_params=SimpleNamespace(battery_cost_per_kwh=500.0),
        avg_module_power_w=400.0,
        emissions_params=None,
    )

    artifacts = run_app_runner(cfg, resolved, deps=SimpleNamespace())

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
    assert range_warnings
    assert len({warning["code"] for warning in range_warnings}) == len(range_warnings)
