"""Tests for the battery module."""

import numpy as np
import pandas as pd
import pytest

from breos.battery import (
    BatteryConfig,
    apply_indoor_temperature_model,
    simulate_energy_balance,
)


class TestBatteryConfig:
    def test_defaults(self):
        cfg = BatteryConfig(nominal_energy_wh=5000)
        assert cfg.max_soc == 0.90
        assert cfg.min_soc == 0.10
        assert cfg.dc_coupled is True
        assert cfg.inverter_efficiency == 0.96
        assert cfg.battery_type == "lfp"
        assert cfg.calendar_model == "naumann_lam_field_calibrated"

    def test_replacement_cost_auto(self):
        cfg = BatteryConfig(nominal_energy_wh=10000)
        # 10 kWh * 500 €/kWh = 5000
        assert cfg.replacement_cost == pytest.approx(5000.0, rel=0.01)

    def test_replacement_cost_zero_battery(self):
        cfg = BatteryConfig(nominal_energy_wh=0)
        assert cfg.replacement_cost == 0.0

    def test_replacement_cost_override(self):
        cfg = BatteryConfig(nominal_energy_wh=5000, replacement_cost=1000.0)
        assert cfg.replacement_cost == 1000.0

    def test_battery_type_accessible(self):
        cfg = BatteryConfig(nominal_energy_wh=5000, battery_type="lfp")
        assert cfg.battery_type == "lfp"


class TestSimulateEnergyBalance:
    def test_no_battery(self, dc_production, sample_load):
        results_df, total_pv, summary_df, rep_cost, n_rep, deg_df = simulate_energy_balance(
            pv_dc=dc_production * 6,
            houseload=sample_load,
            battery_config=None,
            freq="h",
        )
        assert isinstance(results_df, pd.DataFrame)
        assert total_pv > 0
        assert rep_cost == 0.0
        assert n_rep == 0

    def test_with_battery_returns_tuple(self, dc_production, sample_load, battery_config, temperature_series):
        result = simulate_energy_balance(
            pv_dc=dc_production * 6,
            houseload=sample_load,
            battery_config=battery_config,
            freq="h",
            temperature_series=temperature_series,
        )
        assert len(result) == 6
        results_df, total_pv, summary_df, rep_cost, n_rep, deg_df = result
        assert isinstance(results_df, pd.DataFrame)
        assert total_pv > 0

    def test_soc_within_bounds(self, dc_production, sample_load, battery_config, temperature_series):
        results_df, *_ = simulate_energy_balance(
            pv_dc=dc_production * 6,
            houseload=sample_load,
            battery_config=battery_config,
            freq="h",
            temperature_series=temperature_series,
        )
        if "SOC_Absolute" in results_df.columns:
            soc = results_df["SOC_Absolute"]
            # SOC should never exceed nominal capacity
            assert soc.max() <= battery_config.nominal_energy_wh * 1.01  # 1% tolerance

    def test_more_pv_means_more_independence(self, dc_production, sample_load, battery_config, temperature_series):
        gi_values = []
        for n_modules in [3, 10]:
            results_df, *_ = simulate_energy_balance(
                pv_dc=dc_production * n_modules,
                houseload=sample_load,
                battery_config=battery_config,
                freq="h",
                temperature_series=temperature_series,
            )
            total_load = results_df["Houseload"].sum()
            total_import = results_df["Import_From_Grid"].sum()
            gi = (1 - total_import / total_load) * 100 if total_load > 0 else 0
            gi_values.append(gi)
        assert gi_values[1] > gi_values[0]

    def test_load_year_remap_drops_feb_29_when_target_is_not_leap(self):
        pv_idx = pd.date_range("2025-02-28 00:00", periods=48, freq="h", tz="UTC")
        load_idx = pd.date_range("2024-02-28 00:00", periods=72, freq="h", tz="UTC")
        pv_dc = pd.Series(0.0, index=pv_idx)
        houseload = pd.DataFrame({"Load": 1000.0}, index=load_idx)

        results_df, _, summary_df, *_ = simulate_energy_balance(
            pv_dc=pv_dc,
            houseload=houseload,
            battery_config=None,
            freq="h",
        )

        assert len(results_df) == 48
        assert summary_df["Total Load [kWh]"].iloc[0] == pytest.approx(48.0)

    def test_cold_derate_cannot_sell_energy_pv_never_produced(self):
        # A full battery whose temperature drops sees Emax shrink below its
        # stored energy (lfp cold derate). charge_room went negative and the
        # battery silently drained into Sell_To_Grid: 100 Wh of PV exported
        # ~539 Wh in the first cold hour. Export must never exceed PV AC.
        idx = pd.date_range("2025-01-01 00:00", periods=6, freq="h", tz="UTC")
        pv_dc = pd.Series(100.0, index=idx)
        houseload = pd.DataFrame({"Load": 0.0}, index=idx)
        config = BatteryConfig(nominal_energy_wh=10000, battery_type="lfp")
        temperature = pd.Series([25.0, 25.0, 0.0, 0.0, 0.0, 0.0], index=idx)

        results_df, *_ = simulate_energy_balance(
            pv_dc=pv_dc,
            houseload=houseload,
            battery_config=config,
            temperature_series=temperature,
            freq="h",
        )

        pv_ac_max = 100.0 * config.inverter_efficiency
        assert results_df["Sell_To_Grid"].max() <= pv_ac_max + 1e-9
        # Stored energy is clamped into the temperature-derated window
        from breos.battery import lfp_capacity_factor

        emax_cold = 10000 * config.max_soc * lfp_capacity_factor(0.0)
        assert results_df["Battery_Energy"].iloc[-1] <= emax_cold + 1e-9


class TestIndoorTemperatureModel:
    def test_output_shape(self):
        outdoor = pd.Series(np.linspace(-5, 40, 100))
        indoor = apply_indoor_temperature_model(outdoor)
        assert len(indoor) == len(outdoor)

    def test_clamping_floor(self):
        outdoor = pd.Series([-20.0] * 10)
        indoor = apply_indoor_temperature_model(outdoor, floor_c=15.0)
        assert indoor.min() >= 15.0

    def test_clamping_ceiling(self):
        outdoor = pd.Series([50.0] * 10)
        indoor = apply_indoor_temperature_model(outdoor, ceiling_c=35.0)
        assert indoor.max() <= 35.0

    def test_coupling_factor(self):
        outdoor = pd.Series([10.0] * 10)
        # alpha=0 means fully setpoint, alpha=1 means fully outdoor
        indoor_low = apply_indoor_temperature_model(outdoor, setpoint_c=22.0, coupling_alpha=0.0)
        indoor_high = apply_indoor_temperature_model(outdoor, setpoint_c=22.0, coupling_alpha=1.0)
        assert indoor_low.iloc[0] > indoor_high.iloc[0]  # setpoint > outdoor, so less coupling = warmer
