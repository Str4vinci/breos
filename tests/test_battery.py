"""Tests for the battery module."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from breos.battery import (
    BatteryConfig,
    _get_degradation_params,
    apply_indoor_temperature_model,
    simulate_energy_balance,
    update_battery_soh_cyclewise,
)
from breos.constants import LAM_EA_J_MOL, LAM_SOC_EXPONENT_N
from breos.degradation.engine import BlastEngine
from breos.economics import system_ac_production_power
from breos.inverter import calculate_dc_ac_power
from breos.solar import dc_to_ac


class TestBatteryConfig:
    def test_defaults(self):
        cfg = BatteryConfig(nominal_energy_wh=5000)
        assert cfg.max_soc == 0.90
        assert cfg.min_soc == 0.10
        assert cfg.eol_percentage == 0.70
        assert cfg.dc_coupled is True
        assert cfg.inverter_efficiency == 0.96
        assert cfg.battery_type == "lfp"
        assert cfg.calendar_model == "naumann_lam_field_calibrated"

    def test_eol_default_agrees_across_config_surfaces(self):
        # BatteryConfig, the App config DEFAULTS, and the optimizer's
        # battery-spec fallback used to disagree (0.80 / 0.70 / 0.8); the
        # replacement threshold must default to the same value everywhere.
        from breos.app_config import DEFAULTS
        from breos.optimization import _build_battery_config_from_spec

        direct = BatteryConfig(nominal_energy_wh=5000)
        from_spec = _build_battery_config_from_spec({}, nominal_energy_wh=5000)

        assert direct.eol_percentage == DEFAULTS["battery_eol_percentage"]
        assert from_spec.eol_percentage == DEFAULTS["battery_eol_percentage"]

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
        cfg = BatteryConfig(nominal_energy_wh=5000, battery_type="LFP")
        assert cfg.battery_type == "lfp"

    def test_non_lfp_battery_type_rejected(self):
        with pytest.raises(ValueError, match="supports only: lfp"):
            BatteryConfig(nominal_energy_wh=5000, battery_type="nmc")

    def test_ac_coupled_dispatch_rejected(self):
        with pytest.raises(NotImplementedError, match="AC-coupled"):
            BatteryConfig(nominal_energy_wh=5000, dc_coupled=False)

    @pytest.mark.parametrize("field", ["max_charge_power_w", "max_discharge_power_w"])
    @pytest.mark.parametrize("value", [-1.0, float("inf"), float("nan")])
    def test_power_caps_must_be_finite_and_non_negative(self, field, value):
        with pytest.raises(ValueError, match=field):
            BatteryConfig(nominal_energy_wh=5000, **{field: value})

    def test_zero_power_caps_are_valid(self):
        cfg = BatteryConfig(nominal_energy_wh=5000, max_charge_power_w=0.0, max_discharge_power_w=0.0)
        assert cfg.max_charge_power_w == 0.0
        assert cfg.max_discharge_power_w == 0.0

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"nominal_energy_wh": -1.0}, "nominal_energy_wh"),
            ({"nominal_energy_wh": float("inf")}, "nominal_energy_wh"),
            ({"nominal_energy_wh": True}, "not a bool"),
            ({"nominal_energy_wh": 1000.0, "initial_soh": 101.0}, "initial_soh"),
            ({"nominal_energy_wh": 1000.0, "eol_percentage": -0.1}, "eol_percentage"),
            ({"nominal_energy_wh": 1000.0, "min_soc": 0.9, "max_soc": 0.9}, "SOC limits"),
            ({"nominal_energy_wh": 1000.0, "min_soc": -0.1}, "SOC limits"),
            ({"nominal_energy_wh": 1000.0, "charge_efficiency": 0.0}, "charge_efficiency"),
            ({"nominal_energy_wh": 1000.0, "discharge_efficiency": 1.1}, "discharge_efficiency"),
            ({"nominal_energy_wh": 1000.0, "inverter_efficiency": float("nan")}, "inverter_efficiency"),
            ({"nominal_energy_wh": 1000.0, "standby_loss_wh": -1.0}, "standby_loss_wh"),
            ({"nominal_energy_wh": 1000.0, "thermal_resistance_kw": -1.0}, "thermal_resistance_kw"),
            ({"nominal_energy_wh": 1000.0, "max_charge_power_w": True}, "max_charge_power_w"),
            ({"nominal_energy_wh": 1000.0, "dc_coupled": "true"}, "dc_coupled"),
        ],
    )
    def test_physical_configuration_validation(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            BatteryConfig(**kwargs)

    def test_cycle_aging_rejects_non_lfp_battery_type(self):
        soc = pd.Series([0.1, 0.8, 0.2], index=pd.date_range("2025-01-01", periods=3, freq="h"))

        with pytest.raises(ValueError, match="supports only: lfp"):
            update_battery_soh_cyclewise(1.0, soc, 5000.0, battery_type="nca")

    def test_field_calibrated_default_is_v1(self):
        assert _get_degradation_params("naumann_lam_field_calibrated") == _get_degradation_params(
            "naumann_lam_field_calibrated_v1"
        )

    def test_field_calibrated_v2_uses_lam_fixed_params(self):
        default_k0, default_ea, default_b, default_n = _get_degradation_params("naumann_lam_field_calibrated")
        v2_k0, v2_ea, v2_b, v2_n = _get_degradation_params("naumann_lam_field_calibrated_v2")

        assert v2_k0 == pytest.approx(3.6244128958919006e-07)
        assert v2_k0 != default_k0
        assert v2_ea == LAM_EA_J_MOL
        assert v2_b == pytest.approx(0.7108636007561147)
        assert v2_b != default_b
        assert v2_n == LAM_SOC_EXPONENT_N
        assert v2_ea > default_ea
        assert v2_n > default_n


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

    def test_resistance_fade_derates_energy_loop_efficiency(self):
        # enable_resistance_fade must feed the effective RTE back into the
        # charge/discharge flows, not just record it: with substantial
        # resistance growth the same scenario delivers less battery energy.
        idx = pd.date_range("2025-01-01 00:00", periods=24, freq="h", tz="UTC")
        pv = pd.Series([0.0] * 8 + [2000.0] * 8 + [0.0] * 8, index=idx)
        houseload = pd.DataFrame({"Load": [800.0] * 8 + [0.0] * 8 + [800.0] * 8}, index=idx)

        def _run(**kwargs):
            config = BatteryConfig(
                nominal_energy_wh=5000,
                standby_loss_wh=0.0,
                enable_replacement=False,
                **kwargs,
            )
            results_df, *_ = simulate_energy_balance(
                pv_dc=pv,
                houseload=houseload,
                battery_config=config,
                freq="h",
                temperature_series=pd.Series(25.0, index=idx),
            )
            return results_df["Import_From_Grid"].sum()

        import_base = _run()
        import_faded = _run(enable_resistance_fade=True, initial_resistance_growth=1.0)

        # 1.0 relative resistance growth halves the round-trip efficiency
        assert import_faded > import_base

    def test_inverter_ac_capacity_clips_pv_and_export(self):
        idx = pd.date_range("2025-01-01 00:00", periods=4, freq="h", tz="UTC")
        pv_dc = pd.Series([0.0, 2000.0, 3000.0, 1000.0], index=idx)
        houseload = pd.DataFrame({"Load": 500.0}, index=idx)
        config = BatteryConfig(nominal_energy_wh=0, inverter_ac_capacity_w=1000.0)

        results_df, total_pv, *_ = simulate_energy_balance(
            pv_dc=pv_dc, houseload=houseload, battery_config=config, freq="h"
        )

        # AC production saturates at the inverter rating
        assert results_df["PV_Production"].max() == pytest.approx(1000.0)
        # Export uses the headroom left after serving the load
        assert results_df["Sell_To_Grid"].max() == pytest.approx(500.0)
        # total_pv sums the clipped AC production
        expected = sum(calculate_dc_ac_power(value, 1000.0, config.inverter_efficiency).ac_power_w for value in pv_dc)
        assert total_pv == pytest.approx(expected)

    def test_pv_only_dispatch_matches_public_dc_to_ac_curve(self):
        idx = pd.date_range("2025-01-01", periods=5, freq="h", tz="UTC")
        pv_dc = pd.Series([0.0, 50.0, 400.0, 900.0, 1400.0], index=idx)
        houseload = pd.DataFrame({"Load": 0.0}, index=idx)
        config = BatteryConfig(nominal_energy_wh=0, inverter_ac_capacity_w=1000.0)

        results_df, *_ = simulate_energy_balance(
            pv_dc=pv_dc,
            houseload=houseload,
            battery_config=config,
            freq="h",
        )
        expected = dc_to_ac(
            pv_dc,
            pv_peak_power_w=1250.0,
            inverter_loading_ratio=1.25,
            inverter_efficiency=config.inverter_efficiency,
        )

        np.testing.assert_allclose(results_df["PV_Production"], expected)
        np.testing.assert_allclose(results_df["PV_AC_Export"], expected)

    @pytest.mark.parametrize("nominal_energy_wh", [0.0, 1000.0])
    def test_negative_pv_input_is_clamped_before_dispatch(self, nominal_energy_wh):
        idx = pd.date_range("2025-01-01", periods=1, freq="h", tz="UTC")
        config = BatteryConfig(
            nominal_energy_wh=nominal_energy_wh,
            inverter_ac_capacity_w=1000.0,
            standby_loss_wh=0.0,
            enable_replacement=False,
        )

        results, *_ = simulate_energy_balance(
            pv_dc=pd.Series([-100.0], index=idx),
            houseload=pd.DataFrame({"Load": [500.0]}, index=idx),
            battery_config=config,
            temperature_series=pd.Series(25.0, index=idx),
        )

        assert results["PV_DC"].iloc[0] == 0.0
        assert results["PV_DC_To_Battery"].iloc[0] == 0.0
        assert results["PV_DC_To_Inverter"].iloc[0] == 0.0
        assert results["PV_AC_To_Load"].iloc[0] == 0.0
        assert results["Houseload"].iloc[0] == pytest.approx(
            results["Battery_AC_To_Load"].iloc[0] + results["Import_From_Grid"].iloc[0]
        )

    def test_battery_shares_inverter_cap_and_charges_from_clipped_dc(self):
        idx = pd.date_range("2025-01-01 00:00", periods=2, freq="h", tz="UTC")
        pv_dc = pd.Series([0.0, 3000.0], index=idx)
        houseload = pd.DataFrame({"Load": [2000.0, 0.0]}, index=idx)
        config = BatteryConfig(
            nominal_energy_wh=2000,
            inverter_ac_capacity_w=1000.0,
            standby_loss_wh=0.0,
            enable_replacement=False,
        )

        results_df, *_ = simulate_energy_balance(
            pv_dc=pv_dc,
            houseload=houseload,
            battery_config=config,
            freq="h",
            temperature_series=pd.Series(25.0, index=idx),
        )

        # Hour 0: battery discharge AC is capped at the rating; the rest imports
        assert results_df["Import_From_Grid"].iloc[0] == pytest.approx(1000.0)
        # Hour 1: DC surplus above the AC cap still charges the DC-coupled
        # battery back to full, while export clips at the rating
        assert results_df["Battery_Energy"].iloc[1] == pytest.approx(1800.0)
        assert results_df["Sell_To_Grid"].iloc[1] == pytest.approx(1000.0)

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

    def test_blast_degradation_engine_returns_final_state(self):
        idx = pd.date_range("2025-01-01 00:00", periods=48, freq="h", tz="UTC")
        pv_dc = pd.Series(0.0, index=idx)
        houseload = pd.DataFrame({"Load": 0.0}, index=idx)
        temperature = pd.Series(25.0, index=idx)
        config = BatteryConfig(nominal_energy_wh=5000, standby_loss_wh=0.0, enable_replacement=False)

        result = simulate_energy_balance(
            pv_dc=pv_dc,
            houseload=houseload,
            battery_config=config,
            freq="h",
            temperature_series=temperature,
            degradation_engine="blast",
            blast_model="lfp_gr_250ah_prismatic",
            return_degradation_state=True,
        )

        assert len(result) == 7
        results_df, _, summary_df, _, _, degradation_df, degradation_state = result
        assert len(degradation_df) == 2
        assert degradation_df["BLAST_Model"].tolist() == ["lfp_gr_250ah_prismatic"] * 2
        assert "BLAST_Degradation" in degradation_df.columns
        assert summary_df["Final SOH [%]"].iloc[0] == pytest.approx(degradation_df["SOH"].iloc[-1])
        assert degradation_state["degradation_engine"] == "blast"
        assert degradation_state["blast_model"] == "lfp_gr_250ah_prismatic"
        assert degradation_state["blast_engine"]["blast_model_key"] == "lfp_gr_250ah_prismatic"
        assert degradation_state["day_start_soc_absolute"] == pytest.approx(results_df["Battery_SOC_Absolute"].iloc[-1])

    def test_blast_degradation_state_threads_across_calls(self):
        idx = pd.date_range("2025-01-01 00:00", periods=48, freq="h", tz="UTC")
        pv_dc = pd.Series(0.0, index=idx)
        houseload = pd.DataFrame({"Load": 0.0}, index=idx)
        temperature = pd.Series(25.0, index=idx)
        config = BatteryConfig(nominal_energy_wh=5000, standby_loss_wh=0.0, enable_replacement=False)

        full_run = simulate_energy_balance(
            pv_dc=pv_dc,
            houseload=houseload,
            battery_config=config,
            freq="h",
            temperature_series=temperature,
            degradation_engine="blast",
            blast_model="lfp_gr_250ah_prismatic",
        )
        first_day = simulate_energy_balance(
            pv_dc=pv_dc.iloc[:24],
            houseload=houseload.iloc[:24],
            battery_config=config,
            freq="h",
            temperature_series=temperature.iloc[:24],
            degradation_engine="blast",
            blast_model="lfp_gr_250ah_prismatic",
            return_degradation_state=True,
        )
        *_, degradation_state = first_day
        second_day = simulate_energy_balance(
            pv_dc=pv_dc.iloc[24:],
            houseload=houseload.iloc[24:],
            battery_config=config,
            freq="h",
            temperature_series=temperature.iloc[24:],
            degradation_engine="blast",
            blast_model="lfp_gr_250ah_prismatic",
            initial_degradation_state=degradation_state,
        )

        full_degradation = full_run[-1]
        second_degradation = second_day[-1]
        assert second_degradation["SOH"].iloc[-1] == pytest.approx(full_degradation["SOH"].iloc[-1], abs=1e-12)

    def test_blast_split_preserves_stored_energy_and_pv_origin_inventory(self):
        idx = pd.date_range("2025-01-01 00:00", periods=48, freq="h", tz="UTC")
        pv_dc = pd.Series(([0.0] * 8 + [2500.0] * 8 + [0.0] * 8) * 2, index=idx)
        houseload = pd.DataFrame({"Load": ([400.0] * 8 + [0.0] * 8 + [400.0] * 8) * 2}, index=idx)
        temperature = pd.Series(25.0, index=idx)
        config = BatteryConfig(nominal_energy_wh=5000, standby_loss_wh=0.0, enable_replacement=False)

        full_results, *_, full_degradation = simulate_energy_balance(
            pv_dc=pv_dc,
            houseload=houseload,
            battery_config=config,
            freq="h",
            temperature_series=temperature,
            degradation_engine="blast",
            blast_model="lfp_gr_250ah_prismatic",
        )
        first_results, *_, degradation_state = simulate_energy_balance(
            pv_dc=pv_dc.iloc[:24],
            houseload=houseload.iloc[:24],
            battery_config=config,
            freq="h",
            temperature_series=temperature.iloc[:24],
            degradation_engine="blast",
            blast_model="lfp_gr_250ah_prismatic",
            return_degradation_state=True,
        )
        carried_energy = float(first_results["Battery_Energy_End"].iloc[-1])
        carried_pv_origin = float(first_results["Battery_PV_Origin_Energy_End"].iloc[-1])
        second_results, *_, second_degradation = simulate_energy_balance(
            pv_dc=pv_dc.iloc[24:],
            houseload=houseload.iloc[24:],
            battery_config=config,
            freq="h",
            temperature_series=temperature.iloc[24:],
            initial_energy_wh=carried_energy,
            initial_pv_origin_energy_wh=carried_pv_origin,
            degradation_engine="blast",
            blast_model="lfp_gr_250ah_prismatic",
            initial_degradation_state=degradation_state,
        )

        assert carried_energy < config.nominal_energy_wh * config.max_soc
        assert 0.0 < carried_pv_origin <= carried_energy
        assert second_results["Battery_Energy_Beginning"].iloc[0] == pytest.approx(carried_energy)
        assert second_results["Battery_PV_Origin_Energy_Beginning"].iloc[0] == pytest.approx(carried_pv_origin)
        for column in (
            "Battery_Energy_End",
            "Battery_PV_Origin_Energy_End",
            "Battery_AC_To_Load",
            "Battery_AC_To_Load_PV",
            "Import_From_Grid",
            "Sell_To_Grid",
        ):
            np.testing.assert_allclose(
                second_results[column],
                full_results[column].iloc[24:],
                rtol=0.0,
                atol=1e-9,
            )
        assert second_degradation["SOH"].iloc[-1] == pytest.approx(full_degradation["SOH"].iloc[-1], abs=1e-12)

    def test_blast_carried_energy_above_restored_soh_window_is_ledgered(self):
        first_idx = pd.date_range("2025-01-01 00:00", periods=24, freq="h", tz="UTC")
        config = BatteryConfig(nominal_energy_wh=5000, standby_loss_wh=0.0, enable_replacement=False)
        *_, degradation_state = simulate_energy_balance(
            pv_dc=pd.Series(0.0, index=first_idx),
            houseload=pd.DataFrame({"Load": 0.0}, index=first_idx),
            battery_config=config,
            freq="h",
            temperature_series=pd.Series(45.0, index=first_idx),
            degradation_engine="blast",
            blast_model="lfp_gr_250ah_prismatic",
            return_degradation_state=True,
        )
        restored_soh = degradation_state["blast_engine"]["outputs"]["q"][-1]
        assert restored_soh < 1.0

        second_idx = pd.date_range("2025-01-02 00:00", periods=1, freq="h", tz="UTC")
        initial_energy = config.nominal_energy_wh
        initial_pv_origin = initial_energy / 2.0
        results, *_ = simulate_energy_balance(
            pv_dc=pd.Series(0.0, index=second_idx),
            houseload=pd.DataFrame({"Load": 0.0}, index=second_idx),
            battery_config=config,
            freq="h",
            temperature_series=pd.Series(25.0, index=second_idx),
            initial_energy_wh=initial_energy,
            initial_pv_origin_energy_wh=initial_pv_origin,
            degradation_engine="blast",
            blast_model="lfp_gr_250ah_prismatic",
            initial_degradation_state=degradation_state,
        )

        restored_emax = config.nominal_energy_wh * restored_soh * config.max_soc
        assert results["Battery_Energy_Beginning"].iloc[0] == pytest.approx(initial_energy)
        assert results["Capacity_Window_Loss"].iloc[0] == pytest.approx(initial_energy - restored_emax)
        assert results["Battery_Energy_End"].iloc[0] == pytest.approx(restored_emax)
        assert results["Battery_PV_Origin_Energy_End"].iloc[0] == pytest.approx(
            initial_pv_origin * restored_emax / initial_energy
        )

    def test_blast_rejects_native_resistance_fade_at_energy_balance_boundary(self):
        idx = pd.date_range("2025-01-01 00:00", periods=1, freq="h", tz="UTC")
        config = BatteryConfig(nominal_energy_wh=5000, enable_resistance_fade=True)

        with pytest.raises(ValueError, match="cannot be combined with enable_resistance_fade"):
            simulate_energy_balance(
                pv_dc=pd.Series(0.0, index=idx),
                houseload=pd.DataFrame({"Load": 0.0}, index=idx),
                battery_config=config,
                freq="h",
                degradation_engine="blast",
                blast_model="lfp_gr_250ah_prismatic",
            )

    def test_blast_resistance_output_is_diagnostic_only(self):
        model_key = "nmc111_gr_sanyo_2ah"
        baseline_snapshot = BlastEngine(model_key).state_snapshot()
        perturbed_snapshot = deepcopy(baseline_snapshot)

        # Hold every capacity state/output constant while imposing a large,
        # internally consistent resistance increase on this resistance-capable model.
        perturbed_snapshot["states"]["rGain_t"][-1] = 3.0
        perturbed_snapshot["states"]["rGain_EFC"][-1] = 4.0
        perturbed_snapshot["outputs"]["r_t"][-1] = 4.0
        perturbed_snapshot["outputs"]["r_EFC"][-1] = 5.0
        perturbed_snapshot["outputs"]["r"][-1] = 8.0
        for capacity_key in ("q", "q_t", "q_EFC"):
            assert perturbed_snapshot["outputs"][capacity_key] == baseline_snapshot["outputs"][capacity_key]

        idx = pd.date_range("2025-01-01", periods=48, freq="h", tz="UTC")
        pv_dc = pd.Series(([0.0] * 8 + [2500.0] * 8 + [0.0] * 8) * 2, index=idx)
        houseload = pd.DataFrame({"Load": ([600.0] * 8 + [0.0] * 8 + [600.0] * 8) * 2}, index=idx)
        temperature = pd.Series(25.0, index=idx)
        config = BatteryConfig(nominal_energy_wh=5000, standby_loss_wh=0.0, enable_replacement=False)

        def run(snapshot):
            return simulate_energy_balance(
                pv_dc=pv_dc,
                houseload=houseload,
                battery_config=config,
                freq="h",
                temperature_series=temperature,
                degradation_engine="blast",
                blast_model=model_key,
                initial_degradation_state=snapshot,
                return_degradation_state=True,
            )

        baseline = run(baseline_snapshot)
        perturbed = run(perturbed_snapshot)
        baseline_results, _, baseline_summary, _, _, baseline_degradation, baseline_final_state = baseline
        perturbed_results, _, perturbed_summary, _, _, perturbed_degradation, perturbed_final_state = perturbed

        ledger_columns = (
            "PV_DC_To_Battery",
            "PV_DC_To_Inverter",
            "PV_DC_Curtailed",
            "PV_AC_To_Load",
            "PV_AC_Export",
            "Battery_Charge_Input",
            "Battery_Charge_Stored",
            "Battery_Discharge_DC",
            "Battery_AC_To_Load",
            "Battery_AC_To_Load_PV",
            "PV_Origin_Battery_AC_To_Load",
            "PV_Direct_Inverter_Loss",
            "Battery_Inverter_Loss",
            "Inverter_Loss",
            "Standby_Loss",
            "Capacity_Window_Loss",
            "Battery_Replacement_Energy_Removed",
            "Battery_Replacement_Energy_Added",
            "Battery_Energy_Delta",
        )
        pd.testing.assert_frame_equal(
            perturbed_results[list(ledger_columns)],
            baseline_results[list(ledger_columns)],
            check_exact=True,
        )

        economics_columns = ("Houseload", "Import_From_Grid", "Sell_To_Grid")
        pd.testing.assert_frame_equal(
            perturbed_results[list(economics_columns)],
            baseline_results[list(economics_columns)],
            check_exact=True,
        )
        pd.testing.assert_series_equal(
            system_ac_production_power(perturbed_results),
            system_ac_production_power(baseline_results),
            check_exact=True,
        )

        dispatch_and_capacity_columns = (
            "Battery_Energy_Beginning",
            "Battery_Energy_End",
            "Battery_PV_Origin_Energy_Beginning",
            "Battery_PV_Origin_Energy_End",
            "Battery_SOC_Normalized",
            "Battery_SOC_Absolute",
            "Battery_SOH",
        )
        pd.testing.assert_frame_equal(
            perturbed_results[list(dispatch_and_capacity_columns)],
            baseline_results[list(dispatch_and_capacity_columns)],
            check_exact=True,
        )
        pd.testing.assert_series_equal(perturbed_degradation["SOH"], baseline_degradation["SOH"], check_exact=True)
        assert perturbed_summary["Final SOH [%]"].iloc[0] == baseline_summary["Final SOH [%]"].iloc[0]
        for capacity_key in ("q", "q_t", "q_EFC"):
            assert (
                perturbed_final_state["blast_engine"]["outputs"][capacity_key]
                == baseline_final_state["blast_engine"]["outputs"][capacity_key]
            )
        assert (
            perturbed_final_state["blast_engine"]["outputs"]["r"][-1]
            > baseline_final_state["blast_engine"]["outputs"]["r"][-1] + 6.0
        )

    @pytest.mark.parametrize(
        ("contradictory_kwargs", "message"),
        [
            ({"blast_model": "lfp_gr_250ah_prismatic"}, "blast_model requires"),
            ({"initial_degradation_state": {}}, "initial_degradation_state requires"),
        ],
    )
    def test_native_energy_balance_rejects_blast_only_arguments(self, contradictory_kwargs, message):
        idx = pd.date_range("2025-01-01 00:00", periods=1, freq="h", tz="UTC")

        with pytest.raises(ValueError, match=message):
            simulate_energy_balance(
                pv_dc=pd.Series(0.0, index=idx),
                houseload=pd.DataFrame({"Load": 0.0}, index=idx),
                battery_config=BatteryConfig(nominal_energy_wh=5000),
                freq="h",
                degradation_engine="native",
                **contradictory_kwargs,
            )

    def test_blast_state_payload_carries_mid_swing_anchor(self):
        # Pin degradation-anchor bookkeeping under a profile that is still
        # mid-discharge at midnight: the carried day-start anchor must be the
        # true mid-swing SoC, and day-1 EFC must not depend on how long the run
        # continues afterwards. Stored-energy split continuity is covered by
        # the inventory-specific test above.
        idx = pd.date_range("2025-01-01 00:00", periods=48, freq="h", tz="UTC")
        pv_dc = pd.Series(([0.0] * 8 + [2500.0] * 8 + [0.0] * 8) * 2, index=idx)
        houseload = pd.DataFrame({"Load": ([400.0] * 8 + [0.0] * 8 + [400.0] * 8) * 2}, index=idx)
        temperature = pd.Series(25.0, index=idx)
        config = BatteryConfig(nominal_energy_wh=5000, standby_loss_wh=0.0, enable_replacement=False)

        results_df, _, _, _, _, full_degradation, _ = simulate_energy_balance(
            pv_dc=pv_dc,
            houseload=houseload,
            battery_config=config,
            freq="h",
            temperature_series=temperature,
            degradation_engine="blast",
            blast_model="lfp_gr_250ah_prismatic",
            return_degradation_state=True,
        )
        *_, first_state = simulate_energy_balance(
            pv_dc=pv_dc.iloc[:24],
            houseload=houseload.iloc[:24],
            battery_config=config,
            freq="h",
            temperature_series=temperature.iloc[:24],
            degradation_engine="blast",
            blast_model="lfp_gr_250ah_prismatic",
            return_degradation_state=True,
        )

        soc_abs = results_df["Battery_SOC_Absolute"]
        # The boundary is genuinely mid-swing, not saturated at either limit.
        assert abs(soc_abs.iloc[24] - soc_abs.iloc[23]) > 0.01
        anchor = first_state["day_start_soc_absolute"]
        assert abs(anchor - config.max_soc) > 0.05
        assert abs(anchor - config.min_soc) > 0.05
        assert anchor == pytest.approx(soc_abs.iloc[23], abs=1e-12)

        day_1_efc = first_state["blast_engine"]["stressors"]["efc"][-1]
        assert day_1_efc > 0.0
        assert day_1_efc == pytest.approx(full_degradation["Cumulative_FEC"].iloc[0], abs=1e-12)

    def test_blast_cumulative_fec_tracks_engine_efc(self):
        idx = pd.date_range("2025-01-01 00:00", periods=48, freq="h", tz="UTC")
        pv_dc = pd.Series(([0.0] * 8 + [2500.0] * 8 + [0.0] * 8) * 2, index=idx)
        houseload = pd.DataFrame({"Load": ([900.0] * 8 + [0.0] * 8 + [900.0] * 8) * 2}, index=idx)
        temperature = pd.Series(25.0, index=idx)
        config = BatteryConfig(nominal_energy_wh=5000, standby_loss_wh=0.0, enable_replacement=False)

        *_, degradation_df, degradation_state = simulate_energy_balance(
            pv_dc=pv_dc,
            houseload=houseload,
            battery_config=config,
            freq="h",
            temperature_series=temperature,
            degradation_engine="blast",
            blast_model="lfp_gr_250ah_prismatic",
            return_degradation_state=True,
        )

        engine_efc = degradation_state["blast_engine"]["stressors"]["efc"][-1]
        assert degradation_df["Cumulative_FEC"].iloc[-1] > 0.0
        assert degradation_df["Cumulative_FEC"].iloc[-1] == pytest.approx(engine_efc, abs=1e-12)

    def test_blast_replacement_resets_engine_state(self):
        idx = pd.date_range("2025-01-01 00:00", periods=24, freq="h", tz="UTC")
        pv_dc = pd.Series(0.0, index=idx)
        houseload = pd.DataFrame({"Load": 0.0}, index=idx)
        temperature = pd.Series(45.0, index=idx)
        config = BatteryConfig(
            nominal_energy_wh=5000,
            standby_loss_wh=0.0,
            eol_percentage=0.999999999,
            enable_replacement=True,
        )

        results_df, _, summary_df, _, n_replacements, degradation_df, degradation_state = simulate_energy_balance(
            pv_dc=pv_dc,
            houseload=houseload,
            battery_config=config,
            freq="h",
            temperature_series=temperature,
            degradation_engine="blast",
            blast_model="lfp_gr_250ah_prismatic",
            return_degradation_state=True,
        )

        assert n_replacements == 1
        assert bool(results_df["Battery_Replaced"].iloc[-1]) is True
        assert degradation_df["SOH"].iloc[-1] == pytest.approx(100.0)
        assert degradation_df["Cumulative_FEC"].iloc[-1] == pytest.approx(0.0)
        assert summary_df["Final SOH [%]"].iloc[0] == pytest.approx(100.0)
        assert degradation_state["blast_engine"]["outputs"]["q"][-1] == pytest.approx(1.0)
        assert degradation_state["blast_engine"]["stressors"]["efc"][-1] == pytest.approx(0.0)
        assert degradation_state["day_start_soc_absolute"] == pytest.approx(config.max_soc)


class TestEnergyLedger:
    @staticmethod
    def _run(pv_w, load_w, freq="h", **config_kwargs):
        idx = pd.date_range("2025-01-01", periods=len(pv_w), freq=freq, tz="UTC")
        config_kwargs.setdefault("nominal_energy_wh", 4000.0)
        config_kwargs.setdefault("standby_loss_wh", 0.0)
        config_kwargs.setdefault("enable_replacement", False)
        config = BatteryConfig(**config_kwargs)
        results, total_pv, *_ = simulate_energy_balance(
            pv_dc=pd.Series(pv_w, index=idx, dtype=float),
            houseload=pd.DataFrame({"Load": load_w}, index=idx, dtype=float),
            battery_config=config,
            temperature_series=pd.Series(25.0, index=idx),
            freq=freq,
        )
        return results, total_pv, config

    @staticmethod
    def _assert_conservation(results, config):
        np.testing.assert_allclose(
            results["PV_DC"],
            results["PV_DC_To_Battery"] + results["PV_DC_To_Inverter"] + results["PV_DC_Curtailed"],
            atol=1e-8,
        )
        np.testing.assert_allclose(
            results["Houseload"],
            results["PV_AC_To_Load"] + results["Battery_AC_To_Load"] + results["Import_From_Grid"],
            atol=1e-8,
        )
        np.testing.assert_allclose(results["PV_AC_Export"], results["Sell_To_Grid"], atol=1e-8)
        np.testing.assert_allclose(
            results["Battery_Charge_Stored"],
            results["Battery_Charge_Input"] * config.charge_efficiency,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            results["Battery_Energy_Delta"],
            results["Battery_Charge_Stored"]
            - results["Battery_Discharge_DC"]
            - results["Standby_Loss"]
            - results["Capacity_Window_Loss"]
            - results["Battery_Replacement_Energy_Removed"]
            + results["Battery_Replacement_Energy_Added"],
            atol=1e-8,
        )
        np.testing.assert_allclose(
            results["Inverter_Loss"],
            results["PV_Direct_Inverter_Loss"] + results["Battery_Inverter_Loss"],
            atol=1e-8,
        )
        # PV and replacement-added energy are the external inputs. Delivered
        # energy, losses, net battery movement, and energy removed with a
        # replaced pack are outputs.
        rhs = (
            results["PV_AC_To_Load"]
            + results["PV_AC_Export"]
            + results["Battery_AC_To_Load"]
            + results["PV_DC_Curtailed"]
            + results["Battery_Charge_Loss"]
            + results["Battery_Discharge_Loss"]
            + results["PV_Direct_Inverter_Loss"]
            + results["Battery_Inverter_Loss"]
            + results["Standby_Loss"]
            + results["Capacity_Window_Loss"]
            + results["Battery_Replacement_Energy_Removed"]
            + results["Battery_Energy_Delta"]
        )
        lhs = results["PV_DC"] + results["Battery_Replacement_Energy_Added"]
        np.testing.assert_allclose(lhs, rhs, atol=1e-7)

    @pytest.mark.parametrize("freq,repeats", [("h", 1), ("15min", 4)])
    def test_per_step_and_annual_conservation(self, freq, repeats):
        pv = np.repeat([0.0, 7000.0, 2000.0, 0.0, 9000.0, 0.0], repeats)
        load = np.repeat([5000.0, 500.0, 3500.0, 2500.0, 500.0, 4000.0], repeats)
        results, total_pv, config = self._run(
            pv,
            load,
            freq=freq,
            inverter_ac_capacity_w=3000.0,
            max_charge_power_w=1200.0,
            max_discharge_power_w=900.0,
        )
        self._assert_conservation(results, config)
        hours = 0.25 if freq == "15min" else 1.0
        assert total_pv == pytest.approx(results["PV_Production"].sum() * hours)
        assert total_pv == pytest.approx(
            (results["PV_DC"] - results["PV_DC_Curtailed"] - results["PV_Direct_Inverter_Loss"]).sum() * hours
        )

    def test_charge_precedes_export_and_curtailment_is_unusable_dc(self):
        results, _, config = self._run(
            [0.0, 10000.0, 10000.0],
            [20000.0, 500.0, 500.0],
            nominal_energy_wh=2000.0,
            inverter_ac_capacity_w=2000.0,
        )
        self._assert_conservation(results, config)
        assert results["PV_DC_To_Battery"].iloc[1] > 0.0
        assert results["PV_AC_Export"].iloc[1] > 0.0
        assert results["PV_DC_Curtailed"].iloc[1] > 0.0
        assert results["PV_DC_Curtailed"].iloc[2] >= results["PV_DC_Curtailed"].iloc[1]

    def test_shared_inverter_headroom_and_power_caps(self):
        results, _, config = self._run(
            [0.0, 8000.0, 1000.0],
            [10000.0, 0.0, 5000.0],
            inverter_ac_capacity_w=2000.0,
            max_charge_power_w=1000.0,
            max_discharge_power_w=700.0,
        )
        self._assert_conservation(results, config)
        inverter_output = results["PV_AC_To_Load"] + results["PV_AC_Export"] + results["Battery_AC_To_Load"]
        assert inverter_output.max() <= 2000.0 + 1e-8
        assert results["PV_DC_To_Battery"].max() <= 1000.0 + 1e-8
        assert results["Battery_AC_To_Load"].max() <= 700.0 + 1e-8

        simultaneous = results["Battery_AC_To_Load"] > 0.0
        combined_dc = (
            results.loc[simultaneous, "PV_DC_To_Inverter"]
            + results.loc[simultaneous, "Battery_Discharge_DC"] * config.discharge_efficiency
        )
        expected_ac = combined_dc.map(
            lambda value: (
                calculate_dc_ac_power(
                    value,
                    config.inverter_ac_capacity_w,
                    config.inverter_efficiency,
                ).ac_power_w
            )
        )
        np.testing.assert_allclose(inverter_output.loc[simultaneous], expected_ac)

    def test_hourly_and_quarter_hour_energy_agree(self):
        pv_hourly = [0.0, 7000.0, 2000.0, 0.0, 9000.0, 0.0]
        load_hourly = [5000.0, 500.0, 3500.0, 2500.0, 500.0, 4000.0]
        kwargs = {
            "inverter_ac_capacity_w": 3000.0,
            "max_charge_power_w": 1200.0,
            "max_discharge_power_w": 900.0,
        }
        hourly, _, _ = self._run(pv_hourly, load_hourly, **kwargs)
        quarterly, _, _ = self._run(
            np.repeat(pv_hourly, 4),
            np.repeat(load_hourly, 4),
            freq="15min",
            **kwargs,
        )
        for column in ("Import_From_Grid", "Sell_To_Grid", "PV_DC_Curtailed"):
            assert hourly[column].sum() == pytest.approx(quarterly[column].sum() * 0.25)
        assert hourly["Battery_Energy_End"].iloc[-1] == pytest.approx(quarterly["Battery_Energy_End"].iloc[-1])

    def test_pv_origin_inventory_starts_zero_and_mixes_proportionally(self):
        results, _, config = self._run(
            [0.0, 5000.0, 0.0],
            [10000.0, 0.0, 1000.0],
            nominal_energy_wh=1000.0,
            charge_efficiency=0.9,
            discharge_efficiency=0.9,
            inverter_efficiency=0.8,
        )
        self._assert_conservation(results, config)
        assert results["Battery_PV_Origin_Energy_Beginning"].iloc[0] == 0.0
        assert results["Battery_PV_Origin_Energy_End"].iloc[1] > 0.0
        assert results["PV_Origin_Battery_AC_To_Load"].iloc[2] > 0.0
        assert results["PV_Origin_Battery_AC_To_Load"].iloc[2] <= results["Battery_AC_To_Load"].iloc[2]
        np.testing.assert_allclose(results["Battery_AC_To_Load_PV"], results["PV_Origin_Battery_AC_To_Load"])
        assert results["PV_Direct_Inverter_Loss"].sum() > 0.0
        assert results["Battery_Inverter_Loss"].sum() > 0.0
        np.testing.assert_allclose(results["Battery_Charge_Loss"], results["Battery_Charge_Input"] * 0.1)

    def test_standby_is_separate_from_capacity_window_loss(self):
        results, _, config = self._run([0.0], [0.0], standby_loss_wh=20.0)
        self._assert_conservation(results, config)
        assert results["Standby_Loss"].iloc[0] == pytest.approx(20.0)
        assert results["Battery_Standby_Loss"].iloc[0] == pytest.approx(20.0)
        assert results["Capacity_Window_Loss"].iloc[0] == 0.0
        assert results["Battery_Energy_Delta"].iloc[0] == pytest.approx(-20.0)

    def test_temperature_capacity_reduction_is_explicit_loss(self):
        idx = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
        config = BatteryConfig(nominal_energy_wh=10000.0, standby_loss_wh=0.0, enable_replacement=False)
        results, *_ = simulate_energy_balance(
            pv_dc=pd.Series(0.0, index=idx),
            houseload=pd.DataFrame({"Load": 0.0}, index=idx),
            battery_config=config,
            temperature_series=pd.Series([25.0, 0.0], index=idx),
            freq="h",
        )
        self._assert_conservation(results, config)
        assert results["Capacity_Window_Loss"].iloc[1] > 0.0
        assert results["Battery_Energy_End"].iloc[1] < results["Battery_Energy_Beginning"].iloc[1]

    def test_resistance_calendar_uses_daily_mean_cell_temperature(self, monkeypatch):
        import breos.battery as battery_module

        seen_temperatures = []

        def capture_temperature(resistance_growth, T_cell_C, **kwargs):
            seen_temperatures.append(T_cell_C)
            return resistance_growth, 0.0

        monkeypatch.setattr(battery_module, "update_battery_resistance_calendar", capture_temperature)
        idx = pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC")
        simulate_energy_balance(
            pv_dc=pd.Series(0.0, index=idx),
            houseload=pd.DataFrame({"Load": 0.0}, index=idx),
            battery_config=BatteryConfig(
                nominal_energy_wh=1000.0,
                enable_replacement=False,
                enable_resistance_fade=True,
                thermal_resistance_kw=0.0,
            ),
            temperature_series=pd.Series([10.0] * 12 + [30.0] * 12, index=idx),
            freq="h",
        )
        assert seen_temperatures == pytest.approx([20.0])

    def test_replacement_energy_closes_timestep_boundary(self, monkeypatch):
        import breos.battery as battery_module

        def force_eol(soh_start_fraction, cumulative_cal_seconds, **kwargs):
            return 0.5, soh_start_fraction - 0.5, cumulative_cal_seconds + 86400.0

        monkeypatch.setattr(battery_module, "update_battery_soh_calendar", force_eol)
        results, _, config = self._run(
            [0.0] * 24,
            [1000.0] * 24,
            nominal_energy_wh=4000.0,
            enable_replacement=True,
            eol_percentage=0.7,
        )
        self._assert_conservation(results, config)
        final = results.iloc[-1]
        assert bool(final["Battery_Replaced"])
        assert final["Battery_Replacement_Energy_Removed"] == pytest.approx(400.0)
        assert final["Battery_Replacement_Energy_Added"] == pytest.approx(3600.0)
        assert final["Battery_Energy_End"] == pytest.approx(3600.0)
        assert final["Battery_Energy_Delta"] == pytest.approx(3200.0)
        assert final["Battery_SOC_Normalized"] == pytest.approx(1.0)
        assert final["Battery_SOC_Absolute"] == pytest.approx(config.max_soc)
        assert final["Battery_SOH"] == pytest.approx(100.0)
        assert final["Battery_PV_Origin_Energy_End"] == 0.0

    @pytest.mark.parametrize(
        "initial_energy,initial_origin,match",
        [
            (1001.0, 0.0, "initial_energy_wh"),
            (-1.0, 0.0, "initial_energy_wh"),
            (500.0, 501.0, "initial_pv_origin_energy_wh"),
            (500.0, -1.0, "initial_pv_origin_energy_wh"),
            (float("nan"), 0.0, "initial_energy_wh"),
        ],
    )
    def test_invalid_initial_state_cannot_create_energy(self, initial_energy, initial_origin, match):
        idx = pd.date_range("2025-01-01", periods=1, freq="h", tz="UTC")
        with pytest.raises(ValueError, match=match):
            simulate_energy_balance(
                pv_dc=pd.Series(0.0, index=idx),
                houseload=pd.DataFrame({"Load": 0.0}, index=idx),
                battery_config=BatteryConfig(nominal_energy_wh=1000.0, standby_loss_wh=0.0),
                initial_energy_wh=initial_energy,
                initial_pv_origin_energy_wh=initial_origin,
            )

    def test_zero_passed_state_is_not_raised_to_minimum_soc(self):
        idx = pd.date_range("2025-01-01", periods=1, freq="h", tz="UTC")
        results, *_ = simulate_energy_balance(
            pv_dc=pd.Series(0.0, index=idx),
            houseload=pd.DataFrame({"Load": 0.0}, index=idx),
            battery_config=BatteryConfig(nominal_energy_wh=1000.0, standby_loss_wh=0.0),
            initial_energy_wh=0.0,
            initial_pv_origin_energy_wh=0.0,
        )
        assert results["Battery_Energy_Beginning"].iloc[0] == 0.0
        assert results["Battery_Energy_End"].iloc[0] == 0.0

    def test_carried_energy_above_new_soh_window_becomes_capacity_loss(self):
        idx = pd.date_range("2025-01-01", periods=1, freq="h", tz="UTC")
        config = BatteryConfig(
            nominal_energy_wh=1000.0,
            initial_soh=50.0,
            standby_loss_wh=0.0,
            enable_replacement=False,
        )
        results, *_ = simulate_energy_balance(
            pv_dc=pd.Series(0.0, index=idx),
            houseload=pd.DataFrame({"Load": 0.0}, index=idx),
            battery_config=config,
            temperature_series=pd.Series(25.0, index=idx),
            initial_energy_wh=700.0,
            initial_pv_origin_energy_wh=350.0,
        )
        assert results["Battery_Energy_Beginning"].iloc[0] == pytest.approx(700.0)
        assert results["Capacity_Window_Loss"].iloc[0] == pytest.approx(250.0)
        assert results["Battery_Energy_End"].iloc[0] == pytest.approx(450.0)
        assert results["Battery_Energy_Delta"].iloc[0] == pytest.approx(-250.0)
        assert results["Battery_PV_Origin_Energy_End"].iloc[0] == pytest.approx(225.0)
        self._assert_conservation(results, config)

    def test_passed_energy_and_pv_origin_continue_exactly(self):
        config = BatteryConfig(
            nominal_energy_wh=1000.0,
            standby_loss_wh=0.0,
            enable_replacement=False,
            charge_efficiency=0.9,
            discharge_efficiency=0.9,
            inverter_efficiency=0.8,
        )
        first_idx = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
        first, *_ = simulate_energy_balance(
            pv_dc=pd.Series([0.0, 5000.0, 0.0], index=first_idx),
            houseload=pd.DataFrame({"Load": [1000.0, 0.0, 500.0]}, index=first_idx),
            battery_config=config,
            temperature_series=pd.Series(25.0, index=first_idx),
        )
        carried_energy = first["Battery_Energy_End"].iloc[-1]
        carried_origin = first["Battery_PV_Origin_Energy_End"].iloc[-1]
        assert carried_origin > 0.0

        second_idx = pd.date_range("2025-01-01 03:00", periods=2, freq="h", tz="UTC")
        second, *_ = simulate_energy_balance(
            pv_dc=pd.Series(0.0, index=second_idx),
            houseload=pd.DataFrame({"Load": 0.0}, index=second_idx),
            battery_config=config,
            temperature_series=pd.Series(25.0, index=second_idx),
            initial_energy_wh=carried_energy,
            initial_pv_origin_energy_wh=carried_origin,
        )
        assert second["Battery_Energy_Beginning"].iloc[0] == pytest.approx(carried_energy)
        assert second["Battery_PV_Origin_Energy_Beginning"].iloc[0] == pytest.approx(carried_origin)
        assert second["Battery_Energy_End"].iloc[0] == pytest.approx(carried_energy)
        assert second["Battery_PV_Origin_Energy_End"].iloc[0] == pytest.approx(carried_origin)


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
