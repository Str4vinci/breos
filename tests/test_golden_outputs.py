"""Golden-output tests for release-critical simulation paths."""

import pandas as pd
import pytest

from breos.battery import BatteryConfig, simulate_energy_balance
from breos.economics import CostParams, calculate_costs, cost_analysis_projection
from breos.emissions import EmissionsParams, calculate_co2_projection


def _assert_numeric_record(record, expected, *, rel=1e-9, abs=1e-9):
    for key, value in expected.items():
        assert record[key] == pytest.approx(value, rel=rel, abs=abs)


def _golden_battery_inputs():
    idx = pd.date_range("2025-01-01 00:00", periods=48, freq="h", tz="UTC")
    pv_dc = pd.Series(
        ([0.0] * 7 + [900.0, 1600.0, 2200.0, 2400.0, 2000.0, 1500.0, 800.0, 200.0] + [0.0] * 9) * 2,
        index=idx,
    )
    load = pd.DataFrame(
        {"Load": ([450.0] * 6 + [900.0] * 3 + [650.0] * 8 + [1200.0] * 5 + [700.0] * 2) * 2},
        index=idx,
    )
    temperature = pd.Series(22.0, index=idx)
    config = BatteryConfig(
        nominal_energy_wh=3000.0,
        min_soc=0.1,
        max_soc=0.9,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
        inverter_efficiency=0.96,
        enable_replacement=False,
        standby_loss_wh=0.0,
    )
    return pv_dc, load, temperature, config


def test_simulate_energy_balance_battery_golden_output():
    pv_dc, load, temperature, config = _golden_battery_inputs()

    results_df, total_pv, summary_df, replacement_cost, n_replacements, degradation_df = simulate_energy_balance(
        pv_dc=pv_dc,
        houseload=load,
        battery_config=config,
        freq="h",
        temperature_series=temperature,
    )

    _assert_numeric_record(
        summary_df.iloc[0].to_dict(),
        {
            "Total PV [kWh]": 22.272,
            "Total Load [kWh]": 36.0,
            "Sell [kWh]": 7.0425545297401175,
            "Import [kWh]": 19.064139530204148,
            "Import [%]": 52.955943139455975,
            "Grid Independence [%]": 47.044056860544025,
            "Final SOH [%]": 99.76712171800783,
            "N_Replacements": 0,
            "Replacement_Cost": 0.0,
        },
    )
    assert total_pv == pytest.approx(22272.0)
    assert replacement_cost == pytest.approx(0.0)
    assert n_replacements == 0
    assert len(degradation_df) == 2
    assert results_df["Battery_Energy"].iloc[-1] == pytest.approx(297.70796832641753)


def test_simulate_energy_balance_15min_golden_output():
    idx = pd.date_range("2025-01-01 00:00", periods=8, freq="15min", tz="UTC")
    pv_dc = pd.Series([0.0, 400.0, 800.0, 1200.0, 1200.0, 800.0, 400.0, 0.0], index=idx)
    load = pd.DataFrame({"Load": [500.0] * 8}, index=idx)

    results_df, total_pv, summary_df, *_ = simulate_energy_balance(
        pv_dc=pv_dc,
        houseload=load,
        battery_config=None,
        freq="15min",
    )

    _assert_numeric_record(
        summary_df.iloc[0].to_dict(),
        {
            "Total PV [kWh]": 1.152,
            "Total Load [kWh]": 1.0,
            "Sell [kWh]": 0.46,
            "Import [kWh]": 0.308,
            "Import [%]": 30.8,
            "Grid Independence [%]": 69.2,
            "Final SOH [%]": 100.0,
            "N_Replacements": 0,
            "Replacement_Cost": 0.0,
        },
    )
    assert total_pv == pytest.approx(1152.0)
    assert results_df["Import_From_Grid"].sum() * 0.25 / 1000 == pytest.approx(0.308)
    assert results_df["Sell_To_Grid"].sum() * 0.25 / 1000 == pytest.approx(0.46)


def test_cost_analysis_projection_golden_output():
    cost_params = CostParams(
        electricity_cost=0.30,
        electricity_sold_cost=0.05,
        daily_power_cost=0.20,
        module_cost_per_w=0.10,
        battery_cost_per_kwh=400.0,
        dc_ac_ratio=1.25,
        inverter_cost_per_kw=100.0,
        inverter_cost_per_kw_nobatt=50.0,
        installation_cost_per_module=100.0,
        battery_installation_cost=200.0,
        other_cost_per_module=10.0,
        other_cost_fixed=25.0,
        maintenance_cost_per_panel=5.0,
        maintenance_cost_fixed=15.0,
        operation_cost=20.0,
        inflation_rate=0.02,
        sell_price_inflation=0.01,
        discount_rate=0.03,
        pv_degradation_rate=0.005,
    )
    costs = calculate_costs(4, 500.0, 2000.0, cost_params)
    yearly_summary = pd.DataFrame(
        {
            "Year": [1, 2, 3],
            "Load_kWh": [1200.0, 1200.0, 1200.0],
            "PV_Production_kWh": [900.0, 880.0, 860.0],
            "Import_kWh": [450.0, 465.0, 480.0],
            "Export_kWh": [150.0, 145.0, 140.0],
            "PV_Degradation_Factor": [1.0, 0.98, 0.96],
            "Replacement_Cost": [0.0, 0.0, 800.0],
        }
    )

    projection = cost_analysis_projection(
        pd.DataFrame(),
        costs,
        num_years=3,
        inflation_rate=0.02,
        sell_price_inflation=0.01,
        discount_rate=0.03,
        yearly_summary_df=yearly_summary,
        total_replacement_cost=800.0,
        emissions_params=EmissionsParams(average_grid_carbon_intensity_gco2_kwh=200.0, country="Testland"),
    )

    _assert_numeric_record(
        costs,
        {
            "total_initial_cost": 1825.0,
            "annual_operation_cost": 55.0,
            "pv_cost": 200.0,
            "inverter_cost": 160.0,
            "battery_cost": 800.0,
            "installation_cost": 600.0,
            "other_costs": 65.0,
        },
    )
    _assert_numeric_record(
        projection.iloc[0].to_dict(),
        {
            "Cost_No_Sys_Annual": 433.0,
            "Cost_System_Annual": 255.5,
            "Savings_Cumulative_NPV": -1652.6699029126213,
            "CO2_Avoided_Total_kg": 180.0,
            "CO2_Avoided_SelfConsumed_kg": 150.0,
        },
    )
    _assert_numeric_record(
        projection.iloc[2].to_dict(),
        {
            "Cost_Replacement": 832.32,
            "Cost_System_Annual": 1108.1681,
            "Savings_Cumulative_NPV": -2088.5138282480434,
            "CO2_Avoided_Total_Cumulative_kg": 528.0,
            "CO2_Avoided_SelfConsumed_Cumulative_kg": 441.0,
        },
    )
    assert projection.attrs["payback_year"] is None
    assert projection.attrs["total_investment"] == pytest.approx(1825.0)
    assert projection.attrs["total_replacement_cost"] == pytest.approx(800.0)
    assert projection.attrs["final_npv_savings"] == pytest.approx(-2088.5138282480434)


def test_co2_projection_uses_marginal_intensity_golden_output():
    projection = calculate_co2_projection(
        yearly_pv_kwh=pd.Series([1000.0, 900.0]).values,
        yearly_export_kwh=pd.Series([100.0, 120.0]).values,
        emissions_params=EmissionsParams(
            average_grid_carbon_intensity_gco2_kwh=150.0,
            marginal_grid_carbon_intensity_gco2_kwh=400.0,
            country="Testland",
        ),
    )

    assert projection["CO2_Avoided_CI_Type"].tolist() == ["marginal", "marginal"]
    _assert_numeric_record(
        projection.iloc[-1].to_dict(),
        {
            "CO2_Avoided_Total_kg": 360.0,
            "CO2_Avoided_SelfConsumed_kg": 312.0,
            "CO2_Avoided_Total_Cumulative_kg": 760.0,
            "CO2_Avoided_SelfConsumed_Cumulative_kg": 672.0,
            "CO2_Avoided_CI_gCO2_kWh": 400.0,
            "Average_Grid_CI_gCO2_kWh": 150.0,
            "Marginal_Grid_CI_gCO2_kWh": 400.0,
        },
    )
