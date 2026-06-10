"""Simulation orchestration for the public App facade."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from breos.app_config import ResolvedAppConfig, build_costs_dict
from breos.app_inputs import AppRuntimeDependencies, prepare_simulation_inputs
from breos.battery import BatteryConfig, simulate_energy_balance
from breos.economics import calculate_lcoe, cost_analysis_projection, find_payback_year
from breos.utils import get_hours_per_step


@dataclass(frozen=True)
class SimulationArtifacts:
    """Intermediate outputs needed to serialize App results."""

    yearly_df: pd.DataFrame
    first_year_results_df: pd.DataFrame
    cost_projection: pd.DataFrame
    costs: dict[str, float]
    payback_year: int | None
    lcoe: float
    current_soh: float
    total_replacements: int
    total_replacement_cost: float


def run_app_simulation(
    cfg: dict[str, Any],
    resolved: ResolvedAppConfig,
    deps: AppRuntimeDependencies,
) -> SimulationArtifacts:
    """Run the weather/PV/load/battery/economics simulation pipeline."""
    inputs = prepare_simulation_inputs(cfg, resolved, deps)

    freq = cfg["resolution"]
    battery_kwh = cfg["battery_kwh"]
    battery_wh = battery_kwh * 1000
    has_battery = battery_kwh > 0
    projection_years = cfg["projection_years"]
    degradation_rate = cfg["pv_degradation_rate"]
    hours_per_step = get_hours_per_step(freq)

    replacement_cost = resolved.cost_params.battery_cost_per_kwh * battery_kwh

    # Size the inverter AC rating the same way CAPEX does (economics
    # calculate_costs), so the paid-for inverter also clips production.
    pv_peak_w = cfg["n_modules"] * resolved.avg_module_power_w
    loading_ratio = cfg["inverter_loading_ratio"]
    inverter_ac_capacity_w = pv_peak_w / loading_ratio if loading_ratio and loading_ratio > 0 else None

    cumulative_fec = 0.0
    cumulative_cal_seconds = 0.0
    cumulative_resistance_growth = 0.0
    cumulative_cycle_deg = 0.0
    cumulative_cal_deg = 0.0
    current_soh = 100.0
    total_replacements = 0
    total_replacement_cost = 0.0
    yearly_summaries: list[dict[str, Any]] = []
    first_year_results_df: pd.DataFrame | None = None

    for year_idx in range(projection_years):
        pv_degradation_factor = (1 - degradation_rate) ** year_idx
        dc_power = inputs.dc_system_base * pv_degradation_factor

        if has_battery:
            batt_kwargs: dict[str, Any] = {}
            if cfg["battery_rte"] is not None:
                # Split the round-trip efficiency evenly across charge and
                # discharge, matching the BatteryConfig default convention.
                one_way = math.sqrt(cfg["battery_rte"])
                batt_kwargs["charge_efficiency"] = one_way
                batt_kwargs["discharge_efficiency"] = one_way
            batt_cfg = BatteryConfig(
                nominal_energy_wh=battery_wh,
                initial_soh=current_soh,
                eol_percentage=cfg["battery_eol_percentage"],
                max_soc=cfg["battery_max_soc"],
                min_soc=cfg["battery_min_soc"],
                dc_coupled=cfg["dc_coupled"],
                inverter_efficiency=cfg["inverter_efficiency"],
                inverter_ac_capacity_w=inverter_ac_capacity_w,
                enable_replacement=True,
                replacement_cost=replacement_cost,
                calendar_model=cfg["calendar_model"],
                **batt_kwargs,
            )
        else:
            # PV-only runs still flow through the same inverter model so the
            # configured efficiency and AC clipping apply consistently.
            batt_cfg = BatteryConfig(
                nominal_energy_wh=0,
                inverter_efficiency=cfg["inverter_efficiency"],
                inverter_ac_capacity_w=inverter_ac_capacity_w,
            )

        results_df, total_pv, _summary_df, year_rep_cost, year_n_rep, degradation_df = simulate_energy_balance(
            pv_dc=dc_power,
            houseload=inputs.load_data,
            battery_config=batt_cfg,
            freq=freq,
            temperature_series=inputs.temperature_series if has_battery else None,
            initial_fec=cumulative_fec,
            initial_calendar_seconds=cumulative_cal_seconds,
            initial_resistance_growth=cumulative_resistance_growth,
            initial_cumulative_cycle_deg=cumulative_cycle_deg,
            initial_cumulative_cal_deg=cumulative_cal_deg,
        )

        if first_year_results_df is None:
            first_year_results_df = results_df

        if has_battery and not degradation_df.empty:
            cumulative_fec = degradation_df["Cumulative_FEC"].iloc[-1]
            cumulative_cal_seconds = degradation_df["Cumulative_Calendar_Seconds"].iloc[-1]
            cumulative_cycle_deg = degradation_df["Cumulative_Cycle_Degradation"].iloc[-1]
            cumulative_cal_deg = degradation_df["Cumulative_Calendar_Degradation"].iloc[-1]
            current_soh = degradation_df["SOH"].iloc[-1]
            if "Resistance_Growth" in degradation_df.columns:
                cumulative_resistance_growth = degradation_df["Resistance_Growth"].iloc[-1]

        total_replacements += year_n_rep
        total_replacement_cost += year_rep_cost

        total_pv_kwh = total_pv / 1000
        total_load = (results_df["Houseload"].sum() / 1000) * hours_per_step
        total_import = (results_df["Import_From_Grid"].sum() / 1000) * hours_per_step
        total_export = (results_df["Sell_To_Grid"].sum() / 1000) * hours_per_step
        grid_indep = (1 - total_import / total_load) * 100 if total_load > 0 else 0

        yearly_summaries.append(
            {
                "Year": year_idx + 1,
                "PV_Production_kWh": total_pv_kwh,
                "Load_kWh": total_load,
                "Import_kWh": total_import,
                "Export_kWh": total_export,
                "Grid_Independence_%": grid_indep,
                "Battery_SOH_%": current_soh if has_battery else None,
                "Replacements": year_n_rep,
                "Replacement_Cost": year_rep_cost,
                "PV_Degradation_Factor": pv_degradation_factor,
            }
        )

    yearly_df = pd.DataFrame(yearly_summaries)
    if first_year_results_df is None:
        raise RuntimeError("projection_years must be at least 1")

    costs = build_costs_dict(cfg, resolved)
    cost_projection = cost_analysis_projection(
        results_df=first_year_results_df,
        costs=costs,
        num_years=projection_years,
        inflation_rate=cfg["inflation_rate"],
        discount_rate=cfg["discount_rate"],
        freq=freq,
        yearly_summary_df=yearly_df,
        total_replacement_cost=total_replacement_cost,
        emissions_params=resolved.emissions_params,
    )

    year1_pv = yearly_df.iloc[0]["PV_Production_kWh"]
    lcoe = calculate_lcoe(
        total_investment=costs["total_initial_cost"],
        annual_production_kwh=year1_pv,
        annual_operation_cost=costs["annual_operation_cost"],
        lifetime_years=projection_years,
        discount_rate=cfg["discount_rate"],
        degradation_rate=degradation_rate,
    )

    return SimulationArtifacts(
        yearly_df=yearly_df,
        first_year_results_df=first_year_results_df,
        cost_projection=cost_projection,
        costs=costs,
        payback_year=find_payback_year(cost_projection),
        lcoe=lcoe,
        current_soh=current_soh,
        total_replacements=total_replacements,
        total_replacement_cost=total_replacement_cost,
    )
