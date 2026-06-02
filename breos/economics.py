"""
Economics module for cost analysis and projections.

This module handles:
- CAPEX calculations (PV, battery, installation)
- OPEX calculations (maintenance, grid costs)
- Multi-year cost projections with inflation and degradation
- Payback period analysis
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from breos.utils import get_hours_per_step

# Default battery and replacement cost per kWh of battery capacity (€/kWh)
BATTERY_REPLACEMENT_COST_PER_KWH: float = 500.0


@dataclass
class CostParams:
    """Cost parameters for economic analysis."""

    # Electricity prices
    electricity_cost: float = 0.27  # €/kWh purchased
    electricity_sold_cost: float = 0.06  # €/kWh sold to grid
    daily_power_cost: float = 0.30  # € per day connection fee

    # Equipment costs
    module_cost_per_w: float = 0.125  # €/W
    battery_cost_per_kwh: float = BATTERY_REPLACEMENT_COST_PER_KWH  # €/kWh
    dc_ac_ratio: float = 1.25  # DC/AC sizing ratio for inverter CAPEX
    inverter_cost_per_kw: float = 102.58  # €/kW (with battery)
    inverter_cost_per_kw_nobatt: float = 48.37  # €/kW (without battery)
    installation_cost_per_module: float = 350.0  # €/module
    battery_installation_cost: float = 350.0  # € fixed
    other_cost_per_module: float = 50.0  # € cables, etc.
    other_cost_fixed: float = 0.0  # € fixed misc. costs
    land_cost: float = 0.0

    # Operations
    maintenance_cost_per_panel: float = 10.0  # €/panel/year
    maintenance_cost_fixed: float = 0.0  # € fixed /year
    operation_cost: float = 0.0  # € additional /year

    # Thermal system costs (TES + Heat Pump)
    tes_cost_per_kwh_th: float = 0.0  # €/kWh_th (0 = not included)
    heat_pump_cost_per_kw_th: float = 0.0  # €/kW_th (0 = not included)
    hp_maintenance_annual: float = 0.0  # €/year heat pump maintenance
    gas_cost_per_kwh: float = 0.0  # €/kWh baseline heating fuel cost
    tes_installation_cost: float = 0.0  # € fixed TES installation

    # Analysis parameters
    inflation_rate: float = 0.02
    sell_price_inflation: float = 0.0
    discount_rate: float = 0.0
    pv_degradation_rate: float = 0.005


def cost_params_from_config(
    costs_config: Optional[Dict[str, Any]] = None,
    financials_config: Optional[Dict[str, Any]] = None,
) -> CostParams:
    """Build :class:`CostParams` from BREOS cost and financial config keys."""
    costs_config = costs_config or {}
    financials_config = financials_config or {}

    return CostParams(
        electricity_cost=costs_config.get(
            "electricity_cost",
            financials_config.get("electricity_cost", 0.27),
        ),
        electricity_sold_cost=costs_config.get(
            "electricity_sold_cost",
            financials_config.get("electricity_sold_cost", 0.06),
        ),
        daily_power_cost=costs_config.get("daily_power_cost", 0.30),
        module_cost_per_w=costs_config.get("module_cost_per_w", 0.125),
        battery_cost_per_kwh=costs_config.get("storage_cost_per_kwh", BATTERY_REPLACEMENT_COST_PER_KWH),
        dc_ac_ratio=costs_config.get("dc_ac_ratio", 1.25),
        inverter_cost_per_kw=costs_config.get("inverter_cost_per_kw_hybrid", 102.58),
        inverter_cost_per_kw_nobatt=costs_config.get("inverter_cost_per_kw_simple", 48.37),
        installation_cost_per_module=costs_config.get("installation_cost_per_module", 350.0),
        battery_installation_cost=costs_config.get("installation_cost_battery", 350.0),
        other_cost_per_module=costs_config.get("other_cost_per_module", 0.0),
        other_cost_fixed=costs_config.get("other_costs", 0.0),
        maintenance_cost_per_panel=costs_config.get("maintenance_cost_per_panel", 0.0),
        maintenance_cost_fixed=costs_config.get("maintenance_cost", 0.0),
        operation_cost=costs_config.get("operation_cost", 0.0),
        tes_cost_per_kwh_th=costs_config.get("tes_cost_per_kwh_th", 0.0),
        heat_pump_cost_per_kw_th=costs_config.get("heat_pump_cost_per_kw_th", 0.0),
        hp_maintenance_annual=costs_config.get("hp_maintenance_annual", 0.0),
        gas_cost_per_kwh=costs_config.get("gas_cost_per_kwh", 0.0),
        tes_installation_cost=costs_config.get("tes_installation_cost", 0.0),
        inflation_rate=financials_config.get("inflation_rate", 0.02),
        sell_price_inflation=financials_config.get("sell_price_inflation", 0.0),
        discount_rate=financials_config.get("discount_rate", 0.0),
        pv_degradation_rate=financials_config.get("pv_degradation_rate", 0.005),
    )


def calculate_costs(
    n_modules: int,
    module_power_w: float,
    battery_capacity_wh: float = 0.0,
    cost_params: Optional[CostParams] = None,
    tes_capacity_kwh_th: float = 0.0,
    heat_pump_kw_th: float = 0.0,
) -> Dict[str, float]:
    """
    Calculate system costs (CAPEX) and return cost dictionary.

    Args:
        n_modules: Number of PV modules
        module_power_w: Power per module in Watts (STC)
        battery_capacity_wh: Battery capacity in Wh (0 for no battery)
        cost_params: Cost parameters
        tes_capacity_kwh_th: TES capacity in kWh_th (0 for no TES)
        heat_pump_kw_th: Heat pump rated thermal power in kW_th (0 for no HP)

    Returns:
        Dictionary with cost breakdown and totals
    """
    if cost_params is None:
        cost_params = CostParams()

    total_power_kw = n_modules * module_power_w / 1000
    inverter_power_kw = total_power_kw / cost_params.dc_ac_ratio if cost_params.dc_ac_ratio > 0 else total_power_kw
    has_battery = battery_capacity_wh > 1

    # PV module costs
    pv_cost = cost_params.module_cost_per_w * module_power_w * n_modules

    # Installation costs
    installation_cost = cost_params.installation_cost_per_module * n_modules
    if has_battery:
        installation_cost += cost_params.battery_installation_cost

    # Battery costs
    if has_battery:
        battery_cost = (battery_capacity_wh / 1000) * cost_params.battery_cost_per_kwh
        inverter_cost = cost_params.inverter_cost_per_kw * inverter_power_kw
    else:
        battery_cost = 0.0
        inverter_cost = cost_params.inverter_cost_per_kw_nobatt * inverter_power_kw

    # Other costs
    other_costs = (cost_params.other_cost_per_module * n_modules) + cost_params.other_cost_fixed

    # TES + Heat Pump costs
    tes_cost = tes_capacity_kwh_th * cost_params.tes_cost_per_kwh_th
    hp_cost = heat_pump_kw_th * cost_params.heat_pump_cost_per_kw_th
    tes_install = cost_params.tes_installation_cost if (tes_capacity_kwh_th > 0 or heat_pump_kw_th > 0) else 0.0

    # Maintenance
    annual_operation_cost = (
        cost_params.maintenance_cost_per_panel * n_modules
        + cost_params.maintenance_cost_fixed
        + cost_params.operation_cost
        + (cost_params.hp_maintenance_annual if heat_pump_kw_th > 0 else 0.0)
    )

    # Total CAPEX
    total_initial_cost = (
        pv_cost
        + inverter_cost
        + battery_cost
        + installation_cost
        + cost_params.land_cost
        + other_costs
        + tes_cost
        + hp_cost
        + tes_install
    )

    return {
        "electricity_cost": cost_params.electricity_cost,
        "electricity_sold_cost": cost_params.electricity_sold_cost,
        "daily_power_cost": cost_params.daily_power_cost,
        "gas_cost_per_kwh": cost_params.gas_cost_per_kwh,
        "total_initial_cost": total_initial_cost,
        "annual_operation_cost": annual_operation_cost,
        "pv_cost": pv_cost,
        "inverter_cost": inverter_cost,
        "battery_cost": battery_cost,
        "installation_cost": installation_cost,
        "other_costs": other_costs,
        "tes_cost": tes_cost,
        "hp_cost": hp_cost,
        "tes_installation_cost": tes_install,
    }


def cost_analysis_projection(
    results_df: pd.DataFrame,
    costs: Dict[str, float],
    num_years: int = 20,
    inflation_rate: float = 0.03,
    sell_price_inflation: float = 0.0,
    discount_rate: float = 0.02,
    degradation_rate: float = 0.005,
    results_directory: Optional[str] = None,
    scenario_name: str = "",
    freq: str = "h",
    yearly_summary_df: Optional[pd.DataFrame] = None,
    total_replacement_cost: Optional[float] = None,
    emissions_params=None,
) -> pd.DataFrame:
    """
    Perform multi-year cost projection analysis.

    Includes inflation, discount rate, and PV degradation.

    Args:
        results_df: DataFrame with simulation results. Required columns are
            ``Datetime``, ``PV_Production``, ``Houseload``,
            ``Import_From_Grid``, and ``Sell_To_Grid``.
        costs: Dictionary with cost parameters (from calculate_costs())
        num_years: Number of years to project
        inflation_rate: Annual inflation for electricity/operation costs
        sell_price_inflation: Annual inflation for sell price
        discount_rate: Discount rate for NPV calculations
        degradation_rate: Annual PV degradation rate
        results_directory: Optional directory to save results
        scenario_name: Optional name suffix for saved files
        freq: Simulation frequency string ('h', '15min')
        yearly_summary_df: Optional DataFrame from singleyear propagation with
            Year, PV_Production_kWh, Import_kWh, Export_kWh, etc. for each year.
            When provided, uses actual yearly data instead of estimation.
        total_replacement_cost: Total battery replacement cost from propagation

    Returns:
        DataFrame with yearly cost projections
    """

    # If yearly_summary_df provided (from propagation), use actual yearly data
    if yearly_summary_df is not None and not yearly_summary_df.empty:
        # Build projection from actual yearly simulation data
        proj = pd.DataFrame()
        proj["Year"] = range(1, num_years + 1)

        # Map yearly_summary_df to projection (it should already have num_years rows)
        yearly_data = yearly_summary_df.set_index("Year")

        # Get first year data for baseline calculation
        first_year_load = yearly_data["Load_kWh"].iloc[0]
        first_year_days = 365  # Assume full year

        # Factors
        inflation_factors = (1 + inflation_rate) ** (proj["Year"] - 1)
        sell_inflation_factors = (1 + sell_price_inflation) ** (proj["Year"] - 1)
        discount_factors = 1 / ((1 + discount_rate) ** proj["Year"])

        # Baseline (no system) - uses first year's load, scaled by inflation
        # Include gas heating cost if thermal system is present
        gas_cost = costs.get("gas_cost_per_kwh", 0.0)
        thermal_demand_kwh = yearly_data.get("Thermal_Demand_kWh", pd.Series(0.0, index=yearly_data.index))
        first_year_thermal = float(thermal_demand_kwh.iloc[0]) if hasattr(thermal_demand_kwh, "iloc") else 0.0
        baseline_gas_annual = first_year_thermal * gas_cost

        proj["Cost_No_Sys_Annual"] = (
            first_year_load * costs["electricity_cost"]
            + first_year_days * costs["daily_power_cost"]
            + baseline_gas_annual
        ) * inflation_factors
        proj["Cost_No_Sys_Cumulative"] = proj["Cost_No_Sys_Annual"].cumsum()

        # With PV system - Use ACTUAL yearly values from propagation
        proj["PV_Production_kWh"] = yearly_data["PV_Production_kWh"].values
        proj["Export_kWh"] = yearly_data["Export_kWh"].values
        proj["Degradation_Factor"] = yearly_data["PV_Degradation_Factor"].values

        # Cost calculations using actual data
        proj["Cost_Import"] = yearly_data["Import_kWh"].values * costs["electricity_cost"] * inflation_factors
        proj["Revenue_Export"] = (
            yearly_data["Export_kWh"].values * costs["electricity_sold_cost"] * sell_inflation_factors
        )
        proj["Cost_Operation"] = costs["annual_operation_cost"] * inflation_factors
        proj["Cost_Daily"] = first_year_days * costs["daily_power_cost"] * inflation_factors

        # Battery replacement costs from propagation
        proj["Cost_Replacement"] = yearly_data["Replacement_Cost"].values * inflation_factors

        proj["Cost_System_Annual"] = (
            proj["Cost_Import"]
            - proj["Revenue_Export"]
            + proj["Cost_Operation"]
            + proj["Cost_Daily"]
            + proj["Cost_Replacement"]
        )

        proj["Cost_System_Cumulative"] = costs["total_initial_cost"] + proj["Cost_System_Annual"].cumsum()

        # Discounted values (NPV)
        proj["Cost_No_Sys_Annual_NPV"] = proj["Cost_No_Sys_Annual"] * discount_factors
        proj["Cost_System_Annual_NPV"] = proj["Cost_System_Annual"] * discount_factors
        proj["Cost_No_Sys_Cumulative_NPV"] = proj["Cost_No_Sys_Annual_NPV"].cumsum()
        proj["Cost_System_Cumulative_NPV"] = costs["total_initial_cost"] + proj["Cost_System_Annual_NPV"].cumsum()

        # Savings
        proj["Savings_Cumulative"] = proj["Cost_No_Sys_Cumulative"] - proj["Cost_System_Cumulative"]
        proj["Savings_Cumulative_NPV"] = proj["Cost_No_Sys_Cumulative_NPV"] - proj["Cost_System_Cumulative_NPV"]

        # Find payback year
        payback_mask = proj["Savings_Cumulative_NPV"] > 0
        if payback_mask.any():
            payback_year = proj.loc[payback_mask, "Year"].iloc[0]
            proj.attrs["payback_year"] = payback_year
        else:
            proj.attrs["payback_year"] = None

        proj.attrs["total_investment"] = costs["total_initial_cost"]
        proj.attrs["final_npv_savings"] = proj["Savings_Cumulative_NPV"].iloc[-1]
        if total_replacement_cost is not None:
            proj.attrs["total_replacement_cost"] = total_replacement_cost

        # CO2 emissions avoided
        if emissions_params is not None:
            from breos.emissions import calculate_co2_projection

            co2_proj = calculate_co2_projection(
                proj["PV_Production_kWh"].values,
                proj["Export_kWh"].values,
                emissions_params,
            )
            proj["CO2_Avoided_Total_kg"] = co2_proj["CO2_Avoided_Total_kg"].values
            proj["CO2_Avoided_SelfConsumed_kg"] = co2_proj["CO2_Avoided_SelfConsumed_kg"].values
            proj["CO2_Avoided_Total_Cumulative_kg"] = co2_proj["CO2_Avoided_Total_Cumulative_kg"].values
            proj["CO2_Avoided_SelfConsumed_Cumulative_kg"] = co2_proj["CO2_Avoided_SelfConsumed_Cumulative_kg"].values
            for col in (
                "CO2_Avoided_CI_gCO2_kWh",
                "CO2_Avoided_CI_Type",
                "Average_Grid_CI_gCO2_kWh",
                "Marginal_Grid_CI_gCO2_kWh",
            ):
                proj[col] = co2_proj[col].values
            proj.attrs["lifetime_co2_avoided_total_kg"] = float(proj["CO2_Avoided_Total_Cumulative_kg"].iloc[-1])
            proj.attrs["lifetime_co2_avoided_self_consumed_kg"] = float(
                proj["CO2_Avoided_SelfConsumed_Cumulative_kg"].iloc[-1]
            )

        # Save if directory provided
        if results_directory:
            import os

            os.makedirs(results_directory, exist_ok=True)
            suffix = f"_{scenario_name}" if scenario_name else ""
            proj.to_csv(f"{results_directory}/cost_projection{suffix}.csv", index=False)

        return proj

    # ===== LEGACY PATH: Estimate from first year =====
    df = results_df.copy()

    # Prepare datetime index
    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)
        df.set_index("Datetime", inplace=True)

    df["Year"] = df.index.year
    df["Date"] = df.index.normalize()

    # Calculate hours per step for energy conversion
    hours_per_step = get_hours_per_step(freq)

    # Convert W (or whatever units, usually W in results_df) to kW
    # Note: results_df columns like PV_Production are typically in W (Power).
    # To get Energy (kWh), we need to multiply by hours_per_step and divide by 1000.

    # First convert columns to numeric, just in case
    cols_to_numeric = ["PV_Production", "Houseload", "Import_From_Grid", "Sell_To_Grid", "Replacement_Cost"]
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Aggregate first year
    # Summing Power (W) gives sum(Watts). To get Wh, multiply by hours_per_step.
    # To get kWh, divide by 1000.
    # Replacement_Cost is already in currency (EUR), likely summed is correct (not power->energy).
    yearly = df[["PV_Production", "Houseload", "Import_From_Grid", "Sell_To_Grid"]].groupby(df["Year"]).sum()

    # Handle replacement cost separately if present (it's already simple sum, no kWh conversion needed)
    if "Replacement_Cost" in df.columns:
        yearly_replacement = df[["Replacement_Cost"]].groupby(df["Year"]).sum()
    else:
        yearly_replacement = pd.DataFrame(0.0, index=yearly.index, columns=["Replacement_Cost"])

    # Scale to Energy (kWh)
    yearly = yearly * hours_per_step / 1000.0

    daily_counts = df.groupby("Year")["Date"].nunique()

    first_year_load = yearly["Houseload"].iloc[0]
    first_year_import = yearly["Import_From_Grid"].iloc[0]
    first_year_export = yearly["Sell_To_Grid"].iloc[0]
    first_year_pv = yearly["PV_Production"].iloc[0]
    first_year_days = daily_counts.iloc[0]

    # Build projection
    proj = pd.DataFrame()
    proj["Year"] = range(1, num_years + 1)

    # Factors
    inflation_factors = (1 + inflation_rate) ** (proj["Year"] - 1)
    sell_inflation_factors = (1 + sell_price_inflation) ** (proj["Year"] - 1)
    discount_factors = 1 / ((1 + discount_rate) ** proj["Year"])
    degradation_factors = (1 - degradation_rate) ** (proj["Year"] - 1)

    # Baseline (no system)
    proj["Cost_No_Sys_Annual"] = (
        first_year_load * costs["electricity_cost"] + first_year_days * costs["daily_power_cost"]
    ) * inflation_factors
    proj["Cost_No_Sys_Cumulative"] = proj["Cost_No_Sys_Annual"].cumsum()

    # With PV system (including degradation)
    pv_degraded = first_year_pv * degradation_factors
    self_consumption_ratio = 1 - (first_year_export / first_year_pv) if first_year_pv > 0 else 0
    export_degraded = pv_degraded * (1 - self_consumption_ratio)
    pv_reduction = first_year_pv - pv_degraded
    import_adjusted = first_year_import + pv_reduction * self_consumption_ratio

    proj["Cost_Import"] = import_adjusted * costs["electricity_cost"] * inflation_factors
    proj["Revenue_Export"] = export_degraded * costs["electricity_sold_cost"] * sell_inflation_factors
    proj["Cost_Operation"] = costs["annual_operation_cost"] * inflation_factors
    proj["Cost_Daily"] = first_year_days * costs["daily_power_cost"] * inflation_factors

    proj["Cost_System_Annual"] = (
        proj["Cost_Import"] - proj["Revenue_Export"] + proj["Cost_Operation"] + proj["Cost_Daily"]
    )

    proj["Cost_System_Cumulative"] = costs["total_initial_cost"] + proj["Cost_System_Annual"].cumsum()

    # Battery replacement
    # 1. From simulation results (dynamic)
    # We use the yearly_replacement calculated from the FIRST year results as base?
    # NO. The simulation results ONLY cover the simulated period.
    # If this is a PROJECTION (single year -> 20 years), we don't have simulated results for future years.
    # However, if this is a MULTIYEAR simulation, results_df has all years.

    # Let's check if we have data for all years in the projection
    # If results_df has data for year Y, we should use it.

    # Actually, cost_analysis_projection logic currently takes the FIRST year and projects it.
    # It assumes single year simulation.
    # If it is multiyear, 'yearly' dataframe above will have multiple rows.

    # Logic update:
    # If 'yearly' has data for the specific projection year, use actuals.
    # If not (projection derived from first year), we assume no replacement in first year implies no replacement?
    # Or we follow the standard logic.

    # For replacement cost specifically:
    # If we are in multiyear simulation mode, `results_df` contains all events.
    # yearly_replacement has the sum for each year.

    # Add a column for replacement cost to `proj`
    proj["Cost_Replacement"] = 0.0

    # Map available replacement data from simulation to the projection
    # yearly_replacement index is Year (e.g. 2023, 2024...)
    # proj['Year'] is relative year (1, 2, 3...)
    # We need to align them.

    # Get the start year from the data
    start_year = df["Year"].min()

    for relative_year in proj["Year"]:
        actual_year = start_year + relative_year - 1
        if actual_year in yearly_replacement.index:
            # We have simulated data for this year
            cost = yearly_replacement.loc[actual_year, "Replacement_Cost"]
            # Apply inflation if it wasn't already encompassed in the simulation?
            # The simulation outputs nominal cost at time of replacement?
            # Usually simulation just outputs the base cost value. We should apply inflation here.
            inflation_factor = (1 + inflation_rate) ** (relative_year - 1)
            # Actually, check if simulation output `Replacement_Cost` is real or nominal.
            # In `battery.py`, we log `battery_config.replacement_cost`. This is likely the base cost input.
            # So yes, we need to inflate it.
            proj.loc[proj["Year"] == relative_year, "Cost_Replacement"] += cost * inflation_factor

    # Add to annual system cost
    proj["Cost_System_Annual"] += proj["Cost_Replacement"]

    proj["Cost_System_Cumulative"] = costs["total_initial_cost"] + proj["Cost_System_Annual"].cumsum()

    # Discounted values (NPV)
    proj["Cost_No_Sys_Annual_NPV"] = proj["Cost_No_Sys_Annual"] * discount_factors
    proj["Cost_System_Annual_NPV"] = proj["Cost_System_Annual"] * discount_factors
    proj["Cost_No_Sys_Cumulative_NPV"] = proj["Cost_No_Sys_Annual_NPV"].cumsum()
    proj["Cost_System_Cumulative_NPV"] = costs["total_initial_cost"] + proj["Cost_System_Annual_NPV"].cumsum()

    # Savings
    proj["Savings_Cumulative"] = proj["Cost_No_Sys_Cumulative"] - proj["Cost_System_Cumulative"]
    proj["Savings_Cumulative_NPV"] = proj["Cost_No_Sys_Cumulative_NPV"] - proj["Cost_System_Cumulative_NPV"]

    # Tracking columns
    proj["PV_Production_kWh"] = pv_degraded
    proj["Export_kWh"] = export_degraded
    proj["Degradation_Factor"] = degradation_factors

    # Find payback year
    payback_mask = proj["Savings_Cumulative_NPV"] > 0
    if payback_mask.any():
        payback_year = proj.loc[payback_mask, "Year"].iloc[0]
        proj.attrs["payback_year"] = payback_year
    else:
        proj.attrs["payback_year"] = None

    proj.attrs["total_investment"] = costs["total_initial_cost"]
    proj.attrs["final_npv_savings"] = proj["Savings_Cumulative_NPV"].iloc[-1]

    # CO2 emissions avoided
    if emissions_params is not None:
        from breos.emissions import calculate_co2_projection

        co2_proj = calculate_co2_projection(
            proj["PV_Production_kWh"].values,
            proj["Export_kWh"].values,
            emissions_params,
        )
        proj["CO2_Avoided_Total_kg"] = co2_proj["CO2_Avoided_Total_kg"].values
        proj["CO2_Avoided_SelfConsumed_kg"] = co2_proj["CO2_Avoided_SelfConsumed_kg"].values
        proj["CO2_Avoided_Total_Cumulative_kg"] = co2_proj["CO2_Avoided_Total_Cumulative_kg"].values
        proj["CO2_Avoided_SelfConsumed_Cumulative_kg"] = co2_proj["CO2_Avoided_SelfConsumed_Cumulative_kg"].values
        for col in (
            "CO2_Avoided_CI_gCO2_kWh",
            "CO2_Avoided_CI_Type",
            "Average_Grid_CI_gCO2_kWh",
            "Marginal_Grid_CI_gCO2_kWh",
        ):
            proj[col] = co2_proj[col].values
        proj.attrs["lifetime_co2_avoided_total_kg"] = float(proj["CO2_Avoided_Total_Cumulative_kg"].iloc[-1])
        proj.attrs["lifetime_co2_avoided_self_consumed_kg"] = float(
            proj["CO2_Avoided_SelfConsumed_Cumulative_kg"].iloc[-1]
        )

    # Save if directory provided
    if results_directory:
        import os

        os.makedirs(results_directory, exist_ok=True)
        suffix = f"_{scenario_name}" if scenario_name else ""
        proj.to_csv(f"{results_directory}/cost_projection{suffix}.csv", index=False)

    return proj


def find_payback_year(cost_projection: pd.DataFrame) -> Optional[int]:
    """
    Find the payback year from a cost projection DataFrame.

    Args:
        cost_projection: DataFrame from cost_analysis_projection()

    Returns:
        Year number when payback is achieved, or None if never
    """
    if "Savings_Cumulative_NPV" in cost_projection.columns:
        payback = cost_projection[cost_projection["Savings_Cumulative_NPV"] > 0]
        if not payback.empty:
            return int(payback["Year"].iloc[0])
    return None


def calculate_lcoe(
    total_investment: float,
    annual_production_kwh: float,
    annual_operation_cost: float,
    lifetime_years: int = 25,
    discount_rate: float = 0.0,
    degradation_rate: float = 0.005,
) -> float:
    """
    Calculate Levelized Cost of Electricity (LCOE).

    Args:
        total_investment: Total CAPEX (€)
        annual_production_kwh: First year production (kWh)
        annual_operation_cost: Annual O&M cost (€)
        lifetime_years: System lifetime
        discount_rate: Discount rate
        degradation_rate: Annual degradation

    Returns:
        LCOE in €/kWh
    """
    # NPV of costs
    npv_costs = total_investment
    for t in range(1, lifetime_years + 1):
        npv_costs += annual_operation_cost / ((1 + discount_rate) ** t)

    # NPV of production
    npv_production = 0.0
    for t in range(1, lifetime_years + 1):
        year_production = annual_production_kwh * ((1 - degradation_rate) ** (t - 1))
        npv_production += year_production / ((1 + discount_rate) ** t)

    return npv_costs / npv_production if npv_production > 0 else float("inf")
