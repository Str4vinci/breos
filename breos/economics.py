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

from breos.utils import get_hours_per_step, local_datetime_index

# Default battery and replacement cost per kWh of battery capacity (€/kWh)
BATTERY_REPLACEMENT_COST_PER_KWH: float = 500.0

# Where a replacement year carries no recorded instant, the outlay is booked
# at the middle of that year. The simulation records the real instant and the
# runners carry it through, so this covers summaries built before the instant
# was carried and callers that supply none; mid-year is the least biased
# choice for an unknown day. The resolved time is reported back in the
# projection's ``Replacement_Time_Years`` column, so the assumption is visible
# rather than buried here.
DEFAULT_REPLACEMENT_YEAR_FRACTION: float = 0.5

SYSTEM_AC_PRODUCTION_COLUMNS = ("PV_AC_To_Load", "Battery_AC_To_Load_PV")

# Canonical translation from the public cost-catalogue/config vocabulary to
# CostParams attributes. App presets, App ``[costs]`` overrides, and the
# lower-level ``cost_params_from_config`` helper all share this mapping.
COST_CONFIG_KEY_TO_PARAM: dict[str, str] = {
    "electricity_cost": "electricity_cost",
    "electricity_sold_cost": "electricity_sold_cost",
    "daily_power_cost": "daily_power_cost",
    "module_cost_per_w": "module_cost_per_w",
    "storage_cost_per_kwh": "battery_cost_per_kwh",
    "inverter_cost_per_kw_hybrid": "inverter_cost_per_kw",
    "inverter_cost_per_kw_simple": "inverter_cost_per_kw_nobatt",
    "installation_cost_per_module": "installation_cost_per_module",
    "installation_cost_battery": "battery_installation_cost",
    "other_cost_per_module": "other_cost_per_module",
    "other_costs": "other_cost_fixed",
    "land_cost": "land_cost",
    "maintenance_cost_per_panel": "maintenance_cost_per_panel",
    "maintenance_cost": "maintenance_cost_fixed",
    "operation_cost": "operation_cost",
}


def system_ac_production_power(results_df: pd.DataFrame) -> pd.Series:
    """Return usable PV-system AC production in the frame's power unit.

    Prefer the explicit ledger: direct PV to load, PV returned from battery
    to load, and PV exported at the AC boundary. ``Sell_To_Grid`` is accepted
    as the export alias. Older frames fall back to compatibility-only
    ``PV_Production``.
    """
    if all(column in results_df.columns for column in SYSTEM_AC_PRODUCTION_COLUMNS):
        export_column = "PV_AC_Export" if "PV_AC_Export" in results_df.columns else "Sell_To_Grid"
        if export_column in results_df.columns:
            columns = [*SYSTEM_AC_PRODUCTION_COLUMNS, export_column]
            return results_df[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)

    if "PV_Production" in results_df.columns:
        return pd.to_numeric(results_df["PV_Production"], errors="coerce").fillna(0.0)

    required = ", ".join((*SYSTEM_AC_PRODUCTION_COLUMNS, "PV_AC_Export (or Sell_To_Grid)"))
    raise KeyError(f"Results do not contain the AC system-production ledger ({required}) or legacy PV_Production")


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

    # Analysis parameters
    inflation_rate: float = 0.02
    sell_price_inflation: float = 0.0
    discount_rate: float = 0.0
    pv_degradation_rate: float = 0.005


def cost_params_from_config(
    costs_config: Optional[Dict[str, Any]] = None,
    financials_config: Optional[Dict[str, Any]] = None,
) -> CostParams:
    """Build :class:`CostParams` from BREOS cost and financial config keys.

    Missing keys fall back to the :class:`CostParams` dataclass defaults, so
    the config and direct-construction paths cannot diverge.
    """
    costs_config = costs_config or {}
    financials_config = financials_config or {}
    if "panel_wp" in costs_config:
        raise ValueError(
            "costs.panel_wp was removed in 0.7.0: CAPEX is priced at the selected module's own "
            "rating. Remove the key, or select a module with the wattage you meant."
        )
    defaults = CostParams()

    params = {
        param_key: costs_config.get(config_key, getattr(defaults, param_key))
        for config_key, param_key in COST_CONFIG_KEY_TO_PARAM.items()
    }
    # Historical lower-level callers can supply the two energy prices through
    # ``financials_config``; an explicit costs value still takes precedence.
    for key in ("electricity_cost", "electricity_sold_cost"):
        if key not in costs_config:
            params[key] = financials_config.get(key, getattr(defaults, key))
    params.update(
        dc_ac_ratio=costs_config.get("dc_ac_ratio", defaults.dc_ac_ratio),
        inflation_rate=financials_config.get("inflation_rate", defaults.inflation_rate),
        sell_price_inflation=financials_config.get("sell_price_inflation", defaults.sell_price_inflation),
        discount_rate=financials_config.get("discount_rate", defaults.discount_rate),
        pv_degradation_rate=financials_config.get("pv_degradation_rate", defaults.pv_degradation_rate),
    )
    return CostParams(**params)


def calculate_costs(
    n_modules: int,
    module_power_w: float,
    battery_capacity_wh: float = 0.0,
    cost_params: Optional[CostParams] = None,
) -> Dict[str, float]:
    """
    Calculate system costs (CAPEX) and return cost dictionary.

    Args:
        n_modules: Number of PV modules
        module_power_w: Power per module in Watts (STC)
        battery_capacity_wh: Battery capacity in Wh (0 for no battery)
        cost_params: Cost parameters

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

    # Maintenance
    annual_operation_cost = (
        cost_params.maintenance_cost_per_panel * n_modules
        + cost_params.maintenance_cost_fixed
        + cost_params.operation_cost
    )

    # Total CAPEX
    total_initial_cost = (
        pv_cost + inverter_cost + battery_cost + installation_cost + cost_params.land_cost + other_costs
    )

    return {
        "electricity_cost": cost_params.electricity_cost,
        "electricity_sold_cost": cost_params.electricity_sold_cost,
        "daily_power_cost": cost_params.daily_power_cost,
        "total_initial_cost": total_initial_cost,
        "annual_operation_cost": annual_operation_cost,
        "pv_cost": pv_cost,
        "inverter_cost": inverter_cost,
        "battery_cost": battery_cost,
        "installation_cost": installation_cost,
        "other_costs": other_costs,
    }


def replacement_booking_time(
    relative_years,
    year_fractions=None,
    has_replacement=None,
) -> np.ndarray:
    """Project time, in years from commissioning, at which replacements are booked.

    A battery replacement is a single transaction on the day the pack is
    swapped, not a flow spread over the year it falls in. Booking it at a
    calendar-year granularity is what makes the inflation and the discount
    disagree: inflating by ``(1 + i) ** (year - 1)`` values the outlay at the
    start of the year while discounting by ``(1 + d) ** year`` values it at
    the end, so neither matches the swap. This returns the instant both should
    use.

    Args:
        relative_years: 1-based projection years.
        year_fractions: Position of the swap within its year, in ``[0, 1)``.
            ``None`` or a non-finite entry falls back to
            :data:`DEFAULT_REPLACEMENT_YEAR_FRACTION`. Where a year holds more
            than one replacement, the caller supplies the cost-weighted mean
            instant; discounting is not linear in time, so that aggregate is
            exact only when the events share a year-fraction, which is
            adequate for the sub-annual pack lives it would take to reach it.
        has_replacement: Mask of years that carry a replacement. Years outside
            it return NaN.

    Returns:
        Array of project times. A swap 109 days into year 11 gives 10.299.
    """
    years = np.asarray(relative_years, dtype=float)
    if year_fractions is None:
        fractions = np.full(years.shape, np.nan, dtype=float)
    else:
        fractions = np.asarray(year_fractions, dtype=float)
    fractions = np.where(np.isfinite(fractions), fractions, DEFAULT_REPLACEMENT_YEAR_FRACTION)
    booked = years - 1.0 + fractions
    if has_replacement is None:
        return booked
    return np.where(np.asarray(has_replacement, dtype=bool), booked, np.nan)


def _booking_exponents(replacement_time: np.ndarray, relative_years) -> np.ndarray:
    """Fill the NaN of non-replacement years so the power is finite.

    Those years carry a zero outlay, so the exponent cannot reach the result;
    it only has to not be NaN.
    """
    years = np.asarray(relative_years, dtype=float)
    return np.where(np.isfinite(replacement_time), replacement_time, years)


def replacement_fraction_from_steps(replacement_steps, n_steps: int) -> float:
    """Mean within-year position of the steps at which a pack was swapped.

    The step index over the year's step count is the fraction of the year
    elapsed at the swap, on any regular timebase: a swap on day 109 of an
    hourly year is step 2616 of 8760, or 0.2986.

    Returns:
        Fraction in ``[0, 1)``, or NaN when the year holds no replacement.
    """
    steps = np.asarray(replacement_steps, dtype=float)
    if steps.size == 0 or n_steps <= 0:
        return float("nan")
    return float(np.mean(steps / float(n_steps)))


def replacement_fraction_by_year(years, replaced) -> pd.Series:
    """Within-year position of each year's replacement steps, from the ledger.

    ``Battery_Replaced`` marks the step in which the pack was swapped, so the
    step's position in its year is the fraction the economics needs. A year
    holding more than one swap reports the mean position; see
    :func:`replacement_booking_time` for what that aggregate costs.

    Returns:
        Series of fractions in ``[0, 1)`` indexed by the year label, holding
        only the years that carry a replacement.
    """
    frame = pd.DataFrame(
        {
            "Year": np.asarray(years),
            "Replaced": np.asarray(replaced, dtype=bool),
        }
    )
    fractions: Dict[Any, float] = {}
    for year, block in frame.groupby("Year", sort=True):
        fraction = replacement_fraction_from_steps(np.flatnonzero(block["Replaced"].to_numpy()), len(block))
        if np.isfinite(fraction):
            fractions[year] = fraction
    return pd.Series(fractions, dtype=float)


def _discount_annual_with_replacement(
    annual: pd.Series,
    replacement: pd.Series,
    discount_factors: pd.Series,
    replacement_exponents: np.ndarray,
    discount_rate: float,
) -> pd.Series:
    """Discount the annual system cost, taking the replacement from its own instant.

    Every other component is a flow spread over its year and keeps the
    year-end convention it has always had; the replacement is a single dated
    transaction, so it is pulled out, discounted from the swap, and added
    back.
    """
    exponents = np.asarray(replacement_exponents, dtype=float)
    replacement_discount = 1.0 / ((1.0 + discount_rate) ** exponents)
    outlay = pd.Series(np.asarray(replacement, dtype=float), index=annual.index)
    return (annual - outlay) * discount_factors + outlay * replacement_discount


def cost_analysis_projection(
    results_df: Optional[pd.DataFrame],
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
        results_df: DataFrame with ``Datetime``, ``Houseload``,
            ``Import_From_Grid``, and ``Sell_To_Grid``. Required only when
            ``yearly_summary_df`` is not supplied, because it feeds the legacy
            first-year estimation path alone; callers that already have actual
            yearly totals may pass ``None``. System production is
            ``PV_AC_To_Load + Battery_AC_To_Load_PV + PV_AC_Export``
            (``Sell_To_Grid`` is the export alias); legacy ``PV_Production``
            is accepted for compatibility.
        costs: Dictionary with cost parameters (from calculate_costs())
        num_years: Number of years to project
        inflation_rate: Annual inflation for electricity/operation costs
        sell_price_inflation: Annual inflation for sell price
        discount_rate: Discount rate for NPV calculations
        degradation_rate: Annual compound PV degradation rate, counted from
            the start of each year: year ``n`` production is scaled by
            ``(1 - degradation_rate) ** (n - 1)``, so year 1 has none. Used
            only when ``yearly_summary_df`` is not supplied.
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

        first_year_days = 365  # Assume full year

        # Factors
        inflation_factors = (1 + inflation_rate) ** (proj["Year"] - 1)
        sell_inflation_factors = (1 + sell_price_inflation) ** (proj["Year"] - 1)
        discount_factors = 1 / ((1 + discount_rate) ** proj["Year"])

        # Baseline (no system) - use the actual yearly demand from propagation.
        proj["Load_kWh"] = yearly_data["Load_kWh"].values
        proj["Cost_No_Sys_Annual"] = (
            proj["Load_kWh"] * costs["electricity_cost"] + first_year_days * costs["daily_power_cost"]
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

        # Battery replacement costs from propagation. The outlay is booked at
        # the instant the pack is swapped: inflated to it and, below,
        # discounted from it. The surrounding annual flows keep the year-end
        # convention they have always had.
        replacement_base = pd.to_numeric(yearly_data["Replacement_Cost"], errors="coerce").fillna(0.0).to_numpy()
        replacement_time = replacement_booking_time(
            proj["Year"].to_numpy(dtype=float),
            (
                pd.to_numeric(yearly_data["Replacement_Year_Fraction"], errors="coerce").to_numpy()
                if "Replacement_Year_Fraction" in yearly_data.columns
                else None
            ),
            replacement_base > 0.0,
        )
        replacement_exponents = _booking_exponents(replacement_time, proj["Year"].to_numpy(dtype=float))
        proj["Replacement_Time_Years"] = replacement_time
        proj["Cost_Replacement"] = replacement_base * (1 + inflation_rate) ** replacement_exponents

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
        proj["Cost_System_Annual_NPV"] = _discount_annual_with_replacement(
            proj["Cost_System_Annual"],
            proj["Cost_Replacement"],
            discount_factors,
            replacement_exponents,
            discount_rate,
        )
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
        proj.attrs["lcoe_eur_kwh"] = calculate_lcoe_from_projection(
            proj,
            total_investment=costs["total_initial_cost"],
            discount_rate=discount_rate,
        )

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
    if results_df is None:
        raise ValueError(
            "cost_analysis_projection requires results_df when yearly_summary_df is not provided: "
            "the first-year estimation path has nothing to estimate from"
        )
    df = results_df.copy()

    # Prepare datetime index
    if "Datetime" in df.columns:
        df.index = local_datetime_index(df.pop("Datetime"))

    df["Year"] = df.index.year
    df["Date"] = df.index.normalize()

    # Calculate hours per step for energy conversion
    hours_per_step = get_hours_per_step(freq)

    # Convert W (or whatever units, usually W in results_df) to kW
    # Result columns are typically in W (Power).
    # To get Energy (kWh), we need to multiply by hours_per_step and divide by 1000.

    # First convert columns to numeric, just in case
    df["System_AC_Production"] = system_ac_production_power(df)
    cols_to_numeric = ["System_AC_Production", "Houseload", "Import_From_Grid", "Sell_To_Grid", "Replacement_Cost"]
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Aggregate first year
    # Summing Power (W) gives sum(Watts). To get Wh, multiply by hours_per_step.
    # To get kWh, divide by 1000.
    # Replacement_Cost is already in currency (EUR), likely summed is correct (not power->energy).
    yearly = df[["System_AC_Production", "Houseload", "Import_From_Grid", "Sell_To_Grid"]].groupby(df["Year"]).sum()

    # Handle replacement cost separately if present (it's already simple sum, no kWh conversion needed)
    if "Replacement_Cost" in df.columns:
        yearly_replacement = df[["Replacement_Cost"]].groupby(df["Year"]).sum()
    else:
        yearly_replacement = pd.DataFrame(0.0, index=yearly.index, columns=["Replacement_Cost"])

    # The ledger marks the swap step, so the instant does not have to be
    # reconstructed downstream. Without the column the booking falls back to
    # the documented mid-year default.
    if "Battery_Replaced" in df.columns:
        replacement_year_fractions = replacement_fraction_by_year(df["Year"], df["Battery_Replaced"])
    else:
        replacement_year_fractions = pd.Series(dtype=float)

    # Scale to Energy (kWh)
    yearly = yearly * hours_per_step / 1000.0

    daily_counts = df.groupby("Year")["Date"].nunique()

    first_year_load = yearly["Houseload"].iloc[0]
    first_year_import = yearly["Import_From_Grid"].iloc[0]
    first_year_export = yearly["Sell_To_Grid"].iloc[0]
    first_year_pv = yearly["System_AC_Production"].iloc[0]
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

    # Battery replacement, taken from simulated years where available.
    # Simulation results only cover the simulated period: the App's
    # multi-year loop provides per-year replacement events for every
    # projection year, while a single-year run provides at most year 1 and
    # leaves later projection years without replacement costs.
    #
    # yearly_replacement is indexed by calendar year; proj['Year'] is the
    # relative year (1, 2, ...), so align via the simulation start year.
    start_year = df["Year"].min()

    replacement_base = np.zeros(len(proj), dtype=float)
    replacement_fraction = np.full(len(proj), np.nan, dtype=float)
    for position, relative_year in enumerate(proj["Year"]):
        actual_year = start_year + relative_year - 1
        if actual_year in yearly_replacement.index:
            replacement_base[position] = float(yearly_replacement.loc[actual_year, "Replacement_Cost"])
            if actual_year in replacement_year_fractions.index:
                replacement_fraction[position] = float(replacement_year_fractions.loc[actual_year])

    replacement_time = replacement_booking_time(
        proj["Year"].to_numpy(dtype=float),
        replacement_fraction,
        replacement_base > 0.0,
    )
    replacement_exponents = _booking_exponents(replacement_time, proj["Year"].to_numpy(dtype=float))
    proj["Replacement_Time_Years"] = replacement_time
    # The simulation logs replacement at the base (year-1) cost input, so
    # inflate to the swap instant here, and discount from the same instant.
    proj["Cost_Replacement"] = replacement_base * (1 + inflation_rate) ** replacement_exponents

    # Add to annual system cost
    proj["Cost_System_Annual"] += proj["Cost_Replacement"]

    proj["Cost_System_Cumulative"] = costs["total_initial_cost"] + proj["Cost_System_Annual"].cumsum()

    # Discounted values (NPV)
    proj["Cost_No_Sys_Annual_NPV"] = proj["Cost_No_Sys_Annual"] * discount_factors
    proj["Cost_System_Annual_NPV"] = _discount_annual_with_replacement(
        proj["Cost_System_Annual"],
        proj["Cost_Replacement"],
        discount_factors,
        replacement_exponents,
        discount_rate,
    )
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
    proj.attrs["lcoe_eur_kwh"] = calculate_lcoe_from_projection(
        proj,
        total_investment=costs["total_initial_cost"],
        discount_rate=discount_rate,
    )

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


def find_payback_year_exact(cost_projection: pd.DataFrame) -> Optional[float]:
    """Return the discounted payback year, interpolated linearly between years.

    The fractional counterpart of :func:`find_payback_year`, under the same
    rule: payback is where cumulative discounted savings first turn positive.
    A projection whose first row already has positive savings pays back at
    that row's year. A row of exactly zero savings is not payback; if the
    next row is positive, the crossing is at the zero row's year. Returns None
    when the savings never turn positive, or the projection has no savings
    column.
    """
    if "Savings_Cumulative_NPV" not in cost_projection.columns or cost_projection.empty:
        return None

    savings = cost_projection["Savings_Cumulative_NPV"].to_numpy(dtype=float)
    years = cost_projection["Year"].to_numpy(dtype=float)
    if savings[0] > 0.0:
        return float(years[0])

    for idx in range(1, len(savings)):
        if savings[idx] > 0.0:
            # savings[idx - 1] <= 0 < savings[idx], so the fraction is in [0, 1).
            fraction = -savings[idx - 1] / (savings[idx] - savings[idx - 1])
            return float(years[idx - 1] + fraction * (years[idx] - years[idx - 1]))
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
        degradation_rate: Annual compound PV degradation rate, counted from
            the start of each year: year ``t`` produces
            ``annual_production_kwh * (1 - degradation_rate) ** (t - 1)``.

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


def calculate_lcoe_from_projection(
    cost_projection: pd.DataFrame,
    total_investment: Optional[float] = None,
    discount_rate: float = 0.0,
    production_column: str = "PV_Production_kWh",
) -> float:
    """Calculate LCOE from a simulated multi-year projection.

    This variant is intended for simulation outputs that already contain
    year-by-year PV production and replacement costs. It uses system CAPEX,
    operation costs, and replacement costs as the cost basis; grid import
    charges, fixed grid charges, and export revenue are excluded because those
    are tariff outcomes rather than generation costs.

    Args:
        cost_projection: DataFrame from :func:`cost_analysis_projection`.
        total_investment: System CAPEX. If omitted, uses
            ``cost_projection.attrs["total_investment"]`` or infers it from
            the first cumulative/annual system-cost row.
        discount_rate: Discount rate used for production and annual costs.
        production_column: Column containing yearly production in kWh.

    Returns:
        LCOE in €/kWh.
    """
    if cost_projection.empty:
        return float("inf")
    if production_column not in cost_projection.columns:
        raise ValueError(f"cost_projection must include {production_column!r}")

    if total_investment is None:
        total_investment = cost_projection.attrs.get("total_investment")
    if total_investment is None:
        if {"Cost_System_Cumulative", "Cost_System_Annual"}.issubset(cost_projection.columns):
            first = (
                cost_projection.sort_values("Year").iloc[0]
                if "Year" in cost_projection.columns
                else cost_projection.iloc[0]
            )
            total_investment = float(first["Cost_System_Cumulative"] - first["Cost_System_Annual"])
        else:
            raise ValueError("total_investment is required when it cannot be inferred from cost_projection")

    years = (
        pd.to_numeric(cost_projection["Year"], errors="coerce")
        if "Year" in cost_projection.columns
        else pd.Series(range(1, len(cost_projection) + 1), index=cost_projection.index)
    )
    discount_factors = 1 / ((1 + discount_rate) ** years)

    production = pd.to_numeric(cost_projection[production_column], errors="coerce").fillna(0.0)
    operation = (
        pd.to_numeric(cost_projection["Cost_Operation"], errors="coerce").fillna(0.0)
        if "Cost_Operation" in cost_projection.columns
        else pd.Series(0.0, index=cost_projection.index)
    )
    replacement = (
        pd.to_numeric(cost_projection["Cost_Replacement"], errors="coerce").fillna(0.0)
        if "Cost_Replacement" in cost_projection.columns
        else pd.Series(0.0, index=cost_projection.index)
    )

    # The replacement is discounted from the swap instant when the projection
    # carries it, matching how the NPV above books the same outlay.
    if "Replacement_Time_Years" in cost_projection.columns:
        replacement_time = pd.to_numeric(cost_projection["Replacement_Time_Years"], errors="coerce")
        replacement_discount_factors = 1 / ((1 + discount_rate) ** replacement_time.fillna(years))
    else:
        replacement_discount_factors = discount_factors

    npv_costs = float(total_investment) + float(
        ((operation * discount_factors) + (replacement * replacement_discount_factors)).sum()
    )
    npv_production = float((production * discount_factors).sum())

    return npv_costs / npv_production if npv_production > 0 else float("inf")
