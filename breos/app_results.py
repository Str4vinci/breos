"""Result serialization helpers for App simulations."""

from __future__ import annotations

from typing import Any

import pandas as pd

from breos.app_config import ResolvedAppConfig
from breos.app_simulation import SimulationArtifacts
from breos.emissions import calculate_co2_savings
from breos.utils import get_hours_per_step


def monthly_to_dicts(results_df: pd.DataFrame, freq: str) -> list[dict[str, Any]]:
    """Convert first-year timestep results into monthly energy rows."""
    hours_per_step = get_hours_per_step(freq)
    df = results_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Datetime" in df.columns:
            df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)
            df.set_index("Datetime", inplace=True)
        else:
            raise ValueError("results_df must have a DatetimeIndex or Datetime column")

    monthly = df[["PV_Production", "Houseload", "Import_From_Grid", "Sell_To_Grid"]].resample("ME").sum()
    monthly = monthly * hours_per_step / 1000

    rows = []
    for idx, row in monthly.iterrows():
        pv = float(row["PV_Production"])
        consumption = float(row["Houseload"])
        export = float(row["Sell_To_Grid"])
        imported = float(row["Import_From_Grid"])
        self_consumption = pv - export
        rows.append(
            {
                "month": idx.strftime("%b"),
                "pv_kwh": round(pv, 2),
                "consumption_kwh": round(consumption, 2),
                "self_consumption_kwh": round(self_consumption, 2),
                "import_kwh": round(imported, 2),
                "export_kwh": round(export, 2),
                "grid_independence_pct": round((1 - imported / consumption) * 100, 2) if consumption > 0 else 0.0,
            }
        )
    return rows


def financial_to_dicts(cost_proj: pd.DataFrame, total_initial_cost: float) -> list[dict[str, Any]]:
    """Convert BREOS cost projection into the dashboard line-chart shape."""
    rows = [{"year": 0, "balance": round(-float(total_initial_cost), 2), "reference": 0.0}]
    for _, row in cost_proj.iterrows():
        rows.append(
            {
                "year": int(row["Year"]),
                "balance": round(float(row["Savings_Cumulative_NPV"]), 2),
                "reference": 0.0,
                "cost_with_system": round(float(row["Cost_System_Cumulative_NPV"]), 2),
                "cost_without_system": round(float(row["Cost_No_Sys_Cumulative_NPV"]), 2),
            }
        )
    return rows


def yearly_to_dicts(yearly_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert yearly summary DataFrame to a list of plain dicts."""
    rows = []
    for _, row in yearly_df.iterrows():
        item: dict[str, Any] = {
            "year": int(row["Year"]),
            "pv_kwh": round(float(row["PV_Production_kWh"]), 2),
            "consumption_kwh": round(float(row["Load_kWh"]), 2),
            "self_consumption_kwh": round(float(row["PV_Production_kWh"] - row["Export_kWh"]), 2),
            "import_kwh": round(float(row["Import_kWh"]), 2),
            "export_kwh": round(float(row["Export_kWh"]), 2),
            "grid_independence_pct": round(float(row["Grid_Independence_%"]), 2),
        }
        if row["Battery_SOH_%"] is not None:
            item["soh_pct"] = round(float(row["Battery_SOH_%"]), 2)
        rows.append(item)
    return rows


def build_result(
    cfg: dict[str, Any],
    resolved: ResolvedAppConfig,
    artifacts: SimulationArtifacts,
) -> dict[str, Any]:
    """Build the public JSON-serializable App result dictionary."""
    year1 = artifacts.yearly_df.iloc[0]
    yr1_pv = year1["PV_Production_kWh"]
    yr1_export = year1["Export_kWh"]
    yr1_import = year1["Import_kWh"]
    yr1_load = year1["Load_kWh"]
    self_consumption_kwh = yr1_pv - yr1_export
    self_consumption_pct = (self_consumption_kwh / yr1_pv * 100) if yr1_pv > 0 else 0.0
    grid_indep_y1 = year1["Grid_Independence_%"]

    total_initial = artifacts.costs["total_initial_cost"]
    npv_savings = float(artifacts.cost_projection["Savings_Cumulative_NPV"].iloc[-1])

    result: dict[str, Any] = {
        "n_modules": cfg["n_modules"],
        "pv_kwp": round(resolved.system_kwp, 3),
        "battery_kwh": cfg["battery_kwh"],
        "pv_production_kwh": round(float(yr1_pv), 2),
        "consumption_kwh": round(float(yr1_load), 2),
        "self_consumption_kwh": round(float(self_consumption_kwh), 2),
        "grid_import_kwh": round(float(yr1_import), 2),
        "grid_export_kwh": round(float(yr1_export), 2),
        "grid_independence_pct": round(float(grid_indep_y1), 2),
        "self_consumption_pct": round(float(self_consumption_pct), 2),
        "total_investment_eur": round(float(total_initial), 2),
        "payback_year": int(artifacts.payback_year) if artifacts.payback_year is not None else None,
        "npv_savings_eur": round(float(npv_savings), 2),
        "lcoe_eur_kwh": round(float(artifacts.lcoe), 4),
        "yearly": yearly_to_dicts(artifacts.yearly_df),
        "monthly": monthly_to_dicts(artifacts.first_year_results_df, cfg["resolution"]),
        "financial": financial_to_dicts(artifacts.cost_projection, total_initial),
    }

    if resolved.pv_arrays:
        result["pv_arrays"] = [dict(arr) for arr in resolved.pv_arrays]

    if cfg["battery_kwh"] > 0:
        result["battery_soh_end_pct"] = round(float(artifacts.current_soh), 2)
        result["battery_replacements"] = artifacts.total_replacements
        result["battery_replacement_cost_eur"] = round(float(artifacts.total_replacement_cost), 2)

    if resolved.emissions_params is not None:
        co2 = calculate_co2_savings(yr1_pv, self_consumption_kwh, resolved.emissions_params)
        lifetime_co2 = float(artifacts.cost_projection["CO2_Avoided_Total_Cumulative_kg"].iloc[-1])
        result["co2_avoided_year1_kg"] = round(co2["CO2_Avoided_Total_kg"], 2)
        result["co2_avoided_total_kg"] = round(lifetime_co2, 2)

    return result
