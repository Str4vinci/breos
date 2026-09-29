"""Result serialization helpers for App simulations."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from typing import Any, cast

import pandas as pd

from breos.app_config import ResolvedAppConfig
from breos.battery import LEDGER_SCHEMA_VERSION
from breos.economics import projection_rates_record
from breos.result_schema import RESULT_SCHEMA_VERSION
from breos.runners.app import SimulationArtifacts
from breos.tariffs import result_currency
from breos.utils import get_hours_per_step, local_datetime_index


def monthly_to_dicts(results_df: pd.DataFrame, freq: str) -> list[dict[str, Any]]:
    """Convert first-year timestep results into monthly energy rows."""
    hours_per_step = get_hours_per_step(freq)
    df = results_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Datetime" in df.columns:
            df.index = local_datetime_index(df.pop("Datetime"))
        else:
            raise ValueError("results_df must have a DatetimeIndex or Datetime column")

    columns = [
        "PV_DC",
        "PV_Production",
        "PV_AC_To_Load",
        "PV_Origin_Battery_AC_To_Load",
        "Houseload",
        "Import_From_Grid",
        "PV_AC_Export",
        "PV_DC_Curtailed",
    ]
    monthly = df[columns].resample("ME").sum()
    monthly = monthly * hours_per_step / 1000

    rows = []
    for idx, row in monthly.iterrows():
        pv_dc = float(row["PV_DC"])
        direct = float(row["PV_AC_To_Load"])
        battery = float(row["PV_Origin_Battery_AC_To_Load"])
        usable_pv = direct + battery + float(row["PV_AC_Export"])
        legacy_pv = float(row["PV_Production"])
        consumption = float(row["Houseload"])
        export = float(row["PV_AC_Export"])
        imported = float(row["Import_From_Grid"])
        self_consumption = direct + battery
        rows.append(
            {
                "month": cast(pd.Timestamp, idx).strftime("%b"),
                "pv_kwh": round(legacy_pv, 2),
                "pv_dc_generation_kwh": round(pv_dc, 2),
                "direct_pv_ac_load_kwh": round(direct, 2),
                "pv_origin_battery_ac_load_kwh": round(battery, 2),
                "usable_ac_system_production_kwh": round(usable_pv, 2),
                "curtailment_dc_kwh": round(float(row["PV_DC_Curtailed"]), 2),
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
    rows: list[dict[str, Any]] = [{"year": 0, "balance": round(-float(total_initial_cost), 2), "reference": 0.0}]
    for _, row in cost_proj.iterrows():
        rows.append(
            {
                "year": int(row["Year"]),
                "balance": round(float(row["Savings_Cumulative_NPV"]), 2),
                "reference": 0.0,
                "cost_with_system": round(float(row["Cost_System_Cumulative_NPV"]), 2),
                "cost_without_system": round(float(row["Cost_No_Sys_Cumulative_NPV"]), 2),
                # The year's component cashflows, escalated and not discounted
                # (ADR 0003 E7); a replacement is booked at its swap instant.
                "cost_import": round(float(row["Cost_Import"]), 2),
                "revenue_export": round(float(row["Revenue_Export"]), 2),
                "cost_operation": round(float(row["Cost_Operation"]), 2),
                "cost_fixed_charge": round(float(row["Cost_Daily"]), 2),
                "cost_replacement": round(float(row["Cost_Replacement"]), 2),
                "replacement_time_years": (
                    round(float(row["Replacement_Time_Years"]), 4) if pd.notna(row["Replacement_Time_Years"]) else None
                ),
            }
        )
    return rows


def yearly_to_dicts(yearly_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert yearly summary DataFrame to a list of plain dicts."""
    rows = []
    for _, row in yearly_df.iterrows():
        item: dict[str, Any] = {
            "year": int(row["Year"]),
            "pv_kwh": round(float(row["Legacy_PV_Production_kWh"]), 2),
            "pv_dc_generation_kwh": round(float(row["PV_DC_Generation_kWh"]), 2),
            "direct_pv_ac_load_kwh": round(float(row["Direct_PV_AC_Load_kWh"]), 2),
            "pv_origin_battery_ac_load_kwh": round(float(row["PV_Origin_Battery_AC_Load_kWh"]), 2),
            "usable_ac_system_production_kwh": round(float(row["PV_Production_kWh"]), 2),
            "curtailment_dc_kwh": round(float(row["Curtailment_DC_kWh"]), 2),
            "consumption_kwh": round(float(row["Load_kWh"]), 2),
            "self_consumption_kwh": round(float(row["Self_Consumption_kWh"]), 2),
            "import_kwh": round(float(row["Import_kWh"]), 2),
            "export_kwh": round(float(row["Export_kWh"]), 2),
            "grid_independence_pct": round(float(row["Grid_Independence_%"]), 2),
        }
        if row["Battery_SOH_%"] is not None:
            item["soh_pct"] = round(float(row["Battery_SOH_%"]), 2)
        rows.append(item)
    return rows


def _package_version() -> str:
    try:
        return version("breos")
    except PackageNotFoundError:
        return "unknown"


def _provenance(
    cfg: dict[str, Any],
    resolved: ResolvedAppConfig,
    artifacts: SimulationArtifacts,
    input_repairs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    normalized_cfg = {
        **cfg,
        "location": {
            "preset": resolved.loc_key,
            "latitude": resolved.lat,
            "longitude": resolved.lon,
            "timezone": resolved.timezone,
        },
        "tilt": resolved.tilt,
        "azimuth": resolved.azimuth,
        "axis_azimuth": resolved.axis_azimuth,
        "pv_module": cfg.get("pv_module") or resolved.pv_module_key,
    }
    # Round-trip through JSON to guarantee only public, serializable scalar types.
    normalized_cfg = json.loads(json.dumps(normalized_cfg, default=str))
    weather = json.loads(json.dumps(artifacts.weather_metadata, default=str))
    weather.setdefault("latitude", resolved.lat)
    weather.setdefault("longitude", resolved.lon)
    provenance = {
        "breos_version": _package_version(),
        "ledger_schema_version": LEDGER_SCHEMA_VERSION,
        # Every money field is in this currency; BREOS does not convert.
        "currency": result_currency(resolved.tariff),
        "resolved_config": normalized_cfg,
        "weather": weather,
        "load_profile": json.loads(json.dumps(artifacts.load_profile_metadata, default=str)),
        "resolution": cfg["resolution"],
        "timezone": resolved.timezone,
        "start_date": cfg["start_date"],
        "pv_model": {"bifacial": deepcopy(artifacts.pv_loss_waterfall["bifacial"])},
        "degradation": artifacts.degradation_summary,
        # Which dispatch implementation produced these numbers, and the
        # toolchain it ran on. A bit-identity claim cannot be checked after the
        # fact without one, so it is recorded on every run, not only on
        # benchmarks. Same keys as the Monte Carlo block, from the same code.
        "execution": artifacts.execution,
        "economics": projection_rates_record(cfg),
    }
    # Only runs given repair reports carry the key, so existing results are
    # unchanged. An empty list is recorded as given.
    if input_repairs is not None:
        provenance["input_repairs"] = deepcopy(input_repairs)
    # Flat-price runs carry no tariff block, so their results are unchanged.
    if artifacts.tariff is not None:
        provenance["tariff"] = deepcopy(artifacts.tariff)
    if artifacts.smart_charging is not None:
        provenance["smart_charging"] = {
            key: value
            for key, value in artifacts.smart_charging.items()
            if key not in ("initial_stored_energy", "final_stored_energy")
        }
    return provenance


def _round2(value: float) -> float:
    """Round to 0.01; adding 0.0 turns a rounded -0.0 residue into 0.0."""
    return round(float(value), 2) + 0.0


def smart_charging_to_dict(artifacts: SimulationArtifacts) -> dict[str, Any]:
    """The smart-charging block of an App result: grid charge, delivery by origin, and terminal state."""
    assert artifacts.smart_charging is not None
    yearly = artifacts.yearly_df
    rows = []
    for _, row in yearly.iterrows():
        item: dict[str, Any] = {
            "year": int(row["Year"]),
            "grid_charge_ac_kwh": _round2(row["Grid_AC_To_Battery_kWh"]),
            "grid_charge_conversion_loss_kwh": _round2(row["Grid_Charge_Conversion_Loss_kWh"]),
            "battery_ac_to_load_kwh": {
                "pv_origin": _round2(row["PV_Origin_Battery_AC_Load_kWh"]),
                "grid_origin": _round2(row["Grid_Origin_Battery_AC_Load_kWh"]),
                "unattributed": _round2(
                    row["Battery_AC_To_Load_kWh"]
                    - row["PV_Origin_Battery_AC_Load_kWh"]
                    - row["Grid_Origin_Battery_AC_Load_kWh"]
                ),
            },
        }
        # Year-1 prices, like the tariff's other money columns.
        if "Grid_Charge_Cost" in row:
            item["grid_charge_cost_year1_prices"] = _round2(row["Grid_Charge_Cost"])
        rows.append(item)
    stored = {
        key: {name: _round2(value) for name, value in artifacts.smart_charging[key].items()}
        for key in ("initial_stored_energy", "final_stored_energy")
    }
    return {
        "mode": artifacts.smart_charging["mode"],
        "terminal_convention": artifacts.smart_charging["terminal_convention"],
        **stored,
        "yearly": rows,
    }


def build_result(
    cfg: dict[str, Any],
    resolved: ResolvedAppConfig,
    artifacts: SimulationArtifacts,
    *,
    input_repairs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the public JSON-serializable App result dictionary.

    ``input_repairs`` holds strict-JSON repair reports (see
    :func:`breos.repair.input_repair_records`) for ``provenance``.
    """
    year1 = artifacts.yearly_df.iloc[0]
    yr1_pv = year1["PV_Production_kWh"]
    legacy_yr1_pv = year1["Legacy_PV_Production_kWh"]
    yr1_export = year1["Export_kWh"]
    yr1_import = year1["Import_kWh"]
    yr1_load = year1["Load_kWh"]
    self_consumption_kwh = year1["Self_Consumption_kWh"]
    self_consumption_pct = (self_consumption_kwh / yr1_pv * 100) if yr1_pv > 0 else 0.0
    grid_indep_y1 = year1["Grid_Independence_%"]

    total_initial = artifacts.costs["total_initial_cost"]
    npv_savings = float(artifacts.cost_projection["Savings_Cumulative_NPV"].iloc[-1])
    lcoe = float(artifacts.lcoe)

    result: dict[str, Any] = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "n_modules": cfg["n_modules"],
        "pv_kwp": round(resolved.system_kwp, 3),
        "battery_kwh": cfg["battery_kwh"],
        # Compatibility field: legacy potential AC conversion of uncurtailed PV.
        "pv_production_kwh": round(float(legacy_yr1_pv), 2),
        "pv_dc_generation_kwh": round(float(year1["PV_DC_Generation_kWh"]), 2),
        "direct_pv_ac_load_kwh": round(float(year1["Direct_PV_AC_Load_kWh"]), 2),
        "pv_origin_battery_ac_load_kwh": round(float(year1["PV_Origin_Battery_AC_Load_kWh"]), 2),
        "usable_ac_system_production_kwh": round(float(yr1_pv), 2),
        "curtailment_dc_kwh": round(float(year1["Curtailment_DC_kWh"]), 2),
        "consumption_kwh": round(float(yr1_load), 2),
        "self_consumption_kwh": round(float(self_consumption_kwh), 2),
        "grid_import_kwh": round(float(yr1_import), 2),
        "grid_export_kwh": round(float(yr1_export), 2),
        "grid_independence_pct": round(float(grid_indep_y1), 2),
        "self_consumption_pct": round(float(self_consumption_pct), 2),
        "total_investment": round(float(total_initial), 2),
        "payback_year": int(artifacts.payback_year) if artifacts.payback_year is not None else None,
        "npv_savings": round(float(npv_savings), 2),
        "lcoe_per_kwh": round(lcoe, 4) if math.isfinite(lcoe) else None,
        "yearly": yearly_to_dicts(artifacts.yearly_df),
        "monthly": monthly_to_dicts(artifacts.first_year_results_df, cfg["resolution"]),
        "financial": financial_to_dicts(artifacts.cost_projection, total_initial),
        # Copied, so editing a result cannot reach the stored run that
        # App.revalue prices again.
        "pv_loss_waterfall": deepcopy(artifacts.pv_loss_waterfall),
        "provenance": _provenance(cfg, resolved, artifacts, input_repairs),
        "degradation": deepcopy(artifacts.degradation_summary),
    }
    # One copy, as provenance and the top level have always shared it.
    result["provenance"]["degradation"] = result["degradation"]

    # The year-1 money components at year-1 prices, before escalation and
    # discounting (ADR 0003 E7), at the top level so the sweep CSV keeps them.
    result.update(
        {
            "grid_import_cost_year1_prices": _round2(year1["Import_Cost"]),
            "grid_export_revenue_year1_prices": _round2(year1["Export_Revenue"]),
            "fixed_charge_year1_prices": _round2(year1["Fixed_Charge"]),
            "no_system_import_cost_year1_prices": _round2(year1["Baseline_Import_Cost"]),
        }
    )

    if resolved.pv_arrays:
        result["pv_arrays"] = [dict(arr) for arr in resolved.pv_arrays]
    if artifacts.smart_charging is not None:
        result["smart_charging"] = smart_charging_to_dict(artifacts)
        # The part of grid_import_cost_year1_prices bought to charge the battery.
        result["grid_charge_cost_year1_prices"] = _round2(year1["Grid_Charge_Cost"])

    if cfg["battery_kwh"] > 0:
        soh_digits = 1 if cfg["degradation_engine"] == "blast" else 2
        result["battery_soh_end_pct"] = round(float(artifacts.current_soh), soh_digits)
        result["battery_replacements"] = artifacts.total_replacements
        # At t = 0 prices, neither inflated nor discounted; the discounted
        # total is the one the NPV counts.
        result["battery_replacement_cost_t0_prices"] = round(float(artifacts.total_replacement_cost), 2)
        result["battery_replacement_cost_npv"] = round(
            float(artifacts.cost_projection.attrs["replacement_cost_npv"]), 2
        )
        if cfg["degradation_engine"] == "blast":
            for row in result["yearly"]:
                if "soh_pct" in row:
                    row["soh_pct"] = round(float(row["soh_pct"]), 1)

    if resolved.emissions_params is not None:
        # Read from the projection, which computes each year's CO2 once for
        # every runner (#183).
        projection = artifacts.cost_projection
        co2 = {
            column: float(projection[column].iloc[0])
            for column in ("CO2_Avoided_SelfConsumed_kg", "CO2_Avoided_Export_kg", "CO2_Avoided_Total_kg")
        }
        lifetime_self = float(projection.attrs["lifetime_co2_avoided_self_consumed_kg"])
        lifetime_export = float(projection.attrs["lifetime_co2_avoided_export_kg"])
        lifetime_total = float(projection.attrs["lifetime_co2_avoided_total_kg"])
        result.update(
            {
                "co2_avoided_self_consumption_year1_kg": round(co2["CO2_Avoided_SelfConsumed_kg"], 2),
                "co2_avoided_export_year1_kg": round(co2["CO2_Avoided_Export_kg"], 2),
                "co2_avoided_total_year1_kg": round(co2["CO2_Avoided_Total_kg"], 2),
                "co2_avoided_self_consumption_lifetime_kg": round(lifetime_self, 2),
                "co2_avoided_export_lifetime_kg": round(lifetime_export, 2),
                "co2_avoided_total_lifetime_kg": round(lifetime_total, 2),
                # Compatibility aliases retained for the pre-ledger public schema.
                "co2_avoided_year1_kg": round(co2["CO2_Avoided_Total_kg"], 2),
                "co2_avoided_total_kg": round(lifetime_total, 2),
            }
        )

    return result
