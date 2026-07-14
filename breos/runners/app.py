"""Simulation orchestration for the public App facade."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from breos.app_config import ResolvedAppConfig, build_costs_dict
from breos.app_inputs import AppRuntimeDependencies, prepare_simulation_inputs
from breos.battery import BatteryConfig, simulate_energy_balance
from breos.economics import calculate_lcoe_from_projection, cost_analysis_projection, find_payback_year
from breos.solar import PVProductionBreakdown
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
    pv_loss_waterfall: dict[str, Any]
    weather_metadata: dict[str, Any]


LEDGER_SCHEMA_VERSION = "1.0"


def _series_energy_kwh(series: pd.Series, freq: str) -> float:
    """Convert a power series in W to energy in kWh."""
    return float(series.fillna(0.0).sum() * get_hours_per_step(freq) / 1000.0)


def _rounded(value: float, digits: int = 2) -> float:
    """Round JSON-facing floats after normalising pandas/numpy scalars."""
    return round(float(value), digits)


def _waterfall_stage(key: str, label: str, energy_kwh: float, previous_kwh: float | None = None) -> dict[str, Any]:
    """Build one ordered stage row for the public loss waterfall."""
    stage: dict[str, Any] = {
        "key": key,
        "label": label,
        "energy_kwh": _rounded(energy_kwh),
    }
    if previous_kwh is not None:
        delta = energy_kwh - previous_kwh
        stage["delta_kwh"] = _rounded(delta)
        stage["delta_pct_of_previous"] = _rounded((delta / previous_kwh * 100.0) if previous_kwh else 0.0)
    return stage


def _build_pv_loss_waterfall(
    pv_breakdown: PVProductionBreakdown,
    first_year_results_df: pd.DataFrame,
    cfg: dict[str, Any],
    resolved: ResolvedAppConfig,
) -> dict[str, Any]:
    """Build a JSON-serializable year-1 PV loss waterfall."""
    freq = cfg["resolution"]
    horizontal_dc = _series_energy_kwh(pv_breakdown.horizontal_reference_dc, freq)
    poa_dc = _series_energy_kwh(pv_breakdown.poa_global_dc, freq)
    effective_dc = _series_energy_kwh(pv_breakdown.effective_irradiance_dc, freq)
    module_dc = _series_energy_kwh(pv_breakdown.module_dc, freq)
    dc_after_static = _series_energy_kwh(pv_breakdown.dc_after_static_losses, freq)
    dc_after_degradation = _series_energy_kwh(first_year_results_df["PV_DC"], freq)

    pv_peak_w = cfg["n_modules"] * resolved.avg_module_power_w
    loading_ratio = cfg["inverter_loading_ratio"]
    inverter_ac_capacity_w = pv_peak_w / loading_ratio if loading_ratio and loading_ratio > 0 else 0.0

    def e(column: str) -> float:
        return _series_energy_kwh(first_year_results_df[column], freq)

    pv_dc_to_battery = e("PV_DC_To_Battery")
    pv_dc_to_inverter = e("PV_DC_To_Inverter")
    curtailment = e("PV_DC_Curtailed")
    direct_pv_ac = e("PV_AC_To_Load")
    export_ac = e("PV_AC_Export")
    battery_ac = e("Battery_AC_To_Load")
    pv_origin_battery_ac = e("Battery_AC_To_Load_PV")
    inverter_conversion = e("Inverter_Loss")
    direct_pv_conversion = e("PV_Direct_Inverter_Loss")
    battery_discharge_conversion = e("Battery_Inverter_Loss")

    pvwatts_components = {
        name: _rounded(_series_energy_kwh(loss, freq)) for name, loss in pv_breakdown.pvwatts_component_losses.items()
    }
    empty_series = pd.Series(dtype=float)
    dispatch = {
        "curtailment_kwh": _rounded(curtailment),
        "battery_charge_loss_kwh": _rounded(
            _series_energy_kwh(first_year_results_df.get("Battery_Charge_Loss", empty_series), freq)
        ),
        "battery_discharge_loss_kwh": _rounded(
            _series_energy_kwh(first_year_results_df.get("Battery_Discharge_Loss", empty_series), freq)
        ),
        "battery_standby_loss_kwh": _rounded(
            _series_energy_kwh(first_year_results_df.get("Battery_Standby_Loss", empty_series), freq)
        ),
    }
    dispatch["battery_round_trip_loss_kwh"] = _rounded(
        dispatch["battery_charge_loss_kwh"] + dispatch["battery_discharge_loss_kwh"]
    )

    stages = [
        _waterfall_stage("horizontal_reference_dc", "Horizontal irradiance reference", horizontal_dc),
        _waterfall_stage("transposition", "Plane-of-array transposition", poa_dc, horizontal_dc),
        _waterfall_stage("iam", "Incidence-angle modifier", effective_dc, poa_dc),
        _waterfall_stage("temperature", "Cell temperature", module_dc, effective_dc),
        _waterfall_stage("pvwatts_static", "Static PVWatts losses", dc_after_static, module_dc),
        _waterfall_stage("year_1_degradation", "Year 1 PV degradation", dc_after_degradation, dc_after_static),
    ]

    battery_begin = float(first_year_results_df["Battery_Energy_Beginning"].iloc[0]) / 1000.0
    battery_end = float(first_year_results_df["Battery_Energy_End"].iloc[-1]) / 1000.0
    battery_charge_stored = e("Battery_Charge_Stored")
    battery_discharge_dc = e("Battery_Discharge_DC")
    standby = e("Standby_Loss")
    capacity_window = e("Capacity_Window_Loss")
    replacement_removed = e("Battery_Replacement_Energy_Removed")
    replacement_added = e("Battery_Replacement_Energy_Added")
    dispatch.update(
        {
            "capacity_window_loss_kwh": _rounded(capacity_window),
            "replacement_energy_removed_kwh": _rounded(replacement_removed),
            "replacement_energy_added_kwh": _rounded(replacement_added),
            "stored_energy_report": "energy_balance.battery_stored_energy",
        }
    )
    battery_residual = (
        battery_begin
        + battery_charge_stored
        + replacement_added
        - battery_discharge_dc
        - standby
        - capacity_window
        - replacement_removed
        - battery_end
    )

    return {
        "basis": "year_1",
        "unit": "kWh",
        "flow_unit": "kWh per year",
        "state_unit": "kWh at period boundary",
        "ledger_schema_version": LEDGER_SCHEMA_VERSION,
        "stages": stages,
        "pvwatts": {
            "components_pct": {name: float(value) for name, value in pv_breakdown.pvwatts_components_pct.items()},
            "components_kwh": pvwatts_components,
            "combined_pct": _rounded(pv_breakdown.pvwatts_combined_pct, digits=4),
            "combined_kwh": _rounded(module_dc - dc_after_static),
        },
        "inverter": {
            "ac_capacity_kw": _rounded(inverter_ac_capacity_w / 1000.0, digits=3),
            "efficiency_pct": _rounded(cfg["inverter_efficiency"] * 100.0),
            "conversion_loss_kwh": _rounded(inverter_conversion),
            "direct_pv_conversion_loss_kwh": _rounded(direct_pv_conversion),
            "battery_discharge_conversion_loss_kwh": _rounded(battery_discharge_conversion),
        },
        "dispatch": dispatch,
        "energy_balance": {
            "pv_dc": {
                "generation_kwh": _rounded(dc_after_degradation),
                "to_inverter_kwh": _rounded(pv_dc_to_inverter),
                "to_battery_kwh": _rounded(pv_dc_to_battery),
                "curtailed_kwh": _rounded(curtailment),
                "residual_kwh": _rounded(dc_after_degradation - pv_dc_to_inverter - pv_dc_to_battery - curtailment, 6),
            },
            "ac_delivery": {
                "direct_pv_to_load_kwh": _rounded(direct_pv_ac),
                "pv_origin_battery_to_load_kwh": _rounded(pv_origin_battery_ac),
                "battery_to_load_all_origins_kwh": _rounded(battery_ac),
                "export_kwh": _rounded(export_ac),
                "usable_system_production_kwh": _rounded(direct_pv_ac + pv_origin_battery_ac + export_ac),
            },
            "battery_stored_energy": {
                "beginning_kwh": _rounded(battery_begin),
                "charge_stored_kwh": _rounded(battery_charge_stored),
                "discharge_dc_kwh": _rounded(battery_discharge_dc),
                "standby_loss_kwh": _rounded(standby),
                "capacity_window_loss_kwh": _rounded(capacity_window),
                "replacement_energy_removed_kwh": _rounded(replacement_removed),
                "replacement_energy_added_kwh": _rounded(replacement_added),
                "ending_kwh": _rounded(battery_end),
                "residual_kwh": _rounded(battery_residual, 6),
            },
        },
    }


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
    carried_energy_wh: float | None = None
    carried_pv_origin_energy_wh: float | None = None

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
                max_charge_power_w=cfg["battery_max_charge_power_w"],
                max_discharge_power_w=cfg["battery_max_discharge_power_w"],
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

        state_kwargs: dict[str, float] = {}
        if carried_energy_wh is not None:
            state_kwargs = {
                "initial_energy_wh": carried_energy_wh,
                "initial_pv_origin_energy_wh": carried_pv_origin_energy_wh or 0.0,
            }

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
            **state_kwargs,
        )

        if has_battery:
            carried_energy_wh = float(results_df["Battery_Energy_End"].iloc[-1])
            carried_pv_origin_energy_wh = float(results_df["Battery_PV_Origin_Energy_End"].iloc[-1])

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

        pv_dc_kwh = _series_energy_kwh(results_df["PV_DC"], freq)
        legacy_pv_kwh = _series_energy_kwh(results_df["PV_Production"], freq)
        direct_pv_ac_kwh = _series_energy_kwh(results_df["PV_AC_To_Load"], freq)
        pv_origin_battery_ac_kwh = _series_energy_kwh(results_df["Battery_AC_To_Load_PV"], freq)
        total_load = (results_df["Houseload"].sum() / 1000) * hours_per_step
        total_import = (results_df["Import_From_Grid"].sum() / 1000) * hours_per_step
        total_export = (results_df["Sell_To_Grid"].sum() / 1000) * hours_per_step
        total_pv_kwh = direct_pv_ac_kwh + pv_origin_battery_ac_kwh + total_export
        grid_indep = (1 - total_import / total_load) * 100 if total_load > 0 else 0

        yearly_summaries.append(
            {
                "Year": year_idx + 1,
                "PV_Production_kWh": total_pv_kwh,
                "Legacy_PV_Production_kWh": legacy_pv_kwh,
                "PV_DC_Generation_kWh": pv_dc_kwh,
                "Direct_PV_AC_Load_kWh": direct_pv_ac_kwh,
                "PV_Origin_Battery_AC_Load_kWh": pv_origin_battery_ac_kwh,
                "Self_Consumption_kWh": direct_pv_ac_kwh + pv_origin_battery_ac_kwh,
                "Curtailment_DC_kWh": _series_energy_kwh(results_df["PV_DC_Curtailed"], freq),
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
        sell_price_inflation=cfg["sell_price_inflation"],
        discount_rate=cfg["discount_rate"],
        freq=freq,
        yearly_summary_df=yearly_df,
        total_replacement_cost=total_replacement_cost,
        emissions_params=resolved.emissions_params,
    )

    lcoe = calculate_lcoe_from_projection(
        cost_projection,
        total_investment=costs["total_initial_cost"],
        discount_rate=cfg["discount_rate"],
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
        pv_loss_waterfall=_build_pv_loss_waterfall(inputs.pv_breakdown, first_year_results_df, cfg, resolved),
        weather_metadata=dict(
            inputs.weather.attrs.get(
                "breos_weather_metadata",
                {
                    "source": "runtime_dependency_or_unknown",
                    "note": "The injected weather provider did not expose source metadata.",
                },
            )
        ),
    )
