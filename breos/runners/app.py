"""Simulation orchestration for the public App facade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from breos.app_config import DEFAULTS, ResolvedAppConfig, default_module_key
from breos.app_inputs import AppRuntimeDependencies, prepare_simulation_inputs, prepare_simulation_inputs_cached
from breos.battery import LEDGER_SCHEMA_VERSION
from breos.degradation.results import DegradationEngineName, build_degradation_summary_from_state
from breos.economics import find_payback_year
from breos.execution import (
    DEFAULT_EXECUTION_BACKEND,
    aggregate_jit_cache_states,
    backend_provenance,
    is_pv_only_dispatch,
)
from breos.load_profiles import LOAD_PROFILE_METADATA_KEY
from breos.projection import ProjectionYear, run_projection, value_projection
from breos.pv_modules import get_module
from breos.smart_charging import resolve_instructions, smart_charging_provenance, stored_energy_by_origin
from breos.solar import PVProductionBreakdown
from breos.tariffs import tariff_provenance
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
    load_profile_metadata: dict[str, Any]
    degradation_summary: dict[str, Any]
    execution: dict[str, Any]
    # The resolved tariff's provenance; None on flat prices.
    tariff: dict[str, Any] | None = None
    # Smart-charging provenance and the project's first and last stored
    # energy by origin; None without fixed-target smart charging.
    smart_charging: dict[str, Any] | None = None


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


def _bifacial_summary(
    cfg: dict[str, Any],
    resolved: ResolvedAppConfig,
    front_effective_dc_kwh: float,
    rear_gain_dc_kwh: float,
) -> dict[str, Any]:
    """Build JSON-safe bifacial configuration and year-1 gain provenance."""
    default_model = cfg.get("bifacial_model", DEFAULTS["bifacial_model"])
    default_gcr = cfg.get("gcr", DEFAULTS["gcr"])
    default_height = cfg.get("pvrow_height")
    default_pitch = cfg.get("pvrow_pitch")
    resolved_arrays = getattr(resolved, "pv_arrays", None)
    if resolved_arrays:
        arrays = resolved_arrays
    else:
        arrays = [
            {
                "modules": cfg["n_modules"],
                "module": cfg.get("pv_module") or default_module_key(),
                "bifacial_model": default_model,
                "gcr": default_gcr,
                "pvrow_height": default_height,
                "pvrow_pitch": default_pitch,
            }
        ]

    rows: list[dict[str, Any]] = []
    models: set[str] = set()
    for index, array in enumerate(arrays):
        model = str(array.get("bifacial_model", default_model)).strip().lower()
        module_key = array.get("module") or cfg.get("pv_module") or default_module_key()
        module = get_module(module_key)
        models.add(model)
        row: dict[str, Any] = {
            "array_index": index,
            "modules": int(array["modules"]),
            "module": module_key,
            "model": model,
            "bifaciality": float(module.bifaciality) if module.bifaciality is not None else None,
        }
        if model != "none":
            row.update(
                {
                    "gcr": float(array.get("gcr", default_gcr)),
                    "pvrow_height": float(array.get("pvrow_height", default_height)),
                    "pvrow_pitch": float(array.get("pvrow_pitch", default_pitch)),
                }
            )
        rows.append(row)

    rear_gain_dc_kwh = max(0.0, rear_gain_dc_kwh)
    return {
        "enabled": any(model != "none" for model in models),
        "model": next(iter(models)) if len(models) == 1 else "mixed",
        "rear_gain_effective_dc_kwh": _rounded(rear_gain_dc_kwh),
        "rear_gain_pct_of_front_effective": _rounded(
            rear_gain_dc_kwh / front_effective_dc_kwh * 100.0 if front_effective_dc_kwh else 0.0
        ),
        "arrays": rows,
    }


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
    front_effective_dc = _series_energy_kwh(pv_breakdown.front_effective_irradiance_dc, freq)
    rear_gain_dc = _series_energy_kwh(pv_breakdown.rear_gain_dc, freq)
    effective_dc = _series_energy_kwh(pv_breakdown.effective_irradiance_dc, freq)
    module_dc = _series_energy_kwh(pv_breakdown.module_dc, freq)
    dc_after_static = _series_energy_kwh(pv_breakdown.dc_after_static_losses, freq)
    # Module age is counted at the start of each year, so year 1 has no PV
    # degradation and its dispatched PV DC equals the static-loss stage.
    pv_dc_generation = _series_energy_kwh(first_year_results_df["PV_DC"], freq)

    inverter_ac_capacity_w = resolved.inverter_ac_capacity_w or 0.0

    def e(column: str) -> float:
        return _series_energy_kwh(first_year_results_df[column], freq)

    pv_dc_to_battery = e("PV_DC_To_Battery")
    pv_dc_to_inverter = e("PV_DC_To_Inverter")
    curtailment = e("PV_DC_Curtailed")
    direct_pv_ac = e("PV_AC_To_Load")
    export_ac = e("PV_AC_Export")
    battery_ac = e("Battery_AC_To_Load")
    pv_origin_battery_ac = e("PV_Origin_Battery_AC_To_Load")
    inverter_conversion = e("Inverter_Loss")
    direct_pv_conversion = e("PV_Direct_Inverter_Loss")
    battery_discharge_conversion = e("Battery_Inverter_Loss")

    pvwatts_components = {
        name: _rounded(_series_energy_kwh(loss, freq)) for name, loss in pv_breakdown.pvwatts_component_losses.items()
    }
    empty_series = pd.Series(dtype=float)
    dispatch: dict[str, Any] = {
        "curtailment_kwh": _rounded(curtailment),
        "battery_charge_loss_kwh": _rounded(
            _series_energy_kwh(first_year_results_df.get("Battery_Charge_Loss", empty_series), freq)
        ),
        "battery_discharge_loss_kwh": _rounded(
            _series_energy_kwh(first_year_results_df.get("Battery_Discharge_Loss", empty_series), freq)
        ),
        "battery_standby_loss_kwh": _rounded(
            _series_energy_kwh(first_year_results_df.get("Standby_Loss", empty_series), freq)
        ),
    }
    dispatch["battery_round_trip_loss_kwh"] = _rounded(
        dispatch["battery_charge_loss_kwh"] + dispatch["battery_discharge_loss_kwh"]
    )

    stages = [
        _waterfall_stage("horizontal_reference_dc", "Horizontal irradiance reference", horizontal_dc),
        _waterfall_stage("transposition", "Plane-of-array transposition", poa_dc, horizontal_dc),
        _waterfall_stage("iam", "Front-side incidence-angle modifier", front_effective_dc, poa_dc),
        _waterfall_stage("bifacial_rear_gain", "Bifacial rear gain", effective_dc, front_effective_dc),
        _waterfall_stage("temperature", "Cell temperature", module_dc, effective_dc),
        _waterfall_stage("pvwatts_static", "Static PVWatts losses", dc_after_static, module_dc),
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
        "bifacial": _bifacial_summary(cfg, resolved, front_effective_dc, rear_gain_dc),
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
                "generation_kwh": _rounded(pv_dc_generation),
                "to_inverter_kwh": _rounded(pv_dc_to_inverter),
                "to_battery_kwh": _rounded(pv_dc_to_battery),
                "curtailed_kwh": _rounded(curtailment),
                "residual_kwh": _rounded(pv_dc_generation - pv_dc_to_inverter - pv_dc_to_battery - curtailment, 6),
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
    # Resolve the backend before anything is fetched or computed. Input
    # preparation can hit the network for weather, so a missing optional
    # dependency should be reported now rather than after a download.
    # resolve_app_config always supplies this; the default covers direct
    # callers that build a cfg dict themselves, and it is the reference
    # implementation rather than the fast one.
    execution_backend = cfg.get("execution_backend", DEFAULT_EXECUTION_BACKEND)
    battery_kwh = cfg["battery_kwh"]
    battery_wh = battery_kwh * 1000
    # Asked the way the dispatch asks it, so the run and its provenance agree
    # on what counts as PV-only.
    has_battery = not is_pv_only_dispatch(
        battery_wh,
        cfg.get("battery_max_soc", DEFAULTS["battery_max_soc"]),
        cfg.get("battery_min_soc", DEFAULTS["battery_min_soc"]),
    )
    execution = backend_provenance(execution_backend, pv_only=not has_battery)

    inputs = prepare_simulation_inputs_cached(cfg, resolved, deps, prepare=prepare_simulation_inputs)

    projection_years = cfg["projection_years"]
    degradation_rate = cfg["pv_degradation_rate"]

    def year_inputs(year_idx: int) -> ProjectionYear:
        pv_degradation_factor = (1 - degradation_rate) ** year_idx
        return ProjectionYear(
            pv_degradation_factor=pv_degradation_factor,
            pv_dc=inputs.dc_system_base * pv_degradation_factor,
            houseload=inputs.load_data,
            temperature_series=inputs.temperature_series,
        )

    # Every project year replays the start-year calendar (ADR 0002 A2), so the
    # tariff is resolved once, on the simulated index.
    tariff = (
        resolved.tariff.resolve(pd.DatetimeIndex(inputs.dc_system_base.index), resolved.timezone)
        if resolved.tariff
        else None
    )
    # The instructions follow the tariff's calendar, so they too are resolved
    # once and replayed every year.
    spec = resolved.smart_charging
    instructions = resolve_instructions(spec, tariff) if spec is not None and has_battery else None
    projection = run_projection(
        cfg,
        resolved,
        projection_years,
        year_inputs,
        has_battery=has_battery,
        execution_backend=execution_backend,
        observe_jit_per_year=True,
        tariff=tariff,
        instructions=instructions,
    )
    first_year_results_df = cast(pd.DataFrame, projection.first_year_results_df)
    current_soh = projection.carry.soh_pct
    degradation_state = projection.carry.degradation_state
    total_replacements = projection.total_replacements
    total_replacement_cost = projection.total_replacement_cost
    jit_cache_states = projection.jit_cache_states
    degradation_engine = str(cfg.get("degradation_engine", "native")).strip().lower()
    blast_model = cfg.get("blast_model")

    value = value_projection(cfg, resolved, projection)
    costs, cost_projection, lcoe, yearly_df = value.costs, value.cost_projection, value.lcoe, value.yearly_df

    replacement_events = [
        {"year": int(year), "count": int(count)}
        for year, count in zip(yearly_df["Year"], yearly_df["Replacements"], strict=True)
        if count
    ]
    degradation_summary = build_degradation_summary_from_state(
        engine=cast(DegradationEngineName, degradation_engine),
        model_key=str(blast_model) if blast_model is not None else cfg["calendar_model"],
        final_soh_pct=current_soh,
        replacement_events=replacement_events,
        state=degradation_state,
    )

    if execution_backend == "numba":
        execution["jit_cache"] = aggregate_jit_cache_states(jit_cache_states)

    smart_charging = None
    if instructions is not None and spec is not None and tariff is not None:
        first = first_year_results_df.iloc[0]
        carry = projection.carry
        smart_charging = {
            **smart_charging_provenance(spec, instructions, tariff),
            # The terminal convention is physical carry, so the state the
            # project starts and ends in is reported rather than assumed.
            "initial_stored_energy": stored_energy_by_origin(
                first["Battery_Energy_Beginning"],
                first["Battery_PV_Origin_Energy_Beginning"],
                first["Battery_Grid_Origin_Energy_Beginning"],
            ),
            "final_stored_energy": stored_energy_by_origin(
                carry.energy_wh or 0.0, carry.pv_origin_energy_wh or 0.0, carry.grid_origin_energy_wh or 0.0
            ),
        }

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
        load_profile_metadata=dict(
            inputs.load_data.attrs.get(
                LOAD_PROFILE_METADATA_KEY,
                {
                    "source": "runtime_dependency_or_unknown",
                    "note": "The injected load-profile provider did not expose source metadata.",
                },
            )
        ),
        degradation_summary=degradation_summary,
        execution=execution,
        tariff=tariff_provenance(tariff, calendar_year=int(cfg["start_date"][:4])) if tariff is not None else None,
        smart_charging=smart_charging,
    )
