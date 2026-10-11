"""Simulation orchestration for the public App facade."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import Any, cast

import pandas as pd

from breos._daily_persistence import DailyPersistenceController, daily_persistence_provenance
from breos.app_config import ResolvedAppConfig, SimulationPeriod
from breos.app_inputs import (
    AppRuntimeDependencies,
    prepare_simulation_inputs,
    prepare_simulation_inputs_cached,
    unknown_source_metadata,
)
from breos.battery import LEDGER_SCHEMA_VERSION
from breos.degradation.results import DegradationEngineName, build_degradation_summary_from_state
from breos.dispatch_instructions import DispatchInstructions
from breos.economics import TerminalHealthCredit, find_payback_year
from breos.execution import aggregate_jit_cache_states, backend_provenance, config_has_battery
from breos.load_profiles import LOAD_PROFILE_METADATA_KEY
from breos.projection import (
    BASELINE_NETWORK_CREDIT_COLUMNS,
    SYSTEM_NETWORK_CREDIT_COLUMNS,
    ProjectionRun,
    ProjectionValue,
    ProjectionYear,
    YearInstructions,
    effective_reference_escalation,
    price_reference_year_rows,
    price_tariff_year_rows_by_step,
    reprice_network_credit_year_rows,
    reprice_tariff_year_rows,
    run_projection,
    value_projection,
)
from breos.pv_modules import get_module
from breos.smart_charging import (
    PLANNER_MODES,
    FixedTargetDayController,
    resolve_instructions,
    smart_charging_provenance,
    stored_energy_by_origin,
)
from breos.solar import PVProductionBreakdown
from breos.tariffs import ResolvedTariff, reference_tariff_provenance, tariff_provenance
from breos.utils import get_hours_per_step
from breos.weather import WEATHER_METADATA_KEY


@dataclass(frozen=True)
class SimulationArtifacts:
    """Intermediate outputs needed to serialize App results."""

    yearly_df: pd.DataFrame
    first_year_results_df: pd.DataFrame
    # The lifetime economics: None for a [period] window, which runs once.
    cost_projection: pd.DataFrame | None
    costs: dict[str, float]
    payback_year: int | None
    lcoe: float | None
    current_soh: float
    total_replacements: int
    total_replacement_cost: float | None
    pv_loss_waterfall: dict[str, Any]
    weather_metadata: dict[str, Any]
    load_profile_metadata: dict[str, Any]
    degradation_summary: dict[str, Any]
    execution: dict[str, Any]
    # The resolved tariff's provenance; None on flat prices.
    tariff: dict[str, Any] | None = None
    # Smart-charging provenance and the project's first and last stored
    # energy by origin; None without configured grid-charging smart charging.
    smart_charging: dict[str, Any] | None = None
    # What revaluation re-prices from: the unpriced projection, and the
    # tariff and static instructions the run dispatched on (None for a
    # daily-persistence run, whose instructions are decided while it runs).
    projection: ProjectionRun | None = None
    resolved_tariff: ResolvedTariff | None = None
    instructions: DispatchInstructions | None = None
    # A [period] run's window record, and its avoided CO2 by pathway (kg)
    # with emissions on; None for a full-year run.
    period: dict[str, Any] | None = None
    period_co2: dict[str, float] | None = None
    # The [reference_tariff]'s provenance and resolution, which priced the
    # no-system household; None when it pays the system's own prices.
    reference_tariff: dict[str, Any] | None = None
    resolved_reference_tariff: ResolvedTariff | None = None
    terminal_health: TerminalHealthCredit | None = None
    # Each end-of-life crossing of the battery, in project order (ADR 0003 E11).
    end_of_life_events: tuple[dict[str, Any], ...] = ()


# The avoided-CO2 columns a result reports for its first year, or its window.
CO2_COLUMNS = ("CO2_Avoided_SelfConsumed_kg", "CO2_Avoided_Export_kg", "CO2_Avoided_Total_kg")


def _economics_fields(value: ProjectionValue, period: SimulationPeriod | None) -> dict[str, Any]:
    """The artifact fields a priced projection fills.

    A [period] window shorter than a year keeps its year row, priced at
    year-1 prices, and its avoided CO2, and leaves the lifetime economics
    None: a projection would treat the window as a whole project year.
    """
    if period is None:
        return {
            "yearly_df": value.yearly_df,
            "cost_projection": value.cost_projection,
            "costs": value.costs,
            "payback_year": find_payback_year(value.cost_projection),
            "lcoe": value.lcoe,
            "total_replacement_cost": value.total_replacement_cost,
            "period_co2": None,
            "terminal_health": value.terminal_health,
        }
    projection = value.cost_projection
    co2 = (
        {column: float(projection[column].iloc[0]) for column in CO2_COLUMNS}
        if all(column in projection.columns for column in CO2_COLUMNS)
        else None
    )
    return {
        "yearly_df": value.yearly_df,
        "cost_projection": None,
        "costs": value.costs,
        "payback_year": None,
        "lcoe": None,
        "total_replacement_cost": None,
        "period_co2": co2,
        "terminal_health": None,
    }


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
    resolved: ResolvedAppConfig, front_effective_dc_kwh: float, rear_gain_dc_kwh: float
) -> dict[str, Any]:
    """Build JSON-safe bifacial configuration and year-1 gain provenance."""
    cfg = resolved.cfg
    default_model = cfg["bifacial_model"]
    default_gcr = cfg["gcr"]
    default_height = cfg["pvrow_height"]
    default_pitch = cfg["pvrow_pitch"]
    # Without pv_arrays, the top-level settings describe the one array.
    arrays = resolved.pv_arrays or [
        {
            "modules": cfg["n_modules"],
            "module": resolved.pv_module_key,
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
        module_key = array["module"]
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
    pv_breakdown: PVProductionBreakdown, first_year_results_df: pd.DataFrame, resolved: ResolvedAppConfig
) -> dict[str, Any]:
    """Build a JSON-serializable year-1 PV loss waterfall, or the window's for a [period] run."""
    cfg = resolved.cfg
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
    dispatch: dict[str, Any] = {
        "curtailment_kwh": _rounded(curtailment),
        "battery_charge_loss_kwh": _rounded(e("Battery_Charge_Loss")),
        "battery_discharge_loss_kwh": _rounded(e("Battery_Discharge_Loss")),
        "battery_standby_loss_kwh": _rounded(e("Standby_Loss")),
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
        "basis": "year_1" if resolved.period is None else "period",
        "unit": "kWh",
        "flow_unit": "kWh per year" if resolved.period is None else "kWh over the period",
        "state_unit": "kWh at period boundary",
        "ledger_schema_version": LEDGER_SCHEMA_VERSION,
        "stages": stages,
        "bifacial": _bifacial_summary(resolved, front_effective_dc, rear_gain_dc),
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
    resolved: ResolvedAppConfig,
    deps: AppRuntimeDependencies,
    *,
    instructions: YearInstructions | None = None,
) -> SimulationArtifacts:
    """Run the weather/PV/load/battery/economics simulation pipeline.

    ``instructions`` replaces the dispatch instructions the ``[smart_charging]``
    table would give, for a validation tool that replays a schedule of its
    own (``tools/oracles/replay.py``). They must be resolved on the simulated
    index. They are one set every project year replays, one set per project
    year, or a year planner asked as each year begins (ADR 0002 A16); the
    projection's ``year_instructions`` records what each year dispatched on.
    The run then reports no smart-charging provenance, since the table did
    not produce them. App never passes them.
    """
    cfg = resolved.cfg
    # Resolve the backend before anything is fetched or computed. Input
    # preparation can hit the network for weather, so a missing optional
    # dependency should be reported now rather than after a download.
    execution_backend = cfg["execution_backend"]
    # Asked the way the dispatch asks it, so the run and its provenance agree
    # on what counts as PV-only.
    has_battery = config_has_battery(cfg)
    execution = backend_provenance(execution_backend, pv_only=not has_battery)
    spec = resolved.smart_charging
    planned = instructions is None and spec is not None and spec.mode in PLANNER_MODES and has_battery
    if planned and execution_backend == "python" and cfg["resolution"] == "15min":
        # Warned before any input is prepared: the planner solves a dynamic
        # program every simulated day, which the Python backend runs slowly.
        warnings.warn(
            "smart_charging mode = 'daily_persistence' re-plans every day, which is slow on the Python backend "
            "at 15-minute resolution; set execution_backend = 'numba' (the breos[fast] extra) for this mode.",
            UserWarning,
            stacklevel=3,
        )

    inputs = prepare_simulation_inputs_cached(cfg, resolved, deps, prepare=prepare_simulation_inputs)

    # A [period] window runs once. Replayed as project years, it would carry
    # degradation over copies of one window as if each were a year.
    period = resolved.period
    projection_years = cfg["projection_years"] if period is None else 1
    degradation_rate = cfg["pv_degradation_rate"]

    # A window bills the fixed charge on its civil days: a DST day has 23 or
    # 25 hours but is one day of the tariff.
    extra = {"Billed_Days": float(period.days)} if period is not None else {}

    def year_inputs(year_idx: int) -> ProjectionYear:
        pv_degradation_factor = (1 - degradation_rate) ** year_idx
        return ProjectionYear(
            pv_degradation_factor=pv_degradation_factor,
            pv_dc=inputs.dc_system_base * pv_degradation_factor,
            houseload=inputs.load_data,
            temperature_series=inputs.temperature_series,
            extra=extra,
        )

    # Every project year replays the start-year calendar (ADR 0002 A2), so the
    # tariff is resolved once, on the simulated index.
    tariff = (
        resolved.tariff.resolve(pd.DatetimeIndex(inputs.dc_system_base.index), resolved.timezone)
        if resolved.tariff
        else None
    )
    reference_tariff = (
        resolved.reference_tariff.resolve(pd.DatetimeIndex(inputs.dc_system_base.index), resolved.timezone)
        if resolved.reference_tariff
        else None
    )
    # The instructions follow the tariff's calendar, so they too are resolved
    # once and replayed every year.
    from_spec = instructions is None
    day_controller: FixedTargetDayController | DailyPersistenceController | None = None
    if planned:
        # daily_persistence decides each civil day while the run goes (ADR
        # 0002 A12); it has no static instructions to resolve.
        assert spec is not None and tariff is not None
        day_controller = DailyPersistenceController.for_run(
            spec, tariff, freq=cfg["resolution"], execution_backend=execution_backend
        )
    elif from_spec:
        instructions = resolve_instructions(spec, tariff) if spec is not None and has_battery else None
        # The configured table runs through the civil-day controller seam
        # (ADR 0002 A11); a replayed schedule stays on the static path.
        if instructions is not None:
            day_controller = FixedTargetDayController(instructions)
    projection = run_projection(
        cfg,
        resolved,
        projection_years,
        year_inputs,
        has_battery=has_battery,
        execution_backend=execution_backend,
        observe_jit_per_year=True,
        tariff=tariff,
        instructions=None if day_controller is not None else instructions,
        # Kept so App.revalue can re-price the tariff without re-simulating.
        record_period_energy=True,
        day_controller=day_controller,
        # A [period] window is one standalone span; project years replay one
        # calendar, so a civil day cut by a year's end continues next year.
        replay_seam=period is None,
        reference_tariff=reference_tariff,
        record_priced_flows=_price_blind(instructions, day_controller),
    )
    first_year_results_df = cast(pd.DataFrame, projection.first_year_results_df)
    current_soh = projection.carry.soh_pct
    degradation_state = projection.carry.degradation_state
    total_replacements = projection.total_replacements
    jit_cache_states = projection.jit_cache_states
    degradation_engine = cfg["degradation_engine"]
    blast_model = cfg["blast_model"]

    economics = _economics_fields(value_projection(cfg, resolved, projection), period)
    yearly_df = economics["yearly_df"]

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
    if from_spec and spec is not None and tariff is not None and (instructions is not None or planned):
        first = first_year_results_df.iloc[0]
        carry = projection.carry
        # The terminal convention is physical carry, so the state the
        # project starts and ends in is reported rather than assumed.
        stored = {
            "initial_stored_energy": stored_energy_by_origin(
                first["Battery_Energy_Beginning"],
                first["Battery_PV_Origin_Energy_Beginning"],
                first["Battery_Grid_Origin_Energy_Beginning"],
            ),
            "final_stored_energy": stored_energy_by_origin(
                carry.energy_wh or 0.0, carry.pv_origin_energy_wh or 0.0, carry.grid_origin_energy_wh or 0.0
            ),
        }
        if planned:
            executed = projection.controller_instructions
            assert executed is not None
            smart_charging = daily_persistence_provenance(spec, tariff, executed, **stored)
        else:
            assert isinstance(instructions, DispatchInstructions)
            smart_charging = {**smart_charging_provenance(spec, instructions, tariff), **stored}

    return SimulationArtifacts(
        **economics,
        first_year_results_df=first_year_results_df,
        current_soh=current_soh,
        total_replacements=total_replacements,
        end_of_life_events=projection.end_of_life_events,
        pv_loss_waterfall=_build_pv_loss_waterfall(inputs.pv_breakdown, first_year_results_df, resolved),
        weather_metadata=dict(inputs.weather.attrs.get(WEATHER_METADATA_KEY, unknown_source_metadata("weather"))),
        load_profile_metadata=dict(
            inputs.load_data.attrs.get(LOAD_PROFILE_METADATA_KEY, unknown_source_metadata("load-profile"))
        ),
        degradation_summary=degradation_summary,
        execution=execution,
        tariff=tariff_provenance(tariff, calendar_year=int(cfg["start_date"][:4])) if tariff is not None else None,
        smart_charging=smart_charging,
        projection=projection,
        resolved_tariff=tariff,
        # Only one static set can be re-resolved and compared on revaluation.
        instructions=instructions if isinstance(instructions, DispatchInstructions) else None,
        period=period.record() if period is not None else None,
        **_reference_fields(resolved, reference_tariff),
    )


def _price_blind(instructions: object, day_controller: object) -> bool:
    """Whether a run dispatches without reading a tariff, so any schedule can price its stored flows.

    Greedy dispatch, with or without a battery, never sees the tariff.
    Static instructions, per-year instructions, a year planner and a daily
    controller all follow the tariff's periods or prices, so such a run keeps
    no step flows and a new schedule simulates it again.
    """
    return instructions is None and day_controller is None


def _reference_fields(resolved: ResolvedAppConfig, reference: ResolvedTariff | None) -> dict[str, Any]:
    """The artifact fields of a run's no-system reference tariff."""
    if reference is None or resolved.reference_tariff is None:
        return {"reference_tariff": None, "resolved_reference_tariff": None}
    record = reference_tariff_provenance(
        reference,
        calendar_year=int(resolved.cfg["start_date"][:4]),
        import_price_escalation=effective_reference_escalation(resolved),
    )
    return {"reference_tariff": record, "resolved_reference_tariff": reference}


# The year-row columns a tariff fills; without one, economics prices the
# energy at the flat rates instead.
_TARIFF_MONEY_COLUMNS = (
    "Import_Cost",
    "Export_Revenue",
    "Baseline_Import_Cost",
    "Grid_Charge_Cost",
    "Fixed_Charge",
    *SYSTEM_NETWORK_CREDIT_COLUMNS,
    *BASELINE_NETWORK_CREDIT_COLUMNS,
)
# The year-row columns a reference tariff fills; without one, they follow the
# system's prices.
_REFERENCE_MONEY_COLUMNS = ("Baseline_Import_Cost", "Baseline_Fixed_Charge", *BASELINE_NETWORK_CREDIT_COLUMNS)


def _simulated_index(artifacts: SimulationArtifacts) -> pd.DatetimeIndex:
    """The calendar a finished run simulated, which a reference tariff is resolved on."""
    for resolved in (artifacts.resolved_tariff, artifacts.resolved_reference_tariff):
        if resolved is not None:
            return resolved.index
    return pd.DatetimeIndex(artifacts.first_year_results_df["Datetime"])


def _same_step_prices(tariff: ResolvedTariff, old: ResolvedTariff) -> bool:
    """Whether two tariffs on one calendar price every step's import and export alike."""
    return tariff.import_price_per_kwh == old.import_price_per_kwh and (
        tariff.export_price_per_kwh == old.export_price_per_kwh
    )


def revalue_app_simulation(
    resolved: ResolvedAppConfig,
    artifacts: SimulationArtifacts,
    deps: AppRuntimeDependencies,
) -> tuple[SimulationArtifacts, str]:
    """Value a finished run at ``resolved``'s prices; return the artifacts and how they were valued.

    ``resolved`` must differ from the run's configuration in economics keys only
    (App.revalue checks). The stored projection is re-priced (``"repriced"``)
    when the new prices cannot change the dispatch: flat prices, a tariff
    removed, or a tariff on the same schedule whose smart-charging
    instructions are unchanged. A tariff added, or a different schedule or
    calendar, is priced from the stored step flows (``"repriced_by_step"``)
    when the dispatch never read a tariff, and re-simulates the run
    (``"resimulated"``) when it did. Instructions that change re-simulate
    the run. Under ``daily_persistence`` so does any change to the per-step
    import or export prices, which its planner reads; a change to the fixed
    charge alone is re-priced. A reference tariff added, changed or removed
    is always re-priced: it prices only the household load, which no
    dispatch changes. So is an annual network credit, of either household:
    the dispatch never sees it.
    """
    cfg = resolved.cfg
    run, old_tariff = artifacts.projection, artifacts.resolved_tariff
    if run is None:
        raise ValueError("these artifacts carry no projection to re-price")
    tariff = (
        resolved.tariff.resolve(old_tariff.index if old_tariff else _simulated_index(artifacts), resolved.timezone)
        if resolved.tariff
        else None
    )
    # A new schedule, or a first one, gives the stored energy by period no
    # meaning; only a run whose dispatch never read a tariff can be priced
    # on it, step by step.
    new_schedule = tariff is not None and (old_tariff is None or tariff.schedule_hash != old_tariff.schedule_hash)
    if new_schedule and run.priced_flows is None:
        return run_app_simulation(resolved, deps), "resimulated"

    instructions = None
    yearly = run.yearly_df
    if new_schedule:
        assert tariff is not None and run.priced_flows is not None
        yearly = price_tariff_year_rows_by_step(yearly, run.priced_flows, tariff, cfg["resolution"])
    elif tariff is not None and old_tariff is not None:
        planner = resolved.smart_charging is not None and resolved.smart_charging.mode in PLANNER_MODES
        if planner and not _same_step_prices(tariff, old_tariff):
            # The planner chooses each day's target on these prices, so new
            # prices can move the dispatch even on an unchanged schedule.
            return run_app_simulation(resolved, deps), "resimulated"
        if artifacts.instructions is not None and resolved.smart_charging is not None:
            instructions = resolve_instructions(resolved.smart_charging, tariff)
            if instructions is None or instructions.instruction_hash() != artifacts.instructions.instruction_hash():
                return run_app_simulation(resolved, deps), "resimulated"
        if tariff.price_hash != old_tariff.price_hash:
            period_energy = cast(pd.DataFrame, run.period_energy)
            if _same_step_prices(tariff, old_tariff) and (
                tariff.prices.fixed_charge_per_day == old_tariff.prices.fixed_charge_per_day
            ):
                # Only the annual network credit changed: the energy money
                # keeps the floats the simulation summed per step.
                yearly = reprice_network_credit_year_rows(yearly, period_energy, tariff)
            else:
                yearly = reprice_tariff_year_rows(yearly, period_energy, tariff)
    elif old_tariff is not None:
        yearly = yearly.drop(columns=[column for column in _TARIFF_MONEY_COLUMNS if column in yearly.columns])

    reference = (
        resolved.reference_tariff.resolve(_simulated_index(artifacts), resolved.timezone)
        if resolved.reference_tariff is not None
        else None
    )
    if reference is not None:
        houseload = artifacts.first_year_results_df["Houseload"].to_numpy(dtype=float)
        yearly = price_reference_year_rows(yearly, houseload, reference, cfg["resolution"])
    elif artifacts.resolved_reference_tariff is not None and not new_schedule:
        # The reference is gone: the no-system household pays the system's
        # prices again, the tariff's by period or the flat costs.
        yearly = yearly.drop(columns=[column for column in _REFERENCE_MONEY_COLUMNS if column in yearly.columns])
        if tariff is not None:
            baseline = reprice_tariff_year_rows(yearly, cast(pd.DataFrame, run.period_energy), tariff)
            columns = ["Baseline_Import_Cost", *(c for c in BASELINE_NETWORK_CREDIT_COLUMNS if c in baseline)]
            yearly = yearly.assign(**{column: baseline[column] for column in columns})

    value = value_projection(cfg, resolved, replace(run, yearly_df=yearly))
    smart_charging = artifacts.smart_charging
    if smart_charging is not None and instructions is not None and tariff is not None:
        assert resolved.smart_charging is not None
        smart_charging = {**smart_charging, **smart_charging_provenance(resolved.smart_charging, instructions, tariff)}
    revalued = replace(
        artifacts,
        **_economics_fields(value, resolved.period),
        tariff=tariff_provenance(tariff, calendar_year=int(cfg["start_date"][:4])) if tariff is not None else None,
        smart_charging=smart_charging,
        resolved_tariff=tariff,
        **_reference_fields(resolved, reference),
    )
    return revalued, "repriced_by_step" if new_schedule else "repriced"
