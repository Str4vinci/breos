"""
Optimization module for PV system sizing and configuration.

This module provides the NSGA-II multi-objective design search
(:func:`optimize_system_multi_objective`) and the fixed-design projection
(:func:`evaluate_projected_design`) behind it.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from breos.app_inputs import resample_hourly_weather
from breos.battery import BatteryConfig
from breos.dispatch_instructions import DispatchInstructions
from breos.economics import (
    calculate_costs,
    cost_analysis_projection,
    cost_params_from_config,
    find_payback_year_interpolated,
    price_year_rows,
    projection_rates_record,
    replacement_event_cost,
)
from breos.emissions import EmissionsParams
from breos.execution import (
    DEFAULT_EXECUTION_BACKEND,
    limit_worker_threads,
    require_backend,
    validate_execution_backend,
)
from breos.inverter import inverter_ac_capacity_w as inverter_ac_capacity_w_for
from breos.optimization_config import (
    DEFAULT_TIMEZONE,
    adjusted_max_tilt_deg,
    resolve_optimization_config,
    resolve_run_settings,
)
from breos.projection import CarryState, ProjectionYear, project_years
from breos.pv.model_options import configured_pv_model_kwargs
from breos.result_schema import RESULT_SCHEMA_VERSION
from breos.smart_charging import resolve_instructions, smart_charging_provenance
from breos.solar import (
    PVModuleParams,
    calculate_pv_production_dc,
)
from breos.tariffs import ResolvedTariff, reference_tariff_provenance, result_currency, tariff_provenance
from breos.utils import package_version
from breos.weather import build_battery_temperature_series, resample_to_15min, weather_metadata


@dataclass
class OptimizationResult:
    """Result from an optimization run."""

    optimal_value: float
    objective_value: float
    iterations: int
    details: Dict[str, Any]


@dataclass
class ProjectedDesignResult:
    """Detailed result for one fixed design evaluated over a project horizon.

    ``metrics`` contains the projected headline values used by the optimizer.
    ``yearly`` is the simulated annual energy and degradation-state ledger, and
    ``financial`` is the corresponding discounted cost ledger.
    """

    metrics: Dict[str, Any]
    yearly: pd.DataFrame
    financial: pd.DataFrame
    provenance: Dict[str, Any] = field(default_factory=dict)


def _serial_elementwise_runner(func: Callable[[Any], Any], args: list[Any]) -> list[Any]:
    """Fallback pymoo elementwise runner for single-process evaluation."""
    return [func(arg) for arg in args]


def _resolve_max_tilt_deg(constraints: Dict[str, Any], latitude: float) -> float:
    """Resolve the optimization tilt upper bound from constraints."""
    value = constraints["max_tilt_deg"]
    if value == "adjust":
        return adjusted_max_tilt_deg(latitude, constraints["tilt_margin_deg"])
    return float(value)


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

# Constants for defaults (can be overridden by config)
DEFAULT_MODULE_AREA = 1.134 * 2.278


def _pv_params_from_config(params: Dict[str, Any]) -> PVModuleParams:
    """Build PVModuleParams from an inline config mapping."""
    return PVModuleParams(
        Mpp=params["Mpp"],
        Vmp=params["Vmp"],
        Imp=params["Imp"],
        Voc=params["Voc"],
        Isc=params["Isc"],
        T_Pmax_pct=params.get("T_Pmax_pct", -0.34),
        T_Voc_pct=params.get("T_Voc_pct", -0.26),
        T_Isc_pct=params.get("T_Isc_pct", 0.05),
        N_Cells=params.get("N_Cells", 144),
        celltype=params.get("celltype", "monoSi"),
    )


def _dimensions_from_section(section: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if section.get("dimensions"):
        return section["dimensions"]
    if "module_width_m" in section or "module_length_m" in section:
        return {
            "width": section.get("module_width_m"),
            "length": section.get("module_length_m"),
        }
    return None


def _module_area_from_dimensions(dimensions: Optional[Dict[str, Any]]) -> float:
    """Resolve module footprint from config dimensions, falling back only when absent."""
    if not dimensions:
        return DEFAULT_MODULE_AREA

    missing = {key for key in ("width", "length") if key not in dimensions or dimensions[key] is None}
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"PV module dimensions missing required key(s): {missing_list}")

    width = float(dimensions["width"])
    length = float(dimensions["length"])
    area = width * length
    if area <= 0.0:
        raise ValueError(f"PV module dimensions must define a positive area, got width={width}, length={length}")
    return area


def _resolve_pv_module_and_area(config: Dict[str, Any]) -> Tuple[PVModuleParams, float]:
    """Resolve electrical module parameters and physical module area from a resolved config.

    ``pv.params`` gives the module inline; otherwise ``pv.module`` names a
    catalog module, which the resolver defaults as the App does.
    """
    pv_cfg = config["pv"]
    if pv_cfg.get("params"):
        pv_params = _pv_params_from_config(pv_cfg["params"])
    else:
        from breos.pv_modules import get_module

        pv_params = get_module(pv_cfg["module"])
    return pv_params, _module_area_from_dimensions(_dimensions_from_section(pv_cfg))


def _resolve_horizon_and_pv_degradation(config: Dict[str, Any]) -> Tuple[int, float]:
    """Return the scoring horizon (years) and annual PV degradation rate.

    Projected scoring and the fixed-design evaluator read these from here, so
    both rest on the same assumptions: ``simulation.years_projection`` and
    ``pv.degradation_rate``, falling back to ``financials.project_lifespan``
    and ``financials.pv_degradation_rate``.
    """
    return int(config["simulation"]["years_projection"]), float(config["pv"]["degradation_rate"])


def _resolve_degradation_engine_spec(batt_spec: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """Validate ``battery.degradation_engine`` and ``battery.blast_model`` once.

    The same rules as the App's config validation, applied when the problem is
    built, so an invalid BLAST setting fails on either objective basis before
    any candidate is scored.
    """
    engine = str(batt_spec.get("degradation_engine", "native")).strip().lower()
    blast_model = batt_spec.get("blast_model")
    if engine not in ("native", "blast"):
        raise ValueError("battery.degradation_engine must be one of: native, blast")
    if engine == "blast":
        from breos.degradation.profiles import ENABLED_BLAST_MODEL_KEYS

        if blast_model not in ENABLED_BLAST_MODEL_KEYS:
            available = ", ".join(ENABLED_BLAST_MODEL_KEYS)
            raise ValueError(f"Unknown battery.blast_model {blast_model!r}. Available: {available}")
    elif blast_model is not None:
        raise ValueError("battery.blast_model requires battery.degradation_engine = 'blast'")
    return engine, blast_model


def _validated_dc_output_scale(config: Dict[str, Any]) -> float:
    """Read and check the DC-side yield correction from a study config.

    A DC-side factor is applied to the raw array output before dispatch, so
    clipping, charging and the part-load ratio all respond to it. That makes
    it the correct knob for a model that under-predicts measured yield, and
    it is deliberately not bounded above.
    """
    scale = float(config["dc_output_scale"])
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("dc_output_scale must be finite and greater than 0")
    return scale


def _validated_ac_output_scale(config: Dict[str, Any]) -> float:
    """Read and check the AC-side derate from a study config.

    The factor derates AC delivery from inside dispatch, so discharge
    decisions respond to it. Unlike the DC-side factor it lands after the
    inverter nameplate limit, so it is bounded to ``(0, 1]``: above 1 the
    inverter would deliver more than its nameplate and more AC than the DC
    entering it, and the reported inverter loss would pin at zero. Correct an
    under-predicting model with ``dc_output_scale`` instead.

    Rejecting here means an out-of-range study config fails before any
    evaluation starts, rather than being clamped inside the inverter helpers.
    """
    scale = float(config["ac_output_scale"])
    if not np.isfinite(scale) or not 0.0 < scale <= 1.0:
        raise ValueError(
            "ac_output_scale must be finite, greater than 0 and at most 1; it is applied after "
            "the inverter nameplate limit. Use dc_output_scale to correct an under-predicting "
            "model on the DC side"
        )
    return scale


# Battery-section keys forwarded to BatteryConfig under the same name.
_BATTERY_SPEC_KEYS = (
    "min_soc",
    "max_soc",
    "charge_efficiency",
    "discharge_efficiency",
    "standby_loss_wh",
    "eol_percentage",
    "max_charge_power_w",
    "max_discharge_power_w",
    "power_limit_c_rate",
    "calendar_model",
    "enable_resistance_fade",
    "allow_terminal_replacement",
)


def _build_battery_config_from_spec(
    batt_spec: Dict[str, Any],
    nominal_energy_wh: float,
    inverter_efficiency: float = 0.96,
    initial_soh: float = 100.0,
    enable_replacement: bool = False,
    inverter_ac_capacity_w: Optional[float] = None,
    ac_output_scale: float = 1.0,
) -> BatteryConfig:
    """Build a BatteryConfig for optimization paths without dropping supported settings.

    Only the settings the spec names are forwarded. Everything it leaves out
    takes the :class:`BatteryConfig` default, which is the same default the
    App resolves, so the two entry points evaluate the same battery when a
    caller omits a setting.
    """
    configured = {key: batt_spec[key] for key in _BATTERY_SPEC_KEYS if key in batt_spec}
    return BatteryConfig(
        nominal_energy_wh=nominal_energy_wh,
        initial_soh=initial_soh,
        inverter_efficiency=inverter_efficiency,
        inverter_ac_capacity_w=inverter_ac_capacity_w,
        enable_replacement=enable_replacement,
        ac_output_scale=ac_output_scale,
        **configured,
    )


def _battery_replacement_treatment(battery: Mapping[str, Any]) -> Dict[str, Any]:
    """How projected scoring treats battery replacement, for the provenance.

    ``allow_terminal_replacement`` is the configured policy for the final
    period; the earlier years' internal permission is not the user's setting.
    """
    return {
        "method": "simulated_yearly_state_propagation",
        "description": "Projected scoring simulates every year and records actual replacement events.",
        "higher_fidelity_basis": "App multiyear SOH propagation",
        "allow_terminal_replacement": bool(battery.get("allow_terminal_replacement", True)),
        "terminal_period": (
            "The final degradation period of the last project year: the elapsed one-day window of simulation "
            "steps that ends on the horizon's last step, whole or partial. It is always aged and recorded; "
            "allow_terminal_replacement = false skips only its end-of-life replacement. Every earlier period, "
            "including the close of each earlier project year, replaces as usual."
        ),
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return a finite scalar ratio, or zero when the denominator is zero."""
    denominator = float(denominator)
    if abs(denominator) < 1e-12:
        return 0.0
    return float(numerator) / denominator


def _summarize_projected_lifetime_metrics(yearly_summary_df: pd.DataFrame) -> Dict[str, float]:
    """Summarize lifetime metrics from actual simulated yearly values."""
    if yearly_summary_df.empty:
        raise ValueError("yearly_summary_df must contain at least one year")

    load = yearly_summary_df["Load_kWh"].astype(float)
    production = yearly_summary_df["PV_Production_kWh"].astype(float)
    imports = yearly_summary_df["Import_kWh"].astype(float)
    annual_gi = yearly_summary_df["Grid_Independence_%"].astype(float)
    annual_zeb = np.divide(
        production.to_numpy(dtype=float),
        load.to_numpy(dtype=float),
        out=np.zeros(len(yearly_summary_df), dtype=float),
        where=load.to_numpy(dtype=float) > 0.0,
    )

    return {
        "Projected_Grid_Independence_%": 100.0 * (1.0 - _safe_ratio(imports.sum(), load.sum())),
        "Projected_Grid_Independence_Year1_%": float(annual_gi.iloc[0]),
        "Projected_Grid_Independence_FinalYear_%": float(annual_gi.iloc[-1]),
        "Projected_Grid_Independence_Mean_%": float(annual_gi.mean()),
        "Projected_Grid_Independence_Min_%": float(annual_gi.min()),
        "Projected_ZEB_Ratio": _safe_ratio(production.sum(), load.sum()),
        "Projected_ZEB_Ratio_Year1": float(annual_zeb[0]),
        "Projected_ZEB_Ratio_FinalYear": float(annual_zeb[-1]),
        "Projected_ZEB_Ratio_Mean": float(np.mean(annual_zeb)),
        "Projected_ZEB_Ratio_Min": float(np.min(annual_zeb)),
    }


def _projection_rates(fin_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """The ``financials`` rates as ``cost_analysis_projection`` arguments (ADR 0003 E2).

    ``fin_cfg`` is the ``financials`` table of :func:`resolve_optimization_config`,
    which fills ``inflation_rate``, ``sell_price_inflation`` and
    ``discount_rate``; a raw table without them raises ``KeyError``.
    The escalators stay None when unset, so they inherit ``inflation_rate``.
    Rates at or below -1 raise, as the App's do, and the learning rate must
    be in [0, 1).
    """

    def optional(key: str) -> Optional[float]:
        value = fin_cfg.get(key)
        return None if value is None else float(value)

    rates = {
        "inflation_rate": float(fin_cfg["inflation_rate"]),
        "sell_price_inflation": float(fin_cfg["sell_price_inflation"]),
        "import_price_escalation": optional("import_price_escalation"),
        "om_escalation": optional("om_escalation"),
        "replacement_cost_learning": optional("replacement_cost_learning") or 0.0,
        "discount_rate": float(fin_cfg["discount_rate"]),
    }
    for key in ("inflation_rate", "sell_price_inflation", "import_price_escalation", "om_escalation", "discount_rate"):
        if rates[key] is not None and not rates[key] > -1:
            raise ValueError(f"financials.{key} must be greater than -1")
    if not 0 <= rates["replacement_cost_learning"] < 1:
        raise ValueError("financials.replacement_cost_learning must be at least 0 and below 1")
    return rates


def _evaluate_projected_design_metrics(
    *,
    base_dc_power: Union[pd.Series, Sequence[pd.Series]],
    houseload: pd.DataFrame,
    temperature_series: pd.Series,
    pv_params: PVModuleParams,
    batt_spec: Dict[str, Any],
    costs_cfg: Dict[str, Any],
    fin_cfg: Dict[str, Any],
    freq: str,
    years_projection: int,
    degradation_rate: float,
    n_modules: int,
    battery_kwh: float,
    inverter_efficiency: float,
    inverter_ac_capacity_w: Optional[float],
    emissions_params: Optional[EmissionsParams] = None,
    return_tables: bool = False,
    execution_backend: str = DEFAULT_EXECUTION_BACKEND,
    ac_output_scale: float = 1.0,
    tariff: ResolvedTariff | None = None,
    instructions: DispatchInstructions | None = None,
    reference_tariff: ResolvedTariff | None = None,
    reference_import_price_escalation: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluate one design over the projected horizon using production engines.

    ``base_dc_power`` is normally one weather year, repeated for every
    projected year and scaled by PV degradation. Passing a sequence of series
    instead runs a real weather sequence, one entry per projected year, so a
    study can keep observed inter-annual variability rather than repeating a
    single year. The sequence must have exactly ``years_projection`` entries.
    A one-year sequence is equivalent to passing that series directly.

    A ``reference_tariff``, resolved on the tariff's calendar, prices the
    no-system household, so ``Projected_NPV`` is the saving against it. Its
    cost escalates at ``reference_import_price_escalation``; None uses the
    ``financials`` import escalation.
    """
    if years_projection < 1:
        raise ValueError("projected optimization requires at least one project year")
    if isinstance(base_dc_power, pd.Series):
        dc_by_year: Sequence[pd.Series] = [base_dc_power] * years_projection
    else:
        dc_by_year = list(base_dc_power)
        if len(dc_by_year) != years_projection:
            raise ValueError(f"projected weather sequence has {len(dc_by_year)} years, expected {years_projection}")
        if not all(isinstance(series, pd.Series) for series in dc_by_year):
            raise ValueError("every projected weather-sequence entry must be a pandas Series")
    if not 0.0 <= float(degradation_rate) < 1.0:
        raise ValueError("projected PV degradation rate must be between 0 and 1")

    cost_params = cost_params_from_config(costs_cfg, fin_cfg)
    costs = calculate_costs(
        n_modules=n_modules,
        module_power_w=pv_params.Mpp,
        battery_capacity_wh=battery_kwh * 1000.0,
        cost_params=cost_params,
        # The App's price, unless battery.replacement_cost sets one (ADR 0003 E4).
        replacement_cost_each=replacement_event_cost(
            battery_kwh, cost_params.battery_cost_per_kwh, batt_spec.get("replacement_cost")
        ),
    )
    has_battery = battery_kwh > 0.0
    degradation_engine, blast_model = _resolve_degradation_engine_spec(batt_spec)
    if not has_battery:
        # Without a battery there is nothing to age, and BLAST needs a pack.
        degradation_engine, blast_model = "native", None

    def battery_config(soh_pct: float) -> BatteryConfig:
        return _build_battery_config_from_spec(
            batt_spec,
            nominal_energy_wh=battery_kwh * 1000.0,
            inverter_efficiency=inverter_efficiency,
            initial_soh=soh_pct,
            enable_replacement=bool(batt_spec.get("enable_replacement", True)) and has_battery,
            inverter_ac_capacity_w=inverter_ac_capacity_w,
            ac_output_scale=ac_output_scale,
        )

    def year_inputs(year_idx: int) -> ProjectionYear:
        degradation_factor = (1.0 - float(degradation_rate)) ** year_idx
        return ProjectionYear(
            pv_degradation_factor=degradation_factor,
            pv_dc=dc_by_year[year_idx] * degradation_factor,
            houseload=houseload,
            temperature_series=temperature_series,
        )

    projection = project_years(
        years_projection,
        year_inputs,
        battery_config=battery_config,
        freq=freq,
        has_battery=has_battery,
        execution_backend=execution_backend,
        degradation_engine=degradation_engine,
        blast_model=blast_model,
        initial_carry=CarryState(soh_pct=float(batt_spec.get("initial_soh", 100.0)) if has_battery else 100.0),
        tariff=tariff,
        instructions=instructions if has_battery else None,
        reference_tariff=reference_tariff,
    )
    # Priced here, as App and Monte Carlo price theirs (ADR 0003 E4, E7).
    yearly_summary_df = price_year_rows(projection.yearly_df, costs)
    total_replacements = projection.total_replacements
    current_soh = float(projection.carry.soh_pct)
    cost_projection = cost_analysis_projection(
        yearly_summary_df=yearly_summary_df,
        costs=costs,
        num_years=years_projection,
        **_projection_rates(fin_cfg),
        emissions_params=emissions_params,
        currency=result_currency(tariff),
        baseline_import_price_escalation=reference_import_price_escalation,
    )
    payback_year = cost_projection.attrs.get("payback_year")
    payback_interpolated = find_payback_year_interpolated(cost_projection)
    metrics: Dict[str, Any] = {
        **_summarize_projected_lifetime_metrics(yearly_summary_df),
        "Projected_NPV": float(cost_projection["Savings_Cumulative_NPV"].iloc[-1]),
        "Projected_Payback_Year": float(payback_year) if payback_year is not None else np.nan,
        "Projected_Payback_Year_Interpolated": payback_interpolated if payback_interpolated is not None else np.nan,
        "Projected_Initial_Cost": float(costs["total_initial_cost"]),
        "Projected_Replacement_Cost_T0_Prices": float(cost_projection.attrs["total_replacement_cost"]),
        "Projected_Total_Replacements": int(total_replacements),
        "Projected_Final_SOH_%": float(current_soh),
        "Projected_PV_Production_Year1_kWh": float(yearly_summary_df["PV_Production_kWh"].iloc[0]),
        "Projected_PV_Production_FinalYear_kWh": float(yearly_summary_df["PV_Production_kWh"].iloc[-1]),
        "Projected_PV_DC_Year1_kWh": float(yearly_summary_df["PV_DC_Generation_kWh"].iloc[0]),
        "Projected_PV_DC_FinalYear_kWh": float(yearly_summary_df["PV_DC_Generation_kWh"].iloc[-1]),
        "Projected_PV_DC_Curtailed_Year1_kWh": float(yearly_summary_df["Curtailment_DC_kWh"].iloc[0]),
        "Projected_Inverter_Loss_Year1_kWh": float(yearly_summary_df["Inverter_Loss_kWh"].iloc[0]),
        "Projected_LCOE_per_kWh": float(cost_projection.attrs["lcoe_per_kwh"]),
    }
    if "CO2_Avoided_Total_Cumulative_kg" in cost_projection:
        metrics.update(
            {
                "Projected_CO2_Avoided_Total_kg": float(cost_projection["CO2_Avoided_Total_Cumulative_kg"].iloc[-1]),
                "Projected_CO2_Avoided_SelfConsumed_kg": float(
                    cost_projection["CO2_Avoided_SelfConsumed_Cumulative_kg"].iloc[-1]
                ),
            }
        )
    if return_tables:
        metrics["_yearly_summary_df"] = yearly_summary_df
        metrics["_cost_projection_df"] = cost_projection
    return metrics


@dataclass(frozen=True)
class _OptimizationTariff:
    """A search's or design's resolved tariff and smart-charging instructions, shared by every year."""

    tariff: ResolvedTariff | None = None
    instructions: DispatchInstructions | None = None
    smart_charging: Dict[str, Any] | None = None
    # The no-system reference tariff and its configured escalation (None:
    # the financials import escalation), and its provenance record.
    reference_tariff: ResolvedTariff | None = None
    reference_import_price_escalation: float | None = None
    reference_record: Dict[str, Any] | None = None

    def provenance(self) -> Dict[str, Any]:
        # Every money column is in this currency; BREOS does not convert.
        record: Dict[str, Any] = {
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "breos_version": package_version(),
            "currency": result_currency(self.tariff),
        }
        if self.tariff is not None:
            year = self.tariff.index.tz_convert(self.tariff.timezone)[0].year
            record["tariff"] = tariff_provenance(self.tariff, calendar_year=year)
        if self.reference_record is not None:
            record["reference_tariff"] = dict(self.reference_record)
        if self.smart_charging is not None:
            record["smart_charging"] = dict(self.smart_charging)
        return record


def _resolve_optimization_tariff(
    config: dict[str, Any], index: pd.DatetimeIndex, battery_kwh: float, battery_key: str = "battery_kwh"
) -> _OptimizationTariff:
    """Adapt the optimizer's config to the shared tariff and smart-charging validation.

    ``battery_kwh`` is the largest battery the entry point can install: the
    design's for a fixed design, ``constraints.max_battery_kwh`` for a search;
    ``battery_key`` names which, for the error when it is zero.
    The instructions are resolved once, on the tariff's calendar, and replayed
    every project year, as App does. A candidate without a battery ignores
    them. A ``[reference_tariff]`` is resolved on the same calendar and
    prices the no-system household of every candidate.
    """
    from breos.app_config import resolve_reference_tariff_spec, resolve_smart_charging_spec, resolve_tariff_spec

    timezone = config["location"]["timezone"]
    resolution = config["simulation"]["resolution"]
    spec = resolve_tariff_spec(
        {"tariff": config["tariff"], "costs": config["costs"], "resolution": resolution},
        timezone,
    )
    reference_spec = resolve_reference_tariff_spec(
        {"reference_tariff": config.get("reference_tariff"), "resolution": resolution}, timezone, spec
    )
    smart_charging = resolve_smart_charging_spec(
        {"smart_charging": config["smart_charging"], "battery_kwh": battery_kwh}, spec, battery_key
    )
    tariff = spec.resolve(index, timezone) if spec is not None else None
    reference: Dict[str, Any] = {}
    if reference_spec is not None:
        reference_tariff = reference_spec.resolve(index, timezone)
        configured = reference_spec.import_price_escalation
        effective = (
            configured
            if configured is not None
            else projection_rates_record(_projection_rates(config["financials"]))["import_price_escalation"]
        )
        reference = {
            "reference_tariff": reference_tariff,
            "reference_import_price_escalation": configured,
            "reference_record": reference_tariff_provenance(
                reference_tariff,
                calendar_year=reference_tariff.index.tz_convert(reference_tariff.timezone)[0].year,
                import_price_escalation=effective,
            ),
        }
    instructions = resolve_instructions(smart_charging, tariff) if smart_charging is not None else None
    if instructions is None or tariff is None or smart_charging is None:
        return _OptimizationTariff(tariff=tariff, **reference)
    return _OptimizationTariff(
        tariff=tariff,
        instructions=instructions,
        smart_charging=smart_charging_provenance(smart_charging, instructions, tariff),
        **reference,
    )


def _site_location(location: dict[str, Any]) -> Any:
    """Build the pvlib site the optimizer simulates, the way App does.

    Without an ``altitude`` key pvlib looks the elevation up from the
    coordinates, exactly as App's ``Location(lat, lon, tz=...)`` does. The
    elevation sets the air pressure behind the refraction correction of the
    solar position, so modelling every site at sea level (the old default of
    0 m) gave the optimizer slightly different PV output from App for the same
    design.
    """
    from pvlib.location import Location

    location = {"timezone": DEFAULT_TIMEZONE, "altitude": None, "name": "", **location}
    altitude = location["altitude"]
    return Location(
        float(location["latitude"]),
        float(location["longitude"]),
        tz=location["timezone"],
        altitude=None if altitude is None else float(altitude),
        name=str(location["name"]),
    )


def _prepare_study_weather(weather: pd.DataFrame, config: dict[str, Any], loc_obj: Any) -> pd.DataFrame:
    """Resample hourly weather for a 15-minute study with App's helper and policy.

    The site's altitude is passed through, so a configured ``location.altitude``
    sets the clear-sky model as it sets the PV model; without one it is the
    same pvlib lookup App's resampling makes. Weather already at the study
    resolution is returned unchanged.
    """
    return resample_hourly_weather(
        weather,
        str(config["simulation"]["resolution"]),
        latitude=loc_obj.latitude,
        longitude=loc_obj.longitude,
        resample=resample_to_15min,
        irradiance_resampling=config["simulation"]["irradiance_resampling"],
        altitude=loc_obj.altitude,
    )


def evaluate_projected_design(
    tmy_data: pd.DataFrame,
    houseload: pd.DataFrame,
    config: Dict[str, Any],
    *,
    n_modules: int,
    battery_kwh: float,
    tilt: float,
    azimuth: float,
    execution_backend: str = DEFAULT_EXECUTION_BACKEND,
    weather_by_year: Optional[Sequence[pd.DataFrame]] = None,
) -> ProjectedDesignResult:
    """Evaluate one fixed PV-battery design over the projected horizon.

    This is the detailed fixed-design counterpart to projected NSGA-II
    scoring. It uses the same PV, battery, replacement, degradation, and
    economics components as :func:`optimize_system_multi_objective` and
    returns the annual source tables needed for analysis or plotting.

    Args:
        tmy_data: One-year weather DataFrame.
        houseload: One-year load profile.
        config: Nested projected-optimization configuration.
        n_modules: Installed PV module count.
        battery_kwh: Installed nominal battery capacity in kWh.
        tilt: PV surface tilt in degrees.
        azimuth: PV surface azimuth in degrees.
        weather_by_year: Optional real weather sequence, one frame per
            projected year, replacing the repeated ``tmy_data`` year. Each
            frame is run through the same PV model, and PV degradation still
            applies by project year. ``tmy_data`` is still used for the
            battery temperature series and must remain a representative year;
            its temperatures are restamped onto the calendar year of the
            first sequence frame.
            The sequence length must equal the projected horizon.

    Returns:
        Projected metrics, yearly simulation ledger, and financial ledger.
    """
    # Zero modules is a valid grid corner, not an error: the exhaustive
    # lattice enumerates it so the battery-only slice is measured rather than
    # assumed dominated. It produces no PV, so its inverter rating is zero and
    # its LCOE is undefined.
    if int(n_modules) < 0:
        raise ValueError("n_modules must be non-negative")
    if float(battery_kwh) < 0.0:
        raise ValueError("battery_kwh must be non-negative")

    # Before the PV production model runs, not after it. Computing a year of
    # irradiance and then failing on a missing import wastes the expensive part
    # and reports the cheap problem late.
    require_backend(execution_backend)

    config = resolve_optimization_config(config)
    frames = list(weather_by_year) if weather_by_year is not None else None
    loc_obj = _site_location(config["location"])
    freq = str(config["simulation"]["resolution"])
    tmy_data = _prepare_study_weather(tmy_data, config, loc_obj)
    if frames is not None:
        frames = [_prepare_study_weather(frame, config, loc_obj) for frame in frames]
    tariff_index = frames[0].index if frames else tmy_data.index
    pricing = _resolve_optimization_tariff(config, tariff_index, float(battery_kwh))
    financials = config["financials"]
    emissions_config = config["emissions"]
    battery = config["battery"]
    years_projection, degradation_rate = _resolve_horizon_and_pv_degradation(config)
    pv_params, _module_area = _resolve_pv_module_and_area(config)

    # A DC-side yield correction: the array itself produces this much less,
    # or more. Unlike ac_output_scale it is applied before dispatch, so
    # charging, clipping and the part-load ratio all respond to it, which is
    # why it is the correction to reach for when the model under-predicts.
    dc_output_scale = _validated_dc_output_scale(config)

    def _dc_for(weather_frame: pd.DataFrame) -> pd.Series:
        series = calculate_pv_production_dc(
            weather_data=weather_frame,
            location=loc_obj,
            tilt=float(tilt),
            surface_azimuth=float(azimuth),
            n_modules=int(n_modules),
            pv_params=pv_params,
            freq=freq,
            **configured_pv_model_kwargs(config),
        )
        return series if dc_output_scale == 1.0 else series * dc_output_scale

    if frames is None:
        base_dc_power: Union[pd.Series, Sequence[pd.Series]] = _dc_for(tmy_data)
        dc_index = base_dc_power.index
    else:
        if len(frames) != years_projection:
            raise ValueError(f"weather_by_year has {len(frames)} years, expected {years_projection}")
        # Every year must land on the same intra-year index, because the load
        # profile and the battery temperature series are aligned to it once.
        reference = _dc_for(frames[0])
        series = [reference]
        for frame in frames[1:]:
            year_dc = _dc_for(frame)
            if len(year_dc) != len(reference):
                raise ValueError(
                    "every weather_by_year frame must produce the same number of "
                    f"timesteps; got {len(year_dc)} against {len(reference)}"
                )
            year_dc.index = reference.index
            series.append(year_dc)
        base_dc_power = series
        dc_index = reference.index
    # A weather sequence keeps the representative year's temperatures, so they
    # are restamped onto the sequence's calendar year rather than reindexed
    # across years, which found no match and used to fall back to 25 C.
    temperature_series = build_battery_temperature_series(
        battery["temperature"],
        dc_index,
        weather_df=tmy_data,
        indoor_model=battery["indoor_model"],
        align_weather_year=weather_by_year is not None,
    )
    dc_ac_ratio = cost_params_from_config(config["costs"], financials).dc_ac_ratio
    inverter_ac_capacity_w = inverter_ac_capacity_w_for(int(n_modules) * pv_params.Mpp, dc_ac_ratio)
    raw_metrics = _evaluate_projected_design_metrics(
        execution_backend=execution_backend,
        base_dc_power=base_dc_power,
        houseload=houseload,
        temperature_series=temperature_series,
        pv_params=pv_params,
        batt_spec=battery,
        costs_cfg=config["costs"],
        fin_cfg=financials,
        freq=freq,
        years_projection=years_projection,
        degradation_rate=degradation_rate,
        n_modules=int(n_modules),
        battery_kwh=float(battery_kwh),
        inverter_efficiency=float(config["inverter_efficiency"]),
        inverter_ac_capacity_w=inverter_ac_capacity_w,
        emissions_params=EmissionsParams(**emissions_config) if emissions_config else None,
        return_tables=True,
        ac_output_scale=_validated_ac_output_scale(config),
        tariff=pricing.tariff,
        instructions=pricing.instructions,
        reference_tariff=pricing.reference_tariff,
        reference_import_price_escalation=pricing.reference_import_price_escalation,
    )
    yearly = raw_metrics.pop("_yearly_summary_df")
    financial = raw_metrics.pop("_cost_projection_df")
    metrics = {
        "Modules": int(n_modules),
        "Battery_kWh": float(battery_kwh),
        "Tilt": float(tilt),
        "Azimuth": float(azimuth),
        **raw_metrics,
    }
    provenance = {
        **pricing.provenance(),
        "economics": projection_rates_record(_projection_rates(financials)),
        "simulation": dict(config["simulation"]),
        "weather": weather_metadata(tmy_data),
        **({"weather_by_year": [weather_metadata(frame) for frame in frames]} if frames is not None else {}),
        "battery_replacement_treatment": _battery_replacement_treatment(battery),
    }
    return ProjectedDesignResult(metrics=metrics, yearly=yearly, financial=financial, provenance=provenance)


# ==========================================
# 3. PYMOO OPTIMIZATION CLASSES
# ==========================================


def _snap_to_grid_within_bounds(values: np.ndarray, step: float, lower: float, upper: float) -> np.ndarray:
    """Round to the nearest multiple of ``step`` that lies inside the bounds.

    Rounding alone can leave the bounds: 62.9 degrees rounds to 65 under a
    63 degree maximum. A value that rounds past a bound takes the grid point
    just inside it instead, 60 in that example.
    """
    lowest = np.ceil(lower / step) * step
    highest = np.floor(upper / step) * step
    return np.clip(np.round(values / step) * step, lowest, highest)


# pymoo is optional, so these classes exist only when it is installed; without
# it optimize_system_multi_objective raises ImportError. They subclass pymoo
# types at module level so an optimizer result that stores them pickles, which
# means ``import breos`` imports pymoo whenever it is installed.
_PYMOO_IMPORT_ERROR: Optional[ImportError] = None
try:
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.core.repair import Repair
    from pymoo.core.termination import Termination

    class _MinGenerationTermination(Termination):
        """Hold another termination's progress at zero until ``minimum`` generations.

        Defined at module level so an optimizer result that stores it pickles.
        """

        def __init__(self, termination, minimum: int) -> None:
            super().__init__()
            self.termination = termination
            self.minimum = max(1, int(minimum))

        def _update(self, algorithm):
            progress = self.termination.update(algorithm)
            return 0.0 if algorithm.n_gen < self.minimum else progress

    class DiscreteGridRepair(Repair):
        def _do(self, problem, X, **kwargs):
            # pymoo's Repair.do passes the design matrix and writes the result
            # back onto the population. Modules and battery kWh snap to
            # integers, tilt and azimuth to 5 degrees. Every column stays
            # inside the problem bounds.
            steps = (1.0, 1.0, 5.0, 5.0)
            for col in range(X.shape[1]):
                X[:, col] = _snap_to_grid_within_bounds(X[:, col], steps[col], problem.xl[col], problem.xu[col])
            return X

    class SolarDesignProblem(ElementwiseProblem):
        def __init__(
            self,
            tmy_data: pd.DataFrame,
            houseload: pd.DataFrame,
            config: Dict[str, Any],
            *,
            elementwise_runner=None,
            execution_backend: str = DEFAULT_EXECUTION_BACKEND,
        ):
            # Passed explicitly by the caller, never read out of ``config``.
            # Candidate scoring is the hottest loop in the package, which makes
            # it exactly the place where a silently-inherited backend would be
            # hardest to notice and hardest to attribute afterwards.
            self.execution_backend = validate_execution_backend(execution_backend)
            self.houseload = houseload
            # Checked and defaulted once, before any model preparation: an
            # unknown key raises, and every default is in the resolved config.
            config = resolve_optimization_config(config)
            self.config = config
            self.location = config["location"]
            # config['location'] is a plain dict; the pvlib Location that
            # calculate_pv_production_dc needs is constructed once here.
            self.loc_obj = _site_location(self.location)
            self.tmy_data = _prepare_study_weather(tmy_data, config, self.loc_obj)

            self.constraints = config["constraints"]
            # One schedule/price resolution per search, shared by every
            # candidate and project year. Validate before model preparation.
            self.pricing = _resolve_optimization_tariff(
                config,
                self.tmy_data.index,
                float(self.constraints["max_battery_kwh"]),
                battery_key="constraints.max_battery_kwh",
            )
            self.tariff = self.pricing.tariff

            self.budget_limit = self.constraints["budget"]
            self.area_limit = self.constraints["max_area_m2"]
            self.max_battery_kwh = self.constraints["max_battery_kwh"]
            self.max_modules = self.constraints["max_modules"]
            self.min_tilt_deg = float(self.constraints["min_tilt_deg"])
            self.max_tilt_deg = _resolve_max_tilt_deg(self.constraints, self.location["latitude"])
            self.enforce_zeb = bool(self.constraints["enforce_zeb"])
            self.freq = config["simulation"]["resolution"]
            # Resolved once: candidate scoring is the hottest loop here.
            self.model_options = configured_pv_model_kwargs(config)
            self.opt_cfg = config["optimization"]
            # Candidates are scored over the project lifetime only; the
            # resolver refuses the removed annual basis.
            self.objective_basis = self.opt_cfg["objective_basis"]
            self.projected_years, self.projected_degradation_rate = _resolve_horizon_and_pv_degradation(config)
            # Validated here so a bad engine setting fails before the first
            # candidate rather than inside a worker.
            _resolve_degradation_engine_spec(config["battery"])
            self.pv_params, self.module_area_m2 = _resolve_pv_module_and_area(config)
            self.batt_temp_cfg = config["battery"]["temperature"]
            self.indoor_model = config["battery"]["indoor_model"]
            self.emissions_params = EmissionsParams(**config["emissions"]) if config["emissions"] else None
            # Inverter AC rating follows the CAPEX sizing convention
            # (economics.calculate_costs): nameplate = DC peak / dc_ac_ratio.
            # The inverter each candidate pays for is also the one that clips
            # its production — same invariant as the App runner.
            cost_params = cost_params_from_config(config["costs"], config["financials"])
            self.dc_ac_ratio = cost_params.dc_ac_ratio
            self.inverter_efficiency = config["inverter_efficiency"]
            self.ac_output_scale = _validated_ac_output_scale(config)
            self.dc_output_scale = _validated_dc_output_scale(config)

            self.battery_replacement_treatment = _battery_replacement_treatment(config["battery"])

            self.fixed_azimuth = config["mode"]["fixed_azimuth"]

            # --- Dynamic Variable Setup ---
            if self.fixed_azimuth is not None:
                # RETROFIT MODE: 3 Variables
                # x[0]: n_modules (1-max_modules)
                # x[1]: battery_kwh (0-max_battery_kwh)
                # x[2]: surface_tilt (min_tilt_deg-max_tilt_deg)
                n_var = 3
                xl = np.array([1, 0.0, self.min_tilt_deg])
                xu = np.array([self.max_modules, self.max_battery_kwh, self.max_tilt_deg])
            else:
                # PROJECT MODE: 4 Variables (+ Azimuth)
                # Azimuth bounds depend on hemisphere
                lat = self.location["latitude"]
                if lat >= 0:
                    azi_lower, azi_upper = 90.0, 270.0  # Search around South (180°)
                else:
                    azi_lower, azi_upper = -90.0, 90.0  # Search around North (0°)
                n_var = 4
                xl = np.array([1, 0.0, self.min_tilt_deg, azi_lower])
                xu = np.array([self.max_modules, self.max_battery_kwh, self.max_tilt_deg, azi_upper])

            super().__init__(
                n_var=n_var,
                n_obj=2,
                n_ieq_constr=3 if self.enforce_zeb else 2,
                xl=xl,
                xu=xu,
                elementwise_runner=elementwise_runner or _serial_elementwise_runner,
            )

        def __getstate__(self):
            # Exclude elementwise_runner from pickling as it contains the Pool object
            state = self.__dict__.copy()
            state["elementwise_runner"] = None
            return state

        def _evaluate(self, x, out, *args, **kwargs):
            # Extract Genes
            n_modules = int(round(x[0]))
            battery_kwh = int(round(x[1]))
            tilt = x[2]

            if self.fixed_azimuth is not None:
                azimuth = self.fixed_azimuth
            else:
                azimuth = x[3]

            pv_params = self.pv_params
            module_area = self.module_area_m2

            # --- 1. Constraint Check: Area ---
            system_area = n_modules * module_area

            # --- 2. Simulation ---
            # Calculate PV Production (DC)
            dc_production = calculate_pv_production_dc(
                weather_data=self.tmy_data,
                location=self.loc_obj,
                tilt=tilt,
                surface_azimuth=azimuth,
                n_modules=n_modules,
                pv_params=pv_params,
                freq=self.freq,
                **self.model_options,
            )
            # Apply the DC-side correction before scoring. This keeps
            # clipping, charging and the part-load ratio on the corrected raw
            # array output.
            if self.dc_output_scale != 1.0:
                dc_production = dc_production * self.dc_output_scale

            # Load alignment (timezone- and DST-aware year remapping) happens
            # inside simulate_energy_balance — the same code path the App
            # uses. Positionally re-stamping the load onto the PV index here
            # (as the optimizer did before 0.3.4) discarded the load's real
            # timestamps and could shift it against PV by the UTC offset.
            if isinstance(self.houseload, pd.Series):
                houseload_df = self.houseload.to_frame(name="Load")
            else:
                houseload_df = self.houseload

            batt_spec = self.config["battery"]

            # Inverter AC nameplate shared by PV export and battery discharge
            pv_peak_w = n_modules * pv_params.Mpp
            inverter_ac_capacity_w = inverter_ac_capacity_w_for(pv_peak_w, self.dc_ac_ratio)

            temperature_series = build_battery_temperature_series(
                self.batt_temp_cfg,
                dc_production.index,
                weather_df=self.tmy_data,
                indoor_model=self.indoor_model,
            )

            # --- 3. Objective Calculations ---
            projected_metrics = _evaluate_projected_design_metrics(
                execution_backend=self.execution_backend,
                base_dc_power=dc_production,
                houseload=houseload_df,
                temperature_series=temperature_series,
                pv_params=pv_params,
                batt_spec=batt_spec,
                costs_cfg=self.config["costs"],
                fin_cfg=self.config["financials"],
                freq=self.freq,
                years_projection=self.projected_years,
                degradation_rate=self.projected_degradation_rate,
                n_modules=n_modules,
                battery_kwh=float(battery_kwh),
                inverter_efficiency=self.inverter_efficiency,
                inverter_ac_capacity_w=inverter_ac_capacity_w,
                ac_output_scale=self.ac_output_scale,
                emissions_params=self.emissions_params,
                tariff=self.tariff,
                instructions=self.pricing.instructions,
                reference_tariff=self.pricing.reference_tariff,
                reference_import_price_escalation=self.pricing.reference_import_price_escalation,
            )
            out.update(projected_metrics)
            objective_grid_dependence = 1.0 - float(projected_metrics["Projected_Grid_Independence_%"]) / 100.0
            objective_npv = float(projected_metrics["Projected_NPV"])
            objective_zeb = float(projected_metrics["Projected_ZEB_Ratio"])
            # The budget gates the CAPEX the result reports, so the cost a
            # feasible design shows is the cost that was checked.
            objective_capex = float(projected_metrics["Projected_Initial_Cost"])

            out["ZEB_Ratio"] = objective_zeb
            out["Objective_Grid_Independence_%"] = float(projected_metrics["Projected_Grid_Independence_%"])
            out["Objective_NPV"] = objective_npv

            # --- 4. Constraints Calculation ---
            # g1: Price <= Budget (g1 <= 0 means satisfied)
            g1 = objective_capex - self.budget_limit

            # g2: Area <= Max Area
            g2 = system_area - self.area_limit

            constraints = [g1, g2]
            if self.enforce_zeb:
                constraints.append(1.0 - objective_zeb)

            out["F"] = [objective_grid_dependence, -objective_npv]
            out["G"] = constraints

except ImportError as exc:
    _PYMOO_IMPORT_ERROR = exc


def _build_multi_objective_termination(n_gen: int, early_stop: Any):
    """Build the configured pymoo generation cap and objective-space stop."""
    if early_stop in (None, False):
        return ("n_gen", n_gen), None
    if early_stop is True:
        early_stop = {}
    if not isinstance(early_stop, dict):
        raise ValueError("optimization.early_stop must be a boolean or a mapping")
    if early_stop.get("enabled", True) is False:
        return ("n_gen", n_gen), None

    from pymoo.termination.collection import TerminationCollection
    from pymoo.termination.ftol import MultiObjectiveSpaceTermination
    from pymoo.termination.max_gen import MaximumGenerationTermination
    from pymoo.termination.robust import RobustTermination

    ftol = float(early_stop.get("ftol", 0.0025))
    if ftol <= 0.0:
        raise ValueError("optimization.early_stop.ftol must be greater than zero")
    period = max(1, int(early_stop.get("period", 10)))
    n_skip = max(0, int(early_stop.get("n_skip", 0)))
    min_gen = max(1, int(early_stop.get("min_gen", min(20, n_gen))))
    only_feasible = bool(early_stop.get("only_feasible", True))
    objective_stop = RobustTermination(
        MultiObjectiveSpaceTermination(tol=ftol, only_feas=only_feasible, n_skip=n_skip),
        period=period,
    )
    metadata = {
        "n_gen_cap": int(n_gen),
        "ftol": ftol,
        "period": period,
        "n_skip": n_skip,
        "min_gen": min_gen,
        "only_feasible": only_feasible,
    }
    return (
        TerminationCollection(
            MaximumGenerationTermination(n_gen),
            _MinGenerationTermination(objective_stop, minimum=min_gen),
        ),
        metadata,
    )


def optimize_system_multi_objective(
    tmy_data: pd.DataFrame,
    houseload: pd.DataFrame,
    config: Dict[str, Any],
    *,
    pop_size: int | None = None,
    n_gen: int | None = None,
    n_offsprings: int | None = None,
    seed: int | None = None,
    verbose: bool = False,
    n_procs: int = 1,
    execution_backend: str = DEFAULT_EXECUTION_BACKEND,
) -> OptimizationResult:
    """Run NSGA-II multi-objective PV/battery sizing.

    This is the public wrapper around :class:`SolarDesignProblem`. It optimizes
    module count, battery capacity, tilt, and optionally azimuth. It optimizes
    two values, projected lifetime grid independence and projected NPV,
    scoring every candidate over the full project lifetime with PV
    degradation, battery state propagation, and replacement events. ZEB
    remains a diagnostic unless ``constraints.enforce_zeb`` enables it as a
    feasibility constraint.
    Install ``breos[optimization]`` to provide the pymoo dependency.

    Args:
        tmy_data: One-year weather DataFrame.
        houseload: One-year load profile.
        config: Optimization config using the nested keys consumed by
            :class:`SolarDesignProblem` (``location``, ``constraints``,
            ``simulation``, ``pv``, ``battery``, ``costs``, ``financials``).
        pop_size: NSGA-II population size. Each of the four run settings is
            this argument, else the ``[optimization]`` key of the same name,
            else its default; an argument and a key that disagree raise.
            Default 40.
        n_gen: Number of generations. Default 100.
        n_offsprings: Offspring count per generation. Defaults to pymoo's
            algorithm default when ``None``.
        seed: Random seed passed to pymoo. Default 1.
        verbose: Print pymoo progress.
        n_procs: Candidate-evaluation worker processes. The default ``1``
            preserves serial behavior.

    Returns:
        :class:`OptimizationResult` whose ``details["pareto"]`` is a DataFrame
        with ``Modules``, ``Battery_kWh``, ``Tilt``, ``Azimuth``, objective
        values, ZEB diagnostics, and the ``Projected_*`` fields.

    Raises:
        ImportError: If pymoo is not installed.
        RuntimeError: If the optimizer returns no feasible solution.
    """
    if _PYMOO_IMPORT_ERROR is not None:
        raise ImportError(
            "pymoo is required for optimize_system_multi_objective(). Install with: pip install 'breos[optimization]'"
        ) from _PYMOO_IMPORT_ERROR

    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.optimize import minimize

    config = resolve_optimization_config(config)
    settings = resolve_run_settings(config, pop_size=pop_size, n_gen=n_gen, n_offsprings=n_offsprings, seed=seed)
    n_gen, n_offsprings, seed = settings["n_gen"], settings["n_offsprings"], settings["seed"]
    algorithm_kwargs: dict[str, Any] = {
        "pop_size": settings["pop_size"],
        "sampling": FloatRandomSampling(),
        "crossover": SBX(prob=0.9, eta=15),
        "mutation": PM(eta=20),
        "repair": DiscreteGridRepair(),
        "eliminate_duplicates": True,
    }
    if n_offsprings is not None:
        algorithm_kwargs["n_offsprings"] = n_offsprings

    if isinstance(n_procs, bool) or int(n_procs) < 1:
        raise ValueError("n_procs must be a positive integer")
    n_procs = int(n_procs)

    # Before the worker pool exists. An NSGA-II run is long and a missing
    # optional dependency should not surface as a traceback from inside a pool
    # that then has to be torn down.
    require_backend(execution_backend)

    # Invalid tariffs and rates must fail before a worker pool is created.
    economics = projection_rates_record(_projection_rates(config["financials"]))
    problem = SolarDesignProblem(
        tmy_data,
        houseload,
        config,
        execution_backend=execution_backend,
    )
    termination, early_stop_metadata = _build_multi_objective_termination(n_gen, config["optimization"]["early_stop"])
    pool = None
    if n_procs > 1:
        from multiprocessing import Pool

        try:
            from pymoo.parallelization import StarmapParallelization
        except ImportError:  # pymoo 0.6.1 compatibility
            from pymoo.core.problem import StarmapParallelization

        pool = Pool(n_procs, initializer=limit_worker_threads)
        problem.elementwise_runner = StarmapParallelization(pool.starmap)
    try:
        result = minimize(
            problem,
            NSGA2(**algorithm_kwargs),
            termination,
            seed=seed,
            verbose=verbose,
        )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    if result.X is None or result.F is None:
        raise RuntimeError("Multi-objective optimization found no feasible solutions.")

    x = np.atleast_2d(result.X)
    f = np.atleast_2d(result.F)
    fixed_azimuth = config["mode"]["fixed_azimuth"]
    if fixed_azimuth is not None:
        pareto = pd.DataFrame(x, columns=["Modules", "Battery_kWh", "Tilt"])
        pareto["Azimuth"] = fixed_azimuth
    else:
        pareto = pd.DataFrame(x, columns=["Modules", "Battery_kWh", "Tilt", "Azimuth"])

    pareto["Modules"] = pareto["Modules"].round().astype(int)
    pareto["Battery_kWh"] = pareto["Battery_kWh"].round().astype(float)
    pareto["Grid_Independence_%"] = (1 - f[:, 0]) * 100
    pareto["NPV"] = -f[:, 1]
    # pymoo stores every ``out`` value on the evaluated individual, so the
    # Pareto diagnostics are already available even when workers performed
    # the scoring. Enumerate custom data keys to keep optional diagnostics
    # (such as emissions) without re-running each expensive projection.
    diagnostic_keys = sorted(
        {
            key
            for individual in result.opt
            for key in individual.data
            if key.startswith("Projected_") or key.startswith("Objective_") or key == "ZEB_Ratio"
        }
    )
    diagnostics_df = pd.DataFrame({key: result.opt.get(key) for key in diagnostic_keys})
    for column in diagnostics_df.columns:
        pareto[column] = diagnostics_df[column].to_numpy()
    pareto["Grid_Independence_%"] = pareto["Projected_Grid_Independence_%"]
    pareto["NPV"] = pareto["Projected_NPV"]
    pareto["ZEB_Ratio"] = pareto["Projected_ZEB_Ratio"]

    # pymoo advances the counter after its termination update. Report the last
    # completed generation, matching the research workflow's saved metadata.
    actual_generations = max(0, int(getattr(result.algorithm, "n_gen", n_gen + 1)) - 1)
    provenance = {
        **problem.pricing.provenance(),
        "economics": economics,
        "simulation": dict(config["simulation"]),
        "weather": weather_metadata(problem.tmy_data),
        # The search bounds and run settings the search used, defaults included.
        "constraints": dict(config["constraints"]),
        "run_settings": settings,
        "battery_replacement_treatment": dict(problem.battery_replacement_treatment),
    }
    pareto.attrs["currency"] = provenance["currency"]
    return OptimizationResult(
        optimal_value=float("nan"),
        objective_value=float("nan"),
        iterations=actual_generations,
        details={
            "pareto": pareto,
            "pymoo_result": result,
            "problem": problem,
            "objective_basis": problem.objective_basis,
            "objective_names": ["Projected_Grid_Independence_%", "Projected_NPV"],
            "early_stop": early_stop_metadata,
            "n_procs": n_procs,
            "battery_replacement_treatment": problem.battery_replacement_treatment,
            "provenance": provenance,
        },
    )
