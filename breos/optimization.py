"""
Optimization module for PV system sizing and configuration.

This module provides the NSGA-II multi-objective design search
(:func:`optimize_system_multi_objective`) and the fixed-design projection
(:func:`evaluate_projected_design`) behind it.
"""

import warnings
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
    DEFAULT_OBJECTIVE_BASIS,
    EARLY_STOP_DEFAULTS,
    LAYOUTS,
    adjusted_max_tilt_deg,
    resolve_optimization_config,
    resolve_run_settings,
)
from breos.projection import CarryState, ProjectionYear, first_end_of_life_metrics, project_years
from breos.pv.model_options import configured_pv_model_kwargs
from breos.result_schema import RESULT_SCHEMA_VERSION
from breos.smart_charging import resolve_instructions, smart_charging_provenance
from breos.solar import (
    PVModuleParams,
    calculate_pv_production_dc,
)
from breos.tariffs import DEFAULT_CURRENCY, ResolvedTariff, reference_tariff_provenance, tariff_provenance
from breos.utils import package_version
from breos.weather import build_battery_temperature_series, resample_to_15min, weather_metadata


@dataclass
class OptimizationResult:
    """Result from an optimization run."""

    iterations: int
    details: Dict[str, Any]

    @property
    def optimal_value(self) -> float:
        """Always NaN: a multi-objective search has no single optimum.

        .. deprecated:: 0.8.0
           Read the front in ``details["pareto"]``; removed in 0.9.0.
        """
        warnings.warn(
            "OptimizationResult.optimal_value is deprecated and will be removed in BREOS 0.9.0; "
            "read the front in details['pareto']",
            DeprecationWarning,
            stacklevel=2,
        )
        return float("nan")

    @property
    def objective_value(self) -> float:
        """Always NaN: a multi-objective search has no single objective value.

        .. deprecated:: 0.8.0
           Read the front in ``details["pareto"]``; removed in 0.9.0.
        """
        warnings.warn(
            "OptimizationResult.objective_value is deprecated and will be removed in BREOS 0.9.0; "
            "read the front in details['pareto']",
            DeprecationWarning,
            stacklevel=2,
        )
        return float("nan")


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
    "replacement_min_remaining_years",
    "skipped_replacement_action",
)


def _battery_inverter_ac_capacity_w(batt_spec: Mapping[str, Any], battery_kwh: float) -> Optional[float]:
    """A candidate's AC-coupled battery inverter rating in W, or None without one (ADR 0004 C13).

    ``inverter_kw_per_kwh`` follows the candidate's nominal capacity;
    ``inverter_ac_rating_kw`` is fixed for the study.
    """
    # The 1 Wh cutoff calculate_costs prices a battery from.
    if batt_spec.get("coupling", "dc") != "ac" or not float(battery_kwh) * 1000.0 > 1.0:
        return None
    if batt_spec.get("inverter_ac_rating_kw") is not None:
        return float(batt_spec["inverter_ac_rating_kw"]) * 1000.0
    return float(batt_spec["inverter_kw_per_kwh"]) * float(battery_kwh) * 1000.0


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
    caller omits a setting. An AC-coupled candidate with a battery gets its
    battery inverter; without one it runs the PV-only path either way.
    """
    configured = {key: batt_spec[key] for key in _BATTERY_SPEC_KEYS if key in batt_spec}
    battery_inverter_w = _battery_inverter_ac_capacity_w(batt_spec, nominal_energy_wh / 1000.0)
    if battery_inverter_w is not None:
        configured.update(
            coupling="ac",
            battery_inverter_ac_capacity_w=battery_inverter_w,
            battery_inverter_efficiency=batt_spec["inverter_efficiency"],
        )
    return BatteryConfig(
        nominal_energy_wh=nominal_energy_wh,
        initial_soh=initial_soh,
        inverter_efficiency=inverter_efficiency,
        inverter_ac_capacity_w=inverter_ac_capacity_w,
        enable_replacement=enable_replacement,
        ac_output_scale=ac_output_scale,
        **configured,
    )


# The projected metric each first-crossing field is reported under.
_PROJECTED_END_OF_LIFE_METRICS = {
    "first_end_of_life_years": "Projected_First_End_Of_Life_Years",
    "first_end_of_life_action": "Projected_First_End_Of_Life_Action",
    "first_end_of_life_reason": "Projected_First_End_Of_Life_Reason",
    "first_end_of_life_soh_pct": "Projected_First_End_Of_Life_SOH_%",
}


def _battery_replacement_treatment(
    battery: Mapping[str, Any], costs: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    """How projected scoring treats battery replacement, for the provenance.

    ``allow_terminal_replacement`` is the configured policy for the final
    period; the earlier years' internal permission is not the user's setting.
    ``replacement_min_remaining_years`` is the configured minimum service
    time of a new pack, measured to the end of the project.
    ``enable_replacement = false`` skips every swap. An AC-coupled study
    also records its battery inverter's settings and price from ``costs``
    (ADR 0004 C14); each candidate's resolved rating is a result column.
    """
    treatment: Dict[str, Any] = {
        "method": "simulated_yearly_state_propagation",
        "description": "Projected scoring simulates every year and records actual replacement events.",
        "higher_fidelity_basis": "App multiyear SOH propagation",
        "enable_replacement": bool(battery.get("enable_replacement", True)),
        "allow_terminal_replacement": bool(battery.get("allow_terminal_replacement", True)),
        "replacement_min_remaining_years": float(battery.get("replacement_min_remaining_years", 0.0)),
        "skipped_replacement_action": str(battery.get("skipped_replacement_action", "keep")),
        "coupling": str(battery.get("coupling", "dc")),
        "terminal_period": (
            "The final degradation period of the last project year: the elapsed one-day window of simulation "
            "steps that ends on the horizon's last step, whole or partial. It is always aged and recorded; "
            "allow_terminal_replacement = false skips only its end-of-life replacement. Every earlier period, "
            "including the close of each earlier project year, replaces as usual."
        ),
        "minimum_service": (
            "In any project year, an end-of-life replacement that would leave the new pack less than "
            "replacement_min_remaining_years of the project to serve is skipped; 0 skips none. "
            'A skipped swap keeps the old pack in service ("keep") or switches it off for the rest of the '
            'project ("retire"), as skipped_replacement_action says.'
        ),
    }
    if treatment["coupling"] == "ac":
        per_kwh = battery.get("inverter_kw_per_kwh")
        treatment["battery_inverter"] = {
            "inverter_ac_rating_kw": battery.get("inverter_ac_rating_kw"),
            "inverter_kw_per_kwh": per_kwh,
            "rating_basis": "per kWh of each candidate's nominal capacity" if per_kwh is not None else "fixed",
            "inverter_efficiency": battery.get("inverter_efficiency"),
            "cost_per_kw": (costs or {}).get("battery_inverter_cost_per_kw"),
            "rating_column": "Battery_Inverter_AC_Rating_kW",
        }
    return treatment


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
    which checks the rates against the App's bounds and fills
    ``inflation_rate``, ``sell_price_inflation`` and ``discount_rate``; a raw
    table without them raises ``KeyError``. The escalators stay None when
    unset, so they inherit ``inflation_rate``.
    """

    def optional(key: str) -> Optional[float]:
        value = fin_cfg.get(key)
        return None if value is None else float(value)

    return {
        "inflation_rate": float(fin_cfg["inflation_rate"]),
        "sell_price_inflation": float(fin_cfg["sell_price_inflation"]),
        "import_price_escalation": optional("import_price_escalation"),
        "om_escalation": optional("om_escalation"),
        "replacement_cost_learning": optional("replacement_cost_learning") or 0.0,
        "discount_rate": float(fin_cfg["discount_rate"]),
    }


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
    currency: str = DEFAULT_CURRENCY,
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
    battery_inverter_w = _battery_inverter_ac_capacity_w(batt_spec, battery_kwh)
    costs = calculate_costs(
        n_modules=n_modules,
        module_power_w=pv_params.Mpp,
        battery_capacity_wh=battery_kwh * 1000.0,
        cost_params=cost_params,
        # The App's price, unless battery.replacement_cost sets one (ADR 0003 E4).
        replacement_cost_each=replacement_event_cost(
            battery_kwh, cost_params.battery_cost_per_kwh, batt_spec.get("replacement_cost")
        ),
        battery_coupling=batt_spec.get("coupling", "dc"),
        battery_inverter_ac_capacity_w=battery_inverter_w,
    )
    has_battery = battery_kwh > 0.0
    # Checked by resolve_optimization_config.
    degradation_engine, blast_model = batt_spec.get("degradation_engine", "native"), batt_spec.get("blast_model")
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
        currency=currency,
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
        # An AC-coupled study's battery inverter, per candidate (ADR 0004 C14).
        **(
            {
                "Battery_Inverter_AC_Rating_kW": float(battery_inverter_w or 0.0) / 1000.0,
                "Projected_Battery_Inverter_Cost": float(costs["battery_inverter_cost"]),
            }
            if batt_spec.get("coupling", "dc") == "ac"
            else {}
        ),
        "Projected_Replacement_Cost_T0_Prices": float(cost_projection.attrs["total_replacement_cost"]),
        "Projected_Total_Replacements": int(total_replacements),
        "Projected_Final_SOH_%": float(current_soh),
        # The first end-of-life crossing (ADR 0003 E11); NaN and None without one.
        **{
            _PROJECTED_END_OF_LIFE_METRICS[name]: value
            for name, value in first_end_of_life_metrics(projection.end_of_life_events).items()
        },
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

    # The config's resolved currency, of every money input and column.
    currency: str
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
            "currency": self.currency,
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
        {"reference_tariff": config.get("reference_tariff"), "resolution": resolution}, timezone, config["currency"]
    )
    smart_charging = resolve_smart_charging_spec(
        {
            "smart_charging": config["smart_charging"],
            "battery_kwh": battery_kwh,
            "battery_coupling": (config.get("battery") or {}).get("coupling", "dc"),
        },
        spec,
        battery_key,
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
        return _OptimizationTariff(tariff=tariff, currency=config["currency"], **reference)
    return _OptimizationTariff(
        tariff=tariff,
        currency=config["currency"],
        instructions=instructions,
        smart_charging=smart_charging_provenance(smart_charging, instructions, tariff),
        **reference,
    )


def _site_location(location: dict[str, Any]) -> Any:
    """Build the pvlib site the optimizer simulates from a resolved [location], the way App does.

    Without an ``altitude`` key pvlib looks the elevation up from the
    coordinates, exactly as App's ``Location(lat, lon, tz=...)`` does. The
    elevation sets the air pressure behind the refraction correction of the
    solar position, so modelling every site at sea level (the old default of
    0 m) gave the optimizer slightly different PV output from App for the same
    design.
    """
    from pvlib.location import Location

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


# The azimuths of an East-West layout: the first array faces east, the second west.
EAST_WEST_AZIMUTHS = (90.0, 270.0)


def _layout_arrays(layout: str, n_modules: int, tilt: Any, azimuth: Any) -> list[Tuple[int, Any, Any]]:
    """The fixed arrays of one design, as ``(modules, tilt, azimuth)``.

    A ``"single"`` layout is one array. An ``"east_west"`` layout puts
    floor(n / 2) modules at azimuth 90 and the rest at 270, both at ``tilt``,
    so the extra module of an odd count faces west. This is the split of the
    exhaustive lattice the upcoming publication compares the search with. An
    empty array is dropped, as the App drops one, unless every array is empty.
    """
    if layout == "single":
        return [(int(n_modules), tilt, azimuth)]
    if layout != "east_west":
        raise ValueError(f"layout must be one of: {', '.join(LAYOUTS)}; got {layout!r}")
    east = int(n_modules) // 2
    arrays = [(east, tilt, EAST_WEST_AZIMUTHS[0]), (int(n_modules) - east, tilt, EAST_WEST_AZIMUTHS[1])]
    return [array for array in arrays if array[0] > 0] or arrays[:1]


def _layout_dc_production(
    arrays: Sequence[Tuple[int, Any, Any]],
    *,
    weather_data: pd.DataFrame,
    location: Any,
    pv_params: PVModuleParams,
    freq: str,
    model_options: Mapping[str, Any],
) -> pd.Series:
    """The DC output of every array, summed as the App sums ``[[pv_arrays]]``.

    Each array is its own fixed-tilt PV model run, and the series are added
    in order with ``fill_value=0``, as ``calculate_multi_array_production``
    does, so an East-West design gives the App's floats.
    """
    total: Optional[pd.Series] = None
    for modules, tilt, azimuth in arrays:
        series = calculate_pv_production_dc(
            weather_data=weather_data,
            location=location,
            tilt=tilt,
            surface_azimuth=azimuth,
            n_modules=modules,
            pv_params=pv_params,
            freq=freq,
            **model_options,
        )
        total = series if total is None else total.add(series, fill_value=0.0)
    if total is None:
        raise ValueError("a design needs at least one PV array")
    return total


# eq=False: a study is compared by identity; its frames have no truth value.
@dataclass(frozen=True, eq=False)
class _DesignStudy:
    """What every design of one study shares, resolved once: the site, weather, pricing and models.

    :class:`SolarDesignProblem` scores many designs on one study and
    :func:`evaluate_projected_design` scores one; both score through
    :meth:`score`. Nothing in it depends on the design, so a search may score
    its designs in any order, or in worker processes, from the same study.
    """

    config: Dict[str, Any]
    location: Any
    weather: pd.DataFrame
    # One prepared frame per project year, or None to repeat ``weather``.
    weather_by_year: Optional[Tuple[pd.DataFrame, ...]]
    houseload: pd.DataFrame
    pricing: _OptimizationTariff
    pv_params: PVModuleParams
    module_area_m2: float
    model_options: Mapping[str, Any]
    dc_ac_ratio: float
    emissions_params: Optional[EmissionsParams]
    # On the weather grid, which every design's PV output is indexed on.
    temperature_series: pd.Series
    battery_replacement_treatment: Dict[str, Any]
    execution_backend: str

    @classmethod
    def resolve(
        cls,
        tmy_data: pd.DataFrame,
        houseload: Union[pd.DataFrame, pd.Series],
        config: Dict[str, Any],
        *,
        battery_kwh: float,
        battery_key: str = "battery_kwh",
        execution_backend: str,
        weather_by_year: Optional[Sequence[pd.DataFrame]] = None,
    ) -> "_DesignStudy":
        """Prepare a study from a resolved config.

        ``battery_kwh`` is the largest battery the study installs, which
        decides whether smart charging may run; ``battery_key`` names it in
        the error when it is zero.
        """
        location = _site_location(config["location"])
        weather = _prepare_study_weather(tmy_data, config, location)
        frames = None
        if weather_by_year is not None:
            frames = tuple(_prepare_study_weather(frame, config, location) for frame in weather_by_year)
        grid = frames[0].index if frames else weather.index
        pricing = _resolve_optimization_tariff(config, grid, battery_kwh, battery_key)
        years = config["simulation"]["years_projection"]
        if frames is not None and len(frames) != years:
            raise ValueError(f"weather_by_year has {len(frames)} years, expected {years}")
        pv_cfg = config["pv"]
        if pv_cfg.get("params"):
            pv_params = _pv_params_from_config(pv_cfg["params"])
        else:
            from breos.pv_modules import get_module

            pv_params = get_module(pv_cfg["module"])
        battery = config["battery"]
        return cls(
            config=config,
            location=location,
            weather=weather,
            weather_by_year=frames,
            # Load alignment (timezone- and DST-aware year remapping) happens
            # inside the energy balance, the code path the App uses, so the
            # load keeps its own timestamps.
            houseload=houseload.to_frame(name="Load") if isinstance(houseload, pd.Series) else houseload,
            pricing=pricing,
            pv_params=pv_params,
            module_area_m2=_module_area_from_dimensions(_dimensions_from_section(pv_cfg)),
            model_options=configured_pv_model_kwargs(config),
            # The inverter each design pays for is the one that clips its
            # output: nameplate = DC peak / dc_ac_ratio, as App sizes it.
            dc_ac_ratio=cost_params_from_config(config["costs"], config["financials"]).dc_ac_ratio,
            emissions_params=EmissionsParams(**config["emissions"]) if config["emissions"] else None,
            # A weather sequence keeps the representative year's temperatures,
            # so they are restamped onto the sequence's calendar year rather
            # than reindexed across years, which found no match and used to
            # fall back to 25 C.
            temperature_series=build_battery_temperature_series(
                battery["temperature"],
                grid,
                weather_df=weather,
                indoor_model=battery["indoor_model"],
                align_weather_year=frames is not None,
            ),
            battery_replacement_treatment=_battery_replacement_treatment(battery, config["costs"]),
            execution_backend=execution_backend,
        )

    def dc_power(self, arrays: Sequence[Tuple[int, Any, Any]]) -> Union[pd.Series, list[pd.Series]]:
        """The DC output of a design's arrays: one series, or one per project year for a weather sequence.

        ``dc_output_scale`` is applied here, before dispatch, so charging,
        clipping and the part-load ratio all respond to it; that is why it is
        the correction to reach for when the model under-predicts.
        """
        scale = self.config["dc_output_scale"]

        def dc_for(frame: pd.DataFrame) -> pd.Series:
            series = _layout_dc_production(
                arrays,
                weather_data=frame,
                location=self.location,
                pv_params=self.pv_params,
                freq=self.config["simulation"]["resolution"],
                model_options=self.model_options,
            )
            return series if scale == 1.0 else series * scale

        if self.weather_by_year is None:
            return dc_for(self.weather)
        # Every year must land on the same intra-year index, because the load
        # profile and the battery temperature series are aligned to it once.
        reference = dc_for(self.weather_by_year[0])
        series = [reference]
        for frame in self.weather_by_year[1:]:
            year_dc = dc_for(frame)
            if len(year_dc) != len(reference):
                raise ValueError(
                    "every weather_by_year frame must produce the same number of "
                    f"timesteps; got {len(year_dc)} against {len(reference)}"
                )
            year_dc.index = reference.index
            series.append(year_dc)
        return series

    def score(
        self,
        arrays: Sequence[Tuple[int, Any, Any]],
        *,
        n_modules: int,
        battery_kwh: float,
        return_tables: bool = False,
    ) -> Dict[str, Any]:
        """The projected metrics of one design: its arrays, module count and battery."""
        config = self.config
        return _evaluate_projected_design_metrics(
            execution_backend=self.execution_backend,
            base_dc_power=self.dc_power(arrays),
            houseload=self.houseload,
            temperature_series=self.temperature_series,
            pv_params=self.pv_params,
            batt_spec=config["battery"],
            costs_cfg=config["costs"],
            fin_cfg=config["financials"],
            freq=config["simulation"]["resolution"],
            years_projection=int(config["simulation"]["years_projection"]),
            degradation_rate=float(config["pv"]["degradation_rate"]),
            n_modules=n_modules,
            battery_kwh=battery_kwh,
            inverter_efficiency=float(config["inverter_efficiency"]),
            inverter_ac_capacity_w=inverter_ac_capacity_w_for(n_modules * self.pv_params.Mpp, self.dc_ac_ratio),
            emissions_params=self.emissions_params,
            return_tables=return_tables,
            ac_output_scale=config["ac_output_scale"],
            tariff=self.pricing.tariff,
            instructions=self.pricing.instructions,
            reference_tariff=self.pricing.reference_tariff,
            reference_import_price_escalation=self.pricing.reference_import_price_escalation,
            currency=self.pricing.currency,
        )

    def provenance(self, **extra: Any) -> Dict[str, Any]:
        """The study's provenance record; ``extra`` entries go before the battery replacement treatment."""
        config = self.config
        frames = self.weather_by_year
        return {
            **self.pricing.provenance(),
            "economics": projection_rates_record(_projection_rates(config["financials"])),
            "simulation": dict(config["simulation"]),
            "weather": weather_metadata(self.weather),
            **({"weather_by_year": [weather_metadata(frame) for frame in frames]} if frames is not None else {}),
            **extra,
            "battery_replacement_treatment": dict(self.battery_replacement_treatment),
        }


def evaluate_projected_design(
    tmy_data: pd.DataFrame,
    houseload: pd.DataFrame,
    config: Dict[str, Any],
    *,
    n_modules: int,
    battery_kwh: float,
    tilt: float,
    azimuth: Optional[float] = None,
    layout: str = "single",
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
        tilt: PV surface tilt in degrees, of every array.
        azimuth: PV surface azimuth in degrees, for the ``"single"`` layout.
            An East-West layout fixes its azimuths, so it takes none.
        layout: ``"single"``, one array at ``tilt`` and ``azimuth``, or
            ``"east_west"``: floor(n_modules / 2) modules at azimuth 90 and
            the rest at 270, the App's ``[[pv_arrays]]`` result for the same
            two arrays. Area, CAPEX and the inverter rating count every
            module.
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
    if layout == "single" and azimuth is None:
        raise ValueError("the single layout needs an azimuth")
    if layout == "east_west" and azimuth is not None:
        raise ValueError("the east_west layout faces azimuths 90 and 270; do not pass an azimuth")
    arrays = _layout_arrays(layout, int(n_modules), float(tilt), None if azimuth is None else float(azimuth))

    # Before the PV production model runs, not after it. Computing a year of
    # irradiance and then failing on a missing import wastes the expensive part
    # and reports the cheap problem late.
    require_backend(execution_backend)

    study = _DesignStudy.resolve(
        tmy_data,
        houseload,
        resolve_optimization_config(config),
        battery_kwh=float(battery_kwh),
        execution_backend=execution_backend,
        weather_by_year=weather_by_year,
    )
    raw_metrics = study.score(arrays, n_modules=int(n_modules), battery_kwh=float(battery_kwh), return_tables=True)
    yearly = raw_metrics.pop("_yearly_summary_df")
    financial = raw_metrics.pop("_cost_projection_df")
    metrics = {
        "Modules": int(n_modules),
        "Battery_kWh": float(battery_kwh),
        "Tilt": float(tilt),
        # An East-West design has two azimuths, which its layout names.
        "Azimuth": float(azimuth) if azimuth is not None else np.nan,
        "Layout": layout,
        **raw_metrics,
    }
    return ProjectedDesignResult(metrics=metrics, yearly=yearly, financial=financial, provenance=study.provenance())


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
            # integers, tilt and azimuth to 5 degrees, and the layout gene to
            # a layout index. Every column stays inside the problem bounds.
            for col, step in enumerate(problem.gene_steps):
                X[:, col] = _snap_to_grid_within_bounds(X[:, col], step, problem.xl[col], problem.xu[col])
            if problem.layout_gene is not None:
                # An East-West design does not read the single array's tilt
                # and azimuth genes. They are set to their lowest grid value,
                # so two East-West designs that differ only there are the
                # duplicates they are, and NSGA-II does not score both.
                east_west = X[:, problem.layout_gene] == problem.layouts.index("east_west")
                for col in problem.single_orientation_genes:
                    X[east_west, col] = _snap_to_grid_within_bounds(
                        problem.xl[col], problem.gene_steps[col], problem.xl[col], problem.xu[col]
                    )
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
            backend = validate_execution_backend(execution_backend)
            # Checked and defaulted once, before any model preparation: an
            # unknown key raises, and every default is in the resolved config.
            config = resolve_optimization_config(config)
            self.config = config
            constraints = config["constraints"]
            # One site, weather, schedule and price resolution per search,
            # shared by every candidate and project year.
            self.study = _DesignStudy.resolve(
                tmy_data,
                houseload,
                config,
                battery_kwh=float(constraints["max_battery_kwh"]),
                battery_key="constraints.max_battery_kwh",
                execution_backend=backend,
            )
            # Candidates are scored over the project lifetime only; the
            # resolver refuses the removed annual basis.
            self.objective_basis = DEFAULT_OBJECTIVE_BASIS

            self.fixed_azimuth = config["mode"]["fixed_azimuth"]
            # The layouts searched, in LAYOUTS order, and the fixed tilt of
            # an East-West design.
            self.layouts = tuple(config["mode"]["layouts"])
            self.east_west_tilt_deg = float(config["mode"]["east_west_tilt_deg"])

            # --- Dynamic Variable Setup ---
            # x[0]: n_modules (1-max_modules)
            # x[1]: battery_kwh (0-max_battery_kwh)
            lower = [1, 0.0]
            upper = [constraints["max_modules"], constraints["max_battery_kwh"]]
            self.gene_steps: Tuple[float, ...] = (1.0, 1.0)
            # The genes only the single layout reads: tilt, then azimuth
            # unless it is fixed.
            self.single_orientation_genes: Tuple[int, ...] = ()
            if "single" in self.layouts:
                # x[2]: surface_tilt (min_tilt_deg-max_tilt_deg)
                lower.append(float(constraints["min_tilt_deg"]))
                upper.append(_resolve_max_tilt_deg(constraints, config["location"]["latitude"]))
                if self.fixed_azimuth is None:
                    # x[3]: azimuth; the bounds depend on the hemisphere
                    if config["location"]["latitude"] >= 0:
                        lower.append(90.0)  # Search around South (180°)
                        upper.append(270.0)
                    else:
                        lower.append(-90.0)  # Search around North (0°)
                        upper.append(90.0)
                self.single_orientation_genes = tuple(range(2, len(lower)))
                self.gene_steps += (5.0,) * len(self.single_orientation_genes)
            # The last gene picks the layout when more than one is searched:
            # its value is an index into self.layouts.
            self.layout_gene: Optional[int] = None
            if len(self.layouts) > 1:
                self.layout_gene = len(lower)
                lower.append(0)
                upper.append(len(self.layouts) - 1)
                self.gene_steps += (1.0,)
            xl = np.array(lower)
            xu = np.array(upper)

            super().__init__(
                n_var=len(lower),
                n_obj=2,
                n_ieq_constr=3 if constraints["enforce_zeb"] else 2,
                xl=xl,
                xu=xu,
                elementwise_runner=elementwise_runner or _serial_elementwise_runner,
            )
            # Every design scored so far, in the order first scored: the
            # design vector maps to its ``out`` record, and to the number of
            # times the search asked for it.
            self.scored: Dict[Tuple[float, ...], Dict[str, Any]] = {}
            self.score_requests: Dict[Tuple[float, ...], int] = {}

        @property
        def pricing(self) -> _OptimizationTariff:
            return self.study.pricing

        @property
        def tariff(self) -> Optional[ResolvedTariff]:
            return self.study.pricing.tariff

        @property
        def module_area_m2(self) -> float:
            return self.study.module_area_m2

        @property
        def battery_replacement_treatment(self) -> Dict[str, Any]:
            return self.study.battery_replacement_treatment

        def __getstate__(self):
            # Exclude elementwise_runner from pickling as it contains the Pool object.
            # The scored designs stay in the parent: every worker task pickles
            # the problem, and the record grows with the search.
            state = self.__dict__.copy()
            state["elementwise_runner"] = None
            state["scored"] = {}
            state["score_requests"] = {}
            return state

        def _evaluate_elementwise(self, X, out, *args, **kwargs):
            # Simulate each design vector once per problem. NSGA-II drops
            # duplicates within a population and its offspring only, so a
            # design it discarded and finds again would be simulated again;
            # here it takes the record of its first scoring instead. The
            # repair puts every gene on its grid, so equal designs are equal
            # floats, and a design's record depends on its vector alone.
            keys = [tuple(float(value) for value in x) for x in X]
            unscored = list(dict.fromkeys(key for key in keys if key not in self.scored))
            if unscored:
                f = self.elementwise_func(self, args, kwargs)
                records = self.elementwise_runner(f, np.array(unscored, dtype=float))
                self.scored.update(zip(unscored, records, strict=True))
            for key in keys:
                self.score_requests[key] = self.score_requests.get(key, 0) + 1

            # As pymoo's ElementwiseProblem gathers the records.
            for key in keys:
                for name, value in self.scored[key].items():
                    if out.get(name) is None:
                        out[name] = []
                    out[name].append(value)
            for name in out:
                if out[name] is not None:
                    out[name] = np.array(out[name])

        def design_layout(self, x) -> str:
            """The layout a design vector encodes."""
            if self.layout_gene is None:
                return self.layouts[0]
            return self.layouts[int(round(x[self.layout_gene]))]

        def _evaluate(self, x, out, *args, **kwargs):
            n_modules = int(round(x[0]))
            battery_kwh = int(round(x[1]))
            layout = self.design_layout(x)
            if layout == "east_west":
                tilt, azimuth = self.east_west_tilt_deg, None
            else:
                tilt = x[2]
                azimuth = self.fixed_azimuth if self.fixed_azimuth is not None else x[3]

            # Every array's modules: the layouts split one module count.
            projected_metrics = self.study.score(
                _layout_arrays(layout, n_modules, tilt, azimuth), n_modules=n_modules, battery_kwh=float(battery_kwh)
            )
            out.update(projected_metrics)
            objective_grid_dependence = 1.0 - float(projected_metrics["Projected_Grid_Independence_%"]) / 100.0
            objective_npv = float(projected_metrics["Projected_NPV"])
            objective_zeb = float(projected_metrics["Projected_ZEB_Ratio"])
            out["ZEB_Ratio"] = objective_zeb
            out["Objective_Grid_Independence_%"] = float(projected_metrics["Projected_Grid_Independence_%"])
            out["Objective_NPV"] = objective_npv

            out["F"] = [objective_grid_dependence, -objective_npv]
            # Each constraint is met at or below zero. The budget gates the
            # CAPEX the result reports, so the cost a feasible design shows
            # is the cost that was checked.
            constraints = self.config["constraints"]
            out["G"] = [
                float(projected_metrics["Projected_Initial_Cost"]) - constraints["budget"],
                n_modules * self.module_area_m2 - constraints["max_area_m2"],
                *([1.0 - objective_zeb] if constraints["enforce_zeb"] else []),
            ]

except ImportError as exc:
    _PYMOO_IMPORT_ERROR = exc


def _build_multi_objective_termination(n_gen: int, early_stop: Any):
    """Build the configured pymoo generation cap and objective-space stop from a resolved early_stop."""
    if early_stop in (None, False):
        return ("n_gen", n_gen), None
    settings = {**EARLY_STOP_DEFAULTS, "min_gen": min(20, n_gen), **({} if early_stop is True else early_stop)}
    if not settings["enabled"]:
        return ("n_gen", n_gen), None

    from pymoo.termination.collection import TerminationCollection
    from pymoo.termination.ftol import MultiObjectiveSpaceTermination
    from pymoo.termination.max_gen import MaximumGenerationTermination
    from pymoo.termination.robust import RobustTermination

    ftol = float(settings["ftol"])
    period, n_skip, min_gen = settings["period"], settings["n_skip"], settings["min_gen"]
    only_feasible = settings["only_feasible"]
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


def _design_columns(problem: Any, x: np.ndarray) -> pd.DataFrame:
    """The design columns of the Pareto table, one row per design vector.

    ``Modules``, ``Battery_kWh``, ``Tilt``, ``Azimuth`` and ``Layout``. An
    East-West row has the configured tilt and a NaN azimuth, since its two
    arrays face 90 and 270.
    """
    fixed_azimuth = problem.fixed_azimuth
    designs = pd.DataFrame(x[:, :2], columns=["Modules", "Battery_kWh"])
    if problem.single_orientation_genes:
        designs["Tilt"] = x[:, 2]
        designs["Azimuth"] = fixed_azimuth if fixed_azimuth is not None else x[:, 3]
    layouts = [problem.design_layout(row) for row in x]
    designs["Layout"] = layouts
    east_west = np.array([layout == "east_west" for layout in layouts], dtype=bool)
    if east_west.any():
        if "Tilt" not in designs:
            designs["Tilt"] = np.nan
            designs["Azimuth"] = np.nan
        designs["Azimuth"] = designs["Azimuth"].astype(float)
        designs.loc[east_west, "Tilt"] = problem.east_west_tilt_deg
        designs.loc[east_west, "Azimuth"] = np.nan
        designs = designs[["Modules", "Battery_kWh", "Tilt", "Azimuth", "Layout"]]
    return designs


def _design_table(problem: Any, x: np.ndarray, records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """The design columns and scored values of each design vector.

    ``records`` holds each design's ``out`` record, in the order of ``x``.
    Every ``Projected_*`` and ``Objective_*`` key is kept, so optional
    diagnostics such as emissions need no extra projection.
    """
    table = _design_columns(problem, x)
    table["Modules"] = table["Modules"].round().astype(int)
    table["Battery_kWh"] = table["Battery_kWh"].round().astype(float)
    diagnostic_keys = sorted(
        {
            key
            for record in records
            for key in record
            if key.startswith("Projected_")
            or key.startswith("Objective_")
            or key in ("ZEB_Ratio", "Battery_Inverter_AC_Rating_kW")
        }
    )
    table["Grid_Independence_%"] = np.array([record["Projected_Grid_Independence_%"] for record in records])
    table["NPV"] = np.array([record["Projected_NPV"] for record in records])
    for key in diagnostic_keys:
        table[key] = np.array([record.get(key) for record in records])
    return table


# The search's inequality constraints, in the order of ``out["G"]``.
_CONSTRAINT_COLUMNS = ("Constraint_Budget", "Constraint_Area", "Constraint_ZEB")


def _evaluated_designs(problem: Any) -> pd.DataFrame:
    """Every design the search scored, one row each, in the order first scored.

    The columns of the Pareto table, then ``Constraint_Budget``,
    ``Constraint_Area`` and, with ``enforce_zeb``, ``Constraint_ZEB`` (each
    at most zero when met), ``Feasible``, and ``Evaluations``, the number of
    times the search asked for the design.
    """
    keys = list(problem.scored)
    records = [problem.scored[key] for key in keys]
    table = _design_table(problem, np.array(keys, dtype=float).reshape(len(keys), problem.n_var), records)
    constraints = np.array([record["G"] for record in records], dtype=float).reshape(len(keys), problem.n_ieq_constr)
    for column, values in zip(_CONSTRAINT_COLUMNS[: problem.n_ieq_constr], constraints.T, strict=True):
        table[column] = values
    table["Feasible"] = (constraints <= 0.0).all(axis=1)
    table["Evaluations"] = np.array([problem.score_requests[key] for key in keys], dtype=int)
    return table


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
    module count, battery capacity, tilt, and optionally azimuth, and, when
    ``mode.layouts`` names more than one, the layout: one array or an
    East-West roof at ``mode.east_west_tilt_deg``. It optimizes
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
        with ``Modules``, ``Battery_kWh``, ``Tilt``, ``Azimuth``, ``Layout``,
        objective values, ZEB diagnostics, and the ``Projected_*`` fields. An
        East-West row has the configured tilt and a NaN azimuth.
        ``details["evaluated"]`` holds every design the search scored, with
        the same columns plus the constraint values, ``Feasible`` and
        ``Evaluations``. ``details["evaluation_counts"]`` gives the
        evaluations the search asked for, the simulations it ran and the
        distinct designs: a design is simulated once per search, however
        often NSGA-II returns to it.

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

    # Invalid tariffs must fail before a worker pool is created.
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

    # pymoo stores every ``out`` value on the evaluated individual, so the
    # Pareto diagnostics are already available even when workers performed
    # the scoring.
    pareto = _design_table(problem, np.atleast_2d(result.X), [individual.data for individual in result.opt])
    evaluated = _evaluated_designs(problem)
    evaluation_counts = {
        "evaluations": int(sum(problem.score_requests.values())),
        "simulations": len(problem.scored),
        "distinct_designs": len(problem.scored),
    }

    # pymoo advances the counter after its termination update, so this is the
    # last completed generation.
    actual_generations = max(0, int(getattr(result.algorithm, "n_gen", n_gen + 1)) - 1)
    # The search bounds and run settings the search used, defaults included.
    provenance = problem.study.provenance(
        constraints=dict(config["constraints"]),
        mode={**config["mode"], "layouts": list(config["mode"]["layouts"])},
        run_settings=settings,
    )
    pareto.attrs["currency"] = provenance["currency"]
    evaluated.attrs["currency"] = provenance["currency"]
    return OptimizationResult(
        iterations=actual_generations,
        details={
            "pareto": pareto,
            "evaluated": evaluated,
            "evaluation_counts": evaluation_counts,
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
