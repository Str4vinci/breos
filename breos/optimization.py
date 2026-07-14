"""
Optimization module for PV system sizing and configuration.

This module provides:
- Tilt angle optimization
- Battery sizing optimization
- ZEB (Zero Energy Building) sizing
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from breos.battery import BatteryConfig, simulate_energy_balance
from breos.economics import calculate_costs, cost_params_from_config, system_ac_production_power
from breos.solar import PVModuleParams, calculate_pv_production_dc, default_azimuth
from breos.utils import get_hours_per_step


@dataclass
class OptimizationResult:
    """Result from an optimization run."""

    optimal_value: float
    objective_value: float
    iterations: int
    details: Dict[str, Any]


def _serial_elementwise_runner(func: Callable[[Any], Any], args: list[Any]) -> list[Any]:
    """Fallback pymoo elementwise runner for single-process evaluation."""
    return [func(arg) for arg in args]


def _resolve_max_tilt_deg(constraints: Dict[str, Any], latitude: float) -> float:
    """Resolve the optimization tilt upper bound from constraints."""
    value = constraints.get("max_tilt_deg", 90.0)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "adjust":
            margin = float(constraints.get("tilt_margin_deg", 15.0))
            adjusted = 5.0 * round((abs(float(latitude)) + margin) / 5.0)
            return float(np.clip(adjusted, 60.0, 90.0))
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported constraints.max_tilt_deg value: {value!r}. Use a number or 'adjust'."
            ) from exc
    return float(value)


def optimize_tilt(
    weather_data: pd.DataFrame,
    location,
    n_modules: int,
    pv_params: Optional[PVModuleParams] = None,
    surface_azimuth: Optional[float] = None,
    tilt_range: Tuple[float, float] = (0.0, 60.0),
    objective: str = "max_production",
    freq: str = "h",
    n_points: int = 13,
    verbose: bool = True,
) -> OptimizationResult:
    """
    Optimize panel tilt angle for maximum production or self-consumption.

    Args:
        weather_data: Weather DataFrame with solar irradiance
        location: pvlib Location object
        n_modules: Number of PV modules
        pv_params: PV module parameters
        surface_azimuth: Panel azimuth (180=South, 0=North). If None, auto-detected from hemisphere.
        tilt_range: (min_tilt, max_tilt) in degrees
        objective: 'max_production' or 'max_self_consumption'
        freq: Time frequency
        n_points: Number of tilt values to evaluate
        verbose: Print progress

    Returns:
        OptimizationResult with optimal tilt
    """
    if surface_azimuth is None:
        surface_azimuth = default_azimuth(location.latitude)
    tilts = np.linspace(tilt_range[0], tilt_range[1], n_points)
    results = []

    for tilt in tilts:
        try:
            dc_power = calculate_pv_production_dc(
                weather_data=weather_data,
                location=location,
                tilt=tilt,
                surface_azimuth=surface_azimuth,
                n_modules=n_modules,
                pv_params=pv_params,
                freq=freq,
                verbose=False,
            )
            total_production = dc_power.sum() * get_hours_per_step(freq) / 1000  # kWh (DC)
            results.append({"tilt": tilt, "production_kwh": total_production})

            if verbose:
                print(f"  Tilt {tilt:.1f}°: {total_production:.1f} kWh")

        except Exception as e:
            if verbose:
                print(f"  Tilt {tilt:.1f}°: Error - {e}")
            results.append({"tilt": tilt, "production_kwh": 0})

    results_df = pd.DataFrame(results)
    optimal_idx = results_df["production_kwh"].idxmax()
    optimal_tilt = results_df.loc[optimal_idx, "tilt"]
    optimal_production = results_df.loc[optimal_idx, "production_kwh"]

    if verbose:
        print(f"\nOptimal tilt: {optimal_tilt:.1f}° ({optimal_production:.1f} kWh)")

    return OptimizationResult(
        optimal_value=optimal_tilt,
        objective_value=optimal_production,
        iterations=len(tilts),
        details={"all_results": results_df},
    )


def optimize_tilt_brent(
    weather_data: pd.DataFrame,
    location,
    n_modules: int,
    pv_params: Optional[PVModuleParams] = None,
    surface_azimuth: Optional[float] = None,
    tilt_range: Tuple[float, float] = (0.0, 60.0),
    freq: str = "h",
    tol: float = 1.0,
    verbose: bool = True,
) -> OptimizationResult:
    """
    Optimize panel tilt using Brent's method (faster than grid search).

    Args:
        weather_data: Weather DataFrame
        location: pvlib Location object
        n_modules: Number of modules
        pv_params: PV module parameters
        surface_azimuth: Panel azimuth
        tilt_range: Search bounds
        freq: Time frequency
        tol: Optimization tolerance
        verbose: Print progress

    Returns:
        OptimizationResult with optimal tilt
    """
    from scipy.optimize import minimize_scalar

    if surface_azimuth is None:
        surface_azimuth = default_azimuth(location.latitude)

    iterations = [0]

    def objective(tilt):
        iterations[0] += 1
        try:
            dc_power = calculate_pv_production_dc(
                weather_data=weather_data,
                location=location,
                tilt=tilt,
                surface_azimuth=surface_azimuth,
                n_modules=n_modules,
                pv_params=pv_params,
                freq=freq,
                verbose=False,
            )
            # Negative kWh (DC) for minimization
            production = -dc_power.sum() * get_hours_per_step(freq) / 1000

            if verbose:
                print(f"  Iteration {iterations[0]}: tilt={tilt:.2f}°, production={-production:.1f} kWh")

            return production
        except Exception as e:
            if verbose:
                print(f"  Iteration {iterations[0]}: tilt={tilt:.2f}° failed - {e}")
            return np.inf

    result = minimize_scalar(objective, bounds=tilt_range, method="bounded", options={"xatol": tol})

    return OptimizationResult(
        optimal_value=result.x, objective_value=-result.fun, iterations=iterations[0], details={"scipy_result": result}
    )


def optimize_battery_size(
    pv_dc: pd.Series,
    houseload: pd.DataFrame,
    battery_sizes_wh: list,
    start_time: Optional[pd.Timestamp] = None,
    end_time: Optional[pd.Timestamp] = None,
    freq: str = "h",
    objective: str = "max_self_consumption",
    verbose: bool = True,
) -> OptimizationResult:
    """
    Optimize battery size for self-consumption or grid independence.

    Args:
        pv_dc: PV DC production series
        houseload: Load DataFrame
        battery_sizes_wh: List of battery sizes to evaluate
        start_time: Simulation start
        end_time: Simulation end
        freq: Time frequency
        objective: 'max_self_consumption' or 'min_import'
        verbose: Print progress

    Returns:
        OptimizationResult with optimal battery size
    """
    results = []

    for size_wh in battery_sizes_wh:
        config = BatteryConfig(nominal_energy_wh=size_wh)

        try:
            df, total_pv, summary, _, _, _ = simulate_energy_balance(
                pv_dc=pv_dc,
                houseload=houseload,
                battery_config=config,
                start_time=start_time,
                end_time=end_time,
                freq=freq,
                debug=False,
            )

            grid_independence = summary["Grid Independence [%]"].iloc[0]
            import_pct = summary["Import [%]"].iloc[0]
            total_pv_kwh = summary["Total PV [kWh]"].iloc[0]
            export_kwh = summary["Sell [kWh]"].iloc[0]
            self_consumption_pct = ((total_pv_kwh - export_kwh) / total_pv_kwh) * 100 if total_pv_kwh > 0 else 0.0

            results.append(
                {
                    "battery_size_wh": size_wh,
                    "battery_size_kwh": size_wh / 1000,
                    "grid_independence": grid_independence,
                    "import_percent": import_pct,
                    "self_consumption": self_consumption_pct,
                }
            )

            if verbose:
                print(f"  {size_wh / 1000:.1f} kWh: {grid_independence:.1f}% grid independence")

        except Exception as e:
            if verbose:
                print(f"  {size_wh / 1000:.1f} kWh: Error - {e}")

    results_df = pd.DataFrame(results)
    if results_df.empty:
        raise RuntimeError("No battery sizes could be evaluated.")

    if objective == "max_self_consumption":
        optimal_idx = results_df["self_consumption"].idxmax()
        optimal_value = results_df.loc[optimal_idx, "self_consumption"]
    elif objective == "max_grid_independence":
        optimal_idx = results_df["grid_independence"].idxmax()
        optimal_value = results_df.loc[optimal_idx, "grid_independence"]
    elif objective == "min_import":
        optimal_idx = results_df["import_percent"].idxmin()
        optimal_value = results_df.loc[optimal_idx, "import_percent"]
    else:
        raise ValueError("objective must be 'max_self_consumption', 'max_grid_independence', or 'min_import'")

    optimal_size = results_df.loc[optimal_idx, "battery_size_wh"]

    return OptimizationResult(
        optimal_value=optimal_size,
        objective_value=optimal_value,
        iterations=len(battery_sizes_wh),
        details={"all_results": results_df},
    )


def size_for_zeb(houseload: pd.DataFrame, ac_loss: pd.Series, current_n_modules: int) -> Dict[str, float]:
    """
    Calculate PV system size needed for Zero Energy Building (ZEB).

    Args:
        houseload: Annual load profile
        ac_loss: Usable AC system production for the current system (legacy
            parameter name retained for compatibility)
        current_n_modules: Current number of modules

    Returns:
        Dict with ZEB sizing requirements
    """
    yearly_load = houseload.iloc[:, 0].sum()
    yearly_pv = ac_loss.sum()

    if yearly_pv <= 0:
        return {"error": "No PV production", "modules_needed": float("inf")}

    pv_per_module = yearly_pv / current_n_modules if current_n_modules > 0 else yearly_pv
    modules_for_zeb = yearly_load / pv_per_module

    ratio = yearly_pv / yearly_load

    return {
        "yearly_load_wh": yearly_load,
        "yearly_pv_wh": yearly_pv,
        "pv_to_load_ratio": ratio,
        "is_zeb": ratio >= 1.0,
        "modules_needed_for_zeb": int(np.ceil(modules_for_zeb)),
        "additional_modules_needed": int(np.ceil(modules_for_zeb - current_n_modules)),
    }


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

# Constants for defaults (can be overridden by config)
DEFAULT_PANEL_WP = 550
DEFAULT_MODULE_AREA = 1.134 * 2.278
DEFAULT_INFLATION_ELEC = 0.02
DEFAULT_DISCOUNT_RATE = 0.0

DEFAULT_PROJECT_LIFESPAN = 20


def _estimate_battery_replacement_treatment(
    battery_kwh: float,
    annual_soh_loss_pct: float,
    initial_soh_pct: float,
    eol_percentage: float,
    project_lifespan: int,
    replacement_cost_eur: float,
) -> Dict[str, Any]:
    """Approximate replacement years by repeating the simulated year-1 SOH loss.

    The first interval starts at the candidate's configured initial SOH. Each
    replacement resets SOH to 100%, matching :class:`BatteryConfig`; later
    intervals therefore use the full 100%-to-EOL window. This is deliberately
    a steady-state approximation. The App's multiyear projection remains the
    higher-fidelity path because it propagates SOH and records actual events.
    """
    annual_loss = max(0.0, float(annual_soh_loss_pct))
    eol_pct = float(eol_percentage) * 100.0
    treatment: Dict[str, Any] = {
        "method": "repeat_simulated_year_1_soh_loss_to_eol",
        "annual_soh_loss_pct": annual_loss,
        "initial_soh_pct": float(initial_soh_pct),
        "eol_soh_pct": eol_pct,
        "replacement_cost_eur_each": float(replacement_cost_eur),
        "replacement_years": [],
    }
    if battery_kwh <= 0.0 or annual_loss <= 0.0 or replacement_cost_eur <= 0.0:
        return treatment

    first_interval = max(1, int(np.ceil((float(initial_soh_pct) - eol_pct) / annual_loss)))
    repeat_interval = max(1, int(np.ceil((100.0 - eol_pct) / annual_loss)))
    replacement_year = first_interval
    while replacement_year <= project_lifespan:
        treatment["replacement_years"].append(replacement_year)
        replacement_year += repeat_interval
    return treatment


def _year_one_soh_loss_pct(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    initial_soh_pct: float,
    has_battery: bool,
) -> float:
    """Read the candidate's year-one SOH loss from the simulator outputs."""
    if not has_battery:
        return 0.0
    if "Final SOH [%]" in summary_df.columns and not summary_df.empty:
        final_soh = float(summary_df["Final SOH [%]"].iloc[0])
    elif "Battery_SOH" in results_df.columns and not results_df.empty:
        valid_soh = pd.to_numeric(results_df["Battery_SOH"], errors="coerce").dropna()
        if valid_soh.empty:
            return 0.0
        final_soh = float(valid_soh.iloc[-1])
    else:
        return 0.0
    return max(0.0, float(initial_soh_pct) - final_soh)


def _pv_params_from_config(params: Dict[str, Any]) -> PVModuleParams:
    """Build PVModuleParams from an inline config mapping."""
    return PVModuleParams(
        Mpp=params.get("Mpp", 550),
        Vmp=params.get("Vmp", 42.05),
        Imp=params.get("Imp", 13.08),
        Voc=params.get("Voc", 49.88),
        Isc=params.get("Isc", 14.01),
        T_Pmax_pct=params.get("T_Pmax_pct", params.get("T_Pmax", -0.34)),
        T_Voc_pct=params.get("T_Voc_pct", params.get("T_Voc", -0.26)),
        T_Isc_pct=params.get("T_Isc_pct", params.get("T_Isc", 0.05)),
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
    """Resolve electrical module parameters and physical module area from config."""
    pv_spec = config.get("pv_specs", {}) or {}
    pv_cfg = config.get("pv", {}) or {}

    pv_spec_params = pv_spec.get("params") or {}
    pv_cfg_params = pv_cfg.get("params") or {}
    pv_spec_dimensions = _dimensions_from_section(pv_spec)
    pv_cfg_dimensions = _dimensions_from_section(pv_cfg)

    if pv_spec_params:
        pv_params = _pv_params_from_config(pv_spec_params)
        dimensions = pv_spec_dimensions or pv_cfg_dimensions
    elif pv_cfg_params:
        pv_params = _pv_params_from_config(pv_cfg_params)
        dimensions = pv_cfg_dimensions or pv_spec_dimensions
    else:
        from breos.pv_modules import get_module

        module_name = pv_cfg.get("module") or config.get("pv_module") or "Suntech_STP550S_STC"
        pv_params = get_module(module_name)
        dimensions = pv_cfg_dimensions or pv_spec_dimensions

    module_area = _module_area_from_dimensions(dimensions)
    return pv_params, module_area


def _temperature_series_from_config(
    temp_config: Any,
    index: pd.DatetimeIndex,
    weather_df: Optional[pd.DataFrame] = None,
    indoor_model: Optional[Dict[str, Any]] = None,
    default_temp: float = 25.0,
) -> pd.Series:
    """Build a battery temperature series from config, weather, or a fixed value."""
    from breos.weather import build_battery_temperature_series

    return build_battery_temperature_series(
        temp_config=temp_config,
        index=index,
        weather_df=weather_df,
        indoor_model=indoor_model,
        default_temp=default_temp,
    )


def _build_battery_config_from_spec(
    batt_spec: Dict[str, Any],
    nominal_energy_wh: float,
    inverter_efficiency: float = 0.96,
    initial_soh: float = 100.0,
    enable_replacement: bool = False,
    inverter_ac_capacity_w: Optional[float] = None,
) -> BatteryConfig:
    """Build a BatteryConfig for optimization paths without dropping supported settings."""
    return BatteryConfig(
        nominal_energy_wh=nominal_energy_wh,
        battery_type=batt_spec.get("battery_type", "lfp"),
        min_soc=batt_spec.get("min_soc", 0.2),
        max_soc=batt_spec.get("max_soc", 0.8),
        charge_efficiency=batt_spec.get("charge_efficiency", 0.9795),
        discharge_efficiency=batt_spec.get("discharge_efficiency", 0.9795),
        standby_loss_wh=batt_spec.get("standby_loss_wh", 5.0),
        initial_soh=initial_soh,
        eol_percentage=batt_spec.get("eol_percentage", 0.7),
        inverter_efficiency=inverter_efficiency,
        inverter_ac_capacity_w=inverter_ac_capacity_w,
        max_charge_power_w=batt_spec.get("max_charge_power_w"),
        max_discharge_power_w=batt_spec.get("max_discharge_power_w"),
        dc_coupled=batt_spec.get("dc_coupled", True),
        calendar_model=batt_spec.get("calendar_model", "naumann_lam_field_calibrated"),
        enable_replacement=enable_replacement,
        enable_resistance_fade=batt_spec.get("enable_resistance_fade", False),
    )


def calculate_financials(
    n_modules: int,
    battery_kwh: float,
    annual_import_kwh: float,
    annual_export_kwh: float,
    annual_load_kwh: float,
    costs_config: Dict[str, float] = None,
    financials_config: Dict[str, float] = None,
    annual_pv_kwh: Optional[float] = None,
    module_power_w: Optional[float] = None,
    annual_battery_soh_loss_pct: float = 0.0,
    battery_initial_soh_pct: float = 100.0,
    battery_eol_percentage: float = 0.70,
) -> Tuple[float, float]:
    """Calculate initial CAPEX and lifetime NPV of savings for a design.

    Mirrors the year-1-estimation formulas of
    :func:`breos.economics.cost_analysis_projection` (maintenance, PV
    degradation, separate import/export price inflation) so the optimizer
    ranks designs with the same economics the App reports;
    ``tests/test_optimization_parity.py`` enforces the equivalence. The fixed
    daily grid fee is charged identically with and without the system, so it
    cancels out of the savings NPV and is omitted here. Battery replacement
    timing is approximated by repeating the candidate's simulated year-1 SOH
    loss until its configured EOL threshold, resetting to 100% SOH after each
    event. The App projection remains authoritative because it propagates SOH
    and applies actual simulated replacement events year by year.

    Module power for inverter sizing and CAPEX comes from ``module_power_w``
    (pass the selected ``pv_params.Mpp``). An explicitly configured
    ``costs.panel_wp`` remains a cost-model override.

    ``annual_pv_kwh`` apportions degradation between lost export and extra
    import via the year-1 self-consumption ratio. When ``None``, year-1
    energy flows are held flat across the lifespan (pre-0.3.4 behaviour).
    """
    if costs_config is None:
        costs_config = {}
    if financials_config is None:
        financials_config = {}

    panel_wp = costs_config.get(
        "panel_wp",
        module_power_w if module_power_w is not None else DEFAULT_PANEL_WP,
    )
    cost_params = cost_params_from_config(costs_config, financials_config)
    electricity_cost = cost_params.electricity_cost
    electricity_sold_cost = cost_params.electricity_sold_cost
    inflation_rate = financials_config.get("inflation_rate", DEFAULT_INFLATION_ELEC)
    sell_price_inflation = cost_params.sell_price_inflation
    discount_rate = financials_config.get("discount_rate", DEFAULT_DISCOUNT_RATE)
    degradation_rate = cost_params.pv_degradation_rate
    project_lifespan = int(financials_config.get("project_lifespan", DEFAULT_PROJECT_LIFESPAN))

    # 1. CAPEX and yearly O&M (same cost model as the App's build_costs_dict)
    costs = calculate_costs(
        n_modules=n_modules,
        module_power_w=panel_wp,
        battery_capacity_wh=battery_kwh * 1000,
        cost_params=cost_params,
    )
    capex = costs["total_initial_cost"]
    annual_operation_cost = costs["annual_operation_cost"]
    replacement_treatment = _estimate_battery_replacement_treatment(
        battery_kwh=battery_kwh,
        annual_soh_loss_pct=annual_battery_soh_loss_pct,
        initial_soh_pct=battery_initial_soh_pct,
        eol_percentage=battery_eol_percentage,
        project_lifespan=project_lifespan,
        replacement_cost_eur=battery_kwh * cost_params.battery_cost_per_kwh,
    )
    replacement_years = set(replacement_treatment["replacement_years"])

    # 2. Degradation apportioning: lost PV splits into lost export and extra
    # import in proportion to the year-1 self-consumption ratio — the same
    # estimation cost_analysis_projection uses for single-year runs.
    if annual_pv_kwh is not None and annual_pv_kwh > 0:
        self_consumption_ratio = 1.0 - (annual_export_kwh / annual_pv_kwh)
    else:
        self_consumption_ratio = None

    # 3. NPV of savings vs the no-system baseline
    npv = -capex
    for year in range(1, project_lifespan + 1):
        inflation = (1 + inflation_rate) ** (year - 1)
        sell_inflation = (1 + sell_price_inflation) ** (year - 1)

        if self_consumption_ratio is not None:
            pv_year = annual_pv_kwh * (1 - degradation_rate) ** (year - 1)
            export_year = pv_year * (1 - self_consumption_ratio)
            import_year = annual_import_kwh + (annual_pv_kwh - pv_year) * self_consumption_ratio
        else:
            export_year = annual_export_kwh
            import_year = annual_import_kwh

        cost_no_system = annual_load_kwh * electricity_cost * inflation
        cost_with_system = (
            import_year * electricity_cost * inflation
            - export_year * electricity_sold_cost * sell_inflation
            + annual_operation_cost * inflation
            + (replacement_treatment["replacement_cost_eur_each"] * inflation if year in replacement_years else 0.0)
        )
        npv += (cost_no_system - cost_with_system) / ((1 + discount_rate) ** year)

    return capex, npv


# ==========================================
# 3. PYMOO OPTIMIZATION CLASSES
# ==========================================

# Only import pymoo if this module is used for full optimization to avoid overhead
try:
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.core.repair import Repair

    class DiscreteGridRepair(Repair):
        def _do(self, problem, pop, **kwargs):
            # 1. Handle Input Type
            try:
                X = pop.get("X")
                is_population = True
            except AttributeError:
                X = pop
                is_population = False

            # --- 2. Apply Rounding Logic ---

            # Col 0: Modules (Round to integer)
            X[:, 0] = np.round(X[:, 0])

            # Col 1: Battery (Round to nearest 1 kWh - Discrete)
            X[:, 1] = np.round(X[:, 1])

            # Col 2: Tilt (Round to nearest 5 degrees)
            tilt_step = 5.0
            X[:, 2] = np.round(X[:, 2] / tilt_step) * tilt_step

            # Col 3: Azimuth (If it exists, round to nearest 5)
            if X.shape[1] > 3:
                azimuth_step = 5.0
                X[:, 3] = np.round(X[:, 3] / azimuth_step) * azimuth_step

            # --- 3. Return Correct Format ---
            if is_population:
                pop.set("X", X)
                return pop
            else:
                return X

    class SolarDesignProblem(ElementwiseProblem):
        def __init__(
            self,
            tmy_data: pd.DataFrame,
            houseload: pd.DataFrame,
            config: Dict[str, Any],
            results_dir: str,
            elementwise_runner=None,
        ):
            self.tmy_data = tmy_data
            self.houseload = houseload
            self.config = config
            self.results_dir = results_dir

            self.location = config["location"]
            # config['location'] is a plain dict; the pvlib Location that
            # calculate_pv_production_dc needs is constructed once here.
            from pvlib.location import Location

            self.loc_obj = Location(
                self.location["latitude"],
                self.location["longitude"],
                tz=self.location.get("timezone", "UTC"),
                altitude=self.location.get("altitude", 0),
                name=self.location.get("name", ""),
            )

            self.constraints = config.get("constraints", {})
            self.budget_limit = self.constraints.get("budget_eur", 10000)
            self.area_limit = self.constraints.get("max_area_m2", 20)
            self.max_battery_kwh = self.constraints.get("max_battery_kwh", 30)
            self.max_modules = self.constraints.get("max_modules", 60)
            self.max_tilt_deg = _resolve_max_tilt_deg(self.constraints, self.location["latitude"])
            self.freq = config.get("simulation", {}).get("resolution", "h")
            self.pv_params, self.module_area_m2 = _resolve_pv_module_and_area(config)
            self.batt_temp_cfg = config.get("battery", {}).get("temperature", "weather")
            self.indoor_model = config.get("battery", {}).get("indoor_model")
            # Inverter AC rating follows the CAPEX sizing convention
            # (economics.calculate_costs): nameplate = DC peak / dc_ac_ratio.
            # The inverter each candidate pays for is also the one that clips
            # its production — same invariant as the App runner.
            self.dc_ac_ratio = cost_params_from_config(config.get("costs"), config.get("financials")).dc_ac_ratio
            self.inverter_efficiency = config.get(
                "inverter_efficiency",
                config.get("inverter", {}).get("efficiency", 0.96),
            )

            self.battery_replacement_treatment = {
                "method": "repeat_simulated_year_1_soh_loss_to_eol",
                "description": (
                    "Steady-state candidate scoring repeats its simulated year-1 SOH loss, "
                    "replaces at the configured EOL threshold, resets SOH to 100%, and applies "
                    "the configured storage cost in each estimated replacement year."
                ),
                "higher_fidelity_basis": "App multiyear SOH propagation",
            }

            self.fixed_azimuth = config.get("mode", {}).get("fixed_azimuth")

            # Simulation range (derived from TMY data)
            self.start_h = self.tmy_data.index[0]
            self.end_h = self.tmy_data.index[-1]

            # --- Dynamic Variable Setup ---
            if self.fixed_azimuth is not None:
                # RETROFIT MODE: 3 Variables
                # x[0]: n_modules (1-max_modules)
                # x[1]: battery_kwh (0-max_battery_kwh)
                # x[2]: surface_tilt (10-max_tilt_deg)
                n_var = 3
                xl = np.array([1, 0.0, 10.0])
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
                xl = np.array([1, 0.0, 10.0, azi_lower])
                xu = np.array([self.max_modules, self.max_battery_kwh, self.max_tilt_deg, azi_upper])

            super().__init__(
                n_var=n_var,
                n_obj=3,  # Obj1: Grid Indep, Obj2: ROI (NPV), Obj3: ZEB Ratio
                n_ieq_constr=2,  # Constr1: Budget, Constr2: Area
                xl=xl,
                xu=xu,
                elementwise_runner=elementwise_runner or _serial_elementwise_runner,
            )

        def __getstate__(self):
            # Exclude elementwise_runner from pickling as it contains the Pool object
            state = self.__dict__.copy()
            state["elementwise_runner"] = None
            return state

        def __setstate__(self, state):
            self.__dict__.update(state)

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
                verbose=False,
            )

            # Load alignment (timezone- and DST-aware year remapping) happens
            # inside simulate_energy_balance — the same code path the App
            # uses. Positionally re-stamping the load onto the PV index here
            # (the pre-0.3.4 align_load_to_pv call) discarded the load's real
            # timestamps and could shift it against PV by the UTC offset.
            if isinstance(self.houseload, pd.Series):
                houseload_df = self.houseload.to_frame(name="Load")
            else:
                houseload_df = self.houseload

            hours_per_step = get_hours_per_step(self.freq)
            input_load_kwh = float(houseload_df.iloc[:, 0].sum() * hours_per_step / 1000)

            batt_spec = self.config.get("battery", {})

            # Inverter AC nameplate shared by PV export and battery discharge
            pv_peak_w = n_modules * pv_params.Mpp
            inverter_ac_capacity_w = pv_peak_w / self.dc_ac_ratio if self.dc_ac_ratio > 0 else None

            # Configure Battery
            battery_config = _build_battery_config_from_spec(
                batt_spec,
                nominal_energy_wh=battery_kwh * 1000,
                inverter_efficiency=self.inverter_efficiency,
                initial_soh=batt_spec.get("initial_soh", 100),
                enable_replacement=False,
                inverter_ac_capacity_w=inverter_ac_capacity_w,
            )
            temperature_series = _temperature_series_from_config(
                self.batt_temp_cfg,
                dc_production.index,
                weather_df=self.tmy_data,
                indoor_model=self.indoor_model,
            )

            # Run Simulation
            results_df, total_pv_wh, summary_df, _, _, _ = simulate_energy_balance(
                pv_dc=dc_production,
                houseload=houseload_df,
                battery_config=battery_config,
                start_time=self.start_h,
                end_time=self.end_h,
                freq=self.freq,
                temperature_series=temperature_series,
                debug=False,
            )
            total_import = float(summary_df["Import [kWh]"].iloc[0])
            total_export = float(summary_df["Sell [kWh]"].iloc[0])
            if "Houseload" in results_df.columns and not results_df.empty:
                total_load = float(
                    pd.to_numeric(results_df["Houseload"], errors="coerce").fillna(0.0).sum() * hours_per_step / 1000
                )
            elif "Total Load [kWh]" in summary_df.columns and not summary_df.empty:
                total_load = float(summary_df["Total Load [kWh]"].iloc[0])
            else:
                # Compatibility for custom/legacy simulation adapters that
                # expose neither the aligned load ledger nor its aggregate.
                total_load = input_load_kwh
            annual_soh_loss_pct = _year_one_soh_loss_pct(
                results_df,
                summary_df,
                initial_soh_pct=battery_config.initial_soh,
                has_battery=battery_kwh > 0,
            )
            try:
                total_ac_prod = float(system_ac_production_power(results_df).sum() * hours_per_step / 1000)
            except KeyError:
                # Compatibility with older/custom adapters that expose only
                # the established aggregate return value (Wh).
                total_ac_prod = float(total_pv_wh / 1000)

            # --- 3. Objective Calculations ---

            # Obj 1: Grid Independence
            grid_dependence_ratio = total_import / total_load if total_load > 0 else 1.0

            # Obj 2: ROI (NPV)
            capex, npv = calculate_financials(
                n_modules,
                battery_kwh,
                total_import,
                total_export,
                total_load,
                costs_config=self.config.get("costs"),
                financials_config=self.config.get("financials"),
                annual_pv_kwh=total_ac_prod,
                module_power_w=pv_params.Mpp,
                annual_battery_soh_loss_pct=annual_soh_loss_pct,
                battery_initial_soh_pct=battery_config.initial_soh,
                battery_eol_percentage=battery_config.eol_percentage,
            )

            # Obj 3: ZEB Status (Maximize Ratio -> Minimize Negative)
            zeb_ratio = total_ac_prod / total_load if total_load > 0 else 0

            # --- 4. Constraints Calculation ---
            # g1: Price <= Budget (g1 <= 0 means satisfied)
            g1 = capex - self.budget_limit

            # g2: Area <= Max Area
            g2 = system_area - self.area_limit

            # Return
            out["F"] = [grid_dependence_ratio, -npv, -zeb_ratio]
            out["G"] = [g1, g2]

except ImportError:
    # If pymoo is not installed, these classes won't be available
    pass


def optimize_system_multi_objective(
    tmy_data: pd.DataFrame,
    houseload: pd.DataFrame,
    config: Dict[str, Any],
    results_dir: str = "results/optimization",
    pop_size: int = 40,
    n_gen: int = 100,
    n_offsprings: int | None = None,
    seed: int = 1,
    verbose: bool = False,
) -> OptimizationResult:
    """Run NSGA-II multi-objective PV/battery sizing.

    This is the public wrapper around :class:`SolarDesignProblem`. It optimizes
    module count, battery capacity, tilt, and optionally azimuth. Objectives are
    grid independence, NPV, and ZEB ratio. Install ``breos[optimization]`` to
    provide the pymoo dependency.

    Args:
        tmy_data: One-year weather DataFrame.
        houseload: One-year load profile.
        config: Optimization config using the nested keys consumed by
            :class:`SolarDesignProblem` (``location``, ``constraints``,
            ``simulation``, ``pv``, ``battery``, ``costs``, ``financials``).
        results_dir: Directory label retained in the problem object.
        pop_size: NSGA-II population size.
        n_gen: Number of generations.
        n_offsprings: Offspring count per generation. Defaults to pymoo's
            algorithm default when ``None``.
        seed: Random seed passed to pymoo.
        verbose: Print pymoo progress.

    Returns:
        :class:`OptimizationResult` whose ``details["pareto"]`` is a DataFrame
        with ``Modules``, ``Battery_kWh``, ``Tilt``, ``Azimuth``,
        ``Grid_Independence_%``, ``NPV_Eur``, and ``ZEB_Ratio``.

    Raises:
        ImportError: If pymoo is not installed.
        RuntimeError: If the optimizer returns no feasible solution.
    """
    if "SolarDesignProblem" not in globals() or "DiscreteGridRepair" not in globals():
        raise ImportError(
            "pymoo is required for optimize_system_multi_objective(). Install with: pip install 'breos[optimization]'"
        )

    try:
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.operators.sampling.rnd import FloatRandomSampling
        from pymoo.optimize import minimize
    except ImportError as exc:
        raise ImportError(
            "pymoo is required for optimize_system_multi_objective(). Install with: pip install 'breos[optimization]'"
        ) from exc

    algorithm_kwargs: dict[str, Any] = {
        "pop_size": pop_size,
        "sampling": FloatRandomSampling(),
        "crossover": SBX(prob=0.9, eta=15),
        "mutation": PM(eta=20),
        "repair": DiscreteGridRepair(),
        "eliminate_duplicates": True,
    }
    if n_offsprings is not None:
        algorithm_kwargs["n_offsprings"] = n_offsprings

    problem = SolarDesignProblem(tmy_data, houseload, config, results_dir)
    result = minimize(
        problem,
        NSGA2(**algorithm_kwargs),
        ("n_gen", n_gen),
        seed=seed,
        verbose=verbose,
    )

    if result.X is None or result.F is None:
        raise RuntimeError("Multi-objective optimization found no feasible solutions.")

    x = np.atleast_2d(result.X)
    f = np.atleast_2d(result.F)
    fixed_azimuth = (config.get("mode") or {}).get("fixed_azimuth")
    if fixed_azimuth is not None:
        pareto = pd.DataFrame(x, columns=["Modules", "Battery_kWh", "Tilt"])
        pareto["Azimuth"] = fixed_azimuth
    else:
        pareto = pd.DataFrame(x, columns=["Modules", "Battery_kWh", "Tilt", "Azimuth"])

    pareto["Modules"] = pareto["Modules"].round().astype(int)
    pareto["Battery_kWh"] = pareto["Battery_kWh"].round().astype(float)
    pareto["Grid_Independence_%"] = (1 - f[:, 0]) * 100
    pareto["NPV_Eur"] = -f[:, 1]
    pareto["ZEB_Ratio"] = -f[:, 2]

    actual_generations = int(getattr(result.algorithm, "n_gen", n_gen))
    return OptimizationResult(
        optimal_value=float("nan"),
        objective_value=float("nan"),
        iterations=actual_generations,
        details={
            "pareto": pareto,
            "pymoo_result": result,
            "problem": problem,
            "battery_replacement_treatment": problem.battery_replacement_treatment,
        },
    )
