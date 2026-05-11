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
from breos.economics import calculate_costs, cost_params_from_config
from breos.load_profiles import align_load_to_pv
from breos.solar import PVModuleParams, calculate_pv_production_dc, default_azimuth


@dataclass
class OptimizationResult:
    """Result from an optimization run."""

    optimal_value: float
    objective_value: float
    iterations: int
    details: Dict[str, Any]


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
            ac_power = calculate_pv_production_dc(
                weather_data=weather_data,
                location=location,
                tilt=tilt,
                surface_azimuth=surface_azimuth,
                n_modules=n_modules,
                pv_params=pv_params,
                freq=freq,
                verbose=False,
            )
            total_production = ac_power.sum() / 1000  # kWh
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
            ac_power = calculate_pv_production_dc(
                weather_data=weather_data,
                location=location,
                tilt=tilt,
                surface_azimuth=surface_azimuth,
                n_modules=n_modules,
                pv_params=pv_params,
                freq=freq,
                verbose=False,
            )
            production = -ac_power.sum() / 1000  # Negative for minimization

            if verbose:
                print(f"  Iteration {iterations[0]}: tilt={tilt:.2f}°, production={-production:.1f} kWh")

            return production
        except Exception:
            return 0

    result = minimize_scalar(objective, bounds=tilt_range, method="bounded", options={"xatol": tol})

    return OptimizationResult(
        optimal_value=result.x, objective_value=-result.fun, iterations=iterations[0], details={"scipy_result": result}
    )


def optimize_battery_size(
    ac_loss: pd.Series,
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
        ac_loss: PV production series
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
                ac_loss=ac_loss,
                houseload=houseload,
                battery_config=config,
                start_time=start_time,
                end_time=end_time,
                freq=freq,
                debug=False,
            )

            grid_independence = summary["Grid Independence [%]"].iloc[0]
            import_pct = summary["Import [%]"].iloc[0]

            results.append(
                {
                    "battery_size_wh": size_wh,
                    "battery_size_kwh": size_wh / 1000,
                    "grid_independence": grid_independence,
                    "import_percent": import_pct,
                }
            )

            if verbose:
                print(f"  {size_wh / 1000:.1f} kWh: {grid_independence:.1f}% grid independence")

        except Exception as e:
            if verbose:
                print(f"  {size_wh / 1000:.1f} kWh: Error - {e}")

    results_df = pd.DataFrame(results)

    if objective == "max_self_consumption" or objective == "max_grid_independence":
        optimal_idx = results_df["grid_independence"].idxmax()
    else:  # min_import
        optimal_idx = results_df["import_percent"].idxmin()

    optimal_size = results_df.loc[optimal_idx, "battery_size_wh"]
    optimal_value = results_df.loc[optimal_idx, "grid_independence"]

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
        ac_loss: PV production for current system
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
DEFAULT_INFLATION_ELEC = 0.02
DEFAULT_DISCOUNT_RATE = 0.0

DEFAULT_PROJECT_LIFESPAN = 20


def calculate_financials(
    n_modules: int,
    battery_kwh: float,
    annual_import_kwh: float,
    annual_export_kwh: float,
    annual_load_kwh: float,
    costs_config: Dict[str, float] = None,
    financials_config: Dict[str, float] = None,
) -> Tuple[float, float]:
    """
    Calculates Net Present Value (ROI metric) and Initial CAPEX.
    """
    if costs_config is None:
        costs_config = {}
    if financials_config is None:
        financials_config = {}

    panel_wp = costs_config.get("panel_wp", DEFAULT_PANEL_WP)
    cost_params = cost_params_from_config(costs_config, financials_config)
    electricity_cost = cost_params.electricity_cost
    electricity_sold_cost = cost_params.electricity_sold_cost
    inflation_rate = financials_config.get("inflation_rate", DEFAULT_INFLATION_ELEC)
    discount_rate = financials_config.get("discount_rate", DEFAULT_DISCOUNT_RATE)
    project_lifespan = int(financials_config.get("project_lifespan", DEFAULT_PROJECT_LIFESPAN))

    # 1. Calculate CAPEX (Initial Cost)
    costs = calculate_costs(
        n_modules=n_modules,
        module_power_w=panel_wp,
        battery_capacity_wh=battery_kwh * 1000,
        cost_params=cost_params,
    )
    capex = costs["total_initial_cost"]

    # 2. Calculate Annual Savings
    # Baseline cost (if no solar existed)
    cost_no_solar = annual_load_kwh * electricity_cost

    # New cost (Imported energy + Earnings from Export)
    cost_with_solar = (annual_import_kwh * electricity_cost) - (annual_export_kwh * electricity_sold_cost)

    annual_savings = cost_no_solar - cost_with_solar

    # 3. Calculate NPV (Net Present Value)
    npv = -capex
    for year in range(1, project_lifespan + 1):
        # Escalate savings with energy inflation
        savings_y = annual_savings * ((1 + inflation_rate) ** (year - 1))
        # Discount back to present value
        npv += savings_y / ((1 + discount_rate) ** year)

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
            # Ideally passed as pvlib Location, but if dict, construct it?
            # For now assume caller passes data and logic handles location construction if needed
            # Actually calculate_pv_production needs Location object.
            # We'll construct it in _evaluate or pass it in.
            # Better to pass it in or construct once.
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

            self.fixed_azimuth = config.get("mode", {}).get("fixed_azimuth")

            # Simulation range (derived from TMY data)
            self.start_h = self.tmy_data.index[0]
            self.end_h = self.tmy_data.index[-1]

            # --- Dynamic Variable Setup ---
            if self.fixed_azimuth is not None:
                # RETROFIT MODE: 3 Variables
                # x[0]: n_modules (1-60)
                # x[1]: battery_kwh (0-max_battery_kwh)
                # x[2]: surface_tilt (10-60)
                n_var = 3
                xl = np.array([1, 0.0, 10.0])
                xu = np.array([60, self.max_battery_kwh, 60.0])
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
                xu = np.array([60, self.max_battery_kwh, 60.0, azi_upper])

            super().__init__(
                n_var=n_var,
                n_obj=3,  # Obj1: Grid Indep, Obj2: ROI (NPV), Obj3: ZEB Ratio
                n_ieq_constr=2,  # Constr1: Budget, Constr2: Area
                xl=xl,
                xu=xu,
                elementwise_runner=elementwise_runner,
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

            from breos.pv_modules import get_module

            pv_cfg = self.config.get("pv", {})
            module_name = pv_cfg.get("module", "Suntech_STP550S_STC")
            pv_params = get_module(module_name)
            module_area = pv_cfg.get("module_width_m", 1.134) * pv_cfg.get("module_length_m", 2.278)

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
                freq="h",
                verbose=False,
            )

            # Align load to PV index
            if isinstance(self.houseload, pd.Series):
                temp_df = pd.DataFrame({"Load": self.houseload})
                aligned_houseload = align_load_to_pv(temp_df, dc_production)
            else:
                aligned_houseload = align_load_to_pv(self.houseload, dc_production)

            total_prod = float(dc_production.sum() / 1000)

            if isinstance(aligned_houseload, pd.Series):
                total_load = float(aligned_houseload.sum() / 1000)
            else:
                total_load = float(aligned_houseload.iloc[:, 0].sum() / 1000)

            batt_spec = self.config.get("battery", {})

            # Configure Battery
            battery_config = BatteryConfig(
                nominal_energy_wh=battery_kwh * 1000,
                min_soc=batt_spec.get("min_soc", 0.2),
                max_soc=batt_spec.get("max_soc", 0.8),
                charge_efficiency=batt_spec.get("charge_efficiency", 0.9795),
                discharge_efficiency=batt_spec.get("discharge_efficiency", 0.9795),
                initial_soh=batt_spec.get("initial_soh", 100),
                enable_replacement=False,  # Single year optimization does not simulate replacement cycles
            )

            # Run Simulation
            results_df, total_pv_wh, summary_df, _, _, _ = simulate_energy_balance(
                pv_dc=dc_production,
                houseload=aligned_houseload,
                battery_config=battery_config,
                start_time=self.start_h,
                end_time=self.end_h,
                freq="h",
                debug=False,
            )
            total_import = float(summary_df["Import [kWh]"].iloc[0])
            total_export = float(summary_df["Sell [kWh]"].iloc[0])

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
            )

            # Obj 3: ZEB Status (Maximize Ratio -> Minimize Negative)
            zeb_ratio = total_prod / total_load if total_load > 0 else 0

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
