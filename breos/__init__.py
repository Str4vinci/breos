"""
BREOS - Building Renewable Energy Optimization Software

A modular Python library for photovoltaic and battery energy system simulations.
Supports both hourly ('h') and 15-minute ('15min') time resolutions.

Modules:
--------
- weather: Weather data fetching and interpolation
- solar: PV production calculations
- load_profiles: Load profile management
- battery: Energy balance and degradation simulation
- economics: Cost analysis and projections
- optimization: System sizing and slope optimization
- plotting: Visualization utilities
- emissions: CO2 savings calculations

Usage:
------
>>> from breos.weather import fetch_tmy_weather_data
>>> from breos.solar import calculate_pv_production, PVModuleParams
>>> from breos.battery import simulate_energy_balance, BatteryConfig
>>> from breos.load_profiles import load_profile
>>> from breos.economics import calculate_costs, cost_analysis_projection
"""

# Version
__version__ = "0.2.0"

# Public facade
from breos.app import App

# Utils
from breos.utils import (
    is_leap_year,
    count_leap_years,
    number_of_cores,
    get_hours_per_step,
    get_steps_per_day,
    get_steps_per_year,
)

# Constants
from breos.constants import (
    R_GAS,
    T_REF_K,
    NAUMANN_K0_PERCENT,
    NAUMANN_EA_J_MOL,
    NAUMANN_EXPONENT_B,
    NAUMANN_SOC_EXPONENT_N,
    LAM_K0_FRAC,
    LAM_EA_J_MOL,
    LAM_EXPONENT_B,
    LAM_SOC_EXPONENT_N,
    A_Q, B_Q, C_DOC_Q, D_DOC_Q, Z_Q,
    A_R, B_R, C_DOC_R, D_DOC_R, Z_R,
    NAUMANN_K0_R_PERCENT, NAUMANN_EA_R_J_MOL,
    DEFAULT_CHARGE_EFFICIENCY,
    DEFAULT_DISCHARGE_EFFICIENCY,
    DEFAULT_MAX_SOC,
    DEFAULT_MIN_SOC,
    DEFAULT_INDOOR_SETPOINT_C,
    DEFAULT_INDOOR_COUPLING_ALPHA,
    DEFAULT_INDOOR_FLOOR_C,
    DEFAULT_INDOOR_CEILING_C,
)

# Weather
from breos.weather import (
    parse_weather_filename,
    load_weather,
    fetch_tmy_weather_data,
    fetch_tmy_nsrdb,
    fetch_weather_data,
    read_epw_file,
    resample_tmy_to_15min,
    resample_to_15min,
    resample_to_hourly,
    csv_15min_to_hourly,
    csv_hourly_to_15min,
    select_random_year_and_replace_datetime,
    preload_weather_by_year,
    extract_ambient_temperature,
)

# Solar
from breos.solar import (
    calculate_pv_production_dc,
    calculate_pv_production_ac,
    calculate_pv_production_tmy,
    dc_to_ac,
    PVModuleParams,
    estimate_optimal_tilt,
    default_azimuth,
    zeb_sizer,
)

# PV Module Database
from breos.pv_modules import (
    get_module,
    list_modules,
    get_module_info,
    MODULES,
)

# Inverter
from breos.inverter import (
    InverterConfig,
    get_inverter_preset,
    INVERTER_PRESETS,
)

# Load Profiles
from breos.load_profiles import (
    load_profile,
    scale_to_annual_consumption,
    align_load_to_pv,
)

# Battery
from breos.battery import (
    simulate_energy_balance,
    BatteryConfig,
    detect_half_cycles_from_soc_series,
    detect_cycles_rainflow,
    compute_halfcycle_energy_throughput,
    update_battery_soh_cyclewise,
    update_battery_soh_calendar,
    update_battery_soc,
    k_c_rate_Q,
    k_doc_Q,
    k_c_rate_R,
    k_doc_R,
    update_battery_resistance_cyclewise,
    update_battery_resistance_calendar,
    resistance_to_efficiency,
    compute_cell_temperature,
    apply_indoor_temperature_model,
)

# Emissions
from breos.emissions import (
    EmissionsParams,
    calculate_co2_savings,
    calculate_co2_projection,
)

# Economics
from breos.economics import (
    calculate_costs,
    cost_analysis_projection,
    find_payback_year,
    calculate_lcoe,
    CostParams,
)

# Optimization
from breos.optimization import (
    optimize_slope,
    optimize_slope_brent,
    optimize_battery_size,
    size_for_zeb,
    OptimizationResult,
)

# I/O (export/import functions)
from breos.io import (
    export_results,
    export_cost_analysis,
    export_summary,
    save_simulation_report,
    load_results,
    export_monthly_summary,
    export_yearly_summary,
)

# Polysun Degradation (comparison baseline)
from breos.polysun_degradation import (
    PolysunDegradationConfig,
    woehler_cycles_to_failure,
    compute_dod_histogram,
    compute_miner_damage,
    predict_polysun_lifetime,
    simulate_polysun_degradation,
)

# Plotting (try to import, skip if matplotlib not installed)
try:
    from breos.plotting import (
        create_cost_plots,
        monthly_graphs,
        yearly_graphs,
        weekly_graphs,
        degradation_plots,
        plot_timeseries,
        plot_breakeven,
        plot_battery_soh_timeseries,
        plot_slope_optimization,
        plot_monthly_comparison,
        plot_monthly_balance,
        set_presentation_mode,
        plot_weather_monthly_comparison,
        plot_weather_annual_ghi_distribution,
        plot_breakeven_comparison,
        plot_breakeven_two,
        plot_azislope_landscape_2d,
        plot_azislope_landscape_3d,
        plot_azislope_ew_1d,
        plot_pareto_front_analysis,
        plot_resistance_and_efficiency,
        plot_cell_temperature,
        plot_validation_soh_comparison,
        plot_validation_residuals,
        plot_validation_parity,
        plot_validation_multi_system,
        plot_validation_degradation_split,
        plot_loo_cv_summary,
        plot_loo_param_stability,
        plot_loo_predictions,
        # Monte Carlo distributions
        plot_montecarlo_npv_distribution,
        plot_montecarlo_grid_independence_distribution,
        # Sensitivity analysis
        plot_calendar_aging_sensitivity,
        # Batch comparison
        plot_grid_independence_heatmap,
        plot_location_comparison_delta,
        # CO2 savings
        plot_co2_savings,
        # Polysun vs BREOS comparison
        plot_degradation_methodology_comparison,
        plot_lifetime_prediction_comparison,
        plot_temperature_sensitivity_comparison,
    )
except ImportError:
    pass  # matplotlib not installed


# Numba kernels (lazy import to avoid slow import at package load)
def _get_numba():
    """Lazy import of Numba-accelerated kernels."""
    from breos import numba_kernels
    return numba_kernels


__all__ = [
    # Public facade
    "App",
    # Version
    "__version__",
    # Utils
    "is_leap_year",
    "count_leap_years",
    "number_of_cores",
    "get_hours_per_step",
    "get_steps_per_day",
    "get_steps_per_year",
    # Constants
    "R_GAS",
    "T_REF_K",
    "NAUMANN_K0_PERCENT",
    "NAUMANN_EA_J_MOL",
    "LAM_K0_FRAC",
    "LAM_EA_J_MOL",
    "DEFAULT_INDOOR_SETPOINT_C",
    "DEFAULT_INDOOR_COUPLING_ALPHA",
    "DEFAULT_INDOOR_FLOOR_C",
    "DEFAULT_INDOOR_CEILING_C",
    # Weather
    "parse_weather_filename",
    "load_weather",
    "fetch_tmy_weather_data",
    "fetch_tmy_nsrdb",
    "fetch_weather_data",
    "read_epw_file",
    "resample_tmy_to_15min",
    "resample_to_15min",
    "resample_to_hourly",
    "csv_15min_to_hourly",
    "csv_hourly_to_15min",
    "select_random_year_and_replace_datetime",
    "extract_ambient_temperature",
    # Solar
    "calculate_pv_production_dc",
    "calculate_pv_production_ac",
    "calculate_pv_production_tmy",
    "dc_to_ac",
    "PVModuleParams",
    "estimate_optimal_tilt",
    "default_azimuth",
    "zeb_sizer",
    # Load Profiles
    "load_profile",
    "scale_to_annual_consumption",
    "align_load_to_pv",
    # Battery
    "simulate_energy_balance",
    "BatteryConfig",
    "update_battery_soh_cyclewise",
    "update_battery_soh_calendar",
    "update_battery_soc",
    "apply_indoor_temperature_model",
    # Emissions
    "EmissionsParams",
    "calculate_co2_savings",
    "calculate_co2_projection",
    # Economics
    "calculate_costs",
    "cost_analysis_projection",
    "find_payback_year",
    "calculate_lcoe",
    "CostParams",
    # Optimization
    "optimize_slope",
    "optimize_slope_brent",
    "optimize_battery_size",
    "size_for_zeb",
    "OptimizationResult",
    # I/O
    "export_results",
    "export_cost_analysis",
    "export_summary",
    "save_simulation_report",
    "load_results",
    "export_monthly_summary",
    "export_yearly_summary",
    # Polysun Degradation
    "PolysunDegradationConfig",
    "woehler_cycles_to_failure",
    "compute_dod_histogram",
    "compute_miner_damage",
    "predict_polysun_lifetime",
    "simulate_polysun_degradation",
    # Plotting
    "plot_calendar_aging_sensitivity",
    "plot_grid_independence_heatmap",
    "plot_location_comparison_delta",
    "plot_co2_savings",
    "plot_degradation_methodology_comparison",
    "plot_lifetime_prediction_comparison",
    "plot_temperature_sensitivity_comparison",
]
