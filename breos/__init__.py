"""
BREOS - Building Renewable Energy Optimization Software

Python library for PV and battery energy-system simulation and optimization.
Supports both hourly ('h') and 15-minute ('15min') time resolutions.

Modules:
--------
- weather: Weather data fetching and interpolation
- solar: PV production calculations
- load_profiles: Load profile management
- battery: Energy balance and degradation simulation
- economics: Cost analysis and projections
- optimization: System sizing and tilt optimization
- plotting: Visualization utilities
- emissions: CO2 savings calculations

Usage:
------
>>> import breos
>>> app = breos.App({"location": "porto", "n_modules": 10, "annual_consumption_kwh": 4000})
>>> app.simulate()
>>> result = app.result()
"""

# Version — resolved from the installed package metadata so it always matches
# the version declared in pyproject.toml (the single source of truth). The CLI
# and result provenance read the same helper, and docs/conf.py the same
# metadata, which keeps the literal from drifting out of sync with the
# distribution version on a release.
from breos.utils import package_version as _package_version

__version__ = _package_version()

# Public facade
from breos.app import App

# Battery
from breos.battery import (
    BatteryConfig,
    apply_indoor_temperature_model,
    simulate_energy_balance,
)

# Battery degradation model discovery
from breos.degradation import (
    BATTERY_MODEL_REGISTRY,
    BatteryModelProfile,
    get_battery_model_profile,
    list_battery_models,
)

# Economics
from breos.economics import (
    CostParams,
    calculate_costs,
    calculate_lcoe_from_projection,
    cost_analysis_projection,
    cost_params_from_config,
    find_payback_year,
)

# Emissions
from breos.emissions import (
    EmissionsParams,
    calculate_co2_projection,
)

# Inverter
from breos.inverter import (
    InverterConversionResult,
    calculate_dc_ac_power,
    dc_power_for_ac_output,
)

# I/O (export/import functions)
from breos.io import (
    InputRepairReport,
    export_results,
    export_summary,
    load_results,
    repair_series,
)

# Load Profiles
from breos.load_profiles import (
    load_profile,
    scale_to_annual_consumption,
)

# Monte Carlo (weather + demand uncertainty)
from breos.montecarlo import (
    MonteCarloResult,
    MonteCarloSettings,
    MonteCarloYearCache,
    build_year_cache,
    run_montecarlo,
)

# Optimization
from breos.optimization import (
    OptimizationResult,
    ProjectedDesignResult,
    evaluate_projected_design,
    optimize_system_multi_objective,
)

# PV Module Database
from breos.pv_modules import (
    get_module,
    get_module_info,
    list_modules,
)

# Solar
from breos.solar import (
    PVModuleParams,
    calculate_multi_array_production,
    calculate_pv_production_ac,
    calculate_pv_production_dc,
    calculate_pv_production_dc_tracking,
    dc_to_ac,
    default_azimuth,
    estimate_optimal_tilt,
)

# Weather
from breos.weather import (
    fetch_tmy_weather_data,
    fetch_weather_data,
    load_weather,
    read_epw_file,
)

__all__ = [
    # Public facade
    "App",
    # Version
    "__version__",
    # Configuration and result objects
    "BatteryConfig",
    "CostParams",
    "EmissionsParams",
    "InverterConversionResult",
    "OptimizationResult",
    "PVModuleParams",
    # Weather
    "load_weather",
    "fetch_tmy_weather_data",
    "fetch_weather_data",
    "read_epw_file",
    # Solar
    "calculate_pv_production_dc",
    "calculate_pv_production_dc_tracking",
    "calculate_multi_array_production",
    "calculate_pv_production_ac",
    "dc_to_ac",
    "estimate_optimal_tilt",
    "default_azimuth",
    # PV module catalogue
    "get_module",
    "get_module_info",
    "list_modules",
    # Load Profiles
    "load_profile",
    "scale_to_annual_consumption",
    # Inverter
    "calculate_dc_ac_power",
    "dc_power_for_ac_output",
    # Battery
    "simulate_energy_balance",
    "apply_indoor_temperature_model",
    "list_battery_models",
    "get_battery_model_profile",
    "BatteryModelProfile",
    "BATTERY_MODEL_REGISTRY",
    # Emissions
    "calculate_co2_projection",
    # Economics
    "calculate_costs",
    "cost_analysis_projection",
    "cost_params_from_config",
    "find_payback_year",
    "calculate_lcoe_from_projection",
    # Optimization
    "optimize_system_multi_objective",
    "evaluate_projected_design",
    "ProjectedDesignResult",
    # Monte Carlo
    "run_montecarlo",
    "build_year_cache",
    "MonteCarloSettings",
    "MonteCarloResult",
    "MonteCarloYearCache",
    # I/O
    "export_results",
    "export_summary",
    "load_results",
    # Input repair
    "repair_series",
    "InputRepairReport",
]
