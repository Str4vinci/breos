# BREOS - Building Renewable Energy Optimization Software

A modular Python library for photovoltaic (PV) and battery energy system simulations, designed for research and engineering applications.

## Features

- **Weather data**: Fetch TMY data from PVGIS/NSRDB and historical data from Open-Meteo. Support for hourly and 15-minute resolutions with Makima interpolation.
- **PV production**: DC and AC power calculations using pvlib, with built-in module database and inverter presets.
- **Battery simulation**: Energy balance with Numba-accelerated kernels. Calendar and cycle aging models (Naumann 2020, Lam 2025) with field-calibrated LFP parameters. Support for LFP, SIB, VRFB, and solid-state chemistries.
- **Economics**: NPV, LCOE, breakeven analysis, and cost projections with configurable tariffs and inflation.
- **Optimization**: Multi-objective (grid independence, NPV, ZEB ratio) system sizing using pymoo (NSGA-II). Tilt/azimuth optimization via grid search or Brent's method.
- **Emissions**: CO2 savings calculations and projections.
- **Visualization**: Publication-ready plots for energy balances, degradation, breakeven, Pareto fronts, and more.
- **Load profiles**: Support for standard load profiles (BDEW H0, EREDES, REE) and custom profiles.

## Additional Capabilities

BREOS is the open-source core of a broader simulation platform developed as part of PhD research. Additional features not included in this release:

- **Time-of-Use (TOU) tariff optimization** with multi-period pricing and strategy comparison
- **Vehicle-to-Home (V2H)** simulation with EV scheduling and bidirectional charging
- **Multi-chemistry battery support** — Sodium-ion (SIB), Vanadium Redox Flow (VRFB), Solid-State (SSB)
- **Thermal energy storage (TES)** with phase-change material modeling
- **Heat pump integration** with COP modeling and coupled electro-thermal energy balance
- **Community Self-Consumption (CSC)** modeling for multi-building scenarios

These modules may be released in the future or are available for academic collaboration upon request.

## Installation

```bash
pip install breos
```

Or from source:

```bash
git clone https://github.com/Str4vinci/BREOS.git
cd breos
pip install -e .
```

## Quick Start

```python
from breos import (
    fetch_tmy_weather_data,
    calculate_pv_production_dc,
    simulate_energy_balance,
    BatteryConfig,
    PVModuleParams,
    calculate_costs,
)
from pvlib.location import Location

# Define location
location = Location(41.15, -8.61, tz='Europe/Lisbon', altitude=104, name='Porto')

# Fetch TMY weather data
weather = fetch_tmy_weather_data(location.latitude, location.longitude)

# Calculate PV production
pv_dc = calculate_pv_production_dc(
    weather_data=weather,
    location=location,
    slope=35,
    surface_azimuth=180,
    n_modules=10,
)

# Simulate with battery
battery = BatteryConfig(nominal_energy_wh=10000)  # 10 kWh
results_df, total_pv, summary, rep_cost, n_rep, deg_df = simulate_energy_balance(
    pv_dc=pv_dc,
    houseload=load_profile,  # your load data
    battery_config=battery,
)
```

## Weather Data Note

BREOS uses [Open-Meteo](https://open-meteo.com/) for historical weather data. Open-Meteo is free for non-commercial use. For commercial applications, please review their [pricing and terms](https://open-meteo.com/en/pricing).

## Citation

If you use BREOS in your research, please cite:

```bibtex
@software{breos,
  author = {Rodrigues, Leonardo},
  title = {BREOS: Building Renewable Energy Optimization Software},
  year = {2026},
  url = {https://github.com/Str4vinci/BREOS}
}
```

## License

## Contact

For questions, collaboration, or access to additional modules, reach out at lrodrigues@fe.up.pt.

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.
