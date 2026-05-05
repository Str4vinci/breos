# BREOS - Building Renewable Energy Optimization Software

[![Tests](https://github.com/Str4vinci/BREOS/actions/workflows/tests.yml/badge.svg)](https://github.com/Str4vinci/BREOS/actions/workflows/tests.yml)
[![License: BSD-3](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)

A modular Python library for photovoltaic (PV) and battery energy system simulations, designed for research and engineering applications.

## Features

- **Weather data**: Fetch TMY data from PVGIS/NSRDB and historical data from Open-Meteo. Support for hourly and 15-minute resolutions with Makima interpolation.
- **PV production**: DC and AC power calculations using pvlib, with built-in module database and inverter presets.
- **Battery simulation**: Energy balance with Numba-accelerated kernels. Calendar and cycle aging models (Naumann 2020, Lam 2025) with field-calibrated LFP parameters.
- **Economics**: NPV, LCOE, breakeven analysis, and cost projections with configurable tariffs and inflation.
- **Optimization**: Multi-objective (grid independence, NPV, ZEB ratio) system sizing using pymoo (NSGA-II). Tilt/azimuth optimization via grid search or Brent's method.
- **Emissions**: CO2 savings calculations and projections.
- **Visualization**: Publication-ready plots for energy balances, degradation, breakeven, Pareto fronts, and more.
- **Load profiles**: Support for standard load profiles (BDEW H0, EREDES, REE) and custom profiles.

## Installation

```bash
pip install breos
```

Or from source:

```bash
git clone https://github.com/Str4vinci/BREOS.git
cd BREOS
pip install -e .
```

## Quick Start

```python
import breos

app = breos.App({
    "location": "porto",              # preset or {"latitude": ..., "longitude": ..., "timezone": ...}
    "n_modules": 10,
    "annual_consumption_kwh": 4000,
    "battery_kwh": 5.0,               # 0 for no battery
    "cost_preset": "residential_pt",
    "emissions_country": "PT",
})

app.simulate()
result = app.result()

print(f"Grid independence: {result['grid_independence_pct']:.1f}%")
print(f"Payback: {result['payback_year']} years")
print(f"NPV savings: {result['npv_savings_eur']:,.0f} EUR")
print(f"CO2 avoided: {result['co2_avoided_total_kg']:,.0f} kg")
```

`result()` returns a plain Python dict (JSON-serializable, no pandas). See [Configuration](#configuration) for all options.

## Configuration

All keys except `location`, `annual_consumption_kwh`, and either `n_modules` or `pv_arrays` are optional with sensible defaults.

| Key | Default | Description |
|-----|---------|-------------|
| `location` | *required* | Preset key (`"porto"`, `"berlin"`, ...) or dict with `latitude`, `longitude`, `timezone` |
| `n_modules` | *required unless `pv_arrays` is set* | Number of PV modules |
| `pv_arrays` | `None` | Optional list of arrays with `modules`, `module`, `tilt`/`slope`, and `azimuth`; when present, the array module total overrides `n_modules` |
| `annual_consumption_kwh` | *required* | Annual electricity demand (kWh) |
| `battery_kwh` | `0.0` | Battery capacity (0 = no battery) |
| `pv_module` | `None` | Module name from catalogue (`None` = default) |
| `load_profile` | `"6"` | Load profile type (`"1"`–`"8"`) |
| `tilt` | auto | Tilt angle in degrees (auto-estimated from latitude) |
| `azimuth` | auto | Surface azimuth (auto: 180 for northern hemisphere) |
| `resolution` | `"h"` | Time resolution (`"h"` or `"15min"`) |
| `projection_years` | `20` | Economic projection horizon |
| `cost_preset` | `None` | Cost preset from `configs/costs.json` |
| `inflation_rate` | `0.02` | Annual electricity price inflation |
| `discount_rate` | `0.03` | Discount rate for NPV calculations |
| `emissions_country` | `None` | Country code for CO2 calculations (`"PT"`, `"DE"`, `"ES"`, ...) |
| `pv_degradation_rate` | `0.005` | Annual PV degradation (0.5%) |
| `calendar_model` | `"naumann_lam_field_calibrated"` | Battery calendar aging model |
| `dc_coupled` | `True` | DC-coupled / hybrid inverter |
| `inverter_efficiency` | `0.96` | Inverter efficiency |
| `inverter_loading_ratio` | `1.25` | DC/AC oversizing ratio |

## Result

`app.result()` returns a dict with:

| Key | Description |
|-----|-------------|
| `pv_production_kwh` | Year 1 PV production |
| `grid_independence_pct` | Year 1 grid independence (%) |
| `self_consumption_pct` | Year 1 self-consumption ratio (%) |
| `total_investment_eur` | Total CAPEX |
| `payback_year` | Payback year (`None` if not reached) |
| `npv_savings_eur` | NPV savings over projection period |
| `lcoe_eur_kwh` | Levelized cost of electricity |
| `co2_avoided_year1_kg` | Year 1 CO2 avoided |
| `co2_avoided_total_kg` | Lifetime CO2 avoided |
| `battery_soh_end_pct` | Battery state of health at end (if battery) |
| `monthly` | Year 1 monthly balance rows for PV, load, imports, exports, and self-consumption |
| `financial` | Yearly financial projection rows, including year 0 investment |
| `yearly` | List of per-year dicts with detailed breakdown |

### Multi-array PV systems

Use `pv_arrays` when a roof has panels on different faces or orientations:

```python
app = breos.App({
    "location": "porto",
    "annual_consumption_kwh": 4000,
    "pv_arrays": [
        {"modules": 8, "module": "Erlangen_445W", "tilt": 10, "azimuth": 90},
        {"modules": 8, "module": "Erlangen_445W", "tilt": 10, "azimuth": 270},
    ],
})
app.simulate()
```

BREOS calculates production per array and combines the DC output before the
energy balance, so east-west and pitched-roof layouts are not collapsed into a
single representative tilt/azimuth.

## Advanced Usage

For full control over individual simulation steps, you can use the internal modules directly:

```python
from breos.weather import fetch_tmy_weather_data
from breos.solar import calculate_pv_production_dc, PVModuleParams
from breos.battery import simulate_energy_balance, BatteryConfig
from breos.load_profiles import load_profile
from breos.economics import calculate_costs, cost_analysis_projection
from pvlib.location import Location

# Each module can be used independently
weather, metadata = fetch_tmy_weather_data(41.15, -8.63)
location = Location(41.15, -8.63, tz='Europe/Lisbon')
pv_dc = calculate_pv_production_dc(weather, location, tilt=35, surface_azimuth=180, n_modules=10)
# ...
```

## Additional Capabilities

BREOS is the open-source core of a broader simulation platform developed as part of PhD research. Additional features not included in this release:

- **Time-of-Use (TOU) tariff optimization** with multi-period pricing and strategy comparison
- **Vehicle-to-Home (V2H)** simulation with EV scheduling and bidirectional charging
- **Multi-chemistry battery support** — Sodium-ion (SIB), Vanadium Redox Flow (VRFB), Solid-State (SSB)
- **Thermal energy storage (TES)** with phase-change material modeling
- **Heat pump integration** with COP modeling and coupled electro-thermal energy balance
- **Community Self-Consumption (CSC)** modeling for multi-building scenarios

These modules may be released in the future or are available for academic collaboration upon request.

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

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Contact

For questions, collaboration, or access to additional modules, reach out at lrodrigues@fe.up.pt.

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.
