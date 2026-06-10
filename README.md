# BREOS - Building Renewable Energy Optimization Software

[![Tests](https://github.com/Str4vinci/breos/actions/workflows/tests.yml/badge.svg)](https://github.com/Str4vinci/breos/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/breos.svg)](https://pypi.org/project/breos/)
[![License: BSD-3](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

A Python library for PV and battery energy-system simulation and optimization, designed for research and engineering applications.

## Features

- **Weather data**: Fetch TMY data from PVGIS/NSRDB and historical data from Open-Meteo. Support for hourly and 15-minute resolutions with Makima interpolation.
- **PV production**: DC and AC power calculations using pvlib (CEC single-diode model), with a small catalog of example modules and full support for custom module parameters.
- **Battery simulation**: Energy balance with calendar and cycle aging models (Naumann 2020, Lam 2025) and field-calibrated LFP parameters. Optional approximate Numba kernels for fast standalone screening studies.
- **Economics**: NPV, LCOE, breakeven analysis, and cost projections with configurable tariffs and inflation.
- **Optimization**: Multi-objective (grid independence, NPV, ZEB ratio) system sizing using pymoo (NSGA-II). Tilt/azimuth optimization via grid search or Brent's method.
- **Emissions**: CO<sub>2</sub> savings calculations and projections.
- **Visualization**: Publication-ready plots for energy balances, degradation, breakeven, Pareto fronts, and more.
- **Load profiles**: Bundled demandlib-derived H0 examples, plus support for user-supplied BDEW, E-REDES, REE, and custom profiles.

## Installation

```bash
pip install breos
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add breos          # as a project dependency
uvx breos --version   # run the CLI without installing
```

Optional feature groups keep the default install focused on core PV +
battery simulation:

```bash
pip install "breos[plots]"          # publication plots
pip install "breos[optimization]"   # pymoo multi-objective sizing
pip install "breos[weather]"        # Open-Meteo historical weather fetching
pip install "breos[fast]"           # approximate Numba screening kernels (not used by App)
pip install "breos[validation]"     # Excel / Arrow validation workflows
```

To install from source instead:

```bash
git clone https://github.com/Str4vinci/breos.git
cd breos
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

For real studies, bring your own weather/API access where required, licensed
load profiles, PV module/system data, and cost/tariff assumptions. The packaged
defaults are intended to make the tool runnable, not to certify a project.

## Command Line

Run a simulation without writing Python:

```bash
breos run \
  --location porto \
  --n-modules 10 \
  --annual-consumption-kwh 4000 \
  --battery-kwh 5.0 \
  --cost-preset residential-pt \
  --emissions-country pt \
  --output result.json
```

The CLI writes the same JSON-serializable result returned by `App.result()`.
You can also pass a TOML or JSON config file:

```bash
breos run --config configs/examples/quickstart.toml --output result.json
```

Inspect a config before running the full simulation:

```bash
breos validate-config configs/examples/quickstart.toml
breos run --config configs/examples/quickstart.toml --dry-run
```

Discover packaged option keys:

```bash
breos list locations
breos list modules
breos list cost-presets
breos list emissions
breos list load-profiles
```

For non-bundled RLPs, put licensed CSVs in a local directory and pass it through config or flags:

```bash
breos run --config configs/examples/external-rlp.toml --rlp-directory external_rlp
```

## Configuration

All keys except `location`, `annual_consumption_kwh`, and either `n_modules` or `pv_arrays` are optional with sensible defaults.

| Key | Default | Description |
|-----|---------|-------------|
| `location` | *required* | Preset key (`"porto"`, `"berlin"`, ...) or dict with `latitude`, `longitude`, `timezone` |
| `n_modules` | *required unless `pv_arrays` is set* | Number of PV modules |
| `pv_arrays` | `None` | Optional list of arrays with `modules`, `module`, `tilt`, and `azimuth`; when present, the array module total overrides `n_modules` |
| `annual_consumption_kwh` | *required* | Annual electricity demand (kWh) |
| `battery_kwh` | `0.0` | Nominal battery capacity in kWh (0 = no battery). The SOC window sets the usable share — see [Modeling conventions](#modeling-conventions) |
| `pv_module` | `None` | Module name from catalogue (`None` = default) |
| `load_profile` | `"1"` | Bundled demandlib-derived H0 profile. Other standard profiles require caller-supplied CSVs |
| `rlp_directory` | `None` | Directory containing licensed external RLP CSVs for non-bundled load profiles |
| `tilt` | auto | Tilt angle in degrees (auto-estimated from latitude) |
| `azimuth` | auto | Surface azimuth (auto: 180 for northern hemisphere) |
| `tracking` | `"fixed"` | Tracking mode (`"fixed"`, `"single_axis"`, or `"dual_axis"`) |
| `axis_tilt` | `0.0` | Single-axis tracker axis tilt |
| `axis_azimuth` | auto | Tracker axis azimuth (auto from latitude) |
| `max_angle` | `60.0` | Single-axis tracker maximum rotation angle |
| `backtrack` | `True` | Whether single-axis trackers backtrack to avoid row shading |
| `gcr` | `0.35` | Ground coverage ratio for single-axis tracking |
| `cross_axis_tilt` | `0.0` | Cross-axis terrain slope for single-axis tracking |
| `dual_axis_max_tilt` | `90.0` | Maximum panel tilt for dual-axis tracking |
| `resolution` | `"h"` | Time resolution (`"h"` or `"15min"`) |
| `projection_years` | `20` | Economic projection horizon |
| `cost_preset` | `None` | Cost preset key from packaged defaults; editable examples live in `configs/base/` |
| `inflation_rate` | `0.02` | Annual electricity price inflation |
| `discount_rate` | `0.03` | Discount rate for NPV calculations |
| `emissions_country` | `None` | Country code for CO<sub>2</sub> calculations (`"PT"`, `"DE"`, `"ES"`, ...) |
| `pv_degradation_rate` | `0.005` | Annual PV degradation (0.5%) |
| `calendar_model` | `"naumann_lam_field_calibrated"` | Battery calendar aging model |
| `battery_min_soc` | `0.10` | Battery SOC floor (fraction of nominal, SOH-derated capacity) |
| `battery_max_soc` | `0.90` | Battery SOC ceiling (same basis as `battery_min_soc`) |
| `battery_eol_percentage` | `0.70` | SOH fraction that triggers battery replacement |
| `battery_rte` | `None` | Battery round-trip efficiency, split evenly across charge/discharge (`None` = 0.95) |
| `dc_coupled` | `True` | DC-coupled / hybrid inverter |
| `inverter_efficiency` | `0.96` | Inverter efficiency |
| `inverter_loading_ratio` | `1.25` | DC/AC oversizing ratio; also sets the inverter AC rating that clips production |
| `pv_loss_overrides` | `None` | Per-component overrides (percent) for the fixed PVWatts system losses, e.g. `{"shading": 0.0}` |
| `start_date` | `"2023-01-01"` | First simulation date |

### Modeling conventions

- **System losses**: every DC production calculation applies pvlib's PVWatts
  losses with BREOS defaults of soiling 2%, shading 3%, mismatch 2%, wiring 2%,
  connections 0.5%, LID 1.5%, nameplate 1%, and availability 3% — about 14.1%
  combined (`breos.solar.DEFAULT_PVWATTS_LOSSES`). Age-based degradation is
  added separately per simulation year. Override individual components with
  `pv_loss_overrides` (App) or `loss_overrides` (solar functions).
- **Inverter**: the energy balance applies a flat `inverter_efficiency` and
  clips AC output (PV and battery discharge combined) at the inverter rating
  implied by `inverter_loading_ratio` — the same rating used for inverter
  CAPEX. DC surplus above the rating can still charge a DC-coupled battery.
- **Battery SOC window**: `battery_kwh` is the nominal pack capacity. The
  energy balance only cycles the battery between `battery_min_soc` and
  `battery_max_soc`, so the effective storage swing is
  `battery_kwh × (battery_max_soc − battery_min_soc)` — 80% of nominal with
  the defaults. Battery datasheets usually advertise *usable* capacity; to
  match a spec sheet, enter `usable / 0.8` or widen the SOC window. Aging is
  evaluated on absolute SOC, so the window also shapes degradation results —
  the defaults reflect the operating range the field-calibrated aging
  parameters were fit for.

## Result

`app.result()` returns a dict whose main fields are listed below; see
[docs/getting-started/interpreting-results.md](docs/getting-started/interpreting-results.md)
for the full key reference, including system echo fields (`pv_kwp`,
`consumption_kwh`, ...), grid flows, and battery replacement details.

| Key | Description |
|-----|-------------|
| `pv_production_kwh` | Year 1 PV production |
| `grid_independence_pct` | Year 1 grid independence (%) |
| `self_consumption_pct` | Year 1 self-consumption ratio (%) |
| `total_investment_eur` | Total CAPEX |
| `payback_year` | Payback year (`None` if not reached) |
| `npv_savings_eur` | NPV savings over projection period |
| `lcoe_eur_kwh` | Levelized cost of electricity |
| `co2_avoided_year1_kg` | Year 1 CO<sub>2</sub> avoided |
| `co2_avoided_total_kg` | Lifetime CO<sub>2</sub> avoided |
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

For full control over individual simulation steps, use the lower-level modules directly:

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

Two working-directory conventions to be aware of:

- A `weather/` directory in the current working directory is scanned before
  any PVGIS fetch — a file matching the location preset name is used silently
  instead of fetching. Remove or rename it to force a fresh fetch.
- Historical Open-Meteo fetches cache responses in a `.cache.sqlite` file in
  the current working directory (30-day expiry).

Library modules report progress (file discovery, saved files, conversions)
through the standard `logging` module under the `breos.*` logger names —
enable them with `logging.basicConfig(level=logging.INFO)` or silence them
per module. Functions with a `verbose` flag still print to stdout when asked.

## Load Profile Data Note

The public package bundles only demandlib-derived H0 example profiles. E-REDES, REE, and direct BDEW CSVs are supported as user-provided files through `rlp_directory`, but are not redistributed in this repository because their public source terms do not clearly grant package redistribution rights. See [ATTRIBUTIONS.md](ATTRIBUTIONS.md) and [docs/legal/load-profile-data.md](docs/legal/load-profile-data.md).

## Resources

See [docs/resources.md](docs/resources.md) for links to PV modelling references,
RLP sources, weather/solar-resource APIs, and input assumptions to record.

## Citation

If you use BREOS in your research, please cite:

```bibtex
@software{breos,
  author = {Rodrigues, Leonardo},
  title = {BREOS: Building Renewable Energy Optimization Software},
  year = {2026},
  url = {https://github.com/Str4vinci/breos}
}
```

## Roadmap

See [ROADMAP.md](ROADMAP.md) for planned architectural work and capability extensions.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

BREOS uses `develop` as the default development branch. Feature work should be
done on separate branches and opened as pull requests into `develop`.

The `main` branch tracks stable releases only. Use `main` or the GitHub Releases
page when you want the latest stable version.

## Contact

For questions, collaboration, or access to additional modules, reach out at lrodrigues@fe.up.pt.

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.
