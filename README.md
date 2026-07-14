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
- **Monte Carlo**: Multi-year weather-year and demand resampling for NPV, payback, grid-independence, LCOE, and battery SoH distributions.
- **Optimization**: Multi-objective PV/battery sizing with pymoo (NSGA-II), tilt optimization via grid search or Brent's method, simple battery sizing sweeps, and ZEB sizing.
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
pip install "breos[validation]"     # Excel / Arrow dependencies for local validation work
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
You can also pass a TOML or JSON config file. A minimal `quickstart.toml`
looks like this:

```toml
location = "porto"
n_modules = 10
annual_consumption_kwh = 4000
battery_kwh = 5.0
load_profile = "demandlib_h0"
cost_preset = "residential_pt"
emissions_country = "PT"
projection_years = 20
resolution = "h"
```

```bash
breos validate-config quickstart.toml
breos run --config quickstart.toml --dry-run
breos run --config quickstart.toml --output result.json
```

The full simulation may fetch PVGIS TMY weather data and needs internet access
unless a matching local weather cache is present. Source checkouts include the
same config at `configs/examples/quickstart.toml`.

Discover packaged option keys:

```bash
breos list locations
breos list modules
breos list cost-presets
breos list emissions
breos list load-profiles
```

Run a parameter grid from a normal config plus a `[sweep]` section:

```toml
location = "porto"
n_modules = 10
annual_consumption_kwh = 4000
battery_kwh = 0.0
cost_preset = "residential_pt"

[sweep]
n_modules = [8, 10, 12]
battery_kwh = [0.0, 5.0]
```

```bash
breos sweep --config configs/examples/sweep.toml --output sweep_results.csv
```

The command runs every parameter combination and writes one CSV row per run,
including the varied parameters, resolved system sizing, BREOS version, and
top-level scalar result metrics.

Run a Monte Carlo study over weather-year and demand uncertainty:

```bash
breos montecarlo \
  --config configs/examples/montecarlo.toml \
  --weather-file weather/porto_historical_2005_2024_openmeteo.csv \
  --runs 100 \
  --plots
```

The command writes one row per trajectory to `monte_carlo_results.csv` and,
with `--plots`, saves payback, NPV, grid-independence, final-SoH, and LCOE
distributions in `plots/`. BREOS does not bundle the multi-year historical
weather file; provide your own CSV or keep it in the git-ignored local
`weather/` directory.

Multi-objective PV/battery sizing is available from Python with
`breos.optimize_system_multi_objective(...)` after installing
`breos[optimization]`. It returns an `OptimizationResult` whose
`details["pareto"]` table contains the NSGA-II Pareto solutions.

For non-bundled RLPs, put licensed CSVs in a local directory and pass it through config or flags:

```bash
breos run --config configs/examples/external-rlp.toml --rlp-directory external_rlp
```

## Configuration

Only `location`, `annual_consumption_kwh`, and either `n_modules` or
`pv_arrays` are required. Common keys are:

| Key | Default | Description |
|-----|---------|-------------|
| `location` | *required* | Preset key (`"porto"`, `"berlin"`, ...) or dict with `latitude`, `longitude`, `timezone` |
| `n_modules` | *required unless `pv_arrays` is set* | Number of PV modules |
| `annual_consumption_kwh` | *required* | Annual electricity demand (kWh) |
| `battery_kwh` | `0.0` | Nominal battery capacity in kWh (0 = no battery). The SOC window sets the usable share — see [Modeling conventions](#modeling-conventions) |
| `pv_arrays` | `None` | Multiple roof faces, each with `modules`, `module`, `tilt`, and `azimuth` |
| `pv_module` | `None` | Module name from catalogue (`None` = default) |
| `load_profile` | `"1"` | Bundled demandlib-derived H0 profile; `"demandlib_h0"` is the friendly alias |
| `resolution` | `"h"` | Time resolution (`"h"` or `"15min"`) |
| `cost_preset` | `None` | Cost preset key from packaged defaults; editable examples live in `configs/base/` |
| `emissions_country` | `None` | Country code for CO<sub>2</sub> calculations (`"PT"`, `"DE"`, `"ES"`, ...) |
| `projection_years` | `20` | Economic projection horizon |
| `tilt`, `azimuth` | auto | Fixed-array orientation; defaults are estimated from latitude |
| `inverter_loading_ratio` | `1.25` | DC/AC oversizing ratio; also sets the inverter AC rating that clips production |
| `calendar_model` | `"naumann_lam_field_calibrated"` | Battery aging model; default is v1 field calibration |
| `rlp_directory` | `None` | Directory containing licensed external RLP CSVs for non-bundled load profiles |

See [docs/getting-started/configuration.md](docs/getting-started/configuration.md)
for every option, including tracking, the sky-diffusion (transposition) model,
PV loss overrides, battery SOC limits, tariffs, inflation, and discounting.

### Modeling conventions

- **System losses**: every DC production calculation applies pvlib's PVWatts
  losses with BREOS defaults of soiling 2%, shading 3%, mismatch 2%, wiring 2%,
  connections 0.5%, LID 1.5%, nameplate 1%, and availability 3% — about 14.1%
  combined (`breos.solar.DEFAULT_PVWATTS_LOSSES`). Age-based degradation is
  added separately per simulation year. Override individual components with
  `pv_loss_overrides` (App) or `loss_overrides` (solar functions).
- **PV model background**: BREOS uses pvlib for solar position, irradiance
  transposition, cell temperature, and PV performance model pieces. The
  sky-diffusion transposition model is selectable via `transposition_model`
  (default `isotropic`), with `albedo`/`surface_type` for ground reflectance
  and `model_perez` for the Perez coefficient set (see configuration docs). See
  [docs/resources.md](docs/resources.md) for pvlib and PV model references.
- **Inverter**: the energy balance and `dc_to_ac()` use the same PVWatts
  part-load curve, parameterized by nominal `inverter_efficiency`, and clip
  PV and battery discharge at their shared AC rating implied by
  `inverter_loading_ratio` — the same rating used for inverter CAPEX. DC
  surplus above the rating can still charge a DC-coupled battery.
- **Battery SOC window**: `battery_kwh` is the nominal pack capacity. The
  energy balance only cycles the battery between `battery_min_soc` and
  `battery_max_soc`, so the effective storage swing is
  `battery_kwh × (battery_max_soc − battery_min_soc)` — 80% of nominal with
  the defaults. Battery datasheets usually advertise *usable* capacity; to
  match a spec sheet, enter `usable / 0.8` or widen the SOC window. Aging is
  evaluated on absolute SOC, so the window also shapes degradation results —
  the defaults reflect the operating range the field-calibrated aging
  parameters were fit for.
- **Battery degradation calibration**: `calendar_model =
  "naumann_lam_field_calibrated"` is the stable default and maps to the v1
  field calibration. The explicit alias
  `"naumann_lam_field_calibrated_v1"` is equivalent. The v2 option fixes Lam
  `Ea` and `n` while fitting `k0` and `b` to the field data, available as
  `"naumann_lam_field_calibrated_v2"`.

## Result

`app.result()` returns a dict whose main fields are listed below; see
[docs/getting-started/interpreting-results.md](docs/getting-started/interpreting-results.md)
for the full key reference, including system echo fields (`pv_kwp`,
`consumption_kwh`, ...), grid flows, and battery replacement details.

| Key | Description |
|-----|-------------|
| `pv_production_kwh` | Legacy AC-equivalent non-curtailed PV field |
| `pv_ac_system_kwh` | Year 1 usable AC delivered to load or export |
| `pv_dc_generation_kwh` | Year 1 PV DC generation before dispatch |
| `grid_independence_pct` | Year 1 grid independence (%) |
| `self_consumption_pct` | Year 1 self-consumption ratio (%) |
| `total_investment_eur` | Total CAPEX |
| `payback_year` | Payback year (`None` if not reached) |
| `npv_savings_eur` | NPV savings over projection period |
| `lcoe_eur_kwh` | Levelized cost of electricity from system CAPEX, O&M, simulated replacements, and discounted PV production |
| `co2_avoided_self_consumption_year1_kg` | Year 1 behind-the-meter CO<sub>2</sub> benefit |
| `co2_avoided_export_year1_kg` | Year 1 exported-generation CO<sub>2</sub> benefit |
| `co2_avoided_total_year1_kg` | Sum of year 1 pathway benefits |
| `co2_avoided_total_lifetime_kg` | Sum of lifetime pathway benefits |
| `co2_avoided_year1_kg`, `co2_avoided_total_kg` | Compatibility aliases for totals |
| `battery_soh_end_pct` | Battery state of health at end (if battery) |
| `monthly` | Year 1 monthly balance rows for PV, load, imports, exports, and self-consumption |
| `financial` | Yearly financial projection rows, including year 0 investment |
| `yearly` | List of per-year dicts with detailed breakdown |

BREOS 0.3.x models DC-coupled/hybrid stationary batteries only; unsupported
AC coupling fails explicitly. PV and battery discharge share the inverter AC
nameplate, while above-headroom PV can still charge the DC battery. Optional
charge and discharge power limits are available; `None` retains the legacy
unlimited behavior. See the [energy-balance contract](docs/api/energy-balance.md)
for DC/AC bases, conservation identities, and migration guidance for
`PV_Production` consumers.

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

### Opt-in BLAST degradation models

Native Naumann/Lam degradation remains the default. Select BLAST explicitly
with `degradation_engine = "blast"` and a stable `blast_model` key. Discover
the 14 scientifically identified vendored cell models with
`breos.list_battery_models()` or `breos list battery-models`; the output
includes chemistry, form factor, experimental range, citations, capabilities,
and upstream provenance. App configs using the ambiguous legacy
`battery_type` selector now receive an actionable migration error.

BLAST model profiles never invent generic chemistry defaults. Resolution is
user config over sourced profile defaults over global defaults. BLAST remains
opt-in, and BLAST plus Monte Carlo is rejected explicitly rather than falling
back to native degradation.

For full control over individual simulation steps, use the lower-level modules directly:

```python
from breos.weather import fetch_tmy_weather_data
from breos.solar import calculate_pv_production_dc, PVModuleParams
from breos.battery import simulate_energy_balance, BatteryConfig
from breos.load_profiles import load_profile
from breos.economics import calculate_costs, cost_analysis_projection
from pvlib.location import Location

# Each module can be used independently
weather, metadata = fetch_tmy_weather_data(41.15, -8.63, timezone="Europe/Lisbon")
location = Location(41.15, -8.63, tz='Europe/Lisbon')
pv_dc = calculate_pv_production_dc(weather, location, tilt=35, surface_azimuth=180, n_modules=10)
# ...
```

## Version 0.3.4 Scope

BREOS 0.3.4 adds an explicit DC/AC energy ledger, cross-year battery-state
continuity, reconciled PV loss reporting, optimizer/App parity, and a standing
multi-site validation suite. It also adds opt-in solar-position, diffuse-IAM,
and mounting-temperature models while keeping `breos.App` as the stable public
facade and retaining compatibility aliases for existing result fields.

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

If you use BREOS in your research, please cite the preprint:

```bibtex
@misc{rodrigues2026breos,
  author = {Rodrigues, L. and Delgado, J. M. P. Q. and Mendes, A. and Guimar{\~a}es, A. S.},
  title  = {A Modular, Open-Source Python Framework for Household PV-Battery Sizing: Validation, Multi-Objective Optimisation, and Uncertainty Analysis},
  year   = {2026},
  doi    = {10.2139/ssrn.7032064},
  url    = {https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7032064},
  note   = {SSRN preprint}
}
```

You may also cite the software directly:

```bibtex
@software{breos,
  author = {Rodrigues, Leonardo},
  title  = {BREOS: Building Renewable Energy Optimization Software},
  year   = {2026},
  url    = {https://github.com/Str4vinci/breos}
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
