# Troubleshooting

Start with a configuration dry run. It needs no config file, weather download,
or simulation:

```bash
breos run --location porto --n-modules 10 \
  --annual-consumption-kwh 4000 --dry-run
```

If this succeeds, BREOS is installed and its packaged presets are readable.

## A config key or preset is rejected

BREOS rejects unknown top-level keys instead of silently applying a default.
Check packaged option names with:

```bash
breos list locations
breos list modules
breos list cost-presets
breos list emissions
breos list load-profiles
breos list battery-models
```

Use `breos validate-config config.toml` to see a concise resolved summary, or
`breos run --config config.toml --dry-run` for the complete JSON form. Config
files are flat TOML or JSON mappings except for supported `[[pv_arrays]]`,
`[sweep]`, and `[montecarlo]` sections.

## Weather download fails

The quickstart simulation loads a matching file from the current working
directory's `weather/` folder when available; otherwise it fetches PVGIS TMY
weather. Check internet access and retry before changing model settings.

For repeatable or offline work, seed the weather cache as shown in
[Offline runs with cached weather](recipes.md#offline-runs-with-cached-weather).
NSRDB access requires an NREL API key. Custom coordinate-dict locations do not
use a preset cache key and therefore fetch weather when used through `App`.

## An optional command cannot import a dependency

The base package contains the simulation core. Install only the extras needed
by the workflow:

```bash
pip install "breos[plots]"          # Matplotlib plotting helpers
pip install "breos[optimization]"   # pymoo optimization
pip install "breos[weather]"        # Open-Meteo historical weather
pip install "breos[fast]"           # Numba acceleration
pip install "breos[validation]"     # Excel and Arrow validation tools
```

Core imports, help, option discovery, and configuration validation do not load
Matplotlib. If an actual plotting command reports that its configuration
directory is not writable, point Matplotlib at a writable cache location:

```bash
export MPLCONFIGDIR="/tmp/matplotlib-cache"
```

## A run takes longer than expected

The first run may spend time downloading weather. Hourly simulations are the
fastest normal path; 15-minute simulations process four times as many steps.
Long projection horizons, Monte Carlo runs, parameter sweeps, and optimizers
repeat the core simulation and can take substantially longer.

Validate with `--dry-run` first. For an initial real run, use hourly resolution,
a one-year projection, and no sweep or Monte Carlo section, then restore the
study settings after the basic workflow succeeds.

## Results look surprising

Inspect the dry-run output before interpreting the result. It shows the chosen
module, array geometry, inverter rating, battery SOC window, loss assumptions,
cost preset, and emissions preset. Packaged values make examples runnable; they
are not substitutes for project-specific weather, demand, equipment, tariff,
and emissions inputs.

Typical Meteorological Year weather is representative rather than a forecast
for a particular year. See [Required Inputs](inputs.md) and
[Interpreting results](interpreting-results.md) for the assumptions and output
definitions that most often explain unexpected values.

## Reporting a reproducible problem

Include the BREOS version (`breos --version`), Python version, operating system,
the smallest config that reproduces the problem, the complete error, and whether
the no-network dry run succeeds. Do not include private API keys or licensed
load-profile data in a public issue.
