# Monte Carlo

A single simulation answers "what happens in a typical year". Monte Carlo
answers "how wide is the range of outcomes", by running the projection many
times over resampled weather and demand.

Each run is a full multi-year projection. For every projection year, BREOS draws
a weather year at random from your historical file and scales demand by a random
multiplier. Aggregating the runs gives distributions for NPV, payback year, grid
independence, LCOE, and final state of health.

## You have to supply the weather

BREOS ships no weather data, and Monte Carlo needs a multi-year historical CSV
rather than a single TMY. Download one for your site, put it in a local
`weather/` directory, and point `[montecarlo].weather_file` at it. The
`weather/` directory is git-ignored by convention.

Fetch historical data with the `weather` extra:

```bash
pip install "breos[weather]"
```

## Configure a study

The top-level keys are the ordinary scenario, identical to `breos run`. The
`[montecarlo]` section adds the study controls:

```toml
location = "porto"
n_modules = 10
annual_consumption_kwh = 4000
battery_kwh = 5.0
cost_preset = "residential_pt"
emissions_country = "PT"
resolution = "h"
projection_years = 20

[montecarlo]
weather_file = "weather/porto_historical_2005_2024_openmeteo.csv"
n_runs = 100
years_per_run = 20
load_uncertainty = 0.10
load_distribution = "normal"
target_year = 2025
seed = 42
collect_yearly = false
n_procs = 1
```

`load_uncertainty` is the standard deviation of the annual demand multiplier,
which is normal around 1.0 by default. Set `load_distribution = "uniform"` to
draw from `1 - load_uncertainty` to `1 + load_uncertainty` instead.
`weather_start_year` and `weather_end_year` restrict which years are eligible
for sampling when your file covers more history than you want to use.

A runnable version ships as
[`configs/examples/montecarlo.toml`](https://github.com/Str4vinci/breos/blob/main/configs/examples/montecarlo.toml).

## Run it

```bash
breos montecarlo --config configs/examples/montecarlo.toml --runs 100 --plots
```

Every setting in `[montecarlo]` has a command-line override, so
`--runs`, `--seed`, `--years`, `--n-procs`, and `--weather-file` all work
without editing the file. Start with `--runs 10` to check the config resolves,
then raise it.

`--n-procs` runs trajectories in parallel processes and is the setting that
matters most for wall-clock time.

## What you get back

`monte_carlo_results.csv` holds one row per run. Alongside it, BREOS writes a
provenance JSON recording the resolved settings and hashes of the inputs and
outputs, which is what makes a published result auditable later.

`--collect-yearly` adds a second CSV with one row per run and projection year,
carrying the energy, degradation, and discounted-cost ledger. Cost envelopes and
fan charts need it, and it is off by default because it is much larger.

`--plots` writes payback, NPV, grid-independence, final-SoH, and LCOE
distributions into `plots/`. `--json` prints a machine-readable summary to
stdout for scripting.

## Fix the seed

Set `seed` and keep it with the results. Without it, each study draws fresh
randomness and the numbers move between runs, which makes a figure impossible to
reproduce. The seed is recorded in the provenance JSON.

## The optional Numba backend

Monte Carlo repeats the daily dispatch loop millions of times, which is where
the time goes. The `fast` extra installs Numba for an optional dispatch kernel:

```bash
pip install "breos[fast]"
```

```toml
[montecarlo]
execution_backend = "numba"
```

Installing the extra changes nothing on its own. Without
`execution_backend = "numba"`, Monte Carlo uses the Python path.

The kernel carries the production BREOS dispatch and its energy ledger. Rainflow
counting, degradation, resistance growth, and replacement stay in Python, so the
backend accelerates one stage rather than the whole model. It is private, with
no public API, and configuration is the only supported way to select it.

BREOS works fully without Numba. The Python path stays the default and remains
the numerical reference that the backend is checked against. The same compiled
dispatch backend is also available to `breos.App` and multi-objective
optimization through their `execution_backend` option.

## Related pages

- [Recipes](recipes.md) for the single-run scenario keys.
- [Interpreting results](interpreting-results.md) for what each metric means.
- [Optimization](optimization.md) for searching designs rather than sampling
  uncertainty.
