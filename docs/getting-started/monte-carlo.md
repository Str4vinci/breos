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

`target_year` is the study's calendar year. Every sampled weather year is
restamped to it, and the demand profile and any tariff follow its calendar.
The bundled H0 profile therefore puts its weekday, Saturday and Sunday shapes
on that year's days, as an App run with `start_date` on 1 January of that year
does. Monte Carlo does not read `start_date`.

A runnable version ships as
[`configs/examples/montecarlo.toml`](https://github.com/Str4vinci/breos/blob/main/configs/examples/montecarlo.toml).

## Run it

```bash
breos montecarlo --config configs/examples/montecarlo.toml --runs 100 --plots
```

Common settings have command-line overrides. For example, `--runs`, `--seed`,
`--years`, `--n-procs`, and `--weather-file` work without editing the file.
Start with `--runs 10` to check that the config resolves, then raise it.

`--n-procs` runs trajectories in parallel processes and is the setting that
matters most for wall-clock time.

## What you get back

`monte_carlo_results.csv` holds one row per run. Alongside it, BREOS writes a
provenance JSON recording the resolved settings and hashes of the inputs and
outputs, which is what makes a published result auditable later. It and the
`--json` output carry `result_schema_version` and `currency`, the currency of
every money column (see
[Interpreting results](interpreting-results.md#currency-and-schema-version)).

`--collect-yearly` adds a second CSV with one row per run and projection year,
carrying the energy, degradation, and discounted-cost ledger. Cost envelopes and
fan charts need it, and it is off by default because it is much larger.

`--plots` writes payback, NPV, grid-independence, final-SoH, and LCOE
distributions into `plots/`. `--json` prints a machine-readable summary to
stdout for scripting.

Each summary entry gives `count`, the number of runs its statistics cover, out
of `n_runs`. `payback_year` and `payback_year_interpolated` are the sustained
discounted payback within the simulated period, as a whole and as an
interpolated fractional year: the time from which cumulative discounted savings
stay zero or above to the horizon (see
[Interpreting results](interpreting-results.md)). A run that never pays back
within the horizon has no payback year, so the payback statistics cover only
the runs that paid back.
`payback_probability` gives the share of runs that did. Read the two together:
a payback median of 9.8 years means little if only 60% of runs pay back.
The provenance file and the `--json` output are standard JSON, so a statistic
with no defined value is written as `null`, never as `NaN` or `Infinity`.

## Fix the seed

Set `seed` and keep it with the results. Without it, each study draws fresh
randomness and the numbers move between runs, which makes a figure impossible to
reproduce. The seed is recorded in the provenance JSON.

Each run draws from its own stream, spawned from the seed with NumPy's
`SeedSequence`. Studies under different seeds, even adjacent ones such as 42
and 43, share no trajectory, so they can be compared or pooled as independent
samples. The results do not depend on `n_procs`.

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

The kernel is the production BREOS dispatch and its energy ledger, compiled
from the same functions the Python path runs rather than from a copy. Rainflow
counting, degradation, resistance growth, and replacement stay in Python, so the
backend accelerates one stage rather than the whole model. It is private, with
no public API, and configuration is the only supported way to select it.

BREOS works fully without Numba. The Python path stays the default and remains
the numerical reference that the backend is checked against. The same compiled
dispatch backend is also available to `breos.App` and multi-objective
optimization through their `execution_backend` option.

## Sweep designs over one weather file

Before its first trajectory, a study reads the weather file, resamples it, and
computes each weather year's PV production and battery temperature. That setup
does not depend on the number of runs, so with the Numba backend it is often
most of a small study's time. A sweep over designs repeats it for every design.

From Python, build the setup once with `build_year_cache` and pass it to each
study as `year_cache`:

```python
from breos.montecarlo import MonteCarloSettings, build_year_cache, run_montecarlo

settings = MonteCarloSettings(weather_file="weather/porto_2005_2024.csv", n_runs=100, seed=42)
cache = build_year_cache(config, settings)
results = {
    kwh: run_montecarlo({**config, "battery_kwh": kwh}, settings, year_cache=cache)
    for kwh in (2.5, 5.0, 7.5, 10.0)
}
```

The results are the same, bit for bit, as studies run without the cache. The
cache has two layers:

- The weather layer is reused only for the same weather file (its absolute
  path, its contents and those of its `.metadata.json` sidecar), year window,
  `target_year`, resolution, location, `preserve_irradiance_energy`, and
  solar-position method. A study with other weather inputs raises
  `ValueError`.
- The PV layer is reused when the config differs only in keys that do not
  reach PV production or battery temperature: battery sizing and dispatch,
  inverter, degradation, cost, tariff, emissions, and demand settings
  (`breos.montecarlo.YEAR_CACHE_INDEPENDENT_KEYS`). Any other change, such as
  `n_modules`, `tilt`, `battery_temperature`, `battery_indoor_model`, or new
  contents in a `battery_temperature` CSV, rebuilds the PV layer from the
  cached weather and replaces the old one. Run designs grouped by PV
  configuration to get the most reuse.

A study may replace the PV layer, so do not share one cache between threads
running studies at the same time.

## Related pages

- [Recipes](recipes.md) for the single-run scenario keys.
- [Interpreting results](interpreting-results.md) for what each metric means.
- [Optimization](optimization.md) for searching designs rather than sampling
  uncertainty.
