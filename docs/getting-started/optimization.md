# Optimization

`breos.App` simulates one design that you specify. The optimizer searches for
designs instead: it varies module count, battery capacity, tilt, and azimuth,
and returns the trade-off front between energy independence and money.

This is a Python API. There is no `breos optimize` subcommand, so the
command-line workflow in [Recipes](recipes.md) does not reach it.

## Install the extra

NSGA-II comes from pymoo, which the base install omits:

```bash
pip install "breos[optimization]"
```

Without it, {py:func}`~breos.optimization.optimize_system_multi_objective`
raises `ImportError` and names this command.

## The optimizer config is not the App config

This trips people up, so it is worth stating plainly. `App` takes a flat
dictionary of keys such as `n_modules` and `cost_preset`. The optimizer takes a
nested dictionary grouped into sections: `location`, `load`, `pv`, `battery`,
`optimization`, `constraints`, `costs`, `financials`, `emissions`, and
`simulation`. The two shapes are not interchangeable, and a flat `App` config
passed to the optimizer fails on the first missing section.

A ready-to-edit nested config ships as
[`configs/optimization/projected-optimization.toml`](https://github.com/Str4vinci/breos/blob/main/configs/optimization/projected-optimization.toml).
Load it with `tomllib` and pass the resulting dictionary through:

```python
import tomllib

with open("configs/optimization/projected-optimization.toml", "rb") as handle:
    config = tomllib.load(handle)
```

Module count and battery capacity are absent from that file on purpose. The
optimizer chooses them, bounded by `[constraints]`.

## Load the weather and the load profile

The optimizer takes one weather year and one load year as DataFrames, and
reuses them for every candidate and every projection year:

```python
import pandas as pd
from breos.load_profiles import load_profile

weather = pd.read_csv(config["simulation"]["weather_file"], index_col=0)
weather.index = pd.to_datetime(weather.index, utc=True)

load = load_profile(
    config["load"]["profile_type"],
    config["load"]["annual_consumption_kwh"],
    start_date=f"{weather.index[0].year}-01-01",
    freq=config["simulation"]["resolution"],
    timezone=config["location"]["timezone"],
)
```

For 15-minute runs, upsample the weather with
{py:func}`~breos.weather.resample_to_15min` before you pass it in.

## Search the design space

```python
from breos.optimization import optimize_system_multi_objective

result = optimize_system_multi_objective(
    weather,
    load,
    config,
    pop_size=config["optimization"]["pop_size"],
    n_offsprings=config["optimization"]["n_offsprings"],
    n_gen=config["optimization"]["n_gen"],
    seed=config["optimization"]["seed"],
)

pareto = result.details["pareto"]
print(pareto[["Modules", "Battery_kWh", "Tilt", "Azimuth"]])
```

`pareto` is a DataFrame with one row per non-dominated design, holding the
sizing columns above, the objective values, ZEB diagnostics, and both
steady-state and projected fields. There is no single best row. Pick the design
whose balance of independence and cost matches the project.

The optimizer does not read `pop_size`, `n_offsprings`, `n_gen`, or `seed`
from the nested config automatically. Forward them as shown above. Set and
record the seed because NSGA-II is stochastic. Raise the population and
generation counts for a denser front and a longer runtime. `n_procs` above 1
evaluates candidates in parallel processes, which is worth setting for
anything beyond a quick look.

If no candidate satisfies the constraints, the call raises `RuntimeError`.
Loosen `budget_eur`, `max_area_m2`, `max_modules`, or `max_battery_kwh` in
`[constraints]` and run it again.

## Score designs over their projected lifetime

By default the objectives describe year one: annual grid independence, NPV, and
ZEB ratio. A design scored this way ignores what battery degradation does to it
by year fifteen.

Setting `objective_basis` switches the objectives to the whole horizon:

```toml
[optimization]
objective_basis = "projected"
```

The objectives become lifetime grid independence and lifetime NPV, both
computed from simulated yearly values rather than a degradation factor applied
after the fact. ZEB drops to a diagnostic. To keep it as a feasibility
constraint, set `enforce_zeb = true` under `[constraints]`.

This costs real time, because every candidate now runs the full projection
instead of one year. Start with a small `pop_size` and `n_gen` while you check
that the config resolves, then scale up.

## Evaluate one design in detail

Once you have chosen a design, {py:func}`~breos.optimization.evaluate_projected_design`
re-runs it and returns the annual tables behind the headline numbers:

```python
from breos.optimization import evaluate_projected_design

detail = evaluate_projected_design(
    weather,
    load,
    config,
    n_modules=9,
    battery_kwh=5.0,
    tilt=35.0,
    azimuth=180.0,
)

print(detail.metrics["Projected_Grid_Independence_%"])
print(detail.yearly.head())
print(detail.financial.head())
```

`metrics` holds the projected headline values, including year-one, final-year,
mean, and minimum variants of grid independence and ZEB ratio. `yearly` is the
per-year energy and degradation ledger, and `financial` is the matching
discounted cost ledger. Both are DataFrames, so they go straight into a plot or
a CSV.

This function uses the same PV, battery, replacement, degradation, and
economics components as the optimizer, so its numbers agree with the front it
came from.

## Related pages

- [Recipes](recipes.md) for single-design runs through `App` and the CLI.
- [Interpreting results](interpreting-results.md) for the meaning of the
  headline metrics.
- [Optimization API](../api/optimization.md) for the full signatures.
