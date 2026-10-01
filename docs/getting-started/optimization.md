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
nested dictionary grouped into sections: `location`, `pv`, `battery`,
`optimization`, `constraints`, `costs`, `financials`, `emissions`, and
`simulation`, with optional `tariff` and `smart_charging` tables. The two
shapes are not interchangeable. Every table takes a fixed set of keys, so a
flat `App` config passed to the optimizer, or a misspelt key, raises before
any design is scored; the [API reference](../api/optimization.md#configuration-keys)
lists the keys and their defaults. The `financials` section takes the App's rate keys:
`inflation_rate`, `sell_price_inflation`, `discount_rate` and the separate
escalators `import_price_escalation`, `om_escalation` and
`replacement_cost_learning`
([Economic conventions](interpreting-results.md#economic-conventions)).

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
reuses them for every candidate and every projection year. They are not
config keys. BREOS ships no weather data, so point this at your own TMY or
historical CSV:

```python
import pandas as pd
from breos.load_profiles import load_profile

weather = pd.read_csv("weather/porto_tmy_2005_2023_pvgis-sarah3.csv", index_col=0)
weather.index = pd.to_datetime(weather.index, utc=True)

load = load_profile(
    "demandlib_h0",
    4000.0,
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

result = optimize_system_multi_objective(weather, load, config)

pareto = result.details["pareto"]
print(pareto[["Modules", "Battery_kWh", "Tilt", "Azimuth"]])
```

`pareto` is a DataFrame with one row per non-dominated design, holding the
sizing columns above, the objective values, ZEB diagnostics, and the
`Projected_*` fields. There is no single best row. Pick the design
whose balance of independence and cost matches the project.

{py:func}`~breos.plotting.plot_pareto_front` draws the front, two objectives
at a time: `plot_pareto_front(result, "plots", color_by="Battery_kWh")`.

The call reads `pop_size`, `n_offsprings`, `n_gen` and `seed` from
`[optimization]`. You can pass them as arguments instead; an argument that
disagrees with its key raises. Set and record the seed because NSGA-II is
stochastic; `details["provenance"]["run_settings"]` records what the search
used, and `details["provenance"]["constraints"]` its bounds. Raise the population and
generation counts for a denser front and a longer runtime. Pass `n_procs` to
`optimize_system_multi_objective` to evaluate candidates in parallel
processes.

If no candidate satisfies the constraints, the call raises `RuntimeError`.
Loosen `budget`, `max_area_m2`, `max_modules`, or `max_battery_kwh` in
`[constraints]` and run it again.

## Score designs over their projected lifetime

The optimizer scores each candidate over the whole configured horizon:

```toml
[optimization]
objective_basis = "projected"
```

The objectives are lifetime grid independence and lifetime NPV. BREOS computes
both values from the simulated annual ledgers rather than applying a
degradation factor after the simulation. ZEB is a diagnostic. To keep ZEB as a
feasibility constraint, set `enforce_zeb = true` under `[constraints]`.

Projected scoring simulates `years_projection` years for every candidate. Start
with a small `pop_size` and `n_gen` while you check that the config resolves,
then scale up. To screen a wide design space at lower cost, shorten
`years_projection` for the screening run. The single-year `steady_state` basis
was removed in 0.7.0, and a config that still sets it raises an error.

## Price a design with a time-of-use tariff

Add the same `[tariff]` table accepted by App to the optimizer config:

```toml
# Illustrative prices, not a supplier offer.
[tariff]
schedule = "pt_mainland_2026_daily_bi"
currency = "EUR"
import_prices = { peak = 0.28, off_peak = 0.11 }
export_prices = { all = 0.05 }
fixed_charge_per_day = 0.30
```

Remove `electricity_cost`, `electricity_sold_cost`, and `daily_power_cost`
from `[costs]` when you add `[tariff]`. Supplying both raises an error.
The schedule must match `location.timezone`. Schedules with half-hour or
quarter-hour boundaries require `simulation.resolution = "15min"`.
See [Tariffs](../api/tariffs.md) for effective dates and Spanish holiday coverage.

Both `evaluate_projected_design` and `optimize_system_multi_objective` price
each timestep through the shared projection loop. Each project year replays
the input calendar, with PV and battery degradation carried between years.
Tariff prices affect the financial objective. The battery follows
self-consumption dispatch unless a `smart_charging` table sets fixed-target
charging or `discharge_only`, which the optimizer applies as App does. It is checked as App checks
it, with `battery_kwh` taken from the design, or from
`constraints.max_battery_kwh` for a search. The instructions are resolved once
per search; a candidate without a battery ignores them. Results record them
in `provenance["smart_charging"]`. The experimental `daily_persistence` mode
plans each day while a single run goes, so the optimizer refuses it before
any candidate runs. See [Smart charging](configuration.md#smart-charging).

Fixed-design results record the schedule, prices, calendar and hashes in
`result.provenance["tariff"]`. Search results record the same fields in
`result.details["provenance"]["tariff"]`. Tariff-enabled searches also support
`n_procs`, and their results can be pickled.

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
