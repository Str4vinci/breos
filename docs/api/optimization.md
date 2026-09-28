# Optimization

Optimization helpers for system configuration. The supported tilt grid search
and battery-sizing helper cover one-dimensional sizing;
[pymoo](https://pymoo.org/) powers public multi-objective PV/battery sizing (PV
count, battery, cost, grid independence, and ZEB ratio). For end-to-end App
runs over an explicit config grid, use the `breos sweep` CLI command documented
in [Recipes](../getting-started/recipes.md#parameter-sweep).

Install `breos[optimization]` to use pymoo-backed multi-objective sizing.
The one-dimensional helpers use the core scientific stack.

ZEB and financial production use usable AC system energy from the dispatch
ledger, not raw PV DC, so inverter efficiency and clipping affect candidate
scores. Physical size, inverter rating, and CAPEX use the selected module's
`Mpp`.

## Projected objectives

`optimize_system_multi_objective` evaluates every candidate over
`simulation.years_projection` years, or `financials.project_lifespan` when that
key is absent. It optimizes two values: projected lifetime grid independence
and projected NPV. A design is selected for how it performs across the project
lifetime, which is the question a sizing study asks.
`optimization.objective_basis = "projected"` names this basis and is the only
accepted value. The annual `"steady_state"` basis was removed in 0.7.0, and a
config that still sets it raises an error.

Projected mode repeats the configured TMY. Each year applies the configured PV
degradation factor and carries battery stored energy, PV-origin stored energy,
SoH, full-equivalent cycles, calendar time, cycle and calendar degradation,
resistance growth, and supported degradation-engine state into the next year.
Replacement remains enabled, resets the battery through the production battery
engine, and adds the actual event cost to that year's financial ledger.

Lifetime grid independence is calculated from aggregate energy, not from the
mean annual percentage:

```text
GI_lifetime = 100 × (1 − total lifetime grid import / total lifetime load)
```

Projected NPV uses each simulated year's import, export, load, usable AC PV
production, and replacement cost. This is a repeated-TMY scenario, not a
forecast of distinct future weather years.

An optional `tariff` table uses the same schema and validation as App. The
shared projection loop values imports, exports and the no-system baseline at
the timestep prices and bills the fixed charge by simulated duration. The
table replaces the three flat energy/fixed-charge entries in `costs`; giving
both raises. The schedule is resolved once per search in `location.timezone`.
The [optimization guide](../getting-started/optimization.md#price-a-design-with-a-time-of-use-tariff)
shows the configuration and provenance fields. Smart charging is not yet
supported and is rejected.

ZEB remains a reported diagnostic in projected mode. Set
`constraints.enforce_zeb = true` to require a projected lifetime ZEB ratio of
at least one; this adds a feasibility constraint, not a third objective.

Multi-objective sizing accepts these explicit constraint keys:

| Key | Default | Meaning |
| --- | ---: | --- |
| `constraints.budget_eur` | 10,000 | Maximum initial system cost in EUR |
| `constraints.max_area_m2` | 20 | Maximum PV-module frame area in m² |
| `constraints.max_battery_kwh` | 30 | Maximum battery decision-variable value in kWh |
| `constraints.max_modules` | 60 | Maximum PV-module decision-variable value |
| `constraints.max_tilt_deg` | 90 | Maximum tilt, or `"adjust"` for the latitude-based bound |
| `constraints.enforce_zeb` | `false` | Add the ZEB feasibility constraint |

Set the physical and financial limits explicitly for any study you intend to
report. The defaults preserve earlier direct-API behavior; they are not
site-specific recommendations.

Results expose `Projected_*` diagnostics. The ordinary `Grid_Independence_%`
and `NPV_Eur` columns equal `Projected_Grid_Independence_%` and
`Projected_NPV_Eur`. `ZEB_Ratio` mirrors `Projected_ZEB_Ratio` but is not an
objective. `Objective_Grid_Independence_%` and `Objective_NPV_Eur` identify the
metrics sent to NSGA-II explicitly.

Use `evaluate_projected_design` when you need the detailed result for one
fixed design instead of a Pareto search. It returns the projected metrics, the
annual energy and degradation-state ledger, and the matching discounted
financial ledger. LCOE and configured lifetime avoided-emissions totals are
included in the metrics, while the financial table retains their annual source
columns. These tables are intended as stable source data for custom
analysis and plots; BREOS does not require a particular visualization layer.

## Reproducing the upcoming publication

The configurations, drivers, and run records for the upcoming publication
were removed from the repository after 0.6.2. They are preserved in the
[BREOS 0.6.2 archive](https://doi.org/10.5281/zenodo.22938914) and at the
[`v0.6.2` tag](https://github.com/Str4vinci/breos/tree/v0.6.2), under
`validation/article1/` and `tools/`. Reproduce the published numbers from that
release, not from a later version.

## Tilt

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.optimization.optimize_tilt
```

## Multi-objective sizing

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.optimization.optimize_system_multi_objective
   breos.optimization.evaluate_projected_design
```

## Battery sizing

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.optimization.optimize_battery_size
```

## Result type

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.optimization.OptimizationResult
   breos.optimization.ProjectedDesignResult
```
