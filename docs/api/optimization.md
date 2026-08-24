# Optimization

Optimization helpers for system configuration. The supported tilt grid search
and battery-sizing helper cover one-dimensional sizing;
[pymoo](https://pymoo.org/) powers public multi-objective PV/battery sizing (PV
count, battery, cost, grid independence, and ZEB ratio). For end-to-end App
runs over an explicit config grid, use the `breos sweep` CLI command documented
in [Recipes](../getting-started/recipes.md#parameter-sweep).

Install `breos[optimization]` to use pymoo-backed multi-objective sizing.
The one-dimensional helpers use the core scientific stack.

The Brent tilt helper and both standalone ZEB sizing helpers are scheduled for
removal in 0.6.0. See [Deprecations for 0.6.0](../deprecations.md).

ZEB and financial production use usable AC system energy from the dispatch
ledger, not raw PV DC, so inverter efficiency and clipping affect candidate
scores. Physical size, inverter rating, and CAPEX use the selected module's
`Mpp`; an explicit `costs.panel_wp` remains a deliberate cost-model override.

## Annual and projected objectives

`optimize_system_multi_objective` supports two objective bases:

- `optimization.objective_basis = "steady_state"` is the default. It preserves
  the established annual three-objective search: grid independence, NPV, and
  ZEB ratio. Battery replacement is estimated from the first-year SoH loss.
- `optimization.objective_basis = "projected"` evaluates every candidate over
  `simulation.years_projection` years, or `financials.project_lifespan` when
  that key is absent. It optimizes two values: projected lifetime grid
  independence and projected NPV.

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

Set the physical and financial limits explicitly for publication runs. The
defaults preserve earlier direct-API behavior; they are not site-specific
recommendations. The Article configuration pins every applicable limit.

Projected results expose `SteadyState_*` and `Projected_*` diagnostics. The
ordinary `Grid_Independence_%` and `NPV_Eur` columns mirror the values used by
the selected objective basis. In projected mode, they therefore equal
`Projected_Grid_Independence_%` and `Projected_NPV_Eur`. `ZEB_Ratio` mirrors
the corresponding diagnostic but is not an objective. `Objective_*` columns
identify the metrics sent to NSGA-II explicitly; projected output has no
`Objective_ZEB_Ratio` column.

Use `evaluate_projected_design` when you need the detailed result for one
fixed design instead of a Pareto search. It returns the projected metrics, the
annual energy and degradation-state ledger, and the matching discounted
financial ledger. LCOE and configured lifetime avoided-emissions totals are
included in the metrics, while the financial table retains their annual source
columns. These tables are intended as stable source data for custom
analysis and plots; BREOS does not require a particular visualization layer.

## Article 1 reproduction

[`validation/article1/article1-projected-optimization.toml`](../../validation/article1/article1-projected-optimization.toml)
pins the Article 1 15-minute, 20-year configuration, four archived comparison
candidates plus the C5 low-investment benchmark, NSGA-II seed and early
stopping, battery degradation, replacement, and financial assumptions. Its hourly TMY is interpolated with the clear-sky
shape and opt-in hourly-energy conservation; energy conservation remains
opt-in for general resampling. The E-REDES household profile is
licensed external data and is not redistributed.

Run the deterministic fixed candidates before starting NSGA-II:

```bash
uv run python tools/reproduce_article1.py \
  --rlp-directory /path/to/licensed/rlp \
  --output results/article1
```

Add `--full-optimization` to run the configured population. Use
`--smoke-optimization` for a four-candidate generation and `--n-procs N` to
evaluate NSGA-II candidates in parallel. The command writes CSV results plus
`reproduction.json`, which records the resolved config, exact
source revision and dirty flag, command, software versions, and input hashes.
Each fixed candidate also gets `yearly_summary.csv`, `cost_projection.csv`,
and `metrics.json`, with their hashes recorded in the report.

Use `--calendar-model naumann_lam_field_calibrated_v2` or
`--calendar-model naumann_lam` to repeat fixed-design or full optimization
runs with the field-v2 or laboratory calendar-degradation parameters. Put each
model in a separate `--output` directory.
The [reproduction report](https://github.com/Str4vinci/breos/blob/develop/validation/article1/README.md)
explains the numerical changes caused by post-study corrections.

## Tilt

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.optimization.optimize_tilt
   breos.optimization.optimize_tilt_brent
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
   breos.optimization.size_for_zeb
```

## Result type

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.optimization.OptimizationResult
   breos.optimization.ProjectedDesignResult
```
