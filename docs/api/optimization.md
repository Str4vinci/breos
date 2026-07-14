# Optimization

Optimization helpers for system configuration. Brent's method handles smooth
one-dimensional problems (tilt); helper sweeps handle battery sizing and ZEB
sizing; [pymoo](https://pymoo.org/) powers public multi-objective PV/battery
sizing (PV count, battery, cost, grid independence, and ZEB ratio). For
end-to-end App runs over an explicit config grid, use the `breos sweep` CLI
command documented in [Recipes](../getting-started/recipes.md#parameter-sweep).

Install `breos[optimization]` to use pymoo-backed multi-objective sizing.
The one-dimensional helpers use the core scientific stack.

ZEB and financial production use usable AC system energy from the dispatch
ledger, not raw PV DC, so inverter efficiency and clipping affect candidate
scores. Physical size, inverter rating, and CAPEX use the selected module's
`Mpp`; an explicit `costs.panel_wp` remains a deliberate cost-model override.
The optimizer reports its battery-replacement treatment in result metadata;
the App's multi-year projection remains the authoritative full trajectory.

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
```
