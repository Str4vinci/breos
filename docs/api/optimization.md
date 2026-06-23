# Optimization

Optimization helpers for system configuration. Brent's method handles smooth
one-dimensional problems (tilt); simple sweeps handle battery sizing and ZEB
sizing; [pymoo](https://pymoo.org/) powers public multi-objective PV/battery
sizing (PV count, battery, cost, grid independence, and ZEB ratio).

Install `breos[optimization]` to use pymoo-backed multi-objective sizing.
The one-dimensional helpers use the core scientific stack.

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
