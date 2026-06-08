# Optimization

Single- and multi-objective optimization for system configuration. Brent's
method handles smooth one-dimensional problems (tilt); NSGA-II via
[pymoo](https://pymoo.org/) handles multi-objective sizing (PV count vs
battery vs cost vs grid independence).

Install `breos[optimization]` to use pymoo-backed multi-objective classes.
The one-dimensional helpers use the core scientific stack.

## Tilt

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.optimization.optimize_tilt
   breos.optimization.optimize_tilt_brent
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
