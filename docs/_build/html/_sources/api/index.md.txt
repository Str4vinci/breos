# API reference

BREOS's public API is organized by domain "puzzle piece" rather than by
Python module — the [PV reference](pv.md) pulls names from `breos.solar`,
`breos.pv_modules`, and `breos.inverter`, since all three describe the same
piece of a system.

The {py:class}`~breos.App` facade is the recommended entry point. The
pages below cover the lower-level functions you reach for when composing a
custom pipeline.

## Puzzle pieces

::::{grid} 1 2 2 4
:gutter: 3

:::{grid-item-card} Weather
:link: weather
:link-type: doc

TMY and historical data, file I/O, resampling, clear-sky scaling.
:::

:::{grid-item-card} PV
:link: pv
:link-type: doc

DC and AC production, multi-array layouts, module catalogue, inverter
sizing.
:::

:::{grid-item-card} Load profiles
:link: load-profiles
:link-type: doc

Standard residential and commercial profiles, scaling, alignment.
:::

:::{grid-item-card} Energy balance
:link: energy-balance
:link-type: doc

Per-timestep dispatch of PV, load, battery, and grid.
:::

:::{grid-item-card} Battery
:link: battery
:link-type: doc

Configuration, indoor temperature model, calendar and cycle degradation.
:::

:::{grid-item-card} Cost analysis
:link: cost-analysis
:link-type: doc

Cost parameters, NPV / LCOE / payback projections, emissions.
:::

:::{grid-item-card} Optimization
:link: optimization
:link-type: doc

Tilt search, battery sizing, NSGA-II multi-objective Pareto.
:::

:::{grid-item-card} Plotting
:link: plotting
:link-type: doc

Publication-ready figures for production, costs, degradation, and Pareto
fronts.
:::

::::

## Other surfaces

[Appendix](appendix.md) documents the remaining public names — constants,
I/O helpers, utilities, and other modules that are exposed but not
load-bearing for typical use.

```{toctree}
:hidden:

weather
pv
load-profiles
energy-balance
battery
cost-analysis
optimization
plotting
appendix
```
