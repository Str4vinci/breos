# API reference

BREOS's API reference is organized by domain area rather than by Python
module — the [PV reference](pv.md) pulls names from `breos.solar`,
`breos.pv_modules`, and `breos.inverter`, since all three describe the same
part of a system.

The {py:class}`~breos.App` facade is the recommended entry point. The
top-level `breos.__all__` list is intentionally narrower than the full
reference: it marks the stable release surface for `from breos import *`.
The pages below cover lower-level module APIs you reach for when composing a
custom pipeline. Import those lower-level names from their modules, for example
`from breos.solar import calculate_pv_production_dc`.

## Domain areas

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

Per-timestep PV, load, battery, and grid energy flows.
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

Tilt search, battery sizing, ZEB sizing, and NSGA-II multi-objective sizing.
:::

:::{grid-item-card} Plotting
:link: plotting
:link-type: doc

Publication-ready figures for production, costs, degradation, and Pareto
fronts.
:::

::::

## Other surfaces

[Appendix](appendix.md) documents additional module APIs — constants, I/O
helpers, utilities, and research-validation modules that remain importable but
are not part of the narrow top-level release surface.

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
