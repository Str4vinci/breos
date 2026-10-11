# User guide

How to configure a study, search for a design, sample uncertainty, and read
the result. The how-to guides give short configurations to copy for specific
tasks.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Configuration
:link: ../getting-started/configuration
:link-type: doc

Understand every `App` config area, its defaults, and its validation rules.
:::

:::{grid-item-card} Optimization
:link: ../getting-started/optimization
:link-type: doc

Search for a design instead of specifying one: NSGA-II sizing, projected-lifetime
objectives, and detailed evaluation of a chosen candidate.
:::

:::{grid-item-card} Monte Carlo
:link: ../getting-started/monte-carlo
:link-type: doc

Resample weather years and demand to get outcome distributions rather than one
number, with reproducible seeds and provenance.
:::

:::{grid-item-card} Interpreting results
:link: ../getting-started/interpreting-results
:link-type: doc

Read headline KPIs and the yearly, monthly, financial, degradation, and
provenance blocks.
:::

:::{grid-item-card} How-to guides
:link: ../how-to/index
:link-type: doc

Configurations to copy for custom coordinates, your own PV module, partial
years, sweeps, 15-minute runs, external load profiles and offline weather.
:::

::::

Every configuration key, with its default and allowed values, is in the
[configuration key reference](../getting-started/config-reference.md), and the
bundled locations, modules, cost presets and load profiles are on the
[packaged options](../getting-started/options.md) page.

```{toctree}
:hidden:

/getting-started/configuration
/getting-started/optimization
/getting-started/monte-carlo
/getting-started/interpreting-results
/how-to/index
```
