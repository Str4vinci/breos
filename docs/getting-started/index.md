# Guides and recipes

Start with the installation check and quickstart, then browse by the task you
are trying to complete. The guides use the `breos.App` facade and the same
configuration keys accepted by the command-line interface.

## First run

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Install BREOS
:link: installation
:link-type: doc

Install from PyPI or source, choose optional extras, and verify the command-line
entry point without network access.
:::

:::{grid-item-card} Quickstart
:link: quickstart
:link-type: doc

Run a small PV + battery simulation from TOML, Python, or the command line.
:::

:::{grid-item-card} Troubleshooting
:link: troubleshooting
:link-type: doc

Resolve installation, weather access, configuration, cache, and runtime
problems.
:::

::::

## Build a study

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Required inputs
:link: inputs
:link-type: doc

Identify the project weather, load, components, costs, and emissions data you
need for a defensible result.
:::

:::{grid-item-card} Recipes
:link: recipes
:link-type: doc

Copy working configurations for PV-only, PV + battery, east-west roofs,
external load profiles, sweeps, and offline runs.
:::

:::{grid-item-card} Optimization
:link: optimization
:link-type: doc

Search for a design instead of specifying one: NSGA-II sizing, projected-lifetime
objectives, and detailed evaluation of a chosen candidate.
:::

:::{grid-item-card} Monte Carlo
:link: monte-carlo
:link-type: doc

Resample weather years and demand to get outcome distributions rather than one
number, with reproducible seeds and provenance.
:::

:::{grid-item-card} Configuration
:link: configuration
:link-type: doc

Understand every `App` config area, its defaults, and its validation rules.
:::

:::{grid-item-card} Packaged options
:link: options
:link-type: doc

Browse the bundled locations, modules, costs, emissions factors, and load
profiles.
:::

:::{grid-item-card} Interpreting results
:link: interpreting-results
:link-type: doc

Read headline KPIs and the yearly, monthly, financial, degradation, and
provenance blocks.
:::

:::{grid-item-card} Models and assumptions
:link: ../modeling/index
:link-type: doc

Understand physical boundaries, model choices, validation policy, and data
sources before drawing conclusions.
:::

::::
