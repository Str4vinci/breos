# Get started

Install BREOS, run the quickstart, and collect the inputs a study needs. The
pages use the `breos.App` facade and the same configuration keys the
command-line interface accepts.

::::{grid} 1 2 2 2
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

:::{grid-item-card} Required inputs
:link: inputs
:link-type: doc

Identify the project weather, load, components, costs, and emissions data you
need for a defensible result.
:::

:::{grid-item-card} Troubleshooting
:link: troubleshooting
:link-type: doc

Resolve installation, weather access, configuration, cache, and runtime
problems.
:::

::::

## Next steps

- The [examples](../gallery/index.rst) are stored runs that answer common
  questions: what a battery adds, east-west roofs, tariffs, battery
  replacement, Monte Carlo and the optimizer front.
- The [user guide](../user-guide/index.md) covers configuration,
  optimization, Monte Carlo, reading the results, and short how-to guides.
- The [models](../modeling/index.md) pages state the physical boundaries,
  model choices and data sources to check before drawing conclusions.

```{toctree}
:hidden:

installation
quickstart
inputs
troubleshooting
```
