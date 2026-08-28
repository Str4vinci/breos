---
sd_hide_title: true
---

# BREOS

```{image} _static/BREOS_black.svg
:alt: BREOS logo
:width: 220px
:align: center
:class: only-light
```

```{image} _static/BREOS.png
:alt: BREOS logo
:width: 220px
:align: center
:class: only-dark
```

::::{div} sd-text-center sd-fs-2 sd-fw-bold
BREOS
::::

::::{div} sd-text-center sd-fs-4
Building Renewable Energy Optimization Software
::::

::::{div} sd-text-center sd-py-3
A Python library for PV and battery energy-system simulation and
optimization, designed for research and engineering applications.
::::

## Quick example

```bash
pip install breos
```

```python
import breos

app = breos.App({
    "location": "porto",
    "n_modules": 10,
    "annual_consumption_kwh": 4000,
    "battery_kwh": 5.0,
    "cost_preset": "residential_pt",
    "emissions_country": "PT",
})
app.simulate()
result = app.result()

print(f"Grid independence: {result['grid_independence_pct']:.1f}%")
print(f"Payback: {result['payback_year']} years")
print(f"NPV savings: {result['npv_savings_eur']:,.0f} EUR")
```

`result` is a plain JSON-serializable dict — no pandas types leak out.

## Start with a task

::::{grid} 1 2 2 4
:gutter: 3

:::{grid-item-card} Run your first simulation
:link: getting-started/quickstart
:link-type: doc

Verify the install, run a small PV + battery study, and inspect its headline
results.
:::

:::{grid-item-card} Configure a study
:link: getting-started/configuration
:link-type: doc

Choose the location, PV layout, battery, economics, emissions, and model
options.
:::

:::{grid-item-card} Bring your own data
:link: getting-started/inputs
:link-type: doc

Use project weather, load profiles, component data, and financial assumptions.
:::

:::{grid-item-card} Understand the result
:link: getting-started/interpreting-results
:link-type: doc

Read the energy, financial, emissions, degradation, and provenance fields.
:::

::::

## Browse the documentation

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Guides and recipes
:link: getting-started/index
:link-type: doc

Task-oriented guides for common systems, inputs, configuration, and
troubleshooting.
:::

:::{grid-item-card} Models and assumptions
:link: modeling/index
:link-type: doc

Physical boundaries, model choices, data sources, and degradation methods.
:::

:::{grid-item-card} Python API
:link: api/index
:link-type: doc

The stable `breos.App` facade and lower-level functions organized by domain.
:::

::::

## Status

BREOS is pre-1.0 (beta), so the public API may change between minor releases.
The `breos.App` facade is the most stable surface to build on. Use the version
selector to read the docs matching your installation: `stable` follows the
latest published release, while `latest` follows active development. The
documentation describes implemented behavior rather than proposed designs.

## Project direction

The public [roadmap](https://github.com/Str4vinci/breos/blob/main/ROADMAP.md)
shows release intent and larger capabilities under consideration. Detailed
implementation plans, ADRs, and maintainer procedures live in the repository
rather than on this user-documentation site.

```{toctree}
:hidden:
:caption: Learn BREOS

getting-started/index
getting-started/installation
getting-started/quickstart
getting-started/troubleshooting
getting-started/inputs
getting-started/recipes
getting-started/optimization
getting-started/monte-carlo
getting-started/configuration
getting-started/options
getting-started/interpreting-results
```

```{toctree}
:hidden:
:caption: Models and data

modeling/index
resources
legal/load-profile-data
```

```{toctree}
:hidden:
:caption: Reference

api/index
```

```{toctree}
:hidden:
:caption: Project

changelog
```
