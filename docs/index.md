---
sd_hide_title: true
og:description: "Open-source Python library that simulates rooftop PV and battery systems for buildings: energy flows, battery ageing, economics and emissions over the project lifetime."
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
BREOS is an open-source Python library that simulates rooftop PV and battery
systems for buildings and reports their energy flows, battery ageing,
economics and emissions over the project lifetime.
::::

```{figure} _static/landing-energy-balance.png
:alt: Monthly usable PV production, load, grid import and grid export for one year
:align: center

Monthly energy balance in the first year of a 10-module system with a 5 kWh
battery in Porto, from the
[first case example](gallery/getting_started/plot_01_first_home.rst).
```

## Install and run

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

print(f"Grid independence: {result['grid_independence_pct']:.1f} %")
print(f"Payback: {result['payback_year']} years")
print(f"NPV savings: {result['npv_savings']:.0f} EUR")
```

`result` is a plain JSON-serializable dict. The same study runs from a TOML
file with `breos run --config config.toml`; the
[quickstart](getting-started/quickstart.md) shows both.

## Documentation

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} Get started
:link: getting-started/index
:link-type: doc

Install BREOS, run the quickstart, and gather the weather, load, component
and cost inputs a study needs.
:::

:::{grid-item-card} Examples
:link: gallery/index
:link-type: doc

Stored runs that answer one question each, with their configurations and
figures.
:::

:::{grid-item-card} User guide
:link: user-guide/index
:link-type: doc

Configuration, optimization, Monte Carlo, reading the results, and short
how-to guides.
:::

:::{grid-item-card} Models
:link: modeling/index
:link-type: doc

Physical boundaries, model choices, data sources, degradation methods and
external validation.
:::

:::{grid-item-card} API reference
:link: api/index
:link-type: doc

The `breos.App` facade, the lower-level functions by domain, and every
configuration key.
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

Get started <getting-started/index>
Examples <gallery/index>
User guide <user-guide/index>
Models <modeling/index>
API reference <api/index>
changelog
legal/load-profile-data
```
