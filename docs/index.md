---
sd_hide_title: true
---

# BREOS

::::{div} sd-text-center sd-fs-2 sd-fw-bold
BREOS
::::

::::{div} sd-text-center sd-fs-4
Building Renewable Energy Optimization Software
::::

::::{div} sd-text-center sd-py-3
A modular Python library for photovoltaic and battery energy system
simulations, designed for research and engineering applications.
::::

## Quick example

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

## Where to start

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Getting started
:link: getting-started/index
:link-type: doc

Install BREOS, run a first simulation, and learn what every key in the
result dict means.
:::

:::{grid-item-card} API reference
:link: api/index
:link-type: doc

Every public function and class, organized by what it does — weather, PV,
battery, costs, and so on.
:::

:::{grid-item-card} Architecture
:link: architecture/third-party-wrapping
:link-type: doc

Design decisions and refactoring plans for BREOS's internal structure.
:::

::::

## Status

BREOS is pre-1.0 (currently `0.1.0`, beta). The public API may change
between minor releases. The [roadmap](https://github.com/Str4vinci/breos/blob/main/ROADMAP.md)
tracks larger refactoring work — notably, the planned [adapter layer for
third-party libraries](architecture/third-party-wrapping.md) will change
the signatures of `calculate_pv_production_dc` and similar functions that
currently expose `pvlib.Location` directly. The `breos.App` facade is
the most stable surface to build on.

```{toctree}
:hidden:
:caption: Getting started

getting-started/index
```

```{toctree}
:hidden:
:caption: Reference

api/index
```

```{toctree}
:hidden:
:caption: Project

architecture/third-party-wrapping
adr/index
changelog
```
