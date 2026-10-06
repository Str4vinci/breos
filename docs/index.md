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

BREOS is designed for multi-year PV and storage studies where long-term
dynamics govern feasibility: multi-decade battery degradation, tariff
optimization under time-of-use pricing, and performance sensitivity across
historical weather years. It ships with calibrated presets and benchmarks,
while allowing every layer to be replaced with your own data: weather files,
load profiles, PV modules, degradation coefficients, costs, and tariffs.

## What BREOS does

- **Lifetime simulation.** PV production through pvlib and battery dispatch
  at 15-minute or hourly resolution, repeated over every project year with
  the battery ageing as it goes, from native LFP fits or BLAST-Lite cell
  models. See [Battery ageing](gallery/battery/plot_07_ageing.rst).
- **Tariffs and battery control.** Flat and time-of-use prices, a separate
  tariff for the household without the system, opt-in grid charging, and
  battery replacements booked in the cash flow. See
  [Which tariff](gallery/tariffs/plot_09_which_tariff.rst) and
  [Replacement timing](gallery/battery/plot_08_replacement_timing.rst).
- **Uncertainty.** Monte Carlo over historical weather years and annual
  demand, reported as distributions of NPV, payback and battery health. See
  [Monte Carlo](gallery/uncertainty/plot_13_montecarlo.rst).
- **Sizing.** Parameter sweeps and multi-objective PV and battery sizing with
  NSGA-II. See [NSGA-II front](gallery/uncertainty/plot_14_nsga2_front.rst).

The documentation assumes working knowledge of Python and pandas. Studies can
also run from a TOML file with the `breos` command line, without writing
Python.

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

## Citing BREOS

If you use BREOS in published work, please cite the software and the version
you used. The DOI
[10.5281/zenodo.22938913](https://doi.org/10.5281/zenodo.22938913) always
resolves to the latest release, and each release has its own DOI on
[Zenodo](https://doi.org/10.5281/zenodo.22938913).
[`CITATION.cff`](https://github.com/Str4vinci/breos/blob/main/CITATION.cff)
holds the full metadata, which GitHub's "Cite this repository" button reads.

```bibtex
@software{rodrigues_breos,
  author  = {Rodrigues, Leonardo},
  title   = {{BREOS}: Building Renewable Energy Optimization Software},
  doi     = {10.5281/zenodo.22938913},
  url     = {https://github.com/Str4vinci/breos},
  version = {X.Y.Z}
}
```

PV production in BREOS comes from
[pvlib python](https://pvlib-python.readthedocs.io/), so please cite pvlib
alongside BREOS when your results depend on it. BREOS also builds on
BLAST-Lite, demandlib and pymoo;
[ATTRIBUTIONS.md](https://github.com/Str4vinci/breos/blob/main/ATTRIBUTIONS.md)
lists every upstream project and data source with its license.

## Getting help

- Questions about using BREOS, and ideas for features:
  [GitHub Discussions](https://github.com/Str4vinci/breos/discussions).
- Bugs and wrong results: [GitHub issues](https://github.com/Str4vinci/breos/issues).
  Include the BREOS version and the configuration that shows the problem.
- Research collaboration and private enquiries: lrodrigues@fe.up.pt.

## Status

BREOS is pre-1.0 (beta), so the public API may change between minor releases.
The `breos.App` facade is the most stable surface to build on. Use the version
selector to read the docs matching your installation: `stable` follows the
latest published release, while `latest` follows active development. The
documentation describes implemented behavior rather than proposed designs.
BREOS is released under the BSD 3-Clause license.

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
