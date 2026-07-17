# BREOS - Building Renewable Energy Optimization Software

[![Tests](https://github.com/Str4vinci/breos/actions/workflows/tests.yml/badge.svg)](https://github.com/Str4vinci/breos/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/breos.svg)](https://pypi.org/project/breos/)
[![Docs](https://img.shields.io/badge/docs-readthedocs-blue.svg)](https://breos.readthedocs.io/)
[![License: BSD-3](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

BREOS is a Python library for simulating and optimizing household PV + battery
energy systems (weather, PV production, battery aging, economics, emissions,
and multi-objective sizing) behind one stable `breos.App` facade, with
lower-level modules for building custom study pipelines.

**📖 Full documentation: [breos.readthedocs.io](https://breos.readthedocs.io/)**

## Features

- **Weather** — TMY from PVGIS/NSRDB and historical data from Open-Meteo, at hourly or 15-minute resolution.
- **PV production** — pvlib CEC single-diode model, with a small example module catalog to get started.
- **Multi-array roofs** — combine multiple faces/orientations (e.g. east-west) at the DC stage instead of one representative tilt.
- **Battery** — energy balance with calendar + cycle aging (Naumann 2020, Lam 2025) and field-calibrated LFP parameters.
- **Economics** — NPV, LCOE, breakeven, and cost projections with configurable tariffs and inflation.
- **Monte Carlo** — weather-year and demand resampling for NPV, payback, grid-independence, LCOE, and SoH distributions.
- **Optimization** — multi-objective PV/battery sizing (pymoo NSGA-II), tilt optimization, and sizing sweeps.
- **Emissions** — CO<sub>2</sub> savings and projections.
- **Visualization** — publication-ready plots for energy balances, degradation, breakeven, and Pareto fronts.
- **Bring your own data** — every layer accepts custom inputs: PV module parameters, battery degradation coefficients, weather CSVs, residential load profiles, and cost/tariff/emissions assumptions. The packaged presets are starting points, not fixed defaults.

## Installation

```bash
pip install breos
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add breos          # as a project dependency
uvx breos --version   # run the CLI without installing
```

The default install is a lean core. Some workflows need optional extras (e.g.
optimization, historical weather, plots):

```bash
pip install "breos[optimization,weather,plots]"
```

See the [installation guide](docs/getting-started/installation.md)
for the full list of extras and a source/`uv` setup.

## Quick Start

```python
import breos

app = breos.App({
    "location": "porto",              # preset or {"latitude": ..., "longitude": ..., "timezone": ...}
    "n_modules": 10,
    "annual_consumption_kwh": 4000,
    "battery_kwh": 5.0,               # 0 for no battery
    "cost_preset": "residential_pt",
    "emissions_country": "PT",
})

app.simulate()
result = app.result()

print(f"Grid independence: {result['grid_independence_pct']:.1f}%")
print(f"Payback: {result['payback_year']} years")
print(f"NPV savings: {result['npv_savings_eur']:,.0f} EUR")
```

`result()` returns a plain JSON-serializable dict. The
[configuration reference](docs/getting-started/configuration.md)
lists every option, and
[interpreting results](docs/getting-started/interpreting-results.md)
documents every output field.

> For real studies, bring your own weather/API access where required (NSRDB
> needs an NREL API key; PVGIS and Open-Meteo do not), licensed load profiles,
> and your own cost/tariff assumptions. The packaged defaults make the tool
> runnable, not project-grade.

## Command Line

Run a simulation without writing Python:

```bash
breos run --location porto --n-modules 10 --annual-consumption-kwh 4000 \
  --battery-kwh 5.0 --cost-preset residential-pt --emissions-country pt \
  --output result.json
```

The CLI also drives config files, parameter sweeps, and Monte Carlo studies, and
`breos list <category>` shows bundled presets (locations, modules, cost presets,
…). See the [CLI recipes](docs/getting-started/recipes.md).

## Citation

If you use BREOS in your research, please cite the preprint:

```bibtex
@misc{rodrigues2026breos,
  author = {Rodrigues, L. and Delgado, J. M. P. Q. and Mendes, A. and Guimar{\~a}es, A. S.},
  title  = {A Modular, Open-Source Python Framework for Household PV-Battery Sizing: Validation, Multi-Objective Optimisation, and Uncertainty Analysis},
  year   = {2026},
  doi    = {10.2139/ssrn.7032064},
  url    = {https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7032064},
  note   = {SSRN preprint}
}
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Feature work happens on branches off
`develop` and merges via pull request; `main` tracks stable releases only.

## Contact

For questions, collaboration, or access to additional modules: lrodrigues@fe.up.pt.

## License

BSD 3-Clause. See [LICENSE](LICENSE).
