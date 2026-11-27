<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Str4vinci/breos/develop/docs/_static/BREOS.png">
    <img alt="BREOS logo" src="https://raw.githubusercontent.com/Str4vinci/breos/develop/docs/_static/BREOS_black.png" width="200">
  </picture>
</p>

# BREOS - Building Renewable Energy Optimization Software

[![Tests](https://github.com/Str4vinci/breos/actions/workflows/tests.yml/badge.svg)](https://github.com/Str4vinci/breos/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/breos.svg)](https://pypi.org/project/breos/)
[![Docs](https://img.shields.io/badge/docs-readthedocs-blue.svg)](https://breos.readthedocs.io/)
[![License: BSD-3](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://github.com/Str4vinci/breos/blob/main/LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

BREOS simulates and sizes PV + battery systems for buildings: weather, PV
production, battery aging, economics, emissions, and multi-objective sizing.
Use it from the command line or from Python. The PV modeling is built on
[pvlib python](https://pvlib-python.readthedocs.io/).

**Documentation: [breos.readthedocs.io](https://breos.readthedocs.io/)**

## Quick Start

```bash
pip install breos
```

Simulate 10 PV modules and a 5 kWh battery for a house in Porto that uses
4000 kWh a year:

```bash
breos run --location porto \
  --n-modules 10 \
  --annual-consumption-kwh 4000 \
  --battery-kwh 5 \
  --cost-preset residential-pt \
  --emissions-country pt
```

The run fetches a typical weather year from PVGIS and prints a summary:

```text
Year 1 energy
  PV delivered (AC)             8143 kWh
  House load                    4000 kWh
  Self-consumed PV              3157 kWh       38.8 % of delivered PV
  Grid import                    839 kWh       79.0 % grid independence
  Grid export                   4986 kWh
...
Economics, 20 years, EUR, 3 % discount rate
  Investment                    7789 EUR
  Net present value             4888 EUR
  Discounted payback              10 years
```

Add `--output result.json` to save every result field. The
[quickstart](https://breos.readthedocs.io/en/latest/getting-started/quickstart.html)
shows the full summary and how to keep the setup in a config file.

The same run from Python:

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

print(f"Grid independence: {result['grid_independence_pct']:.1f} %")
print(f"Payback: {result['payback_year']} years")
print(f"NPV savings: {result['npv_savings']:.0f} EUR")
```

`result()` returns a plain JSON-serializable dict. Money fields are in the
currency recorded in `result["provenance"]["currency"]` (EUR for the bundled
cost presets). The
[configuration reference](https://breos.readthedocs.io/en/latest/getting-started/configuration.html)
lists every option, and
[interpreting results](https://breos.readthedocs.io/en/latest/getting-started/interpreting-results.html)
documents every output field.

> For real studies, bring your own weather and API access where required,
> licensed load profiles,
> and your own cost/tariff assumptions. The packaged defaults make the tool
> runnable, not project-grade.

## Installation

The default install is a lean core. Optimization, historical weather, and
plots need extras:

```bash
pip install "breos[optimization,weather,plots]"
```

With [uv](https://docs.astral.sh/uv/):

```bash
uv add breos          # as a project dependency
uvx breos --version   # run the CLI without installing
```

See the [installation guide](https://breos.readthedocs.io/en/latest/getting-started/installation.html)
for the full list of extras and a source setup.

## Features

- **Weather:** TMY from PVGIS, local weather files, and historical data from Open-Meteo at hourly or 15-minute resolution.
- **PV production:** pvlib CEC single-diode model, with a small example module catalog to get started.
- **Multi-array systems:** combine multiple faces/orientations (e.g. an east-west roof) at the DC stage instead of one representative tilt.
- **Battery:** energy balance with calendar + cycle aging (Naumann 2020, Lam 2025) and field-calibrated LFP parameters, or one of 14 vendored BLAST-Lite aging models.
- **Battery control:** greedy self-consumption by default; opt-in fixed-target grid charging and discharge-only windows under time-of-use prices.
- **Economics:** NPV, LCOE, breakeven, and cost projections with flat or time-of-use tariffs, a separate tariff for the no-system baseline, and separate price escalators.
- **Monte Carlo:** weather-year and demand resampling for NPV, payback, grid-independence, LCOE, and SoH distributions.
- **Optimization:** multi-objective PV/battery sizing and orientation (pymoo NSGA-II), and parameter-grid sweeps.
- **Emissions:** CO<sub>2</sub> savings and projections.
- **Visualization:** plots for energy balances, degradation, breakeven, Monte Carlo distributions, sweep heatmaps, orientation landscapes and Pareto fronts.
- **Bring your own data:** every layer accepts custom inputs (PV module parameters, battery degradation coefficients, weather CSVs, load profiles, and cost/tariff/emissions assumptions). The packaged presets are starting points, not fixed defaults.

The CLI also drives config files, parameter sweeps, and Monte Carlo studies, and
`breos list <category>` shows bundled presets (locations, modules, cost presets,
…). See the [how-to guides](https://breos.readthedocs.io/en/latest/how-to/index.html) and the
[case examples](https://breos.readthedocs.io/en/latest/gallery/index.html).

## Citation

If you use BREOS in your research, please cite the software and the version you
used. GitHub's "Cite this repository" button reads the metadata in
[`CITATION.cff`](https://github.com/Str4vinci/breos/blob/main/CITATION.cff), or use:

```bibtex
@software{rodrigues_breos,
  author = {Rodrigues, Leonardo},
  title  = {{BREOS}: Building Renewable Energy Optimization Software},
  url    = {https://github.com/Str4vinci/breos},
  note   = {Version X.Y.Z}
}
```

BREOS results that depend on PV production also depend on pvlib. Please cite it
alongside BREOS:

```bibtex
@article{anderson2023pvlib,
  author  = {Anderson, K. and Hansen, C. and Holmgren, W. and Jensen, A. and Mikofski, M. and Driesse, A.},
  title   = {pvlib python: 2023 project update},
  journal = {Journal of Open Source Software},
  volume  = {8},
  number  = {92},
  pages   = {5994},
  year    = {2023},
  doi     = {10.21105/joss.05994}
}
```

pvlib additionally asks that you cite the Zenodo DOI for the specific pvlib
version you used.

## Acknowledgements

BREOS stands on work done by others:

- **[pvlib python](https://pvlib-python.readthedocs.io/)** — the PV modeling
  foundation: solar position, irradiance transposition, IAM, cell temperature,
  CEC single-diode evaluation, PVWatts losses, tracking, and inverter helpers.
  BREOS composes these into a production pipeline with staged losses,
  degradation, and multi-array handling; the underlying physics is pvlib's.
- **[BLAST-Lite](https://github.com/NatLabRockies/BLAST-Lite)** (NREL) —
  vendored battery life models.
- **[demandlib](https://demandlib.readthedocs.io/)** — basis for the bundled
  example H0 load profiles.
- **[pymoo](https://pymoo.org/)** — NSGA-II multi-objective optimization.
- **PVGIS (EU JRC)** and **Open-Meteo:** weather and solar
  resource data.

Model choices, defaults, and any errors in how these are combined are BREOS's
own, not those of the upstream projects. See
[ATTRIBUTIONS.md](https://github.com/Str4vinci/breos/blob/main/ATTRIBUTIONS.md) for the full list with licenses and terms.

## Contributing

See [CONTRIBUTING.md](https://github.com/Str4vinci/breos/blob/main/CONTRIBUTING.md). Feature work happens on branches off
`develop` and merges via pull request; `main` tracks stable releases only.

## Contact

Usage questions and feature ideas:
[GitHub Discussions](https://github.com/Str4vinci/breos/discussions).
Bugs: [issues](https://github.com/Str4vinci/breos/issues).
Research collaboration or private enquiries: lrodrigues@fe.up.pt.

## License

BSD 3-Clause. See [LICENSE](https://github.com/Str4vinci/breos/blob/main/LICENSE).
