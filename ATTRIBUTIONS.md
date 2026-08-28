# Attributions and Third-Party Notices

BREOS bundles or relies on third-party data, services, software, and published
methods. This document lists each source, its license posture, and any
redistribution, commercial-use, citation, or attribution caveats.

This is a project-maintainer note, not legal advice.

## Bundled reference load profiles

| File | Source | License / Terms |
|------|--------|-----------------|
| `breos/data/rlp/h0SLP_demandlib_1000kwh_hourly.csv`, `breos/data/rlp/h0SLP_demandlib_1000kwh_15min.csv` | Generated with [demandlib](https://demandlib.readthedocs.io/) H0 logic | demandlib documents itself as MIT-licensed free software. Preserve demandlib attribution and license notices when redistributing derived profile examples. |

## Bundled third-party software

| Files | Source | License / Terms |
|-------|--------|-----------------|
| `breos/degradation/blast/` | BLAST-Lite 1.1.0, vendored from the clean NREL source history at commit `d789e00` (`Correct Tesla Model 3 data source in README`). The GitHub organization has since redirected from `NREL/BLAST-Lite` to `NatLabRockies/BLAST-Lite`. | BSD-3-Clause. The vendored BLAST `LICENSE` and DOE `NOTICE` are preserved at `breos/degradation/blast/LICENSE` and `breos/degradation/blast/NOTICE`. Phase 0 applies only mechanical BREOS vendoring transforms: `np.trapz` to `np.trapezoid`, pandas/matplotlib import trim, package-relative imports, and extraction of `rescale_soc`. Golden parity fixtures were generated from the local BLAST-Lite prep commit `b12e8f3`. Per-file source/result hashes and the complete transformation record are in `breos/degradation/blast/VENDORED.md`. |

## Supported but not redistributed

BREOS can load the following profile families when users provide their own licensed local copies through `breos.load_profile(..., rlp_directory="...")`:

| Profile family | Why not bundled in this public release |
|----------------|-----------------------------------------|
| Direct BDEW Standardlastprofile exports (`h0_SLP.csv`, `bdew_h0_2025_15min.csv`) | BDEW publishes downloadable SLP files, but its public site terms reserve copyright rights and limit downloads/copies to private, non-commercial use unless written permission is granted. |
| E-REDES BTN profiles (`EREDES_2025_BTN_*.csv`) | Public website terms reviewed for this release do not provide a clear redistribution grant for bundling derived CSVs in an OSS package. |
| REE 2.0TD profiles (`REE_2026_2.0TD_*.csv`) | REE legal terms reserve intellectual-property rights and do not clearly authorize republishing derived CSV datasets in this package. |

Users can still provide these files locally through `rlp_directory` when their source terms permit their use case. If written redistribution permission is granted, store the permission text with the release record before adding the files back to package data.

## Runtime data sources (fetched on demand)

| Service | Used by | License | Caveats |
|---------|---------|---------|---------|
| **Open-Meteo** Historical & Forecast API | `breos/weather.py` (`fetch_*_openmeteo`) | Data licensed **CC-BY 4.0**. | **Free API tier is non-commercial.** Commercial workloads require a paid Open-Meteo subscription. Attribution required: "Weather data by Open-Meteo.com". |
| **PVGIS** (JRC) | `breos/weather.py` (PVGIS endpoints) | Governed by Commission Decision 2011/833/EU on reuse of Commission documents — free reuse including commercial, with attribution. | Attribution: "© European Union, [year], PVGIS". |

## Python dependencies

BREOS's Python dependencies are open-source packages under their respective
licenses. See `pyproject.toml`, `uv.lock`, and each package's own metadata for
the authoritative license text. Core and optional dependencies currently include:

- **geopy** — MIT
- **joblib** — BSD 3-Clause
- **matplotlib** — Matplotlib / PSF-style license terms
- **numba** — BSD
- **numpy** — BSD 3-Clause
- **openmeteo-requests** — MIT
- **pandas** — BSD 3-Clause
- **pvlib** — BSD 3-Clause
- **pymoo** — Apache 2.0
- **rainflow** — MIT
- **requests-cache** — BSD 2-Clause
- **scipy** — BSD 3-Clause
- **timezonefinder** — MIT

## Scientific and model credits

BREOS implements or wraps methods from the photovoltaic, battery, optimization,
and reliability literature. These credits are separate from software-license
requirements, but they should be preserved in papers, reports, and downstream
documentation where the relevant models affect results.

| Area | Used by | Credit / citation note |
|------|---------|------------------------|
| PV modelling | `breos/solar.py`, `breos/weather.py` | BREOS uses [pvlib python](https://pvlib-python.readthedocs.io/) for solar position, irradiance transposition, temperature, CEC single-diode evaluation (`calcparams_cec`, `max_power_point`), PVWatts losses, tracking, and inverter helpers. Cite pvlib in published work that relies on these calculations — see the citation below. |
| CEC PV parameter fitting | `breos/cec_fit.py`, `breos/solar.py`, `breos/pv_modules.py` | The CEC 6-parameter coefficient calculator follows A. Dobos, "An Improved Coefficient Calculator for the California Energy Commission 6 Parameter Photovoltaic Module Model", J. Solar Energy Eng. 134 (2012), DOI:10.1115/1.4005759. |
| Multi-objective optimization | `breos/optimization.py` | BREOS uses [pymoo](https://pymoo.org/) for NSGA-II multi-objective optimization. Cite pymoo where optimizer behavior is material to the study. |
| Rainflow cycle counting | `breos/battery.py` | BREOS uses the `rainflow` Python package and ASTM E1049-style rainflow counting for battery cycle detection in the reference path. |
| Battery cycle and calendar ageing | `breos/battery.py`, `breos/constants.py` | Naumann et al. (2020) parameterization and equations are used for cycle ageing and selected calendar/resistance ageing behavior. |
| LFP calendar ageing calibration | `breos/constants.py`, `breos/battery.py` | Lam et al. (2025) LFP calendar ageing behavior informs the `naumann_lam*` calendar-model variants and field-calibrated defaults. |
| BLAST-Lite battery ageing models | `breos/degradation/blast/` | BLAST-Lite model classes preserve DOI-cited empirical degradation models for LFP-Gr, NMC-Gr, NMC-GrSi, NMC-LTO, NCA-Gr, NCA-GrSi, and LMO-Gr cells. Primary source DOIs preserved from BLAST-Lite include `10.1016/j.est.2018.01.019`, `10.1016/j.jpowsour.2019.227666`, `10.1149/1945-7111/ac86a8`, `10.1109/EEEIC/ICPSEUROPE54979.2022.9854784`, `10.1016/j.est.2020.101695`, `10.1149/2.0411609jes`, `10.1149/1945-7111/abae37`, `10.1016/j.jpowsour.2022.232498`, `10.1016/j.jpowsour.2020.228566`, `10.1016/j.jpowsour.2014.02.012`, `10.1016/j.est.2023.109042`, and `10.1149/1945-7111/ac2ebd`. |

### Citing pvlib

pvlib's recommended citation:

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

pvlib also asks that you cite the Zenodo DOI for the specific pvlib version
used. BREOS composes pvlib primitives into its own production pipeline (staged
losses, age degradation, multi-array combination, model-selection surface);
those choices and their defaults are BREOS's responsibility, not pvlib's.

## Notes for downstream users

If you redistribute BREOS or derived datasets:

1. Preserve attributions above.
2. If you call Open-Meteo from a commercial deployment, obtain a paid Open-Meteo subscription.
3. Do not assume a public download page grants redistribution rights; keep externally sourced RLPs outside public package artifacts unless the source license is explicit.
4. Cite the scientific/model references that materially affect published or customer-facing results.
