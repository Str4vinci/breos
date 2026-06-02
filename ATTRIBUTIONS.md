# Attributions

BREOS bundles or relies on third-party data and tools. This document lists each source, its license posture, and any redistribution or commercial-use caveats.

This is a project-maintainer note, not legal advice.

## Bundled reference load profiles

| File | Source | License / Terms |
|------|--------|-----------------|
| `rlp/h0SLP_demandlib_1000kwh_hourly.csv`, `rlp/h0SLP_demandlib_1000kwh_15min.csv`, `breos/data/rlp/h0SLP_demandlib_*.csv` | Generated with demandlib H0 logic | demandlib documents itself as MIT-licensed free software. Preserve demandlib attribution and license notices when redistributing derived profile examples. |

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
| **NREL NSRDB** (via pvlib) | `breos/weather.py:fetch_tmy_nsrdb` | NSRDB data is a U.S. government work, generally in the public domain in the U.S. | Citation requested by NREL. Refer to the NSRDB "How to cite" page for the current canonical citation. Users must obtain their own NREL API key. |
| **PVGIS** (JRC) | `breos/weather.py` (PVGIS endpoints) | Governed by Commission Decision 2011/833/EU on reuse of Commission documents — free reuse including commercial, with attribution. | Attribution: "© European Union, [year], PVGIS". |

## Python dependencies

All Python dependencies are open source under their respective licenses (see `pyproject.toml` and `uv.lock`). Notable ones with non-trivial redistribution implications:

- **pvlib** — BSD 3-Clause
- **NREL-PySAM** — Refer to the NREL-PySAM license; bindings around SAM (proprietary NREL software, free to use).
- **pymoo** — Apache 2.0

## Notes for downstream users

If you redistribute BREOS or derived datasets:

1. Preserve attributions above.
2. If you call Open-Meteo from a commercial deployment, obtain a paid Open-Meteo subscription.
3. Do not assume a public download page grants redistribution rights; keep externally sourced RLPs outside public package artifacts unless the source license is explicit.
