# Attributions

BREOS bundles or relies on third-party data and tools. This document lists each source, its license, and any redistribution or commercial-use caveats.

## Bundled reference load profiles (`rlp/`)

| File | Source | License / Terms |
|------|--------|-----------------|
| `bdew_h0_2025_15min.csv`, `h0_SLP.csv`, `h0SLP_demandlib_*.csv` | BDEW H0 Standardlastprofil (German residential standard load profile, published by BDEW — Bundesverband der Energie- und Wasserwirtschaft) | Publicly published profile; widely redistributed in research. Attribution: BDEW. |
| `EREDES_2025_BTN_*.csv` | E-REDES (Portuguese DSO) public BTN consumption profiles | Publicly released by the Portuguese DSO. Attribution: E-REDES. |
| `REE_2026_2.0TD_*.csv` | Red Eléctrica de España standard 2.0TD profile | Public profile published by REE. Attribution: REE. |

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
