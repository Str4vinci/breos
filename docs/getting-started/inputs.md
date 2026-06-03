# Required Inputs

The quickstart runs with packaged defaults: a Porto location preset, a bundled
demandlib-derived H0 load profile, packaged cost/emissions presets, and the
default PV module catalogue entry.

For a real study, bring your own project inputs and keep a record of where
they came from.

## Weather and API access

BREOS can fetch TMY/weather data through supported weather providers, or load
local weather files through the lower-level weather helpers.

- PVGIS-based TMY fetches do not require a user API key.
- NSRDB fetches require your own NREL API key.
- Open-Meteo access is subject to Open-Meteo terms; commercial use may require
  a paid subscription.
- For reproducible or offline studies, keep the exact weather file/source
  used for the run.

## Load profiles

BREOS bundles only the demandlib-derived H0 example profile. It is suitable
for examples and baseline residential simulations.

For E-REDES, REE, direct BDEW, measured smart-meter data, or any other custom
profile, provide licensed local CSV files and pass `rlp_directory`:

```toml
load_profile = "6"
rlp_directory = "/path/to/licensed/rlp/files"
resolution = "15min"
```

See [Load Profile Data](../legal/load-profile-data.md) for expected filenames
and the redistribution policy.

## PV system data

At minimum, provide the module count and either a module key from the built-in
catalogue or enough module data to extend the catalogue/lower-level PV
parameters.

For real systems, record:

- Module manufacturer/model and datasheet electrical parameters.
- Module count, tilt, azimuth, and any multi-array roof layout.
- Tracking mode, if applicable.
- Inverter/coupling assumptions, including DC/AC ratio and efficiency.

## Battery and financial assumptions

Battery, cost, tariff, and emissions defaults are examples, not universal
truth. For publishable or customer-facing studies, provide:

- Battery capacity, chemistry, usable SOC window, efficiency, and degradation
  model assumptions.
- Installed PV/battery costs, replacement costs, maintenance costs, tariffs,
  export compensation, inflation, and discount rate.
- Grid-emissions factor or country preset appropriate for the study year.

## Minimum reproducibility checklist

For each simulation, save:

- BREOS version and config file.
- Weather source or local weather file.
- Load profile source/license and annual consumption.
- PV module/inverter/battery datasheets or assumptions.
- Cost, tariff, and emissions assumptions.

See [Resources](../resources.md) for links to commonly used PV, RLP, weather,
and solar-resource sources.
