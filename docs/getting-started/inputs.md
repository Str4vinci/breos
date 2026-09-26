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
- To use NSRDB data, download it with your own NREL credentials and load the
  saved weather file locally. BREOS does not call the NSRDB API.
- Open-Meteo access is subject to Open-Meteo terms; commercial use may require
  a paid subscription.
- For reproducible or offline studies, keep the exact weather file/source
  used for the run.

## Load profiles

BREOS bundles only the demandlib-derived H0 example profile. It is suitable
for examples and baseline residential simulations. Use
`load_profile = "demandlib_h0"` or the equivalent canonical key `"1"` for that
bundled profile.

For E-REDES, REE, direct BDEW, measured smart-meter data, or any other custom
profile, provide licensed local CSV files and pass `rlp_directory`:

```toml
load_profile = "6"
rlp_directory = "/path/to/licensed/rlp/files"
resolution = "15min"
```

See [Load Profile Data](../legal/load-profile-data.md) for expected filenames
and the redistribution policy.

## Repairing measured data

The simulation refuses PV and load input with a gap, a NaN or infinite value,
or negative load. It does not fill them with zero, because a missing reading
is not a real zero. Measured data often has such readings, so BREOS provides
an explicit repair step, {py:func}`breos.io.repair_series <breos.repair.repair_series>`, that you run
before the simulation. It changes only what you ask it to change, refuses
what it cannot repair safely, and returns a report of every change.

```python
import pandas as pd
from breos import App
from breos.io import repair_series

measured = pd.read_csv("meter.csv", index_col=0, parse_dates=True)["W"]
repaired, report = repair_series(measured, kind="load", gap_fill="nearby_days")
repaired.to_csv("external_rlp/REE_2026_2.0TD_1000kwh_hourly.csv")

app = App(
    {
        "location": "porto",
        "n_modules": 10,
        "annual_consumption_kwh": 4000,
        "load_profile": "8",
        "rlp_directory": "external_rlp",
    },
    input_repairs=[report],
)
app.simulate()
app.result()["provenance"]["input_repairs"]  # the report, as JSON
```

The file name and profile key `"8"` select the generic single-column format
described in [Load Profile Data](../legal/load-profile-data.md). The App still
scales that profile to `annual_consumption_kwh`, so the energies in the report
are those of the measured series before scaling.

What it does:

- **Index.** The index must be regular at one step (`freq`, inferred when
  omitted). Timestamps missing from that grid count as gaps; duplicate
  timestamps, an unsorted index, or timestamps off the grid raise. A
  timezone-aware index is stepped in absolute time, so a local index with a
  23-hour and a 25-hour day is regular. Pass `index=` (for example the
  simulation year) to also treat missing steps at the start or end as gaps.
- **Small negative readings** are clipped to zero. The default threshold is
  `negative_clip_w=10.0` W, and a negative stretch may last at most
  `max_negative_run="1h"`. More negative readings, or a longer stretch, raise:
  for load they usually mean the meter recorded net flow (load minus on-site
  PV), and gross demand cannot be recovered without the PV data.
- **Gaps** (missing timestamps, NaN, and ±inf) raise by default
  (`gap_fill="raise"`). With `gap_fill="nearby_days"`, each missing step gets
  the mean of the same time of day on the `neighbour_days=2` nearest days,
  within `window_days=7` either side, that have a valid reading there. Load
  prefers days of the same type (weekday or weekend) and uses the other type
  only when no same-type day is in the window. Load follows the index's own
  clock, so 08:00 is 08:00 on both sides of a DST change; PV
  (`kind="pv"`) follows UTC, which is closer to solar time, and ignores the
  day type. Only original readings are used to fill, never filled ones. A step
  with no valid day in the window raises. Linear interpolation and zero fill
  are not offered.

The report ({py:class}`~breos.repair.InputRepairReport`) lists each repaired run:
its issue (`"gap"` or `"negative"`), first and last timestamp, number of
steps, how many timestamps were missing and how many values were non-finite,
the method, the minimum, mean, and maximum value written, and the energy
added in Wh (a gap counts as zero before the repair). It also carries the
options used and the series energy before and after. `report.to_dict()` is
strict JSON. `App` records the reports it is given, unchanged, under
`provenance.input_repairs`; it does not check them against the file it loads.
Runs without `input_repairs` have no such key.

Weather files are not repaired by this step: the PV path has its own rules for
weather gaps. There is no config key or CLI option for repair yet, so it is a
Python step.

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
