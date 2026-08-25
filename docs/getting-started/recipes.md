# Recipes

Copy-paste starting points for common setups. Save any block below as
`config.toml`, then:

```bash
breos validate-config config.toml                  # check resolved choices first
breos run --config config.toml --output result.json
```

Every key works identically as a Python dict passed to
{py:class}`~breos.App`. Valid option keys for locations, modules, cost
presets, emissions countries, and load profiles are listed on the
[packaged options](options.md) page or via `breos list`.

## PV-only home

Set `battery_kwh = 0` to disable storage. Investment, payback, and NPV then
reflect the PV system alone, and battery-specific result keys are omitted:

```toml
location = "porto"
n_modules = 10
annual_consumption_kwh = 4000
battery_kwh = 0.0
cost_preset = "residential_pt"
emissions_country = "PT"
```

## PV plus battery

The packaged quickstart, [configs/examples/quickstart.toml](https://github.com/Str4vinci/breos/blob/main/configs/examples/quickstart.toml):

```toml
location = "porto"
n_modules = 10
annual_consumption_kwh = 4000
battery_kwh = 5.0
load_profile = "demandlib_h0"
cost_preset = "residential_pt"
emissions_country = "PT"
projection_years = 20
resolution = "h"
```

See the [quickstart](quickstart.md) for representative output values.

## Custom latitude / longitude / timezone

Any site works without a packaged preset — pass coordinates and an IANA
timezone instead of a location key:

```toml
location = { latitude = 48.2082, longitude = 16.3738, timezone = "Europe/Vienna" }
n_modules = 12
annual_consumption_kwh = 4500
battery_kwh = 5.0
cost_preset = "residential_de"
emissions_country = "AT"
```

Tilt and azimuth are auto-estimated from the latitude when not set. There is
no Austrian cost preset yet, so this example borrows the German one — replace
it with your own tariffs for real economics.

## Use your own PV module

`pv_module` accepts catalogue keys only. An unknown name fails configuration
validation and lists what is available instead. To simulate hardware BREOS does
not ship, register it with {py:func}`~breos.pv_modules.add_module` first, then
reference it by the name you registered:

```python
import breos
from breos.pv_modules import add_module, PVModuleParams

add_module(
    "MySupplier_540W",
    PVModuleParams(
        Mpp=540.0,          # W at STC
        Vmp=41.3,           # V
        Imp=13.08,          # A
        Voc=49.4,           # V
        Isc=13.86,          # A
        T_Pmax_pct=-0.34,   # %/degC
        T_Voc_pct=-0.25,    # %/degC
        T_Isc_pct=0.045,    # %/degC
        N_Cells=144,
        Name="MySupplier 540W mono PERC",
        Module_Efficiency=0.209,
    ),
)

app = breos.App({
    "location": "porto",
    "n_modules": 10,
    "annual_consumption_kwh": 4000,
    "battery_kwh": 5.0,
    "pv_module": "MySupplier_540W",
    "cost_preset": "residential_pt",
    "emissions_country": "PT",
})
app.simulate()
```

Those nine electrical and thermal values are required, and a datasheet at STC
supplies all of them. `N_Cells` is the cell count, so 144 for a half-cut
72-cell module. The three `T_*_pct` values are temperature coefficients in
percent per degree Celsius, and `T_Pmax_pct` and `T_Voc_pct` are negative on
almost every module.

Everything after `N_Cells` is optional. `Module_Efficiency` feeds the PVsyst and
SAM cell-temperature models, and `noct-sam` refuses to run without both it and a
`NOCT` value. Set `bifaciality` if you also plan to turn on
`bifacial_model`, because the metadata alone changes nothing.

To confirm the registration took effect, resolve the same config in the current
Python process before running a full simulation. Ten 540 W modules give 5.4
kWp:

```python
from breos.app_config import resolve_app_config

resolved = resolve_app_config({
    "location": "porto",
    "n_modules": 10,
    "annual_consumption_kwh": 4000,
    "pv_module": "MySupplier_540W",
})
assert resolved.system_kwp == 5.4
```

`add_module` writes the module into the in-memory catalogue and persists
nothing. Call it before you construct `App`, and call it again in every new
process, including optimizer workers when `n_procs` is above 1. The command-line
interface starts a separate process and has no registration hook, so a custom
module needs the Python API rather than `breos run --config`.

## East-west roof with `pv_arrays`

Each array is simulated independently and the DC output is combined before
the energy balance, so an east-west layout is not collapsed into one
representative orientation. `n_modules` is derived from the array totals:

```toml
location = "porto"
annual_consumption_kwh = 4000
battery_kwh = 5.0
cost_preset = "residential_pt"
emissions_country = "PT"

[[pv_arrays]]
modules = 8
module = "Erlangen_445W"
tilt = 10
azimuth = 90    # east

[[pv_arrays]]
modules = 8
module = "Erlangen_445W"
tilt = 10
azimuth = 270   # west
```

## Anisotropic sky-diffusion model

The default `isotropic` transposition underestimates plane-of-array
irradiance on clear days. Switch to an anisotropic model — here Perez — to
capture circumsolar and horizon brightening. No extra weather inputs are
needed; see [Sky-diffusion model](configuration.md#sky-diffusion-transposition-model):

```toml
location = "porto"
n_modules = 10
annual_consumption_kwh = 4000
cost_preset = "residential_pt"
emissions_country = "PT"
transposition_model = "perez"
surface_type = "grass"          # or a numeric albedo, e.g. albedo = 0.2
```

`surface_type` (or a numeric `albedo`) sets the ground reflectance that feeds
the ground-diffuse component; a snowy or sandy foreground (`"snow"`, `"sand"`)
raises annual yield further. Leave both unset to keep pvlib's 0.25 default.

From the CLI, the equivalent flag is `--transposition-model perez`
(alias `--sky-model`).

## Parameter sweep

Use `breos sweep` when you want to run the same scenario over an explicit grid
of App config values. The top-level keys define the base scenario; every key
under `[sweep]` replaces the matching key for each run. Quote dotted keys to
vary values inside a table such as `[costs]`. The command runs the Cartesian
product and writes one CSV row per combination:

```toml
location = "porto"
n_modules = 10
annual_consumption_kwh = 4000
battery_kwh = 0.0
load_profile = "demandlib_h0"
cost_preset = "residential_pt"
emissions_country = "PT"
projection_years = 20
resolution = "h"

[sweep]
n_modules = [8, 10, 12]
battery_kwh = [0.0, 5.0]
"costs.electricity_cost" = [0.20, 0.30]
```

```bash
breos sweep --config config.toml --output sweep_results.csv
```

The output includes the varied parameters (`param_*` columns, including for
example `param_costs.electricity_cost`), resolved system
sizing, the BREOS version, and top-level scalar result metrics such as grid
independence, NPV, payback, LCOE, and battery replacement totals. This is
explicit enumeration, not an optimizer; use the optimization API for searching
over objectives and constraints.

## 15-minute resolution

Hourly weather is interpolated to 15-minute steps (Makima), and the bundled
H0 profile has a native 15-minute variant. Simulations take correspondingly
longer:

```toml
location = "porto"
n_modules = 10
annual_consumption_kwh = 4000
battery_kwh = 5.0
resolution = "15min"
cost_preset = "residential_pt"
emissions_country = "PT"
```

## External load profile (E-REDES, BDEW, REE)

Only the demandlib-derived H0 profile (`"1"`, alias `"demandlib_h0"`) ships
with BREOS. For the other standard profiles, download the source CSVs yourself
under terms that permit your use, put them in a local directory, and point
`rlp_directory` at it.
[Load Profile Data](../legal/load-profile-data.md) lists the exact expected
filenames per profile key:

```toml
rlp_directory = "external_rlp"
location = "porto"
n_modules = 10
annual_consumption_kwh = 4000
battery_kwh = 5.0
load_profile = "6"   # E-REDES BTN C
resolution = "15min"
cost_preset = "residential_pt"
emissions_country = "PT"
```

A runnable template also ships in the repository as
`configs/examples/external-rlp.toml`.

## Apply a custom terrain horizon

Use azimuth/elevation pairs to model far-horizon direct-beam obstruction
without a 3D scene:

```toml
location = "porto"
n_modules = 8
annual_consumption_kwh = 4000

horizon_profile = [
  [0, 4],
  [60, 9],
  [120, 14],
  [180, 3],
  [270, 6],
]
```

The profile is circular: BREOS interpolates from the last point back to the
first across north, so a repeated `360` endpoint is unnecessary. With a fresh
PVGIS fetch, configuring the profile automatically requests provider data
without PVGIS's own horizon. Cached weather must have a matching provenance
sidecar that explicitly records `not_applied`; legacy CSVs are rejected because
their horizon treatment cannot be established safely. The normalized profile
and number of shaded timesteps appear under
`provenance.weather.horizon.profile`.

## Offline runs with cached weather

When the config uses a location *preset key*, BREOS scans a `weather/`
directory in the current working directory before fetching from PVGIS, and
silently reuses a file named `<location>_tmy_<year0>_<year1>_<source>.csv`.
Seed the cache once while online:

```python
from pathlib import Path
from breos.weather import fetch_tmy_weather_data

Path("weather").mkdir(exist_ok=True)
tmy, _ = fetch_tmy_weather_data(
    latitude=41.1579,
    longitude=-8.6291,
    timezone="Europe/Lisbon",
)
tmy.to_csv("weather/porto_tmy_2005_2023_pvgis-sarah3.csv")
```

This manual CSV export is compatible with existing offline runs, but it has no
provenance sidecar and is therefore loaded with an `unknown` horizon status.
Calling `fetch_tmy_weather_data(..., save_to_file=True)` writes both the CSV
and its digest-bound `.csv.metadata.json` sidecar. Keep the two files together
and rename both with the same CSV basename if you adapt the generated filename
to a location preset.

Subsequent runs from the same working directory work without network access
(the log line `Found local weather file` confirms the cache hit). Custom
coordinate-dict locations always fetch; delete or rename the file to force a
fresh fetch. The filename's year and source parts only need to match the
pattern — they are metadata, not lookup keys.
