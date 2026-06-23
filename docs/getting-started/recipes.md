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

Subsequent runs from the same working directory work without network access
(the log line `Found local weather file` confirms the cache hit). Custom
coordinate-dict locations always fetch; delete or rename the file to force a
fresh fetch. The filename's year and source parts only need to match the
pattern — they are metadata, not lookup keys.
