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
vary one value inside a table: `[costs]`, `[battery_indoor_model]`,
`[tariff]` or `[smart_charging]`. A tariff's price maps take one more level,
the period name, as in `"tariff.import_prices.off_peak"`, or two on a
schedule with month seasons, the season and the period, as in
`"tariff.import_prices.q1.high"`. Any other key inside a table, or a level
past those, is refused. The command runs
the Cartesian product and writes one CSV row per combination:

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
independence, NPV, payback, LCOE, battery replacement totals, and the
[year-1 money components](interpreting-results.md#year-1-money-keys). This is
explicit enumeration, not an optimizer; use the optimization API for searching
over objectives and constraints.

{py:func}`~breos.plotting.plot_sweep_heatmap` draws one column of a
two-parameter sweep CSV as a heatmap, and
{py:func}`~breos.plotting.plot_orientation_landscape` draws a `tilt` ×
`azimuth` sweep ([Plotting](../api/plotting.md#sweeps-and-the-optimizer-front)).
The CSV does not record the currency, so pass `currency=` to label a money
column such as `npv_savings` with it.

Every combination is validated before the first run starts, so a bad one,
such as a charge period the tariff schedule does not have, stops the sweep
at once. `breos validate-config` checks every combination too.

Runs that differ only in settings the input stage never reads share one
preparation of weather, PV, load and battery temperature. Those settings are
the battery and inverter settings, degradation, the projection length, prices,
`[tariff]`, `[smart_charging]`, emissions and the execution backend. A tariff
or battery-size comparison therefore fetches PVGIS weather and runs the PV
model once per PV design, not once per run, and writes the same CSV as
preparing each run afresh. To do so it runs the grid grouped by PV design,
while the CSV keeps the grid order. A warning from the input stage appears
once, for the run that prepares those inputs.

To compare tariffs, see [Compare tariffs](#compare-tariffs).

## Compare tariffs

A tariff comparison prices the same system and load under each offer. The
dispatch does not depend on the price unless `[smart_charging]` is set, so
the energy flows are the same and only the money differs. Two comparisons
answer different questions:

- **Which offer is cheapest?** Compare what the household pays under each.
  For the first year, the
  [year-1 money keys](interpreting-results.md#year-1-money-keys) give the
  bill: import cost plus fixed charge minus export revenue. Over the project,
  the last `financial` row's `cost_with_system` is the cumulative discounted
  cost with the system: investment, energy, fixed charge, O&M and
  replacements.
- **Under which offer does the system pay most?** `npv_savings` is the
  system's saving against no system *under the same offer*. Each offer has
  its own no-system bill, so a higher `npv_savings` does not mean a cheaper
  offer. An offer with an expensive peak can make PV save more while still
  costing more overall.

### From the CLI

Give `tariff` a list of whole tables under `[sweep]`; each replaces the
tariff for its runs. A simple (single-price) offer is any schedule with an
`all` price.
[configs/examples/tariff-comparison.toml](https://github.com/Str4vinci/breos/blob/develop/configs/examples/tariff-comparison.toml)
compares a simple, a bi-hourly and a tri-hourly offer, with and without a
battery:

```toml
location = "porto"
n_modules = 10
annual_consumption_kwh = 4000
cost_preset = "residential_pt"
emissions_country = "PT"
resolution = "15min"   # the tri-hourly schedule changes on the half hour

[sweep]
battery_kwh = [0.0, 5.0]

[[sweep.tariff]]
schedule = "pt_mainland_2026_daily_bi"
currency = "EUR"
import_prices = { all = 0.1950 }
export_prices = { all = 0.0500 }
fixed_charge_per_day = 0.30

[[sweep.tariff]]
schedule = "pt_mainland_2026_daily_bi"
currency = "EUR"
import_prices = { peak = 0.2310, off_peak = 0.1210 }
export_prices = { all = 0.0500 }
fixed_charge_per_day = 0.30

[[sweep.tariff]]
schedule = "pt_mainland_2026_daily_tri"
currency = "EUR"
import_prices = { peak = 0.2890, mid_peak = 0.1920, off_peak = 0.1210 }
export_prices = { all = 0.0500 }
fixed_charge_per_day = 0.30
```

```bash
breos sweep --config configs/examples/tariff-comparison.toml --output tariff_comparison.csv
```

The six runs share one preparation of weather and PV. In the CSV,
`param_tariff` holds each run's tariff table as JSON, so a spreadsheet or
pandas can label the rows by its `schedule` and prices. The year-1 money
columns compare the first-year bills; the CSV carries top-level scalars only,
so for the project-long cost use the Python route below. The prices above are
illustrative; put the offers you are comparing in their place.

To vary one price instead of the whole offer, keep one `[tariff]` in the base
scenario and sweep a dotted key:

```toml
[tariff]
schedule = "pt_mainland_2026_daily_bi"
currency = "EUR"
import_prices = { peak = 0.2310, off_peak = 0.1210 }
export_prices = { all = 0.0500 }

[sweep]
"tariff.import_prices.off_peak" = [0.1010, 0.1210, 0.1410]
```

### From Python

A loop over {py:class}`~breos.App` runs does the same, and can also compare
against the cost preset's flat prices, which a tariff table replaces:

```python
from breos import App

base = {
    "location": "porto",
    "n_modules": 10,
    "annual_consumption_kwh": 4000,
    "battery_kwh": 5.0,
    "cost_preset": "residential_pt",
    "emissions_country": "PT",
}
offers = {
    "preset flat prices": {},
    "bi-hourly": {
        "tariff": {
            "schedule": "pt_mainland_2026_daily_bi",
            "currency": "EUR",
            "import_prices": {"peak": 0.2310, "off_peak": 0.1210},
            "export_prices": {"all": 0.0500},
            "fixed_charge_per_day": 0.30,
        }
    },
}

for name, offer in offers.items():
    app = App({**base, **offer})
    app.simulate()
    result = app.result()
    bill = (
        result["grid_import_cost_year1_prices"]
        + result["fixed_charge_year1_prices"]
        - result["grid_export_revenue_year1_prices"]
    )
    project_cost = result["financial"][-1]["cost_with_system"]
    print(
        f"{name}: year-1 bill {bill:.2f}, project cost {project_cost:.2f}, "
        f"NPV savings vs no system {result['npv_savings']:.2f}"
    )
```

The year-1 bill leaves out O&M, which does not depend on the offer.

The flat run uses the preset's `electricity_cost`, `electricity_sold_cost` and
`daily_power_cost`; a tariff run must not set them. Each run with the same PV
design repeats the weather and PV preparation, which `breos sweep` shares.

## Revalue a run at other prices

`App.revalue` returns the result a finished run would give at other prices,
without simulating the energy balance again when the prices cannot change
the dispatch. It accepts the economics keys only: `costs`, `cost_preset`,
`tariff`, `discount_rate`, `inflation_rate` and the escalators. A nested
table changes only the keys it sets, a key set to `None` in it is removed,
and `{"tariff": None}` removes the tariff. A price list
(`tariff.import_prices`, `tariff.export_prices`) replaces the old one whole.

```python
from breos import App

app = App(
    {
        "location": "porto",
        "n_modules": 10,
        "annual_consumption_kwh": 4000,
        "battery_kwh": 5.0,
        "cost_preset": "residential_pt",
    }
)
app.simulate()

for storage_cost in (500, 400, 300):
    result = app.revalue({"costs": {"storage_cost_per_kwh": storage_cost}, "discount_rate": 0.05})
    print(storage_cost, result["npv_savings"], result["provenance"]["revaluation"]["method"])
```

`provenance.revaluation.method` says what happened:

- `"repriced"`: the stored run was priced again. This is the case for flat
  prices, for a tariff removed, and for new prices on the same tariff
  schedule when the smart-charging instructions do not change (fixed-target
  instructions follow the periods, not the prices). Flat prices give the
  same floats as a new run. A re-priced tariff sums each year's energy by
  period instead of by step, so it agrees with a new run to rounding.
- `"resimulated"`: the run was simulated again, because a tariff was added,
  the schedule changed, or the instructions would change. Under the
  experimental `daily_persistence` smart charging, whose planner reads the
  prices, any change to the import or export prices simulates again; a
  change to the fixed charge alone is re-priced.

A key that changes the simulation, such as `battery_kwh` or
`projection_years`, raises `ValueError`; build a new `App` for it.
`revalue` leaves the App and its `result()` unchanged.

## 15-minute resolution

Hourly weather is interpolated to 15-minute steps (Makima), and the bundled
H0 profile has a native 15-minute variant. An external profile supplied only
as hourly means is interpolated between hour midpoints and keeps each hour's
mean exactly. Simulations take correspondingly
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

## Simulate part of a year

A `[period]` table simulates a window shorter than a year, for example one
week in June:

```toml
location = "porto"
n_modules = 10
annual_consumption_kwh = 4000
battery_kwh = 5.0
start_date = "2025-01-01"

[period]
start = 2025-06-01
end = 2025-06-08
```

As a Python dict, give the dates as ISO strings or `datetime.date` values:
`"period": {"start": "2025-06-01", "end": "2025-06-08"}`.

- `start` and `end` are dates in the location's timezone, in the year of
  `start_date`. The window starts at local midnight of `start` and ends at
  local midnight of `end`, so `end` is exclusive: the example covers 1 to 7
  June, and one day is `start = 2025-06-01`, `end = 2025-06-02`. `end` may be
  1 January of the next year. A window of the whole year raises; omit
  `[period]` for that.
- The load profile is built for the whole year and scaled to
  `annual_consumption_kwh` as usual, and the window gets its share of it.
  Weather, PV, load and battery temperature are then cut to the window.
- The weather must cover the whole window. It may cover more (a TMY covers
  the year), but a window it does not reach raises `ValueError` naming the
  missing span. Two cases follow from the window being in civil time:
  - Weather that covers one UTC year, such as an Open-Meteo or other UTC CSV
    file, cannot serve a window that touches the civil year edge away from
    UTC. A Berlin window starting on 1 January misses one hour before
    01:00 UTC, and a New York window ending on 1 January of the next year
    misses five hours. PVGIS TMY weather, on the fixed offset of 1 January,
    covers both.
  - Weather stamped at half past the hour (NSRDB-style) cannot place a
    window that starts at local midnight, so no window runs on it.

  Both raise `ValueError` before anything is simulated.
- The window runs once, from the battery's initial state, and nothing is
  repeated or carried over. `projection_years` is not used: setting it
  alongside `[period]` gives a warning, and the result's `period` block
  records `projection_years_used = 1`.
- The fixed charge (`costs.daily_power_cost`, or a tariff's
  `fixed_charge_per_day`) is billed on the window's civil days, so a window
  over a DST change bills whole days, not 23 or 25 hours.
- Energy results cover the window. The lifetime economics (NPV, payback,
  LCOE, the `financial` projection and the replacement costs) are `None`; see
  [Period runs](interpreting-results.md#period-runs).
- `monthly` groups the window by the location's civil months, while a
  full-year run groups by the weather's clock. On weather whose clock is not
  the civil one (a PVGIS fixed offset in summer), a window's `"Jun"` row can
  therefore differ from the same month of a full-year run by the hour at
  each month edge.

PV, load and every other flow of a PV-only run equal the same steps of a
full-year run exactly. With a battery the dispatch differs, because the
window starts from the battery's initial state of charge and health, not
from the state the full-year run reaches on that date.

Monte Carlo and the optimizer reject `period`: they rank designs on lifetime
economics. `breos sweep` accepts it, and can vary `period.start` and
`period.end`. `App.revalue` re-prices a period run's window; its lifetime
economics stay `None`.

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
load_profile = "eredes_btn_c"
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
fresh fetch. The filename's year part only needs to match the pattern; it is
metadata, not a lookup key.

If the directory holds more than one TMY file for the location, for example a
PVGIS and an NSRDB export for Porto, BREOS does not pick one: the run stops and
lists the candidates. Set `weather_source` to the filename's source part to
choose one:

```python
App({"location": "porto", "n_modules": 10, "annual_consumption_kwh": 4000,
     "weather_source": "pvgis-sarah3"})
```

or `breos run --config config.toml --weather-source pvgis-sarah3` from the
command line. A `weather_source` with no matching file is an error rather than
a PVGIS fetch. The file that was used, with its SHA-256 digest and the parsed
filename (including the source), is recorded under `provenance.weather`.

The file is restamped onto the year of `start_date` and must then cover that
whole calendar year. A file missing its first or last rows raises `ValueError`
that names the file and the missing span, rather than simulating a shorter
year. With a [`[period]`](#simulate-part-of-a-year), the file needs to cover
only the window.
