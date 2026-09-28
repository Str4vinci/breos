# Configuration

The {py:class}`~breos.App` constructor accepts a single `config` dict. Only
three keys are strictly required:

- `location`
- `annual_consumption_kwh`
- `n_modules` — *or* `pv_arrays` for multi-array systems

Every other key has a sensible default.

Defaults are useful for examples. For real studies, provide project-specific
weather/data access, load profiles, PV system data, and cost assumptions; see
[Required Inputs](inputs.md).

## All keys

The [Configuration key reference](config-reference.md) lists every key, with
its default, CLI flag, allowed values and nested-table keys. It is generated
from the configuration registry that validates a config, so it cannot miss a
key. The sections below explain how the keys work together.

Real calendar-year load profiles follow `start_date`: leap years contain
8,784 hourly (35,136 quarter-hourly) intervals and preserve exact annual
energy. An 8,760-hour TMY restamped onto a leap year gets the same treatment:
29 February is a copy of 28 February, for the weather and the load alike, and
1 March onwards keeps its own data. The result's weather provenance records
the copied day under `leap_day`.

Unknown top-level keys are rejected at load time. A misspelled key such as
`batery_kwh` raises an error listing the offending key rather than being
silently ignored (which would quietly fall back to the default). The optional
`[sweep]` and `[montecarlo]` sections used by their dedicated CLI commands are
recognised and allowed.

## Battery capacity and the SOC window

`battery_kwh` is the **nominal** pack capacity. The energy balance only
cycles the battery between `battery_min_soc` and `battery_max_soc`, so the
effective storage swing is:

```
usable swing = battery_kwh × (battery_max_soc − battery_min_soc)
```

With the defaults (0.10–0.90) that is 80% of nominal: `battery_kwh = 5.0`
gives a 4.0 kWh swing at full state of health.

Battery datasheets usually advertise *usable* capacity. To match a spec
sheet, either enter `usable / 0.8` as `battery_kwh` or widen the SOC window.
Keep in mind that calendar and cycle aging are evaluated on the absolute SOC,
so the window also shapes degradation results — the defaults reflect the
operating range the field-calibrated aging parameters were fit for, and
simulating a 0–1.00 window models a battery management system that no real
product ships.

Battery temperature is also a degradation input. By default, BREOS derives it
from the weather data and applies the indoor-buffering model. A numeric
`battery_temperature` is treated as an outdoor or supplied temperature and is
still buffered unless the indoor model is disabled. For a study that assumes
an exact constant battery temperature, configure both values explicitly:

```python
breos.App({
    # ...required project inputs...
    "battery_temperature": 25.0,
    "battery_indoor_model": {"enabled": False},
})
```

The mapping also accepts `setpoint_c`, `coupling_alpha`, `floor_c`, and
`ceiling_c`. Set `coupling_alpha` between 0 and 1, and do not set `floor_c`
above `ceiling_c`.

A CSV `battery_temperature` needs a timestamp column (`date`, `datetime`, or
`time`) and a temperature column (`temp`, `temperature`, `t_cell`, or `t_amb`)
that cover the simulated year. Naive timestamps are read as UTC. Each step
takes the latest reading within the file's own sampling interval, so hourly
readings can drive a 15-minute run. A file that is missing or unreadable, that
comes from another calendar year, or that has gaps raises an error instead of
falling back to a default temperature.

## Battery degradation calibration

`calendar_model = "naumann_lam_field_calibrated"` is the stable default and
maps to the v1 field calibration. The explicit
`"naumann_lam_field_calibrated_v1"` alias is equivalent. Use
`"naumann_lam_field_calibrated_v2"` for the v2 field-calibrated fit with Lam
`Ea`/`n` fixed and `k0`/`b` fitted to field data.

The native BREOS degradation path is calibrated for LFP cells only. App config
must not use the ambiguous legacy `battery_type` selector: omit
`degradation_engine` for native behavior, or set `degradation_engine="blast"`
and a stable `blast_model` key. Lower-level
`BatteryConfig(battery_type="LFP")` still normalizes to `"lfp"` for native
compatibility; it does not select BLAST. See the
[degradation model reference](../api/degradation-models.md) for discovery,
precedence, provenance, and migration details.

## Discovering available options

Use the CLI to list packaged option keys:

```bash
breos list locations
breos list modules
breos list cost-presets
breos list emissions
breos list battery-models
breos list load-profiles
```

Add `--json` to any `breos list` command for machine-readable output.

Before running a full simulation, validate and inspect a config:

```bash
breos validate-config quickstart.toml
breos run --config quickstart.toml --dry-run
```

These commands resolve packaged presets, modules, inverter sizing, battery
settings, load-profile choices, emissions settings, and the static PVWatts loss
stack without fetching weather or simulating. In JSON output, `pv.losses`
contains the resolved component percentages plus the combined PVWatts loss
percentage after applying any `pv_loss_overrides`.

## Recommended PV-model starting point

BREOS keeps historical defaults stable so existing studies remain reproducible.
For a new hourly study using interval-averaged weather, the explicit profile in
`configs/examples/recommended-pv.toml` is a stronger starting point:

| Choice | Compatible default | Recommended starting point |
|---|---|---|
| Sky transposition | `isotropic` | `perez` |
| Solar position | `interval-start` | `weather` when the input has content-bound timing metadata; otherwise choose a label-aware explicit method |
| Beam IAM | `ashrae` | `physical` |
| Diffuse IAM | `none` | `marion` |
| Cell temperature | `faiman` open rack | A mount-appropriate PVsyst or SAPM preset |

These are explicit modeling assumptions, not universally correct replacements.
Match the timestamp convention and label direction to the weather source:
`mid-interval` only represents an interval centre when the label marks the
interval start. Match the temperature preset to the physical construction.
The example uses a close rooftop mount and a catalog module with sourced
efficiency; a free-standing array should select a free-standing/open-rack
thermal model instead.

## Incidence-angle modifier (IAM)

`iam_model` controls the optical loss applied to direct irradiance. The
historical default, `"ashrae"`, remains the compatible choice. Set it to
`"physical"` for pvlib's physical glass/refraction model or `"martin_ruiz"`
for its empirical model. BREOS deliberately uses pvlib's published default
parameters for both alternatives; it does not fabricate module-specific
optical inputs.

For `diffuse_iam = "marion"`, fixed-tilt arrays use pvlib's exact Marion
diffuse integration with that same selected IAM model. Tracking arrays
evaluate the integrated IAM on a cached 0.5 degree tilt grid and interpolate
per timestep, avoiding thousands of repeated integrations while preserving a
smooth tracker response.

## Cell-temperature model choices

`"faiman"` remains the default, with its historical open-rack coefficients.
The three `"pvsyst-*"` presets model free-standing, semi-integrated, and
insulated mounting.

PVsyst's heat balance takes a module efficiency, which is a physical input
rather than a tuning constant: it sets the share of absorbed energy that leaves
the module as electricity instead of heat. BREOS supplies a module's sourced
`Module_Efficiency` when it has one, and a representative 20% for modern
crystalline silicon when it does not. Both are deliberate — pvlib's own 0.1
default is a legacy placeholder that would model a module as converting 10% and
shedding the other 90% as heat, which runs cell temperatures roughly 2.5 °C hot
at 800 W/m². Anywhere in the realistic 19–22% band shifts cell temperature by at
most about 0.5 °C, so the exact figure matters much less than not inheriting
0.1.

Two details worth knowing if you are comparing against PVsyst itself. The value
is defined at the operating point, and BREOS uses the datasheet STC efficiency
as a stand-in; PVsyst re-evaluates it each timestep, which is worth a further
0.4–0.6 °C at high irradiance. And efficiency only reaches the `pvsyst-*` and
`"noct-sam"` thermal models — it plays no part in the single-diode DC
calculation, which works from the full IV parameters.

The four `"sapm-*"` choices are named exactly for pvlib's Sandia construction
and mounting coefficient sets:

- `"sapm-open-rack-glass-glass"`
- `"sapm-close-mount-glass-glass"`
- `"sapm-open-rack-glass-polymer"`
- `"sapm-insulated-back-glass-polymer"`

`"noct-sam"` is deliberately stricter. It requires both a sourced `NOCT` in
°C and a sourced module-efficiency fraction. No bundled catalog module has a
verified NOCT value yet, so selecting it with a bundled module fails during
configuration validation instead of guessing a thermal input. It is available
to direct `breos.solar` callers who provide complete metadata in
`PVModuleParams`.

## Bifacial rear gain

Bifacial modeling is deliberately opt-in. A module's `bifaciality` metadata
never changes production by itself; set `bifacial_model = "infinite_sheds"`
and provide complete row geometry to activate rear irradiance:

```python
breos.App({
    "location": "porto",
    "n_modules": 10,
    "annual_consumption_kwh": 4000,
    "pv_module": "Generic_600W_Bifacial",
    "bifacial_model": "infinite_sheds",
    "albedo": 0.2,
    "gcr": 0.35,
    "pvrow_height": 1.5,
    "pvrow_pitch": 6.0,
})
```

`pvrow_height` is the row-center height and `pvrow_pitch` is the distance
between rows. Their absolute unit is arbitrary, but both values must use the
same unit. For `pv_arrays`, each array may override the model and geometry;
this permits mixed front-only and bifacial systems.

BREOS keeps its existing unshaded front-side transposition chain and uses
pvlib's infinite-sheds row geometry for the rear side only. This hybrid is a
good approximation in the low-GCR limit, but it is front-optimistic for dense
ground-mount rows because front-side row shading is not modeled. The rear
estimate uses Hay-Davies when the front selects it and isotropic transposition
for every other front-side model. Rear irradiance is included in the thermal
balance as well as in DC power, so the cell-temperature model sees the
bifaciality-weighted rear gain and the resulting temperature rise offsets part
of it. No extra `pvfactors`/Shapely dependency is required.

The year-1 result reports the modeled contribution under
`pv_loss_waterfall.bifacial`, as an ordered `bifacial_rear_gain` waterfall
stage, and under `provenance.pv_model.bifacial`.

## Custom location

Pass an explicit coordinate dict instead of a preset key:

```python
breos.App({
    "location": {
        "latitude": 41.1579,
        "longitude": -8.6291,
        "timezone": "Europe/Lisbon",
    },
    "n_modules": 10,
    "annual_consumption_kwh": 4000,
})
```

## Multi-array PV systems

For roofs with panels facing different directions, use `pv_arrays`. Each
array is simulated independently and its DC output combined before the
energy balance — east-west or pitched-roof layouts are not collapsed into
one representative tilt/azimuth:

```python
breos.App({
    "location": "porto",
    "annual_consumption_kwh": 4000,
    "pv_arrays": [
        {"modules": 8, "module": "Erlangen_445W", "tilt": 10, "azimuth": 90},
        {"modules": 8, "module": "Erlangen_445W", "tilt": 10, "azimuth": 270},
    ],
})
```

When `pv_arrays` is set, `n_modules` is computed from the array totals and
any explicit `n_modules` key is ignored.

Each array may also set its own `transposition_model`, overriding the
top-level default for that array only.

Arrays inherit `tracking` and the tracker geometry (`axis_tilt`,
`axis_azimuth`, `max_angle`, `backtrack`, `cross_axis_tilt`, and
`dual_axis_max_tilt`) from the top level, and an array may override any of
them. So a top-level `tracking = "single_axis"` makes every array a tracker
unless the array sets `tracking = "fixed"`. An array entry accepts only the
keys named in this section and the sky, ground, and bifacial keys; any other
key, such as a misspelled `tlt`, is rejected.

## Sky-diffusion (transposition) model

To compute plane-of-array (POA) irradiance, BREOS transposes the horizontal
irradiance components (GHI/DHI/DNI) onto the tilted module surface using a
*sky-diffusion* (transposition) model. The default, `"isotropic"`, treats
diffuse sky radiance as uniform — simple and robust, but it underestimates POA
on clear days because it ignores circumsolar and horizon brightening.

Anisotropic models capture those effects and are generally more accurate; over
a full year, Perez can raise modeled POA by a few percent relative to
isotropic at mid-latitude sites. Set `transposition_model` to any of:

`isotropic` (default), `klucher`, `haydavies`, `reindl`, `king`, `perez`,
`perez-driesse`.

```python
breos.App({
    "location": "porto",
    "n_modules": 10,
    "annual_consumption_kwh": 4000,
    "transposition_model": "perez",
})
```

The extra inputs the anisotropic models need (extraterrestrial DNI and, for
the Perez variants, relative airmass) are derived internally from the time
index and solar position, so no additional weather columns are required. All
models are provided by
[`pvlib.irradiance.get_total_irradiance`](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.irradiance.get_total_irradiance.html).

### Ground reflectance (albedo)

Every transposition model adds a ground-reflected diffuse component, which
depends on how reflective the ground around the array is. By default BREOS
uses pvlib's 0.25 albedo. If you know your site, set it explicitly — either a
numeric `albedo` (0-1) or a named `surface_type` that pvlib maps to an albedo
(`"snow"` ≈ 0.65, `"sea"`, `"grass"`, `"sand"`, `"urban"`, …). Set one or the
other, not both. A snowy or sandy foreground can add a few percent to annual
POA on tilted arrays.

```python
breos.App({
    "location": "berlin",
    "n_modules": 10,
    "annual_consumption_kwh": 4000,
    "transposition_model": "perez",
    "surface_type": "snow",     # or: "albedo": 0.65
})
```

### Perez coefficient set

The `perez` model uses an empirically fitted coefficient set. `model_perez`
selects it (default `"allsitescomposite1990"`); the other sets are
location/era-specific fits from the Perez papers and are only consulted when
`transposition_model = "perez"`.

## Cost and emissions presets

Built-in presets are packaged with BREOS. Editable copies and examples live
in `configs/base/` and `configs/examples/`.
Pass the key, then use the optional `costs` table for project-specific values.
Explicit overrides win over the named preset; preset values win over
{py:class}`~breos.CostParams` defaults:

```python
breos.App({
    "location": "porto",
    "n_modules": 10,
    "annual_consumption_kwh": 4000,
    "cost_preset": "residential_pt",
    "costs": {
        "electricity_cost": 0.22,
        "storage_cost_per_kwh": 425.0,
    },
    "emissions_country": "PT",
})
```

TOML uses a dedicated table:

```toml
cost_preset = "residential_pt"

[costs]
electricity_cost = 0.22
storage_cost_per_kwh = 425.0
```

The accepted keys follow the packaged cost-catalogue names; the
[key reference](config-reference.md#costs) lists them with their defaults.
Unknown keys and negative or non-finite values are rejected before simulation.

For full control, build a {py:class}`~breos.CostParams` and
{py:class}`~breos.EmissionsParams` yourself and call the lower-level
functions documented in the [Cost and emissions API](../api/cost-analysis.md).

## Time-of-use tariffs

A `[tariff]` table prices energy by period instead of at one flat rate. It
names a bundled schedule, which fixes the periods in local civil time, and
you give the prices, which BREOS does not bundle:

```toml
[tariff]
schedule = "pt_mainland_2026_daily_bi"   # see the schedule list in the Tariffs API page
currency = "EUR"
import_prices = { peak = 0.28, off_peak = 0.11 }
export_prices = { all = 0.05 }
fixed_charge_per_day = 0.25              # optional, default 0
# study_date = 2027-07-01                # needed for a 2027 schedule on an earlier simulated year
```

- Every period of the schedule needs an import and an export price, or an
  `all` price for every period. A period the schedule does not have is an
  error; there is no fallback to another schedule.
- The schedule must be defined in the location's timezone, and the
  resolution fine enough for its boundaries: the Portuguese tri-hourly and
  2027 schedules change on the half hour, so they need `resolution = "15min"`.
- A tariff replaces the flat `costs.electricity_cost`,
  `costs.electricity_sold_cost` and `costs.daily_power_cost`, so setting
  those as well is an error. CAPEX, O&M and replacement costs still come from
  the cost preset, in the same currency.
- Dispatch does not change: the battery still maximises self-consumption.
  The tariff changes what the energy costs. Each year row records its import
  cost, export revenue, no-system import cost and fixed charge at year-1
  prices; the projection escalates and discounts them as it does flat prices.
- Every project year replays the start-year calendar, so weekdays and
  holidays do not advance; provenance records this as
  `calendar_policy = "replay_start_year"`, with the schedule, prices and
  their hashes.

Monte Carlo prices every trajectory with the same tariff. Projected
optimization accepts the same tariff table in its nested config; see
[Optimization](optimization.md#price-a-design-with-a-time-of-use-tariff).
To compare several offers, see [Compare tariffs](recipes.md#compare-tariffs).

## Smart charging

A `[smart_charging]` table sets when the battery may discharge and when the
grid may charge it, by tariff period. It needs a `[tariff]` and a battery.
Omitting it, or setting `mode = "disabled"`, is greedy self-consumption with
unchanged results:

```toml
[smart_charging]
mode = "fixed_target"               # or "disabled"
target_usable_fraction = 0.50       # 0 is battery_min_soc, 1 is battery_max_soc
charge_periods = ["off_peak"]
discharge_periods = ["mid_peak", "peak"]
grid_charge_efficiency = 0.95       # required: AC-to-DC conversion of the grid-charging path
grid_import_limit_w = 5000          # optional: grid charging keeps total import below this
```

- In a charge period the grid may charge the battery toward
  `target_usable_fraction` of the usable window. In a discharge period the
  battery may discharge to the load. In a period in neither list it does
  neither. PV may charge the battery in every period.
- `target_usable_fraction` is a fraction of the usable window between
  `battery_min_soc` and `battery_max_soc`, not of nominal capacity. The window
  shrinks with temperature and state of health, and the target moves with it.
- The period names must exist in the tariff's schedule, and the two lists
  must not share a period: every step either charges or discharges.
- `grid_charge_efficiency` has no default, because the inverter model has no
  AC-to-DC path to derive one from. Stored energy then also passes through
  the battery's own charge efficiency.
- `grid_import_limit_w` limits grid charging only: in each step it may import
  up to the limit minus what the load already imports. Load import is never
  cut, so a load above the limit still imports in full. Omit it for no site
  limit. Grid charging is also bounded by the battery's charge power and by
  the inverter's AC rating, which PV output uses first.
- `mode = "disabled"` accepts no other key. Unknown keys are errors.
- Grid charging runs after PV in each step and never while PV is exported,
  so it never takes PV self-consumption. The grid-charge import is part of
  `grid_import_kwh`, and the tariff prices it like any import.

Results gain a `smart_charging` block: per year, the grid-charge AC energy,
its conversion loss, its cost at year-1 prices, and battery delivery to load
split into PV, grid and unattributed origin. It also gives the stored energy
by origin at the start and end of the project, since stored energy carries
from year to year (`terminal_convention = "physical_carry"`). Only PV-origin
battery delivery counts as self-consumption. Avoided emissions use net
exchange: grid energy shifted through the battery is imported, so it earns
nothing, and its round-trip loss counts against the system.
`provenance.smart_charging` records the parameters, the hash of the resolved
instructions and the tariff's schedule hash.

Monte Carlo applies the same instructions to every trajectory, and projected
optimization to every candidate design with a battery; both record the same
provenance. See `configs/examples/smart-charging-portugal.toml`.

## Load profiles

The public package default is `load_profile = "demandlib_h0"`, a
demandlib-derived H0 example bundled with BREOS. The other standard profiles,
`eredes_btn_a`, `eredes_btn_b`, `eredes_btn_c`, `bdew_h0` and `ree_2.0td`, are
supported when you provide the required CSV files yourself through
`rlp_directory`. `load_profile = "custom"` reads any CSV you name with
`load_profile_file`, `load_profile_column` and `load_profile_unit`. Keys are
case-insensitive; the numeric keys (`"1"` to `"8"`) and the aliases `h0`,
`default` and `crest` were removed in 0.7.0.

```python
breos.App({
    "location": "porto",
    "n_modules": 10,
    "annual_consumption_kwh": 4000,
    "load_profile": "eredes_btn_c",
    "rlp_directory": "/path/to/licensed/rlp/files",
    "resolution": "15min",
})
```

Use external BDEW, E-REDES, REE, or custom profiles only under terms that
permit your intended use. See [Load Profile Data](../legal/load-profile-data.md)
for the expected filenames and the reason these CSVs are not bundled.
