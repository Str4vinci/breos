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

Unknown keys are rejected at load time, at the top level and inside every
table. A misspelled key such as `batery_kwh` raises an error listing the
offending key rather than being silently ignored (which would quietly fall
back to the default). The `[sweep]` and `[montecarlo]` runner sections are
allowed in an App config; App checks the `[montecarlo]` keys, and
`breos sweep` checks the `[sweep]` keys.

## Leap years

Real calendar-year load profiles follow `start_date`: leap years contain
8,784 hourly (35,136 quarter-hourly) intervals and preserve exact annual
energy. An 8,760-hour TMY restamped onto a leap year gets the same treatment:
29 February is a copy of 28 February for the weather, and 1 March onwards
keeps its own data. External load profiles placed by position likewise copy
28 February. The bundled demandlib H0 instead uses the nearest source day of
the leap day's weekday, Saturday or Sunday type, and a dated E-REDES file the
nearest of its working-day, Saturday or Sunday/holiday class. The result's weather
provenance records the copied weather day under `leap_day`.

## Hourly weather at 15-minute resolution

With `resolution = "15min"`, hourly weather is reconstructed at quarter-hour
representative times. The top-level `irradiance_resampling` key (CLI:
`--irradiance-resampling`) applies in App and Monte Carlo; projected
optimization reads the same values from `simulation.irradiance_resampling`:

- `"auto"` (default) uses `"clear_sky_energy_conserving"` for weather declaring
  `radiation_time_basis = "interval_mean"`, and `"clear_sky"` otherwise.
- `"clear_sky"` interpolates each component's clear-sky ratio with makima,
  retaining a 5 W/m² denominator regulariser and no upper ratio cap. Where a
  component's clear-sky value at either bracketing source point is at most
  5 W/m², that component is interpolated linearly instead, which guards dawn
  and dusk. Negative overshoot is clipped and zero clear-sky support stays
  zero.
- `"clear_sky_energy_conserving"` then scales each source hour's four quarters
  to reproduce that hour's mean, independently for GHI, DNI and DHI. It
  raises for instantaneous or undeclared input. Positive source hours with
  no reconstructed support receive a flat fill, recorded in provenance.

PVGIS TMY is instantaneous with a provider offset, so `"auto"` resolves to
`"clear_sky"`. Open-Meteo interval means and EPW use hourly conservation.
Without coordinates, direct interpolation replaces clear-sky reconstruction;
the policy still controls hourly conservation. Already-quarter-hour weather
is unchanged.

No GHI/DNI/DHI closure is enforced, and DHI is never clipped to GHI.
Weather metadata and result provenance record requested and resolved policy,
interpolation method, the daylight quarter-hours filled by the dawn/dusk
fallback and the zero-support hour counts, per component. With coordinates and all three components, `irradiance_closure`
records before/after conservation diagnostics over daylight quarter-hours:
the GHI-weighted mean absolute residual in W/m² and the absolute residual
energy divided by GHI energy. Solar position uses apparent zenith at the
same representative times as reconstruction.

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

The native BREOS degradation path is calibrated for LFP cells only. Omit
`degradation_engine` for native behavior, or set `degradation_engine="blast"`
and a stable `blast_model` key. See the
[degradation model reference](../api/degradation-models.md) for discovery,
precedence, provenance, and engine selection.

## Battery replacement at the end of the horizon

BREOS replaces the battery when its state of health falls to
`battery_eol_percentage`. The check runs when each degradation period closes.
A degradation period is a fixed window of one day of simulation steps (24
hourly or 96 fifteen-minute steps), counted from the start of the simulated
span; it is not a civil or tariff day. By default the check also runs when
the horizon's final period closes. A pack that reaches end of life there is
bought and priced, but it serves no step inside the horizon, so the result
depends on whether the crossing falls just before or just after the horizon
ends.

`battery_allow_terminal_replacement = false` skips only that final
replacement:

- The final period is the one that ends on the last simulated step of the
  last project year. When the span is whole days, it is the last whole day.
  When the span ends with a partial day, it is that partial period, and the
  whole day before it can still replace, because the new pack serves the
  remaining steps. A span shorter than a day, such as a short `[period]`,
  has one partial period, and that period is the final one.
- The final period is still dispatched, aged and recorded, and its remaining
  rainflow cycles are still counted. The result reports the old pack's state
  of health, cycles, resistance and stored energy. No replacement, replaced
  capacity or replacement cost is recorded for it.
- Every earlier period replaces as usual. This includes the close of each
  earlier project year, because the next year uses the new pack. A
  replacement earlier in the final year is still counted and priced.
- Smart-charging decisions do not change. The key controls only the
  replacement in the simulated battery.

Only a horizon whose final period reaches end of life gives a different
result. Monte Carlo applies the key to the final year of each trajectory. The
optimizer takes it as `[battery] allow_terminal_replacement`. The default is
`true`. The resolved value is in `provenance.resolved_config`, and the
optimizer records it in `battery_replacement_treatment`. A direct
{py:class}`~breos.battery.BatteryConfig` call treats its own span as the horizon, so a
caller that splits one horizon across several calls must keep
`allow_terminal_replacement=True` on every span except the last.

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

Built-in presets are packaged with BREOS; [Packaged options](options.md)
lists them, and `configs/examples/` has runnable configs that use them.
Pass the key, then use the optional `costs` table for project-specific values.
Explicit overrides win over the named preset; preset values win over
{py:class}`~breos.economics.CostParams` defaults:

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

For full control, build a {py:class}`~breos.economics.CostParams` and
{py:class}`~breos.emissions.EmissionsParams` yourself and call the lower-level
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
  error; there is no fallback to another schedule. A custom schedule with
  month seasons is priced by season instead: each season's table prices the
  periods that season's rules use, or gives `all`, and pricing a period the
  season never uses is an error.
- Instead of `schedule`, you can define `[tariff.custom_schedule]` inline.
  Set `identifier`, `version`, `timezone`, `cycle`, `periods`, and one or
  more `[[tariff.custom_schedule.rules]]` tables. Do not set both schedule
  forms. The rules must cover every day type and season exactly once, and
  each rule's intervals must cover the whole local day without gaps or
  overlaps. See [Custom App schedules](../api/tariffs.md#custom-app-schedules)
  for a complete example.
- A custom schedule can name calendar-month `seasons`, such as quarters,
  instead of the standard/DST seasons. Its rules then select a season by
  name, and each price list may give a table of period prices for every
  season. See [Month seasons](../api/tariffs.md#month-seasons).
- `currency` must be `"EUR"`, the only currency BREOS accepts so far, and
  the currency of the cost presets. BREOS does not convert.
- Holidays of a custom schedule are optional and explicit. The
  `[tariff.custom_schedule.holidays]` table needs `day_type`, the day type
  holidays follow, and `dates`, which maps each covered year to its dates;
  provide the complete calendar you intend for each year the simulation can
  use. A run in a year absent from that map fails rather than guessing or
  reusing dates. The bundled `es_2_0td` carries Spanish national holidays for
  2026 only.
- The schedule must be defined in the location's timezone, and the
  resolution fine enough for its boundaries. Of the bundled schedules, only
  `pt_mainland_2026_daily_bi` and `es_2_0td` run at hourly resolution; the
  others need `resolution = "15min"`. See the boundary-step column in
  [Bundled schedules](../api/tariffs.md#bundled-schedules).
- A tariff replaces the flat `costs.electricity_cost`,
  `costs.electricity_sold_cost` and `costs.daily_power_cost`, so setting
  those as well is an error. CAPEX, O&M and replacement costs still come from
  the cost preset, in the same currency.
- Without [`[smart_charging]`](#smart-charging) the dispatch does not
  change: the battery still maximises self-consumption, and the tariff
  changes only what the energy costs. Each year row records its import
  cost, export revenue, no-system import cost and fixed charge at year-1
  prices; the projection escalates and discounts them as it does flat prices.
- Every project year replays the start-year calendar, so weekdays and
  holidays do not advance; provenance records this as
  `calendar_policy = "replay_start_year"`, with the schedule, prices and
  their hashes.

Monte Carlo prices every trajectory with the same tariff. Projected
optimization accepts the same tariff table in its nested config; see
[Optimization](optimization.md#price-a-design-with-a-time-of-use-tariff).
To compare several offers, see {doc}`Simple, bi-hourly and tri-hourly tariffs <../gallery/tariffs/plot_09_which_tariff>`.

## Estimated battery residual value

At the end of the project, BREOS can estimate the installed battery's
residual value by scaling its replacement cost by the fraction of capacity
health remaining above the replacement threshold. This assumes the value
declines linearly with that fraction. The discounted estimate is reported
beside the NPV excluding residual value; it does not estimate resale
proceeds. The table and the result fields keep their names,
`[terminal_value]` and `terminal_health_credit*`:

```toml
[terminal_value]
basis = "battery_health_fraction"
```

The default `basis = "none"`, also used when the table or key is omitted,
leaves the residual value out. No other keys are accepted. It inherits
`battery_eol_percentage` and the resolved full replacement-pack price,
including cost overrides; there is no separate threshold or price.

With final SOH fraction `h` and replacement threshold `h*`, the fraction is
`clip((h - h*) / (1 - h*), 0, 1)`. SOH is measured after the last step's
aging, terminal cycle finalization and any replacement. A fresh pack is
valued at the full replacement price; at or below threshold it is valued at
zero. Inputs must be finite and `h* < 1`. For example, a pack at 90 % SOH
with an 80 % threshold is valued at half the replacement price. That is half
of the capacity-health margin, not necessarily half of the pack's remaining
service life or market value: degradation can be nonlinear, and capacity
health omits resistance-related limits.

The nominal residual value at exactly project time `T` is the fraction times the
replacement price `C0 × (1 + inflation_rate)^T ×
(1 - replacement_cost_learning)^T`. The present value divides it by
`(1 + discount_rate)^T`. This is the existing replacement-outlay
convention with a t = 0 price and exponent T, rather than the T − 1
escalation used for annual energy prices. PV modules, the inverter and
stored energy are excluded.

Only the pack present at the end is valued. Every replacement outlay
remains, and replacement policy is unchanged. If
`battery_allow_terminal_replacement = false` skips the final swap, the old
pack remains and earns zero at or below threshold. Enabled PV-only and
zero-capacity runs explicitly report zero. A partial `[period]` run keeps
all lifetime credit fields null.

App reports the three residual-value fields beside the NPV excluding
residual value. `App.revalue` can enable, disable or reprice the estimate using retained
final health without simulating again for it. Monte Carlo computes it for
each trajectory before aggregation. Projected optimization accepts but
ignores this table, reports no adjusted values for evaluated designs and
continues ranking on the NPV excluding residual value. Cashflows, paybacks,
LCOE, emissions, dispatch and aging exclude it and stay unchanged. See
[Residual value in the results](interpreting-results.md#estimated-battery-residual-value).

## No-system reference tariff

The savings of a system are measured against the household without it. By
default that household pays the system's own prices: the `[tariff]`, or the
flat `costs` prices, and the same fixed charge. A `[reference_tariff]` prices
the household without the system on its own tariff instead, for example the
offer it has today while the system runs on a time-of-use offer:

```toml
[reference_tariff]
schedule = "pt_mainland_2026_daily_bi"   # or custom_schedule, or neither for one flat price
currency = "EUR"
import_prices = { peak = 0.2310, off_peak = 0.1210 }
fixed_charge_per_day = 0.30              # required; use 0 for no fixed charge
# import_price_escalation = 0.03         # optional, default the system's import escalation
```

- The no-system cost of each year is the whole household load at the
  reference import prices, plus the reference fixed charge, both escalated at
  `reference_tariff.import_price_escalation`. Without that key they escalate
  at the system's import escalation: `import_price_escalation` or, when that
  is unset, `inflation_rate`. An explicit `0` keeps the reference prices
  constant.
- The reference has no export prices: the household without a system exports
  nothing.
- `fixed_charge_per_day` is required. Set it to the fixed charge the household
  pays without the system, or explicitly to `0` when there is no fixed charge.
- Without `schedule` or `custom_schedule` the reference is one flat price,
  `import_prices = { all = <price> }`, and takes no `boundary_policy` or
  `study_date`. With a schedule, the prices follow the `[tariff]` rules:
  every period priced, or `all`, a schedule in the location's timezone, and a
  resolution fine enough for the schedule's boundaries. A custom schedule
  with calendar-month `seasons` also accepts prices per season and period,
  such as `import_prices = { q1 = { peak = 0.30, off_peak = 0.12 }, ... }`.
  Price every season and exactly the periods its rules use, or give `all`
  within that season. Bundled, DST-season and flat references take prices
  per period only. See [month seasons](../api/tariffs.md#month-seasons).
- The reference can be set with or without a `[tariff]`. Its `currency` must
  be the result's currency: the `[tariff]` currency, or EUR with flat prices.
  BREOS does not convert.
- The reference never changes the dispatch, and the costs with the system do
  not change. Every project year replays the start-year calendar, as the
  system tariff does. A `[period]` run bills the reference fixed charge on the
  window's civil days.
- `App.revalue` re-prices a reference that is added, changed or removed. It
  does not simulate again for the reference, also under `daily_persistence`.
- Monte Carlo prices the sampled load of each trajectory at the reference, so
  the costs with and without the system stay paired. Projected optimization
  accepts the same table in its nested config, and its NPV objective is the
  saving against the reference.

Results gain `provenance.reference_tariff`, with the schedule, prices, fixed
charge, calendar policy and the escalation used, and the month partition
`seasons` when configured. The no-system cost
components are reported with or without a reference; see [Year-1 money
keys](interpreting-results.md#year-1-money-keys). BREOS does not choose the
cheapest offer the household could have had: to compare candidates, run each
one as the reference.

## Annual network credit

Some network tariffs reduce a household's network charges by an annual
amount that cannot exceed what the household paid for the network that
year. Stromnetz Berlin's §14a EnWG Modul 1 is one: 146.58 EUR a year, gross,
capped at the connection's network charges. A household with PV and storage
imports less, so the cap can bind and it receives less than the same
household without the system. An `annual_network_credit` table under
`[tariff]` gives the system household its credit:

```toml
[tariff.annual_network_credit]
amount_per_year = 146.58           # the annual reduction
network_fixed_per_year = 39.70     # network part of the fixed charge
network_import_prices = { low = 0.0311, standard = 0.0888, high = 0.1659 }
```

- The household's eligible network charges for a year are its grid import
  times the network price of each step, plus `network_fixed_per_year`. Its
  credit is `amount_per_year`, capped at those charges.
- The network values are the network parts of the tariff's own prices, with
  the same taxes. They set the cap and are never added to the bill: the
  import cost and the fixed charge do not change. A period's network price
  may not exceed its import price, and `network_fixed_per_year` may not
  exceed 365 days of `fixed_charge_per_day`.
- `network_import_prices` has the shape of `import_prices`: by period, or by
  month season and period, with `all` as the fallback. All three keys are
  required; an explicit 0 is valid.
- The household without the system gets its own credit, capped at its own
  network charges on the whole load. Without a `[reference_tariff]` it is the
  `[tariff]` table. With one, it is
  `[reference_tariff.annual_network_credit]`, and without that table the
  household has no credit. A reference without a schedule takes
  `network_import_prices = { all = <price> }`.
- Each credit escalates with its household's import prices and lowers that
  household's yearly cost before NPV and payback. LCOE leaves it out, as it
  leaves out the other tariff outcomes.
- A simulated year gets the annual amounts in full, also a leap year. A
  `[period]` window of `d` civil days gets `d / 365` of them, or `d / 366`
  in a leap year.
- The credit never changes the dispatch. `App.revalue` re-prices a credit
  that is added, changed or removed without simulating again, also under
  `daily_persistence`. Monte Carlo caps each trajectory at its own import
  and load, and the projected optimizer's NPV objective includes both
  credits.
- This is not a billing engine: metering, control-box and other connection
  charges are not modelled.

Results then carry each household's eligible network charges and credit;
see [Year-1 money keys](interpreting-results.md#year-1-money-keys).

## Smart charging

A `[smart_charging]` table sets when the battery may discharge and when the
grid may charge it, by tariff period. Every mode except `disabled` needs a
`[tariff]` and `battery_kwh > 0`. Omitting the table, or setting
`mode = "disabled"`, is greedy self-consumption with unchanged results:

```toml
[smart_charging]
mode = "fixed_target"               # or "disabled", "discharge_only", or the experimental "daily_persistence"
target_usable_fraction = 0.50       # 0 is battery_min_soc, 1 is battery_max_soc
charge_periods = ["off_peak"]
discharge_periods = ["mid_peak", "peak"]
grid_charge_efficiency = 0.95       # required: AC-to-DC conversion of the grid-charging path
grid_import_limit_w = 5000          # optional: grid charging keeps total import below this
overlap_policy = "reject"           # default; "hold_target" permits overlap
```

- In a charge period the grid may charge the battery toward
  `target_usable_fraction` of the usable window. In a discharge period the
  battery may discharge to the load. In a period in neither list it does
  neither. PV may charge the battery in every period.
- `target_usable_fraction` is a fraction of the usable window between
  `battery_min_soc` and `battery_max_soc`, not of nominal capacity. The window
  shrinks with temperature and state of health, and the target moves with it.
- The period names must exist in the tariff's schedule, and the two lists
  must not share a period under the default `overlap_policy = "reject"`.
  With `overlap_policy = "hold_target"` (`fixed_target` and
  `daily_persistence`), the grid target is also the discharge floor on steps
  in both lists: above it the battery may discharge down to it; below it the
  grid may charge up to it. It never charges and discharges in the same
  step. Both bounds move together with temperature and health. PV may still
  charge above the target. Steps in only one list keep their usual behavior.
  The floor holds the energy up to the target, not all the stored energy:
  to keep the battery from discharging in a period at all, leave that period
  out of `discharge_periods`.
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
instructions, `overlap_policy` and the tariff's schedule hash.

To allow discharge in every period while retaining an off-peak target, use
`charge_periods = ["off_peak"]`, list every tariff period in
`discharge_periods`, and set `overlap_policy = "hold_target"`.
`disabled` and `discharge_only` refuse `hold_target` because they have no grid
target. `daily_persistence` accepts it; see
[Daily persistence](#daily-persistence-experimental).

Monte Carlo applies the same instructions to every trajectory, and projected
optimization to every candidate design with a battery; both record the same
provenance. See `configs/examples/smart-charging-portugal.toml`.

### Discharge only

`mode = "discharge_only"` restricts when the battery discharges, without
any grid charging:

```toml
[smart_charging]
mode = "discharge_only"
discharge_periods = ["peak"]        # required: the battery holds its charge in every other period
```

- In a discharge period the battery may discharge to the load. In every
  other period it holds its charge. PV may charge the battery in every
  period, and the grid never does.
- The mode takes `discharge_periods` only. `charge_periods`,
  `target_usable_fraction`, `grid_charge_efficiency`, `grid_import_limit_w`
  and the planner settings are errors, since no grid charging takes place.
- A peak-only policy lists the peak period; a selected-period policy lists
  several. Discharge in every period is greedy self-consumption: use
  `mode = "disabled"`, or list every period, which gives the same results.
- It runs on the same dispatch instructions as `fixed_target`, in App, Monte
  Carlo and projected optimization, on both execution backends.
  `provenance.smart_charging` records the mode, the discharge periods and
  the instruction and schedule hashes, with the grid-charge settings unset.

### Daily persistence (experimental)

`mode = "daily_persistence"` keeps the fixed-target layout but chooses the
grid-charge target once per day while the run goes, instead of fixing it in
the configuration. It is experimental and runs in `App` only:

```toml
[smart_charging]
mode = "daily_persistence"
charge_periods = ["off_peak"]
discharge_periods = ["peak"]
grid_charge_efficiency = 0.95       # required, as for fixed_target
grid_import_limit_w = 5000          # optional
# Optional planner settings, shown at their defaults:
forecast_horizon_days = 2           # days the planner looks ahead, today included
target_levels = 11                  # candidate targets 0, 0.1, ..., 1 of the usable window
soc_states = 21                     # stored-energy grid points of the planner
wear_cost_per_kwh = 0               # planning weight per kWh the battery discharges
decision_boundary = "civil_day"     # or "charge_window_start"
```

At each local midnight in the tariff's timezone the mode:

1. forecasts PV, load and temperature for the planning window by repeating
   the last complete local day that has been simulated. Slots match by local
   wall-clock time, so a 23- or 25-hour day keeps every tariff period in
   place: a repeated fall-back hour reuses the observed hour, and the hour a
   spring-forward day skipped is interpolated between its neighbours;
2. solves a small dynamic program over that forecast and the known tariff
   prices, from the battery's measured stored energy, state of health and
   efficiencies, choosing one target per day from `target_levels` candidates;
3. applies the first day's target to that day's charge periods, and plans
   the next day again from what actually happened.

The table takes the same `charge_periods`, `discharge_periods`,
`grid_charge_efficiency`, `grid_import_limit_w` and `overlap_policy` as
`fixed_target`, and refuses `target_usable_fraction`, since the planner
chooses it.
`forecast_horizon_days`, `target_levels` and `soc_states` are integers (not
booleans or `2.0`), `wear_cost_per_kwh` is a number of at least 0 (see
"Wear cost in the planner" below), and `decision_boundary` is `"civil_day"`
or `"charge_window_start"` (see "Deciding at the start of each charge
window" below); `fixed_target`,
`discharge_only` and `disabled` refuse every planner setting. `target_levels = 1` is valid and selects the sole target 0.
The run starts with no complete day to repeat, so until one
complete local day has been observed it sets no grid target
(`warm_start_policy = "no_grid_until_one_complete_local_day"`); PV charging
and the discharge periods run as configured. A partial first or last day
does not count as observed, and the two ends of a `[period]` are never
joined into one day.

The planner's terminal target is the day's starting energy, capped at the
max-SOC capacity at the final forecast temperature and current state of
health. Energy that ends the window below that target is priced at the
cheapest import price of a step that may grid-charge, through both charge
efficiencies (`planner_terminal_policy = "preserve_start_energy"`). When no
step permits grid charging, it uses the cheapest import price of any step.
The same policy applies at the project's final horizon.
This only keeps a rolling plan from treating an empty battery at the end of
its window as free. It is not an instruction: the simulated battery still
carries its stored energy, origins and degradation from year to year
(`terminal_convention = "physical_carry"`).

To discharge in every period while the planner still chooses an off-peak
target, list every tariff period in `discharge_periods` and set
`overlap_policy = "hold_target"`:

```toml
charge_periods = ["off_peak"]
discharge_periods = ["off_peak", "peak"]
overlap_policy = "hold_target"
```

On a step in both lists, the day's planned target is also the discharge
floor, as under `fixed_target`: above it the battery may discharge down to
it; below it the grid may charge up to it. The planner tries every candidate
target with its own floor, so the instructions it executes are the ones it
planned with. On a day with no grid target, which is every warm-start day,
such a step has no floor, and the battery may discharge to
`battery_min_soc` there.

The target changes from day to day. A shared period that crosses midnight
holds the previous day's target before midnight and the new day's after it,
so when the new target is lower the battery may discharge, in that same
period, grid energy it bought the evening before. The planner counts that
cost when it chooses the targets. Deciding at the start of each charge
window, below, avoids this split.

Keep these modelling assumptions in mind when reading the results:

- The forecast is naive persistence, not a weather or load forecast.
- The end-of-window price is a simple continuation value, not a learned one.
  It assumes missing energy can be bought back at the cheapest chargeable
  price within the window, and is biased when recharge prices beyond the
  window differ.
- State of health and efficiencies stay fixed inside each short solve; the
  simulation still ages the battery every day.
- The planner weighs no battery ageing unless `wear_cost_per_kwh` is set.

`provenance.smart_charging` of such a run records `experimental = true`, the
controller and planner versions, the effective planner settings, the
forecast, warm-start and terminal policies, the schedule hash, an
`instruction_hash` of the instructions the run executed in every project
year, and the stored energy by origin at the start and end of the project.
[`App.revalue`](../api/app.md#revalue-a-finished-run) simulates the run
again when the import or export prices change, since they move the plan; a
change to the fixed charge alone is re-priced. Monte Carlo and projected
optimization refuse the mode, because they share one set of static
instructions across trajectories and candidate designs.

The planner simulates each candidate target with the same dispatch step as
the run, several hundred times a day, so the mode is much slower than
`fixed_target`. Use `execution_backend = "numba"` (the `breos[fast]` extra); the Python
backend at 15 minutes gives a warning.

#### Wear cost in the planner

Without a wear cost the planner charges from the grid whenever the price
spread pays for the conversion and battery losses, however little it saves.
`wear_cost_per_kwh` adds a price, in the tariff's currency, for each kWh of
DC energy the battery discharges (`Battery_Discharge_DC`) to the cost each
daily solve minimises. Grid energy is then bought only when the spread also
pays for the cycling it causes. The default is 0, which gives the same
results as a table without the key.

It is a planning weight, not a degradation model. It changes only which
targets the planner picks. It does not enter the import cost, the export
revenue or the NPV, and the battery still ages and is replaced as the
degradation model gives. The weight applies to all discharge, including
PV-charged energy, because a target the planner raises adds cycling
whatever the energy's origin. Charged energy that the battery still holds
at the end of the planning window carries no wear cost; the terminal value
prices it as energy only.

BREOS does not compute a value for you. One simple estimate is the price of
a replacement pack divided by the energy the battery can discharge before it
reaches end of life:

```text
wear_cost_per_kwh = replacement pack price / usable discharge throughput to end of life
                  = 2500 / 20 000 = 0.125 per kWh
```

Here the throughput is taken from a warranty, or from the expected cycles to
end of life times the average usable energy per cycle. This counts all of the
pack's ageing against cycling. Part of it is calendar ageing, which happens
whether the battery cycles or not, so a lower value can be argued. Compare
runs at several values rather than reading one as exact.
`provenance.smart_charging.wear_cost_per_kwh` records the value a run used.

#### Deciding at the start of each charge window

By default the planner decides at local midnight, so an off-peak window
that crosses midnight, such as Bi-hourly 22:00–08:00, is split between two
days' targets. With

```toml
decision_boundary = "charge_window_start"
```

the planner decides one target at the start of each charge window instead,
and the target holds until the next window starts:

- A charge window is a run of consecutive steps whose period is in
  `charge_periods`. Adjacent charge periods form one window, a day can have
  several windows, and a window can cross midnight or last a whole weekend:
  under the 2026 weekly cycle, off-peak from Saturday 22:00 to Monday 07:00
  is one window with one target.
- The decision sees the battery's state at the window start and only the
  data observed before it. Its forecast repeats the local day before the
  decision: today's observed hours, and the rest from the last complete
  day.
- Each planning stage is one window. The planner looks ahead over the
  windows that start within `forecast_horizon_days` days.
- Under `overlap_policy = "hold_target"` the floor is the window's one
  target for the whole window, across midnight.
- A window that starts before one complete local day has been observed
  gets no grid target, and holds none until the next window.
- At least one tariff period must be outside `charge_periods`, or a window
  would never end.

`provenance.smart_charging.decision_boundary` records the boundary; under
`charge_window_start` the controller and planner versions are `"2"`. The
daily-target oracle takes the same choice as `--decision-boundary`, so a
perfect-information replay stays comparable with the controller.

## Load profiles

The public package default is `load_profile = "demandlib_h0"`, a
demandlib-derived H0 example bundled with BREOS. The other standard profiles,
`eredes_btn_a`, `eredes_btn_b`, `eredes_btn_c`, `bdew_h0` and `ree_2.0td`, are
supported when you provide the required CSV files yourself through
`rlp_directory`. `load_profile = "custom"` reads any CSV you name with
`load_profile_file`, `load_profile_column` and `load_profile_unit`. Keys are
case-insensitive. The numeric keys and the aliases `h0`, `default` and
`crest` were removed in 0.7.0;
[Migrating from BREOS 0.6.2](interpreting-results.md#migrating-from-breos-062)
maps them to the current keys.

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
  June. `end` may be 1 January of the next year. A window of the whole year
  raises; omit `[period]` for that.
- The load profile is built for the whole year and scaled to
  `annual_consumption_kwh` as usual, and the window gets its share of it.
  Weather, PV, load and battery temperature are then cut to the window.
- The weather must cover the whole window, in civil time. Weather that
  covers one UTC year, such as an Open-Meteo CSV, cannot serve a window that
  touches the civil year edge away from UTC, and weather stamped at half
  past the hour cannot place a window that starts at local midnight. Both
  raise `ValueError` before anything is simulated. PVGIS TMY weather covers
  both year edges.
- The window runs once, from the battery's initial state; nothing is
  repeated or carried over. `projection_years` is not used: setting it gives
  a warning, and the result records `projection_years_used = 1`.
- The fixed charge (`costs.daily_power_cost`, or a tariff's
  `fixed_charge_per_day`) is billed on the window's civil days.
- Energy results cover the window, and the lifetime economics are `None`;
  see [Period runs](interpreting-results.md#period-runs).

A PV-only window equals the same steps of a full-year run exactly. With a
battery the dispatch differs, because the window starts from the battery's
initial state of charge and health.

Monte Carlo and the optimizer reject `[period]`, because they rank designs on
lifetime economics. `breos sweep` accepts it and can vary `period.start` and
`period.end`. `App.revalue` re-prices the window; its lifetime economics stay
`None`.
