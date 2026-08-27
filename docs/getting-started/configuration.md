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

| Key | Default | Description |
|---|---|---|
| `location` | *required* | Preset key (e.g. `"porto"`, `"berlin"`) or `{"latitude": ..., "longitude": ..., "timezone": ...}` |
| `n_modules` | *required unless `pv_arrays` is set* | Number of PV modules |
| `pv_arrays` | `None` | List of arrays with `modules`, `module`, `tilt`, and `azimuth`. When present, the array module total overrides `n_modules` |
| `annual_consumption_kwh` | *required* | Annual electricity demand (kWh) |
| `battery_kwh` | `0.0` | Nominal battery capacity in kWh (`0` = no battery). The SOC window sets the usable share — see [below](#battery-capacity-and-the-soc-window) |
| `pv_module` | `None` | Module key from the built-in catalogue. `None` uses the first available |
| `load_profile` | `"1"` | Bundled demandlib-derived H0 profile; `"demandlib_h0"` is the friendly alias (see {py:func}`~breos.load_profiles.load_profile`) |
| `rlp_directory` | `None` | Directory containing licensed external RLP CSVs for non-bundled load profiles |
| `tilt` | auto | Tilt angle (degrees). Auto-estimated from latitude when `None` |
| `azimuth` | auto | Surface azimuth (degrees). Auto-set to 180 in the northern hemisphere |
| `tracking` | `"fixed"` | Tracking mode (`"fixed"`, `"single_axis"`, or `"dual_axis"`) |
| `axis_tilt` | `0.0` | Single-axis tracker axis tilt |
| `axis_azimuth` | auto | Tracker axis azimuth. Auto-set from latitude when `None` |
| `max_angle` | `60.0` | Single-axis tracker maximum rotation angle |
| `backtrack` | `True` | Whether single-axis trackers backtrack to avoid row shading |
| `gcr` | `0.35` | Ground coverage ratio for single-axis tracking and infinite-sheds bifacial geometry |
| `cross_axis_tilt` | `0.0` | Cross-axis terrain slope for single-axis tracking |
| `dual_axis_max_tilt` | `90.0` | Maximum panel tilt for dual-axis tracking |
| `transposition_model` | `"isotropic"` | Sky-diffusion model used to project GHI/DHI/DNI onto the plane of array (see [below](#sky-diffusion-transposition-model)) |
| `albedo` | `None` | Ground reflectance (0-1) for the ground-diffuse component; `None` uses pvlib's 0.25 default. Mutually exclusive with `surface_type` |
| `surface_type` | `None` | Named ground cover (e.g. `"snow"`, `"sea"`, `"grass"`) mapped to an albedo; an alternative to `albedo` |
| `model_perez` | `"allsitescomposite1990"` | Perez coefficient set; only used when `transposition_model = "perez"` |
| `solar_position` | `"interval-start"` | Where within each timestep the sun position is evaluated. `"mid-interval"` matches the PVWatts/SAM convention for interval-averaged weather (hourly value labelled 07:00 = 07:00–08:00 average → 07:30 sun) |
| `horizon_profile` | `None` | Optional `[[azimuth_deg, elevation_deg], ...]` far-horizon profile. Points are circularly interpolated; direct beam is removed while the sun is on or below the terrain line. Requires weather explicitly marked as unshaded |
| `iam_model` | `"ashrae"` | Beam incidence-angle modifier. `"physical"` uses pvlib's physical optics model and `"martin_ruiz"` its empirical model; the Ashrae default preserves historical results |
| `diffuse_iam` | `"none"` | Whether the incidence-angle modifier is also applied to the diffuse POA components. `"marion"` weighs sky- and ground-diffuse with the view-factor-integrated selected IAM model (Marion 2017); the default applies IAM to beam only, a known ~0.5–1% overestimate |
| `temperature_model` | `"faiman"` | Cell-temperature model / mounting preset. `"pvsyst-*"` and `"sapm-*"` expose documented mounting/construction coefficients; `"noct-sam"` requires sourced module NOCT and efficiency metadata (not yet available for bundled modules). The default Faiman open-rack result is unchanged |
| `bifacial_model` | `"none"` | Rear-irradiance model. `"none"` preserves front-only production; `"infinite_sheds"` requires sourced module bifaciality plus `gcr`, `pvrow_height`, and `pvrow_pitch` |
| `pvrow_height` | `None` | Height of the PV row center above ground; required by `"infinite_sheds"` and expressed in the same unit as `pvrow_pitch` |
| `pvrow_pitch` | `None` | Distance between adjacent PV rows; required by `"infinite_sheds"` and expressed in the same unit as `pvrow_height` |
| `resolution` | `"h"` | Time resolution (`"h"` or `"15min"`) |
| `projection_years` | `20` | Economic projection horizon |
| `cost_preset` | `None` | Cost preset key from packaged defaults |
| `costs` | *unset* | Optional cost overrides layered over the selected preset and built-in defaults; see [below](#cost-and-emissions-presets) |
| `inflation_rate` | `0.02` | Annual electricity price inflation |
| `sell_price_inflation` | `0.0` | Annual inflation of the grid export (sell) price |
| `discount_rate` | `0.03` | Discount rate for NPV |
| `emissions_country` | `None` | Country code for CO2 calculations (`"PT"`, `"DE"`, `"ES"`, ...) |
| `export_emissions_factor_gco2_kwh` | `None` | Optional displacement factor for exported PV. `None` uses the preset's avoided-grid factor and reports that fallback explicitly |
| `pv_degradation_rate` | `0.005` | Annual PV degradation rate (0.5% / year) |
| `calendar_model` | `"naumann_lam_field_calibrated"` | Battery calendar aging model. Default is the v1 field calibration; use `"naumann_lam_field_calibrated_v2"` for the v2 field-calibrated fit with Lam `Ea`/`n` fixed and `k0`/`b` fitted |
| `degradation_engine` | `"native"` | `"native"` keeps Naumann/Lam; `"blast"` explicitly opts into a vendored BLAST cell model |
| `blast_model` | `None` | Stable BLAST model key; required with `degradation_engine="blast"` and invalid with the native engine |
| `battery_min_soc` | `0.10` | Battery SOC floor (fraction of nominal, SOH-derated capacity) |
| `battery_max_soc` | `0.90` | Battery SOC ceiling (same basis as `battery_min_soc`) |
| `battery_eol_percentage` | `0.70` | SOH fraction that triggers battery replacement |
| `battery_rte` | `None` | Battery round-trip efficiency (`None` = 0.95), split evenly across charge/discharge |
| `battery_max_charge_power_w` | `None` | Maximum DC power entering the battery charge path; `None` is unlimited |
| `battery_max_discharge_power_w` | `None` | Maximum battery AC power delivered to load; `None` is unlimited |
| `battery_temperature` | `"weather"` | Battery temperature used for degradation: `"weather"`, a fixed temperature in °C, or a timestamped CSV path |
| `battery_indoor_model` | `None` | Optional indoor-temperature model settings. `None` applies the default indoor buffering; use `{"enabled": false}` to use `battery_temperature` without remapping |
| `dc_coupled` | `True` | DC-coupled / hybrid inverter. `False` is currently unsupported and raises |
| `inverter_efficiency` | `0.96` | Nominal inverter efficiency used by the PVWatts part-load curve |
| `inverter_loading_ratio` | `1.25` | DC/AC oversizing ratio; also sets the inverter AC rating that clips production |
| `pv_loss_overrides` | `None` | Per-component overrides (percent) for the fixed PVWatts system losses, e.g. `{"shading": 0.0}` |
| `start_date` | `"2023-01-01"` | First simulation date |

Real calendar-year load profiles follow `start_date`: leap years contain
8,784 hourly (35,136 quarter-hourly) intervals and preserve exact annual
energy. Conventional 8,760-hour TMY weather remains a separate weather-data
convention and is not blindly expanded to 8,784 rows.

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
| Solar position | `interval-start` | `mid-interval` for interval-averaged weather |
| Beam IAM | `ashrae` | `physical` |
| Diffuse IAM | `none` | `marion` |
| Cell temperature | `faiman` open rack | A mount-appropriate PVsyst or SAPM preset |

These are explicit modeling assumptions, not universally correct replacements.
Match the timestamp convention to the weather source and the temperature preset
to the physical construction. The example uses a close rooftop mount and a
catalog module with sourced efficiency; a free-standing array should select a
free-standing/open-rack thermal model instead.

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

The accepted keys follow the packaged cost-catalogue names:
`electricity_cost`, `electricity_sold_cost`, `daily_power_cost`,
`module_cost_per_w`, `storage_cost_per_kwh`,
`inverter_cost_per_kw_hybrid`, `inverter_cost_per_kw_simple`,
`installation_cost_per_module`, `installation_cost_battery`,
`other_cost_per_module`, `other_costs`, `land_cost`,
`maintenance_cost_per_panel`, `maintenance_cost`, and `operation_cost`.
Unknown keys and negative or non-finite values are rejected before simulation.

For full control, build a {py:class}`~breos.CostParams` and
{py:class}`~breos.EmissionsParams` yourself and call the lower-level
functions documented in the [Cost and emissions API](../api/cost-analysis.md).

## Load profiles

The public package default is `load_profile = "1"`, a demandlib-derived H0
example bundled with BREOS. `load_profile = "demandlib_h0"` is the same
profile under a readable alias and is preferred in examples. Other standard
profile keys remain supported when you provide the required CSV files yourself
through `rlp_directory`:

```python
breos.App({
    "location": "porto",
    "n_modules": 10,
    "annual_consumption_kwh": 4000,
    "load_profile": "6",
    "rlp_directory": "/path/to/licensed/rlp/files",
    "resolution": "15min",
})
```

Use external BDEW, E-REDES, REE, or custom profiles only under terms that
permit your intended use. See [Load Profile Data](../legal/load-profile-data.md)
for the expected filenames and the reason these CSVs are not bundled.
