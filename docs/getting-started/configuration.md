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
| `gcr` | `0.35` | Ground coverage ratio for single-axis tracking |
| `cross_axis_tilt` | `0.0` | Cross-axis terrain slope for single-axis tracking |
| `dual_axis_max_tilt` | `90.0` | Maximum panel tilt for dual-axis tracking |
| `transposition_model` | `"isotropic"` | Sky-diffusion model used to project GHI/DHI/DNI onto the plane of array (see [below](#sky-diffusion-transposition-model)) |
| `albedo` | `None` | Ground reflectance (0-1) for the ground-diffuse component; `None` uses pvlib's 0.25 default. Mutually exclusive with `surface_type` |
| `surface_type` | `None` | Named ground cover (e.g. `"snow"`, `"sea"`, `"grass"`) mapped to an albedo; an alternative to `albedo` |
| `model_perez` | `"allsitescomposite1990"` | Perez coefficient set; only used when `transposition_model = "perez"` |
| `resolution` | `"h"` | Time resolution (`"h"` or `"15min"`) |
| `projection_years` | `20` | Economic projection horizon |
| `cost_preset` | `None` | Cost preset key from packaged defaults |
| `inflation_rate` | `0.02` | Annual electricity price inflation |
| `discount_rate` | `0.03` | Discount rate for NPV |
| `emissions_country` | `None` | Country code for CO2 calculations (`"PT"`, `"DE"`, `"ES"`, ...) |
| `pv_degradation_rate` | `0.005` | Annual PV degradation rate (0.5% / year) |
| `calendar_model` | `"naumann_lam_field_calibrated"` | Battery calendar aging model. Default is the v1 field calibration; use `"naumann_lam_field_calibrated_v2"` for the v2 field-calibrated fit with Lam `Ea`/`n` fixed and `k0`/`b` fitted |
| `battery_min_soc` | `0.10` | Battery SOC floor (fraction of nominal, SOH-derated capacity) |
| `battery_max_soc` | `0.90` | Battery SOC ceiling (same basis as `battery_min_soc`) |
| `battery_eol_percentage` | `0.70` | SOH fraction that triggers battery replacement |
| `battery_rte` | `None` | Battery round-trip efficiency (`None` = 0.95), split evenly across charge/discharge |
| `dc_coupled` | `True` | DC-coupled / hybrid inverter (vs AC-coupled) |
| `inverter_efficiency` | `0.96` | Inverter efficiency |
| `inverter_loading_ratio` | `1.25` | DC/AC oversizing ratio; also sets the inverter AC rating that clips production |
| `pv_loss_overrides` | `None` | Per-component overrides (percent) for the fixed PVWatts system losses, e.g. `{"shading": 0.0}` |
| `start_date` | `"2023-01-01"` | First simulation date |

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

## Battery degradation calibration

`calendar_model = "naumann_lam_field_calibrated"` is the stable default and
maps to the v1 field calibration. The explicit
`"naumann_lam_field_calibrated_v1"` alias is equivalent. Use
`"naumann_lam_field_calibrated_v2"` for the v2 field-calibrated fit with Lam
`Ea`/`n` fixed and `k0`/`b` fitted to field data.

## Discovering available options

Use the CLI to list packaged option keys:

```bash
breos list locations
breos list modules
breos list cost-presets
breos list emissions
breos list load-profiles
```

Add `--json` to any `breos list` command for machine-readable output.

Before running a full simulation, validate and inspect a config:

```bash
breos validate-config quickstart.toml
breos run --config quickstart.toml --dry-run
```

These commands resolve packaged presets, modules, inverter sizing, battery
settings, load-profile choices, and emissions settings without fetching weather
or simulating.

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
Pass the key:

```python
breos.App({
    "location": "porto",
    "n_modules": 10,
    "annual_consumption_kwh": 4000,
    "cost_preset": "residential_pt",
    "emissions_country": "PT",
})
```

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
