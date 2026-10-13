# Choose a sky model and a terrain horizon

## Anisotropic sky-diffusion model

The default `isotropic` transposition underestimates plane-of-array
irradiance on clear days. Switch to an anisotropic model — here Perez — to
capture circumsolar and horizon brightening. No extra weather inputs are
needed; see [Sky-diffusion model](../getting-started/configuration.md#sky-diffusion-transposition-model):

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

## Apply near shading from a layout tool

Objects near the array, such as chimneys, dormers, neighbouring roofs and
trees, shade it at some sun positions and not at others. A layout tool that
holds the 3D scene computes, for each array, the fraction of the direct beam
they shade at each sun position, and BREOS applies it step by step:

```toml
[[pv_arrays]]
modules = 12
tilt = 30
azimuth = 180

[pv_arrays.shading]
sky_view_factor = 0.93
azimuth_deg = [0, 60, 90, 120, 150, 180, 210, 240, 270, 300]
elevation_deg = [0, 10, 20, 30, 45, 60, 90]
beam_shaded_fraction = [
  # one row per elevation, one value per azimuth
  [1.0, 1.0, 1.0, 0.6, 0.3, 0.2, 0.3, 0.6, 1.0, 1.0],
  [0.4, 0.4, 0.8, 0.4, 0.1, 0.0, 0.1, 0.4, 0.8, 0.4],
  [0.0, 0.0, 0.3, 0.1, 0.0, 0.0, 0.0, 0.1, 0.3, 0.0],
  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
]
```

Azimuth is in degrees clockwise from north and elevation above the horizon,
as for `horizon_profile`. Both axes take any increasing values, so a table
can be fine where the shade edge moves and coarse where it does not. BREOS
reads the table bilinearly. When the gap from the last azimuth round through
north to the first is no wider than the widest step, as above, the axis is
periodic and the last column interpolates to the first; otherwise the
nearest edge holds outside the grid, as it does above and below the
elevation axis. An axis that runs from 0 to 360 must give the same column at
both ends.

The beam shaded fraction reduces the array's direct plane-of-array
irradiance, and `sky_view_factor` (1 when omitted) the sky diffuse, before
the incidence-angle modifier. With `transposition_model = "haydavies"`,
`"perez"` or `"perez-driesse"` the circumsolar part of the sky diffuse comes
from around the sun's disc, so it takes the beam's shaded fraction instead.
The ground-reflected and rear-side irradiance are not shaded. The table
depends on the geometry only, so one table serves hourly and 15-minute runs
and every Monte Carlo weather year.

Give every array a table, or none: an unshaded array takes an all-zero
table, such as `{azimuth_deg = [0], elevation_deg = [0],
beam_shaded_fraction = [[0.0]]}`. With the tables, the flat PVWatts
`shading` loss defaults to 0; a `pv_loss_overrides.shading` you state stays
and applies on top. The loss appears as the `near_shading` stage of
`pv_loss_waterfall`, and `pv_loss_waterfall.near_shading` reports each
array's table. A table given with `horizon_profile` should hold near shading
only, since the horizon has already removed the beam behind the terrain.

How fine a grid needs to be depends on the shade. In a test against an exact
shade function at Porto, a long wall's shade, which grows smoothly over the
array, came within 0.04 percentage points of the annual beam loss on a
5-degree grid. A small object whose shade switches fully on and off at
single sun positions, 10 degrees wide, was the worst case: its 1 % annual
loss came out 0.2 percentage points high on a 1-degree grid and more than
doubled on a 5-degree one. Steps of 1 or 2 degrees around small objects, and
values that are the shaded share of the array's area rather than a single
point, keep the table close.

## Model shading module by module

A shadow across one cell costs more than its share of the light: the cells of
a bypass-diode substring carry one current, so the most-shaded cell sets it,
and the string's other modules share the string's current too. With
string-level energy (`electrical_fidelity = "string"`), a string can carry
each module's own shading instead of its array's table:

```toml
[[pv_strings]]
inverter = 0
mppt = 0
array = 0
modules = 3

[pv_strings.module_shading]
azimuth_deg = [0, 90, 180, 270]
elevation_deg = [0, 30, 45]

# Module 1: a chimney's shadow crosses its first substring in the morning.
[[pv_strings.module_shading.modules]]
[[pv_strings.module_shading.modules.substrings]]
beam_shaded_fraction = [[0, 0.1, 0, 0], [0, 0.05, 0, 0], [0, 0, 0, 0]]
limiting_shaded_fraction = [[0, 1, 0, 0], [0, 0.6, 0, 0], [0, 0, 0, 0]]
[[pv_strings.module_shading.modules.substrings]]
beam_shaded_fraction = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
[[pv_strings.module_shading.modules.substrings]]
beam_shaded_fraction = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

# Modules 2 and 3: unshaded.
[[pv_strings.module_shading.modules]]
beam_shaded_fraction = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
[[pv_strings.module_shading.modules]]
beam_shaded_fraction = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
```

One grid of sun positions serves the string, with the rules of the array
table above. Each module entry is either one `beam_shaded_fraction` table,
for shade spread evenly over the module, or `substrings`, one entry per
bypass diode (`bypass_diodes`, 3 when omitted). A substring's
`beam_shaded_fraction` is the shaded share of its area, and its optional
`limiting_shaded_fraction` the shaded share of its most-shaded cell; in a
half-cell module, give the mean of the two halves' most-shaded cells; for
shade on one half only, this overstates the module's power (see
[Module layouts and technologies](../modeling/pv-modules.md#module-layouts-and-technologies)). The
most-shaded cell alone shades its share of the substring, so the mean must
be at least the limiting fraction over the substring's cells (48 for a
144-cell module with three diodes); taking both from the same cell samples
keeps them consistent, and BREOS refuses tables that are not. Each entry may
carry its own `sky_view_factor`.

BREOS builds each substring from its most-shaded cell and its other cells,
lets the bypass diode take the current a shaded substring cannot carry
(`electrical_environment.bypass_diode_voltage_v`, 0.5 V when unstated), and
runs each input at the global maximum of its strings' curve. The array's
irradiance stage, `near_shading`, is the mean of its modules, and the rest of
the cost is the `shading_mismatch` stage. `pv_loss_waterfall.shading_mismatch`
reports it, with each string's own loss measured from that string's mean
irradiance, so stringings can be compared: a string whose modules are shaded
alike loses little to mismatch. Strings shaded differently but on one input also
share its voltage; the `parallel_mismatch` stage reports what that costs, which,
limits aside, is what a separate input for each would save. Where an input's
or the inverter's power cap binds anyway, a separate input may save less.

Every string of an array carries `module_shading` or none does, such a string
has `parallel = 1`, and an array whose strings carry it takes no `shading`
table of its own.
