# 0008 — Shading as an input

- **Status:** Proposed. Not implemented; the maintainer approves the
  decisions before any code.
- **Date:** 2026-10-12

## Context

BREOS models shading in two ways today. `horizon_profile` is far shading: a
site-wide terrain horizon, as `[azimuth, elevation]` pairs, that blocks the
beam when the sun is below it. Everything nearer, such as chimneys,
neighbouring roofs, trees and other rows, falls into the flat PVWatts
`shading` loss, 3 % by default, the same in every hour and for every array.
Professional tools compute near shading from a 3D scene and apply it hour by
hour, and their users expect the same of BREOS.

BREOS computes no geometry (2026-10-12). Layout tools hold the scene and the
module positions, so they compute the shading. BREOS applies it physically:
to the irradiance each array receives, and, at string level, to the IV curves
of partly shaded strings. This record fixes that contract. It also leaves
room for module-level power electronics (MLPE), which change the electrical
effect of shading and are planned after it.

A layout tool's shading output varies. Some produce an hourly series for one
year; some produce a factor per sun position; some only an annual figure per
module. A typical engine samples points on each module, computes the beam
shaded fraction at every step of a typical year, and deduplicates the
geometry onto sun-position bins, a few degrees wide, because the shading
depends on the sun's position and not on the weather. Few compute a sky-view
factor. An annual figure carries no time structure and is already served by
`pv_loss_overrides.shading`. The contract below asks for the time structure
and accepts the two common forms of it.

## Decisions

### S1. Two levels, introduced in order

1. **Array level (irradiance).** Per array, the fraction of the beam
   irradiance that is shaded, and the fraction of the sky dome the array
   sees. These reduce the array's plane-of-array beam and sky-diffuse
   components before the incidence-angle modifier. Energy stays linear in
   irradiance, as PVsyst's "linear shading" is.
2. **Module level (electrical).** Per module, the fraction of its beam that is
   shaded, and optionally per bypass-diode substring. With string-level
   energy, a string's IV curve is the series sum of its modules' curves with
   bypass diodes, so one shaded module costs more than its share of the
   irradiance. This is PVsyst's "electrical effect" and needs
   `electrical_fidelity = "string"`.

Level 1 comes first, because it serves aggregate runs and needs nothing below
the array. Level 2 follows, built on ADR 0007's string model.

### S2. The shading table: beam shading against sun position

The primary form is a **shading table**, as PVsyst and many layout tools
exchange it: the shaded fraction of the beam, from 0 (fully lit) to 1 (fully
shaded), on a grid of sun azimuth and sun elevation. BREOS interpolates it at
each step's sun position, as `horizon_profile` already reads the sun.

```toml
[[pv_arrays]]
modules = 12
tilt = 30
azimuth = 180

[pv_arrays.shading]
sky_view_factor = 0.93
azimuth_deg = [60, 90, 120, 150, 180, 210, 240, 270, 300]
elevation_deg = [5, 10, 20, 30, 45, 60, 90]
beam_shaded_fraction = [
    # one row per elevation, one value per azimuth
    [1.0, 0.8, 0.4, 0.1, 0.0, 0.1, 0.4, 0.8, 1.0],
    ...
]
```

The table does not depend on the weather year or the resolution, so one table
serves App, Monte Carlo over many weather years, and 15-minute and hourly runs
alike. Both axes take any increasing values: shade edges are sharp, so a layout
tool may send steps of a few degrees where the shade changes and wider ones
where it does not. BREOS interpolates bilinearly in azimuth and elevation.
When the azimuth axis spans the full circle, it is periodic and the last
column interpolates to the first across north; otherwise, and on the
elevation axis, the nearest edge holds outside the grid. A full circle matters
at higher latitudes, where the summer sun rises north of east. The sun below
the horizon carries no beam.

`sky_view_factor` is the share of the isotropic sky diffuse the array still
sees, from 0 to 1. It is optional and 1 when omitted, since many layout tools
do not compute it; the result then reports that diffuse shading was not
given. It scales the sky-diffuse component at every
step. The ground-reflected component is not shaded; a layout tool that knows
better can lower `albedo` for the array.

### S3. The time-series form, deferred until a tool needs it

This form is recorded so a later tool can use it without a new contract; it
is not built with the table unless the maintainer asks for it (question 1).

A layout tool that has already simulated the year may give a **shading
series** instead: one column per array of beam-shaded fractions, on the
weather's timestamps, in a CSV or Parquet file named by
`shading_file`. Each column is named for its array's position (`array_0`,
`array_1`, …). A series is a function of the sun's position, so it is aligned to the run by
the time of year, as a load profile is: each value is restamped from its own
year to the simulated year by month, day and time of day, in UTC. It must
cover every simulated step after that, or the run refuses it. A series at the
run's resolution applies step for step; an hourly series in a 15-minute run
applies its hour's value to the hour's four steps, an approximation the result
reports. A series computed for one weather file still applies to another, as
long as the site and the time stamps' convention (interval start or middle)
match; the result records the series' file and hash. Monte Carlo samples
other years with other sun positions at the same clock time only through the
calendar, which the restamp handles, so it accepts a series too; a table
remains the exact form.

An array takes a table or a series column, not both.

### S4. Where shading applies

At array level, the step's beam shaded fraction `f` and the sky view factor
`v` act on the plane-of-array components before the incidence-angle
modifier:

    poa_direct  := poa_direct  * (1 - f)
    poa_sky     := poa_sky     * v
    poa_ground  := poa_ground

`horizon_profile` still applies first: a sun behind the terrain is not
shaded again. A table or series given with `horizon_profile` must therefore
hold near shading only; one that also includes the terrain counts it twice. A bifacial array's rear irradiance is not shaded by this input.
The loss appears as a new `near_shading` stage of the PV loss waterfall,
between transposition and the incidence-angle modifier.

### S5. The flat `shading` loss steps aside

When any array has a shading input, the PVWatts `shading` component defaults
to 0 for the system, as the flat `wiring` loss does with cable data (ADR 0007
D8). A `pv_loss_overrides.shading` the caller states stays, and applies on top
of the input, for whatever the input does not cover. Both then appear in the
result, the input as the `near_shading` stage and the flat loss in the PVWatts
block, so a caller that sends its annual shading loss and its shading series
together sees the double count rather than hiding it. Every array must then carry a shading input, or none,
so that an unshaded array is stated as such (an all-zero table), not assumed.

### S6. Module level: a shaded fraction per module

At level 2, the shading table or series is given per module rather than per
array: `pv_strings[i].module_shading` lists, for each module of each string
in series order, a beam shading table or series column of its own, with an
optional `substrings` count (3 for a typical 60- or 120-half-cell module) and
one table per substring, in the module's own substring order. Bypass-diode
substrings run along the module's long axis, so where they lie on the roof
depends on portrait or landscape mounting; the layout tool maps them, and
BREOS assumes no geometry. BREOS then builds each module's IV curve from its own
irradiance, adds the bypass-diode branch per substring (pvlib's `bishop88`
with reverse-bias breakdown), sums the curves in series at equal current and
in parallel at equal voltage, and tracks the input's maximum power point
within the limits ADR 0007 D7 already applies. A string without
`module_shading` takes its array's level-1 shading evenly on every module.

The per-module form is what the electrical effect needs, and it is also what
MLPE needs (S7). A layout tool that knows each panel's shading, as the
module-level shading engines in layout tools do, gives it directly.

### S7. Room for MLPE

A later record will add optimizers and microinverters. With
per-module shading in hand, an optimizer string's energy is the sum of each
module's own maximum power point less the optimizer's efficiency, with no
series mismatch; a microinverter's is each module's through its own AC
conversion. S6's per-module inputs are therefore the inputs MLPE needs, and
this record's contract does not change when MLPE arrives.

## What this record does not decide

- How a layout tool computes shading. BREOS does not trace rays.
- Shading derates applied to the module DC as a whole. A layout tool that cuts
  beam and diffuse alike with its beam fraction overstates the loss; under
  this contract the beam fraction cuts the beam only, and diffuse shading is
  the sky view factor's.
- Diffuse shading that varies with the sun, such as circumsolar shading. The
  sky view factor is isotropic; a refinement can make it a table later.
- Snow and soiling, which vary in time but are not shading. They stay on the
  deferred list.

## Questions for the maintainer

1. **Table first, series second, or both at once?** The table serves Monte
   Carlo and both resolutions; the series is what a tool that simulates the
   year already has. Recommended: the table only, with the series (S3)
   deferred until a layout tool asks for it. The layout tools consulted
   compute on sun positions or clear-sky geometry and have no weather
   timeline to fill a series with.
2. **Grid convention.** PVsyst writes its table in azimuth from south and
   elevation; pvlib uses azimuth from north. Recommended: pvlib's convention,
   as `horizon_profile` already does, and an importer for PVsyst's table later.
3. **Every array or none (S5).** Recommended, as with DC cables, so no array is
   silently unshaded.
4. **Level 2 scope.** Per module, with optional substrings, as S6 sets out;
   or per string only (the fraction of each string's modules shaded), which is
   lighter but cannot serve MLPE. Recommended: per module, with substrings
   optional, since layout tools rarely sample along the bypass-diode
   substrings today.
5. **Hourly series in 15-minute runs (S3).** Repeat the hour's value, as
   proposed, or refuse and ask for a table or a 15-minute series.

## Consequences

- Aggregate runs gain hourly near shading per array without any change to the
  electrical model, and Monte Carlo keeps working through the table.
- Results gain a `near_shading` waterfall stage and, at level 2, the
  electrical mismatch of partial shading within the existing `mppt_limits`
  cost, or in a stage of its own if the maintainer prefers.
- The flat 3 % shading default no longer double-counts when a shading input is
  given.
- Level 2 is the expensive part: an IV curve per module and step for shaded
  strings. It runs only where `module_shading` is given, and only for steps
  where some module is shaded.
