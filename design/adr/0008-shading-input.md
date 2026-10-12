# 0008 — Shading as an input

- **Status:** Accepted 2026-10-12. Levels 1 and 2 are implemented with the
  shading table; the series form (S3) is deferred.
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

    poa_direct       := poa_direct       * (1 - f)
    poa_circumsolar  := poa_circumsolar  * (1 - f)
    poa_sky_rest     := poa_sky_rest     * v
    poa_ground       := poa_ground

The circumsolar diffuse comes from around the sun's disc, so an object that
shades the beam shades it too. Hay-Davies and the Perez models separate it,
and with them it takes the beam's fraction; the rest of the sky diffuse
(isotropic, and Perez's horizon band) takes the sky view factor. The other
transposition models do not separate it, and their whole sky diffuse takes
the sky view factor.

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

At level 2, the shading is given per module rather than per array.
`pv_strings[i].module_shading` holds one grid of sun positions for the
string (`azimuth_deg`, `elevation_deg`, with S2's rules), an optional
`bypass_diodes` count (3 when omitted, reported as assumed), and `modules`:
one entry per module of the string, in series order. An entry gives either
one `beam_shaded_fraction` table for the whole module, shaded evenly, or
`substrings`, one table per bypass-diode substring in the module's own
order. Bypass-diode substrings run along the module's long axis, so where
they lie on the roof depends on portrait or landscape mounting; the layout
tool maps them, and BREOS assumes no geometry. Each entry may also carry a
`sky_view_factor`.

A substring's `beam_shaded_fraction` is its area mean, and its optional
`limiting_shaded_fraction` the shaded fraction of its most-shaded series
cell. Series cells share one current, so the worst cell sets the
substring's current: one fully shaded cell out of twenty bypasses the
substring, about a third of the module, while the area mean says 5 %. The
limiting fraction defaults to the mean, which is even shade; BREOS uses the
mean where it is given lower. The worst cell alone shades its share of the
substring's area, so the mean must be at least the limiting fraction over
the substring's cell count at every node, and a substring that gives a
limiting fraction needs at least two cells; BREOS refuses tables that break
either. A layout tool that takes both values from the same cell samples
meets the first. In a half-cell module the two halves run in
parallel, so the limiting fraction is the mean of the two halves' worst
cells.

BREOS models each substring as its most-shaded cell at the limiting
irradiance in series with its other cells at the irradiance that keeps the
area mean, with `N_Cells / bypass_diodes` cells per substring. Each part is a
single-diode curve with its share of the module's series resistance, shunt
resistance and diode voltage, at the array's cell temperature. The bypass
diode holds the substring at minus its forward voltage
(`electrical_environment.bypass_diode_voltage_v`, 0.5 V when unstated, the
value pvlib's partial-shading example assumes, reported as assumed), so a
shaded cell in reverse bias never sees more than the rest of its substring;
before that its reverse characteristic is the shunt alone, without
avalanche breakdown. The string's substrings carry one current and add
their voltages, less the cable's drop. The input runs at the global maximum
of its strings' summed curve within the limits ADR 0007 D7 applies: the
maximum can be the lower peak, with a substring bypassed.

Each substring sees the array's unshaded irradiance after the incidence-angle
modifier: the beam and circumsolar part reduced by its shaded fraction, the
other sky diffuse scaled by its module's sky view factor, and the ground and
rear parts unchanged. The array's own irradiance stage is the mean of its
modules', so `near_shading` stays the linear loss of level 1, and a new
`shading_mismatch` stage takes what the uneven shading costs on top of it:
the strings' power at their array's mean irradiance less their power from
each module's own. Each string's own mismatch, measured from the string's
own mean irradiance, is reported so a layout tool can compare stringings.

Rules: module shading needs `electrical_fidelity = "string"` and
`pv_arrays`; such a string has `parallel = 1`; every string of an array
carries it or none; an array whose strings carry it takes no `shading`
table; and every array has a table or module-shaded strings, or none does.

Steps where every substring of a string sees the same irradiance use the
module's own curve, exactly as without module shading; the others build the
string's curve on a grid of currents, dense below each part's photocurrent
where the curve has its knees. A lone string on its input is solved on its
exact curve: its best local maxima within the voltage range and current
limit, refined where a limit cuts them. Strings in parallel share the
input's voltage and are solved on their tabled curves, within 0.01 % of a
brute-force solve in the tests. Each search refines the best few local
maxima of its table (four on a string's curve, six on an input's summed
curve); a curve with more peaks than that, or a peak narrower than the
table's spacing, can still be missed. The open-circuit voltage that starts
the inverter is where the strings' signed currents cancel, the table
reaching into reverse current down to 1.5 times the module's Isc for it.
A substring holds at least one cell: `bypass_diodes` above `N_Cells` is
refused.

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

## Decisions of the maintainer (2026-10-12)

1. The table first. The series (S3) stays deferred, and more accurate
   modelling remains a goal (see "Accuracy" below).
2. pvlib's convention for the grid.
3. Every array or none.
4. Level 2 per module, with substrings optional.
5. Hourly first; 15-minute runs are wanted too, after the performance of
   both resolutions is measured at each level. The table serves both, so
   question 5, which concerns the series only, waits with it.

## Accuracy

For a fixed scene, the beam shaded fraction is a function of the sun's
direction alone, so tracing the sun through the year gives the table's own
values at the positions the sun visits. The table loses nothing but the
interpolation between its nodes, and a layout tool controls that with its
grid step. Measured against an exact shade function at Porto (2026-10-12),
a long wall's smoothly growing shade came within 0.04 percentage points of
the annual beam loss on a 5-degree grid and 0.01 on a 2-degree grid; a
10-degree object that switches the beam fully on and off, sampled as single
points, was 0.2 points high on a 1-degree grid and more than doubled on a
5-degree one. Nodes that hold the shaded share of the array's area, not a
single ray, keep the error small.

What a table cannot hold, in the order of the gain expected from adding it:

1. The electrical effect of partial shading (level 2). One shaded cell
   string costs a whole bypass-diode substring, far more than its share of
   the irradiance, so this is the largest gap for roofs with near objects.
2. Diffuse shading that depends on direction. The sky view factor is one
   number for the whole dome; the circumsolar part already follows the beam.
   A table of sky view by sky region would refine the rest.
3. Shading that changes with time: deciduous trees in leaf or bare,
   construction. A table per season, or the series, carries it.
4. Interpolation error at sharp edges, which a finer grid removes.

## Questions for the maintainer (answered above)

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

## Performance

Level 1 reads the table once per step and array. On a one-year single-array
run it added 3 to 5 ms to about 40 ms hourly, and 5 to 9 ms to about 140 ms
at 15 minutes, for a 180 by 46 table (2026-10-12).

Level 2 builds a curve for each string at each step where its substrings see
different light. For one string of ten 144-half-cell modules at Porto, on an
83 by 26 grid with a chimney's shadow crossing the modules in turn, a
one-year run of the PV stage took (2026-10-12, one core):

| Shading | Hourly | 15 minutes |
|---|---|---|
| None, string-level energy | 0.04 s | 0.13 s |
| Level 1 table | 0.04 s | 0.14 s |
| Level 2, whole modules | 0.20 s | 0.56 s |
| Level 2, substrings with limiting cells | 0.84 s | 2.14 s |

The two resolutions gave the same mismatch loss within 0.2 %, 95 kWh with
whole modules and 142 kWh with limiting cells, so hourly runs are enough to
size the loss and 15-minute runs add little for it. The cost grows with the
number of distinct substrings and of steps where they differ; steps where a
substring is lit like the string's brightest reuse that curve.

## Consequences

- Aggregate runs gain hourly near shading per array without any change to the
  electrical model, and Monte Carlo keeps working through the table.
- Results gain a `near_shading` waterfall stage and, at level 2, a
  `shading_mismatch` stage with each string's own mismatch. Where strings
  in parallel on one input are shaded differently, the voltage they then
  share costs the `parallel_mismatch` stage, as for parallel arrays (ADR 0007
  D7), so a layout tool can see what a second input would save before the
  inverter's limits; a binding power cap can leave less to recover.
- The flat 3 % shading default no longer double-counts when a shading input is
  given.
- Level 2 is the expensive part: a curve per string and step from its
  substrings. It runs only where `module_shading` is given, and only for
  steps where the string's substrings see different irradiance.
