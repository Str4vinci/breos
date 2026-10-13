# How BREOS models PV modules

BREOS has two levels of electrical detail. By default
(`electrical_fidelity = "aggregate"`) the PV system is one block: the modules'
DC power, less flat losses, through one inverter. That level suits studies of
economics and trends. With `electrical_fidelity = "string"`, energy depends on
how the modules are wired: which modules form a string, which strings share an
inverter input, and what the input's voltage and current limits allow. This
page explains the physics that level models, which module technologies it
represents, and where it errs and in which direction. The configuration keys
are in [Configuration](../getting-started/configuration.md#string-level-energy)
and [Model shading module by module](../how-to/sky-model-and-horizon.md#model-shading-module-by-module).

## Cells, substrings and bypass diodes

A crystalline silicon cell is a large diode that light turns into a current
source. Its current is close to proportional to irradiance; its voltage, about
0.6 to 0.7 V open circuit, depends weakly on irradiance and falls as the cell
warms. A module is a chain of cells in series, and cells in series carry one
current, so the chain carries no more than its most-shaded cell can pass.

A shaded cell still generates while the string's current stays below its own
short-circuit current. Above it, the lit cells drive the extra current through
the cell and push it into reverse bias, where it absorbs power instead of
producing it.
Modules therefore split their chain into substrings, usually three, each with
a bypass diode across it. Once a shaded substring's voltage falls below minus
the diode's forward voltage, the diode conducts and the current flows around
the substring: the module loses about a third of its voltage and keeps its
current. A shaded string's power-voltage curve therefore has several peaks, one
for each set of substrings that are bypassed or not.

BREOS builds each substring from its most-shaded cell and its other cells,
each a single-diode curve with its share of the module's parameters, and lets
the bypass diode take the current a shaded substring cannot carry.

## Reverse bias

How much current a shaded cell passes in reverse before its bypass diode takes
over decides how much a small shadow costs. BREOS lets each part of a substring
carry at most its own short-circuit current; beyond it, the bypass diode
conducts. Reverse leakage and avalanche breakdown are not modelled.

Whether that holds depends on the cell's reverse characteristic, which
datasheets rarely state. In a 20-cell substring the diode holds a fully shaded
cell near minus the voltage of the rest of the substring plus the diode's
forward drop, about -12 to -15 V. A cell that breaks down beyond that, as
TOPCon and heterojunction cells typically do, never reaches breakdown, and the
model holds. PERC cells break down at about -10 to -20 V, so some leak before
the diode takes over and lose a little less than BREOS computes. Some
back-contact cells (IBC and its variants) are designed to break down at a few
volts ([Calcabrini et al., 2022](https://doi.org/10.1016/j.xcrp.2022.101155)):
a shaded cell then passes the current itself before its substring's diode is
needed, and the module loses that cell's generation and the power it
dissipates in breakdown rather than the whole substring. For modules with such
cells BREOS overstates the shading loss.

## Module layouts and technologies

A module's response to shade depends on two independent properties: how its
cells are wired, and how its cells behave in reverse bias. A half-cut module
can have front-contact or back-contact cells.

| Module | Wiring | Shading in BREOS |
|---|---|---|
| Full-cell (60 or 72 cells) | One chain, three substrings, three diodes | Represented |
| Half-cut (120 or 144 half-cells) | Two half-chains in parallel on each diode, top and bottom | Approximated: see below |
| Back-contact with low breakdown voltage (some IBC and variants) | Either of the above | Represented as cells without breakdown; losses overstated |
| Shingled | Cell strips overlapped into parallel strip-strings | Not represented |
| Thin film (CdTe, CIGS) | Long narrow cells across the module, few or no diodes | Not represented |

**Half-cut modules.** BREOS takes a half-cut module as a chain of half-cells
in series, and the shading input gives the mean of the two halves' most-shaded
cells. That reproduces the substring's current limit but behaves as if each
top half-cell were wired in parallel with its bottom partner, which real
modules are not. When the shadow covers the bottom half only, as row-to-row
shade does on modules mounted in portrait, the real top half keeps carrying
its current at the section's voltage while the shaded bottom half contributes
little, and the module delivers less than BREOS computes. In a test against a
cell-level model of a 144 half-cell module, with one to six bottom rows
shaded, BREOS overstated module power by 12 to 15 % when every substring was
shaded and by 6 to 7 % when one was. With half the module shaded the
difference was under 1 %, and in landscape, where the shadow crosses both
halves alike, there was none.

The temperature coefficients come from the module's datasheet, so the
technology's temperature behaviour enters through them. `celltype` only seeds
the fit of the single-diode parameters.

## Strings, inputs and the inverter

Modules in series add their voltages and share one current. An inverter input
(MPPT) holds one or more strings in parallel, which share one voltage and add
their currents. That fixes the sizing rules the
[electrical check](../getting-started/interpreting-results.md#electrical-check)
applies: on the coldest morning the strings' open-circuit voltage must stay
below the inverter's maximum DC voltage, in hot weather their maximum power
voltage must stay above the input's MPPT minimum, and their current must stay
within the input's limits.

Each input's tracker runs its strings at the best point of their summed
current-voltage curve inside its voltage window and current limits. Strings in
parallel that see different light, or different shade, each work away from
their own maximum power point at the shared voltage; the
`parallel_mismatch` stage of the
[PV loss waterfall](../getting-started/interpreting-results.md#pv-loss-waterfall)
reports that loss before the input's limits and the inverter's clipping. A
separate input for each string would avoid it, though where a limit or the
clipping binds anyway the energy gained is smaller.

Under shade, the curve has several peaks. Trackers that climb towards the
nearest peak can stop on a lower one; inverters that sweep the whole curve
every few minutes find the highest. BREOS assumes the highest peak inside the
window on every step: hourly and 15-minute steps cannot resolve what a tracker
does between sweeps. Its search refines the best few peaks of each curve, so a
curve with more peaks, or a very narrow one, can still have its highest peak
missed.

With several inverters, each clips and converts its own strings' DC, and
their AC outputs add up; a battery then waits for AC coupling (see
[Several inverters](../getting-started/configuration.md#several-inverters)).
The PVWatts `mismatch` loss still
applies, for differences between modules of one model, which a model of
identical modules does not produce.

## Bifacial modules

With `bifacial_model = "infinite_sheds"`, the rear side's irradiance comes from
a view-factor model of infinite rows. Near shading does not reduce it, and no
loss is applied for rear structure shading or the uneven light across the rear,
so the bifacial gain is an upper estimate.

## Microinverters

A microinverter with one tracker per module is, electrically, a small inverter
whose inputs each hold a one-module string. String-level energy represents one
such microinverter directly: one entry in `inverters`, one MPPT input per
module, and `modules = 1` on each string. With one module per input there is
no series mismatch, so set the PVWatts `mismatch` loss to 0
(`pv_loss_overrides = {"mismatch": 0.0}`). Systems with
several microinverters, and power optimizers, are not represented.
