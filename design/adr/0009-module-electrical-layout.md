# 0009 — Module electrical layout

- **Status:** Accepted 2026-10-13. M1, M2, M3 and M6 are implemented.
- **Date:** 2026-10-13

## Context

Module-level shading (ADR 0008 S6) builds each bypass-diode substring from
the module's cells. How a module answers shade depends on two properties that
combine freely:

- **Layout**: how the cells are wired. A full-cell module is one chain of 60
  or 72 cells in three substrings. A half-cut module has 120 or 144
  half-cells: each bypass diode spans two half-chains in parallel, top and
  bottom. Shingled modules overlap cell strips into parallel strip-strings,
  and thin-film modules are one monolithic chain of narrow cells.
- **Reverse behaviour**: the voltage at which a shaded cell breaks down, and
  what it leaks before. The bypass diode holds a fully shaded cell at about
  -12 to -15 V in a 20-cell substring. Where a cell breaks down beyond that
  and leaks little before, as TOPCon and heterojunction cells typically do,
  the no-leakage curve of ADR 0008 S6 is a close approximation; PERC cells,
  which break down from about -10 V, can leak before the diode takes over
  and lose a little less than it computes. Some
  back-contact cells are designed to break down at a few volts
  (Calcabrini et al., 2022), so a shaded cell passes the current itself and
  the module loses that cell's generation and what it dissipates, not the
  whole substring. Datasheets do not state the breakdown voltage, and PAN
  files carry a reverse-curve coefficient, not a breakdown voltage.

A half-cut module can carry front-contact or back-contact cells, so a single
technology label would need every combination.

ADR 0008 S6 models a half-cut module as 144 half-voltage cells in series,
with the mean of the two halves' worst cells as the limiting fraction. That
reproduces the current limit but ties each top half-cell to its bottom
partner. Against a cell-level model with explicit halves, shade on the bottom
rows in portrait came out 12 to 15 % high in module power with every
substring shaded, 6 to 7 % with one; landscape was exact.

`bypass_diodes` sat on `pv_strings[i].module_shading`, so one module could
be given different diode counts in different strings.

## Decisions

### M1. Physical layout fields on the module record

`PVModuleParams` describes the layout with physical fields, not a
technology label: `N_Cells`, the total count a datasheet prints;
`bypass_diodes`, 3 by default; and, with M2, `cell_strings_in_parallel`, 1 by
default and 2 for half-cut. Cells in series per substring are
`N_Cells / (cell_strings_in_parallel * bypass_diodes)`, a whole number: a
record where it is not is refused. `celltype` stays the
cell material that seeds the single-diode fit; temperature coefficients come
from the datasheet, so the technology's thermal behaviour enters through
them. A caller's equipment data maps its own labels onto these fields.

`bypass_diodes` moves to the module record and is removed from
`module_shading`. A string's shading tables are bound to its array's module
when the configuration resolves: a module given by one table shares it among
its substrings, and one given by substrings must hold one table per bypass
diode of the module.

### M2. Explicit half-cut halves

With `cell_strings_in_parallel = 2`, each diode section is two half-chains
in parallel that share the section's voltage. A substring's shading may give
a `halves` list, one entry per parallel chain in the module record's order,
each with its own `beam_shaded_fraction` and `limiting_shaded_fraction`; the
order carries no top or bottom meaning, since portrait and landscape swap
them. A substring without `halves` is shaded alike in both, so its
limiting fraction is each chain's worst cell. The ADR 0008 S6 convention is
removed.

A substring whose chains are shaded alike keeps ADR 0008 S6's closed form
with `N_Cells / cell_strings_in_parallel` cell positions, each the chains'
cells side by side. One whose chains differ is solved as the chains' summed
current at each voltage: each chain is its parts in series, its shade
concentrated as M6 states, capped at its smallest part's short-circuit current,
and the bypass diode clamps the substring
(`breos.pv.mismatch._HalfSection`). Above its open circuit a chain takes
current back from the others, up to the brightest one's. Against a
cell-level model of a 144 half-cell module with explicit parallel chains
(tests/cell_reference.py), portrait shade on any number of bottom rows came
within 0.05 % of module power, and a ten-module string with three such
modules within 0.1 % (M6).

### M3. An optional breakdown voltage

`breakdown_voltage_v`, optional and negative, is the cell's breakdown
voltage. A shaded group of k cells then clamps at k times it, where that is
reached before the bypass diode's clamp. Unstated, the no-leakage curve
applies, which overstates the loss of modules with low-breakdown cells; the
documentation says so.

The breakdown is hard: past its short-circuit current a part's voltage is
its cells' breakdown voltage (`breos.pv.mismatch._part_voltage`), and a
half-cut chain's table runs up to its largest part's short-circuit current.
Against PVMismatch 4.1's avalanche model (Bishop) with no leakage before it,
on ten 72-cell modules with one cell at 0.05 to 0.5 suns in 1, 3 or 30
substrings, the hard breakdown came within 0.3 % of string power from -2 to
-5.5 V; at -15 V, where breakdown and the bypass clamp nearly meet, the
avalanche model's earlier onset made it up to 0.8 percentage points of
string power lower. Against the cell-level model of M2 with the same hard
breakdown (tests/test_cell_breakdown.py), one dark cell per chain came
within 0.05 %.

Row shade whose edge falls between cell rows darkens several cells of a
chain, and they break down together (M6): one to four portrait rows at -3 V come within 0.05 %. Soft
shade around a dark cell does not; see M6.

### M4. Not represented

Shingled and thin-film layouts need geometry inputs this contract does not
carry: which strips a shadow crosses. They are documented as not
represented. One microinverter that tracks each module is ADR 0007's one
inverter with one-module strings (ADR 0008 S7); optimizers and fleets of
microinverters wait for the MLPE record.

### M5. What stays

The tracker takes the global maximum within its window (ADR 0008 S6),
stated as an assumption. No runtime warning names a technology, since the
fields cannot tell a back-contact module from another. A module-shaded
string keeps `parallel = 1`.

### M6. A chain's shade is concentrated

The shading input gives each chain two numbers, its area mean and its
most-shaded cell's fraction. ADR 0008 S6 first read them as the most-shaded
cell in series with the other cells at the irradiance that keeps the mean,
which is exact for one dark cell but spreads a shadow that darkens several
over the lit ones. Without a breakdown voltage, eight to eleven portrait
bottom rows came out up to 1.1 % high; with `breakdown_voltage_v = -3`, one
portrait row darkens two cells per chain, which break down as one, and came
out 26 % high.

A chain of n cells is now `floor(n * mean / limiting)` cells at the limiting
fraction, one cell at the remainder, `n * mean` less the dark cells' share,
and the other cells clear (`breos.pv.mismatch._concentrated`). A shadow that
leaves at most one cell of a chain partly shaded, every other cell lit or
dark, is described exactly: every
portrait and landscape case of the cell-level model, with or without
breakdown, comes within 0.05 %. The input format is unchanged.

The trade-off is soft shade, where several cells are partly shaded:
concentrating them into fewer darker cells understates the module. Without
breakdown a penumbra or a gradient over four rows comes within 0.15 %. With
breakdown, soft shade around a dark cell is concentrated into several dark
cells that break down together: three rows shaded by half and one dark cell
come out 15 % low at -3 V, where the first form was within 0.6 %; a dark cell
with two half-shaded neighbours comes out 8.5 % low at -5.5 V, where the
first form was 9.5 % high. A diagonal edge leaves cells like that last
case, and so does an edge that cuts through a row of a half-cut chain, two
cells wide: only shade with at most one partly shaded cell per chain is
exact, not every hard shadow. Two numbers per chain cannot describe both; a
third, such as the count of cells above half shade, would, and waits for a
caller that needs it.

## Consequences

- `module_shading.bypass_diodes` is removed; a configuration that gives it
  is refused as an unknown key. The result's shading summary reports the
  module's `bypass_diodes` and no longer `bypass_diodes_stated`.
- Callers that send three substrings per module for three-diode modules are
  unaffected by M1. M2 adds an optional key, and the bundled modules, all
  half-cut, state `cell_strings_in_parallel = 2`; on them a substring's
  limiting fraction now means each chain's worst cell, no longer the mean of
  the two chains' worst cells.
- M6 changes every module-shaded result where a chain's mean exceeds its
  limiting fraction over its cell count, without a new key.
- A module whose `N_Cells` does not divide into whole chains on every diode
  is refused.
