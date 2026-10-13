# 0009 — Module electrical layout

- **Status:** Accepted 2026-10-13. M1 and M2 are implemented; M3 is next.
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
`N_Cells / (cell_strings_in_parallel * bypass_diodes)`. `celltype` stays the
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
current at each voltage: each chain is its parts in series, its worst cell
and its other cells, capped at its smallest part's short-circuit current,
and the bypass diode clamps the substring
(`breos.pv.mismatch._HalfSection`). Against a cell-level model of a 144
half-cell module with explicit parallel chains (tests/test_half_cut_halves.py),
portrait shade on one to twelve bottom rows came within 0.2 % of module
power, and a ten-module string with three such modules within 0.1 %. A chain
is described by its mean and its worst cell, so a shadow that darkens every
other cell of a chain, as one landscape row along the long edge does, comes
out 0.9 % high.

### M3. An optional breakdown voltage

`breakdown_voltage_v`, optional and negative, is the cell's breakdown
voltage. A shaded group of k cells then clamps at k times it, where that is
reached before the bypass diode's clamp. Unstated, the no-leakage curve
applies, which overstates the loss of modules with low-breakdown cells; the
documentation says so. The implementation is validated against an
independent model with breakdown.

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

## Consequences

- `module_shading.bypass_diodes` is removed; a configuration that gives it
  is refused as an unknown key. The result's shading summary reports the
  module's `bypass_diodes` and no longer `bypass_diodes_stated`.
- Callers that send three substrings per module for three-diode modules are
  unaffected by M1. M2 adds an optional key, and the bundled modules, all
  half-cut, state `cell_strings_in_parallel = 2`; on them a substring's
  limiting fraction now means each chain's worst cell, no longer the mean of
  the two chains' worst cells.
