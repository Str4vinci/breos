# 0007 — String topology and electrical checks

- **Status:** Proposed. D1–D6 are implemented on this branch; D7 is the next
  increment and has no code.
- **Date:** 2026-10-11

## Context

BREOS models PV at the array level. A system has arrays of modules, and one
inverter AC rating set by `inverter_loading_ratio` or `inverter_ac_rating_kw`
clips their summed output (`design/architecture/string-inverter-sizing.md`,
Phase 1). That is enough for sizing and economics, but it says nothing about
whether the system can be built. A string that is too long exceeds the DC
voltage rating on a cold morning. A string that is too short drops out of MPPT
tracking, or cannot restart the inverter, on a hot afternoon. Too many strings
on one input exceed its short-circuit rating. And an aggregate run reports the
energy of a system whose strings may not exist.

Layout tools decide where modules go and how they are wired. They need one
authoritative answer to "is this wiring buildable with this equipment at this
site?", and then an energy figure for the system as wired. The physics belongs
in the engine that already models the modules: BREOS carries their datasheet
values and fits the CEC single-diode model from them. Placement and routing do
not belong in BREOS, and neither do cable, fuse and protection selection, or
any jurisdiction's rules.

This record fixes the contract for specified equipment and string topology,
the checks run on it, and how App reports them. It also scopes the next
increment, which makes energy depend on the strings.

## Decisions

### D1. Equipment: `InverterSpec` with one `MpptSpec` per input

`breos.equipment.InverterSpec` holds an inverter's AC nameplate (W), optional
efficiency, and DC limits: maximum DC voltage and power, the MPPT voltage range
and the start-up voltage. These voltage limits apply to every input. `mppts`
lists the inputs in datasheet order. Each `MpptSpec` has its own operating
current, short-circuit current and string count, because asymmetric inverters
publish different limits per input. A Fronius Symo GEN24 6.0, for example, has
25 A and 12.5 A inputs. `isc_comparison_factor` is the multiplier the
manufacturer applies to array Isc before comparing it with an input's rating:
some require 1.25 × Isc, others compare Isc as it is.

Three states are kept apart:

| Value | Meaning |
|---|---|
| `None` (omitted) | Unknown: not published or not transcribed |
| `math.inf` (upper limit) or 0 (lower limit) | The equipment has no such limit |
| A number | The published limit |

JSON has no infinity: JavaScript writes `Infinity` as `null`, which here
means unknown. So "no limit" cannot cross a JSON boundary as a number. Equipment
without limits, such as an idealised inverter in an early design, should not be
sent as a topology at all: the caller leaves the three keys out and sizes the
inverter by `inverter_ac_rating_kw` or the loading ratio. TOML states `inf`
directly, and the Python API takes `math.inf`.

### D2. Topology is counted, not placed

`breos.electrical.PVString(inverter, mppt, array, modules, parallel=1)` is
`parallel` identical strings of `modules` modules in series. All of them come
from one PV array and connect to one input. The three references are 0-based
positions. Panel identities, positions and routing stay with the caller: BREOS
needs only how many modules of which array are in series on which input.

A string belongs to one array, and an array has one orientation, so a string
that mixes orientations cannot be declared. The caller's layout decides which
faces are grouped. Module-level power electronics (optimizers, micro-inverters)
are out of scope.

Entries on the same input add up. A caller may send each string as its own
entry with `parallel = 1`, or group identical strings; the check is the same.

### D3. Every module is strung exactly once

The strings must take exactly each array's module count. Modules left over are
an `unconnected_modules` error, and strings that take more modules than an array
has are an `array_overcommitted` error. To leave modules out, the caller lowers
the array's count. An unwired module therefore never counts toward production or
cost unless the caller says it is there.

### D4. Checks, temperatures and severities

Temperatures follow the usual string-sizing rules. The cell rises above ambient
by a mounting-dependent amount (`MOUNTING_TEMPERATURE_RISE_C`: flush 30 °C,
tilted rack 25 °C, ground mount 20 °C).

| Check | Temperature | Severity |
|---|---|---|
| `modules × Voc` ≤ min(inverter max DC, module rating) | Coldest ambient (no irradiance) | error |
| `modules × Voc` ≥ start-up voltage | Hottest cell | error |
| `modules × Vmp` ≥ MPPT minimum | Hottest cell | error |
| `modules × Vmp` ≤ MPPT maximum | Coldest operating cell (cold ambient + rise) | warning |
| Strings on an input ≤ its `max_strings` | — | error |
| Strings on an input share one array and one length | — | error |
| Σ design Isc × `isc_comparison_factor` ≤ input short-circuit rating | Hottest cell | error |
| Σ Imp ≤ input operating current | STC | warning |
| Strung STC power ≤ `max_dc_power_w` | — | warning |
| DC/AC ratio inside a caller's range (optional) | — | warning |

Start-up is checked at the hottest cell, not at STC, because the inverter must
also restart after a midday trip, when Voc is lowest. The upper MPPT check reads
the coldest *operating* cell because at the coldest ambient there is no
irradiance and the module is not producing. This matches PVsyst's `TOperMin` of
20 °C at the default −5 °C, tilted rack.

Design Isc is the hot-cell Isc, never below STC, times
`1 + bifaciality × 0.135`. 0.135 is 135 W/m² behind 1000 W/m², the bifacial
nameplate irradiance of IEC TS 60904-1-2. The flat 1.25 often seen here is the
NEC conductor factor, which belongs to cable sizing, not to the inverter's
limit.

Severities:

- `error`: a hard limit or the wiring is broken, and the design is not
  buildable as declared.
- `warning`: an operating limit is exceeded. The inverter holds its limit and
  loses some energy, but the design is safe.
- `missing`: a value the check needs is unknown, so that check did not run.

The status is `fail` on any error, then `incomplete` on any missing value, then
`warning`, then `pass`.

A missing value is never replaced by a default. The one exception is a derived
value, which is not a default. Datasheets rarely publish the Vmp temperature
coefficient, so `PVModuleParams.T_Vmp_pct_effective` derives it as
`T_Pmax_pct − T_Isc_pct`: to first order, since `Pmp = Vmp × Imp`, the power
coefficient is the sum of the Vmp and Imp coefficients. A published `T_Vmp_pct`
wins. The module's maximum system voltage has no such derivation, and a check
without it reports `missing` while still applying the inverter's limit.

### D5. App reports the check; energy stays aggregate

App takes `inverters`, `pv_strings` and `electrical_environment` together, or
none of them. With them:

- The inverters size the AC side: the AC rating is the sum of the nameplates,
  so `inverter_loading_ratio` and `inverter_ac_rating_kw` are refused. When
  every inverter states its efficiency, their nameplate-weighted mean replaces
  `inverter_efficiency`. Setting both is refused, and so is setting it on only
  some inverters.
- The check runs when the configuration is resolved, and the result carries an
  `electrical` block with `fidelity`, `status` and `findings`.
- A design that fails still simulates. Its numbers explain why it loses against
  a buildable design, but they do not describe a system that can be built.
  Whether a failing result may be ranked, exported or shown is the caller's
  policy, derived from `status`. BREOS does not encode it.

Every result carries the block. Without a topology it is
`{"fidelity": "aggregate", "status": "not_evaluated", "findings": []}`, so a
reader can always tell whether a check ran. In this increment `fidelity` is
always `"aggregate"`: with or without a topology, energy comes from the
array-level model and one AC rating. A run with a topology gives the same
numbers as the equivalent `inverter_ac_rating_kw` run, and a test pins that.

These are added fields, so the result format stays `"1"`.

### D6. Pure function first

`check_topology(arrays, inverters, strings, environment)` takes plain objects
and has no App dependency. Layout tools can call it on every edit without a
simulation, and App calls the same function, so the two can never disagree.

### D7. Next increment: energy per MPPT (not implemented)

`fidelity = "string"` will make energy depend on the strings:

1. Evaluate each array's single-diode model at every step for `v_mp` and
   `i_mp`, as it already does for power, and keep them.
2. Per input: the string voltage is `modules × v_mp`, and the current is the
   sum over its strings of `parallel × i_mp`, after the irradiance-type losses
   (soiling, shading) that act on current. Outside the MPPT range, operate at
   the nearest bound on the IV curve. Above the operating current, limit the
   current.
3. Per inverter: sum its inputs' DC and clip at its own AC rating, with its own
   efficiency. Each inverter's AC output is then added; inputs share their
   inverter's rating, with no fixed per-input AC cap.
4. String mode replaces the flat PVWatts `mismatch` loss, which stands in for
   what the string model computes, so it is not counted twice.

Several inverters with a battery need the converter topology of ADR 0004
(#419), which decides which inverter the battery shares, so string mode with a
battery waits for it. Partial shading, bypass diodes and cable losses are
separate increments.

## Consequences

- Callers with real equipment get one authoritative check. A layout tool may
  keep a fast local copy for immediate feedback, but only BREOS's result should
  be quoted as validated.
- Aggregate behaviour, defaults and numbers are unchanged when the three keys
  are absent. The result and the resolved configuration gain fields.
- Equipment data quality now matters: a check is only as good as the
  transcribed datasheet. BREOS ships no inverter catalogue. Callers bring
  equipment, and the tests use values from cited manufacturer sources.
- The module catalogue gains two optional fields that only the check reads, so
  existing modules and their energy are unchanged.
