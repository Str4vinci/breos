# 0007 — String topology and electrical checks

- **Status:** Proposed. D1–D7 are implemented, D7 for one inverter; several
  inverters are the next step.
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
25 A and 12.5 A inputs. An input is an independent tracker.
`isc_comparison_factor` is the multiplier the manufacturer applies to the
array's STC Isc before comparing it with an input's rating: some require
1.25 × Isc, others compare Isc as it is and are given 1. Omitted, it is
reported missing like any other unknown value. `max_dc_power_w` is the maximum
PV generator power the manufacturer permits; `recommended_dc_power_w` is its
recommendation. Datasheets publish one or the other, so neither is reported
missing when omitted.

Some datasheets publish further limits, and a Fronius Verto Plus publishes all
of them. On an input: each string connection's operating and short-circuit
current (`max_string_input_current_a`, `max_string_short_circuit_current_a`),
the PV generator power it may carry (`max_dc_power_w`) and the DC power it
converts (`max_usable_dc_power_w`). On the inverter: the short-circuit current
over all its inputs (`max_short_circuit_current_a`) and the DC power it
converts over all of them (`max_usable_dc_power_w`). Most datasheets publish
none of these, so they are not reported missing when omitted. Connection counts
beyond the permitted parallel strings are not modelled: `max_strings` is what
the inverter accepts.

Three states are kept apart:

| Value | Meaning |
|---|---|
| `None` (omitted) | Unknown: not published or not transcribed |
| `math.inf` (upper limit) or 0 (lower limit) | The equipment has no such limit |
| A number | The published limit |

JSON has no infinity: JavaScript writes `Infinity` as `null`, which here
means unknown. So "no limit" crosses a JSON boundary as the string `"inf"`,
which the configuration accepts for every upper limit, and the result's
provenance writes an unbounded limit the same way, so results stay strict
JSON. TOML states `inf` directly, and the Python API takes `math.inf`.
Equipment that is idealised altogether, such as a placeholder inverter in an
early design, is better sent without a topology: the caller leaves the three
keys out and sizes the inverter by `inverter_ac_rating_kw` or the loading
ratio.

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
Entries from different arrays may share an input when they have the same module
and the same length, as the two halves of a roof face split by a window, or an
east-west pair, often do. They share the input's voltage, and D7 models the
energy their different curves cost.

### D3. Every module is strung exactly once

The strings must take exactly each array's module count. Modules left over are
an `unconnected_modules` error, and strings that take more modules than an array
has are an `array_overcommitted` error. To leave modules out, the caller lowers
the array's count. An unwired module therefore never counts toward production or
cost unless the caller says it is there.

### D4. Checks, temperatures and severities

Temperatures follow the usual string-sizing rules. The cell rises above ambient
by a mounting-dependent amount (`MOUNTING_TEMPERATURE_RISE_C`: flush 30 °C,
tilted rack 25 °C, ground mount 20 °C). These are screening values, not bounds:
an insulated roof at zero wind runs hotter. A caller with a thermal study or
local rules states `hot_cell_temp_c` and `cold_operating_cell_temp_c` instead.
`min_design_temp_c` is the extreme annual minimum on the jurisdiction's or
manufacturer's basis, not a heating-design percentile, which is warmer and
would clear strings that are too long.

| Check | Temperature | Severity |
|---|---|---|
| `modules × Voc` ≤ inverter max DC, and ≤ module rating (one finding per ceiling) | Extreme minimum ambient, before appreciable solar heating | error |
| `modules × Voc` ≥ start-up voltage | Hottest cell | warning |
| `modules × Vmp` ≥ MPPT minimum | Hottest cell | warning |
| `modules × Vmp` ≤ MPPT maximum | Coldest operating cell (cold ambient + rise) | warning |
| Strings on an input ≤ its `max_strings` | — | error |
| Strings on an input share one module and one length | — | error |
| Σ max(factor × STC Isc, hot-cell Isc), with bifacial rear gain ≤ input short-circuit rating | STC and hottest cell | error |
| Σ Imp ≤ input operating current | STC | warning |
| Each string's design Isc ≤ its connection's short-circuit rating | STC and hottest cell | error |
| Each string's Imp ≤ its connection's operating current | STC | warning |
| Strung STC power on an input ≤ its `max_dc_power_w` | — | error |
| Σ design Isc over the inverter ≤ its short-circuit rating | STC and hottest cell | error |
| Strung STC power ≤ `max_dc_power_w` | — | error |
| Strung STC power ≤ `recommended_dc_power_w` | — | warning |
| DC/AC ratio inside a caller's range (optional) | — | warning |

Start-up is checked at the hottest cell, not at STC, because the inverter must
also restart after a midday trip, when Voc is lowest. A string below the start-up
voltage or below the MPPT minimum is a yield and availability problem, not
damage: the inverter works off the maximum power point or waits to start. Both
are warnings, so `fail` means a hard limit or a wiring inconsistency only.
D7 models the low-voltage case as lost energy. The upper MPPT check reads the
coldest *operating* cell, the usual winter operating temperature; PVsyst's
`TOperMin` is 20 °C, which the default −5 °C, tilted rack gives. A cold, sunny
morning can be colder, which a caller states with
`cold_operating_cell_temp_c`.

The short-circuit check takes the larger of two currents per string. The
manufacturer's rule multiplies the STC Isc by `isc_comparison_factor`; the
physical maximum is the hot-cell Isc, never below STC. Both include the bifacial
rear gain, `1 + bifaciality × bifacial_rear_irradiance_w_m2 / 1000`. The
default 135 W/m² is the bifacial nameplate irradiance of IEC TS 60904-1-2; a
site with more rear irradiance states it. Voc and Isc move with the coefficients
the energy model uses, `beta_voc` and `alpha_sc`, which honour absolute
overrides.

Severities:

- `error`: a hard limit or the wiring is broken, and the design is not
  buildable as declared.
- `warning`: an operating or yield limit is exceeded. The equipment is safe, but
  it loses energy or may not restart on the hottest days.
- `missing`: a value the check needs is unknown, so that check did not run.

The status is `fail` on any error, then `incomplete` on any missing value, then
`warning`, then `pass`.

Every finding carries `unit` (`"V"`, `"A"`, `"W"`, `"count"` or `"ratio"`) and
`limit_field`, the datasheet value its limit came from, so a client never reads
units or limits from the message. `breos.electrical.FINDING_CODES` publishes each
code's severity, unit and location fields. Codes and severities are stable; a
later release may add codes.

A missing value is never replaced by a default. The one exception is a derived
value, which is not a default. Datasheets rarely publish the Vmp temperature
coefficient, so `PVModuleParams.T_Vmp_pct_effective` derives it as the power
coefficient minus the Isc one, from `gamma_pmp_effective` and `alpha_sc`: to
first order, since `Pmp = Vmp × Imp`, the power coefficient is the sum of the
Vmp and Imp coefficients. It is an estimate for crystalline silicon, so a
published `T_Vmp_pct` wins and is preferred near a sizing boundary. The module's maximum system voltage has no such derivation, and a check
without it reports `missing` while still applying the inverter's limit.

### D5. App reports the check; energy stays aggregate by default

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
reader can always tell whether a check ran. `fidelity` is `"aggregate"` unless
D7's string mode is asked for: energy then comes from the array-level model and
one AC rating, and a run with a topology gives the same numbers as the
equivalent `inverter_ac_rating_kw` run. A test pins that.

These are added fields, so the result format stays `"1"`.

### D6. Pure function first

`check_topology(arrays, inverters, strings, environment)` takes plain objects
and has no App dependency. Layout tools can call it on every edit without a
simulation, and App calls the same function, so the two can never disagree.

### D7. Energy per MPPT input, opt-in

`electrical_fidelity = "string"` makes energy depend on the strings. The
default stays `"aggregate"`, so a topology alone changes no number.

1. Each array's single-diode model keeps, at every step, the five parameters
   and the maximum power point of one module
   (`PVProductionBreakdown.module_operating`).
2. The static PVWatts losses scale the current. Per array and step, the factor
   is the DC after losses over the module DC, so an input's unconstrained power
   equals the aggregate model's. Availability is the exception: it is time out
   of service, not less current while running, so it applies after the limits
   and cannot hide a current limit.
3. Per input, the strings run at their maximum power point when it lies inside
   the MPPT voltage range and the current is under the input's and each string
   connection's operating limit. Otherwise the inverter moves along the IV
   curve to the best point that respects them: the voltage at the current limit
   or the nearer end of the range. When no point of the curve satisfies them,
   the input has no usable operating point and delivers nothing for that step.
   A brute-force search of the IV curve pins this, including the empty case.
4. Strings from different arrays on one input share its voltage, so the input
   runs at the best point of their summed IV curve, found by a grid and a
   golden-section search after a bisection for the current limits. Where their
   irradiance or temperature differ, that point lies below the sum of their
   maximum powers, and the difference is lost. A brute-force search pins it.
5. An input delivers at most its usable DC power, and the inverter at most its
   own. The inverter can shed power at any operating point, so a power limit
   costs only the excess, however the inverter shares it among its inputs.
6. The inverter starts at the first producing step of a day on which an
   input's open-circuit voltage reaches `startup_voltage_v`, and runs until
   production stops. Once running, it works down to the MPPT minimum, so the
   start-up voltage costs energy only before the first start of a spell. String
   mode therefore needs the start-up voltage; 0 states none.
7. The lost DC is `mppt_limit_loss`, a waterfall stage after the static losses.
   Where no limit binds the DC is the aggregate model's, bit for bit.

The PVWatts `mismatch` loss stays. It stands for differences between modules
in a string, which a model of identical modules does not produce, so the string
model adds to it rather than replacing it.

The model runs for one inverter, whose rating, part-load curve and battery
coupling are the existing aggregate ones. For several inverters, each would need
its own AC clipping and, with a battery, the converter topology of ADR 0004
(#419). Until then, string mode refuses them. It also refuses a wiring with
structural errors and an input without the limits it needs, rather than fall
back to the aggregate model. A design that breaks a hard limit but is wired
definitely, such as one with an overvoltage, still gets string energy as a
diagnostic.

Later years scale the first year's limited DC by module degradation, as the
aggregate model does. A limit that binds in year 1 is assumed to bind as much
later, which slightly understates later years when it binds on current. Partial
shading of single modules, bypass diodes and cable losses are separate
increments.

### Open items

These are additive and do not change a field above:

- Per-array mounting and cell temperatures. The energy model takes one thermal
  model for every array by design, so a roof array and a carport share their
  cell temperatures; the check takes one mounting to match.
- Several inverters in string mode, after ADR 0004 (#419).

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
