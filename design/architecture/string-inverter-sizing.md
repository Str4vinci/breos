# String Inverter Sizing Design Note

BREOS currently simulates PV production at the array level: a system has a
module count, module model, tilt, azimuth, optional tracking, and an inverter
loading ratio. That is enough for fast techno-economic simulation, but it does
not prove that a design is electrically buildable.

This note scopes future string-inverter support for the BREOS engine. The goal
is to let BREOS validate and, where possible, model declared PV string topology
using module, inverter, site-temperature, and MPPT data supplied by callers.

## Why This Matters

Ignoring string design can produce optimistic or invalid results in several
cases:

- A module count may imply a string length that exceeds inverter or module
  maximum DC voltage during cold weather.
- A short string may fall below the inverter MPPT window or startup voltage
  during hot weather or low-light conditions.
- Multiple parallel strings can exceed MPPT short-circuit current limits, which
  is a hard electrical constraint.
- Operating current limits and inverter AC power limits can clip energy output.
- Multi-orientation systems can be modeled too optimistically if separate roof
  faces are combined without knowing whether they use separate MPPTs, parallel
  strings, optimizers, or an invalid mixed-orientation series string.

BREOS should not present aggregate PV simulations as electrical design
certification. It can, however, become the source of truth for electrical
feasibility checks and string-aware energy modeling once the required topology
and datasheet inputs exist.

## Proposed Scope

### Phase 1: Aggregate Inverter Clipping

Apply inverter AC power limits consistently in the main simulation path.
BREOS already has a `dc_to_ac()` helper that applies a PVWatts inverter model,
but `App.simulate()` currently passes array DC production into the energy
balance, where inverter efficiency is applied without an AC clipping limit.

This is the highest-priority accuracy improvement because it affects normal
simulations even when no explicit string topology is available.

### Phase 2: Electrical Feasibility Validation

Add an optional validation module that checks whether a declared module,
inverter, environment, and string configuration is electrically plausible. This
module should return structured warnings/errors rather than silently changing
the design.

Suggested inputs:

- Module specs: `Pmp`, `Voc`, `Vmp`, `Isc`, `Imp`, voltage/current temperature
  coefficients, and maximum system voltage.
- Inverter specs: rated AC power, max DC voltage, MPPT voltage range, startup
  voltage, number of MPPTs, per-MPPT max operating current, and per-MPPT max
  short-circuit current.
- Environment: minimum design temperature, maximum ambient temperature, and a
  mounting-temperature assumption for hot-cell voltage checks.
- String topology: strings per MPPT, modules per string, module model per
  string, and optional orientation metadata.

Core checks:

| Check | Condition | Severity |
| --- | --- | --- |
| Cold overvoltage | `N * Voc_cold > min(inverter_max_dc_voltage, module_max_system_voltage)` | Error |
| Hot MPPT minimum | `N * Vmp_hot < inverter_mppt_min_voltage` | Warning or error |
| Cold MPPT maximum | `N * Vmp_cold > inverter_mppt_max_voltage` | Warning |
| Startup voltage | `N * Voc_stc < inverter_startup_voltage` | Warning |
| Short-circuit current | `parallel_strings * Isc > mppt_max_short_circuit_current` | Error |
| Operating current | `parallel_strings * Imp > mppt_max_input_current` | Warning |
| Parallel string mismatch | unequal module count or mixed module models on one MPPT | Error |
| DC/AC ratio | `total_pdc_stc / inverter_rated_ac_power` outside configured guidance | Warning |

Temperature-adjusted voltage formulas:

```text
Voc_cold = Voc_stc * (1 + (T_min - 25) * Tk_Voc_pct / 100)
Vmp_hot  = Vmp_stc * (1 + (T_hot_cell - 25) * Tk_Vmp_pct / 100)
Vmp_cold = Vmp_stc * (1 + (T_min - 25) * Tk_Vmp_pct / 100)
```

String length guidance:

```text
N_max = floor(min(inverter_max_dc_voltage, module_max_system_voltage) / Voc_cold)
N_min = ceil(inverter_mppt_min_voltage / Vmp_hot)
```

### Phase 3: String-Aware Multi-Array Modeling

Once callers can provide MPPT and string topology, BREOS can model multi-array
systems with more fidelity:

- Separate MPPTs: simulate each array/string group independently, then combine
  power at the inverter AC limit.
- Parallel strings on one MPPT: require equal module counts and matching module
  models; optionally apply current clipping or mismatch assumptions.
- Mixed orientations in one series string: reject by default unless an explicit
  advanced model or optimizer architecture is provided.
- Power sharing: for multi-MPPT inverters, avoid fixed per-MPPT AC caps unless
  the inverter datasheet actually imposes them.

This phase should not invent geometric routing. BREOS should validate and model
topology supplied by a caller; layout tools can decide how panels are physically
grouped and routed.

### Phase 4: Datasheet Catalogs and Presets

String-aware validation only becomes useful with reliable component data.
BREOS will need a stronger module and inverter catalog before exposing this as
a user-facing guarantee.

Needed catalog fields:

- Module maximum system voltage.
- Module temperature coefficient for `Vmp` if different from power/Voc
  coefficients.
- Inverter absolute max DC voltage.
- Inverter MPPT operating range and startup voltage.
- Per-MPPT current limits, including separate short-circuit and operating
  current ratings.
- Number of MPPTs and any asymmetric MPPT limits.
- Rated AC power and maximum DC oversizing guidance where published.

## Non-Goals

- BREOS should not claim code compliance or installer certification.
- BREOS should not size conductors, fuses, breakers, rapid shutdown equipment,
  roof setbacks, or jurisdiction-specific protection requirements in this
  roadmap item.
- BREOS should not auto-route panel wiring unless a separate layout subsystem
  supplies panel geometry and that feature is explicitly scoped.

## Recommended Order

1. Add aggregate inverter AC clipping to the main `App` energy flow.
2. Add a pure validation API for string length, voltage, current, startup, and
   DC/AC ratio checks.
3. Extend PV module and inverter data models/catalogs with required datasheet
   fields.
4. Add optional `pv_strings` / `mppt_layout` inputs for callers that know the
   topology.
5. Use topology to improve multi-array and multi-MPPT energy modeling.
