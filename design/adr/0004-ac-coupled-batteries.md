# 0004 — AC-coupled batteries: a battery inverter on the AC bus

- **Status:** Proposed. No code yet; implementation is for a later release,
  after the maintainer accepts or amends the decisions below (#379).
- **Date:** 2026-10-03

## Context

BREOS models one battery topology: a DC-coupled hybrid inverter. PV and the
battery share one DC bus and one inverter. PV DC charges the battery without
crossing the inverter, and PV and battery discharge share the inverter's AC
rating and one part-load operating point. Many residential batteries are not
built that way. A battery added to an existing PV system, or sold as a unit
with its own inverter, connects to the house AC wiring through a second,
bidirectional inverter. Its energy then crosses more converters, the PV
inverter clips before the battery can take anything, and grid charging no
longer competes with PV for the PV inverter's rating. A study that compares
retrofit options, or that sizes a battery for an existing PV system, needs
that topology.

0.7.0 removed the App key `dc_coupled`, the CLI flag `--dc-coupled` and
`BatteryConfig.dc_coupled` because they could only say `True`
(`CHANGELOG.md:2422-2430`). An earlier release made `dc_coupled=False` fail
instead of silently running the DC-coupled model (`CHANGELOG.md:3131-3132`).
This record adds the topology as a real choice, with a new key rather than
the removed boolean.

ADR 0002 A6 left one question open that this topology answers in part. A6
counts PV AC output and grid-charge AC input together against the hybrid
inverter's rating, and defers netting because it "needs a converter topology
that says which paths share which power stage". In AC coupling the paths are
separate converters, so the PV and battery inverter limits are independent.

This record specifies the topology, the models, the configuration, the
accounting, the costs and the tests. It changes no code. It also estimates
the implementation effort, as the issue asks, before any work is scheduled.

## Current behaviour this record relies on

Line numbers are at `2e06c99` (`origin/develop`, 2026-10-03).

**Inverter cores.** `_dc_ac` (`breos/_dispatch.py:43-103`) is the one
DC-to-AC conversion. It applies the PVWatts part-load curve with
`pdc0 = inverter_ac_power / inverter_efficiency` (`:85`), clips input above
`pdc0` (`:86`, `:101`), applies `ac_output_scale` after the curve and
nameplate (`:100`), and returns `(ac, conversion_loss, clipping_dc)`. With a
non-finite rating it is a flat efficiency with no clipping (`:77-79`); a zero
rating or efficiency converts nothing (`:68-69`). `_dc_for_ac`
(`:106-141`) is its inverse: the input needed for a requested output, capped
at `pdc0` (`:125-127`). `math.pow(zeta, pow_two)` with a run-time `pow_two`
keeps the Python and Numba results identical (`:55-61`, `:96`).
`_calculate_dc_ac_power_arrays` (`breos/inverter.py:135-171`) is the
vectorised form, used by the PV-only path.

**The DC step.** `_dispatch_dc_step` (`breos/_dispatch.py:520-711`):

- PV is converted once at the shared inverter (`:577`). When PV AC covers the
  load, the DC needed for the load is solved by `_dc_for_ac` (`:581`), the DC
  surplus charges the battery before any conversion (`:582-583`), and the
  rest is converted again and exported (`:585-592`).
- When PV alone saturates the inverter, the DC above `pdc0` charges the
  battery before it is curtailed (`:598-607`): DC-coupled clipping recovery.
  The battery cannot discharge then, since the inverter has no headroom.
- Otherwise the battery discharges toward the load through the same inverter
  at one shared operating point (`_combined_conversion`, `:498-517`, used at
  `:627-646`). A binding `max_discharge_power_w` is solved by a 40-step
  bisection on the battery's AC share (`:626-642`). The loss is split between
  PV and battery in proportion to their DC (`:660-668`).
- `_charge` (`:438-455`) limits DC drawn to the window room, the DC
  charge-input limit and the stored-energy (C-rate) limit.
- Grid charging runs after PV, only when no PV is exported and the battery
  has not discharged (`:674`). It gets what PV left of the charge-input and
  stored-energy limits, the inverter rating less the step's PV AC output, and
  the site import limit less the load's import (`:679-690`). `_grid_charge`
  (`:458-495`) converts grid AC `a` to DC `a × grid_eff` and stores it at the
  charge efficiency.
- `hold_target` (`:575`) is set when discharge is allowed and the discharge
  floor is at or above a finite grid target (ADR 0002 A14). It lands a
  binding floor exactly (`:648-657`) and a reached target exactly
  (`:493-494`).

**The day loop.** `_dispatch_day` (`:714-948`) applies the capacity window
and standby loss (`_apply_capacity_window`, `:360-435`), turns instruction
fractions into energies (`:800-806`), takes one origin fraction per step
before dispatch (`:811-819`), refuses a step that both charged and discharged
(`:858-859`), adds PV charge to the PV origin as `pv_dc_to_battery ×
eff_charge` and grid charge to the grid origin (`:860-873`), runs the thermal
model on the DC charge input and discharge (`:881-892`), and writes every row
(`:899-948`). `PV_Production` is `pv_dc − curtailed − direct inverter loss`
(`:877`). `_day_arguments` (`:951-1019`) packs both backends' arguments, and
reads the inverter fields from `BatteryConfig` (`:1005`, `:1012`) and the two
grid-charging scalars from the instructions (`:1017-1018`).

**Numba.** `breos/_numba_dispatch_kernels.py:35-49` registers the scalar
helpers and compiles `_dispatch_day` itself with `fastmath=False` and an
on-disk cache keyed on `breos/_dispatch.py`. The backend contract is in
[Numba dispatch backend](../architecture/numba-dispatch-backend.md).

**Ledger.** `_LEDGER_COLUMNS` (`breos/_dispatch.py:192-233`); the version is
`LEDGER_SCHEMA_VERSION = "3.0"`, bumped for any visible change and by a minor
step for additive ones (`:144-164`). `tests/energy_conservation.py:81-150`
asserts the per-step identities, including the whole-system balance with PV
DC, grid AC and replacement-added energy as inputs (`:131-150`), and
`:166-202` reconciles the three origins.

**`BatteryConfig`** (`breos/battery.py:93-322`) says "Only DC-coupled
systems (hybrid inverters) are modelled" (`:98-102`). `max_charge_power_w` is
DC into the battery path, `max_discharge_power_w` is battery AC delivered and
`power_limit_c_rate` limits stored energy (`:104-112`, `:208-215`).
`inverter_ac_capacity_w` is the shared AC rating, None for no clipping
(`:205-207`). `ac_output_scale` derates inverter AC output inside dispatch
(`:114-121`, `:216-224`). Limits become per-step energies in `_step_energy_cap`
(`:762-768`) and are set once per span in `_simulate_core` (`:2000-2006`).

**PV-only and retired steps.** A run without a usable battery takes the
vectorised `_dispatch_no_battery_vectorized` (`:1185-1251`, called at
`:2060-2083`). A retired pack's steps take the same path with the same
inverter (`:2120-2132`, ADR 0003 E12), so they equal a PV-only run bit for
bit. `AlignedSimulationInputs.with_pv_only_chain` (`:520-564`) memoizes that
conversion; `_simulate_core` checks the memo key only for PV-only runs
(`:1902-1905`), and Monte Carlo builds the memo only without a battery
(`breos/montecarlo.py:552-553`).

**App and projection.** `build_battery_config` (`breos/projection.py:53-89`)
passes `inverter_efficiency` and the resolved AC rating;
`build_pv_only_battery_config` (`:92-104`) does the same for a PV-only year.
The rating comes from `inverter_ac_rating_kw` or the DC peak over
`inverter_loading_ratio` (`breos/app_config.py:2701-2706`), and only one of
them may be set (`:2671-2674`). Year rows report self-consumption as direct
PV AC plus PV-origin battery AC (`breos/projection.py:289`) and diagnostics
by ledger column (`:211-225`). The App PV loss waterfall reads the inverter
and PV DC columns (`breos/runners/app.py:240-249`, `:314-335`).

**Smart charging.** `grid_charge_efficiency` is required for the
grid-charging modes and has no default, because the inverter model has no
AC-to-DC path to derive it from (`breos/app_config.py:1876-1886`; ADR 0002
A6). `DispatchInstructions` requires it in `(0, 1]`
(`breos/dispatch_instructions.py:52`, `:89-91`). The daily-target planner
dispatches candidate days through the same day loop with the config's caps
(`breos/_daily_targets.py:382-397`) and prices terminal refill at
`grid_charge_efficiency × eff_charge` (`:505-508`).

**Costs.** `calculate_costs` (`breos/economics.py:157-243`) prices the
inverter at `inverter_cost_per_kw` ("hybrid") with a battery and at
`inverter_cost_per_kw_nobatt` ("simple") without (`:205-210`), on the
dispatch's AC rating (`:188-193`). The config keys are
`inverter_cost_per_kw_hybrid` and `inverter_cost_per_kw_simple`
(`:33-49`; presets in `breos/data/configs/costs.json`). A battery replacement
is priced per kWh of storage (`replacement_event_cost`, `:130-155`). BREOS
prices no inverter replacement.

**Optimizer.** `_build_battery_config_from_spec`
(`breos/optimization.py:251-275`) forwards the `[battery]` keys
(`breos/optimization_config.py:209-246`) and sizes the inverter from the
design's DC peak and `dc_ac_ratio` (`breos/optimization.py:857`, `:1148`).

## Decision

### C1. Topology

An AC-coupled system has two inverters on the house AC bus:

```text
PV array --DC--> PV inverter --AC--+--> load
                                   +--> grid (export)
                                   +<-> battery inverter <--DC--> battery
grid -----------------------AC-----+
```

- The **PV inverter** converts PV DC to AC. It is the inverter BREOS models
  today for a PV-only system, with the same rating, efficiency, part-load
  curve and clipping.
- The **battery inverter** is bidirectional. Charging, it converts AC from
  the bus (PV surplus or grid) to DC for the battery. Discharging, it
  converts battery DC to AC for the load.
- Energy never passes from the PV array to the battery as DC. PV energy that
  is stored crosses three conversions before it reaches the load: PV
  inverter, battery inverter charging, battery inverter discharging.
- Battery discharge never exports, as today. Grid charge is AC from the grid
  into the battery inverter.

### C2. Configuration shape and defaults

One new choice, `coupling`, selects the topology. It is a string so a later
topology can add a value; the removed boolean is not restored.

| Where | Key | Values | Default |
|---|---|---|---|
| App, Monte Carlo | `battery_coupling` | `"dc"`, `"ac"` | `"dc"` |
| Optimizer | `[battery] coupling` | `"dc"`, `"ac"` | `"dc"` |
| `BatteryConfig` | `coupling` | `"dc"`, `"ac"` | `"dc"` |

The battery inverter has three settings. They apply only to `"ac"`, and
`"dc"` refuses them, so no setting is silently ignored:

| Setting | App, Monte Carlo | Optimizer `[battery]` | `BatteryConfig` |
|---|---|---|---|
| AC rating, absolute | `battery_inverter_ac_rating_kw` | `inverter_ac_rating_kw` | `battery_inverter_ac_capacity_w` |
| AC rating per kWh of nominal capacity | `battery_inverter_kw_per_kwh` | `inverter_kw_per_kwh` | (resolved to the absolute rating) |
| Nominal efficiency | `battery_inverter_efficiency` | `inverter_efficiency` | `battery_inverter_efficiency` |

- App, Monte Carlo and the optimizer require exactly one of the two rating
  keys with `"ac"` and a battery, as `inverter_ac_rating_kw` and
  `inverter_loading_ratio` size the PV inverter today
  (`breos/app_config.py:2671-2674`). The per-kWh form lets a capacity sweep
  keep one inverter-to-battery ratio, as `power_limit_c_rate` keeps one
  C-rate.
- `BatteryConfig.battery_inverter_ac_capacity_w = None` means no rating: the
  flat-efficiency, unclipped conversion `_dc_ac` already gives a missing PV
  inverter rating (`breos/_dispatch.py:77-79`). App never uses it; the
  reference cases below do, because flat conversion can be computed by hand.
- The efficiency defaults to 0.96, the PV inverter's default
  (`breos/app_config.py:626-627`). It is not tied to `inverter_efficiency`:
  they are two products.
- With `"ac"`, `inverter_efficiency`, `inverter_ac_rating_kw`,
  `inverter_loading_ratio` and `ac_output_scale` describe the PV inverter
  only (C3, C6).
- The existing battery keys keep their meaning in both topologies:
  `battery_max_charge_power_w` is DC into the battery, so in AC coupling the
  DC side of the battery inverter; `battery_max_discharge_power_w` is battery
  AC delivered, so the battery inverter's AC output; `battery_power_limit_c_rate`
  limits stored energy.
- A config without `battery_coupling` is `"dc"` and resolves exactly as
  today. A PV-only config (`battery_kwh = 0`) accepts `"ac"` and the battery
  inverter keys, does not require a rating, and runs the PV-only path; its
  CAPEX prices no battery inverter (C9), so a sweep over capacities from 0
  is consistent.

### C3. PV inverter

In AC coupling the PV inverter is the PV-only system's inverter. Its
conversion for the whole span is computed once, before the day loop, by
`_calculate_dc_ac_power_arrays` with the PV rating, `inverter_efficiency` and
`ac_output_scale`, exactly as the PV-only path computes it
(`breos/battery.py:1213-1218`), or taken from the `with_pv_only_chain` memo.
The day loop receives the three arrays (PV AC, conversion loss, DC clipping)
and the AC step never converts PV itself.

This is what makes the PV side independent of the battery: whatever the
battery does, the PV inverter's clipping, loss and AC output are the PV-only
run's, element for element. It also reuses the existing memo for battery
runs (C13).

### C4. Battery inverter model

The battery inverter uses the same PVWatts part-load core in both
directions, with its own rating `P` and nominal efficiency `η_b`. `P` is an AC
rating: the AC output when discharging and the AC input when charging, as
residential battery inverter datasheets state it. The part-load ratio is the
input over the rated input in both directions.

- **Discharging** (DC to AC): `ac = _dc_ac(dc, P, η_b, 1.0, pow_two)`, so
  `pdc0 = P / η_b`. The inverse is `_dc_for_ac(ac, P, η_b, 1.0)`.
- **Charging** (AC to DC): `dc = _dc_ac(ac, P × η_b, η_b, 1.0, pow_two)`,
  so the input limit `pdc0` is `P` itself and the DC output is at most
  `P × η_b`. The inverse is `_dc_for_ac(dc, P × η_b, η_b, 1.0)`.
- With `P` None both are flat: `ac = dc × η_b` and `dc = ac × η_b`.
- `ac_output_scale` is 1.0 for the battery inverter (C6).

Using one core keeps DC-coupled and AC-coupled results comparable: the DC
step already discharges through this curve, so a comparison between the two
topologies shows the topology, not two converter models. The curve is an
empirical inverter curve; using it in the charging direction is an
assumption, stated as such in the docs. At light load it is markedly less
efficient than `η_b`: 300 W from a 3000 W, 0.96 inverter needs 324.75 W of DC,
an efficiency of 92.4 % (reference case AC6).

### C5. The AC step: conversion paths and their order

A new scalar function `_dispatch_ac_step`, in `breos/_dispatch.py` beside
`_dispatch_dc_step`, dispatches one AC-coupled step in Wh. It keeps the
greedy order and the instruction semantics of the DC step. With
`pv_ac`, `pv_loss` and `pv_clip` from C3, `E` the stored energy after the
capacity window, `room = max(0, emax − E)` and
`available = max(0, E − discharge_floor)`:

```text
pv_ac_to_battery = pv_dc_charge = battery_dc = battery_ac = 0.0
pv_ac_to_load = min(pv_ac, load)
surplus_ac    = pv_ac - pv_ac_to_load
deficit       = load - pv_ac_to_load

# 1. PV surplus charges through the battery inverter
if surplus_ac > 0 and room > 0:
    dc_wanted        = min(room / eff_c, cap_charge_in, cap_stored / eff_c)
    pv_ac_to_battery = min(surplus_ac, charge_inverse(dc_wanted), P)
    pv_dc_charge     = charge_forward(pv_ac_to_battery)
    E               += pv_dc_charge * eff_c
pv_ac_export = surplus_ac - pv_ac_to_battery

# 2. the battery serves the deficit through the battery inverter
#    (a step with a surplus has no deficit)
if deficit > 0 and discharge_allowed and available > 0:
    ac_target  = min(deficit, cap_discharge_ac, P)
    battery_dc = min(available * eff_d, discharge_inverse(ac_target), cap_stored * eff_d)
    battery_ac = discharge_forward(battery_dc)
    draw       = battery_dc / eff_d          # hold_target landing as at :648-657
    E         -= draw
grid_import = max(0, deficit - battery_ac)

# 3. grid charge (C7)
```

- PV serves the load first, then charges, then exports, as today. A step
  with a surplus has no deficit, so a step still charges or discharges, never
  both, and the check at `:858-859` stays.
- `cap_discharge_ac` and `P` bound the AC target directly. No bisection is
  needed: the battery inverter does not share an operating point with PV.
- When the PV surplus is larger than the battery can take, the rest is
  exported. When the battery is full, `pv_ac_to_battery` is 0 and
  `pv_ac_export = surplus_ac − 0.0 = surplus_ac`. With no battery flow the
  step's PV flows are therefore the PV-only path's expressions
  (`breos/battery.py:1223-1225`) applied to the same arrays, which gives
  invariant P3 below.
- The step returns the DC step's sixteen values plus `pv_ac_to_battery` and
  the charge conversion loss of the PV share, and it returns the DC that PV
  put into the battery separately from `pv_dc_to_battery` (C8).

### C6. Power limits and clipping

| Limit | DC-coupled (today) | AC-coupled |
|---|---|---|
| PV inverter rating | shared by PV output, battery discharge and grid-charge input (A6) | PV output only |
| Battery inverter rating `P` | none | battery AC in (PV surplus plus grid) and battery AC out |
| `max_charge_power_w` | DC into the battery, PV plus grid | the same: DC side of the battery inverter |
| `max_discharge_power_w` | battery AC share at the shared operating point | battery inverter AC output |
| `power_limit_c_rate` | stored energy | stored energy |
| `grid_import_limit_w` | load import plus grid charge | the same |
| `ac_output_scale` | shared inverter output, PV and battery | PV inverter output only |

**Clipping.** In AC coupling, PV DC above the PV inverter's `pdc0` is
curtailed whatever the battery state: `PV_DC_Curtailed` is the PV-only run's
clipping. DC-coupled clipping recovery (`breos/_dispatch.py:598-607`) has no
equivalent, because the battery sees only AC. On a system with a high
loading ratio this is a real yield difference between the topologies, and it
shows in reference case AC2.

**Discharge while PV saturates.** In DC coupling an inverter saturated by PV
has no headroom, so the battery cannot discharge even when the load exceeds
the PV output (`:598-607`). In AC coupling the battery inverter has its own
rating, so the battery serves the remaining deficit.

**`ac_output_scale`.** It stands in for AC shortfall of the PV system such
as availability and downstream wiring (`breos/battery.py:114-121`). In AC
coupling it applies to the PV inverter only. Applying it to the battery
inverter would derate battery output for PV availability.

### C7. Grid charging

Grid charging keeps A6's rules on when it runs and what it may use: after PV
allocation, only when no PV is exported and the battery has not discharged,
toward the instruction's target, within the charge-input, stored-energy and
site limits PV left. In AC coupling the grid charge passes through the
battery inverter, not the PV inverter:

- **Conversion.** The battery inverter's charging conversion (C4). PV surplus
  and grid AC entering it in the same step share one operating point, as PV
  and battery share one in `_combined_conversion`: the total AC is converted
  once and the DC is split in proportion to each source's AC.

  ```text
  room_t          = min(target, emax) - E_after_pv
  dc_total_wanted = min(pv_dc_charge + room_t / eff_c, cap_charge_in, cap_stored / eff_c)
  grid_ac         = min(charge_inverse(dc_total_wanted) - pv_ac_to_battery,
                        P - pv_ac_to_battery,
                        grid_import_cap - grid_import)
  # nothing more happens if room_t <= 0 or grid_ac <= 0
  total_ac        = pv_ac_to_battery + grid_ac
  dc_total        = charge_forward(total_ac)
  grid_dc         = dc_total * grid_ac / total_ac
  pv_dc_charge    = dc_total - grid_dc
  E               = E_before_charge + dc_total * eff_c   # E_before_charge: after the capacity window
  ```

  With `hold_target` and no limit binding (`grid_ac` equal to its first
  term) `E` lands on `min(target, emax)` exactly, as `_grid_charge` does at
  `:493-494`.
- **Rating.** The PV inverter's rating does not limit grid charge. A6's
  summed-throughput rule applies to the battery inverter instead: PV and grid
  AC into it together are at most `P`.
- **`grid_charge_efficiency`.** The battery inverter model is the AC-to-DC
  path A6 said BREOS lacked. With `"ac"` the `[smart_charging]` key
  `grid_charge_efficiency` is refused, and `DispatchInstructions` carry
  `grid_charge_efficiency = None`, which only an AC-coupled run accepts; a DC
  run still requires a value in `(0, 1]`. A value the step would ignore is
  never accepted.

With flat efficiencies and `η_b` equal to a DC run's `grid_charge_efficiency`,
night-time grid charging is the same in both topologies (AC3). The
topologies differ when PV is running: AC4 shows a step where DC coupling can
neither grid charge (the inverter is full of PV) nor curtail, and AC coupling
grid charges while it clips PV.

### C8. Ledger and energy-origin accounting

**New columns.** Two ledger columns, added after `Grid_Charge_Conversion_Loss`
and zero in DC-coupled and PV-only runs:

- `PV_AC_To_Battery`: PV inverter AC that enters the battery inverter.
- `PV_Charge_Conversion_Loss`: the battery inverter's charging loss on that
  AC.

The grid trio keeps its meaning: `Grid_AC_To_Battery`, `Grid_DC_To_Battery`
and `Grid_Charge_Conversion_Loss` are the grid's AC, its DC and the loss of
the conversion between them, now in the battery inverter.

**Existing columns in AC coupling.**

- `PV_DC_To_Battery` is 0: no PV DC reaches the battery. Reusing it for the
  DC that PV-origin AC becomes would break the PV DC split identity and give
  one column two meanings.
- `PV_DC_To_Inverter`, `PV_DC_Curtailed` and `PV_Direct_Inverter_Loss`
  describe the PV inverter, and equal the PV-only run's.
- `PV_AC_To_Load` and `PV_AC_Export` keep their meaning.
- `Battery_Charge_Input` is the DC into the battery from both sources:
  `PV_DC_To_Battery + (PV_AC_To_Battery − PV_Charge_Conversion_Loss) +
  Grid_DC_To_Battery`.
- `Battery_Inverter_Loss` is the battery inverter's discharging loss, and
  `Inverter_Loss` stays `PV_Direct_Inverter_Loss + Battery_Inverter_Loss`.
  Charging losses are in the two conversion-loss columns, as grid charging's
  already is.
- `PV_Production` stays `PV_DC − PV_DC_Curtailed − PV_Direct_Inverter_Loss`
  (`breos/_dispatch.py:877`), which in AC coupling is the PV inverter's AC
  output.

**Origins (ADR 0002 A8).** The three origins and their proportional removals
are unchanged. Each source's charge adds to its own origin: PV-origin stored
energy rises by the DC PV put into the battery times the charge efficiency,
and grid-origin by `grid_dc × eff_c`. The day loop today computes the PV
share as `pv_dc_to_battery × eff_charge` (`:862`). It will use the step's DC
from PV instead, which a DC-coupled step returns as the same float it
returns for `pv_dc_to_battery`, so DC-coupled origins are unchanged bit for
bit. Self-consumption stays direct PV AC plus PV-origin battery AC
(`breos/projection.py:289`), and avoided emissions stay on net exchange
(A10).

**Versions.** The ledger schema becomes `"3.1"`: two columns are added and
none renamed, the precedent of 1.1. The result format stays `"1"`.

**Reports.** Year rows gain the diagnostics `PV_AC_To_Battery_kWh` and
`PV_Charge_Conversion_Loss_kWh` (`breos/projection.py:211-225`). The App PV
loss waterfall's `inverter` block stays the PV inverter's; a
`battery_inverter` block (rating, efficiency, charging and discharging
conversion loss) is added for AC coupling, and `energy_balance.ac_delivery`
gains `pv_to_battery_kwh` (`breos/runners/app.py:314-335`).

### C9. Costs

- **PV inverter.** With `"ac"` it is priced at `inverter_cost_per_kw_simple`
  on its rating, with or without a battery. `inverter_cost_per_kw_hybrid`
  prices only a DC-coupled system with a battery, as today
  (`breos/economics.py:205-210`).
- **Battery inverter.** A new cost key `battery_inverter_cost_per_kw`
  (`CostParams.battery_inverter_cost_per_kw`) prices the battery inverter's
  AC rating, only with `"ac"` and a battery. `calculate_costs` gains the
  rating and the coupling, and reports `battery_inverter_cost` (0.0 for DC
  coupling) inside `total_initial_cost`. The key has no default and no preset
  value: BREOS has no sourced price for it, so an AC-coupled run with a
  battery and without the key is refused, with a message naming it. Adding
  a sourced preset value is an open question.
- **Installation.** `installation_cost_battery` applies in both topologies.
- **Replacement.** A battery replacement prices the pack, per kWh of storage
  (`replacement_event_cost`), in both topologies (ADR 0003 E3, E4). The
  battery inverter is not replaced with the pack and no inverter replacement
  is priced, as for the PV or hybrid inverter today. Products that sell pack
  and inverter as one unit are an open question.
- **Terminal credit.** ADR 0003 E10 credits only the pack; both inverters are
  excluded, as the inverter is now.
- **O&M.** BREOS has no per-component O&M, so nothing is added.

DC-coupled CAPEX, replacement outlays and every cash flow are unchanged.

### C10. DC-coupled results stay bit for bit

- **Config.** `"dc"` is the default everywhere, and the AC-only keys are
  refused with it. `provenance.resolved_config` gains `battery_coupling:
  "dc"` (and the AC keys as null), so the six App golden fixtures and the
  stored gallery results are regenerated for that field only; no number
  changes. Config docs are regenerated with `tools/generate_config_docs.py`.
- **Dispatch selection.** One day loop serves both topologies. `_dispatch_day`
  gains a coupling flag, the battery inverter's rating and efficiency, and
  the three PV chain arrays (empty arrays for DC coupling). Each step calls
  `_dispatch_dc_step` or `_dispatch_ac_step` on that flag. `_dispatch_dc_step`
  is not edited, the DC branch writes 0.0 to the two new rows, and every
  existing expression in the day loop keeps its operands and order. Two day
  loops were rejected: they would duplicate the capacity window, origins,
  thermal model and row writes that the backend note keeps in one place.
- **Numba.** `_dispatch_ac_step` is registered with `register_jitable` beside
  the DC helpers (`breos/_numba_dispatch_kernels.py:35-46`), and both
  branches assign the same float variables, so the kernel keeps one
  signature. The edit to `breos/_dispatch.py` invalidates the on-disk cache
  once. The parity harness gains AC-coupled scenarios and keeps its DC ones.
- **Gates.** The App goldens, `tests/test_golden_outputs.py`, the Numba
  parity tests and `tools/parity/harness.py --instructions` must pass
  unchanged for DC coupling. Kernel changes need the slow and macOS/Windows
  workflow dispatched before merging.

### C11. Smart charging, `hold_target` and planners

- **Instructions.** `discharge_allowed`, `reserve_fraction` and
  `grid_target_fraction` mean the same in both topologies, including A7's
  moving window and A14's shared fraction. `discharge_only` needs nothing
  new.
- **`hold_target`.** The flag is computed as at `:575`, lands a binding
  floor as at `:648-657` and a reached target as in C7.
- **Daily persistence and the daily-target planner.** Both dispatch candidate
  days through the canonical day loop, so they model AC coupling once the
  loop does. `_DayEvaluator` must pass the battery inverter caps and a PV
  chain computed from the forecast PV (`breos/_daily_targets.py:382-397`).
  The terminal refill price (`:505-508`) uses `η_b × eff_charge` in place of
  `grid_charge_efficiency × eff_charge`. It is a flat price per stored
  kWh, as today; the part-load curve is not used there.
- **Controllers** carry the instructions' scalars and require them equal
  across a run (`breos/_controller.py:346-351`); `None` is one value there
  like any other.

### C12. Retirement and replacement

- **Retirement (ADR 0003 E12).** A retired AC-coupled pack's steps take the
  PV-only path with the PV inverter, as now (`breos/battery.py:2120-2132`).
  They equal the PV-only run bit for bit and, by C3, equal the AC-coupled
  steps of an idle battery too. The battery inverter is idle; no money
  changes.
- **Replacement (E3, E4, E11).** Unchanged: the swap is booked at the same
  instant and prices the pack (C9). The new pack is behind the same battery
  inverter.

### C13. Monte Carlo and the optimizer

- **Monte Carlo** builds its battery through `build_battery_config`, so the
  keys reach every trajectory. Nothing new is sampled. Because the PV
  inverter does not depend on the battery (C3), the PV chain memo
  (`breos/montecarlo.py:534-575`) can serve AC-coupled battery runs; today it
  is built only without a battery (`:552-553`). This is an optional
  performance step; results are identical with and without it.
- **Optimizer.** `coupling` is fixed for a study, not a decision variable:
  comparing topologies is two studies. With `inverter_kw_per_kwh` the battery
  inverter rating follows each candidate's capacity; with
  `inverter_ac_rating_kw` it is fixed. Each candidate's CAPEX includes its
  battery inverter. The design-invariant year cache is unchanged. For AC
  coupling the PV chain depends only on the module count, layout, weather and
  project year, not on the battery, so it can be hoisted across battery
  capacities, another optional step.

### C14. Provenance

`provenance.resolved_config` records the coupling and the battery inverter
keys. The optimizer's battery treatment records the coupling and the
resolved inverter rating per candidate. The ledger schema version is
reported as today.

## Energy closure

Every identity of `tests/energy_conservation.py` holds for both topologies,
with two changes that are zero for DC coupling:

1. PV DC split: `PV_DC = PV_DC_To_Battery + PV_DC_To_Inverter +
   PV_DC_Curtailed` (unchanged; `PV_DC_To_Battery = 0` in AC coupling).
2. Charge input split: `Battery_Charge_Input = PV_DC_To_Battery +
   PV_AC_To_Battery − PV_Charge_Conversion_Loss + Grid_DC_To_Battery`
   (changed: adds the PV AC path).
3. Whole-system balance: the outputs gain `PV_Charge_Conversion_Loss`. The
   inputs stay PV DC, grid AC to the battery and replacement-added energy
   (changed). `PV_AC_To_Battery` is internal and appears in neither.

Two identities and two bounds are new, asserted for AC coupling:

4. PV inverter: `PV_DC_To_Inverter − PV_Direct_Inverter_Loss = PV_AC_To_Load
   + PV_AC_Export + PV_AC_To_Battery`.
5. Battery inverter, charging: `PV_AC_To_Battery + Grid_AC_To_Battery −
   PV_Charge_Conversion_Loss − Grid_Charge_Conversion_Loss =
   Battery_Charge_Input`.
6. Battery inverter rating: `PV_AC_To_Battery + Grid_AC_To_Battery ≤ P` and
   `Battery_AC_To_Load ≤ P`, in average W, to rounding.
7. Origins: the PV-origin charge stored is `(PV_AC_To_Battery −
   PV_Charge_Conversion_Loss) × eff_charge` without resistance fade; the
   three-origin reconciliation is unchanged.

If identity 4 also holds for DC coupling, the implementation asserts it for
both. This record does not rely on that.

## Reference cases

Every case uses `BatteryConfig(nominal_energy_wh=10000, min_soc=0,
max_soc=1, charge_efficiency=0.95, discharge_efficiency=0.95,
inverter_efficiency=0.96, standby_loss_wh=0, thermal_resistance_k_per_w=0,
enable_replacement=False)` with `battery_inverter_efficiency=0.96`, hourly
steps at 25 °C (capacity factor 1, so `emin = 0` and `emax = 10 000 Wh`),
and no power limits. Ratings are None (flat conversion) unless given.
Values are Wh per one-hour step. The DC-coupled values were computed with
the current code at `2e06c99`.

**AC1 — PV through the battery and back.** Initial energy 0. Step 1: PV DC
3000, load 1000. Step 2: PV 0, load 1500.

| Column | AC step 1 | AC step 2 | DC step 1 | DC step 2 |
|---|---|---|---|---|
| `PV_Direct_Inverter_Loss` | 120 | 0 | 41.666667 | 0 |
| `PV_AC_To_Load` | 1000 | 0 | 1000 | 0 |
| `PV_DC_To_Battery` | 0 | 0 | 1958.333333 | 0 |
| `PV_AC_To_Battery` | 1880 | 0 | 0 | 0 |
| `PV_Charge_Conversion_Loss` | 75.2 | 0 | 0 | 0 |
| `Battery_Charge_Input` | 1804.8 | 0 | 1958.333333 | 0 |
| `Battery_Charge_Stored` | 1714.56 | 0 | 1860.416667 | 0 |
| `Battery_AC_To_Load` | 0 | 1500 | 0 | 1500 |
| `Battery_Discharge_DC` | 0 | 1644.736842 | 0 | 1644.736842 |
| `Battery_Inverter_Loss` | 0 | 62.5 | 0 | 62.5 |
| `PV_Origin_Battery_AC_To_Load` | 0 | 1500 | 0 | 1500 |
| `Battery_Energy` | 1714.56 | 69.823158 | 1860.416667 | 215.679825 |

Step 1 closes: 3000 = 1000 + 120 + 75.2 + 90.24 (charge loss) + 1714.56.
Delivering 1500 Wh from the battery costs the same in both. From the same PV
surplus, AC coupling stores `η × η_b = 0.9216` of the DC-coupled stored
energy. PV DC to load through the
battery is `0.96 × 0.96 × 0.95 × 0.95 × 0.96 = 0.798` in AC coupling against
`0.95 × 0.95 × 0.96 = 0.866` in DC coupling.

**AC2 — clipping is not recovered.** PV inverter 2000 W, PV DC 3000, load 0,
initial energy 5000. At full load the PVWatts curve returns the rating
exactly, so these values are exact.

| Column | AC | AC, battery inverter 1500 W | DC |
|---|---|---|---|
| `PV_DC_Curtailed` | 916.666667 | 916.666667 | 0 |
| `PV_Direct_Inverter_Loss` | 83.333333 | 83.333333 | 0 |
| `PV_AC_To_Battery` | 2000 | 1500 | 0 |
| `PV_Charge_Conversion_Loss` | 80 | 60 | 0 |
| `PV_DC_To_Battery` | 0 | 0 | 3000 |
| `Battery_Charge_Stored` | 1824 | 1368 | 2850 |
| `PV_AC_Export` | 0 | 500 | 0 |
| `Battery_Energy` | 6824 | 6368 | 7850 |

**AC3 — grid charge at night.** PV 0, load 500, initial energy 2000,
`grid_target_fraction = 0.4` (4000 Wh), discharge not allowed. DC coupling
uses `grid_charge_efficiency = 0.96`; both topologies give
`Grid_AC_To_Battery` 2192.982456, `Grid_Charge_Conversion_Loss` 87.719298,
`Battery_Charge_Input` 2105.263158, `Battery_Charge_Stored` 2000,
`Import_From_Grid` 2692.982456 and `Battery_Energy` 4000.

**AC4 — saturated PV inverter and grid charge.** PV inverter 2000 W, PV DC
2500, load 2000, initial energy 2000, the AC3 target, discharge not allowed.

| Column | AC | DC |
|---|---|---|
| `PV_AC_To_Load` | 2000 | 2000 |
| `PV_DC_Curtailed` | 416.666667 | 0 |
| `PV_DC_To_Battery` | 0 | 416.666667 |
| `Grid_AC_To_Battery` | 2192.982456 | 0 |
| `Import_From_Grid` | 2192.982456 | 0 |
| `Battery_Energy` | 4000 | 2395.833333 |

DC coupling stores the clipped DC and has no inverter headroom left for the
grid (A6); AC coupling clips the PV and grid charges through its own
inverter.

**AC5 — battery inverter rating limits discharge.** PV 0, load 3000,
initial energy 5000, battery inverter 2000 W. `Battery_AC_To_Load` 2000,
`Battery_Inverter_Loss` 83.333333, `Battery_Discharge_DC` 2192.982456,
`Battery_Discharge_Loss` 109.649123, `Import_From_Grid` 1000,
`Battery_Energy` 2807.017544. A DC-coupled system with a 2000 W hybrid
inverter and no PV gives the same numbers.

**AC6 — part load.** PV 0, load 300, initial energy 5000, battery inverter
3000 W. `Battery_AC_To_Load` 300 (to 1e-12), DC from the battery 324.751952,
`Battery_Inverter_Loss` 24.751952, `Battery_Discharge_DC` 341.844160,
`Battery_Energy` 4658.155840.

**Invariants**, tested on App-sized cases at hourly and 15-minute steps,
Python and Numba:

- P1. In any AC-coupled run, `PV_DC_Curtailed`, `PV_DC_To_Inverter` and
  `PV_Direct_Inverter_Loss` equal the PV-only run's bit for bit, and
  `PV_AC_To_Load + PV_AC_Export + PV_AC_To_Battery` equals its AC output to
  1e-9 relative.
- P2. An AC-coupled config with `battery_kwh = 0` gives the PV-only results
  bit for bit.
- P3. An AC-coupled battery that does not move, held full at `max_soc`
  with discharge not allowed and no standby loss, gives the PV-only flows bit
  for bit.
- P4. A retired AC-coupled pack's steps equal the PV-only run bit for bit.
- P5. DC-coupled App goldens, golden outputs and the parity harness are
  unchanged; Python and Numba agree bit for bit for both topologies.
- P6. Every ledger identity above holds to the existing tolerances, the
  origins reconcile, and no step both charges and discharges.

## Alternatives considered

- **Restore `dc_coupled` as a boolean.** Rejected: 0.7.0 removed it, and a
  string leaves room for another topology without a second flag.
- **Flat battery inverter efficiencies, one per direction.** Simple and close
  to what some tools do, but DC coupling already discharges through the
  part-load curve. Two converter models would mix model differences into a
  topology comparison. It remains a possible amendment (open question 1).
- **Convert PV inside the AC step, per step.** Rejected: the vectorised and
  scalar conversions are not promised identical to the last bit, so P1, P3
  and P4 would hold only to rounding, and the PV chain memo could not serve
  battery runs.
- **A second day loop for AC coupling.** Rejected (C10).
- **Keep `grid_charge_efficiency` as an extra loss in AC coupling.** Rejected:
  it would count the battery inverter's conversion twice.
- **Coupling as an optimizer decision variable.** Deferred: it doubles the
  search for a question two studies answer, and the battery inverter keys
  only exist for one value.

## Implementation plan and effort

Five PRs, about 11 working days for one developer, plus review and the
platform CI runs each kernel change needs. A sixth, optional PR adds the
performance steps. Once the work ships, this section is replaced by an
implementation status, as in ADR 0002.

| PR | Scope | Effort | Main risk |
|---|---|---|---|
| 1 | `BatteryConfig.coupling` and the battery inverter fields with validation (DC refuses AC fields); the two ledger columns written as zeros; ledger schema 3.1; conservation helper with the new identities. No behaviour change. | 1.5 days | Golden or schema tests that list columns. |
| 2 | `_dispatch_ac_step`, the day-loop branch, PV chain arrays in `_day_arguments` and `_simulate_core`, Numba registration, the PV origin input; reference cases AC1–AC6, invariants P1–P6, parity harness scenarios. | 4 days | DC bit identity, and Numba cold-compile time with a second step function. |
| 3 | App, Monte Carlo and optimizer keys; rating resolution; smart-charging rules for `grid_charge_efficiency`; planner evaluator and refill price; retirement test; provenance; regenerated config docs and golden provenance. | 2.5 days | Planner and controller paths that build `BatteryConfig` from a dict. |
| 4 | Costs: `battery_inverter_cost_per_kw`, PV inverter at the simple rate, `battery_inverter_cost`, optimizer CAPEX per candidate. | 1.5 days | A missing price must fail early, not price zero. |
| 5 | Docs: energy-balance and battery pages, configuration guide, results guide (waterfall block, year-row diagnostics), CHANGELOG; ADR status. | 1.5 days | None significant. |
| 6 (optional) | PV chain memo for AC-coupled battery runs in Monte Carlo; PV chain hoisted across battery capacities in the optimizer. | 1 day | Memo key must include everything the PV inverter depends on. |

Risks across the work:

- **DC bit identity.** The day loop's signature and the origin input change.
  `_dispatch_dc_step` stays untouched and every gate in C10 is run on PR 2.
- **Performance.** The Python backend gains one branch and a few arguments
  per step; the Numba kernel compiles both steps. Measure with
  `tools/benchmark_montecarlo.py` and the cold-compile time before and after
  PR 2.
- **Model validity.** The part-load curve in the charging direction is not
  validated against measured battery inverters. The docs must say so.
- **Cost data.** There is no sourced battery inverter price for the presets.

## Open questions for the maintainer

1. Battery inverter model: the PVWatts curve in both directions (proposed),
   or flat charge and discharge efficiencies?
2. `grid_charge_efficiency` with AC coupling: `None` in the instructions and
   the key refused (proposed), or a fixed 1.0 that the AC step ignores?
3. `ac_output_scale` with AC coupling: PV inverter only (proposed), or both
   inverters?
4. Battery inverter rating: both the absolute and the per-kWh key
   (proposed), or one? Should App derive a default from
   `battery_power_limit_c_rate` instead of requiring one?
5. `battery_inverter_cost_per_kw`: required with no default (proposed), or a
   preset value, and from which source?
6. Units that sell pack and inverter together: should a battery replacement
   in AC coupling be able to include the battery inverter, and should App
   gain an explicit replacement price as the optimizer has?
7. Inverter replacement and inverter standby consumption are not modelled
   for either topology. Are they out of scope for this work?
8. Target release.

## Consequences

- DC-coupled results, configs and costs are unchanged; resolved-config
  provenance gains the coupling field.
- An AC-coupled system can be simulated, priced and optimized through the
  same `breos.App` facade, with every flow in the same ledger and the same
  closure tests.
- The ledger schema moves to 3.1 with two additive columns; the result
  format stays "1".
- A6's summed-throughput rule keeps its scope, the hybrid inverter. AC
  coupling applies it to the battery inverter alone.
