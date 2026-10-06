# 0005 — Vehicle-to-home: the vehicle as a second, part-time AC-coupled store

- **Status:** Proposed. No code yet; implementation is planned for 0.8.0
  after the AC-coupled battery work (#419), once the maintainer accepts or
  amends the decisions below (#420).
- **Date:** 2026-10-06

## Context

BREOS has no electric vehicle model. A household with a vehicle charges it at
home, and its charging demand is often larger than the rest of the house's
consumption. With a bidirectional charger the vehicle's battery can also
serve the house while it is plugged in: vehicle-to-home (V2H). It charges
from PV or the grid, and discharges to the house load through the charger.
ADR 0002 lists V2H as outside 0.7.0, and the 0.7.0 design note repeats it
([Outside 0.7.0](../architecture/tariffs-and-smart-charging.md#outside-070)).

A bidirectional charger is a converter between the house AC wiring and a
battery, as the battery inverter of ADR 0004 is. Seen from the house, a
plugged-in vehicle is an AC-coupled battery whose availability is limited:
it is present on some steps, absent on others, and it comes back with less
energy than it left with. #420 asks for a record of the decisions first.
Its contract:

- vehicle inputs: plug-in availability per day (arrival and departure times
  or a schedule file), battery capacity, the required state of charge at
  departure, and the energy used while away (as the state of charge at
  arrival or as kWh driven);
- charger inputs: bidirectional power rating, efficiency and price; smart
  charging without discharge (V1G) is a mode of the same model;
- the vehicle's charging demand is served in every case, and the benefit of
  V2H is measured against a household that still charges the vehicle;
- the dispatch order with a stationary battery, and the interaction with
  `[smart_charging]` and time-of-use tariffs;
- the vehicle battery's degradation, and how wear from home use is separated
  from wear from driving in the economics;
- energy closure per step, with the vehicle away counted as neither load nor
  storage;
- default behaviour unchanged and bit-identical; Python and Numba agree;
- results with the energy to and from the vehicle, unmet departure targets
  and the cost of extra wear.

This record answers each point or lists it as an open decision. It changes
no code. Its options carry a recommendation; the maintainer's answers
will replace the open decisions, as they did for ADR 0004.

## Current behaviour this record relies on

Line numbers are at `567df2a` (`origin/develop`, 2026-10-06). ADR 0004 is
accepted and not implemented; where this record relies on it, it cites its
sections (C1–C14), not code.

**One store per site.** `BatteryConfig` (`breos/battery.py:94-322`) describes
one stationary pack. `_simulate_core` (`breos/battery.py:1725-2241`) runs it
day by day. A run without a usable pack takes the vectorised
`_dispatch_no_battery_vectorized` (`:1197-1263`, branch at `:2087-2105`),
which never enters the day loop. With a pack, each positional day window
(`:2120-2230`) dispatches through `_dispatch_day` and then closes health in
Python (`_apply_daily_degradation`, `:1266-1437`, called at `:2195`).

**The step.** `_dispatch_dc_step` (`breos/_dispatch.py:520-711`) takes PV DC
and load in Wh, serves the load from PV, charges from the surplus, exports
the rest (`:579-593`), or discharges toward the deficit (`:594-671`). Grid
charging runs after PV, only when no PV is exported and the pack has not
discharged, within what PV left of each shared limit, including the site
import limit (`:674-692`). The step returns the stored energy and fifteen
flows; nothing after it in the day loop changes a flow.

**Conversion cores.** `_dc_ac` (`breos/_dispatch.py:43-103`) and its inverse
`_dc_for_ac` (`:106-141`) are the PVWatts part-load conversion. ADR 0004 C4
uses them for a battery inverter in both directions with a rating `P_wh` per
step and a nominal efficiency `η_b`; at full load the curve returns exactly
`η_b × P_wh`.

**Day loop and ledger.** `_dispatch_day` (`breos/_dispatch.py:714-948`)
applies the capacity window and standby (`_apply_capacity_window`,
`:360-435`, with the LFP temperature factor `lfp_capacity_factor`,
`:296-318`), takes one origin fraction per step (`:811-819`), refuses a step
that both charged and discharged (`:858-859`), books origins (`:860-873`) and
writes every row (`:899-948`). `_day_arguments` (`:951-1019`) packs both
backends' arguments with one float signature. `_LEDGER_COLUMNS`
(`:192-233`) is the ledger; `LEDGER_SCHEMA_VERSION` is `"3.0"` (`:164`),
and ADR 0004 C8 moves it to `"3.1"`. `tests/energy_conservation.py:60-150`
asserts the per-step identities and `:166-202` the three origins.

**Numba.** `breos/_numba_dispatch_kernels.py` compiles `_dispatch_day` and
its scalar helpers with `fastmath=False`
([Numba dispatch backend](../architecture/numba-dispatch-backend.md)).
Degradation, replacement and every state change between days stay in Python.

**Degradation.** `DegradationLifecycle` (`breos/degradation/protocol.py`)
is the boundary between dispatch and an ageing engine. A day close passes it
the window's absolute SOC and cell temperature (`breos/battery.py:1321-1332`).
The native engine is the Naumann/Lam LFP model; BLAST offers cell models,
NMC, NCA and LMO among them (`breos/degradation/profiles.py:121-369`). The
policy note says a BLAST key names one empirical cell model, not a pack
([Battery degradation policy](../architecture/battery-degradation-policy.md)).

**Load and the no-system baseline.** App reads one household load
(`load_consumption_profile`, `breos/app_inputs.py:489-506`). The no-system
household buys that whole load from the grid: `Baseline_Import_Cost` is
`Houseload` at the import prices (`breos/projection.py:361-377`). Avoided
emissions use net exchange (ADR 0002 A10; `breos/emissions.py:61-95`).
Monte Carlo scales the load by a sampled factor
(`breos/montecarlo.py:136`, `:612-626`).

**Smart charging.** `DispatchInstructions` (`breos/dispatch_instructions.py`)
gate the stationary pack per step. `daily_persistence` plans with the
canonical day loop (`_DayEvaluator`, `breos/_daily_targets.py:361`) and is
refused by Monte Carlo (`breos/montecarlo.py:429`).

**Terminal credit.** ADR 0003 E10 credits the final pack's health above the
end-of-life threshold, `f = clip((h − h*) / (1 − h*), 0, 1)`, priced at the
replacement price escalated to `T` and discounted from it
(`terminal_health_credit`, `breos/economics.py:539`).

## Regulatory setting

The model needs no regulatory input to run. The rules below decide what a
study can express with it and what it leaves out. Each was checked on
2026-10-06; an item marked "not verified" is stated as an assumption.

**Germany.**

- **§14a EnWG.** Under the Bundesnetzagentur determinations BK6-22-300 (27
  November 2023) and BK8-22/010-A (23 November 2023), a non-public charging
  point or a storage unit above 4.2 kW commissioned from 1 January 2024 is a
  controllable consumption device. The network operator may reduce its power
  in a grid emergency. In return the household gets a lower network charge:
  Modul 1, a flat annual reduction that depends on the network area;
  Modul 2, a 60 % reduction of the network working price for a separately
  metered device; or, from 1 April 2025 and only with Modul 1, Modul 3,
  time-variable network charges. A study expresses Modul 3 as a custom
  tariff schedule and Modul 1 as an `annual_network_credit` (ADR 0002 A15,
  whose example is Stromnetz Berlin's Modul 1). Modul 2 needs a second
  meter for the charger, which BREOS's one-meter site does not have, and the
  operator's power reduction is not modelled (V7).
- **Electricity tax.** Since 1 January 2026, §5a(3) StromStG (BGBl. 2025 I
  Nr. 340) says that electricity a vehicle feeds back through bidirectional
  charging and that is used at the charging point's site, without the public
  grid, bears no electricity tax. That is V2H. BREOS prices only the grid
  import and export the user's prices apply to, and V2H discharge to the
  house is neither, so the rule needs nothing in the model; it confirms that
  V2H energy is not taxed a second time.
- **Network charges on energy fed back.** Since 23 December 2025, §118(6)
  EnWG with §21 EnFG (BGBl. 2025 I Nr. 347) treats charging points as storage
  and exempts grid energy that is fed back into the grid from network
  charges, in proportion to what is fed back. It concerns vehicle-to-grid.
  V2H feeds nothing back, so it does not apply; this reading is ours, not
  a statement of the act.
- **EEG.** The Solarspitzengesetz (BGBl. 2025 I Nr. 51, in force 25 February
  2025) extends the EEG's metering options for mixed storage to charging
  points (§19(3) EEG). They concern payment for exported stored PV energy,
  which V2H never exports.

**Portugal.**

- **Electric mobility.** Decreto-Lei n.º 93/2025, de 14 de agosto, replaces
  Decreto-Lei n.º 39/2010 and defines bidirectional charging, framed as
  vehicle-to-grid. We found no rule specific to V2H behind a household
  meter. This record treats home charging and V2H as consumption and storage
  behind the household's meter, as BREOS treats the stationary battery under
  the self-consumption regime of Decreto-Lei n.º 15/2022. That is an
  assumption. The full text of Decreto-Lei n.º 93/2025 was not read for this
  record; the summary rests on secondary sources.
- **Self-consumption with a charging point.** Article 7 of the ERSE
  self-consumption regulation (Regulamento n.º 815/2023) admits charging
  points of the public electric mobility network to self-consumption only
  through pilot projects. A private home charger is not such a point; not
  verified whether Decreto-Lei n.º 93/2025 changes this.

**Hardware.** V2H needs a vehicle and a charger that both support
bidirectional power transfer, for example under ISO 15118-20:2022 or
CHAdeMO. BREOS takes that support as given by the user's configuration and
models no protocol.

## Decision

### V1. Scope

- One vehicle per site. Its battery is a second store beside the optional
  stationary battery, connected to the house AC bus through its own
  bidirectional charger.
- **V2H only.** The vehicle discharges to the house load and never exports,
  as the stationary battery never does (ADR 0004 C1). Vehicle-to-grid
  (export from the vehicle, or market services) is out of scope.
- Charging and driving away from home are outside the site. The energy a
  trip uses is the net energy the vehicle's battery loses between departure
  and arrival, whatever charged it on the way.
- The model applies to a DC bidirectional charger (the converter sits in the
  wall unit) and to an AC one (the converter is the vehicle's on-board
  charger) alike: in both, one bidirectional converter sits between the AC
  bus and the vehicle battery. BREOS does not model the difference between
  them.

### V2. Configuration shape

A new top-level App table, `[vehicle]`, and a lower-level `VehicleConfig`
beside `BatteryConfig`. Without the table nothing changes (V12).

```toml
[vehicle]
mode = "v2h"                      # "uncontrolled", "v1g" or "v2h"
battery_kwh = 60                  # nominal capacity
min_soc = 0.10                    # lowest state of charge the vehicle allows
max_soc = 1.00
departure_soc = 0.80              # required at every departure
v2h_min_soc = 0.40                # V2H never discharges below this
charge_efficiency = 0.95          # battery, DC in to stored
discharge_efficiency = 0.95       # battery, stored to DC out
charger_ac_rating_kw = 7.4        # bidirectional, AC side
charger_efficiency = 0.94         # nominal, no default
charge_periods = ["off_peak"]     # optional; needs [tariff]
discharge_periods = ["peak"]      # optional; needs [tariff]; "v2h" only
reference_mode = "v1g"            # the household the benefit is measured against
charger_cost = 2500               # extra cost of this charger (V10)
battery_cost_per_kwh = 120        # values the vehicle pack's wear (V10)

[vehicle.schedule]
weekday = { arrive = "18:30", depart = "07:45", trip_kwh = 9.0 }
weekend = { arrive = "14:00", depart = "10:00", trip_kwh = 12.0 }
# or: file = "trips.csv"  (departure, arrival, trip_kwh per row)
```

- **Modes.** `"uncontrolled"` charges at the charger's full rating from
  arrival until the departure target, from whatever the AC bus has: PV
  surplus first, then the grid. `"v1g"` charges from PV surplus whenever it
  can, from the grid in `charge_periods`, and from the grid as late as the
  departure target allows (V5); it never discharges. `"v2h"` is `"v1g"` plus
  discharge to the house load (V4, V5). The values are strings, so a later
  mode such as vehicle-to-grid adds one.
- **State of charge** is the fraction of the vehicle battery's current
  capacity (nominal times its health), as a vehicle displays it, not a
  fraction of the usable window. `0 ≤ min_soc ≤ v2h_min_soc ≤ max_soc ≤ 1`
  and `min_soc ≤ departure_soc ≤ max_soc`. `v2h_min_soc` is refused outside
  `"v2h"`; its default is `min_soc`.
- **No defaults for equipment facts.** `battery_kwh`, `departure_soc`,
  `charger_ac_rating_kw` and `charger_efficiency` are required. BREOS has no
  sourced value for any of them, and a charger efficiency is not the PV
  inverter's 0.96, as ADR 0004 C2 says of the battery inverter. The battery
  efficiencies default to the stationary battery's (`DEFAULT_CHARGE_EFFICIENCY`,
  `DEFAULT_DISCHARGE_EFFICIENCY`).
- **Rating units** follow ADR 0004 C4: the rating is AC, the charger's
  input when charging and its output when discharging, and it enters every
  per-step limit as `P_wh = P × hours_per_step`.
- The lower level takes `VehicleConfig` and a resolved `VehicleSchedule`
  (V3) through new optional `_simulate_core` arguments, as instructions and
  controllers arrive today.

### V3. Plug-in schedule and trips

**Input forms.** Options:

- (a) Arrival and departure times for weekdays and weekend days, each with
  the trip energy, plus an optional file of trips.
- (b) A per-step plugged-in mask from a file.
- (c) Statistical trip generation (sampled arrival, departure and distance).

Recommendation: (a). Times per day type cover the common study, and the
file covers measured logbooks. A mask (b) cannot carry the energy of each
absence, which the trip needs, and generation (c) is a behaviour model BREOS
has no data for; a study can generate a trip file itself.

- **Trip file.** One row per absence: `departure`, `arrival` (timestamps
  with an offset or in the location's timezone, as load files are read) and
  `trip_kwh`. Rows must not overlap. Within the simulated year, a file
  replaces the weekday and weekend rules; it is replayed every project year
  like the tariff calendar (ADR 0002 A2).
- **Civil time.** Times are local wall-clock times in the configured
  timezone, resolved as a tariff schedule is (ADR 0002 A1), never from the
  index's own offset. A day type is the configured-zone civil date's
  weekday; holidays are not modelled.
- **Steps.** A step is plugged in only if the vehicle is home for the whole
  step. An arrival rounds up to the next step boundary and a departure rounds
  down. At hourly steps an arrival at 18:30 plugs in at 19:00 and a
  departure at 07:45 leaves at 07:00. Options: this conservative rounding,
  or ADR 0002 A3's strict policy, which refuses a time off the step grid.
  Recommendation: conservative rounding, recorded in provenance. A schedule
  is behaviour data, not a regulatory boundary; refusing 07:45 at hourly
  steps would force invented times. The rounding never gives the vehicle more
  time at home than the schedule says.
- **Resolution.** The schedule resolves once, on the simulated index, to
  per-step arrays: `plugged` (bool), `steps_to_departure` (the number of
  plugged steps left before the next departure, this one included; 0 when
  away), and `trip_wh` at each departure step. They are hashed for
  provenance as instructions are. Every project year replays them, and a
  plug-in that crosses the year end continues at the head of the year
  (A2's seam). A `[period]` run takes the next departure after its end from
  the same rules.

**Energy while away.** Options:

- (a) Trip energy in kWh per absence, at the battery (DC, out of the store).
- (b) The state of charge at arrival.

(b) makes the trip's energy depend on the departure state: a vehicle that
leaves fuller after V1G drives the same distance but would lose more. The
difference between two households would then include energy that no trip
used. Recommendation: (a) is the model. (b) is accepted as input and
converted to (a) against the departure target:
`trip_wh = (departure_soc − arrival_soc) × capacity`, at the vehicle's
current capacity. With every target met, both forms give the same run.

### V4. Dispatch order

Each step dispatches the stationary battery first, exactly as today, and
the vehicle second, on the AC flows the stationary step left:

```text
1. stationary step (_dispatch_dc_step, or ADR 0004's AC step, or none)
   -> pv_ac_export, grid_import, grid_ac (stationary grid charge)
2. vehicle step on the AC residual:
   surplus      = pv_ac_export                  # PV the house would export
   deficit      = grid_import - grid_ac         # load the house would import
   site_room    = max(0, grid_import_cap_wh - grid_import)   # load import is never cut
   -> PV_AC_To_Vehicle, Grid_AC_To_Vehicle, Vehicle_AC_To_Load
   PV_AC_Export    = pv_ac_export - PV_AC_To_Vehicle
   Import_From_Grid = grid_import - Vehicle_AC_To_Load + Grid_AC_To_Vehicle
```

Options for the order:

- (a) Stationary battery first, vehicle second (recommended).
- (b) Vehicle first, stationary battery second.
- (c) One joint step that shares PV surplus and deficit between the two by
  a rule (for example in proportion to their power ratings).

(a) leaves the stationary step untouched: it reads nothing the vehicle
writes, so every stationary flow is the same with and without a vehicle
(invariant Q2). Stationary cycles are cheaper than vehicle cycles, and the
vehicle is absent when PV is strongest on working days, so using the
stationary battery first also reflects how home energy managers are usually
set up. That is an assumption about controllers, not a measured fact. (b)
cycles the vehicle more, and would change the stationary flows of every
run with a vehicle. (c) needs a sharing rule BREOS has no basis for, and
mixes both stores' limits in one step. A priority option can be added later
as an amendment.

**No storage-to-storage transfer.** The stationary battery never charges the
vehicle, and the vehicle never charges the stationary battery or exports.
Order (a) gives this without a new rule: the stationary step sees only the
house load, and the vehicle's discharge serves only `deficit`, which
excludes the stationary battery's grid charge. Energy moved from one store
to the other would pay two conversions and two battery efficiencies for no
service.

**Which PV inverter.** With a DC-coupled stationary battery, the vehicle
takes PV AC after the hybrid inverter. With an AC-coupled one, or none, the
house side is ADR 0004's AC path: the PV inverter's output comes from the PV
chain arrays computed before the day loop (ADR 0004 C3). A run with a
vehicle and no stationary battery therefore enters the day loop with the
AC step and an absent battery, whose house flows equal the PV-only run's
bit for bit (ADR 0004 P2, P3).

### V5. The vehicle step and the departure target

`_dispatch_vehicle_step`, in `breos/_dispatch.py` beside the other steps,
dispatches one plugged-in step in Wh. Let `cap_v` be the vehicle's nominal
capacity times its health, `E` its stored energy, `T = departure_soc ×
cap_v`, `F = max(v2h_min_soc, min_soc) × cap_v`, `emax_v = max_soc × cap_v`,
`room = emax_v − E` (`T − E` for `"uncontrolled"`), `k = steps_to_departure`,
and `s_max = charge_forward(P_cap) × eff_c`, the energy one full-power step
stores, with `P_cap = min(P_wh, grid_import_cap_wh)` (`P_wh` without a site
limit). `charge_forward` and its inverse are ADR 0004 C4's charging
conversion with the charger's rating and efficiency; `discharge_forward` and
`discharge_inverse` are its discharging conversion:

```text
need      = max(0, T - E)
mandatory = min(s_max, max(0, need - (k - 1) * s_max))   # "v1g", "v2h"
mandatory = min(s_max, need)                             # "uncontrolled"

# 1. PV surplus ("v1g", "v2h": up to emax_v; "uncontrolled": up to T)
pv_ac  = min(surplus, charge_inverse(room / eff_c), P_wh)
# 2. grid: what step 1 left of the mandatory charge, then, in a
#    charge period, toward T
grid_ac = min(grid AC for the rest of the mandatory or period charge,
              P_wh - pv_ac, site_room)
#    PV and grid AC share one charger operating point (ADR 0004 C7)
# 3. discharge ("v2h", no charge this step, discharge period or none set)
d_max   = min(E - F, E - T + (k - 1) * s_max)      # stored energy it may give
ac_out  = min(deficit, P_wh)
dc_out  = min(d_max * eff_d, discharge_inverse(ac_out))   # DC into the charger
draw    = dc_out / eff_d                                  # Vehicle_Discharge_DC
#    a binding guard lands E exactly on T - (k - 1) * s_max
```

- **Deadline guard.** The departure target is a hard constraint. `mandatory`
  is the least energy that must be stored now so that full-power charging on
  the remaining plugged steps still reaches `T`: "as late as possible"
  charging. V2H discharges only while the target stays reachable after the
  discharge (`d_max`). A step whose mandatory charge reaches `T` lands on it
  exactly, as `hold_target` lands a reached target (ADR 0002 A14), and a
  discharge the guard binds lands exactly on `T − (k − 1) × s_max`, so a
  rounding residue cannot leave the last step one ulp short. A departure
  counts as short only when it misses `T` by more than 1e-9 of `cap_v`.
- **Site limit.** With `grid_import_limit_w` set, `s_max` uses the smaller
  of the charger rating and the limit. The house load can still take part
  of the import room on a later step, which the guard cannot know without
  reading future load. A departure that V1G or V2H leaves short while the
  uncontrolled reference would not is therefore possible under a binding
  site limit; the result reports it (V8), and Q5 holds only without one.
- **Why a guard and not a planner.** The guard uses only the schedule, which
  the household knows, and the vehicle's own state; it never reads future PV,
  load or prices. It is causal, cheap and the same in both backends. A
  planner that also chooses when to charge from forecasts is a later mode.
- **Grid charging in charge periods** is opportunistic: toward `T`, never
  above it, so cheap grid energy does not fill the vehicle beyond what the
  next trip needs. PV may fill it to `max_soc`.
- **Unmet targets.** A target that cannot be reached (the vehicle arrived
  too empty, stayed too briefly, or the site limit bound) is not refused. The
  vehicle charges at full power and leaves short; the shortfall is reported
  (V11). Options: report it (recommended), refuse the run, or buy the
  shortfall at an outside charging price. Refusing would make a whole study
  fail on one late arrival. An outside price is a study assumption, so it is
  an optional key, `vehicle.outside_charging_price`, with no default: set, it
  prices unserved trip energy in every household (R0, R1, R2) as a cost of
  that household, so households that drive the same trips stay comparable.
  Unset, the shortfall and unserved trip energy are reported only, and the
  V2H benefit is flagged as not comparable when R2 leaves more driving
  unserved than R1 (V8).
- **One direction per step,** as A8 requires: a step that charges does not
  discharge. The guard gives `d_max ≤ 0` whenever `mandatory > 0`.
- **Limits.** `P_wh` bounds charger AC in and AC out; `site_room` bounds
  vehicle grid charging, so `grid_import_limit_w` counts the vehicle too.
  The vehicle has no capacity-temperature derating, standby loss or
  self-heating model: `lfp_capacity_factor` describes an LFP stationary pack
  in its enclosure, and a vehicle's thermal management is outside BREOS. This
  is an assumption, and the docs state it.

### V6. Departure, absence and arrival

- **Departure** happens at the start of the first step that is not plugged
  in. The vehicle's stored energy `E_dep` leaves the site: the ledger books
  `Vehicle_Departure_Energy = E_dep`, and the in-site stored energy becomes 0.
  On board, `trip_wh` is removed at once. If it exceeds the usable energy
  `E_dep − min_soc × cap_v`, the on-board energy stops at `min_soc × cap_v`
  and the rest is `Vehicle_Trip_Unserved`.
- **Away,** the vehicle is neither load nor storage: every vehicle flow is
  0 and the in-site stored energy is 0. The on-board energy is reported as a
  state, for the ageing model and the reader, and does not enter any balance.
- **Arrival** is the first plugged-in step after an absence. The on-board
  energy enters the site: `Vehicle_Arrival_Energy`, booked before that
  step's dispatch.
- **Timing of the trip energy.** Removing it at departure means the ageing
  model sees the arrival state of charge during the absence. Options: at
  departure (recommended), at arrival, or spread linearly over the absence.
  The linear form is closest to driving but invents a time profile the
  input does not have; the two step forms bracket it. At departure is chosen
  so that an absence that ends after the simulated year needs no trip
  energy from the next year.
- **Origins.** The vehicle's stored energy has the three origins of ADR 0002
  A8. A trip removes them in proportion; the arrival energy keeps the
  departure shares. PV-origin energy used by trips is reported as PV used
  for driving.
- **Start.** The vehicle starts the project at its departure target,
  unattributed, plugged in or away as the schedule says at the first step.
  Between project years the on-board energy, its origins and its health
  carry over as the stationary battery's do (`physical_carry`).

### V7. Smart charging and tariffs

- `[smart_charging]` keeps its meaning and gates the stationary battery
  only. The vehicle's own `charge_periods` and `discharge_periods` select
  tariff periods, need a `[tariff]`, must be disjoint, and are refused with
  `"uncontrolled"`; `discharge_periods` is refused with `"v1g"`. Without
  them, `"v1g"` charges from the grid only as the guard requires, and
  `"v2h"` may discharge on any step.
- **Greedy V2H under flat prices.** With no `discharge_periods`, V2H
  discharges whenever the guard allows and buys the energy back before
  departure. Under a flat price that loses the round trip; reference case
  VH3 shows it. The comparison with the reference household (V8) reports
  that loss rather than hiding it, and the docs say to set
  `discharge_periods` for time-of-use studies.
- **`daily_persistence`** is refused with a `[vehicle]` table in 0.8.0. The
  planner's day loop would model the stationary battery exactly, since the
  vehicle acts after it, but its cost objective would ignore the vehicle's
  effect on import. Lifting the refusal needs the planner to carry the
  vehicle; an amendment can do it.
- **Germany, §14a EnWG.** A study expresses a Modul 3 time-variable network
  charge as a custom schedule and the Modul 1 reduction as an
  `annual_network_credit` (ADR 0002 A15). The operator's right to reduce the
  charging power in a grid emergency is not modelled.

### V8. The reference household and the comparison

Three households are defined, all with the same vehicle and trips:

| Household | System | Vehicle charging | Used for |
|---|---|---|---|
| R0, no system | none | `"uncontrolled"`, from the grid | NPV baseline, emissions baseline |
| R1, reference | the configured PV and battery | `reference_mode` | V2H benefit |
| R2, study | the configured PV and battery | `mode` | every reported result |

- **R0** extends today's no-system household: it buys the house load and the
  uncontrolled vehicle charging from the grid. `Baseline_Import_Cost` prices
  both, so a system's NPV is not charged for owning a vehicle. R0's vehicle
  charging comes from the vehicle step on a site with no PV and no battery,
  simulated once per project year with the vehicle's ageing.
- **R1** is a second projection of the configured system with
  `vehicle.mode = reference_mode`. Options for `reference_mode`: (a) a key
  that defaults to `"v1g"` for `"v2h"` and to `"uncontrolled"` for `"v1g"`,
  with `"none"` to skip R1 (recommended); (b) always `"uncontrolled"`;
  (c) always `"v1g"`. (a) isolates the value of discharge for V2H by
  default, which is #420's question, lets a study measure V1G against
  uncontrolled charging, and lets a sweep skip the second run. (b) would mix
  the value of smart charging into the value of V2H.
- **Benefit.** The V2H benefit is R1's discounted cost less R2's, less the
  extra wear of V9 and the extra charger cost of V10's CAPEX. It is reported
  with its parts. R1 doubles the dispatch time of a run; App runs both. A
  `reference_mode` equal to `mode` is refused.
- **Same driving.** The benefit is only meaningful if both households drive
  the same trips. R1 and R2 report their unserved trip energy and departure
  shortfalls. With `vehicle.outside_charging_price` set, unserved trip energy
  is a cost in each household and the benefit includes the difference (V5).
  Without it, a benefit where R2 leaves more trip energy unserved than R1 is
  reported with `comparable = false`, and the docs say why.

### V9. Vehicle battery degradation and wear from home use

**Model.** Options:

- (a) A second degradation lifecycle for the vehicle pack, stepped at the
  same positional day windows as the stationary pack, on the vehicle's
  on-board state of charge, trips included, and its temperature. The engine
  is chosen as for the stationary battery: native by default, BLAST by key.
- (b) No ageing model; a wear price per kWh of V2H discharge, as the
  planner's `wear_cost_per_kwh` (ADR 0002 A17).
- (c) No wear.

Recommendation: (a). It reuses the `DegradationLifecycle` boundary, runs in
Python between day windows as the stationary pack's does, and so leaves the
kernel's ageing-free contract intact. It captures calendar ageing, which
V2H can lower as well as raise by changing the average state of charge;
(b) cannot. Limits stated in the docs: the native model is calibrated on LFP
cells, many vehicles use NMC or NCA, and a BLAST key names one cell model,
not a vehicle pack (battery degradation policy). The vehicle has its own
`degradation_engine`, `blast_model` and `calendar_model` keys, defaulting to
the stationary battery's defaults.

**Separating home use from driving.** Driving wear and V2H wear act on the
same cells at the same time, and a cycle-counting model has no additive
split. The record therefore uses the counterfactual: R1 and R2 have the same
trips, so the wear from home use is the difference between the vehicle's
health at the end of R1 and of R2. It includes every change V2H makes to the
vehicle's state of charge, charging as well as discharging. It can be
negative, when V2H keeps the pack at a lower average state of charge.
Options: this difference (recommended), or a throughput share (V2H discharge
over all discharge, times total wear), which assumes every kWh wears the
same and ignores calendar ageing.

**Temperature.** `vehicle.battery_temperature` takes `"weather"`, a fixed
value or a CSV, as `battery_temperature` does, without the indoor model.
Default `"weather"`. A vehicle's thermal management keeps its cells warmer
in winter and cooler in summer than the air; this is an assumption, stated.

**Lifetime.** One vehicle pack serves the whole project; it is not replaced,
and its fading capacity limits dispatch and lowers `T` (A7's rule, applied
to the vehicle). Options: one pack for the horizon (recommended for 0.8.0),
or a `vehicle_holding_years` after which a new vehicle arrives at full
health. A real household changes vehicles; a study can set the projection
horizon to the holding period meanwhile. The holding period can follow as an
amendment.

### V10. Costs and wear valuation

- **Charger.** `vehicle.charger_cost` is the extra cost of the configured
  charger over the standard charger every household is assumed to own
  already, R0 included, since every household charges the vehicle. It is
  required with `"v2h"`, has no default and no preset (BREOS has no sourced
  price), and an explicit 0 is valid. It joins the system's CAPEX.
  `"uncontrolled"` and `"v1g"` accept it too, so a study can price a smart
  charger's premium; there, omitted means 0, because a standard charger is
  assumed to support controlled charging. That is an assumption, stated in
  the docs. Options: an incremental price (recommended), or
  two prices (configured and standard charger). The NPV uses only their
  difference, so two keys would be two required inputs for one number. The
  risk of one key is reading it as the full price, so its documentation
  says "extra cost over a standard charger".
- **Wear.** With `h` the vehicle pack's final health in each household and
  `h*` the vehicle's `eol_percentage` (default the stationary default, 0.70),
  the extra wear is `C_v × (h_R1 − h_R2) / (1 − h*)`, where
  `C_v = battery_cost_per_kwh × battery_kwh`: the share of the pack's useful
  health range that home use consumed, at the pack's price, as ADR 0003 E10
  values health. It is booked at `t = T`, escalated and discounted as E10's
  credit is. Unlike E10's credit it is not clipped at `h*`: a single pack
  kept for the whole project can fall below its threshold in both
  households, and a clipped value would then read 0 however much more V2H
  aged it. It is negative when V2H ages the pack less.
  `battery_cost_per_kwh` is required with `"v2h"`, with no default.
- **Where the wear goes.** Options: (a) reported beside the unadjusted NPV
  and inside the V2H benefit (recommended); (b) inside the system's NPV. (a)
  follows E10: the NPV keeps its meaning (cash flows of the system), and the
  vehicle's value belongs to the household, not the system. The V2H benefit
  is where the wear matters, and it includes it.
- No vehicle O&M, no charger replacement and no charger standby consumption,
  for the same reasons as ADR 0004 decision 7.

### V11. Ledger and results

**Ledger columns.** Options: (a) vehicle rows in the one ledger, zero
without a vehicle (recommended); (b) a second, vehicle-only frame. (a) keeps
one closure test over one frame and one row-write path in the day loop, as
ADR 0004 C8 did. Its cost is memory and write time for every run; PR 2
measures it, and a later PR may drop all-zero vehicle rows from summary
buffers as the PV-only summary buffer does.

New state rows: `Vehicle_Plugged` (0 or 1), `Vehicle_Energy` (in-site, Wh,
0 away), `Vehicle_Energy_Onboard` (Wh), `Vehicle_SOC_Absolute`,
`Vehicle_SOH`, `Vehicle_PV_Origin_Energy_End`,
`Vehicle_Grid_Origin_Energy_End`. New flow rows (average W):
`PV_AC_To_Vehicle`, `Grid_AC_To_Vehicle`, `Vehicle_Charge_Conversion_Loss`,
`Vehicle_Charge_Input`, `Vehicle_Charge_Loss`, `Vehicle_Discharge_DC`,
`Vehicle_Discharge_Loss`, `Vehicle_Charger_Loss`, `Vehicle_AC_To_Load`,
`PV_Origin_Vehicle_AC_To_Load`, `Grid_Origin_Vehicle_AC_To_Load`,
`Vehicle_Departure_Energy`, `Vehicle_Arrival_Energy`, `Vehicle_Trip_Energy`,
`PV_Origin_Vehicle_Trip_Energy`, `Grid_Origin_Vehicle_Trip_Energy`,
`Vehicle_Trip_Unserved`, `Vehicle_Departure_Shortfall`. The ledger schema
becomes `"3.2"` (additive, after ADR 0004's 3.1). The result format stays
`"1"`.

**Existing columns.** `PV_AC_Export` and `Import_From_Grid` are the site's
final exchange, after the vehicle. `Houseload` stays the house load without
the vehicle. `Grid_AC_To_Battery` stays the stationary battery's.

**Year rows** gain the vehicle's AC charged from PV and from the grid, AC to
the house load, trip energy, PV used for driving, unserved trip energy,
departures short of target (count and kWh) and the vehicle's health.
`Self_Consumption_kWh` adds PV-origin vehicle AC to the load, an AC
quantity like its other terms. PV used for driving is stored DC energy, so
it is reported on its own, not added to it. Avoided emissions keep A10's net
exchange, with R0's import in place of the load:
`(Import_R0 − Import_From_Grid − B_u) × CI + PV_AC_Export × CI_export`,
where `B_u` is the stationary battery's unattributed energy to the load, as
in A10. Without a vehicle `Import_R0` is the load and the formula is A10's,
bit for bit. Grid energy that passes through the vehicle and back to the
house enters through `Import_From_Grid`, so its losses count against the
system. Both households start with the vehicle at its target, so its
unattributed starting energy cancels. `calculate_co2_projection` gains
R0's yearly import as an input.

**App result** gains a `vehicle` block: the configured modes, those year-row
totals, the end health in R1 and R2, the wear and its present value, and the
V2H benefit with its parts. `provenance.vehicle` records the resolved table,
the schedule hash and its rounding.

### V12. Default behaviour, backends and entry points

- **No `[vehicle]`, no change.** `_simulate_core` takes the vehicle as an
  optional argument; without it the day loop's vehicle branch is skipped,
  the vehicle rows are written as zeros, `_dispatch_dc_step` and ADR 0004's
  AC step are not edited, and the PV-only path stays vectorised. App goldens,
  golden outputs and the parity harness pass unchanged; the stored gallery
  results and goldens are regenerated for the provenance field
  (`vehicle: null`) and the ledger version only.
- **Numba.** `_dispatch_vehicle_step` is registered with `register_jitable`.
  `_dispatch_day` gains the vehicle scalars and the three schedule arrays
  (empty without a vehicle), so the kernel keeps one signature. The vehicle's
  degradation, departures' effect on health and the carry between years stay
  in Python. Python and Numba must agree bit for bit with a vehicle too.
- **Monte Carlo.** Each trajectory runs R2 and, with a `reference_mode`, R1
  on the same sampled weather and load, so the benefit is paired. The load
  scale applies to the house load only; trips are not sampled.
- **Optimizer.** Refused with a `[vehicle]` table in 0.8.0. The vehicle is
  not a design variable, and the reference run would double every candidate.
  Options: refuse (recommended), or accept with R1 off. A study can sweep
  App runs meanwhile.

## Energy closure

Per step, with the vehicle away contributing nothing:

1. Load supply: `Houseload = PV_AC_To_Load + Battery_AC_To_Load +
   Vehicle_AC_To_Load + Import_From_Grid − Grid_AC_To_Battery −
   Grid_AC_To_Vehicle`.
2. PV AC. AC coupling or no stationary battery (ADR 0004 identity 4,
   extended): `PV_DC_To_Inverter − PV_Direct_Inverter_Loss = PV_AC_To_Load +
   PV_AC_Export + PV_AC_To_Battery + PV_AC_To_Vehicle`. DC coupling: the
   stationary step's PV AC export, before the vehicle, equals
   `PV_AC_Export + PV_AC_To_Vehicle`, and the DC step's own identities are
   unchanged.
3. Charger, charging: `PV_AC_To_Vehicle + Grid_AC_To_Vehicle −
   Vehicle_Charge_Conversion_Loss = Vehicle_Charge_Input`.
4. Charger, discharging: `Vehicle_Discharge_DC − Vehicle_Discharge_Loss −
   Vehicle_Charger_Loss = Vehicle_AC_To_Load`.
5. Vehicle store, in site: `ΔVehicle_Energy = Vehicle_Charge_Input ×
   eff_c − Vehicle_Discharge_DC − Vehicle_Departure_Energy +
   Vehicle_Arrival_Energy`.
6. Trip: `Vehicle_Trip_Energy = Vehicle_Departure_Energy −` the on-board
   energy after departure.
7. Whole system: the inputs gain `Vehicle_Arrival_Energy`; the outputs gain
   the four vehicle losses, `Vehicle_Departure_Energy` and
   `ΔVehicle_Energy`. `PV_AC_To_Vehicle` and `Grid_AC_To_Vehicle` are
   internal and grid AC enters through `Import_From_Grid` as today.
8. Origins: the three vehicle origins reconcile step by step, with trips
   removing them in proportion.
9. Rating: `PV_AC_To_Vehicle + Grid_AC_To_Vehicle ≤ P` and
   `Vehicle_AC_To_Load ≤ P`, in average W, to rounding.

## Reference cases

Every case uses a vehicle of 40 000 Wh at full health, `min_soc = 0.1`,
`max_soc = 1`, `departure_soc = 0.8` (`T` = 32 000 Wh), `v2h_min_soc = 0.3`,
battery efficiencies 0.95, and a 7400 W charger at 0.95 on the part-load
curve, hourly steps, no stationary battery and no site limit. One
full-power step stores `s_max = 7400 × 0.95 × 0.95 = 6678.5` Wh. Values
are Wh per step, computed with `_dc_ac` and `_dc_for_ac` at `567df2a`
following the pseudocode of V5.

**VH1 — PV surplus to the vehicle.** PV inverter without a rating at 0.96,
PV DC 5000, load 1000, `E` = 20 000, ten steps to departure (no mandatory
charge). `PV_AC_To_Vehicle` 3800, `Vehicle_Charge_Conversion_Loss`
181.415687 (part-load ratio 0.514), `Vehicle_Charge_Input` 3618.584313,
`Vehicle_Charge_Loss` 180.929216, stored 3437.655098, `PV_AC_Export` 0,
`Vehicle_Energy` 23 437.655098.

**VH2 — the deadline guard charges as late as possible.** No PV, load 500,
`E` = 20 000 at four steps before departure, flat prices.

| Step (`k`) | `"v1g"` `Grid_AC_To_Vehicle` | `"v1g"` `Vehicle_Energy` | `"uncontrolled"` `Grid_AC_To_Vehicle` | `"uncontrolled"` `Vehicle_Energy` |
|---|---|---|---|---|
| 4 | 0 | 20 000 | 7400 | 26 678.5 |
| 3 | 0 | 20 000 | 5885.422167 | 32 000 (exactly) |
| 2 | 5885.422167 | 25 321.5 | 0 | 32 000 |
| 1 | 7400 | 32 000 (exactly) | 0 | 32 000 |

At `k` = 2 the guard stores `12 000 − 6678.5 = 5321.5`, at a part-load ratio
of 0.795 and a conversion loss of 283.843220; at `k` = 1 the charger runs at
full load. Both modes buy the same energy; V1G buys it later, when a
time-of-use tariff or PV could make it cheaper. Without the exact landing
`Vehicle_Energy` would end at 32 000.000000000004 (`"v1g"`) and
32 000.000000000007 (`"uncontrolled"`).

**VH3 — V2H within the guard.** No PV, load 3000, `E` = 30 000 at three
steps before departure, `"v2h"`, no discharge periods.

| Step (`k`) | `d_max` | `Vehicle_AC_To_Load` | `Vehicle_Discharge_DC` | `Grid_AC_To_Vehicle` | `Import_From_Grid` | `Vehicle_Energy` |
|---|---|---|---|---|---|---|
| 3 | 11 357 | 3000 | 3320.753657 | 0 | 0 | 26 679.246343 |
| 2 | 1357.746343 | 1204.751230 | 1357.746343 | 0 | 1795.248770 | 25 321.5 |
| 1 | — | 0 | 0 | 7400 | 10 400 | 32 000 (exactly) |

At `k` = 2 the guard lets the vehicle discharge to exactly `T − s_max`, so
the last step can still reach the target. Over the three steps the house
avoids 4204.751230 Wh of import and the vehicle buys back 7400 Wh; R1
(`"v1g"`) would buy 2221.635598 on the last step, the AC for the 2000 Wh it
lacks at a part-load ratio of 0.300. Under a
flat price this V2H run costs more than V1G; with `discharge_periods` on a
peak and the buy-back in an off-peak period it can cost less. The
comparison of V8 reports either.

**VH4 — departure, absence and arrival.** At departure `E` = 32 000, of which
PV origin 16 000 and grid origin 8000; `trip_wh` = 9000.
`Vehicle_Departure_Energy` 32 000, `Vehicle_Trip_Energy` 9000 (PV origin
4500, grid origin 2250), `Vehicle_Energy` 0 and every vehicle flow 0 while
away, `Vehicle_Energy_Onboard` 23 000 (PV origin 11 500, grid origin 5750),
`Vehicle_Arrival_Energy` 23 000 on the arrival step. The whole-system
balance holds on each step with the house side unchanged.

**VH5 — an unmet target.** `E` = 10 000 on the last plugged step (`k` = 1):
the charger runs at full power, stores 6678.5, and the vehicle leaves with
16 678.5. `Vehicle_Departure_Shortfall` 15 321.5. With `trip_wh` = 9000 the
trip is served and arrives with 7678.5. With `trip_wh` = 15 000 only
`16 678.5 − 4000 = 12 678.5` is usable: the vehicle arrives with 4000 and
`Vehicle_Trip_Unserved` is 2321.5.

The reference-case tests compare with the full-precision helper values to
1e-9 relative, and the exact landings exactly, as ADR 0004's AC7 does.

**Invariants**, tested at hourly and 15-minute steps, Python and Numba:

- Q1. Without `[vehicle]`, every existing column and value is bit-identical
  to the release before (App goldens, golden outputs, parity harness); the
  new vehicle rows are 0 and provenance gains `vehicle: null`.
- Q2. With a vehicle, every stationary battery column and every stationary
  instruction equals the same run without the vehicle, bit for bit.
- Q3. A vehicle that is never plugged in gives the house flows of the run
  without a vehicle bit for bit (for a run without a stationary battery,
  the PV-only flows, by ADR 0004 P2).
- Q4. Every identity above holds to the existing tolerances; no step both
  charges and discharges the vehicle; the vehicle never exports.
- Q5. With no site limit and every target reachable, no departure is short
  in any mode.
- Q6. R0's vehicle charging does not depend on the system configured.

## Alternatives considered

- **The vehicle as a load profile with a flexible part.** Rejected: it
  cannot discharge, so it cannot be V2H, and the energy it holds would not
  close.
- **Reuse `BatteryConfig` with an availability mask.** Rejected: the
  stationary pack's capacity window, temperature derating, standby,
  replacement and retirement rules do not apply to a vehicle, and two
  stores in one config would change the stationary contract.
- **Vehicle first, or a joint step.** Rejected for 0.8.0 (V4).
- **A forecast-based vehicle planner.** Deferred: the guard is causal and
  serves the departure target; a planner is a later mode.
- **Vehicle-to-grid.** Out of scope (V1). It needs export from storage,
  market or tariff rules for it, and a different reference.
- **Arrival state of charge as the model.** Rejected (V3).

## Implementation plan and effort

Six PRs, about 18 working days for one developer, plus review and the
platform CI runs each kernel change needs. Once the work ships, this section
is replaced by an implementation status, as in ADR 0002.

**Dependency on ADR 0004 (#419).** V2H reuses ADR 0004's charging and
discharging conversion (C4), its rating units (`P_wh`), the PV chain arrays
in the day loop (C3) and the AC step with an absent battery (C5, P2, P3),
and the ledger schema 3.1 (C8). Precisely:

- PR 1 below needs nothing from #419 and may start at once.
- PR 2 changes the day loop and the kernel, so it starts after #419 PR 2
  merges (the AC step, the PV chain arrays, ledger 3.1, AC1–AC7 passing).
  Building both kernel changes at once would put two branches in one
  function and break each other's bit-identity gates.
- PR 4 follows #419 PR 4, so App exposes the vehicle only after App
  exposes AC coupling, and the cost refusal follows the pattern of #419
  PR 3.
- The reference cases and goldens are computed after #417 and #418, on the
  final PV defaults, as ADR 0004's are.

**Rollout gates.** No merged state runs a vehicle half-built:

- PR 1 adds `VehicleConfig`, the `[vehicle]` table and the schedule, and
  refuses to simulate a vehicle.
- PR 2 lifts that refusal for a direct `_simulate_core` call only.
- PR 4 is the first PR that exposes `[vehicle]` in App, with R0, R1 and
  every required price.

| PR | Scope | Effort | Main risk |
|---|---|---|---|
| 1 | `VehicleConfig` and `[vehicle]` validation; schedule resolution to per-step arrays (civil time, rounding, year seam, trip file), its hash; arrival SOC converted to trip energy; simulation refused. No behaviour change. | 2.5 days | Civil-time edge cases (DST, year seam). |
| 2 | `_dispatch_vehicle_step` with the deadline guard and exact landing; departure and arrival; vehicle rows and ledger schema 3.2; origins; the day-loop branch and its arguments in `_day_arguments`; Numba registration; closure helper; reference cases VH1–VH5 and invariants Q1–Q5; parity scenarios. | 5 days | Stationary bit identity (Q2), and the kernel's compile time and memory with more rows. |
| 3 | The vehicle's degradation lifecycle between day windows, its temperature, its carry between years; refusal of `daily_persistence` with a vehicle. | 2.5 days | A second lifecycle in the day-close path without moving the stationary one. |
| 4 | App: R0's vehicle charging in the baseline (invariant Q6), R1 as a second projection, the V2H benefit and its comparability flag, `outside_charging_price`, wear valuation, `charger_cost` and `battery_cost_per_kwh` required and refused when missing, emissions, year rows, the `vehicle` result block, provenance, regenerated config docs and golden provenance. | 4.5 days | A path that reaches economics before a price is checked. |
| 5 | Monte Carlo: paired R1 per trajectory, load scale on the house only; optimizer refusal. | 1.5 days | Memory of two projections per trajectory. |
| 6 | Docs: configuration guide (`[vehicle]`, the assumptions in V5 and V9), energy-balance and results pages, a gallery case, CHANGELOG; ADR implementation status. | 1.5 days | None significant. |

Risks across the work:

- **Stationary bit identity.** Order V4 makes Q2 a test, not a hope; it runs
  on PR 2 with every C10 gate of ADR 0004.
- **Performance.** R1 doubles a V2H run. The new rows add memory to every
  run; PR 2 measures it with `tools/benchmark_montecarlo.py` and the
  cold-compile time.
- **Model validity.** The charging direction of the part-load curve (ADR
  0004 C4), the vehicle's temperature and the use of cell ageing models for
  a vehicle pack are assumptions, and the docs say so.
- **Data.** Charger prices, charger efficiencies and vehicle pack prices are
  user inputs; BREOS ships none.

## Open decisions for the maintainer

1. **Dispatch order (V4):** stationary battery first, vehicle second, on the
   AC residual; no storage-to-storage transfer (proposed); or vehicle
   first; or a joint rule?
2. **Charger model (V2, V5):** ADR 0004's battery inverter core in both
   directions, with a required rating and efficiency (proposed); or flat
   efficiencies?
3. **Schedule (V3):** weekday and weekend times plus an optional trip file,
   with conservative rounding to whole steps (proposed); or strict
   boundaries as for tariffs?
4. **Energy while away (V3):** trip energy in kWh as the model, arrival SOC
   converted to it against the target (proposed); or arrival SOC as a
   state?
5. **Departure target (V5):** a hard constraint through the deadline guard,
   with unmet targets reported and not refused; an optional
   `outside_charging_price` that prices unserved trip energy in every
   household, and without it a benefit flagged as not comparable when V2H
   leaves more driving unserved (proposed)? Or refuse the run?
6. **Trip timing (V6):** trip energy removed at departure (proposed), at
   arrival, or spread over the absence?
7. **Smart charging (V7):** vehicle periods in `[vehicle]`, `[smart_charging]`
   for the stationary battery only, `daily_persistence` refused with a
   vehicle in 0.8.0 (proposed)?
8. **Reference (V8):** R0 with uncontrolled charging as the no-system
   baseline; R1 with `reference_mode` defaulting to `"v1g"` for `"v2h"`
   (proposed); or always uncontrolled?
9. **Degradation (V9):** a second lifecycle on the on-board SOC, native by
   default, wear from home use as the R1 − R2 health difference (proposed);
   or a throughput wear price?
10. **Vehicle lifetime (V9):** one pack for the horizon (proposed); or a
    holding period with a new vehicle?
11. **Costs (V10):** `charger_cost` as the extra cost over a standard
    charger, required for `"v2h"` with no preset, optional in the other
    modes; wear valued at the pack price on the R1 − R2 health difference,
    unclipped, booked at `T`, reported beside the NPV and inside the V2H
    benefit (proposed)?
12. **Ledger (V11):** vehicle rows in the one ledger, schema 3.2 (proposed);
    or a separate frame?
13. **Entry points (V12):** App and Monte Carlo, optimizer refused in 0.8.0
    (proposed)?
14. **Target release:** 0.8.0 after #419, with the gates above.

## Consequences

- A household with a vehicle can be simulated in App with uncontrolled,
  smart (V1G) or V2H charging, and its V2H benefit is reported against the
  same household without V2H, including the vehicle's extra wear.
- The no-system baseline buys the vehicle's charging too, so NPVs with a
  vehicle compare like with like.
- Runs without `[vehicle]` keep every existing value bit for bit; the
  ledger schema moves to 3.2 with additive rows and the result format stays
  "1".
- The stationary battery's dispatch does not change when a vehicle is
  added.
- Vehicle-to-grid, a vehicle planner, vehicle replacement, charger power
  reduction by the network operator and vehicle thermal management stay
  outside BREOS.
