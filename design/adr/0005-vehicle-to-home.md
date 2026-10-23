# 0005 — Vehicle-to-home: the vehicle as a second, part-time AC-coupled store

- **Status:** Proposed, with the decisions taken by the maintainer
  on 2026-10-06, 2026-10-07 and 2026-10-22 (see [Decisions](#decisions)).
  Implemented for 0.8.0 after the AC-coupled battery work (#419), pending
  maintainer acceptance. Tracked in #420.
- **Date:** 2026-10-06; decisions taken 2026-10-06, 2026-10-07 and
  2026-10-22; amended 2026-10-22

## Context

When this record was proposed, BREOS had no electric vehicle model. A
household with a vehicle charges it at home, and its charging demand is
often larger than the rest of the house's
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

This record answers each point. It changes no code. The maintainer took
its fourteen decisions on 2026-10-06 and, after a second design review,
revised four of them on 2026-10-07: trip energy is gross, with optional
charging away from home (decision 4); it is withdrawn over driving legs
(decision 6); the vehicle pack is replaced at its threshold (decision 10);
and wear is valued through replacement outlays and terminal credits
(decision 11). The sections below follow the decisions, and
[Decisions](#decisions) lists them. The contract's "state of charge at
arrival" is no longer an input: arrival is an outcome of driving and away
charging.

## Current behaviour this record relies on

Line numbers are at `567df2a` (`origin/develop`, 2026-10-06). At that
revision ADR 0004 was accepted and not implemented; this historical account
cites its sections (C1–C14), not code. See the
[Amendments of 2026-10-22](#amendments-of-2026-10-22) for the implemented base.

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

**Replacement.** At a day close, a stationary pack at or below
`eol_percentage` is replaced by `_apply_battery_replacement`
(`breos/battery.py:1092-1145`), which books the removed and added energy
and starts the new pack at `max_soc` (`:1127`), unless a guard skips the
swap (`_replacement_skipped_by`, `:1440`; ADR 0003 E11) and the pack is kept
or retired (E12). Economics books each swap at its instant
(`replacement_booking_time`, `breos/economics.py:277`; E3, E4).

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
- **Away from home** the vehicle drives and may charge, at work or at a
  public charger. Both change the energy on board and cross no site meter:
  the site's balance sees the vehicle leave and come back, and the vehicle's
  own on-board balance books the driving and the away charging (V6). Away
  purchases are a separate source of energy, priced and counted in emissions
  in every household (V8, V11).
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
departure_soc = 0.80              # required at every departure from home
v2h_min_soc = 0.40                # V2H never discharges below this
charge_efficiency = 0.95          # battery, DC in to stored
discharge_efficiency = 0.95       # battery, stored to DC out
charger_ac_rating_kw = 7.4        # bidirectional, AC side
charger_efficiency = 0.94         # nominal, no default
charge_periods = ["off_peak"]     # optional; needs [tariff]
discharge_periods = ["peak"]      # optional; needs [tariff]; "v2h" only
reference_mode = "v1g"            # R1, the household V2H is compared with (V8)
charger_cost = 2500               # installed premium over the reference charger (V10)
battery_cost_per_kwh = 120        # vehicle pack price, for replacements and wear (V10)
eol_percentage = 0.70             # vehicle pack replacement threshold (V9)
outside_charging_price = 0.60     # optional penalty per kWh of unserved driving (V5)

[vehicle.away_charging]           # optional (V6)
target_soc = 0.70                 # stopping target, not a ceiling
power_kw = 3.7                    # at the purchased-energy boundary
efficiency = 0.90                 # stored energy per purchased energy
price = 0.0                       # away_charging_price per kWh purchased; 0 = free

[vehicle.schedule]
weekday = { arrive = "18:30", depart = "07:45", trip_kwh = 9.0 }
weekend = { arrive = "14:00", depart = "10:00", trip_kwh = 12.0 }
# or: file = "trips.csv"  (V3)
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
  `"v2h"`; its default is `min_soc`. The state of charge at arrival is an
  outcome of the run, never an input.
- **No defaults for equipment facts.** `battery_kwh`, `departure_soc`,
  `charger_ac_rating_kw` and `charger_efficiency` are required. BREOS has no
  sourced value for any of them, and a charger efficiency is not the PV
  inverter's 0.96, as ADR 0004 C2 says of the battery inverter. The battery
  efficiencies default to the stationary battery's (`DEFAULT_CHARGE_EFFICIENCY`,
  `DEFAULT_DISCHARGE_EFFICIENCY`). With `[vehicle.away_charging]`, all four
  of its keys are required; `price = 0` is valid (free charging at work).
- **Rating units** follow ADR 0004 C4: the rating is AC, the charger's
  input when charging and its output when discharging, and it enters every
  per-step limit as `P_wh = P × hours_per_step`.
- **Replacement keys** mirror the stationary battery's (V9):
  `enable_replacement`, `replacement_min_remaining_years`,
  `skipped_replacement_action` and an optional `battery_replacement_cost`
  (one swap's invoice, as ADR 0004 C9 gives App).
- The lower level takes `VehicleConfig` and a resolved `VehicleSchedule`
  (V3) through new optional `_simulate_core` arguments, as instructions and
  controllers arrive today.

### V3. Plug-in schedule, trips and away charging windows

**Input forms.** Decided 2026-10-06 (decision 3). The options were:

- (a) Arrival and departure times for weekdays and weekend days, each with
  the trip energy, plus an optional file of trips.
- (b) A per-step plugged-in mask from a file.
- (c) Statistical trip generation (sampled arrival, departure and distance).

(a) is decided. Times per day type cover the common study, and the file
covers measured logbooks. A mask (b) cannot carry the energy of each
absence, and generation (c) is a behaviour model BREOS has no data for; a
study can generate a trip file itself.

- **Trip energy.** `trip_kwh` is the gross energy the trip withdraws from
  the energy stored in the pack, driving losses included (decision 4). It
  does not depend on away charging: charging on the way is a separate input
  to the pack (V6), never a reduction of `trip_kwh`. There is no
  arrival-state input.
- **Day types.** Without a file, each absence follows the weekday or weekend
  rule of its departure's civil date, and the template of V6 places its
  energy in two driving legs with a parked interval between. Each day type
  may set `away_charging = false` to have no charging window.
- **Trip file.** One row per segment of an absence, in time order:
  `start`, `end` (timestamps with an offset or in the location's timezone,
  as load files are read), `kind` (`"drive"` or `"park"`), `energy_kwh` for
  a drive (gross, as `trip_kwh`), and `away_charging` (true or false) for a
  park. An absence is the run of segments between two home periods. Rows
  must not overlap. A segment may cross days and the year end. Within the
  simulated year a file replaces the weekday and weekend rules, and it is
  replayed every project year like the tariff calendar (ADR 0002 A2).
- **Civil time.** Times are local wall-clock times in the configured
  timezone, resolved as a tariff schedule is (ADR 0002 A1), never from the
  index's own offset. A day type is the configured-zone civil date's
  weekday; holidays are not modelled.
- **Steps.** A step is plugged in only if the vehicle is home for the whole
  step. An arrival rounds up to the next step boundary and a departure rounds
  down. At hourly steps an arrival at 18:30 plugs in at 19:00 and a
  departure at 07:45 leaves at 07:00. The rounding never gives the vehicle
  more time at home than the schedule says, and never gives it more away
  charging time either: a charging window's hours in a step are the parked
  hours the schedule gives inside that step, not the whole step. The other
  option was ADR 0002 A3's strict policy, which refuses a time off the step
  grid; a schedule is behaviour data, not a regulatory boundary, and
  refusing 07:45 at hourly steps would force invented times.
- **Resolution.** The schedule resolves once, on the simulated index, to
  per-step arrays: `plugged` (bool), `steps_to_departure` (the number of
  plugged steps left before the next departure, this one included; 0 when
  away), and, for away steps, `drive_first_wh`, `park_hours` (charging
  window hours) and `drive_second_wh`: the step's driving before its
  parked time, its parked charging time, and its driving after. They are
  hashed for provenance as instructions are. Every project year replays
  them, and a plug-in or an absence that crosses the year end continues at
  the head of the year (A2's seam). A `[period]` run takes the next
  departure after its end from the same rules.
- **Sub-step order.** A step with more alternations of driving and parking
  than drive, park, drive is resolved to those three in order: the driving
  before its first parked time, all its parked time, and the rest of its
  driving. At 15-minute steps this is rare; it is an approximation the docs
  state.

### V4. Dispatch order

Each step dispatches the stationary battery first, exactly as today, and
the vehicle second, on the AC flows the stationary step left:

```text
1. stationary step (_dispatch_dc_step, or ADR 0004's AC step, or none)
   -> pv_ac_export, grid_import, grid_ac (stationary grid charge)
2. vehicle step on the AC residual:
   surplus      = pv_ac_export                  # PV the house would export
   deficit      = max(0, grid_import - grid_ac - night)  # house load only (A5)
   site_room    = max(0, grid_import_cap_wh - grid_import)   # load import is never cut
   -> PV_AC_To_Vehicle, Grid_AC_To_Vehicle, Vehicle_AC_To_Load
   PV_AC_Export    = pv_ac_export - PV_AC_To_Vehicle
   Import_From_Grid = grid_import - Vehicle_AC_To_Load + Grid_AC_To_Vehicle
```

Decided 2026-10-06 (decision 1): (a). The options were:

- (a) Stationary battery first, vehicle second.
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
excludes the stationary battery's grid charge and the PV inverter's night
draw. The grid keeps that draw as a documented simplification (ADR 0004
M3; A5). Energy moved from one store to the other would pay two
conversions and two battery efficiencies for no service.

**Which PV inverter.** With a DC-coupled stationary battery, the vehicle
takes PV AC after the hybrid inverter. With an AC-coupled one, or none, the
house side is ADR 0004's AC path: the PV inverter's output comes from the PV
chain arrays computed before the day loop (ADR 0004 C3). A run with a
vehicle and no stationary battery therefore enters the day loop with the
AC step and an absent battery, whose house flows equal the PV-only run's
bit for bit (ADR 0004 P2, P3). With several PV inverters, the vehicle
uses their summed AC residual, with no stationary battery or beside an
AC-coupled one. A DC stationary battery beside several inverters stays
refused, as does Monte Carlo with several inverters (A6).

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
- **Site limit.** `[smart_charging].grid_import_limit_w` is the one site
  limit key, accepted without a stationary battery when a vehicle is
  configured (A12). With it set, `s_max` uses the smaller
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
  (V11). Decided 2026-10-06 (decision 5); the other options were to refuse
  the run, which would make a whole study fail on one late arrival, or to
  buy the shortfall at a price. A short departure shows up on the road: the
  trip then runs out of energy sooner, and `Vehicle_Drive_Unserved` records
  it (V6). `vehicle.outside_charging_price` is an optional penalty per kWh
  of unserved driving, with no default: set, it is a cost in every
  household (R0, R1, R2), so households that drive the same trips stay
  comparable. It is money only; it restores no energy. Away charging (V6)
  is the physical alternative, power-limited and priced at what it buys.
  Unset, shortfalls and unserved driving are reported only, and a
  comparison is flagged as not comparable when R2 leaves more driving
  unserved (V8).
- **One direction per step,** as A8 requires: a step that charges does not
  discharge. The guard gives `d_max ≤ 0` whenever `mandatory > 0`.
- **Limits.** `P_wh` bounds charger AC in and AC out; `site_room` bounds
  vehicle grid charging, so `grid_import_limit_w` counts the vehicle too.
  The vehicle has no capacity-temperature derating, standby loss or
  self-heating model: `lfp_capacity_factor` describes an LFP stationary pack
  in its enclosure, and a vehicle's thermal management is outside BREOS. This
  is an assumption, and the docs state it.

### V6. Departure, driving, away charging and arrival

Decided 2026-10-07 (decisions 4 and 6). Trip energy is withdrawn over
driving legs, with parked time between them in which the vehicle may charge
away from home.

- **Departure** happens at the start of the first step that is not plugged
  in. The vehicle's stored energy `E_dep` leaves the site: the site ledger
  books `Vehicle_Departure_Energy = E_dep`, and the in-site stored energy
  becomes 0. The on-board energy is unchanged by crossing the boundary.
- **Template legs.** Without a trip file, half of `trip_kwh` is withdrawn
  in the first away step and half in the last; the steps between are parked.
  An absence of one step holds both halves, the first before and the second
  after any parked time in that step (there is none in the template). This
  is a coarse template, not a measured driving profile, and the docs say so.
  A trip file gives the legs, their energy, their duration and the charging
  windows instead; a drive segment's energy is spread over its steps in
  proportion to the time it spends in each.
- **Driving within a step** runs in the order of V3: `drive_first_wh`, then
  the parked time, then `drive_second_wh`. Each withdrawal is served down to
  `min_soc × cap_v` and no further. What cannot be served is
  `Vehicle_Drive_Unserved`; a withdrawal never borrows from charging later
  in the same step or a later step, and the energy on board never goes
  negative.
- **Away charging.** In a parked window with `[vehicle.away_charging]`, the
  vehicle charges toward `target_soc × cap_v`. Per step it stores at most
  `power_kw × park_hours × efficiency`, and never more than the room to the
  target. `target_soc` is a stopping target, not a ceiling: a vehicle above
  it keeps its energy, does not charge, and later driving may take it below.
  The target need not be reached. `power_kw` and `efficiency` are at the
  purchased-energy boundary: `Vehicle_Away_Charge_Purchased` is the energy
  bought, `Vehicle_Away_Charge_Loss` the part that is not stored. The home
  charger's rating and efficiency are not assumed for away charging.
- **Away charging and the departure target.** Away charging happens only
  away. It cannot meet a departure target, which must be met before the
  vehicle leaves home (V5); the guard counts home steps only.
- **Arrival** is the first plugged-in step after an absence. The on-board
  energy enters the site as `Vehicle_Arrival_Energy`, booked before that
  step's dispatch. The arrival state of charge is whatever driving and away
  charging left.
- **Absences across days and years.** Legs and charging windows of a
  multi-day absence carry over civil days and the year seam, as the
  schedule arrays do (V3).
- **Origins.** The vehicle's stored energy has the three origins of ADR 0002
  A8. Driving removes them in proportion. Away charging adds to the grid
  origin: it is grid energy, bought off site. PV-origin energy used by
  driving is reported as PV used for driving.
- **Start.** The vehicle starts the project at its departure target,
  unattributed, plugged in or away as the schedule says at the first step.
  Between project years the on-board energy, its origins, its health and
  its replacement state carry over as the stationary battery's do
  (`physical_carry`).

The options for the timing of the trip energy were: all at departure, all
at arrival, spread uniformly over the absence, or over driving legs with
parked intervals between. All at departure holds the state of charge low
through the absence and lets no away charging serve the way back; all at
arrival holds it high and lets later charging pay for a trip the pack could
not have made; a uniform spread makes a parked car drive all day, and with
away charging it charges while driving. Driving legs order driving and
charging correctly, and the template keeps the weekday and weekend input
simple. Endpoint deductions do not bracket the leg model's wear or
replacement cost, since ageing is not linear in the state of charge;
studies test their sensitivity with a trip file.

### V7. Smart charging and tariffs

Decided 2026-10-06 (decision 7).

- `[smart_charging]` keeps its meaning and gates the stationary battery
  only, except that its `grid_import_limit_w` also limits vehicle grid
  charging and is accepted without a stationary battery when `[vehicle]`
  is configured (A12). No new site-limit key is added, and stationary
  instructions do not change. The vehicle's own `charge_periods` and
  `discharge_periods` select tariff periods, need a `[tariff]`, must be
  disjoint, and are refused with
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

### V8. The households and the comparisons

Decided 2026-10-06 (decision 8). Three households are defined, all with the
same vehicle, trips, away-charging rules and prices, and the same vehicle
replacement policy:

| Household | System | Vehicle charging at home | Used for |
|---|---|---|---|
| R0, no system | none | `"uncontrolled"`, from the grid | NPV baseline, emissions baseline, household-adjusted NPV |
| R1, reference | the configured PV and battery | `reference_mode` | V2H benefit |
| R2, study | the configured PV and battery | `mode` | every reported result |

- **R0** extends today's no-system household: it buys the house load and the
  uncontrolled vehicle charging from the grid. `Baseline_Import_Cost` prices
  both, so a system's NPV is not charged for owning a vehicle. R0 is a
  vehicle-only projection (no PV, no battery) with the vehicle's ageing and
  replacements, simulated once per project year. It runs whenever
  `[vehicle]` is set.
- **R1** is a second projection of the configured system with
  `vehicle.mode = reference_mode`. `reference_mode` defaults to `"v1g"` for
  `"v2h"` and to `"uncontrolled"` for `"v1g"`; `"none"` skips R1 and the
  V2H benefit. A `reference_mode` equal to `mode` is refused, and
  `"uncontrolled"` has no R1.
- **Run time.** A V2H run is three household trajectories: R0, a light
  vehicle-only run, and R1 and R2, full system runs. Each has its own
  vehicle degradation lifecycle and swap dates, so a V2H run costs about
  twice a run without a vehicle in dispatch and three vehicle lifecycles in
  degradation. App runs all three. Monte Carlo runs all three per
  trajectory on the same sampled weather and load (V12).
- **System NPV** keeps today's meaning: the system's cash flows against R0's
  bill. It does not include vehicle wear or away purchases.
- **Household-adjusted NPV**, reported beside it and labelled as such: the
  system NPV, less the present value of R2's away purchases over R0's, less
  the vehicle wear `W_{2−0}` (V10), less R2's priced unserved driving over
  R0's when `outside_charging_price` is set. It answers what the household
  gains from the system once its vehicle is counted.
- **V2H benefit**, R2 against R1: operating savings (R1's home electricity
  bill, export revenue, away purchases and priced unserved driving, less
  R2's), less `charger_cost` (V10), less the vehicle wear `W_{2−1}`. It is
  reported with its parts.
- **Same driving.** Both comparisons are only meaningful if the households
  drive the same trips. Each household reports its unserved driving and
  departure shortfalls. With `outside_charging_price` set, unserved driving
  is a cost in each household (V5). Without it, a comparison in which R2
  leaves more driving unserved than its comparator is reported with
  `comparable = false`, and the docs say why.

The other options for `reference_mode` were always `"uncontrolled"`, which
would mix the value of smart charging into the value of V2H, and always
`"v1g"`, which leaves no comparison for a V1G study.

### V9. Vehicle battery degradation and pack replacement

**Model.** Decided 2026-10-06 (decision 9): a second degradation lifecycle
for the vehicle pack, stepped at the same positional day windows as the
stationary pack, on the vehicle's real on-board state of charge (driving,
away charging and home use included) and its temperature. The engine is
chosen as for the stationary battery: native by default, BLAST by key. It
reuses the `DegradationLifecycle` boundary and runs in Python between day
windows, as the stationary pack's does, so the kernel stays free of ageing.
It captures calendar ageing, which V2H can lower as well as raise. The other
options were a wear price per kWh of V2H discharge, as the planner's
`wear_cost_per_kwh` (ADR 0002 A17), which misses calendar ageing, and no
wear. Limits the docs state: the native model is calibrated on LFP cells,
many vehicles use NMC or NCA, and a BLAST key names one cell model, not a
vehicle pack (battery degradation policy). The vehicle has its own
`degradation_engine`, `blast_model` and `calendar_model` keys, defaulting to
the stationary battery's defaults.

**Temperature.** `vehicle.battery_temperature` takes `"weather"`, a fixed
value or a CSV, as `battery_temperature` does, without the indoor model.
Default `"weather"`. A vehicle's thermal management keeps its cells warmer
in winter and cooler in summer than the air; this is an assumption, stated.

**Capacity.** Fade lowers `cap_v`, and with it `T`, `F` and the window
(A7's rule, applied to the vehicle). Stored energy above the new
`max_soc × cap_v` at a day close is booked as
`Vehicle_Capacity_Window_Loss`, as `_apply_capacity_window` books it for the
stationary pack, without the temperature factor.

**Replacement.** Decided 2026-10-07 (decision 10). The vehicle pack is
replaced when its health reaches `eol_percentage`, with the stationary
pack's rules: the check at a day close, the minimum-remaining-years guard
(ADR 0003 E11), the terminal-period guard, and the keep or retire action for
a skipped swap (E12). `"retire"` switches the pack off: the vehicle neither
charges nor discharges, and every later trip is unserved driving. Every
household (R0, R1, R2) applies the same policy and simulates its own swap
dates.

- **Replacement energy** is explicit. A swap removes the old pack's
  on-board energy (`Vehicle_Replacement_Energy_Removed`, with its origin
  shares) and adds the new pack's (`Vehicle_Replacement_Energy_Added`,
  unattributed). The new pack starts at the old pack's state of charge, so
  a swap neither fills nor empties the vehicle; the stationary pack's
  start at `max_soc` (`breos/battery.py:1127`) would give the vehicle energy
  no one bought. A swap may fall while the vehicle is away: both rows then
  belong to the on-board balance only, and the site balance does not see
  them. While plugged in they belong to both.
- **A study policy, not a prediction.** Owners do not replace packs at a
  universal state-of-health threshold. The U.S. Department of Energy says
  that "EV batteries are generally designed to last for the expected
  lifetime of the vehicle but will wear out eventually"
  ([Electric Vehicles for Consumers](https://afdc.energy.gov/vehicles/electric-consumers),
  checked 2026-10-07). The threshold is what makes the wear of different
  households comparable over a project longer than a pack's life; the
  docs say so.
- **The pack, not the car.** BREOS replaces the pack. Replacing the whole
  vehicle would need its purchase price, resale value and non-battery
  depreciation; putting a car's price into the pack formula would charge
  them to battery wear.

The draft's one pack for the whole project was rejected: over 20 years
both households' packs can pass their thresholds, and a comparison of final
health alone then compares a worn pack with a new one (V10, VH7).

### V10. Costs and wear valuation

Decided 2026-10-07 (decision 11).

- **Charger.** `vehicle.charger_cost` is the installed premium of the
  configured charger over the reference charger, which R0 and R1 own and
  which is assumed to support V1G. It joins the system's CAPEX, so the
  system NPV (R2 against R0) counts it once, and the V2H benefit (R2
  against R1) subtracts it once; no comparison counts it twice. A study of
  V1G against uncontrolled charging sets it to the smart charger's premium;
  a study on hardware already owned sets it to 0 explicitly. It is required
  with `"v2h"`, with no default and no preset (BREOS has no sourced price);
  in the other modes it is optional and omitted means 0. The other option,
  a configured and a standard price, gives a clearer procurement breakdown,
  but only the difference enters any result.
- **Replacement outlays.** For household `r`, with swaps `j` at simulated
  instants `t_rj` (years from commissioning, as `Replacement_Time_Years`
  books a stationary swap), general inflation `i`, replacement-cost learning
  `ℓ` and the nominal discount rate `d`:

  ```text
  P_r = Σ_j C_swap × (1 + i)^t_rj × (1 − ℓ)^t_rj / (1 + d)^t_rj
  ```

  `C_swap` is `battery_replacement_cost` when set (it may include
  installation), else `battery_cost_per_kwh × battery_kwh`. Booking time,
  escalation, learning and discounting follow ADR 0003 E3, E4 and E11.
- **Terminal credit.** The pack-only credit of ADR 0003 E10, on the final
  installed pack's health `h_r(T)` and the vehicle's `h* = eol_percentage`:

  ```text
  H_r = C_pack × (1 + i)^T × (1 − ℓ)^T / (1 + d)^T × clip((h_r(T) − h*) / (1 − h*), 0, 1)
  ```

  `C_pack = battery_cost_per_kwh × battery_kwh`, never the swap invoice, as
  ADR 0004 C9 amends E10. Each credit is clipped; at or below the threshold
  it is 0. It is computed for this valuation only: it does not switch on
  `[terminal_value]`, which stays optional and off by default, and it
  changes no NPV.
- **Wear.** The wear of household `a` against household `b` is

  ```text
  W_{a−b} = P_a − P_b + H_b − H_a
  ```

  The difference is not clipped. Positive means `a` costs the vehicle more;
  negative means it saves vehicle value. The V2H benefit uses `W_{2−1}`,
  the household-adjusted NPV `W_{2−0}`. The outlays-only difference
  `P_a − P_b` is reported beside it: the cash the household actually pays,
  which a swap just before the horizon swings strongly.
- **Who pays.** The household pays the replacement invoices. Warranty
  reimbursement, resale value and the vehicle's downtime during a swap are
  excluded.
- The other options for valuing wear were a price per kWh discharged, which
  misses calendar ageing and replacement timing, and the draft's
  final-health difference, which after a swap compares different packs and
  can reward the household that has just bought one (VH7). Neither is added
  on top.
- No vehicle O&M, no charger replacement and no charger standby
  consumption, for the same reasons as ADR 0004 decision 7.

### V11. Ledger and results

**Ledger columns.** Decided 2026-10-06 (decision 12): vehicle rows in the
one ledger, zero without a vehicle. The other option was a second,
vehicle-only frame. One ledger keeps one closure test over one frame and
one row-write path in the day loop, as ADR 0004 C8 did. Its cost is memory
and write time for every run; PR 2 measures it, and a later PR may drop
all-zero vehicle rows from summary buffers as the PV-only summary buffer
does.

- **State rows:** `Vehicle_Plugged` (0 or 1), `Vehicle_Energy` (in site,
  Wh, 0 away), `Vehicle_Energy_Onboard` (Wh), `Vehicle_SOC_Absolute`,
  `Vehicle_SOH`, `Vehicle_PV_Origin_Energy_End`,
  `Vehicle_Grid_Origin_Energy_End`.
- **Home flows** (average W): `PV_AC_To_Vehicle`, `Grid_AC_To_Vehicle`,
  `Vehicle_Charge_Conversion_Loss`, `Vehicle_Charge_Input`,
  `Vehicle_Charge_Loss`, `Vehicle_Discharge_DC`, `Vehicle_Discharge_Loss`,
  `Vehicle_Charger_Loss`, `Vehicle_AC_To_Load`,
  `PV_Origin_Vehicle_AC_To_Load`, `Grid_Origin_Vehicle_AC_To_Load`.
- **Site boundary:** `Vehicle_Departure_Energy`, `Vehicle_Arrival_Energy`,
  `Vehicle_Departure_Shortfall`.
- **Away flows:** `Vehicle_Drive_Energy` (served), `Vehicle_Drive_Unserved`,
  `Vehicle_Away_Charge_Purchased`, `Vehicle_Away_Charge_Loss`, and the PV-
  and grid-origin shares of the driving.
- **Removals and additions:** `Vehicle_Capacity_Window_Loss`,
  `Vehicle_Replacement_Energy_Removed`, `Vehicle_Replacement_Energy_Added`,
  with the PV- and grid-origin shares of each removal, as A9 has for the
  stationary pack.

The vehicle columns join the unreleased ledger `"3.1"` (see A2), as ADR
0004 M1 did. If V2H ships after 0.8.0, they become `"3.2"` then. The
result format stays `"1"`.

**Existing columns.** `PV_AC_Export` and `Import_From_Grid` are the site's
final exchange, after the vehicle. `Houseload` stays the house load without
the vehicle. `Grid_AC_To_Battery` stays the stationary battery's. Away
purchases are never part of `Import_From_Grid`: they are a source of their
own, off the site's meter.

**Year rows** gain the vehicle's AC charged from PV and from the grid, AC to
the house load, driving served and unserved, PV used for driving, away
energy purchased and its cost at `away_charging.price`, departures short of
target (count and kWh), the vehicle's health, and its swaps (count,
replaced capacity and `Replacement_Time_Years`, as the stationary pack's).
`Self_Consumption_kWh` adds PV-origin vehicle AC to the load, an AC
quantity like its other terms; PV used for driving is stored DC energy, so
it is reported on its own.

**Emissions.** Away purchases are grid energy and count at the grid factor
in every household. Avoided emissions keep A10's net exchange, with R0's
energy in place of the load:

```text
(Import_R0 + Away_R0 − Import_From_Grid − Away − B_u) × CI + PV_AC_Export × CI_export
```

`Away` is `Vehicle_Away_Charge_Purchased`, and `B_u` the stationary
battery's unattributed energy to the load, as in A10. Without a vehicle the
formula is A10's, bit for bit. Grid energy that passes through the vehicle
and back to the house enters through `Import_From_Grid`, so its losses count
against the system. Equal starting energy does not make the households'
vehicle energy cancel, since away purchases and replacement energy differ
between them; the formula counts what each one bought.
`calculate_co2_projection` gains R0's yearly import and away purchases and
the household's away purchases as inputs.

**App result** gains a `vehicle` block: the configured modes; those year-row
totals for R0, R1 and R2; each household's replacement history (dates,
reasons, outlays) and final pack health; `P_r`, `H_r`, the wear `W_{2−1}`
and `W_{2−0}` and the outlays-only differences; the household-adjusted NPV;
the V2H benefit with its parts; and the `comparable` flags.
`provenance.vehicle` records the resolved table, the schedule hash and its
rounding.

### V12. Default behaviour, backends and entry points

Decided 2026-10-06 (decision 13).

- **No `[vehicle]`, no change.** `_simulate_core` takes the vehicle as an
  optional argument; without it the day loop's vehicle branch is skipped,
  the vehicle rows are written as zeros, `_dispatch_dc_step` and ADR 0004's
  AC step are not edited, and the PV-only path stays vectorised. Every
  existing column and value is unchanged; the App golden fixtures and the
  stored gallery results are regenerated for the new zero rows, the ledger
  version and the provenance field (`vehicle: null`) only.
- **Numba.** `_dispatch_vehicle_step` is registered with `register_jitable`.
  `_dispatch_day` gains the vehicle scalars and the schedule arrays of V3
  (empty without a vehicle), so the kernel keeps one signature. Driving and
  away charging are part of the vehicle step, since they change the
  on-board energy the next home step reads. The vehicle's degradation,
  replacement and the carry between years stay in Python. Python and Numba
  must agree bit for bit with a vehicle too.
- **Monte Carlo.** Each trajectory runs R0, R2 and, unless `reference_mode
  = "none"`, R1 on the same sampled weather and load, so both comparisons
  are paired. The load scale applies to the house load only; trips are not
  sampled. A trajectory then costs about twice the dispatch and three times
  the vehicle ageing of one without a vehicle (V8). The existing BLAST
  refusal applies: a Monte Carlo vehicle uses native ageing; requesting
  BLAST is refused, never silently substituted (A15).
- **Optimizer.** Refused with a `[vehicle]` table in 0.8.0. The vehicle is
  not a design variable, and the three households would triple every
  candidate's vehicle work. A study can sweep App runs meanwhile.

## Energy closure

Two balances close per step: the site's, in which the vehicle away is
neither load nor storage, and the vehicle's on-board balance, which follows
the pack everywhere.

1. Load supply: `Houseload = PV_AC_To_Load + Battery_AC_To_Load +
   Vehicle_AC_To_Load + Import_From_Grid − Grid_AC_To_Battery −
   Grid_AC_To_Vehicle − Inverter_Night_Consumption`.
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
5. On board: `ΔVehicle_Energy_Onboard = Vehicle_Charge_Input −
   Vehicle_Charge_Loss − Vehicle_Discharge_DC +
   Vehicle_Away_Charge_Purchased − Vehicle_Away_Charge_Loss −
   Vehicle_Drive_Energy − Vehicle_Capacity_Window_Loss −
   Vehicle_Replacement_Energy_Removed + Vehicle_Replacement_Energy_Added`.
   Crossing the site boundary is not a change on board.
6. Site boundary: `Vehicle_Energy` equals `Vehicle_Energy_Onboard` on a
   plugged-in step and 0 away. `Vehicle_Departure_Energy` is the on-board
   energy at the start of the departure step, and `Vehicle_Arrival_Energy`
   the on-board energy at the start of the arrival step.
7. Whole site: the inputs gain `Vehicle_Arrival_Energy` and, on a plugged-in
   step, `Vehicle_Replacement_Energy_Added`; the outputs gain the four home
   vehicle losses, `Vehicle_Departure_Energy`, on a plugged-in step
   `Vehicle_Capacity_Window_Loss` and `Vehicle_Replacement_Energy_Removed`,
   and `ΔVehicle_Energy`. `PV_AC_To_Vehicle` and `Grid_AC_To_Vehicle` are
   internal, grid AC enters through `Import_From_Grid` as today, and away
   flows are not in the site balance.
8. Driving: `Vehicle_Drive_Energy + Vehicle_Drive_Unserved` equals the
   step's scheduled driving, and driving never takes the on-board energy
   below `min_soc × cap_v`.
9. Away charging: the energy stored, `Vehicle_Away_Charge_Purchased −
   Vehicle_Away_Charge_Loss`, is `efficiency` times the purchase, at most
   `power_kw × park_hours × efficiency`, and never takes the on-board energy
   above the larger of `target_soc × cap_v` and its value before.
10. Origins: the three vehicle origins reconcile on board step by step, with
    driving, capacity-window and replacement removals in proportion and away
    charging added to the grid origin.
11. Rating: `PV_AC_To_Vehicle + Grid_AC_To_Vehicle ≤ P` and
    `Vehicle_AC_To_Load ≤ P`, in average W, to rounding.

## Reference cases

Every case uses a vehicle of 40 000 Wh at full health, `min_soc = 0.1`,
`max_soc = 1`, `departure_soc = 0.8` (`T` = 32 000 Wh), `v2h_min_soc = 0.3`,
battery efficiencies 0.95, and a 7400 W charger at 0.95 on the part-load
curve, hourly steps, no stationary battery and no site limit. One
full-power step stores `s_max = 7400 × 0.95 × 0.95 = 6678.5` Wh. Values
are Wh per step. VH1–VH3 are home steps; VH4–VH6 follow the vehicle away
(V6) and through a swap (V9); VH7 values wear (V10).

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

**VH4 — driving legs and away charging.** The vehicle leaves with 32 000,
of which PV origin 16 000 and grid origin 8000, on an absence of four hourly
steps with `trip_kwh` = 9 (the template: 4500 in the first and in the last
away step), and away charging with `target_soc` 0.7 (28 000), 3.7 kW and
efficiency 0.9 in the two parked steps.

| Away step | Driving served | Away purchased | Away loss | On board | PV origin | Grid origin |
|---|---|---|---|---|---|---|
| 1, first leg | 4500 | 0 | 0 | 27 500 | 13 750 | 6875 |
| 2, parked | 0 | 555.555556 | 55.555556 | 28 000 | 13 750 | 7375 |
| 3, parked | 0 | 0 | 0 | 28 000 | 13 750 | 7375 |
| 4, last leg | 4500 | 0 | 0 | 23 500 | 11 540.178571 | 6189.732143 |

`Vehicle_Departure_Energy` is 32 000 on the first away step and
`Vehicle_Arrival_Energy` 23 500 on the next home step; every home flow and
`Vehicle_Energy` are 0 while away. Step 3 holds at the target, a stopping
target, and the last leg takes the vehicle below it. PV-origin driving is
2250 + 2209.821429 = 4459.821429.

**VH5 — an unmet target and unserved driving.** `E` = 10 000 on the last
plugged step (`k` = 1): the charger runs at full power, stores 6678.5, and
the vehicle leaves with 16 678.5. `Vehicle_Departure_Shortfall` 15 321.5.
The absence has three steps and `trip_kwh` = 15 (7500 per leg).

| Away charging | After first leg | Purchased while parked | Driving served | `Vehicle_Drive_Unserved` | Arrival |
|---|---|---|---|---|---|
| none | 9178.5 | 0 | 12 678.5 | 2321.5 | 4000 |
| 0.7, 3.7 kW, 0.9 | 9178.5 | 3700 | 15 000 | 0 | 5008.5 |

Without away charging the last leg stops at `min_soc` (4000). With it, the
parked hour stores 3330 and the trip is served. The departure shortfall is
15 321.5 in both: away charging cannot meet a target that had to be met at
home.

**VH6 — a swap while away.** At a day close the vehicle is away with 20 000
on board, and its pack's health reaches 0.70 (capacity 28 000, state of
charge 0.714286). The swap books `Vehicle_Replacement_Energy_Removed`
20 000, with its origin shares, and `Vehicle_Replacement_Energy_Added`
28 571.428571, unattributed: a new 40 000 pack at the same state of charge.
The on-board balance closes with both rows; `Vehicle_Energy` stays 0 and the
site balance is unchanged.

**VH7 — wear valuation.** Pack price `C_pack = C_swap` = 40 kWh × 120 =
4800, `i` = 0.02, `ℓ` = 0, `d` = 0.03, `T` = 20, `h*` = 0.70. R1 keeps its
first pack and ends at health 0.78; R2 swaps once at `t` = 12.5 and ends at
0.88.

| Quantity | Value |
|---|---|
| `P_1` | 0 |
| `P_2` | 4248.915318 |
| `H_1` | 1053.098298 |
| `H_2` | 2369.471170 |
| `W_{2−1} = P_2 − P_1 + H_1 − H_2` | 2932.542446 |
| Outlays only, `P_2 − P_1` | 4248.915318 |

The draft's final-health formula would have given −1316.372872: it would
have credited V2H for the new pack its extra wear made the household buy.

VH1–VH6 were first computed with `_dc_ac` and `_dc_for_ac` at `567df2a`
by the script that accompanies the PR, following V5 and V6; VH7 follows
V10. Recomputed at `4a3cc04` on 2026-10-22 with the current converter
arguments, all published values are unchanged (amendment A16). The
tests compare with the full-precision values to 1e-9 relative, and the
exact landings exactly, as ADR 0004's AC7 does.

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
  charges and discharges the vehicle at home; the vehicle never exports;
  driving never takes the on-board energy below `min_soc × cap_v`.
- Q5. With no site limit and every target reachable, no departure is short
  in any mode.
- Q6. R0's vehicle (its charging, driving, away purchases and swaps) does
  not depend on the system configured.
- Q7. Energy enters or leaves the pack at a swap only through the two
  replacement rows, plugged in or away.

## Alternatives considered

- **The vehicle as a load profile with a flexible part.** Rejected: it
  cannot discharge, so it cannot be V2H, and the energy it holds would not
  close.
- **Reuse `BatteryConfig` with an availability mask.** Rejected: the
  stationary pack's capacity window, temperature derating and standby rules
  do not apply to a vehicle, and two stores in one config would change the
  stationary contract.
- **Vehicle first, or a joint step.** Rejected (V4).
- **A forecast-based vehicle planner.** Deferred: the guard is causal and
  serves the departure target; a planner is a later mode.
- **Vehicle-to-grid.** Out of scope (V1). It needs export from storage,
  market or tariff rules for it, and a different reference.
- **Arrival state of charge as an input.** Rejected (decision 4): it made
  the trip's energy depend on the departure state; the arrival state is an
  outcome.
- **Trip energy at departure, at arrival or spread uniformly.** Rejected
  for driving legs (V6, decision 6).
- **One vehicle pack for the whole project, and wear from final health
  alone.** Rejected (V9, V10, decisions 10 and 11; VH7).
- **Whole-vehicle replacement.** Rejected (V9): it needs purchase, resale
  and non-battery depreciation that are not battery wear.

## Implementation status

PRs 1–6 are implemented. App and paired Monte Carlo run complete vehicle
households; the optimizer, `daily_persistence` with a vehicle, and BLAST or
several PV inverters with Monte Carlo remain refused as specified. The plan
table below records the implementation commits; PR 6 is this uncommitted
documentation changeset. This section is the final implementation account,
with the original plan retained as history. The ADR remains **Proposed**:
implementation does not substitute for the maintainer's acceptance.

PR 6 adds the configuration guide, site/on-board ledger documentation and
results interpretation, including the model-validity and user-data limits.
The `vehicle_to_home` gallery case stores R0/R1/R2 for a three-year hourly
Porto study, using only the committed TMY and bundled H0 load and bi-hourly
schedule. Its charger premium, efficiencies, pack price and energy prices
are explicitly illustrative user inputs. The page reads stored files only;
the offline case regenerates in 2.1 seconds. CHANGELOG describes the complete
feature. The guide treats threshold replacement as study policy and states
the charging-curve, thermal, driving-template and cell-ageing assumptions.

PR 1 implements configuration, schedule resolution and the App rollout refusal.
PR 2(a) implements home dispatch in `_simulate_core` and the shared Python/Numba
day loop: stationary first, then the vehicle on the AC residual, including
missing and retired stationary packs. Those paths retain the vectorised house
rows and run the shared loop with stationary dispatch disabled; no stationary
capacity, health or standby state is invented. Without a vehicle the PV-only
path remains vectorised. Ledger 3.1 includes all V11 vehicle columns, zero
without a vehicle, plus PV/grid stored-charge and discharge-DC origin flows
for stepwise reconciliation. States are Wh, SOC and vehicle SOH are fractions,
and transfers are average W. PR 4 adds their year reductions and App results.

PR 3 adds the independent vehicle lifecycle and lifts the lower-level
replacement refusal. PR 4 lifts the App refusal. The vehicle still
has no capacity-temperature derating, indoor temperature model, standby or
resistance-fade model.
PR 2(b) implements departure and arrival bookings, the ordered driving legs,
and parked away charging in the shared Python/Numba loop. A departure books
the pre-driving on-board energy and a shortfall only above `1e-9 × cap_v`;
arrival books the pre-home-dispatch energy. Driving removes each origin in
proportion, stops at `min_soc`, and reports the rest as unserved without
borrowing from charging later in the step. Away charging uses its own power
and purchased-energy efficiency, with stored energy added to the grid origin.
Away purchases and driving never cross the site meter.

`VehicleCarry` holds on-board Wh, PV/grid origins and the previous plug state.
It is returned by `_simulate_core`, detailed runs and summary reductions, and
passed back through `vehicle_carry`. The shared `project_years` loop accepts a
lower-level vehicle config plus each `ProjectionYear.vehicle_schedule`, and
carries that state independently of a stationary pack on both frame and
aligned-summary paths. PR 4 uses independent copies for App's R0/R1/R2. Closure checks now include
identities 5–10 across home/away transitions, with Wh/W scaling, ordered
withdrawals, parked-time power limits and a full site balance excluding away
flows. VH4/VH5 and full-trip Q4/Q5 run hourly and at 15 minutes on both backends,
including multi-day, day-window and year seams.

PR 3 advances a second `DegradationLifecycle` through `DegradationDay` at
positional day closes, on real on-board SOC at home and away. Native LFP
ageing is the default; a configured BLAST cell model uses its own snapshot.
`breos.weather.build_battery_temperature_series` resolves weather, fixed and
CSV inputs with the indoor model disabled. An independently prepared
`vehicle_temperature_series` can supply vehicle temperatures alongside the
stationary input; projection years carry it on both execution paths.

Health changes the next day's capacity, targets and floors. Day close books
excess energy as `Vehicle_Capacity_Window_Loss`, with proportional origins.
A same-SOC swap explicitly removes the outgoing energy and origins and adds
the successor's energy unattributed, at home or away. Replacement uses the
shared stationary horizon guards, while a skipped swap keeps or retires the
vehicle pack. A kept pack at zero health retires too. Retirement removes its
remaining energy explicitly and disables home/away charging and discharge;
subsequent scheduled driving is unserved.

`VehicleCarry` now includes health, lifecycle/SOC/temperature endpoints,
unresolved native rainflow state, retirement state, lifetime FEC and the
project's swap history. Pending half cycles are finalized only at project
end or when a pack leaves service. The year loop preserves this state on
frame and aligned-summary paths. A run's `vehicle_end_of_life_events` uses
local step positions (also JSON records in frame attrs); carried history
uses cumulative project step positions. Core, detailed and summary results
retain the vehicle's degradation tracking separately from the stationary
tracking. Direct daily-target controllers/planners remain refused with a
vehicle, regardless of forecast or decision boundary. VH6 and Q7 cover home
and away swaps; VH1–VH5 retain their specified full-health fixture, with
separate native/BLAST ageing, parity and stationary-bit-identity tests.

PR 4 exposes the complete vehicle in App. The runner resolves its schedule
and raw vehicle temperature once, then runs independent R0/R1/R2 projections
with independent lifecycles and carry. R0 uses a zero-capacity stationary
configuration without PV equipment, conversion overrides or inverter night
draw. R1 is omitted when the reference mode is none. Static stationary
instructions run directly through the core with a vehicle; the existing
fixed-target controller produces the same instructions, but its daily-target
controller seam remains refused with a vehicle. PR 5 lifts the native-vehicle
Monte Carlo gate; the optimizer remains explicitly refused.

Each projection retains first-year ledger rows, each year's priced exchange
flows and all vehicle end-of-life events with individual booking instants.
R0's actual import and network-credit cap replace the house-only baseline on
flat, tariff and reference-tariff paths, including revalue. Vehicle year rows
report home energy, driving, away purchases, missed departures, health and
swaps. The vehicle result block contains the three household trajectories,
outlays P, pack credits H, wear and both comparisons. Top-level
`npv_savings_household_adjusted` and `v2h_benefit_npv` repeat the labelled
comparison values beside the existing system NPV, with comparable flags.
Away purchases enter avoided emissions in each household; the R2 emissions
caller now uses R0's site and away purchases. PV-origin vehicle AC delivered
to the house joins yearly/monthly self-consumption and the loss waterfall.

`provenance.vehicle` is null without a vehicle; otherwise it records the typed
configuration, resolved schedule hash, rounding and household definitions.
The six existing App goldens gain only that null provenance key. Revalue
accepts vehicle money changes only and restates every nonzero vehicle price
on a currency change. It reuses the three households' flows when dispatch
cannot change; a vehicle period-control schedule change, or changed stationary
instructions, runs the households again. A changed reference tariff is pricing
only. PR 6 documents this behaviour and the remaining entry-point limits.

PR 5 pairs independent R0/R1/R2 projections inside each Monte Carlo trajectory.
The existing weather-year and house-load sampling callback is materialised
exactly once per project year. Every household reads those same inputs, and
the trajectory's existing persistent PV model factor is shared too. R0
removes PV equipment and its night draw; load scaling changes house demand
only. Each sampled weather year supplies the vehicle's own temperature,
independently of stationary battery temperatures. All households keep their
own vehicle lifecycle and carry across project years. BLAST and several PV
inverters remain refused before weather is read.

Monte Carlo uses the same retained-flow household valuation as App, including
reference tariffs, annual network credits, replacement booking instants,
pack-only credits, away purchases, outside-charging penalties and comparable
flags. The runs table retains ordinary system `npv_savings` and adds
`npv_savings_household_adjusted`, `household_adjusted_comparable`,
`v2h_benefit_npv` and `v2h_benefit_comparable`. New `vehicle_` columns contain
R2/R0 and R2/R1 wear and outlays-only differences, adjusted away/unserved cost
differences, and the V2H operating/home/export/away/unserved savings and charger
premium. Each `vehicle_r0_`, `vehicle_r1_` and `vehicle_r2_` family includes
App's energy totals (home charge and discharge with origins, driving served
and unserved, away purchases, departure shortfalls, capacity-window loss,
replacement transfers), missed-departure and swap counts, replaced capacity,
final health and retirement, home imports/exports, avoided emissions where
configured, and discounted bills, exports, away purchases, unserved-driving
penalties, operating costs, replacement outlays P and terminal pack credits H.
Every finite new column gets the existing summary statistics and quantiles.
Vehicle provenance records the configuration, schedule hash, households and
pairing rule. No-vehicle runs keep their columns, provenance and aligned
summary execution path.

Implementation interpretations for PR 5:

- The PV uncertainty factor keeps its existing persistent per-trajectory
  child stream; it is not redrawn per household or year. Materialising the
  year callback adds no random draws. Adjacent-seed tests compare weather,
  load scale, PV output and stationary lifecycle quantities with and without
  a vehicle.
- Vehicle trajectories use the frame path so the common App valuation can
  price actual exchanges. They bypass the PV-only conversion memo; they do
  not cache or share dispatch-dependent results. Only one trajectory's three
  households are retained per worker, then released after valuation. Sampled
  year inputs are retained for that trajectory so the callback cannot resample.
- R1 is omitted when `reference_mode = "none"`. Its column family and V2H
  components are omitted, while V2H comparison scalars are NaN and the benefit
  comparable flag is null. All-missing columns have no summary. Boolean
  summaries use 0/1, so their means report the fraction comparable or retired.
- All PR 4 money conventions above apply unchanged. There is no new pricing
  rule or house-only fallback for Monte Carlo. Exact no-noise tests compare
  App's entire unrounded vehicle block and ordinary NPV with one trajectory,
  for flat and time-of-use prices, both backends, hourly and 15-minute steps,
  and no, DC or AC stationary storage.

Implementation interpretations and money conventions for PR 4:

- Replacement keys mirror the stationary policy: `vehicle.allow_terminal_replacement`
  is a bool, default true, and is used by all three households. A lower-level
  explicit override still takes precedence. Earlier project-year boundaries
  permit swaps; the configured terminal rule applies at the final close.
- The vehicle and its initial pack are already owned by every household.
  Only the stated installed charger premium joins system CAPEX at commissioning;
  it is subtracted once in the R2/R1 comparison too. There is no extra vehicle
  purchase, charger O&M, charger replacement or terminal onboard-energy credit.
- System NPV uses R0's actual home bill, with its reference tariff when set,
  and R2's import, export and existing system cash flows. It includes the
  charger premium once, and excludes vehicle invoices, credits, away purchases
  and unserved-driving penalties. Household-adjusted NPV starts from that
  ordinary NPV, independently of the optional stationary terminal-value sensitivity.
- R0's home purchases, fixed charge and network credit use the reference
  tariff's escalation when supplied; otherwise they use the system import
  escalation. R1 and R2 use system prices. Away purchases and outside-charging
  penalties use their stated common year-1 prices and the system import-price
  escalation in every household, even when R0 has a separate home tariff.
  These annual operating amounts are discounted at year end, as home bills are.
  The outside price is a money penalty only: it restores neither driving energy
  nor emissions, and it does not change dispatch.
- Every vehicle swap invoice is escalated by general inflation and replacement
  learning from t = 0 and discounted at its own simulated swap instant; multiple
  swaps are never collapsed to a mean booking time. Booking is at step end,
  year_idx + (step + 1) / n_steps, with the closing sample's date reported,
  following the stationary convention. Kept/retired events have
  zero outlay. A stated invoice may include installation; absent one, only
  nominal kWh times the required pack price is used. Replacement inventory
  additions/removals get no separate electricity charge or revenue. No equipment-price default
  or currency conversion is introduced; explicit zero prices remain valid.
- H uses the nominal pack-only price, never installation or the swap invoice,
  and is clipped individually before differences are formed. A retired pack
  has no terminal credit. The permitted edge policy h* = 1 has no health above
  its threshold and earns zero credit. Credits are valuation terms only:
  they do not enable terminal_value or enter ordinary system NPV. W and the
  discounted outlays-only differences are signed and are never clipped.
- R2/R1 operating savings include home import, fixed charges, network credits,
  export revenue, away purchases and priced unserved driving. Their configured
  system CAPEX, O&M and stationary replacements are identical and cancel; no
  second PV or stationary-pack cost is charged to V2H.
- Comparability tests total gross unserved driving over the projection, with
  a 1e-9 kWh numerical tolerance. Only more unserved driving by R2 makes a
  comparison non-comparable when no outside price is stated. A stated penalty,
  including explicit zero, makes it priced and comparable. Departure misses
  remain reported and do not independently invalidate a comparison. Missing
  R1 gives a null V2H benefit and null top-level benefit flags.
- A period window keeps energy and year-1-price bills and invoices, with no
  lifetime comparison, P/H NPV, nominal escalated invoice or discounted invoice.
  Replacement history still reports its t = 0 invoice. Vehicle monetary
  components retain full precision internally and in the detailed block;
  headline NPVs use the existing two-decimal result convention.

Implementation interpretations for PR 2(a):

- A plugged step with `steps_to_departure = 0` (an empty trip file) has no
  deadline. V1G/V2H use PV and charge periods; V2H may discharge to its floor.
  Uncontrolled charging still charges toward the target.
- The lower-level `grid_import_limit_w` argument carries the existing
  `[smart_charging]` key without constructing stationary instructions for a
  vehicle-only run. When instructions also provide a limit, the vehicle uses
  the smaller limit and leaves the stationary instructions and their hash
  unchanged. App wiring remains PR 4 work.
- Exact charge landings require that the requested stored energy fits one
  full charger step: the converter inverse saturates above its rating and
  cannot establish that an arbitrarily large target was reached.
- ADR 0004 M9 refines identity 1's night term to grid-served night draw
  (`night - min(PV AC surplus before charging, night)`). Identity 2 includes
  that PV-served draw as an output before stationary/vehicle charging and
  export. The vehicle still uses decision 16's literal deficit expression.

Implementation interpretations for PR 2(b):

- A fresh simulation starts where its first schedule step says. Starting
  already away creates no synthetic departure, shortfall or arrival at the
  initial instant, and deducts no driving that occurred before the span.
  Explicit carry supplies the previous plug state when resuming a span.
- The resolved arrays supply the remaining leg energy and parked hours;
  the carried plug state supplies the position inside an absence needed for
  boundary detection. Continuing away across a day/year seam never repeats
  departure; the next home step books arrival once, before charging.
- The away charging stopping target applies after the first driving leg,
  before the second: an initially above-target vehicle may fall below the
  target on its first leg and then charge. Away charging does not revise
  the departure shortfall already booked at the start of the absence.

Implementation interpretations for PR 3:

- PR 3 had no VehicleConfig terminal-swap key. Direct `_simulate_core` calls
  used `vehicle_allow_terminal_replacement` (default true, as for the stationary
  pack) and `vehicle_replacement_years_after_span`; `project_years` supplies
  the remaining horizon and allows earlier year-end swaps. The vehicle's
  existing minimum-remaining-years key uses the same span/year convention
  as the stationary policy. No App configuration keys are added here.
- Outgoing unresolved cycles are settled before inventory changes. Any
  energy above the outgoing pack's faded maximum is clipped and booked
  first; its remaining SOC determines the new pack's energy.
- At exactly zero outgoing capacity, SOC is undefined. An enabled swap uses
  the last measured SOC bounded to the configured vehicle window, and books
  all successor energy as replacement-added energy. A skipped kept pack at
  zero health retires instead.
- Closing ledger rows show the updated health, SOC and inventory. Lifecycle
  endpoints retain the actual pre-close samples consumed by the engine;
  the following sample sees the new capacity. Rewriting an ingested
  rainflow endpoint would break continuation and is not done. A replacement
  resets the lifecycle and starts it at the successor's same SOC.

PR 2(a) performance measured on 2026-10-23, against unchanged `2443c37`,
using the same Python 3.12 environment and offline synthetic weather.
`tools.generate_app_golden.run_scenario` ran `native_h_battery` (5 kWh)
and `native_h_pv_only` with `projection_years = 1`, Python execution, one
warm-up and five timed repetitions; the table reports medians. Cold and warm
Numba first calls use `tools.benchmark_montecarlo.measure_compilation`, with
a fresh temporary cache for the cold call and the same cache for the warm
call. These are single-machine samples, not a platform performance guarantee.

| No-vehicle measurement | Before | After |
|---|---:|---:|
| Hourly App year, stationary battery | 0.122921 s | 0.123874 s |
| Hourly App year, PV-only | 0.036595 s | 0.037687 s |
| Cold Numba first kernel call | 1.781 s | 2.369 s |
| Warm-cache Numba first kernel call | 0.170 s | 0.175 s |

PR 2(b) performance measured on 2026-10-23 against unchanged `a680092`,
using the same offline synthetic App-year setup and compilation probes as
part (a), with one warm-up and five repetitions for each Python App median.
The small App differences are within the ranges of these single-machine
samples; no-vehicle ledger values remain pinned by the golden and digest
checks. Numba first calls use an isolated temporary cache.

| No-vehicle measurement | Before part (b) | After part (b) |
|---|---:|---:|
| Hourly App year, stationary battery | 0.124105 s | 0.123167 s |
| Hourly App year, PV-only | 0.037053 s | 0.036862 s |
| Cold Numba first kernel call | 2.383 s | 2.528 s |
| Warm-cache Numba first kernel call | 0.170 s | 0.166 s |

PR 3 performance measured on 2026-10-23 against unchanged `04275e2`,
using the same offline, hourly Python App-year scenarios as above. Three
alternating before/after batches each warmed once and timed five runs;
the App figures are medians of those 15 observations. Three alternating
fresh-cache compilation probes per tree supply the Numba medians. No tests
ran concurrently. Earlier isolated batches varied together by about 30 ms
per App run, including PV-only, so the paired measurements below are used
instead. These samples show no measurable no-vehicle regression.

| No-vehicle measurement | Before PR 3 | After PR 3 |
|---|---:|---:|
| Hourly App year, stationary battery | 0.124205 s | 0.123454 s |
| Hourly App year, PV-only | 0.037863 s | 0.037595 s |
| Cold Numba first kernel call | 2.503 s | 2.512 s |
| Warm-cache Numba first kernel call | 0.169 s | 0.168 s |

PR 4 performance measured on 2026-10-23 against unchanged `01b9ba7`,
Python 3.12, offline synthetic weather, hourly steps, one project year.
The same `tools.generate_app_golden.run_scenario` wrapper (including JSON
serialization and flattening), one warm-up and five timed repetitions were
used before and after; figures are medians. Vehicle runs add the 60 kWh V2H
vehicle from the config tests, its default V1G reference and uncontrolled
R0, with native ageing. Cold/warm calls use the same isolated-cache
`tools.benchmark_montecarlo.measure_compilation` probe as earlier PRs.
No tests ran concurrently with the timed samples. The no-vehicle sample
ranges overlap; there is no material no-vehicle regression in these samples.

| Hourly App-year measurement | Before PR 4, no vehicle | After PR 4, no vehicle | After PR 4, vehicle |
|---|---:|---:|---:|
| 5 kWh stationary battery | 0.123759 s | 0.124328 s | 0.441431 s |
| PV-only system | 0.038057 s | 0.037877 s | 0.250728 s |

| Numba probe | Before PR 4 | After PR 4 |
|---|---:|---:|
| Cold first kernel call | 2.582 s | 2.474 s |
| Warm-cache first kernel call | 0.169 s | 0.164 s |

Three households cost 3.55 times the battery App-year sample and 6.62 times
the PV-only sample. In addition to two configured-system dispatches, there
are three vehicle lifecycles; a no-vehicle PV-only run keeps its vectorised
path. These measured costs replace the rough runtime estimate in V8 for
this implementation and environment.

Memory measured separately on fresh App simulations after import/timing
warm-ups, using `tracemalloc` for process-local allocations. Retained arrays
sum each first-year DataFrame's deep memory usage and every distinct retained
priced-flow array once; this excludes Python objects and model snapshots.
The three projections retain one complete first-year ledger each; later
years retain priced exchange arrays, year reductions and lifecycle carry.

| Hourly projection memory | One household, no vehicle | R0/R1/R2 with vehicle |
|---|---:|---:|
| One year: retained frame/flow arrays | 6.49 MiB | 19.47 MiB |
| One year: traced peak | 27.45 MiB | 41.14 MiB |
| Twenty years: retained frame/flow arrays | 9.03 MiB | 25.82 MiB |
| Twenty years: traced peak | 42.70 MiB | 60.43 MiB |

At twenty years, R0 retains 7.76 MiB of arrays, R1 and R2 9.03 MiB each.
Traced live allocations after the run were 9.32 MiB without a vehicle and
26.94 MiB with one. Memory tracing was not used for the runtime figures.

PR 5 performance measured on 2026-10-23, Python 3.12/native ageing, Python
dispatch, 100 trajectories of three hourly project years, seed 42, two offline
synthetic weather years, 5% house-load uncertainty and 5% persistent PV model
uncertainty. The system has eight modules, a 5 kWh stationary battery and
4,000 kWh/year house demand. Vehicle runs add the config tests' 60 kWh V2H
vehicle, uncontrolled R0 and V1G R1. Yearly tables are not collected. Each
fresh process builds a reusable year cache and warms one serial trajectory
before timing. No tests or other measurements ran concurrently with timings.

The worker count is the machine's multiprocessing default CPU count, 20,
selected explicitly with `n_procs = os.cpu_count()`; the public settings/CLI
default remains one worker. Existing pool initialization caps native BLAS and
OpenMP pools to one thread per worker (`breos.execution.limit_worker_threads`,
covered by `tests/test_worker_threads.py`), avoiding nested thread pools.

| 100 trajectories, three years each | No vehicle | R0/R1/R2 with vehicle |
|---|---:|---:|
| Timed study, excluding year-cache build | 3.869 s | 15.391 s |
| Separate year-cache build | 0.081 s | 0.080 s |
| Parent peak RSS during timing | 178.97 MiB | 230.40 MiB |
| Largest worker peak RSS during timing | 144.01 MiB | 182.11 MiB |
| Separate sampled process-tree peak PSS | 517.31 MiB | 1,269.17 MiB |
| One serial trajectory, traced allocation peak | 7.40 MiB | 54.19 MiB |
| One serial trajectory, traced live after return | 0.62 MiB | 1.33 MiB |

RSS uses `resource.getrusage` in fresh processes; worker RSS is the maximum
child high-water mark, not the sum. PSS is sampled separately every 50 ms
from `/proc/*/smaps_rollup` for the parent and its worker descendants, so
shared fork pages count once. Per-trajectory peaks use `tracemalloc` after
warm-up, include study setup and valuation, and exclude imports and the
prepared year cache. Tracing and PSS sampling are absent from timed runs.
These are single-machine samples, not a platform performance guarantee.

## Implementation plan and effort

Six PRs, about 22.5 working days for one developer, plus review and the
platform CI runs each kernel change needs; the decisions of 2026-10-07
(driving legs, away charging, vehicle pack replacement and the
replacement-aware wear valuation) added about 4 to the draft's 18. The
2026-10-22 decision to support several PV inverters adds about 0.5 day to
PR 2. These were planning estimates. The table is retained as history;
[Implementation status](#implementation-status) is the final account.

**Dependency on ADR 0004 (#419).** At `4a3cc04`, #419 is implemented,
including its amendments M1–M8 and App exposure; #417 and #418 have also
landed. PR 2's wait for #419 PR 2 and PR 4's wait for #419 PR 4 are
satisfied. V2H reuses the implemented conversion helpers, `P_wh`, the
four-array PV chain and AC step. The six PRs below retain their own order
and rollout gates, with the amendments decided on 2026-10-22, including
ledger 3.1 (see A2). VH1–VH7 have been recomputed on this base and are
unchanged; App goldens and invariants must use the current PV defaults.
ADR 0006 remains planned after this work. See the amendments for the current
edit locations. PR 2 added the no-stationary-pack routing described above.

**Rollout gates.** No merged state runs a vehicle half-built:

- PR 1 adds `VehicleConfig`, the `[vehicle]` table and the schedule, and
  refuses to simulate a vehicle.
- PR 2 lifts that refusal for a direct `_simulate_core` call only, with
  replacement off.
- PR 3 adds the vehicle's ageing and replacement; a direct call may then
  turn replacement on.
- PR 4 is the first PR that exposes `[vehicle]` in App, with R0, R1, R2 and
  every required price.

| PR | Implementation | Scope | Original effort | Main risk |
|---|---|---|---|---|
| 1 | Done: `c977927` | `VehicleConfig` and `[vehicle]` validation, with `[vehicle.away_charging]` and the replacement keys; accept `[smart_charging].grid_import_limit_w` without a stationary battery when a vehicle is configured, leaving stationary instructions unchanged (A12); schedule resolution to per-step arrays (civil time, rounding, year seam, the leg template, the trip file's drive and park segments, charging windows), its hash; simulation refused. No behaviour change. | 3 days | Civil-time edge cases (DST, year seam) and multi-day absences. |
| 2 | Done: home `a680092`, trips `04275e2` | `_dispatch_vehicle_step` with the deadline guard and exact landings; departure, driving legs, away charging and arrival; vehicle rows in ledger schema 3.1 (see A2); on-board and site balances; origins; the day-loop branch and its arguments in `_day_arguments`; Numba registration; closure helper; reference cases VH1–VH5 and invariants Q1–Q5, with Q1–Q4 also on the several-PV-inverter path (A6); parity scenarios. | 6.5 days | Stationary bit identity (Q2), and the kernel's compile time and memory with more rows. |
| 3 | Done: `01b9ba7` | The vehicle's degradation lifecycle between day windows, its temperature, capacity-window loss, replacement with the stationary rules and explicit replacement energy, plugged in or away, its carry between years; VH6 and Q7; refusal of `daily_persistence` with a vehicle. | 3.5 days | A second lifecycle and a second swap path in the day-close code without moving the stationary one. |
| 4 | Done: `5c52ba5`; golden gallery `fa6f8f8` | App: R0 as a vehicle-only projection (invariant Q6), R1 and R2; the baseline with R0's charging; away purchases; `P_r`, `H_r` and the wear `W`; the household-adjusted NPV, the V2H benefit and the `comparable` flags; `outside_charging_price`; `charger_cost` and `battery_cost_per_kwh` required and refused when missing; emissions with away purchases; year rows, the `vehicle` result block, provenance, regenerated config docs and golden provenance; VH7. | 5.5 days | A path that reaches economics before a price is checked, and three projections' memory. |
| 5 | Done: `9ae659b` | Monte Carlo: R0, R1 and R2 paired per trajectory, load scale on the house only; optimizer refusal. | 2 days | Memory and run time of three households per trajectory. |
| 6 | Done: this changeset (uncommitted; no commit yet) | Docs: configuration guide (`[vehicle]`, the leg template and its approximation, away charging, replacement as a study policy, the assumptions in V5 and V9), energy-balance and results pages, a gallery case, CHANGELOG; ADR implementation status. | 2 days | None significant. |

Risks across the work:

- **Stationary bit identity.** Order V4 makes Q2 a test, not a hope; it runs
  on PR 2 with every C10 gate of ADR 0004.
- **Performance.** A V2H run is three households (V8). The new rows add
  memory to every run; PR 2 measures it with `tools/benchmark_montecarlo.py`
  and the cold-compile time, and PR 5 measures Monte Carlo with three
  households.
- **Model validity.** The charging direction of the part-load curve (ADR
  0004 C4), the vehicle's temperature, the leg template, the use of cell
  ageing models for a vehicle pack and replacement at a fixed threshold are
  assumptions, and the docs say so.
- **Data.** Charger premiums, charger efficiencies, away charging prices
  and vehicle pack prices are user inputs; BREOS ships none.

## Amendments of 2026-10-22

Checked against `4a3cc04` (`refactor/0.8.0-deslop`). ADR 0004, including
M1–M8, is implemented there, as are the electrical work of ADR 0007–0009,
#416–#418 and the 0.8.0 refactors. These notes give the implementation base
for V1–V12 and the first three implementation PRs. The line references in **Current behaviour
this record relies on** remain at `567df2a`; every code reference in this
section is at `4a3cc04`. Where these amendments correct an implementation
assumption above, they apply. The maintainer settled the four open
questions on 2026-10-22 and confirmed the existing BLAST rule; the
[Decisions](#decisions) section records the outcomes, preserving decisions
1–14 with a dated amendment under decision 12.

- **A1. The dependencies have landed.** #419's AC step and App exposure
  are available (`breos/_dispatch.py:918-1109`,
  `breos/projection.py:56-123`, `breos/runners/app.py:620-815`). PR 2 no
  longer waits for #419 PR 2, nor PR 4 for #419 PR 4. The six V2H PRs keep
  their own rollout gates and order. The before/after baseline for Q1, Q2
  and ADR 0004's gates is now `4a3cc04`, including both topologies and the
  electrical options, rather than the pre-#419 release.
- **A2. Vehicle columns join unreleased ledger 3.1.** Decided 2026-10-22:
  follow ADR 0004 M1 and add the vehicle columns to `"3.1"`, with no new
  version for 0.8.0 (`breos/_dispatch.py:255-285`, `:313-357`). If V2H
  ships after 0.8.0, it becomes `"3.2"` then. This amends decision 12;
  PR 2's schema assertions and PR 4's goldens follow it. The result format
  remains `"1"` (`breos/result_schema.py:16`).
- **A3. The charger conversion is still PVWatts only.** `_dc_ac` and
  `_dc_for_ac` retain their names, but now require curve arrays, a curve
  row and an AC cable coefficient (`breos/_dispatch.py:113-252`). Reuse
  `_battery_inverter_charge` and `_battery_inverter_charge_input`
  (`:899-915`) for V5's charging direction, and the scalar cores for
  discharge, with empty curve arrays, row 0, cable 0 and scale 1. This is
  ADR 0004 M4's inverter model, as decision 2 requires: no measured charger
  curve, temperature derating, AC cable or night consumption is introduced.
  Its finite AC rating still becomes Wh per step.
- **A4. The AC step consumes the complete PV chain.** `_dispatch_ac_step`
  takes PV DC, PV AC, conversion loss, clipping and night draw; it returns
  the DC step's sixteen values plus `pv_ac_to_battery` and `pv_dc_charge`,
  not a separate returned PV charging-loss value
  (`breos/_dispatch.py:918-1109`). The day loop derives that loss and books
  PV origin from `pv_dc_charge` (`:1240-1356`). The four-array chain comes
  from `_pv_inverter_conversion` (`breos/battery.py:1308-1358`,
  `:2352-2363`): per-step curve row, derating, AC cable, `ac_output_scale`
  and each PV inverter's own conversion, as M2 specifies. V4 must consume
  these outputs, never reconstruct PV AC with a scalar efficiency.
- **A5. The grid keeps PV inverter night draw.** Decided 2026-10-22:
  preserve ADR 0004 M3's documented simplification. The vehicle's deficit
  is `max(0, grid_import - grid_ac - night)`; night draw still counts
  against `site_room` and stays stationary-independent (Q2)
  (`breos/battery.py:1415-1419`, `breos/_dispatch.py:1046-1048`,
  `:1322-1331`). Closure identity 1 subtracts
  `Inverter_Night_Consumption`, as the current helper does
  (`tests/energy_conservation.py:115-128`); a whole-site balance using
  total grid import includes that draw as an output. Identity 2 already
  includes PV cable/scale losses in `PV_Direct_Inverter_Loss`.
- **A6. Several PV inverters are supported.** Decided 2026-10-22: the
  vehicle uses their summed AC residual, with no stationary battery or
  beside an AC one, extending M6's implemented bus
  (`breos/battery.py:1323-1335`). A DC stationary battery beside several
  inverters stays refused (`breos/app_config.py:3544-3559`), and Monte
  Carlo keeps its refusal (`breos/montecarlo.py:522-534`). PR 2 runs Q1–Q4
  on this path, including one inverter producing while another draws at
  night; add about 0.5 day to its estimate.
- **A7. No stationary pack is not yet an AC day-loop mode.**
  `_simulate_core` routes a missing/zero usable pack straight to the
  vectorised path (`breos/battery.py:2130-2152`, `:2326-2347`); setting
  `coupling = "ac"` on a zero-capacity config does not change that.
  A retired pack also bypasses `_dispatch_day` (`:2394-2410`). PR 2 must
  arrange a vehicle step on both paths while preserving their stationary
  outputs and the no-vehicle fast path. Do not fake a positive stationary
  capacity to reach the loop: it would create health, standby and origin
  state. The AC step itself can serve an absent pack; the surrounding
  routing and stationary state handling still need work (V4, V12, Q3).
- **A8. One matrix, with explicit units.** `_ROW_COLUMNS` now combines
  `_STATE_COLUMNS` and `_LEDGER_COLUMNS`, with module-level row indices
  (`breos/_dispatch.py:290-423`), and both full and reduced summary buffers
  use that layout (`breos/battery.py:910-947`, `:997-1040`). PR 2 adds the
  vehicle states and flows there and extends summary reductions as well as
  frame writes. Vehicle stored-energy states are Wh; energy transfers,
  including departure, arrival, driving and replacement, are accumulated
  in Wh and written as average W, like stationary replacement flows
  (`breos/battery.py:1248-1254`). Closure identities 5–7 and origin
  reconciliation must multiply those flows by `hours_per_step` when
  comparing them with differences of stored-energy states.
- **A9. Ageing carries unresolved cycles.** V9's second lifecycle must
  use `DegradationDay` and `DegradationLifecycle`
  (`breos/degradation/protocol.py:226-273`), including the incremental
  native rainflow state (`:24-222`). Carry the on-board SOC endpoint,
  temperature endpoint and lifecycle snapshot across windows and years;
  finalise half cycles only at the project end or when a pack leaves
  service (`breos/battery.py:1259-1273`, `:1676-1702`,
  `breos/projection.py:1130-1154`). The stationary close still changes no
  dispatch flow except replacement/retirement; its faded capacity window
  is enforced in the next step (`breos/_dispatch.py:492-567`). V9's
  vehicle clipping at day close remains a vehicle-specific operation,
  booked in its own rows with consistent on-board state and carry.
- **A10. Replacement and terminal credit are implemented foundations.**
  The stationary swap still starts at `max_soc`
  (`breos/battery.py:1203-1256`); PR 3 needs V9's distinct same-SOC vehicle
  swap, including away swaps. Reuse the policy checks, not the stationary
  energy reset (`:1625-1639`). Retirement removes the old energy without
  adding a pack (`:1276-1305`); a kept pack also retires at zero health
  (`:1561-1584`), so a vehicle then serves no further trips. App and Monte
  Carlo now accept `battery_replacement_cost`, while terminal credit uses
  the pack-only price (`breos/projection.py:1388-1407`). V10 already
  matches this distinction. Use the existing booking and credit helpers
  (`breos/economics.py:344-383`, `:599-690`) with the vehicle's own inputs;
  its wear credit remains independent of the optional system credit.
- **A11. Configuration and temperature use the current boundaries.**
  PR 1 registers `[vehicle]` in `APP_CONFIG_FIELDS` and the nested table
  specifications, and extends `ValidatedSpecs`/`ResolvedAppConfig`
  (`breos/app_config.py:305-1299`, `:1525-1576`, `:1969-2014`,
  `:2578-2588`, `:3302-3421`). It must validate equipment, prices, mode
  combinations and the simulation refusal before input preparation.
  Temperature construction lives in `breos.weather`, including the fixed,
  weather and CSV forms (`breos/weather.py:1709-1786`), not the deprecated battery-module
  alias. Load preparation is `breos/app_inputs.py:538-556`. The common
  weather timestamp parser (`breos/weather.py:437-472`) is not a vehicle
  trip parser: V3's configured-zone, DST and conservative-rounding rules
  still need their own resolution.
- **A12. One site-limit key; stationary instructions unchanged.** Decided
  2026-10-22: `[smart_charging].grid_import_limit_w` stays the one key.
  PR 1 accepts it without a stationary battery when a vehicle is configured
  (`breos/app_config.py:2459-2554`, `:2591-2626`); no new key is added.
  Stationary instructions and their hashes stay unchanged, including AC
  efficiency None/NaN and controller binding
  (`breos/_dispatch.py:1497-1570`, `breos/_controller.py:307-394`).
  `daily_persistence` remains refused with a vehicle for every forecast
  and decision boundary (`breos/smart_charging.py:195-275`).
- **A13. R0 must reach every pricing path.** `project_years` is the shared
  App/Monte Carlo/optimizer loop, with separate frame and aligned-summary
  runners and `CarryState` (`breos/projection.py:126-225`, `:902-1012`,
  `:1026-1262`). Extend it for independent vehicle carry in each household.
  `Baseline_Import_Cost` still prices `Houseload` (`:379-387`); R0's actual
  import must replace that baseline in tariff, reference-tariff, flat-price
  and annual-network-credit paths. Ageing means R0's charging is not one
  first-year profile replayed unchanged. R0 must also omit PV equipment and
  its night draw, rather than only zeroing the PV input
  (`breos/battery.py:1355-1358`). Extend the emissions caller as well as
  the helper (`breos/economics.py:904-935`, `breos/emissions.py:61-114`).
- **A14. Revaluation must retain the three households.** App now reprices
  from retained per-period/per-step flows, or simulates again when a tariff
  can change dispatch (`breos/runners/app.py:818-826`, `:873-967`). With a
  vehicle, that choice must also inspect its tariff-period controls even
  without stationary smart charging. Repricing needs R0's own import in
  each year, R1/R2's exchange, away purchases and separate swap histories;
  it must not restore the current house-only baseline. Vehicle prices use
  the run's currency, and currency changes must restate its non-zero
  amounts as for other equipment (`breos/app.py:84-125`). No new equipment
  price default is implied.
- **A15. Pair Monte Carlo inputs; retain the existing BLAST rule.** Draw
  each year's weather, house-load scale and trajectory PV factor once for
  R0/R1/R2 (`breos/montecarlo.py:722-775`); its PV-chain memo remains
  PV-only (`:657-699`). Confirmed 2026-10-22, with no new decision: App's
  existing BLAST-with-Monte-Carlo refusal applies to the vehicle too
  (`breos/app_config.py:2903-2913`). A Monte Carlo vehicle uses native
  ageing; requesting BLAST is refused, never silently substituted. Add the
  optimizer's vehicle refusal before its shared study/scoring path
  (`breos/optimization_config.py:484-522`,
  `breos/optimization.py:733-928`).
- **A16. PV defaults and the reference cases.** `solar_position = "auto"`
  is now the default: it uses the weather's stated radiation time basis,
  falling back to interval-start with a warning when that basis is absent
  (`breos/pv/model_options.py:70-96`). PV preparation uses the single array
  engine (`breos/app_inputs.py:474-535`, `breos/solar.py:1526-1574`).
  Future App goldens and Q1–Q3 compare against these defaults, including
  #417's weather handling; they must not restore the old solar position.
  VH1–VH7 were recomputed with the supplied script adapted only for the
  new converter arguments and import path. Every published number is
  unchanged, including the exact target landings: these cases supply PV
  DC directly (VH1 uses an unbounded flat PV inverter), so neither the PV
  defaults nor the new optional electrical effects change their inputs.
  This is a calculation of the proposal, not a test of implemented V2H.

Current edit locations for the first three PRs, at `4a3cc04`:

| PR | Existing boundary to extend | File:line range |
|---|---|---|
| 1 | Lower-level config beside `BatteryConfig` | `breos/battery.py:85-349` |
| 1 | App field/table registry, validation and resolution | `breos/app_config.py:305-1299`, `:1525-1576`, `:1969-2014`, `:2578-2588`, `:3302-3421` |
| 1 | Prepared inputs and calendar for schedule resolution | `breos/app_inputs.py:68-84`, `:561-602`; `breos/battery.py:494-633` |
| 2 | Converter helpers and stationary steps beside the new vehicle step | `breos/_dispatch.py:113-252`, `:658-1109` |
| 2 | Row layout, day loop, shared backend arguments and registration | `breos/_dispatch.py:290-423`, `:1112-1442`, `:1445-1571`; `breos/_numba_dispatch_kernels.py:36-56` |
| 2 | Buffers, PV conversion, no-pack/retired routing and day orchestration | `breos/battery.py:910-1040`, `:1308-1448`, `:1909-2494` |
| 2 | Energy and origin closure assertions | `tests/energy_conservation.py:63-202`, `:218-254` |
| 3 | Lifecycle construction, close, cycle settlement, replacement, retirement and snapshot | `breos/battery.py:805-859`, `:1203-1305`, `:1451-1639`, `:1676-1702`; `breos/degradation/protocol.py:226-273` |
| 3 | Physical carry and project-year guards; vehicle temperature | `breos/projection.py:126-225`, `:1026-1262`; `breos/weather.py:1709-1786` |

## Decisions

Taken by the maintainer on 2026-10-06, and for decisions 4, 6, 10 and 11
on 2026-10-07, after a second design review. The body sections follow them.
Decisions 4, 6, 10 and 11 changed the proposal. On 2026-10-22 the
maintainer amended decision 12, settled the four amendment questions in
decisions 15–18 and confirmed the existing BLAST rule below.

1. **Dispatch order** (2026-10-06): the stationary battery first, the
   vehicle second on the AC residual; no storage-to-storage transfer (V4).
2. **Charger model** (2026-10-06): ADR 0004's battery inverter model in both
   directions; rating and efficiency required (V2, V5).
3. **Schedule** (2026-10-06): weekday and weekend times or a trip file,
   with conservative whole-step rounding (V3).
4. **Energy while away** (2026-10-07, changed): `trip_kwh` is the gross
   energy withdrawn from the pack, driving losses included. Optional away
   charging tops up toward a stopping target in parked windows, with a power
   limit, an efficiency at the purchased-energy boundary and a price (0 for
   free charging at work); its purchases are priced the same in R0, R1 and
   R2, tracked apart from site import and counted in emissions. The arrival
   state of charge is an outcome; the arrival-state input and its
   conversion are removed. Away charging and `outside_charging_price` are
   distinct (V2, V3, V6, V8, V11).
5. **Departure target** (2026-10-06): a hard constraint through the deadline
   guard; misses reported, not refused; optional `outside_charging_price`;
   the `comparable` flag. Away charging cannot meet a target that had to be
   met before departure (V5, V6, V8).
6. **Trip timing** (2026-10-07, changed): trip energy is withdrawn over
   driving legs, by default half in the first away step and half in the
   last, parked between; a one-step absence combines them; a trip file can
   give legs, durations and charging windows; multi-day trips carry over
   days and years. Driving never goes below `min_soc`, never borrows from
   later charging, and its unserved remainder is reported. The intervals are
   computed once. It is an approximation (V3, V6).
7. **Smart charging** (2026-10-06): vehicle periods in `[vehicle]`;
   `[smart_charging]` for the stationary battery only; `daily_persistence`
   refused with a vehicle (V7).
8. **Households** (2026-10-06): R0 without a system and with uncontrolled
   charging; R1 with `reference_mode`, `"v1g"` by default; R2 the study;
   all three run in App and paired in Monte Carlo (V8, V12).
9. **Degradation** (2026-10-06): a second lifecycle on the real on-board
   state of charge (V9).
10. **Vehicle pack replacement** (2026-10-07, changed): the pack is replaced
    at `eol_percentage` with the stationary pack's rules, the same policy in
    R0, R1 and R2 and independent swap dates; replacement energy is explicit
    in the on-board balance, also away; replacement is a study policy, not a
    prediction (V9).
11. **Costs and wear** (2026-10-07, changed): one `charger_cost`, the
    installed premium over the reference charger, never counted twice; the
    system NPV unchanged, with a household-adjusted NPV (R2 against R0) and
    the V2H benefit (R2 against R1) beside it; wear
    `W = P_2 − P_1 + H_1 − H_2` from replacement outlays at their swap
    instants and pack-only terminal credits, each credit clipped and the
    difference not, with the outlays-only difference also reported; the
    household pays, and warranty, resale and downtime are excluded (V10).
12. **Ledger** (2026-10-06): vehicle rows in the one ledger, schema 3.2
    (V11).

    Amended 2026-10-22: the vehicle columns join unreleased ledger 3.1,
    as ADR 0004 M1 did; no 3.2 for 0.8.0. If V2H ships after 0.8.0,
    it becomes 3.2 then (A2, decision 15).
13. **Entry points** (2026-10-06): App and Monte Carlo; the optimizer
    refuses `[vehicle]` (V12).
14. **Target release** (2026-10-06): 0.8.0, after #419 (see
    [Implementation plan and effort](#implementation-plan-and-effort)).

15. **Ledger version** (Decided 2026-10-22): the vehicle columns join
    unreleased ledger 3.1, as ADR 0004 M1 did; no 3.2 for 0.8.0. If V2H
    ships after 0.8.0, it becomes 3.2 then. This amends decision 12 (A2).
16. **PV inverter night draw** (Decided 2026-10-22): the grid keeps the
    draw, as ADR 0004 M3. The vehicle's discharge deficit is
    `max(0, grid_import - grid_ac - night)`, and closure identity 1
    subtracts `Inverter_Night_Consumption`. This is a documented
    simplification (A5).
17. **Several PV inverters** (Decided 2026-10-22): supported. The vehicle
    uses the summed AC residual, with no stationary battery or beside an
    AC one. A DC stationary battery beside several inverters stays refused,
    and Monte Carlo keeps its refusal of several inverters. PR 2 runs
    Q1–Q4 on this path, adding about 0.5 day to its estimate (A6).
18. **Site import limit** (Decided 2026-10-22):
    `[smart_charging].grid_import_limit_w` stays the one key, accepted
    without a stationary battery when a vehicle is configured. PR 1 makes
    this validation change; stationary instructions do not change. No new
    key is added (A12).

**BLAST in Monte Carlo — existing rule, confirmed 2026-10-22.** No new
decision is needed: App refuses BLAST with Monte Carlo, so a Monte Carlo
vehicle uses native ageing. Requesting BLAST is refused; nothing silently
substitutes an engine (A15).

## Consequences

- A household with a vehicle can be simulated in App with uncontrolled,
  smart (V1G) or V2H charging, with driving and charging away from home,
  and its V2H benefit is reported against the same household without V2H,
  including the vehicle's extra wear valued through its replacements.
- The no-system baseline buys the vehicle's charging too, so NPVs with a
  vehicle compare like with like. The system NPV keeps its meaning; a
  labelled household-adjusted NPV adds the vehicle's wear and away
  purchases.
- Runs without `[vehicle]` keep every existing value bit for bit; the
  vehicle columns join ledger 3.1 (see A2), and the result format stays
  "1". If V2H ships after 0.8.0, its ledger version becomes 3.2 then.
- The stationary battery's dispatch does not change when a vehicle is
  added.
- A V2H run simulates three households; Monte Carlo pairs them per
  trajectory.
- Vehicle-to-grid, a vehicle planner, whole-vehicle replacement, warranty
  and resale, charger power reduction by the network operator and vehicle
  thermal management stay outside BREOS.
