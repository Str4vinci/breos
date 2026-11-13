# Energy balance

Per-timestep energy-flow accounting for PV, load, battery, and grid. The
energy-balance engine runs the same way for PV-only and PV+battery systems —
when no battery is configured, it simply skips the storage path.

## Main entry point

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.battery.simulate_energy_balance
```

The function returns a five-tuple of `(results_df, total_pv_wh,
summary_df, n_replacements, degradation_df)`, with the degradation state as
a sixth item when `return_degradation_state=True`. Battery-specific outputs
are empty when running without storage.

The physics carries no money. A replacement is reported where it happens:
`Battery_Replaced` marks the step, and `Battery_Replaced_Capacity_Wh` holds
the nominal capacity swapped in (ledger schema 3.0). The economics prices it
at `costs["replacement_cost_each"]`; see
{py:func}`~breos.economics.replacement_event_cost`.
`BatteryConfig(allow_terminal_replacement=False)` skips a replacement in the
final degradation period of the call's span, which ends on its last step.
`BatteryConfig(replacement_min_remaining_years=...)` skips any replacement
that leaves the new pack less than that many project years: the call's span
counts as one project year, and `replacement_years_after_span` more follow
it. See
[Battery replacement at the end of the horizon](../getting-started/configuration.md#battery-replacement-at-the-end-of-the-horizon).
`BatteryConfig(skipped_replacement_action="retire")` switches off a pack
whose replacement is skipped: from the next step the span dispatches as a
PV-only system. A later span continued from its returned degradation state
stays retired, or takes `battery_retired=True`.
Each end-of-life crossing, replaced or not, is a
{py:class}`~breos.battery.EndOfLifeEvent`: the summary's
`end_of_life_events`, and on the detailed path the degradation frame's
`attrs["end_of_life_events"]`, as JSON-safe
{py:meth}`~breos.battery.EndOfLifeEvent.to_record` dicts (pandas writes
attrs as JSON in `to_parquet`).

## Physical boundary and coupling

BREOS currently implements **DC-coupled/hybrid dispatch only**. PV and the
battery share one inverter AC nameplate. This is the only supported App
dispatch model; there is no configuration option to select AC coupling.

Inputs and flow columns are average power in W over each interval. Dispatch
converts them once to Wh (`W × interval hours`), applies all limits and
conservation equations in energy units, then converts flows back to W.
Stored-energy state columns remain Wh.

Direct PV and battery discharge use the same PVWatts part-load inverter
curve as `breos.solar.dc_to_ac`. The configured AC nameplate remains shared:
PV output consumes headroom before battery discharge, and combined delivery
cannot exceed the rating. A lower-level `BatteryConfig` with no inverter
nameplate keeps the legacy unbounded flat-efficiency fallback because an
inverter loading fraction cannot be defined without rated power.

## DC routing and limits

Direct PV serves AC load first. Surplus PV DC charges the battery before
export; remaining DC exports within unused inverter headroom. Only DC that
cannot serve load, enter storage, or export is curtailed. PV routed to the
battery has not crossed the inverter and is never classified as clipping.

`max_charge_power_w` limits DC input to the battery path before charge loss.
`max_discharge_power_w` limits battery AC delivered to load after cell and
inverter losses. Both limits scale with the timestep. `None` means unlimited
for backward compatibility; users should configure product nameplate limits.

`power_limit_c_rate` replaces both with one limit on the stored energy, the
way a cell current rating works. At 1 C a 5 kWh pack stores or releases at
most 5 kW in either direction. Its DC input while charging is higher by the
charge loss, and its AC output while discharging is lower by the discharge and
inverter losses.

## Dispatch instructions and grid charging

`simulate_energy_balance(..., dispatch_instructions=...)` takes a
{py:class}`~breos.dispatch_instructions.DispatchInstructions`, which gates
the greedy step without replacing it. Its first three fields are arrays with
one entry per simulation step; the last two are scalars.

- `discharge_allowed`: whether the battery may discharge in that step.
- `reserve_fraction`: the usable fraction kept before discharge. The battery
  discharges down to `emin + f × (emax − emin)`, not to `emin`.
- `grid_target_fraction`: a grid-charge target as a usable fraction, or NaN
  for no grid charge in that step.
- `grid_charge_efficiency`: the hybrid inverter's AC-to-DC conversion on the
  charging path.
- `grid_import_limit_w`: the site import limit that grid charging respects,
  counting the load's own import against it (`math.inf` for none). Load
  import itself is never cut.

A step that allows discharge and has a grid target must keep
`reserve_fraction` at or above `grid_target_fraction`, so the target is also
the discharge floor; this is how `overlap_policy = "hold_target"` runs. A
step never charges from the grid and discharges at once. Omitting the
instructions, or passing `DispatchInstructions.noop(n)`, is greedy
self-consumption, bit for bit.

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.dispatch_instructions.DispatchInstructions
```

Fractions apply to each step's capacity window, which scales with the
temperature capacity factor every step and with SOH every day. The same
target therefore names less energy on a cold night and falls as the pack
fades.

Grid charging runs after PV allocation in the same step. Grid AC `a` becomes
DC charge input `a × grid_charge_efficiency`, which then passes through the
cell charge efficiency like PV charge input. It is booked as
`Battery_Charge_Input`, so cell losses, self-heating and both aging engines
see it. PV keeps priority on every limit the two share:

- the charge-input limit (`max_charge_power_w`) and the stored-energy limit
  (`power_limit_c_rate`) bound PV and grid charge together;
- grid-charge AC input is at most the inverter AC rating minus the step's PV
  AC output. This summed-throughput rule is conservative: a real hybrid
  inverter may net the two flows internally;
- grid-charge import is at most `grid_import_limit_w` minus the step's load
  import.

There is no grid charging in a step that exports PV or discharges the
battery. Grid-charge AC is part of `Import_From_Grid`.

| Column | Unit/basis | Definition |
|---|---|---|
| `Grid_AC_To_Battery` | W, AC | Grid import for charging, part of `Import_From_Grid` |
| `Grid_DC_To_Battery` | W, DC | That import after AC-to-DC conversion, part of `Battery_Charge_Input` |
| `Grid_Charge_Conversion_Loss` | W | `Grid_AC_To_Battery − Grid_DC_To_Battery` |
| `Grid_Origin_Battery_Charge_Stored` | W-equivalent | Grid charge stored after charge loss |

## Ledger schema

| Column | Unit/basis | Definition |
|---|---|---|
| `PV_DC` | W, DC | PV generated before dispatch |
| `PV_DC_To_Inverter` | W, DC | Direct PV entering the inverter |
| `PV_DC_To_Battery` | W, DC | Charge-path input, before charge loss |
| `PV_DC_Curtailed` | W, DC | PV that cannot be routed |
| `PV_AC_To_Load` | W, AC | Direct PV delivered to load |
| `PV_AC_Export` | W, AC | Direct PV exported |
| `Battery_Charge_Stored` | W-equivalent | Increase due to charging after charge loss |
| `Battery_Discharge_DC` | W-equivalent, stored DC | Energy removed from storage |
| `Battery_AC_To_Load` | W, AC | All battery energy delivered to load |
| `PV_Direct_Inverter_Loss` | W | Direct-PV inverter conversion loss |
| `Battery_Inverter_Loss` | W | Battery-discharge inverter loss |
| `Battery_Charge_Loss` / `Battery_Discharge_Loss` | W | Cell conversion losses |
| `Standby_Loss` | W | Storage standby loss |
| `Capacity_Window_Loss` | W | Energy explicitly removed when temperature/SOH lowers `Emax` |
| `Battery_Energy_Beginning` / `Battery_Energy_End` | Wh | Stored energy at interval boundaries |
| `Battery_Energy_Delta` | W-equivalent | End minus beginning, including explicit boundary adjustments |

Ledger schema 2.0 removed four columns that repeated another under a second
name. Read the second name of each pair instead: `Sell_To_Grid` →
`PV_AC_Export`, `PV_Curtailment` → `PV_DC_Curtailed`, `Battery_Standby_Loss`
→ `Standby_Loss`, and `Battery_AC_To_Load_PV` →
`PV_Origin_Battery_AC_To_Load`.

## Energy origins

Stored energy has three origins: PV, grid, and an unattributed remainder. A
fresh battery starts full with unattributed energy, and a replacement pack's
energy is unattributed too, so initial SOC is never credited as PV. PV
charging adds to the PV origin, and grid charging to the grid origin.
Discharge, standby loss, capacity-window loss and replacement each take from
all three origins in proportion to their shares at the start of that
operation. A step either charges or discharges the battery, never both, so one
share per step is exact.

| Column | Unit/basis | Definition |
|---|---|---|
| `Battery_PV_Origin_Energy_Beginning` / `_End` | Wh | PV-origin stored energy at interval boundaries |
| `Battery_Grid_Origin_Energy_Beginning` / `_End` | Wh | Grid-origin stored energy at interval boundaries |
| `PV_Origin_Battery_Charge_Stored`, `Grid_Origin_Battery_Charge_Stored` | W-equivalent | Each source's charge stored after charge loss |
| `PV_Origin_Battery_Discharge_DC`, `Grid_Origin_Battery_Discharge_DC` | W-equivalent, stored DC | Each origin's share of `Battery_Discharge_DC` |
| `PV_Origin_Battery_AC_To_Load`, `Grid_Origin_Battery_AC_To_Load` | W, AC | Each origin's share of `Battery_AC_To_Load` |
| `PV_Origin_Standby_Loss`, `Grid_Origin_Standby_Loss` | W | Each origin's share of `Standby_Loss` |
| `PV_Origin_Capacity_Window_Loss`, `Grid_Origin_Capacity_Window_Loss` | W | Each origin's share of `Capacity_Window_Loss` |
| `PV_Origin_Replacement_Energy_Removed`, `Grid_Origin_Replacement_Energy_Removed` | W-equivalent | Each origin's share of `Battery_Replacement_Energy_Removed` |

The unattributed share of any flow is its total minus the PV and grid shares,
and the unattributed balance is `Battery_Energy` minus both origin balances.
Each origin reconciles step by step from these columns alone, to rounding:
the ending balance is the beginning balance plus charge stored, minus discharge, standby,
capacity-window and replacement removal. Only PV-origin discharge counts as
self-consumption.

App and Monte Carlo projections carry total stored energy and both origin
balances from one simulated year into the next. They do not reset the battery
to a free full state at calendar boundaries.

## Compatibility fields and KPIs

`PV_Production` is retained as an AC-equivalent compatibility field. With a
finite inverter it is non-curtailed PV minus the explicit direct-PV inverter
loss; lower-level callers that omit a nameplate retain the exact legacy
`(PV_DC − PV_DC_Curtailed) × inverter_efficiency` calculation. It is not
physical AC delivery through storage, and new KPIs do not derive from it.

Self-consumed PV is `PV_AC_To_Load + PV_Origin_Battery_AC_To_Load`. Usable system AC
generation is self-consumed PV plus `PV_AC_Export`. Grid independence is
`1 − Import_From_Grid / Houseload`.

The ledger enforces PV routing, AC load, battery state-transition, inverter
sub-balances, and whole-system conservation at each timestep and annually,
including non-zero ending SOC and explicit capacity-window/replacement terms.
