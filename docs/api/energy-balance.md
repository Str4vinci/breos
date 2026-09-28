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

The function returns a six-tuple of `(results_df, total_pv_wh,
summary_df, total_replacement_cost, n_replacements, degradation_df)`.
Battery-specific outputs are empty when running without storage.

## Physical boundary and coupling

BREOS currently implements **DC-coupled/hybrid dispatch only**. PV and the
battery share one inverter AC nameplate. `BatteryConfig(dc_coupled=False)` and
`App({..., "dc_coupled": False})` raise rather than silently running the DC
model under an AC-coupled label.

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
charging adds to the PV origin. Nothing charges the grid origin yet: it holds
only energy carried in with `initial_grid_origin_energy_wh`, until grid
charging lands.
Discharge, standby loss, capacity-window loss and replacement each take from
all three origins in proportion to their shares at the start of that
operation. A step either charges or discharges the battery, never both, so one
share per step is exact.

| Column | Unit/basis | Definition |
|---|---|---|
| `Battery_PV_Origin_Energy_Beginning` / `_End` | Wh | PV-origin stored energy at interval boundaries |
| `Battery_Grid_Origin_Energy_Beginning` / `_End` | Wh | Grid-origin stored energy at interval boundaries |
| `PV_Origin_Battery_Charge_Stored` | W-equivalent | PV charge stored after charge loss |
| `PV_Origin_Battery_Discharge_DC`, `Grid_Origin_Battery_Discharge_DC` | W-equivalent, stored DC | Each origin's share of `Battery_Discharge_DC` |
| `PV_Origin_Battery_AC_To_Load`, `Grid_Origin_Battery_AC_To_Load` | W, AC | Each origin's share of `Battery_AC_To_Load` |
| `PV_Origin_Standby_Loss`, `Grid_Origin_Standby_Loss` | W | Each origin's share of `Standby_Loss` |
| `PV_Origin_Capacity_Window_Loss`, `Grid_Origin_Capacity_Window_Loss` | W | Each origin's share of `Capacity_Window_Loss` |
| `PV_Origin_Replacement_Energy_Removed`, `Grid_Origin_Replacement_Energy_Removed` | W-equivalent | Each origin's share of `Battery_Replacement_Energy_Removed` |

The unattributed share of any flow is its total minus the PV and grid shares,
and the unattributed balance is `Battery_Energy` minus both origin balances.
Each origin reconciles step by step from these columns alone, to rounding:
the ending balance is the beginning balance plus charge stored (PV only, for
now), minus discharge, standby, capacity-window and replacement removal. Only PV-origin discharge counts as
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
