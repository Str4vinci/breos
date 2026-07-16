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

BREOS 0.3.x implements **DC-coupled/hybrid dispatch only**. PV and the
battery share one inverter AC nameplate. `BatteryConfig(dc_coupled=False)`
and `App({..., "dc_coupled": False})` raise rather than silently running the
DC model under an AC-coupled label.

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

## Ledger schema

| Column | Unit/basis | Definition |
|---|---|---|
| `PV_DC` | W, DC | PV generated before dispatch |
| `PV_DC_To_Inverter` | W, DC | Direct PV entering the inverter |
| `PV_DC_To_Battery` | W, DC | Charge-path input, before charge loss |
| `PV_DC_Curtailed` | W, DC | PV that cannot be routed |
| `PV_AC_To_Load` | W, AC | Direct PV delivered to load |
| `PV_AC_Export` | W, AC | Direct PV exported (`Sell_To_Grid` alias) |
| `Battery_Charge_Stored` | W-equivalent | Increase due to charging after charge loss |
| `Battery_Discharge_DC` | W-equivalent, stored DC | Energy removed from storage |
| `Battery_AC_To_Load` | W, AC | All battery energy delivered to load |
| `Battery_AC_To_Load_PV` | W, AC | PV-origin share of battery delivery |
| `PV_Direct_Inverter_Loss` | W | Direct-PV inverter conversion loss |
| `Battery_Inverter_Loss` | W | Battery-discharge inverter loss |
| `Battery_Charge_Loss` / `Battery_Discharge_Loss` | W | Cell conversion losses |
| `Standby_Loss` | W | Storage standby loss |
| `Capacity_Window_Loss` | W | Energy explicitly removed when temperature/SOH lowers `Emax` |
| `Battery_Energy_Beginning` / `Battery_Energy_End` | Wh | Stored energy at interval boundaries |
| `Battery_Energy_Delta` | W-equivalent | End minus beginning, including explicit boundary adjustments |

PV-origin inventory begins at zero at the reporting boundary and is mixed
proportionally with stored energy. This prevents initial SOC from being
credited as PV and makes ending PV inventory visible rather than crediting it
as self-consumption.

App and Monte Carlo projections carry both total stored energy and PV-origin
inventory from one simulated year into the next. They do not reset the battery
to a free full state at calendar boundaries.

## Compatibility fields and KPIs

`PV_Production` is retained as an AC-equivalent compatibility field. With a
finite inverter it is non-curtailed PV minus the explicit direct-PV inverter
loss; lower-level callers that omit a nameplate retain the exact legacy
`(PV_DC − PV_DC_Curtailed) × inverter_efficiency` calculation. It is not
physical AC delivery through storage, and new KPIs do not derive from it.

Self-consumed PV is `PV_AC_To_Load + Battery_AC_To_Load_PV`. Usable system AC
generation is self-consumed PV plus `PV_AC_Export`. Grid independence is
`1 − Import_From_Grid / Houseload`.

The ledger enforces PV routing, AC load, battery state-transition, inverter
sub-balances, and whole-system conservation at each timestep and annually,
including non-zero ending SOC and explicit capacity-window/replacement terms.
