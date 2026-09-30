# Battery

Configuration, indoor temperature modelling, and the calendar and cycle
degradation primitives used by the energy balance.

The default calendar model, `naumann_lam_field_calibrated`, maps to the v1
field-calibrated parameters. `naumann_lam_field_calibrated_v1` is the explicit
alias. `naumann_lam_field_calibrated_v2` selects the v2 field-calibrated fit
with Lam `Ea`/`n` fixed and `k0`/`b` fitted to field data.

## Configuration

BREOS currently supports stationary DC-coupled/hybrid batteries only.
AC-coupled dispatch is not implemented and fails explicitly. Optional charge
and discharge nameplate limits default to unlimited for backward
compatibility; configure them for realistic sizing studies. Charge power is
measured at the DC charge-path input. Discharge power is measured as AC
delivered to load.

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.battery.BatteryConfig
```

## Temperature model

The indoor temperature model couples ambient air temperature to a damped
indoor series — relevant for calendar aging, which is strongly temperature
dependent.

Temperature- or SOH-driven reductions in the maximum energy window are
reported as `Capacity_Window_Loss`; they are not silently discarded or folded
into standby loss. Resistance calendar aging uses daily mean cell temperature
and daily mean absolute SOC.

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.battery.apply_indoor_temperature_model
   breos.battery.compute_cell_temperature
```

## Degradation primitives

Low-level update functions for the degradation model. The energy balance
ages the pack once per daily degradation window, not each timestep. With the
native engine, it calls `update_battery_soh_calendar` once per window, with that window's mean cell
temperature and mean absolute SOC. When resistance fade is enabled, it also
calls the two resistance functions once per window. It does not call
`update_battery_soh_cyclewise`: its cycle step takes the window's cycles from
an incremental rainflow counter. `update_battery_soh_cyclewise` is a
standalone equivalent of that cycle step for one whole SOC series. It closes
the rainflow residue at the end of the series, so calling it once per day
does not reproduce a simulation. Use these functions directly only when
reproducing or critiquing the degradation model.

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.battery.update_battery_soh_calendar
   breos.battery.update_battery_soh_cyclewise
   breos.battery.update_battery_resistance_calendar
   breos.battery.update_battery_resistance_cyclewise
   breos.battery.resistance_to_efficiency
```
