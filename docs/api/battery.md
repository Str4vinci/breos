# Battery

Configuration, indoor temperature modelling, and the calendar and cycle
degradation primitives used by the energy balance.

## Configuration

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.battery.BatteryConfig
```

## Temperature model

The indoor temperature model couples ambient air temperature to a damped
indoor series — relevant for calendar aging, which is strongly temperature
dependent.

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.battery.apply_indoor_temperature_model
   breos.battery.compute_cell_temperature
```

## Cycle detection

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.battery.detect_cycles_rainflow
   breos.battery.detect_half_cycles_from_soc_series
   breos.battery.compute_halfcycle_energy_throughput
```

## Degradation primitives

Low-level update functions that the energy balance loop calls each
timestep. Use these directly only when reproducing or critiquing the
degradation model.

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.battery.update_battery_soc
   breos.battery.update_battery_soh_calendar
   breos.battery.update_battery_soh_cyclewise
   breos.battery.update_battery_resistance_calendar
   breos.battery.update_battery_resistance_cyclewise
   breos.battery.resistance_to_efficiency
```
