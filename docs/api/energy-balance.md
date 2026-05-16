# Energy balance

Per-timestep dispatch of PV, load, battery, and grid. The dispatch engine
runs the same way for PV-only and PV+battery systems — when no battery is
configured, it simply skips the storage path.

## Main entry point

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.battery.simulate_energy_balance
```

The function returns a six-tuple of `(results_df, total_pv_wh,
summary_df, total_replacement_cost, n_replacements, degradation_df)`.
Battery-specific outputs are empty when running without storage.

## Coupling

Whether the inverter is DC-coupled (hybrid) or AC-coupled affects when
losses are applied. The {py:class}`~breos.BatteryConfig` `dc_coupled` flag
controls this — see the [Battery reference](battery.md) for the full
configuration.
