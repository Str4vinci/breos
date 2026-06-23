# Plotting

Publication-ready matplotlib figures grouped by what they visualize.
All functions write a PNG to a results directory and accept optional
styling overrides.

## Time series

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.plotting.plot_timeseries
   breos.plotting.plot_monthly_balance
   breos.plotting.plot_monthly_comparison
   breos.plotting.monthly_graphs
   breos.plotting.weekly_graphs
   breos.plotting.yearly_graphs
```

## Cost and breakeven

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.plotting.plot_breakeven
   breos.plotting.plot_breakeven_two
   breos.plotting.plot_breakeven_comparison
   breos.plotting.create_cost_plots
```

## Battery degradation

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.plotting.plot_battery_soh_timeseries
   breos.plotting.plot_resistance_and_efficiency
   breos.plotting.plot_cell_temperature
   breos.plotting.degradation_plots
```

## Tilt and azimuth optimization

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.plotting.plot_tilt_optimization
   breos.plotting.plot_azitilt_ew_1d
   breos.plotting.plot_azitilt_landscape_2d
   breos.plotting.plot_azitilt_landscape_3d
```

## Pareto front

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.plotting.plot_pareto_front_analysis
```

## Sensitivity and Monte Carlo

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.plotting.plot_calendar_aging_sensitivity
   breos.plotting.plot_temperature_sensitivity_comparison
   breos.plotting.plot_montecarlo_simulation
   breos.plotting.plot_montecarlo_npv_distribution
   breos.plotting.plot_montecarlo_grid_independence_distribution
   breos.plotting.plot_montecarlo_final_soh_distribution
```

## Batch comparison

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.plotting.plot_grid_independence_heatmap
   breos.plotting.plot_location_comparison_delta
```

## CO2

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.plotting.plot_co2_savings
```

## Weather visualization

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.plotting.plot_weather_annual_ghi_distribution
   breos.plotting.plot_weather_monthly_comparison
```

## Validation plots

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.plotting.plot_validation_parity
   breos.plotting.plot_validation_residuals
   breos.plotting.plot_validation_soh_comparison
   breos.plotting.plot_validation_degradation_split
   breos.plotting.plot_validation_multi_system
```

## Presentation styling

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.plotting.set_presentation_mode
```
