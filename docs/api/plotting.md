# Plotting

Matplotlib figures of BREOS results, grouped by what they visualize. They
need the `plots` extra (`pip install "breos[plots]"`). Most functions take a
results directory and write one or more PNG files there;
`plot_pv_loss_waterfall` returns the figure and saves it only when given an
`output_path`. Use `set_presentation_mode` to enlarge the fonts of every
figure.

## Time series

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.plotting.plot_timeseries
   breos.plotting.plot_monthly_balance
   breos.plotting.plot_monthly_comparison
   breos.plotting.weekly_graphs
   breos.plotting.yearly_graphs
```

## PV loss diagnostics

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.plotting.plot_pv_loss_waterfall
```

## Cost and breakeven

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.plotting.plot_breakeven
   breos.plotting.plot_breakeven_comparison
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

   breos.plotting.plot_azitilt_ew_1d
   breos.plotting.plot_azitilt_landscape_2d
```

## Pareto front

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.plotting.plot_pareto_front_analysis
```

## Monte Carlo

```{eval-rst}
.. autosummary::
   :toctree: generated/

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

## Presentation styling

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.plotting.set_presentation_mode
```
