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

`plot_breakeven_comparison` takes `App.result()` dicts or cost projection
frames, one per scenario:

```python
results = []
for battery_kwh in (0.0, 5.0, 10.0):
    app = App({**config, "battery_kwh": battery_kwh})
    app.simulate()
    results.append(app.result())
plot_breakeven_comparison(results, ["PV only", "PV + 5 kWh", "PV + 10 kWh"], "plots")
```

Each payback line is labelled with its year. Scenarios that share a
no-system cost share one baseline, named "No system" when all of them share
it. An App result records its currency; a cost projection read back from CSV
does not, so pass `currency=` to label it, or its amounts show no currency
code.

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

## Sweeps and the optimizer front

These read the CSV that `breos sweep` writes, or its DataFrame. A swept key
can be named as in the config (`n_modules`) or by its column
(`param_n_modules`). `plot_sweep_heatmap` draws one result column over a
two-parameter grid; with `diff=`, a second sweep over the same grid, it draws
the difference, for example between two locations.
`plot_orientation_landscape` maps a tilt × azimuth sweep and draws the
east-west profile at the best tilt; a tilt-only sweep, such as an east-west
roof, gets the tilt profile. `plot_pareto_front` draws two objectives of the
optimizer's front (the `OptimizationResult` or its `details["pareto"]` frame),
or of any table of designs, and marks the designs no other one beats.

```python
from breos.plotting import plot_orientation_landscape, plot_pareto_front, plot_sweep_heatmap

plot_sweep_heatmap("porto.csv", "grid_independence_pct", "plots", diff="berlin.csv", labels=("Porto", "Berlin"))
plot_orientation_landscape("orientation.csv", "pv_production_kwh", "plots")
plot_pareto_front(result, "plots", color_by="Battery_kWh")
```

A swept key names the swept `param_` column, not the result column of the
same name, which holds the App's resolved value. A difference of a
percentage, such as grid independence, is labelled in percentage points. A
difference, and a metric with both gains and losses such as `npv_savings`,
use a diverging colour scale centred on zero.

Sweep CSVs do not record their currency. Pass `currency="EUR"` (or another
code) to label their money; without it, money labels name no currency, such
as "NPV savings". A DataFrame can record it in `attrs["currency"]`, as the
optimizer's `details["pareto"]` frame does.

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.plotting.plot_sweep_heatmap
   breos.plotting.plot_orientation_landscape
   breos.plotting.plot_pareto_front
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

## CO2

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.plotting.plot_co2_savings
```

## Weather visualization

Compare a TMY with the historical years a Monte Carlo study samples. Both
plots take the TMY as a weather DataFrame, such as
`breos.weather.load_weather(..., data_type="tmy")` returns, and the
historical weather as the study's `weather_file` path or the per-year frames of
`breos.weather.preload_weather_by_year`. The monthly minimum and maximum are
each month's lowest and highest value over the historical years, so the two
can come from different years.

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
