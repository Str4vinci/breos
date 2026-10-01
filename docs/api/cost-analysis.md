# Cost analysis

Cost parameters, NPV / LCOE / payback projections, and CO2 emissions
projections. Emissions and economics share the same projection function
because both produce per-year cumulative discounted series from the same
energy balance.

## Cost parameters

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.economics.CostParams
   breos.economics.cost_params_from_config
```

## CAPEX and projection

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.economics.calculate_costs
   breos.economics.replacement_event_cost
   breos.economics.cost_analysis_projection
```

`cost_analysis_projection` runs in stages, each public, so a caller can
re-price stored year rows without repeating the rest:

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.economics.price_year_rows
   breos.economics.value_year_rows
   breos.economics.discount_cashflows
   breos.economics.add_co2_projection
   breos.economics.write_cost_projection
```

`price_year_rows` and `value_year_rows` turn each year's energy into component
cashflows, `discount_cashflows` accumulates and discounts them and sets the
payback, NPV and LCOE, `add_co2_projection` adds the avoided emissions, and
`write_cost_projection` writes `cost_projection.csv`. App, Monte Carlo and the
optimizer read LCOE and lifetime CO2 from the projection these stages build,
so each is computed once per run. To value a finished App run at other
prices, use `App.revalue` ({doc}`example <../gallery/tariffs/plot_12_price_scenarios>`).

## Metrics

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.economics.calculate_lcoe_from_projection
   breos.economics.find_payback_year
   breos.economics.find_payback_year_interpolated
```

The `lcoe_per_kwh` that the App, Monte Carlo and the optimizer report comes
from `calculate_lcoe_from_projection`, which reads O&M and replacement costs
from the projection, where they escalate with inflation.

`find_payback_year` and `find_payback_year_interpolated` give the sustained
discounted payback within the simulated period: the first time cumulative
discounted savings, starting at minus the investment in year 0, reach zero and
stay nonnegative to the horizon. The fractional value is interpolated linearly
between annual points.

## Emissions

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.emissions.EmissionsParams
   breos.emissions.calculate_co2_projection
```
