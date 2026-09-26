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
   breos.economics.cost_analysis_projection
```

## Metrics

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.economics.calculate_lcoe
   breos.economics.calculate_lcoe_from_projection
   breos.economics.find_payback_year
   breos.economics.find_payback_year_exact
```

`calculate_lcoe` is a real-terms (constant-price) LCOE: it holds O&M at the
first-year cost and takes a real discount rate. The `lcoe_eur_kwh` that the
App, Monte Carlo and the optimizer report comes from
`calculate_lcoe_from_projection`, which reads O&M and replacement costs from
the projection, where they escalate with inflation. The two agree when
inflation is zero and there is no replacement.

`find_payback_year` and `find_payback_year_exact` give the sustained
discounted payback within the simulated period: the first time cumulative
discounted savings, starting at minus the investment in year 0, reach zero and
stay nonnegative to the horizon. The fractional value is interpolated linearly
between annual points.

## Emissions

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.emissions.EmissionsParams
   breos.emissions.calculate_co2_savings
   breos.emissions.calculate_co2_projection
```
