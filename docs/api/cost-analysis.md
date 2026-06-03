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
   breos.economics.find_payback_year
```

## Emissions

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.emissions.EmissionsParams
   breos.emissions.calculate_co2_savings
   breos.emissions.calculate_co2_projection
```
