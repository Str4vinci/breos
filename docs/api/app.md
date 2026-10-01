# App facade

{py:class}`~breos.App` is the stable entry point. It takes one configuration
dict, the same keys as a TOML file for `breos run`
([Configuration](../getting-started/configuration.md),
[key reference](../getting-started/config-reference.md)), runs weather, PV,
load, battery, economics and emissions with `simulate()`, and returns a plain
JSON-serializable dict from `result()`
([Interpreting results](../getting-started/interpreting-results.md)).

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.App
```

## Revalue a finished run

`App.revalue(changes)` returns the result a finished run would give at other
prices, without changing the App or its `result()`. `changes` may set only
the economics keys: `costs`, `cost_preset`, `tariff`, `reference_tariff`,
`terminal_value`, `discount_rate`, `inflation_rate` and the escalators
(`import_price_escalation`, `om_escalation`, `sell_price_inflation`,
`replacement_cost_learning`). Any other key, such as `battery_kwh` or
`projection_years`, raises `ValueError`; build a new `App` for it.

A nested table changes only the keys it sets, and a key set to `None` in it
is removed; `{"tariff": None}` removes the table. A price list
(`tariff.import_prices`, `tariff.export_prices`,
`reference_tariff.import_prices`) replaces the old one whole.

```python
from breos import App

app = App(
    {
        "location": "porto",
        "n_modules": 10,
        "annual_consumption_kwh": 4000,
        "battery_kwh": 5.0,
        "cost_preset": "residential_pt",
    }
)
app.simulate()

for storage_cost in (500, 400, 300):
    result = app.revalue({"costs": {"storage_cost_per_kwh": storage_cost}, "discount_rate": 0.05})
    print(storage_cost, result["npv_savings"], result["provenance"]["revaluation"]["method"])
```

`provenance.revaluation` records the keys that changed and the `method`:

- `"repriced"`: the stored simulation was priced again. This is the case for
  flat prices, a tariff removed, new prices on the same tariff schedule when
  the smart-charging instructions do not change, any change to
  `reference_tariff`, and `terminal_value`, which is recomputed from the
  retained final battery health. Flat prices give the same floats as a new
  run. A re-priced tariff sums each year's energy by period instead of by
  step, so it agrees with a new run to rounding.
- `"resimulated"`: the run was simulated again, because a tariff was added,
  the schedule changed, or the smart-charging instructions would change.
  Under the experimental `daily_persistence` smart charging, whose planner
  reads the prices, any change to the import or export prices simulates
  again; a change to the fixed charge alone is re-priced.

A [`[period]`](../getting-started/configuration.md#simulate-part-of-a-year)
run is re-priced over its window, and its lifetime economics stay `None`.
