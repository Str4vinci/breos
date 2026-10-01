# Simulate part of a year

A `[period]` table simulates a window shorter than a year, for example one
week in June:

```toml
location = "porto"
n_modules = 10
annual_consumption_kwh = 4000
battery_kwh = 5.0
start_date = "2025-01-01"

[period]
start = 2025-06-01
end = 2025-06-08
```

As a Python dict, give the dates as ISO strings or `datetime.date` values:
`"period": {"start": "2025-06-01", "end": "2025-06-08"}`.

- `start` and `end` are local dates in the year of `start_date`, and `end` is
  exclusive: the example covers 1 to 7 June.
- The window runs once, from the battery's initial state. Energy results cover
  the window; the lifetime economics (NPV, payback, LCOE and the `financial`
  projection) are `None`.
- Monte Carlo and the optimizer reject `[period]`. `breos sweep` and
  `App.revalue` accept it.

[Simulate part of a year](../getting-started/configuration.md#simulate-part-of-a-year)
in the configuration guide gives every rule: weather coverage, the fixed
charge on civil days, monthly grouping, and how a window's dispatch differs
from the same days of a full-year run.
