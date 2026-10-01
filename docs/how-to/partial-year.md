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

- `start` and `end` are dates in the location's timezone, in the year of
  `start_date`. The window starts at local midnight of `start` and ends at
  local midnight of `end`, so `end` is exclusive: the example covers 1 to 7
  June, and one day is `start = 2025-06-01`, `end = 2025-06-02`. `end` may be
  1 January of the next year. A window of the whole year raises; omit
  `[period]` for that.
- The load profile is built for the whole year and scaled to
  `annual_consumption_kwh` as usual, and the window gets its share of it.
  Weather, PV, load and battery temperature are then cut to the window.
- The weather must cover the whole window. It may cover more (a TMY covers
  the year), but a window it does not reach raises `ValueError` naming the
  missing span. Two cases follow from the window being in civil time:
  - Weather that covers one UTC year, such as an Open-Meteo or other UTC CSV
    file, cannot serve a window that touches the civil year edge away from
    UTC. A Berlin window starting on 1 January misses one hour before
    01:00 UTC, and a New York window ending on 1 January of the next year
    misses five hours. PVGIS TMY weather, on the fixed offset of 1 January,
    covers both.
  - Weather stamped at half past the hour (NSRDB-style) cannot place a
    window that starts at local midnight, so no window runs on it.

  Both raise `ValueError` before anything is simulated.
- The window runs once, from the battery's initial state, and nothing is
  repeated or carried over. `projection_years` is not used: setting it
  alongside `[period]` gives a warning, and the result's `period` block
  records `projection_years_used = 1`.
- The fixed charge (`costs.daily_power_cost`, or a tariff's
  `fixed_charge_per_day`) is billed on the window's civil days, so a window
  over a DST change bills whole days, not 23 or 25 hours.
- Energy results cover the window. The lifetime economics (NPV, payback,
  LCOE, the `financial` projection and the replacement costs) are `None`; see
  [Period runs](../getting-started/interpreting-results.md#period-runs).
- `monthly` groups the window by the location's civil months, while a
  full-year run groups by the weather's clock. On weather whose clock is not
  the civil one (a PVGIS fixed offset in summer), a window's `"Jun"` row can
  therefore differ from the same month of a full-year run by the hour at
  each month edge.

PV, load and every other flow of a PV-only run equal the same steps of a
full-year run exactly. With a battery the dispatch differs, because the
window starts from the battery's initial state of charge and health, not
from the state the full-year run reaches on that date.

Monte Carlo and the optimizer reject `period`: they rank designs on lifetime
economics. `breos sweep` accepts it, and can vary `period.start` and
`period.end`. `App.revalue` re-prices a period run's window; its lifetime
economics stay `None`.
