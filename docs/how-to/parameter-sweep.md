# Run a parameter sweep

Use `breos sweep` when you want to run the same scenario over an explicit grid
of App config values. The top-level keys define the base scenario; every key
under `[sweep]` replaces the matching key for each run. Quote dotted keys to
vary one value inside a table: `[costs]`, `[battery_indoor_model]`,
`[tariff]`, `[reference_tariff]`, `[smart_charging]`, `[terminal_value]` or
`[period]`, as in `"period.start"`. A tariff's price maps take one more level,
the period name, as in `"tariff.import_prices.off_peak"`, or two on a
schedule with month seasons, the season and the period, as in
`"tariff.import_prices.q1.high"`. Any other key inside a table, or a level
past those, is refused. The command runs
the Cartesian product and writes one CSV row per combination:

```toml
location = "porto"
n_modules = 10
annual_consumption_kwh = 4000
battery_kwh = 0.0
load_profile = "demandlib_h0"
cost_preset = "residential_pt"
emissions_country = "PT"
projection_years = 20
resolution = "h"

[sweep]
n_modules = [8, 10, 12]
battery_kwh = [0.0, 5.0]
"costs.electricity_cost" = [0.20, 0.30]
```

```bash
breos sweep --config config.toml --output sweep_results.csv
```

The output includes the varied parameters (`param_*` columns, including for
example `param_costs.electricity_cost`), resolved system
sizing, the BREOS version, and top-level scalar result metrics such as grid
independence, NPV, payback, LCOE, battery replacement totals, and the
[year-1 money components](../getting-started/interpreting-results.md#year-1-money-keys). This is
explicit enumeration, not an optimizer; use the optimization API for searching
over objectives and constraints.

{py:func}`~breos.plotting.plot_sweep_heatmap` draws one column of a
two-parameter sweep CSV as a heatmap, and
{py:func}`~breos.plotting.plot_orientation_landscape` draws a `tilt` ×
`azimuth` sweep ([Plotting](../api/plotting.md#sweeps-and-the-optimizer-front)).
The CSV does not record the currency, so pass `currency=` to label a money
column such as `npv_savings` with it.

Every combination is validated before the first run starts, so a bad one,
such as a charge period the tariff schedule does not have, stops the sweep
at once. `breos validate-config` checks every combination too.

Runs that differ only in settings the input stage never reads share one
preparation of weather, PV, load and battery temperature. Those settings are
the battery and inverter settings, degradation, the projection length, prices,
`[tariff]`, `[smart_charging]`, emissions and the execution backend. A tariff
or battery-size comparison therefore fetches PVGIS weather and runs the PV
model once per PV design, not once per run, and writes the same CSV as
preparing each run afresh. To do so it runs the grid grouped by PV design,
while the CSV keeps the grid order. A warning from the input stage appears
once, for the run that prepares those inputs.

## Sweep one price inside a table

To vary one price instead of the whole tariff, keep one `[tariff]` in the base
scenario and sweep a dotted key:

```toml
[tariff]
schedule = "pt_mainland_2026_daily_bi"
currency = "EUR"
import_prices = { peak = 0.2310, off_peak = 0.1210 }
export_prices = { all = 0.0500 }

[sweep]
"tariff.import_prices.off_peak" = [0.1010, 0.1210, 0.1410]
```

Each run simulates again. For price scenarios on a finished run,
{py:meth}`breos.App.revalue` re-prices the stored simulation instead, also on
another tariff schedule when the run has no smart charging; see
{doc}`../gallery/tariffs/plot_12_price_scenarios`. To compare whole offers,
see {doc}`../gallery/tariffs/plot_09_which_tariff`.
