# Interpreting results

`App.result()` returns a plain Python dict, JSON-serializable, with no
pandas or numpy types. The same dict is written by the CLI's `--output`
flag. For the step-by-step flows of the first year, use
[`App.timeseries()`](../api/app.md).

## Top-level keys

| Key | Description |
|---|---|
| `result_schema_version` | Format number of the result's names (see [Currency and result format](#currency-and-result-format)) |
| `n_modules` | Number of PV modules used in the simulation |
| `pv_kwp` | System DC nameplate capacity (kWp) |
| `battery_kwh` | Battery capacity (kWh) |
| `usable_ac_system_production_kwh` | PV-origin AC delivered to load or export in year 1 |
| `pv_dc_generation_kwh` | PV DC generated before dispatch |
| `direct_pv_ac_load_kwh` | Direct PV AC delivered to load |
| `pv_origin_battery_ac_load_kwh` | PV-origin AC delivered from storage to load |
| `curtailment_dc_kwh` | PV DC that could not serve load, charge storage, or export |
| `consumption_kwh` | Year 1 load |
| `self_consumption_kwh` | Direct PV AC plus PV-origin battery AC delivered to load |
| `grid_import_kwh` | Year 1 energy bought from the grid |
| `grid_export_kwh` | Year 1 energy sold to the grid |
| `grid_independence_pct` | Year 1 grid independence ratio |
| `self_consumption_pct` | Year 1 self-consumption ratio |
| `total_investment` | Total CAPEX |
| `payback_year` | Sustained discounted payback within the simulated period, as a whole year: the year from which cumulative NPV savings are zero or above and stay so to the horizon (`None` if not reached) |
| `npv_savings` | Cumulative NPV savings over the projection horizon |
| `terminal_health_credit` | Optional nominal credit at the end of year T; null when disabled or on a partial period |
| `terminal_health_credit_npv` | Present value of that credit |
| `npv_savings_terminal_adjusted` | Unadjusted NPV plus credit present value, before rounding |
| `lcoe_per_kwh` | Levelized cost of electricity from system CAPEX, O&M, simulated replacements, and discounted PV production |
| `monthly` | Year 1 monthly energy balance rows |
| `financial` | Yearly financial projection rows (year 0 = investment) |
| `yearly` | Per-year breakdown of production, load, imports, exports |
| `pv_loss_waterfall` | Year 1 PV loss waterfall from the irradiance reference through the static PVWatts losses, with inverter and dispatch blocks beside it (see [PV loss waterfall](#pv-loss-waterfall)) |
| `degradation` | Battery degradation engine, model, initial and final state of health and replacement events |
| `smart_charging` | Present with [smart charging](configuration.md#smart-charging): grid charging, delivery by origin and stored energy by origin |
| `pv_arrays` | Present with `pv_arrays` (see [Multi-array systems](#multi-array-systems)) |
| `period` | Present on a [period run](#period-runs) |
| `provenance` | BREOS version, currency, normalized resolved config, ledger schema version, weather/location metadata, resolution, timezone, and start date; `input_repairs` holds the reports passed as `App(..., input_repairs=...)`, and is present only then (see [Repairing measured data](inputs.md#repairing-measured-data)) |

## Year-1 money keys

These keys give the first project year's money components at year-1 prices:
the prices of the first project year, before escalation and discounting. They
are in the run's currency and rounded to 0.01. Without a `[tariff]` they use
the flat `costs` prices; with one, each step's energy is priced at that
step's tariff price. Flat and tariff runs report the same five keys. With a
[`[reference_tariff]`](configuration.md#no-system-reference-tariff), the two
no-system keys use the reference's prices instead.

| Key | Description |
|---|---|
| `grid_import_cost_year1_prices` | Cost of the year-1 grid import, `grid_import_kwh` |
| `grid_export_revenue_year1_prices` | Revenue from the year-1 grid export, `grid_export_kwh` |
| `fixed_charge_year1_prices` | The fixed charge for the simulated duration of year 1: the daily charge times the simulated hours / 24 |
| `grid_charge_cost_year1_prices` | Present only with smart charging (`smart_charging.mode = "fixed_target"`, `"daily_persistence"` or `"discharge_only"`): the part of `grid_import_cost_year1_prices` bought to charge the battery. Always 0 with `discharge_only`, which never charges from the grid |
| `no_system_import_cost_year1_prices` | Import cost of the household without a system, which buys its whole year-1 load, `consumption_kwh`. It is the import cost only; the fixed charge is `no_system_fixed_charge_year1_prices` |
| `no_system_fixed_charge_year1_prices` | The fixed charge of the household without a system for year 1: `fixed_charge_year1_prices`, or the reference tariff's fixed charge for the same days when a `[reference_tariff]` is set |

With an [annual network credit](configuration.md#annual-network-credit),
each household that has one also reports its eligible network charges, the
cap basis, and its credit:

| Key | Description |
|---|---|
| `network_charge_year1_prices` | The system household's eligible network charges for year 1: its grid import at the network prices, plus the network fixed amount |
| `network_credit_year1_prices` | The system household's credit for year 1: the annual amount, capped at `network_charge_year1_prices` |
| `no_system_network_charge_year1_prices` | The eligible network charges of the household without a system, on its whole year-1 load |
| `no_system_network_credit_year1_prices` | The credit of the household without a system, capped at `no_system_network_charge_year1_prices` |

The eligible network charges are part of the import cost and the fixed
charge, not an addition to them. Each credit lowers its household's bill.

`grid_charge_cost_year1_prices` is already included in
`grid_import_cost_year1_prices`, so do not add the two. It is the same value
as `smart_charging.yearly[0].grid_charge_cost_year1_prices`.

Without a `[reference_tariff]`, the same fixed charge applies with or
without the system, so `no_system_fixed_charge_year1_prices` equals
`fixed_charge_year1_prices`. The projection's no-system annual cost is the
no-system import cost plus the no-system fixed charge, so the year-1
no-system bill is
`no_system_import_cost_year1_prices + no_system_fixed_charge_year1_prices`.

Year 1 is not escalated, so these values match the `cost_import`,
`revenue_export` and `cost_fixed_charge` of the year-1 `financial` row, and
the two no-system keys match its `no_system_cost_import` and
`no_system_cost_fixed_charge`; the later rows escalate. `breos sweep` copies every top-level scalar key into its
CSV, so the sweep CSV carries these columns too.

## Battery-specific keys

Present only with a battery that can dispatch (`battery_kwh > 0`, and more
than 1 Wh):

| Key | Description |
|---|---|
| `battery_soh_end_pct` | State of health at the end of the projection horizon |
| `battery_replacements` | Total number of replacements over the projection |
| `battery_replacement_cost_t0_prices` | Total replacement cost at t = 0 prices, neither inflated nor discounted |
| `battery_replacement_cost_npv` | The same replacements inflated to and discounted from each swap instant, as `npv_savings` counts them |
| `battery_end_of_life_events` | Every end-of-life crossing over the projection, in order; see [End-of-life events](#end-of-life-events) |
| `battery_first_end_of_life_years` | Time of the first crossing, in years from commissioning; None without one |
| `battery_first_end_of_life_action` | What was done at the first crossing: `"replaced"`, `"kept"` or `"retired"`; None without one |
| `battery_first_end_of_life_reason` | Why, as in `battery_end_of_life_events`; None without one |
| `battery_first_end_of_life_soh_pct` | State of health at the first crossing; None without one |
| `battery_degradation_history` | The battery's state at the end of each project year, with the year's use and stress; see [Degradation history](#degradation-history) |

With `battery_allow_terminal_replacement = false`, a pack that reaches end of
life in the final degradation period of the horizon is not replaced. The
replacement count and costs then leave out that one swap, and
`battery_soh_end_pct` can end below the end-of-life threshold. With
`battery_replacement_min_remaining_years` above 0, every swap that would leave
the new pack less than that many project years is left out in the same way,
and the old pack ages below the threshold until the end of the project. See
[Battery replacement at the end of the horizon](configuration.md#battery-replacement-at-the-end-of-the-horizon).

### End-of-life events

A crossing is a degradation period that closes with the installed pack at or
below `battery_eol_percentage`. Each entry of `battery_end_of_life_events`
records one:

| Field | Description |
|---|---|
| `year` | Project year of the crossing, from 1 |
| `time_years` | Years from commissioning to the end of the closing step: the instant a replacement is booked at, equal to the `financial` row's `replacement_time_years` |
| `date` | Calendar date of the closing step; every project year replays the simulated calendar, moved forward by its year |
| `action` | `"replaced"`: a new pack was installed. `"kept"`: no pack was bought and the old one stays in service below its threshold. `"retired"`: no pack was bought and the old one was switched off; the project finishes PV-only (`battery_skipped_replacement_action = "retire"`) |
| `reason` | `"end_of_life"` for a replacement. For a skipped one, `"replacement_disabled"` (`battery_enable_replacement = false`), `"min_remaining_years"` (less than `battery_replacement_min_remaining_years` was left) or `"terminal_period"` (the final period with `battery_allow_terminal_replacement = false`) |
| `soh_pct` | State of health at the crossing, the value compared with the threshold |

A replaced pack's successor can cross again, so a long horizon can list
several replacements. A pack that is kept or retired crosses once: it stays below its
threshold, and no later period records it again. A skipped replacement is
therefore always the last entry. For example, with 20 project years and
`battery_replacement_min_remaining_years = 1.0`, a pack that reaches end of
life at 19.3 years shows as `{"year": 20, "time_years": 19.3, "action":
"kept", "reason": "min_remaining_years", ...}`: the replacement was skipped
and the old battery was retained. With
`battery_skipped_replacement_action = "retire"` the same entry has `"action":
"retired"`: the old battery was switched off at 19.3 years.

Monte Carlo reports the first crossing of each trajectory as
`first_end_of_life_years`, `first_end_of_life_action`,
`first_end_of_life_reason` and `first_end_of_life_soh_pct`; the projected
optimizer reports it as `Projected_First_End_Of_Life_Years`,
`Projected_First_End_Of_Life_Action`, `Projected_First_End_Of_Life_Reason`
and `Projected_First_End_Of_Life_SOH_%`. Without a crossing the times and
health are NaN and the text fields None.

### Degradation history

`battery_degradation_history` has one entry per project year. The state of
health, capacities, `cumulative_fec` and loss split belong to the pack
installed at the end of the year, so they restart after a replacement. The
year's own figures cover every pack that served in it.

| Field | Description |
|---|---|
| `year` | Project year, from 1 |
| `soh_pct` | State of health at the end of the year, as `yearly[].soh_pct` |
| `capacity_kwh` | `battery_kwh` times the state of health |
| `usable_capacity_kwh` | `capacity_kwh` times the SOC window, `battery_max_soc - battery_min_soc` |
| `replacements` | Replacements in the year |
| `charge_throughput_kwh` | Energy stored in the cells in the year, after charging losses |
| `discharge_throughput_kwh` | Energy drawn from the cells in the year, before inverter losses |
| `fec` | Full equivalent cycles in the year, over every pack, as the degradation model counts them (below) |
| `cumulative_fec` | Full equivalent cycles of the installed pack since it was installed |
| `mean_soc_pct` | Mean state of charge over the year, as a share of the pack's aged capacity; the SOC the ageing model sees |
| `mean_cell_temperature_c` | Mean cell temperature over the year, the temperature the ageing model sees |
| `cycle_loss_pct` | State of health the installed pack has lost to cycling, in percentage points. Native engine only; None with BLAST |
| `calendar_loss_pct` | The same for calendar ageing. Native engine only; None with BLAST |
| `resistance_growth_pct` | Growth of the installed pack's internal resistance, in percent. Only with `enable_resistance_fade` (native engine); None otherwise |
| `round_trip_efficiency` | Cell round-trip efficiency at that resistance, as the dispatch applies it. Only with `enable_resistance_fade`; None otherwise |

A field is None when the degradation model does not supply it, never 0. The
BLAST models report a state of health but no split into cycle and calendar
loss, and only the native engine has a resistance model. For the native
engine `cycle_loss_pct + calendar_loss_pct` is the state of health lost
since the pack was installed, to rounding. The year rows behind the history
also carry `Battery_Cell_Temperature_Mean_C`.

The cycle counts are the model's own. The native engine counts rainflow
cycles of the state of charge, which is a share of the aged capacity, so a
full cycle of a worn pack counts as one. BLAST scales each cycle by the
capacity left, so it counts cycles of the new pack's capacity, and the same
use gives fewer cycles as the pack ages.

## Estimated battery residual value

The optional [`[terminal_value]`](configuration.md#estimated-battery-residual-value)
estimate adds three fields beside the unchanged `npv_savings`, the NPV
excluding residual value. The configuration page gives the formula, its
assumption and what is valued.

`terminal_health_credit` is nominal year-T money, priced with the resolved
replacement-pack price at t = 0, inflated and reduced by replacement
learning to exactly t = T. `terminal_health_credit_npv` discounts it from T.
`npv_savings_terminal_adjusted`, the NPV including the estimated residual
value, adds that present value to the unrounded NPV excluding it; each money field is then rounded to two decimals, so the
reported scalars can differ by a cent from adding rounded values.
`financial`, paybacks and LCOE exclude the residual value.

Disabled App runs and partial `[period]` runs report all three fields as
null and carry no `provenance.terminal_value`. Enabled PV-only runs report
a zero residual value and equal NPVs with and without it. Enabled lifetime
runs record basis, formula version, unrounded final SOH fraction, physical
threshold, credited fraction, full replacement price at t = 0 and T,
inflation, learning, discount rate, horizon, booking time and replacement
policy in `provenance.terminal_value`.

Monte Carlo's `runs` frame has the three fields per trajectory; its
`summary` reports their existing mean, spread, percentile, range and count
statistics. Its `provenance.terminal_value.trajectories` records the inputs
for each numbered run. Disabled values are NaN with no statistics or
terminal-value provenance. Projected optimization ignores the table and
ranks on the NPV excluding residual value.

## Emissions keys

Present only when `emissions_country` is set:

| Key | Description |
|---|---|
| `co2_avoided_self_consumption_year1_kg` | Year 1 behind-the-meter benefit |
| `co2_avoided_export_year1_kg` | Year 1 exported-generation benefit |
| `co2_avoided_total_year1_kg` | Sum of the two year 1 pathways |
| `co2_avoided_self_consumption_lifetime_kg` | Lifetime behind-the-meter benefit |
| `co2_avoided_export_lifetime_kg` | Lifetime exported-generation benefit |
| `co2_avoided_total_lifetime_kg` | Sum of the two lifetime pathways |

Self-consumption uses the preset's avoided-grid factor. Export uses
`export_emissions_factor_gco2_kwh` when configured; otherwise it explicitly
falls back to the same avoided-grid factor. Curtailed energy, conversion and
storage losses, initial SOC, and PV energy remaining stored at the reporting
boundary receive no credit.

## Multi-array systems

When `pv_arrays` is set, the result also contains a `pv_arrays` list with
each array's resolved configuration: `modules`, `module`, `tilt`, `azimuth`,
and `tracking`, plus the resolved tracker geometry for a tracking array.

## PV loss waterfall

`pv_loss_waterfall` reports the year 1 PV production chain in kWh. Its
ordered `stages` cover only the linear PV-model chain: horizontal reference,
transposition, front-side incidence-angle modifier, optional bifacial rear
gain, cell temperature, and static PVWatts losses. There is no degradation
stage because year 1 has none (see [Module aging](../api/pv.md#module-aging)),
so the last stage equals `energy_balance.pv_dc.generation_kwh`. Dispatch
is a branching flow and is therefore reported under `energy_balance`, not
forced into a misleading linear stage.

The `bifacial` block identifies the resolved model and per-array module factor
and geometry. It reports rear gain in effective-DC-equivalent kWh and as a
percentage of front effective irradiance. The same block is retained under
`provenance.pv_model.bifacial` so serialized results carry the assumptions that
produced the gain.

The `pvwatts` block contains fixed-loss percentages and attributed kWh. The
`inverter` block reports AC rating plus separate direct-PV and
battery-discharge conversion losses. `energy_balance.pv_dc` reconciles PV
routing; `energy_balance.ac_delivery` reconciles delivered/exported AC; and
`energy_balance.battery_stored_energy` reconciles beginning/end energy,
charge, discharge, standby, capacity-window, and replacement boundary flows.

Use {py:func}`breos.plotting.plot_pv_loss_waterfall` to render the same
block as a PV loss diagram.

## Economic conventions

The projection is in nominal terms: `inflation_rate`, the escalators and
`discount_rate` are nominal annual rates. `result()["provenance"]["economics"]`
records the rates a run used, each escalator after inheriting from
`inflation_rate`, and the implied real discount rate,
`(1 + discount_rate) / (1 + inflation_rate) − 1`. To run a real study, give
real rates and zero inflation; the arithmetic is the same, and BREOS records
the rates rather than the intent.

Timing:

- The initial investment is at year 0.
- Energy, the fixed charge and O&M are at year-1 prices, escalated
  `(1 + rate)^(n − 1)` in year `n` and booked at the end of the year, so
  discounted by `(1 + discount_rate)^n`. When an escalator equals the
  discount rate, one year of discounting still remains.
- A battery replacement is priced at today's (t = 0) storage cost, inflated
  to the instant of the swap and discounted from that instant, not from a
  year boundary. The simulation reports only when a pack was swapped and its
  capacity; the economics prices it, the same way for the App,
  Monte Carlo and the optimizer.
- The daily fixed charge is billed on the simulated duration: 365 days for a
  common year, 366 for a leap year.

## Monthly and yearly breakdowns

### `monthly`

A list of dicts, one per month of year 1 (12 rows; a period run has one per
civil month of its window). From the quickstart run:

```python
{
    "month": "Jan",
    "pv_dc_generation_kwh": 572.25,
    "direct_pv_ac_load_kwh": 157.52,
    "pv_origin_battery_ac_load_kwh": 99.62,
    "usable_ac_system_production_kwh": 538.94,
    "curtailment_dc_kwh": 0.0,
    "consumption_kwh": 407.38,
    "self_consumption_kwh": 257.15,
    "grid_import_kwh": 146.36,
    "grid_export_kwh": 281.8,
    "grid_independence_pct": 64.07,
}
```

### `yearly`

A list of one dict per simulation year (length `projection_years`). Each
row contains the same fields as `monthly` aggregated to a year, plus
`soh_pct` when a battery is present.

### `financial`

A list of dicts with one row per year (year 0 is the investment row). From
the quickstart run:

```python
{"year": 0, "balance": -7788.85, "reference": 0.0}
{"year": 1, "balance": -6899.8, "reference": 0.0, "cost_with_system": 8008.83,
 "cost_without_system": 1109.03, "no_system_cost_import": 1032.8,
 "no_system_cost_fixed_charge": 109.5, "cost_import": 216.91,
 "revenue_export": 199.84, "cost_operation": 100.0, "cost_fixed_charge": 109.5,
 "cost_replacement": 0.0, "replacement_time_years": None}
# ...
```

`balance` is the cumulative NPV savings, `cost_without_system` minus
`cost_with_system`. `cost_with_system` and `cost_without_system` are the
cumulative discounted costs with and without the system; `cost_with_system`
includes the investment. `replacement_time_years` is the project time, in
years, of a battery replacement in that year, or `None`. From year 1,
`no_system_cost_import` and
`no_system_cost_fixed_charge` give that year's no-system cost by component,
escalated and not discounted, beside the system's `cost_import`,
`revenue_export`, `cost_operation`, `cost_fixed_charge` and
`cost_replacement`. With a
[no-system reference tariff](configuration.md#no-system-reference-tariff),
they are at the reference's prices and escalation. With an
[annual network credit](configuration.md#annual-network-credit), the rows of
a household with a credit also carry `network_charge` and `network_credit`,
or `no_system_network_charge` and `no_system_network_credit`, escalated and
not discounted; each credit is already in that household's cost.
`payback_year` is the sustained discounted
payback within the simulated period: the year from which `balance ≥ 0` holds
to the end of the horizon. The series starts at year 0, so a system that
recovers its investment during year 1 reports 1. If a battery replacement
turns `balance` negative again, payback is the later recovery, and a
`balance` that is negative in the last year means no payback.
`economics.find_payback_year_interpolated` gives the same crossing as a fractional
year, interpolated linearly between the annual points; it is an estimate
from year-end values, not an exact date.

## Period runs

A run with a [`[period]`](configuration.md#simulate-part-of-a-year) window
simulates that window once, with no projection years. Its result keeps the
full-year keys, with these differences:

- The energy keys (`usable_ac_system_production_kwh`, `grid_import_kwh`,
  `self_consumption_kwh` and the rest), the [year-1 money
  keys](#year-1-money-keys), the year-1 CO2 keys and the PV loss waterfall
  cover the window. The waterfall's `basis` is `"period"`. The fixed charge
  is billed on the window's civil days.
- `yearly` has one row, labelled with the window's `period_start` and
  `period_end` (exclusive). `monthly` groups the window by the location's
  civil months, so a window inside June has one `"Jun"` row. A full-year
  run groups by the weather's clock instead, so where that clock is not the
  civil one, a window's month can differ from the full-year run's by the
  hour at the month edge.
- The lifetime economics are `None`: `npv_savings`, `payback_year`,
  the three residual-value fields,
  `lcoe_per_kwh`, `financial`, `battery_replacement_cost_t0_prices`,
  `battery_replacement_cost_npv` and the lifetime CO2 keys. A window has no
  project lifetime to escalate, discount or pay back over. `total_investment`
  is still reported.
- `battery_soh_end_pct` is the state of health at the end of the window,
  including the rainflow cycles still open at its end, which the last day
  of any run counts.
- A top-level `period` block, also recorded as `provenance.period`, describes
  the window:

```python
{
    "start": "2025-06-01",
    "end": "2025-06-08",
    "end_exclusive": True,
    "timezone": "Europe/Lisbon",
    "days": 7,
    "start_time": "2025-06-01T00:00:00+01:00",
    "end_time": "2025-06-08T00:00:00+01:00",
    "projection_years_used": 1,
    "simulated_hours": 168.0,
    "lifetime_economics": "skipped",
    "lifetime_economics_reason": "A period shorter than a year runs once, ...",
    "skipped_fields": [
        "payback_year",
        "npv_savings",
        "terminal_health_credit",
        "terminal_health_credit_npv",
        "npv_savings_terminal_adjusted",
        "lcoe_per_kwh",
        "financial",
    ],
}
```

`skipped_fields` lists the keys of this result that are `None` for that
reason; the example is a PV-only run without emissions.

## Currency and result format

Money keys carry no currency. Every money value in a result is in the run's
currency, which `provenance.currency` records: the tariff's `currency` when
the run has a `[tariff]` table, otherwise `EUR`, the currency of the bundled
cost catalogue. BREOS does not convert currencies. Summary and plot labels
read the recorded currency.

`result_schema_version` is the result's format number, independent of the
ledger schema. The format changes, to the next integer, only when a field is
renamed or removed, a change that breaks existing readers. Added fields do
not change it: the
[changelog](https://github.com/Str4vinci/breos/blob/develop/CHANGELOG.md)
of each release lists them, and `breos_version` in App, Monte Carlo and optimizer
provenance identifies the release that wrote a result. BREOS 0.7.0 writes format `"1"`, the first
released format. Results from earlier BREOS versions carry no format number;
the [migration table](#migrating-from-breos-062) maps their names.

## Provenance records

Besides the [top-level keys](#top-level-keys), results record the following,
by feature:

- **Economics.** `provenance.economics` holds the rates a projection used
  ([Economic conventions](#economic-conventions)) in App, Monte Carlo and
  optimizer provenance. The year rows of Monte Carlo trajectories and
  optimizer tables carry `Replaced_Capacity_kWh` and the year-1-price money
  columns. The cost projection and Monte Carlo trajectories carry the export
  CO2 columns, `CO2_Avoided_Export_kg` and `CO2_Avoided_Export_Cumulative_kg`.
- **Revaluation.** Results of
  [`App.revalue`](../api/app.md#revalue-a-finished-run) carry
  `provenance.revaluation`.
- **Optimizer.** The optimizer's provenance carries `constraints` and
  `run_settings`; the Pareto rows of a search with `[emissions]` carry the
  `Projected_CO2_*` columns.
- **Inverter.** `inverter_ac_rating_kw` is in the `resolved_config` of App
  and Monte Carlo results.
- **Period runs.** A run over part of a year carries the
  [`period` keys](#period-runs).
- **Load year.** Monte Carlo's `provenance.load_profile` records
  `calendar_year`, the `target_year` the load was built for.
- **Custom schedules.** App and Monte Carlo results that set an inline
  [custom schedule](../api/tariffs.md#custom-app-schedules) carry
  `tariff.custom_schedule` in `resolved_config`. A schedule with calendar-month
  [seasons](../api/tariffs.md#month-seasons) records `seasons` in
  `resolved_config.tariff.custom_schedule` and, as the month partition, in
  `provenance.tariff`. With month seasons, `import_prices` and
  `export_prices` in the resolved config and in `provenance.tariff` may map
  each season to its period prices instead of each period to a price.
- **Smart charging.** `provenance.smart_charging` records `overlap_policy`
  in App, Monte Carlo and optimizer results, including the default `reject`;
  `hold_target` permits overlapping periods in `fixed_target` and
  `daily_persistence` and keeps the grid target as the discharge floor. A
  [`discharge_only`](configuration.md#discharge-only) run records its
  discharge periods, an empty `charge_periods`, and `None` for
  `target_usable_fraction`, `grid_charge_efficiency` and
  `grid_import_limit_w`; the App's top-level `smart_charging` block reports
  it as it reports the other modes. An App run with the experimental
  [daily-persistence smart charging](configuration.md#daily-persistence-experimental)
  records `experimental`, `decision_boundary`, `controller_version`,
  `planner_version`, `forecast_horizon_days`, `target_levels`, `soc_states`,
  `wear_cost_per_kwh`, `forecast_policy`, `warm_start_policy`,
  `planner_terminal_policy`, and the `initial_stored_energy` and
  `final_stored_energy` by origin. Under
  `decision_boundary = "charge_window_start"` the two versions are `"2"`.
- **Battery replacement at the end of the horizon.**
  `battery_allow_terminal_replacement` and
  `battery_replacement_min_remaining_years` are in the `resolved_config` of
  App and Monte Carlo results, and the provenance of a projected design and
  of an optimizer search carries `battery_replacement_treatment`, with its
  `allow_terminal_replacement` policy, a `terminal_period` description, its
  `replacement_min_remaining_years` and a `minimum_service` description. An
  enabled `[terminal_value]` records both in its `replacement_policy`.
  See
  [battery replacement at the end of the horizon](configuration.md#battery-replacement-at-the-end-of-the-horizon).
- **No-system costs.** Results carry `no_system_fixed_charge_year1_prices`
  (see [Year-1 money keys](#year-1-money-keys)), `no_system_cost_import` and
  `no_system_cost_fixed_charge` in the `financial` rows, the
  `Cost_No_Sys_Import` and `Cost_No_Sys_Fixed_Charge` cost-projection columns,
  the `Baseline_Fixed_Charge` year-row column of Monte Carlo and optimizer
  tables, and `reference_tariff` in the `resolved_config` of App and Monte
  Carlo results. Results with a
  [no-system reference tariff](configuration.md#no-system-reference-tariff)
  also carry `provenance.reference_tariff`.
- **Annual network credit (0.7.1).** A household with an
  [annual network credit](configuration.md#annual-network-credit) carries
  its year-1 keys (see [Year-1 money keys](#year-1-money-keys)), its
  `financial` row fields, the `Network_Charge` and `Network_Credit`, or
  `Baseline_Network_Charge` and `Baseline_Network_Credit`, year-row columns,
  and the `Cost_Network_Charge` and `Cost_Network_Credit`, or
  `Cost_No_Sys_Network_Charge` and `Cost_No_Sys_Network_Credit`,
  cost-projection columns. `provenance.tariff.annual_network_credit` and
  `provenance.reference_tariff.annual_network_credit` record the table.
  Without the table none of these are present.
- **Estimated battery residual value.** The three residual-value fields,
  `terminal_value` in the resolved config and the optional
  `provenance.terminal_value` are described [above](#estimated-battery-residual-value).
- **Irradiance resampling.** App `provenance.weather` and Monte Carlo
  `settings` and `runtime_weather.metadata` record `irradiance_resampling`
  (requested) and `irradiance_resampling_resolved`, with per-component
  fallback and zero-support counts and the observational
  `irradiance_closure` residuals. `irradiance_resampling` is in
  `resolved_config` and in the optimizer's `simulation` config, and
  optimizer provenance carries `simulation`, `weather` and, for a real
  weather sequence, `weather_by_year`. See
  [hourly weather at 15-minute resolution](configuration.md#hourly-weather-at-15-minute-resolution).

## Migrating from BREOS 0.6.2

BREOS 0.7.0 removes the old names below without compatibility aliases; a
removed config key raises an error that names its replacement. Results of
BREOS 0.6.2 carry no `result_schema_version`; 0.7.0 results carry format
`"1"` (see [Currency and result format](#currency-and-result-format)).

### Result fields

| 0.6.2 field | 0.7.0 field or action |
|---|---|
| `total_investment_eur`, `npv_savings_eur`, `lcoe_eur_kwh`, `battery_replacement_cost_eur` (App result and sweep CSV) | `total_investment`, `npv_savings`, `lcoe_per_kwh`, `battery_replacement_cost_t0_prices` |
| Monte Carlo `npv_savings_eur`, `lcoe_eur_kwh`, `total_replacement_cost_eur`, `payback_year_exact` (runs, summary and CSV) | `npv_savings`, `lcoe_per_kwh`, `total_replacement_cost_t0_prices`, `payback_year_interpolated` |
| Optimizer `NPV_Eur`, `Objective_NPV_Eur`, `Projected_NPV_Eur`, `Projected_Initial_Cost_Eur`, `Projected_Replacement_Cost_Eur`, `Projected_LCOE_Eur_kWh` | `NPV`, `Objective_NPV`, `Projected_NPV`, `Projected_Initial_Cost`, `Projected_Replacement_Cost_T0_Prices`, `Projected_LCOE_per_kWh` |
| Optimizer `Projected_Breakeven_Year`, `Projected_Breakeven_Year_Exact` | `Projected_Payback_Year`, `Projected_Payback_Year_Interpolated` |
| Optimizer `SteadyState_*` columns and `Objective_ZEB_Ratio` | Remove; the steady-state basis is gone and ZEB is not an objective |
| `evaluate_projected_design(...).yearly` `PV_DC_kWh`, `PV_DC_Curtailed_kWh` | `PV_DC_Generation_kWh`, `Curtailment_DC_kWh` |
| `breos list cost-presets --json` `*_eur_kwh` | `*_per_kwh`, with a `currency` field |
| `co2_avoided_year1_kg` | Remove; read the existing `co2_avoided_total_year1_kg` field |
| `co2_avoided_total_kg` | Remove; read the existing `co2_avoided_total_lifetime_kg` field |
| Top-level `pv_production_kwh` | Remove; read the existing `usable_ac_system_production_kwh` field, which reports a different quantity |
| `monthly[].pv_kwh`, `yearly[].pv_kwh` | Remove; read each row's existing `usable_ac_system_production_kwh`, which reports a different quantity |
| `Legacy_PV_Production_kWh` in Monte Carlo trajectory and yearly tables and optimizer year tables | Remove; read the existing `PV_Production_kWh`, which already means usable AC production |
| Monte Carlo `mean_pv_production_kwh` | Remove; read the existing `mean_usable_ac_system_production_kwh` |
| `monthly[].import_kwh`, `yearly[].import_kwh` | Corresponding `grid_import_kwh` row field; values are unchanged |
| `monthly[].export_kwh`, `yearly[].export_kwh` | Corresponding `grid_export_kwh` row field; values are unchanged |
| `pv_loss_waterfall.stages` entry `year_1_degradation` | Remove; year 1 has no PV degradation, so `pvwatts_static` is the last stage |
| `preserve_irradiance_energy` in App `provenance.weather` and Monte Carlo `settings` and weather metadata | `irradiance_resampling` (requested) and `irradiance_resampling_resolved` |
| `provenance.resolved_config.dc_coupled` | Remove the read |
| `BatteryModelProfile.operating_defaults`, discovery JSON `operating_defaults`, and serialized `model_profile.operating_defaults` | Remove the read. Profiles did not define any defaults; this field was always empty. |

The removed legacy PV fields and `usable_ac_system_production_kwh` report
different quantities. The old value counted PV DC sent into storage before
storage losses, along with direct and exported AC. The existing usable-AC
field counts direct AC delivered to load, PV-origin battery AC delivered to
load, and exported AC. This is also the retained definition of annual
`PV_Production_kWh`. The timestep ledger still carries its `PV_Production`
field, and dispatch values do not change. The ledger schema in provenance is
now `"3.0"`.

`provenance.resolved_config` stores `calendar_model` and `load_profile` in
their canonical spelling, for example `"demandlib_h0"` for `"DemandLib_H0"`.
Other keys keep the spelling you configured.

### Configuration keys and CLI flags

| 0.6.2 key or flag | 0.7.0 replacement or action |
|---|---|
| App `dc_coupled`, `breos run --dc-coupled`, optimizer `battery.dc_coupled` | Remove. BREOS always uses its DC-coupled/hybrid dispatch. |
| `load_profile = "1"` or `"default"` | `"demandlib_h0"` |
| `load_profile = "4"`, `"5"`, `"6"` | `"eredes_btn_a"`, `"eredes_btn_b"`, `"eredes_btn_c"` |
| `load_profile = "7"`, `"8"` | `"bdew_h0"`, `"ree_2.0td"` |
| `load_profile = "h0"` | `"demandlib_h0"` (bundled) or `"bdew_h0"` (the BDEW publication) |
| `load_profile = "crest"` | `"custom"` with `load_profile_file` for a CREST export; `"crest"` loaded the demandlib H0 profile |
| `load_profile = "bdew_h0"` without `rlp_directory` | `bdew_h0` now reads the BDEW publication file you supply; use `"demandlib_h0"` for the bundled profile |
| `[montecarlo] preserve_irradiance_energy`, `breos montecarlo --preserve-irradiance-energy` | Top-level `irradiance_resampling`, `--irradiance-resampling`; see [Hourly weather at 15-minute resolution](configuration.md#hourly-weather-at-15-minute-resolution) |
| Optimizer `constraints.budget_eur` | `constraints.budget` |
| Optimizer `optimization.objective_basis = "steady_state"` | Remove; `"projected"` is the only basis |
| Optimizer `pv.params` `T_Pmax`, `T_Voc`, `T_Isc` | `T_Pmax_pct`, `T_Voc_pct`, `T_Isc_pct` |
| Optimizer `name`, `[load]`, `[pv_specs]`, `simulation.weather_file`, `optimization.algorithm`, `costs.panel_wp`, `battery.battery_type` | Remove; the weather and load are function arguments, see [Optimization](optimization.md) |
| Optimizer top-level `execution_backend` | Pass `execution_backend=` to the function |

`breos list load-profiles` lists the named keys; its `aliases` field is
removed. The [optimizer configuration keys](../api/optimization.md#configuration-keys)
lists every accepted optimizer key.

### Python API

The top-level `breos` namespace is now exactly `breos.__all__`; import
lower-level names from their modules. Removed functions include
`optimize_tilt`, `optimize_battery_size`,
`calculate_lcoe`, `calculate_financials`, `calculate_co2_savings`,
`align_load_to_pv`, the inverter presets, `resample_tmy_to_15min`, the
`preserve_irradiance_energy` argument of `resample_to_15min`, and many
plotting helpers. The `Removed` and `Changed` sections of the
[changelog](https://github.com/Str4vinci/breos/blob/develop/CHANGELOG.md)
list each one with its replacement.
