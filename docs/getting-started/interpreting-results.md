# Interpreting results

`App.result()` returns a plain Python dict, JSON-serializable, with no
pandas or numpy types. The same dict is written by the CLI's `--output`
flag.

## Currency and schema version

Money keys carry no currency. Every money value in a result is in the run's
currency, which `provenance.currency` records: the tariff's `currency` when
the run has a `[tariff]` table, otherwise `EUR`, the currency of the bundled
cost catalogue. BREOS does not convert currencies. Summary and plot labels
read the recorded currency.

`result_schema_version` versions the result's names, independently of the
ledger schema. Version `"1.0"` dropped the `_eur` suffixes and renamed the
`_exact` payback fields to `_interpolated`, with no aliases (the
[changelog](https://github.com/Str4vinci/breos/blob/develop/CHANGELOG.md)
lists every rename). Version `"1.1"` adds the
[year-1 money keys](#year-1-money-keys), `"1.2"` adds `provenance.economics`
([Economic conventions](#economic-conventions)), `"1.3"` adds
`Replaced_Capacity_kWh` to the year rows of Monte Carlo trajectories and
optimizer tables, `"1.4"` adds `provenance.revaluation` to the results of
[`App.revalue`](recipes.md#revalue-a-run-at-other-prices) and the export CO2
columns (`CO2_Avoided_Export_kg`, `CO2_Avoided_Export_Cumulative_kg`) to the
cost projection and Monte Carlo trajectories, `"1.5"` adds `constraints` and
`run_settings` to the optimizer's provenance and the `Projected_CO2_*`
columns to the Pareto rows of a search with `[emissions]`, `"1.6"` adds
`inverter_ac_rating_kw` to the `resolved_config` of App and Monte Carlo
results, `"1.7"` adds the [`period` keys](#period-runs) of a run over part
of a year, and `"1.8"` adds `calendar_year`, the
`target_year` the load was built for, to Monte Carlo's
`provenance.load_profile`.
Version "2.0" removes configuration and metadata fields without an
operational effect, removes duplicate CO2 and legacy PV keys, and renames
monthly/yearly grid fields and optimizer payback columns. The migration table
below lists every change.
Version "2.1" adds `tariff.custom_schedule` to the `resolved_config` of App
and Monte Carlo results that set an inline
[custom schedule](../api/tariffs.md#custom-app-schedules).
Version "2.2" adds the record of the experimental
[daily-persistence smart charging](configuration.md#daily-persistence-experimental)
to `provenance.smart_charging` of App results that configure it:
`experimental`, `controller_version`, `planner_version`,
`forecast_horizon_days`, `target_levels`, `soc_states`, `forecast_policy`,
`warm_start_policy`, `planner_terminal_policy`, and the
`initial_stored_energy` and `final_stored_energy` by origin. Other results
are unchanged.
Version "2.3" adds `battery_allow_terminal_replacement` to the
`resolved_config` of App and Monte Carlo results, and
`battery_replacement_treatment`, with its `allow_terminal_replacement`
policy and a `terminal_period` description, to the provenance of a projected
design and of an optimizer search. See
[battery replacement at the end of the horizon](configuration.md#battery-replacement-at-the-end-of-the-horizon).
Default results are otherwise unchanged.
Version "2.4" adds calendar-month
[seasons](../api/tariffs.md#month-seasons) to custom schedules: `seasons` in
`resolved_config.tariff.custom_schedule` and, as the month partition, in
`provenance.tariff`. With month seasons, `import_prices` and `export_prices`
in the resolved config and in `provenance.tariff` may map each season to its
period prices instead of each period to a price. Results without month
seasons are unchanged.
Version "2.5" adds the [`discharge_only`](configuration.md#discharge-only)
smart-charging mode to `provenance.smart_charging`: such a run records its
discharge periods, an empty `charge_periods`, and `None` for
`target_usable_fraction`, `grid_charge_efficiency` and
`grid_import_limit_w`. The App's top-level `smart_charging` block reports it
as it reports the other modes. Other results are unchanged.
Version "2.6" adds the no-system cost components: `no_system_fixed_charge_year1_prices`
(see [Year-1 money keys](#year-1-money-keys)), `no_system_cost_import` and
`no_system_cost_fixed_charge` in the `financial` rows, the
`Cost_No_Sys_Import` and `Cost_No_Sys_Fixed_Charge` cost-projection columns
and the `Baseline_Fixed_Charge` year-row column of Monte Carlo and optimizer
tables, and `reference_tariff` in the
`resolved_config` of App and Monte Carlo results. Results with a
[no-system reference tariff](configuration.md#no-system-reference-tariff)
also carry `provenance.reference_tariff`. Without one, every existing value is
unchanged.
A renamed or removed key bumps the
major version, an added key the minor. A result without the key predates 1.0.

## Top-level keys

| Key | Description |
|---|---|
| `result_schema_version` | Version of the result's names (see above) |
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
| `lcoe_per_kwh` | Levelized cost of electricity from system CAPEX, O&M, simulated replacements, and discounted PV production |
| `monthly` | Year 1 monthly energy balance rows |
| `financial` | Yearly financial projection rows (year 0 = investment) |
| `yearly` | Per-year breakdown of production, load, imports, exports |
| `pv_loss_waterfall` | Year 1 PV loss waterfall from irradiance reference through PVWatts losses, inverter losses, and dispatch losses |
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
| `no_system_import_cost_year1_prices` | Import cost of the household without a system, which buys its whole year-1 load, `consumption_kwh`. It is the import cost only; it does not include the fixed charge |
| `grid_charge_cost_year1_prices` | Present only with smart charging (`smart_charging.mode = "fixed_target"`, `"daily_persistence"` or `"discharge_only"`): the part of `grid_import_cost_year1_prices` bought to charge the battery. Always 0 with `discharge_only`, which never charges from the grid |
| `no_system_import_cost_year1_prices` | Import cost of the household without a system, which buys its whole year-1 load, `consumption_kwh`. It is the import cost only; the fixed charge is `no_system_fixed_charge_year1_prices` |
| `no_system_fixed_charge_year1_prices` | The fixed charge of the household without a system for year 1: `fixed_charge_year1_prices`, or the reference tariff's fixed charge for the same days when a `[reference_tariff]` is set |
| `grid_charge_cost_year1_prices` | Present only with grid-charging smart charging (`smart_charging.mode = "fixed_target"` or `"daily_persistence"`): the part of `grid_import_cost_year1_prices` bought to charge the battery |

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

Present only when `battery_kwh > 0`:

| Key | Description |
|---|---|
| `battery_soh_end_pct` | State of health at the end of the projection horizon |
| `battery_replacements` | Total number of replacements over the projection |
| `battery_replacement_cost_t0_prices` | Total replacement cost at t = 0 prices, neither inflated nor discounted |
| `battery_replacement_cost_npv` | The same replacements inflated to and discounted from each swap instant, as `npv_savings` counts them |

With `battery_allow_terminal_replacement = false`, a pack that reaches end of
life in the final degradation period of the horizon is not replaced. The
replacement count and costs then leave out that one swap, and
`battery_soh_end_pct` can end below the end-of-life threshold. See
[Battery replacement at the end of the horizon](configuration.md#battery-replacement-at-the-end-of-the-horizon).

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

## Migrating from result schema 1.8

Schema 2.0 removes the old names without compatibility aliases. Update readers
using the following mappings:

| Schema 1.8 field | Schema 2.0 field or action |
|---|---|
| App config `dc_coupled` and `provenance.resolved_config.dc_coupled` | Remove the config key; it is now unknown. BREOS always uses its supported DC-coupled/hybrid dispatch. |
| `BatteryModelProfile.operating_defaults`, discovery JSON `operating_defaults`, and serialized `model_profile.operating_defaults` | Remove the read. Profiles did not define any defaults; this field was always empty. |
| `co2_avoided_year1_kg` | Remove; read the existing `co2_avoided_total_year1_kg` field |
| `co2_avoided_total_kg` | Remove; read the existing `co2_avoided_total_lifetime_kg` field |
| Top-level `pv_production_kwh` | Remove; read the existing `usable_ac_system_production_kwh` field, which reports a different quantity |
| `monthly[].pv_kwh`, `yearly[].pv_kwh` | Remove; read each row's existing `usable_ac_system_production_kwh`, which reports a different quantity |
| Shared annual `Legacy_PV_Production_kWh` | Remove; read the existing `PV_Production_kWh`, which already means usable AC production |
| Monte Carlo `mean_pv_production_kwh` | Remove; read the existing `mean_usable_ac_system_production_kwh` |
| `monthly[].import_kwh`, `yearly[].import_kwh` | Corresponding `grid_import_kwh` row field; values are unchanged |
| `monthly[].export_kwh`, `yearly[].export_kwh` | Corresponding `grid_export_kwh` row field; values are unchanged |
| Optimizer `Projected_Breakeven_Year` | `Projected_Payback_Year` |
| Optimizer `Projected_Breakeven_Year_Interpolated` | `Projected_Payback_Year_Interpolated` |

The removed legacy PV fields and `usable_ac_system_production_kwh` report
different quantities. The old value counted PV DC sent into storage before
storage losses, along with direct and exported AC. The existing usable-AC
field counts direct AC delivered to load, PV-origin battery AC delivered to
load, and exported AC. This is also the retained definition of annual
`PV_Production_kWh`. The timestep ledger still carries its `PV_Production`
field, and dispatch values do not change.

Schema 2.0 does not normalize enum spelling in echoed provenance: for example,
the configured spelling remains in `provenance.resolved_config`.

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

Timing (ADR 0003 E3):

- The initial investment is at year 0.
- Energy, the fixed charge and O&M are at year-1 prices, escalated
  `(1 + rate)^(n − 1)` in year `n` and booked at the end of the year, so
  discounted by `(1 + discount_rate)^n`. When an escalator equals the
  discount rate, one year of discounting still remains.
- A battery replacement is priced at today's (t = 0) storage cost, inflated
  to the instant of the swap and discounted from that instant, not from a
  year boundary. The simulation reports only when a pack was swapped and its
  capacity; the economics prices it (ADR 0003 E4), the same way for the App,
  Monte Carlo and the optimizer.
- The daily fixed charge is billed on the simulated duration: 365 days for a
  common year, 366 for a leap year.

## Monthly and yearly breakdowns

### `monthly`

A list of 12 dicts, one per month of year 1:

```python
{
    "month": "Jan",
    "usable_ac_system_production_kwh": 245.3,
    "consumption_kwh": 412.5,
    "self_consumption_kwh": 180.2,
    "grid_import_kwh": 232.3,
    "grid_export_kwh": 65.1,
    "grid_independence_pct": 43.7,
}
```

### `yearly`

A list of one dict per simulation year (length `projection_years`). Each
row contains the same fields as `monthly` aggregated to a year, plus
`soh_pct` when a battery is present.

### `financial`

A list of dicts with one row per year (year 0 is the investment row):

```python
{"year": 0, "balance": -8500.0, "reference": 0.0}
{"year": 1, "balance": -7950.4, "reference": 0.0, "cost_with_system": 542.1, "cost_without_system": 1092.5}
# ...
```

`balance` is the cumulative NPV savings; `cost_with_system` and
`cost_without_system` are the cumulative discounted costs of operating with
and without the BREOS-sized system. From year 1, `no_system_cost_import` and
`no_system_cost_fixed_charge` give that year's no-system cost by component,
escalated and not discounted, beside the system's `cost_import`,
`revenue_export`, `cost_operation`, `cost_fixed_charge` and
`cost_replacement`. With a
[no-system reference tariff](configuration.md#no-system-reference-tariff),
they are at the reference's prices and escalation. `payback_year` is the sustained discounted
payback within the simulated period: the year from which `balance ≥ 0` holds
to the end of the horizon. The series starts at year 0, so a system that
recovers its investment during year 1 reports 1. If a battery replacement
turns `balance` negative again, payback is the later recovery, and a
`balance` that is negative in the last year means no payback.
`economics.find_payback_year_interpolated` gives the same crossing as a fractional
year, interpolated linearly between the annual points; it is an estimate
from year-end values, not an exact date.

## Period runs

A run with a [`[period]`](recipes.md#simulate-part-of-a-year) window
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
    "skipped_fields": ["payback_year", "npv_savings", "lcoe_per_kwh", "financial"],
}
```

`skipped_fields` lists the keys of this result that are `None` for that
reason; the example is a PV-only run without emissions.
