# Interpreting results

`App.result()` returns a plain Python dict, JSON-serializable, with no
pandas or numpy types. The same dict is written by the CLI's `--output`
flag.

## Top-level keys

| Key | Description |
|---|---|
| `n_modules` | Number of PV modules used in the simulation |
| `pv_kwp` | System DC nameplate capacity (kWp) |
| `battery_kwh` | Battery capacity (kWh) |
| `pv_production_kwh` | Year 1 PV production |
| `consumption_kwh` | Year 1 load |
| `self_consumption_kwh` | Year 1 PV directly self-consumed |
| `grid_import_kwh` | Year 1 energy bought from the grid |
| `grid_export_kwh` | Year 1 energy sold to the grid |
| `grid_independence_pct` | Year 1 grid independence ratio |
| `self_consumption_pct` | Year 1 self-consumption ratio |
| `total_investment_eur` | Total CAPEX |
| `payback_year` | First year with positive cumulative NPV (`None` if not reached) |
| `npv_savings_eur` | Cumulative NPV savings over the projection horizon |
| `lcoe_eur_kwh` | Levelized cost of electricity |
| `monthly` | Year 1 monthly energy balance rows |
| `financial` | Yearly financial projection rows (year 0 = investment) |
| `yearly` | Per-year breakdown of production, load, imports, exports |

## Battery-specific keys

Present only when `battery_kwh > 0`:

| Key | Description |
|---|---|
| `battery_soh_end_pct` | State of health at the end of the projection horizon |
| `battery_replacements` | Total number of replacements over the projection |
| `battery_replacement_cost_eur` | Total replacement cost |

## Emissions keys

Present only when `emissions_country` is set:

| Key | Description |
|---|---|
| `co2_avoided_year1_kg` | Year 1 CO2 savings |
| `co2_avoided_total_kg` | Lifetime CO2 savings over the projection horizon |

## Multi-array systems

When `pv_arrays` is set, the result also contains a `pv_arrays` list
echoing each array's configuration (`modules`, `module`, `tilt`,
`azimuth`).

## Monthly and yearly breakdowns

### `monthly`

A list of 12 dicts, one per month of year 1:

```python
{
    "month": "Jan",
    "pv_kwh": 245.3,
    "consumption_kwh": 412.5,
    "self_consumption_kwh": 180.2,
    "import_kwh": 232.3,
    "export_kwh": 65.1,
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
and without the BREOS-sized system. The crossover point of `balance ≥ 0`
defines `payback_year`.
