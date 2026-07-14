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
| `pv_production_kwh` | Legacy AC-equivalent non-curtailed PV compatibility field |
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
| `total_investment_eur` | Total CAPEX |
| `payback_year` | First year with positive cumulative NPV (`None` if not reached) |
| `npv_savings_eur` | Cumulative NPV savings over the projection horizon |
| `lcoe_eur_kwh` | Levelized cost of electricity from system CAPEX, O&M, simulated replacements, and discounted PV production |
| `monthly` | Year 1 monthly energy balance rows |
| `financial` | Yearly financial projection rows (year 0 = investment) |
| `yearly` | Per-year breakdown of production, load, imports, exports |
| `pv_loss_waterfall` | Year 1 PV loss waterfall from irradiance reference through PVWatts losses, inverter losses, and dispatch losses |
| `provenance` | BREOS version, normalized resolved config, ledger schema version, weather/location metadata, resolution, timezone, and start date |

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
| `co2_avoided_self_consumption_year1_kg` | Year 1 behind-the-meter benefit |
| `co2_avoided_export_year1_kg` | Year 1 exported-generation benefit |
| `co2_avoided_total_year1_kg` | Sum of the two year 1 pathways |
| `co2_avoided_self_consumption_lifetime_kg` | Lifetime behind-the-meter benefit |
| `co2_avoided_export_lifetime_kg` | Lifetime exported-generation benefit |
| `co2_avoided_total_lifetime_kg` | Sum of the two lifetime pathways |
| `co2_avoided_year1_kg`, `co2_avoided_total_kg` | Compatibility aliases for the corresponding total fields |

Self-consumption uses the preset's avoided-grid factor. Export uses
`export_emissions_factor_gco2_kwh` when configured; otherwise it explicitly
falls back to the same avoided-grid factor. Curtailed energy, conversion and
storage losses, initial SOC, and PV energy remaining stored at the reporting
boundary receive no credit.

## Multi-array systems

When `pv_arrays` is set, the result also contains a `pv_arrays` list
echoing each array's configuration (`modules`, `module`, `tilt`,
`azimuth`).

## PV loss waterfall

`pv_loss_waterfall` reports the year 1 PV production chain in kWh. Its
ordered `stages` cover only the linear PV-model chain: horizontal reference,
transposition, incidence-angle modifier, cell temperature, static PVWatts
losses, and year 1 degradation. Dispatch is a branching flow and is therefore
reported under `energy_balance`, not forced into a misleading linear stage.

The `pvwatts` block contains fixed-loss percentages and attributed kWh. The
`inverter` block reports AC rating plus separate direct-PV and
battery-discharge conversion losses. `energy_balance.pv_dc` reconciles PV
routing; `energy_balance.ac_delivery` reconciles delivered/exported AC; and
`energy_balance.battery_stored_energy` reconciles beginning/end energy,
charge, discharge, standby, capacity-window, and replacement boundary flows.

Use {py:func}`breos.plotting.plot_pv_loss_waterfall` to render the same
block as a PV loss diagram.

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
