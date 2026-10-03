# Optimization

[pymoo](https://pymoo.org/) powers multi-objective PV/battery sizing over
module count, battery capacity, tilt and azimuth, and optionally an East–West
layout, against projected grid
independence and NPV; ZEB is a diagnostic. The
[optimization guide](../getting-started/optimization.md) walks through a
search. For end-to-end App
runs over an explicit config grid, including one-dimensional tilt or battery
sweeps, use the `breos sweep` CLI command documented in
[Run a parameter sweep](../how-to/parameter-sweep.md).

Install `breos[optimization]` to use pymoo-backed multi-objective sizing.
`evaluate_projected_design` uses the core scientific stack.

ZEB and financial production use usable AC system energy from the dispatch
ledger, not raw PV DC, so inverter efficiency and clipping affect candidate
scores. System kWp, inverter rating, and CAPEX use the selected module's
`Mpp`; module frame area uses `pv.dimensions`.

## Projected objectives

`optimize_system_multi_objective` evaluates every candidate over
`simulation.years_projection` years, or `financials.project_lifespan` when that
key is absent. It optimizes two values: projected lifetime grid independence
and projected NPV. A design is selected for how it performs across the project
lifetime, which is the question a sizing study asks.
`optimization.objective_basis = "projected"` names this basis and is the only
accepted value. The annual `"steady_state"` basis was removed in 0.7.0, and a
config that still sets it raises an error.

Projected mode repeats the configured TMY. Each year applies the configured PV
degradation factor and carries battery stored energy, PV-origin stored energy,
SoH, full-equivalent cycles, calendar time, cycle and calendar degradation,
resistance growth, and supported degradation-engine state into the next year.
Replacement remains enabled, resets the battery through the production battery
engine, and adds the actual event cost to that year's financial ledger.
`battery.allow_terminal_replacement = false` skips only a replacement in the
final degradation period of the last project year.
`battery.replacement_min_remaining_years` skips any replacement that leaves
the new pack less than that many project years before the end of the
projection; see
[Battery replacement at the end of the horizon](../getting-started/configuration.md#battery-replacement-at-the-end-of-the-horizon).

Lifetime grid independence is calculated from aggregate energy, not from the
mean annual percentage:

```text
GI_lifetime = 100 × (1 − total lifetime grid import / total lifetime load)
```

Projected NPV uses each simulated year's import, export, load, usable AC PV
production, and replacement cost. This is a repeated-TMY scenario, not a
forecast of distinct future weather years.

Optional `tariff`, `reference_tariff` and `smart_charging` tables use the
App's schema and validation; the
[optimization guide](../getting-started/optimization.md#price-a-design-with-a-time-of-use-tariff)
describes how a search uses them. The experimental, App-only
`daily_persistence` smart-charging mode raises `ValueError`.

ZEB remains a reported diagnostic in projected mode. Set
`constraints.enforce_zeb = true` to require a projected lifetime ZEB ratio of
at least one; this adds a feasibility constraint, not a third objective.

### Configuration keys

`breos.optimization_config.resolve_optimization_config` checks the nested
config once, before any candidate is scored, and fills in every default by
name. Every table takes a fixed set of keys, and an unknown key raises, so a
misspelt key cannot be ignored. The weather and load are not config keys: the
call takes them as DataFrames.

| Table | Keys (default) |
| --- | --- |
| `location` | `latitude`, `longitude` (required); `timezone` (`"UTC"`); `altitude` (looked up from the coordinates); `name` (`""`) |
| `pv` | `module` (the App's default module); `params` (an inline module: `Mpp`, `Vmp`, `Imp`, `Voc`, `Isc` required; temperature coefficients `T_Pmax_pct`, `T_Voc_pct` and `T_Isc_pct`, `N_Cells` and `celltype` optional); `dimensions` or `module_width_m` and `module_length_m` (1.134 × 2.278 m); `degradation_rate` (0.005, or `financials.pv_degradation_rate`) |
| `battery` | the `BatteryConfig` keys `min_soc`, `max_soc`, `charge_efficiency`, `discharge_efficiency`, `standby_loss_wh`, `eol_percentage`, `max_charge_power_w`, `max_discharge_power_w`, `power_limit_c_rate`, `calendar_model`, `enable_resistance_fade` (the App's defaults); `temperature` (`"weather"`); `indoor_model` (the App's `battery_indoor_model` table); `degradation_engine` (`"native"`); `blast_model`; `replacement_cost` (storage cost per kWh times capacity); `enable_replacement` (`true`); `allow_terminal_replacement` (`true`); `replacement_min_remaining_years` (0); `initial_soh` (100) |
| `costs` | the App's `costs` keys, plus `dc_ac_ratio` (1.25), the DC peak over the inverter AC rating |
| `financials` | `inflation_rate` (0.02), `sell_price_inflation` (0), `import_price_escalation`, `om_escalation`, `replacement_cost_learning`, `discount_rate` (0.03), `project_lifespan`, `pv_degradation_rate`, and the flat-price fallbacks `electricity_cost` and `electricity_sold_cost` |
| `constraints` | see below |
| `mode` | `fixed_azimuth` (none: azimuth is searched, 90–270° in the northern hemisphere, −90–90° in the southern); `layouts` (`["single"]`; add `"east_west"` to search East–West roofs too); `east_west_tilt_deg` (10), the fixed tilt of an East–West design |
| `optimization` | `objective_basis` (`"projected"`); `early_stop` (off; `true` or a table turns it on, and a table takes `enabled` (`true`), `ftol` 0.0025, `period` 10, `n_skip` 0, `min_gen` min(20, `n_gen`), `only_feasible` `true`); `pop_size` (40), `n_gen` (100), `n_offsprings` (pymoo's), `seed` (1) |
| `simulation` | `resolution` (`"h"`, or `"15min"`); `irradiance_resampling` (`"auto"`); `years_projection` (20, or `financials.project_lifespan`) |
| `inverter` | `efficiency`, used when the top-level `inverter_efficiency` (0.96) is not set |
| `emissions` | the `EmissionsParams` fields; the search then reports `Projected_CO2_*` for every Pareto row |
| `tariff`, `reference_tariff`, `smart_charging`, `terminal_value` | the App's tables; `terminal_value` is accepted and ignored |
| top level | `pv_module`, `inverter_efficiency`, `dc_output_scale` (1), `ac_output_scale` (1), and the App's PV model keys (`transposition_model`, `albedo`, `iam_model`, ...) |

Where two keys set one thing, the first one set wins:
`simulation.years_projection` over `financials.project_lifespan`,
`pv.degradation_rate` over `financials.pv_degradation_rate`,
`inverter_efficiency` over `inverter.efficiency`, and `pv.module` over
`pv_module`. `optimize_system_multi_objective` takes `pop_size`, `n_gen`,
`n_offsprings` and `seed` as arguments too; an argument and its
`[optimization]` key that disagree raise. The optimizer never reads
`execution_backend` from the config: pass it to the function.

`dc_output_scale` and `ac_output_scale` are optimizer-only keys: the App and
the Monte Carlo runner do not take them.

The search bounds:

| Key | Default | Meaning |
| --- | ---: | --- |
| `constraints.budget` | 10,000 | Maximum initial system cost, in the run's currency. `budget_eur`, its name before 0.7.0, is an error |
| `constraints.max_area_m2` | 20 | Maximum PV-module frame area in m² |
| `constraints.max_battery_kwh` | 30 | Maximum battery decision-variable value in kWh |
| `constraints.max_modules` | 60 | Maximum PV-module decision-variable value |
| `constraints.min_tilt_deg` | 10 | Minimum tilt in degrees |
| `constraints.max_tilt_deg` | 90 | Maximum tilt, or `"adjust"` for the latitude-based bound |
| `constraints.tilt_margin_deg` | 15 | Margin over the latitude for `"adjust"`: 5° × round((\|latitude\| + margin) / 5), between 60 and 90 |
| `constraints.enforce_zeb` | `false` | Add the ZEB feasibility constraint |

Set the physical and financial limits explicitly for any study you intend to
report. The defaults preserve earlier direct-API behavior; they are not
site-specific recommendations. The search records the bounds it used,
defaults included, in `details["provenance"]["constraints"]`, and its run
settings in `details["provenance"]["run_settings"]`.

With more than one entry in `mode.layouts`, a layout gene is added after the
tilt and azimuth genes. An East–West design does not read those two genes,
and the repair sets them to their lowest grid value so that NSGA-II drops
East–West duplicates. An East–West design puts floor(n / 2) modules at
azimuth 90 and the rest at azimuth 270, at `east_west_tilt_deg`; its DC output
is the sum of the two arrays, computed as the App computes `[[pv_arrays]]`.
The Pareto table's `Layout` column names each design's layout; an East–West
row has the configured `Tilt` and a NaN `Azimuth`.
`details["provenance"]["mode"]` records the layouts and tilt used.
`evaluate_projected_design` takes `layout="east_west"` with a `tilt` and no
`azimuth`, and its metrics carry `Layout` too.

Results expose `Projected_*` diagnostics. The ordinary `Grid_Independence_%`
and `NPV` columns equal `Projected_Grid_Independence_%` and
`Projected_NPV`. `ZEB_Ratio` mirrors `Projected_ZEB_Ratio` but is not an
objective. `Objective_Grid_Independence_%` and `Objective_NPV` identify the
metrics sent to NSGA-II explicitly.

`details["provenance"]`, and the `provenance` of `evaluate_projected_design`,
carry `result_schema_version` and `currency`, the currency of every money
column (see [Interpreting results](../getting-started/interpreting-results.md#currency-and-result-format)),
plus the tariff and smart-charging records when the config has them. Their
`battery_replacement_treatment` records the replacement method, the configured
`allow_terminal_replacement` and what the terminal period is, and the
configured `replacement_min_remaining_years` with a `minimum_service`
description;
`details["battery_replacement_treatment"]` holds the same record.

Use `evaluate_projected_design` when you need the detailed result for one
fixed design instead of a Pareto search. It returns the projected metrics, the
annual energy and degradation-state ledger, and the matching discounted
financial ledger. LCOE and configured lifetime avoided-emissions totals are
included in the metrics, while the financial table retains their annual source
columns. These tables are intended as stable source data for custom
analysis and plots; BREOS does not require a particular visualization layer.

>>>

## Multi-objective sizing

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.optimization.optimize_system_multi_objective
   breos.optimization.evaluate_projected_design
   breos.optimization_config.resolve_optimization_config
```

## Result type

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.optimization.OptimizationResult
   breos.optimization.ProjectedDesignResult
```
