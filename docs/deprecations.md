# Deprecations for 0.6.0

BREOS 0.5.1 keeps the APIs below working. Deprecated callables emit a
`DeprecationWarning` when called, `PolysunDegradationConfig` warns when
instantiated, and directly importing `breos.numba_kernels` warns. The
`breos[fast]` extra and comparison-only constants cannot warn on use and are
announced here and in the changelog. All are scheduled for removal in BREOS
0.6.0. Python hides `DeprecationWarning` by default; run tests with `-W default`
or `-W error::DeprecationWarning` to find calls before upgrading.

The {py:class}`~breos.App` facade and its configuration are unaffected.

## Accelerated screening kernels

`breos.numba_kernels` and the `breos[fast]` optional extra are deprecated.
These approximate standalone kernels are not called by `App` or by
{py:func}`breos.battery.simulate_energy_balance`, and their degradation and
dispatch behavior does not match the reference simulation. Use
{py:func}`breos.battery.simulate_energy_balance` for supported results. There
is no supported accelerated replacement in 0.5.x.

## Polysun comparison baseline

The article-scoped `breos.polysun_degradation` module, its comparison-only
constants in `breos.constants`, and its three comparison plots are deprecated
without a package replacement:

- `PolysunDegradationConfig`, `woehler_cycles_to_failure`,
  `compute_dod_histogram`, `compute_miner_damage`,
  `predict_polysun_lifetime`, and `simulate_polysun_degradation`
- `plot_degradation_methodology_comparison`,
  `plot_lifetime_prediction_comparison`, and
  `plot_temperature_sensitivity_comparison`
- `WOEHLER_LFP_CONSERVATIVE_A`, `WOEHLER_LFP_CONSERVATIVE_B`,
  `WOEHLER_LFP_TYPICAL_A`, `WOEHLER_LFP_TYPICAL_B`,
  `WOEHLER_LFP_OPTIMISTIC_A`, `WOEHLER_LFP_OPTIMISTIC_B`,
  `POLYSUN_CALENDAR_LIFE_LION`, and `POLYSUN_CALENDAR_LIFE_LEAD`

Copy the comparison implementation into the research artifact that needs it
before moving to 0.6.0. BREOS's supported degradation models are documented in
[Degradation models](api/degradation-models.md).

## Undocumented plotting helpers

The following unverified plotting helpers are deprecated without a direct
replacement:

- `plot_smart_charging_sweep`
- `plot_optimization_results_2d` and `plot_optimization_results_3d`
- `plot_loo_cv_summary`, `plot_loo_param_stability`, and
  `plot_loo_predictions`

The supported plotting surface remains in [Plotting](api/plotting.md).

## Orphaned module helpers

| Deprecated API | Migration |
|---|---|
| `breos.io.save_simulation_report` | Call `export_results`, `export_summary`, and `export_cost_analysis` for the artifacts needed by the application. |
| `breos.io.export_monthly_summary` | Aggregate numeric columns with `DataFrame.resample("ME").sum()`, then write the result with pandas. |
| `breos.io.export_yearly_summary` | Aggregate numeric columns with `DataFrame.resample("YE").sum()`, then write the result with pandas. |
| `breos.weather.resample_to_hourly` | Use `DataFrame.resample("h")` with the required aggregation. |
| `breos.weather.csv_15min_to_hourly` | Read and write the CSV with pandas and use `DataFrame.resample("h")`. |
| `breos.weather.csv_hourly_to_15min` | Read and write the CSV with pandas around `breos.weather.resample_to_15min`. |
| `breos.weather.fetch_tmy_nsrdb` | Use `breos.weather.fetch_tmy_weather_data` for the supported PVGIS TMY path. |
| `breos.solar.calculate_pv_production_tmy` | Call `breos.solar.calculate_pv_production_dc`; TMY data needs no special production wrapper. |
| `breos.solar.zeb_sizer` | Compute the annual usable-AC-production to load ratio in application code. |
| `breos.optimization.optimize_tilt_brent` | Use the supported `breos.optimization.optimize_tilt` grid search. |
| `breos.optimization.size_for_zeb` | Compute the annual usable-AC-production to load ratio in application code. |
| `breos.utils.count_leap_years` | Sum `breos.utils.is_leap_year(year)` over the required range. |
| `breos.utils.number_of_cores` | Use `os.cpu_count()` and apply the desired worker policy in application code. |
| `breos.battery.compute_halfcycle_energy_throughput` | Compute throughput directly from the cycle boundaries when maintaining a custom degradation model. |
| `breos.battery.k_c_rate_Q` | Keep this equation with the custom degradation model that uses it. |
| `breos.battery.k_doc_Q` | Keep this equation with the custom degradation model that uses it. |
| `breos.battery.update_battery_soc` | Derive SOC from the energy ledger in application code; BREOS's simulation handles this internally. |

All deprecated functions retain their 0.5.0 signatures and return values
during 0.5.x. The warnings identify 0.6.0 as the earliest removal release.
