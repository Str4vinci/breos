# Changelog

All notable changes to BREOS are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

## [0.4.1] - 2026-07-23

### Changed
- Extracted BLAST experimental-range and aging-horizon validation into a focused
  internal warning collector while preserving warning records, snapshot
  continuation, replacement-reset behavior, and the `breos.App` result schema.
- Centralized native and BLAST degradation-result construction in a focused
  schema-aware builder while preserving public fields, engine-specific SOH
  precision, and shared result/provenance data.
- Decomposed `breos.App` configuration validation into focused subsystem
  validators while preserving validation order, public errors, normalization,
  defaults, and resolution precedence.
- Deferred optional plotting imports until a plotting compatibility attribute
  is first accessed, keeping core imports and non-plotting CLI commands quiet
  while preserving existing top-level names.
- Improved first-run onboarding with a no-file/no-network dry run,
  troubleshooting guidance, corrected discovery commands, and synchronized
  support and release documentation.
- Added grouped weekly dependency updates, core-package coverage reporting that
  excludes vendored BLAST-Lite, and lightweight macOS/Windows public-entrypoint
  smoke tests.

## [0.4.0] - 2026-07-20

### Added
- Vendored BLAST-Lite degradation models as an opt-in `degradation_engine="blast"`
  path with the `BlastEngine` adapter, daily endpoint-grid integration,
  cross-year snapshot threading, replacement reset handling, and validation that keeps
  BLAST disabled for Monte Carlo and resistance-fade runs until those paths are
  explicitly supported.
- A declarative 14-model BLAST registry with stable keys, chemistry and cell
  metadata, experimental ranges, study citations, output capabilities,
  upstream provenance, Python discovery (`list_battery_models()`), CLI
  discovery (`breos list battery-models`), explicit CLI model selection, and
  versioned JSON-safe engine snapshots. BLAST result/provenance blocks identify
  the cell model, calibration basis, initial/final SOH, replacements, warnings,
  and state schema.
- All 14 BLAST models enabled with multi-condition parameter and trajectory
  parity against the pinned upstream source, deduplicated experimental-range
  warnings, and sourced aging-horizon warnings carried through snapshots.

### Changed
- The App energy balance and public `dc_to_ac()` helper now share the same
  PVWatts part-load inverter curve. Dispatch therefore accounts for loading-
  dependent conversion losses when serving load, exporting PV, and
  discharging the battery, while preserving the shared AC nameplate,
  charge-before-export behavior, and explicit 0.3.4 energy ledger. Lower-
  level `BatteryConfig` callers that omit an inverter nameplate retain the
  legacy unbounded flat-efficiency fallback because part load is undefined
  without a rated power.
- App configuration now resolves user values over sourced battery-profile
  defaults over global defaults. Native Naumann/Lam remains the default;
  `blast_model` requires explicit `degradation_engine="blast"`. The ambiguous
  App-level `battery_type` selector, which strict App validation already
  rejected as unknown in 0.3.4, now raises targeted migration guidance instead
  of being repurposed for chemistry/model selection. The lower-level
  `BatteryConfig(battery_type="LFP")` API remains supported.

### Fixed
- BLAST time-varying state updates are guarded against trajectory-inversion
  domain overshoot. When day-varying stressors shrink a state's rate
  coefficient or sigmoid asymptote between updates, the accumulated state can
  fall outside the domain of the trajectory inversion; the next update
  previously returned NaN that silently corrupted SoH (`nmc_lto_10ah` around
  day 3 and `nca_grsi_sonymurata_2p5ah` around year 9 of real multi-year
  profiles). A saturated sigmoid state (`y0 >= y_inf` after `y_inf` shrank) now
  holds its accumulated loss by returning a zero increment instead of snapping
  it back down to `y_inf`; the earlier clamp produced a negative increment that
  decreased an accumulated degradation state and manufactured artificial
  capacity recovery. Increments for the supported positive sigmoid-loss
  trajectories are therefore never negative. Constant and periodic profiles
  never hit the guards, so the golden and multi-condition parity fixtures are
  unchanged.
- `BlastEngine.step` now raises `BlastNumericalError` — reporting the model key,
  elapsed days, and offending field names — whenever any newest state or output
  value is non-finite, not only when capacity `q` is, matching the documented
  fail-loud contract instead of propagating a corrupt state.

### Notes
- The Panasonic NCA model (`nca_gr_panasonic_3ah`) emits a
  `BlastAgingHorizonWarning` once a projection extends past its sourced 300-day
  aging-data horizon, so multi-year projections with that cell warn that they
  extrapolate beyond the calibration window.

### Acknowledgments
- Thanks to Paul Gasper at NLR for suggesting the BLAST-Lite integration.

## [0.3.4] - 2026-07-14

### Added
- A versioned, explicit DC/AC timestep energy ledger now reports PV routing,
  direct and battery inverter losses, cell conversion losses, standby,
  capacity-window adjustments, storage boundary state, and PV-origin battery
  delivery. Optional battery charge (DC input) and discharge (AC delivered)
  power limits are available through `BatteryConfig`, `App`, Monte Carlo, and
  CLI configuration.
- `App.result()` separates behind-the-meter, export, and total CO2 benefits
  and includes JSON-serializable model/configuration provenance.
- `App.result()` now includes `pv_loss_waterfall`, a year-1 diagnostic that
  reports the PV chain from horizontal irradiance reference through
  transposition, IAM, cell temperature, static PVWatts losses, year-1
  degradation, inverter clipping/conversion, surplus curtailment, and battery
  dispatch losses. `breos run` JSON output includes the same block, and
  `breos.plotting.plot_pv_loss_waterfall` renders it as a PV loss diagram.
- `temperature_model` App config key, `--temperature-model` CLI flag, and
  `temperature_model=` parameter on every solar-chain function. The
  `"pvsyst-freestanding"`, `"pvsyst-semi-integrated"`, and
  `"pvsyst-insulated"` presets use pvlib's PVsyst cell-temperature model
  with its documented mounting coefficient sets. The default `"faiman"`
  (open-rack Faiman coefficients, bit-for-bit unchanged) runs cool for
  roof-mounted systems — BREOS's primary audience — and systematically
  overestimates their yield; rooftop studies should pick a roof preset.
  The validation suite runs a `perez_roof` (semi-integrated) config so the
  rooftop yield delta is documented per site.
- `diffuse_iam` App config key, `--diffuse-iam` CLI flag, and `diffuse_iam=`
  parameter on every solar-chain function. `"marion"` applies the
  incidence-angle modifier to the sky- and ground-diffuse POA components via
  pvlib's view-factor-integrated `iam.marion_diffuse` (Marion 2017), using
  the same ashrae model as the beam IAM. Beam-only IAM (diffuse passing at
  1.0) was a known ~0.5–1.5% systematic overestimate; across the validation
  suite `"marion"` lowers annual yield 1.1–2.0% and moves BREOS toward the
  PVGIS reference at all seven sites (e.g. Porto +9.4% → +7.8%). The default
  `"none"` reproduces prior behaviour bit-for-bit; the validation suite now
  runs a `perez_diffuse` config so the effect is tracked per site.
- `solar_position` App config key, `--solar-position` CLI flag, and
  `solar_position=` parameter on every solar-chain function
  (`calculate_pv_production_dc`/`_dc_tracking`/`_ac`/`_tmy`,
  `calculate_multi_array_production`). `"mid-interval"` evaluates the sun
  half a timestep after each label — the PVWatts/SAM convention for
  interval-averaged irradiance (an hourly value labelled 07:00 representing
  the 07:00–08:00 average pairs with the 07:30 sun) — and also drives
  tracker rotation angles. The default `"interval-start"` reproduces prior
  behaviour bit-for-bit. The validation suite now runs a third
  `perez_mid` config so the effect is measured per site.
- A standing validation suite under `validation/` (repo-side, not shipped):
  seven sites on four continents with committed PVGIS TMY weather inputs
  (trimmed to the five columns BREOS reads and gzipped, ~90 KB per site),
  independent PVGIS PVcalc reference results (PVWatts v8 fetcher included,
  references pending network access to `developer.nrel.gov`), a comparison
  report generator, and `tests/test_validation_drift.py`, which fails CI when
  BREOS output drifts >0.1% from its committed baseline or falls outside a
  ±10% gross-error band around the PVGIS reference.

### Changed
- `PV_Production` remains as `(PV_DC - curtailed DC) × inverter efficiency`
  for compatibility. New self-consumption, emissions, loss reporting, and
  optimizer objectives use explicit AC ledger flows. Consumers should migrate
  to `pv_ac_system_kwh`, direct/load/export fields, and the versioned ledger.
- Real calendar-year load profiles now include leap day (8,784 hourly or
  35,136 quarter-hourly intervals) while preserving exact annual energy.
- `diffuse_iam="marion"` now keeps fixed-tilt arrays on pvlib's exact Marion
  integration path but uses a cached 0.5° tilt grid for tracking arrays,
  avoiding thousands of repeated sky/ground diffuse IAM integrations per run.
- POA transposition now receives the refraction-corrected apparent zenith,
  matching pvlib's `ModelChain`. Previously one call mixed zenith
  definitions: the AOI/IAM step used `apparent_zenith` while
  `get_total_irradiance` got the true `zenith`. Annual yields move by well
  under 0.1% (refraction only matters near the horizon); the validation
  baseline was regenerated accordingly.

### Fixed
- Unsupported AC-coupled dispatch (`dc_coupled=False`) now fails early instead
  of silently executing the DC-coupled model.
- The DC-coupled dispatcher now shares inverter headroom between PV and battery
  discharge, routes above-headroom PV to storage before curtailment, records
  battery-discharge inverter loss, closes with non-zero delta SOC, and reports
  temperature/SOH capacity-window changes explicitly.
- Resistance calendar aging now uses daily mean cell temperature rather than
  the final timestep's temperature.
- Multiyear App and Monte Carlo runs now carry stored energy and PV-origin
  inventory across year boundaries instead of silently resetting to a full,
  unknown-origin battery each January.
- Optimizer candidates now use the App's top-level inverter efficiency and
  the simulated/aligned load when computing objectives.
- The NSGA-II optimizer (`optimize_system_multi_objective`) scored candidate
  designs with a different model than the App reports, in three ways, all
  fixed:
  - candidates were simulated **without AC clipping** (no
    `inverter_ac_capacity_w`), biasing the Pareto front toward high DC/AC
    ratios whose clipping losses were never seen. Candidates now get the
    CAPEX-matched nameplate (`pv_peak / costs.dc_ac_ratio`) — the inverter a
    design pays for is the one that clips it;
  - `calculate_financials` ignored maintenance, PV degradation, and the
    separate export-price inflation. It now mirrors the year-1-estimation
    formulas of `cost_analysis_projection` exactly (equivalence enforced by
    `tests/test_optimization_parity.py`); the fixed daily grid fee cancels
    out of the savings NPV and remains omitted by construction. NPV values
    and Pareto fronts change (lower, more realistic NPVs). Candidate battery
    replacements use a documented year-1-SOH projection; App multiyear
    propagation remains the higher-fidelity basis;
  - the load was positionally re-stamped onto the PV index via
    `align_load_to_pv`, ignoring timezones (a UTC-offset shift of the whole
    profile against PV). The raw load now reaches
  `simulate_energy_balance`, whose internal alignment is timezone- and
  DST-aware — the same code path the App uses. `align_load_to_pv` keeps
  its behaviour for external callers but now carries a docstring warning.

## [0.3.3] - 2026-07-02

### Removed
- The `Suntech_STP550S_NOMT` catalog module. Its datasheet points were NMOT
  ratings (800 W/m², Mpp = 415 W), but the CEC single-diode fit interprets
  `Vmp`/`Imp`/`Voc`/`Isc` as STC values, so the entry produced silently wrong
  model parameters. Configs referencing it now fail with the standard
  "Module '...' not found. Available: ..." error; use `Suntech_STP550S_STC`
  (the same physical module at STC) instead.

### Added
- `breos sweep`, a serial parameter-grid CLI command that expands a `[sweep]`
  section in a normal App config and writes one combined CSV with varied
  parameters, resolved system sizing, BREOS version, and scalar result metrics.
- `configs/examples/sweep.toml` as a runnable sweep example over module count
  and battery size.
- Release-smoke tests for the README quickstart, the Monte Carlo example path,
  and the pymoo-backed multi-objective optimization helper.
- `breos.solar.resolve_pvwatts_losses`, used by dry-run/config inspection to
  report resolved PVWatts loss components and their combined percentage.
- `sell_price_inflation` App config key and `--sell-price-inflation` CLI flag
  (default `0.0`). `CostParams` and `cost_analysis_projection` already
  supported an annual export-price inflation, but no config key existed and
  neither the App runner nor the Monte Carlo runner passed it, so the public
  paths always projected with `0.0`. The value is validated in
  `validate_config`, threaded through both projection call sites, and shown
  in `breos run --dry-run` / `validate-config --json`. The `0.0` default
  reproduces existing results bit-for-bit.

### Changed
- `breos run --dry-run` and `breos validate-config --json` now include the
  fully resolved static PVWatts loss stack instead of only echoing
  `pv_loss_overrides`.
- `BatteryConfig.battery_type` is now explicit about the native degradation
  model being LFP-only: `"LFP"` normalizes to `"lfp"`, while unsupported
  chemistries raise instead of silently reusing LFP cycle-aging parameters.
- `BatteryConfig.eol_percentage` now defaults to `0.70`, aligning with the
  App config default `battery_eol_percentage = 0.70` and the optimizer's
  battery-spec fallback (previously `0.80` and `0.8` respectively — three
  surfaces, two values). App and CLI results are unchanged (they always pass
  the config value explicitly), but direct `BatteryConfig` users who relied
  on the implicit `0.80` will now see batteries replaced later, at 70% SOH;
  pass `eol_percentage=0.8` to keep the old threshold. The same applies to
  optimization battery specs without an explicit `eol_percentage`.

### Documentation
- Updated the README and `CITATION.cff` to cite the SSRN preprint DOI
  (`10.2139/ssrn.7032064`).
- Added the BLAST degradation-engine design note and refreshed roadmap
  priorities around model accuracy, validation, energy-loss accounting, and
  future default-model profiles.

### Fixed
- `dc_to_ac` (and therefore `calculate_pv_production_ac`) clipped ~4% below
  the intended inverter AC nameplate: it passed the nameplate
  (`pv_peak_power_w / inverter_loading_ratio`) as pvlib's `pdc0`, which is a
  DC-input limit whose AC nameplate is `eta_inv_nom * pdc0`. The DC limit is
  now derived as `nameplate / eta_inv_nom`, so clipping happens at the same
  AC rating used by `InverterConfig.size_from_pv`, the App energy balance,
  `economics.calculate_costs`, and the CLI's reported `ac_rating_kw`. This
  raises `dc_to_ac` / `calculate_pv_production_ac` outputs slightly at every
  operating point (most visibly during clipping hours); App simulation
  results are unchanged because the App path converts DC through
  `simulate_energy_balance`, not `dc_to_ac`.
- `PVModuleParams` no longer discards a user-supplied `gamma_pmp`: the
  constructor argument existed but `__post_init__` unconditionally overwrote
  it with `T_Pmax_pct`. It now only defaults to `T_Pmax_pct` when not given,
  matching the `alpha_sc_abs` / `beta_voc_abs` override pattern. Catalog
  modules and configs that never set `gamma_pmp` are unaffected.

## [0.3.2] - 2026-06-26

> **Upgrading:** config validation is now strict — a config with an unknown
> top-level key (e.g. a typo like `batery_kwh`) that silently defaulted in
> 0.3.1 now raises listing the offending key(s). Fix or remove stray keys
> before upgrading. All other changes preserve prior behaviour by default.

### Removed
- The `nrel-pysam` runtime dependency. It was only ever reached transitively,
  through pvlib's `fit_cec_sam`, to fit the CEC single-diode parameters on the
  default PV path. `nrel-pysam` publishes no Python 3.14 wheel or sdist and was
  the sole blocker to running BREOS on 3.14.

### Added
- `breos.cec_fit.fit_cec_params`: a pure-`scipy`/`pvlib` implementation of the
  CEC 6-parameter coefficient calculator (Dobos 2012, DOI:10.1115/1.4005759),
  a drop-in for `pvlib.ivtools.sdm.fit_cec_sam`. Across every bundled module it
  reproduces the SAM fit to within 0.03% on maximum power over a
  temperature x irradiance grid and 0.004% on annual energy, so model results
  are unchanged. Validated against the `nrel-pysam` oracle by
  `tools/validate_cec_fit.py`.
- Python 3.14 support: the `3.14` classifier and CI matrix entry, now that the
  `nrel-pysam` blocker is gone.
- Config validation now rejects unknown top-level keys. A typo such as
  `batery_kwh` previously slipped through `merge_defaults` and silently
  defaulted (e.g. the battery to `0`), producing plausible-but-wrong results;
  it now raises listing the offending key(s). The optional `montecarlo`
  section is recognised so Monte Carlo configs still validate.
- Configurable sky-diffusion (transposition) model via a `transposition_model`
  config key and `--transposition-model` / `--sky-model` CLI flag, threaded
  through `calculate_pv_production_dc`, the tracking and multi-array variants,
  and the `App` config surface. Supports `isotropic` (default), `klucher`,
  `haydavies`, `reindl`, `king`, `perez`, and `perez-driesse` via pvlib's
  `get_total_irradiance`; the extra inputs the anisotropic models need
  (extraterrestrial DNI, relative airmass) are derived internally. The default
  `isotropic` reproduces prior results bit-for-bit. Per-array overrides are
  supported in `pv_arrays`.
- Configurable ground reflectance and Perez coefficients to drive those models
  with real site information: `albedo` (0-1) or a named `surface_type`
  (`"snow"`, `"sea"`, `"grass"`, ...) sets the ground-diffuse reflectance for
  every model (previously fixed at pvlib's 0.25), and `model_perez` selects
  the Perez coefficient set. All three are App config keys with matching
  `--albedo` / `--surface-type` / `--perez-model` CLI flags and per-array
  overrides; not setting them leaves the previous defaults unchanged.

### Changed
- The default PV path fits CEC parameters via `breos.cec_fit.fit_cec_params`
  instead of `pvlib.ivtools.sdm.fit_cec_sam`; `breos/solar.py` and the public
  API are otherwise unchanged.
- The two placeholder `Generic_400W` and `Generic_600W_Bifacial` catalog
  modules now carry realistic mono-PERC datasheet specifications (their
  previous made-up values fit cleanly under SAM only via an internal
  short-circuit-current heuristic); their nameplate power and keys are
  unchanged.
- `resolve_pv_system` no longer mutates the merged config in place to record
  the derived `n_modules`; the resolved count is materialised into a fresh
  dict by `resolve_app_config`, so the dict wrapped by the frozen
  `ResolvedAppConfig` is built once and the caller's input dict is left
  untouched.

## [0.3.1] - 2026-06-25

### Changed
- Pinned `requires-python` to `>=3.11,<3.14`. The transitive `nrel-pysam`
  dependency (reached through pvlib's CEC fit) publishes no Python 3.14 wheel
  or sdist, so installs on 3.14 could not resolve. This is a stopgap; 0.3.2
  removes the `nrel-pysam` dependency and lifts the cap.

## [0.3.0] - 2026-06-24

### Fixed
- **TMY timezone misalignment (results-changing):** `fetch_tmy_weather_data`
  relabeled PVGIS's UTC-ordered rows with local-time labels, shifting
  irradiance against the computed solar position by the location's UTC offset
  (~1 h for Berlin, ~10 h for Melbourne; UTC+0 locations were unaffected).
  Rows are now rolled to start at local midnight while each timestamp keeps
  its correct UTC instant.
- **Battery phantom export (results-changing):** when temperature derating or
  daily SOH decline shrank `Emax` below the stored energy, the negative
  charge room silently drained the battery into `Sell_To_Grid`. Stored energy
  is now clamped into the derated window, mirroring the Numba kernel.
- Load profiles are pinned to the location's wall clock instead of the UTC
  clock, so H0 morning/evening peaks land at the correct local hours across
  DST (previously ~1 h off in Iberia during summer).
- `optimize_tilt`/`optimize_tilt_brent` reported "kWh" without accounting for
  the timestep (4x off at 15-minute resolution; ranking was unaffected).
- `BatteryConfig.initial_resistance_growth` was never read by
  `simulate_energy_balance`; it now seeds the resistance state when the
  continuation argument is not supplied.
- `get_module_info` printed the efficiency fraction as a percent and crashed
  on modules without efficiency metadata.
- Removed three dead, shadowed plotting functions and an undefined
  `MONTH_LABELS` reference that crashed the TMY-vs-historical monthly plot.

### Added
- Inverter AC clipping in the `App` energy pipeline: PV output, export, and
  battery discharge now saturate at the AC rating implied by
  `inverter_loading_ratio` — the same rating used for inverter CAPEX. DC
  surplus above the rating still charges a DC-coupled battery
  (`BatteryConfig.inverter_ac_capacity_w`, `None` = legacy uncapped model).
- Configurable PVWatts system losses: `breos.solar.DEFAULT_PVWATTS_LOSSES`
  (~14.1% combined) with a `loss_overrides` hook on the production functions
  and a `pv_loss_overrides` App config key.
- Battery operating parameters as App config keys: `battery_min_soc`,
  `battery_max_soc`, `battery_eol_percentage`, and `battery_rte` (previously
  hardcoded to 0.10/0.90/0.70/sqrt(0.95)).
- `enable_resistance_fade` now feeds the resistance-derated round-trip
  efficiency back into the energy loop (previously tracking-only).
- Battery degradation calibration variants are explicit for the 0.3.0
  release: `naumann_lam_field_calibrated` remains the default v1 field
  calibration, `naumann_lam_field_calibrated_v1` is an equivalent explicit
  alias, and `naumann_lam_field_calibrated_v2` exposes the v2
  field-calibrated fit with Lam `Ea`/`n` fixed and `k0`/`b` fitted to field
  data.
- Parity tests for the optional Numba kernels: the duplicated LFP derate
  constants against `battery.lfp_capacity_factor`, and the energy-balance
  kernel against the reference path under shared-model conditions.
- CLI discovery and inspection commands: `breos list
  {locations,modules,cost-presets,emissions,load-profiles}` prints packaged
  option keys, `breos validate-config <config>` checks a config file and
  summarizes the resolved choices, and `breos run --dry-run` writes the
  resolved configuration as JSON without running a simulation. `list` and
  `validate-config` accept `--json` for machine-readable output.
- PyPI distribution: 0.3.0 is the first release installable with
  `pip install breos`. Tagged `v*` releases on `main` now publish to PyPI
  through GitHub Actions trusted publishing (OIDC), running the release
  artifact verifier before upload, with a manually triggered TestPyPI
  dry-run path.
- Projection-based LCOE support:
  `breos.economics.calculate_lcoe_from_projection` computes LCOE from the
  simulated multi-year cost projection, and the batch location comparison
  tool now writes `lcoe_eur_kwh` plus LCOE heatmaps.

### Changed
- Renamed remaining pre-release "PVBAT" branding to BREOS in the Polysun
  comparison plots: the `plot_degradation_methodology_comparison` first
  argument is now `breos_soh`, the scenario/location dicts passed to
  `plot_lifetime_prediction_comparison` and
  `plot_temperature_sensitivity_comparison` use the `breos_eol_year` key,
  legend labels read "BREOS (Naumann)", and the SOH comparison figure is
  saved as `polysun_breos_soh_comparison*.png`.
- Cost defaults are single-sourced from the `CostParams` dataclass:
  `cost_params_from_config` and the App preset fallbacks no longer carry
  their own diverging literals (packaged presets are unaffected).
- Config validation rejects out-of-range values at load time: negative
  `battery_kwh`, top-level `tilt`/`azimuth`, `inverter_efficiency`,
  `inverter_loading_ratio`, `projection_years`, `pv_degradation_rate`, and
  the new battery keys.
- PV-only App runs construct an explicit inverter model, so a configured
  `inverter_efficiency` now applies without a battery (previously ignored).
- `App.result()["lcoe_eur_kwh"]` now uses the simulated projection, including
  O&M and battery replacement costs, instead of the simpler CAPEX + fixed
  annual O&M helper.
- Library progress messages (weather file discovery, saved files, CSV
  conversions) go through `logging` under `breos.*` logger names instead of
  unconditional `print()`. Functions with a `verbose` flag still print.
- Slimmed the default runtime dependency set to the BREOS core simulation
  stack and moved heavier workflow packages behind extras: `plots`,
  `optimization`, `weather`, `fast`, `validation`, and `location-tools`.
  NREL-PySAM stays in the core set because the default PV model fits CEC
  single-diode parameters at runtime via pvlib's `fit_cec_sam`. (Removed in
  0.3.2.)
- The `dev` extra now installs optional feature dependencies so contributor
  test runs continue to cover optional paths.

### Documentation
- Install snippets in the README and docs point at PyPI (`pip install breos`)
  instead of git tag installs, and the quickstart gained a "10-minute first
  run" walkthrough with a pip-friendly inline config, the matching
  `configs/examples/quickstart.toml` source-checkout example, the new
  option-discovery commands, and a representative output excerpt with
  plausibility ranges.
- New recipes page with validated copy-paste configs: PV-only home, PV plus
  battery, custom latitude/longitude/timezone, east-west roof with
  `pv_arrays`, 15-minute resolution, external E-REDES/BDEW/REE load
  profiles, and offline runs with cached weather.
- New generated "Packaged options" reference page listing locations, PV
  modules, cost presets, emissions factors, and load profiles. It is built
  by `tools/generate_option_docs.py` from the packaged data and source
  constants, and a test fails CI when the page drifts.
- README documents the fixed PVWatts loss components, the inverter clipping
  convention, the `weather/` working-directory override, the Open-Meteo
  `.cache.sqlite` file, logging configuration, and the new config keys.
- README describes the Numba kernels honestly as approximate standalone
  screening engines that `breos.App` does not use; the module docstring
  carries the same warning.
- Clarified that the `bdew_h0` alias maps to the bundled demandlib
  BDEW-H0-shaped profile `"1"`, distinct from the external BDEW H0 2025
  dataset (profile `"7"`).
- Replaced stream-of-consciousness working notes in `economics`,
  `optimization`, and `plotting` with factual comments, and
  fixed mislabeled docstrings (`total_pv` is post-inverter AC; the Suntech
  NOMT catalog entry documents its NMOT-condition rating).

## [0.2.3] - 2026-06-08

### Changed
- Lowered the minimum supported Python from 3.13 to 3.11 — the real floor, set by
  pandas, timezonefinder, and stdlib `tomllib`. CI now runs a 3.11/3.12/3.13 matrix.
- Relaxed the pvlib constraint from `==0.14.0` to `>=0.14.0,<0.16` after verifying
  the full API surface and the test suite against pvlib 0.15.1.
- `breos.__version__` is now resolved from installed package metadata
  (`importlib.metadata`) instead of a hardcoded literal, so it can no longer drift
  from `pyproject.toml`.

### Added
- `CITATION.cff`, `CODE_OF_CONDUCT.md`, and `SECURITY.md` for open-source release
  readiness.

### Removed
- Duplicate top-level `rlp/*.csv` load-profile files (byte-identical to the
  packaged `breos/data/rlp/` copies that runtime actually uses). `rlp/README.md` is
  retained as external-RLP guidance.

### Documentation
- README badge and installation docs now state Python 3.11+.
- Trimmed `ATTRIBUTIONS.md` to reference only the packaged load-profile paths.

## [0.2.2] - 2026-06-07

### Documentation
- Expanded third-party notices with dependency credits, runtime data-source
  caveats, and scientific/model attribution guidance.

## [0.2.1] - 2026-06-03

### Documentation
- Updated installation guidance to use the stable GitHub tag until PyPI
  publishing is available.
- Added PyPI trusted publishing to the roadmap.
- Documented the full CI/release validation gates in the contributor guide.
- Standardized API documentation wording around domain areas.

## [0.2.0] - 2026-06-03

### Changed
- Narrowed the top-level `breos.__all__` release surface to the stable facade,
  key configuration/result objects, and core composition helpers. Lower-level
  module APIs remain importable from their modules.

## [0.1.0] - 2026-04-30

### Added
- Public API facade (`breos.App`) — single entry point for simulations: config dict in, plain dict out.
- Command line entry point (`breos run`) for running simulations from shell flags or TOML/JSON config files.
- Test suite — pytest coverage of the public API, battery, economics, emissions, and solar modules (all offline).
- GitHub Actions CI on every push/PR.
- `cost_params_from_config()` — config parser for `CostParams`.
- Marginal grid carbon intensity support in `EmissionsParams` for more accurate CO₂ avoidance accounting.

### Changed
- Renamed PV `slope` → `tilt` everywhere: function parameters, dataclass fields, docstrings, CLI/config keys, public API. Includes `optimize_slope()` → `optimize_tilt()` and the `tools/azitilt_optimizer.py` script.
- Calendar model name canonicalized to `naumann_lam_field_calibrated` (the legacy alias `naumann_lam_calibrated` has been removed).
- Constants renamed: `LAM_NAUMANN_FIELD_CALIBRATED_*` → `NAUMANN_LAM_FIELD_CALIBRATED_*`; alias indirections (`LAM_CAL_K0_FRAC` …) dropped.
- Configs modernized: per-unit cost keys (`maintenance_cost_per_panel`, `other_cost_per_module`); emissions schema renamed and country list expanded.
- Polysun degradation now tracks the actual `last_replacement_year` instead of approximating with `n_replacements × int(total_life)` — handles fractional lifetimes and cycle-driven replacements correctly.
- Numba degradation kernel now treats SOC reversals as half-cycles (rainflow-aligned) and applies LFP temperature derating per timestep, matching the Python reference path.

### Fixed
- NPV discount factor now uses `(1 + r) ** Year` (time-0 NPV) instead of `(1 + r) ** (Year - 1)`. Affects all `cost_analysis_projection` outputs.

### Removed
- Out-of-scope kernels (`combined_energy_balance_kernel`, `batch_combined_energy_balance_kernel`) and non-core energy-system code paths. BREOS focuses on PV + battery simulation.
