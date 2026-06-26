# Changelog

All notable changes to BREOS are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

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
  single-diode parameters at runtime via pvlib's `fit_cec_sam`. (Removed after
  0.3.0 — see the Unreleased section above.)
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
