# Changelog

All notable changes to BREOS are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Changed
- Slimmed the default runtime dependency set to the BREOS core simulation
  stack and moved heavier workflow packages behind extras: `plots`,
  `optimization`, `weather`, `fast`, `cec-fit`, `validation`, and
  `location-tools`.
- The `dev` extra now installs optional feature dependencies so contributor
  test runs continue to cover optional paths.

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
- Out-of-scope kernels (`combined_energy_balance_kernel`, `batch_combined_energy_balance_kernel`) and any thermal storage / heat-pump / V2H / time-of-use code paths. BREOS focuses on PV + battery simulation.
