# Changelog

All notable changes to BREOS are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.2.0] - 2026-04-04

### Added
- **Public API facade** (`breos.App`) — single entry point for simulations with config dict in, plain dict out
- **Test suite** — 62 tests covering the public API, battery, economics, emissions, and solar modules (all offline, ~8s)
- **CI/CD** — GitHub Actions runs tests on every push/PR to `main` and `develop`
- **CONTRIBUTING.md** — development setup, branching workflow, and PR guidelines
- `battery_type` field on `BatteryConfig` (was referenced internally but missing)
- `nrel-pysam` as an explicit dependency (required by pvlib's `fit_cec_sam`)

### Changed
- README rewritten with `breos.App` as the primary Quick Start example
- Configuration and Result reference tables added to README

## [0.1.0] - 2026-04-04

### Added
- Initial open-source release
- Weather data fetching (PVGIS, Open-Meteo) with hourly and 15-minute resolution
- PV production calculations (DC/AC) with built-in module database
- Battery simulation with calendar and cycle aging models (Naumann 2020, Lam 2025)
- Economic analysis: NPV, LCOE, breakeven, cost projections
- Multi-objective optimization (NSGA-II) for system sizing
- CO2 emissions savings calculations
- Standard load profiles (BDEW H0, EREDES, REE)
- Publication-ready plotting utilities
