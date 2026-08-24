# BREOS Roadmap

Planned work that is not yet scheduled. These are intentions, not commitments.
See GitHub issues for active work, and `design/architecture/` for the detailed
plans behind individual items.

## Next releases

- **0.6.0** — projected-lifetime PV-battery optimization, detailed fixed-design
  source tables, reproducible historical-weather Monte Carlo, and the planned
  0.5.x deprecation removals. Annual optimization remains the compatible
  default. See
  [design/architecture/0.6x-projected-optimization-plan.md](design/architecture/0.6x-projected-optimization-plan.md).
- **0.7.0** — currency concept; time-of-use tariff valuation with static,
  provenance-bound schedules; opt-in fixed-target smart charging. Flat pricing
  and greedy self-consumption stay the compatible defaults. See
  [design/architecture/0.7x-phd-porting-plan.md](design/architecture/0.7x-phd-porting-plan.md).
- **0.7.x** — economic scenario and sensitivity analysis (scenarios,
  switching values, and probabilistic inputs), then broader price-aware
  dispatch strategies.
- **1.0** — flip to the recommended model defaults with a documented upgrade
  note.

## Model accuracy and validation

- Standing validation suite comparing annual and monthly yields against
  SAM/PVWatts and measured public datasets, with per-location deltas tracked in
  CI.
- A documented "recommended" model profile (haydavies/perez transposition,
  mid-interval sun position, diffuse IAM, mount-appropriate thermal
  coefficients) to become the default at 1.0.
- More pvlib physics behind the self-contained PV stage: extended
  cell-temperature and IAM options, and optional DC-side loss models (ohmic,
  soiling, snow).
- String-aware inverter validation and modeling. See
  [design/architecture/string-inverter-sizing.md](design/architecture/string-inverter-sizing.md).

## Economics

- Currency concept plus non-EU cost and grid-emission presets.
- Time-of-use tariff structures as pluggable price time series.
- Economic scenario and sensitivity analysis: escalator decomposition, a
  scenario runner, switching values, and economic uncertainty in Monte Carlo.

## Battery

- Per-chemistry aging for NMC and NCA alongside the native LFP model.
- BLAST under Monte Carlo (candidate 0.8.0).

## Performance and portability

- Extend the optional Numba dispatch backend introduced for Monte Carlo in
  0.6.0 to `App` and projected optimization, behind the same reference-path
  parity requirements.
- Worker controls (`--workers`) and conservative auto-defaults for CPU and
  memory, with care for fanless Apple Silicon machines.
- A startup diagnostic and a benchmark/smoke mode for long runs.

## Onboarding and tooling

- Keep install snippets, config tables, and version-specific text aligned with
  the current PyPI release.
- Multi-config parameter sweeps and parallel batch runs.
- An offline `breos demo` command using clearly labeled synthetic inputs.

## Architecture

- Wrap third-party modules (pvlib, scipy, rainflow) behind a `breos.adapters`
  layer so upstream API churn touches one file. Deprioritized; see
  [#11](https://github.com/Str4vinci/breos/issues/11) and
  [design/architecture/third-party-wrapping.md](design/architecture/third-party-wrapping.md).

## Reference load profiles pending license verification

These sample profiles were removed from `rlp/` and `breos.load_profiles` before
the open-source release because their redistribution terms were not confirmed.
They can return once written permission or a clear license is obtained.

- **SynPRO Family profile** (Fraunhofer ISE) — was profile key `"2"` /
  `family_profile_SynPro.csv`. Contact: synpro@ise.fraunhofer.de.
- **LoadProfileGenerator family-with-3-kids profile** (Noah Pflugradt, FZJ
  IEK-3) — was profile key `"3"` / `LoadProfileGenerator_family_3kids.csv`. Tool
  is MIT-licensed; output redistribution policy needs author confirmation.
