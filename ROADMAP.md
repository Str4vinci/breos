# BREOS Roadmap

Planned work that is not yet scheduled. These are intentions, not commitments.
See GitHub issues for active work, and `design/architecture/` for the detailed
plans behind individual items.

## Next releases

- **0.7.0** — currency concept; time-of-use tariff valuation with static,
  provenance-bound schedules; opt-in fixed-target smart charging; descriptive
  load-profile keys replacing the numeric ones. Flat pricing and greedy
  self-consumption stay the compatible defaults. See
  [design/architecture/tariffs-and-smart-charging.md](design/architecture/tariffs-and-smart-charging.md).
- **0.7.x** — economic scenario and sensitivity analysis (scenarios,
  switching values, and probabilistic inputs), then broader price-aware
  dispatch strategies.
- **1.0** — flip to the recommended model defaults with a documented upgrade
  note.

## Model accuracy and validation

- Bring the measured-data checks on the
  [external validation](docs/modeling/validation.md) page into the standing
  drift suite, so per-dataset deltas are tracked in CI alongside PVGIS and
  PVWatts.
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

- Worker controls (`--workers`) and conservative auto-defaults for CPU and
  memory, with care for fanless Apple Silicon machines.
- A startup diagnostic and a benchmark/smoke mode for long runs.

## Onboarding and tooling

- Keep install snippets, config tables, and version-specific text aligned with
  the current PyPI release.
- Multi-config parameter sweeps and parallel batch runs.
- An offline `breos demo` command using clearly labeled synthetic inputs.

## Assumptions to bound

These stay as they are, but each needs a stated boundary before results lean
on it.

- Replacement thresholds are study decisions. Keep 70% as a documented
  default, but make replacement-disabled sensitivity reachable from the
  stable facade; the lower-level projected path already supports it.
- Battery temperature needs an unambiguous meaning. An explicit 40 C becomes
  27.4 C under the default indoor transformation, so ambient, enclosure and
  measured cell temperatures should not share one indistinguishable input
  contract.
- BLAST selection changes aging while dispatch still applies the LFP
  temperature-capacity function. Exposing an NMC or NCA aging model must not
  imply that the whole pack model became chemistry-specific.
- Bifacial and tracking limits stand: the bifacial path keeps an unshaded
  front and adds rear irradiance, and dual-axis tracking has no row
  self-shading. Validate the existing behaviour before adding geometry.
- Monte Carlo represents only the uncertainty actually sampled. Historical
  weather resampling and annual demand multipliers say nothing about
  uncertainty in tariffs, degradation models, load timing, or future climate.

## Under consideration

Suggestions borrowed from the islanded-HRES literature, recorded so they are
not lost. None of these is decided. They belong in the sections above only
after a deliberate call to take them on, and the notes below are the arguments
for and against, not designs.

- **Excess-energy index.** `EEI = excess / PV production`, with excess as
  `PV_AC_Export` plus `PV_DC_Curtailed`. Both are already in the ledger, so
  the diagnostic is nearly free. As an optimization objective it is close to
  degenerate under flat feed-in pricing, where NPV already values exported
  energy, so it earns its place only alongside an export cap. Open question:
  does BREOS want to model export-capped markets at all?

- **Pareto-front quality diagnostics.** Hypervolume, spacing, and maximum
  spread over repeated seeds, to say whether an NSGA-II front converged and
  how much it varies run to run. Hypervolume needs only a documented reference
  point. IGD needs a reference front, which for this problem can only be the
  pooled non-dominated set across the runs being compared, making it a
  within-experiment number rather than a score. Open question: is the audience
  a reviewer, who wants the full statistics, or a user, who wants a
  converged-or-not flag for much less work?

- **Knee-point selector.** One recommended design off the front, by minimum
  normalized distance to the utopia point. The objectives are a percentage, a
  euro amount, and a ratio, so the normalization decides the answer and would
  have to be a stated convention. Open question: choosing a knee makes BREOS
  opinionated about trading grid independence against NPV, which is a product
  decision before it is a feature.

- **Autonomy days as a battery parameterization.** `E_b_max = P_load_avg *
  A_d / (DoD * eta)`, resolved to `battery_kwh` at config load. Not a better
  decision variable, since the search space is unchanged, but closer to how
  practitioners state a requirement or a bound.

- **Profile-complementarity metric.** Correlation-style diagnostics over two
  or more normalized load profiles, to say whether building-to-building
  transfer could pay before any transfer is modeled. Needs no simulation and
  no dispatch change, and gates whether multi-building work is worth starting.
  The multi-building dispatch model itself is out of scope for core.

- **Export-limited and islanded operation.** Every dispatch path assumes an
  unconstrained bidirectional connection: surplus sold, deficit imported
  without limit. An export cap is the small half, routing blocked surplus to
  the existing curtailment column with economics unchanged. Unserved load is
  the large half: LPSP requires dispatch to be allowed to fail, a ledger
  column for the shortfall, and an economics path that does not assume every
  deficit is a billable import. Open question: "simulator for PV and storage
  for buildings" currently means grid-connected buildings throughout the code,
  and declining this is a legitimate answer.

## Reference load profiles

The profile set is settled for 0.7.0 (#182): demandlib H0 (bundled), E-REDES
BTN A/B/C, the BDEW H0 publication and REE 2.0TD (external files only), plus
`custom` for any other CSV.

- **SynPRO Family profile** (Fraunhofer ISE), once profile key `"2"`, is
  dropped for good.
- **LoadProfileGenerator** output (Noah Pflugradt, FZJ IEK-3), once profile key
  `"3"`, is user input through `load_profile = "custom"`, as are CREST exports
  and measured data. A named LoadProfileGenerator adapter can follow later.
