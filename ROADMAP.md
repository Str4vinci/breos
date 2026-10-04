# BREOS Roadmap

What the current release line delivers and what is planned next. Future items
are intentions, not commitments. See GitHub issues for active work, and
`design/` for the decision records and design notes behind individual items.

## 0.7.0

0.7.0 adds tariff-aware economics and opt-in battery control. Flat pricing and
greedy self-consumption stay the defaults.

- Time-of-use tariff valuation through a `[tariff]` table, with packaged
  Portuguese and Spanish schedules and custom schedules that can change prices
  and windows by calendar month.
- A separate `[reference_tariff]` for the household without the system.
- Opt-in `[smart_charging]`: fixed-target grid charging, discharge-only
  windows, and an experimental daily persistence planner.
- Separate escalators for import prices, export prices and O&M, a replacement
  cost learning rate, and one default discount and inflation rate everywhere.
- An optional estimated battery residual value reported beside the unadjusted
  NPV, and `App.revalue` for price scenarios on a finished run.
- Currency-neutral result names, with the result currency in provenance. EUR
  is the only supported currency so far.
- Descriptive load-profile keys: demandlib H0 (bundled), E-REDES BTN A/B/C,
  the BDEW H0 publication and REE 2.0TD (external files only), plus `custom`
  for any other CSV. The numeric keys are gone. The SynPRO family profile is
  dropped; LoadProfileGenerator, CREST and measured data are read through
  `custom`.
- One explicit `irradiance_resampling` policy for hourly-to-15-minute weather
  in App, Monte Carlo and optimization, without the old cap on diffuse
  irradiance.
- Sub-year simulation windows (`[period]`) and an absolute inverter AC rating.

See [CHANGELOG.md](CHANGELOG.md) for the complete list, and
[design/architecture/tariffs-and-smart-charging.md](design/architecture/tariffs-and-smart-charging.md)
for the tariff and smart-charging design.

## Next releases

- **0.7.1** — study tooling: scripted, reproducible study workflows on the
  public API, such as paired Monte Carlo comparisons across designs and
  tariffs, and reusable revaluation scenarios. A general capability that these
  workflows need goes into the library itself.
- **0.8** — 15-minute irradiance from hourly data by quadrature
  reconstruction, with GHI derived from DNI and DHI. The `solar_position`
  default changes from `"interval-start"` to `"weather"`, which reads the
  representative time of each step from the weather metadata. Both change
  results and will ship with an upgrade note. BLAST degradation under Monte
  Carlo is a candidate.
- **1.0** — flip to the recommended model defaults with a documented upgrade
  note.

## Model accuracy and validation

- Bring the measured-data checks on the
  [external validation](docs/modeling/validation.md) page into the standing
  drift suite, so per-dataset deltas are tracked in CI alongside PVGIS and
  PVWatts.
- A documented "recommended" model profile (haydavies/perez transposition,
  diffuse IAM, mount-appropriate thermal coefficients) to become the default
  at 1.0.
- More pvlib physics behind the self-contained PV stage: extended
  cell-temperature and IAM options, and optional DC-side loss models (ohmic,
  soiling, snow).
- String-aware inverter validation and modeling. See
  [design/architecture/string-inverter-sizing.md](design/architecture/string-inverter-sizing.md).

## Economics

- More currencies, and non-EU cost presets.
- Economic sensitivity analysis on top of `App.revalue`: switching values and
  economic uncertainty in Monte Carlo.
- Broader price-aware dispatch strategies.

## Battery

- Per-chemistry aging for NMC and NCA in the native model, alongside LFP.
- BLAST under Monte Carlo (candidate for 0.8).
- AC-coupled batteries: a battery with its own inverter on the AC bus,
  beside the DC-coupled hybrid model. See
  [design/adr/0004-ac-coupled-batteries.md](design/adr/0004-ac-coupled-batteries.md).

## Performance and portability

- Conservative automatic worker defaults for CPU and memory, in place of the
  explicit `n_procs` (default 1) of Monte Carlo and optimization, with care for
  fanless laptops.
- A startup diagnostic and a benchmark/smoke mode for long runs.

## Onboarding and tooling

- Multi-config parameter sweeps and parallel batch runs.
- An offline `breos demo` command using clearly labeled synthetic inputs.
- A named LoadProfileGenerator adapter, beside the `custom` CSV path.

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
