# BREOS Roadmap

Planned work that is not yet scheduled. These are intentions, not commitments.
See GitHub issues for active work, and `design/architecture/` for the detailed
plans behind individual items.

## Next releases

- **0.7.0** — currency concept; time-of-use tariff valuation with static,
  provenance-bound schedules; opt-in fixed-target smart charging; descriptive
  load-profile keys replacing the numeric ones. Flat pricing and greedy
  self-consumption stay the compatible defaults. See
  [design/architecture/0.7x-tariffs-and-smart-charging-plan.md](design/architecture/0.7x-tariffs-and-smart-charging-plan.md).
- **0.7.x** — economic scenario and sensitivity analysis (scenarios,
  switching values, and probabilistic inputs), then broader price-aware
  dispatch strategies.
- **1.0** — flip to the recommended model defaults with a documented upgrade
  note.

## Model accuracy and validation

- **Weekday-aware load-profile alignment and an E-REDES source-file converter.
  Prerequisite for 0.7.0 time-of-use tariffs.** `load_profile` restamps a source
  CSV onto the simulation year positionally (`df.index = new_index`), so row 0
  becomes 1 January regardless of the weekday in the source file or target
  year. Each profile inherits the weekday phase of the year that generated its
  file. The bundled demandlib H0 files are 2023 with a Sunday start,
  `EREDES_2025_BTN_1000kwh_15min.csv` is 2025 with a Wednesday start, and
  `EREDES_2025_BTN_1000kwh_hourly.csv` is 2023 despite its name. The two
  E-REDES files therefore disagree by four days.

  Under flat pricing and TMY weather this has little effect because weather has
  no weekday structure. On the H0 profile, it moves PV-only grid independence
  by 0.19 pp, with no systematic sign. Under a TOU schedule, the error is
  between the load's weekday phase and the tariff calendar, which TMY does
  nothing to protect. On an illustrative Portuguese tri-horario weekly
  cycle the worst case across all seven phases is 0.50% of the annual bill for
  E-REDES BTN C and 1.86% for demandlib H0. The error has a consistent sign for
  a given file but remains small because the phase error changes which load
  shape meets each tariff period without changing the number of days in each
  period. Price-aware dispatch is untested and could produce a larger
  difference because a wrong day type gives the controller wrong charge and
  discharge periods on two days in seven.

  The raw E-REDES publication (`Perfil_Consumo_Injecao_E-REDES_<year>.csv`)
  makes this straightforward, and a converter for it is the first step:

  - It carries the true weekday in column 2, verified against the real calendar
    for all 35040 rows of the 2026 vintage. Parsing the published dates instead
    of restamping positionally removes the whole bug class for E-REDES, with no
    whole-week rolling heuristic. Profiles without a published calendar still
    need that heuristic, which trades up to three days of day-of-year phase for
    correct weekday phase. On H0, the residual after realignment is 1.27% of
    mean load, compared with 10.9% before.
  - Timestamps are interval-end values. The first row is `00:15`, and the last
    is `24:00` on 31 December. The existing repository file was restamped onto
    interval-start labels, which is correct for BREOS but currently implicit
    and easy to invert. Make the conversion explicit and test it with the
    existing solar-timing conventions.
  - It is `latin-1`, has four header rows, and ends with a trailing all-NaN row
    that needs `dropna(how="all")`. The current repository files also contain
    that row, with 35041 rows for 35040 intervals. `load_profile` truncates the
    extra row, but calling `.sum()` on those files returns NaN.
  - Values are already normalised to exactly 1000 kWh/yr, in kWh per interval.
    The repository format multiplies these values by 1000 and labels the column
    `- Wh`. `_load_profile_csv` then renames the column to `Electrical
    Consumption [W]`, although Wh per 15 minutes is not W. The converter should
    correct the label. This label does not affect current calculations because
    `scale_to_annual_consumption` renormalises the values afterwards.
  - The source file also contains `RESP (MW)`, `IP`, `MP`, and six UPAC
    self-consumption and injection profiles that BREOS does not currently
    expose. These profiles may support later self-consumption work.

  Also regenerate or re-stamp the E-REDES hourly file so both resolutions share
  one phase, and check the interaction with the existing leap-day insertion in
  `load_profile`, which assumes positional alignment.

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

- **Absolute inverter AC rating as an App input (0.7.x).** `InverterConfig`
  already carries `nominal_power_w`, and `calculate_dc_ac_power` already clips
  against it with full DC-side bookkeeping, but the App path never sets it:
  `runners/app.py` derives the ceiling as `pv_peak_w / inverter_loading_ratio`,
  so the AC rating is a consequence of module choice rather than a stated
  quantity. On a large roof that is a reasonable sizing default. On a small
  system it is wrong in a way the caller cannot correct: 2 x 445 Wp at the
  default 1.25 ratio lands at 712 W by coincidence, and swapping to 550 W
  modules moves the ceiling to 880 W with no input having said so.

  The driving case is plug-in balcony PV, where the AC rating is not a sizing
  choice but a legal ceiling — 800 W in Portugal (Decreto-Lei 130/2026), Great
  Britain (SI 2026/848), Germany and Austria — and DC oversizing behind it is
  the normal configuration, so the clipped energy is a headline result rather
  than a rounding error. A ratio cannot express "exactly 800 W whatever the
  modules are".

  Ask: an optional absolute nominal AC power on App config, taking precedence
  over `inverter_loading_ratio` when supplied, honoured by the existing
  clipping path, and reported in the results the way `inverter.ac_capacity_kw`
  already is. Pairs naturally with the export cap described under "Under
  consideration": the same installations that need a hard AC ceiling are the
  ones legally required not to inject.

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
  `Sell_To_Grid` plus `PV_DC_Curtailed`. Both are already in the ledger, so
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

## Reference load profiles pending license verification

These sample profiles were removed from `rlp/` and `breos.load_profiles` before
the open-source release because their redistribution terms were not confirmed.
They can return once written permission or a clear license is obtained.

- **SynPRO Family profile** (Fraunhofer ISE) — was profile key `"2"` /
  `family_profile_SynPro.csv`. Contact: synpro@ise.fraunhofer.de.
- **LoadProfileGenerator family-with-3-kids profile** (Noah Pflugradt, FZJ
  IEK-3) — was profile key `"3"` / `LoadProfileGenerator_family_3kids.csv`. Tool
  is MIT-licensed; output redistribution policy needs author confirmation.
