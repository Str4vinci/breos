# BREOS Roadmap

Planned work that is not yet scheduled. These are intentions, not commitments.
See GitHub issues for active work, and `design/architecture/` for the detailed
plans behind individual items.

## Next releases

- **0.6.0** — projected-lifetime PV-battery optimization, detailed fixed-design
  source tables, reproducible historical-weather Monte Carlo, and the planned
  0.5.x deprecation removals. Projected optimization is the default; annual
  steady-state optimization remains available for lower-cost screening. See
  [design/architecture/0.6x-projected-optimization-plan.md](design/architecture/0.6x-projected-optimization-plan.md).
- **0.7.0** — currency concept; time-of-use tariff valuation with static,
  provenance-bound schedules; opt-in fixed-target smart charging. Flat pricing
  and greedy self-consumption stay the compatible defaults. See
  [design/architecture/0.7x-tariffs-and-smart-charging-plan.md](design/architecture/0.7x-tariffs-and-smart-charging-plan.md).
- **0.7.x** — economic scenario and sensitivity analysis (scenarios,
  switching values, and probabilistic inputs), then broader price-aware
  dispatch strategies.
- **1.0** — flip to the recommended model defaults with a documented upgrade
  note.

## Astra's comments (0.6.2 audit, 2026-09-07)

An external review of 991b651 covering configuration, weather and load
handling, PV production, dispatch, degradation, economics, optimization,
Monte Carlo, results, and release tooling. The verdict was that 0.6.2 has a
sound core with reproducible correctness problems worth clearing before
dispatch features expand; no rewrite. Recorded here so the findings are not
lost, with the items I re-derived against the code marked as confirmed.
Nothing below is scheduled yet.

- **Weather resamplers assume nanosecond timestamps. Confirmed, and the
  highest priority of the set.** Both `resample_to_15min`
  (`breos/weather.py:761`) and the TMY resampler (`breos/weather.py:649`)
  build their interpolation abscissa with `index.astype("int64") // 10**9`,
  which is only correct for a nanosecond-resolution `DatetimeIndex`. This is
  not hypothetical: on pandas 3.x, `to_datetime`, `read_csv(parse_dates=)`
  and `date_range` all default to `datetime64[us]`, so the default path is
  already wrong wherever pandas 3 is installed, and `pyproject.toml` pins
  `pandas>=2.3.3`, where 2.x still gives nanoseconds. The same input file
  therefore produces different irradiance depending on the installed pandas.
  Measured on `dev/article1-inputs/weather/porto_historical_2005_2024_openmeteo.csv`
  (native index unit `us`), microsecond against nanosecond: annual DNI sum
  -0.157%, GHI -0.147%, DHI -0.076%, with instantaneous differences up to
  136 W/m2 and 1.1 C. Second-resolution input raises `ValueError: 'x' must be
  strictly increasing` rather than interpolating. `breos/battery.py:2400`
  already carries the right convention (`_TICKS_PER_SECOND` keyed on
  `index.unit`), so the fix is to reuse an in-repo pattern: derive elapsed
  seconds from timedeltas, consolidate the two implementations, and test the
  same data at `ns`, `us` and `s`.

- **Native cycle counting does not cross day boundaries. Confirmed, but the
  impact on published numbers is likely nil.** The native adapter
  (`breos/degradation/protocol.py:137`) runs rainflow over each day's
  post-dispatch SOC samples alone, without the starting SOC and without
  carrying an unfinished cycle into the next day; the trailing partial day
  never closes a window (`breos/battery.py:1979`). A four-day probe whose
  discharge straddles midnight counted 0.3831 FEC daily against 0.4000
  continuous, -4.2%. A full synthetic year of ordinary diurnal residential
  dispatch counted 292.325 both ways, exactly, because that dispatch resets
  SOC to its daily minimum at the boundary, which is where rainflow would
  close the cycle anyway. So fix it for correctness, but do not expect it to
  move SOH or replacement timing here, and measure before touching
  calibration. The change is small: `DegradationDay` already carries
  `start_soc` and `start_temperature_c`, and `build_endpoint_day`
  (`breos/degradation/engine.py:92`) already prepends the anchor for the
  BLAST adapter. Only the native adapter drops them.

- **The simulation boundary repairs invalid input instead of rejecting it.
  Confirmed, and it breaks the ledger, not only the totals.**
  `_align_input_arrays` (`breos/battery.py:400`) substitutes zero for missing
  PV and load and does not reject negative load. Probes over 24 hours: a
  complete 1 kW load gives 24 kWh consumption; supplying only the first 12
  hours silently gives 12 kWh; a -100 W load with zero PV gives
  `Houseload -2.4 kWh`, `PV_AC_To_Load -2.4 kWh` and `PV_AC_Export
  +2.4 kWh`, so PV delivers negative energy with no PV present. A simulator
  taking user data has to separate a missing measurement from a real zero,
  and a negative net-meter reading cannot stand in for gross demand.
  Validate coverage, regularity, finiteness and sign at the boundary, and
  make any gap filling an explicit preprocessing step.

- **A non-January `start_date` silently changes annual consumption.**
  `prepare_simulation_inputs` (`breos/app_inputs.py:243`) takes
  `start_year = int(cfg["start_date"][:4])` and prepares a calendar year of
  weather, while the demand profile starts on the full supplied date; the
  zero-fill above then covers the uncovered remainder. Reported by the audit
  as 4,000.00 kWh for `2025-01-01` against 2,095.13 kWh for `2025-07-01` on
  the same synthetic weather and the same requested 4,000 kWh/year. I
  confirmed the mechanism in the code but did not re-run the App. The cheap
  correction is to reject non-January starts for the repeated calendar-year
  workflow; genuine arbitrary project start dates are a separate feature.

- **Temperature input failures silently select a different model input.
  Confirmed.** In `build_battery_temperature_series`
  (`breos/weather.py:1115`) a missing file, an unreadable file, or missing
  expected columns falls back to 25 C, which the default indoor
  transformation then turns into 22.9 C. The same blanket 25 C appears when
  representative weather and the simulation window sit in different calendar
  years: `reindex` produces all-NaN, `ffill` is a no-op, and `fillna` fills
  the whole series, wiping out a real 5 C series. Related restamping lives at
  `breos/optimization.py:966`. This feeds degradation and replacement timing
  directly. Explicit file inputs should fail when unusable, representative
  years need deliberate calendar alignment, and if a weather sequence is
  meant to keep representative temperatures then provenance should say so.

- **App and optimization resolve different battery defaults. Confirmed.**
  `app_config.py:372` gives 0.10/0.90 for min and max SOC;
  `_build_battery_config_from_spec` (`breos/optimization.py:470`) defaults to
  0.2/0.8 with 0.9795 each way, a 95.942% round trip against the App's 95%.
  A new 5 kWh pack therefore has a 4 kWh usable window in one entry point and
  3 kWh in the other. The publication configuration supplies these values
  explicitly, so this does not implicate the accepted candidates. Delete the
  optimizer's duplicated numerics and resolve battery settings through one
  shared path.

- **Optimization can accept a design that violates its own budget or
  bounds.** The budget constraint (`breos/optimization.py:1489`) uses CAPEX
  from the steady-state financial calculation while projected evaluation
  computes CAPEX separately, and the two treat `costs.panel_wp` differently:
  with a 400 W cost override, a 550 W module and a 480 EUR budget, the
  constraint accepted 465.48 EUR while the projected result reported 490.03
  EUR. Separately, discrete repair (`breos/optimization.py:1158`) rounds tilt
  to the nearest 5 degrees with no clip to the configured bounds, so a valid
  62.9 candidate becomes 65 under a 63 maximum — confirmed by reading;
  `X[:, 2] = np.round(X[:, 2] / 5.0) * 5.0` has no clip. Use the evaluated
  design's own CAPEX for feasibility, and build the grid so repair stays
  inside the bounds.

- **"Symmetric 1 C" applies at two different electrical boundaries.
  Confirmed, with a wider spread than the audit reported.** `BatteryConfig`
  (`breos/battery.py:226`) derives both limits from
  `power_limit_c_rate * nominal_energy_wh`, but charge is limited at DC into
  the battery path and discharge at AC delivered to the load, so equal
  numbers are not equal cell-side limits. A 5 kWh pack at 1 C on 15-minute
  steps, measured: discharge delivers 5,000 W AC while drawing 5,553 W from
  the cells, 1.111 C; charge takes 5,000 W DC into the path while storing
  4,873 W in the cells, 0.975 C. A 14% spread between directions under one
  setting. This matters to 0.6.2 specifically because
  `validation/article1/article1-projected-optimization.toml:42` motivates the
  setting from battery DC voltage and current ratings. Define where the limit
  applies and derive the rest through conversion efficiencies; keep the
  historical configuration and measure the sensitivity before revising any
  scientific conclusion.

- **Terrain shading ignores weather-defined solar timing.** The PV
  calculation honours `solar_position="weather"`, but the horizon helper
  (`breos/pv/horizon.py:79`) only shifts timestamps for `"mid-interval"`.
  With metadata declaring interval means, the audit's probe kept 600 W/m2 DNI
  under `"mid-interval"` and removed it entirely under `"weather"`. Not
  re-verified here. Resolve solar-position timing once and use the result for
  both shading and transposition.

- **Monte Carlo payback summaries hide unsuccessful trajectories.
  Confirmed.** `_summarize` (`breos/montecarlo.py:642`) drops NaN before
  taking quantiles, so nine trajectories that never pay back plus one that
  pays back in year five report `mean`, `p5`, `p50`, `p95`, `min` and `max`
  all equal to 5.0, with no success count. Those statistics are conditional
  on achieving payback and the summary does not say so; the plotting path
  already distinguishes achieved from not achieved. Report the probability of
  payback within the horizon and label conditional statistics as conditional.
  Also validate sampling bounds — `max_load_scale=-1` currently generates
  negative demand.

Smaller items from the same review, best folded into the work above:

- A valid 100% PV-loss setting produces an infinite LCOE, so
  `json.dumps(result, allow_nan=False)` fails on App output. Represent
  undefined metrics explicitly. `breos/app_results.py:155`.
- Annual financial rows are matched positionally, so passing years `[2, 1]`
  assigns their values to output years 1 and 2. Align by validated year
  labels. `breos/economics.py:265`.
- Weather selection depends on directory enumeration: `candidates[0]` wins,
  and an uncovered historical request can fall back to a non-covering file.
  Prefer an explicit file, or reject ambiguity. `breos/weather.py:287`.
- Inverter presets are shared mutable objects, so modifying a returned preset
  changes later callers' defaults. Return a copy. `breos/inverter.py:224`.
- The load-profile weekday problem is the one already recorded under "Model
  accuracy and validation" above; the review independently reached the same
  conclusion and the same prerequisite ordering against 0.7.0 tariffs.

Deletions the review proposed, consistent with the project's delete-rather-
than-deprecate convention and with public compatibility preserved through an
explicit migration where one is needed:

- Automatic zero and default substitution for explicitly supplied invalid
  inputs. It hides errors and adds behaviour to maintain.
- `align_load_to_pv`'s positional alignment path. Its own docstring warns
  against using it before the simulator; keep one authoritative path.
- The alternative `use_rainflow=False` aging path. The same SOC excursion
  counts 0.6 FEC with rainflow and 1.2 without; production already uses
  rainflow.
- The unconditional extra steady-state evaluation inside projected
  optimization. Every candidate pays for an additional year simulation and a
  separate financial calculation before its lifetime evaluation; make the
  screening diagnostics opt-in.
- Duplicate resampling and payback implementations, kept as thin wrappers
  over one implementation.
- The unused resistance-component accumulators on `_AgingState`, documented
  as neither reported nor used.
- `default_order` bookkeeping in the config registry.
- Tests that assert source-code spellings or textual call order, replaced by
  observable failure-order and propagation checks.

After those, the principal simplification is one shared configuration
resolver and lifetime evaluator, with App, fixed-design evaluation,
optimization and Monte Carlo supplying their own inputs — ordinary functions
and a small state object, not an execution framework. The review also
recommended closing the `breos.adapters` item under "Architecture" outright:
BREOS builds deliberately on pvlib and the scientific Python stack, and
wrapping every upstream concept would cost more than it returns. Fix actual
compatibility problems where they occur instead.

Assumptions the review wanted bounded rather than changed:

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

The review explicitly left alone the small App facade, the config registry's
shared key definitions, packaged-resource loading, the explicit energy
ledger, PV-origin battery accounting, the degradation lifecycle boundary, and
the optional Numba backend with its parity tests. It also recommended keeping
the scientific evidence and reproduction assets, and archiving historical
publication scripts deliberately if they are ever retired.

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
