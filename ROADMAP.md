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
