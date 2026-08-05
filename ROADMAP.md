# BREOS Roadmap

This document tracks planned work that is not yet scheduled. Items here
are intentions, not commitments — see GitHub issues for active work.

## Release sequencing outlook

Recorded 2026-07 after the 0.4.0 release. Like everything here these are
intentions, not commitments; reassess after each release.

- **0.4.1** — behavior-preserving degradation and configuration refactors
  (warning collection, public degradation results, and validation boundaries).
- **0.4.2** — completed behavior-preserving maintainability and onboarding
  work: lazy optional imports, a quiet no-network installation check, first-run
  troubleshooting, dependency hygiene, coverage automation, cross-platform
  smoke tests, reproducible BLAST parity tooling, and an internal degradation
  lifecycle protocol. No scientific-model, dispatch, or public result changes.
- **0.5.0** — completed PV-fidelity release, delivered as independently
  reviewable slices: opt-in bifacial rear gain (activation key, explicit
  waterfall stage, and benchmark rows); a behavior-preserving internal
  PV-model-core refactor; selectable pvlib IAM models; sourced
  cell-temperature fidelity; and a recommended-configuration example with an
  equal-nameplate 2×600 W bifacial versus 3×400 W monofacial comparison.
  Shipped alongside them: validated inverter datasheet limits, extracted
  battery dispatch seams, a reorganized documentation site, and broader
  package metadata. Default results stay bit-for-bit compatible apart from
  three documented correctness fixes, each confined to a non-default
  configuration: multi-array tracking now honours the top-level `gcr`, an
  out-of-range `gcr` is rejected instead of silently backtracking, and the
  `pvsyst-*` temperature presets model a realistic module efficiency.
- **0.5.x** — the declarative config schema (behavior-preserving, and
  deliberately before TOU adds another cluster of config keys);
  horizon-profile input; the cost-override seam (phase 0 of economic
  scenario analysis, which TOU does not invalidate); and further internal
  maintainability work if needed.
- **0.6.0** — the currency concept plus time-of-use tariff
  valuation and static presets; flat pricing preserved bit-for-bit.
- **0.6.x / 0.7.0** — economic scenario and sensitivity analysis phases 1–3
  (escalator decomposition, scenario runner, switching values), sequenced
  after TOU restructures the same price surface.

## Model accuracy and validation

The goal is a gold-standard *engine* — PVsyst/HelioScope-class results
without the 3D scene modeling. That standard is earned two ways: closing
known systematic modeling gaps, and publishing reproducible evidence that
the numbers are right. This work takes priority over architectural
refactoring (see the deferred adapter layer at the bottom of this document).

### 0.5.0 PV-fidelity delivery (completed)

Delivered as a PR stack that kept the public `breos.App` facade and
`breos.solar` function signatures stable:

1. **Bifacial rear gain:** inert validated module `bifaciality` metadata,
   then opt-in `bifacial_model="infinite_sheds"` rear gain for fixed,
   tracking, and mixed multi-array systems, with rear irradiance feeding both
   DC power and cell temperature, an explicit loss-waterfall stage, provenance,
   a runnable ground-mount example, and paired front/rear benchmark rows.
2. **PV model core refactor:** one internal resolved-options object for
   transposition, ground reflectance, IAM, temperature, and bifacial choices,
   plus narrow IAM and temperature kernels, leaving public defaults, errors,
   and numerical results unchanged. It was a targeted seam for the two
   features that followed, not the deferred project-wide third-party-adapter
   rewrite.
3. **IAM selection:** pvlib's `ashrae`, `physical`, and `martin_ruiz` beam
   models are selectable via `iam_model`, with `ashrae` retaining the
   historical default. When diffuse IAM is enabled, pvlib's Marion
   solid-angle integration uses the same selected IAM model so beam and
   diffuse optics cannot silently disagree.
4. **Temperature fidelity:** PVsyst consumes sourced module efficiency, named
   SAPM construction/mounting presets are selectable, and `noct-sam` strictly
   requires NOCT plus efficiency metadata. No bundled catalog entry has a
   sourced NOCT yet, so catalog activation remains intentionally deferred.
5. **Guidance and evidence:** the recommended PV configuration leaves defaults
   unchanged; IAM/temperature choices have seven-site benchmark rows; and the
   equal-nameplate comparison separates 2×600 W bifacial front-only and
   rear-gain results from 3×400 W monofacial results. Its albedo, GCR, row
   height/pitch, inverter loading, and front-shading limitation are recorded
   beside the result.

Each slice landed with its own compatibility tests and benchmark evidence.
Horizon profiles, string-aware electrical validation, currency, and
time-of-use tariffs stayed outside 0.5.0, as planned.

### Standing validation and benchmark suite

Build on the existing seven-site `validation/` harness and
`tests/test_validation_drift.py` safeguards with broader reproducible evidence.

The single highest-leverage credibility item. PVsyst's authority comes from
decades of published validation; BREOS needs a reproducible harness that
compares its annual and monthly yields against SAM/PVWatts and against
measured public datasets (e.g. NREL PVDAQ), per location and per model
choice, with deltas documented and tracked over time.

- Start from the existing seeds: `tools/validate_cec_fit.py` and
  `tools/batch_compare_locations.py`.
- Publish expected-delta tables per transposition / cell-temperature / IAM
  choice, and wire a CI job that fails when a delta drifts beyond a stated
  tolerance.
- Every new physics capability (bifacial, cell-temperature models, IAM
  models) lands with its row in the benchmark table — this generalizes the
  per-item "validate against baseline" notes elsewhere in this roadmap.

### Horizon-profile input

Far-horizon shading without any 3D: PVGIS TMY is already fetched with
`usehorizon=True`, so PVGIS-sourced weather accounts for it implicitly —
but user-supplied weather files and custom horizons get nothing. Accept a
horizon profile (azimuth/elevation pairs) and apply it via pvlib's horizon
tools, documenting the PVGIS overlap so shading is not double-counted.

Planned as a standalone 0.5.x feature, deliberately outside the 0.5.0
bifacial gate: the PVGIS double-counting semantics and the input format are
distinct enough to review on their own.

### Recommended model profile and future defaults

Isotropic transposition, label-timestamp sun position, beam-only IAM, and
free-standing Faiman coefficients are all kept as defaults for bit-for-bit
compatibility — but they are not what a gold-standard engine should
recommend. Define a documented "recommended" profile (haydavies/perez
transposition, mid-interval sun position, diffuse IAM, mount-appropriate
thermal coefficients), steer new users to it in the quickstart, and plan
the default flip for a major version (targeted: 1.0) with a clear upgrade
note, together with the battery power-limit default C-rate decision (~0.5C,
which changes results for small batteries paired with large arrays and so
must ship with a documented yield/self-consumption delta).

## Architecture

### Declarative config schema with strict validation

The public `App` config surface is currently defined and checked in four
separate places: the `DEFAULTS` dict and imperative `validate_config` in
`breos.app_config`, plus the `argparse` flag definitions and the
`_add_override` calls in `breos.cli`. Adding one parameter means editing all
four, which is drift-prone, and the hand-rolled validation is hard to keep in
sync with the defaults. Replace it with a single declarative schema (a
dataclass with field metadata, or `pydantic`) so defaults, types, bounds, and
documentation live in one place.

- **Full step (pending, targeted at a 0.5.x behavior-preserving release):**
  collapse `DEFAULTS`, the validation rules, and the CLI flag definitions
  into the schema so a new parameter is added once, not four times. This is
  deliberately scheduled *before* the 0.6.0 TOU/currency work adds another
  cluster of config keys, and deserves its own release slot rather than
  riding along a feature release.
- **Coordination with the [function-level refactor plan](design/architecture/0.4x-refactor-plan.md):** earlier internal
  validation cleanup should create reusable boundaries for the full schema,
  not throwaway helpers that need another rewrite in 0.5.x.
- **The hard part is error-message parity**, not the schema itself: the
  acceptance bar is the same exception types with equally actionable
  "Unknown X. Available: ..." messages. Off-the-shelf pydantic messages do
  not meet it, so plan for either a dataclass-with-field-metadata schema
  with hand-rolled errors, or pydantic behind a message-translation layer.
- Keep all error messages actionable; preserve current behaviour for valid
  configs (regression-test the example configs in `configs/examples/`).

## Performance and portability

### Resource controls and Apple Silicon hygiene

BREOS already runs on macOS/Apple Silicon when installed in a native ARM
Python environment, but longer optimization and Monte Carlo workflows need
clearer resource controls so laptops and small-memory machines do not
oversubscribe CPU threads or memory. Future work should make parallelism
explicit, reproducible, and visible at startup.

- Add CLI and config-level worker controls for simulation batches,
  optimization, and Monte Carlo runs, for example `--workers 4` and an
  equivalent config key.
- Set conservative auto-defaults based on CPU count and available memory, with
  particular care for fanless or low-memory Apple Silicon machines.
- Control nested threading for Numba and scientific BLAS/OpenMP libraries, and
  document `NUMBA_NUM_THREADS`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
  `MKL_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS`.
- Print a compact startup diagnostic for long runs: platform/architecture, CPU
  count, selected worker count, Numba thread count, and detected memory.
- Add a benchmark/smoke mode for comparing machines without launching a full
  production run, e.g. reduced generations/population for optimizers.

## Onboarding and discoverability

### Keep the first successful run easy to trust

Continue improving the existing quickstart, discovery commands, configuration
inspection, and packaged-options reference through the following work.

Ongoing docs hygiene:

- Keep install snippets and docs status text aligned with the current release
  on PyPI.
- Keep the README and getting-started configuration tables in sync with the
  public `App` config surface, including battery SOC/EOL/RTE keys and
  `pv_loss_overrides`.
- Keep the representative quickstart output excerpt in the docs close to what
  current dependency versions actually produce.
- Keep security support tables, release-matrix headings, example commands, and
  version-specific limitation text aligned with the latest release.

Completed for 0.5.0:

- Reorganized the documentation site around task-oriented guides, model
  assumptions, and API reference, moved internal design plans, ADRs, and
  maintainer procedures to repository-only documentation, and replaced stale
  0.3.x capability references with version-neutral wording.
- Broadened PyPI classifiers and keywords, credited pvlib and the other
  upstream projects and data sources, and routed usage questions and feature
  ideas to GitHub Discussions.

Completed for 0.4.2:

- Added a no-file, no-network `breos run ... --dry-run` installation check next
  to the install command.
- Kept optional plotting imports lazy so core imports, `--help`, option
  discovery, and configuration validation do not initialize Matplotlib.
- Added a short first-run troubleshooting page covering weather/network
  failures, preset discovery, optional extras, cache behavior, and expected
  runtimes.
- Kept automated dependency maintenance focused on security updates, added
  core-package coverage that does not treat vendored BLAST as ordinary
  application code, and added lightweight macOS/Windows public-API smoke checks.
- Added reproducible tooling and provenance for the reviewed BLAST parity
  fixture, plus a narrow internal degradation lifecycle protocol that preserves
  dispatch and public result schemas.

Option discovery work:

- Add matching Python helpers where useful, building on `list_modules()`.
- Curate small `good first issue` tasks around docs, examples, and tooling so
  the contribution guide leads to approachable work.

Future additive onboarding work:

- Consider a deterministic offline `breos demo` command using clearly labeled
  synthetic inputs. Because this adds a public CLI surface, schedule it for a
  feature release (0.5.x) rather than forcing it into a 0.4.x patch.
- Consider a compact human-readable completion summary while retaining the
  full JSON result as the machine-readable output contract.

Agent and contributor setup:

- Keep `AGENTS.md` and `CONTRIBUTING.md` aligned on branch flow, test gates,
  and the public `breos.App` facade as the preferred extension point.
- Consider a single `make`/`just`/script entry for local validation if command
  drift becomes a recurring issue.

## Capability extensions

### Full pvlib PV modeling behind a self-contained PV stage

BREOS deliberately treats PV production as a *self-contained stage* with a single
output contract: **(weather + system config) → a DC-power time series in watts**,
summed to system level (or per-array for multi-array). Everything pvlib does —
transposition, tracking, cell-temperature, IAM, spectral, bifacial rear-gain —
lives *inside* that stage. The downstream chain (inverter/AC conversion, battery
dispatch, economics, emissions, Monte Carlo, NSGA-II sizing) consumes only the DC
series and must stay invariant as new physics is switched on. Keeping that
boundary firm is what lets us progressively "turn on" more of pvlib without
destabilising the rest of the engine.

This is the functional, data-flow counterpart to "Wrap third-party modules behind
adapters" ([#11](https://github.com/Str4vinci/breos/issues/11)) below: adapters
own the *types* crossing the boundary; this item owns the *contract* that boundary
guarantees.

Principles:

- One output contract: a watts DC `pd.Series` on the run's time index. A new
  capability must not add new required inputs to any downstream stage.
- New physics is opt-in and defaults to current behaviour, so existing configs
  reproduce bit-for-bit (regression-test the example configs in
  `configs/examples/`).
- Each capability declares the *extra inputs* it needs and fails loudly (in the
  existing "Unknown X. Available: ..." style) when a config selects a model whose
  inputs are missing — never silently fall back to different physics.
- Validate every new model against the current baseline on at least one reference
  location and document the expected annual-yield delta.

Capabilities still to bring online (transposition is already selectable via
`transposition_model` / `sky_model` since 0.3.2, and tracking is wired
end-to-end through `build_dc_system_base` and the multi-array path):

- **Bifacial rear-gain** — see the dedicated item below.
- **Cell-temperature model choice (0.5.0):** expose named SAPM presets and,
  where the module catalog has complete sourced inputs, `noct_sam` alongside
  the existing Faiman and PVsyst presets. Let the PVsyst path consume real
  module efficiency instead of pvlib's 0.1 fallback. `noct_sam` must require
  NOCT plus efficiency rather than assuming either value.
- **IAM model choice (delivered for 0.5.0):** `iam_model` exposes pvlib's
  `martin_ruiz` and `physical` beam models alongside the historical `ashrae`
  path, and Marion diffuse IAM follows that same selection. SAPM IAM remains
  later work because its module-specific polynomial coefficients are not in
  BREOS's CEC-style catalog.
- **DC-side loss refinements** — optional time-series ohmic/soiling/snow models in
  place of (parts of) the flat PVWatts loss stack, where inputs allow.
- Non-goal: replacing the CEC single-diode core or the PVWatts loss model as the
  defaults. This is about *optional* fidelity, not a new default engine.

### Bifacial rear-gain modeling

Several catalog modules are labelled "Bifacial" but BREOS models them front-side
only, so the label is currently cosmetic. pvlib *can* model rear irradiance — the
gap is inputs, not capability. Add real rear-gain so a bifacial module's extra
yield (typically ~5–15%, dominated by ground albedo and mounting height) is
actually simulated. This is the first concrete new capability under "Full pvlib PV
modeling" above.

- **Module input ("unless the panel states it, we can't model it"):** add a
  `bifaciality` factor (rear/front efficiency ratio, ~0.7–0.85 for TOPCon) to
  `PVModuleParams` and the module catalog in `pv_modules.py` — as *datasheet
  metadata only*. Metadata never activates modeling: adding `bifaciality` to a
  catalog module must not change any existing configuration's results.
- **Activation is a separate config key:** `bifacial_model = "none" |
  "infinite_sheds"` (default `"none"`), following the `temperature_model` /
  `diffuse_iam` pattern. Row geometry is required only when the model is
  activated, and selecting `"infinite_sheds"` for a module without a sourced
  `bifaciality` raises in the "Unknown X. Available: ..." style rather than
  assuming a typical value.
- **Site/array inputs:** ground `albedo` / `surface_type` already exist
  (0.3.2); the new inputs are row geometry. Tracking arrays already carry
  `gcr`; fixed arrays need the infinite-sheds geometry (gcr, height, pitch).
- **Model:** use `pvlib.bifacial.infinite_sheds.get_irradiance_poa` (pure
  pvlib, no new dependencies) called for the *back* surface only (flipped
  tilt/azimuth), and apply `bifaciality` in BREOS code. Do not use
  `get_irradiance`: its `poa_global` already folds in a silent
  `bifaciality=0.8` default, which risks double counting (or modeling a rear
  gain nobody configured). Its narrower transposition support (isotropic /
  haydavies) then constrains only the rear estimate — the front chain keeps
  BREOS's full transposition set. Prefer all of this over
  `pvlib.bifacial.pvfactors.*`, which drags in the `pvfactors`/`shapely`
  stack — out of scope for the default install.
- **Documented hybrid limitation:** the front side stays BREOS's existing
  chain (an unshaded, isolated array) while the rear side sees the
  infinite-sheds row geometry. That is correct in the small-gcr rooftop
  limit — BREOS's primary audience — and front-optimistic for dense
  ground-mount rows. Document it, and consider a warning for tight pitch.
- **Integration:** compute rear POA inside
  `_compute_irradiance_and_cell_temp_detail` and blend
  `effective_irradiance += bifaciality * poa_rear` before the CEC DC model. The
  stage's output contract (DC-watts series) is unchanged, so nothing downstream
  moves — the whole point of the self-contained PV stage above.
- **Diagnostics:** rear gain is an explicit `pv_loss_waterfall` and
  provenance stage. Both rear gain and the front chain land in effective
  irradiance before the CEC model, so without its own bucket the gain would
  be silently absorbed into the transposition stage (or masquerade as an IAM
  change).
- Validate against the front-only baseline (`bifacial_model="none"` must
  reproduce it bit-for-bit) and document expected rear-gain deltas versus
  albedo and mounting height.

### String-aware inverter validation and modeling

BREOS currently models PV systems at the aggregate array level. That is useful
for fast production, battery, and economics studies, but it does not prove that
a proposed PV layout is electrically buildable. Future work should add
string-aware validation and, later, string-aware inverter modeling when callers
provide module, inverter, environment, MPPT, and string-topology data.

- Design note: [design/architecture/string-inverter-sizing.md](design/architecture/string-inverter-sizing.md)
- First, add a pure validation API for string voltage windows, startup
  voltage, MPPT current limits, parallel-string compatibility, and DC/AC ratio
  warnings.
- Then extend module and inverter catalogs with the datasheet fields needed
  for those checks.
- Later, accept optional MPPT/string topology from callers and use it to
  improve multi-array energy modeling.
- Non-goal: code-compliance certification, conductor/fuse sizing, and physical
  wiring auto-routing.

### Parameter sweeps and batch runs

Extend the current single-config, serial `breos sweep --config ... --output ...`
workflow for research runs that need more than one parameter grid:

- Accept a glob/list of config files resolved into one combined CSV/JSON of
  results.
- Reuse the worker controls planned under "Resource controls and Apple
  Silicon hygiene" for parallel execution of independent runs.
- Optionally echo a fuller resolved-config payload per row when users need
  more than the current resolved sizing columns.
- Non-goal: this is explicit enumeration, not optimization — the `optimization`
  module's NSGA-II sizing already covers searching for good designs.
- Related: phase 0 of "Economic scenario and sensitivity analysis" below adds
  dotted-key support to `[sweep]`, without which no economic parameter below the
  top level can be enumerated at all. Do the two together if they land in the
  same release.

### Globalization: economics and grid emissions beyond Europe

BREOS ships cost and grid-emission presets that are entirely European. The cost
catalog (`breos/data/configs/costs.json`) covers only `residential_de`,
`residential_es`, and `residential_pt` — all Eurozone, priced implicitly in EUR
with no currency field — and the emissions catalog
(`breos/data/configs/emissions.json`) covers only the ~36 ENTSO-E countries.
Adding other countries' economics would let non-European users get realistic
LCOE, payback, and CO₂ results without hand-entering every cost.

- Introduce an explicit `currency` concept. Today every cost is bare EUR; adding
  other-country presets first needs a currency field per cost preset, surfaced in
  results and plots, so a BRL or USD preset cannot be silently mixed with EUR
  defaults.
- Add non-EU cost presets to `costs.json` (electricity tariff, feed-in / sold
  price, module / inverter / storage capex, install and maintenance), each citing
  its source and year and following the existing `residential_<cc>` key
  convention.
- Add non-European grid-emission factors to `emissions.json` beyond the ENTSO-E
  set (for example BR, US, AU, IN, CN, JP), keyed by the same ISO country codes,
  with source and vintage documented.
- Keep the "Unknown cost preset '...'. Available: ..." and "Unknown emissions
  country '...'" errors actionable as the catalogs grow.
- Non-goal: live tariff / FX feeds — this is static, documented, per-country
  presets, not a market-data integration. Time-of-use tariff *structures* are
  no longer a non-goal; see the dedicated item below.

### Time-of-use tariff structures

Flat import/export prices cannot value a battery correctly in markets where
time-of-use tariffs are standard (ES 2.0TD periods, PT bi/tri-horária, DE
dynamic tariffs) — and battery economics is BREOS's differentiator.
Restructure the economics layer so a tariff is a pluggable price time series
rather than a single scalar, and ship static, documented TOU presets per
country following the existing `residential_<cc>` convention. Phase-1
valuation is targeted at 0.6.0, together with the currency concept from the
globalization item above — both restructure the same preset/economics
surface, and doing TOU presets first in bare EUR would mean touching every
preset twice.

- Requires per-timestep import/export pricing in the results path; the
  flat-price path must reproduce current results bit-for-bit.
- Valuation does not require touching `simulate_energy_balance()`, but it is
  not economics-only either: the App runner already re-simulates every year
  at full timestep resolution, yet retains only year 1's timestep frame and
  reduces each year to annual import/export totals before economics runs.
  Phase 1 therefore computes per-year price-weighted import cost and export
  revenue *inside* the runner's year loop (loop-local aggregation alongside
  the existing yearly summaries — no retention or schema change).
- Tariff periods are defined in local wall-clock time (ES 2.0TD, PT
  horária), so tariff resolution must reuse the timezone-aware alignment
  machinery from 0.3.4 and survive DST transitions and the
  TMY-year-replayed-N-times pattern.
- Currency: existing EUR-named surfaces (`npv_savings_eur`, `lcoe_eur_kwh`,
  `total_investment_eur`, `battery_replacement_cost_eur`, the financial
  dicts, sweep CSV columns, plot labels) remain as compatibility aliases for
  EUR configurations while currency-neutral fields and explicit currency
  metadata are introduced.
- Later (0.7.0 target): TOU-aware dispatch (charge/discharge on price
  signals) as an opt-in strategy — greedy self-consumption stays the
  default. This needs an explicit dispatch-strategy contract, specified in a
  design doc before implementation (à la
  `design/architecture/string-inverter-sizing.md`). The seam begins around
  `_dispatch_dc_step` in `breos/battery.py`, which is per-step and
  memoryless; price-aware dispatch needs lookahead, and since TOU presets
  are static and the simulation deterministic, a perfect-foresight day-ahead
  schedule (strategy sees the price series, emits charge/discharge windows
  or SOC targets; the step function stays dumb) is the honest v1 contract.
  Planned internal session refactoring may make the surrounding energy loop
  easier to reason about, but it is not itself the dispatch seam.
- Non-goal: live tariff APIs, dynamic hourly market prices, FX feeds.

### Economic scenario and sensitivity analysis

BREOS answers "what does *this* system cost and yield under *one* set of
economic assumptions". The assumptions that dominate the answer — electricity
price, capex, feed-in tariff, discount rate, and how each escalates over the
project lifetime — are the ones a user is least able to know and most wants to
interrogate. Make them explicit, varyable, and reportable, so BREOS can produce
the four standard techno-economic analyses:

- **Sensitivity analysis** (one-at-a-time): which assumption does the answer
  hinge on? Reported as a tornado diagram of NPV swing per parameter.
- **Scenario analysis**: coherent *bundles* of assumptions varied together
  (the IEA STEPS/APS pattern), not one lever at a time.
- **Switching values** (break-even / threshold analysis): invert the question —
  what feed-in tariff or capex makes NPV zero, or payback ≤ N years?
- **Probabilistic analysis**: distributions over economic inputs, reported as
  P10/P50/P90 NPV and LCOE, alongside the weather-year and load uncertainty
  Monte Carlo already samples.

**Sequencing.** Phase 0 is a 0.5.x item; phases 1–3 follow the 0.6.0 TOU and
currency work, which restructures the same preset/economics surface from scalar
prices to price time series. Building the scenario layer on today's scalar
surface would mean building it twice — the same reasoning that put the
declarative config schema before TOU.

**Architectural note: economics is downstream of the energy balance.** Dispatch
is currently price-blind (`breos/battery.py` carries `replacement_cost` only as
a bookkeeping tag on replacement events, never as a dispatch input), so varying
prices, capex, tariffs, or the discount rate changes no simulated energy flow.
A pure-economics scenario run must therefore simulate the physics *once* and
re-run only `cost_analysis_projection()` per scenario — hundreds of scenarios in
seconds rather than hundreds of full simulations. This shortcut becomes wrong
the moment TOU-aware dispatch (0.7.0 target, above) makes price an input to
dispatch, so the scenario runner must detect a price-aware dispatch strategy and
fall back to full re-simulation. Treat that guard as a correctness requirement,
not an optimisation detail.

Phases:

- **Phase 0 — cost-override seam (0.5.x).** Today every price and capex term
  (`electricity_cost`, `electricity_sold_cost`, `module_cost_per_w`,
  `storage_cost_per_kwh`, …) reaches `CostParams` *only* through `cost_preset`
  in `resolve_costs`; only `inflation_rate`, `sell_price_inflation`, and
  `discount_rate` are top-level keys. And `_sweep` merges the grid at the top
  level (`{**config, **varied}`), so it cannot reach a nested key. The result is
  that electricity price and capex cannot be swept at all without authoring one
  throwaway preset per value. Add a `[costs]` override table layered over
  preset → dataclass defaults, and dotted-key support in `[sweep]`, both with
  the existing `Unknown key '...'. Available: ...` error style. Small,
  independently useful, and unaffected by the TOU restructure.
- **Phase 1 — escalator decomposition (0.6.x).** A single `inflation_rate`
  currently escalates import cost, O&M, the standing charge, *and* battery
  replacement capex alike. That conflates general inflation with energy-price
  escalation and with capex learning, and the last one is arguably backwards:
  battery capex has followed a declining experience curve, so inflating
  replacement cost at ~2%/yr systematically penalises storage. Split into named
  escalators — electricity import, export/feed-in, O&M, and a replacement-capex
  learning rate (declining by default) — each reproducing current results
  bit-for-bit when set to today's values.
- **Phase 2 — scenario definition and runner (0.6.x/0.7.0).** Named coherent
  bundles in config (`[scenarios.high_price_low_capex]`), each a validated
  override set over phase 0's surface; one result row per scenario × design.
  Implements the simulate-once/re-cost-many path and the price-aware-dispatch
  guard above. Ship a tornado diagram and a scenario-comparison plot.
- **Phase 3 — switching values.** A 1-D root-find over any scalar economic knob
  for a target (`npv_savings == 0`, `payback_year <= N`). Cheap once phases 0–2
  exist, and the highest-value output for a policy audience: "the feed-in tariff
  at which residential storage turns NPV-positive in PT is X".
- **Phase 4 (optional) — economic uncertainty in Monte Carlo.** Extend
  `MonteCarloSettings`, which today samples only weather year and load scale, to
  sample economic inputs from distributions and report P10/P50/P90 NPV and LCOE.

- Non-goal: BREOS models a *single system*. Scenario analysis says whether one
  investment pencils out under a given assumption set; it does not model
  adoption, deployment volume, or total programme cost. Those need an uptake
  model layered on top and are out of scope.
- Non-goal: live price, FX, or market-data feeds — the same boundary the
  globalization and TOU items draw. Scenarios are user-declared and static.

### Additional Li-ion battery chemistries

The battery degradation model is calibrated for LFP only. Calendar aging uses the
Naumann 2020 LFP parameter sets (`naumann_lam_field_calibrated` default and
variants in `breos/constants.py`), and cycle aging uses LFP Wöhler curves
(`WOEHLER_LFP_CONSERVATIVE` / `_TYPICAL` / `_OPTIMISTIC`, consumed in
`breos/polysun_degradation.py`). This is the same "label without the physics" gap
as bifacial modules. In 0.3.3 the native `BatteryConfig.battery_type` selector
was made honest: `LFP` normalizes to `lfp`, and unsupported values now raise
instead of silently reusing LFP cycle-aging parameters. Add real per-chemistry
aging so NMC / NCA packs degrade on their own parameters.

- Add a `battery_chemistry` config key (defaulting to `lfp`) that selects the
  calendar and Wöhler parameter sets, validated in the existing "Unknown X.
  Available: ..." style.
- Add NMC and NCA parameter sets (calendar-aging coefficients and Wöhler `a` / `b`
  cycle coefficients), each with a documented source; consider LTO and
  sodium-ion as later additions.
- Allow per-chemistry calendar-life and round-trip-efficiency defaults where they
  differ from LFP (for example NMC's higher energy density but shorter cycle
  life).
- Default `lfp` must reproduce current results bit-for-bit — regression-test the
  example configs in `configs/examples/`, exactly as the PV-capability items above
  require.
- Non-goal: electrochemical / physics-based (single-particle, P2D) models — this
  stays an empirical Wöhler-plus-calendar approach, just parameterised per
  chemistry.

**Priority note (2026-07):** the 0.4.0 BLAST integration lowers this item's
urgency — anyone needing non-LFP degradation can opt into a sourced,
cell-specific BLAST model today. Lower priority, not obsolete: BLAST
provides specific cells, not generic NMC/NCA defaults, and it remains
unavailable under Monte Carlo. Native per-chemistry parameter sets stay on
the roadmap for the generic-default and MC use cases.

### BLAST under Monte Carlo

BLAST is explicitly rejected in Monte Carlo runs (0.4.0). Enabling it is a
candidate 0.8.0 headline, after the degradation-protocol and snapshot-codec
refactors have settled. This is real work, not just removing the rejection:
per-draw continuation semantics, provenance that identifies the sampled
configuration per trajectory, and performance (N draws × daily model
stepping) all need design and testing.

### Workflow hardening

The public surface focuses on deterministic PV + stationary-battery
simulation, economic analysis, Monte Carlo uncertainty studies, and
multi-objective PV/battery sizing. Near-term work should harden those
workflows before adding new feature families:

- Keep Monte Carlo outputs and plots aligned with the public result schema.
- Add small example datasets or documented download steps for reproducible MC
  demos without committing large weather files.
- Improve multi-objective sizing examples, result serialization, and Pareto
  plotting documentation.

## Distribution and release automation

### Release tag protection

- Protect `v*` tags in GitHub repository settings so only maintainers can
  create release tags.

## Deferred

### Wrap third-party modules behind adapters

**Deprioritized 2026-07:** the model-accuracy and validation work above
delivers more user-visible value per week of effort; pvlib API churn is
modest and the `pvlib>=0.14,<0.16` pin already contains it. Revisit once
the accuracy items land.

Concentrate every direct import of `pvlib`, `scipy`, `rainflow`, and other
third-party scientific libraries in a small `breos.adapters` layer so that
upstream API changes only affect a single file rather than the whole
package. The current `Location` parameter exposed by
`solar.calculate_pv_production_dc` (and several other public functions) is
a `pvlib.Location`, which means BREOS does not own its own public API.

- Tracking issue: [#11](https://github.com/Str4vinci/breos/issues/11)
- Design: [design/architecture/third-party-wrapping.md](design/architecture/third-party-wrapping.md)
- Scope: pvlib first (Phase 1), then scipy / rainflow (Phase 2), then IO
  clients (Phase 3). Pandas, numpy, and matplotlib are kept direct.
- Estimated effort: ~3–4 weeks of focused work, split into many small
  PRs.

## Reference load profiles pending license verification

The following sample load profiles were removed from `rlp/` and `breos.load_profiles` before the open-source release because their redistribution terms were not confirmed. They can be re-added once written permission or a clear license is obtained from the upstream providers.

- **SynPRO Family profile** (Fraunhofer ISE) — was profile key `"2"` / `family_profile_SynPro.csv`. Contact: synpro@ise.fraunhofer.de.
- **LoadProfileGenerator family-with-3-kids profile** (Noah Pflugradt, FZJ IEK-3) — was profile key `"3"` / `LoadProfileGenerator_family_3kids.csv`. Tool itself is MIT-licensed; output redistribution policy needs author confirmation.
