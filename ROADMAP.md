# BREOS Roadmap

This document tracks planned work that is not yet scheduled. Items here
are intentions, not commitments — see GitHub issues for active work.

## Model accuracy and validation

The goal is a gold-standard *engine* — PVsyst/HelioScope-class results
without the 3D scene modeling. That standard is earned two ways: closing
known systematic modeling gaps, and publishing reproducible evidence that
the numbers are right. This work takes priority over architectural
refactoring (see the deferred adapter layer at the bottom of this document).

### Standing validation and benchmark suite

**Status: seeded 2026-07** — `validation/` holds a seven-site worldwide
harness (PVGIS TMY inputs, PVGIS PVcalc references, PVWatts v8 fetcher) and
`tests/test_validation_drift.py` fails CI on any unintended model drift
(0.1% self-baseline) or gross error (±10% vs PVGIS). Remaining work below.

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

### Mid-interval solar position for hourly weather

`solar._prepare_solarpos_and_weather` computes solar position at the
interval *labels* of hourly averaged irradiance. PVWatts/SAM (and PVsyst,
equivalently) evaluate sun position at the interval midpoint; using labels
causes a systematic transposition/AOI error that is worst near
sunrise/sunset and for east/west-facing arrays, and it biases every
anisotropic transposition model. This is the cheapest meaningful accuracy
win available.

- Add a half-interval shift for interval-averaged weather, opt-in at first
  so existing configs reproduce bit-for-bit, then fold it into the
  recommended profile (below).
- While in there, pass `apparent_zenith` consistently: AOI already uses it,
  but `get_total_irradiance` currently receives the true `zenith`.
- Document the expected annual-yield delta via the benchmark suite.

### One inverter model everywhere (App, dc_to_ac, optimizer)

Three inverter representations exist today and they disagree: the `App`
energy balance uses a flat efficiency plus an AC cap treated as the
nameplate; `solar.dc_to_ac` uses pvlib's pvwatts part-load curve but passes
the intended AC nameplate as pvlib's `pdc0` (a *DC input* limit), so it
clips ~4% low; and the NSGA-II optimizer applies no AC clipping at all and
prices systems without the daily connection fee or battery replacements.
Designs are picked by one model and reported by another, which biases the
Pareto front toward high DC/AC ratios whose clipping is never seen.

- Unify on a single conversion model (the pvwatts part-load curve is the
  better physics) with one definition of the AC nameplate.
- Make "the optimizer scores designs with the same engine that reports
  them" a regression-tested invariant; this includes aligning
  `align_load_to_pv` with the App's wall-clock/UTC load alignment.
- This extends Phase 1 of "String-aware inverter validation and modeling"
  below and subsumes its App-only scope.

### Battery charge/discharge power limits

The dispatch loop has no C-rate limits: charging is unbounded (a 5 kWh pack
absorbs any surplus in a single step) and discharge is bounded only by the
shared inverter AC rating. Credible sizing studies need power limits.

- Add max charge/discharge power keys (or a C-rate key, default ~0.5C),
  validated in the existing "Unknown X. Available: ..." style.
- Unlimited stays available explicitly, and the default must be introduced
  with a documented yield/self-consumption delta — it will change results
  for small batteries paired with large arrays.

### Energy loss waterfall

PVsyst's loss diagram is its most-loved diagnostic, and it is pure engine
work: report the chain transposition gain → IAM → temperature → static
PVWatts stack (already componentized) → inverter clipping → conversion →
battery round-trip → curtailment. Today curtailed energy in the surplus
branch vanishes without a trace, and `inverter.InverterConversionResult`
already carries clipping bookkeeping but has no production caller. Surface
the waterfall in `App.result()` and the CLI JSON output.

### Horizon-profile input

Far-horizon shading without any 3D: PVGIS TMY is already fetched with
`usehorizon=True`, so PVGIS-sourced weather accounts for it implicitly —
but user-supplied weather files and custom horizons get nothing. Accept a
horizon profile (azimuth/elevation pairs) and apply it via pvlib's horizon
tools, documenting the PVGIS overlap so shading is not double-counted.

### Recommended model profile and future defaults

Isotropic transposition, label-timestamp sun position, beam-only IAM, and
free-standing Faiman coefficients are all kept as defaults for bit-for-bit
compatibility — but they are not what a gold-standard engine should
recommend. Define a documented "recommended" profile (haydavies/perez
transposition, mid-interval sun position, diffuse IAM, mount-appropriate
thermal coefficients), steer new users to it in the quickstart, and plan
the default flip for a major version with a clear upgrade note.

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

- **Near-term first step — shipped in 0.3.2:** unknown top-level config keys
  are now rejected. A typo such as `batery_kwh` previously slipped through
  `merge_defaults` and silently defaulted (e.g. the battery to `0.0`),
  producing plausible-but-wrong results; it now raises listing the offending
  key(s) in the existing "Unknown X. Available: ..." style.
- **Side cleanup — shipped in 0.3.2:** resolution no longer mutates the
  caller's input config. The derived `n_modules` is materialised into a fresh
  dict instead of being written back into the dict wrapped by the frozen
  `ResolvedAppConfig`.
- **Full step (pending):** collapse `DEFAULTS`, the validation rules, and the
  CLI flag definitions into the schema so a new parameter is added once, not
  four times.
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

### Make the first successful run easier to trust

The 0.3.0 onboarding pass added the "10-minute first run" quickstart,
the required-inputs page, the recipes page, the generated packaged-options
reference, the `breos list` discovery commands, and `breos validate-config` /
`breos run --dry-run` config inspection with `--json` output. Remaining work:

Ongoing docs hygiene:

- Keep install snippets and docs status text aligned with the current release
  on PyPI.
- Keep the README and getting-started configuration tables in sync with the
  public `App` config surface, including battery SOC/EOL/RTE keys and
  `pv_loss_overrides`.
- Keep the representative quickstart output excerpt in the docs close to what
  current dependency versions actually produce.

Option discovery work:

- Add matching Python helpers where useful, building on `list_modules()`.

Config inspection work:

- **Shipped in 0.3.3:** the dry-run / `validate-config --json` summary now
  includes the fully resolved static PVWatts loss components and combined loss
  percentage after applying `pv_loss_overrides`.

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
adapters" ([#11](https://github.com/Str4vinci/breos/issues/11)) above: adapters
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

Capabilities to bring fully online, roughly in order (tracking is already wired
end-to-end through `build_dc_system_base` and the multi-array path):

- **Configurable transposition models** — shipped in 0.3.2 (see the dedicated
  item below).
- **Bifacial rear-gain** — see the dedicated item below.
- **Cell-temperature model choice** — `faiman` is currently hardcoded in
  `_compute_effective_irradiance_and_cell_temp`; expose `sapm`, `pvsyst`, and
  `noct_sam` with their parameter sets. Lead with mount-type parameter
  presets (open rack vs roof mount): the current free-standing coefficients
  run cool for rooftop residential systems — BREOS's primary audience — and
  systematically overestimate their yield.
- **IAM model choice** — `ashrae` is currently hardcoded; expose `martin_ruiz`,
  `physical`, and the SAPM IAM. Also apply IAM to the diffuse components:
  effective irradiance currently applies IAM to beam only (diffuse passes at
  1.0), a ~0.5–1% systematic overestimate; pvlib's `iam.marion_diffuse`
  covers the sky/ground diffuse terms.
- **DC-side loss refinements** — optional time-series ohmic/soiling/snow models in
  place of (parts of) the flat PVWatts loss stack, where inputs allow.
- Non-goal: replacing the CEC single-diode core or the PVWatts loss model as the
  defaults. This is about *optional* fidelity, not a new default engine.

### Configurable sky-diffusion (transposition) models

**Shipped in 0.3.2.** The sky-diffusion (transposition) model is now selectable
via a `transposition_model` (a.k.a. `sky_model`) config key and
`--transposition-model` / `--sky-model` CLI flag, threaded through
`calculate_pv_production_dc`, the tracking and multi-array variants, and the
`App` config surface, with per-array overrides. Supported models: `isotropic`
(default, reproduces prior results bit-for-bit), `klucher`, `haydavies`,
`reindl`, `king`, `perez`, and `perez-driesse`, with the extra inputs the
anisotropic models need (extraterrestrial DNI, relative airmass) derived
internally. Configurable ground reflectance (`albedo` / `surface_type`) and
Perez coefficient set (`model_perez`) shipped alongside. See the CHANGELOG for
details.

Spectral irradiance modeling remains a non-goal for this item; bifacial
rear-gain is tracked separately — see "Bifacial rear-gain modeling" below.

### Bifacial rear-gain modeling

Several catalog modules are labelled "Bifacial" but BREOS models them front-side
only, so the label is currently cosmetic. pvlib *can* model rear irradiance — the
gap is inputs, not capability. Add real rear-gain so a bifacial module's extra
yield (typically ~5–15%, dominated by ground albedo and mounting height) is
actually simulated. This is the first concrete new capability under "Full pvlib PV
modeling" above.

- **Module input ("unless the panel states it, we can't model it"):** add a
  `bifaciality` factor (rear/front efficiency ratio, ~0.7–0.85 for TOPCon) to
  `PVModuleParams` and the module catalog in `pv_modules.py`. Absent or zero ⇒
  front-only, exactly as today.
- **Site/array inputs:** ground `albedo` (the single biggest driver; ~0.2 grass to
  ~0.6 snow/sand) and row geometry. Tracking arrays already carry `gcr`; fixed
  arrays need the `infinite_shed` geometry (gcr, height, pitch).
- **Model:** use `pvlib.bifacial.infinite_shed.get_irradiance` (pure pvlib, no new
  dependencies). Prefer it over `pvlib.bifacial.pvfactors.*`, which drags in the
  `pvfactors`/`shapely` stack — out of scope for the default install.
- **Integration:** compute rear POA inside
  `_compute_effective_irradiance_and_cell_temp` and blend
  `effective_irradiance += bifaciality * poa_rear` before the CEC DC model. The
  stage's output contract (DC-watts series) is unchanged, so nothing downstream
  moves — the whole point of the self-contained PV stage above.
- Validate against the front-only baseline (`bifaciality=0` must reproduce it
  bit-for-bit) and document expected rear-gain deltas versus albedo and mounting
  height.

### String-aware inverter validation and modeling

BREOS currently models PV systems at the aggregate array level. That is useful
for fast production, battery, and economics studies, but it does not prove that
a proposed PV layout is electrically buildable. Future work should add
string-aware validation and, later, string-aware inverter modeling when callers
provide module, inverter, environment, MPPT, and string-topology data.

- Design note: [docs/architecture/string-inverter-sizing.md](docs/architecture/string-inverter-sizing.md)
- Phase 1: apply aggregate inverter AC clipping consistently in the main
  `App` energy flow. Shipped for the `App` path; extended and superseded by
  "One inverter model everywhere" under Model accuracy and validation.
- Phase 2: add a pure validation API for string voltage windows, startup
  voltage, MPPT current limits, parallel-string compatibility, and DC/AC ratio
  warnings.
- Phase 3: extend module and inverter catalogs with the datasheet fields needed
  for those checks.
- Phase 4: accept optional MPPT/string topology from callers and use it to
  improve multi-array energy modeling.
- Non-goal: code-compliance certification, conductor/fuse sizing, and physical
  wiring auto-routing.

### Parameter sweeps and batch runs

The CLI runs one config to one result (`breos run --config ... --output ...`).
Research workflows often set up *many* configs — parametric sweeps over module
count, battery size, tilt, tariffs, and so on — which today means scripting the
Python `App` in a loop by hand. Add a first-class way to enumerate a parameter
grid (or a set of config files) and collect the per-run results into one table.

- **MVP shipped in 0.3.3:** `breos sweep --config ... --output ...` expands a
  `[sweep]` parameter grid in a normal App config, runs the Cartesian product
  serially, and writes one combined CSV with varied parameters, resolved system
  sizing, the BREOS version, and top-level scalar result metrics.
- Remaining work: accept a glob/list of config files resolved into one combined
  CSV/JSON of results.
- Remaining work: reuse the worker controls planned under "Resource controls
  and Apple Silicon hygiene" for parallel execution of independent runs.
- Remaining work: optionally echo a fuller resolved-config payload per row when
  users need more than the current resolved sizing columns.
- Non-goal: this is explicit enumeration, not optimization — the `optimization`
  module's NSGA-II sizing already covers searching for good designs.

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
country following the existing `residential_<cc>` convention.

- Requires per-timestep import/export pricing in the results path; the
  flat-price path must reproduce current results bit-for-bit.
- Later: TOU-aware dispatch (charge/discharge on price signals) as an opt-in
  strategy — greedy self-consumption stays the default.
- Non-goal: live tariff APIs, dynamic hourly market prices, FX feeds.

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

### 0.3.0 workflow hardening

The 0.3.0 public surface focuses on deterministic PV + stationary-battery
simulation, economic analysis, Monte Carlo uncertainty studies, and
multi-objective PV/battery sizing. Near-term roadmap items should harden those
workflows before adding new feature families:

- Keep Monte Carlo outputs and plots aligned with the public result schema.
- Add small example datasets or documented download steps for reproducible MC
  demos without committing large weather files.
- Improve multi-objective sizing examples, result serialization, and Pareto
  plotting documentation.
- **Shipped in 0.3.3:** release smoke tests cover the README quickstart, the
  Monte Carlo example path with generated local weather, and the pymoo-backed
  multi-objective optimization helper.

## Distribution and release automation

### PyPI trusted publishing

Shipped in 0.3.0: `.github/workflows/publish.yml` publishes `v*` tags that
point at `main` to PyPI via GitHub Actions OIDC trusted publishing, re-runs
the release artifact verifier before upload, and offers a manually triggered
TestPyPI dry-run. The one-time index and environment configuration is
documented in [docs/release.md](docs/release.md). Remaining work:

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
- Design: [docs/architecture/third-party-wrapping.md](docs/architecture/third-party-wrapping.md)
- Scope: pvlib first (Phase 1), then scipy / rainflow (Phase 2), then IO
  clients (Phase 3). Pandas, numpy, and matplotlib are kept direct.
- Estimated effort: ~3–4 weeks of focused work, split into many small
  PRs.

## Reference load profiles pending license verification

The following sample load profiles were removed from `rlp/` and `breos.load_profiles` before the open-source release because their redistribution terms were not confirmed. They can be re-added once written permission or a clear license is obtained from the upstream providers.

- **SynPRO Family profile** (Fraunhofer ISE) — was profile key `"2"` / `family_profile_SynPro.csv`. Contact: synpro@ise.fraunhofer.de.
- **LoadProfileGenerator family-with-3-kids profile** (Noah Pflugradt, FZJ IEK-3) — was profile key `"3"` / `LoadProfileGenerator_family_3kids.csv`. Tool itself is MIT-licensed; output redistribution policy needs author confirmation.
