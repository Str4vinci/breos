# BREOS Roadmap

This document tracks planned work that is not yet scheduled. Items here
are intentions, not commitments — see GitHub issues for active work.

## Release sequencing outlook

Recorded 2026-07 after the 0.4.0 release. Like everything here these are
intentions, not commitments; reassess after each release.

- **0.4.1 / 0.4.2** — maintainability refactors only, following a separate
  working refactor plan for selected BREOS functions. No feature or behavior
  changes.
- **0.5.0** — bifacial rear-gain is the firm scope (activation key, explicit
  rear-gain loss-waterfall stage, benchmark rows). Independently complete
  PV-fidelity items (the pvsyst real-efficiency fix, additional IAM and
  cell-temperature models) may ride along if each lands with its own
  compatibility and benchmark tests.
- **0.5.x** — the declarative config schema (behavior-preserving, and
  deliberately before TOU adds another cluster of config keys);
  horizon-profile input; and further internal maintainability work if needed.
- **0.6.0** — the currency concept plus time-of-use tariff
  valuation and static presets; flat pricing preserved bit-for-bit.

## Model accuracy and validation

The goal is a gold-standard *engine* — PVsyst/HelioScope-class results
without the 3D scene modeling. That standard is earned two ways: closing
known systematic modeling gaps, and publishing reproducible evidence that
the numbers are right. This work takes priority over architectural
refactoring (see the deferred adapter layer at the bottom of this document).

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
- **Coordination with the function-level refactor plan:** earlier internal
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

Option discovery work:

- Add matching Python helpers where useful, building on `list_modules()`.

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
- **Cell-temperature model choice** — expose `sapm` and `noct_sam` alongside
  the existing Faiman and PVsyst mounting presets, with their parameter sets,
  and let the PVsyst path take the module's real efficiency instead of
  pvlib's 0.1 default — a natural 0.5.0 ride-along, since
  `PVModuleParams.Module_Efficiency` already exists as explicitly unused
  metadata.
- **IAM model choice** — expose `martin_ruiz`, `physical`, and the SAPM IAM
  for the beam term, extending the existing diffuse-IAM option.
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
  `_compute_effective_irradiance_and_cell_temp` and blend
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

- Design note: [docs/architecture/string-inverter-sizing.md](docs/architecture/string-inverter-sizing.md)
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
  `docs/architecture/string-inverter-sizing.md`). The seam begins around
  `_dispatch_dc_step` in `breos/battery.py`, which is per-step and
  memoryless; price-aware dispatch needs lookahead, and since TOU presets
  are static and the simulation deterministic, a perfect-foresight day-ahead
  schedule (strategy sees the price series, emits charge/discharge windows
  or SOC targets; the step function stays dumb) is the honest v1 contract.
  Planned internal session refactoring may make the surrounding energy loop
  easier to reason about, but it is not itself the dispatch seam.
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
- Design: [docs/architecture/third-party-wrapping.md](docs/architecture/third-party-wrapping.md)
- Scope: pvlib first (Phase 1), then scipy / rainflow (Phase 2), then IO
  clients (Phase 3). Pandas, numpy, and matplotlib are kept direct.
- Estimated effort: ~3–4 weeks of focused work, split into many small
  PRs.

## Reference load profiles pending license verification

The following sample load profiles were removed from `rlp/` and `breos.load_profiles` before the open-source release because their redistribution terms were not confirmed. They can be re-added once written permission or a clear license is obtained from the upstream providers.

- **SynPRO Family profile** (Fraunhofer ISE) — was profile key `"2"` / `family_profile_SynPro.csv`. Contact: synpro@ise.fraunhofer.de.
- **LoadProfileGenerator family-with-3-kids profile** (Noah Pflugradt, FZJ IEK-3) — was profile key `"3"` / `LoadProfileGenerator_family_3kids.csv`. Tool itself is MIT-licensed; output redistribution policy needs author confirmation.
