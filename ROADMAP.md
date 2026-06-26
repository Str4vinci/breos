# BREOS Roadmap

This document tracks planned work that is not yet scheduled. Items here
are intentions, not commitments — see GitHub issues for active work.

## Architecture

### Wrap third-party modules behind adapters

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

### Declarative config schema with strict validation

The public `App` config surface is currently defined and checked in four
separate places: the `DEFAULTS` dict and imperative `validate_config` in
`breos.app_config`, plus the `argparse` flag definitions and the
`_add_override` calls in `breos.cli`. Adding one parameter means editing all
four, which is drift-prone, and the hand-rolled validation is hard to keep in
sync with the defaults. Replace it with a single declarative schema (a
dataclass with field metadata, or `pydantic`) so defaults, types, bounds, and
documentation live in one place.

- **Near-term first step (do independently of the full refactor): reject
  unknown config keys.** Today `merge_defaults` does `{**DEFAULTS, **config}`
  and `validate_config` only checks *known* keys, so a typo such as
  `batery_kwh` is silently dropped and the battery defaults to `0.0` — the run
  succeeds and returns plausible-but-wrong numbers with no warning. This is a
  real footgun for parametric/batch studies driven by many config files. Add an
  allowed-key check that raises listing the unknown key(s), in the existing
  "Unknown X. Available: ..." style. Small, self-contained, worth a test.
- **Full step:** collapse `DEFAULTS`, the validation rules, and the CLI flag
  definitions into the schema so a new parameter is added once, not four times.
- **Side cleanup:** stop mutating the input config during resolution —
  `resolve_pv_system` writes `cfg["n_modules"] = sum(...)` into the same dict
  the `frozen=True` `ResolvedAppConfig` wraps, which makes the immutability
  partly cosmetic and the data flow harder to follow.
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

- Extend the dry-run summary with the fully resolved PVWatts loss components
  (today it only echoes `pv_loss_overrides`).

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

- **Configurable transposition models** — see the dedicated item below (0.3.2).
- **Bifacial rear-gain** — see the dedicated item below.
- **Cell-temperature model choice** — `faiman` is currently hardcoded in
  `_compute_effective_irradiance_and_cell_temp`; expose `sapm`, `pvsyst`, and
  `noct_sam` with their parameter sets.
- **IAM model choice** — `ashrae` is currently hardcoded; expose `martin_ruiz`,
  `physical`, and the SAPM IAM.
- **DC-side loss refinements** — optional time-series ohmic/soiling/snow models in
  place of (parts of) the flat PVWatts loss stack, where inputs allow.
- Non-goal: replacing the CEC single-diode core or the PVWatts loss model as the
  defaults. This is about *optional* fidelity, not a new default engine.

### Configurable sky-diffusion (transposition) models

**Target: 0.3.2.** BREOS currently hardcodes the isotropic sky-diffusion model
when transposing GHI/DHI/DNI to plane-of-array irradiance (`model="isotropic"`
in `solar._compute_effective_irradiance_and_cell_temp`). The isotropic model is
simple and robust but underestimates POA on clear days; anisotropic models
(Hay-Davies, Reindl, King, Perez, Perez-Driesse) are more accurate and are all
available in `pvlib.irradiance.get_total_irradiance`. Callers should be able to
select the model via config and the public production APIs.

- Expose a `transposition_model` (a.k.a. `sky_model`) option threaded through
  `calculate_pv_production_dc`, the tracking variant, and the `App` config
  surface, defaulting to `isotropic` for backward compatibility.
- Supply the extra inputs the anisotropic models need that the current
  `get_total_irradiance` call omits: extraterrestrial DNI, relative airmass,
  and (for Perez) the appropriate coefficient set.
- Validate against the isotropic baseline on at least one reference location and
  document the expected annual-yield differences (Perez vs isotropic can shift
  annual POA by a few percent).
- Add a docs entry and a recipe showing how to switch models.
- Non-goal for this item: spectral irradiance modeling. Bifacial rear-gain is now
  planned separately — see "Bifacial rear-gain modeling" below.

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
  `App` energy flow.
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

- For example a `breos sweep` command over a base config plus a parameter grid,
  or a glob of config files resolved into one combined CSV/JSON of results.
- Reuse the worker controls planned under "Resource controls and Apple Silicon
  hygiene" for parallel execution of independent runs.
- Echo the resolved config (and `breos` version) per row so a sweep output is
  self-describing and reproducible.
- Non-goal: this is explicit enumeration, not optimization — the `optimization`
  module's NSGA-II sizing already covers searching for good designs.

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
- Add release smoke tests for the exact README quickstart, MC example, and MOO
  example.

## Distribution and release automation

### PyPI trusted publishing

Shipped in 0.3.0: `.github/workflows/publish.yml` publishes `v*` tags that
point at `main` to PyPI via GitHub Actions OIDC trusted publishing, re-runs
the release artifact verifier before upload, and offers a manually triggered
TestPyPI dry-run. The one-time index and environment configuration is
documented in [docs/release.md](docs/release.md). Remaining work:

- Protect `v*` tags in GitHub repository settings so only maintainers can
  create release tags.

## Reference load profiles pending license verification

The following sample load profiles were removed from `rlp/` and `breos.load_profiles` before the open-source release because their redistribution terms were not confirmed. They can be re-added once written permission or a clear license is obtained from the upstream providers.

- **SynPRO Family profile** (Fraunhofer ISE) — was profile key `"2"` / `family_profile_SynPro.csv`. Contact: synpro@ise.fraunhofer.de.
- **LoadProfileGenerator family-with-3-kids profile** (Noah Pflugradt, FZJ IEK-3) — was profile key `"3"` / `LoadProfileGenerator_family_3kids.csv`. Tool itself is MIT-licensed; output redistribution policy needs author confirmation.
