# BLAST Degradation Engine

**Status:** Phase 0-1 implemented locally; Phase 2+ planned
**Relates to:** ROADMAP "Additional Li-ion battery chemistries"
**Owner:** BREOS maintainers

## Problem

BREOS's battery degradation is calibrated for **LFP only**. Calendar aging uses
the Naumann/Lam power-law (`k0, Ea, b, n`) in `update_battery_soh_calendar`, and
cycle aging uses an LFP Naumann/Wöhler form in `update_battery_soh_cyclewise`.
Before 0.3.3, a config could name a non-LFP pack (`BatteryConfig.battery_type`)
but still age it on LFP curves. In 0.3.3 the native path was made explicit:
`BatteryConfig(battery_type="LFP")` normalizes to `"lfp"`, and unsupported
chemistries now raise instead of silently reusing LFP cycle-aging parameters.
The runner (`breos/runners/app.py`) exposes BLAST through explicit
`degradation_engine` / `blast_model` config keys; chemistry-profile defaults
remain a Phase 2 task.

NREL's [BLAST-Lite](https://github.com/NREL/BLAST-Lite) (BSD-3) provides **14
lab-calibrated, DOI-cited degradation models** (in the version vendored) spanning
the exact chemistries the roadmap names (NMC111/622/811, NCA, NCA-Si, LMO, LTO,
2nd-life). They are empirical (calendar + cycle), not electrochemical/P2D —
matching our stated non-goal.

## Decision: vendor as a parallel engine, do not re-map parameters

BLAST's parameterization is **structurally different** from BREOS's, so
translating BLAST numbers into BREOS's `(k0,Ea,b,n)` + Wöhler form is
lossy-to-impossible:

| BREOS knob | BLAST equivalent | Maps? |
|---|---|---|
| `k0` calendar rate | `qcal_A` (absorbs T_ref + %↔fraction) | ✅ algebra |
| `Ea` Arrhenius | `qcal_B = −Ea/R` | ✅ algebra |
| `b` time exponent | `qcal_p` | ✅ same role |
| `n` (`soc^n` stress) | `exp(qcal_C·soc/T)` — exponential, T-coupled (or via `Ua`) | ❌ different function |
| Wöhler/Naumann cycle `(a,b,c,d,z)` | `(qcyc_A…E, qcyc_p)` — **T-dependent**, linear DOD | ❌ different model |

BREOS cycle aging is temperature-*independent*; BLAST's is not. Worse, several
target chemistries (NMC111 Kokam, NMC622 DENSO, LFP Sony-Murata, NCA-Si Sony)
split capacity loss into **separate LLI / LAM / resistance modes**, with LAM a
**sigmoid "knee"** and DENSO an **exponential break-in** — shapes BREOS's
single-bucket power-law cannot represent at all. A BLAST parameter is only
meaningful paired with its exact equation + trajectory kernel.

Therefore: **vendor BLAST's model classes and run them as an opt-in alternative
engine behind a config selector.** Default LFP path stays bit-for-bit.

## Architecture

```
breos/degradation/
  __init__.py
  blast/                     # vendored, BSD-3 header + NOTICE preserved
    degradation_model.py     # BatteryDegradationModel base + trajectory kernels
    rainflow.py              # (or reuse breos's `rainflow` dep — see note)
    functions.py             # rescale_soc ONLY — see vendoring notes below
    models/                  # all 14 chemistry classes (enabled in phases)
  engine.py                  # BlastEngine adapter — uniform step() interface
```

- **Vendor, do not add `blast-lite` as a PyPI dependency.** The PyPI package
  pulls `matplotlib`/`pandas` and carries all 14 models. Vendoring lets us trim
  to a **numpy-only** subset: `degradation_model.py` imports `matplotlib` (only a
  commented test block) and `pandas` (only the DataFrame input path BREOS won't
  use) — both removable. Keeps the core install lean, consistent with our
  optional-extras philosophy. **One exception:** `lfp_gr_SonyMurata3Ah` (P3b)
  imports `scipy.stats` for cell-to-cell variability sampling — `scipy` is
  already a BREOS core dependency, so no new dep; the Phase 0/1 flagship + POC
  (LFP 250Ah, NCA Panasonic) are genuinely numpy-only.
- **NumPy 2 rename (required, Phase 0).** Upstream pins `numpy<2.0.0` and calls
  `np.trapz` at ~40 sites (`degradation_model.py:537` plus ~12 of the 14
  models); BREOS requires `numpy>=2.0`, where `np.trapz` was **removed**.
  Rename `np.trapz` → `np.trapezoid` throughout the vendored files — the two
  are numerically identical, so Phase 0 stays behavior-neutral. No other
  NumPy-2-removed APIs (`np.NaN`, `np.float_`, `np.in1d`, …) appear in the
  vendoring scope (checked 2026-07-01).
- **Extract `rescale_soc` only — do not vendor `blast/utils/functions.py`
  wholesale.** The full file imports `h5pyd`, `geopy`, and `scipy.spatial` for
  NSRDB-fetching/demo helpers and would fail at import inside BREOS; the base
  class needs only the ~9-line `rescale_soc`.
- **Pull from clean upstream `github.com/NREL/BLAST-Lite`** (org renamed —
  redirects to `NatLabRockies/BLAST-Lite`; record the exact commit vendored in
  `ATTRIBUTIONS.md`), NOT the local `work/BLAST-Lite` checkout — that copy has
  a botched `NREL→NLR` find/replace (`nlr.gov`, broken FASTSim links,
  `Paul.Gasper@nlr.gov`) plus a fork-local `numpy<2.0.0` pin.

### Adapter contract (`breos/degradation/engine.py`)

A thin `BlastEngine` wraps one BLAST model instance and exposes what the energy
loop needs, mirroring the current native flow:

```python
class BlastEngine:
    def __init__(self, blast_model_key: str): ...     # instantiates the class
    def step(self, t_secs_day, soc_abs_day, T_cell_day_C) -> float:
        # calls model.update_battery_state(...); returns SoH fraction = outputs['q'][-1]
    def soh(self) -> float: ...                        # current q
    def state_snapshot(self) -> dict: ...              # for cross-year threading
    @classmethod
    def from_snapshot(cls, key, snapshot) -> "BlastEngine": ...
    def reset(self): ...                               # on replacement (fresh instance)
```

BLAST's `update_battery_state(t_secs, soc, T_celsius)` is already designed for
**incremental chunks** — it appends to internal state arrays and tracks
cumulative `t_days`/`efc` itself. We pass **per-day relative seconds**
(`[0, 3600, …, 86400]`); only the per-chunk delta matters, and the model
accumulates total time. SoH = `outputs['q'][-1]` (fraction of nominal, 1.0→0),
same semantics as `battery_soh_decimal`.

## Integration (per-day incremental — "Strategy A")

`simulate_energy_balance` already runs degradation on a **daily cadence**: it
buffers a day of absolute SoC and, once `soc_buf_idx >= steps_per_day`, calls
`update_battery_soh_cyclewise` + `update_battery_soh_calendar`
(`breos/battery.py:429-464`). The BLAST path slots in at exactly that point:

```python
if degradation_engine == "blast":
    battery_soh_decimal = blast_engine.step(t_secs_day, soc_series.values, mean_T_cell_or_series)
else:
    # existing native cyclewise + calendar calls, unchanged
```

This preserves the **degradation→dispatch feedback** (usable capacity shrinks via
`update_battery_soc` as SoH drops) — a post-processing "run BLAST once over the
whole series" approach would break that feedback and is rejected.

### Daily time grid (correctness)

BLAST derives elapsed time from the chunk itself —
`delta_t_days = t_days[-1] - t_days[0]` (`degradation_model.py:523`). But the loop
buffers only `steps_per_day` **post-step** SoC samples (`battery.py:413`), which
for hourly data span 0..23 h — `delta_t_days` would be 23/24, undercounting
calendar aging (~4 %/day) and dropping the final cycle segment.

The adapter must build a **full-day grid of `steps_per_day + 1` endpoints**:

- **SoC:** prepend the day's start anchor (the prior day's last `soc_absolute`,
  or the initial SoC on day 0) to the buffered values → 25 points for hourly
  spanning `t_secs = [0, 3600, …, 86400]`, so `delta_t_days == 1.0` exactly.
- **Boundary sharing:** day N's last sample *is* day N+1's anchor. Because BLAST
  computes `delta_efc` per chunk as `sum(|diff(soc)|)/2`, sharing the boundary
  endpoint counts each SoC segment exactly once across the two chunks — no gap,
  no double-count.
- **Temperature:** pass the matching `T_cell` series on the same grid (BLAST
  trapz-integrates it); the existing `T_cell_day_sum` daily mean is not enough for
  the series path.

So the BLAST path must retain one extra carry variable: the start-of-day SoC and
`T_cell` anchors.

### Cross-year state threading (critical)

The runner calls `simulate_energy_balance` **once per simulated year**, threading
degradation state via `initial_fec`, `initial_calendar_seconds`, etc.
(`breos/runners/app.py:103-114`). BLAST state is richer than those scalars, so:

- Add `initial_degradation_state: dict | None` param to `simulate_energy_balance`
  and return a `final_degradation_state` in its tuple (or thread the live
  `BlastEngine` instance through the year loop).
- On year N+1, rebuild via `BlastEngine.from_snapshot(...)` so cumulative
  `t_days`/`efc`/states continue seamlessly.

### Replacement reset

On replacement (`breos/battery.py:507`), the native path zeroes the scalar
accumulators. BLAST path instead calls `blast_engine.reset()` (fresh instance).

### Resistance fade

Phase 1: **disable `enable_resistance_fade` for the BLAST path** (raise if both
set) — BLAST multi-mode models expose `outputs['r']`, but mapping resistance→RTE
derate is its own task. Defer to Phase 4.

## Config plumbing

New keys (added in all the places the ROADMAP "declarative schema" item flags as
drift-prone — keep additions minimal until that lands):

1. `breos/app_config.py` `DEFAULTS`: `"degradation_engine": "native"`,
   `"blast_model": None`.
2. `breos/app_config.py` `validate_config`: `degradation_engine ∈ {native, blast}`;
   if `blast`, `blast_model` must be a currently-enabled key (actionable
   "Unknown … Available:" error listing the enabled models). Also **reject
   `degradation_engine="blast"` together with Monte Carlo** until Phase 4 — raise
   a clear error, never silently fall back to native (see the Monte Carlo note in
   Performance / Phasing).
3. `breos/runners/app.py`: pass both into `BatteryConfig`. **Retire or fully
   deprecate the legacy `battery_type` field** (see Open decisions) — the new
   `degradation_engine` / `blast_model` keys replace it; don't overload a field
   that currently means native-LFP-only.
4. `breos/cli.py`: `--degradation-engine` / `--blast-model` flags via
   `_add_override`.

`degradation_engine="native"` (default) ⇒ existing behavior, **bit-for-bit**.

When `blast_model` is set, its chemistry profile (see below) is resolved into the
`BatteryConfig` defaults *before* user overrides are applied, following the
precedence rule in "Chemistry profile registry."

## Model catalog (vendor all 14, enable in phases)

Vendor **all 14** model files — they are tiny and share one base class, so the
marginal cost of the full catalog over a subset is negligible. *Enabling* a key
(exposing it as a supported `blast_model` value) is gated on a passing smoke test
+ a surfaced `experimental_range`, so the engine is honest about which cells a
stationary, low-C-rate study is extrapolating.

Enable order is by degradation-form complexity: the 2-bucket power-law models
need no new kernels; the sigmoid / break-in / multi-mode models exercise the
`_update_sigmoid_state` / `_update_exponential_relax_state` / `_update_power_B_state`
kernels and multi-output handling, so they validate last.

| Key | Class | Form | Enable |
|---|---|---|---|
| `lfp_gr_250ah_prismatic` | `Lfp_Gr_250AhPrismatic` | 2-bucket power | **P1 — LFP flagship (stationary)** |
| `nca_gr_panasonic_3ah` | `Nca_Gr_Panasonic3Ah_Battery` | 2-bucket power | **P1 — POC** |
| `lmo_gr_nissanleaf_66ah_2nd` | `Lmo_Gr_NissanLeaf66Ah_2ndLife_Battery` | 2-bucket power | P3a (2nd-life) |
| `nmc811_grsi_lgm50_5ah` | `Nmc811_GrSi_LGM50_5Ah_Battery` | 2-bucket power | P3a |
| `nmc811_grsi_lgmj1_4ah` | `Nmc811_GrSi_LGMJ1_4Ah_Battery` | 2-bucket power | P3a |
| `nmc_gr_50ah_b1` | `NMC_Gr_50Ah_B1` | 2-bucket power | P3a |
| `nmc_gr_50ah_b2` | `NMC_Gr_50Ah_B2` | 2-bucket power | P3a |
| `nmc_gr_75ah_a` | `NMC_Gr_75Ah_A` | 2-bucket power | P3a |
| `nmc111_gr_sanyo_2ah` | `Nmc111_Gr_Sanyo2Ah_Battery` | 3× power (q + R) | P3a |
| `nmc_lto_10ah` | `Nmc_Lto_10Ah_Battery` | 3× power (incl. qGain rise) | P3a |
| `lfp_gr_sonymurata_3ah` | `Lfp_Gr_SonyMurata3Ah_Battery` | sigmoid + power_B, multi-mode | P3b |
| `nca_grsi_sonymurata_2p5ah` | `NCA_GrSi_SonyMurata2p5Ah_Battery` | 2× sigmoid | P3b |
| `nmc111_gr_kokam_75ah` | `Nmc111_Gr_Kokam75Ah_Battery` | power×4 + sigmoid (LLI+LAM+R) | P3b |
| `nmc622_gr_denso_50ah` | `Nmc622_Gr_DENSO50Ah_Battery` | power + exp break-in | P3b |

Note: several keys (Panasonic, Sony-Murata cylindrical, the NMC fast-charge
pouches) are EV / high-power cells tested well above stationary C-rates — they
run, but lean on the out-of-range warning below.

## Chemistry profile registry (per-chemistry settings)

There are **three tiers** of per-chemistry data; only the third is a user
*setting*. Conflating them is the trap to avoid.

1. **Degradation parameters** (`qcal_*` / `qcyc_*`) — baked into the vendored
   BLAST class, calibrated to papers. **Never user-tunable.**
2. **Validity ranges** (`experimental_range`, e.g. the 250Ah LFP declares
   `cycling_temperature: [10, 45]`, `dod: [0.8, 1]`, `max_rate_charge: 0.65`) —
   also baked in; drives **warnings, not tuning**.
3. **Operating-envelope defaults** — RTE, SoC window (`min_soc`/`max_soc`),
   `eol_percentage`, C-rate limits, energy density. These **differ by chemistry**
   (LFP tolerates 0–100% DoD + long calendar life; NMC/NCA prefer narrower
   windows; LTO huge cycle life / low energy density; 2nd-life Leaf starts below
   100% SoH) and **are the legitimate "settings per chemistry."**

### Design: a declarative registry feeding existing `BatteryConfig` fields

A small `breos/degradation/chemistry_profiles.py` (or JSON in `breos/data/configs/`,
mirroring the existing `costs.json` / `emissions.json` preset pattern) keyed by
`blast_model`, supplying **only tier-3 defaults**. Selecting a model auto-loads
its profile; every field stays independently overridable.

Precedence (explicit, least-surprising):

```
explicit user config  >  chemistry profile default  >  global BatteryConfig default
```

**Merge-order implementation note.** `resolve_app_config` currently does
`merge_defaults(config)` *then* validates (`app_config.py:382-385`), so by
validation time the raw user keys are indistinguishable from defaults. To honor
the precedence above, capture the **raw user key set** before merging and resolve
as `{**DEFAULTS, **chemistry_profile, **raw_user_config}` — the profile fills only
keys the user did not set. (This is also a prerequisite the ROADMAP "declarative
schema" item will need.)

### Rules

- **Don't fabricate tier-3 numbers.** Ship a per-chemistry default only where a
  source supports it; otherwise inherit the global default. (Same "documented
  source" bar the ROADMAP sets — a made-up per-chemistry RTE is worse than the
  honest global default.)
- **Cost stays out of the chemistry profile.** $/kWh already lives in the
  cost-preset system; duplicating it here creates two sources of truth. The
  profile owns the *electrochemical* envelope only.
- **Warn, don't block, on conflicts.** If a user picks NMC and forces
  `max_soc=1.0`, emit an `experimental_range` warning — don't reject it.

Net: BLAST class owns the *physics*, the chemistry profile owns *policy
defaults*, the user keeps the final say.

## Known integration risks to verify

- **Throughput double-counting.** BLAST rescales `delta_efc`/`Crate` by current
  SoH *internally* (`_extract_stressors` multiplies by `outputs['q'][-1]` to
  convert to nominal-normalized units). BREOS's `soc_absolute` is normalized by
  **current** capacity — `Battery_Energy_Wh / (nominal_energy_wh ×
  battery_soh_decimal)` (`battery.py:407`) — which is exactly the input BLAST's
  internal rescale assumes, so the composition is correct by construction: no
  double-derate. The adapter-parity validation case must still assert this
  invariant (per-day `delta_efc` × nominal ≈ energy actually cycled that day).
- **Time-base continuity** across daily chunks and across yearly
  `simulate_energy_balance` calls — assert cumulative `t_days` is monotonic and
  matches wall-clock.
- **Temperature input granularity.** Native calendar uses daily-mean cell temp;
  BLAST `_extract_stressors` can take the intraday series (it trapz-integrates).
  Decide per-day series vs mean; prefer passing the day's `T_cell` series.

## Testing & validation

- **Vendoring smoke (Phase 0):** every vendored module imports under BREOS's
  `numpy>=2.0` and each model runs one `update_battery_state` chunk — catches
  the upstream `np.trapz` usage (removed in NumPy 2) and any missed heavy
  imports.
- **Regression (gate):** default `native` path reproduces current results
  bit-for-bit on `configs/examples/` — same rule as every PV-capability item.
- **Adapter parity:** a constant 25 °C / fixed-SoC profile through `BlastEngine`
  matches BLAST standalone (`model.simulate_battery_life`) to ≤1e-6 — proves the
  adapter doesn't distort the model.
- **Cross-year continuity:** 1×20yr run == 20×1yr runs threaded through the
  snapshot API (SoH trajectory identical).
- **Per-chemistry smoke:** each enabled key runs a 20-yr sim; SoH monotonic
  non-increasing (except LTO qGain early-life), ends in a plausible band; no NaNs.
- **Replacement:** EoL triggers `reset()` and SoH returns to ~1.0.

## Performance note

The BLAST path does per-day rainflow + trapz; it does **not** use the
`numba_kernels` fast path. Fine for single studies (~7300 daily calls / 20 yr).
For Monte Carlo / NSGA-II inner loops it will be slower — defer a fast mode
(BLAST's own `is_constant_input` repeat-accumulate, or numba) to Phase 4.

Monte Carlo also has its **own** year loop with separate state threading
(`montecarlo.py:182`), which the Phase 1 runner changes do not touch. So
`degradation_engine="blast"` + Monte Carlo is **rejected at validation** until
Phase 4 wires that loop — never silently run as native.

## Licensing / attribution

- Preserve the BSD-3 copyright header (`Alliance for Energy Innovation, LLC`) and
  the DOE-contract `NOTICE` text in every vendored file.
- Add a BLAST-Lite entry to `ATTRIBUTIONS.md` (source, commit/version vendored,
  DOIs of the model papers).

## Phasing

- **Phase 0** — Vendor **all 14** model files + base class + rainflow +
  `rescale_soc` (numpy-only trim; **`np.trapz` → `np.trapezoid`** NumPy-2
  rename); license/NOTICE/ATTRIBUTIONS with the upstream commit pinned. No
  behavior change (`np.trapezoid` is numerically identical).
- **Phase 1** — `BlastEngine` adapter (incl. the daily-grid endpoint
  construction) + minimal App-level `degradation_engine`/`blast_model` config;
  **cross-year state threading + replacement reset** — *required here, not
  deferred*: the runner loops `simulate_energy_balance` once per year
  (`runners/app.py:68`), so without threading every simulated year silently
  resets BLAST. Enable the two simple 2-bucket-power models end-to-end —
  **LFP 250Ah prismatic (flagship)** + **NCA Panasonic (POC)**; default path
  untouched. Adapter-parity, cross-year-continuity, and regression tests.
- **Phase 2** — The **chemistry profile registry** (precedence + raw-key
  merge-order) + full config/CLI plumbing (`--degradation-engine` /
  `--blast-model`; **retire** the legacy `battery_type` selector — see the
  resolved decision below).
- **Phase 3a** — Enable the remaining **power-law** chemistries (LMO 2nd-life,
  NMC811 M50/MJ1, NMC 50Ah B1/B2, NMC 75Ah A, NMC111 Sanyo, NMC-LTO);
  per-chemistry smoke tests + `experimental_range` out-of-bounds warnings.
- **Phase 3b** — Enable the **sigmoid / break-in / multi-mode** chemistries
  (LFP Sony-Murata, NCA-Si Sony-Murata, NMC111 Kokam, NMC622 DENSO) — exercises
  the sigmoid/exp/power-B kernels and multi-output handling.
- **Phase 4 (later)** — **Monte Carlo BLAST support** (thread state through
  `montecarlo.py`'s own year loop — until then `blast` + MC raises);
  resistance-fade mapping for multi-mode models; fast/repeat mode for Monte
  Carlo; ROADMAP + docs update.

## Open decisions

Settle before implementation begins:

- None currently.

Resolved:

- **[DECIDED] Cross-year state carrier uses serialized snapshots.** Thread a
  `state_snapshot()` dict through `simulate_energy_balance`'s return tuple
  instead of passing a live `BlastEngine` through the runner's year loop. This is
  consistent with the existing `initial_fec` / `final` scalar pattern, keeps
  mutable engine objects out of the function signature, and gives Phase 4 Monte
  Carlo a picklable representation. Prototype tests in the BLAST-Lite prep
  branch proved snapshot continuity across all 14 BLAST models, including the
  P3b multi-mode models.
- **[DECIDED] Do not repurpose `battery_type` for BLAST chemistry.** In 0.3.3 it
  became a guarded native-LFP selector instead of a silent no-op for non-LFP
  values. The new `degradation_engine` / `blast_model` keys should select BLAST
  chemistry rather than overloading `battery_type`.

## Non-goals

- Replacing the native Naumann/Lam LFP path as the default.
- Electrochemical / P2D models.
- Re-mapping BLAST parameters into BREOS's equation form.
