# BLAST degradation engine

**Status:** Implemented in 0.4.0. All 14 vendored BLAST-Lite cell models are
enabled. Monte Carlo support, a resistance-fade mapping and acceleration are
deferred.

This note records the design of the opt-in BLAST degradation path. The
default, validation, cell-to-pack, resistance and provenance rules are in
[Battery degradation policy](battery-degradation-policy.md). User-facing
model selection is documented in
[Battery degradation models](../../docs/api/degradation-models.md).

## Problem

BREOS's native degradation is calibrated for LFP only. Calendar aging uses the
Naumann/Lam power law (`k0, Ea, b, n`) and cycle aging uses an LFP
Naumann/Wöhler form. Before 0.3.3 a configuration could name a non-LFP pack
and still age it on LFP curves. 0.3.3 made unsupported chemistries raise, and
0.7.0 removed the `BatteryConfig.battery_type` field, which only accepted
`"lfp"`.

NREL's [BLAST-Lite](https://github.com/NREL/BLAST-Lite) (BSD-3) provides 14
lab-calibrated, DOI-cited empirical degradation models (calendar plus cycle)
for NMC111/622/811, NCA, NCA-Si, LMO, LTO, LFP and second-life cells. They are
empirical, not electrochemical, which matches BREOS's scope.

## Decision: a parallel engine, without re-mapping parameters

BLAST's parameterization is structurally different from BREOS's, so
translating BLAST numbers into BREOS's `(k0, Ea, b, n)` and Wöhler form loses
information or is impossible:

| BREOS knob | BLAST equivalent | Maps? |
|---|---|---|
| `k0` calendar rate | `qcal_A` (absorbs T_ref and the percent-to-fraction scale) | Yes, by algebra |
| `Ea` Arrhenius | `qcal_B = −Ea/R` | Yes, by algebra |
| `b` time exponent | `qcal_p` | Yes, same role |
| `n` (`soc^n` stress) | `exp(qcal_C·soc/T)`: exponential and temperature-coupled (or via `Ua`) | No, different function |
| Wöhler/Naumann cycle `(a, b, c, d, z)` | `(qcyc_A…E, qcyc_p)`: temperature-dependent, linear in DOD | No, different model |

BREOS cycle aging does not depend on temperature; BLAST's does. Several models
(NMC111 Kokam, NMC622 DENSO, LFP Sony-Murata, NCA-Si Sony) split capacity loss
into separate LLI, LAM and resistance modes, with a sigmoid LAM knee or an
exponential break-in. BREOS's single power law cannot represent these shapes.
A BLAST parameter means something only together with its own equation and
trajectory kernel.

BREOS therefore vendors BLAST's model classes and runs them as an opt-in
engine. The default native path is unchanged bit for bit.

## Vendoring

The vendored source is in `breos/degradation/blast/`.
`breos/degradation/blast/VENDORED.md` records the pinned upstream commit, the
transformation of each file and the SHA-256 hashes. The upstream `LICENSE` and
`NOTICE` are bundled unchanged.

- BREOS vendors the source and does not depend on the `blast-lite` package.
  The package pulls in `matplotlib` and `pandas`. The vendored subset needs
  NumPy only, plus `scipy.stats` for the Sony-Murata LFP model, and SciPy is
  already a core dependency.
- Upstream calls `np.trapz`, which NumPy 2 removed. The vendored files use
  `np.trapezoid`, which gives the same values.
- Only `rescale_soc` is extracted from upstream `blast/utils/functions.py`. The
  full file imports NSRDB and geocoding helpers that BREOS does not need.

## Engine and lifecycle

`breos/degradation/engine.py` defines `BlastEngine`. It wraps one model
instance and has the methods `step`, `soh`, `reset`, `state_snapshot` and
`from_snapshot`. `breos/degradation/protocol.py` puts the native and BLAST
engines behind one internal lifecycle contract (`DegradationLifecycle`), so
the energy-balance loop calls both the same way.

- **Daily cadence, with feedback.** Degradation runs once per positional
  degradation day, inside the simulation. Usable capacity shrinks as state
  of health falls, and dispatch sees that change. Running BLAST once over a
  finished series would remove this feedback, so BREOS does not do it.
- **A full-day time grid.** BLAST derives elapsed time from the chunk
  (`t_days[-1] - t_days[0]`). Hourly post-step samples would cover only 23
  hours and undercount calendar aging by about 4 % a day. The adapter
  prepends the day's start anchor (the previous day's last SOC and
  temperature), so a day has `steps_per_day + 1` points and lasts exactly one
  day. Consecutive days share their boundary point, so BLAST's
  `sum(|diff(soc)|)/2` counts every SOC segment once.
- **Temperature.** BLAST receives the intraday cell-temperature series on the
  same grid, not a daily mean.
- **Throughput normalization.** BREOS's absolute SOC is stored energy over
  current capacity. BLAST rescales its stressors by its own current `q`
  internally, which expects exactly that input, so throughput is not derated
  twice.
- **Cross-year state.** The runner calls the energy balance once per project
  year. The engine state passes between years as a serialized snapshot, not
  as a live object. This follows the existing scalar-carry pattern, keeps
  mutable engines out of the function signature and gives a picklable
  payload. The snapshot also carries the end-of-year SOC and temperature
  anchors, so a split run and a continuous run agree.
- **Replacement.** A replacement resets the engine to a fresh model instance.
- **Resistance.** BLAST resistance outputs are diagnostic only.
  `enable_resistance_fade` together with BLAST is refused. See the
  [resistance policy](battery-degradation-policy.md#resistance-and-screening-paths).

## Configuration

`degradation_engine = "native"` (the default) or `"blast"`, with a named
`blast_model`. The App, the CLI (`--degradation-engine`, `--blast-model`) and
the projected optimizer accept them. Validation refuses an unknown model, a
`blast_model` without the BLAST engine, BLAST without a battery, and BLAST
together with `enable_resistance_fade`. Monte Carlo refuses BLAST: its
trajectory loop does not yet carry the BLAST state. BREOS never falls back to
native degradation silently. The `battery_type` key was never reused to
select a BLAST chemistry.

## Cell-model profile registry

`breos/degradation/profiles.py` is the single declarative registry for model
identity, citations, experimental ranges, output capabilities and engine-class
lookup. The Python and CLI discovery commands read it. Profile data has three
tiers:

1. **Degradation parameters** (`qcal_*`, `qcyc_*`) are part of the vendored
   class and are never user-tunable.
2. **Experimental ranges** (for example temperature, DOD and C-rate) come
   from the model and drive warnings. They never change the user's settings.
3. **Operating-envelope defaults** (round-trip efficiency, SOC window, EOL
   threshold, power limits) are empty for every profile, because the sources
   do not give generic pack defaults.

Settings therefore resolve as explicit user configuration over the global App
defaults. Costs stay in the cost presets. Several models (Panasonic, the
Sony-Murata cylindrical cells, the NMC fast-charge pouches) are EV or
high-power cells that were tested well above stationary C-rates. They run,
and the out-of-range warning flags them.

## Tests

- Every vendored module imports under NumPy 2, and every model runs one
  update.
- The default native path stays bit for bit identical (App golden fixture).
- Adapter parity: a constant-temperature, fixed-SOC profile through
  `BlastEngine` matches standalone BLAST.
- Cross-year continuity: one multi-year run equals the same years threaded
  through snapshots.
- Every enabled model runs a 20-year simulation with no NaNs, and its state
  of health does not rise (except for the LTO early-life capacity gain).
- A replacement resets state of health.
- Parity fixtures come from the pinned, unmodified upstream source (see the
  [policy](battery-degradation-policy.md#provenance-legal-records-and-fixtures)).

## Deferred work

- **Monte Carlo.** Thread the BLAST snapshot through the Monte Carlo
  trajectory loop. Until then the combination is refused.
- **Resistance mapping.** Couple a model's resistance output to dispatch only
  after its metric and test protocol have a validated mapping to pack
  behavior.
- **Performance.** The BLAST path does per-day rainflow counting and
  trapezoidal integration in Python. This is acceptable for one study (about
  7,300 daily calls over 20 years). Acceleration for large Monte Carlo or
  optimization loops is not planned yet.

## Non-goals

- Replacing native Naumann/Lam LFP degradation as the default.
- Electrochemical or P2D models.
- Re-mapping BLAST parameters into BREOS's equation form.
