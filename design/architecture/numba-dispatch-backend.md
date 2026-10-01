# Numba dispatch backend

**Status:** Implemented. Added in 0.6.0 for Monte Carlo, App and projected
optimization. In 0.7.0 the compiled kernel became the Python day loop itself
and gained the smart-charging instruction arrays.

BREOS keeps the Python battery simulation as its numerical reference. Repeated
Monte Carlo and optimization runs can select an optional Numba backend for the
within-day dispatch loop, which is the measured simulation bottleneck. The
backend is an execution choice, not a second battery model.

## Boundary

The Numba backend compiles `breos._dispatch._dispatch_day`, the function the
Python backend runs, and the scalar helpers it calls. It does not compile a
copy. Both backends pack their arguments through `_day_arguments`, so the
per-step dispatch instructions (ADR 0002) and the PV, grid and unattributed
origin accounting are one source for both. Python owns every state
transition between days and years:

- rainflow counting;
- calendar and cycle degradation;
- resistance growth;
- replacement decisions; and
- the stored energy and origin balances carried across days and years.

The Python backend is the default. Selecting `numba`, through the App and
Monte Carlo `execution_backend` key or the optimizer's `execution_backend`
argument, validates the optional dependency before a simulation or
trajectory begins.
`breos[fast]` installs it. The `breos` import does not import Numba.
`breos/_numba_dispatch_kernels.py` is imported only when the compiled
backend is selected.

The removed `breos.numba_kernels` module (deleted in 0.6.0) compiled whole
projects, including degradation and replacement. It is not coming back:
it duplicated science that must stay in one place.

## Numerical contract

The target is bit-identical output from the Python and Numba backends on the
same machine and toolchain. BREOS does not promise bit identity across Python,
NumPy, Numba, LLVM, operating-system, or platform-libm versions, so provenance
records the selected backend and the Numba and llvmlite versions.

The kernel is compiled with `fastmath=False` and without automatic
parallelization, so LLVM keeps the Python operation order. One non-obvious
safeguard is the PVWatts part-load polynomial: CPython evaluates `zeta ** 2`
through libm, while LLVM can fold a constant square to `zeta * zeta` and
change the result by one ULP. The kernel receives the exponent as a runtime
value to keep the reference libm operation. A focused test pins an input
where the two forms differ.

Parity coverage compares:

- detailed and summary-only simulation;
- Python and Numba timestep and annual ledgers;
- serial and pooled Monte Carlo bundles;
- App-assembled outputs across battery, PV-only, constrained-power, and
  15-minute scenarios;
- smart-charging instruction scenarios (`tools/parity/harness.py --instructions`);
- replacement timing and carried energy state; and
- temperatures across the LFP derating thresholds and indoor-model bounds.

Only backend-identifying provenance fields are excluded from numerical
comparison. The harness also verifies that the two runs actually selected
different backends.

## Summary-only simulation

Monte Carlo and projected optimization do not need a per-timestep data frame.
Their summary path returns annual energy totals, grid independence, degradation
state, replacements, carried energy, and temperature diagnostics directly. The
detailed path remains available for ordinary App results and parity checks.

The summary path also exposes separate loss and replacement ledgers so a parity
failure can be localized without weakening the public energy-balance contract.

## Compilation cache and provenance

The day loop and its helpers are defined at module scope in
`breos/_dispatch.py`. Numba's cache key for a function defined in a factory
closure includes the closure cell contents. These are not stable across
processes, so such a function would recompile and add a cache entry in every
new process. With module-level functions, later processes and pooled workers
reuse one on-disk artifact. Numba freezes module globals at compile time: after
editing a constant imported into `breos/_dispatch.py`, clear the cache or the
compiled backend keeps the old value. The parity tests detect this.

Provenance records an observed JIT cache state of `cold`, `warm`, or `unknown`.
Cache observation is diagnostic metadata and cannot abort an otherwise valid
study. Cross-process tests cover cold-to-warm reuse and guard the Numba-internal
counters used for this observation.

## Performance scope

A representative 20-year Monte Carlo benchmark measured about a 9.8x
trajectory speedup with both one and ten workers. This is an order-of-magnitude
measurement from one machine and configuration, not a performance guarantee.
Use `tools/benchmark_montecarlo.py` and `tools/benchmark_optimization.py` to
measure the workload and hardware that matter to a deployment.

Cold compilation is visible for a single App run but is amortized across a
large study and cached for later processes. Once dispatch is compiled, most
remaining trajectory time is Python-side degradation work. Extending compilation
across that boundary would duplicate scientifically sensitive model logic and is
outside this design.

Parallelism has one owner. Use the serial day kernel inside process-parallel
candidate or trajectory evaluation; do not add a parallel kernel on top and
oversubscribe the CPUs.

## Repository validation material

`tools/parity/` contains development-only comparison commands for the App,
Monte Carlo, detailed-versus-summary simulation and instruction scenarios.
They do not add generated results, licensed inputs, or another runtime
implementation to the package.
