# Study 15 — Peak memory and wall time across the config space

**Question.** One run once used up this machine's 23 GB (a 1-second detailed frame, long-run study
2). Which ordinary settings make BREOS slow or memory-hungry?

**Answer.** None of the shipped configs. Over 104 successful runs, peak memory was 245–475 MB per
process and no simulation took more than 7 s, except Monte Carlo (0.32 s per trajectory). Memory
depends almost only on resolution (about 290 MB hourly, 380 MB at 15-min) and hardly at all on
horizon, module count or Monte Carlo size. The known costly settings remain the ones measured
elsewhere: 1-second data (study 2) and per-module shading (run C).

## Setup

Each spec ran in its own process with `ru_maxrss` recorded, three in parallel on the i7-8700. Weather
was the local PVGIS TMY for preset locations. Numba's on-disk cache was warm. A cold first run in a
fresh environment adds about 7 s of compilation (seen in the smoke test).

## Results

| | Peak RSS | Simulate time |
|---|---|---|
| Examples, hourly, 1–25 years | 245–315 MB | 0.2–5.4 s |
| Examples, 15-min, 1–25 years | 336–462 MB | 0.6–7.1 s |
| Community (ACC), 15-min, 1–25 years | 402–473 MB | 11–19 s |
| 10 → 1,000 modules, hourly / 15-min | 291–293 / 379–383 MB | 0.3–1.3 s |
| Monte Carlo 10 / 100 / 1,000 runs, hourly | 280 / 286 / 287 MB | 5 / 36 / 316 s |
| Monte Carlo 10 / 100 runs, 15-min | 333 / 357 MB | 9 / 57 s |

- Import of `breos` takes about 0.9 s and 145 MB of the baseline.
- The slowest examples are those with per-step decisions: vehicle-to-home (7.1 s at 15-min over
  25 years) and smart charging (4.7 s). Their time grows with the horizon; the others hardly do.
- The detailed timeseries is 6 MB hourly and 25 MB at 15-min, for every horizon (one year is kept).
- Monte Carlo memory doesn't grow with the number of runs: per-run results are summarised, not kept.
- The external-RLP example failed by design in all six runs: it needs an E-REDES file BREOS doesn't
  ship, and says so in its error.

## Takeaways for BREOS

1. Nothing in the normal config space needs a memory warning. A warning belongs only on sub-minute
   resolution with the detailed frame.
2. A cold numba compile (about 7 s) is longer than most simulations. That's worth mentioning in the
   performance docs, since a CLI user pays it on first use.
