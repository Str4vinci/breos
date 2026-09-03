# Data availability and citation scope

Generated results are not committed. `results/` is ignored, and the
deterministic bundle alone is 863 MB. The bundles behind the forthcoming
publication are deposited at <!-- TODO: Zenodo DOI and record URL --> and that
deposit is the authoritative copy.

## Which version produced which result

The results were not all produced by one release. Cite the version that
generated each one.

| Result | Deposited as | BREOS version | Commit |
| --- | --- | --- | --- |
| Deterministic C1-C5 tables and the seeded Monte Carlo study | `publication-1c` | 0.6.2 | `b3d0034` |
| Battery-cost and degradation cross-sensitivity | `cost-degradation-cross` | 0.6.2 | `b3d0034` |
| Task 3 end-of-life sweep | `task3-eol-0.6.2` | 0.6.2 | `30c35cb` |
| Figure 2 gate, Task 2 Porto, the Task 4 lattice and NSGA-II validation, Tasks 5, 5b, 5f, 6 and 7, and the figures | see the 0.6.1 deposit | 0.6.1 | — |

Splitting the citation this way is what the provenance files record, not a
convenience. Every `reproduction.json` and `provenance.json` inside the three
0.6.2 bundles records the commit named above with
`"tracked_worktree_dirty": false`, alongside the resolved configuration, the
input hashes, and the Python, NumPy, pandas, pvlib and SciPy versions used.

BREOS 0.6.2 changes no model behaviour relative to 0.6.1. Regenerating
`validation/baselines/breos_baseline.json` on 0.6.2 changed only the recorded
version string; every validation figure is unchanged.

## What the deposit contains

<!-- TODO: state whether the per-year trajectories are included. -->

Each Monte Carlo case directory holds:

- `summary.json` — the case's aggregate metrics.
- `runs.csv` — one row per trajectory: 10,000 rows, 18 columns.
- `yearly.csv` — one row per trajectory-year: 200,000 rows, 65 columns, holding
  the energy, degradation-state and discounted-cost paths behind the summaries.

The five `yearly.csv` files account for 850 MB of the 863 MB total; the five
`runs.csv` files account for 13 MB.

## Reproducing a bundle

Check out the commit named in the table above, then follow the
[source-data runbook](runbook.md). `tools/verify_article1_bundle.py` checks a
generated bundle against its recorded provenance.
