# Long-run studies 11-16 — plan and progress

Requested 2026-11-26 ("go for them"). Worktree ../breos-long-studies-2, branch docs/0.9.0-long-studies-2
from develop 82c9e6c (0.8.1 merged). Commit locally only. data/ and output/ are gitignored.
Shared helpers: _shared/sizing.py (sites, SARAH3 2005-2023 years + PVGIS TMY of long-run study 3,
household, design study). Heavy runs: systemd user services with MemoryMax=16G.

| # | Study | Runner | Status | Headline |
|---|---|---|---|---|
| 11 | Sizing on TMY vs actual years (regret) | 11_sizing_regret/run_grid.py | done | NPV-best size robust (≤14 €); targets missed in 30–50 % of years by ~1 pp; >1 k€ regrets (5 %) are replacement cliffs, min_remaining 2 y cuts them to 1 % |
| 12 | Order of weather years in a projection | 12_year_order/run_order.py (queue) | running | |
| 13 | Replacement-rule family vs default (#403) | 13_replacement_policy/run_policy.py (queue) | queued | |
| 14 | NSGA-II with a third objective (lookup replays) | 14_three_objectives/{enumerate_space,run_3obj}.py (queue) | queued | |
| 15 | Peak memory and wall time across configs | 15_resources/run_resources.py (queue) | queued | |
| 16 | Result drift across Python / dependency versions | 16_version_drift/{build_envs.sh,run_gallery.py} (queue) | queued | |

Queue: studies/queue.sh as service breos-s12-16-queue (waits for breos-s11-grid), log studies/queue.log.
Check: `systemctl --user status breos-s11-grid breos-s12-16-queue`; `tail studies/queue.log`.

## Notes log

- 2026-11-26: the optimizer has two objectives only (n_obj=2), so study 14 asks whether a third would
  work by replaying NSGA-II on exact lookup tables (az180 space, 6,820 designs) with CO2 avoided or CAPEX added.
- The optimizer ignores [terminal_value]; study 11/13 use replacement_min_remaining_years instead.
- BLAST NMC 50Ah b1 never reaches EOL at 25 C in Porto; study 13 uses the LMO (Nissan Leaf 2nd life) cell.
- Smoke: Porto v1, 10 modules + 5 kWh: never replacing beats the default rule by ~2.9 k EUR (projected NPV, no terminal value).
- Study 15 child processes include the numba compile in simulate_s (a fresh process, as a CLI user sees it).
- Study 16 envs: lock py3.11/3.12/3.13, pandas 2.3.3, pvlib 0.14, numpy 2.2, lowest-direct py3.11, highest py3.13.
  lock-py313 reproduces the stored first_home result.json bit for bit.
