# Long-run studies 11-16 — plan and progress

Requested 2026-11-26 ("go for them"). Worktree ../breos-long-studies-2, branch docs/0.9.0-long-studies-2
from develop 82c9e6c (0.8.1 merged). Commit locally only. data/ and output/ are gitignored.
Shared helpers: _shared/sizing.py (sites, SARAH3 2005-2023 years + PVGIS TMY of long-run study 3,
household, design study). Heavy runs: systemd user services with MemoryMax=16G.

| # | Study | Runner | Status | Headline |
|---|---|---|---|---|
| 11 | Sizing on TMY vs actual years (regret) | 11_sizing_regret/run_grid.py | done | NPV-best size robust (≤14 €); targets missed in 30–50 % of years by ~1 pp; >1 k€ regrets (5 %) are replacement cliffs, min_remaining 2 y cuts them to 1 % |
| 12 | Order of weather years in a projection | 12_year_order/run_order.py (queue) | done | 305 orders of the same 19 years: NPV range 50–140 € (σ≈10 €), replacement count never changes; only outlier is a TMY-vs-sequence replacement cliff (1.2 k€) |
| 13 | Replacement-rule family vs default (#403) | 13_replacement_policy/run_policy.py (queue) | done | best rule in all 126 cases buys no replacement: +0.7–9.7 k€ NPV vs default (v1 5 kWh +2.9 k€), −2–8 pp self-sufficiency; retire never beats keep |
| 14 | NSGA-II with a third objective (lookup replays) | 14_three_objectives/{enumerate_space,run_3obj}.py (queue) | done | 3 objectives at shipped budget: returned front HV 0.984 (2 obj: 0.996), exact 3D fronts 85–252 pts vs pop 40; all_evaluated restores 0.998 at 820 sims |
| 15 | Peak memory and wall time across configs | 15_resources/run_resources.py (queue) | done | all 104 runs ≤475 MB, ≤7 s (MC 0.32 s/run); memory set by resolution only (~290 MB h, ~380 MB 15-min); external-rlp fails by design |
| 16 | Result drift across Python / dependency versions | 16_version_drift/{build_envs.sh,run_gallery.py} (queue) | done | py3.11–3.13 bit-identical on the lock; pandas 2.3.3, pvlib 0.14 identical; numpy 2.2/2.0 move values ≤1.55 € and reroute NSGA-II (front HV 0.998) |

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
- 2026-11-27: study 14 enumeration failed on the old SolarDesignProblem API (pv_params/budget_limit moved to .study/.config); the replay Pool then respawned failing workers 19:35-02:00. Fixed, guarded, queue restarted 02:04.
- 2026-11-27: study 15 failed in 14 s (TOML start_date not JSON serializable); fixed with default=str, rerun as service breos-s15 after the queue ends.
- 2026-11-27: spot check on develop bd6c74b (after #517-#519, GHI decomposition, four-period tariffs): study 11 and 13 scores bit-identical.
