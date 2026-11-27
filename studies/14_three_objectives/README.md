# Study 14 — NSGA-II with a third objective (lookup replays)

Ported from ~/pythonprojects/nsga_tou_study (common.py, metrics.py, enumerate_space.py,
nsga_runs.py), pointed at this worktree and updated for the current optimizer API.

- `enumerate_space.py SCENARIO SPACE`: exact table of the az180 space (modules x battery x tilt,
  6,820 designs) with CO2 and CAPEX kept. 5 cases (Porto, Berlin, Colorado, high load, cheap
  battery) x 2 tariffs (flat, bi-hourly greedy). About 2 min each on 10 workers.
- `run_3obj.py SPACE SCENARIOS`: the optimizer's NSGA-II (operators, repair, termination from
  breos.optimization) replayed on the tables with 2 objectives (as shipped) and with a third
  (-CO2 avoided, or CAPEX). Budgets prod (40/20/40 + ftol stop), prod_noes, long (200 gen),
  big (pop 100, 200 gen); 50 seeds each. 6,000 runs, about 13 min.
- `analyze_3obj.py` -> `output/summary.md`, `output/replays.csv`.
