# Study 15 — Peak memory and wall time across configs

- `run_resources.py`: 110 specs, each in a fresh process (`child.py`), 3 at a time: every App example
  in configs/examples at hourly and 15-min over 1, 10 and 25 years, the community example (15-min),
  Monte Carlo with 10/100/1000 runs (15-min: 10/100), and 10/100/1000 modules over 1 and 20 years.
  About 7 min.
- `analyze.py` -> `output/summary.md`, `output/results.csv`.
