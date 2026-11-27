# Study 13 — Battery replacement rule (evidence for #403)

- `run_policy.py`: 7 sites x 3 ageing models (field v1 default, field v2, BLAST LMO Nissan Leaf
  2nd life) x 3 cost cases (base 500 EUR/kWh, 300 EUR/kWh, 5 %/yr replacement-price learning) x
  92 replacement policies (eol 0.50-0.80 x min remaining 0-7 y x keep/retire, plus never) x
  5 and 10 kWh, 10 modules, TMY weather, 20 years. About 30 min on 10 workers.
- `analyze.py` -> `output/summary.md`, `output/best.csv`.
