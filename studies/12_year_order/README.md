# Study 12 — Order of weather years in a projection

- `run_order.py`: 7 sites x 2 calendar models (v1 default, v2) x 305 orders of the same 19 SARAH3
  years (2005-2023), each a 19-year `weather_by_year` sequence: chronological, reversed, worst-first,
  best-first (by year-1 DC yield), 300 seeded shuffles, plus the TMY repeated 19 times. Five designs
  (10 modules x 0/5/10 kWh, 6 x 5 kWh, 16 x 10 kWh). About 40 min on 10 workers.
- `analyze.py` -> `output/summary.md`, `output/orders.csv`.
