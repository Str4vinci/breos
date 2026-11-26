# Study 11 — Sizing on a TMY vs actual years

Question: a user sizes PV and battery on one PVGIS TMY. How much does that choice cost in the
years that actually happen?

- `run_grid.py`: 7 sites x (TMY + 19 SARAH3 years 2005-2023) x 3 battery variants x 304 designs
  (2-20 modules x 0-15 kWh, south, tilt 30), scored through the optimizer's own design study
  (`_shared/sizing.py`). Each weather year is repeated over the 20-year projection. 127,680 scores,
  about 1 h 45 min on 10 workers.
- `analyze.py`: decisions made on the TMY and scored in every actual year; robust (best-mean) picks;
  the TMY front scored against each year's own front. Output in `output/` (gitignored).
