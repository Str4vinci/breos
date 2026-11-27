# Study 16 — Result drift across Python and dependency versions

- `build_envs.sh`: 8 venvs under data/envs: uv.lock on Python 3.11/3.12/3.13; the lock with pandas
  2.3.3, pvlib 0.14 or numpy 2.2; lowest-direct (Python 3.11); highest (Python 3.13).
- `run_gallery.py OUT`: every gallery case except montecarlo through
  tools/regenerate_gallery_results.py's run_case. About 20 min per env (2 h 45 min total).
- `compare.py` -> `output/summary.md`, `output/cases.csv`, `output/fields.csv`.
