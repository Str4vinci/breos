"""Study 11: score every design on the grid for every (site, weather, variant).

Weather is the PVGIS TMY or one actual SARAH3 year 2005-2023, repeated over the 20-year
projection (as a user running BREOS on that year would get). The design grid is
sizing.MODULES x sizing.BATTERY_KWH at a fixed south-facing tilt. One job = one
(site, weather, variant) and writes data/grid/<site>_<weather>_<variant>.pkl, so the
run resumes after an interruption.

Usage: uv run python run_grid.py [--workers N]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _shared import sizing as S  # noqa: E402

OUT = HERE / "data" / "grid"
VARIANTS = {
    "default": {},
    "min_remaining_2y": {"battery": {"replacement_min_remaining_years": 2.0}},
    "v2": {"battery": {"calendar_model": "naumann_lam_field_calibrated_v2"}},
}
WEATHERS = ["tmy", *S.YEARS]


def job(args) -> str:
    site, wkey, variant = args
    path = OUT / f"{site}_{wkey}_{variant}.pkl"
    if path.exists():
        return f"skip {path.name}"
    warnings.simplefilter("ignore")
    t = time.time()
    study = S.design_study(site, S.site_config(site, VARIANTS[variant]), S.weather(site, wkey))
    rows = []
    for m in S.MODULES:
        for b in S.BATTERY_KWH:
            r = S.score(study, m, b)
            rows.append({"site": site, "weather": str(wkey), "variant": variant, "modules": m, "battery_kwh": b, **r})
    tmp = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_pickle(tmp)
    tmp.rename(path)
    return f"done {path.name} {time.time() - t:.0f}s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(s, w, v) for v in VARIANTS for s in S.SITES for w in WEATHERS]
    print(f"{len(jobs)} jobs, {len(S.MODULES) * len(S.BATTERY_KWH)} designs each", flush=True)
    with mp.Pool(a.workers, maxtasksperchild=4) as pool:
        for i, msg in enumerate(pool.imap_unordered(job, jobs), 1):
            print(f"[{i}/{len(jobs)}] {time.strftime('%H:%M:%S')} {msg}", flush=True)


if __name__ == "__main__":
    main()
