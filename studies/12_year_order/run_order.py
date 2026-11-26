"""Study 12: does the order of weather years change a projection's result?

Every sequence holds the same 19 actual SARAH3 years (2005-2023), one per project year, so
the horizon is 19 years. Orders: chronological, reversed, worst-first and best-first by annual
plane-of-array DC yield of the reference design, and N_RANDOM seeded shuffles. Any spread
between them comes from the order alone: battery ageing is path dependent, and discounting
weighs early years more. Reference rows: the TMY repeated 19 times.

One job = (site, order, variant) scoring DESIGNS; output data/order/<site>_<variant>.pkl per
(site, variant) after all its orders finish (job results are collected in the parent).

Usage: uv run python run_order.py [--workers N] [--random N]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _shared import sizing as S  # noqa: E402

OUT = HERE / "data" / "order"
HORIZON = len(S.YEARS)
DESIGNS = [(10, 0.0), (10, 5.0), (10, 10.0), (6, 5.0), (16, 10.0)]
VARIANTS = {
    "default": {},
    "v2": {"battery": {"calendar_model": "naumann_lam_field_calibrated_v2"}},
}


def cfg(site: str, variant: str) -> dict:
    return S.site_config(site, {**VARIANTS[variant], "simulation": {"years_projection": HORIZON}})


def year_rank(site: str) -> list[int]:
    """Years sorted by the reference design's year-1 DC yield (worst first)."""
    yields = {}
    for y in S.YEARS:
        st = S.design_study(site, cfg(site, "default"), S.weather(site, y))
        yields[y] = S.score(st, 10, 0.0)["Projected_PV_DC_Year1_kWh"]
    return sorted(S.YEARS, key=yields.get), yields


def orders(site: str, n_random: int) -> dict[str, list]:
    worst_first, _ = year_rank(site)
    out = {
        "chronological": list(S.YEARS),
        "reversed": list(S.YEARS)[::-1],
        "worst_first": worst_first,
        "best_first": worst_first[::-1],
        "tmy_repeated": None,
    }
    rng = np.random.default_rng(12)
    for i in range(n_random):
        out[f"random_{i:03d}"] = [int(y) for y in rng.permutation(S.YEARS)]
    return out


def job(args):
    site, name, seq, variant = args
    warnings.simplefilter("ignore")
    frames = None if seq is None else [S.weather(site, y) for y in seq]
    st = S.design_study(site, cfg(site, variant), S.tmy(site), weather_by_year=frames)
    rows = []
    for m, b in DESIGNS:
        r = S.score(st, m, b, tables=True)
        yearly = r.pop("_yearly_summary_df")
        for k in [k for k in r if k.startswith("_")]:
            r.pop(k)
        soh = yearly.filter(like="SOH").iloc[:, 0].tolist() if len(yearly.filter(like="SOH").columns) else []
        rows.append({
            "site": site, "order": name, "variant": variant, "sequence": "" if seq is None else ",".join(map(str, seq)),
            "modules": m, "battery_kwh": b, "soh_by_year": soh, **r,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--random", type=int, default=300)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    warnings.simplefilter("ignore")
    for site in S.SITES:
        todo = [v for v in VARIANTS if not (OUT / f"{site}_{v}.pkl").exists()]
        if not todo:
            continue
        t = time.time()
        ords = orders(site, a.random)
        jobs = [(site, n, s, v) for v in todo for n, s in ords.items()]
        with mp.Pool(a.workers, maxtasksperchild=20) as pool:
            rows = [r for chunk in pool.imap_unordered(job, jobs, chunksize=4) for r in chunk]
        df = pd.DataFrame(rows)
        for v in todo:
            df[df.variant == v].to_pickle(OUT / f"{site}_{v}.pkl")
        print(f"{time.strftime('%H:%M:%S')} {site}: {len(jobs)} sequences in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
