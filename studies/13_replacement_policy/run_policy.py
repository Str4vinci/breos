"""Study 13: how much NPV does the end-of-life replacement rule leave on the table? (evidence for #403)

BREOS replaces a battery when its health crosses eol_percentage, optionally skipping swaps
with fewer than replacement_min_remaining_years left (then keeping or retiring the pack).
#403 proposes an economic rule instead. This study brute-forces the rule family that exists:
eol threshold x min remaining years x skipped action, plus never replacing, for each site,
degradation model, battery size and cost case, on the TMY. The best policy found is a lower
bound on what an economic rule could gain over the default (0.70, 0 years, keep).

One job = (site, model, cost case, policy) scoring BATTERIES; output data/policy/<site>_<model>.pkl.

Usage: uv run python run_policy.py [--workers N]
"""

from __future__ import annotations

import argparse
import itertools
import multiprocessing as mp
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _shared import sizing as S  # noqa: E402

OUT = HERE / "data" / "policy"
MODULES = 10
BATTERIES = (5.0, 10.0)
MODELS = {
    "v1": {},
    "v2": {"battery": {"calendar_model": "naumann_lam_field_calibrated_v2"}},
    "blast_lmo": {"battery": {"degradation_engine": "blast", "blast_model": "lmo_gr_nissanleaf_66ah_2nd"}},
}
COSTS = {
    "base": {},
    "cheap_storage": {"costs": {"storage_cost_per_kwh": 300.0}},
    "learning_5pct": {"financials": {"replacement_cost_learning": 0.05}},
}
EOL = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
MIN_REMAINING = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0)
ACTIONS = ("keep", "retire")


def policies() -> dict[str, dict]:
    out = {"never": {"battery": {"enable_replacement": False}}}
    for eol, rem, act in itertools.product(EOL, MIN_REMAINING, ACTIONS):
        if rem == 0.0 and act == "retire":
            continue  # nothing is skipped, so the action never applies
        out[f"eol{eol:.2f}_rem{rem:g}_{act}"] = {
            "battery": {"eol_percentage": eol, "replacement_min_remaining_years": rem, "skipped_replacement_action": act}
        }
    return out


POLICIES = policies()


def job(args):
    site, model, cost, policy = args
    warnings.simplefilter("ignore")
    over = S.merge(S.merge(S.merge({}, MODELS[model]), COSTS[cost]), POLICIES[policy])
    st = S.design_study(site, S.site_config(site, over), S.tmy(site))
    rows = []
    for b in BATTERIES:
        r = S.score(st, MODULES, b)
        rows.append({"site": site, "model": model, "cost": cost, "policy": policy, "battery_kwh": b, **r})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for site in S.SITES:
        for model in MODELS:
            path = OUT / f"{site}_{model}.pkl"
            if path.exists():
                continue
            t = time.time()
            jobs = [(site, model, c, p) for c in COSTS for p in POLICIES]
            with mp.Pool(a.workers, maxtasksperchild=50) as pool:
                rows = [r for chunk in pool.imap_unordered(job, jobs, chunksize=4) for r in chunk]
            pd.DataFrame(rows).to_pickle(path)
            print(f"{time.strftime('%H:%M:%S')} {site} {model}: {len(jobs)} jobs in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
