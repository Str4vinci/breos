#!/usr/bin/env python3
"""Task 6(a): the exact bootstrap distribution of the v1 field calibration.

The v1 calendar model fits four parameters -- k0_frac, Ea, cal_b, n -- to five
residential LFP systems carrying seventeen capacity-test points between them.
Reviewer question 6 asks how much of the field-versus-laboratory design
difference survives that calibration's uncertainty.

Resampling five systems with replacement has only C(9,5) = 126 distinct
multisets, so drawing a thousand resamples would be a thousand samples of one
hundred and twenty-six refits. This enumerates all 126 and carries each one's
multinomial weight instead, which is the complete bootstrap distribution rather
than a sample of it, at an eighth of the cost and with no Monte Carlo error.
Differential evolution is seeded, so the spread reported here is resampling
variance alone.

The calibration pipeline and its dataset live in the research repository, not
in breos, so this must run under that repository's interpreter:

    ~/code/phd/dev/.venv/bin/python tools/revision/task6_bootstrap_calibration.py ...

Stage two (`task6_replay_candidates.py`) reads this stage's CSV under the breos
interpreter and replays C2, C3 and C4 for every parameter set.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import shlex
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

LFP_SYSTEMS = (14, 15, 17, 20, 21)
PARAM_NAMES = ("k0_frac", "Ea", "cal_b", "n")


def _git_commit(repo: Path) -> dict:
    """Record a repository's HEAD and whether its tracked tree is dirty."""
    try:
        commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True
            ).strip()
        )
        return {"commit": commit, "tracked_worktree_dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "tracked_worktree_dirty": None}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def multiset_weight(counts: tuple[int, ...], n: int) -> float:
    """Probability of drawing this multiset when resampling n systems.

    The multinomial coefficient over n^n equally likely ordered draws. The 126
    weights sum to one, so weighted quantiles over them are exact.
    """
    coefficient = math.factorial(n)
    for c in counts:
        coefficient //= math.factorial(c)
    return coefficient / (n**n)


def weighted_quantile(values, weights, q: float) -> float:
    """Quantile of a weighted empirical distribution, by cumulative weight."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    cumulative = 0.0
    total = sum(weights)
    for i in order:
        cumulative += weights[i] / total
        if cumulative >= q:
            return float(values[i])
    return float(values[order[-1]])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--research-repo", type=Path, default=Path.home() / "code/phd/dev")
    ap.add_argument("--data-dir", type=Path, required=True, help="the 12091223 dataset root")
    ap.add_argument("--cache-dir", type=Path, required=True, help="15-minute parquet cache")
    ap.add_argument("--resolution", default="15min")
    ap.add_argument("--calibration-variant", default="current", help="'current' is the v1 fit")
    ap.add_argument("--optimizer-mode", default="final")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(args.research_repo))
    from tools.validate_degradation import (  # noqa: E402
        calibrate_field_parameters,
        load_zenodo_home_dataset,
    )

    print(f"Loading {len(LFP_SYSTEMS)} systems from cache...", flush=True)
    by_id = {
        sid: load_zenodo_home_dataset(str(args.data_dir), str(sid), str(args.cache_dir), resolution=args.resolution)
        for sid in LFP_SYSTEMS
    }
    truth_counts = {sid: len(by_id[sid]["soh_ground_truth"]) for sid in LFP_SYSTEMS}
    print(f"SOH points per system: {truth_counts}; total {sum(truth_counts.values())}", flush=True)

    multisets = sorted(set(itertools.combinations_with_replacement(LFP_SYSTEMS, len(LFP_SYSTEMS))))
    print(f"{len(multisets)} distinct resamples; weights sum to 1 by construction", flush=True)

    rows = []
    start = time.time()
    for index, multiset in enumerate(multisets, start=1):
        counts = Counter(multiset)
        # A duplicated system appears twice in the list the objective averages
        # over, which is exactly the weight a bootstrap resample gives it.
        systems_data = [by_id[sid] for sid in multiset]
        fit = calibrate_field_parameters(
            systems_data,
            calibration_variant=args.calibration_variant,
            optimizer_mode=args.optimizer_mode,
            optimizer_workers=1,
        )
        params = fit["params"]
        pinned = sorted(fit.get("bound_proximity") or {})
        row = {
            "multiset_id": index,
            "systems": "-".join(str(s) for s in multiset),
            "n_distinct": len(counts),
            "n_soh_points": sum(truth_counts[s] for s in multiset),
            "weight": multiset_weight(tuple(counts[s] for s in LFP_SYSTEMS), len(LFP_SYSTEMS)),
            **{f"n_{sid}": counts[sid] for sid in LFP_SYSTEMS},
            **{name: float(params[name]) for name in PARAM_NAMES},
            "mean_RMSE": float(fit["aggregate_metrics"]["mean_RMSE"]),
            "max_RMSE": float(fit["aggregate_metrics"]["max_RMSE"]),
            "n_bound_pinned": len(pinned),
            "bound_pinned": ",".join(pinned),
        }
        rows.append(row)
        if index % 10 == 0 or index == len(multisets):
            rate = (time.time() - start) / index
            print(
                f"  [{index}/{len(multisets)}] {rate:.1f}s/fit, "
                f"{rate * (len(multisets) - index) / 60:.1f} min remaining",
                flush=True,
            )

    table = pd.DataFrame(rows)
    assert abs(table["weight"].sum() - 1.0) < 1e-12, table["weight"].sum()

    args.output.mkdir(parents=True, exist_ok=True)
    params_path = args.output / "task6_bootstrap_params.csv"
    table.to_csv(params_path, index=False)

    # Weighted confidence intervals over the complete distribution.
    ci_rows = []
    for name in PARAM_NAMES:
        values, weights = table[name].tolist(), table["weight"].tolist()
        point = weighted_quantile(values, weights, 0.50)
        lo = weighted_quantile(values, weights, 0.025)
        hi = weighted_quantile(values, weights, 0.975)
        ci_rows.append(
            {
                "parameter": name,
                "bootstrap_median": point,
                "ci95_low": lo,
                "ci95_high": hi,
                "ci95_width_over_median": (hi - lo) / abs(point) if point else float("nan"),
                "min": min(values),
                "max": max(values),
            }
        )
    ci = pd.DataFrame(ci_rows)
    ci_path = args.output / "task6_bootstrap_parameter_ci.csv"
    ci.to_csv(ci_path, index=False)

    print()
    print(ci.to_string(index=False))
    pinned_weight = table.loc[table["n_bound_pinned"] > 0, "weight"].sum()
    thin = table.loc[table["n_distinct"] <= 2, "weight"].sum()
    print(f"\nweight on resamples with any parameter pinned to a bound: {pinned_weight:.4f}")
    print(f"weight on resamples covering <= 2 distinct systems:        {thin:.4f}")

    provenance = {
        "schema": "breos-task6-bootstrap-v1",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "breos_source": _git_commit(Path(__file__).resolve().parents[2]),
        "research_source": _git_commit(args.research_repo),
        "calibration_variant": args.calibration_variant,
        "optimizer_mode": args.optimizer_mode,
        "fitted_parameters": list(PARAM_NAMES),
        "systems": list(LFP_SYSTEMS),
        "soh_points_per_system": truth_counts,
        "soh_points_total": sum(truth_counts.values()),
        "resamples": {
            "scheme": "exhaustive multiset enumeration, multinomially weighted",
            "distinct": len(multisets),
            "equivalent_random_draws": "complete distribution; no sampling error",
        },
        "cache_files": {p.name: _sha256(p) for p in sorted(args.cache_dir.glob(f"{args.resolution}_system_*.parquet"))},
        "outputs": {params_path.name: _sha256(params_path), ci_path.name: _sha256(ci_path)},
    }
    (args.output / "task6_bootstrap_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"\nWrote {params_path}\nWrote {ci_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
