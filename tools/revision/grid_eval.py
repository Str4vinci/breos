#!/usr/bin/env python3
"""Evaluate a rectangular design grid under the projected objective basis.

Used as a convergence check against an NSGA-II front (Task 1) and as the
exhaustive benchmark (Task 4c). Applies the same roof and budget constraints
the optimiser applies.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import itertools
import json
import platform
import shlex
import sys
import tomllib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

_spec = importlib.util.spec_from_file_location("repro", PROJECT_ROOT / "tools/reproduce_article1.py")
repro = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(repro)

import breos  # noqa: E402
from breos.optimization import (  # noqa: E402
    _resolve_pv_module_and_area,
    evaluate_projected_design,
)

GI = "Projected_Grid_Independence_%"
NPV = "Projected_NPV_Eur"

_STATE: dict = {}


def _frange(spec: str) -> list[float]:
    lo, hi, step = (float(x) for x in spec.split(":"))
    n = int(round((hi - lo) / step))
    return [lo + i * step for i in range(n + 1)]


def _init(config, rlp_directory, backend):
    weather, load, _, _ = repro._load_inputs(config, Path(rlp_directory))
    _STATE.update(config=config, weather=weather, load=load, backend=backend)


def _run(point):
    modules, battery_kwh, tilt, azimuth = point
    r = evaluate_projected_design(
        _STATE["weather"],
        _STATE["load"],
        _STATE["config"],
        n_modules=int(modules),
        battery_kwh=float(battery_kwh),
        tilt=float(tilt),
        azimuth=float(azimuth),
        execution_backend=_STATE["backend"],
    )
    m = r.metrics
    rep = r.yearly.index[r.yearly["Replacements"] > 0]
    return {
        "Modules": int(modules),
        "Battery_kWh": float(battery_kwh),
        "Tilt": float(tilt),
        "Azimuth": float(azimuth),
        GI: m[GI],
        NPV: m[NPV],
        "Replacement_Year": None if len(rep) == 0 else int(r.yearly.loc[rep[0], "Year"]),
        "Projected_Total_Replacements": m["Projected_Total_Replacements"],
        "Projected_Breakeven_Year_Exact": m.get("Projected_Breakeven_Year_Exact"),
        "Projected_Initial_Cost_Eur": m["Projected_Initial_Cost_Eur"],
        "Projected_Final_SOH_%": m["Projected_Final_SOH_%"],
    }


def pareto_mask(gi: np.ndarray, npv: np.ndarray) -> np.ndarray:
    """True where no other point is >= on both objectives and > on one."""
    order = np.lexsort((-npv, -gi))
    keep = np.zeros(len(gi), dtype=bool)
    best_npv = -np.inf
    for i in order:
        if npv[i] > best_npv + 1e-9:
            keep[i] = True
            best_npv = npv[i]
    return keep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=repro.DEFAULT_CONFIG)
    ap.add_argument("--rlp-directory", type=Path, required=True)
    ap.add_argument("--calendar-model")
    ap.add_argument("--modules", required=True, help="lo:hi:step")
    ap.add_argument("--battery", required=True, help="lo:hi:step (kWh)")
    ap.add_argument("--tilt", required=True, help="lo:hi:step (deg)")
    ap.add_argument("--azimuth", required=True, help="lo:hi:step (deg)")
    ap.add_argument("--n-procs", type=int, default=8)
    ap.add_argument("--execution-backend", default="numba")
    ap.add_argument("--label", default="grid")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    config_bytes = args.config.read_bytes()
    config = tomllib.loads(config_bytes.decode("utf-8"))
    diff = {}
    if args.calendar_model:
        config.setdefault("battery", {})["calendar_model"] = args.calendar_model
        diff["battery.calendar_model"] = args.calendar_model

    constraints = config.get("constraints", {})
    budget = float(constraints.get("budget_eur", float("inf")))
    max_area = float(constraints.get("max_area_m2", float("inf")))
    _pv, module_area = _resolve_pv_module_and_area(config)

    points = [
        p
        for p in itertools.product(
            [int(x) for x in _frange(args.modules)],
            _frange(args.battery),
            _frange(args.tilt),
            _frange(args.azimuth),
        )
        if p[0] * module_area <= max_area + 1e-9
    ]
    print(f"module area {module_area:.4f} m2; roof cap {max_area} m2 -> max {int(max_area // module_area)} modules")
    print(f"{len(points)} grid points after the roof constraint; budget filter applied post-hoc")

    with ProcessPoolExecutor(
        max_workers=args.n_procs,
        initializer=_init,
        initargs=(config, str(args.rlp_directory), args.execution_backend),
    ) as pool:
        rows = list(pool.map(_run, points, chunksize=4))

    table = pd.DataFrame(rows)
    table["Within_Budget"] = table["Projected_Initial_Cost_Eur"] <= budget + 1e-9
    feasible = table[table["Within_Budget"]].reset_index(drop=True)
    mask = pareto_mask(feasible[GI].to_numpy(), feasible[NPV].to_numpy())
    feasible["On_Grid_Pareto_Front"] = mask
    print(f"{len(feasible)} within budget {budget}; {int(mask.sum())} on the exhaustive front")

    args.output.mkdir(parents=True, exist_ok=True)
    all_path = args.output / f"{args.label}_all.csv"
    front_path = args.output / f"{args.label}_front.csv"
    table.to_csv(all_path, index=False)
    feasible[mask].sort_values(GI).to_csv(front_path, index=False)

    provenance = {
        "breos_version": breos.__version__,
        "breos_source": repro._git_revision(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "config": str(args.config.relative_to(PROJECT_ROOT)),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "config_diff_vs_base": diff,
        "grid": {
            "modules": args.modules,
            "battery_kwh": args.battery,
            "tilt_deg": args.tilt,
            "azimuth_deg": args.azimuth,
        },
        "module_area_m2": module_area,
        "constraints": {"budget_eur": budget, "max_area_m2": max_area},
        "points_evaluated": len(points),
        "points_within_budget": int(len(feasible)),
        "front_size": int(mask.sum()),
        "execution_backend": args.execution_backend,
        "outputs": {
            all_path.name: repro._sha256(all_path),
            front_path.name: repro._sha256(front_path),
        },
    }
    (args.output / f"{args.label}_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Wrote {all_path}\nWrote {front_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
