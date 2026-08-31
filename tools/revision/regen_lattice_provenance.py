#!/usr/bin/env python3
"""Rebuild a lattice bundle's provenance record after grid_eval died writing it.

The v1 and v2 lattice runs evaluated every design, wrote both result CSVs and
every archive shard, and then raised in the provenance block because
``--config`` was passed as a repository-relative path (fixed in c7d04da). The
results are intact; only the JSON is missing.

Everything the original record held is recoverable without re-simulating. The
grid, the roof and budget constraints and the CAPEX pre-filter are arithmetic
over the config, and the CAPEX check is re-derived from the results table
rather than copied, so this re-verifies the pre-filter instead of asserting it
again on trust.

The record states that it was regenerated and which commit the run itself
used, because that commit is not this file's HEAD.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import tomllib
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

_spec = importlib.util.spec_from_file_location("repro", PROJECT_ROOT / "tools/reproduce_article1.py")
repro = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(repro)

import breos  # noqa: E402
from breos.economics import calculate_costs, cost_params_from_config  # noqa: E402
from breos.optimization import _resolve_pv_module_and_area  # noqa: E402


def _frange(spec: str) -> list[float]:
    lo, hi, step = (float(x) for x in spec.split(":"))
    n = int(round((hi - lo) / step))
    return [lo + i * step for i in range(n + 1)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--calendar-model", required=True)
    ap.add_argument("--run-commit", required=True, help="commit the run used, not today's HEAD")
    ap.add_argument("--modules", required=True)
    ap.add_argument("--battery", required=True)
    ap.add_argument("--tilt", required=True)
    ap.add_argument("--azimuth", required=True)
    ap.add_argument("--command", required=True, help="the command line the run was launched with")
    ap.add_argument("--execution-backend", default="numba")
    args = ap.parse_args()

    config_bytes = args.config.read_bytes()
    config = tomllib.loads(config_bytes.decode("utf-8"))
    config.setdefault("battery", {})["calendar_model"] = args.calendar_model

    constraints = config.get("constraints", {})
    budget = float(constraints.get("budget_eur", float("inf")))
    max_area = float(constraints.get("max_area_m2", float("inf")))
    pv_params, module_area = _resolve_pv_module_and_area(config)
    cost_params = cost_params_from_config(config.get("costs", {}) or {}, config.get("financials", {}) or {})

    module_values = [int(x) for x in _frange(args.modules)]
    battery_values = _frange(args.battery)
    orientations = len(_frange(args.tilt)) * len(_frange(args.azimuth))
    roof_ok = sum(1 for m in module_values if m * module_area <= max_area + 1e-9)

    capex_by_pair, affordable = {}, []
    for modules in module_values:
        if modules * module_area > max_area + 1e-9:
            continue
        for battery_kwh in battery_values:
            capex = calculate_costs(
                n_modules=modules,
                module_power_w=pv_params.Mpp,
                battery_capacity_wh=battery_kwh * 1000.0,
                cost_params=cost_params,
            )["total_initial_cost"]
            capex_by_pair[(modules, battery_kwh)] = capex
            if capex <= budget + 1e-9:
                affordable.append((modules, battery_kwh))

    all_path = args.bundle / f"{args.label}_all.csv"
    front_path = args.bundle / f"{args.label}_front.csv"
    table = pd.read_csv(all_path)
    front = pd.read_csv(front_path)

    predicted = table.apply(lambda r: capex_by_pair[(int(r["Modules"]), float(r["Battery_kWh"]))], axis=1)
    worst = float((predicted - table["Projected_Initial_Cost_Eur"]).abs().max())
    over = int((table["Projected_Initial_Cost_Eur"] > budget + 1e-9).sum())
    if worst >= 1e-6 or over:
        raise SystemExit(f"re-verification failed: max CAPEX disagreement {worst}, {over} over budget")
    print(f"re-verified: max CAPEX disagreement {worst:.3e} EUR; {over} designs over budget")

    archive = args.bundle / "archive"
    shards = sorted(p.name for p in archive.glob("annual_*.csv")) if archive.is_dir() else []
    total_after_roof = roof_ok * len(battery_values) * orientations

    provenance = {
        "breos_version": breos.__version__,
        "breos_source": {"commit": args.run_commit, "tracked_worktree_dirty": False},
        "provenance_regenerated": {
            "reason": (
                "grid_eval raised writing this record when --config was relative; "
                "results, front and archive were already written. Fixed in c7d04da."
            ),
            "tool": "tools/revision/regen_lattice_provenance.py",
            "regenerated_at_commit": repro._git_revision().get("commit"),
            "capex_check_recomputed_from_results": True,
        },
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "command": args.command,
        "config": repro._display_path(args.config),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "config_diff_vs_base": {"battery.calendar_model": args.calendar_model},
        "grid": {
            "modules": args.modules,
            "battery_kwh": args.battery,
            "tilt_deg": args.tilt,
            "azimuth_deg": args.azimuth,
        },
        "module_area_m2": module_area,
        "constraints": {"budget_eur": budget, "max_area_m2": max_area},
        "points_evaluated": int(len(table)),
        "points_skipped_by_budget_prefilter": total_after_roof - int(len(table)),
        "points_within_budget": int(len(table)),
        "budget_prefilter": {
            "basis": "CAPEX depends only on (modules, battery); tilt and azimuth do not enter calculate_costs",
            "affordable_pairs": len(affordable),
            "max_capex_disagreement_eur": worst,
        },
        "archive": {
            "directory": str(archive),
            "format": "csv",
            "shards": shards,
            "compression": None,
            "rows_per_design": "one per projected year",
        },
        "front_size": int(len(front)),
        "execution_backend": args.execution_backend,
        "outputs": {all_path.name: repro._sha256(all_path), front_path.name: repro._sha256(front_path)},
    }
    out = args.bundle / f"{args.label}_provenance.json"
    out.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Wrote {out}  (front {len(front)}, evaluated {len(table)}, shards {len(shards)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
