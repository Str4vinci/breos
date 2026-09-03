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
import csv
import hashlib
import importlib.util
import json
import platform
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

_spec = importlib.util.spec_from_file_location("repro", PROJECT_ROOT / "tools/reproduce_article1.py")
repro = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(repro)

import breos  # noqa: E402
from breos.economics import calculate_costs, cost_params_from_config  # noqa: E402
from breos.optimization import _resolve_pv_module_and_area  # noqa: E402
from tools.revision.grid_eval import GI, NPV, pareto_mask  # noqa: E402


def _frange(spec: str) -> list[float]:
    lo, hi, step = (float(x) for x in spec.split(":"))
    n = int(round((hi - lo) / step))
    return [lo + i * step for i in range(n + 1)]


def _validate_result_grid(
    table: pd.DataFrame,
    affordable: list[tuple[int, float]],
    tilt_values: list[float],
    azimuth_values: list[float],
) -> set[int]:
    """Require one result for every affordable grid point and no others."""
    required = {"Design_ID", "Modules", "Battery_kWh", "Tilt", "Azimuth", GI, NPV}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"result table is missing columns: {missing}")

    expected_order = [
        (int(modules), float(battery_kwh), float(tilt), float(azimuth))
        for modules, battery_kwh in affordable
        for tilt in tilt_values
        for azimuth in azimuth_values
    ]
    expected = set(expected_order)

    numeric = {
        column: pd.to_numeric(table[column], errors="coerce")
        for column in ("Design_ID", "Modules", "Battery_kWh", "Tilt", "Azimuth")
    }
    for column, values in numeric.items():
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"result table contains non-finite or non-numeric {column} values")
    for column in ("Design_ID", "Modules"):
        values = numeric[column].to_numpy(dtype=float)
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"result table contains non-integral {column} values")

    ids = numeric["Design_ID"].astype("int64")
    modules = numeric["Modules"].astype("int64")
    actual_rows = list(
        zip(
            modules,
            numeric["Battery_kWh"].astype(float),
            numeric["Tilt"].astype(float),
            numeric["Azimuth"].astype(float),
        )
    )
    actual = set(actual_rows)
    duplicate_points = len(actual_rows) - len(actual)
    missing_points = expected - actual
    extra_points = actual - expected
    if duplicate_points or missing_points or extra_points:
        raise ValueError(
            "result grid does not match the requested affordable grid: "
            f"{duplicate_points} duplicates, {len(missing_points)} missing, {len(extra_points)} extra"
        )

    if ids.duplicated().any():
        raise ValueError(f"result table contains {int(ids.duplicated().sum())} duplicate Design_ID values")
    expected_ids = set(range(len(expected)))
    actual_ids = set(ids)
    if actual_ids != expected_ids:
        raise ValueError(
            f"result Design_ID coverage is incomplete: {len(expected_ids - actual_ids)} missing, "
            f"{len(actual_ids - expected_ids)} extra"
        )

    expected_id_by_point = {point: design_id for design_id, point in enumerate(expected_order)}
    wrong_mappings = sum(int(design_id) != expected_id_by_point[point] for design_id, point in zip(ids, actual_rows))
    if wrong_mappings:
        raise ValueError(
            "result Design_ID mapping does not match grid_eval enumeration: "
            f"{wrong_mappings} coordinates have the wrong Design_ID"
        )
    return actual_ids


def _validate_front(table: pd.DataFrame, front: pd.DataFrame) -> int:
    """Require the stored front to equal a fresh Pareto reduction."""
    if "Design_ID" not in front:
        raise ValueError("front table is missing Design_ID")
    if front["Design_ID"].duplicated().any():
        raise ValueError("front table contains duplicate Design_ID values")

    mask = pareto_mask(table[GI].to_numpy(), table[NPV].to_numpy())
    expected_ids = set(table.loc[mask, "Design_ID"].astype(int))
    actual_ids = set(front["Design_ID"].astype(int))
    if actual_ids != expected_ids:
        raise ValueError(
            f"front does not match a fresh Pareto reduction: {len(expected_ids - actual_ids)} missing, "
            f"{len(actual_ids - expected_ids)} extra"
        )

    compare = ["Modules", "Battery_kWh", "Tilt", "Azimuth", GI, NPV]
    expected = table.set_index("Design_ID").loc[sorted(expected_ids), compare]
    actual = front.set_index("Design_ID").loc[sorted(actual_ids), compare]
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=True)
    return len(expected_ids)


def _validate_csv_archive(archive: Path, expected_ids: set[int], years_projection: int) -> dict:
    """Verify archive headers, hashes, and complete design-year coverage."""
    shards = sorted(archive.glob("annual_*.csv")) if archive.is_dir() else []
    if not shards:
        raise ValueError(f"archive contains no annual CSV shards: {archive}")

    pairs = []
    manifest = []
    for shard in shards:
        with shard.open(newline="") as handle:
            header = next(csv.reader(handle), [])
        duplicates = sorted({column for column in header if header.count(column) > 1})
        if duplicates:
            raise ValueError(f"archive shard {shard.name} has duplicate columns: {duplicates}")
        missing = sorted({"Design_ID", "Year"} - set(header))
        if missing:
            raise ValueError(f"archive shard {shard.name} is missing columns: {missing}")

        frame = pd.read_csv(shard, usecols=["Design_ID", "Year"])
        pairs.append(frame)
        manifest.append(
            {
                "file": shard.name,
                "sha256": repro._sha256(shard),
                "bytes": shard.stat().st_size,
                "rows": len(frame),
            }
        )

    coverage = pd.concat(pairs, ignore_index=True)
    coverage["Design_ID"] = coverage["Design_ID"].astype(int)
    coverage["Year"] = coverage["Year"].astype(int)
    duplicates = int(coverage.duplicated(["Design_ID", "Year"]).sum())
    actual_ids = set(coverage["Design_ID"])
    invalid_years = int((~coverage["Year"].between(1, years_projection)).sum())
    expected_rows = len(expected_ids) * years_projection
    if duplicates or actual_ids != expected_ids or len(coverage) != expected_rows or invalid_years:
        raise ValueError(
            "archive coverage is incomplete: "
            f"{duplicates} duplicate design-years, {len(expected_ids - actual_ids)} designs missing, "
            f"{len(actual_ids - expected_ids)} designs extra, {invalid_years} invalid years, "
            f"{len(coverage)} of {expected_rows} rows"
        )

    counts = coverage.groupby("Design_ID")["Year"].nunique()
    if not (counts == years_projection).all():
        raise ValueError("archive does not contain every projected year for every design")
    return {"shards": manifest, "rows": len(coverage), "designs": len(actual_ids)}


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
    tilt_values = _frange(args.tilt)
    azimuth_values = _frange(args.azimuth)
    orientations = len(tilt_values) * len(azimuth_values)
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
    table = pd.read_csv(all_path, float_precision="round_trip")
    front = pd.read_csv(front_path, float_precision="round_trip")

    result_ids = _validate_result_grid(table, affordable, tilt_values, azimuth_values)
    front_size = _validate_front(table, front)
    years_projection = int(
        config.get("simulation", {}).get("years_projection", config.get("financials", {}).get("project_lifespan", 20))
    )
    archive = args.bundle / "archive"
    archive_verification = _validate_csv_archive(archive, result_ids, years_projection)

    predicted = table.apply(lambda r: capex_by_pair[(int(r["Modules"]), float(r["Battery_kWh"]))], axis=1)
    worst = float((predicted - table["Projected_Initial_Cost_Eur"]).abs().max())
    over = int((table["Projected_Initial_Cost_Eur"] > budget + 1e-9).sum())
    if worst >= 1e-6 or over:
        raise SystemExit(f"re-verification failed: max CAPEX disagreement {worst}, {over} over budget")
    print(f"re-verified: max CAPEX disagreement {worst:.3e} EUR; {over} designs over budget")

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
            "result_grid_recomputed": True,
            "pareto_front_recomputed": True,
            "archive_coverage_recomputed": True,
            "run_commit_evidence": "operator-supplied; the result files do not encode their source commit",
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
            "shards": archive_verification["shards"],
            "compression": None,
            "rows": archive_verification["rows"],
            "designs": archive_verification["designs"],
            "rows_per_design": years_projection,
        },
        "front_size": front_size,
        "execution_backend": args.execution_backend,
        "outputs": {all_path.name: repro._sha256(all_path), front_path.name: repro._sha256(front_path)},
    }
    out = args.bundle / f"{args.label}_provenance.json"
    out.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Wrote {out}  (front {front_size}, evaluated {len(table)}, shards {len(archive_verification['shards'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
