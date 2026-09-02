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
from breos.economics import calculate_costs, cost_params_from_config  # noqa: E402
from breos.optimization import (  # noqa: E402
    ProjectedDesignResult,
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


def _archive_format(requested: str) -> str:
    """Resolve the archive format, preferring parquet when an engine exists.

    Parquet halves the bytes and lets Task 7 project the handful of columns a
    re-ranking needs instead of parsing every row, but it needs pyarrow, which
    is not a declared dependency of this project. Rather than mutate the
    environment, fall back to CSV and say so.
    """
    if requested == "csv":
        return "csv"
    try:
        import pyarrow  # noqa: F401

        return "parquet"
    except ImportError:
        if requested == "parquet":
            raise SystemExit("--archive-format parquet needs pyarrow; install it or use csv")
        print("no parquet engine (pyarrow) available; archiving as CSV instead")
        return "csv"


def _prepare_archive_directory(path: Path) -> None:
    """Create a new archive directory and refuse to mix in stale shards."""
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to reuse nonempty archive directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _init(config, rlp_directory, backend, archive, archive_format):
    weather, load, _, _ = repro._load_inputs(config, Path(rlp_directory))
    _STATE.update(
        config=config,
        weather=weather,
        load=load,
        backend=backend,
        archive=archive,
        archive_format=archive_format,
    )


def _annual_archive_frame(design_id: int, result: ProjectedDesignResult) -> pd.DataFrame:
    """Join one design's annual energy ledger to its annual financial ledger.

    The two frames overlap. Concatenating them unchanged emits
    ``PV_Production_kWh`` and ``Export_kWh`` twice, which makes column
    selection depend on which duplicate a reader's CSV parser happens to
    keep. The financial projection copies those columns straight from the
    yearly summary, so one canonical copy is enough -- but "should be equal"
    is checked rather than assumed, because a silent disagreement would mean
    the two ledgers had drifted apart and every archived cash flow would be
    suspect. Nothing else is dropped: every itemised, undiscounted financial
    component the projection reports is kept.
    """
    yearly = result.yearly.reset_index(drop=True)
    financial = result.financial.reset_index(drop=True)

    shared = [column for column in financial.columns if column in yearly.columns]
    for column in shared:
        left = pd.to_numeric(yearly[column], errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(financial[column], errors="coerce").to_numpy(dtype=float)
        if not np.array_equal(left, right, equal_nan=True):
            worst = float(np.nanmax(np.abs(left - right)))
            raise ValueError(f"design {design_id}: yearly and financial ledgers disagree on {column} by up to {worst}")

    annual = pd.concat([yearly, financial.drop(columns=shared)], axis=1)
    annual.insert(0, "Design_ID", design_id)
    return annual


def _reject_duplicate_columns(frame: pd.DataFrame) -> None:
    """Fail before a shard is written rather than archive an ambiguous header."""
    counts = frame.columns.value_counts()
    duplicated = sorted(counts[counts > 1].index)
    if duplicated:
        raise ValueError(f"annual archive shard has duplicate column names: {duplicated}")


def _run_chunk(task):
    """Evaluate one chunk of designs, archiving the annual series as a shard.

    Chunking is what gives each worker a natural shard boundary. Returning the
    annual frames to the parent instead would push roughly 700 MB per model
    through pickling for no gain, since nothing in the parent reads them.
    """
    chunk_index, points = task
    summaries, archive_frames = [], []
    for design_id, point in points:
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
        summaries.append(
            {
                "Design_ID": design_id,
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
        )

        if _STATE["archive"] is not None:
            archive_frames.append(_annual_archive_frame(design_id, r))

    if _STATE["archive"] is not None and archive_frames:
        frame = pd.concat(archive_frames, ignore_index=True)
        _reject_duplicate_columns(frame)
        if _STATE["archive_format"] == "parquet":
            frame.to_parquet(
                Path(_STATE["archive"]) / f"annual_{chunk_index:05d}.parquet", index=False, compression=None
            )
        else:
            frame.to_csv(Path(_STATE["archive"]) / f"annual_{chunk_index:05d}.csv", index=False)

    return summaries


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
    ap.add_argument("--archive", type=Path, help="write per-design annual series as parquet shards")
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--archive-format", choices=("auto", "parquet", "csv"), default="auto")
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

    module_values = [int(x) for x in _frange(args.modules)]
    battery_values = _frange(args.battery)
    tilt_values = _frange(args.tilt)
    azimuth_values = _frange(args.azimuth)

    # CAPEX is a function of module count and battery size alone -- tilt and
    # azimuth never enter calculate_costs -- so the budget can be applied to the
    # (modules, battery) pairs before the orientation axes are expanded. That is
    # exact rather than a heuristic, and it is why the filter can run before
    # evaluation instead of discarding a fifth of the results afterwards.
    cost_params = cost_params_from_config(config.get("costs", {}) or {}, config.get("financials", {}) or {})
    pv_params, _ = _resolve_pv_module_and_area(config)
    capex_by_pair, affordable_pairs = {}, []
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
                affordable_pairs.append((modules, battery_kwh))

    orientations = list(itertools.product(tilt_values, azimuth_values))
    points = [
        (modules, battery_kwh, tilt, azimuth)
        for modules, battery_kwh in affordable_pairs
        for tilt, azimuth in orientations
    ]
    roof_ok = sum(1 for m in module_values if m * module_area <= max_area + 1e-9)
    total_after_roof = roof_ok * len(battery_values) * len(orientations)
    print(f"module area {module_area:.4f} m2; roof cap {max_area} m2 -> max {int(max_area // module_area)} modules")
    print(f"{total_after_roof} grid points after the roof constraint")
    print(
        f"{len(affordable_pairs)} of {roof_ok * len(battery_values)} (modules, battery) pairs within "
        f"budget {budget} -> {len(points)} evaluated, {total_after_roof - len(points)} skipped "
        f"({100.0 * (total_after_roof - len(points)) / total_after_roof:.1f}%)"
    )

    archive_format = _archive_format(args.archive_format) if args.archive else None
    if args.archive:
        _prepare_archive_directory(args.archive)

    indexed = list(enumerate(points))
    chunks = [(i // args.chunk_size, indexed[i : i + args.chunk_size]) for i in range(0, len(indexed), args.chunk_size)]
    with ProcessPoolExecutor(
        max_workers=args.n_procs,
        initializer=_init,
        initargs=(
            config,
            str(args.rlp_directory),
            args.execution_backend,
            str(args.archive) if args.archive else None,
            archive_format,
        ),
    ) as pool:
        rows = [row for batch in pool.map(_run_chunk, chunks) for row in batch]

    table = pd.DataFrame(rows)
    # The pre-filter is only sound if the CAPEX it predicted is the CAPEX the
    # simulation reports, so check rather than trust it.
    predicted = table.apply(lambda r: capex_by_pair[(int(r["Modules"]), float(r["Battery_kWh"]))], axis=1)
    worst = float((predicted - table["Projected_Initial_Cost_Eur"]).abs().max())
    assert worst < 1e-6, f"budget pre-filter CAPEX disagrees with simulated CAPEX by {worst}"
    over = int((table["Projected_Initial_Cost_Eur"] > budget + 1e-9).sum())
    assert over == 0, f"{over} evaluated designs exceed the budget"
    print(f"pre-filter check: max CAPEX disagreement {worst:.3e} EUR; 0 evaluated designs over budget")
    table["Within_Budget"] = True
    feasible = table.reset_index(drop=True)
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
        # relative_to raises on a relative --config, which is how this is
        # normally typed; _display_path resolves first and falls back to an
        # absolute path for a config outside the repository.
        "config": repro._display_path(args.config),
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
        "points_skipped_by_budget_prefilter": total_after_roof - len(points),
        "points_within_budget": int(len(feasible)),
        "budget_prefilter": {
            "basis": "CAPEX depends only on (modules, battery); tilt and azimuth do not enter calculate_costs",
            "affordable_pairs": len(affordable_pairs),
            "max_capex_disagreement_eur": worst,
        },
        "archive": (
            None
            if not args.archive
            else {
                "directory": str(args.archive),
                "format": archive_format,
                "shards": sorted(p.name for p in args.archive.glob(f"annual_*.{archive_format}")),
                "compression": None,
                "rows_per_design": "one per projected year",
            }
        ),
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
