#!/usr/bin/env python3
"""Task 3: how much of storage economics is the battery replacement assumption.

Ported unchanged in method from the 0.6.1 revision runner
(`results/revision-0.6.1/task3/task3_eol_sweep.py`) so the sweep can be re-run
on a later release. The models, end-of-life settings, grid, candidate set and
every verification check are identical; only the commit pin, the lattice
location and the output default are now CLI options.

Runs the three degradation models over a reduced exhaustive grid at each
model's accepted knee orientation, under three end-of-life settings, and
replays the final C1 to C5 set under every one of them.

Nothing in `breos/` or `tools/` is modified. The evaluation, Pareto filter and
representative selection all come from tracked source, so the results are
comparable with the accepted lattice by construction.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shlex
import sys
import time
import tomllib
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

_spec = importlib.util.spec_from_file_location("repro", PROJECT_ROOT / "tools/reproduce_article1.py")
repro = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(repro)

import breos  # noqa: E402
from breos.optimization import evaluate_projected_design  # noqa: E402
from tools.revision.grid_eval import pareto_mask  # noqa: E402

GI = "Projected_Grid_Independence_%"
NPV = "Projected_NPV_Eur"

ACCEPTED_CONFIG_SHA256 = "b7b467ece194c89e047a9eec97e849eea2dd9cab3d7e909a7ac489889ca319bc"
# The 0.6.1 run pinned its own HEAD. This port is meant to run on a later
# release, so the commit is recorded rather than pinned; pass --expect-commit
# to restore the hard check. The config hash stays pinned: the same file must
# drive the sweep for the results to be comparable.
ACCEPTED_LATTICE = Path("/home/leo/code/breos/results/revision-0.6.1/lattice-1c")

# Each model is swept at its own accepted headline knee orientation.
MODELS = (
    ("field-v1", "naumann_lam_field_calibrated_v1", 35.0, 195.0),
    ("field-v2", "naumann_lam_field_calibrated_v2", 40.0, 200.0),
    ("laboratory", "naumann_lam", 45.0, 190.0),
)

# 60% is not run: the brief excludes it for want of a first-life residential
# basis in the selected literature.
SETTINGS = (
    ("eol-80", {"eol_percentage": 0.80, "enable_replacement": True}, "end of life at 80% state of health"),
    (
        "eol-70",
        {"eol_percentage": 0.70, "enable_replacement": True},
        "end of life at 70% state of health, the accepted base",
    ),
    (
        "no-replacement",
        {"eol_percentage": 0.70, "enable_replacement": False},
        "replacement disabled through year 20; the pack keeps ageing past its threshold",
    ),
)

# The final representative set, with C2 fixed by the Gate 2 decision.
CANDIDATES = (
    ("C1", "maximum NPV", 6, 0.0, 30.0, 200.0),
    ("C2", "best-value storage", 9, 7.0, 35.0, 200.0),
    ("C3", "knee", 9, 9.0, 35.0, 195.0),
    ("C4", "maximum GI", 9, 20.0, 50.0, 185.0),
    ("C5", "low-investment off-front benchmark", 4, 0.0, 35.0, 180.0),
)

MODULE_VALUES = tuple(range(0, 10))
BATTERY_VALUES = tuple(float(k) for k in range(0, 21))

_STATE: dict = {}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _init(config: dict, rlp_directory: str, backend: str) -> None:
    weather, load, _, _ = repro._load_inputs(config, Path(rlp_directory))
    _STATE.update(weather=weather, load=load, backend=backend)


def _run_chunk(task):
    """Evaluate one chunk of (run_key, config, design) tuples."""
    run_key, config, points = task
    rows = []
    for point in points:
        kind, label, modules, battery_kwh, tilt, azimuth = point
        result = evaluate_projected_design(
            _STATE["weather"],
            _STATE["load"],
            config,
            n_modules=int(modules),
            battery_kwh=float(battery_kwh),
            tilt=float(tilt),
            azimuth=float(azimuth),
            execution_backend=_STATE["backend"],
        )
        metrics = result.metrics
        replaced = result.yearly.index[result.yearly["Replacements"] > 0]
        rows.append(
            {
                "run_key": run_key,
                "kind": kind,
                "label": label,
                "Modules": int(modules),
                "Battery_kWh": float(battery_kwh),
                "Tilt": float(tilt),
                "Azimuth": float(azimuth),
                GI: metrics[GI],
                NPV: metrics[NPV],
                "Replacement_Year": (None if len(replaced) == 0 else int(result.yearly.loc[replaced[0], "Year"])),
                "Projected_Total_Replacements": metrics["Projected_Total_Replacements"],
                "Projected_Breakeven_Year_Exact": metrics.get("Projected_Breakeven_Year_Exact"),
                "Projected_Initial_Cost_Eur": metrics["Projected_Initial_Cost_Eur"],
                "Projected_Final_SOH_%": metrics["Projected_Final_SOH_%"],
            }
        )
    return rows


def _status(path: Path, **fields) -> None:
    path.write_text(json.dumps(fields, indent=2, default=str) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "validation/article1/revision-0.6.1/article1-power-1c.toml"
    )
    ap.add_argument("--rlp-directory", type=Path, default=PROJECT_ROOT / "dev/article1-inputs/rlp")
    ap.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/task3-eol")
    ap.add_argument(
        "--lattice",
        type=Path,
        default=ACCEPTED_LATTICE,
        help="Accepted 1 C lattice bundle to cross-check the eol-70 slice against",
    )
    ap.add_argument(
        "--expect-commit",
        help="Require HEAD to be this commit. Omitted by default so the sweep can run on a later release.",
    )
    ap.add_argument("--n-procs", type=int, default=16)
    ap.add_argument("--execution-backend", default="numba")
    ap.add_argument("--chunk-size", type=int, default=30)
    args = ap.parse_args()

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "status.json"
    started = time.time()
    _status(
        status_path,
        state="running",
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        stage="loading inputs",
        pid=os.getpid(),
    )

    try:
        config_bytes = args.config.read_bytes()
        base_config = tomllib.loads(config_bytes.decode("utf-8"))
        config_sha = hashlib.sha256(config_bytes).hexdigest()
        if config_sha != ACCEPTED_CONFIG_SHA256:
            raise SystemExit(f"config sha256 {config_sha} is not the accepted {ACCEPTED_CONFIG_SHA256}")
        source_revision = repro._git_revision()
        if args.expect_commit and source_revision["commit"] != args.expect_commit:
            raise SystemExit(f"HEAD {source_revision['commit']} is not the required {args.expect_commit}")
        if source_revision["tracked_worktree_dirty"]:
            raise SystemExit("refusing to run: the tracked worktree is dirty")

        # Build every task up front so the run is one flat parallel sweep.
        run_configs, tasks, diffs = {}, [], {}
        for model_key, calendar_model, tilt, azimuth in MODELS:
            for setting_key, overrides, _description in SETTINGS:
                run_key = f"{model_key}|{setting_key}"
                config = tomllib.loads(config_bytes.decode("utf-8"))
                battery = config.setdefault("battery", {})
                battery["calendar_model"] = calendar_model
                battery.update(overrides)
                # The projection must not stop early anywhere in this task.
                config.get("optimization", {}).pop("early_stop", None)
                run_configs[run_key] = config
                diffs[run_key] = {
                    "battery.calendar_model": calendar_model,
                    **{f"battery.{k}": v for k, v in overrides.items()},
                    "optimization.early_stop": "removed",
                }
                points = [("grid", "", m, b, tilt, azimuth) for m in MODULE_VALUES for b in BATTERY_VALUES]
                points += [("candidate", label, m, b, t, a) for label, _role, m, b, t, a in CANDIDATES]
                for i in range(0, len(points), args.chunk_size):
                    tasks.append((run_key, config, points[i : i + args.chunk_size]))

        total_points = sum(len(t[2]) for t in tasks)
        print(f"{len(MODELS)} models x {len(SETTINGS)} settings = {len(run_configs)} runs")
        print(f"{total_points} evaluations in {len(tasks)} chunks on {args.n_procs} workers")
        _status(
            status_path,
            state="running",
            stage="evaluating",
            runs=len(run_configs),
            evaluations=total_points,
            chunks=len(tasks),
            pid=os.getpid(),
        )

        with ProcessPoolExecutor(
            max_workers=args.n_procs,
            initializer=_init,
            initargs=(base_config, str(args.rlp_directory), args.execution_backend),
        ) as pool:
            rows = [row for batch in pool.map(_run_chunk, tasks) for row in batch]

        elapsed = time.time() - started
        print(f"{len(rows)} evaluations in {elapsed:.1f} s ({len(rows) / elapsed:.2f}/s)")

        table = pd.DataFrame(rows)
        table[["model", "setting"]] = table["run_key"].str.split("|", expand=True)

        grid = table[table["kind"] == "grid"].drop(columns=["label"]).reset_index(drop=True)
        candidates = table[table["kind"] == "candidate"].reset_index(drop=True)
        candidates["role"] = candidates["label"].map({c[0]: c[1] for c in CANDIDATES})

        # Representatives, selected by the repository's own definitions.
        rep_rows, front_sizes = [], {}
        for run_key, block in grid.groupby("run_key", sort=False):
            block = block.reset_index(drop=True)
            mask = pareto_mask(block[GI].to_numpy(), block[NPV].to_numpy())
            front = block[mask].reset_index(drop=True)
            front_sizes[run_key] = int(mask.sum())
            reps = repro._select_pareto_representatives(front)
            for row in reps.to_dict(orient="records"):
                rep_rows.append({"run_key": run_key, **row})
        representatives = pd.DataFrame(rep_rows)
        representatives[["model", "setting"]] = representatives["run_key"].str.split("|", expand=True)
        grid["On_Grid_Pareto_Front"] = False
        for run_key, block in grid.groupby("run_key", sort=False):
            mask = pareto_mask(block[GI].to_numpy(), block[NPV].to_numpy())
            grid.loc[block.index, "On_Grid_Pareto_Front"] = mask

        # How much of storage NPV is the replacement assumption alone.
        sens_rows = []
        for model_key, _cm, _t, _a in MODELS:
            for label, role, modules, battery_kwh, tilt, azimuth in CANDIDATES:
                sel = candidates[(candidates.model == model_key) & (candidates.label == label)]
                by = {r["setting"]: r for r in sel.to_dict(orient="records")}
                base, none_, strict = by["eol-70"], by["no-replacement"], by["eol-80"]
                sens_rows.append(
                    {
                        "model": model_key,
                        "candidate": label,
                        "role": role,
                        "battery_kwh": battery_kwh,
                        "npv_eol_80": strict[NPV],
                        "npv_eol_70": base[NPV],
                        "npv_no_replacement": none_[NPV],
                        "npv_spread_eur": float(
                            max(strict[NPV], base[NPV], none_[NPV]) - min(strict[NPV], base[NPV], none_[NPV])
                        ),
                        "npv_no_replacement_minus_eol_70": float(none_[NPV] - base[NPV]),
                        "npv_eol_80_minus_eol_70": float(strict[NPV] - base[NPV]),
                        "gi_spread_pp": float(
                            max(strict[GI], base[GI], none_[GI]) - min(strict[GI], base[GI], none_[GI])
                        ),
                        "replacements_eol_80": strict["Projected_Total_Replacements"],
                        "replacements_eol_70": base["Projected_Total_Replacements"],
                        "replacements_no_replacement": none_["Projected_Total_Replacements"],
                        "replacement_year_eol_80": strict["Replacement_Year"],
                        "replacement_year_eol_70": base["Replacement_Year"],
                        "final_soh_no_replacement_pct": none_["Projected_Final_SOH_%"],
                    }
                )
        sensitivity = pd.DataFrame(sens_rows)

        paths = {}
        for name, frame in (
            ("task3_grid.csv", grid),
            ("task3_representatives.csv", representatives),
            ("task3_candidates.csv", candidates),
            ("task3_storage_npv_sensitivity.csv", sensitivity),
        ):
            path = out / name
            frame.to_csv(path, index=False)
            paths[name] = path

        # --- verification -------------------------------------------------
        checks: list[dict] = []

        def check(name, passed, detail):
            checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})

        check("config sha256 is the accepted one", True, config_sha)
        check(
            "HEAD is the commit this run records",
            (not args.expect_commit) or source_revision["commit"] == args.expect_commit,
            source_revision["commit"],
        )
        check("tracked worktree is clean", not source_revision["tracked_worktree_dirty"], "clean")
        check(
            "every model and setting ran", len(front_sizes) == len(MODELS) * len(SETTINGS), f"{len(front_sizes)} runs"
        )
        check(
            "grid evaluation count is exact",
            len(grid) == len(MODELS) * len(SETTINGS) * len(MODULE_VALUES) * len(BATTERY_VALUES),
            f"{len(grid)} rows",
        )
        check(
            "candidate replay count is exact",
            len(candidates) == len(MODELS) * len(SETTINGS) * len(CANDIDATES),
            f"{len(candidates)} rows",
        )
        check("no evaluation returned a null objective", not (grid[GI].isna().any() or grid[NPV].isna().any()), "none")
        check(
            "disabling replacement never replaces a pack",
            int(candidates.loc[candidates.setting == "no-replacement", "Projected_Total_Replacements"].sum()) == 0
            and int(grid.loc[grid.setting == "no-replacement", "Projected_Total_Replacements"].sum()) == 0,
            "0 replacements recorded",
        )
        check(
            "an 80% threshold never yields fewer replacements than a 70% threshold",
            bool((sensitivity["replacements_eol_80"] >= sensitivity["replacements_eol_70"]).all()),
            "80% >= 70% replacement count for every candidate",
        )
        replacement_year_80 = pd.to_numeric(sensitivity["replacement_year_eol_80"], errors="coerce")
        replacement_year_70 = pd.to_numeric(sensitivity["replacement_year_eol_70"], errors="coerce")
        replacement_order = replacement_year_70.isna() | (
            replacement_year_80.notna() & (replacement_year_80 <= replacement_year_70)
        )
        check(
            "an 80% threshold never triggers its first replacement later than a 70% threshold",
            bool(replacement_order.all()),
            "first 80% replacement year <= first 70% replacement year for every candidate",
        )

        # Cross-check against the accepted lattice. The reduced grid is one
        # orientation slice of it, so at eol-70 -- the accepted base setting --
        # every design on that slice must reproduce the accepted lattice
        # exactly. Comparing against the accepted full-grid representatives
        # instead would be wrong: those live at their own best orientations,
        # and each model's slice is fixed at its knee orientation.
        accepted_reps = pd.read_csv(args.lattice / "representatives.csv", float_precision="round_trip")
        bundle_by_model = {"field-v1": "base-v1", "field-v2": "field-v2", "laboratory": "laboratory"}
        slice_rows = []
        for model_key, _cm, tilt, azimuth in MODELS:
            bundle_dir = args.lattice / bundle_by_model[model_key]
            lattice_source = next(iter(sorted(bundle_dir.glob("*_all.csv"))))
            accepted_all = pd.read_csv(lattice_source, float_precision="round_trip")
            accepted_slice = accepted_all[(accepted_all.Tilt == tilt) & (accepted_all.Azimuth == azimuth)]
            mine = grid[(grid.model == model_key) & (grid.setting == "eol-70")]
            merged = mine.merge(
                accepted_slice, on=["Modules", "Battery_kWh"], suffixes=("_mine", "_accepted"), validate="one_to_one"
            )
            check(
                f"{model_key} slice at {tilt:g}/{azimuth:g} covers every accepted design",
                len(merged) == len(accepted_slice) == len(mine),
                f"{len(merged)} merged, {len(accepted_slice)} accepted, {len(mine)} reduced",
            )
            worst_gi = float((merged[f"{GI}_mine"] - merged[f"{GI}_accepted"]).abs().max())
            worst_npv = float((merged[f"{NPV}_mine"] - merged[f"{NPV}_accepted"]).abs().max())
            check(
                f"{model_key} slice reproduces the accepted lattice bit-for-bit",
                worst_gi == 0.0 and worst_npv == 0.0,
                f"max |dGI| {worst_gi:.3e} pp, max |dNPV| {worst_npv:.3e} EUR",
            )

            # The same slice, filtered and reduced by the same tracked code.
            sl = accepted_slice.reset_index(drop=True)
            sl_mask = pareto_mask(sl[GI].to_numpy(), sl[NPV].to_numpy())
            sl_reps = repro._select_pareto_representatives(sl[sl_mask].reset_index(drop=True))
            mine_reps = representatives[(representatives.model == model_key) & (representatives.setting == "eol-70")]
            accepted_full = accepted_reps[accepted_reps.bundle == bundle_by_model[model_key]]
            for rep in ("max_npv", "knee", "max_gi"):
                sr = sl_reps[sl_reps.Representative == rep].iloc[0]
                mr = mine_reps[mine_reps.Representative == rep].iloc[0]
                check(
                    f"{model_key} {rep} matches the accepted lattice on the same slice",
                    int(sr["Modules"]) == int(mr["Modules"])
                    and float(sr["Battery_kWh"]) == float(mr["Battery_kWh"])
                    and float(sr[NPV]) == float(mr[NPV]),
                    f"{int(mr['Modules'])} mod / {mr['Battery_kWh']:g} kWh, NPV {mr[NPV]!r}",
                )
                fr = accepted_full[accepted_full.representative == rep].iloc[0]
                slice_rows.append(
                    {
                        "model": model_key,
                        "representative": rep,
                        "slice_tilt_deg": tilt,
                        "slice_azimuth_deg": azimuth,
                        "accepted_full_grid_modules": int(fr.modules),
                        "accepted_full_grid_battery_kwh": float(fr.battery_kwh),
                        "accepted_full_grid_tilt_deg": float(fr.tilt_deg),
                        "accepted_full_grid_azimuth_deg": float(fr.azimuth_deg),
                        "accepted_full_grid_npv_eur": float(fr.npv_eur),
                        "slice_modules": int(mr["Modules"]),
                        "slice_battery_kwh": float(mr["Battery_kWh"]),
                        "slice_npv_eur": float(mr[NPV]),
                        "sizing_matches_full_grid": (
                            int(fr.modules) == int(mr["Modules"]) and float(fr.battery_kwh) == float(mr["Battery_kWh"])
                        ),
                    }
                )
        slice_reference = pd.DataFrame(slice_rows)

        # The knee is the representative each slice orientation was chosen for,
        # so it must reproduce the accepted full-grid sizing. The other two are
        # reported, not required: they live at their own best orientations.
        knees = slice_reference[slice_reference.representative == "knee"]
        check(
            "every model's knee sizing matches the accepted full-grid knee",
            bool(knees["sizing_matches_full_grid"].all()),
            "; ".join(f"{r.model} {r.slice_modules} mod / {r.slice_battery_kwh:g} kWh" for r in knees.itertuples()),
        )

        # PV-only designs cannot depend on the battery settings.
        pv_only = grid[grid["Battery_kWh"] == 0.0]
        spread = pv_only.groupby(["Modules", "Tilt", "Azimuth"])[NPV].agg(lambda s: s.max() - s.min())
        check(
            "PV-only designs are identical across all settings and models",
            float(spread.max()) == 0.0,
            f"max NPV spread {float(spread.max()):.3e} EUR",
        )

        slice_path = out / "task3_slice_reference.csv"
        slice_reference.to_csv(slice_path, index=False)
        paths["task3_slice_reference.csv"] = slice_path

        manifest = pd.DataFrame(
            [
                {
                    "file": name,
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                    "rows": sum(1 for _ in path.open()) - 1,
                }
                for name, path in paths.items()
            ]
        )
        manifest_path = out / "task3_manifest.csv"
        manifest.to_csv(manifest_path, index=False)

        provenance = {
            "task": "Task 3: battery end-of-life threshold and replacement assumption",
            "breos_version": breos.__version__,
            "breos_source": source_revision,
            "accepted_lattice": str(args.lattice),
            "accepted_lattice_representatives_sha256": sha256(args.lattice / "representatives.csv"),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "command": shlex.join([sys.executable, *sys.argv]),
            "config": repro._display_path(args.config),
            "config_sha256": config_sha,
            "config_diff_vs_accepted_base": diffs,
            "rlp_directory": str(args.rlp_directory),
            "models": [{"key": k, "calendar_model": c, "tilt_deg": t, "azimuth_deg": a} for k, c, t, a in MODELS],
            "settings": [{"key": k, "overrides": o, "description": d} for k, o, d in SETTINGS],
            "candidates": [
                {"label": lbl, "role": role, "modules": m, "battery_kwh": b, "tilt_deg": t, "azimuth_deg": a}
                for lbl, role, m, b, t, a in CANDIDATES
            ],
            "grid": {
                "modules": list(MODULE_VALUES),
                "battery_kwh": list(BATTERY_VALUES),
                "orientation": "fixed per model at its accepted knee orientation",
            },
            "front_sizes": front_sizes,
            "execution_backend": args.execution_backend,
            "n_procs": args.n_procs,
            "power_limit_c_rate": base_config["battery"]["power_limit_c_rate"],
            "early_termination": "removed from every run config",
            "evaluations": len(rows),
            "elapsed_s": elapsed,
            "reader": 'pandas.read_csv(..., float_precision="round_trip")',
            "outputs": {name: sha256(path) for name, path in paths.items()},
        }
        (out / "task3_provenance.json").write_text(json.dumps(provenance, indent=2, default=str) + "\n")
        verification = {
            "checks": checks,
            "passed": sum(c["passed"] for c in checks),
            "total": len(checks),
            "all_passed": all(c["passed"] for c in checks),
        }
        (out / "task3_verification.json").write_text(json.dumps(verification, indent=2) + "\n")

        for c in checks:
            print(f"[{'PASS' if c['passed'] else 'FAIL'}] {c['check']}: {c['detail']}")
        print(f"\n{verification['passed']} of {verification['total']} checks passed")

        _status(
            status_path,
            state="succeeded" if verification["all_passed"] else "failed",
            stage="complete",
            elapsed_s=elapsed,
            evaluations=len(rows),
            checks_passed=verification["passed"],
            checks_total=verification["total"],
            outputs={name: sha256(path) for name, path in paths.items()},
            manifest=sha256(manifest_path),
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        )
        return 0 if verification["all_passed"] else 1

    except BaseException as error:  # noqa: BLE001 - the runner must record why it stopped
        _status(
            status_path,
            state="failed",
            stage="exception",
            error=repr(error),
            traceback=traceback.format_exc(),
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
