#!/usr/bin/env python3
"""Task 5(f): does the constant-power assumption move any replacement year?

Replays every design on each constant-power (P0) Pareto front under all three
power models, holding the degradation model fixed to the one that produced that
front. Only the battery power limit varies.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import platform
import shlex
import sys
import tomllib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

_spec = importlib.util.spec_from_file_location("repro", PROJECT_ROOT / "tools/reproduce_article1.py")
repro = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(repro)

import breos  # noqa: E402
from breos.optimization import evaluate_projected_design  # noqa: E402

GI = "Projected_Grid_Independence_%"
NPV = "Projected_NPV_Eur"

POWER_MODELS = {
    "P0_constant_4352W": "validation/article1/article1-projected-optimization.toml",
    "P1_1C_symmetric": "validation/article1/revision-0.6.1/article1-power-1c.toml",
    "P2_0p5C_symmetric": "validation/article1/revision-0.6.1/article1-power-0p5c.toml",
}
DEGRADATION = {
    "base-v1": "naumann_lam_field_calibrated_v1",
    "field-v2": "naumann_lam_field_calibrated_v2",
    "laboratory": "naumann_lam",
}

_STATE: dict = {}


def _init(configs, rlp_directory, backend):
    any_cfg = next(iter(configs.values()))
    weather, load, _, _ = repro._load_inputs(any_cfg, Path(rlp_directory))
    _STATE.update(configs=configs, weather=weather, load=load, backend=backend)


def _run(job):
    front, power_model, modules, battery_kwh, tilt, azimuth = job
    cfg = _STATE["configs"][(front, power_model)]
    r = evaluate_projected_design(
        _STATE["weather"],
        _STATE["load"],
        cfg,
        n_modules=int(modules),
        battery_kwh=float(battery_kwh),
        tilt=float(tilt),
        azimuth=float(azimuth),
        execution_backend=_STATE["backend"],
    )
    hit = r.yearly.index[r.yearly["Replacements"] > 0]
    return {
        "Front": front,
        "Calendar_Model": DEGRADATION[front],
        "Power_Model": power_model,
        "Modules": int(modules),
        "Battery_kWh": float(battery_kwh),
        "Tilt": float(tilt),
        "Azimuth": float(azimuth),
        GI: r.metrics[GI],
        NPV: r.metrics[NPV],
        "Replacement_Year": None if len(hit) == 0 else int(r.yearly.loc[hit[0], "Year"]),
        "Total_Replacements": r.metrics["Projected_Total_Replacements"],
        "Final_SOH_%": r.metrics["Projected_Final_SOH_%"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--front-root", type=Path, required=True, help="Directory holding the P0 front bundles")
    ap.add_argument("--rlp-directory", type=Path, required=True)
    ap.add_argument("--n-procs", type=int, default=16)
    ap.add_argument("--execution-backend", default="numba")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    configs: dict[tuple[str, str], dict] = {}
    config_shas: dict[str, str] = {}
    for power_model, rel in POWER_MODELS.items():
        raw = (PROJECT_ROOT / rel).read_bytes()
        config_shas[power_model] = hashlib.sha256(raw).hexdigest()
        parsed = tomllib.loads(raw.decode("utf-8"))
        for front, calendar_model in DEGRADATION.items():
            cfg = copy.deepcopy(parsed)
            cfg["battery"]["calendar_model"] = calendar_model
            configs[(front, power_model)] = cfg

    jobs = []
    for front in DEGRADATION:
        designs = pd.read_csv(args.front_root / front / "pareto_results.csv")
        for _, d in designs.iterrows():
            for power_model in POWER_MODELS:
                jobs.append((front, power_model, d["Modules"], d["Battery_kWh"], d["Tilt"], d["Azimuth"]))
    print(f"{len(jobs)} evaluations ({len(jobs) // len(POWER_MODELS)} designs x {len(POWER_MODELS)} power models)")

    with ProcessPoolExecutor(
        max_workers=args.n_procs,
        initializer=_init,
        initargs=(configs, str(args.rlp_directory), args.execution_backend),
    ) as pool:
        rows = list(pool.map(_run, jobs, chunksize=4))

    table = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    long_path = args.output / "task5f_replacement_year_sweep.csv"
    table.to_csv(long_path, index=False)

    key = ["Front", "Calendar_Model", "Modules", "Battery_kWh", "Tilt", "Azimuth"]
    # A plain pivot, not pivot_table(dropna=False): the latter expands the index
    # to the cartesian product of its levels and invents designs that were never run.
    wide = table.pivot(index=key, columns="Power_Model", values="Replacement_Year").reset_index()
    reps = table.pivot(index=key, columns="Power_Model", values="Total_Replacements").reset_index()
    for objective, suffix in ((GI, "GI"), (NPV, "NPV")):
        piv = table.pivot(index=key, columns="Power_Model", values=objective).reset_index()
        for arm in ("P1_1C_symmetric", "P2_0p5C_symmetric"):
            wide[f"{suffix}_Delta_{arm}"] = piv[arm] - piv["P0_constant_4352W"]
    for arm in ("P1_1C_symmetric", "P2_0p5C_symmetric"):
        wide[f"Total_Replacements_Delta_{arm}"] = reps[arm] - reps["P0_constant_4352W"]
    wide_path = args.output / "task5f_replacement_year_by_design.csv"
    wide.to_csv(wide_path, index=False)

    print("\n=== Part f: does the constant-power assumption move any replacement year? ===")
    base = "P0_constant_4352W"
    print(f"{len(wide)} distinct designs across the three constant-power fronts")
    print(f"  under {base}: {int(wide[base].notna().sum())} replaced, {int(wide[base].isna().sum())} never replaced")
    for label, arm in (("1 C", "P1_1C_symmetric"), ("0.5 C", "P2_0p5C_symmetric")):
        # NaN means never replaced; treat NaN-vs-NaN as agreement, NaN-vs-year as a change.
        same = (wide[arm].isna() & wide[base].isna()) | (wide[arm] == wide[base])
        moved = wide[~same]
        print(f"  {label} vs constant: {len(moved)} of {len(wide)} designs change replacement year")
        if len(moved):
            print(moved[key + [base, arm]].to_string(index=False))
        n_rep = wide[f"Total_Replacements_Delta_{arm}"].fillna(0)
        print(f"  {label} vs constant: {int((n_rep != 0).sum())} designs change replacement COUNT")
        print(
            f"  {label} vs constant: max |dGI| {wide[f'GI_Delta_{arm}'].abs().max():.6f} pp, "
            f"max |dNPV| {wide[f'NPV_Delta_{arm}'].abs().max():.4f} EUR"
        )

    provenance = {
        "breos_version": breos.__version__,
        "breos_source": repro._git_revision(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "power_model_configs": POWER_MODELS,
        "power_model_config_sha256": config_shas,
        "degradation_models": DEGRADATION,
        "designs_per_front": {f: int((table["Front"] == f).sum() // len(POWER_MODELS)) for f in DEGRADATION},
        "evaluations": len(jobs),
        "execution_backend": args.execution_backend,
        "outputs": {long_path.name: repro._sha256(long_path), wide_path.name: repro._sha256(wide_path)},
    }
    (args.output / "task5f_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"\nWrote {long_path}\nWrote {wide_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
