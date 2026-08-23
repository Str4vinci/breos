#!/usr/bin/env python3
"""Reproduce the deterministic Article 1 projected optimization results."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import importlib.metadata
import json
import platform
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import breos  # noqa: E402
from breos.load_profiles import load_profile  # noqa: E402
from breos.optimization import evaluate_projected_design, optimize_system_multi_objective  # noqa: E402
from breos.weather import resample_to_15min  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "validation/article1/article1-projected-optimization.toml"
RESEARCH_COMMIT = "a0db6aae1e8d04a8260f51a34543b23bd82a1762"
RESEARCH_PARETO_SHA256 = "5334b8361b2395f0f19b6839005964b0b61bfa0d00e5ea28f450cfb4cde0a225"
RESEARCH_WEATHER_SHA256 = "d2258dc7ea0d6432a6ddf69f748e67e88a36b235a906d5c6ab7da96e8e6911e0"


def _sha256(path: Path, *, decompress_gzip: bool = False) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if decompress_gzip else Path.open
    with opener(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> dict[str, str | bool]:
    """Return the exact source revision and whether tracked files differ from it."""
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked_status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=no"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"commit": revision, "tracked_worktree_dirty": bool(tracked_status)}


def _dependency_versions() -> dict[str, str]:
    """Record the numerical packages that can affect deterministic output."""
    return {package: importlib.metadata.version(package) for package in ("numpy", "pandas", "pvlib", "scipy", "pymoo")}


def _display_path(path: Path) -> str:
    """Prefer a repository-relative input path without hiding external inputs."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _load_inputs(
    config: dict,
    rlp_directory: Path,
    weather_override: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    simulation = config["simulation"]
    location = config["location"]
    load_config = config["load"]
    weather_path = weather_override or PROJECT_ROOT / simulation["weather_file"]
    rlp_path = rlp_directory / load_config["external_filename"]
    if not weather_path.is_file():
        raise FileNotFoundError(f"Article 1 weather file not found: {weather_path}")
    if not rlp_path.is_file():
        raise FileNotFoundError(
            f"Article 1 external RLP not found: {rlp_path}. Pass --rlp-directory with the licensed E-REDES file."
        )

    weather = pd.read_csv(weather_path, index_col=0)
    weather.index = pd.to_datetime(weather.index, utc=True)
    if simulation["resolution"] == "15min":
        resampling = simulation.get("irradiance_resampling", "clear_sky")
        if resampling not in {"clear_sky", "clear_sky_energy_conserving"}:
            raise ValueError(f"Unsupported simulation.irradiance_resampling: {resampling!r}")
        weather = resample_to_15min(
            weather,
            latitude=float(location["latitude"]),
            longitude=float(location["longitude"]),
            preserve_irradiance_energy=resampling == "clear_sky_energy_conserving",
        )
    load = load_profile(
        str(load_config["profile_type"]),
        float(load_config["annual_consumption_kwh"]),
        start_date=str(weather.index[0].year) + "-01-01",
        freq=str(simulation["resolution"]),
        rlp_directory=str(rlp_directory),
        timezone=str(location["timezone"]),
    )
    return weather, load, weather_path, rlp_path


def _fixed_candidates(
    config: dict,
    weather: pd.DataFrame,
    load: pd.DataFrame,
    selected_labels: set[str],
    output_directory: Path,
) -> pd.DataFrame:
    rows = []
    for reference in config["reference_candidates"]:
        if selected_labels and reference["label"] not in selected_labels:
            continue
        result = evaluate_projected_design(
            weather,
            load,
            config,
            n_modules=int(reference["modules"]),
            battery_kwh=float(reference["battery_kwh"]),
            tilt=float(reference["tilt"]),
            azimuth=float(reference["azimuth"]),
        )
        output = result.metrics
        reproduced_gi = float(output["Projected_Grid_Independence_%"])
        reproduced_npv = float(output["Projected_NPV_Eur"])
        reference_gi = float(reference.get("projected_grid_independence_pct", float("nan")))
        reference_npv = float(reference.get("projected_npv_eur", float("nan")))
        row = {
            "Label": reference["label"],
            "Modules": int(reference["modules"]),
            "Battery_kWh": float(reference["battery_kwh"]),
            "Tilt": float(reference["tilt"]),
            "Azimuth": float(reference["azimuth"]),
            "Reference_Projected_GI_%": reference_gi,
            "BREOS_Projected_GI_%": reproduced_gi,
            "GI_Difference_pp": reproduced_gi - reference_gi,
            "Reference_Projected_NPV_Eur": reference_npv,
            "BREOS_Projected_NPV_Eur": reproduced_npv,
            "NPV_Difference_Eur": reproduced_npv - reference_npv,
        }
        for key, value in output.items():
            if key.startswith("Projected_"):
                row[key] = value
        rows.append(row)

        candidate_directory = output_directory / "candidates" / str(reference["label"]).lower()
        candidate_directory.mkdir(parents=True, exist_ok=True)
        result.yearly.to_csv(candidate_directory / "yearly_summary.csv", index=False)
        result.financial.to_csv(candidate_directory / "cost_projection.csv", index=False)
        (candidate_directory / "metrics.json").write_text(json.dumps(output, indent=2, default=str) + "\n")
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rlp-directory", type=Path, required=True)
    parser.add_argument("--weather-file", type=Path, help="Override the configured weather file")
    parser.add_argument("--candidate", action="append", default=[], help="Run only the named fixed candidate")
    parser.add_argument("--skip-fixed", action="store_true", help="Skip fixed-design evaluations")
    parser.add_argument(
        "--calendar-model",
        help="Override battery.calendar_model (for example naumann_lam_field_calibrated_v2 or naumann_lam)",
    )
    optimization_group = parser.add_mutually_exclusive_group()
    optimization_group.add_argument("--smoke-optimization", action="store_true")
    optimization_group.add_argument("--full-optimization", action="store_true")
    parser.add_argument("--n-procs", type=int, default=1, help="Optimization worker processes")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/article1")
    args = parser.parse_args()

    config_bytes = args.config.read_bytes()
    config = tomllib.loads(config_bytes.decode("utf-8"))
    if args.calendar_model:
        config.setdefault("battery", {})["calendar_model"] = args.calendar_model
    weather, load, weather_path, rlp_path = _load_inputs(config, args.rlp_directory, args.weather_file)
    args.output.mkdir(parents=True, exist_ok=True)

    fixed = pd.DataFrame()
    fixed_path = args.output / "fixed_candidates.csv"
    if not args.skip_fixed:
        fixed = _fixed_candidates(config, weather, load, set(args.candidate), args.output)
        fixed.to_csv(fixed_path, index=False)
        print(fixed.to_string(index=False))
        print(f"\nWrote {fixed_path}")

    report = {
        "breos_version": breos.__version__,
        "breos_source": _git_revision(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "dependency_versions": _dependency_versions(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "research_commit": RESEARCH_COMMIT,
        "research_pareto_sha256": RESEARCH_PARETO_SHA256,
        "research_weather_sha256": RESEARCH_WEATHER_SHA256,
        "config": _display_path(args.config),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "resolved_config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "resolved_config": config,
        "weather": _display_path(weather_path),
        "weather_file_sha256": _sha256(weather_path),
        "weather_uncompressed_sha256": _sha256(
            weather_path,
            decompress_gzip=weather_path.suffix == ".gz",
        ),
        "external_rlp_filename": rlp_path.name,
        "external_rlp_sha256": _sha256(rlp_path),
        "fixed_candidates": fixed.to_dict(orient="records"),
        "fixed_candidate_artifacts": (
            {
                str(row["Label"]): {
                    "yearly_summary": f"candidates/{str(row['Label']).lower()}/yearly_summary.csv",
                    "yearly_summary_sha256": _sha256(
                        args.output / "candidates" / str(row["Label"]).lower() / "yearly_summary.csv"
                    ),
                    "cost_projection": f"candidates/{str(row['Label']).lower()}/cost_projection.csv",
                    "cost_projection_sha256": _sha256(
                        args.output / "candidates" / str(row["Label"]).lower() / "cost_projection.csv"
                    ),
                    "metrics": f"candidates/{str(row['Label']).lower()}/metrics.json",
                    "metrics_sha256": _sha256(args.output / "candidates" / str(row["Label"]).lower() / "metrics.json"),
                }
                for row in fixed.to_dict(orient="records")
            }
            if not fixed.empty
            else {}
        ),
    }

    if args.full_optimization or args.smoke_optimization:
        run_config = copy.deepcopy(config)
        optimization = run_config["optimization"]
        if args.smoke_optimization:
            optimization.update({"pop_size": 4, "n_offsprings": 2, "n_gen": 1, "early_stop": False})
        result = optimize_system_multi_objective(
            weather,
            load,
            run_config,
            results_dir=str(args.output / "optimization"),
            pop_size=int(optimization["pop_size"]),
            n_gen=int(optimization["n_gen"]),
            n_offsprings=int(optimization["n_offsprings"]),
            seed=int(optimization["seed"]),
            verbose=True,
            n_procs=args.n_procs,
        )
        pareto_path = args.output / "pareto_results.csv"
        result.details["pareto"].to_csv(pareto_path, index=False)
        report["optimization"] = {
            "run_type": "full" if args.full_optimization else "smoke",
            "settings": optimization,
            "iterations": result.iterations,
            "objective_basis": result.details["objective_basis"],
            "objective_names": result.details["objective_names"],
            "early_stop": result.details["early_stop"],
            "n_procs": result.details["n_procs"],
            "pareto_csv": pareto_path.name,
            "pareto_sha256": _sha256(pareto_path),
        }
        print(f"Wrote {pareto_path}")

    report_path = args.output / "reproduction.json"
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
