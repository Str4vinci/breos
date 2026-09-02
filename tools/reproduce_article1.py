#!/usr/bin/env python3
"""Reproduce projected optimization results for the forthcoming publication."""

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
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import breos  # noqa: E402
from breos.execution import (  # noqa: E402
    DEFAULT_EXECUTION_BACKEND,
    EXECUTION_BACKENDS,
    backend_provenance,
)
from breos.load_profiles import PROFILE_FILES, PROFILE_FILES_15MIN, load_profile  # noqa: E402
from breos.optimization import evaluate_projected_design, optimize_system_multi_objective  # noqa: E402
from breos.pv.model_options import resolve_configured_pv_model_options  # noqa: E402
from breos.pv_modules import get_module  # noqa: E402
from breos.weather import (  # noqa: E402
    read_weather_csv,
    resample_to_15min,
    weather_metadata,
    weather_representative_time_offset,
)

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


def _pv_module_provenance(config: dict) -> dict:
    """Resolve the electrical and physical PV inputs for the publication study."""
    pv = config["pv"]
    width = float(pv.get("module_width_m", 1.134))
    length = float(pv.get("module_length_m", 2.278))
    return {
        "catalog_key": str(pv["module"]),
        "parameters": asdict(get_module(str(pv["module"]))),
        "width_m": width,
        "length_m": length,
        "area_m2": width * length,
    }


def _display_path(path: Path) -> str:
    """Prefer a repository-relative input path without hiding external inputs."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _battery_cost_slug(cost: float) -> str:
    """Return a stable, filesystem-safe label for a battery-cost scenario."""
    return f"{cost:g}".replace(".", "p")


def _battery_cost_scenarios(requested: list[float] | None) -> list[float | None]:
    """Resolve optional CLI cost scenarios while preserving the default layout."""
    if not requested:
        return [None]
    scenarios: list[float] = []
    for cost in requested:
        if cost < 0.0:
            raise ValueError("--battery-cost must be non-negative")
        if cost not in scenarios:
            scenarios.append(cost)
    return scenarios


def _resolved_rlp_path(config: dict, rlp_directory: Path) -> Path | None:
    profile_type = str(config["load"]["profile_type"])
    if profile_type in {"1", "default", "demandlib_h0", "h0", "bdew_h0"}:
        return None
    resolution = str(config["simulation"]["resolution"])
    filename = PROFILE_FILES_15MIN[profile_type] if resolution == "15min" else PROFILE_FILES[profile_type]
    return rlp_directory / filename


def _load_inputs(
    config: dict,
    rlp_directory: Path,
    weather_override: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path | None]:
    simulation = config["simulation"]
    location = config["location"]
    load_config = config["load"]
    weather_path = weather_override or PROJECT_ROOT / simulation["weather_file"]
    rlp_path = _resolved_rlp_path(config, rlp_directory)
    uses_packaged_profile = rlp_path is None
    if not weather_path.is_file():
        raise FileNotFoundError(f"Weather file for the forthcoming publication not found: {weather_path}")
    if rlp_path is not None and not rlp_path.is_file():
        raise FileNotFoundError(
            "External RLP for the forthcoming publication not found: "
            f"{rlp_path}. Pass --rlp-directory with the licensed E-REDES file."
        )

    weather = read_weather_csv(weather_path)
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
        rlp_directory=None if uses_packaged_profile else str(rlp_directory),
        timezone=str(location["timezone"]),
    )
    return weather, load, weather_path, rlp_path


def _fixed_candidates(
    config: dict,
    weather: pd.DataFrame,
    load: pd.DataFrame,
    selected_labels: set[str],
    output_directory: Path,
    *,
    compare_reference: bool = True,
    execution_backend: str = DEFAULT_EXECUTION_BACKEND,
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
            execution_backend=execution_backend,
        )
        output = result.metrics
        reproduced_gi = float(output["Projected_Grid_Independence_%"])
        reproduced_npv = float(output["Projected_NPV_Eur"])
        reference_gi = (
            float(reference.get("projected_grid_independence_pct", float("nan"))) if compare_reference else float("nan")
        )
        reference_npv = float(reference.get("projected_npv_eur", float("nan"))) if compare_reference else float("nan")
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


def _select_pareto_representatives(pareto: pd.DataFrame) -> pd.DataFrame:
    """Select the five Article 1 designs used to describe the Pareto front."""
    if pareto.empty:
        raise ValueError("Cannot select representatives from an empty Pareto front")
    npv_column = "Projected_NPV_Eur"
    gi_column = "Projected_Grid_Independence_%"
    battery_column = "Battery_kWh"
    missing = {npv_column, gi_column, battery_column} - set(pareto.columns)
    if missing:
        raise ValueError(f"Pareto front is missing projected metric columns: {', '.join(sorted(missing))}")

    npv = pareto[npv_column].to_numpy(dtype=float)
    gi = pareto[gi_column].to_numpy(dtype=float)
    battery = pareto[battery_column].to_numpy(dtype=float)

    def maximum_position(values: np.ndarray, eligible: np.ndarray, description: str) -> int:
        positions = np.flatnonzero(eligible)
        if positions.size == 0:
            raise ValueError(f"Pareto front has no {description} design")
        return int(positions[int(np.argmax(values[positions]))])

    npv_span = float(np.max(npv) - np.min(npv))
    gi_span = float(np.max(gi) - np.min(gi))
    npv_normalized = (npv - np.min(npv)) / (npv_span if npv_span > 0.0 else 1.0)
    gi_normalized = (gi - np.min(gi)) / (gi_span if gi_span > 0.0 else 1.0)
    knee_distance = np.hypot(1.0 - npv_normalized, 1.0 - gi_normalized)
    selections = (
        ("max_npv", int(np.argmax(npv))),
        ("max_npv_battery", maximum_position(npv, battery > 0.0, "PV-plus-battery")),
        ("knee", int(np.argmin(knee_distance))),
        ("max_gi_positive_npv", maximum_position(gi, npv > 0.0, "positive-NPV")),
        ("max_gi", int(np.argmax(gi))),
    )

    rows = []
    for label, position in selections:
        row = pareto.iloc[position].to_dict()
        row = {"Representative": label, **row}
        rows.append(row)
    return pd.DataFrame(rows)


def _export_pareto_representatives(
    config: dict,
    weather: pd.DataFrame,
    load: pd.DataFrame,
    pareto: pd.DataFrame,
    output_directory: Path,
    execution_backend: str = DEFAULT_EXECUTION_BACKEND,
    representative_names: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    """Replay selected Pareto designs and export their plot-independent tables."""
    representatives = _select_pareto_representatives(pareto)
    if representative_names:
        available = set(representatives["Representative"])
        unknown = representative_names - available
        if unknown:
            raise ValueError(f"Unknown Pareto representative: {', '.join(sorted(unknown))}")
        representatives = representatives[representatives["Representative"].isin(representative_names)]
    summaries = []
    artifacts = {}
    for representative in representatives.to_dict(orient="records"):
        label = str(representative["Representative"])
        result = evaluate_projected_design(
            weather,
            load,
            config,
            n_modules=int(representative["Modules"]),
            battery_kwh=float(representative["Battery_kWh"]),
            tilt=float(representative["Tilt"]),
            azimuth=float(representative["Azimuth"]),
            execution_backend=execution_backend,
        )
        directory = output_directory / "representatives" / label
        directory.mkdir(parents=True, exist_ok=True)
        yearly_path = directory / "yearly_summary.csv"
        financial_path = directory / "cost_projection.csv"
        metrics_path = directory / "metrics.json"
        result.yearly.to_csv(yearly_path, index=False)
        result.financial.to_csv(financial_path, index=False)
        metrics_path.write_text(json.dumps(result.metrics, indent=2, default=str) + "\n")
        summaries.append({**representative, **result.metrics})
        artifacts[label] = {
            "yearly_summary": str(yearly_path.relative_to(output_directory)),
            "yearly_summary_sha256": _sha256(yearly_path),
            "cost_projection": str(financial_path.relative_to(output_directory)),
            "cost_projection_sha256": _sha256(financial_path),
            "metrics": str(metrics_path.relative_to(output_directory)),
            "metrics_sha256": _sha256(metrics_path),
        }
    return pd.DataFrame(summaries), artifacts


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
    parser.add_argument("--resolution", choices=("h", "15min"), help="Override simulation.resolution")
    parser.add_argument(
        "--load-profile",
        choices=("eredes", "h0"),
        help="Override the configured household profile; h0 uses BREOS packaged data",
    )
    parser.add_argument(
        "--battery-cost",
        action="append",
        type=float,
        help=(
            "Override costs.storage_cost_per_kwh. Repeat the option to run several "
            "cost scenarios in separate battery-cost-* directories."
        ),
    )
    optimization_group = parser.add_mutually_exclusive_group()
    optimization_group.add_argument("--smoke-optimization", action="store_true")
    optimization_group.add_argument("--full-optimization", action="store_true")
    parser.add_argument("--n-procs", type=int, default=1, help="Optimization worker processes")
    parser.add_argument(
        "--execution-backend",
        choices=EXECUTION_BACKENDS,
        default=DEFAULT_EXECUTION_BACKEND,
        help=(
            "Within-day dispatch implementation. 'python' is the numerical reference and the "
            "default; 'numba' is a compiled path that reproduces it bit for bit and needs "
            'pip install "breos[fast]".'
        ),
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/article1")
    args = parser.parse_args()

    config_bytes = args.config.read_bytes()
    base_config = tomllib.loads(config_bytes.decode("utf-8"))
    if args.calendar_model:
        base_config.setdefault("battery", {})["calendar_model"] = args.calendar_model
    if args.resolution:
        base_config.setdefault("simulation", {})["resolution"] = args.resolution
    if args.load_profile:
        base_config.setdefault("load", {})["profile_type"] = "6" if args.load_profile == "eredes" else "1"
    weather, load, weather_path, rlp_path = _load_inputs(base_config, args.rlp_directory, args.weather_file)
    configured_cost = float(base_config["costs"]["storage_cost_per_kwh"])

    for requested_cost in _battery_cost_scenarios(args.battery_cost):
        config = copy.deepcopy(base_config)
        scenario_output = args.output
        if requested_cost is not None:
            config["costs"]["storage_cost_per_kwh"] = requested_cost
            scenario_output = args.output / f"battery-cost-{_battery_cost_slug(requested_cost)}"
        scenario_output.mkdir(parents=True, exist_ok=True)
        scenario_cost = float(config["costs"]["storage_cost_per_kwh"])
        compare_reference = (
            scenario_cost == configured_cost
            and not args.calendar_model
            and not args.resolution
            and not args.load_profile
        )

        fixed = pd.DataFrame()
        fixed_path = scenario_output / "fixed_candidates.csv"
        if not args.skip_fixed:
            fixed = _fixed_candidates(
                config,
                weather,
                load,
                set(args.candidate),
                scenario_output,
                compare_reference=compare_reference,
                execution_backend=args.execution_backend,
            )
            fixed.to_csv(fixed_path, index=False)
            print(fixed.to_string(index=False))
            print(f"\nWrote {fixed_path}")

        report = {
            "breos_version": breos.__version__,
            "breos_source": _git_revision(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "dependency_versions": _dependency_versions(),
            "execution": backend_provenance(args.execution_backend),
            "command": shlex.join([sys.executable, *sys.argv]),
            "battery_cost_scenario_eur_per_kwh": scenario_cost,
            "research_commit": RESEARCH_COMMIT,
            "research_pareto_sha256": RESEARCH_PARETO_SHA256,
            "research_weather_sha256": RESEARCH_WEATHER_SHA256,
            "config": _display_path(args.config),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "resolved_config_sha256": hashlib.sha256(
                json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "resolved_config": config,
            "resolved_pv_module": _pv_module_provenance(config),
            "effective_runtime_pv_model_options": resolve_configured_pv_model_options(
                config, bifaciality=get_module(str(config["pv"]["module"])).bifaciality
            ),
            "weather_metadata": weather_metadata(weather),
            "solar_position_offset_minutes": weather_representative_time_offset(
                weather, str(config["simulation"]["resolution"])
            ).total_seconds()
            / 60.0,
            "input_weather_resolution": "h",
            "output_weather_resolution": str(config["simulation"]["resolution"]),
            "irradiance_resampling_method": (
                str(config["simulation"].get("irradiance_resampling", "clear_sky"))
                if str(config["simulation"]["resolution"]) == "15min"
                else "none"
            ),
            "weather": _display_path(weather_path),
            "weather_file_sha256": _sha256(weather_path),
            "weather_uncompressed_sha256": _sha256(
                weather_path,
                decompress_gzip=weather_path.suffix == ".gz",
            ),
            "external_rlp_filename": rlp_path.name if rlp_path is not None else None,
            "external_rlp_sha256": _sha256(rlp_path) if rlp_path is not None else None,
            "fixed_candidates": fixed.to_dict(orient="records"),
            "fixed_candidate_artifacts": (
                {
                    str(row["Label"]): {
                        "yearly_summary": f"candidates/{str(row['Label']).lower()}/yearly_summary.csv",
                        "yearly_summary_sha256": _sha256(
                            scenario_output / "candidates" / str(row["Label"]).lower() / "yearly_summary.csv"
                        ),
                        "cost_projection": f"candidates/{str(row['Label']).lower()}/cost_projection.csv",
                        "cost_projection_sha256": _sha256(
                            scenario_output / "candidates" / str(row["Label"]).lower() / "cost_projection.csv"
                        ),
                        "metrics": f"candidates/{str(row['Label']).lower()}/metrics.json",
                        "metrics_sha256": _sha256(
                            scenario_output / "candidates" / str(row["Label"]).lower() / "metrics.json"
                        ),
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
                results_dir=str(scenario_output / "optimization"),
                pop_size=int(optimization["pop_size"]),
                n_gen=int(optimization["n_gen"]),
                n_offsprings=int(optimization["n_offsprings"]),
                seed=int(optimization["seed"]),
                verbose=True,
                n_procs=args.n_procs,
                execution_backend=args.execution_backend,
            )
            pareto_path = scenario_output / "pareto_results.csv"
            pareto = result.details["pareto"]
            pareto.to_csv(pareto_path, index=False)
            representatives, representative_artifacts = _export_pareto_representatives(
                config,
                weather,
                load,
                pareto,
                scenario_output,
                execution_backend=args.execution_backend,
            )
            representatives_path = scenario_output / "pareto_representatives.csv"
            representatives.to_csv(representatives_path, index=False)
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
                "representatives_csv": representatives_path.name,
                "representatives_sha256": _sha256(representatives_path),
                "representative_artifacts": representative_artifacts,
            }
            print(f"Wrote {pareto_path}")
            print(f"Wrote {representatives_path}")

        report_path = scenario_output / "reproduction.json"
        report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
