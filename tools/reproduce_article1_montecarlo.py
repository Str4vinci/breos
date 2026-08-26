#!/usr/bin/env python3
"""Generate Monte Carlo source tables for the forthcoming publication study with public BREOS APIs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import shlex
import subprocess
import sys
import tomllib
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import breos  # noqa: E402
from breos.montecarlo import MonteCarloSettings, run_montecarlo  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "validation/article1/article1-montecarlo.toml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> dict[str, str | bool]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=no"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"commit": revision, "tracked_worktree_dirty": bool(status)}


def _selected_cases(cases: dict, requested: list[str]) -> list[str]:
    if not requested or any(value.lower() == "all" for value in requested):
        return list(cases)
    normalized = {value.upper() for value in requested}
    unknown = normalized - set(cases)
    if unknown:
        raise ValueError(f"Unknown case(s): {', '.join(sorted(unknown))}; choose from {', '.join(cases)} or all")
    return [case for case in cases if case in normalized]


def _settings(config: dict, args: argparse.Namespace) -> MonteCarloSettings:
    values = config["montecarlo"]
    return MonteCarloSettings(
        weather_file=str(args.weather_file),
        n_runs=int(args.runs if args.runs is not None else values["n_runs"]),
        years_per_run=int(values["years_per_run"]),
        load_uncertainty=float(values["load_uncertainty"]),
        load_distribution=str(values["load_distribution"]),
        target_year=int(values["target_year"]),
        weather_start_year=int(values["weather_start_year"]),
        weather_end_year=int(values["weather_end_year"]),
        seed=int(values["seed"]),
        min_load_scale=float(values["min_load_scale"]),
        max_load_scale=float(values["max_load_scale"]),
        preserve_irradiance_energy=bool(values["preserve_irradiance_energy"]),
        collect_yearly=bool(values["collect_yearly"]),
        n_procs=int(args.n_procs if args.n_procs is not None else values["n_procs"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rlp-directory", type=Path, required=True)
    parser.add_argument("--weather-file", type=Path, required=True)
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        help="C1-C5 or all; repeat to select cases",
    )
    parser.add_argument("--calendar-model", help="Override the configured native calendar-degradation model")
    parser.add_argument("--runs", type=int, help="Override the configured number of trajectories")
    parser.add_argument("--n-procs", type=int, help="Worker processes for independent trajectories")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/article1-montecarlo")
    args = parser.parse_args()

    config_bytes = args.config.read_bytes()
    full_config = tomllib.loads(config_bytes.decode("utf-8"))
    cases = full_config.pop("cases")
    if args.calendar_model:
        full_config["calendar_model"] = args.calendar_model
    full_config["rlp_directory"] = str(args.rlp_directory.resolve())
    settings = _settings(full_config, args)

    weather_file = args.weather_file.resolve()
    if not weather_file.is_file():
        raise FileNotFoundError(weather_file)
    rlp_file = args.rlp_directory.resolve() / "EREDES_2025_BTN_1000kwh_15min.csv"
    if not rlp_file.is_file():
        raise FileNotFoundError(rlp_file)

    selected = _selected_cases(cases, args.case)
    for case_id in selected:
        case = cases[case_id]
        case_config = copy.deepcopy(full_config)
        case_config.update(
            {
                "n_modules": int(case["n_modules"]),
                "battery_kwh": float(case["battery_kwh"]),
                "tilt": float(case["tilt"]),
                "azimuth": float(case["azimuth"]),
            }
        )
        result = run_montecarlo(case_config, settings)
        case_directory = args.output / case_id.lower()
        case_directory.mkdir(parents=True, exist_ok=True)
        runs_path = case_directory / "runs.csv"
        yearly_path = case_directory / "yearly.csv"
        summary_path = case_directory / "summary.json"
        provenance_path = case_directory / "provenance.json"
        result.runs.to_csv(runs_path, index=False)
        if result.yearly is None:
            raise RuntimeError("The Monte Carlo run for the forthcoming publication requires collect_yearly=true")
        result.yearly.to_csv(yearly_path, index=False)
        summary_path.write_text(json.dumps(result.summary, indent=2) + "\n")

        resolved_hash = hashlib.sha256(
            json.dumps(case_config, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()
        report = {
            "case": case_id,
            "case_label": case["label"],
            "breos_version": breos.__version__,
            "breos_source": _git_revision(),
            "python_version": platform.python_version(),
            "command": shlex.join([sys.executable, *sys.argv]),
            "base_config": str(args.config),
            "base_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "resolved_config_sha256": resolved_hash,
            "resolved_config": case_config,
            "settings": asdict(settings),
            "available_weather_years": result.available_years,
            "weather_file": str(weather_file),
            "weather_file_sha256": _sha256(weather_file),
            "external_rlp_file": rlp_file.name,
            "external_rlp_sha256": _sha256(rlp_file),
            "runs_csv": runs_path.name,
            "runs_csv_sha256": _sha256(runs_path),
            "yearly_csv": yearly_path.name,
            "yearly_csv_sha256": _sha256(yearly_path),
            "summary_json": summary_path.name,
            "summary_json_sha256": _sha256(summary_path),
            "montecarlo_provenance": result.provenance,
        }
        provenance_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(f"{case_id}: wrote {case_directory}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
