#!/usr/bin/env python3
"""Replay selected Article 1 designs from an existing Pareto result package."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

import breos  # noqa: E402
from breos.execution import (  # noqa: E402
    DEFAULT_EXECUTION_BACKEND,
    EXECUTION_BACKENDS,
    backend_provenance,
)
from tools.reproduce_article1 import (  # noqa: E402
    _dependency_versions,
    _display_path,
    _export_pareto_representatives,
    _git_revision,
    _load_inputs,
    _select_pareto_representatives,
    _sha256,
)

REPRESENTATIVE_NAMES = (
    "max_npv",
    "max_npv_battery",
    "knee",
    "max_gi_positive_npv",
    "max_gi",
)


def _source_files(source_directory: Path) -> tuple[Path, Path]:
    pareto_path = source_directory / "pareto_results.csv"
    reproduction_path = source_directory / "reproduction.json"
    for path in (pareto_path, reproduction_path):
        if not path.is_file():
            raise FileNotFoundError(f"Article 1 source result is missing: {path}")
    return pareto_path, reproduction_path


def _verify_source_pareto(report: dict, pareto_path: Path) -> str:
    actual = _sha256(pareto_path)
    expected = report.get("optimization", {}).get("pareto_sha256")
    if not expected:
        raise ValueError("Source reproduction.json does not record optimization.pareto_sha256")
    if actual != expected:
        raise ValueError(f"Source Pareto hash mismatch: expected {expected}, got {actual}")
    return actual


def _refuse_existing_outputs(output_directory: Path, names: set[str]) -> None:
    for name in names:
        directory = output_directory / "representatives" / name
        existing = [path for path in directory.glob("*") if path.is_file()] if directory.is_dir() else []
        if existing:
            raise FileExistsError(f"Representative output already exists: {directory}")


def _verify_replayed_representatives(source: pd.DataFrame, replayed: pd.DataFrame) -> dict[str, float]:
    """Require the replay to reproduce each selected source row."""
    comparison = source.merge(
        replayed,
        on="Representative",
        suffixes=("_source", "_replay"),
        validate="one_to_one",
    )
    exact_columns = ("Modules", "Battery_kWh", "Tilt", "Azimuth")
    metric_columns = ("Projected_Grid_Independence_%", "Projected_NPV_Eur")
    for column in exact_columns:
        if not comparison[f"{column}_source"].eq(comparison[f"{column}_replay"]).all():
            raise ValueError(f"Representative replay changed {column}")
    deltas = {
        column: float((comparison[f"{column}_source"] - comparison[f"{column}_replay"]).abs().max())
        for column in metric_columns
    }
    for column, delta in deltas.items():
        if delta > 1e-9:
            raise ValueError(f"Representative replay changed {column} by up to {delta:.3e}")
    return deltas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_directory", type=Path, help="Existing result directory with Pareto and provenance files"
    )
    parser.add_argument("--rlp-directory", type=Path, required=True)
    parser.add_argument(
        "--weather-file", type=Path, help="Override the weather file recorded in the source configuration"
    )
    parser.add_argument(
        "--representative",
        action="append",
        choices=REPRESENTATIVE_NAMES,
        required=True,
        help="Representative to replay. Repeat this option to run more than one.",
    )
    parser.add_argument(
        "--execution-backend",
        choices=EXECUTION_BACKENDS,
        default=DEFAULT_EXECUTION_BACKEND,
    )
    parser.add_argument("--output", type=Path, help="Output directory. Defaults to the source result directory.")
    args = parser.parse_args()

    source_directory = args.source_directory.resolve()
    output_directory = (args.output or source_directory).resolve()
    pareto_path, reproduction_path = _source_files(source_directory)
    source_report = json.loads(reproduction_path.read_text())
    config = source_report.get("resolved_config")
    if not isinstance(config, dict):
        raise ValueError("Source reproduction.json does not contain a resolved_config object")
    pareto_hash = _verify_source_pareto(source_report, pareto_path)
    pareto = pd.read_csv(pareto_path)
    selected_names = set(args.representative)
    selected = _select_pareto_representatives(pareto)
    selected = selected[selected["Representative"].isin(selected_names)]
    if len(selected) != len(selected_names):
        missing = selected_names - set(selected["Representative"])
        raise ValueError(f"Could not select Pareto representative: {', '.join(sorted(missing))}")

    output_directory.mkdir(parents=True, exist_ok=True)
    _refuse_existing_outputs(output_directory, selected_names)
    replay_csv = output_directory / "representative_replay.csv"
    replay_report = output_directory / "representative_replay.json"
    for path in (replay_csv, replay_report):
        if path.exists():
            raise FileExistsError(f"Replay output already exists: {path}")

    weather, load, weather_path, rlp_path = _load_inputs(config, args.rlp_directory, args.weather_file)
    representatives, artifacts = _export_pareto_representatives(
        config,
        weather,
        load,
        pareto,
        output_directory,
        execution_backend=args.execution_backend,
        representative_names=selected_names,
    )
    source_replay_delta = _verify_replayed_representatives(selected, representatives)
    representatives.to_csv(replay_csv, index=False)

    report = {
        "breos_version": breos.__version__,
        "breos_source": _git_revision(),
        "dependency_versions": _dependency_versions(),
        "execution": backend_provenance(args.execution_backend),
        "command": shlex.join([sys.executable, *sys.argv]),
        "source_reproduction": _display_path(reproduction_path),
        "source_reproduction_sha256": _sha256(reproduction_path),
        "source_breos_source": source_report.get("breos_source"),
        "source_pareto": _display_path(pareto_path),
        "source_pareto_sha256": pareto_hash,
        "source_resolved_config": config,
        "weather": _display_path(weather_path),
        "weather_file_sha256": _sha256(weather_path),
        "external_rlp_filename": rlp_path.name if rlp_path is not None else None,
        "external_rlp_sha256": _sha256(rlp_path) if rlp_path is not None else None,
        "representatives_csv": replay_csv.name,
        "representatives_sha256": _sha256(replay_csv),
        "representatives": representatives.to_dict(orient="records"),
        "source_replay_max_abs_delta": source_replay_delta,
        "representative_artifacts": artifacts,
    }
    replay_report.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(representatives.to_string(index=False))
    print(f"\nWrote {replay_csv}")
    print(f"Wrote {replay_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
