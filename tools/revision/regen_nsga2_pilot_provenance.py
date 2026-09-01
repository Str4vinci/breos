#!/usr/bin/env python3
"""Rebuild provenance for a completed NSGA-II pilot without re-simulating.

The Batch B3 pilot wrote every CSV after generation 40, then raised while
formatting a repository-relative ``--spec`` path against an absolute project
root. This tool validates the saved scientific outputs and reconstructs the
missing provenance record. It does not run the optimizer or the simulator.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import shlex
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

_repro_spec = importlib.util.spec_from_file_location("article1_repro", PROJECT_ROOT / "tools/reproduce_article1.py")
repro = importlib.util.module_from_spec(_repro_spec)
assert _repro_spec.loader is not None
_repro_spec.loader.exec_module(repro)

import breos  # noqa: E402
from breos.execution import backend_provenance, require_backend  # noqa: E402

OUTPUT_NAMES = (
    "evaluations.csv",
    "final_population.csv",
    "final_population_front.csv",
    "external_archive.csv",
    "first_published_budget_front.csv",
    "comparison.csv",
    "generation_metrics.csv",
)
DESIGN_COLUMNS = ["Modules", "Battery_kWh", "Tilt", "Azimuth"]


def _load_pilot(path: Path):
    module_spec = importlib.util.spec_from_file_location("nsga2_archive_pilot", path)
    if module_spec is None or module_spec.loader is None:
        raise SystemExit(f"cannot import pilot implementation from {path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pilot_source_at_run(fixed_source: bytes) -> bytes:
    """Restore the one path expression that was present during the failed run."""
    fixed = b'"pilot_spec": repro._display_path(args.spec),'
    original = b'"pilot_spec": str(args.spec.relative_to(PROJECT_ROOT)),'
    if fixed_source.count(fixed) != 1:
        raise SystemExit("cannot reconstruct the pilot source used by the run")
    return fixed_source.replace(fixed, original, 1)


def _sorted(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in DESIGN_COLUMNS if column in frame]
    return frame.sort_values(columns, kind="stable").reset_index(drop=True)


def _assert_frame_matches(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    label: str,
    *,
    tolerance: float = 1e-12,
) -> float:
    actual = _sorted(actual)
    expected = _sorted(expected)
    if list(actual.columns) != list(expected.columns):
        raise SystemExit(f"{label}: columns differ: actual={list(actual.columns)}, expected={list(expected.columns)}")
    try:
        pd.testing.assert_frame_equal(
            actual,
            expected,
            check_dtype=False,
            check_exact=False,
            rtol=tolerance,
            atol=tolerance,
        )
    except AssertionError as error:
        raise SystemExit(f"{label}: saved output does not recompute\n{error}") from error

    numeric = actual.select_dtypes(include=[np.number]).columns
    if not len(numeric):
        return 0.0
    left = actual[numeric].to_numpy(float)
    right = expected[numeric].to_numpy(float)
    return float(np.nanmax(np.abs(left - right))) if left.size else 0.0


def _validate_evaluation_ledger(evaluations: pd.DataFrame, settings: dict) -> dict[str, object]:
    population = int(settings["pop_size"])
    offspring = int(settings["n_offsprings"])
    generations = int(settings["n_gen"])
    expected_rows = population + (generations - 1) * offspring
    if len(evaluations) != expected_rows:
        raise SystemExit(f"evaluation ledger has {len(evaluations)} rows; expected {expected_rows}")

    sequence = evaluations["Evaluation"].to_numpy(int)
    if not np.array_equal(sequence, np.arange(1, expected_rows + 1)):
        raise SystemExit("evaluation ledger is not sequential from 1 through the expected budget")

    counts = evaluations.groupby("Generation", sort=True).size().to_dict()
    expected_counts = {1: population, **{generation: offspring for generation in range(2, generations + 1)}}
    if counts != expected_counts:
        raise SystemExit(f"generation counts differ: actual={counts}, expected={expected_counts}")
    return {
        "expected_evaluations": expected_rows,
        "generations": generations,
        "generation_row_counts": counts,
    }


def _recompute_analysis(pilot, spec: dict, frames: dict[str, pd.DataFrame]) -> tuple[dict, dict]:
    settings = spec["optimization"]
    evaluations = frames["evaluations.csv"]
    ledger_validation = _validate_evaluation_ledger(evaluations, settings)

    comparison = spec["comparison"]
    exhaustive = pd.read_csv(PROJECT_ROOT / comparison["exhaustive_front"])
    original = pd.read_csv(PROJECT_ROOT / comparison["original_nsga_front"])
    original["Constraint_Violation"] = 0.0
    original["Feasible"] = True
    budget = int(comparison["original_evaluation_budget"])

    exhaustive_f = pilot.objective_array(exhaustive)
    ideal = exhaustive_f.min(axis=0)
    span = exhaustive_f.max(axis=0) - ideal
    span[span == 0.0] = 1.0
    ref_point = np.asarray(comparison["hypervolume_reference_normalized"], dtype=float)

    from pymoo.indicators.hv import HV

    exhaustive_hv = float(HV(ref_point=ref_point)((exhaustive_f - ideal) / span))
    budget_front = pilot.nondominated_from_evaluations(evaluations.iloc[:budget].copy())
    archive = pilot.nondominated_from_evaluations(evaluations.copy())
    population_front = pilot.nondominated_from_evaluations(frames["final_population.csv"].copy())

    deltas = {
        "first_published_budget_front_max_abs_delta": _assert_frame_matches(
            frames["first_published_budget_front.csv"],
            budget_front,
            "first published-budget front",
        ),
        "external_archive_max_abs_delta": _assert_frame_matches(
            frames["external_archive.csv"],
            archive[frames["external_archive.csv"].columns],
            "external archive",
        ),
        "final_population_front_max_abs_delta": _assert_frame_matches(
            frames["final_population_front.csv"],
            population_front,
            "final population front",
        ),
    }

    rows = [
        pilot.compare_front("published_nsga", original, exhaustive, ideal, span, ref_point, exhaustive_hv),
        pilot.compare_front(
            f"improved_first_{budget}_evaluations",
            budget_front,
            exhaustive,
            ideal,
            span,
            ref_point,
            exhaustive_hv,
        ),
        pilot.compare_front(
            "improved_final_population",
            frames["final_population_front.csv"],
            exhaustive,
            ideal,
            span,
            ref_point,
            exhaustive_hv,
        ),
        pilot.compare_front(
            "improved_external_archive",
            frames["external_archive.csv"],
            exhaustive,
            ideal,
            span,
            ref_point,
            exhaustive_hv,
        ),
    ]
    deltas["comparison_max_abs_delta"] = _assert_frame_matches(
        frames["comparison.csv"], pd.DataFrame(rows), "comparison table"
    )

    generation_rows = []
    for generation in range(1, int(settings["n_gen"]) + 1):
        through_generation = evaluations[evaluations["Generation"] <= generation]
        front = pilot.nondominated_from_evaluations(through_generation.copy())
        row = pilot.compare_front(
            f"generation_{generation}",
            front,
            exhaustive,
            ideal,
            span,
            ref_point,
            exhaustive_hv,
        )
        generation_rows.append({"Generation": generation, "Evaluations": len(through_generation), **row})
    deltas["generation_metrics_max_abs_delta"] = _assert_frame_matches(
        frames["generation_metrics.csv"],
        pd.DataFrame(generation_rows),
        "generation metrics",
    )

    validation = {
        **ledger_validation,
        "saved_outputs_recomputed": True,
        "maximum_absolute_numeric_delta": max(deltas.values()),
        "checks": deltas,
    }
    hypervolume = {
        "objectives": ["1 - GI_fraction", "-NPV_Eur"],
        "normalization_ideal": ideal.tolist(),
        "normalization_span": span.tolist(),
        "reference_point": ref_point.tolist(),
    }
    return validation, hypervolume


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--pilot-script", type=Path, required=True)
    parser.add_argument("--run-commit", required=True, help="commit checked out when the pilot ran")
    parser.add_argument(
        "--run-tracked-worktree-dirty",
        action="store_true",
        help="record that tracked files were dirty when the pilot ran",
    )
    parser.add_argument("--queue-status", type=Path)
    args = parser.parse_args()

    spec_path = args.spec.resolve()
    pilot_path = args.pilot_script.resolve()
    spec_bytes = spec_path.read_bytes()
    spec = tomllib.loads(spec_bytes.decode("utf-8"))
    output = (PROJECT_ROOT / spec["output"]).resolve()
    frames = {name: pd.read_csv(output / name) for name in OUTPUT_NAMES}

    pilot = _load_pilot(pilot_path)
    validation, hypervolume = _recompute_analysis(pilot, spec, frames)
    fixed_pilot_source = pilot_path.read_bytes()
    run_pilot_source = _pilot_source_at_run(fixed_pilot_source)

    base_path = PROJECT_ROOT / spec["base_config"]
    base_bytes = base_path.read_bytes()
    base_config = tomllib.loads(base_bytes.decode("utf-8"))
    base_config.setdefault("battery", {})["calendar_model"] = spec["calendar_model"]
    base_config.setdefault("optimization", {})["early_stop"] = False
    require_backend(spec["execution_backend"])
    _, _, weather_path, rlp_path = repro._load_inputs(base_config, PROJECT_ROOT / spec["rlp_directory"])

    queue_status = None
    if args.queue_status:
        queue_status = args.queue_status.read_text().strip()
    outputs = {name: pilot.sha256(output / name) for name in OUTPUT_NAMES}
    settings = spec["optimization"]
    provenance = {
        "purpose": "NSGA-II algorithm diagnostic matched to the constant-power v1 lattice",
        "not_a_physical_model_result": True,
        "provenance_regenerated": {
            "reason": (
                "The optimizer completed and wrote every CSV, then the original provenance "
                "writer called Path.relative_to() with a relative spec and an absolute project root."
            ),
            "tool": repro._display_path(Path(__file__)),
            "regenerated_at": datetime.now(UTC).isoformat(),
            "regenerated_at_commit": _git_commit(),
            "original_timing_not_recoverable": True,
            "queue_status": queue_status,
        },
        "reconstruction_command": shlex.join([sys.executable, *sys.argv]),
        "original_command": shlex.join(
            [sys.executable, repro._display_path(pilot_path), "--spec", repro._display_path(spec_path)]
        ),
        "python_version": platform.python_version(),
        "breos_version": breos.__version__,
        "breos_source": {
            "commit": args.run_commit,
            "tracked_worktree_dirty": args.run_tracked_worktree_dirty,
        },
        "execution": backend_provenance(spec["execution_backend"]),
        "pilot_script": repro._display_path(pilot_path),
        "pilot_script_sha256_at_run": _sha256_bytes(run_pilot_source),
        "pilot_script_sha256_after_path_fix": _sha256_bytes(fixed_pilot_source),
        "pilot_script_post_run_change": (
            "Only pilot_spec path display changed: the failing relative_to call now uses "
            "the repository's resolve-first _display_path helper."
        ),
        "pilot_spec": repro._display_path(spec_path),
        "pilot_spec_sha256": _sha256_bytes(spec_bytes),
        "base_config": spec["base_config"],
        "base_config_sha256": _sha256_bytes(base_bytes),
        "config_diff": {
            "battery.calendar_model": spec["calendar_model"],
            "optimization.pop_size": int(settings["pop_size"]),
            "optimization.n_offsprings": int(settings["n_offsprings"]),
            "optimization.n_gen": int(settings["n_gen"]),
            "optimization.seed": int(settings["seed"]),
            "optimization.early_stop": False,
            "optimization.external_archive": settings["external_archive"],
        },
        "weather": repro._display_path(weather_path),
        "weather_sha256": pilot.sha256(weather_path),
        "rlp": repro._display_path(rlp_path),
        "rlp_sha256": pilot.sha256(rlp_path),
        "evaluations": int(len(frames["evaluations.csv"])),
        "unique_evaluated_designs": len(pilot.design_keys(frames["evaluations.csv"])),
        "scientific_output_validation": validation,
        "outputs": outputs,
        "hypervolume_definition": hypervolume,
    }
    provenance_path = output / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        f"Validated {provenance['evaluations']} evaluations through generation "
        f"{validation['generations']}; maximum recomputation delta "
        f"{validation['maximum_absolute_numeric_delta']:.3e}"
    )
    print(f"Wrote {provenance_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
