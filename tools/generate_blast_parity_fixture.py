"""Regenerate the reviewed BLAST-Lite multicondition parity fixture.

This is an explicit maintainer tool, not part of ordinary tests or CI.  It
loads models from a clean, separately checked-out BLAST-Lite repository so the
fixture measures unmodified upstream behavior rather than BREOS's vendored
copy.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "blast" / "blast_parity_multicondition.json"
MANIFEST_PATH = FIXTURE_PATH.with_suffix(".manifest.json")

PINNED_BLAST_VERSION = "1.1.0"
PINNED_BLAST_COMMIT = "d789e00bca60f628de640745c18eb724b07358bd"
FIXTURE_SCHEMA = "blast-lite-breos-parity-multicondition-v1"
MANIFEST_SCHEMA = "breos-blast-parity-generation-manifest-v1"
DAYS_PER_CONDITION = 60
GENERATOR_PATH = "tools/generate_blast_parity_fixture.py"
CANONICAL_COMMAND = "python tools/generate_blast_parity_fixture.py --blast-checkout <CLEAN_BLAST_LITE_CHECKOUT>"

# Stable BREOS fixture key -> public class exported by upstream blast.models.
MODEL_CLASS_NAMES = {
    "lfp_gr_250ah_prismatic": "Lfp_Gr_250AhPrismatic",
    "nca_gr_panasonic_3ah": "Nca_Gr_Panasonic3Ah_Battery",
    "lmo_gr_nissanleaf_66ah_2nd": "Lmo_Gr_NissanLeaf66Ah_2ndLife_Battery",
    "nmc811_grsi_lgm50_5ah": "Nmc811_GrSi_LGM50_5Ah_Battery",
    "nmc811_grsi_lgmj1_4ah": "Nmc811_GrSi_LGMJ1_4Ah_Battery",
    "nmc_gr_50ah_b1": "NMC_Gr_50Ah_B1",
    "nmc_gr_50ah_b2": "NMC_Gr_50Ah_B2",
    "nmc_gr_75ah_a": "NMC_Gr_75Ah_A",
    "nmc111_gr_sanyo_2ah": "Nmc111_Gr_Sanyo2Ah_Battery",
    "nmc_lto_10ah": "Nmc_Lto_10Ah_Battery",
    "lfp_gr_sonymurata_3ah": "Lfp_Gr_SonyMurata3Ah_Battery",
    "nca_grsi_sonymurata_2p5ah": "NCA_GrSi_SonyMurata2p5Ah_Battery",
    "nmc111_gr_kokam_75ah": "Nmc111_Gr_Kokam75Ah_Battery",
    "nmc622_gr_denso_50ah": "Nmc622_Gr_DENSO50Ah_Battery",
}

PROFILE_DESCRIPTIONS = {
    "hot_storage": "constant 90% SoC at 45 C, sampled hourly for 60 days",
    "cold_storage": "constant 30% SoC at 5 C, sampled hourly for 60 days",
    "deep_cycle": "daily triangular 95%-5%-95% SoC cycle at 25 C for 60 days",
    "shallow_fast_cycle": "four daily sinusoidal 60%-90% SoC cycles at 40 C for 60 days",
    "tvar_deep_cycle": "daily triangular 95%-5%-95% SoC cycle with 15-35 C sinusoid for 60 days",
}


class ParityGenerationError(RuntimeError):
    """Raised when provenance cannot be established or outputs drift."""


def _git(checkout: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(checkout), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) and exc.stderr else str(exc)
        raise ParityGenerationError(f"cannot inspect BLAST-Lite checkout {checkout}: {detail}") from exc
    return result.stdout.strip()


def validate_checkout(
    checkout: Path,
    *,
    expected_commit: str = PINNED_BLAST_COMMIT,
    allow_unexpected_commit: bool = False,
) -> tuple[Path, str]:
    """Return the resolved checkout and HEAD after strict provenance checks."""
    checkout = checkout.expanduser().resolve()
    if not checkout.is_dir():
        raise ParityGenerationError(f"BLAST-Lite checkout does not exist or is not a directory: {checkout}")

    repository_root = Path(_git(checkout, "rev-parse", "--show-toplevel")).resolve()
    if repository_root != checkout:
        raise ParityGenerationError(
            f"--blast-checkout must name the BLAST-Lite repository root; got {checkout}, root is {repository_root}"
        )
    if not (checkout / "blast" / "models" / "__init__.py").is_file():
        raise ParityGenerationError(f"checkout does not contain blast/models/__init__.py: {checkout}")

    dirty = _git(checkout, "status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        first_change = dirty.splitlines()[0]
        raise ParityGenerationError(f"BLAST-Lite checkout must be clean; first reported change is {first_change!r}")

    actual_commit = _git(checkout, "rev-parse", "HEAD")
    if actual_commit != expected_commit and not allow_unexpected_commit:
        raise ParityGenerationError(
            "BLAST-Lite checkout is at unexpected commit "
            f"{actual_commit}; expected {expected_commit}. "
            "Use --allow-unexpected-commit only in a reviewed upstream-pin update."
        )
    return checkout, actual_commit


def build_profiles() -> dict[str, dict[str, Any]]:
    """Build the named deterministic stress profiles used by the fixture."""
    hours = np.arange(25, dtype=float)
    time_s = hours * 3600.0
    deep_cycle = np.concatenate((np.linspace(0.95, 0.05, 13), np.linspace(0.05, 0.95, 13)[1:]))
    return {
        "hot_storage": {
            "time_s": time_s.tolist(),
            "soc": np.full_like(hours, 0.9).tolist(),
            "temperature_c": np.full_like(hours, 45.0).tolist(),
            "days": DAYS_PER_CONDITION,
        },
        "cold_storage": {
            "time_s": time_s.tolist(),
            "soc": np.full_like(hours, 0.3).tolist(),
            "temperature_c": np.full_like(hours, 5.0).tolist(),
            "days": DAYS_PER_CONDITION,
        },
        "deep_cycle": {
            "time_s": time_s.tolist(),
            "soc": deep_cycle.tolist(),
            "temperature_c": np.full_like(hours, 25.0).tolist(),
            "days": DAYS_PER_CONDITION,
        },
        "shallow_fast_cycle": {
            "time_s": time_s.tolist(),
            "soc": (0.75 + 0.15 * np.sin(hours * np.pi / 3.0)).tolist(),
            "temperature_c": np.full_like(hours, 40.0).tolist(),
            "days": DAYS_PER_CONDITION,
        },
        "tvar_deep_cycle": {
            "time_s": time_s.tolist(),
            "soc": deep_cycle.tolist(),
            "temperature_c": (25.0 + 10.0 * np.sin(hours * np.pi / 12.0)).tolist(),
            "days": DAYS_PER_CONDITION,
        },
    }


def load_upstream_models(checkout: Path) -> dict[str, type]:
    """Import the public model classes from the validated upstream checkout."""
    sys.path.insert(0, str(checkout))
    try:
        models_module = importlib.import_module("blast.models")
    except Exception as exc:
        raise ParityGenerationError(f"cannot import blast.models from {checkout}: {exc}") from exc
    finally:
        sys.path.pop(0)

    module_path = Path(models_module.__file__).resolve()
    if not module_path.is_relative_to(checkout):
        raise ParityGenerationError(f"blast.models resolved outside the requested checkout: {module_path}")

    try:
        return {key: getattr(models_module, class_name) for key, class_name in MODEL_CLASS_NAMES.items()}
    except AttributeError as exc:
        raise ParityGenerationError(f"upstream BLAST-Lite model export is missing: {exc}") from exc


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def generate_fixture(model_classes: dict[str, type], source_commit: str) -> dict[str, Any]:
    """Run every upstream model over each deterministic stress profile."""
    profiles = build_profiles()
    conditions: dict[str, Any] = {}
    parameters: dict[str, Any] = {}

    for condition_name, profile in profiles.items():
        trajectories: dict[str, Any] = {}
        final_outputs: dict[str, Any] = {}
        for model_key, model_class in model_classes.items():
            model = model_class()
            q: list[float] = []
            efc: list[float] = []
            for _ in range(profile["days"]):
                model.update_battery_state(
                    np.asarray(profile["time_s"], dtype=float),
                    np.asarray(profile["soc"], dtype=float),
                    np.asarray(profile["temperature_c"], dtype=float),
                )
                q.append(float(model.outputs["q"][-1]))
                efc.append(float(model.stressors["efc"][-1]))
            trajectories[model_key] = {"q": q, "efc": efc}
            final_outputs[model_key] = {
                output_key: float(output_values[-1]) for output_key, output_values in model.outputs.items()
            }
            if model_key not in parameters:
                parameters[model_key] = {
                    "cap": _json_safe(model.cap),
                    "params_life": _json_safe(model._params_life),
                    "experimental_range": _json_safe(model.experimental_range),
                }
        conditions[condition_name] = {
            "profile": profile,
            "trajectories": trajectories,
            "final_outputs": final_outputs,
        }

    return {
        "schema": FIXTURE_SCHEMA,
        "metadata": {
            "source": f"local untransformed BLAST-Lite checkout (blast/ at commit {source_commit})",
            "numpy_version": np.__version__,
            "python_version": ".".join(str(part) for part in sys.version_info[:3]),
            "days_per_condition": DAYS_PER_CONDITION,
        },
        "conditions": conditions,
        "parameters": parameters,
    }


def _compact_json_bytes(document: Any) -> bytes:
    return (json.dumps(document, allow_nan=False) + "\n").encode()


def _canonical_json_bytes(document: Any) -> bytes:
    return json.dumps(document, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_manifest(
    fixture: dict[str, Any],
    fixture_bytes: bytes,
    *,
    source_commit: str,
    source_version: str,
) -> dict[str, Any]:
    """Build the small provenance sidecar bound to the fixture bytes."""
    profiles = {name: condition["profile"] for name, condition in fixture["conditions"].items()}
    return {
        "schema": MANIFEST_SCHEMA,
        "fixture": {
            "path": str(FIXTURE_PATH.relative_to(PROJECT_ROOT)),
            "schema": fixture["schema"],
            "sha256": _sha256(fixture_bytes),
        },
        "source": {
            "project": "BLAST-Lite",
            "version": source_version,
            "commit": source_commit,
        },
        "environment": {
            "python": fixture["metadata"]["python_version"],
            "numpy": fixture["metadata"]["numpy_version"],
        },
        "generation": {
            "tool": GENERATOR_PATH,
            "command": CANONICAL_COMMAND,
        },
        "profiles": {
            "sha256": _sha256(_canonical_json_bytes(profiles)),
            "definitions": {
                name: {
                    "description": PROFILE_DESCRIPTIONS[name],
                    "sha256": _sha256(_canonical_json_bytes(profile)),
                }
                for name, profile in profiles.items()
            },
        },
    }


def _manifest_bytes(manifest: dict[str, Any]) -> bytes:
    return (json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()


def validate_manifest(fixture: dict[str, Any], fixture_bytes: bytes, manifest: dict[str, Any]) -> None:
    """Reject drift between a committed fixture and its provenance sidecar."""
    source_commit = manifest.get("source", {}).get("commit")
    source_version = manifest.get("source", {}).get("version")
    if not source_commit or not source_version:
        raise ParityGenerationError("parity manifest is missing source version or commit")
    expected = build_manifest(
        fixture,
        fixture_bytes,
        source_commit=source_commit,
        source_version=source_version,
    )
    if manifest != expected:
        raise ParityGenerationError("parity manifest does not match fixture contents or generator definitions")
    expected_source = f"local untransformed BLAST-Lite checkout (blast/ at commit {source_commit})"
    if fixture.get("metadata", {}).get("source") != expected_source:
        raise ParityGenerationError("fixture source metadata does not match the manifest commit")
    days = {condition["profile"]["days"] for condition in fixture["conditions"].values()}
    if days != {fixture["metadata"]["days_per_condition"]}:
        raise ParityGenerationError("fixture condition durations do not match its metadata")


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def _check_file(path: Path, expected: bytes) -> None:
    try:
        actual = path.read_bytes()
    except FileNotFoundError as exc:
        raise ParityGenerationError(f"generated artifact is missing: {path}") from exc
    if actual != expected:
        raise ParityGenerationError(f"generated artifact is stale: {path}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blast-checkout",
        required=True,
        type=Path,
        help="Path to the clean, unmodified BLAST-Lite repository checkout.",
    )
    parser.add_argument("--output", type=Path, default=FIXTURE_PATH, help="Fixture output path.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH, help="Generation-manifest output path.")
    parser.add_argument(
        "--source-version",
        help=f"BLAST-Lite version recorded in the manifest (default: {PINNED_BLAST_VERSION} at the current pin).",
    )
    parser.add_argument(
        "--allow-unexpected-commit",
        action="store_true",
        help="Allow a commit other than BREOS's pin; only for a reviewed upstream-pin update.",
    )
    parser.add_argument("--check", action="store_true", help="Validate committed outputs without writing them.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        checkout, source_commit = validate_checkout(
            args.blast_checkout,
            allow_unexpected_commit=args.allow_unexpected_commit,
        )
        if source_commit != PINNED_BLAST_COMMIT and args.source_version is None:
            raise ParityGenerationError(
                "--source-version is required with --allow-unexpected-commit so an upstream update cannot "
                "silently retain the old version"
            )
        source_version = args.source_version or PINNED_BLAST_VERSION
        model_classes = load_upstream_models(checkout)
        fixture = generate_fixture(model_classes, source_commit)
        fixture_bytes = _compact_json_bytes(fixture)
        manifest = build_manifest(
            fixture,
            fixture_bytes,
            source_commit=source_commit,
            source_version=source_version,
        )
        validate_manifest(fixture, fixture_bytes, manifest)
        manifest_bytes = _manifest_bytes(manifest)
        if args.check:
            _check_file(args.output, fixture_bytes)
            _check_file(args.manifest, manifest_bytes)
            print("BLAST parity fixture and generation manifest are reproducible")
        else:
            _write_atomic(args.output, fixture_bytes)
            _write_atomic(args.manifest, manifest_bytes)
            print(f"Wrote {args.output} and {args.manifest}")
    except ParityGenerationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
