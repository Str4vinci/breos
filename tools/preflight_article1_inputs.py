#!/usr/bin/env python3
"""Verify and inventory Article 1 inputs without running a simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RLP_15MIN_FILENAME = "EREDES_2025_BTN_1000kwh_15min.csv"
RLP_HOURLY_FILENAME = "EREDES_2025_BTN_1000kwh_hourly.csv"
EXPECTED_SHA256 = {
    "external_rlp_15min": "23becc5a7bfc927b1f7604156e0e4953dcc6bb65268ca947b38db3dc4f2b28bc",
    "external_rlp_hourly": "6ae15efed9b179537349ee1c1c5747065f18db876920e073e8670bf412d20d6b",
    # Open-Meteo 2005-2024 snapshot with preceding-hour radiation means.
    "historical_weather": "71c26d072c09faf16dab37230cfe8b2d430bd39344333227d00c7be4e76a188a",
    "bundled_tmy": "bf84e31b02ad9bf39f331a5ce8629b1ea8f80cd1597748e72a87a0fce56b4f15",
    # Corrected Esposende Figure 2 data from the provenance-clean rerun at
    # phd commit e1558f1678946c4a8d2146a4bd62ad38e068080d.
    "validation_monthly": "84cce17d0d51f745895bad1c98adaa3ef3d6043f076ce69d3a1dff5ecad1a526",
    "validation_weekly": "07c7b938da100ca5c5301a0ef9a6988b24616f7e5eb5166408056afd1ae79375",
    "validation_daily": "2d12468d982f0b59af875841bf8b9e228a532c6f4060070245907bbb243b0bfb",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> dict[str, str | bool]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=no"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"commit": revision, "tracked_worktree_dirty": bool(status)}


def _checked_input(label: str, path: Path) -> dict[str, str | int]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    actual = _sha256(resolved)
    expected = EXPECTED_SHA256[label]
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual} ({resolved})")
    return {"path": str(resolved), "sha256": actual, "size_bytes": resolved.stat().st_size}


def _checked_weather_metadata(
    weather_path: Path, *, radiation_time_basis: str, timestamp_label_basis: str
) -> dict[str, object]:
    sidecar = Path(f"{weather_path}.metadata.json").resolve()
    if not sidecar.is_file():
        raise FileNotFoundError(f"weather metadata sidecar not found: {sidecar}")
    payload = json.loads(sidecar.read_text())
    weather_sha256 = _sha256(weather_path.resolve())
    if payload.get("schema_version") != 1 or payload.get("weather_sha256") != weather_sha256:
        raise ValueError(f"weather metadata sidecar is not bound to {weather_path.resolve()}")
    metadata = payload.get("breos_weather_metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"weather metadata sidecar has no metadata object: {sidecar}")
    if metadata.get("radiation_time_basis") != radiation_time_basis:
        raise ValueError(f"weather metadata sidecar has the wrong radiation time basis: {sidecar}")
    if metadata.get("timestamp_label_basis") != timestamp_label_basis:
        raise ValueError(f"weather metadata sidecar has the wrong timestamp label basis: {sidecar}")
    return {
        "path": str(sidecar),
        "sha256": _sha256(sidecar),
        "weather_sha256": weather_sha256,
        "metadata": metadata,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rlp-directory", type=Path, required=True)
    parser.add_argument("--historical-weather-file", type=Path, required=True)
    parser.add_argument(
        "--validation-directory",
        type=Path,
        required=True,
        help="Directory containing the external monthly, weekly, and daily Figure 2 comparison CSVs",
    )
    parser.add_argument(
        "--copy-validation-to",
        type=Path,
        help="Copy the three verified Figure 2 CSVs into this ignored result directory",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON manifest path")
    args = parser.parse_args()

    validation = args.validation_directory
    inputs = {
        "external_rlp_15min": _checked_input("external_rlp_15min", args.rlp_directory / RLP_15MIN_FILENAME),
        "external_rlp_hourly": _checked_input("external_rlp_hourly", args.rlp_directory / RLP_HOURLY_FILENAME),
        "historical_weather": _checked_input("historical_weather", args.historical_weather_file),
        "bundled_tmy": _checked_input(
            "bundled_tmy",
            PROJECT_ROOT / "validation/data/weather/porto_tmy_2005_2023_pvgis-sarah3.csv.gz",
        ),
        "validation_monthly": _checked_input("validation_monthly", validation / "monthly_results.csv"),
        "validation_weekly": _checked_input("validation_weekly", validation / "weekly_results.csv"),
        "validation_daily": _checked_input("validation_daily", validation / "daily_results.csv"),
    }
    inputs["historical_weather_metadata"] = _checked_weather_metadata(
        args.historical_weather_file,
        radiation_time_basis="interval_mean",
        timestamp_label_basis="right",
    )
    bundled_tmy = PROJECT_ROOT / "validation/data/weather/porto_tmy_2005_2023_pvgis-sarah3.csv.gz"
    inputs["bundled_tmy_metadata"] = _checked_weather_metadata(
        bundled_tmy,
        radiation_time_basis="instant",
        timestamp_label_basis="provider_hour",
    )
    if args.copy_validation_to is not None:
        args.copy_validation_to.mkdir(parents=True, exist_ok=True)
        for label, filename in (
            ("validation_monthly", "monthly_results.csv"),
            ("validation_weekly", "weekly_results.csv"),
            ("validation_daily", "daily_results.csv"),
        ):
            destination = args.copy_validation_to / filename
            if destination.is_file() and _sha256(destination) != EXPECTED_SHA256[label]:
                raise ValueError(f"refusing to overwrite a different file: {destination.resolve()}")
            if not destination.exists():
                shutil.copy2(Path(inputs[label]["path"]), destination)
            inputs[label]["bundle_copy"] = str(destination.resolve())
    manifest = {
        "schema": "breos-article1-input-manifest-v1",
        "breos_source": _git_revision(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "inputs": inputs,
    }
    rendered = json.dumps(manifest, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(f"Article 1 input preflight passed; wrote {args.output}")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
