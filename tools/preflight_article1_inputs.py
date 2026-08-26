#!/usr/bin/env python3
"""Verify and inventory Article 1 inputs without running a simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RLP_15MIN_FILENAME = "EREDES_2025_BTN_1000kwh_15min.csv"
RLP_HOURLY_FILENAME = "EREDES_2025_BTN_1000kwh_hourly.csv"
EXPECTED_SHA256 = {
    "external_rlp_15min": "23becc5a7bfc927b1f7604156e0e4953dcc6bb65268ca947b38db3dc4f2b28bc",
    "external_rlp_hourly": "6ae15efed9b179537349ee1c1c5747065f18db876920e073e8670bf412d20d6b",
    "historical_weather": "71c26d072c09faf16dab37230cfe8b2d430bd39344333227d00c7be4e76a188a",
    "bundled_tmy": "bf84e31b02ad9bf39f331a5ce8629b1ea8f80cd1597748e72a87a0fce56b4f15",
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rlp-directory", type=Path, required=True)
    parser.add_argument("--historical-weather-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Optional JSON manifest path")
    args = parser.parse_args()

    inputs = {
        "external_rlp_15min": _checked_input("external_rlp_15min", args.rlp_directory / RLP_15MIN_FILENAME),
        "external_rlp_hourly": _checked_input("external_rlp_hourly", args.rlp_directory / RLP_HOURLY_FILENAME),
        "historical_weather": _checked_input("historical_weather", args.historical_weather_file),
        "bundled_tmy": _checked_input(
            "bundled_tmy",
            PROJECT_ROOT / "validation/data/weather/porto_tmy_2005_2023_pvgis-sarah3.csv.gz",
        ),
    }
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
