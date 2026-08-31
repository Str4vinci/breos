#!/usr/bin/env python3
"""Emit the resolved-config diff of a run bundle against the article base config."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE = PROJECT_ROOT / "validation/article1/article1-projected-optimization.toml"


def flatten(d: dict, prefix: str = "") -> dict:
    out: dict = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def diff_one(resolved: dict) -> dict:
    base = flatten(tomllib.loads(BASE.read_bytes().decode("utf-8")))
    run = flatten(resolved)
    return {
        k: {"base": base.get(k, "<absent>"), "run": run.get(k, "<absent>")}
        for k in sorted(set(base) | set(run))
        if base.get(k, "<absent>") != run.get(k, "<absent>")
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bundles", nargs="+", type=Path, help="Directories containing reproduction.json")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    payload = {}
    for bundle in args.bundles:
        report = json.loads((bundle / "reproduction.json").read_text())
        payload[bundle.name] = {
            "base_config": str(BASE.relative_to(PROJECT_ROOT)),
            "resolved_config_sha256": report["resolved_config_sha256"],
            "diff_vs_base": diff_one(report["resolved_config"]),
            "command": report["command"],
            "weather_uncompressed_sha256": report["weather_uncompressed_sha256"],
            "external_rlp_sha256": report["external_rlp_sha256"],
            "solar_position_offset_minutes": report.get("solar_position_offset_minutes"),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
