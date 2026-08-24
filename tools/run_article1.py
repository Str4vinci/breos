#!/usr/bin/env python3
"""Run the complete Article 1 reproduction workflow with local defaults."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "dev/article1-inputs"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results/article1"
ARTICLE_VERSION = "0.6.0"


def _python_tool(name: str, *args: object) -> list[str]:
    return [sys.executable, str(PROJECT_ROOT / "tools" / name), *(str(arg) for arg in args)]


def _preflight_command(input_root: Path, output_root: Path) -> list[str]:
    return _python_tool(
        "preflight_article1_inputs.py",
        "--rlp-directory",
        input_root / "rlp",
        "--historical-weather-file",
        input_root / "weather/porto_historical_2005_2024_openmeteo.csv",
        "--validation-directory",
        input_root / "validation",
        "--copy-validation-to",
        output_root / "external-validation",
        "--output",
        output_root / "input-manifest.json",
    )


def _fixed_command(input_root: Path, output_root: Path) -> list[str]:
    return _python_tool(
        "reproduce_article1.py",
        "--rlp-directory",
        input_root / "rlp",
        "--output",
        output_root / "base-v1",
    )


def _analysis_commands(input_root: Path, output_root: Path, n_procs: int) -> list[list[str]]:
    rlp = input_root / "rlp"
    weather = input_root / "weather/porto_historical_2005_2024_openmeteo.csv"
    return [
        _python_tool(
            "reproduce_article1.py",
            "--rlp-directory",
            rlp,
            "--battery-cost",
            350,
            "--battery-cost",
            500,
            "--battery-cost",
            711,
            "--skip-fixed",
            "--full-optimization",
            "--n-procs",
            n_procs,
            "--output",
            output_root / "battery-cost-sensitivity",
        ),
        _python_tool(
            "reproduce_article1.py",
            "--rlp-directory",
            rlp,
            "--resolution",
            "h",
            "--skip-fixed",
            "--full-optimization",
            "--n-procs",
            n_procs,
            "--output",
            output_root / "hourly-v1",
        ),
        _python_tool(
            "reproduce_article1.py",
            "--rlp-directory",
            rlp,
            "--load-profile",
            "h0",
            "--candidate",
            "C2",
            "--output",
            output_root / "load-profile-h0",
        ),
        _python_tool(
            "reproduce_article1.py",
            "--rlp-directory",
            rlp,
            "--calendar-model",
            "naumann_lam_field_calibrated_v2",
            "--candidate",
            "C2",
            "--full-optimization",
            "--n-procs",
            n_procs,
            "--output",
            output_root / "field-v2",
        ),
        _python_tool(
            "reproduce_article1.py",
            "--rlp-directory",
            rlp,
            "--calendar-model",
            "naumann_lam",
            "--candidate",
            "C2",
            "--full-optimization",
            "--n-procs",
            n_procs,
            "--output",
            output_root / "laboratory",
        ),
        _python_tool(
            "reproduce_article1_context.py",
            "--output",
            output_root / "orientation",
            "orientation",
        ),
        _python_tool(
            "reproduce_article1_context.py",
            "--output",
            output_root / "weather-comparison",
            "weather-comparison",
            "--historical-weather-file",
            weather,
        ),
    ]


def _monte_carlo_commands(
    input_root: Path,
    output_root: Path,
    n_procs: int,
    runs: int | None,
) -> list[list[str]]:
    common: list[object] = [
        "--rlp-directory",
        input_root / "rlp",
        "--weather-file",
        input_root / "weather/porto_historical_2005_2024_openmeteo.csv",
        "--n-procs",
        n_procs,
        "--output",
        output_root / "monte-carlo-v1",
    ]
    if runs is not None:
        common.extend(("--runs", runs))
    return [
        _python_tool("reproduce_article1_montecarlo.py", "--case", "C2", *common),
        _python_tool(
            "reproduce_article1_montecarlo.py",
            "--case",
            "C1",
            "--case",
            "C3",
            "--case",
            "C4",
            "--case",
            "C5",
            *common,
        ),
    ]


def commands_for_stage(
    stage: str,
    input_root: Path,
    output_root: Path,
    n_procs: int,
    runs: int | None = None,
) -> list[list[str]]:
    """Return the ordered commands for one workflow stage."""
    preflight = [_preflight_command(input_root, output_root)]
    if stage == "check":
        return preflight
    if stage == "fixed":
        return [*preflight, _fixed_command(input_root, output_root)]
    if stage == "analysis":
        return [*preflight, *_analysis_commands(input_root, output_root, n_procs)]
    if stage == "deterministic":
        return [
            *preflight,
            _fixed_command(input_root, output_root),
            *_analysis_commands(input_root, output_root, n_procs),
        ]
    if stage == "monte-carlo":
        return [*preflight, *_monte_carlo_commands(input_root, output_root, n_procs, runs)]
    if stage == "verify":
        return [_python_tool("verify_article1_bundle.py", output_root)]
    if stage == "all":
        return [
            *preflight,
            _fixed_command(input_root, output_root),
            *_analysis_commands(input_root, output_root, n_procs),
            *_monte_carlo_commands(input_root, output_root, n_procs, runs),
            _python_tool("verify_article1_bundle.py", output_root),
        ]
    raise ValueError(f"Unknown stage: {stage}")


def _require_clean_article_version() -> None:
    version = importlib.metadata.version("breos")
    if version != ARTICLE_VERSION:
        raise RuntimeError(f"Article 1 requires BREOS {ARTICLE_VERSION}; the active environment reports {version}")
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=no"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("Commit or restore tracked changes before running Article 1 simulations")


def _run(commands: list[list[str]], *, dry_run: bool) -> None:
    for index, command in enumerate(commands, start=1):
        print(f"\n[{index}/{len(commands)}] {shlex.join(command)}", flush=True)
        if not dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Stages:
  check          Verify and inventory the local inputs without simulation.
  fixed          Run C1-C5 without NSGA-II.
  analysis       Run every deterministic analysis except C1-C5.
  deterministic Run C1-C5 and every deterministic analysis (default).
  monte-carlo    Run C2 first, then C1, C3, C4, and C5.
  verify         Verify the completed result bundle.
  all            Run deterministic, Monte Carlo, and verification stages.
""",
    )
    parser.add_argument(
        "stage",
        nargs="?",
        default="deterministic",
        choices=("check", "fixed", "analysis", "deterministic", "monte-carlo", "verify", "all"),
        help="Workflow stage. The default runs every deterministic analysis.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Local input bundle (default: dev/article1-inputs)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Result bundle (default: results/article1)",
    )
    parser.add_argument(
        "--n-procs",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Worker processes for optimization and Monte Carlo (default: up to 8)",
    )
    parser.add_argument("--mc-runs", type=int, help="Override 10,000 Monte Carlo trajectories for each case")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without running them")
    args = parser.parse_args()

    if args.n_procs < 1:
        parser.error("--n-procs must be at least 1")
    if args.mc_runs is not None and args.mc_runs < 1:
        parser.error("--mc-runs must be at least 1")
    if args.mc_runs is not None and args.stage != "monte-carlo":
        parser.error("--mc-runs applies only to the monte-carlo stage")

    commands = commands_for_stage(
        args.stage,
        args.input_root.resolve(),
        args.output.resolve(),
        args.n_procs,
        args.mc_runs,
    )
    if not args.dry_run:
        _require_clean_article_version()
    _run(commands, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
