#!/usr/bin/env python3
"""Run the complete Article 1 reproduction workflow with local defaults."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from breos.execution import (  # noqa: E402
    DEFAULT_EXECUTION_BACKEND,
    EXECUTION_BACKENDS,
    backend_provenance,
)

DEFAULT_INPUT_ROOT = PROJECT_ROOT / "dev/article1-inputs"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results/article1"
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "validation/article1"
DETERMINISTIC_CONFIG = "article1-projected-optimization.toml"
MONTE_CARLO_CONFIG = "article1-montecarlo.toml"
ARTICLE_VERSION = "0.6.0"
CALENDAR_MODELS = (
    "naumann",
    "naumann_lam",
    "naumann_lam_field_calibrated",
    "naumann_lam_field_calibrated_v1",
    "naumann_lam_field_calibrated_v2",
)


def _python_tool(name: str, *args: object) -> list[str]:
    return [sys.executable, str(PROJECT_ROOT / "tools" / name), *(str(arg) for arg in args)]


def _config_args(config_dir: Path | None, name: str) -> tuple[str, ...]:
    """The config override, or nothing at all when the shipped one applies.

    Passing nothing on the default keeps an unswept run's commands identical
    to the published ones, so provenance records no override that never was.
    """
    return () if config_dir is None else ("--config", str(config_dir / name))


def _calendar_args(calendar_model: str | None) -> tuple[str, ...]:
    """The model override, or nothing at all when the article default applies.

    Passing nothing rather than the default name keeps an unswept run
    byte-identical to the published bundle: the tools record the flag they
    were given, so a redundant override would show up in provenance.
    """
    return () if calendar_model is None else ("--calendar-model", calendar_model)


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


def _monte_carlo_validation_command(input_root: Path, config_dir: Path | None) -> list[str]:
    return _python_tool(
        "reproduce_article1_montecarlo.py",
        *_config_args(config_dir, MONTE_CARLO_CONFIG),
        "--case",
        "all",
        "--rlp-directory",
        input_root / "rlp",
        "--weather-file",
        input_root / "weather/porto_historical_2005_2024_openmeteo.csv",
        "--validate-only",
    )


def _fixed_command(
    input_root: Path,
    output_root: Path,
    execution_backend: str,
    calendar_model: str | None,
    config_dir: Path | None,
) -> list[str]:
    return _python_tool(
        "reproduce_article1.py",
        *_config_args(config_dir, DETERMINISTIC_CONFIG),
        "--rlp-directory",
        input_root / "rlp",
        "--execution-backend",
        execution_backend,
        *_calendar_args(calendar_model),
        "--output",
        output_root / "base-v1",
    )


def _analysis_commands(
    input_root: Path,
    output_root: Path,
    n_procs: int,
    execution_backend: str,
    calendar_model: str | None,
    config_dir: Path | None,
) -> list[list[str]]:
    """Deterministic analyses.

    Only the ``reproduce_article1.py`` invocations take a backend.
    ``reproduce_article1_context.py`` builds orientation and weather-comparison
    tables without ever entering the within-day dispatch loop, so forwarding
    the flag there would record a claim about code that never ran.
    """
    rlp = input_root / "rlp"
    weather = input_root / "weather/porto_historical_2005_2024_openmeteo.csv"
    return [
        _python_tool(
            "reproduce_article1.py",
            *_config_args(config_dir, DETERMINISTIC_CONFIG),
            "--execution-backend",
            execution_backend,
            "--rlp-directory",
            rlp,
            *_calendar_args(calendar_model),
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
            *_config_args(config_dir, DETERMINISTIC_CONFIG),
            "--execution-backend",
            execution_backend,
            "--rlp-directory",
            rlp,
            *_calendar_args(calendar_model),
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
            *_config_args(config_dir, DETERMINISTIC_CONFIG),
            "--execution-backend",
            execution_backend,
            "--rlp-directory",
            rlp,
            *_calendar_args(calendar_model),
            "--load-profile",
            "h0",
            "--candidate",
            "C2",
            "--output",
            output_root / "load-profile-h0",
        ),
        # The two model sensitivities contrast one calendar model with the
        # article's default. A pipeline already swept onto another model has
        # nothing left to contrast, and would file its results under a
        # directory naming a model that did not run.
        *(
            []
            if calendar_model is not None
            else [
                _python_tool(
                    "reproduce_article1.py",
                    *_config_args(config_dir, DETERMINISTIC_CONFIG),
                    "--execution-backend",
                    execution_backend,
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
                    *_config_args(config_dir, DETERMINISTIC_CONFIG),
                    "--execution-backend",
                    execution_backend,
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
            ]
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
    execution_backend: str,
    calendar_model: str | None,
    config_dir: Path | None,
) -> list[list[str]]:
    common: list[object] = [
        *_config_args(config_dir, MONTE_CARLO_CONFIG),
        "--execution-backend",
        execution_backend,
        *_calendar_args(calendar_model),
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
    execution_backend: str = DEFAULT_EXECUTION_BACKEND,
    calendar_model: str | None = None,
    config_dir: Path | None = None,
) -> list[list[str]]:
    """Return the ordered commands for one workflow stage.

    The backend reaches only the stages that simulate. The preflight,
    validation and verification commands read and check inputs and outputs;
    they never run the dispatch loop, so they take no backend.
    """
    checks = [
        _preflight_command(input_root, output_root),
        _monte_carlo_validation_command(input_root, config_dir),
    ]
    fixed = _fixed_command(input_root, output_root, execution_backend, calendar_model, config_dir)
    analysis = _analysis_commands(input_root, output_root, n_procs, execution_backend, calendar_model, config_dir)
    monte_carlo = _monte_carlo_commands(
        input_root, output_root, n_procs, runs, execution_backend, calendar_model, config_dir
    )
    verify = _python_tool("verify_article1_bundle.py", output_root)

    if stage == "check":
        return checks
    if stage == "fixed":
        return [*checks, fixed]
    if stage == "analysis":
        return [*checks, *analysis]
    if stage == "deterministic":
        return [*checks, fixed, *analysis]
    if stage == "monte-carlo":
        return [*checks, *monte_carlo]
    if stage == "verify":
        return [verify]
    if stage == "all":
        # verify_article1_bundle.py asserts the article's own calendar model
        # and thermal assumption, so it can only pass against a bundle that
        # ran the shipped configuration.
        if calendar_model is not None or config_dir is not None:
            return [*checks, fixed, *analysis, *monte_carlo]
        return [*checks, fixed, *analysis, *monte_carlo, verify]
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
    parser.add_argument(
        "--execution-backend",
        choices=EXECUTION_BACKENDS,
        default=DEFAULT_EXECUTION_BACKEND,
        help=(
            "Within-day dispatch implementation for the simulating stages. 'python' is the "
            "numerical reference and the default; 'numba' is a compiled path that reproduces "
            'it bit for bit and needs pip install "breos[fast]".'
        ),
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        help=(
            f"Directory holding {DETERMINISTIC_CONFIG} and {MONTE_CARLO_CONFIG}. Omit it to run "
            "the shipped Article configuration. Setting it drops bundle verification and "
            "requires an explicit --output."
        ),
    )
    parser.add_argument(
        "--calendar-model",
        choices=CALENDAR_MODELS,
        help=(
            "Sweep every simulating stage onto one native calendar-degradation model. "
            "Omit it to run the article as published. Setting it drops the two model "
            "sensitivities and bundle verification, and requires an explicit --output."
        ),
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
    config_dir = args.config_dir
    if config_dir is not None:
        config_dir = config_dir.resolve()
        for name in (DETERMINISTIC_CONFIG, MONTE_CARLO_CONFIG):
            if not (config_dir / name).is_file():
                parser.error(f"--config-dir has no {name}: {config_dir}")
        if config_dir == DEFAULT_CONFIG_DIR.resolve():
            config_dir = None
    overrides = [
        name
        for name, value in (("--calendar-model", args.calendar_model), ("--config-dir", config_dir))
        if value is not None
    ]
    if overrides:
        # The published bundle is the parity reference for the compiled
        # backend. An override written over it would destroy that quietly.
        joined = " and ".join(overrides)
        if args.output.resolve() == DEFAULT_OUTPUT_ROOT.resolve():
            parser.error(
                f"{joined} needs an explicit --output; another configuration written into "
                f"{DEFAULT_OUTPUT_ROOT} would overwrite the published bundle"
            )
        if args.stage == "verify":
            parser.error(f"{joined} cannot be verified: verify asserts the article's own configuration")

    commands = commands_for_stage(
        args.stage,
        args.input_root.resolve(),
        args.output.resolve(),
        args.n_procs,
        args.mc_runs,
        args.execution_backend,
        args.calendar_model,
        config_dir,
    )
    # Say what will run before it runs. Each tool records its own execution
    # provenance in its own output, but a bundle assembled over hours is much
    # easier to read back if the run that produced it announced the choice.
    print(f"execution: {json.dumps(backend_provenance(args.execution_backend), sort_keys=True)}", flush=True)
    if args.calendar_model is not None:
        print(f"calendar model: {args.calendar_model} (sensitivities and verification skipped)", flush=True)
    if config_dir is not None:
        print(f"config dir: {config_dir} (verification skipped)", flush=True)
    if not args.dry_run:
        _require_clean_article_version()
    _run(commands, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
