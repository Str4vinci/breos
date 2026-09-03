"""Run the recoverable publication validation packages without CPU overlap.

The runner can wait for another process to exit, then reconstruct the local-data
packages in a new output root. It never deletes or overwrites an existing
package. The NIST 2016 replay is a separate manual workflow because the portal
requires a dynamic archive download.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RECOVERY_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RECOVERY_ROOT.parents[2]
DEFAULT_BREOS_ROOT = Path("/tmp/breos-article1-0.6.0")
# The DKASC archive is downloaded separately; --dkasc-raw-dir or
# BREOS_VALIDATION_DKASC_RAW says where it was unpacked.
DEFAULT_DKASC_RAW = Path(os.environ.get("BREOS_VALIDATION_DKASC_RAW", "datasets/DKA_Alice_Springs")).expanduser()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class RecoveryRun:
    def __init__(self, output_root: Path, breos_root: Path, python: Path) -> None:
        self.output_root = output_root
        self.breos_root = breos_root
        self.python = python
        self.status_path = output_root / "status.json"
        self.status: dict[str, Any] = {
            "state": "initializing",
            "started_at": _now(),
            "updated_at": _now(),
            "article_breos_root": str(breos_root),
            "article_commit": "f62f4f5bf3c14140ab189d35ea2885e6fcc60c6b",
            "packages": {},
            "nist": {
                "state": "separate_manual_workflow",
                "workflow": "tools/validation/recovery/nist",
                "reason": (
                    "This runner neither executes nor verifies NIST; use the separate "
                    "manual replay workflow and record its output package independently."
                ),
            },
            "esposende": {
                "state": "excluded",
                "reason": "Excluded at the user's request.",
            },
        }

    def save(self, **updates: Any) -> None:
        self.status.update(updates)
        self.status["updated_at"] = _now()
        _write_json(self.status_path, self.status)

    def package_state(self, name: str, state: str, **details: Any) -> None:
        self.status["packages"][name] = {
            "state": state,
            "updated_at": _now(),
            **details,
        }
        self.save()

    def command(
        self,
        name: str,
        command: list[str],
        *,
        cwd: Path,
        log_path: Path,
        env: dict[str, str] | None = None,
    ) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command_env = os.environ.copy()
        if env:
            command_env.update(env)
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"started_at={_now()}\n")
            log.write("command=" + " ".join(command) + "\n")
            log.flush()
            result = subprocess.run(
                command,
                cwd=cwd,
                env=command_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            log.write(f"\nfinished_at={_now()}\nexit_code={result.returncode}\n")
        if result.returncode:
            raise RuntimeError(f"{name} failed with exit code {result.returncode}; see {log_path}")

    def run_replayed_driver(self, name: str, source_dir: str, driver_name: str) -> None:
        package = self.output_root / name
        if package.exists():
            raise FileExistsError(f"Refusing existing package: {package}")
        package.mkdir()
        driver_dir = package / "drivers"
        driver_dir.mkdir()
        source = RECOVERY_ROOT / source_dir
        driver = driver_dir / driver_name
        shutil.copy2(source / "drivers" / driver_name, driver)
        shutil.copy2(source / "README.md", package / "README.md")
        self.command(
            name,
            [str(self.python), str(driver), "--force"],
            cwd=REPO_ROOT,
            log_path=package / "run.log",
            env={
                "BREOS_VALIDATION_ROOT": str(self.breos_root),
                "BREOS_VALIDATION_OUTPUT": str(package),
                "MPLCONFIGDIR": str(self.output_root / ".matplotlib"),
            },
        )

    def run_hkust(self) -> None:
        name = "validation_hkust_timing-corrected-exploratory-v4_recovered_20260902"
        package = self.output_root / name
        if package.exists():
            raise FileExistsError(f"Refusing existing package: {package}")
        driver = RECOVERY_ROOT / "hkust" / "drivers" / "hkust_validate.py"
        self.command(
            name,
            [str(self.python), str(driver), "--output-dir", str(package)],
            cwd=REPO_ROOT,
            log_path=self.output_root / "logs" / f"{name}.log",
            env={
                "BREOS_VALIDATION_ROOT": str(self.breos_root),
                "MPLCONFIGDIR": str(self.output_root / ".matplotlib"),
            },
        )
        shutil.copy2(self.output_root / "logs" / f"{name}.log", package / "run.log")

    def run_dkasc(self, raw_dir: Path) -> None:
        name = "validation_dkasc_recovered_20260902"
        package = self.output_root / name
        if package.exists():
            raise FileExistsError(f"Refusing existing package: {package}")
        package.mkdir()
        drivers = package / "drivers"
        drivers.mkdir()
        source = RECOVERY_ROOT / "dkasc"
        for path in sorted(source.glob("*.py")):
            shutil.copy2(path, drivers / path.name)
        shutil.copy2(source / "README-dkasc.md", package / "README.md")
        shutil.copy2(source / "FINDINGS-dkasc.md", package / "FINDINGS.md")
        data_dir = package / "data"
        results_dir = package / "results"
        logs = package / "logs"
        data_dir.mkdir()
        results_dir.mkdir()
        logs.mkdir()
        env = {
            "PYTHONPATH": f"{self.breos_root}:{drivers}",
            "MPLCONFIGDIR": str(self.output_root / ".matplotlib"),
        }

        self.command(
            "dkasc facts",
            [str(self.python), str(drivers / "dkasc_facts.py"), "--raw-dir", str(raw_dir)],
            cwd=REPO_ROOT,
            log_path=logs / "facts.log",
            env=env,
        )
        data_2016 = data_dir / "dkasc_2016.csv.gz"
        self.command(
            "dkasc 2016 build",
            [
                str(self.python),
                str(drivers / "dkasc_build.py"),
                "--raw-dir",
                str(raw_dir),
                "--out",
                str(data_2016),
                "--start-year",
                "2016",
                "--end-year",
                "2016",
                "--sources",
                "100",
                "81",
                "84",
                "92",
                "91",
                "96",
            ],
            cwd=REPO_ROOT,
            log_path=logs / "build-2016.log",
            env=env,
        )
        data_long = data_dir / "dkasc_2009_2020.csv.gz"
        self.command(
            "dkasc long build",
            [
                str(self.python),
                str(drivers / "dkasc_build.py"),
                "--raw-dir",
                str(raw_dir),
                "--out",
                str(data_long),
                "--start-year",
                "2009",
                "--end-year",
                "2020",
                "--sources",
                "100",
                "81",
                "84",
                "92",
                "91",
            ],
            cwd=REPO_ROOT,
            log_path=logs / "build-2009-2020.log",
            env=env,
        )
        self.command(
            "dkasc transposition",
            [
                str(self.python),
                str(drivers / "dkasc_transposition.py"),
                "--gti-data",
                str(data_2016),
                "--pair-data",
                str(data_long),
                "--outdir",
                str(results_dir),
            ],
            cwd=REPO_ROOT,
            log_path=logs / "transposition.log",
            env=env,
        )
        self.command(
            "dkasc analysis",
            [
                str(self.python),
                str(drivers / "dkasc_analysis.py"),
                "--data",
                str(data_long),
                "--outdir",
                str(results_dir),
                "--ladder-years",
                "2009:2014",
                "--tracker-years",
                "2014:2016",
            ],
            cwd=REPO_ROOT,
            log_path=logs / "analysis.log",
            env=env,
        )

        raw_files = sorted(path for path in raw_dir.iterdir() if path.is_file())
        (package / "input_manifest.sha256").write_text(
            "".join(f"{_sha256(path)}  {path.name}\n" for path in raw_files),
            encoding="utf-8",
        )
        _write_json(
            package / "run_config.json",
            {
                "raw_dir": str(raw_dir),
                "breos_root": str(self.breos_root),
                "breos_commit": "f62f4f5bf3c14140ab189d35ea2885e6fcc60c6b",
                "commands": "See logs/*.log and the recovered drivers.",
            },
        )

    def write_manifest(self) -> None:
        excluded = {self.status_path, self.output_root / "artifact_manifest.sha256"}
        files = sorted(path for path in self.output_root.rglob("*") if path.is_file() and path not in excluded)
        (self.output_root / "artifact_manifest.sha256").write_text(
            "".join(f"{_sha256(path)}  {path.relative_to(self.output_root).as_posix()}\n" for path in files),
            encoding="utf-8",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--breos-root", type=Path, default=DEFAULT_BREOS_ROOT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--dkasc-raw-dir", type=Path, default=DEFAULT_DKASC_RAW)
    parser.add_argument(
        "--wait-pid",
        type=int,
        help="wait until this process exits before starting CPU-intensive work",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing non-empty output root: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    run = RecoveryRun(args.output_root, args.breos_root, args.python)
    run.save()

    if args.wait_pid:
        run.save(
            state="waiting_for_task4",
            wait_pid=args.wait_pid,
            detail="No validation workload will start until the Task 4 process exits.",
        )
        while _pid_exists(args.wait_pid):
            time.sleep(60)

    if not args.breos_root.is_dir():
        raise FileNotFoundError(args.breos_root)
    if not args.dkasc_raw_dir.is_dir():
        raise FileNotFoundError(args.dkasc_raw_dir)

    run.save(state="running", detail="Running local-data validation recovery")
    tasks = [
        (
            "validation_sandia_task13_recovered_20260902",
            lambda: run.run_replayed_driver(
                "validation_sandia_task13_recovered_20260902",
                "sandia_task13",
                "sandia_thermal_validate.py",
            ),
        ),
        (
            "validation_pcoe_recovered_20260902",
            lambda: run.run_replayed_driver("validation_pcoe_recovered_20260902", "pcoe", "pcoe_validate.py"),
        ),
        (
            "validation_reunion_microgrid_recovered_20260902",
            lambda: run.run_replayed_driver(
                "validation_reunion_microgrid_recovered_20260902",
                "reunion_microgrid",
                "reunion_validate.py",
            ),
        ),
        (
            "validation_orientation_diversity_recovered_20260902",
            lambda: run.run_replayed_driver(
                "validation_orientation_diversity_recovered_20260902",
                "orientation_diversity",
                "orientation_screen.py",
            ),
        ),
        (
            "validation_dkasc_recovered_20260902",
            lambda: run.run_dkasc(args.dkasc_raw_dir),
        ),
        (
            "validation_hkust_timing-corrected-exploratory-v4_recovered_20260902",
            run.run_hkust,
        ),
    ]

    failures: list[str] = []
    for name, task in tasks:
        run.package_state(name, "running", started_at=_now())
        try:
            task()
        except Exception as exc:  # keep independent recoveries running
            failures.append(name)
            run.package_state(name, "failed", error=f"{type(exc).__name__}: {exc}")
        else:
            run.package_state(name, "complete", finished_at=_now())

    verification_log = args.output_root / "verification.log"
    try:
        run.command(
            "recovery verification",
            [
                str(args.python),
                str(RECOVERY_ROOT / "verify_recovery.py"),
                str(args.output_root),
            ],
            cwd=REPO_ROOT,
            log_path=verification_log,
        )
    except Exception as exc:
        failures.append("verification")
        run.status["verification"] = {
            "state": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "log": str(verification_log),
        }
    else:
        run.status["verification"] = {
            "state": "complete",
            "log": str(verification_log),
        }

    run.write_manifest()
    run.save(
        state="complete_with_failures" if failures else "complete",
        failures=failures,
        finished_at=_now(),
        detail="Local-data recovery complete; NIST is recorded as a separate manual replay. Esposende was excluded.",
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
