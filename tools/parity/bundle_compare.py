"""Compare a Monte Carlo case directory against a preserved reference bundle.

The release gate for the compiled backend: an accelerated re-run of an Article
case must reproduce the preserved Python-path bundle exactly, field by field,
trajectory by trajectory.

Both ``runs.csv`` and ``yearly.csv`` are compared. Floats are compared bitwise
via their raw bytes rather than with a tolerance, because the claim being gated
is bit identity and "close" would let a reassociated sum through. NaN equals
NaN: it is a legitimate value here -- ``final_soh_pct`` is NaN for a case with
no battery -- and a comparison that treated it as a mismatch would fail C1 for
being correct.

Provenance is reported, never compared: the reference records ``execution:
null`` because it predates the backend, and the re-run records the compiled
backend and its toolchain. That difference is the point of the exercise.

Usage:
    python tools/parity/bundle_compare.py <reference_case_dir> <candidate_case_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TABLES = ("runs.csv", "yearly.csv")


def _identical(left: pd.Series, right: pd.Series) -> tuple[bool, str]:
    """Return whether two columns are bit-identical, and how they differ."""
    if len(left) != len(right):
        return False, f"length {len(left)} != {len(right)}"

    if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
        a = left.to_numpy(dtype=np.float64)
        b = right.to_numpy(dtype=np.float64)
        same = a.view(np.uint64) == b.view(np.uint64)
        both_nan = np.isnan(a) & np.isnan(b)
        differing = ~(same | both_nan)
        if not differing.any():
            return True, ""
        count = int(differing.sum())
        with np.errstate(invalid="ignore"):
            max_abs = float(np.nanmax(np.abs(a[differing] - b[differing])))
        first = int(np.argmax(differing))
        return False, f"{count}/{len(a)} elements differ, max abs {max_abs:.6g}, first at row {first}"

    unequal = (left.astype(str) != right.astype(str)).to_numpy()
    if not unequal.any():
        return True, ""
    return False, f"{int(unequal.sum())}/{len(left)} elements differ"


def _execution_block(directory: Path) -> dict | None:
    """Return a case's execution provenance, wherever the writer put it.

    App and Monte Carlo write it at the top level; the Article reproduction
    tool nests it under ``montecarlo_provenance``. Looking in one place only
    reports ``null`` for a run that did record its backend, which would defeat
    the point of checking.
    """
    provenance = json.loads((directory / "provenance.json").read_text())
    execution = provenance.get("execution")
    if execution is None:
        execution = (provenance.get("montecarlo_provenance") or {}).get("execution")
    return execution


def compare_case(reference: Path, candidate: Path) -> tuple[int, list[str]]:
    problems: list[str] = []
    compared = 0

    for table in TABLES:
        left_path, right_path = reference / table, candidate / table
        if not left_path.exists() or not right_path.exists():
            missing = left_path if not left_path.exists() else right_path
            problems.append(f"{table}: missing {missing}")
            continue

        left = pd.read_csv(left_path)
        right = pd.read_csv(right_path)

        for column in sorted(set(left.columns) - set(right.columns)):
            problems.append(f"{table}: column missing from candidate: {column}")
        for column in sorted(set(right.columns) - set(left.columns)):
            problems.append(f"{table}: column added by candidate: {column}")

        for column in sorted(set(left.columns) & set(right.columns)):
            compared += 1
            ok, detail = _identical(left[column], right[column])
            if not ok:
                problems.append(f"{table}::{column}: {detail}")

        print(f"  {table}: {len(left)} rows x {len(left.columns)} columns", flush=True)

    summary_left = json.loads((reference / "summary.json").read_text())
    summary_right = json.loads((candidate / "summary.json").read_text())
    if summary_left != summary_right:
        problems.append("summary.json differs")

    return compared, problems


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    reference, candidate = Path(sys.argv[1]), Path(sys.argv[2])

    print(f"reference: {reference}")
    print(f"candidate: {candidate}")
    backends = {}
    for label, directory in (("reference", reference), ("candidate", candidate)):
        execution = _execution_block(directory)
        backends[label] = (execution or {}).get("execution_backend")
        print(f"  {label} execution: {json.dumps(execution)}")
    print()

    compared, problems = compare_case(reference, candidate)

    # Without this the gate can silently compare two Python runs and call it a
    # pass. The candidate must say it ran the compiled path, and the two sides
    # must not report the same backend.
    if backends["candidate"] is None:
        problems.append("candidate records no execution backend — cannot prove the compiled path ran")
    elif backends["candidate"] == backends["reference"]:
        problems.append(f"both sides report backend {backends['candidate']!r} — the comparison proves nothing")
    print()
    if problems:
        for line in problems:
            print(line)
        print(f"\nFAIL: {len(problems)} problem(s) across {compared} compared columns")
        return 1
    print(f"PASS: {compared} columns bit-identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
