"""Shared input and output for the oracle command lines.

Every oracle writes a JSON summary and, on request, a CSV of its schedule.
Both carry a schema tag: the JSON under ``"schema"``, the CSV on a first
comment line (read it with ``pandas.read_csv(path, comment="#")``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from breos.cli import _load_config
from breos.io import nonfinite_to_none


def load_config(path: str | Path) -> dict[str, Any]:
    """An App configuration from a TOML or JSON file, read as ``breos run --config`` reads it."""
    return _load_config(Path(path))


def write_json(payload: dict[str, Any], path: str | Path | None) -> None:
    """Write ``payload`` as strict JSON to ``path``, or to standard output for None or ``-``.

    A non-finite float, such as an absent power limit, is written as null.
    """
    text = json.dumps(nonfinite_to_none(payload), indent=2, allow_nan=False) + "\n"
    if path is None or str(path) == "-":
        sys.stdout.write(text)
        return
    Path(path).write_text(text, encoding="utf-8")


def write_csv(frame: pd.DataFrame, schema: str, path: str | Path) -> None:
    """Write ``frame`` to ``path`` under a ``# schema: <schema>`` first line."""
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"# schema: {schema}\n")
        frame.to_csv(handle, index=False, lineterminator="\n")
