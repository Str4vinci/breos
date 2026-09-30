"""
I/O module for data export and import.

This module provides functions for:
- Exporting simulation results to CSV/TXT
- Generating formatted summary reports
- Preparing result payloads for strict JSON
- Repairing measured load and PV series before a simulation

``repair_series``, ``InputRepairReport`` and ``RepairEvent`` are re-exported
from :mod:`breos.repair`.
"""

import math
import os
from pathlib import Path
from typing import Any, List, Union

import numpy as np
import pandas as pd

from breos.repair import InputRepairReport, RepairEvent, repair_series  # noqa: F401 - public re-export
from breos.utils import local_datetime_index


def nonfinite_to_none(value: Any) -> Any:
    """Return ``value`` with every non-finite float replaced by ``None``.

    Walks dicts, lists and tuples (tuples become lists, as JSON writes them).
    Python and NumPy floats that are NaN or infinite become ``None``; finite
    NumPy floats become Python floats. Everything else is returned unchanged.

    Use it on result payloads where a metric can be legitimately undefined,
    such as the LCOE of a system with no production, so the payload can be
    written as strict JSON (``allow_nan=False``), where ``null`` is the only
    way to say "no value".
    """
    if isinstance(value, dict):
        return {key: nonfinite_to_none(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [nonfinite_to_none(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def export_results(
    results_df: pd.DataFrame,
    results_directory: str,
    prefix: str = "",
    suffix: str = "",
    format: str = "csv",
    index: bool = False,
) -> str:
    """
    Export simulation results to CSV or TXT.

    Args:
        results_df: DataFrame with simulation results
        results_directory: Directory to save the file
        prefix: Optional prefix for filename
        suffix: Optional suffix for filename
        format: Output format ('csv' or 'txt')
        index: Whether to include DataFrame index

    Returns:
        Path to the saved file
    """
    os.makedirs(results_directory, exist_ok=True)

    # Build filename
    parts = [p for p in [prefix, "results", suffix] if p]
    filename = "_".join(parts) + f".{format}"
    filepath = os.path.join(results_directory, filename)

    if format == "csv":
        results_df.to_csv(filepath, index=index)
    elif format == "txt":
        results_df.to_csv(filepath, index=index, sep="\t")
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'txt'.")

    return filepath


def export_summary(
    summary_df: pd.DataFrame,
    results_directory: str,
    prefix: str = "",
    suffix: str = "",
    format: str = "txt",
) -> str:
    """
    Export summary statistics as formatted text or CSV.

    Args:
        summary_df: Summary DataFrame (typically single row with key metrics)
        results_directory: Directory to save the file
        prefix: Optional prefix for filename
        suffix: Optional suffix for filename
        format: Output format ('txt' for formatted text, 'csv' for raw)

    Returns:
        Path to the saved file
    """
    os.makedirs(results_directory, exist_ok=True)

    parts = [p for p in [prefix, "summary", suffix] if p]
    filename = "_".join(parts) + f".{format}"
    filepath = os.path.join(results_directory, filename)

    if format == "txt":
        with open(filepath, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("SIMULATION SUMMARY\n")
            f.write("=" * 60 + "\n\n")

            for col in summary_df.columns:
                value = summary_df[col].iloc[0]
                if isinstance(value, float):
                    f.write(f"{col}: {value:.2f}\n")
                else:
                    f.write(f"{col}: {value}\n")

            f.write("\n" + "=" * 60 + "\n")
    else:
        summary_df.to_csv(filepath, index=False)

    return filepath


def load_results(filepath: Union[str, os.PathLike], parse_dates: Union[bool, List[str]] = True) -> pd.DataFrame:
    """
    Load simulation results from CSV or TXT file.

    Args:
        filepath: Path to the results file, as a string or path-like
        parse_dates: Whether to parse datetime columns (True, False, or list of column names)

    Returns:
        DataFrame with loaded results. A ``Datetime`` column becomes the index,
        on the results' own calendar: a run in a DST zone, whose CSV mixes
        UTC offsets, keeps each row's wall-clock time. That index repeats an
        hour in autumn and skips one in spring, so such a file also gets a
        ``Datetime_UTC`` column with each row's absolute time, which the
        energy plots use to find the step length.
    """
    if Path(filepath).suffix == ".txt":
        df = pd.read_csv(filepath, sep="\t", parse_dates=parse_dates)
    else:
        df = pd.read_csv(filepath, parse_dates=parse_dates)

    # Try to set Datetime as index if present
    if "Datetime" in df.columns:
        raw = df["Datetime"]
        df["Datetime"] = local_datetime_index(raw)
        if df["Datetime"].dt.tz is None and not pd.api.types.is_datetime64_any_dtype(raw):
            # Naive text parses to the same values as UTC; text with mixed
            # offsets does not, and its wall-clock index loses the instants.
            instants = pd.to_datetime(raw, utc=True)
            if not (instants.dt.tz_localize(None).to_numpy() == df["Datetime"].to_numpy()).all():
                df["Datetime_UTC"] = instants
        df.set_index("Datetime", inplace=True)

    return df
