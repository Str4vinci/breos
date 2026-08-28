"""
I/O module for data export and import.

This module provides functions for:
- Exporting simulation results to CSV/TXT
- Saving cost analysis reports
- Generating formatted summary reports
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


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


def export_cost_analysis(
    cost_df: pd.DataFrame,
    results_directory: str,
    prefix: str = "",
    suffix: str = "",
    format: str = "csv",
    index: bool = False,
) -> str:
    """
    Export cost projection analysis to CSV or TXT.

    Args:
        cost_df: DataFrame from cost_analysis_projection()
        results_directory: Directory to save the file
        prefix: Optional prefix for filename
        suffix: Optional suffix for filename
        format: Output format ('csv' or 'txt')
        index: Whether to include DataFrame index

    Returns:
        Path to the saved file
    """
    os.makedirs(results_directory, exist_ok=True)

    parts = [p for p in [prefix, "cost_analysis", suffix] if p]
    filename = "_".join(parts) + f".{format}"
    filepath = os.path.join(results_directory, filename)

    if format == "csv":
        cost_df.to_csv(filepath, index=index)
    elif format == "txt":
        cost_df.to_csv(filepath, index=index, sep="\t")
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'txt'.")

    return filepath


def export_summary(
    summary_df: pd.DataFrame,
    results_directory: str,
    prefix: str = "",
    suffix: str = "",
    format: str = "txt",
    extra_metrics: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Export summary statistics as formatted text or CSV.

    Args:
        summary_df: Summary DataFrame (typically single row with key metrics)
        results_directory: Directory to save the file
        prefix: Optional prefix for filename
        suffix: Optional suffix for filename
        format: Output format ('txt' for formatted text, 'csv' for raw)
        extra_metrics: Optional label -> value pairs appended as additional
            summary fields (e.g. ``{"LCOE [EUR/kWh]": "0.1327"}``). Pre-format
            float values as strings to control their displayed precision.

    Returns:
        Path to the saved file
    """
    os.makedirs(results_directory, exist_ok=True)

    if extra_metrics:
        summary_df = summary_df.copy()
        for label, value in extra_metrics.items():
            summary_df[label] = value

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


def _economics_summary_metrics(cost_projection_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Pull headline economics figures from a cost projection's ``attrs``.

    The projection produced by :func:`breos.economics.cost_analysis_projection`
    stamps LCOE, payback, NPV savings, and total investment onto
    ``DataFrame.attrs``. This surfaces them as summary fields without any
    recomputation. Missing figures are skipped; an absent projection yields an
    empty dict.
    """
    if cost_projection_df is None:
        return {}

    attrs = cost_projection_df.attrs
    metrics: Dict[str, Any] = {}

    lcoe = attrs.get("lcoe_eur_kwh")
    if lcoe is not None and np.isfinite(lcoe):
        metrics["LCOE [EUR/kWh]"] = f"{float(lcoe):.4f}"

    total_investment = attrs.get("total_investment")
    if total_investment is not None:
        metrics["Total Investment [EUR]"] = f"{float(total_investment):.2f}"

    npv = attrs.get("final_npv_savings")
    if npv is not None:
        metrics["NPV Savings [EUR]"] = f"{float(npv):.2f}"

    if "payback_year" in attrs:
        payback = attrs.get("payback_year")
        metrics["Payback [year]"] = "N/A" if payback is None else int(payback)

    return metrics


def load_results(filepath: str, parse_dates: Union[bool, List[str]] = True) -> pd.DataFrame:
    """
    Load simulation results from CSV or TXT file.

    Args:
        filepath: Path to the results file
        parse_dates: Whether to parse datetime columns (True, False, or list of column names)

    Returns:
        DataFrame with loaded results
    """
    if filepath.endswith(".txt"):
        df = pd.read_csv(filepath, sep="\t", parse_dates=parse_dates)
    else:
        df = pd.read_csv(filepath, parse_dates=parse_dates)

    # Try to set Datetime as index if present
    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df.set_index("Datetime", inplace=True)

    return df
