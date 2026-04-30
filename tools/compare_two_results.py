#!/usr/bin/env python3
"""
Compare Break-even: Two Simulation Results

Usage:
    python compare_two_results.py <folder1> <folder2> [label1] [label2] [output_dir]

Examples:
    python compare_two_results.py results/singleyear_porto_15min results/singleyear_porto_15min_nobatt
    python compare_two_results.py results/singleyear_porto_15min results/singleyear_porto_15min_nobatt "With Battery" "No Battery"
"""

import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from breos.plotting import plot_breakeven_two


def _interpolate_breakeven(df: pd.DataFrame):
    savings = df["Savings_Cumulative_NPV"].values
    years = df["Year"].values
    for i in range(1, len(savings)):
        if savings[i] >= 0 and savings[i - 1] < 0:
            frac = -savings[i - 1] / (savings[i] - savings[i - 1])
            return years[i - 1] + frac
    return None


def _fmt(be):
    if be is None:
        return "N/A"
    y, m = int(be), int((be - int(be)) * 12)
    return f"{y}y" if m == 0 else f"{y}y {m}m"


def compare_two(
    folder1: str,
    folder2: str,
    label1: str = None,
    label2: str = None,
    output_dir: str = "results/comparison",
):
    """Generate comparison break-even plot for two simulation results."""
    if label1 is None:
        label1 = os.path.basename(folder1.rstrip("/"))
    if label2 is None:
        label2 = os.path.basename(folder2.rstrip("/"))

    df1 = pd.read_csv(f"{folder1}/cost_projection.csv")
    df2 = pd.read_csv(f"{folder2}/cost_projection.csv")
    be1 = _interpolate_breakeven(df1)
    be2 = _interpolate_breakeven(df2)

    safe_name = f"{label1}_vs_{label2}".replace(" ", "_").replace("/", "_")
    filename = f"breakeven_{safe_name}.png"
    os.makedirs(output_dir, exist_ok=True)

    plot_breakeven_two(df1, label1, be1, df2, label2, be2, output_dir, filename)
    output_path = os.path.join(output_dir, filename)

    max_year = int(df1["Year"].max())
    cost1 = df1.loc[df1["Year"] == max_year, "Cost_System_Cumulative_NPV"].values[0]
    cost2 = df2.loc[df2["Year"] == max_year, "Cost_System_Cumulative_NPV"].values[0]

    print(f"Saved: {output_path}")
    print()
    print("Summary:")
    print(f"{'System':<30} {'Break-even':<12} {f'{max_year}-yr Cost':<12}")
    print("-" * 54)
    print(f"{label1:<30} {_fmt(be1):<12} {cost1:>10,.0f}€")
    print(f"{label2:<30} {_fmt(be2):<12} {cost2:>10,.0f}€")
    print()
    print(f"Difference ({label2} - {label1}): {cost2 - cost1:+,.0f}€")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    compare_two(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3] if len(sys.argv) > 3 else None,
        sys.argv[4] if len(sys.argv) > 4 else None,
        sys.argv[5] if len(sys.argv) > 5 else "results/comparison",
    )
