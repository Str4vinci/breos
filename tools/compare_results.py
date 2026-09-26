#!/usr/bin/env python3
"""
Compare Break-even: Multiple Simulation Results

Usage:
    python compare_results.py <folder1> [folder2] [folder3] ... [--output DIR] [--labels "Label1,Label2,..."]

Examples:
    # Two folders (uses folder names as labels)
    python compare_results.py results/singleyear_porto_15min results/singleyear_porto_15min_nobatt

    # Three folders with custom labels
    python compare_results.py results/pv_battery results/pv_only results/high_consumption --labels "PV + Battery,PV Only,High Consumption"

    # Custom output directory
    python compare_results.py results/folder1 results/folder2 --output results/my_comparison
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from breos.economics import find_payback_year_exact
from breos.plotting import plot_breakeven_comparison
from breos.utils import format_years_months

# Keep script presentation choices local instead of importing a private symbol
# from the public plotting module.
BREAKEVEN_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b")


def compare_results(folders: list, labels: list = None, output_dir: str = "results/comparison"):
    """Generate comparison break-even plot for multiple simulation results."""
    if not folders:
        print("Error: At least one folder required")
        return

    if labels is None or len(labels) != len(folders):
        labels = [os.path.basename(f.rstrip("/")) for f in folders]

    cost_dfs = []
    valid_labels = []
    valid_colors = []
    max_year = 20

    for i, (folder, label) in enumerate(zip(folders, labels)):
        try:
            df = pd.read_csv(f"{folder}/cost_projection.csv")
            cost_dfs.append(df)
            valid_labels.append(label)
            valid_colors.append(BREAKEVEN_COLORS[i % len(BREAKEVEN_COLORS)])
            max_year = int(df["Year"].max())
        except FileNotFoundError:
            print(f"Warning: {folder}/cost_projection.csv not found, skipping")
        except Exception as e:
            print(f"Error loading {folder}: {e}")

    if not cost_dfs:
        print("No valid result folders found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    plot_breakeven_comparison(cost_dfs, valid_labels, valid_colors, output_dir, "breakeven_comparison.png")
    output_path = os.path.join(output_dir, "breakeven_comparison.png")
    print(f"Saved: {output_path}")

    # Summary
    print()
    print("Summary:")
    print(f"{'System':<35} {'Break-even':<12} {f'{max_year}-yr Cost':<14} {'No System Cost':<14}")
    print("-" * 77)
    summary = []
    for df, label in zip(cost_dfs, valid_labels):
        be = find_payback_year_exact(df)
        cost = df.loc[df["Year"] == max_year, "Cost_System_Cumulative_NPV"].values[0]
        no_sys_cost = df.loc[df["Year"] == max_year, "Cost_No_Sys_Cumulative_NPV"].values[0]
        summary.append((label, be, cost, no_sys_cost))
    for label, be, cost, no_sys_cost in sorted(summary, key=lambda x: x[2]):
        print(f"{label:<35} {format_years_months(be):<12} {cost:>12,.0f}€  {no_sys_cost:>12,.0f}€")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Compare multiple simulation results")
    parser.add_argument("folders", nargs="+", help="Result folders to compare")
    parser.add_argument("--output", "-o", default="results/comparison", help="Output directory")
    parser.add_argument("--labels", "-l", help="Comma-separated labels for each folder")
    args = parser.parse_args()
    labels = args.labels.split(",") if args.labels else None
    compare_results(args.folders, labels, args.output)


if __name__ == "__main__":
    main()
