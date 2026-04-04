#!/usr/bin/env python3
"""
Compare battery degradation models over 20 years for Porto.

Runs the multiyear simulation with 5 calendar models:
  - lam_calibrated (field-calibrated LFP, default)
  - modern_lfp (projected 2020+ cells, 50% k0)
  - lam (lab-derived LFP)
  - lam_calibrated_hourly (field-calibrated, hourly resolution legacy)
  - naumann (Naumann 2020, NMC/LFP lab)

Configuration: 5 panels, 5 kWh battery, Porto, hourly, 20 years.

Outputs comparison plots to results/degradation_model_comparison/
"""

import copy
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

# ── Configuration template (Porto, 5 panels, 5 kWh) ───────────────────────
BASE_CONFIG = {
    "name": "Degradation Model Comparison",
    "simulation_type": "multiyear",
    "extends": {
        "location": "base/locations.json#porto",
        "costs": "base/costs.json#residential_pt"
    },
    "pv": {
        "module": "Suntech_STP550S_STC",
        "n_modules": 5,
        "slope": 35,
        "azimuth": 180
    },
    "battery": {
        "nominal_kwh": 5,
        "dc_coupled": True,
        "initial_soh": 100,
        "eol_percentage": 0.70,
        "enable_replacement": True,
        "replacement_cost": "calculate",
        "max_soc": 0.90,
        "min_soc": 0.10,
        "charge_efficiency": 0.974679434,
        "discharge_efficiency": 0.974679434,
        "calendar_model": "lam_calibrated"
    },
    "inverter": {
        "dc_ac_ratio": 1.25,
        "efficiency": 0.96
    },
    "load": {
        "source": "6",
        "annual_consumption_kwh": 5000
    },
    "simulation": {
        "start_year": 2005,
        "years": 20,
        "weather_file": "weather/porto_historical_2005_2024_openmeteo.csv",
        "resolution": "h"
    },
    "output": {
        "folder": "",
        "plots": False,
        "csv": True,
        "breakeven": False
    }
}

MODELS = [
    ("lam_calibrated",       "Lam calibrated (field)"),
    ("modern_lfp",           "Modern LFP (2020+)"),
    ("lam",                  "Lam (lab)"),
    ("lam_calibrated_hourly","Lam calibrated (hourly)"),
    ("naumann",              "Naumann 2020"),
]

COLORS = {
    "lam_calibrated":        "#1f77b4",
    "modern_lfp":            "#2ca02c",
    "lam":                   "#d62728",
    "lam_calibrated_hourly": "#ff7f0e",
    "naumann":               "#9467bd",
}
LINESTYLES = {
    "lam_calibrated":        "-",
    "modern_lfp":            "--",
    "lam":                   "-.",
    "lam_calibrated_hourly": ":",
    "naumann":               (0, (3, 1, 1, 1)),
}

OUTPUT_DIR = str(PROJECT_ROOT / "results" / "degradation_model_comparison")


def resolve_config(raw_config: dict) -> dict:
    """Resolve extends directives in config."""
    from run_simulation import resolve_extends
    config_dir = PROJECT_ROOT / "configs"
    return resolve_extends(raw_config, config_dir)


def run_single_model(model_key: str, label: str) -> dict:
    """Run the 20-year simulation for one calendar model and return paths."""
    from run_simulation import run_multiyear

    model_dir = os.path.join(OUTPUT_DIR, model_key)
    os.makedirs(model_dir, exist_ok=True)

    cfg = copy.deepcopy(BASE_CONFIG)
    cfg["battery"]["calendar_model"] = model_key
    cfg["output"]["folder"] = model_dir
    cfg["output"]["plots"] = True
    cfg["output"]["csv"] = True
    cfg["output"]["breakeven"] = True
    cfg["name"] = f"Porto 5p5kWh – {label}"

    # Resolve extends
    cfg = resolve_config(cfg)

    print(f"\n{'='*60}")
    print(f"  Running: {label} ({model_key})")
    print(f"{'='*60}")
    run_multiyear(cfg)

    return {
        "model": model_key,
        "label": label,
        "dir": model_dir,
        "degradation_csv": os.path.join(model_dir, "degradation_data.csv"),
        "cost_csv": os.path.join(model_dir, "cost_projection.csv"),
    }


def load_results(runs: list) -> dict:
    """Load degradation and cost CSVs for all runs."""
    data = {}
    for r in runs:
        model = r["model"]
        deg_path = r["degradation_csv"]
        cost_path = r["cost_csv"]

        deg_df = None
        if os.path.exists(deg_path):
            deg_df = pd.read_csv(deg_path, parse_dates=["Datetime"])

        cost_df = None
        if os.path.exists(cost_path):
            cost_df = pd.read_csv(cost_path)

        data[model] = {
            "label": r["label"],
            "deg": deg_df,
            "cost": cost_df,
        }
    return data


def plot_comparison(data: dict):
    """Generate all comparison plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fs, lfs = 11, 12

    # ── 1. SOH over 20 years ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for model, label in MODELS:
        d = data.get(model)
        if d is None or d["deg"] is None:
            continue
        deg = d["deg"]
        years = (deg["Datetime"] - deg["Datetime"].iloc[0]).dt.total_seconds() / (365.25 * 86400)
        ax.plot(years, deg["SOH"], color=COLORS[model],
                linestyle=LINESTYLES[model], linewidth=2, label=d["label"])
    ax.set_xlabel("Years", fontsize=lfs)
    ax.set_ylabel("SOH (%)", fontsize=lfs)
    ax.set_xlim(0, 20)
    ax.set_xticks(range(0, 21))
    ax.legend(fontsize=fs, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=fs)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "soh_comparison.png"), dpi=200)
    plt.close(fig)
    print("Saved soh_comparison.png")

    # ── 2. Calendar vs Cycle degradation components ───────────────────────
    # One plot per model (otherwise too chaotic)
    for model, label in MODELS:
        d = data.get(model)
        if d is None or d["deg"] is None:
            continue
        deg = d["deg"]
        years = (deg["Datetime"] - deg["Datetime"].iloc[0]).dt.total_seconds() / (365.25 * 86400)

        # Use Global columns if available, otherwise Cumulative
        cal_col = "Global_Calendar_Degradation" if "Global_Calendar_Degradation" in deg.columns else "Cumulative_Calendar_Degradation"
        cyc_col = "Global_Cycle_Degradation" if "Global_Cycle_Degradation" in deg.columns else "Cumulative_Cycle_Degradation"

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.fill_between(years, 0, deg[cal_col] * 100, alpha=0.4,
                         color="#1f77b4", label="Calendar aging")
        ax.fill_between(years, deg[cal_col] * 100,
                         (deg[cal_col] + deg[cyc_col]) * 100,
                         alpha=0.4, color="#d62728", label="Cycle aging")
        ax.plot(years, (deg[cal_col] + deg[cyc_col]) * 100,
                color="black", linewidth=1.5, label="Total")
        ax.set_xlabel("Years", fontsize=lfs)
        ax.set_ylabel("Cumulative capacity fade (%)", fontsize=lfs)
        ax.set_xlim(0, 20)
        ax.set_xticks(range(0, 21))
        ax.legend(fontsize=fs)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=fs)
        fig.tight_layout()
        fname = f"degradation_components_{model}.png"
        fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=200)
        plt.close(fig)
        print(f"Saved {fname}")

    # ── 3. All models stacked: calendar vs cycle split (grouped bar) ──────
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(MODELS))
    width = 0.6
    cal_vals = []
    cyc_vals = []
    labels = []
    for model, label in MODELS:
        d = data.get(model)
        if d is None or d["deg"] is None:
            cal_vals.append(0)
            cyc_vals.append(0)
            labels.append(label)
            continue
        deg = d["deg"]
        cal_col = "Global_Calendar_Degradation" if "Global_Calendar_Degradation" in deg.columns else "Cumulative_Calendar_Degradation"
        cyc_col = "Global_Cycle_Degradation" if "Global_Cycle_Degradation" in deg.columns else "Cumulative_Cycle_Degradation"
        cal_vals.append(deg[cal_col].iloc[-1] * 100)
        cyc_vals.append(deg[cyc_col].iloc[-1] * 100)
        labels.append(label)

    bars_cal = ax.bar(x, cal_vals, width, label="Calendar", color="#1f77b4", alpha=0.8)
    bars_cyc = ax.bar(x, cyc_vals, width, bottom=cal_vals, label="Cycle", color="#d62728", alpha=0.8)

    # Add total labels
    for i, (c, cy) in enumerate(zip(cal_vals, cyc_vals)):
        total = c + cy
        ax.text(i, total + 0.5, f"{total:.1f}%", ha="center", fontsize=fs - 1, fontweight="bold")

    ax.set_ylabel("Total capacity fade at year 20 (%)", fontsize=lfs)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=fs - 1, rotation=15, ha="right")
    ax.legend(fontsize=fs)
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(labelsize=fs)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "degradation_split_bar.png"), dpi=200)
    plt.close(fig)
    print("Saved degradation_split_bar.png")

    # ── 4. Economic progression: cumulative cost with vs without system ───
    fig, ax = plt.subplots(figsize=(10, 6))
    baseline_plotted = False
    for model, label in MODELS:
        d = data.get(model)
        if d is None or d["cost"] is None:
            continue
        cost = d["cost"]
        if not baseline_plotted and "Cost_No_Sys_Cumulative" in cost.columns:
            ax.plot(cost["Year"], cost["Cost_No_Sys_Cumulative"],
                    color="grey", linewidth=2.5, label="No system (grid only)")
            baseline_plotted = True
        if "Cost_System_Cumulative" in cost.columns:
            ax.plot(cost["Year"], cost["Cost_System_Cumulative"],
                    color=COLORS[model], linestyle=LINESTYLES[model],
                    linewidth=2, label=d["label"])
    ax.set_xlabel("Year", fontsize=lfs)
    ax.set_ylabel("Cumulative cost (€)", fontsize=lfs)
    ax.set_xlim(1, 20)
    ax.set_xticks(range(1, 21))
    ax.legend(fontsize=fs - 1)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=fs)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "economic_cumulative_cost.png"), dpi=200)
    plt.close(fig)
    print("Saved economic_cumulative_cost.png")

    # ── 5. Cumulative savings (NPV) ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for model, label in MODELS:
        d = data.get(model)
        if d is None or d["cost"] is None:
            continue
        cost = d["cost"]
        npv_col = "Savings_Cumulative_NPV" if "Savings_Cumulative_NPV" in cost.columns else "Savings_Cumulative"
        if npv_col in cost.columns:
            ax.plot(cost["Year"], cost[npv_col],
                    color=COLORS[model], linestyle=LINESTYLES[model],
                    linewidth=2, label=d["label"])
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Year", fontsize=lfs)
    ax.set_ylabel("Cumulative savings NPV (€)", fontsize=lfs)
    ax.set_xlim(1, 20)
    ax.set_xticks(range(1, 21))
    ax.legend(fontsize=fs - 1)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=fs)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "economic_savings_npv.png"), dpi=200)
    plt.close(fig)
    print("Saved economic_savings_npv.png")

    # ── 6. Summary table ──────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print(f"{'Model':<30s} {'Final SOH':>10s} {'Cal fade':>10s} {'Cyc fade':>10s} "
          f"{'Replacements':>12s} {'NPV Savings':>12s} {'Breakeven':>10s}")
    print("=" * 100)
    for model, label in MODELS:
        d = data.get(model)
        if d is None or d["deg"] is None:
            print(f"{label:<30s} {'N/A':>10s}")
            continue
        deg = d["deg"]
        cost = d["cost"]

        final_soh = deg["SOH"].iloc[-1]
        cal_col = "Global_Calendar_Degradation" if "Global_Calendar_Degradation" in deg.columns else "Cumulative_Calendar_Degradation"
        cyc_col = "Global_Cycle_Degradation" if "Global_Cycle_Degradation" in deg.columns else "Cumulative_Cycle_Degradation"
        cal_fade = deg[cal_col].iloc[-1] * 100
        cyc_fade = deg[cyc_col].iloc[-1] * 100

        # Count replacements (SOH jumps back up)
        soh_series = deg["SOH"].values
        replacements = sum(1 for i in range(1, len(soh_series)) if soh_series[i] > soh_series[i-1] + 5)

        npv_savings = "N/A"
        breakeven = "N/A"
        if cost is not None:
            npv_col = "Savings_Cumulative_NPV" if "Savings_Cumulative_NPV" in cost.columns else None
            if npv_col:
                npv_savings = f"€{cost[npv_col].iloc[-1]:,.0f}"
                positive = cost[cost[npv_col] > 0]
                if len(positive) > 0:
                    breakeven = f"Year {int(positive['Year'].iloc[0])}"
                else:
                    breakeven = "> 20y"

        print(f"{label:<30s} {final_soh:>9.1f}% {cal_fade:>9.1f}% {cyc_fade:>9.1f}% "
              f"{replacements:>12d} {npv_savings:>12s} {breakeven:>10s}")
    print("=" * 100)
    print(f"\nAll plots saved to: {OUTPUT_DIR}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Run all 5 models
    runs = []
    for model_key, label in MODELS:
        result = run_single_model(model_key, label)
        runs.append(result)

    # Load and plot
    data = load_results(runs)
    plot_comparison(data)


if __name__ == "__main__":
    main()
