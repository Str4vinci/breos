#!/usr/bin/env python3
"""
Compare battery degradation models across multiple scenarios.

Runs 5 calendar models × 4 scenarios (20 simulations total):
  Scenarios:
    - Porto 5p/5kWh  (E-Redes BTN C residential load)
    - Berlin 5p/5kWh (H0SLP demandlib load)
    - Porto 10p/10kWh
    - Berlin 10p/10kWh

  Calendar models:
    - naumann_lam_calibrated (Naumann cycle + Lam calendar, field-calibrated, default)
    - naumann_lam_calibrated_hourly (field-calibrated, hourly resolution)
    - naumann_lam_modern (projected 2020+ cells, 0.5×k₀)
    - naumann_lam (Naumann cycle + Lam lab calendar)
    - naumann (pure Naumann 2020, NMC/LFP lab)

Outputs to results/degradation_comparison/
"""

import copy
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

# ── Scenarios ─────────────────────────────────────────────────────────────────
SCENARIOS = {
    "porto_5p5kwh": {
        "label": "Porto 5p/5kWh",
        "extends": {
            "location": "base/locations.json#porto",
            "costs": "base/costs.json#residential_pt",
        },
        "pv": {
            "module": "Suntech_STP550S_STC",
            "n_modules": 5,
            "slope": 35,
            "azimuth": 180,
        },
        "battery": {"nominal_kwh": 5},
        "load": {"source": "6", "annual_consumption_kwh": 5000},
        "simulation": {
            "start_year": 2005,
            "years": 20,
            "weather_file": "weather/porto_historical_2005_2024_openmeteo.csv",
            "resolution": "h",
        },
    },
    "berlin_5p5kwh": {
        "label": "Berlin 5p/5kWh",
        "extends": {
            "location": "base/locations.json#berlin",
            "costs": "base/costs.json#residential_de",
        },
        "pv": {
            "module": "Suntech_STP550S_STC",
            "n_modules": 5,
            "slope": 40,
            "azimuth": 180,
        },
        "battery": {"nominal_kwh": 5},
        "load": {"source": "1", "annual_consumption_kwh": 5000},
        "simulation": {
            "start_year": 2005,
            "years": 20,
            "weather_file": "weather/berlin_historical_2005_2024_openmeteo.csv",
            "resolution": "h",
        },
    },
    "porto_10p10kwh": {
        "label": "Porto 10p/10kWh",
        "extends": {
            "location": "base/locations.json#porto",
            "costs": "base/costs.json#residential_pt",
        },
        "pv": {
            "module": "Suntech_STP550S_STC",
            "n_modules": 10,
            "slope": 35,
            "azimuth": 180,
        },
        "battery": {"nominal_kwh": 10},
        "load": {"source": "6", "annual_consumption_kwh": 5000},
        "simulation": {
            "start_year": 2005,
            "years": 20,
            "weather_file": "weather/porto_historical_2005_2024_openmeteo.csv",
            "resolution": "h",
        },
    },
    "berlin_10p10kwh": {
        "label": "Berlin 10p/10kWh",
        "extends": {
            "location": "base/locations.json#berlin",
            "costs": "base/costs.json#residential_de",
        },
        "pv": {
            "module": "Suntech_STP550S_STC",
            "n_modules": 10,
            "slope": 40,
            "azimuth": 180,
        },
        "battery": {"nominal_kwh": 10},
        "load": {"source": "1", "annual_consumption_kwh": 5000},
        "simulation": {
            "start_year": 2005,
            "years": 20,
            "weather_file": "weather/berlin_historical_2005_2024_openmeteo.csv",
            "resolution": "h",
        },
    },
}

# ── Calendar models (grouped) ─────────────────────────────────────────────────
MODEL_GROUPS = {
    "Field-calibrated": [
        ("naumann_lam_calibrated",        "Naumann-Lam (field-calibrated)"),
        ("naumann_lam_calibrated_hourly", "Naumann-Lam (field, hourly res.)"),
    ],
    "Projected": [
        ("naumann_lam_modern",            "Naumann-Lam modern (projected, 0.5\u00d7k\u2080)"),
    ],
    "Lab-derived": [
        ("naumann_lam",                   "Naumann-Lam (lab)"),
        ("naumann",                       "Naumann (2020, lab)"),
    ],
}

# Flat list preserving group order
MODELS = []
for _group_models in MODEL_GROUPS.values():
    MODELS.extend(_group_models)

# Model → group mapping
MODEL_TO_GROUP = {}
for group_name, group_models in MODEL_GROUPS.items():
    for model_key, _ in group_models:
        MODEL_TO_GROUP[model_key] = group_name

# Colors: blue family for field, green for projected, red family for lab
COLORS = {
    "naumann_lam_calibrated":        "#1a5276",   # dark blue (field)
    "naumann_lam_calibrated_hourly": "#5dade2",   # light blue (field)
    "naumann_lam_modern":            "#27ae60",   # green (projected)
    "naumann_lam":                   "#c0392b",   # dark red (lab)
    "naumann":                       "#e74c3c",   # light red (lab)
}
LINESTYLES = {
    "naumann_lam_calibrated":        "-",
    "naumann_lam_calibrated_hourly": "--",
    "naumann_lam_modern":            "-",
    "naumann_lam":                   "-",
    "naumann":                       "--",
}
# Group colors for bar charts
GROUP_COLORS = {
    "Field-calibrated": "#2471a3",
    "Projected":        "#27ae60",
    "Lab-derived":      "#c0392b",
}

BATTERY_DEFAULTS = {
    "dc_coupled": True,
    "initial_soh": 100,
    "eol_percentage": 0.70,
    "enable_replacement": True,
    "replacement_cost": "calculate",
    "max_soc": 0.90,
    "min_soc": 0.10,
    "charge_efficiency": 0.974679434,
    "discharge_efficiency": 0.974679434,
    "calendar_model": "naumann_lam_calibrated",
}

OUTPUT_ROOT = str(PROJECT_ROOT / "results" / "degradation_comparison")


def resolve_config(raw_config: dict) -> dict:
    from run_simulation import resolve_extends
    config_dir = PROJECT_ROOT / "configs"
    return resolve_extends(raw_config, config_dir)


def build_config(scenario_key: str, model_key: str, model_label: str) -> dict:
    """Build a full simulation config for one scenario + model combination."""
    sc = SCENARIOS[scenario_key]
    cfg = {
        "name": f"{sc['label']} – {model_label}",
        "simulation_type": "multiyear",
        "extends": copy.deepcopy(sc["extends"]),
        "pv": copy.deepcopy(sc["pv"]),
        "battery": {**BATTERY_DEFAULTS, **sc["battery"], "calendar_model": model_key},
        "inverter": {"dc_ac_ratio": 1.25, "efficiency": 0.96},
        "load": copy.deepcopy(sc["load"]),
        "simulation": copy.deepcopy(sc["simulation"]),
        "output": {
            "folder": os.path.join(OUTPUT_ROOT, scenario_key, model_key),
            "plots": True,
            "csv": True,
            "breakeven": True,
        },
    }
    return cfg


def run_single(scenario_key: str, model_key: str, model_label: str) -> dict:
    """Run one 20-year simulation."""
    from run_simulation import run_multiyear

    cfg = build_config(scenario_key, model_key, model_label)
    os.makedirs(cfg["output"]["folder"], exist_ok=True)

    cfg = resolve_config(cfg)

    sc_label = SCENARIOS[scenario_key]["label"]
    print(f"\n{'='*60}")
    print(f"  {sc_label} | {model_label} ({model_key})")
    print(f"{'='*60}")
    run_multiyear(cfg)

    out_dir = cfg["output"]["folder"]
    return {
        "scenario": scenario_key,
        "model": model_key,
        "label": model_label,
        "dir": out_dir,
        "degradation_csv": os.path.join(out_dir, "degradation_data.csv"),
        "cost_csv": os.path.join(out_dir, "cost_projection.csv"),
    }


def load_results(runs: list) -> dict:
    """Load all results keyed by (scenario, model)."""
    data = {}
    for r in runs:
        key = (r["scenario"], r["model"])
        deg_df = None
        if os.path.exists(r["degradation_csv"]):
            deg_df = pd.read_csv(r["degradation_csv"], parse_dates=["Datetime"])
        cost_df = None
        if os.path.exists(r["cost_csv"]):
            cost_df = pd.read_csv(r["cost_csv"])
        data[key] = {"label": r["label"], "deg": deg_df, "cost": cost_df}
    return data


def _grouped_legend(ax, fs):
    """Create a legend with group headers."""
    from matplotlib.lines import Line2D
    handles, labels = [], []
    for group_name, group_models in MODEL_GROUPS.items():
        # Group header (invisible line, bold label)
        handles.append(Line2D([], [], color="none"))
        labels.append(f"$\\bf{{{group_name}}}$")
        for model_key, model_label in group_models:
            handles.append(Line2D([0], [0], color=COLORS[model_key],
                                   linestyle=LINESTYLES[model_key], linewidth=2))
            labels.append(f"  {model_label}")
    ax.legend(handles, labels, fontsize=fs, loc="lower left",
              handlelength=2.5, labelspacing=0.4)


def _get_deg_cols(deg):
    """Return (cal_col, cyc_col) names from degradation df."""
    cal = "Global_Calendar_Degradation" if "Global_Calendar_Degradation" in deg.columns else "Cumulative_Calendar_Degradation"
    cyc = "Global_Cycle_Degradation" if "Global_Cycle_Degradation" in deg.columns else "Cumulative_Cycle_Degradation"
    return cal, cyc


def _get_years(deg):
    """Convert Datetime column to fractional years from start."""
    return (deg["Datetime"] - deg["Datetime"].iloc[0]).dt.total_seconds() / (365.25 * 86400)


def plot_all(data: dict):
    """Generate comparison plots per scenario and cross-scenario."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    fs, lfs = 11, 12

    # ── Per-scenario plots ────────────────────────────────────────────────
    for sc_key, sc_info in SCENARIOS.items():
        sc_dir = os.path.join(OUTPUT_ROOT, sc_key)
        os.makedirs(sc_dir, exist_ok=True)
        sc_label = sc_info["label"]

        # 1. SOH comparison (grouped legend)
        fig, ax = plt.subplots(figsize=(10, 6))
        for model_key, model_label in MODELS:
            d = data.get((sc_key, model_key))
            if d is None or d["deg"] is None:
                continue
            deg = d["deg"]
            ax.plot(_get_years(deg), deg["SOH"], color=COLORS[model_key],
                    linestyle=LINESTYLES[model_key], linewidth=2)
        ax.set_xlabel("Years", fontsize=lfs)
        ax.set_ylabel("SOH (%)", fontsize=lfs)
        ax.set_xlim(0, 20)
        ax.set_xticks(range(0, 21))
        _grouped_legend(ax, fs)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=fs)
        fig.tight_layout()
        fig.savefig(os.path.join(sc_dir, "soh_comparison.png"), dpi=200)
        plt.close(fig)
        print(f"[{sc_label}] Saved soh_comparison.png")

        # 2. Degradation components per model
        for model_key, model_label in MODELS:
            d = data.get((sc_key, model_key))
            if d is None or d["deg"] is None:
                continue
            deg = d["deg"]
            years = _get_years(deg)
            cal_col, cyc_col = _get_deg_cols(deg)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.fill_between(years, 0, deg[cal_col] * 100, alpha=0.4,
                            color="#2471a3", label="Calendar aging")
            ax.fill_between(years, deg[cal_col] * 100,
                            (deg[cal_col] + deg[cyc_col]) * 100,
                            alpha=0.4, color="#c0392b", label="Cycle aging")
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
            fname = f"degradation_components_{model_key}.png"
            fig.savefig(os.path.join(sc_dir, fname), dpi=200)
            plt.close(fig)
            print(f"[{sc_label}] Saved {fname}")

        # 3. Degradation split bar (grouped with separators)
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_positions = []
        bar_cal, bar_cyc, bar_labels, bar_colors = [], [], [], []
        pos = 0
        for group_name, group_models in MODEL_GROUPS.items():
            for model_key, model_label in group_models:
                d = data.get((sc_key, model_key))
                if d is not None and d["deg"] is not None:
                    deg = d["deg"]
                    cal_col, cyc_col = _get_deg_cols(deg)
                    bar_cal.append(deg[cal_col].iloc[-1] * 100)
                    bar_cyc.append(deg[cyc_col].iloc[-1] * 100)
                else:
                    bar_cal.append(0)
                    bar_cyc.append(0)
                bar_labels.append(model_label)
                bar_colors.append(GROUP_COLORS[group_name])
                bar_positions.append(pos)
                pos += 1
            pos += 0.5  # gap between groups

        x = np.array(bar_positions)
        width = 0.7
        # Calendar bars with group-specific color
        for i in range(len(x)):
            ax.bar(x[i], bar_cal[i], width, color=bar_colors[i], alpha=0.7)
            ax.bar(x[i], bar_cyc[i], width, bottom=bar_cal[i],
                   color=bar_colors[i], alpha=0.35, hatch="//")
        # Total labels
        for i in range(len(x)):
            total = bar_cal[i] + bar_cyc[i]
            ax.text(x[i], total + 0.5, f"{total:.1f}%", ha="center",
                    fontsize=fs - 1, fontweight="bold")

        # Custom legend: calendar (solid) + cycle (hatched)
        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, fc="grey", alpha=0.7, label="Calendar aging"),
            plt.Rectangle((0, 0), 1, 1, fc="grey", alpha=0.35, hatch="//", label="Cycle aging"),
        ]
        # Add group color patches
        for group_name in MODEL_GROUPS:
            legend_handles.append(
                plt.Rectangle((0, 0), 1, 1, fc=GROUP_COLORS[group_name], alpha=0.7,
                              label=group_name))
        ax.legend(handles=legend_handles, fontsize=fs - 1)

        ax.set_ylabel("Total capacity fade at year 20 (%)", fontsize=lfs)
        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, fontsize=fs - 1, rotation=20, ha="right")
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(labelsize=fs)
        fig.tight_layout()
        fig.savefig(os.path.join(sc_dir, "degradation_split_bar.png"), dpi=200)
        plt.close(fig)
        print(f"[{sc_label}] Saved degradation_split_bar.png")

        # 4. Economic: cumulative cost (grouped legend)
        fig, ax = plt.subplots(figsize=(10, 6))
        baseline_plotted = False
        for model_key, model_label in MODELS:
            d = data.get((sc_key, model_key))
            if d is None or d["cost"] is None:
                continue
            cost = d["cost"]
            if not baseline_plotted and "Cost_No_Sys_Cumulative" in cost.columns:
                ax.plot(cost["Year"], cost["Cost_No_Sys_Cumulative"],
                        color="grey", linewidth=2.5, label="No system (grid only)")
                baseline_plotted = True
            if "Cost_System_Cumulative" in cost.columns:
                ax.plot(cost["Year"], cost["Cost_System_Cumulative"],
                        color=COLORS[model_key], linestyle=LINESTYLES[model_key],
                        linewidth=2)
        # Build grouped legend with baseline
        handles_leg, labels_leg = [], []
        handles_leg.append(Line2D([0], [0], color="grey", linewidth=2.5))
        labels_leg.append("No system (grid only)")
        for group_name, group_models in MODEL_GROUPS.items():
            handles_leg.append(Line2D([], [], color="none"))
            labels_leg.append(f"$\\bf{{{group_name}}}$")
            for model_key, model_label in group_models:
                handles_leg.append(Line2D([0], [0], color=COLORS[model_key],
                                          linestyle=LINESTYLES[model_key], linewidth=2))
                labels_leg.append(f"  {model_label}")
        ax.legend(handles_leg, labels_leg, fontsize=fs - 1,
                  handlelength=2.5, labelspacing=0.4)
        ax.set_xlabel("Year", fontsize=lfs)
        ax.set_ylabel("Cumulative cost (\u20ac)", fontsize=lfs)
        ax.set_xlim(1, 20)
        ax.set_xticks(range(1, 21))
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=fs)
        fig.tight_layout()
        fig.savefig(os.path.join(sc_dir, "economic_cumulative_cost.png"), dpi=200)
        plt.close(fig)
        print(f"[{sc_label}] Saved economic_cumulative_cost.png")

        # 5. Economic: NPV savings (grouped legend)
        fig, ax = plt.subplots(figsize=(10, 6))
        for model_key, model_label in MODELS:
            d = data.get((sc_key, model_key))
            if d is None or d["cost"] is None:
                continue
            cost = d["cost"]
            npv_col = "Savings_Cumulative_NPV" if "Savings_Cumulative_NPV" in cost.columns else "Savings_Cumulative"
            if npv_col in cost.columns:
                ax.plot(cost["Year"], cost[npv_col],
                        color=COLORS[model_key], linestyle=LINESTYLES[model_key],
                        linewidth=2)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        _grouped_legend(ax, fs - 1)
        ax.set_xlabel("Year", fontsize=lfs)
        ax.set_ylabel("Cumulative savings NPV (\u20ac)", fontsize=lfs)
        ax.set_xlim(1, 20)
        ax.set_xticks(range(1, 21))
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=fs)
        fig.tight_layout()
        fig.savefig(os.path.join(sc_dir, "economic_savings_npv.png"), dpi=200)
        plt.close(fig)
        print(f"[{sc_label}] Saved economic_savings_npv.png")

    # ── Cross-scenario plots ──────────────────────────────────────────────
    sc_colors = {"porto_5p5kwh": "#1a5276", "berlin_5p5kwh": "#c0392b",
                  "porto_10p10kwh": "#2471a3", "berlin_10p10kwh": "#e74c3c"}
    sc_styles = {"porto_5p5kwh": "-", "berlin_5p5kwh": "-",
                  "porto_10p10kwh": "--", "berlin_10p10kwh": "--"}

    # SOH comparison: one plot per model, all scenarios
    for model_key, model_label in MODELS:
        fig, ax = plt.subplots(figsize=(10, 6))
        for sc_key, sc_info in SCENARIOS.items():
            d = data.get((sc_key, model_key))
            if d is None or d["deg"] is None:
                continue
            deg = d["deg"]
            ax.plot(_get_years(deg), deg["SOH"], color=sc_colors[sc_key],
                    linestyle=sc_styles[sc_key], linewidth=2, label=sc_info["label"])
        ax.set_xlabel("Years", fontsize=lfs)
        ax.set_ylabel("SOH (%)", fontsize=lfs)
        ax.set_xlim(0, 20)
        ax.set_xticks(range(0, 21))
        ax.legend(fontsize=fs, loc="lower left")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=fs)
        fig.tight_layout()
        fname = f"cross_scenario_soh_{model_key}.png"
        fig.savefig(os.path.join(OUTPUT_ROOT, fname), dpi=200)
        plt.close(fig)
        print(f"[Cross] Saved {fname}")

    # Cross-scenario: field-calibrated model, all scenarios, SOH + NPV
    fig, ax = plt.subplots(figsize=(10, 6))
    for sc_key, sc_info in SCENARIOS.items():
        d = data.get((sc_key, "naumann_lam_calibrated"))
        if d is None or d["deg"] is None:
            continue
        deg = d["deg"]
        ax.plot(_get_years(deg), deg["SOH"], color=sc_colors[sc_key],
                linestyle=sc_styles[sc_key], linewidth=2, label=sc_info["label"])
    ax.set_xlabel("Years", fontsize=lfs)
    ax.set_ylabel("SOH (%)", fontsize=lfs)
    ax.set_xlim(0, 20)
    ax.set_xticks(range(0, 21))
    ax.legend(fontsize=fs, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=fs)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_ROOT, "cross_scenario_soh_field_calibrated.png"), dpi=200)
    plt.close(fig)
    print("[Cross] Saved cross_scenario_soh_field_calibrated.png")

    # Cross-scenario: NPV for field-calibrated
    fig, ax = plt.subplots(figsize=(10, 6))
    for sc_key, sc_info in SCENARIOS.items():
        d = data.get((sc_key, "naumann_lam_calibrated"))
        if d is None or d["cost"] is None:
            continue
        cost = d["cost"]
        npv_col = "Savings_Cumulative_NPV" if "Savings_Cumulative_NPV" in cost.columns else "Savings_Cumulative"
        if npv_col in cost.columns:
            ax.plot(cost["Year"], cost[npv_col], color=sc_colors[sc_key],
                    linestyle=sc_styles[sc_key], linewidth=2, label=sc_info["label"])
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Year", fontsize=lfs)
    ax.set_ylabel("Cumulative savings NPV (\u20ac)", fontsize=lfs)
    ax.set_xlim(1, 20)
    ax.set_xticks(range(1, 21))
    ax.legend(fontsize=fs)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=fs)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_ROOT, "cross_scenario_npv_field_calibrated.png"), dpi=200)
    plt.close(fig)
    print("[Cross] Saved cross_scenario_npv_field_calibrated.png")

    # ── Summary tables (grouped) ─────────────────────────────────────────
    for sc_key, sc_info in SCENARIOS.items():
        sc_label = sc_info["label"]
        print(f"\n{'='*120}")
        print(f"  {sc_label}")
        print(f"{'='*120}")
        print(f"{'Group':<20s} {'Model':<35s} {'Final SOH':>10s} {'Cal fade':>10s} {'Cyc fade':>10s} "
              f"{'Repl.':>6s} {'NPV Savings':>12s} {'Breakeven':>10s}")
        print("-" * 120)
        for group_name, group_models in MODEL_GROUPS.items():
            for i, (model_key, model_label) in enumerate(group_models):
                group_str = group_name if i == 0 else ""
                d = data.get((sc_key, model_key))
                if d is None or d["deg"] is None:
                    print(f"{group_str:<20s} {model_label:<35s} {'N/A':>10s}")
                    continue
                deg = d["deg"]
                cost = d["cost"]
                final_soh = deg["SOH"].iloc[-1]
                cal_col, cyc_col = _get_deg_cols(deg)
                cal_fade = deg[cal_col].iloc[-1] * 100
                cyc_fade = deg[cyc_col].iloc[-1] * 100
                soh_series = deg["SOH"].values
                replacements = sum(1 for j in range(1, len(soh_series)) if soh_series[j] > soh_series[j-1] + 5)
                npv_savings = "N/A"
                breakeven = "N/A"
                if cost is not None:
                    npv_col = "Savings_Cumulative_NPV" if "Savings_Cumulative_NPV" in cost.columns else None
                    if npv_col:
                        npv_savings = f"\u20ac{cost[npv_col].iloc[-1]:,.0f}"
                        positive = cost[cost[npv_col] > 0]
                        if len(positive) > 0:
                            breakeven = f"Year {int(positive['Year'].iloc[0])}"
                        else:
                            breakeven = "> 20y"
                print(f"{group_str:<20s} {model_label:<35s} {final_soh:>9.1f}% {cal_fade:>9.1f}% {cyc_fade:>9.1f}% "
                      f"{replacements:>6d} {npv_savings:>12s} {breakeven:>10s}")
            print("-" * 120)
    print(f"\nAll plots saved to: {OUTPUT_ROOT}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare degradation models across scenarios")
    parser.add_argument("--scenarios", nargs="*", default=None,
                        help="Scenario keys to run (default: all). Options: " +
                             ", ".join(SCENARIOS.keys()))
    parser.add_argument("--models", nargs="*", default=None,
                        help="Model keys to run (default: all)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip simulations, just re-plot from existing CSVs")
    args = parser.parse_args()

    scenarios_to_run = args.scenarios or list(SCENARIOS.keys())
    models_to_run = args.models or [m[0] for m in MODELS]

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    if not args.plot_only:
        runs = []
        for sc_key in scenarios_to_run:
            for model_key, model_label in MODELS:
                if model_key not in models_to_run:
                    continue
                result = run_single(sc_key, model_key, model_label)
                runs.append(result)
    else:
        # Build run list from existing directories
        runs = []
        for sc_key in scenarios_to_run:
            for model_key, model_label in MODELS:
                if model_key not in models_to_run:
                    continue
                out_dir = os.path.join(OUTPUT_ROOT, sc_key, model_key)
                runs.append({
                    "scenario": sc_key,
                    "model": model_key,
                    "label": model_label,
                    "dir": out_dir,
                    "degradation_csv": os.path.join(out_dir, "degradation_data.csv"),
                    "cost_csv": os.path.join(out_dir, "cost_projection.csv"),
                })

    data = load_results(runs)
    plot_all(data)


if __name__ == "__main__":
    main()
