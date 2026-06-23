"""
Batch recalculate economics for all results directories with updated energy prices.

Reads existing yearly_summary.csv (or hourly_results.csv as fallback) to preserve
the simulated energy balance, then re-runs cost_analysis_projection() with the
updated prices from configs/base/costs.json.

Usage:
    uv run python tools/recalculate_economics.py [--dry-run] [--joao-only]
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from breos.economics import cost_analysis_projection

RESULTS_DIR = ROOT / "results"
COSTS_FILE = ROOT / "configs" / "base" / "costs.json"

with open(COSTS_FILE) as f:
    _COSTS_CFG = json.load(f)

NEW_PRICES = {
    "pt": {
        "electricity_cost": _COSTS_CFG["residential_pt"]["electricity_cost"],
        "electricity_sold_cost": _COSTS_CFG["residential_pt"]["electricity_sold_cost"],
    },
    "es": {
        "electricity_cost": _COSTS_CFG["residential_es"]["electricity_cost"],
        "electricity_sold_cost": _COSTS_CFG["residential_es"]["electricity_sold_cost"],
    },
    "de": {
        "electricity_cost": _COSTS_CFG["residential_de"]["electricity_cost"],
        "electricity_sold_cost": _COSTS_CFG["residential_de"]["electricity_sold_cost"],
    },
}

# Directories to skip (no standard cost_projection.csv or different structure)
_SKIP_PREFIXES = (
    "_test",
    "validation",
    "weather_comparison",
    "model_comparison",
    "degradation_comparison",
    "degradation_model_comparison",
    "batch_comparison",
    "calendar_sensitivity",
    "ea_sensitivity",
    "montecarlo",
    "webapp_test",
    "tariff_sensitivity",
    "optimization_run",
    "azitilt_optimization",
)
_SKIP_EXACT = {"comparison", "joao_tables", "joao_pv_battery_optimization", "reference_case_spain"}


def detect_country(results_dir: Path) -> str:
    export_file = results_dir / "export.txt"
    if export_file.exists():
        text = export_file.read_text(errors="ignore").lower()
        if any(k in text for k in ("germany", "berlin", "münchen")):
            return "de"
        if any(k in text for k in ("spain", "madrid", "seville", "sevilla")):
            return "es"
    name = results_dir.name.lower()
    if any(k in name for k in ("berlin", "_de", "germany")):
        return "de"
    if any(k in name for k in ("spain", "_es", "madrid")):
        return "es"
    return "pt"


def extract_params(proj_df: pd.DataFrame) -> dict:
    """Extract cost and projection parameters from an existing cost_projection.csv."""
    r1 = proj_df[proj_df["Year"] == 1].iloc[0]

    total_initial = float(r1["Cost_System_Cumulative"] - r1["Cost_System_Annual"])
    operation = float(r1.get("Cost_Operation", 0.0))
    daily = float(r1.get("Cost_Daily", 109.5)) / 365.0
    num_years = int(proj_df["Year"].max())

    inflation = 0.02
    discount = 0.0
    if len(proj_df) > 1:
        r2 = proj_df[proj_df["Year"] == 2].iloc[0]
        if r1["Cost_No_Sys_Annual"] > 0:
            inflation = round(float(r2["Cost_No_Sys_Annual"] / r1["Cost_No_Sys_Annual"]) - 1.0, 4)
        if "Cost_No_Sys_Annual_NPV" in proj_df.columns and r2["Cost_No_Sys_Annual"] > 0:
            npv_ratio = float(r2["Cost_No_Sys_Annual_NPV"] / r2["Cost_No_Sys_Annual"])
            if abs(npv_ratio - 1.0) > 0.0001:
                discount = round(1.0 / npv_ratio - 1.0, 4)

    return {
        "total_initial_cost": total_initial,
        "annual_operation_cost": operation,
        "daily_power_cost": daily,
        "num_years": num_years,
        "inflation_rate": inflation,
        "discount_rate": discount,
    }


def detect_freq(hourly_file: Path) -> str:
    df = pd.read_csv(hourly_file, nrows=3)
    if "Datetime" in df.columns and len(df) >= 2:
        dates = pd.to_datetime(df["Datetime"].iloc[:2], dayfirst=True)
        delta_min = (dates.iloc[1] - dates.iloc[0]).total_seconds() / 60
        if delta_min <= 16:
            return "15min"
    return "h"


def recalculate_dir(results_dir: Path, dry_run: bool = False) -> str:
    """Recalculate economics for one results directory. Returns a status string."""
    proj_file = results_dir / "cost_projection.csv"
    yearly_file = results_dir / "yearly_summary.csv"
    hourly_file = results_dir / "hourly_results.csv"

    if not proj_file.exists():
        return "SKIP (no cost_projection.csv)"
    if not hourly_file.exists():
        return "SKIP (no hourly_results.csv)"

    try:
        proj_df = pd.read_csv(proj_file)
        params = extract_params(proj_df)
    except Exception as exc:
        return f"ERROR extracting params: {exc}"

    country = detect_country(results_dir)
    prices = NEW_PRICES[country]

    costs = {
        "electricity_cost": prices["electricity_cost"],
        "electricity_sold_cost": prices["electricity_sold_cost"],
        "total_initial_cost": params["total_initial_cost"],
        "annual_operation_cost": params["annual_operation_cost"],
        "daily_power_cost": params["daily_power_cost"],
    }

    summary = (
        f"[{country.upper()}] {params['num_years']}yr  "
        f"buy={prices['electricity_cost']:.4f}  "
        f"sell={prices['electricity_sold_cost']:.4f}  "
        f"initial={params['total_initial_cost']:.0f}€"
    )

    if dry_run:
        mode = "propagation" if yearly_file.exists() else "legacy"
        return f"DRY-RUN ({mode}) {summary}"

    try:
        if yearly_file.exists():
            yearly_df = pd.read_csv(yearly_file)
            cost_analysis_projection(
                results_df=pd.DataFrame(),
                costs=costs,
                num_years=params["num_years"],
                inflation_rate=params["inflation_rate"],
                discount_rate=params["discount_rate"],
                results_directory=str(results_dir),
                yearly_summary_df=yearly_df,
            )
        else:
            freq = detect_freq(hourly_file)
            results_df = pd.read_csv(hourly_file)
            cost_analysis_projection(
                results_df=results_df,
                costs=costs,
                num_years=params["num_years"],
                inflation_rate=params["inflation_rate"],
                discount_rate=params["discount_rate"],
                results_directory=str(results_dir),
                freq=freq,
            )
        return f"OK {summary}"
    except Exception as exc:
        return f"ERROR: {exc}"


def should_skip(d: Path) -> bool:
    name = d.name
    return name in _SKIP_EXACT or any(name.startswith(p) for p in _SKIP_PREFIXES) or not d.is_dir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without writing files")
    parser.add_argument("--joao-only", action="store_true", help="Only process joao_* directories")
    args = parser.parse_args()

    all_dirs = sorted(RESULTS_DIR.iterdir())
    joao_dirs = [d for d in all_dirs if d.is_dir() and d.name.startswith("joao_")]
    other_dirs = [d for d in all_dirs if d.is_dir() and not d.name.startswith("joao_")]

    dirs = joao_dirs if args.joao_only else joao_dirs + other_dirs
    dirs = [d for d in dirs if not should_skip(d)]

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Processing {len(dirs)} directories...\n")

    counts = {"ok": 0, "skip": 0, "error": 0}
    for d in dirs:
        status = recalculate_dir(d, dry_run=args.dry_run)
        icon = (
            "✓" if status.startswith("OK") or status.startswith("DRY") else ("·" if status.startswith("SKIP") else "✗")
        )
        print(f"  {icon} {d.name}: {status}")
        key = "ok" if status.startswith(("OK", "DRY")) else ("skip" if status.startswith("SKIP") else "error")
        counts[key] += 1

    label = "[DRY RUN] " if args.dry_run else ""
    print(f"\n{label}Done: {counts['ok']} updated, {counts['skip']} skipped, {counts['error']} errors")


if __name__ == "__main__":
    main()
