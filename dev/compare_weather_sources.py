#!/usr/bin/env python3
"""
Weather Source Comparison Tool
================================
Compare TMY vs Historical weather data for a location, producing monthly
analysis with PV generation estimates, statistical summaries, and plots.

Usage:
    python tools/compare_weather_sources.py --location porto --slope 35 --azimuth 180
    python tools/compare_weather_sources.py --location porto  # uses defaults
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from pvlib.location import Location
from breos.weather import parse_weather_filename
from breos.solar import calculate_pv_production_dc, PVModuleParams
from breos.pv_modules import get_module
from breos.plotting import (
    set_presentation_mode,
    plot_weather_monthly_comparison,
    plot_weather_annual_ghi_distribution,
)

WEATHER_DIR = PROJECT_ROOT / "weather"
LOCATIONS_PATH = PROJECT_ROOT / "configs" / "base" / "locations.json"


def discover_weather_files(location: str) -> dict:
    """Find all weather files for a given location, grouped by type."""
    files = {"tmy": [], "historical": []}

    for fname in sorted(os.listdir(WEATHER_DIR)):
        parsed = parse_weather_filename(fname)
        if parsed is None or parsed["location"] != location:
            continue
        parsed["filepath"] = str(WEATHER_DIR / fname)
        files[parsed["type"]].append(parsed)

    return files


def load_location(location: str) -> dict:
    """Load location info from locations.json."""
    with open(LOCATIONS_PATH) as f:
        locations = json.load(f)
    if location not in locations:
        print(f"ERROR: '{location}' not found in {LOCATIONS_PATH}")
        print(f"Available: {', '.join(locations.keys())}")
        sys.exit(1)
    return locations[location]


def _load_weather_csv(filepath: str) -> pd.DataFrame:
    """Load a weather CSV and return DataFrame with DatetimeIndex.

    Handles three formats produced by this codebase:
    - TMY (naive):    unnamed first col is naive datetime string
    - TMY (tz-aware): unnamed first col has mixed tz offsets (e.g., +01:00/+02:00)
    - Historical:     unnamed integer index col, then named 'date' col
    """
    df = pd.read_csv(filepath, index_col=0)

    # Try parsing the first column as datetime (TMY case)
    try:
        # utc=True handles mixed-offset tz strings (DST transitions)
        idx = pd.to_datetime(df.index, utc=True)
        if idx.year.min() > 1971:   # sanity check — not epoch noise
            df.index = idx
            return df
    except Exception:
        pass

    # Historical case: unnamed integer index, datetime in a named column
    df = pd.read_csv(filepath)
    for col_name in ["date", "time", "Datetime"]:
        if col_name in df.columns:
            df[col_name] = pd.to_datetime(df[col_name], utc=True)
            df.set_index(col_name, inplace=True)
            return df

    raise ValueError(f"Could not parse datetime index from {filepath}")


load_tmy = _load_weather_csv
load_historical = _load_weather_csv


def compute_pv_for_year(
    weather_df: pd.DataFrame, location: Location,
    slope: float, azimuth: float, pv_params: PVModuleParams
) -> pd.Series:
    """Run PV DC calculation for a single year of weather data."""
    # Ensure timezone
    if weather_df.index.tz is None:
        weather_df = weather_df.copy()
        weather_df.index = weather_df.index.tz_localize("UTC")

    dc_power = calculate_pv_production_dc(
        weather_data=weather_df,
        location=location,
        slope=slope,
        surface_azimuth=azimuth,
        n_modules=1,
        pv_params=pv_params,
        freq="h",
    )
    # Convert W to kWh (hourly data → each step = 1 hour)
    return dc_power / 1000.0


def monthly_stats(yearly_monthly: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    """Compute mean, std, CI, min, max per month from yearly data."""
    z = 1.96  # ~95% CI
    n = yearly_monthly.count()
    mean = yearly_monthly.mean()
    std = yearly_monthly.std()
    ci = z * std / np.sqrt(n)
    return pd.DataFrame({
        "mean": mean,
        "std": std,
        "ci_low": mean - ci,
        "ci_high": mean + ci,
        "min": yearly_monthly.min(),
        "max": yearly_monthly.max(),
        "n": n,
    })


def main():
    parser = argparse.ArgumentParser(description="Compare TMY vs Historical weather sources")
    parser.add_argument("--location", required=True, help="Location key (e.g., 'porto')")
    parser.add_argument("--slope", type=float, default=35, help="PV panel tilt (degrees)")
    parser.add_argument("--azimuth", type=float, default=180, help="PV panel azimuth (degrees)")
    parser.add_argument("--module", type=str, default="Suntech_STP550S_STC", help="PV module name")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")

    set_presentation_mode(True)

    loc_info = load_location(args.location)
    location = Location(
        latitude=loc_info["latitude"],
        longitude=loc_info["longitude"],
        tz=loc_info.get("timezone", "UTC"),
    )
    pv_params = get_module(args.module)

    # Discover files
    files = discover_weather_files(args.location)
    if not files["tmy"]:
        print(f"ERROR: No TMY files found for '{args.location}' in {WEATHER_DIR}")
        sys.exit(1)
    if not files["historical"]:
        print(f"ERROR: No historical files found for '{args.location}' in {WEATHER_DIR}")
        sys.exit(1)

    print(f"Location: {args.location} ({loc_info['latitude']}, {loc_info['longitude']})")
    print(f"PV: slope={args.slope} deg, azimuth={args.azimuth} deg, module={pv_params.Mpp}W")
    print(f"TMY files: {len(files['tmy'])}")
    print(f"Historical files: {len(files['historical'])}")

    # Output directory
    results_dir = PROJECT_ROOT / "results" / f"weather_comparison_{args.location}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Process TMY ---
    tmy_info = files["tmy"][0]
    print(f"\nProcessing TMY: {os.path.basename(tmy_info['filepath'])}")
    tmy_df = load_tmy(tmy_info["filepath"])

    # GHI column name detection
    ghi_col = "ghi" if "ghi" in tmy_df.columns else "shortwave_radiation"
    
    # Use groupby to handle potential timezone shifts pushing data to previous year
    tmy_monthly_ghi = tmy_df[ghi_col].groupby(tmy_df.index.month).sum() / 1000.0  # kWh/m2

    tmy_pv = compute_pv_for_year(tmy_df, location, args.slope, args.azimuth, pv_params)
    tmy_monthly_pv = tmy_pv.groupby(tmy_pv.index.month).sum()

    # --- Process Historical (year by year) ---
    hist_info = files["historical"][0]
    print(f"Processing Historical: {os.path.basename(hist_info['filepath'])}")
    hist_df = load_historical(hist_info["filepath"])

    hist_ghi_col = "ghi" if "ghi" in hist_df.columns else "shortwave_radiation"
    years = sorted(hist_df.index.year.unique())
    print(f"Historical years: {years[0]}-{years[-1]} ({len(years)} years)")

    yearly_ghi = {}  # year -> 12 monthly values
    yearly_pv = {}

    for year in years:
        year_data = hist_df[hist_df.index.year == year].copy()
        # Skip incomplete years
        if len(year_data) < 8000:
            print(f"   Skipping {year}: only {len(year_data)} hours")
            continue

        # Monthly GHI
        monthly_ghi = year_data[hist_ghi_col].groupby(year_data.index.month).sum() / 1000.0
        yearly_ghi[year] = monthly_ghi.values

        # Monthly PV
        pv_series = compute_pv_for_year(year_data, location, args.slope, args.azimuth, pv_params)
        monthly_pv = pv_series.groupby(pv_series.index.month).sum()
        yearly_pv[year] = monthly_pv.values

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Build DataFrames: rows=years, columns=months
    ghi_matrix = pd.DataFrame(yearly_ghi, index=month_labels).T
    pv_matrix = pd.DataFrame(yearly_pv, index=month_labels).T

    ghi_stats = monthly_stats(ghi_matrix)
    pv_stats = monthly_stats(pv_matrix)

    # TMY monthly values aligned to month labels
    tmy_ghi_vals = tmy_monthly_ghi.values[:12]
    tmy_pv_vals = tmy_monthly_pv.values[:12]

    # --- Annual totals ---
    tmy_annual_ghi = tmy_ghi_vals.sum()
    tmy_annual_pv = tmy_pv_vals.sum()
    hist_annual_ghi_mean = ghi_stats["mean"].sum()
    hist_annual_ghi_std = np.sqrt((ghi_stats["std"] ** 2).sum())
    hist_annual_pv_mean = pv_stats["mean"].sum()
    hist_annual_pv_std = np.sqrt((pv_stats["std"] ** 2).sum())

    # Annual per-year totals for min/max
    annual_ghi_per_year = ghi_matrix.sum(axis=1)
    annual_pv_per_year = pv_matrix.sum(axis=1)

    ghi_diff_pct = (tmy_annual_ghi - hist_annual_ghi_mean) / hist_annual_ghi_mean * 100
    pv_diff_pct = (tmy_annual_pv - hist_annual_pv_mean) / hist_annual_pv_mean * 100

    # --- Console output ---
    print(f"\n{'='*60}")
    print(f"  Annual Summary: {args.location}")
    print(f"{'='*60}")
    print(f"                    {'TMY':>10}  {'Hist Mean':>10}  {'Hist Std':>10}  {'Diff %':>8}")
    print(f"  GHI (kWh/m2)     {tmy_annual_ghi:10.1f}  {hist_annual_ghi_mean:10.1f}  {hist_annual_ghi_std:10.1f}  {ghi_diff_pct:+8.1f}%")
    print(f"  PV  (kWh/kWp)    {tmy_annual_pv:10.1f}  {hist_annual_pv_mean:10.1f}  {hist_annual_pv_std:10.1f}  {pv_diff_pct:+8.1f}%")
    print(f"  Hist min year    {annual_ghi_per_year.idxmin()} (GHI: {annual_ghi_per_year.min():.0f})  "
          f"{annual_pv_per_year.idxmin()} (PV: {annual_pv_per_year.min():.0f})")
    print(f"  Hist max year    {annual_ghi_per_year.idxmax()} (GHI: {annual_ghi_per_year.max():.0f})  "
          f"{annual_pv_per_year.idxmax()} (PV: {annual_pv_per_year.max():.0f})")

    # --- Save CSV ---
    summary_df = pd.DataFrame({
        "Month": month_labels,
        "TMY_GHI_kWh_m2": tmy_ghi_vals,
        "Hist_GHI_mean": ghi_stats["mean"].values,
        "Hist_GHI_std": ghi_stats["std"].values,
        "Hist_GHI_min": ghi_stats["min"].values,
        "Hist_GHI_max": ghi_stats["max"].values,
        "TMY_PV_kWh_kWp": tmy_pv_vals,
        "Hist_PV_mean": pv_stats["mean"].values,
        "Hist_PV_std": pv_stats["std"].values,
        "Hist_PV_min": pv_stats["min"].values,
        "Hist_PV_max": pv_stats["max"].values,
    })
    csv_path = results_dir / "monthly_comparison.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved monthly statistics to {csv_path}")

    # --- Plots (via pvbat.plotting) ---
    tmy_source = tmy_info.get("source", "TMY")
    results_dir_str = str(results_dir)

    plot_weather_monthly_comparison(
        tmy_ghi_vals, ghi_stats, "GHI (kWh/m²)", tmy_source,
        results_dir_str, "monthly_ghi_comparison.png",
    )
    print("Saved GHI comparison plot")

    plot_weather_monthly_comparison(
        tmy_pv_vals, pv_stats, "PV Generation (kWh/kWp)", tmy_source,
        results_dir_str, "monthly_pv_comparison.png",
    )
    print("Saved PV generation comparison plot")

    plot_weather_annual_ghi_distribution(
        annual_ghi_per_year, tmy_annual_ghi, hist_annual_ghi_mean,
        results_dir_str, "annual_ghi_distribution.png",
    )
    print("Saved annual GHI distribution plot")

    print(f"\nAll outputs saved to {results_dir}")


if __name__ == "__main__":
    main()
