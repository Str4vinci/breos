#!/usr/bin/env python3
"""
Compare Polysun (Miner/Wöhler) vs PVBAT (Naumann) degradation methodologies.

Runs PVBAT multi-year simulations to get SOC profiles and degradation curves,
then feeds the same SOC profiles into Polysun's Miner-rule degradation model.
Compares predicted lifetimes, SOH trajectories, and economic impact.

Scenarios:
  - Porto 5kWp/5kWh  (residential, warm climate)
  - Berlin 5kWp/5kWh (residential, cold climate)

Wöhler sensitivity: conservative / typical / optimistic LFP parameters.

Outputs to results/polysun_comparison/
"""

import copy
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from breos.constants import (
    WOEHLER_LFP_CONSERVATIVE_A, WOEHLER_LFP_CONSERVATIVE_B,
    WOEHLER_LFP_TYPICAL_A, WOEHLER_LFP_TYPICAL_B,
    WOEHLER_LFP_OPTIMISTIC_A, WOEHLER_LFP_OPTIMISTIC_B,
    POLYSUN_CALENDAR_LIFE_LION,
    DEFAULT_MAX_SOC, DEFAULT_MIN_SOC, DEFAULT_CHARGE_EFFICIENCY,
)
from breos.polysun_degradation import (
    PolysunDegradationConfig,
    simulate_polysun_degradation,
)

# ── Scenarios ─────────────────────────────────────────────────────────────────
SCENARIOS = {
    "porto_5p5kwh": {
        "label": "Porto 5kWp/5kWh",
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
        "label": "Berlin 5kWp/5kWh",
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
    "indoor_model": {
        "enabled": True,
        "setpoint_c": 22.0,
        "coupling_alpha": 0.3,
        "floor_c": 15.0,
        "ceiling_c": 35.0,
    },
}

WOEHLER_SETS = {
    "conservative": {
        "label": "Wöhler conservative",
        "a": WOEHLER_LFP_CONSERVATIVE_A,
        "b": WOEHLER_LFP_CONSERVATIVE_B,
    },
    "typical": {
        "label": "Wöhler typical",
        "a": WOEHLER_LFP_TYPICAL_A,
        "b": WOEHLER_LFP_TYPICAL_B,
    },
    "optimistic": {
        "label": "Wöhler optimistic",
        "a": WOEHLER_LFP_OPTIMISTIC_A,
        "b": WOEHLER_LFP_OPTIMISTIC_B,
    },
}

OUTPUT_ROOT = str(PROJECT_ROOT / "results" / "polysun_comparison")


def resolve_config(raw_config: dict) -> dict:
    from run_simulation import resolve_extends
    config_dir = PROJECT_ROOT / "configs"
    return resolve_extends(raw_config, config_dir)


def build_config(scenario_key: str) -> dict:
    sc = SCENARIOS[scenario_key]
    cfg = {
        "name": f"Polysun comparison – {sc['label']}",
        "simulation_type": "multiyear",
        "extends": copy.deepcopy(sc["extends"]),
        "pv": copy.deepcopy(sc["pv"]),
        "battery": {**BATTERY_DEFAULTS, **sc["battery"]},
        "inverter": {"dc_ac_ratio": 1.25, "efficiency": 0.96},
        "load": copy.deepcopy(sc["load"]),
        "simulation": copy.deepcopy(sc["simulation"]),
        "output": {
            "folder": os.path.join(OUTPUT_ROOT, scenario_key, "pvbat"),
            "plots": False,
            "csv": True,
            "breakeven": False,
        },
    }
    return cfg


def run_pvbat_simulation(scenario_key: str) -> dict:
    """Run PVBAT multi-year simulation. Returns paths to result files."""
    from run_simulation import run_multiyear

    cfg = build_config(scenario_key)
    os.makedirs(cfg["output"]["folder"], exist_ok=True)
    cfg = resolve_config(cfg)

    sc_label = SCENARIOS[scenario_key]["label"]
    print(f"\n{'='*60}")
    print(f"  PVBAT: {sc_label}")
    print(f"{'='*60}")
    run_multiyear(cfg)

    out_dir = cfg["output"]["folder"]
    return {
        "results_csv": os.path.join(out_dir, "hourly_results.csv"),
        "degradation_csv": os.path.join(out_dir, "degradation_data.csv"),
        "dir": out_dir,
    }


def extract_year1_soc(results_csv: str) -> np.ndarray:
    """Extract first year of SOC data from PVBAT results."""
    df = pd.read_csv(results_csv)
    soc = df["Battery_SOC_Normalized"].values
    # First year = first 8760 hours (or 8784 for leap year)
    year1_len = min(8784, len(soc))
    return soc[:year1_len]


def find_pvbat_eol_year(degradation_csv: str, eol_soh: float = 80.0) -> float:
    """Find year when PVBAT SOH first drops below EOL threshold."""
    df = pd.read_csv(degradation_csv, parse_dates=["Datetime"])
    soh = df["SOH"].values
    dt = df["Datetime"]
    years = (dt - dt.iloc[0]).dt.total_seconds() / (365.25 * 86400)

    below_eol = np.where(soh <= eol_soh)[0]
    if len(below_eol) > 0:
        return float(years.iloc[below_eol[0]])
    return float(years.iloc[-1])  # Never reached EOL


def get_pvbat_yearly_soh(degradation_csv: str, n_years: int = 20) -> pd.DataFrame:
    """Extract yearly SOH values from PVBAT degradation data."""
    df = pd.read_csv(degradation_csv, parse_dates=["Datetime"])
    dt = df["Datetime"]
    years_frac = (dt - dt.iloc[0]).dt.total_seconds() / (365.25 * 86400)

    rows = []
    for year in range(1, n_years + 1):
        # Find closest point to each year boundary
        idx = (years_frac - year).abs().idxmin()
        rows.append({
            "Year": year,
            "SOH": df.loc[idx, "SOH"],
        })
    return pd.DataFrame(rows)


def get_mean_temperature(scenario_key: str) -> float:
    """Get annual mean ambient temperature for a scenario's location."""
    sc = SCENARIOS[scenario_key]
    weather_file = sc["simulation"].get("weather_file", "")
    if os.path.exists(weather_file):
        wdf = pd.read_csv(weather_file, nrows=8760)
        for col in ["temperature_2m", "temp_air", "Temperature"]:
            if col in wdf.columns:
                return float(wdf[col].mean())
    # Fallback rough estimates
    if "porto" in scenario_key:
        return 15.5
    elif "berlin" in scenario_key:
        return 10.0
    return 15.0


def run_comparison():
    from breos.plotting import (
        plot_degradation_methodology_comparison,
        plot_lifetime_prediction_comparison,
        plot_temperature_sensitivity_comparison,
    )

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    n_years = 20

    # ── Step 1: Run PVBAT simulations ─────────────────────────────────────
    pvbat_runs = {}
    for sc_key in SCENARIOS:
        out_dir = os.path.join(OUTPUT_ROOT, sc_key, "pvbat")
        results_csv = os.path.join(out_dir, "hourly_results.csv")
        deg_csv = os.path.join(out_dir, "degradation_data.csv")

        # Skip if already run
        if os.path.exists(results_csv) and os.path.exists(deg_csv):
            print(f"\n  [SKIP] PVBAT results exist for {SCENARIOS[sc_key]['label']}")
            pvbat_runs[sc_key] = {
                "results_csv": results_csv,
                "degradation_csv": deg_csv,
                "dir": out_dir,
            }
        else:
            pvbat_runs[sc_key] = run_pvbat_simulation(sc_key)

    # ── Step 2: Run Polysun degradation on year-1 SOC ─────────────────────
    polysun_results = {}  # (scenario, woehler_set) -> DataFrame

    for sc_key in SCENARIOS:
        soc_year1 = extract_year1_soc(pvbat_runs[sc_key]["results_csv"])
        sc_dir = os.path.join(OUTPUT_ROOT, sc_key)

        for ws_key, ws_params in WOEHLER_SETS.items():
            cfg = PolysunDegradationConfig(
                woehler_a=ws_params["a"],
                woehler_b=ws_params["b"],
                calendar_life_years=POLYSUN_CALENDAR_LIFE_LION,
            )
            polysun_df = simulate_polysun_degradation(soc_year1, cfg, n_years=n_years)
            polysun_results[(sc_key, ws_key)] = polysun_df

            # Save CSV
            ws_dir = os.path.join(sc_dir, f"polysun_{ws_key}")
            os.makedirs(ws_dir, exist_ok=True)
            polysun_df.to_csv(os.path.join(ws_dir, "polysun_degradation.csv"), index=False)

            total_life = polysun_df.iloc[0]["Total_Life_Years"]
            annual_damage = polysun_df.iloc[0]["Damage_Annual"]
            print(f"  [{SCENARIOS[sc_key]['label']}] {ws_params['label']}: "
                  f"D_annual={annual_damage:.4f}, "
                  f"cycle_life={polysun_df.iloc[0]['Cycle_Life_Years']:.1f}y, "
                  f"total_life={total_life:.1f}y")

    # ── Step 3: Generate comparison plots ─────────────────────────────────
    print("\n" + "="*60)
    print("  Generating comparison plots")
    print("="*60)

    for sc_key in SCENARIOS:
        sc_label = SCENARIOS[sc_key]["label"]
        sc_dir = os.path.join(OUTPUT_ROOT, sc_key)
        pvbat_soh_df = get_pvbat_yearly_soh(pvbat_runs[sc_key]["degradation_csv"], n_years)

        # Plot SOH comparison for each Wöhler parameter set
        for ws_key, ws_params in WOEHLER_SETS.items():
            polysun_df = polysun_results[(sc_key, ws_key)]
            plot_degradation_methodology_comparison(
                pvbat_soh=pvbat_soh_df,
                polysun_df=polysun_df,
                results_directory=sc_dir,
                scenario_label=f"{sc_label} ({ws_params['label']})",
                suffix=f"_{ws_key}",
            )
            print(f"  [{sc_label}] Saved SOH comparison ({ws_key})")

    # ── Step 4: Lifetime comparison bar chart ─────────────────────────────
    # Use typical Wöhler for the summary comparison
    lifetime_scenarios = {}
    for sc_key in SCENARIOS:
        sc_label = SCENARIOS[sc_key]["label"]
        pvbat_eol = find_pvbat_eol_year(pvbat_runs[sc_key]["degradation_csv"], eol_soh=80.0)
        polysun_df = polysun_results[(sc_key, "typical")]
        lifetime_scenarios[sc_label] = {
            "pvbat_eol_year": pvbat_eol,
            "polysun_total_life": polysun_df.iloc[0]["Total_Life_Years"],
            "polysun_cycle_life": polysun_df.iloc[0]["Cycle_Life_Years"],
            "polysun_calendar_life": polysun_df.iloc[0]["Calendar_Life_Years"],
        }

    plot_lifetime_prediction_comparison(
        lifetime_scenarios, OUTPUT_ROOT, suffix="_typical"
    )
    print("  Saved lifetime prediction comparison")

    # ── Step 5: Temperature sensitivity comparison ────────────────────────
    temp_locations = {}
    for sc_key in SCENARIOS:
        sc_label = SCENARIOS[sc_key]["label"]
        pvbat_eol = find_pvbat_eol_year(pvbat_runs[sc_key]["degradation_csv"], eol_soh=80.0)
        polysun_df = polysun_results[(sc_key, "typical")]
        mean_temp = get_mean_temperature(sc_key)
        temp_locations[sc_label] = {
            "pvbat_eol_year": pvbat_eol,
            "polysun_total_life": polysun_df.iloc[0]["Total_Life_Years"],
            "mean_temp_c": mean_temp,
        }

    plot_temperature_sensitivity_comparison(
        temp_locations, OUTPUT_ROOT
    )
    print("  Saved temperature sensitivity comparison")

    # ── Step 6: Summary table ─────────────────────────────────────────────
    print("\n" + "="*100)
    print("  SUMMARY: Polysun vs PVBAT Degradation Predictions")
    print("="*100)
    print(f"{'Scenario':<25s} {'Wöhler set':<20s} {'PVBAT EOL (y)':>15s} "
          f"{'Polysun life (y)':>18s} {'Cycle life (y)':>15s} {'Cal life (y)':>13s}")
    print("-"*100)

    for sc_key in SCENARIOS:
        sc_label = SCENARIOS[sc_key]["label"]
        pvbat_eol = find_pvbat_eol_year(pvbat_runs[sc_key]["degradation_csv"], eol_soh=80.0)
        for ws_key, ws_params in WOEHLER_SETS.items():
            polysun_df = polysun_results[(sc_key, ws_key)]
            row = polysun_df.iloc[0]
            print(f"{sc_label:<25s} {ws_params['label']:<20s} {pvbat_eol:>15.1f} "
                  f"{row['Total_Life_Years']:>18.1f} {row['Cycle_Life_Years']:>15.1f} "
                  f"{row['Calendar_Life_Years']:>13.0f}")
        print("-"*100)

    # Wöhler sensitivity: show how different parameters affect cycle counts
    print("\n  Wöhler Curve Reference:")
    for ws_key, ws_params in WOEHLER_SETS.items():
        a, b = ws_params["a"], ws_params["b"]
        n100 = a
        n80 = a * 0.8**(-b)
        n50 = a * 0.5**(-b)
        n20 = a * 0.2**(-b)
        print(f"    {ws_params['label']:<25s} N(100%)={n100:.0f}  N(80%)={n80:.0f}  "
              f"N(50%)={n50:.0f}  N(20%)={n20:.0f}")

    print(f"\nAll results saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Compare Polysun vs PVBAT battery degradation methodologies"
    )
    parser.add_argument("--scenarios", nargs="*", default=None,
                        help=f"Scenarios to run (default: all). Options: {', '.join(SCENARIOS.keys())}")
    parser.add_argument("--skip-simulation", action="store_true",
                        help="Skip PVBAT simulation, reuse existing results")
    args = parser.parse_args()

    if args.scenarios:
        SCENARIOS = {k: v for k, v in SCENARIOS.items() if k in args.scenarios}

    run_comparison()
