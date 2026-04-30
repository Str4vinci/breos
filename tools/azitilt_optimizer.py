#!/usr/bin/env python3
"""
Azimuth and Tilt Optimizer
===========================
Optimizes PV system orientation (Azimuth, Tilt) for maximum production or grid independence.

Usage:
    python tools/azitilt_optimizer.py configs/scenarios/azitilt_porto.json
"""

import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from pvlib.location import Location

from breos import load_profile
from breos.battery import BatteryConfig, simulate_energy_balance
from breos.plotting import (
    plot_azitilt_ew_1d,
    plot_azitilt_landscape_2d,
    plot_azitilt_landscape_3d,
)
from breos.pv_modules import get_module
from breos.solar import calculate_pv_production_dc, default_azimuth
from breos.weather import fetch_tmy_weather_data

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================


def load_json(filepath: Path) -> Dict[str, Any]:
    """Load a JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def resolve_extends(config: Dict[str, Any], config_dir: Path) -> Dict[str, Any]:
    """
    Resolve 'extends' references to base configs.
    Format: "base/locations.json#porto" -> loads locations.json, extracts 'porto' key
    """
    if "extends" not in config:
        return config

    extends = config.pop("extends")
    resolved = {}

    for key, ref in extends.items():
        # Parse reference: "base/locations.json#porto"
        if "#" in ref:
            filepath, subkey = ref.split("#", 1)
        else:
            filepath = ref
            subkey = None

        # Resolve relative to configs/ directory
        # If config_dir is inside configs/ (e.g. configs/scenarios), we need to check properly
        # We assume ref is relative to 'configs/' root if it starts with 'base/'?
        # Let's try resolving relative to the config file first.

        full_path = config_dir / filepath

        # If not found, try resolving relative to PROJECT_ROOT/configs
        if not full_path.exists():
            full_path = PROJECT_ROOT / "configs" / filepath

        if not full_path.exists():
            raise FileNotFoundError(f"Base config not found: {full_path}")

        base_data = load_json(full_path)

        if subkey:
            if subkey not in base_data:
                raise KeyError(f"Key '{subkey}' not found in {filepath}")
            resolved[key] = base_data[subkey]
        else:
            resolved[key] = base_data

    # Merge resolved configs with main config (main config overrides resolved)
    return {**resolved, **config}


# ==========================================
# 2. OBJECTIVE FUNCTION
# ==========================================


class OptimizerContext:
    """Holds shared data for the objective function to avoid pickling large dataframes repeatedly."""

    def __init__(
        self,
        weather_data: pd.DataFrame,
        location: Location,
        pv_params: Any,
        n_modules: int,
        load_data: Optional[pd.DataFrame],
        battery_config: Optional[BatteryConfig],
        objective_type: str,
        mode: str = "standard",
    ):
        self.weather_data = weather_data
        self.location = location
        self.pv_params = pv_params
        self.n_modules = n_modules
        self.load_data = load_data
        self.battery_config = battery_config
        self.objective_type = objective_type
        self.mode = mode

        # Cache for plotting
        self.history = []

    def evaluate(self, x):
        """
        Objective function for optimization.
        Standard Mode: x[0] = Azimuth, x[1] = Tilt
        East-West Mode: x[0] = Tilt (Azimuths fixed at 90 and 270)
        """
        if self.mode == "east_west":
            tilt = x[0]
            # Split modules: half East (90), half West (270)
            n_east = self.n_modules // 2
            n_west = self.n_modules - n_east

            # West (270)
            dc_west = calculate_pv_production_dc(
                weather_data=self.weather_data,
                location=self.location,
                tilt=tilt,
                surface_azimuth=270,
                n_modules=n_west,
                pv_params=self.pv_params,
                freq="h",
                verbose=False,
            )

            # East (90)
            dc_east = calculate_pv_production_dc(
                weather_data=self.weather_data,
                location=self.location,
                tilt=tilt,
                surface_azimuth=90,
                n_modules=n_east,
                pv_params=self.pv_params,
                freq="h",
                verbose=False,
            )

            dc_power = dc_west + dc_east
            azimuth_repr = "East-West"

        else:
            azimuth = x[0]
            tilt = x[1]
            dc_power = calculate_pv_production_dc(
                weather_data=self.weather_data,
                location=self.location,
                tilt=tilt,
                surface_azimuth=azimuth,
                n_modules=self.n_modules,
                pv_params=self.pv_params,
                freq="h",
                verbose=False,
            )
            azimuth_repr = azimuth

        try:
            total_pv_kwh = dc_power.sum() / 1000.0

            # 2. Return Metric based on objective
            if self.objective_type == "max_production":
                metric = total_pv_kwh
                # We want to MAXIMIZE, so return NEGATIVE for minimizer
                result = -metric

            elif self.objective_type == "max_grid_independence":
                if self.load_data is None:
                    raise ValueError("Load profile required for grid independence optimization")

                # Check battery
                final_batt_config = self.battery_config
                if final_batt_config is None:
                    # Default: No battery, just direct self-consumption
                    final_batt_config = BatteryConfig(nominal_energy_wh=0)

                # Run energy balance
                results_df, _, summary_df, _, _, _ = simulate_energy_balance(
                    pv_dc=dc_power, houseload=self.load_data, battery_config=final_batt_config, freq="h", debug=False
                )

                # Extract Independence
                # summary_df cols: "Import [kWh]", "Sell [kWh]", etc.
                # Grid Independence = (1 - Import/Load) * 100

                # Alternatively, get it from summary df if it exists
                if "Grid Independence [%]" in summary_df.columns:
                    independence = summary_df["Grid Independence [%]"].iloc[0]
                else:
                    # Recalc manually if needed
                    total_load = results_df["Houseload"].sum()
                    total_import = results_df["Import_From_Grid"].sum()
                    if total_load > 0:
                        independence = (1 - total_import / total_load) * 100
                    else:
                        independence = 100.0

                metric = independence
                result = -metric

            else:
                raise ValueError(f"Unknown objective: {self.objective_type}")

            return result

        except Exception as e:
            print(f"Error evaluating ({azimuth_repr}, {tilt:.2f}): {e}")
            return 999999  # Huge penalty


# ==========================================
# 3. MAIN SCRIPT
# ==========================================


def main():
    parser = argparse.ArgumentParser(description="Optimize Azimuth and Tilt")
    parser.add_argument("config", type=Path, help="Path to JSON configuration file")
    args = parser.parse_args()

    # 1. Load Config
    print(f"Loading config from {args.config}...")
    try:
        raw_config = load_json(args.config)
        config = resolve_extends(raw_config, args.config.parent)
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    # 2. Extract settings
    out_cfg = config.get("output", {})
    results_dir = Path(out_cfg.get("folder", "results/azitilt"))
    results_dir.mkdir(parents=True, exist_ok=True)

    loc_cfg = config["location"]
    pv_cfg = config["pv"]
    opt_cfg = config.get("optimization", {})

    # 3. Setup Location & Weather
    location = Location(
        latitude=loc_cfg["latitude"],
        longitude=loc_cfg["longitude"],
        tz=loc_cfg.get("timezone", "UTC"),
        name=loc_cfg.get("name", "Unknown"),
    )
    print(f"Location: {location.name} ({location.latitude}, {location.longitude})")

    print("Fetching weather data...")
    tmy_data, meta = fetch_tmy_weather_data(latitude=location.latitude, longitude=location.longitude, freq="h")

    # 4. Setup PV
    module_name = pv_cfg.get("module", "Suntech_STP550S_STC")
    n_modules = pv_cfg.get("n_modules", 10)
    pv_params = get_module(module_name)
    print(f"PV: {n_modules} x {module_name}")

    # 5. Setup Load & Battery (if needed)
    objective = opt_cfg.get("objective", "max_production")

    load_data = None
    battery_config = None

    if objective == "max_grid_independence":
        print("Objective: Maximize Grid Independence (Loading demand profile...)")
        # Load profile
        load_cfg = config.get("load", {})
        # Defaults if missing?
        if not load_cfg:
            # Try to infer default load
            print("Warning: No load config found for grid independence. Using default 4000kWh/yr.")
            load_data = load_profile(profile_type="crest", annual_consumption_kwh=4000, freq="h")
        else:
            load_data = load_profile(
                profile_type=load_cfg.get("source", "crest"),
                annual_consumption_kwh=load_cfg.get("annual_consumption_kwh", 4000),
                freq="h",
            )

        # Battery
        batt_cfg = config.get("battery", {})
        if batt_cfg:
            battery_config = BatteryConfig(
                nominal_energy_wh=batt_cfg.get("nominal_kwh", 5) * 1000,
                max_soc=batt_cfg.get("max_soc", 0.8),
                min_soc=batt_cfg.get("min_soc", 0.2),
                charge_efficiency=batt_cfg.get("charge_efficiency", 0.95),
                discharge_efficiency=batt_cfg.get("discharge_efficiency", 0.95),
                dc_coupled=batt_cfg.get("dc_coupled", True),
            )
            print(f"Battery: {batt_cfg.get('nominal_kwh')} kWh")
    else:
        print("Objective: Maximize PV Production")

    # 6. Optimization Loop
    modes_to_run = ["standard", "east_west"]
    all_results = {}

    elapsed_total = 0

    for mode in modes_to_run:
        print("\n" + "=" * 60)
        print(f"RUNNING OPTIMIZATION: {mode.upper()}")
        print("=" * 60)

        # Setup Bounds per mode
        bounds_cfg = opt_cfg.get("bounds", {})
        tilt_bounds = tuple(bounds_cfg.get("tilt", [10, 90]))

        if mode == "east_west":
            # One variable: Tilt
            bounds = [tilt_bounds]
            print(f"Bounds: Tilt {tilt_bounds} (Fixed Azimuths: 90 / 270)")
        else:
            # Two variables: Azimuth, Tilt
            default_azi_bounds = [90, 270] if location.latitude >= 0 else [-90, 90]
            azi_bounds = tuple(bounds_cfg.get("azimuth", default_azi_bounds))
            bounds = [azi_bounds, tilt_bounds]
            print(f"Bounds: Azimuth {azi_bounds}, Tilt {tilt_bounds}")

        # Initialize Context
        ctx = OptimizerContext(
            weather_data=tmy_data,
            location=location,
            pv_params=pv_params,
            n_modules=n_modules,
            load_data=load_data,
            battery_config=battery_config,
            objective_type=objective,
            mode=mode,
        )

        # Run Optimization
        print("Starting optimization (Differential Evolution)...")
        start_time = time.time()

        # Use 'workers=-1' to use all available cores
        result = differential_evolution(
            ctx.evaluate,
            bounds,
            strategy="best1bin",
            maxiter=opt_cfg.get("max_iter", 20),
            popsize=opt_cfg.get("pop_size", 10),
            tol=0.01,
            workers=-1,  # PARALLEL EXECUTION
            disp=True,
            polish=True,
        )

        elapsed = time.time() - start_time
        elapsed_total += elapsed
        print(f"Optimization complete in {elapsed:.2f}s")

        # Extract Results
        res_data = {
            "mode": mode,
            "elapsed": elapsed,
            "raw_metric": -result.fun,
            "nfev": result.nfev,
            "message": result.message,
        }

        if mode == "east_west":
            res_data["raw_tilt"] = result.x[0]
            res_data["raw_azimuth"] = "East-West"

            res_data["opt_tilt"] = round(res_data["raw_tilt"])
            res_data["discrete_tilt"] = round(res_data["opt_tilt"] / 5) * 5

            # Eval Rounded
            res_data["opt_metric"] = -ctx.evaluate((res_data["opt_tilt"],))
            res_data["discrete_metric"] = -ctx.evaluate((res_data["discrete_tilt"],))

            res_data["opt_azimuth"] = "East-West"
            res_data["discrete_azimuth"] = "East-West"

        else:
            res_data["raw_azimuth"] = result.x[0]
            res_data["raw_tilt"] = result.x[1]

            res_data["opt_azimuth"] = round(res_data["raw_azimuth"])
            res_data["opt_tilt"] = round(res_data["raw_tilt"])
            res_data["opt_metric"] = -ctx.evaluate((res_data["opt_azimuth"], res_data["opt_tilt"]))

            res_data["discrete_azimuth"] = round(res_data["opt_azimuth"] / 5) * 5
            res_data["discrete_tilt"] = round(res_data["opt_tilt"] / 5) * 5
            res_data["discrete_metric"] = -ctx.evaluate((res_data["discrete_azimuth"], res_data["discrete_tilt"]))

        all_results[mode] = res_data

        # Visualization per mode
        if mode == "standard":
            print("Generating surface plot (Grid Evaluation)...")

            # Grid resolution
            n_azi = 20
            n_tilt = 20

            azi_vals = np.linspace(azi_bounds[0], azi_bounds[1], n_azi)
            tilt_vals = np.linspace(tilt_bounds[0], tilt_bounds[1], n_tilt)

            grid_points = []
            for a in azi_vals:
                for s in tilt_vals:
                    grid_points.append((a, s))

            with concurrent.futures.ProcessPoolExecutor() as executor:
                # ctx.evaluate returns negative metric (minimization)
                neg_metrics = list(executor.map(ctx.evaluate, grid_points))

            grid_results = []
            for pt, neg_m in zip(grid_points, neg_metrics):
                grid_results.append((pt[0], pt[1], -neg_m))  # Convert back to positive

            # Convert to DataFrame for plotting
            df_grid = pd.DataFrame(grid_results, columns=["Azimuth", "Tilt", "Metric"])
            df_grid.to_csv(results_dir / "grid_search_standard.csv", index=False)

            try:
                plot_azitilt_landscape_2d(
                    df_grid,
                    res_data["opt_azimuth"],
                    res_data["opt_tilt"],
                    str(results_dir),
                    "optimization_landscape_2d.png",
                )
                plot_azitilt_landscape_3d(
                    df_grid,
                    res_data["opt_azimuth"],
                    res_data["opt_tilt"],
                    res_data["opt_metric"],
                    str(results_dir),
                    "optimization_landscape_3d.png",
                )
                print("Generated standard plots.")
            except Exception as e:
                print(f"Error plotting standard: {e}")

        elif mode == "east_west":
            print("Generating 1D performance plot...")
            tilt_vals = np.linspace(tilt_bounds[0], tilt_bounds[1], 20)
            points = [(s,) for s in tilt_vals]

            with concurrent.futures.ProcessPoolExecutor() as executor:
                neg_metrics = list(executor.map(ctx.evaluate, points))

            metrics = [-m for m in neg_metrics]

            try:
                plot_azitilt_ew_1d(
                    tilt_vals,
                    metrics,
                    res_data["opt_tilt"],
                    res_data["opt_metric"],
                    str(results_dir),
                    "optimization_1d_tilt_ew.png",
                )
                print("Generated East-West plot.")
            except Exception as e:
                print(f"Error plotting EW: {e}")

    # 7. Final Report
    print("\n" + "=" * 80)
    print(f"FINAL REPORT: {objective}")
    print("=" * 80)

    # Header
    print(f"{'Mode':<15} | {'Type':<10} | {'Azimuth':<15} | {'Tilt':<10} | {'Metric':<10} | {'Diff':<10}")
    print("-" * 80)

    for mode in modes_to_run:
        res = all_results[mode]
        # Precise
        print(
            f"{mode:<15} | {'Raw':<10} | {res['raw_azimuth']:<15} | {res['raw_tilt']:<10.2f} | {res['raw_metric']:<10.2f} | -"
        )
        # Integer
        diff_int = res["opt_metric"] - res["raw_metric"]
        print(
            f"{'':<15} | {'1 deg':<10} | {res['opt_azimuth']:<15} | {res['opt_tilt']:<10.0f} | {res['opt_metric']:<10.2f} | {diff_int:<10.2f}"
        )
        # Discrete
        diff_dis = res["discrete_metric"] - res["raw_metric"]
        print(
            f"{'':<15} | {'5 deg':<10} | {res['discrete_azimuth']:<15} | {res['discrete_tilt']:<10.0f} | {res['discrete_metric']:<10.2f} | {diff_dis:<10.2f}"
        )
        print("-" * 80)

    # Save Report
    results_txt = results_dir / "optimization_results.txt"
    with open(results_txt, "w") as f:
        f.write(f"Objective: {objective}\n")
        f.write(f"Total Time: {elapsed_total:.2f}s\n\n")

        for mode in modes_to_run:
            res = all_results[mode]
            f.write(f"MODE: {mode.upper()}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Raw Tilt: {res['raw_tilt']:.4f}\n")
            if mode == "standard":
                f.write(f"Raw Azimuth: {res['raw_azimuth']:.4f}\n")
            f.write(f"Raw Metric: {res['raw_metric']:.4f}\n")
            f.write(f"Optimal (1deg): {res['opt_metric']:.4f}\n")
            f.write(f"Discrete (5deg): {res['discrete_metric']:.4f}\n")
            f.write("\n")

    print(f"Saved full report to {results_txt}")


if __name__ == "__main__":
    main()
