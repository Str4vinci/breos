#!/usr/bin/env python3
"""Generate Article 1 orientation and weather-comparison source tables."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
from pvlib.location import Location
from scipy.optimize import differential_evolution

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import breos  # noqa: E402
from breos.pv_modules import get_module  # noqa: E402
from breos.solar import calculate_pv_production_dc  # noqa: E402
from breos.weather import preload_weather_by_year, resample_to_15min  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "validation/article1/article1-projected-optimization.toml"
MONTH_NAMES = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
HISTORICAL_COLUMN_MAP = {
    "shortwave_radiation": "ghi",
    "direct_normal_irradiance": "dni",
    "diffuse_radiation": "dhi",
    "temperature_2m": "temp_air",
    "wind_speed_10m": "wind_speed",
}


def _sha256(path: Path, *, decompress_gzip: bool = False) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if decompress_gzip else Path.open
    with opener(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> dict[str, str | bool]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=no"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"commit": revision, "tracked_worktree_dirty": bool(status)}


def _load_config(path: Path) -> tuple[dict, bytes]:
    content = path.read_bytes()
    return tomllib.loads(content.decode("utf-8")), content


def _location(config: dict) -> Location:
    location = config["location"]
    return Location(
        float(location["latitude"]),
        float(location["longitude"]),
        tz=str(location["timezone"]),
        name=str(location["name"]),
    )


def _load_tmy(
    config: dict, weather_override: Path | None = None, *, resolution: str = "h"
) -> tuple[pd.DataFrame, Path]:
    weather_path = weather_override or PROJECT_ROOT / config["simulation"]["weather_file"]
    if not weather_path.is_file():
        raise FileNotFoundError(weather_path)
    weather = pd.read_csv(weather_path, index_col=0)
    weather.index = pd.to_datetime(weather.index, utc=True)
    if resolution == "15min":
        location = config["location"]
        weather = resample_to_15min(
            weather,
            latitude=float(location["latitude"]),
            longitude=float(location["longitude"]),
            preserve_irradiance_energy=True,
        )
    return weather, weather_path


def _annual_module_production(
    weather: pd.DataFrame,
    config: dict,
    *,
    tilt: float,
    azimuth: float,
    resolution: str = "h",
) -> float:
    module = get_module(str(config["pv"]["module"]))
    dc = calculate_pv_production_dc(
        weather,
        _location(config),
        tilt=float(tilt),
        surface_azimuth=float(azimuth),
        n_modules=1,
        pv_params=module,
        freq=resolution,
        verbose=False,
    )
    hours_per_step = 0.25 if resolution == "15min" else 1.0
    return float(dc.sum() * hours_per_step / 1000.0)


def _inclusive_values(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0.0:
        raise ValueError("grid steps must be positive")
    return np.arange(start, stop + step * 0.5, step, dtype=float)


def _orientation_source(args: argparse.Namespace, config: dict) -> dict:
    weather, weather_path = _load_tmy(config, args.weather_file, resolution=args.resolution)
    module_area = float(args.module_width_m * args.module_length_m)
    rows = []
    for tilt in _inclusive_values(args.tilt_min, args.tilt_max, args.tilt_step):
        for azimuth in _inclusive_values(args.azimuth_min, args.azimuth_max, args.azimuth_step):
            production = _annual_module_production(
                weather, config, tilt=tilt, azimuth=azimuth, resolution=args.resolution
            )
            rows.append(
                {
                    "Tilt_deg": tilt,
                    "Azimuth_deg": azimuth,
                    "Annual_Module_DC_Production_kWh": production,
                    "Annual_DC_Production_kWh_m2": production / module_area,
                }
            )
    grid = pd.DataFrame(rows)
    grid_path = args.output / "orientation_grid.csv"
    grid.to_csv(grid_path, index=False)

    def objective(values: np.ndarray) -> float:
        azimuth, tilt = values
        return -_annual_module_production(
            weather, config, tilt=float(tilt), azimuth=float(azimuth), resolution=args.resolution
        )

    optimum = differential_evolution(
        objective,
        bounds=((args.azimuth_min, args.azimuth_max), (args.tilt_min, args.tilt_max)),
        seed=args.seed,
        workers=1,
        polish=True,
        updating="immediate",
    )
    optimum_payload = {
        "azimuth_deg": float(optimum.x[0]),
        "tilt_deg": float(optimum.x[1]),
        "annual_module_dc_production_kwh": float(-optimum.fun),
        "annual_dc_production_kwh_m2": float(-optimum.fun / module_area),
        "success": bool(optimum.success),
        "message": str(optimum.message),
        "evaluations": int(optimum.nfev),
    }
    optimum_path = args.output / "orientation_optimum.json"
    optimum_path.write_text(json.dumps(optimum_payload, indent=2) + "\n")
    return {
        "analysis": "orientation",
        "weather_file": str(weather_path.resolve()),
        "weather_file_sha256": _sha256(weather_path),
        "weather_uncompressed_sha256": _sha256(weather_path, decompress_gzip=weather_path.suffix == ".gz"),
        "resolution": args.resolution,
        "module_area_m2": module_area,
        "grid": {
            "tilt_min_deg": args.tilt_min,
            "tilt_max_deg": args.tilt_max,
            "tilt_step_deg": args.tilt_step,
            "azimuth_min_deg": args.azimuth_min,
            "azimuth_max_deg": args.azimuth_max,
            "azimuth_step_deg": args.azimuth_step,
            "rows": len(grid),
        },
        "orientation_grid_csv": grid_path.name,
        "orientation_grid_sha256": _sha256(grid_path),
        "orientation_optimum_json": optimum_path.name,
        "orientation_optimum_sha256": _sha256(optimum_path),
        "optimum": optimum_payload,
    }


def _historical_weather(path: Path, start_year: int, end_year: int) -> dict[int, pd.DataFrame]:
    available = preload_weather_by_year(str(path), target_year=2025)
    selected = {}
    for year, frame in available.items():
        if start_year <= year <= end_year:
            weather = frame.rename(columns=HISTORICAL_COLUMN_MAP).copy()
            weather["date"] = pd.to_datetime(weather["date"], utc=True)
            selected[year] = weather.set_index("date")
    if not selected:
        raise ValueError(f"No complete historical weather years found from {start_year} through {end_year}")
    return selected


def _monthly_weather_rows(
    weather_by_year: dict[int, pd.DataFrame], config: dict, *, tilt: float, azimuth: float
) -> pd.DataFrame:
    module = get_module(str(config["pv"]["module"]))
    rows = []
    for year, weather in sorted(weather_by_year.items()):
        dc = calculate_pv_production_dc(
            weather,
            _location(config),
            tilt=tilt,
            surface_azimuth=azimuth,
            n_modules=1,
            pv_params=module,
            freq="h",
            verbose=False,
        )
        for month in range(1, 13):
            mask = weather.index.month == month
            rows.append(
                {
                    "Year": year,
                    "Month": month,
                    "GHI_kWh_m2": float(weather.loc[mask, "ghi"].sum() / 1000.0),
                    "Temperature_C": float(weather.loc[mask, "temp_air"].mean()),
                    "PV_DC_kWh_kWp": float(dc.loc[mask].sum() / module.Mpp),
                }
            )
    return pd.DataFrame(rows)


def _weather_comparison_source(args: argparse.Namespace, config: dict) -> dict:
    tmy, tmy_path = _load_tmy(config, args.tmy_weather_file, resolution="h")
    historical_path = args.historical_weather_file.resolve()
    if not historical_path.is_file():
        raise FileNotFoundError(historical_path)
    historical = _historical_weather(historical_path, args.start_year, args.end_year)
    historical_monthly = _monthly_weather_rows(historical, config, tilt=args.tilt, azimuth=args.azimuth)
    tmy_monthly = _monthly_weather_rows({2025: tmy}, config, tilt=args.tilt, azimuth=args.azimuth)

    rows = []
    for month in range(1, 13):
        samples = historical_monthly[historical_monthly["Month"] == month]
        tmy_row = tmy_monthly[tmy_monthly["Month"] == month].iloc[0]
        row = {"Month": MONTH_NAMES[month - 1]}
        for column, label in (
            ("GHI_kWh_m2", "GHI_kWh_m2"),
            ("Temperature_C", "Temperature_C"),
            ("PV_DC_kWh_kWp", "PV_DC_kWh_kWp"),
        ):
            values = samples[column].to_numpy(dtype=float)
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1))
            half_width = float(1.96 * std / np.sqrt(len(values)))
            row.update(
                {
                    f"TMY_{label}": float(tmy_row[column]),
                    f"Historical_{label}_Mean": mean,
                    f"Historical_{label}_Std": std,
                    f"Historical_{label}_CI95_Low": mean - half_width,
                    f"Historical_{label}_CI95_High": mean + half_width,
                    f"Historical_{label}_Min": float(np.min(values)),
                    f"Historical_{label}_Max": float(np.max(values)),
                }
            )
        rows.append(row)
    comparison = pd.DataFrame(rows)
    raw_path = args.output / "weather_monthly_by_year.csv"
    comparison_path = args.output / "weather_monthly_comparison.csv"
    historical_monthly.to_csv(raw_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    return {
        "analysis": "weather-comparison",
        "historical_weather_file": str(historical_path),
        "historical_weather_sha256": _sha256(historical_path),
        "tmy_weather_file": str(tmy_path.resolve()),
        "tmy_weather_sha256": _sha256(tmy_path),
        "tmy_weather_uncompressed_sha256": _sha256(tmy_path, decompress_gzip=tmy_path.suffix == ".gz"),
        "historical_years": sorted(historical),
        "tilt_deg": args.tilt,
        "azimuth_deg": args.azimuth,
        "weather_monthly_by_year_csv": raw_path.name,
        "weather_monthly_by_year_sha256": _sha256(raw_path),
        "weather_monthly_comparison_csv": comparison_path.name,
        "weather_monthly_comparison_sha256": _sha256(comparison_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/article1-context")
    subparsers = parser.add_subparsers(dest="analysis", required=True)

    orientation = subparsers.add_parser("orientation", help="Generate Figure 3 source data")
    orientation.add_argument("--weather-file", type=Path)
    orientation.add_argument("--resolution", choices=("h", "15min"), default="h")
    orientation.add_argument("--tilt-min", type=float, default=10.0)
    orientation.add_argument("--tilt-max", type=float, default=90.0)
    orientation.add_argument("--tilt-step", type=float, default=5.0)
    orientation.add_argument("--azimuth-min", type=float, default=100.0)
    orientation.add_argument("--azimuth-max", type=float, default=260.0)
    orientation.add_argument("--azimuth-step", type=float, default=5.0)
    orientation.add_argument("--module-width-m", type=float, default=1.134)
    orientation.add_argument("--module-length-m", type=float, default=2.278)
    orientation.add_argument("--seed", type=int, default=1)

    weather = subparsers.add_parser("weather-comparison", help="Generate Figure 6 source data")
    weather.add_argument("--historical-weather-file", type=Path, required=True)
    weather.add_argument("--tmy-weather-file", type=Path)
    weather.add_argument("--start-year", type=int, default=2005)
    weather.add_argument("--end-year", type=int, default=2023)
    weather.add_argument("--tilt", type=float, default=35.0)
    weather.add_argument("--azimuth", type=float, default=180.0)
    args = parser.parse_args()

    config, config_bytes = _load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    analysis = (
        _orientation_source(args, config)
        if args.analysis == "orientation"
        else _weather_comparison_source(args, config)
    )
    report = {
        "breos_version": breos.__version__,
        "breos_source": _git_revision(),
        "python_version": platform.python_version(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "config": str(args.config),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        **analysis,
    }
    report_path = args.output / "provenance.json"
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
