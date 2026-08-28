#!/usr/bin/env python3
"""Compare mean and instant radiation inputs for the forthcoming publication."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shlex
import subprocess
import sys
import tomllib
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from pvlib.location import Location

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import breos  # noqa: E402
from breos.pv.model_options import (  # noqa: E402
    configured_pv_model_kwargs,
    resolve_configured_pv_model_options,
)
from breos.pv_modules import get_module  # noqa: E402
from breos.solar import calculate_pv_production_dc  # noqa: E402
from breos.weather import (  # noqa: E402
    preload_weather_by_year,
    weather_file_metadata,
    weather_metadata,
    weather_representative_time_offset,
)

DEFAULT_CONFIG = PROJECT_ROOT / "validation/article1/article1-projected-optimization.toml"
COLUMN_MAP = {
    "shortwave_radiation": "ghi",
    "direct_normal_irradiance": "dni",
    "diffuse_radiation": "dhi",
    "temperature_2m": "temp_air",
    "wind_speed_10m": "wind_speed",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolved_config_sha256(config: dict) -> str:
    rendered = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _location(config: dict) -> Location:
    location = config["location"]
    return Location(
        float(location["latitude"]),
        float(location["longitude"]),
        tz=str(location["timezone"]),
        name=str(location["name"]),
    )


def _weather_years(path: Path, start_year: int, end_year: int) -> dict[int, pd.DataFrame]:
    years = preload_weather_by_year(str(path.resolve()), target_year=2025)
    selected = {}
    for year, frame in sorted(years.items()):
        if start_year <= year <= end_year:
            metadata = weather_metadata(frame)
            weather = frame.rename(columns=COLUMN_MAP).copy()
            weather["date"] = pd.to_datetime(weather["date"], utc=True)
            weather = weather.set_index("date")
            weather.attrs["breos_weather_metadata"] = metadata
            selected[year] = weather
    expected = set(range(start_year, end_year + 1))
    if set(selected) != expected:
        raise ValueError(f"Expected complete weather years {start_year}-{end_year}, got {sorted(selected)}")
    return selected


def _annual_row(
    weather: pd.DataFrame, config: dict, *, scenario: str, year: int, tilt: float, azimuth: float
) -> dict[str, float | int | str]:
    module = get_module(str(config["pv"]["module"]))
    dc = calculate_pv_production_dc(
        weather,
        _location(config),
        tilt=tilt,
        surface_azimuth=azimuth,
        n_modules=1,
        pv_params=module,
        freq="h",
        verbose=False,
        **configured_pv_model_kwargs(config),
    )
    return {
        "scenario": scenario,
        "weather_year": year,
        "ghi_kwh_m2": float(weather["ghi"].sum() / 1000.0),
        "dni_kwh_m2": float(weather["dni"].sum() / 1000.0),
        "dhi_kwh_m2": float(weather["dhi"].sum() / 1000.0),
        "fixed_design_dc_kwh_per_module": float(dc.sum() / 1000.0),
    }


def _git_revision() -> dict[str, str | bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=no"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"commit": commit, "tracked_worktree_dirty": bool(status)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--interval-mean-weather", type=Path, required=True)
    parser.add_argument("--instant-weather", type=Path, required=True)
    parser.add_argument("--start-year", type=int, default=2005)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--tilt", type=float, default=35.0)
    parser.add_argument("--azimuth", type=float, default=180.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty result directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    config = tomllib.loads(args.config.read_text())
    sources = {
        "preceding_hour_mean_right_labeled": args.interval_mean_weather,
        "instant_at_label": args.instant_weather,
    }
    rows = []
    runtime_weather = {}
    for scenario, path in sources.items():
        years = _weather_years(path, args.start_year, args.end_year)
        representative = next(iter(years.values()))
        runtime_weather[scenario] = {
            "representative_source_year": int(next(iter(years))),
            "solar_position_offset_minutes": weather_representative_time_offset(representative, "h").total_seconds()
            / 60.0,
            "metadata": weather_metadata(representative),
        }
        for year, weather in years.items():
            rows.append(
                _annual_row(
                    weather,
                    config,
                    scenario=scenario,
                    year=year,
                    tilt=args.tilt,
                    azimuth=args.azimuth,
                )
            )

    annual = pd.DataFrame(rows)
    annual_path = args.output / "annual_radiation_and_fixed_pv.csv"
    summary_path = args.output / "summary.json"
    provenance_path = args.output / "provenance.json"
    annual.to_csv(annual_path, index=False)
    summary = annual.groupby("scenario").agg(
        {
            "ghi_kwh_m2": ["mean", "std", "min", "max"],
            "dni_kwh_m2": ["mean", "std", "min", "max"],
            "dhi_kwh_m2": ["mean", "std", "min", "max"],
            "fixed_design_dc_kwh_per_module": ["mean", "std", "min", "max"],
        }
    )
    summary.columns = ["_".join(column) for column in summary.columns]
    summary_path.write_text(json.dumps(summary.reset_index().to_dict(orient="records"), indent=2, default=str) + "\n")

    module = get_module(str(config["pv"]["module"]))
    provenance = {
        "breos_version": breos.__version__,
        "breos_source": _git_revision(),
        "python_version": platform.python_version(),
        "dependency_versions": {
            package: importlib.metadata.version(package) for package in ("numpy", "pandas", "pvlib", "scipy")
        },
        "command": shlex.join([sys.executable, *sys.argv]),
        "resolved_config": config,
        "resolved_config_sha256": _resolved_config_sha256(config),
        "effective_runtime_pv_model_options": resolve_configured_pv_model_options(
            config, bifaciality=module.bifaciality
        ),
        "resolved_pv_module": asdict(module),
        "tilt_deg": args.tilt,
        "azimuth_deg": args.azimuth,
        "input_resolution": "h",
        "output_resolution": "h",
        "irradiance_resampling_method": "none",
        "weather_sources": {
            scenario: {
                "path": str(path.resolve()),
                "sha256": _sha256(path.resolve()),
                "metadata": weather_file_metadata(path.resolve()),
            }
            for scenario, path in sources.items()
        },
        "effective_runtime_weather": runtime_weather,
        "annual_csv": annual_path.name,
        "annual_csv_sha256": _sha256(annual_path),
        "summary_json": summary_path.name,
        "summary_json_sha256": _sha256(summary_path),
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, default=str) + "\n")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
