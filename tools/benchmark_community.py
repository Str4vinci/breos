"""Offline community wall time and process peak RSS, in fresh worker processes.

Run ``python tools/benchmark_community.py --participants 10 50 --years 2``.
Participants repeat load-only, DC battery, AC battery and PV-only sites; one
shared PV asset is added. Weather is a synthetic full year at 15 minutes.
Each worker warms all site types for one project year before timing the run.
Reported peak RSS includes imports and warm-up; timing includes input loading,
PV preparation, full-horizon physics and settlement, but excludes parsing and
warm-up. All yearly settlements and pricing flows remain retained, with full
dispatch-frame retention disabled (the runner default).
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from breos.app_inputs import AppRuntimeDependencies
from breos.community.config import parse_community_config
from breos.community.runner import run_community
from breos.load_profiles import load_profile
from breos.weather import build_battery_temperature_series, resample_to_15min


def case(count: int, years: int, backend: str) -> dict:
    participants = []
    for i in range(count):
        site = {"id": f"home-{i:03d}", "annual_consumption_kwh": 4000}
        kind = i % 4
        if kind:
            site["n_modules"] = 8
        if kind in (1, 2):
            site["battery_kwh"] = 5.0
        if kind == 2:
            site.update(
                battery_coupling="ac",
                battery_inverter_ac_rating_kw=2.5,
                battery_inverter_efficiency=0.95,
                costs={"battery_inverter_cost_per_kw": 0},
            )
        participants.append(site)
    return {
        "community": {
            "location": "porto",
            "start_date": "2026-01-01",
            "resolution": "15min",
            "projection_years": years,
            "execution_backend": backend,
            "allocation": "proportional",
            "shared_energy_price": 0.1,
            "shared_network_charge": 0.02,
            "surplus_price": 0.04,
        },
        "participants": participants,
        "assets": [
            {
                "id": "shared-roof",
                "n_modules": 40,
                "ownership": "external",
                "costs": {"electricity_cost": 0.2, "daily_power_cost": 0},
            }
        ],
    }


def synthetic_weather() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", "2027-01-01", inclusive="left", freq="15min", tz="UTC")
    hours = np.arange(len(index)) / 4
    daylight = np.clip(np.sin((hours % 24 - 6) / 12 * np.pi), 0, 1)
    season = 0.6 + 0.4 * np.sin((hours / 24 - 80) / 365 * 2 * np.pi)
    ghi = daylight * season * 800
    frame = pd.DataFrame(
        {"ghi": ghi, "dni": ghi * 0.7, "dhi": ghi * 0.3, "temp_air": 20.0, "wind_speed": 3.0}, index=index
    )
    frame.attrs["breos_weather_metadata"] = {
        "source": "synthetic_community_benchmark",
        "radiation_time_basis": "instant",
        "timestamp_label_basis": "instant",
        "irradiance_time_offset_hours": 0.0,
    }
    return frame


def worker(count: int, years: int, backend: str) -> dict:
    weather = synthetic_weather()
    deps = AppRuntimeDependencies(
        load_profile=load_profile,
        load_weather=lambda **_kwargs: None,
        fetch_tmy_weather_data=lambda **_kwargs: (weather.copy(deep=True), {}),
        resample_to_15min=resample_to_15min,
        build_battery_temperature_series=build_battery_temperature_series,
    )
    warm = run_community(parse_community_config(case(4, 1, backend)), deps=deps)
    del warm
    gc.collect()
    parsed = parse_community_config(case(count, years, backend))
    started = time.perf_counter()
    run = run_community(parsed, deps=deps)
    elapsed = time.perf_counter() - started
    return {
        "participants": count,
        "assets": 1,
        "project_years": years,
        "steps_per_year": len(weather),
        "backend": backend,
        "seconds": round(elapsed, 3),
        "peak_rss_mib": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 if sys.platform != "darwin" else 1024**2), 1
        ),
        "settlement_rows": sum(len(frame) for frame in run.settlements.values()),
        "retained_settlement_mib": round(
            sum(frame.memory_usage(deep=True).sum() for frame in run.settlements.values()) / 1024**2, 1
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participants", nargs="+", type=int, default=[10, 50])
    parser.add_argument("--years", type=int, default=2)
    parser.add_argument("--backend", choices=["python", "numba"], default="numba")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.participants) < 1 or args.years < 1:
        parser.error("participants and years must be positive")
    if args.worker:
        print(json.dumps(worker(args.participants[0], args.years, args.backend)))
        return
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "platform": platform.platform(),
            }
        ),
        flush=True,
    )
    for count in args.participants:
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--participants",
                str(count),
                "--years",
                str(args.years),
                "--backend",
                args.backend,
            ],
            cwd=PROJECT_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        print(result.stdout.strip(), flush=True)


if __name__ == "__main__":
    main()
