"""Run a small seeded Monte Carlo study and dump every numeric output.

Used for validation steps 4 and 6: the same seeded study must produce
identical run metrics and identical yearly trajectories across the pre-refactor
baseline, the summary path, and the compiled backend. The five cases mirror the
C1-C5 shape used by the forthcoming publication study: no battery, battery
only, PV only, PV plus battery, and a larger PV-plus-battery system.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CASES = {
    # (n_modules, battery_kwh). Shaped like the forthcoming publication study
    # set: two cases with no battery, three with one. A configuration needs at
    # least one module, so
    # the battery-led case uses a minimal array rather than none.
    "C1": (8, 0.0),
    "C2": (4, 5.0),
    "C3": (8, 5.0),
    "C4": (12, 10.0),
    "C5": (12, 0.0),
}


def write_weather(path, years=(2021, 2022, 2023)):
    frames = []
    for year in years:
        idx = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
        idx = idx[~((idx.month == 2) & (idx.day == 29))]
        hour = idx.hour.to_numpy()
        doy = idx.dayofyear.to_numpy()
        daylight = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None)
        seasonal = 0.6 + 0.4 * np.cos((doy - 172) / 365.0 * 2 * np.pi)
        ghi = 750.0 * daylight * seasonal * (1.0 + 0.05 * ((year - 2021) - 1))
        frames.append(
            pd.DataFrame(
                {
                    "date": idx,
                    "temperature_2m": 14.0 + 9.0 * daylight - 6.0 * np.cos((doy - 200) / 365.0 * 2 * np.pi),
                    "wind_speed_10m": 2.0,
                    "shortwave_radiation": ghi,
                    "direct_normal_irradiance": 0.8 * ghi,
                    "diffuse_radiation": 0.2 * ghi,
                }
            )
        )
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    return path


def config(n_modules: int, battery_kwh: float) -> dict:
    return {
        "location": "porto",
        "n_modules": n_modules,
        "annual_consumption_kwh": 4000,
        "battery_kwh": battery_kwh,
        "cost_preset": "residential_pt",
        "emissions_country": "PT",
        "resolution": "h",
        "projection_years": 4,
    }


def dump(path: str, backend: str, weather: str, n_procs: int = 1) -> None:
    """Dump every numeric Monte Carlo output for one backend.

    ``n_procs`` is part of the parity claim, not just a speed knob: the pooled
    path seeds and aggregates in workers, so a study that matches serially can
    still diverge when pooled. Both are compared.
    """
    from breos.montecarlo import MonteCarloSettings, run_montecarlo

    payload: dict[str, np.ndarray] = {}
    for case, (n_modules, battery_kwh) in CASES.items():
        kwargs = dict(
            weather_file=weather,
            n_procs=n_procs,
            n_runs=6,
            years_per_run=4,
            seed=20260824,
            load_uncertainty=0.05,
            load_distribution="uniform",
            min_load_scale=0.95,
            max_load_scale=1.05,
            collect_yearly=True,
        )
        if backend != "baseline":
            kwargs["execution_backend"] = backend
        result = run_montecarlo(config(n_modules, battery_kwh), MonteCarloSettings(**kwargs))
        for col in result.runs.columns:
            values = pd.to_numeric(result.runs[col], errors="coerce").to_numpy(dtype=np.float64)
            payload[f"{case}::runs::{col}"] = values
        for col in result.yearly.columns:
            values = pd.to_numeric(result.yearly[col], errors="coerce").to_numpy(dtype=np.float64)
            if np.isnan(values).all() and result.yearly[col].notna().any():
                continue  # non-numeric diagnostic column
            payload[f"{case}::yearly::{col}"] = values
        for metric, stats in result.summary.items():
            for stat, value in stats.items():
                payload[f"{case}::summary::{metric}::{stat}"] = np.array([value], dtype=np.float64)
        print(f"  {case}: {len(result.runs)} runs x {len(result.yearly)} yearly rows", flush=True)
    np.savez(path, **payload)
    print(f"wrote {path} ({len(payload)} arrays)")


if __name__ == "__main__":
    out, backend, weather = sys.argv[1], sys.argv[2], sys.argv[3]
    procs = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    if not Path(weather).exists():
        write_weather(weather)
    dump(out, backend, weather, procs)
