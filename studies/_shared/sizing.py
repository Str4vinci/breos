"""Shared setup for studies 11-16: sites, weather years, the household and the design study.

Weather: PVGIS v5.3 SARAH3 hourly 2005-2023 and the PVGIS TMY for seven European sites (the
data of long-run study 3, copied into _shared/data). Every year is restamped onto STUDY_YEAR with
BREOS's own remap_tmy_year, so all frames share one intra-year index, as weather_by_year needs.

Household: configs/optimization/projected-optimization.toml (residential_pt costs, 20-year
projection, hourly, field-calibrated v1 calendar model), 4,000 kWh/yr demandlib H0 load, south
facing at a fixed tilt. Only the location changes between sites; prices stay Portuguese so that
the sites differ by weather alone.
"""

from __future__ import annotations

import copy
import tomllib
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
REPO = HERE.parent.parent
BASE_CONFIG = REPO / "configs/optimization/projected-optimization.toml"

STUDY_YEAR = 2026
YEARS = tuple(range(2005, 2024))
ANNUAL_KWH = 4000.0
TILT = 30.0
AZIMUTH = 180.0

SITES = {
    "porto": (41.1579, -8.6291, "Europe/Lisbon"),
    "lisbon": (38.7223, -9.1393, "Europe/Lisbon"),
    "madrid": (40.4168, -3.7038, "Europe/Madrid"),
    "athens": (37.9838, 23.7275, "Europe/Athens"),
    "erlangen": (49.5897, 11.0078, "Europe/Berlin"),
    "berlin": (52.52, 13.405, "Europe/Berlin"),
    "stockholm": (59.3293, 18.0686, "Europe/Stockholm"),
}

# The enumerated design grid: module count x nominal battery kWh.
MODULES = tuple(range(2, 21))
BATTERY_KWH = tuple(float(b) for b in range(0, 16))


def merge(target: dict, overrides: dict) -> dict:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def site_config(site: str, overrides: dict | None = None) -> dict:
    with open(BASE_CONFIG, "rb") as fh:
        config = tomllib.load(fh)
    lat, lon, tz = SITES[site]
    config["location"] = {"latitude": lat, "longitude": lon, "timezone": tz, "name": site}
    return merge(config, overrides or {})


def _prepare(frame: pd.DataFrame, site: str) -> pd.DataFrame:
    from breos.app_inputs import remap_tmy_year, require_full_year_weather

    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    out = remap_tmy_year(frame, STUDY_YEAR)
    require_full_year_weather(out, STUDY_YEAR, "h", SITES[site][2])
    return out


@lru_cache(maxsize=None)
def tmy(site: str) -> pd.DataFrame:
    from breos.weather import read_weather_csv

    return _prepare(read_weather_csv(DATA / f"{site}_tmy_pvgis.csv"), site)


@lru_cache(maxsize=None)
def _series(site: str) -> pd.DataFrame:
    from breos.weather import read_weather_csv

    return read_weather_csv(DATA / f"{site}_pvgis_sarah3_2005_2023.csv")


@lru_cache(maxsize=None)
def year(site: str, y: int) -> pd.DataFrame:
    from breos.weather import WEATHER_METADATA_KEY

    s = _series(site)
    part = s[s.index.year == y].copy()
    part.attrs[WEATHER_METADATA_KEY] = s.attrs.get(WEATHER_METADATA_KEY, {})
    return _prepare(part, site)


def weather(site: str, key) -> pd.DataFrame:
    """key: "tmy" or an int year."""
    return tmy(site) if key == "tmy" else year(site, int(key))


@lru_cache(maxsize=None)
def load(site: str, annual_kwh: float = ANNUAL_KWH) -> pd.DataFrame:
    from breos.load_profiles import load_profile

    return load_profile("demandlib_h0", annual_kwh, start_date=f"{STUDY_YEAR}-01-01", freq="h", timezone=SITES[site][2])


def design_study(site: str, config: dict, tmy_frame: pd.DataFrame, weather_by_year=None, max_battery_kwh: float = 15.0):
    """A breos.optimization._DesignStudy: the code path evaluate_projected_design scores through.

    One study caches DC output per (frame, module count), so a grid scores much faster through it.
    """
    from breos.execution import require_backend
    from breos.optimization import _DesignStudy, resolve_optimization_config

    return _DesignStudy.resolve(
        tmy_frame,
        load(site),
        resolve_optimization_config(config, battery_kwh=max_battery_kwh),
        battery_kwh=max_battery_kwh,
        execution_backend=require_backend("numba"),
        weather_by_year=weather_by_year,
    )


def score(study, n_modules: int, battery_kwh: float, tables: bool = False) -> dict:
    from breos.optimization import _layout_arrays

    arrays = _layout_arrays("single", int(n_modules), TILT, AZIMUTH)
    out = study.score(arrays, n_modules=int(n_modules), battery_kwh=float(battery_kwh), return_tables=tables)
    if not tables:
        out = {k: v for k, v in out.items() if not k.startswith("_")}
    return out


def pareto_mask(obj: np.ndarray) -> np.ndarray:
    """Non-dominated rows of a minimisation matrix."""
    n = len(obj)
    keep = np.ones(n, bool)
    for i in range(n):
        if keep[i]:
            dom = np.all(obj <= obj[i], axis=1) & np.any(obj < obj[i], axis=1)
            if dom.any():
                keep[i] = False
    return keep
