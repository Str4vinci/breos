"""Shared setup for the NSGA-II vs exact-enumeration study on TOU tariffs.

Every scenario is the shipped projected-optimization config (Porto, 20-year
projection, hourly) with one tariff + dispatch strategy swapped in. The
optimizer's discrete design space (modules int, battery int kWh, tilt and
azimuth on 5 degree steps) is finite, so it is enumerated exactly.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tomllib
from functools import lru_cache
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

BREOS_ROOT = Path(__file__).resolve().parents[2]
STUDY_ROOT = Path(__file__).resolve().parent
RUNS = STUDY_ROOT / "data" / "runs"
sys.path.insert(0, str(BREOS_ROOT))

BASE_CONFIG = BREOS_ROOT / "configs/optimization/projected-optimization.toml"
WEATHER_FILE = BREOS_ROOT / "validation/data/weather/porto_tmy_2005_2023_pvgis-sarah3.csv.gz"
ANNUAL_KWH = 4000.0
STUDY_YEAR = 2026
OBJECTIVES = ["f_dependence", "f_neg_npv"]

BI = "pt_mainland_2026_daily_bi"
# Prices of configs/examples/smart-charging-portugal.toml (illustrative).
BI_PRICES = {"import_prices": {"peak": 0.2310, "off_peak": 0.1210}, "export_prices": {"all": 0.05}}
# A wide spread where grid arbitrage clearly pays (benchmark_optimization.py prices).
WIDE_PRICES = {"import_prices": {"peak": 0.50, "off_peak": 0.10}, "export_prices": {"all": 0.03}}
FIXED_TARGET = {
    "mode": "fixed_target",
    "target_usable_fraction": 0.5,
    "charge_periods": ["off_peak"],
    "discharge_periods": ["peak"],
    "grid_charge_efficiency": 0.95,
    "grid_import_limit_w": 6900,
}
DISCHARGE_ONLY = {"mode": "discharge_only", "discharge_periods": ["peak"]}
HOLD_TARGET = {
    **FIXED_TARGET,
    "discharge_periods": ["peak", "off_peak"],
    "overlap_policy": "hold_target",
}


def _tou(prices, smart=None):
    out = {"tariff": {"schedule": BI, "currency": "EUR", **prices, "fixed_charge_per_day": 0.30}}
    if smart is not None:
        out["smart_charging"] = smart
    return out


# name -> overrides. "flat" is the non-TOU control.
SCENARIOS: dict[str, dict] = {
    "flat": {},
    "bi_greedy": _tou(BI_PRICES),
    "bi_discharge_only": _tou(BI_PRICES, DISCHARGE_ONLY),
    "bi_fixed_target": _tou(BI_PRICES, FIXED_TARGET),
    "bi_hold_target": _tou(BI_PRICES, HOLD_TARGET),
    "wide_greedy": _tou(WIDE_PRICES),
    "wide_fixed_target": _tou(WIDE_PRICES, FIXED_TARGET),
}


# Design spaces: "full" searches tilt and azimuth; "az180" fixes azimuth south
# through the optimizer's own mode.fixed_azimuth.
SPACES = {"full": {}, "az180": {"fixed_azimuth": 180}}

WEATHER_DIR = BREOS_ROOT / "validation/data/weather"
# Robustness cases (added 2026-11-06). A space "az180-<case>" is the az180
# space on that case's site, load and costs; a bare space is Porto, unchanged.
# Each case: weather location key (file prefix in WEATHER_DIR), the site, the
# annual H0 load, and config overrides deep-merged into the base config.
CASES: dict[str, dict] = {
    "porto": {
        "weather": "porto", "latitude": 41.1579, "longitude": -8.6291, "timezone": "Europe/Lisbon",
        "annual_kwh": ANNUAL_KWH, "overrides": {},
    },
    "berlin": {
        "weather": "berlin", "latitude": 52.52, "longitude": 13.405, "timezone": "Europe/Berlin",
        "annual_kwh": ANNUAL_KWH, "overrides": {"location": {"name": "Berlin, Germany"}},
    },
    "colorado": {
        "weather": "golden_colorado", "latitude": 39.7555, "longitude": -105.2211, "timezone": "America/Denver",
        "annual_kwh": ANNUAL_KWH, "overrides": {"location": {"name": "Golden, Colorado"}},
    },
    "high_load": {
        "weather": "porto", "latitude": 41.1579, "longitude": -8.6291, "timezone": "Europe/Lisbon",
        "annual_kwh": 8000.0, "overrides": {},
    },
    "cheap_battery": {
        "weather": "porto", "latitude": 41.1579, "longitude": -8.6291, "timezone": "Europe/Lisbon",
        "annual_kwh": ANNUAL_KWH, "overrides": {"costs": {"storage_cost_per_kwh": 250.0}},
    },
}


def split_space(space: str) -> tuple[str, str]:
    """'az180-berlin' -> ('az180', 'berlin'); a bare space is Porto."""
    base, _, case = space.partition("-")
    return base, case or "porto"


def weather_file(space: str) -> Path:
    key = CASES[split_space(space)[1]]["weather"]
    (path,) = sorted(WEATHER_DIR.glob(f"{key}_tmy_*.csv.gz"))
    return path


def _merge(target: dict, overrides: dict) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def scenario_config(name: str, space: str = "full") -> dict:
    with open(BASE_CONFIG, "rb") as fh:
        config = tomllib.load(fh)
    overrides = copy.deepcopy(SCENARIOS[name])
    if "tariff" in overrides:
        for key in ("electricity_cost", "electricity_sold_cost", "daily_power_cost"):
            config["costs"].pop(key, None)
    config.update(overrides)
    base, case = split_space(space)
    spec = CASES[case]
    if case != "porto":
        config["location"].update(latitude=spec["latitude"], longitude=spec["longitude"], timezone=spec["timezone"])
        _merge(config, spec["overrides"])
        tariff = config.get("tariff")
        if tariff and spec["timezone"] != "Europe/Lisbon":
            # BREOS will not move a schedule to another zone. The case keeps
            # the PT daily bi-hourly windows (off-peak 22-08) in local time.
            assert tariff.pop("schedule") == BI
            tariff["custom_schedule"] = {
                "identifier": f"pt_daily_bi_windows_{case}",
                "version": "2026",
                "timezone": spec["timezone"],
                "cycle": "daily",
                "periods": ["off_peak", "peak"],
                "source": "pt_mainland_2026_daily_bi windows in local civil time (study case)",
                "rules": [{
                    "days": "all",
                    "season": "all",
                    "intervals": {"off_peak": [["00:00", "08:00"], ["22:00", "24:00"]], "peak": [["08:00", "22:00"]]},
                }],
            }
    if SPACES[base]:
        config.setdefault("mode", {}).update(SPACES[base])
    return config


@lru_cache(maxsize=None)
def inputs(space: str = "full") -> tuple[pd.DataFrame, pd.DataFrame]:
    import tools.benchmark_optimization as bench

    spec = CASES[split_space(space)[1]]
    # The benchmark builds the H0 load and checks the year in its module
    # TIMEZONE (Europe/Lisbon); a case at another site uses its own.
    bench.TIMEZONE = spec["timezone"]
    weather, load, _ = bench.prepare_case_inputs(
        weather_file(space),
        spec["weather"],
        "h",
        latitude=spec["latitude"],
        longitude=spec["longitude"],
        annual_consumption_kwh=spec["annual_kwh"],
    )
    return weather, load


def build_problem(name: str, space: str = "full", backend: str = "numba"):
    from breos.optimization import SolarDesignProblem

    weather, load = inputs(space)
    return SolarDesignProblem(weather, load, scenario_config(name, space), execution_backend=backend)


def problem_levels(problem) -> list[np.ndarray]:
    """Every value each gene can take after DiscreteGridRepair."""
    from breos.optimization import _snap_to_grid_within_bounds

    levels = []
    for col, step in enumerate(problem.gene_steps):
        lo = _snap_to_grid_within_bounds(np.array([problem.xl[col]]), step, problem.xl[col], problem.xu[col])[0]
        hi = _snap_to_grid_within_bounds(np.array([problem.xu[col]]), step, problem.xl[col], problem.xu[col])[0]
        levels.append(np.round(np.arange(lo, hi + step / 2, step), 6))
    return levels


def full_grid(problem) -> np.ndarray:
    return np.asarray(list(product(*problem_levels(problem))), dtype=float)


def evaluate(problem, x: np.ndarray) -> dict:
    out: dict = {}
    problem._evaluate(np.asarray(x, dtype=float), out)
    return out


def design_key(x) -> tuple:
    return tuple(np.round(np.asarray(x, dtype=float), 6))


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=float))


def capex_and_area(problem, modules: np.ndarray, battery_kwh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Projected_Initial_Cost and array area exactly as SolarDesignProblem computes them.

    CAPEX depends only on (modules, battery), so infeasible designs get their
    exact constraint values without running the dispatch.
    """
    from breos.economics import calculate_costs, cost_params_from_config, replacement_event_cost

    cost_params = cost_params_from_config(problem.config["costs"], problem.config["financials"])
    batt_spec = problem.config["battery"]
    capex = np.empty(len(modules))
    cache: dict = {}
    for i, (m, b) in enumerate(zip(modules, battery_kwh)):
        key = (int(round(m)), float(int(round(b))))
        if key not in cache:
            cache[key] = float(
                calculate_costs(
                    n_modules=key[0],
                    module_power_w=problem.pv_params.Mpp,
                    battery_capacity_wh=key[1] * 1000.0,
                    cost_params=cost_params,
                    replacement_cost_each=replacement_event_cost(
                        key[1], cost_params.battery_cost_per_kwh, batt_spec.get("replacement_cost")
                    ),
                )["total_initial_cost"]
            )
        capex[i] = cache[key]
    area = np.round(modules).astype(int) * problem.module_area_m2
    return capex, area
