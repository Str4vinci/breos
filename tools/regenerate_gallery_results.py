#!/usr/bin/env python3
"""Run the example-gallery cases and store their results for the docs.

The pages in ``docs/examples/`` are reports of stored runs: the docs build
loads what this tool writes under ``docs/examples/_results/<case>/`` and draws
the figures, without simulating or fetching weather. Run it once per release,
after the version bump, and commit the result directories::

    uv run python tools/regenerate_gallery_results.py            # every case
    uv run python tools/regenerate_gallery_results.py first_home  # some cases
    uv run python tools/regenerate_gallery_results.py --check    # rerun the cheap cases and compare
    uv run python tools/regenerate_gallery_results.py --list

Each case directory holds its result files (JSON and small CSVs) and a
``manifest.json`` recording the BREOS version and ``git describe`` that made
them, the UTC date, the runtime, every config file with its SHA-256, the
weather files with their SHA-256 and attribution, and the dependency
versions. Absolute local paths are replaced by repository-relative ones.

Weather is the PVGIS TMYs committed under ``validation/data/weather``. The
Monte Carlo case needs a multi-year history: it fetches Open-Meteo data
(CC BY 4.0) at run time, or reads ``--mc-weather PATH``, and records the
file's SHA-256 in the manifest. The history is not committed.

``--check`` reruns the cases marked cheap into a temporary directory and
compares every number with the stored one at a relative and absolute
tolerance of 1e-9. It exits 1 on a difference and changes nothing. That
holds on the machine that stored the results. BREOS does not promise bit
identity across machines: on other hardware a rounded value can move by one
unit in the last decimal place it is written with. ``--cross-machine``
accepts that and nothing more. A field's precision is the most decimal
places written for it in either file, so 0.01 for a value rounded to cents
and 0.0001 for a state of charge; a sum of rounded values may move by one
unit per term. Integers, counts, years, text and booleans compare exactly.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as dt
import decimal
import functools
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
import warnings
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "docs" / "examples" / "_results"
WEATHER_DIR = REPO / "validation" / "data" / "weather"
TOLERANCE = 1e-9

PVGIS_ATTRIBUTION = (
    "PVGIS typical meteorological year 2005-2023 (PVGIS-SARAH3). PVGIS data © European Union, 2001-2024, "
    "reused under Commission Decision 2011/833/EU."
)
OPEN_METEO_ATTRIBUTION = "Weather data by Open-Meteo.com, licensed CC BY 4.0."
TMY_FILES = {
    "porto": "porto_tmy_2005_2023_pvgis-sarah3.csv.gz",
    "berlin": "berlin_tmy_2005_2023_pvgis-sarah3.csv.gz",
}
MC_HISTORY = {"location": "porto", "start": "2005-01-01", "end": "2024-12-31"}
DEPENDENCIES = ("numpy", "pandas", "pvlib", "scipy", "numba", "pymoo")

# The first-year step columns a stored week keeps, in W (power) or as a fraction.
WEEK_COLUMNS = (
    "PV_Production",
    "Houseload",
    "PV_AC_To_Load",
    "Battery_AC_To_Load",
    "Import_From_Grid",
    "PV_AC_Export",
    "Grid_AC_To_Battery",
    "Battery_Energy",
    "Battery_SOC_Normalized",
    "PV_DC",
    "PV_DC_Curtailed",
)
# Keys whose values describe the machine, not the run; --check ignores them.
VOLATILE_KEYS = frozenset({"breos_version", "python", "numpy", "pandas", "pvlib", "scipy", "numba", "pymoo"})


# --------------------------------------------------------------------------- helpers


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO).as_posix()


@functools.cache
def git_describe() -> str:
    """The checkout's ``git describe``, read once before any result file is written."""
    out = subprocess.run(
        ["git", "-C", str(REPO), "describe", "--tags", "--always", "--dirty"],
        capture_output=True,
        text=True,
        check=False,
    )
    return out.stdout.strip() or "unknown"


def dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for name in DEPENDENCIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def load_toml(relative: str) -> dict[str, Any]:
    with open(REPO / relative, "rb") as handle:
        return tomllib.load(handle)


def with_overrides(config: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """``config`` with top-level ``overrides`` applied; a value of None removes the key."""
    merged = copy.deepcopy(dict(config))
    for key, value in overrides.items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def first_year_frame(app: Any) -> pd.DataFrame:
    """The first simulated year, one row per step, from the public accessor."""
    return app.timeseries()


def week_slice(frame: pd.DataFrame, timezone: str, iso_week: int) -> pd.DataFrame:
    """One ISO week of ``frame`` on the local clock, with UTC timestamps, rounded for storage."""
    instants = pd.DatetimeIndex(pd.to_datetime(frame["Datetime"], utc=True))
    local = instants.tz_convert(timezone)
    mask = np.asarray(local.isocalendar().week == iso_week)
    columns = [column for column in WEEK_COLUMNS if column in frame.columns]
    week = frame.loc[mask, columns].astype(float).reset_index(drop=True)
    week = week.round({column: 4 if column == "Battery_SOC_Normalized" else 1 for column in columns})
    week.insert(0, "Datetime", [stamp.isoformat() for stamp in instants[mask]])
    return week


def year1_bill(result: Mapping[str, Any]) -> float:
    """What the household pays in year 1 with the system: import, fixed charge, minus export."""
    return round(
        result["grid_import_cost_year1_prices"]
        + result["fixed_charge_year1_prices"]
        - result["grid_export_revenue_year1_prices"],
        2,
    )


def no_system_bill(result: Mapping[str, Any]) -> float:
    return round(result["no_system_import_cost_year1_prices"] + result["no_system_fixed_charge_year1_prices"], 2)


SCALAR_KEYS = (
    "pv_kwp",
    "battery_kwh",
    "pv_dc_generation_kwh",
    "usable_ac_system_production_kwh",
    "curtailment_dc_kwh",
    "consumption_kwh",
    "self_consumption_kwh",
    "grid_import_kwh",
    "grid_export_kwh",
    "grid_independence_pct",
    "self_consumption_pct",
    "total_investment",
    "payback_year",
    "npv_savings",
    "lcoe_per_kwh",
    "battery_soh_end_pct",
    "battery_replacements",
    "battery_replacement_cost_npv",
    "terminal_health_credit",
    "terminal_health_credit_npv",
    "npv_savings_terminal_adjusted",
    "co2_avoided_total_year1_kg",
    "co2_avoided_total_lifetime_kg",
    "grid_import_cost_year1_prices",
    "grid_export_revenue_year1_prices",
    "fixed_charge_year1_prices",
    "no_system_import_cost_year1_prices",
    "no_system_fixed_charge_year1_prices",
    "grid_charge_cost_year1_prices",
)


def scalars(result: Mapping[str, Any]) -> dict[str, Any]:
    """The headline numbers of ``result`` a comparison table needs."""
    row = {key: result.get(key) for key in SCALAR_KEYS if key in result}
    if "grid_import_cost_year1_prices" in result:
        row["bill_year1"] = year1_bill(result)
        row["no_system_bill_year1"] = no_system_bill(result)
    return row


def financial_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The discounted cumulative ledger a break-even plot reads, year 0 first."""
    keys = ("year", "balance", "cost_with_system", "cost_without_system")
    return [{key: row.get(key) for key in keys} for row in result["financial"]]


def replacement_times(result: Mapping[str, Any]) -> list[float]:
    """Project times (years) of the battery swaps, from the financial ledger."""
    return [row["replacement_time_years"] for row in result.get("financial") or [] if row.get("replacement_time_years")]


def sanitise(value: Any, roots: Mapping[str, str]) -> Any:
    """``value`` with every string under one of ``roots`` (absolute path -> relative) made relative."""
    if isinstance(value, dict):
        return {key: sanitise(item, roots) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitise(item, roots) for item in value]
    if isinstance(value, str):
        for root, replacement in roots.items():
            if value == root:
                return replacement or "."
            if value.startswith(root + os.sep) or value.startswith(root + "/"):
                rest = Path(value[len(root) + 1 :]).as_posix()
                return f"{replacement}/{rest}" if replacement else rest
    return value


def json_ready(value: Any) -> Any:
    """``value`` with numpy scalars and non-finite floats made JSON-safe."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


# --------------------------------------------------------------------------- context


@dataclass
class Output:
    """What a case stores: file name -> JSON-ready object or DataFrame, plus manifest fields."""

    files: dict[str, Any] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)


class Context:
    """Runs App configs from a scratch directory that holds the committed TMYs."""

    def __init__(self, work: Path, options: argparse.Namespace) -> None:
        self.work = work
        self.options = options
        self.configs: dict[str, str] = {}
        self.weather: dict[str, dict[str, str]] = {}
        self.currencies: set[str] = set()
        self.timezones: set[str] = set()
        (work / "weather").mkdir(parents=True, exist_ok=True)

    def config(self, relative: str) -> dict[str, Any]:
        """Load a committed TOML and record it in the manifest."""
        self.configs[relative] = sha256(REPO / relative)
        return load_toml(relative)

    def stage_tmy(self, location: str) -> Path:
        """Copy the committed PVGIS TMY for ``location`` (and its sidecar) into ``weather/``, where App looks."""
        source = WEATHER_DIR / TMY_FILES[location]
        target = self.work / "weather" / source.name
        if not target.exists():
            shutil.copyfile(source, target)
            sidecar = source.with_name(source.name + ".metadata.json")
            if sidecar.exists():
                shutil.copyfile(sidecar, target.with_name(sidecar.name))
        self.weather[repo_relative(source)] = {"sha256": sha256(source), "attribution": PVGIS_ATTRIBUTION}
        return target

    @contextlib.contextmanager
    def inside(self) -> Iterator[None]:
        with contextlib.chdir(self.work), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield

    def simulate(self, config: Mapping[str, Any]) -> Any:
        """A simulated ``breos.App`` for ``config``, its weather staged first."""
        import breos

        location = config.get("location")
        if isinstance(location, str) and location in TMY_FILES:
            self.stage_tmy(location)
        with self.inside():
            app = breos.App(dict(config))
            app.simulate()
        provenance = app.result()["provenance"]
        self.currencies.add(provenance["currency"])
        self.timezones.add(provenance["timezone"])
        return app

    def revalue(self, app: Any, changes: Mapping[str, Any]) -> dict[str, Any]:
        with self.inside():
            return app.revalue(changes)

    def roots(self) -> dict[str, str]:
        """Absolute prefixes to rewrite: a staged TMY becomes its committed path, the rest repo-relative."""
        staged = {}
        for committed in self.weather:
            if committed.startswith("validation/"):
                name = Path(committed).name
                for work in (self.work, self.work.resolve()):
                    staged[str(work / "weather" / name)] = committed
        return {**staged, str(self.work.resolve()): "", str(self.work): "", str(REPO): ""}


@dataclass(frozen=True)
class Case:
    name: str
    title: str
    run: Callable[[Context], Output]
    cheap: bool = False


CASES: dict[str, Case] = {}


def case(name: str, title: str, *, cheap: bool = False) -> Callable[[Callable[[Context], Output]], Callable]:
    def register(function: Callable[[Context], Output]) -> Callable[[Context], Output]:
        CASES[name] = Case(name, title, function, cheap)
        return function

    return register


# --------------------------------------------------------------------------- cases

QUICKSTART = "configs/examples/quickstart.toml"


@case("first_home", "PV and battery system in Porto", cheap=True)
def first_home(ctx: Context) -> Output:
    config = ctx.config(QUICKSTART)
    app = ctx.simulate(config)
    result = app.result()
    frame = first_year_frame(app)
    timezone = result["provenance"]["timezone"]
    weeks = {"winter": 3, "summer": 27}
    files: dict[str, Any] = {"result.json": result}
    for label, week in weeks.items():
        files[f"week_{label}.csv"] = week_slice(frame, timezone, week)
    return Output(files, {"iso_weeks": weeks})


@case("battery_worth", "Battery size and storage price", cheap=True)
def battery_worth(ctx: Context) -> Output:
    base = ctx.config(QUICKSTART)
    sizes = (0.0, 5.0, 10.0)
    storage_prices = list(range(150, 625, 25))
    designs, rows = [], []
    for size in sizes:
        app = ctx.simulate(with_overrides(base, {"battery_kwh": size}))
        result = app.result()
        designs.append(
            {
                **scalars(result),
                "financial": financial_rows(result),
                "provenance": {"currency": result["provenance"]["currency"]},
            }
        )
        if size == 0:
            continue
        for price in storage_prices:
            revalued = ctx.revalue(app, {"costs": {"storage_cost_per_kwh": price}})
            rows.append(
                {
                    "battery_kwh": size,
                    "storage_cost_per_kwh": price,
                    "npv_savings": revalued["npv_savings"],
                    "payback_year": revalued["payback_year"],
                    "total_investment": revalued["total_investment"],
                    "method": revalued["provenance"]["revaluation"]["method"],
                }
            )
    return Output(
        {"designs.json": designs, "storage_prices.csv": pd.DataFrame(rows)},
        {
            "variants": {f"{size:g} kWh": {"battery_kwh": size} for size in sizes},
            "storage_prices": storage_prices,
            "preset_storage_cost_per_kwh": _preset_cost(result, "storage_cost_per_kwh"),
        },
    )


@case("sun_prices_carbon", "Porto and Berlin compared", cheap=True)
def sun_prices_carbon(ctx: Context) -> Output:
    base = ctx.config(QUICKSTART)
    variants = {
        "Porto": {},
        "Berlin, Porto prices and grid": {"location": "berlin"},
        "Berlin": {"location": "berlin", "cost_preset": "residential_de", "emissions_country": "DE"},
    }
    sites = []
    for name, overrides in variants.items():
        result = ctx.simulate(with_overrides(base, overrides)).result()
        sites.append(
            {
                "site": name,
                **scalars(result),
                "co2_avoided_self_consumption_lifetime_kg": result["co2_avoided_self_consumption_lifetime_kg"],
                "co2_avoided_export_lifetime_kg": result["co2_avoided_export_lifetime_kg"],
                "tilt": result["provenance"]["resolved_config"]["tilt"],
                "electricity_cost": _preset_cost(result, "electricity_cost"),
                "electricity_sold_cost": _preset_cost(result, "electricity_sold_cost"),
                "grid_carbon": _grid_carbon(result),
                "monthly_production_kwh": [row["usable_ac_system_production_kwh"] for row in result["monthly"]],
                "monthly_consumption_kwh": [row["consumption_kwh"] for row in result["monthly"]],
                "financial": financial_rows(result),
                "provenance": {"currency": result["provenance"]["currency"]},
            }
        )
    return Output({"sites.json": sites}, {"variants": variants})


def _preset_cost(result: Mapping[str, Any], key: str) -> float:
    from breos.resources import load_config_json

    preset = result["provenance"]["resolved_config"]["cost_preset"]
    return float(load_config_json("costs.json")[preset][key])


def _grid_carbon(result: Mapping[str, Any]) -> dict[str, Any]:
    from breos.resources import load_config_json

    country = result["provenance"]["resolved_config"]["emissions_country"]
    entry = load_config_json("emissions.json")[country]
    return {"country": country, **entry}


@case("east_west", "East-west and south-facing arrays", cheap=True)
def east_west(ctx: Context) -> Output:
    base = ctx.config("configs/examples/east-west-roof.toml")
    modules = sum(array["modules"] for array in base["pv_arrays"])
    tilt = base["pv_arrays"][0]["tilt"]
    south = 35
    designs = {
        f"East-West {tilt}°": {},
        f"South {tilt}°": {"pv_arrays": None, "n_modules": modules, "tilt": tilt, "azimuth": 180},
        f"South {south}°": {"pv_arrays": None, "n_modules": modules, "tilt": south, "azimuth": 180},
    }
    rows, summer, winter = [], None, None
    for battery in (0.0, base["battery_kwh"]):
        for name, overrides in designs.items():
            app = ctx.simulate(with_overrides(base, {**overrides, "battery_kwh": battery}))
            result = app.result()
            resolved = result["provenance"]["resolved_config"]
            arrays = resolved.get("pv_arrays") or [resolved]
            rows.append(
                {
                    "design": name,
                    **scalars(result),
                    "tilts": ";".join(f"{array['tilt']:g}" for array in arrays),
                    "azimuths": ";".join(f"{array['azimuth']:g}" for array in arrays),
                }
            )
            if battery:
                continue
            frame = first_year_frame(app)
            timezone = result["provenance"]["timezone"]
            for label, week in (("summer", 27), ("winter", 3)):
                piece = week_slice(frame, timezone, week)[["Datetime", "PV_Production", "Houseload"]]
                piece = piece.rename(columns={"PV_Production": name})
                if label == "summer":
                    summer = piece if summer is None else summer.join(piece[[name]])
                else:
                    winter = piece if winter is None else winter.join(piece[[name]])
    assert summer is not None and winter is not None
    order = ["Datetime", "Houseload", *designs]
    return Output(
        {"designs.csv": pd.DataFrame(rows), "week_summer.csv": summer[order], "week_winter.csv": winter[order]},
        {
            "variants": designs,
            "battery_kwh": [0.0, base["battery_kwh"]],
            "iso_weeks": {"summer": 27, "winter": 3},
        },
    )


# The optical and solar-position choices of configs/examples/recommended-pv.toml,
# applied to the PV-only example so that only the sky and optics model changes.
PEREZ_MARION = {
    "transposition_model": "perez",
    "diffuse_iam": "marion",
    "iam_model": "physical",
    "solar_position": "weather",
}


# What the orientation page reads from each of its ~1900 runs.
ORIENTATION_FIELDS = ("usable_ac_system_production_kwh", "self_consumption_pct", "npv_savings")


@case("orientation", "Array tilt and azimuth", cheap=False)
def orientation(ctx: Context) -> Output:
    base = ctx.config("configs/examples/pv-only.toml")
    models = {"default": {}, "perez_marion": PEREZ_MARION}
    tilts = list(range(0, 65, 5))
    azimuths = list(range(90, 275, 5))
    fine_tilts = list(range(30, 45))
    fine_azimuths = list(range(170, 201))
    coarse, fine, defaults = [], [], {}
    for model, overrides in models.items():
        config = with_overrides(base, overrides)
        for grid, rows in (((tilts, azimuths), coarse), ((fine_tilts, fine_azimuths), fine)):
            for tilt in grid[0]:
                for azimuth in grid[1]:
                    result = ctx.simulate(with_overrides(config, {"tilt": tilt, "azimuth": azimuth})).result()
                    rows.append(
                        {
                            "model": model,
                            "tilt": tilt,
                            "azimuth": azimuth,
                            **{key: result[key] for key in ORIENTATION_FIELDS},
                        }
                    )
        best = max(
            (row for row in fine if row["model"] == model), key=lambda row: row["usable_ac_system_production_kwh"]
        )
        if best["tilt"] in (fine_tilts[0], fine_tilts[-1]) or best["azimuth"] in (fine_azimuths[0], fine_azimuths[-1]):
            raise RuntimeError(f"the {model} optimum {best['tilt']}/{best['azimuth']} is on the fine grid's edge")
        result = ctx.simulate(config).result()
        resolved = result["provenance"]["resolved_config"]
        defaults[model] = {"tilt": resolved["tilt"], "azimuth": resolved["azimuth"], **scalars(result)}
    return Output(
        {"orientations.csv": pd.DataFrame(coarse), "fine.csv": pd.DataFrame(fine), "default.json": defaults},
        {
            "models": models,
            "tilts": tilts,
            "azimuths": azimuths,
            "fine_tilts": fine_tilts,
            "fine_azimuths": fine_azimuths,
        },
    )


@case("clipping", "DC/AC ratio and clipping at 15 minutes", cheap=False)
def clipping(ctx: Context) -> Output:
    base = with_overrides(ctx.config(QUICKSTART), {"resolution": "15min"})
    ratios = [1.0, 1.25, 1.5, 1.75, 2.0]
    rows, week, distribution = [], None, None
    for battery in (0.0, base["battery_kwh"]):
        for ratio in ratios:
            app = ctx.simulate(with_overrides(base, {"battery_kwh": battery, "inverter_loading_ratio": ratio}))
            result = app.result()
            inverter = result["pv_loss_waterfall"]["inverter"]
            rows.append(
                {
                    "inverter_loading_ratio": ratio,
                    **scalars(result),
                    "inverter_ac_kw": inverter["ac_capacity_kw"],
                    "inverter_conversion_loss_kwh": inverter["conversion_loss_kwh"],
                }
            )
            if battery == 0 and ratio == min(ratios):
                distribution = dc_power_distribution(first_year_frame(app), result["pv_kwp"])
            if battery == 0 and ratio == max(ratios):
                piece = week_slice(first_year_frame(app), result["provenance"]["timezone"], 27)
                week = piece[["Datetime", "PV_DC", "PV_DC_Curtailed", "PV_Production", "Houseload"]]
    return Output(
        {"ratios.csv": pd.DataFrame(rows), "week_summer.csv": week, "dc_power_distribution.csv": distribution},
        {
            "overrides": {"resolution": "15min"},
            "ratios": ratios,
            "week_ratio": max(ratios),
            "iso_week": 27,
            "inverter_efficiency": result["provenance"]["resolved_config"]["inverter_efficiency"],
        },
    )


def dc_power_distribution(frame: pd.DataFrame, pv_kwp: float, width: float = 0.025) -> pd.DataFrame:
    """Year-1 DC energy by DC power as a fraction of the array's DC rating, in bins of ``width``."""
    instants = pd.to_datetime(frame["Datetime"], utc=True)
    step_h = (instants.iloc[1] - instants.iloc[0]).total_seconds() / 3600
    fraction = frame["PV_DC"] / (pv_kwp * 1000)
    edges = np.arange(0.0, max(1.0, float(fraction.max())) + width, width)
    bins = pd.cut(fraction[fraction > 0], edges, right=False)
    energy = (frame.loc[fraction > 0, "PV_DC"] * step_h / 1000).groupby(bins, observed=False).sum()
    return pd.DataFrame(
        {
            "dc_fraction_low": edges[:-1].round(3),
            "dc_fraction_high": edges[1:].round(3),
            "energy_kwh": energy.to_numpy().round(2),
        }
    )


@case("quarterly_berlin", "Discharge-only dispatch on a quarterly tariff in Berlin", cheap=True)
def quarterly_berlin(ctx: Context) -> Output:
    base = ctx.config("configs/examples/quarterly-tariff-berlin.toml")
    strategies = {
        "greedy": {},
        "discharge_only high": {"smart_charging": {"mode": "discharge_only", "discharge_periods": ["high"]}},
        "discharge_only standard and high": {
            "smart_charging": {"mode": "discharge_only", "discharge_periods": ["standard", "high"]}
        },
    }
    rows = []
    for name, overrides in strategies.items():
        result = ctx.simulate(with_overrides(base, overrides)).result()
        rows.append(
            {
                "strategy": name,
                **scalars(result),
                "monthly_battery_to_load_kwh": [row["pv_origin_battery_ac_load_kwh"] for row in result["monthly"]],
                "monthly_import_kwh": [row["grid_import_kwh"] for row in result["monthly"]],
            }
        )
    return Output({"strategies.json": rows}, {"variants": strategies})


# The ageing page's degradation models: the native engine's two field fits and
# three BLAST cell models, by key, label and the config keys that select them.
AGEING_MODELS = (
    ("native_v1", "Native LFP, v1 field fit", {"calendar_model": "naumann_lam_field_calibrated_v1"}),
    ("native_v2", "Native LFP, v2 field fit", {"calendar_model": "naumann_lam_field_calibrated_v2"}),
    (
        "lfp_gr_250ah_prismatic",
        "BLAST LFP-Gr 250 Ah",
        {"degradation_engine": "blast", "blast_model": "lfp_gr_250ah_prismatic"},
    ),
    (
        "lfp_gr_sonymurata_3ah",
        "BLAST LFP-Gr 3 Ah",
        {"degradation_engine": "blast", "blast_model": "lfp_gr_sonymurata_3ah"},
    ),
    ("nmc_gr_50ah_b1", "BLAST NMC-Gr 50 Ah", {"degradation_engine": "blast", "blast_model": "nmc_gr_50ah_b1"}),
)
# One pack serves the whole projection, so its health can be followed past end
# of life. The PV array keeps its first-year output, so the only feedback into
# the dispatch is the battery's own fade.
AGEING_OVERRIDES = {"battery_enable_replacement": False, "pv_degradation_rate": 0.0}
# The BLAST inputs the ageing page sets against each model's experimental-range limits.
AGEING_INPUTS = ("c_rate_charge", "dod", "temperature_c")


@contextlib.contextmanager
def blast_inputs(record: dict[str, dict[str, list[float]]]) -> Iterator[None]:
    """Record, per BLAST model key, each period's charge C-rate, depth of discharge and cell temperature.

    The result keeps only the first input outside a model's limits.
    This keeps every period's, as the experimental-range check
    (``BlastWarningCollector.check_experimental_range``) receives them. The
    check measures the C-rate against the aged capacity; the rate times the
    state of health at the period's start is the same rate against the
    nominal capacity.
    """
    from unittest import mock

    from breos.degradation.engine import BlastEngine
    from breos.degradation.validation import BlastWarningCollector

    step, check = BlastEngine.step, BlastWarningCollector.check_experimental_range
    start_soh: dict[str, float] = {}

    def stepping(self: Any, t_secs_day: Any, soc_abs_day: Any, t_cell_day_c: Any) -> float:
        start_soh[self.blast_model_key] = self.soh()
        return step(self, t_secs_day, soc_abs_day, t_cell_day_c)

    def recording(self: Any, t_secs: np.ndarray, soc: np.ndarray, temperature_c: np.ndarray) -> None:
        periods = record.setdefault(
            self.blast_model_key,
            {"c_rate_charge": [], "c_rate_charge_nominal": [], "dod": [], "t_min": [], "t_max": []},
        )
        c_rate = max(0.0, float(np.max(np.diff(soc) / (np.diff(t_secs) / 3600.0))))
        periods["c_rate_charge"].append(c_rate)
        periods["c_rate_charge_nominal"].append(c_rate * start_soh[self.blast_model_key])
        periods["dod"].append(float(np.ptp(soc)))
        periods["t_min"].append(float(np.min(temperature_c)))
        periods["t_max"].append(float(np.max(temperature_c)))
        check(self, t_secs, soc, temperature_c)

    with (
        mock.patch.object(BlastEngine, "step", stepping),
        mock.patch.object(BlastWarningCollector, "check_experimental_range", recording),
    ):
        yield


def input_ranges(
    periods: Mapping[str, list[float]], limits: Mapping[str, Any], years: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Each input in ``AGEING_INPUTS``: its limits, the range the run gave it, and the periods outside.

    ``limits`` is the model's ``experimental_range``, which the range check
    warns outside of. A charge C-rate is limited to a maximum, so it has no
    minimum; it is given against the aged capacity, as the check measures it,
    and against the nominal capacity. A period that does not move the state of
    charge has no depth of discharge, as in the check itself, so the depth has
    no minimum either. ``first_year_outside`` is the project year of the first
    period outside, for a run of ``years`` equal years. Also returns, per
    input with a period outside, the first such value as the check's warning
    records it (``observed``).
    """
    c_rate, nominal = np.asarray(periods["c_rate_charge"]), np.asarray(periods["c_rate_charge_nominal"])
    dod = np.asarray(periods["dod"])
    moved = dod > 1e-12
    t_min, t_max = np.asarray(periods["t_min"]), np.asarray(periods["t_max"])
    c_limit = limits["max_c_rate_charge"]
    dod_low, dod_high = min(limits["dod"]), max(limits["dod"])
    t_low, t_high = min(limits["cycling_temperature_c"]), max(limits["cycling_temperature_c"])
    rows = (
        ("c_rate_charge", None, c_limit, None, c_rate.max(), c_rate > c_limit + 1e-12, lambda i: float(c_rate[i])),
        ("dod", dod_low, dod_high, None, dod.max(), moved & ((dod < dod_low - 1e-12) | (dod > dod_high + 1e-12)),
         lambda i: [float(dod[i]), float(dod[i])]),
        ("temperature_c", t_low, t_high, t_min.min(), t_max.max(), (t_min < t_low - 1e-12) | (t_max > t_high + 1e-12),
         lambda i: [float(t_min[i]), float(t_max[i])]),
    )  # fmt: skip
    ranges, observed = [], {}
    for name, low, high, run_low, run_high, outside, value in rows:
        first = int(np.argmax(outside)) if outside.any() else None
        if first is not None:
            observed[name] = value(first)
        ranges.append(
            {
                "input": name,
                "limit_min": json_ready(low),
                "limit_max": json_ready(high),
                "run_min": None if run_low is None else round(float(run_low), 4),
                "run_max": round(float(run_high), 4),
                "periods_outside": int(outside.sum()),
                "periods": len(dod),
                "first_year_outside": None if first is None else first * years // len(dod) + 1,
                "run_max_nominal": round(float(nominal.max()), 4) if name == "c_rate_charge" else None,
                "periods_outside_nominal": int((nominal > c_limit + 1e-12).sum()) if name == "c_rate_charge" else None,
            }
        )
    return ranges, observed


def imposed_ageing(
    selection: Mapping[str, Any], soc: np.ndarray, temperature_c: np.ndarray, *, years: int, step_seconds: float,
    start_soc: float, eol_fraction: float,
) -> tuple[list[dict[str, Any]], float | None]:  # fmt: skip
    """Age one model under one year of SOC and cell temperature, repeated for ``years``.

    The stress history is imposed: it does not respond to the model's fade.
    Each day is handed to the same degradation lifecycle the simulation
    steps (``breos.battery._build_degradation_lifecycle``), in the same
    one-day periods, and the rainflow residue is closed after the last day.
    Returns the unrounded year-end rows and the time of the first end-of-life
    crossing in years, defined as in the simulation: the end of the first
    period that closes at or below ``eol_fraction``, or None.
    """
    from breos.battery import BatteryConfig, _build_degradation_lifecycle
    from breos.degradation.protocol import DegradationDay

    engine = selection.get("degradation_engine", "native")
    battery = BatteryConfig(
        nominal_energy_wh=1000.0, **({"calendar_model": selection["calendar_model"]} if engine == "native" else {})
    )
    lifecycle, day_start_soc, day_start_t = _build_degradation_lifecycle(
        engine,
        battery,
        battery_soh_decimal=1.0,
        has_battery=True,
        blast_model=selection.get("blast_model"),
        initial_degradation_state=None,
        initial_fec=0.0,
        initial_calendar_seconds=0.0,
        default_day_start_soc=start_soc,
        default_day_start_t_cell=float(temperature_c[0]),
    )
    per_day = int(round(86400.0 / step_seconds))
    ticks = np.arange(per_day, dtype=float) * step_seconds
    cycle_loss = calendar_loss = 0.0
    first_end_of_life = None
    rows = []
    for year in range(1, years + 1):
        for start in range(0, len(soc), per_day):
            day_soc, day_t = soc[start : start + per_day], temperature_c[start : start + per_day]
            step = lifecycle.step(
                DegradationDay(
                    soc=day_soc,
                    time_ticks=ticks[: len(day_soc)],
                    ticks_per_second=1.0,
                    temperature_c=day_t,
                    step_seconds=step_seconds,
                    start_soc=day_start_soc,
                    start_temperature_c=day_start_t,
                    finalize_cycles=year == years and start + per_day >= len(soc),
                )
            )
            cycle_loss += step.cycle_degradation
            calendar_loss += step.calendar_degradation
            day_start_soc, day_start_t = float(day_soc[-1]), float(day_t[-1])
            if first_end_of_life is None and step.soh_fraction <= eol_fraction:
                first_end_of_life = year - 1 + (start + len(day_soc)) / len(soc)
        rows.append(
            {
                "year": year,
                "soh_pct": step.soh_fraction * 100.0,
                "cumulative_fec": step.fec,
                # BLAST reports no split; its zero increments mean "not supplied".
                "cycle_loss_pct": cycle_loss * 100.0 if engine == "native" else None,
                "calendar_loss_pct": calendar_loss * 100.0 if engine == "native" else None,
            }
        )
    return rows, first_end_of_life


@case("ageing", "Battery ageing with different degradation models", cheap=False)
def ageing(ctx: Context) -> Output:
    import breos

    base = with_overrides(ctx.config(QUICKSTART), AGEING_OVERRIDES)
    years = int(base["projection_years"])
    battery_wh = float(base["battery_kwh"]) * 1000.0
    models, simulated, imposed = [], [], []
    runs: dict[str, tuple[pd.DataFrame, dict[str, Any]]] = {}
    inputs: dict[str, dict[str, list[float]]] = {}
    for key, label, selection in AGEING_MODELS:
        with blast_inputs(inputs):
            app = ctx.simulate(with_overrides(base, selection))
        result = app.result()
        runs[key] = first_year_frame(app), result
        for entry in result["battery_degradation_history"]:
            simulated.append({"model": key, **entry})
        events = result["battery_end_of_life_events"]
        degradation = result["degradation"]
        warned = sorted(
            {
                warning.get("field") or warning.get("code")
                for warning in degradation.get("experimental_range_warnings", [])
            }
        )
        profile = breos.get_battery_model_profile(selection["blast_model"]) if "blast_model" in selection else None
        ranges = None
        if profile is not None:
            ranges, first_outside = input_ranges(inputs[profile.key], profile.experimental_range, years)
            # The page shows these inputs only, so they must be every input the result warns about,
            # and the recorder must find the first value outside where the check found it.
            recorded = {
                warning["field"]: warning["observed"] for warning in degradation.get("experimental_range_warnings", [])
            }
            if first_outside != recorded:
                raise RuntimeError(f"{key}: recorded first inputs outside {first_outside}, warnings {recorded}")
        models.append(
            {
                "model": key,
                "label": label,
                "engine": degradation["engine"],
                "model_key": degradation["model_key"],
                "chemistry": profile.chemistry if profile is not None else "LFP/graphite",
                "cell_capacity_ah": profile.nominal_capacity_ah if profile is not None else None,
                "basis": "laboratory cell model"
                if profile is not None
                else "field calendar fit, laboratory cycle model",
                "first_end_of_life_years": events[0]["time_years"] if events else None,
                "battery_soh_end_pct": result["battery_soh_end_pct"],
                "experimental_range_warnings": warned,
                "input_ranges": ranges,
                "aging_horizon_days": profile.aging_horizon_days if profile is not None else None,
            }
        )
    resolved = runs[AGEING_MODELS[0][0]][1]["provenance"]["resolved_config"]
    eol_fraction = float(resolved["battery_eol_percentage"])

    def stress(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return frame["Battery_SOC_Absolute"].to_numpy(dtype=float), frame["T_cell"].to_numpy(dtype=float)

    with ctx.inside():
        for (key, _label, selection), entry in zip(AGEING_MODELS, models, strict=True):
            frame, result = runs[key]
            options = {
                "step_seconds": 3600.0,
                "start_soc": float(frame["Battery_Energy_Beginning"].iloc[0]) / battery_wh,
                "eol_fraction": eol_fraction,
            }
            # The result's own precision: two decimals for the native engine, one for a BLAST cell model.
            digits = 2 if entry["engine"] == "native" else 1
            # Each model's own first simulated year, imposed, must give that run's first-year health
            # to within the rounding of the stored result. Two years, so the first does not close the
            # rainflow residue the simulation carries on.
            own, _ = imposed_ageing(selection, *stress(frame), years=2, **options)
            first = result["battery_degradation_history"][0]["soh_pct"]
            if abs(own[0]["soh_pct"] - first) > 0.5 * 10.0**-digits + 1e-9:
                raise RuntimeError(
                    f"{key}: imposed year 1 ({own[0]['soh_pct']}) does not reproduce the simulation ({first})"
                )
            # Every model is given the first model's first simulated year.
            rows, first_end_of_life = imposed_ageing(
                selection, *stress(runs[AGEING_MODELS[0][0]][0]), years=years, **options
            )
            imposed.extend(
                {
                    "model": key,
                    "year": row["year"],
                    "soh_pct": round(row["soh_pct"], digits),
                    "cumulative_fec": round(row["cumulative_fec"], 2),
                    **{
                        column: None if row[column] is None else round(row[column], digits)
                        for column in ("cycle_loss_pct", "calendar_loss_pct")
                    },
                }
                for row in rows
            )
            entry["imposed_first_end_of_life_years"] = (
                None if first_end_of_life is None else round(first_end_of_life, 4)
            )
    stress_year = runs[AGEING_MODELS[0][0]][1]["battery_degradation_history"][0]
    return Output(
        {
            "models.json": models,
            "simulated.csv": pd.DataFrame(simulated),
            "imposed.csv": pd.DataFrame(imposed),
        },
        {
            "overrides": AGEING_OVERRIDES,
            "battery_eol_percentage": eol_fraction,
            "soc_window": [resolved["battery_min_soc"], resolved["battery_max_soc"]],
            "imposed_stress": {
                "source": AGEING_MODELS[0][0],
                "mean_soc_pct": stress_year["mean_soc_pct"],
                "mean_cell_temperature_c": stress_year["mean_cell_temperature_c"],
                "fec_per_year": stress_year["fec"],
            },
        },
    )


@case("replacement_timing", "Battery replacement timing", cheap=False)
def replacement_timing(ctx: Context) -> Output:
    base = with_overrides(ctx.config(QUICKSTART), {"terminal_value": {"basis": "battery_health_fraction"}})

    def row(result: Mapping[str, Any], **extra: Any) -> dict[str, Any]:
        swaps = replacement_times(result)
        return {
            **extra,
            "npv_savings": result["npv_savings"],
            "npv_savings_terminal_adjusted": result["npv_savings_terminal_adjusted"],
            "terminal_health_credit_npv": result["terminal_health_credit_npv"],
            "battery_soh_end_pct": result["battery_soh_end_pct"],
            "battery_replacements": result["battery_replacements"],
            "battery_replacement_cost_npv": result["battery_replacement_cost_npv"],
            "last_replacement_years": swaps[-1] if swaps else None,
            "replacement_times_years": ";".join(f"{value:g}" for value in swaps),
        }

    horizons = list(range(6, 26))
    horizon_rows = [
        row(ctx.simulate(with_overrides(base, {"projection_years": years})).result(), projection_years=years)
        for years in horizons
    ]
    module_counts = list(range(8, 15))
    module_rows = []
    for count in module_counts:
        for allow in (True, False):
            config = with_overrides(base, {"n_modules": count, "battery_allow_terminal_replacement": allow})
            result = ctx.simulate(config).result()
            module_rows.append(
                row(
                    result,
                    n_modules=count,
                    pv_kwp=result["pv_kwp"],
                    terminal_replacement=allow,
                    projection_years=len(result["yearly"]),
                )
            )
    return Output(
        {"horizons.csv": pd.DataFrame(horizon_rows), "modules.csv": pd.DataFrame(module_rows)},
        {
            "overrides": {"terminal_value": {"basis": "battery_health_fraction"}},
            "projection_years": horizons,
            "n_modules": module_counts,
        },
    )


@case("which_tariff", "Simple, bi-hourly and tri-hourly tariffs", cheap=False)
def which_tariff(ctx: Context) -> Output:
    sweep_config = ctx.config("configs/examples/tariff-comparison.toml")
    sweep = sweep_config.pop("sweep")
    names = ("simple", "bi-hourly", "tri-hourly")
    offers = dict(zip(names, sweep["tariff"], strict=True))
    today = offers["simple"]
    reference = {
        "currency": today["currency"],
        "import_prices": dict(today["import_prices"]),
        "fixed_charge_per_day": today["fixed_charge_per_day"],
    }
    rows = []
    for battery in sweep["battery_kwh"]:
        for name, tariff in offers.items():
            app = ctx.simulate(with_overrides(sweep_config, {"battery_kwh": battery, "tariff": tariff}))
            for baseline, result in (
                ("same offer", app.result()),
                ("today's simple offer", ctx.revalue(app, {"reference_tariff": reference})),
            ):
                rows.append(
                    {
                        "offer": name,
                        "battery_kwh": battery,
                        "baseline": baseline,
                        "npv_savings": result["npv_savings"],
                        "payback_year": result["payback_year"],
                        "bill_year1": year1_bill(result),
                        "no_system_bill_year1": no_system_bill(result),
                        "project_cost": result["financial"][-1]["cost_with_system"],
                        "no_system_project_cost": result["financial"][-1]["cost_without_system"],
                        "grid_import_kwh": result["grid_import_kwh"],
                    }
                )
    return Output(
        {"offers.csv": pd.DataFrame(rows)},
        {"offers": offers, "reference_tariff": reference, "battery_kwh": sweep["battery_kwh"]},
    )


@case("dispatch_strategies", "Dispatch strategies on a bi-hourly tariff", cheap=False)
def dispatch_strategies(ctx: Context) -> Output:
    base = ctx.config("configs/examples/smart-charging-portugal.toml")
    fixed = base["smart_charging"]
    strategies = {
        "greedy": {"smart_charging": None},
        "discharge_only": {"smart_charging": {"mode": "discharge_only", "discharge_periods": ["peak"]}},
        "fixed_target": {},
        "hold_target": {
            "smart_charging": {
                **fixed,
                "discharge_periods": ["peak", "off_peak"],
                "overlap_policy": "hold_target",
            }
        },
    }
    rows, week = [], None
    for name, overrides in strategies.items():
        app = ctx.simulate(with_overrides(base, overrides))
        result = app.result()
        rows.append({"strategy": name, **scalars(result)})
        piece = week_slice(first_year_frame(app), result["provenance"]["timezone"], 6)
        piece = piece[["Datetime", "Houseload", "PV_Production", "Import_From_Grid", "Grid_AC_To_Battery",
                       "Battery_SOC_Normalized"]]  # fmt: skip
        piece.columns = ["Datetime", *(f"{name}:{column}" for column in piece.columns[1:])]
        week = piece if week is None else week.join(piece.drop(columns="Datetime"))
    return Output(
        {"strategies.csv": pd.DataFrame(rows), "week_winter.csv": week},
        {"variants": strategies, "iso_week": 6},
    )


@case("price_scenarios", "Price scenarios with App.revalue", cheap=True)
def price_scenarios(ctx: Context) -> Output:
    base = ctx.config("configs/examples/time-of-use-portugal.toml")
    started = time.perf_counter()
    app = ctx.simulate(base)
    simulate_s = time.perf_counter() - started
    off_peak = base["tariff"]["import_prices"]["off_peak"]
    escalations = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    peaks = [0.18, base["tariff"]["import_prices"]["peak"], 0.28, 0.33]
    rows = []
    started = time.perf_counter()
    for escalation in escalations:
        for peak in peaks:
            changes = {
                "import_price_escalation": escalation,
                "tariff": {"import_prices": {"peak": peak, "off_peak": off_peak}},
            }
            result = ctx.revalue(app, changes)
            rows.append(
                {
                    "import_price_escalation": escalation,
                    "peak_price": peak,
                    "npv_savings": result["npv_savings"],
                    "payback_year": result["payback_year"],
                    "method": result["provenance"]["revaluation"]["method"],
                }
            )
    revalue_s = time.perf_counter() - started
    # One scenario simulated from scratch, to show revaluation gives the same answer.
    probe = {"import_price_escalation": 0.03, "tariff": {"import_prices": {"peak": 0.28, "off_peak": off_peak}}}
    fresh_config = with_overrides(
        base, {"import_price_escalation": 0.03, "tariff": {**base["tariff"], **probe["tariff"]}}
    )
    fresh = ctx.simulate(fresh_config).result()
    revalued = ctx.revalue(app, probe)
    parity = {
        "changes": probe,
        "revalued_npv_savings": revalued["npv_savings"],
        "simulated_npv_savings": fresh["npv_savings"],
        "base_npv_savings": app.result()["npv_savings"],
    }
    return Output(
        {"scenarios.csv": pd.DataFrame(rows), "parity.json": parity},
        {
            "scenarios": len(rows),
            "base_peak_price": base["tariff"]["import_prices"]["peak"],
            "base_import_price_escalation": app.result()["provenance"]["economics"]["import_price_escalation"],
            "simulate_s": round(simulate_s, 2),
            "revalue_all_s": round(revalue_s, 2),
        },
    )


@case("montecarlo", "Monte Carlo analysis of weather and demand", cheap=False)
def montecarlo(ctx: Context) -> Output:
    from breos.montecarlo import MonteCarloSettings, run_montecarlo
    from breos.weather import load_weather

    config = ctx.config("configs/examples/montecarlo.toml")
    mc = dict(config.pop("montecarlo"))
    mc.pop("weather_file")
    history = _mc_history(ctx)
    history_name = f"weather/{history.name}"
    ctx.weather[history_name] = {
        "sha256": sha256(history),
        "attribution": OPEN_METEO_ATTRIBUTION,
        "note": "fetched at regeneration time; not committed",
    }
    settings = MonteCarloSettings(weather_file=str(history), **mc)
    with ctx.inside():
        study = run_montecarlo(config, settings)
    deterministic = ctx.simulate(with_overrides(config, {"start_date": f"{settings.target_year}-01-01"})).result()

    ctx.stage_tmy(MC_HISTORY["location"])
    with ctx.inside():
        hourly = load_weather(MC_HISTORY["location"], data_type="historical")
        tmy = load_weather(MC_HISTORY["location"], data_type="tmy")
    if hourly is None or tmy is None:
        raise RuntimeError("the staged Monte Carlo history or TMY did not load")
    # Open-Meteo's hourly means are labelled at the end of their hour.
    years = (hourly.index - pd.Timedelta(hours=1)).year
    ghi_by_year = (hourly["shortwave_radiation"].groupby(years).sum() / 1000.0).round(2)
    weather_years = pd.DataFrame({"year": ghi_by_year.index.astype(int), "ghi_kwh_m2": ghi_by_year.to_numpy()})
    summary = {
        "summary": study.summary,
        "available_years": list(study.available_years),
        "settings": {
            key: getattr(study.settings, key)
            for key in (
                "n_runs",
                "years_per_run",
                "load_uncertainty",
                "load_distribution",
                "target_year",
                "seed",
                "execution_backend",
            )
        },  # fmt: skip
        "tmy_ghi_kwh_m2": round(float(tmy["ghi"].sum()) / 1000.0, 2),
        "tmy_file": TMY_FILES[MC_HISTORY["location"]],
        "tmy_result": scalars(deterministic),
    }
    return Output(
        {"runs.csv": study.runs, "summary.json": summary, "weather_years.csv": weather_years},
        {"history": {"location": MC_HISTORY["location"], "start": MC_HISTORY["start"], "end": MC_HISTORY["end"]}},
    )


def _mc_history(ctx: Context) -> Path:
    if ctx.options.mc_weather:
        source = Path(ctx.options.mc_weather).resolve()
        target = ctx.work / "weather" / source.name
        shutil.copyfile(source, target)
        return target
    from breos.resources import load_config_json
    from breos.weather import fetch_weather_data

    site = load_config_json("locations.json")[MC_HISTORY["location"]]
    with ctx.inside():
        fetch_weather_data(
            site["latitude"],
            site["longitude"],
            MC_HISTORY["start"],
            MC_HISTORY["end"],
            tilt=35,
            azimuth=0,
            location_name=MC_HISTORY["location"],
            output_dir="weather",
        )
    start, end = MC_HISTORY["start"][:4], MC_HISTORY["end"][:4]
    return ctx.work / "weather" / f"{MC_HISTORY['location']}_historical_{start}_{end}_openmeteo.csv"


@case("nsga2_front", "Multi-objective sizing with NSGA-II", cheap=False)
def nsga2_front(ctx: Context) -> Output:
    from breos.load_profiles import load_profile
    from breos.optimization import optimize_system_multi_objective
    from breos.weather import load_weather

    config = ctx.config("configs/optimization/projected-optimization.toml")
    ctx.stage_tmy("porto")
    annual_kwh = 4000.0
    with ctx.inside():
        weather = load_weather("porto", data_type="tmy")
    if weather is None:
        raise RuntimeError("the staged Porto TMY did not load")
    with ctx.inside():
        load = load_profile(
            "demandlib_h0",
            annual_kwh,
            start_date=f"{weather.index[0].year}-01-01",
            freq=config["simulation"]["resolution"],
            timezone=config["location"]["timezone"],
        )
        result = optimize_system_multi_objective(weather, load, config, n_procs=ctx.options.procs)
    if result.details["pareto"].attrs.get("currency"):
        ctx.currencies.add(str(result.details["pareto"].attrs["currency"]))
    pareto = result.details["pareto"].sort_values("Battery_kWh").reset_index(drop=True)
    return Output(
        {"pareto.csv": pareto},
        {
            "annual_consumption_kwh": annual_kwh,
            "load_profile": "demandlib_h0",
            "n_procs": ctx.options.procs,
            "generations": result.iterations,
            "constraints": config["constraints"],
            "module_area_m2": round(config["pv"]["module_width_m"] * config["pv"]["module_length_m"], 4),
        },
    )


# --------------------------------------------------------------------------- storage


def write_case(entry: Case, output: Output, target: Path, ctx: Context, runtime_s: float) -> None:
    import breos

    target.mkdir(parents=True, exist_ok=True)
    for stale in target.iterdir():
        if stale.is_file():
            stale.unlink()
    roots = ctx.roots()
    files = {}
    for name, value in output.files.items():
        path = target / name
        if isinstance(value, pd.DataFrame):
            frame = value.copy()
            for column in frame.columns:
                if frame[column].dtype == object:
                    frame[column] = frame[column].map(lambda item: sanitise(item, roots))
            frame.to_csv(path, index=False)
        else:
            text = json.dumps(json_ready(sanitise(value, roots)), indent=1, ensure_ascii=False)
            path.write_text(text + "\n", encoding="utf-8")
        files[name] = sha256(path)
    manifest = {
        "case": entry.name,
        "title": entry.title,
        "breos_version": breos.__version__,
        "git_describe": git_describe(),
        "generated_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
        "runtime_s": round(runtime_s, 1),
        "cheap": entry.cheap,
        "configs": dict(sorted(ctx.configs.items())),
        "weather": dict(sorted(ctx.weather.items())),
        "dependencies": dependency_versions(),
        **({"currency": next(iter(ctx.currencies))} if len(ctx.currencies) == 1 else {}),
        **({"timezone": next(iter(ctx.timezones))} if len(ctx.timezones) == 1 else {}),
        **json_ready(sanitise(output.manifest, roots)),
        "files": files,
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    leaked = [name for name in files if any(root in (target / name).read_text() for root in roots if root)]
    if leaked:
        raise RuntimeError(f"{entry.name}: an absolute local path survived in {', '.join(leaked)}")


def run_case(entry: Case, target: Path, options: argparse.Namespace) -> float:
    with tempfile.TemporaryDirectory(prefix="breos-gallery-") as scratch:
        ctx = Context(Path(scratch), options)
        started = time.perf_counter()
        output = entry.run(ctx)
        runtime_s = time.perf_counter() - started
        write_case(entry, output, target, ctx, runtime_s)
    return runtime_s


# --------------------------------------------------------------------------- check


# Counts and years: exact under --cross-machine too, also where a CSV column stores them as floats.
DISCRETE_FIELDS = frozenset({"payback_year", "battery_replacements", "n_modules", "modules", "count", "year",
                             "projection_years"})  # fmt: skip
# Values this tool sums from rounded App results: one unit in the last place per rounded term.
ROUNDED_TERMS = {"bill_year1": 3, "no_system_bill_year1": 2}


def _number(value: Any) -> decimal.Decimal | None:
    """``value`` (a parsed JSON number or a CSV cell) as the exact decimal it was written as."""
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return None
    try:
        number = decimal.Decimal(value if isinstance(value, str) else json.dumps(value))
    except decimal.InvalidOperation:
        return None
    return number if number.is_finite() else None


def _places(number: decimal.Decimal) -> int:
    exponent = number.as_tuple().exponent
    return max(0, -exponent) if isinstance(exponent, int) else 0


def _field_places(tree: Any, field: str = "", found: dict[str, int] | None = None) -> dict[str, int]:
    """The most decimal places written for each JSON key in ``tree``."""
    found = {} if found is None else found
    if isinstance(tree, dict):
        for key, item in tree.items():
            _field_places(item, key, found)
    elif isinstance(tree, list):
        for item in tree:
            _field_places(item, field, found)
    elif (number := _number(tree)) is not None and not isinstance(tree, str):
        found[field] = max(found.get(field, 0), _places(number))
    return found


def _same_number(stored: decimal.Decimal, fresh: decimal.Decimal, field: str, places: int | None) -> bool:
    """Whether two written numbers agree; ``places`` is the field's precision under --cross-machine."""
    if math.isclose(float(stored), float(fresh), rel_tol=TOLERANCE, abs_tol=TOLERANCE):
        return True
    if not places or field in DISCRETE_FIELDS:
        return False
    return abs(stored - fresh) <= ROUNDED_TERMS.get(field, 1) * decimal.Decimal(10) ** -places


def _differences(stored: Any, fresh: Any, where: str, field: str, places: Mapping[str, int] | None) -> Iterator[str]:
    if isinstance(stored, dict) and isinstance(fresh, dict):
        for key in sorted(stored.keys() | fresh.keys()):
            if key in VOLATILE_KEYS:
                continue
            if key not in stored or key not in fresh:
                yield f"{where}.{key}: present in only one"
                continue
            yield from _differences(stored[key], fresh[key], f"{where}.{key}", key, places)
    elif isinstance(stored, list) and isinstance(fresh, list):
        if len(stored) != len(fresh):
            yield f"{where}: length {len(stored)} != {len(fresh)}"
            return
        for index, (left, right) in enumerate(zip(stored, fresh)):
            yield from _differences(left, right, f"{where}[{index}]", field, places)
    elif isinstance(stored, str) or isinstance(fresh, str) or (left := _number(stored)) is None:
        if stored != fresh:
            yield f"{where}: {stored!r} != {fresh!r}"
    elif (right := _number(fresh)) is None or not _same_number(
        left, right, field, None if places is None else places.get(field)
    ):
        yield f"{where}: {stored!r} != {fresh!r}"


def compare_file(stored: Path, fresh: Path, cross_machine: bool = False) -> list[str]:
    """Differences between a stored result file and a fresh one, at 1e-9 or under ``cross_machine``."""
    if stored.suffix == ".json":
        left = json.loads(stored.read_text(encoding="utf-8"))
        right = json.loads(fresh.read_text(encoding="utf-8"))
        field_places = None
        if cross_machine:
            field_places = _field_places(left)
            for key, value in _field_places(right).items():
                field_places[key] = max(field_places.get(key, 0), value)
        return list(_differences(left, right, stored.name, "", field_places))
    left_frame = pd.read_csv(stored, dtype=str, keep_default_na=False)
    right_frame = pd.read_csv(fresh, dtype=str, keep_default_na=False)
    if list(left_frame.columns) != list(right_frame.columns) or len(left_frame) != len(right_frame):
        return [f"{stored.name}: columns or row count differ"]
    problems: list[str] = []
    for column in left_frame.columns:
        cells = list(zip(left_frame[column], right_frame[column]))
        numbers = [(_number(old), _number(new)) for old, new in cells]
        written = [_places(number) for pair in numbers for number in pair if number is not None]
        column_places = max(written, default=0) if cross_machine else None
        for row, ((old, new), (old_number, new_number)) in enumerate(zip(cells, numbers)):
            if old == new:
                continue
            if (
                old_number is None
                or new_number is None
                or not _same_number(old_number, new_number, column, column_places)
            ):
                problems.append(f"{stored.name}[{row}, {column}]: {old!r} != {new!r}")
                break
    return problems


def check_case(entry: Case, options: argparse.Namespace) -> list[str]:
    stored = RESULTS_DIR / entry.name
    if not (stored / "manifest.json").exists():
        return [f"{entry.name}: no stored results; run the tool without --check first"]
    manifest = json.loads((stored / "manifest.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="breos-gallery-check-") as scratch:
        fresh = Path(scratch) / entry.name
        run_case(entry, fresh, options)
        problems = []
        for name in manifest["files"]:
            problems.extend(
                f"{entry.name}/{line}" for line in compare_file(stored / name, fresh / name, options.cross_machine)
            )
        fresh_manifest = json.loads((fresh / "manifest.json").read_text(encoding="utf-8"))
        if fresh_manifest["configs"] != manifest["configs"]:
            print(f"note: {entry.name}: a config file changed since the results were stored", file=sys.stderr)
    return problems


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("cases", nargs="*", help="Cases to run (default: all; with --check: the cheap ones).")
    parser.add_argument("--check", action="store_true", help="Rerun cases and compare with the stored results.")
    parser.add_argument("--list", action="store_true", help="List the cases and exit.")
    parser.add_argument("--mc-weather", help="Local Open-Meteo history CSV for the Monte Carlo case (no fetch).")
    parser.add_argument(
        "--cross-machine",
        action="store_true",
        help="With --check, accept one unit in the last written decimal place (results stored on another machine).",
    )
    parser.add_argument("--procs", type=int, default=min(8, os.cpu_count() or 1), help="Optimizer worker processes.")
    options = parser.parse_args(argv)

    if options.list:
        for entry in CASES.values():
            print(f"{entry.name:22s} {'cheap' if entry.cheap else '     '}  {entry.title}")
        return 0
    unknown = [name for name in options.cases if name not in CASES]
    if unknown:
        parser.error(f"unknown case(s): {', '.join(unknown)}; see --list")

    git_describe()  # before writing: a rewritten result file would mark the tree dirty
    if options.check:
        names = options.cases or [entry.name for entry in CASES.values() if entry.cheap]
        problems = []
        for name in names:
            started = time.perf_counter()
            found = check_case(CASES[name], options)
            status = "ok" if not found else f"{len(found)} difference(s)"
            print(f"{name}: {status} ({time.perf_counter() - started:.1f} s)")
            problems.extend(found)
        for line in problems[:50]:
            print(f"  {line}")
        return 1 if problems else 0

    for name in options.cases or list(CASES):
        runtime_s = run_case(CASES[name], RESULTS_DIR / name, options)
        size_kb = sum(path.stat().st_size for path in (RESULTS_DIR / name).iterdir()) / 1024
        print(f"{name}: {runtime_s:.1f} s, {size_kb:.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
