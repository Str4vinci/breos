"""Monte Carlo simulation over weather and demand uncertainty.

Each Monte Carlo *run* is a full multi-year projection, like the deterministic
:class:`breos.App`. The difference is that for every projection year the inputs
are resampled:

* an annual weather realization is drawn (with replacement) from a multi-year
  weather file, and
* the demand is scaled by a random multiplier ``~ Normal(1, load_uncertainty)``.

Battery state-of-health carries across years exactly as in the deterministic
pipeline, so degradation compounds over each trajectory. Aggregating many runs
gives the spread of NPV savings, payback year, grid independence, and
end-of-life state-of-health.

BREOS does not bundle weather data: point ``weather_file`` at your own
multi-year historical CSV (see ``configs/examples/montecarlo.toml``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from breos.app_config import ResolvedAppConfig, build_costs_dict, resolve_app_config
from breos.app_inputs import (
    AppRuntimeDependencies,
    build_dc_system_base,
    load_consumption_profile,
)
from breos.battery import BatteryConfig, simulate_energy_balance
from breos.economics import (
    calculate_lcoe_from_projection,
    cost_analysis_projection,
    find_payback_year,
)
from breos.load_profiles import load_profile
from breos.utils import get_hours_per_step
from breos.weather import (
    build_battery_temperature_series,
    fetch_tmy_weather_data,
    load_weather,
    preload_weather_by_year,
    resample_to_15min,
)

# Metrics summarized across runs (column in the per-run frame -> output label).
_SUMMARY_METRICS = {
    "npv_savings_eur": "npv_savings_eur",
    "payback_year": "payback_year",
    "lcoe_eur_kwh": "lcoe_eur_kwh",
    "final_soh_pct": "final_soh_pct",
    "mean_grid_independence_pct": "mean_grid_independence_pct",
    "total_replacements": "total_replacements",
}


@dataclass(frozen=True)
class MonteCarloSettings:
    """Knobs controlling a Monte Carlo study."""

    weather_file: str
    n_runs: int = 100
    years_per_run: int | None = None  # None -> use config projection_years
    load_uncertainty: float = 0.10
    target_year: int = 2025
    seed: int | None = None
    min_load_scale: float = 0.0
    max_load_scale: float | None = None


@dataclass
class MonteCarloResult:
    """Outcome of a Monte Carlo study."""

    runs: pd.DataFrame  # one row per run
    summary: dict[str, dict[str, float]]  # metric -> {mean, std, p5, p50, p95, min, max}
    settings: MonteCarloSettings
    available_years: list[int] = field(default_factory=list)


def _runtime_dependencies() -> AppRuntimeDependencies:
    return AppRuntimeDependencies(
        load_profile=load_profile,
        load_weather=load_weather,
        fetch_tmy_weather_data=fetch_tmy_weather_data,
        resample_to_15min=resample_to_15min,
        build_battery_temperature_series=build_battery_temperature_series,
    )


def _sample_load_scale(
    rng: np.random.Generator, load_uncertainty: float, min_scale: float, max_scale: float | None
) -> float:
    """Draw a demand multiplier ~ Normal(1, load_uncertainty), clipped to bounds."""
    scale = float(rng.normal(1.0, load_uncertainty))
    scale = max(float(min_scale), scale)
    if max_scale is not None:
        scale = min(float(max_scale), scale)
    return scale


def _index_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Turn a ``preload_weather_by_year`` frame (with a ``date`` column) into a
    UTC-indexed weather DataFrame matching the deterministic pipeline."""
    w = df.copy()
    w["date"] = pd.to_datetime(w["date"])
    w = w.set_index("date")
    if w.index.tz is None:
        w.index = w.index.tz_localize("UTC")
    else:
        w.index = w.index.tz_convert("UTC")
    return w


def _precompute_year_caches(
    cfg: dict[str, Any], resolved: ResolvedAppConfig, settings: MonteCarloSettings
) -> tuple[dict[int, pd.Series], dict[int, pd.Series]]:
    """Build per-year undegraded DC production and battery temperature series."""
    freq = cfg["resolution"]
    weather_by_year = preload_weather_by_year(settings.weather_file, target_year=settings.target_year)
    if not weather_by_year:
        raise ValueError(
            f"No complete years found in weather file: {settings.weather_file}. "
            "Provide a multi-year historical CSV with a 'date' column."
        )

    dc_by_year: dict[int, pd.Series] = {}
    temp_by_year: dict[int, pd.Series] = {}
    for year, df in weather_by_year.items():
        weather = _index_weather(df)
        if freq == "15min":
            inferred = pd.infer_freq(weather.index[:10])
            if inferred and "h" in inferred.lower() and "15" not in inferred:
                weather = resample_to_15min(weather, latitude=resolved.lat, longitude=resolved.lon)
        dc_by_year[year] = build_dc_system_base(cfg, resolved, weather)
        temp_by_year[year] = build_battery_temperature_series(
            "weather", index=dc_by_year[year].index, weather_df=weather
        )
    return dc_by_year, temp_by_year


def _simulate_trajectory(
    cfg: dict[str, Any],
    resolved: ResolvedAppConfig,
    base_load: pd.Series,
    dc_by_year: dict[int, pd.Series],
    temp_by_year: dict[int, pd.Series],
    available_years: np.ndarray,
    years_per_run: int,
    settings: MonteCarloSettings,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Run one Monte Carlo trajectory and return its summary metrics."""
    freq = cfg["resolution"]
    hours_per_step = get_hours_per_step(freq)
    degradation_rate = cfg["pv_degradation_rate"]
    battery_kwh = cfg["battery_kwh"]
    battery_wh = battery_kwh * 1000
    has_battery = battery_kwh > 0

    replacement_cost = resolved.cost_params.battery_cost_per_kwh * battery_kwh
    pv_peak_w = cfg["n_modules"] * resolved.avg_module_power_w
    loading_ratio = cfg["inverter_loading_ratio"]
    inverter_ac_capacity_w = pv_peak_w / loading_ratio if loading_ratio and loading_ratio > 0 else None

    cumulative_fec = 0.0
    cumulative_cal_seconds = 0.0
    cumulative_resistance_growth = 0.0
    cumulative_cycle_deg = 0.0
    cumulative_cal_deg = 0.0
    current_soh = 100.0
    total_replacements = 0
    total_replacement_cost = 0.0
    yearly_summaries: list[dict[str, Any]] = []
    first_year_results_df: pd.DataFrame | None = None

    for year_idx in range(years_per_run):
        pv_degradation_factor = (1 - degradation_rate) ** year_idx
        year = int(available_years[rng.integers(len(available_years))])
        dc_power = dc_by_year[year] * pv_degradation_factor
        load_scale = _sample_load_scale(
            rng, settings.load_uncertainty, settings.min_load_scale, settings.max_load_scale
        )
        houseload = base_load * load_scale

        if has_battery:
            batt_kwargs: dict[str, Any] = {}
            if cfg["battery_rte"] is not None:
                one_way = math.sqrt(cfg["battery_rte"])
                batt_kwargs["charge_efficiency"] = one_way
                batt_kwargs["discharge_efficiency"] = one_way
            batt_cfg = BatteryConfig(
                nominal_energy_wh=battery_wh,
                initial_soh=current_soh,
                eol_percentage=cfg["battery_eol_percentage"],
                max_soc=cfg["battery_max_soc"],
                min_soc=cfg["battery_min_soc"],
                dc_coupled=cfg["dc_coupled"],
                inverter_efficiency=cfg["inverter_efficiency"],
                inverter_ac_capacity_w=inverter_ac_capacity_w,
                enable_replacement=True,
                replacement_cost=replacement_cost,
                calendar_model=cfg["calendar_model"],
                **batt_kwargs,
            )
        else:
            batt_cfg = BatteryConfig(
                nominal_energy_wh=0,
                inverter_efficiency=cfg["inverter_efficiency"],
                inverter_ac_capacity_w=inverter_ac_capacity_w,
            )

        results_df, total_pv, _summary_df, year_rep_cost, year_n_rep, degradation_df = simulate_energy_balance(
            pv_dc=dc_power,
            houseload=houseload,
            battery_config=batt_cfg,
            freq=freq,
            temperature_series=temp_by_year[year] if has_battery else None,
            initial_fec=cumulative_fec,
            initial_calendar_seconds=cumulative_cal_seconds,
            initial_resistance_growth=cumulative_resistance_growth,
            initial_cumulative_cycle_deg=cumulative_cycle_deg,
            initial_cumulative_cal_deg=cumulative_cal_deg,
        )

        if first_year_results_df is None:
            first_year_results_df = results_df

        if has_battery and not degradation_df.empty:
            cumulative_fec = degradation_df["Cumulative_FEC"].iloc[-1]
            cumulative_cal_seconds = degradation_df["Cumulative_Calendar_Seconds"].iloc[-1]
            cumulative_cycle_deg = degradation_df["Cumulative_Cycle_Degradation"].iloc[-1]
            cumulative_cal_deg = degradation_df["Cumulative_Calendar_Degradation"].iloc[-1]
            current_soh = degradation_df["SOH"].iloc[-1]
            if "Resistance_Growth" in degradation_df.columns:
                cumulative_resistance_growth = degradation_df["Resistance_Growth"].iloc[-1]

        total_replacements += year_n_rep
        total_replacement_cost += year_rep_cost

        total_pv_kwh = total_pv / 1000
        total_load = (results_df["Houseload"].sum() / 1000) * hours_per_step
        total_import = (results_df["Import_From_Grid"].sum() / 1000) * hours_per_step
        total_export = (results_df["Sell_To_Grid"].sum() / 1000) * hours_per_step
        grid_indep = (1 - total_import / total_load) * 100 if total_load > 0 else 0

        yearly_summaries.append(
            {
                "Year": year_idx + 1,
                "PV_Production_kWh": total_pv_kwh,
                "Load_kWh": total_load,
                "Import_kWh": total_import,
                "Export_kWh": total_export,
                "Grid_Independence_%": grid_indep,
                "Battery_SOH_%": current_soh if has_battery else None,
                "Replacements": year_n_rep,
                "Replacement_Cost": year_rep_cost,
                "PV_Degradation_Factor": pv_degradation_factor,
                "Weather_Year": year,
                "Load_Scale": load_scale,
            }
        )

    yearly_df = pd.DataFrame(yearly_summaries)
    costs = build_costs_dict(cfg, resolved)
    cost_projection = cost_analysis_projection(
        results_df=first_year_results_df,
        costs=costs,
        num_years=years_per_run,
        inflation_rate=cfg["inflation_rate"],
        discount_rate=cfg["discount_rate"],
        freq=freq,
        yearly_summary_df=yearly_df,
        total_replacement_cost=total_replacement_cost,
        emissions_params=resolved.emissions_params,
    )
    lcoe = calculate_lcoe_from_projection(
        cost_projection,
        total_investment=costs["total_initial_cost"],
        discount_rate=cfg["discount_rate"],
    )
    payback_year = find_payback_year(cost_projection)
    npv_savings = float(cost_projection["Savings_Cumulative_NPV"].iloc[-1])

    return {
        "npv_savings_eur": npv_savings,
        "payback_year": payback_year if payback_year is not None else float("nan"),
        "lcoe_eur_kwh": float(lcoe),
        "final_soh_pct": float(current_soh) if has_battery else float("nan"),
        "mean_grid_independence_pct": float(yearly_df["Grid_Independence_%"].mean()),
        "total_replacements": int(total_replacements),
        "total_replacement_cost_eur": float(total_replacement_cost),
        "mean_pv_production_kwh": float(yearly_df["PV_Production_kWh"].mean()),
        "mean_import_kwh": float(yearly_df["Import_kWh"].mean()),
        "mean_export_kwh": float(yearly_df["Export_kWh"].mean()),
    }


def _summarize(runs: pd.DataFrame) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for col in _SUMMARY_METRICS:
        if col not in runs.columns:
            continue
        series = runs[col].dropna()
        if series.empty:
            continue
        summary[col] = {
            "mean": float(series.mean()),
            "std": 0.0 if len(series) == 1 else float(series.std()),
            "p5": float(series.quantile(0.05)),
            "p50": float(series.quantile(0.50)),
            "p95": float(series.quantile(0.95)),
            "min": float(series.min()),
            "max": float(series.max()),
        }
    return summary


def run_montecarlo(config: dict[str, Any], settings: MonteCarloSettings) -> MonteCarloResult:
    """Run a Monte Carlo study over weather years and demand uncertainty.

    Args:
        config: An App configuration dict (same keys as :class:`breos.App`).
        settings: Monte Carlo controls (weather file, runs, uncertainty, seed).

    Returns:
        A :class:`MonteCarloResult` with one row per run and summary statistics.
    """
    resolved = resolve_app_config(config)
    cfg = resolved.cfg
    years_per_run = settings.years_per_run or cfg["projection_years"]

    dc_by_year, temp_by_year = _precompute_year_caches(cfg, resolved, settings)
    available_years = np.array(sorted(dc_by_year.keys()))

    deps = _runtime_dependencies()
    base_load = load_consumption_profile(cfg, deps, timezone=resolved.timezone)

    rows: list[dict[str, Any]] = []
    for run_idx in range(settings.n_runs):
        seed = None if settings.seed is None else settings.seed + run_idx
        rng = np.random.default_rng(seed)
        metrics = _simulate_trajectory(
            cfg,
            resolved,
            base_load,
            dc_by_year,
            temp_by_year,
            available_years,
            years_per_run,
            settings,
            rng,
        )
        metrics = {"run": run_idx + 1, **metrics}
        rows.append(metrics)

    runs_df = pd.DataFrame(rows)
    return MonteCarloResult(
        runs=runs_df,
        summary=_summarize(runs_df),
        settings=settings,
        available_years=[int(y) for y in available_years],
    )
