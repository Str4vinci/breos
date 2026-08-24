"""Monte Carlo simulation over weather and demand uncertainty.

Each Monte Carlo *run* is a full multi-year projection, like the deterministic
:class:`breos.App`. The difference is that for every projection year the inputs
are resampled:

* an annual weather realization is drawn (with replacement) from a multi-year
  weather file, and
* the demand is scaled by a random multiplier from the configured normal or
  bounded-uniform distribution.

Battery state-of-health carries across years exactly as in the deterministic
pipeline, so degradation compounds over each trajectory. Aggregating many runs
gives the spread of NPV savings, payback year, grid independence, and
end-of-life state-of-health.

BREOS does not bundle weather data: point ``weather_file`` at your own
multi-year historical CSV (see ``configs/examples/montecarlo.toml``).
"""

from __future__ import annotations

import math
import platform
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from multiprocessing import Pool
from typing import Any

import numpy as np
import pandas as pd

from breos.app_config import ResolvedAppConfig, build_costs_dict, resolve_app_config
from breos.app_inputs import (
    AppRuntimeDependencies,
    build_dc_system_base,
    load_consumption_profile,
)
from breos.battery import EXECUTION_BACKENDS, BatteryConfig, simulate_energy_balance_summary
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
    "payback_year_exact": "payback_year_exact",
    "lcoe_eur_kwh": "lcoe_eur_kwh",
    "final_soh_pct": "final_soh_pct",
    "mean_grid_independence_pct": "mean_grid_independence_pct",
    "lifetime_grid_independence_pct": "lifetime_grid_independence_pct",
    "total_replacements": "total_replacements",
}


@dataclass(frozen=True)
class MonteCarloSettings:
    """Knobs controlling a Monte Carlo study."""

    weather_file: str
    n_runs: int = 100
    years_per_run: int | None = None  # None -> use config projection_years
    load_uncertainty: float = 0.10
    load_distribution: str = "normal"
    target_year: int = 2025
    weather_start_year: int | None = None
    weather_end_year: int | None = None
    seed: int | None = None
    min_load_scale: float = 0.0
    max_load_scale: float | None = None
    preserve_irradiance_energy: bool = False
    collect_yearly: bool = False
    n_procs: int = 1
    # "python" is the reference implementation and the default. "numba"
    # selects the optional compiled within-day dispatch kernel and requires
    # breos[fast]; it is checked before any trajectory starts.
    execution_backend: str = "python"


@dataclass
class MonteCarloResult:
    """Outcome of a Monte Carlo study."""

    runs: pd.DataFrame  # one row per run
    summary: dict[str, dict[str, float]]  # metric -> descriptive statistics and quantiles
    settings: MonteCarloSettings
    available_years: list[int] = field(default_factory=list)
    yearly: pd.DataFrame | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


def _runtime_dependencies() -> AppRuntimeDependencies:
    return AppRuntimeDependencies(
        load_profile=load_profile,
        load_weather=load_weather,
        fetch_tmy_weather_data=fetch_tmy_weather_data,
        resample_to_15min=resample_to_15min,
        build_battery_temperature_series=build_battery_temperature_series,
    )


def _sample_load_scale(
    rng: np.random.Generator,
    load_uncertainty: float,
    min_scale: float,
    max_scale: float | None,
    distribution: str = "normal",
) -> float:
    """Draw a demand multiplier and enforce the configured physical bounds."""
    if distribution == "normal":
        scale = float(rng.normal(1.0, load_uncertainty))
    elif distribution == "uniform":
        scale = float(rng.uniform(1.0 - load_uncertainty, 1.0 + load_uncertainty))
    else:
        raise ValueError("load_distribution must be 'normal' or 'uniform'")
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
    if settings.weather_start_year is not None:
        weather_by_year = {
            year: frame for year, frame in weather_by_year.items() if year >= settings.weather_start_year
        }
    if settings.weather_end_year is not None:
        weather_by_year = {year: frame for year, frame in weather_by_year.items() if year <= settings.weather_end_year}
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
                weather = resample_to_15min(
                    weather,
                    latitude=resolved.lat,
                    longitude=resolved.lon,
                    preserve_irradiance_energy=settings.preserve_irradiance_energy,
                )
        dc_by_year[year] = build_dc_system_base(cfg, resolved, weather)
        temp_by_year[year] = build_battery_temperature_series(
            cfg["battery_temperature"],
            index=dc_by_year[year].index,
            weather_df=weather,
            indoor_model=cfg["battery_indoor_model"],
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
) -> tuple[dict[str, Any], pd.DataFrame]:
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
    carried_energy_wh: float | None = None
    carried_pv_origin_energy_wh: float | None = None

    for year_idx in range(years_per_run):
        pv_degradation_factor = (1 - degradation_rate) ** year_idx
        year = int(available_years[rng.integers(len(available_years))])
        dc_power = dc_by_year[year] * pv_degradation_factor
        load_scale = _sample_load_scale(
            rng,
            settings.load_uncertainty,
            settings.min_load_scale,
            settings.max_load_scale,
            settings.load_distribution,
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
                max_charge_power_w=cfg["battery_max_charge_power_w"],
                max_discharge_power_w=cfg["battery_max_discharge_power_w"],
                enable_resistance_fade=cfg.get("enable_resistance_fade", False),
                **batt_kwargs,
            )
        else:
            batt_cfg = BatteryConfig(
                nominal_energy_wh=0,
                inverter_efficiency=cfg["inverter_efficiency"],
                inverter_ac_capacity_w=inverter_ac_capacity_w,
            )

        state_kwargs: dict[str, float] = {}
        if carried_energy_wh is not None:
            state_kwargs = {
                "initial_energy_wh": carried_energy_wh,
                "initial_pv_origin_energy_wh": carried_pv_origin_energy_wh or 0.0,
            }

        summary = simulate_energy_balance_summary(
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
            execution_backend=settings.execution_backend,
            **state_kwargs,
        )
        year_rep_cost = summary.total_replacement_cost
        year_n_rep = summary.n_replacements
        totals = summary.column_sums

        if has_battery:
            carried_energy_wh = summary.carried_energy_wh
            carried_pv_origin_energy_wh = summary.carried_pv_origin_energy_wh

        if has_battery and summary.has_degradation_rows:
            cumulative_fec = summary.fec_cum
            cumulative_cal_seconds = summary.cumulative_calendar_seconds
            cumulative_cycle_deg = summary.cumulative_cycle_degradation
            cumulative_cal_deg = summary.cumulative_calendar_degradation
            current_soh = summary.final_soh_percent
            # The detailed path only reported resistance growth when the fade
            # model was enabled; without it the carried value stays put.
            if batt_cfg.enable_resistance_fade:
                cumulative_resistance_growth = summary.resistance_growth

        total_replacements += year_n_rep
        total_replacement_cost += year_rep_cost

        # Each expression keeps the scaling order the detailed path used, so
        # these are the same floats it produced, not merely equivalent ones.
        pv_dc_kwh = float(totals["PV_DC"] * hours_per_step / 1000)
        legacy_pv_kwh = float(totals["PV_Production"] * hours_per_step / 1000)
        direct_pv_ac_kwh = float(totals["PV_AC_To_Load"] * hours_per_step / 1000)
        pv_origin_battery_ac_kwh = float(totals["Battery_AC_To_Load_PV"] * hours_per_step / 1000)
        total_load = (totals["Houseload"] / 1000) * hours_per_step
        total_import = (totals["Import_From_Grid"] / 1000) * hours_per_step
        total_export = (totals["Sell_To_Grid"] / 1000) * hours_per_step
        total_pv_kwh = direct_pv_ac_kwh + pv_origin_battery_ac_kwh + total_export
        grid_indep = (1 - total_import / total_load) * 100 if total_load > 0 else 0

        yearly_summaries.append(
            {
                "Year": year_idx + 1,
                "PV_Production_kWh": total_pv_kwh,
                "Legacy_PV_Production_kWh": legacy_pv_kwh,
                "PV_DC_Generation_kWh": pv_dc_kwh,
                "Direct_PV_AC_Load_kWh": direct_pv_ac_kwh,
                "PV_Origin_Battery_AC_Load_kWh": pv_origin_battery_ac_kwh,
                "Self_Consumption_kWh": direct_pv_ac_kwh + pv_origin_battery_ac_kwh,
                "Curtailment_DC_kWh": float(totals["PV_DC_Curtailed"] * hours_per_step / 1000),
                "Load_kWh": total_load,
                "Import_kWh": total_import,
                "Export_kWh": total_export,
                "Grid_Independence_%": grid_indep,
                "Battery_SOH_%": current_soh if has_battery else None,
                "Battery_Cumulative_FEC": cumulative_fec,
                "Battery_Cumulative_Calendar_Seconds": cumulative_cal_seconds,
                "Battery_Cumulative_Cycle_Degradation": cumulative_cycle_deg,
                "Battery_Cumulative_Calendar_Degradation": cumulative_cal_deg,
                "Battery_Resistance_Growth": cumulative_resistance_growth,
                "Replacements": year_n_rep,
                "Replacement_Cost": year_rep_cost,
                "PV_Degradation_Factor": pv_degradation_factor,
                "Weather_Year": year,
                "Load_Scale": load_scale,
                # Diagnostics. These are reductions of ledger columns the
                # detailed frame already exposed; they are reported here so a
                # Monte Carlo run can be compared field by field against
                # another execution path without rerunning it.
                "PV_Direct_Inverter_Loss_kWh": float(totals["PV_Direct_Inverter_Loss"] * hours_per_step / 1000),
                "Battery_Inverter_Loss_kWh": float(totals["Battery_Inverter_Loss"] * hours_per_step / 1000),
                "Battery_Charge_Input_kWh": float(totals["Battery_Charge_Input"] * hours_per_step / 1000),
                "Battery_Discharge_DC_kWh": float(totals["Battery_Discharge_DC"] * hours_per_step / 1000),
                "Battery_AC_To_Load_kWh": float(totals["Battery_AC_To_Load"] * hours_per_step / 1000),
                "Battery_Charge_Loss_kWh": float(totals["Battery_Charge_Loss"] * hours_per_step / 1000),
                "Battery_Discharge_Loss_kWh": float(totals["Battery_Discharge_Loss"] * hours_per_step / 1000),
                "Battery_Standby_Loss_kWh": float(totals["Battery_Standby_Loss"] * hours_per_step / 1000),
                "Capacity_Window_Loss_kWh": float(totals["Capacity_Window_Loss"] * hours_per_step / 1000),
                "Replacement_Energy_Removed_kWh": float(
                    totals["Battery_Replacement_Energy_Removed"] * hours_per_step / 1000
                ),
                "Replacement_Energy_Added_kWh": float(
                    totals["Battery_Replacement_Energy_Added"] * hours_per_step / 1000
                ),
                "Battery_Carried_Energy_Wh": float(carried_energy_wh) if has_battery else None,
                "Battery_Carried_PV_Origin_Energy_Wh": (float(carried_pv_origin_energy_wh) if has_battery else None),
                # Within-year replacement timing, as timestep indices.
                "Replacement_Steps": ";".join(str(step) for step in summary.replacement_steps),
            }
        )

    yearly_df = pd.DataFrame(yearly_summaries)
    costs = build_costs_dict(cfg, resolved)
    cost_projection = cost_analysis_projection(
        # The projection is built entirely from yearly_summary_df below; the
        # per-timestep frame is only consulted by the legacy first-year
        # estimation path, which a Monte Carlo run never takes.
        results_df=None,
        costs=costs,
        num_years=years_per_run,
        inflation_rate=cfg["inflation_rate"],
        sell_price_inflation=cfg["sell_price_inflation"],
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
    payback_year_exact = _interpolate_payback_year(cost_projection)
    npv_savings = float(cost_projection["Savings_Cumulative_NPV"].iloc[-1])

    trajectory = yearly_df.merge(cost_projection, on="Year", how="left", suffixes=("", "_Financial"))
    lifetime_load = float(yearly_df["Load_kWh"].sum())
    lifetime_import = float(yearly_df["Import_kWh"].sum())
    lifetime_gi = 100.0 * (1.0 - lifetime_import / lifetime_load) if lifetime_load > 0.0 else 0.0

    metrics = {
        "npv_savings_eur": npv_savings,
        "payback_year": payback_year if payback_year is not None else float("nan"),
        "payback_year_exact": payback_year_exact if payback_year_exact is not None else float("nan"),
        "lcoe_eur_kwh": float(lcoe),
        "final_soh_pct": float(current_soh) if has_battery else float("nan"),
        "mean_grid_independence_pct": float(yearly_df["Grid_Independence_%"].mean()),
        "lifetime_grid_independence_pct": lifetime_gi,
        "total_replacements": int(total_replacements),
        "total_replacement_cost_eur": float(total_replacement_cost),
        "mean_pv_production_kwh": float(yearly_df["Legacy_PV_Production_kWh"].mean()),
        "mean_pv_dc_generation_kwh": float(yearly_df["PV_DC_Generation_kWh"].mean()),
        "mean_direct_pv_ac_load_kwh": float(yearly_df["Direct_PV_AC_Load_kWh"].mean()),
        "mean_pv_origin_battery_ac_load_kwh": float(yearly_df["PV_Origin_Battery_AC_Load_kWh"].mean()),
        "mean_self_consumption_kwh": float(yearly_df["Self_Consumption_kWh"].mean()),
        "mean_usable_ac_system_production_kwh": float(yearly_df["PV_Production_kWh"].mean()),
        "mean_import_kwh": float(yearly_df["Import_kWh"].mean()),
        "mean_export_kwh": float(yearly_df["Export_kWh"].mean()),
    }
    return metrics, trajectory


def _interpolate_payback_year(cost_projection: pd.DataFrame) -> float | None:
    """Return the linearly interpolated discounted-payback year."""
    savings = cost_projection["Savings_Cumulative_NPV"].to_numpy(dtype=float)
    years = cost_projection["Year"].to_numpy(dtype=float)
    if len(savings) == 0:
        return None
    if savings[0] >= 0.0:
        return float(years[0])
    for idx in range(1, len(savings)):
        if savings[idx] >= 0.0 and savings[idx - 1] < 0.0:
            change = savings[idx] - savings[idx - 1]
            return float(years[idx]) if abs(change) < 1e-12 else float(years[idx - 1] - savings[idx - 1] / change)
    return None


def _resolve_backend(execution_backend: str) -> dict[str, Any]:
    """Check the backend is usable and record its toolchain.

    A bit-identity claim is only meaningful against a stated toolchain, and it
    cannot be checked after the fact without one, so the compiler versions and
    runtime versions are recorded for every run rather than only for
    benchmarks. Workers report the JIT cache outcome after they call the
    compiled dispatcher.
    """
    provenance: dict[str, Any] = {
        "execution_backend": execution_backend,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    if execution_backend == "numba":
        from breos._numba_dispatch import numba_versions, require_numba_dispatch_day

        require_numba_dispatch_day()
        provenance.update(numba_versions())
    return provenance


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
            "p2_5": float(series.quantile(0.025)),
            "p50": float(series.quantile(0.50)),
            "p95": float(series.quantile(0.95)),
            "p97_5": float(series.quantile(0.975)),
            "min": float(series.min()),
            "max": float(series.max()),
        }
    return summary


_WORKER_CONTEXT: tuple[Any, ...] | None = None


def _initialize_worker(*context: Any) -> None:
    """Install read-only trajectory inputs once per worker process."""
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = context


def _run_trajectory_index(run_idx: int) -> tuple[int, dict[str, Any], pd.DataFrame, str | None]:
    """Evaluate one deterministic per-run random stream in a worker."""
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Monte Carlo worker context was not initialized")
    cfg, resolved, base_load, dc_by_year, temp_by_year, available_years, years_per_run, settings = _WORKER_CONTEXT
    if settings.execution_backend == "numba":
        from breos._numba_dispatch import reset_jit_cache_observation

        reset_jit_cache_observation()
    seed = None if settings.seed is None else settings.seed + run_idx
    rng = np.random.default_rng(seed)
    metrics, trajectory = _simulate_trajectory(
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
    jit_cache_state = None
    if settings.execution_backend == "numba":
        from breos._numba_dispatch import observed_jit_cache_state

        # None means no compiled dispatch call was observed in this worker --
        # a trajectory can legitimately never enter the kernel. Record that as
        # "unknown" rather than aborting: the trajectory's results are valid
        # either way, and provenance that admits it could not tell is more
        # useful than no results at all.
        jit_cache_state = observed_jit_cache_state() or "unknown"
    return run_idx, metrics, trajectory, jit_cache_state


def _aggregate_jit_cache_states(states: list[str]) -> str:
    """Summarise the workers' JIT cache observations for provenance.

    ``cold`` wins over ``warm`` because one worker compiling means the study
    paid for a compile. ``unknown`` wins over both: if any worker could not
    tell, the study-level claim cannot be trusted either. An empty list is
    ``unknown`` as well -- it means nothing was observed, not that the cache
    was warm. This never raises; provenance bookkeeping must not be able to
    fail a completed study.
    """
    if not states or any(state not in {"warm", "cold"} for state in states):
        return "unknown"
    return "cold" if "cold" in states else "warm"


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
    if settings.n_runs < 1:
        raise ValueError("n_runs must be at least 1")
    if settings.years_per_run is not None and settings.years_per_run < 1:
        raise ValueError("years_per_run must be at least 1")
    if settings.load_uncertainty < 0.0:
        raise ValueError("load_uncertainty must be non-negative")
    if settings.load_distribution not in {"normal", "uniform"}:
        raise ValueError("load_distribution must be 'normal' or 'uniform'")
    if settings.n_procs < 1:
        raise ValueError("n_procs must be at least 1")
    if settings.execution_backend not in EXECUTION_BACKENDS:
        raise ValueError(f"execution_backend must be one of {EXECUTION_BACKENDS}, got {settings.execution_backend!r}")
    if (
        settings.weather_start_year is not None
        and settings.weather_end_year is not None
        and settings.weather_start_year > settings.weather_end_year
    ):
        raise ValueError("weather_start_year must not be later than weather_end_year")
    if cfg["degradation_engine"] == "blast":
        raise ValueError("degradation_engine='blast' is not supported with Monte Carlo yet")
    if cfg["horizon_profile"] is not None:
        raise ValueError(
            "'horizon_profile' is not supported with Monte Carlo weather files yet because their "
            "terrain-horizon provenance is unknown"
        )
    years_per_run = settings.years_per_run or cfg["projection_years"]

    # Resolve the dispatch backend before any input is loaded, so a missing
    # optional dependency stops a 10,000-trajectory study immediately rather
    # than hours into it.
    backend_provenance = _resolve_backend(settings.execution_backend)

    dc_by_year, temp_by_year = _precompute_year_caches(cfg, resolved, settings)
    available_years = np.array(sorted(dc_by_year.keys()))

    deps = _runtime_dependencies()
    base_load = load_consumption_profile(cfg, deps, timezone=resolved.timezone)

    context = (
        cfg,
        resolved,
        base_load,
        dc_by_year,
        temp_by_year,
        available_years,
        years_per_run,
        settings,
    )
    if settings.n_procs == 1:
        _initialize_worker(*context)
        outputs = [_run_trajectory_index(run_idx) for run_idx in range(settings.n_runs)]
    else:
        with Pool(settings.n_procs, initializer=_initialize_worker, initargs=context) as pool:
            outputs = pool.map(_run_trajectory_index, range(settings.n_runs))

    rows: list[dict[str, Any]] = []
    yearly_frames: list[pd.DataFrame] = []
    jit_cache_states: list[str] = []
    for run_idx, metrics, trajectory, jit_cache_state in outputs:
        rows.append({"run": run_idx + 1, **metrics})
        if jit_cache_state is not None:
            jit_cache_states.append(jit_cache_state)
        if settings.collect_yearly:
            trajectory.insert(0, "run", run_idx + 1)
            yearly_frames.append(trajectory)

    if settings.execution_backend == "numba":
        backend_provenance["jit_cache"] = _aggregate_jit_cache_states(jit_cache_states)

    runs_df = pd.DataFrame(rows)
    yearly_df = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else None
    try:
        breos_version = version("breos")
    except PackageNotFoundError:
        breos_version = "unknown"
    return MonteCarloResult(
        runs=runs_df,
        summary=_summarize(runs_df),
        settings=settings,
        available_years=[int(y) for y in available_years],
        yearly=yearly_df,
        provenance={
            "breos_version": breos_version,
            "resolved_config": cfg,
            "settings": asdict(settings),
            "available_weather_years": [int(y) for y in available_years],
            "random_stream": "numpy.default_rng(base_seed + zero_based_run_index)",
            "execution": backend_provenance,
        },
    )
