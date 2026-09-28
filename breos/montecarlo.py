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
import multiprocessing
from dataclasses import asdict, dataclass, field, replace
from importlib.metadata import PackageNotFoundError, version
from multiprocessing import Pool
from typing import Any, cast

import numpy as np
import pandas as pd

from breos.app_config import DEFAULTS, ResolvedAppConfig, resolve_app_config
from breos.app_inputs import (
    AppRuntimeDependencies,
    build_dc_system_base,
    load_consumption_profile,
)
from breos.battery import LEDGER_SCHEMA_VERSION, AlignedSimulationInputs, align_simulation_inputs
from breos.dispatch_instructions import DispatchInstructions
from breos.economics import find_payback_year, find_payback_year_exact
from breos.execution import (
    aggregate_jit_cache_states,
    is_pv_only_dispatch,
    observed_jit_cache_state,
    reset_jit_cache_observation,
    validate_execution_backend,
)
from breos.execution import (
    backend_provenance as _backend_provenance,
)
from breos.load_profiles import LOAD_PROFILE_METADATA_KEY, load_profile
from breos.projection import ProjectionYear, build_pv_only_battery_config, run_projection, value_projection
from breos.pv.model_options import DEFAULT_SOLAR_POSITION, resolve_solar_position_method, solar_position_time_offset
from breos.smart_charging import resolve_instructions, smart_charging_provenance
from breos.tariffs import ResolvedTariff, tariff_provenance
from breos.weather import (
    build_battery_temperature_series,
    fetch_tmy_weather_data,
    load_weather,
    preload_weather_by_year,
    resample_to_15min,
    weather_metadata,
    weather_representative_time_offset,
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
# A run without a payback year did not pay back within the horizon.
_PAYBACK_METRICS = ("payback_year", "payback_year_exact")


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
    # breos[fast]; it is checked before any trajectory starts. None inherits
    # the App config's top-level ``execution_backend``, which itself defaults
    # to "python"; see :func:`run_montecarlo`.
    execution_backend: str | None = None


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
    cfg: dict[str, Any],
    resolved: ResolvedAppConfig,
    settings: MonteCarloSettings,
    *,
    runtime_weather: dict[str, Any] | None = None,
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
        input_frequency = pd.infer_freq(weather.index[:10]) if len(weather.index) >= 3 else None
        if input_frequency is None and len(weather.index) >= 2:
            input_frequency = pd.tseries.frequencies.to_offset(weather.index[1] - weather.index[0]).freqstr
        if freq == "15min":
            if input_frequency and "h" in input_frequency.lower() and "15" not in input_frequency:
                weather = resample_to_15min(
                    weather,
                    latitude=resolved.lat,
                    longitude=resolved.lon,
                    preserve_irradiance_energy=settings.preserve_irradiance_energy,
                )
        if runtime_weather is not None and not runtime_weather:
            # The same resolution the PV model applies, so a spelling such as
            # "Mid-Interval" is recorded with the offset it actually gets.
            method = resolve_solar_position_method(cfg.get("solar_position", DEFAULT_SOLAR_POSITION))
            offset = solar_position_time_offset(method, weather, freq)
            runtime_weather.update(
                {
                    "representative_source_year": int(year),
                    "input_resolution": input_frequency,
                    "output_resolution": str(freq),
                    "solar_position_method": method,
                    "solar_position_offset_minutes": offset.total_seconds() / 60.0,
                    "metadata": weather_metadata(weather),
                }
            )
        dc_by_year[year] = build_dc_system_base(cfg, resolved, weather)
        temp_by_year[year] = build_battery_temperature_series(
            cfg["battery_temperature"],
            index=dc_by_year[year].index,
            weather_df=weather,
            indoor_model=cfg["battery_indoor_model"],
        )
    return dc_by_year, temp_by_year


def _align_years(
    cfg: dict[str, Any],
    base_load: pd.DataFrame,
    dc_by_year: dict[int, pd.Series],
    temp_by_year: dict[int, pd.Series],
    *,
    has_battery: bool,
) -> dict[int, AlignedSimulationInputs]:
    """Align each weather year's inputs once for the whole study.

    Every trajectory simulates the same nineteen-odd weather years against
    the same load profile, differing only by two scalars: a PV degradation
    factor set by the project year and a sampled load multiplier. The
    alignment behind that -- building the calendar, reindexing three series
    onto it and year-shifting the load profile onto the weather year -- does
    not depend on either, so a study was deriving one of nineteen answers
    once per simulated year of every trajectory.

    Aligning here rather than in the worker matters twice: the work happens
    once for the study instead of once per worker, and on a forking platform
    the arrays are shared with the workers rather than copied into each.
    """
    freq = cfg["resolution"]
    return {
        year: align_simulation_inputs(
            dc_power,
            base_load,
            temp_by_year[year] if has_battery else None,
            freq=freq,
        )
        for year, dc_power in dc_by_year.items()
    }


# A PV-only study memoizes the DC-to-AC conversion for every distinct
# (weather year, project year) pair. That is bounded work, but it is not
# bounded memory: four arrays per pair (the PV input and three conversion
# outputs), at eight bytes per timestep. The Article's 19 weather years over a
# 20-year project at 15-minute resolution come to about 407 MiB. Past this
# budget the study runs without the cache
# rather than exhausting the machine -- same numbers, less speed.
_PV_CHAIN_CACHE_MAX_BYTES = 1 << 30

# Below this many trajectories per distinct weather year there is not enough
# reuse to pay for building the cache at all.
_PV_CHAIN_CACHE_MIN_REUSE = 4


def _pv_chain_cache_is_worthwhile(
    n_runs: int,
    n_years: int,
    years_per_run: int,
    n_steps: int,
    n_procs: int,
) -> bool:
    """Decide whether memoizing the PV chain pays for this study.

    Three things have to hold. The study has to reuse each pair often enough
    to earn the build; the cache has to fit the budget; and the workers have
    to be able to share it. That last one is why the start method matters:
    ``fork`` hands a worker the parent's pages copy-on-write, so one cache
    serves every worker, while a start method that pickles the pool's
    ``initargs`` would send a full copy to each one and turn a saving into a
    per-worker cost.
    """
    if n_runs < _PV_CHAIN_CACHE_MIN_REUSE * n_years:
        return False
    if 4 * n_years * years_per_run * n_steps * 8 > _PV_CHAIN_CACHE_MAX_BYTES:
        return False
    return n_procs == 1 or multiprocessing.get_start_method() == "fork"


def _prepare_pv_chains(
    cfg: dict[str, Any],
    resolved: ResolvedAppConfig,
    aligned_by_year: dict[int, AlignedSimulationInputs],
    settings: MonteCarloSettings,
    years_per_run: int,
) -> dict[tuple[int, int], AlignedSimulationInputs] | None:
    """Memoize the PV-only DC-to-AC conversion per weather and project year.

    A PV-only trajectory converts DC to AC before it looks at the load, so
    the conversion depends only on which weather year was drawn and which
    project year it is being aged to -- a few hundred distinct answers that a
    ten-thousand-trajectory study was recomputing two hundred thousand times.

    Returns ``None`` when the study is not shaped to benefit, in which case
    every run computes its own conversion exactly as before.
    """
    if _has_battery(cfg):
        return None
    any_year = next(iter(aligned_by_year.values()), None)
    if any_year is None:
        return None
    if not _pv_chain_cache_is_worthwhile(
        settings.n_runs,
        len(aligned_by_year),
        years_per_run,
        len(any_year.index),
        settings.n_procs,
    ):
        return None

    batt_cfg = build_pv_only_battery_config(cfg, resolved)
    degradation_rate = cfg["pv_degradation_rate"]
    freq = cfg["resolution"]
    return {
        (year, year_idx): aligned.scaled(pv_factor=(1 - degradation_rate) ** year_idx).with_pv_only_chain(
            batt_cfg, freq=freq
        )
        for year, aligned in aligned_by_year.items()
        for year_idx in range(years_per_run)
    }


def _simulate_trajectory(
    cfg: dict[str, Any],
    resolved: ResolvedAppConfig,
    available_years: np.ndarray,
    years_per_run: int,
    settings: MonteCarloSettings,
    rng: np.random.Generator,
    aligned_by_year: dict[int, AlignedSimulationInputs],
    pv_chains: dict[tuple[int, int], AlignedSimulationInputs] | None,
    tariff: ResolvedTariff | None = None,
    instructions: DispatchInstructions | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run one Monte Carlo trajectory and return its summary metrics."""
    degradation_rate = cfg["pv_degradation_rate"]
    has_battery = _has_battery(cfg)

    def year_inputs(year_idx: int) -> ProjectionYear:
        pv_degradation_factor = (1 - degradation_rate) ** year_idx
        year = int(available_years[rng.integers(len(available_years))])
        load_scale = _sample_load_scale(
            rng,
            settings.load_uncertainty,
            settings.min_load_scale,
            settings.max_load_scale,
            settings.load_distribution,
        )
        # Scaling after alignment rather than before it: elementwise the same
        # product, because reindexing only selects elements and zero-fills.
        # With a PV chain memoized for this pair the PV side is already
        # scaled, and scaling the load does not invalidate the chain.
        if pv_chains is None:
            aligned = aligned_by_year[year].scaled(pv_factor=pv_degradation_factor, load_factor=load_scale)
        else:
            aligned = pv_chains[(year, year_idx)].scaled(load_factor=load_scale)
        return ProjectionYear(
            pv_degradation_factor=pv_degradation_factor,
            aligned=aligned,
            extra={"Weather_Year": year, "Load_Scale": load_scale},
        )

    projection = run_projection(
        cfg,
        resolved,
        years_per_run,
        year_inputs,
        has_battery=has_battery,
        execution_backend=cast(str, settings.execution_backend),
        tariff=tariff,
        instructions=instructions,
    )
    current_soh = projection.carry.soh_pct
    total_replacements = projection.total_replacements
    total_replacement_cost = projection.total_replacement_cost
    value = value_projection(cfg, resolved, projection)
    cost_projection, lcoe, yearly_df = value.cost_projection, value.lcoe, value.yearly_df
    payback_year = find_payback_year(cost_projection)
    payback_year_exact = find_payback_year_exact(cost_projection)
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


def _has_battery(cfg: dict[str, Any]) -> bool:
    """Ask the dispatch's own question of a study config.

    A study asks this in four places -- the chain cache, alignment, the
    trajectory loop and the provenance record -- and a study whose answer
    differed between them would cache one shape of work and report another.
    """
    return not is_pv_only_dispatch(
        cfg["battery_kwh"] * 1000,
        cfg.get("battery_max_soc", DEFAULTS["battery_max_soc"]),
        cfg.get("battery_min_soc", DEFAULTS["battery_min_soc"]),
    )


def _resolve_backend(execution_backend: str, *, pv_only: bool = False) -> dict[str, Any]:
    """Check the backend is usable and record its toolchain.

    Thin wrapper over :func:`breos.execution.backend_provenance` so that App
    and Monte Carlo record the same keys from the same code.
    """
    return _backend_provenance(execution_backend, pv_only=pv_only)


def _summarize(runs: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Describe each metric across runs.

    Each entry gives ``count``, the number of runs its statistics cover, out
    of ``n_runs``. Runs without a value are left out of the statistics. For
    the payback metrics, such a run did not pay back within the horizon, so
    their statistics are conditional on payback. ``payback_probability`` is
    the unconditional share of runs that paid back. A payback entry is kept
    with only these counts when no run paid back.
    """
    summary: dict[str, dict[str, float]] = {}
    n_runs = len(runs)
    for col in _SUMMARY_METRICS:
        if col not in runs.columns:
            continue
        # A non-finite value (an infinite LCOE with no production) is no
        # value, like NaN: it would make the mean infinite and the spread NaN.
        values = pd.to_numeric(runs[col], errors="coerce")
        series = values[np.isfinite(values)]
        is_payback = col in _PAYBACK_METRICS
        if series.empty and not is_payback:
            continue
        entry: dict[str, float] = {}
        if not series.empty:
            entry = {
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
        entry["count"] = len(series)
        entry["n_runs"] = n_runs
        if is_payback:
            entry["payback_probability"] = len(series) / n_runs
        summary[col] = entry
    return summary


_WORKER_CONTEXT: tuple[Any, ...] | None = None


def _resolve_study_tariff(
    resolved: ResolvedAppConfig, aligned_by_year: dict[int, AlignedSimulationInputs]
) -> ResolvedTariff | None:
    """Resolve the configured tariff once for the whole study.

    Every sampled weather year is restamped to the target year, so all of them
    share one calendar and one resolved tariff (ADR 0002 A2).
    """
    if resolved.tariff is None:
        return None
    calendars = [inputs.index for inputs in aligned_by_year.values()]
    reference = calendars[0]
    if any(not calendar.equals(reference) for calendar in calendars[1:]):
        raise ValueError("Monte Carlo weather years do not share one calendar, so one tariff cannot price them")
    return resolved.tariff.resolve(reference, resolved.timezone)


def _initialize_worker(*context: Any) -> None:
    """Install read-only trajectory inputs once per worker process."""
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = context


def _run_trajectory_index(run_idx: int) -> tuple[int, dict[str, Any], pd.DataFrame | None, str | None]:
    """Evaluate one deterministic per-run random stream in a worker."""
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Monte Carlo worker context was not initialized")
    (
        cfg,
        resolved,
        available_years,
        years_per_run,
        settings,
        aligned_by_year,
        pv_chains,
        tariff,
        instructions,
    ) = _WORKER_CONTEXT
    # One observation window per trajectory: that is the unit of work whose
    # compile cost is being attributed. A no-op on the Python backend.
    reset_jit_cache_observation(settings.execution_backend)
    # Each run takes its own child of the base seed's SeedSequence, the same
    # stream as SeedSequence(seed).spawn(n_runs)[run_idx], so studies under
    # different base seeds share no trajectory. Without a seed, every run
    # draws fresh entropy.
    rng = np.random.default_rng(np.random.SeedSequence(settings.seed, spawn_key=(run_idx,)))
    metrics, trajectory = _simulate_trajectory(
        cfg,
        resolved,
        available_years,
        years_per_run,
        settings,
        rng,
        aligned_by_year,
        pv_chains,
        tariff,
        instructions,
    )
    # None from a numba run means no compiled dispatch call was observed -- a
    # trajectory can legitimately never enter the kernel. Record that as
    # "unknown" rather than aborting: the trajectory's results are valid either
    # way, and provenance that admits it could not tell is more useful than no
    # results at all. On the Python backend there is nothing to report.
    jit_cache_state = None
    if settings.execution_backend == "numba":
        jit_cache_state = observed_jit_cache_state(settings.execution_backend) or "unknown"
    return run_idx, metrics, trajectory if settings.collect_yearly else None, jit_cache_state


def _aggregate_jit_cache_states(states: list[str]) -> str:
    """Summarise the workers' JIT cache observations for provenance.

    Thin wrapper over :func:`breos.execution.aggregate_jit_cache_states`.
    """
    return aggregate_jit_cache_states(states)


def run_montecarlo(config: dict[str, Any], settings: MonteCarloSettings) -> MonteCarloResult:
    """Run a Monte Carlo study over weather years and demand uncertainty.

    Args:
        config: An App configuration dict (same keys as :class:`breos.App`).
        settings: Monte Carlo controls (weather file, runs, uncertainty, seed).

    The dispatch backend is ``settings.execution_backend`` when set, else the
    config's top-level ``execution_backend``, else ``"python"``. The CLI
    applies the same order after its own ``--execution-backend`` flag and
    ``[montecarlo].execution_backend``, so a study selects the same backend
    from Python and from ``breos montecarlo``. The returned result's
    ``settings`` records the backend that ran.

    Returns:
        A :class:`MonteCarloResult` with one row per run and summary statistics.
    """
    resolved = resolve_app_config(config)
    cfg = resolved.cfg
    if settings.execution_backend is None:
        settings = replace(settings, execution_backend=cfg["execution_backend"])
    if settings.n_runs < 1:
        raise ValueError("n_runs must be at least 1")
    if settings.years_per_run is not None and settings.years_per_run < 1:
        raise ValueError("years_per_run must be at least 1")
    if not (math.isfinite(settings.load_uncertainty) and settings.load_uncertainty >= 0.0):
        raise ValueError("load_uncertainty must be a finite, non-negative number")
    # The load scale multiplies demand, so a negative bound would make demand negative.
    if not (math.isfinite(settings.min_load_scale) and settings.min_load_scale >= 0.0):
        raise ValueError(f"min_load_scale must be a finite, non-negative number, got {settings.min_load_scale}")
    # None is the way to leave demand unbounded above. An infinite bound would
    # say the same thing, but it cannot be written to strict JSON provenance.
    if settings.max_load_scale is not None and not math.isfinite(settings.max_load_scale):
        raise ValueError(
            "max_load_scale must be a finite number, or None to leave the load scale unbounded, "
            f"got {settings.max_load_scale}"
        )
    if settings.max_load_scale is not None and not settings.max_load_scale >= settings.min_load_scale:
        raise ValueError(
            f"max_load_scale must be at least min_load_scale ({settings.min_load_scale}), got {settings.max_load_scale}"
        )
    if settings.load_distribution not in {"normal", "uniform"}:
        raise ValueError("load_distribution must be 'normal' or 'uniform'")
    if settings.n_procs < 1:
        raise ValueError("n_procs must be at least 1")
    validate_execution_backend(settings.execution_backend)
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
    has_battery = _has_battery(cfg)
    backend_provenance = _resolve_backend(settings.execution_backend, pv_only=not has_battery)

    runtime_weather: dict[str, Any] = {}
    dc_by_year, temp_by_year = _precompute_year_caches(
        cfg,
        resolved,
        settings,
        runtime_weather=runtime_weather,
    )
    available_years = np.array(sorted(dc_by_year.keys()))

    deps = _runtime_dependencies()
    base_load = load_consumption_profile(cfg, deps, timezone=resolved.timezone)
    aligned_by_year = _align_years(
        cfg,
        base_load,
        dc_by_year,
        temp_by_year,
        has_battery=has_battery,
    )
    pv_chains = _prepare_pv_chains(cfg, resolved, aligned_by_year, settings, years_per_run)
    tariff = _resolve_study_tariff(resolved, aligned_by_year)
    # Every sampled weather year shares the tariff's calendar, so one set of
    # smart-charging instructions serves every trajectory and year.
    spec = resolved.smart_charging
    instructions = resolve_instructions(spec, tariff) if spec is not None and has_battery else None

    # The weather frames and the raw load profile are not passed to the
    # workers: alignment consumed them here, and a worker only ever reads the
    # aligned arrays.
    context = (
        cfg,
        resolved,
        available_years,
        years_per_run,
        settings,
        aligned_by_year,
        pv_chains,
        tariff,
        instructions,
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
        if trajectory is not None:
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
            "runtime_weather": runtime_weather,
            "load_profile": dict(base_load.attrs.get(LOAD_PROFILE_METADATA_KEY, {})),
            "random_stream": (
                "numpy.random.default_rng(numpy.random.SeedSequence(base_seed).spawn(n_runs)[zero_based_run_index])"
            ),
            "execution": backend_provenance,
            "ledger_schema_version": LEDGER_SCHEMA_VERSION,
            **({"tariff": tariff_provenance(tariff, calendar_year=settings.target_year)} if tariff is not None else {}),
            **(
                {"smart_charging": smart_charging_provenance(spec, instructions, tariff)}
                if spec is not None and instructions is not None and tariff is not None
                else {}
            ),
        },
    )
