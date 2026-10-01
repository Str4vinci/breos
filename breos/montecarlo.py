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

import json
import math
import multiprocessing
import os
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from multiprocessing import Pool
from typing import Any, cast

import numpy as np
import pandas as pd

from breos.app import App
from breos.app_config import ResolvedAppConfig, resolve_app_config
from breos.app_inputs import (
    INPUT_INDEPENDENT_KEYS,
    build_dc_system_base,
    config_cache_key,
    load_consumption_profile,
    resample_hourly_weather,
    weather_input_frequency,
)
from breos.battery import LEDGER_SCHEMA_VERSION, AlignedSimulationInputs, align_simulation_inputs
from breos.dispatch_instructions import DispatchInstructions
from breos.economics import find_payback_year, find_payback_year_interpolated, projection_rates_record
from breos.execution import (
    aggregate_jit_cache_states,
    backend_provenance,
    config_has_battery,
    observed_jit_cache_state,
    reset_jit_cache_observation,
    validate_execution_backend,
)
from breos.load_profiles import LOAD_PROFILE_METADATA_KEY
from breos.projection import (
    ProjectionYear,
    build_pv_only_battery_config,
    effective_reference_escalation,
    run_projection,
    value_projection,
)
from breos.pv.model_options import DEFAULT_SOLAR_POSITION, resolve_solar_position_method, solar_position_time_offset
from breos.result_schema import RESULT_SCHEMA_VERSION
from breos.smart_charging import PLANNER_MODES, resolve_instructions, smart_charging_provenance
from breos.tariffs import ResolvedTariff, reference_tariff_provenance, result_currency, tariff_provenance
from breos.utils import package_version
from breos.weather import (
    _weather_file_sha256,
    _weather_metadata_sidecar_path,
    build_battery_temperature_series,
    fill_leap_day,
    preload_weather_by_year,
    resample_to_15min,
    weather_metadata,
)

# Per-run columns summarized across runs, under the same names.
_SUMMARY_METRICS = (
    "npv_savings",
    "terminal_health_credit",
    "terminal_health_credit_npv",
    "npv_savings_terminal_adjusted",
    "payback_year",
    "payback_year_interpolated",
    "lcoe_per_kwh",
    "final_soh_pct",
    "mean_grid_independence_pct",
    "lifetime_grid_independence_pct",
    "total_replacements",
)
# A run without a payback year did not pay back within the horizon.
_PAYBACK_METRICS = ("payback_year", "payback_year_interpolated")


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
    UTC-indexed weather DataFrame matching the deterministic pipeline.

    The years are read without 29 February, so a leap target year gets a copy
    of 28 February, as an App run of that year does. The day is filled on the
    file's own clock, where it was dropped: in UTC, a file written at +01:00
    would still have an hour dated 29 February, and the fill would be skipped.
    """
    w = df.copy()
    w["date"] = pd.to_datetime(w["date"])
    w = fill_leap_day(w.set_index("date"))
    if w.index.tz is None:
        w.index = w.index.tz_localize("UTC")
    else:
        w.index = w.index.tz_convert("UTC")
    return w


def _load_weather_years(
    cfg: dict[str, Any],
    resolved: ResolvedAppConfig,
    settings: MonteCarloSettings,
    *,
    runtime_weather: dict[str, Any] | None = None,
) -> dict[int, pd.DataFrame]:
    """Read the weather file into per-year frames at the study resolution."""
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

    indexed_by_year: dict[int, pd.DataFrame] = {}
    for year, df in weather_by_year.items():
        weather = _index_weather(df)
        input_frequency = weather_input_frequency(weather)
        weather = resample_hourly_weather(
            weather,
            freq,
            latitude=resolved.lat,
            longitude=resolved.lon,
            resample=resample_to_15min,
            irradiance_resampling=cfg.get("irradiance_resampling", "auto"),
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
        indexed_by_year[year] = weather
    return indexed_by_year


def _build_pv_years(
    cfg: dict[str, Any],
    resolved: ResolvedAppConfig,
    weather_by_year: dict[int, pd.DataFrame],
) -> tuple[dict[int, pd.Series], dict[int, pd.Series]]:
    """Build each year's undegraded DC production and battery temperature series.

    Both read the weather frames without writing to them, which is what lets
    a :class:`MonteCarloYearCache` build them again from the frames it holds.
    """
    dc_by_year: dict[int, pd.Series] = {}
    temp_by_year: dict[int, pd.Series] = {}
    for year, weather in weather_by_year.items():
        dc_by_year[year] = build_dc_system_base(cfg, resolved, weather)
        temp_by_year[year] = build_battery_temperature_series(
            cfg["battery_temperature"],
            index=dc_by_year[year].index,
            weather_df=weather,
            indoor_model=cfg["battery_indoor_model"],
        )
    return dc_by_year, temp_by_year


def _precompute_year_caches(
    cfg: dict[str, Any],
    resolved: ResolvedAppConfig,
    settings: MonteCarloSettings,
    *,
    runtime_weather: dict[str, Any] | None = None,
) -> tuple[dict[int, pd.Series], dict[int, pd.Series]]:
    """Build per-year undegraded DC production and battery temperature series."""
    weather_by_year = _load_weather_years(cfg, resolved, settings, runtime_weather=runtime_weather)
    return _build_pv_years(cfg, resolved, weather_by_year)


# Config keys the PV layer of the year cache never reads, directly or through
# the ResolvedAppConfig fields it reads. They are the input-independent keys
# of the App sweep, plus the demand keys and the runner's own section: a
# Monte Carlo study loads its demand apart from the year cache, and reads its
# weather from MonteCarloSettings, not the config. Each is pinned by a test
# that changes it and compares the year cache. A key not listed here is part
# of the cache key, so a new key costs a rebuild, never a wrong reuse.
YEAR_CACHE_INDEPENDENT_KEYS: frozenset[str] = INPUT_INDEPENDENT_KEYS | frozenset(
    {
        # Demand.
        "annual_consumption_kwh",
        "load_profile",
        "load_profile_file",
        "load_profile_column",
        "load_profile_unit",
        "rlp_directory",
        "start_date",
        # The [montecarlo] section, read by the CLI into MonteCarloSettings.
        "montecarlo",
    }
)


def _weather_cache_key(
    cfg: dict[str, Any], resolved: ResolvedAppConfig, settings: MonteCarloSettings
) -> dict[str, Any]:
    """Everything the weather layer of the year cache reads.

    The path is kept as given, made absolute but not resolved through links,
    because that is the path the study's provenance records. The metadata
    sidecar carries the file's timestamp and radiation timing, which move
    the sun, so it is part of the weather too.
    """
    path = os.path.abspath(os.fspath(settings.weather_file))
    sidecar = _weather_metadata_sidecar_path(path)
    return {
        "weather_file": path,
        "weather_file_sha256": _weather_file_sha256(path),
        "weather_metadata_sidecar_sha256": _weather_file_sha256(sidecar) if sidecar.is_file() else None,
        "target_year": settings.target_year,
        "weather_start_year": settings.weather_start_year,
        "weather_end_year": settings.weather_end_year,
        "resolution": cfg["resolution"],
        "latitude": resolved.lat,
        "longitude": resolved.lon,
        "irradiance_resampling": cfg.get("irradiance_resampling", "auto"),
        "solar_position": resolve_solar_position_method(cfg.get("solar_position", DEFAULT_SOLAR_POSITION)),
    }


def _pv_cache_key(cfg: dict[str, Any]) -> str | None:
    """The resolved config without :data:`YEAR_CACHE_INDEPENDENT_KEYS`, as canonical JSON.

    A ``battery_temperature`` CSV is keyed on its contents as well as its
    path, so a file rewritten in place is read again. None when a value is
    not plain JSON data, which the key could not represent faithfully; the
    PV layer is then built afresh for every run.
    """
    key = config_cache_key(cfg, YEAR_CACHE_INDEPENDENT_KEYS)
    temperature = cfg.get("battery_temperature")
    if key is None or not isinstance(temperature, (str, os.PathLike)) or str(temperature).lower() == "weather":
        return key
    return json.dumps([key, _weather_file_sha256(temperature)])


class MonteCarloYearCache:
    """The per-year weather and PV inputs of a Monte Carlo study, for reuse.

    Build one with :func:`build_year_cache` and pass it to
    :func:`run_montecarlo` as ``year_cache`` to run many designs over one
    weather file without preparing the weather again for each.

    It holds two layers. The weather layer is each weather year read,
    restamped and resampled to the study resolution. It is keyed on the
    weather file's absolute path, its SHA-256 and that of its metadata
    sidecar, the year window and target year, the resolution, the
    coordinates, ``irradiance_resampling`` and the solar-position
    method, and a study whose weather key differs is refused. The PV layer
    is each year's DC production and battery temperature. It is keyed on the
    resolved config without :data:`YEAR_CACHE_INDEPENDENT_KEYS`, such as the
    battery sizing and dispatch, inverter, cost and demand settings, and on
    the contents of a ``battery_temperature`` CSV. A study with another PV
    key, such as another module count or battery temperature, builds the PV
    layer again from the cached weather and keeps it in place of the old
    one, so run the designs grouped by PV configuration to reuse each PV
    layer fully.

    The weather file is read once, when the cache is built: a warning it
    raises then does not repeat for each study. A study may replace the PV
    layer, so the cache is not safe to share between threads running
    studies at once.
    """

    def __init__(
        self,
        weather_key: dict[str, Any],
        weather_by_year: dict[int, pd.DataFrame],
        runtime_weather: dict[str, Any],
        pv_key: str | None,
        dc_by_year: dict[int, pd.Series],
        temp_by_year: dict[int, pd.Series],
    ) -> None:
        self._weather_key = weather_key
        self._weather_by_year = weather_by_year
        self._runtime_weather = runtime_weather
        self._pv_key = pv_key
        self._dc_by_year = dc_by_year
        self._temp_by_year = temp_by_year

    @property
    def weather_key(self) -> dict[str, Any]:
        """The weather inputs this cache was built from."""
        return dict(self._weather_key)

    @property
    def pv_key(self) -> str | None:
        """The PV configuration of the PV layer it holds, or None if it cannot be reused."""
        return self._pv_key

    @property
    def available_years(self) -> list[int]:
        """The weather years a study samples from."""
        return sorted(int(year) for year in self._weather_by_year)

    def __repr__(self) -> str:
        return f"MonteCarloYearCache(weather_file={self._weather_key['weather_file']!r}, years={self.available_years})"

    def _years_for(
        self, cfg: dict[str, Any], resolved: ResolvedAppConfig, settings: MonteCarloSettings
    ) -> tuple[dict[int, pd.Series], dict[int, pd.Series], dict[str, Any]]:
        """The DC and temperature series and weather record for one study."""
        weather_key = _weather_cache_key(cfg, resolved, settings)
        if weather_key != self._weather_key:
            differing = ", ".join(name for name in weather_key if weather_key[name] != self._weather_key.get(name))
            raise ValueError(
                f"year_cache was built for other weather inputs ({differing} differ); "
                "build one for this study with build_year_cache(config, settings)"
            )
        pv_key = _pv_cache_key(cfg)
        if pv_key is None or pv_key != self._pv_key:
            self._dc_by_year, self._temp_by_year = _build_pv_years(cfg, resolved, self._weather_by_year)
            self._pv_key = pv_key
        # New dicts over the shared series: a study reads them and never
        # writes to them, and its provenance keeps its own weather record.
        return dict(self._dc_by_year), dict(self._temp_by_year), deepcopy(self._runtime_weather)


def _reject_period(cfg: dict[str, Any]) -> None:
    """Refuse a [period] window, before any weather is loaded or cached."""
    if cfg.get("period") is not None:
        raise ValueError(
            "'period' is not supported with Monte Carlo: each trajectory simulates whole weather years and "
            "their lifetime economics. Remove 'period', or run the window with breos.App."
        )


def _reject_planned_smart_charging(resolved: ResolvedAppConfig) -> None:
    """Refuse daily-persistence smart charging, before any weather is loaded or any trajectory runs.

    Every trajectory replays one set of static instructions, and that mode
    decides its instructions day by day from each run's own observations.
    """
    spec = resolved.smart_charging
    if spec is not None and spec.mode in PLANNER_MODES:
        raise ValueError(
            f"smart_charging mode = '{spec.mode}' is experimental and runs in breos.App only; Monte Carlo "
            "shares one set of static instructions across trajectories. Use mode = 'fixed_target' or "
            "'discharge_only', or run the design with breos.App."
        )


def build_year_cache(config: dict[str, Any], settings: MonteCarloSettings) -> MonteCarloYearCache:
    """Prepare the per-year weather and PV inputs once, for many Monte Carlo studies.

    Args:
        config: An App configuration dict, as for :func:`run_montecarlo`.
        settings: Monte Carlo controls. Only the weather settings are read:
            ``weather_file``, ``target_year``, ``weather_start_year`` and
            ``weather_end_year``. The irradiance policy comes from the App
            config's ``irradiance_resampling``.

    Returns:
        A :class:`MonteCarloYearCache` to pass to :func:`run_montecarlo` as
        ``year_cache``. Its results are the same, bit for bit, as a study
        run without it.
    """
    resolved = resolve_app_config(config)
    cfg = resolved.cfg
    _reject_period(cfg)
    _reject_planned_smart_charging(resolved)
    weather_key = _weather_cache_key(cfg, resolved, settings)
    runtime_weather: dict[str, Any] = {}
    weather_by_year = _load_weather_years(cfg, resolved, settings, runtime_weather=runtime_weather)
    dc_by_year, temp_by_year = _build_pv_years(cfg, resolved, weather_by_year)
    return MonteCarloYearCache(
        weather_key, weather_by_year, runtime_weather, _pv_cache_key(cfg), dc_by_year, temp_by_year
    )


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
# outputs), at eight bytes per timestep. Nineteen weather years over a
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
    if config_has_battery(cfg):
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
    reference_tariff: ResolvedTariff | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run one Monte Carlo trajectory and return its summary metrics.

    A ``reference_tariff`` prices the trajectory's no-system household on its
    own sampled load, so the with- and without-system costs stay paired.
    """
    degradation_rate = cfg["pv_degradation_rate"]
    has_battery = config_has_battery(cfg)

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
        reference_tariff=reference_tariff,
    )
    current_soh = projection.carry.soh_pct
    total_replacements = projection.total_replacements
    value = value_projection(cfg, resolved, projection)
    cost_projection, lcoe, yearly_df = value.cost_projection, value.lcoe, value.yearly_df
    total_replacement_cost = value.total_replacement_cost
    payback_year = find_payback_year(cost_projection)
    payback_year_interpolated = find_payback_year_interpolated(cost_projection)
    npv_savings = float(cost_projection["Savings_Cumulative_NPV"].iloc[-1])

    trajectory = yearly_df.merge(cost_projection, on="Year", how="left", suffixes=("", "_Financial"))
    lifetime_load = float(yearly_df["Load_kWh"].sum())
    lifetime_import = float(yearly_df["Import_kWh"].sum())
    lifetime_gi = 100.0 * (1.0 - lifetime_import / lifetime_load) if lifetime_load > 0.0 else 0.0

    metrics = {
        "npv_savings": npv_savings,
        "payback_year": payback_year if payback_year is not None else float("nan"),
        "payback_year_interpolated": payback_year_interpolated
        if payback_year_interpolated is not None
        else float("nan"),
        "lcoe_per_kwh": float(lcoe),
        "final_soh_pct": float(current_soh) if has_battery else float("nan"),
        "mean_grid_independence_pct": float(yearly_df["Grid_Independence_%"].mean()),
        "lifetime_grid_independence_pct": lifetime_gi,
        "total_replacements": int(total_replacements),
        "total_replacement_cost_t0_prices": float(total_replacement_cost),
        "mean_pv_dc_generation_kwh": float(yearly_df["PV_DC_Generation_kWh"].mean()),
        "mean_direct_pv_ac_load_kwh": float(yearly_df["Direct_PV_AC_Load_kWh"].mean()),
        "mean_pv_origin_battery_ac_load_kwh": float(yearly_df["PV_Origin_Battery_AC_Load_kWh"].mean()),
        "mean_self_consumption_kwh": float(yearly_df["Self_Consumption_kWh"].mean()),
        "mean_usable_ac_system_production_kwh": float(yearly_df["PV_Production_kWh"].mean()),
        "mean_import_kwh": float(yearly_df["Import_kWh"].mean()),
        "mean_export_kwh": float(yearly_df["Export_kWh"].mean()),
    }
    terminal = value.terminal_health
    metrics.update(
        terminal_health_credit=terminal.nominal if terminal else float("nan"),
        terminal_health_credit_npv=terminal.npv if terminal else float("nan"),
        npv_savings_terminal_adjusted=terminal.adjusted_npv if terminal else float("nan"),
    )
    if terminal is not None:
        metrics["_terminal_value_provenance"] = terminal.provenance
    return metrics, trajectory


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
    return resolved.tariff.resolve(_study_calendar(aligned_by_year), resolved.timezone)


def _resolve_study_reference_tariff(
    resolved: ResolvedAppConfig, aligned_by_year: dict[int, AlignedSimulationInputs]
) -> ResolvedTariff | None:
    """Resolve the configured no-system reference tariff once for the whole study, as the tariff is."""
    if resolved.reference_tariff is None:
        return None
    return resolved.reference_tariff.resolve(_study_calendar(aligned_by_year), resolved.timezone)


def _study_calendar(aligned_by_year: dict[int, AlignedSimulationInputs]) -> pd.DatetimeIndex:
    calendars = [inputs.index for inputs in aligned_by_year.values()]
    calendar = calendars[0]
    if any(not other.equals(calendar) for other in calendars[1:]):
        raise ValueError("Monte Carlo weather years do not share one calendar, so one tariff cannot price them")
    return pd.DatetimeIndex(calendar)


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
        reference_tariff,
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
        reference_tariff,
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


def run_montecarlo(
    config: dict[str, Any],
    settings: MonteCarloSettings,
    *,
    year_cache: MonteCarloYearCache | None = None,
) -> MonteCarloResult:
    """Run a Monte Carlo study over weather years and demand uncertainty.

    Args:
        config: An App configuration dict (same keys as :class:`breos.App`).
        settings: Monte Carlo controls (weather file, runs, uncertainty, seed).
        year_cache: Per-year weather and PV inputs from :func:`build_year_cache`,
            reused instead of prepared again. A sweep over designs builds it
            once and passes it to every study. Raises ``ValueError`` when it
            was built for other weather inputs; see
            :class:`MonteCarloYearCache` for what it reuses.

    The weather, the load and any tariff share the ``settings.target_year``
    calendar. The load is the one an App run of that year would build. The
    config's ``start_date`` is validated but not used for the load or weather,
    and ``provenance["load_profile"]["calendar_year"]`` records the year the
    load was built for. A leap target year's 29 February copies 28 February's
    weather, as in the App.

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
    _reject_planned_smart_charging(resolved)
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
    _reject_period(cfg)
    if cfg["horizon_profile"] is not None:
        raise ValueError(
            "'horizon_profile' is not supported with Monte Carlo weather files yet because their "
            "terrain-horizon provenance is unknown"
        )
    years_per_run = settings.years_per_run or cfg["projection_years"]

    # Resolve the dispatch backend before any input is loaded, so a missing
    # optional dependency stops a 10,000-trajectory study immediately rather
    # than hours into it.
    has_battery = config_has_battery(cfg)
    execution = backend_provenance(settings.execution_backend, pv_only=not has_battery)

    if year_cache is None:
        runtime_weather: dict[str, Any] = {}
        dc_by_year, temp_by_year = _precompute_year_caches(
            cfg,
            resolved,
            settings,
            runtime_weather=runtime_weather,
        )
    else:
        dc_by_year, temp_by_year, runtime_weather = year_cache._years_for(cfg, resolved, settings)
    available_years = np.array(sorted(dc_by_year.keys()))

    deps = App._runtime_dependencies()
    # Every weather year is restamped to target_year, so the load is built on
    # that calendar too: H0 day types then match the study year's weekdays, as
    # in an App run of that year. The year of start_date does not enter.
    base_load = load_consumption_profile(
        {**cfg, "start_date": f"{settings.target_year}-01-01"}, deps, timezone=resolved.timezone
    )
    aligned_by_year = _align_years(
        cfg,
        base_load,
        dc_by_year,
        temp_by_year,
        has_battery=has_battery,
    )
    pv_chains = _prepare_pv_chains(cfg, resolved, aligned_by_year, settings, years_per_run)
    tariff = _resolve_study_tariff(resolved, aligned_by_year)
    reference_tariff = _resolve_study_reference_tariff(resolved, aligned_by_year)
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
        reference_tariff,
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
    terminal_records = []
    for run_idx, metrics, trajectory, jit_cache_state in outputs:
        terminal_record = metrics.pop("_terminal_value_provenance", None)
        if terminal_record is not None:
            terminal_records.append({"run": run_idx + 1, **terminal_record})
        rows.append({"run": run_idx + 1, **metrics})
        if jit_cache_state is not None:
            jit_cache_states.append(jit_cache_state)
        if trajectory is not None:
            trajectory.insert(0, "run", run_idx + 1)
            yearly_frames.append(trajectory)

    if settings.execution_backend == "numba":
        execution["jit_cache"] = aggregate_jit_cache_states(jit_cache_states)

    runs_df = pd.DataFrame(rows)
    currency = result_currency(resolved.tariff)
    # Plot labels read the currency from the frame.
    runs_df.attrs["currency"] = currency
    yearly_df = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else None
    return MonteCarloResult(
        runs=runs_df,
        summary=_summarize(runs_df),
        settings=settings,
        available_years=[int(y) for y in available_years],
        yearly=yearly_df,
        provenance={
            "breos_version": package_version(),
            "result_schema_version": RESULT_SCHEMA_VERSION,
            # Every money column and summary is in this currency; BREOS does not convert.
            "currency": currency,
            "resolved_config": cfg,
            "settings": asdict(settings),
            "available_weather_years": [int(y) for y in available_years],
            "runtime_weather": runtime_weather,
            # The load is built for target_year; resolved_config keeps the
            # start_date the user gave, which Monte Carlo does not use.
            "load_profile": {
                **base_load.attrs.get(LOAD_PROFILE_METADATA_KEY, {}),
                "calendar_year": settings.target_year,
            },
            "random_stream": (
                "numpy.random.default_rng(numpy.random.SeedSequence(base_seed).spawn(n_runs)[zero_based_run_index])"
            ),
            "execution": execution,
            "economics": projection_rates_record(cfg),
            **(
                {"terminal_value": {"basis": "battery_health_fraction", "trajectories": terminal_records}}
                if terminal_records
                else {}
            ),
            "ledger_schema_version": LEDGER_SCHEMA_VERSION,
            **({"tariff": tariff_provenance(tariff, calendar_year=settings.target_year)} if tariff is not None else {}),
            **(
                {
                    "reference_tariff": reference_tariff_provenance(
                        reference_tariff,
                        calendar_year=settings.target_year,
                        import_price_escalation=effective_reference_escalation(resolved),
                    )
                }
                if reference_tariff is not None
                else {}
            ),
            **(
                {"smart_charging": smart_charging_provenance(spec, instructions, tariff)}
                if spec is not None and instructions is not None and tariff is not None
                else {}
            ),
        },
    )
