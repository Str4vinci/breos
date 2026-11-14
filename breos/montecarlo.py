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
    InverterSteps,
    build_inverter_ac_derate,
    build_inverter_curve_row,
    build_pv_production_breakdown,
    config_cache_key,
    load_consumption_profile,
    resample_hourly_weather,
    weather_input_frequency,
)
from breos.battery import (
    LEDGER_SCHEMA_VERSION,
    AlignedSimulationInputs,
    align_simulation_inputs,
    simulate_energy_balance_summary,
)
from breos.dispatch_instructions import DispatchInstructions
from breos.economics import find_payback_year, find_payback_year_interpolated, projection_rates_record
from breos.exceedance import yield_exceedance
from breos.execution import (
    aggregate_jit_cache_states,
    backend_provenance,
    config_has_battery,
    limit_worker_threads,
    observed_jit_cache_state,
    reset_jit_cache_observation,
    resolve_execution_backend,
    validate_requested_backend,
)
from breos.load_profiles import LOAD_PROFILE_METADATA_KEY
from breos.projection import (
    ProjectionYear,
    build_pv_only_battery_config,
    first_end_of_life_metrics,
    run_projection,
    tariff_records,
    value_projection,
)
from breos.pv._solver import SolverDiagnostics
from breos.pv.model_options import (
    DEFAULT_SOLAR_POSITION,
    resolve_solar_position_method,
    solar_position_record,
    solar_position_warning_scope,
    warn_solar_position_fallback,
)
from breos.result_schema import RESULT_SCHEMA_VERSION
from breos.smart_charging import PLANNER_MODES, resolve_instructions, smart_charging_provenance
from breos.tariffs import ResolvedTariff
from breos.utils import package_version
from breos.vehicle import VehicleConfig, VehicleSchedule, resolve_vehicle_schedule
from breos.vehicle.households import (
    VehicleHouseholds,
    household_configs,
    run_vehicle_baseline,
    value_vehicle_households,
    vehicle_provenance,
)
from breos.weather import (
    _weather_file_sha256,
    _weather_metadata_sidecar_path,
    _weather_years,
    build_battery_temperature_series,
    fill_leap_day,
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
_VEHICLE_COMPARISON_METRICS = (
    "npv_savings_household_adjusted",
    "household_adjusted_comparable",
    "vehicle_benefit_npv",
    "vehicle_benefit_comparable",
)


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
    # "python" is the reference implementation. "numba" selects the optional
    # compiled within-day dispatch kernel and requires breos[fast]; it is
    # checked before any trajectory starts. "auto" runs "numba" when Numba is
    # installed and "python" otherwise. None inherits the App config's
    # top-level ``execution_backend``, which itself defaults to "auto"; see
    # :func:`run_montecarlo`.
    execution_backend: str | None = None
    # One-sigma uncertainty of the PV model's annual yield, in percent: the
    # irradiance data, transposition, PV model and loss assumptions together.
    # Stated by the caller; None leaves it out. Each trajectory then draws one
    # persistent yield factor, and the yield exceedance values include it.
    model_uncertainty_pct: float | None = None

    def __post_init__(self) -> None:
        # Checked when the settings are made, so build_year_cache refuses
        # what run_montecarlo refuses.
        if self.n_runs < 1:
            raise ValueError("n_runs must be at least 1")
        if self.years_per_run is not None and self.years_per_run < 1:
            raise ValueError("years_per_run must be at least 1")
        if not (math.isfinite(self.load_uncertainty) and self.load_uncertainty >= 0.0):
            raise ValueError("load_uncertainty must be a finite, non-negative number")
        if self.model_uncertainty_pct is not None and not (
            math.isfinite(self.model_uncertainty_pct) and self.model_uncertainty_pct >= 0.0
        ):
            raise ValueError("model_uncertainty_pct must be a finite, non-negative number or None")
        # The load scale multiplies demand, so a negative bound would make demand negative.
        if not (math.isfinite(self.min_load_scale) and self.min_load_scale >= 0.0):
            raise ValueError(f"min_load_scale must be a finite, non-negative number, got {self.min_load_scale}")
        # None is the way to leave demand unbounded above. An infinite bound would
        # say the same thing, but it cannot be written to strict JSON provenance.
        if self.max_load_scale is not None and not math.isfinite(self.max_load_scale):
            raise ValueError(
                "max_load_scale must be a finite number, or None to leave the load scale unbounded, "
                f"got {self.max_load_scale}"
            )
        if self.max_load_scale is not None and not self.max_load_scale >= self.min_load_scale:
            raise ValueError(
                f"max_load_scale must be at least min_load_scale ({self.min_load_scale}), got {self.max_load_scale}"
            )
        if self.load_distribution not in {"normal", "uniform"}:
            raise ValueError("load_distribution must be 'normal' or 'uniform'")
        if self.n_procs < 1:
            raise ValueError("n_procs must be at least 1")
        if self.execution_backend is not None:
            validate_requested_backend(self.execution_backend)
        if (
            self.weather_start_year is not None
            and self.weather_end_year is not None
            and self.weather_start_year > self.weather_end_year
        ):
            raise ValueError("weather_start_year must not be later than weather_end_year")


def _solver_summary(by_weather_year: dict[int, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Recovery counts for prepared source years, counted once per weather year."""
    years = deepcopy(by_weather_year or {})
    stages: dict[str, int] = {}
    for record in years.values():
        for stage, count in record["by_stage"].items():
            stages[stage] = stages.get(stage, 0) + count
    return {
        "by_weather_year": years,
        "total": {
            "recovered_steps": sum(record["recovered_steps"] for record in years.values()),
            "by_stage": dict(sorted(stages.items())),
        },
    }


@dataclass
class MonteCarloResult:
    """Outcome of a Monte Carlo study.

    ``solver`` reports recoveries by source weather year and their total,
    counted once per prepared year rather than per trajectory draw.
    """

    runs: pd.DataFrame  # one row per run
    summary: dict[str, dict[str, float]]  # metric -> descriptive statistics and quantiles
    settings: MonteCarloSettings
    available_years: list[int] = field(default_factory=list)
    yearly: pd.DataFrame | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    # P50/P75/P90/P99 annual yield over the weather years, for one year and
    # the study horizon (breos.exceedance); None with fewer than two years.
    yield_exceedance: dict[str, Any] | None = None
    # Recoveries in the parent's prepared inputs, by source weather year and
    # summed over those years; not multiplied by trajectory sampling.
    solver: dict[str, Any] = field(default_factory=_solver_summary)


def _sample_pv_model_factor(settings: MonteCarloSettings, run_idx: int) -> float:
    """One trajectory's PV yield factor from the model uncertainty; 1 without one.

    Drawn from the run's own child stream, ``SeedSequence(seed).spawn(n_runs)[run_idx].spawn(1)[0]``,
    so stating an uncertainty leaves every weather year and load scale draw as it was.
    """
    sigma = settings.model_uncertainty_pct
    if not sigma:
        return 1.0
    rng = np.random.default_rng(np.random.SeedSequence(settings.seed, spawn_key=(run_idx, 0)))
    return max(0.0, 1.0 + float(rng.normal(0.0, sigma / 100.0)))


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
    """A weather year, indexed on the file's clock, in UTC as the deterministic pipeline has it.

    The years are read without 29 February, so a leap target year gets a copy
    of 28 February, as an App run of that year does. The day is filled on the
    file's own clock, where it was dropped: in UTC, a file written at +01:00
    would still have an hour dated 29 February, and the fill would be skipped.
    """
    w = fill_leap_day(df)
    return w.tz_localize("UTC") if w.index.tz is None else w.tz_convert("UTC")


def _load_weather_years(
    cfg: dict[str, Any],
    resolved: ResolvedAppConfig,
    settings: MonteCarloSettings,
    *,
    runtime_weather: dict[str, Any] | None = None,
) -> dict[int, pd.DataFrame]:
    """Read the weather file into per-year frames at the study resolution."""
    freq = cfg["resolution"]
    weather_by_year = _weather_years(settings.weather_file, target_year=settings.target_year)
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
            # "Mid-Interval" is recorded with the offset it actually gets, and
            # "auto" with the method it resolves to on this weather.
            solar_position = solar_position_record(cfg.get("solar_position", DEFAULT_SOLAR_POSITION), weather, freq)
            runtime_weather.update(
                {
                    "representative_source_year": int(year),
                    "input_resolution": input_frequency,
                    "output_resolution": str(freq),
                    "solar_position_requested": solar_position["requested"],
                    "solar_position_method": solar_position["resolved"],
                    "solar_position_offset_minutes": solar_position["offset_minutes"],
                    "metadata": weather_metadata(weather),
                }
            )
        indexed_by_year[year] = weather
    return indexed_by_year


_PvYears = tuple[dict[int, pd.Series], dict[int, pd.Series], dict[int, InverterSteps]]


# Every weather year computes the sun position, and an "auto" fallback warns
# once for the study, not once per year.
@solar_position_warning_scope()
def _build_pv_years(
    cfg: dict[str, Any],
    resolved: ResolvedAppConfig,
    weather_by_year: dict[int, pd.DataFrame],
    *,
    solver_by_year: dict[int, dict[str, Any]] | None = None,
) -> _PvYears:
    """Build each year's undegraded DC production, battery temperature and inverter series.

    All read the weather frames without writing to them, which is what lets
    a :class:`MonteCarloYearCache` build them again from the frames it holds.
    The inverter series are its temperature derating and its efficiency
    curve's row at each step's operating voltage, each None when not modelled.
    ``solver_by_year`` retains the recovery record from each prepared year.
    """
    dc_by_year: dict[int, pd.Series] = {}
    temp_by_year: dict[int, pd.Series] = {}
    inverter_by_year: dict[int, InverterSteps] = {}
    for year, weather in weather_by_year.items():
        breakdown = build_pv_production_breakdown(cfg, resolved, weather)
        if solver_by_year is not None:
            solver_by_year[year] = deepcopy(breakdown.solver_diagnostics or SolverDiagnostics().result())
        dc_by_year[year] = breakdown.dc_after_losses
        temp_by_year[year] = build_battery_temperature_series(
            cfg["battery_temperature"],
            index=dc_by_year[year].index,
            weather_df=weather,
            indoor_model=cfg["battery_indoor_model"],
        )
        inverter_by_year[year] = InverterSteps(
            build_inverter_ac_derate(resolved, weather, dc_by_year[year].index),
            build_inverter_curve_row(resolved, breakdown),
        )
    return dc_by_year, temp_by_year, inverter_by_year


def _precompute_year_caches(
    cfg: dict[str, Any],
    resolved: ResolvedAppConfig,
    settings: MonteCarloSettings,
    *,
    runtime_weather: dict[str, Any] | None = None,
    solver_by_year: dict[int, dict[str, Any]] | None = None,
) -> _PvYears:
    """Build per-year undegraded DC production, battery temperature and inverter series."""
    weather_by_year = _load_weather_years(cfg, resolved, settings, runtime_weather=runtime_weather)
    return _build_pv_years(cfg, resolved, weather_by_year, solver_by_year=solver_by_year)


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
        inverter_by_year: dict[int, InverterSteps],
        solver_by_year: dict[int, dict[str, Any]] | None = None,
    ) -> None:
        self._weather_key = weather_key
        self._weather_by_year = weather_by_year
        self._runtime_weather = runtime_weather
        self._pv_key = pv_key
        self._dc_by_year = dc_by_year
        self._temp_by_year = temp_by_year
        self._inverter_by_year = inverter_by_year
        self._solver_by_year = (
            deepcopy(solver_by_year)
            if solver_by_year is not None
            else {year: SolverDiagnostics().result() for year in dc_by_year}
        )

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
        self,
        cfg: dict[str, Any],
        resolved: ResolvedAppConfig,
        settings: MonteCarloSettings,
        *,
        solver_by_year: dict[int, dict[str, Any]] | None = None,
    ) -> tuple[dict[int, pd.Series], dict[int, pd.Series], dict[int, InverterSteps], dict[str, Any]]:
        """The DC, temperature and inverter series and weather record for one study."""
        weather_key = _weather_cache_key(cfg, resolved, settings)
        if weather_key != self._weather_key:
            differing = ", ".join(name for name in weather_key if weather_key[name] != self._weather_key.get(name))
            raise ValueError(
                f"year_cache was built for other weather inputs ({differing} differ); "
                "build one for this study with build_year_cache(config, settings)"
            )
        pv_key = _pv_cache_key(cfg)
        if pv_key is None or pv_key != self._pv_key:
            rebuilt_solver: dict[int, dict[str, Any]] = {}
            self._dc_by_year, self._temp_by_year, self._inverter_by_year = _build_pv_years(
                cfg, resolved, self._weather_by_year, solver_by_year=rebuilt_solver
            )
            self._solver_by_year = rebuilt_solver
            self._pv_key = pv_key
        if solver_by_year is not None:
            solver_by_year.update(deepcopy(self._solver_by_year))
        # New dicts over the shared series: a study reads them and never
        # writes to them, and its provenance keeps its own weather record.
        return (
            dict(self._dc_by_year),
            dict(self._temp_by_year),
            dict(self._inverter_by_year),
            deepcopy(self._runtime_weather),
        )


def _reject_period(cfg: dict[str, Any]) -> None:
    """Refuse a [period] window, before any weather is loaded or cached."""
    if cfg.get("period") is not None:
        raise ValueError(
            "'period' is not supported with Monte Carlo: each trajectory simulates whole weather years and "
            "their lifetime economics. Remove 'period', or run the window with breos.App."
        )


def _reject_weather_file(cfg: dict[str, Any]) -> None:
    """Refuse the App's one-year ``weather_file``: a study samples ``MonteCarloSettings.weather_file``."""
    if cfg.get("weather_file") is not None:
        raise ValueError(
            "'weather_file' names the App's one-year weather and is not used by Monte Carlo, which samples the "
            "years of MonteCarloSettings.weather_file ([montecarlo].weather_file in a config file). "
            "Remove 'weather_file' from the config."
        )


def _reject_several_string_inverters(resolved: ResolvedAppConfig) -> None:
    """Refuse string-level energy with several inverters, before any weather is loaded or any trajectory runs.

    Each trajectory carries one DC series, converted by one inverter; several
    inverters each convert their own share (ADR 0007 D10), in breos.App only.
    """
    electrical = resolved.electrical
    if electrical is not None and electrical.fidelity == "string" and len(electrical.inverters) > 1:
        raise ValueError(
            f"electrical_fidelity = 'string' with {len(electrical.inverters)} inverters runs in breos.App only; "
            "Monte Carlo converts each trajectory's DC through one inverter. Run the design with breos.App, or "
            "use electrical_fidelity = 'aggregate'."
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


def _reject_vehicle_blast(resolved: ResolvedAppConfig) -> None:
    if resolved.vehicle is not None and resolved.vehicle.degradation_engine == "blast":
        raise ValueError("vehicle.degradation_engine=blast is not supported with Monte Carlo yet")


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
    _reject_vehicle_blast(resolved)
    cfg = resolved.cfg
    _reject_period(cfg)
    _reject_weather_file(cfg)
    _reject_planned_smart_charging(resolved)
    _reject_several_string_inverters(resolved)
    weather_key = _weather_cache_key(cfg, resolved, settings)
    runtime_weather: dict[str, Any] = {}
    weather_by_year = _load_weather_years(cfg, resolved, settings, runtime_weather=runtime_weather)
    solver_by_year: dict[int, dict[str, Any]] = {}
    dc_by_year, temp_by_year, inverter_by_year = _build_pv_years(
        cfg, resolved, weather_by_year, solver_by_year=solver_by_year
    )
    return MonteCarloYearCache(
        weather_key,
        weather_by_year,
        runtime_weather,
        _pv_cache_key(cfg),
        dc_by_year,
        temp_by_year,
        inverter_by_year,
        solver_by_year,
    )


def _align_years(
    cfg: dict[str, Any],
    base_load: pd.DataFrame,
    dc_by_year: dict[int, pd.Series],
    temp_by_year: dict[int, pd.Series],
    inverter_by_year: dict[int, InverterSteps],
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
            inverter_ac_derate=inverter_by_year[year].ac_derate,
            inverter_curve_row=inverter_by_year[year].curve_row,
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
    if config_has_battery(cfg) or resolved.vehicle is not None or settings.model_uncertainty_pct:
        # Vehicle households use the frame path to retain exchanges for App's
        # valuation, rather than this aligned PV-only conversion memo.
        # A trajectory's model factor scales its PV, which a memoized
        # conversion cannot follow.
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


@dataclass(frozen=True)
class _Study:
    """What every trajectory of a study reads: built once, and installed once per worker process."""

    cfg: dict[str, Any]
    resolved: ResolvedAppConfig
    available_years: np.ndarray
    years_per_run: int
    settings: MonteCarloSettings
    aligned_by_year: dict[int, AlignedSimulationInputs]
    # Each (weather year, project year)'s PV chain, already scaled; None
    # when memoizing them does not pay (_pv_chain_cache_is_worthwhile).
    pv_chains: dict[tuple[int, int], AlignedSimulationInputs] | None = None
    tariff: ResolvedTariff | None = None
    instructions: DispatchInstructions | None = None
    # Prices each trajectory's no-system household on its own sampled load,
    # so the with- and without-system costs stay paired.
    reference_tariff: ResolvedTariff | None = None
    vehicle_temperature_by_year: dict[int, pd.Series] | None = None
    vehicle_schedule: VehicleSchedule | None = None


def _vehicle_frame_year(year: ProjectionYear, schedule: VehicleSchedule, temperature: pd.Series) -> ProjectionYear:
    """Present one sampled draw to App's retained-flow household path."""
    aligned = year.aligned
    assert aligned is not None and aligned.pv_chain is None
    index = aligned.index
    return replace(
        year,
        aligned=None,
        pv_dc=pd.Series(aligned.pv_dc_w, index=index),
        houseload=pd.DataFrame({"Load": aligned.load_w}, index=index),
        temperature_series=pd.Series(aligned.temperature_c, index=index),
        inverter_ac_derate=None
        if aligned.inverter_ac_derate is None
        else pd.Series(aligned.inverter_ac_derate, index=index),
        inverter_curve_row=None
        if aligned.inverter_curve_row is None
        else pd.Series(aligned.inverter_curve_row, index=index),
        vehicle_schedule=schedule,
        vehicle_temperature_series=temperature,
    )


def _vehicle_run_metrics(block: dict[str, Any]) -> dict[str, Any]:
    """Flatten App's numeric household results for runs tables and statistics."""
    adjusted, benefit = block["household_adjusted_npv"], block["vehicle_benefit"]
    metrics = {
        "npv_savings_household_adjusted": adjusted["npv"],
        "household_adjusted_comparable": adjusted["comparable"],
        "vehicle_benefit_npv": math.nan if benefit is None else benefit["npv"],
        "vehicle_benefit_comparable": None if benefit is None else benefit["comparable"],
    }
    for label, comparison in (("r2_r0", adjusted), ("r2_r1", benefit)):
        for output, source in (
            ("wear", "wear_npv"),
            ("replacement_outlays_difference", "replacement_outlays_difference_npv"),
        ):
            metrics[f"vehicle_{output}_{label}_npv"] = math.nan if comparison is None else comparison[source]
    for name in ("away_cost_difference_npv", "unserved_cost_difference_npv"):
        metrics[f"vehicle_household_adjusted_{name}"] = adjusted[name]
    if benefit is not None:
        for name in (
            "operating_savings_npv",
            "home_bill_savings_npv",
            "export_revenue_gain_npv",
            "away_savings_npv",
            "unserved_savings_npv",
            "charger_premium",
        ):
            metrics[f"vehicle_benefit_{name}"] = benefit[name]
    for key, household in block["households"].items():
        prefix = f"vehicle_{key.lower()}_"
        metrics.update({prefix + name: value for name, value in household["totals"].items()})
        for name in (
            "final_soh_fraction",
            "retired",
            "replacement_outlays_npv",
            "terminal_pack_credit_npv",
            "home_bill_npv",
            "export_revenue_npv",
            "away_purchases_npv",
            "unserved_driving_cost_npv",
            "operating_cost_npv",
        ):
            metrics[prefix + name] = household[name]
        for name in ("grid_import_kwh", "grid_export_kwh"):
            metrics[prefix + name] = sum(year[name] for year in household["yearly"])
    return metrics


def _simulate_trajectory(
    study: _Study, rng: np.random.Generator, pv_model_factor: float = 1.0
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run one Monte Carlo trajectory and return its summary metrics.

    ``pv_model_factor`` scales the trajectory's PV in every year: a model
    bias persists, unlike the weather.
    """
    cfg, resolved, settings = study.cfg, study.resolved, study.settings
    available_years, aligned_by_year, pv_chains = study.available_years, study.aligned_by_year, study.pv_chains
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
            aligned = aligned_by_year[year].scaled(
                pv_factor=pv_degradation_factor * pv_model_factor, load_factor=load_scale
            )
        else:
            aligned = pv_chains[(year, year_idx)].scaled(load_factor=load_scale)
        return ProjectionYear(
            pv_degradation_factor=pv_degradation_factor,
            aligned=aligned,
            extra={"Weather_Year": year, "Load_Scale": load_scale},
        )

    vehicle_result = None
    if resolved.vehicle is None:
        projection = run_projection(
            cfg,
            resolved,
            study.years_per_run,
            year_inputs,
            has_battery=has_battery,
            execution_backend=cast(str, settings.execution_backend),
            tariff=study.tariff,
            instructions=study.instructions,
            reference_tariff=study.reference_tariff,
        )
        value = value_projection(cfg, resolved, projection)
    else:
        # Consume precisely the original year/scale draws, once. Household
        # callbacks only read these materialised years; none resamples them.
        sampled = tuple(year_inputs(i) for i in range(study.years_per_run))
        assert study.vehicle_schedule is not None and study.vehicle_temperature_by_year is not None
        years = tuple(
            _vehicle_frame_year(
                year, study.vehicle_schedule, study.vehicle_temperature_by_year[int(year.extra["Weather_Year"])]
            )
            for year in sampled
        )
        configs = household_configs(resolved.vehicle)
        limit = (cfg.get("smart_charging") or {}).get("grid_import_limit_w")
        grid_limit = math.inf if limit is None else float(limit)

        def system_run(vehicle: VehicleConfig):
            return run_projection(
                cfg,
                resolved,
                study.years_per_run,
                years.__getitem__,
                has_battery=has_battery,
                execution_backend=cast(str, settings.execution_backend),
                tariff=study.tariff,
                instructions=study.instructions,
                reference_tariff=study.reference_tariff,
                vehicle_config=vehicle,
                grid_import_limit_w=grid_limit,
                record_priced_flows=True,
            )

        projection = system_run(configs["R2"])
        runs = {
            "R0": run_vehicle_baseline(
                study.years_per_run,
                years.__getitem__,
                configs["R0"],
                freq=cfg["resolution"],
                execution_backend=cast(str, settings.execution_backend),
                grid_import_limit_w=grid_limit,
            ),
            "R2": projection,
        }
        if "R1" in configs:
            runs["R1"] = system_run(configs["R1"])
        value, vehicle_result = value_vehicle_households(
            cfg, resolved, VehicleHouseholds(runs, study.vehicle_schedule), study.tariff, study.reference_tariff
        )

    current_soh = projection.carry.soh_pct
    total_replacements = projection.total_replacements
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
        "pv_model_factor": float(pv_model_factor),
        # The trajectory's first end-of-life crossing (ADR 0003 E11); NaN
        # and None without one.
        **first_end_of_life_metrics(projection.end_of_life_events),
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
    if vehicle_result is not None:
        metrics.update(_vehicle_run_metrics(vehicle_result))
    return metrics, trajectory


def _summarize(runs: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Describe each metric across runs.

    The ``p*`` keys are quantiles: ``p10`` is the value 10 % of runs fall
    below. A bankable P90 of a quantity that should be high, such as
    ``npv_savings``, is its ``p10``; of one that should be low, such as
    ``lcoe_per_kwh``, its ``p90``.

    Each entry gives ``count``, the number of runs its statistics cover, out
    of ``n_runs``. Runs without a value are left out of the statistics. For
    the payback metrics, such a run did not pay back within the horizon, so
    their statistics are conditional on payback. ``payback_probability`` is
    the unconditional share of runs that paid back. A payback entry is kept
    with only these counts when no run paid back.
    """
    summary: dict[str, dict[str, float]] = {}
    n_runs = len(runs)
    vehicle_metrics = tuple(
        column for column in runs if column.startswith("vehicle_") or column in _VEHICLE_COMPARISON_METRICS
    )
    for col in (*_SUMMARY_METRICS, *vehicle_metrics):
        if col not in runs.columns:
            continue
        # A non-finite value (an infinite LCOE with no production) is no
        # value, like NaN: it would make the mean infinite and the spread NaN.
        values = pd.to_numeric(runs[col], errors="coerce")
        if pd.api.types.is_bool_dtype(values.dtype):
            values = values.astype(float)
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
                "p10": float(series.quantile(0.10)),
                "p25": float(series.quantile(0.25)),
                "p50": float(series.quantile(0.50)),
                "p75": float(series.quantile(0.75)),
                "p90": float(series.quantile(0.90)),
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


_WORKER_STUDY: _Study | None = None


def _yield_exceedance(
    cfg: dict[str, Any],
    resolved: ResolvedAppConfig,
    aligned_by_year: dict[int, AlignedSimulationInputs],
    settings: MonteCarloSettings,
    years_per_run: int,
) -> dict[str, Any] | None:
    """P50/P75/P90/P99 annual AC yield over the weather years, for one year and ``years_per_run``.

    Each weather year's yield is the system's first-year, undegraded AC output
    without a battery, so it measures the weather and the PV system alone.
    None with fewer than two weather years.
    """
    if len(aligned_by_year) < 2:
        return None
    battery_config = build_pv_only_battery_config(cfg, resolved)
    yields = {
        year: simulate_energy_balance_summary(aligned=aligned, battery_config=battery_config).total_pv_wh / 1000.0
        for year, aligned in aligned_by_year.items()
    }
    result = yield_exceedance(yields, model_uncertainty_pct=settings.model_uncertainty_pct)
    return result.as_dict(horizons=(1, years_per_run))


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


def _initialize_worker(study: _Study) -> None:
    """Install read-only trajectory inputs once per worker process."""
    global _WORKER_STUDY
    _WORKER_STUDY = study


def _initialize_pool_worker(study: _Study) -> None:
    """Initialise a pool worker: one native thread, then the trajectory inputs."""
    limit_worker_threads()
    _initialize_worker(study)


def _run_trajectory_index(run_idx: int) -> tuple[int, dict[str, Any], pd.DataFrame | None, str | None]:
    """Evaluate one deterministic per-run random stream in a worker."""
    study = _WORKER_STUDY
    if study is None:
        raise RuntimeError("Monte Carlo worker context was not initialized")
    settings = study.settings
    # One observation window per trajectory: that is the unit of work whose
    # compile cost is being attributed. A no-op on the Python backend.
    reset_jit_cache_observation(settings.execution_backend)
    # Each run takes its own child of the base seed's SeedSequence, the same
    # stream as SeedSequence(seed).spawn(n_runs)[run_idx], so studies under
    # different base seeds share no trajectory. Without a seed, every run
    # draws fresh entropy.
    rng = np.random.default_rng(np.random.SeedSequence(settings.seed, spawn_key=(run_idx,)))
    metrics, trajectory = _simulate_trajectory(study, rng, pv_model_factor=_sample_pv_model_factor(settings, run_idx))
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
    config's top-level ``execution_backend``, else ``"auto"``. The CLI
    applies the same order after its own ``--execution-backend`` flag and
    ``[montecarlo].execution_backend``, so a study selects the same backend
    from Python and from ``breos montecarlo``. ``"auto"`` is resolved once,
    before any trajectory, so every worker runs the parent's backend. The
    returned result's ``settings`` records the backend that ran, and
    ``provenance["execution"]["requested_execution_backend"]`` the name asked for.

    Returns:
        A :class:`MonteCarloResult` with one row per run and summary statistics.
    """
    resolved = resolve_app_config(config)
    _reject_vehicle_blast(resolved)
    cfg = resolved.cfg
    _reject_planned_smart_charging(resolved)
    _reject_several_string_inverters(resolved)
    if settings.execution_backend is None:
        requested_backend = resolved.requested_execution_backend
        settings = replace(settings, execution_backend=cfg["execution_backend"])
    else:
        requested_backend = settings.execution_backend
        settings = replace(settings, execution_backend=resolve_execution_backend(settings.execution_backend))
    if cfg["degradation_engine"] == "blast":
        raise ValueError("degradation_engine='blast' is not supported with Monte Carlo yet")
    _reject_period(cfg)
    _reject_weather_file(cfg)
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
    execution = backend_provenance(
        settings.execution_backend,
        pv_only=not has_battery and resolved.vehicle is None,
        requested=requested_backend,
    )

    # One "auto" fallback warning per study. A year cache that reuses its PV
    # layer builds nothing that warns, so the study warns from its weather
    # record; a study that built its PV layer and warned does not warn again.
    solver_by_year: dict[int, dict[str, Any]] = {}
    with solar_position_warning_scope():
        if year_cache is None:
            runtime_weather: dict[str, Any] = {}
            if resolved.vehicle is None:
                dc_by_year, temp_by_year, inverter_by_year = _precompute_year_caches(
                    cfg, resolved, settings, runtime_weather=runtime_weather, solver_by_year=solver_by_year
                )
            else:
                weather_by_year = _load_weather_years(cfg, resolved, settings, runtime_weather=runtime_weather)
                dc_by_year, temp_by_year, inverter_by_year = _build_pv_years(
                    cfg, resolved, weather_by_year, solver_by_year=solver_by_year
                )
        else:
            dc_by_year, temp_by_year, inverter_by_year, runtime_weather = year_cache._years_for(
                cfg, resolved, settings, solver_by_year=solver_by_year
            )
            if resolved.vehicle is not None:
                weather_by_year = year_cache._weather_by_year
        warn_solar_position_fallback(
            runtime_weather.get("solar_position_requested", ""), runtime_weather.get("solar_position_method", "")
        )
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
        inverter_by_year,
        has_battery=has_battery,
    )
    vehicle_temperature_by_year = None
    vehicle_schedule = None
    if resolved.vehicle is not None:
        vehicle_temperature_by_year = {
            year: build_battery_temperature_series(
                resolved.vehicle.battery_temperature,
                index=dc_by_year[year].index,
                weather_df=weather,
                indoor_model={"enabled": False},
            )
            for year, weather in weather_by_year.items()
        }
        vehicle_schedule = resolve_vehicle_schedule(
            resolved.vehicle,
            _study_calendar(aligned_by_year),
            timezone=resolved.timezone,
            reference_year=settings.target_year,
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
    study = _Study(
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
        vehicle_temperature_by_year,
        vehicle_schedule,
    )
    if settings.n_procs == 1:
        _initialize_worker(study)
        outputs = [_run_trajectory_index(run_idx) for run_idx in range(settings.n_runs)]
    else:
        with Pool(settings.n_procs, initializer=_initialize_pool_worker, initargs=(study,)) as pool:
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
    currency = resolved.currency
    # Plot labels read the currency from the frame.
    runs_df.attrs["currency"] = currency
    yearly_df = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else None
    exceedance = _yield_exceedance(cfg, resolved, aligned_by_year, settings, years_per_run)
    solver = _solver_summary(solver_by_year)
    return MonteCarloResult(
        runs=runs_df,
        summary=_summarize(runs_df),
        settings=settings,
        available_years=[int(y) for y in available_years],
        yearly=yearly_df,
        yield_exceedance=exceedance,
        solver=solver,
        provenance={
            "breos_version": package_version(),
            "result_schema_version": RESULT_SCHEMA_VERSION,
            # Every money column and summary is in this currency; BREOS does not convert.
            "currency": currency,
            # The config's own request, "auto" by default; settings and the
            # execution block name the backend that ran.
            "resolved_config": {**cfg, "execution_backend": resolved.requested_execution_backend},
            "settings": asdict(settings),
            "available_weather_years": [int(y) for y in available_years],
            "solver": deepcopy(solver),
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
            **(
                {
                    "vehicle": {
                        **vehicle_provenance(resolved.vehicle, vehicle_schedule),
                        "pairing": "weather/load drawn once per project year; persistent trajectory PV factor; shared by R0/R1/R2",
                        "load_scale_applies_to": "house load only",
                    }
                }
                if resolved.vehicle is not None and vehicle_schedule is not None
                else {}
            ),
            **tariff_records(resolved, tariff, reference_tariff, calendar_year=settings.target_year),
            **(
                {"smart_charging": smart_charging_provenance(spec, instructions, tariff)}
                if spec is not None and instructions is not None and tariff is not None
                else {}
            ),
        },
    )
