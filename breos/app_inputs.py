"""Input loading and preparation for App simulations."""

from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, cast

import pandas as pd
from pvlib.location import Location

from breos.app_config import ResolvedAppConfig, resolve_app_config
from breos.pv.horizon import apply_terrain_horizon_profile
from breos.pv.model_options import DEFAULT_SOLAR_POSITION
from breos.solar import (
    PVProductionBreakdown,
    calculate_multi_array_production_breakdown,
    calculate_pv_production_breakdown,
    calculate_pv_production_tracking_breakdown,
)
from breos.utils import get_hours_per_step, remap_datetime_index_years
from breos.weather import AmbiguousWeatherError, fill_leap_day, warn_if_naive_weather_timestamps


@dataclass(frozen=True)
class AppRuntimeDependencies:
    """Runtime callables supplied by breos.app for monkeypatch-friendly tests."""

    load_profile: Callable[..., Any]
    load_weather: Callable[..., pd.DataFrame | None]
    fetch_tmy_weather_data: Callable[..., tuple[pd.DataFrame, dict]]
    resample_to_15min: Callable[..., pd.DataFrame]
    build_battery_temperature_series: Callable[..., pd.Series]


@dataclass(frozen=True)
class PreparedSimulationInputs:
    """Prepared weather, PV, demand, and battery-temperature series."""

    weather: pd.DataFrame
    dc_system_base: pd.Series
    pv_breakdown: PVProductionBreakdown
    load_data: pd.DataFrame
    temperature_series: pd.Series


def _ensure_weather_horizon_metadata(weather: pd.DataFrame) -> None:
    """Give injected or legacy weather an explicit conservative horizon state."""
    metadata = deepcopy(weather.attrs.get("breos_weather_metadata"))
    if not isinstance(metadata, dict):
        metadata = {
            "source": "runtime_dependency_or_unknown",
            "note": "The injected weather provider did not expose source metadata.",
        }
    horizon = metadata.get("horizon")
    if not isinstance(horizon, dict) or horizon.get("status") not in {"applied", "not_applied", "unknown"}:
        metadata["horizon"] = {"status": "unknown", "provider": None, "profile": None}
    else:
        horizon = deepcopy(horizon)
        horizon.setdefault("provider", None)
        horizon.setdefault("profile", None)
        metadata["horizon"] = horizon
    weather.attrs["breos_weather_metadata"] = metadata


def remap_tmy_year(df: pd.DataFrame, target_year: int) -> pd.DataFrame:
    """Remap a TMY DatetimeIndex to target_year.

    A leap target year gets a 29 February copied from 28 February, so a
    non-leap TMY covers the leap year the load profile already covers.
    """
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) == 0:
        return df
    was_tz = idx.tz
    idx_utc = idx.tz_convert("UTC") if was_tz is not None else idx.tz_localize("UTC")
    dominant_year = cast(int, idx_utc.year.value_counts().idxmax())
    offset = target_year - dominant_year
    if offset == 0:
        return fill_leap_day(df)
    weather_metadata = deepcopy(df.attrs.get("breos_weather_metadata"))
    remapped = df.copy()
    remapped.index = idx_utc
    remapped = remap_datetime_index_years(remapped, offset)
    new_idx = remapped.index
    new_idx = new_idx.tz_convert(was_tz) if was_tz is not None else new_idx.tz_localize(None)
    remapped.index = new_idx
    if weather_metadata is not None:
        remapped.attrs["breos_weather_metadata"] = weather_metadata
    return fill_leap_day(remapped)


def _weather_source_label(weather: pd.DataFrame) -> str:
    """Name the weather's origin for an error message: its file, else its source."""
    metadata = weather.attrs.get("breos_weather_metadata")
    if isinstance(metadata, dict):
        if metadata.get("path"):
            return str(metadata["path"])
        if metadata.get("source") and metadata["source"] != "runtime_dependency_or_unknown":
            return str(metadata["source"])
    return "the injected weather provider"


def require_full_year_weather(weather: pd.DataFrame, year: int, freq: str, timezone: str) -> None:
    """Raise unless the weather covers the whole calendar year the App simulates.

    The simulation calendar runs from the first to the last weather row, so
    weather missing its first or last days would quietly simulate a shorter
    year against a full year of economics. A gap in the middle is already an
    error in the PV model; this rejects missing leading and trailing rows.

    The year may be read on three clocks: the weather index's own, UTC, and
    the location's ``timezone``. A PVGIS TMY covers one fixed-offset year,
    CSV weather often one UTC year, and local weather one civil year, and all
    three are complete. The weather must cover the year on at least one of
    them, to within one simulation step at each end, so labels offset from
    the hour by less than a step still count.
    """
    index = weather.index
    if not isinstance(index, pd.DatetimeIndex) or index.empty:
        raise ValueError(f"Weather from {_weather_source_label(weather)} has no timestamped rows")
    step = pd.Timedelta(hours=get_hours_per_step(freq))
    first, last = index.min(), index.max()

    shortfalls = []
    clocks = {str(clock): clock for clock in (index.tz, "UTC", timezone)}
    for clock in clocks.values():
        year_start = pd.Timestamp(f"{year}-01-01", tz=clock)
        year_end = pd.Timestamp(f"{year + 1}-01-01", tz=clock)
        leading = max(first - year_start, pd.Timedelta(0))
        trailing = max(year_end - (last + step), pd.Timedelta(0))
        if leading < step and trailing < step:
            return
        shortfalls.append((max(leading, trailing), leading, trailing, year_start, year_end))

    # Report on the clock the weather comes closest to covering.
    _, leading, trailing, year_start, year_end = min(shortfalls, key=lambda item: item[0])
    missing = []
    if leading >= step:
        missing.append(
            f"the leading {leading} ({int(leading // step)} steps, {year_start} to "
            f"{first.tz_convert(year_start.tz) - step})"
        )
    if trailing >= step:
        missing.append(
            f"the trailing {trailing} ({int(trailing // step)} steps, "
            f"{last.tz_convert(year_end.tz) + step} to {year_end - step})"
        )
    raise ValueError(
        f"Weather from {_weather_source_label(weather)} does not cover the simulated year {year}: "
        f"after restamping onto {year} it runs from {first} to {last}, so {' and '.join(missing)} "
        f"{'is' if len(missing) == 1 else 'are'} missing. The App simulates the whole calendar year "
        "of start_date; supply weather for the full year, or fill the missing rows explicitly."
    )


def load_weather_for_simulation(
    resolved: ResolvedAppConfig,
    freq: str,
    start_year: int,
    deps: AppRuntimeDependencies,
    weather_dir: Path | None = None,
    *,
    horizon_profile: Any = None,
    solar_position: str = DEFAULT_SOLAR_POSITION,
    weather_source: str | None = None,
) -> pd.DataFrame:
    """Load TMY weather, falling back to PVGIS fetch.

    When ``weather_dir`` is not given, a ``weather/`` directory in the
    current working directory is scanned first: a file matching the
    location preset key takes precedence over the PVGIS fetch. Remove or
    rename the directory (or its files) to force a fresh fetch.

    Several matching TMY files are an error rather than a guess;
    ``weather_source`` (the App ``weather_source`` key) picks one by the
    filename's source part. A requested source with no matching file is also
    an error, so it never silently becomes a PVGIS fetch.
    """
    weather = None
    weather_path = weather_dir or Path.cwd() / "weather"

    if resolved.loc_key and weather_path.is_dir():
        try:
            weather = deps.load_weather(
                location=resolved.loc_key,
                data_type="tmy",
                source=weather_source,
                weather_dir=str(weather_path),
            )
        except AmbiguousWeatherError as exc:
            raise ValueError(
                f"Several cached TMY weather files match location {resolved.loc_key!r} in {weather_path}: "
                f"{', '.join(exc.filenames)}. Set the 'weather_source' config key (CLI: --weather-source) "
                f"to one of: {', '.join(exc.sources)}; or leave one matching file in the directory."
            ) from exc

    if weather is None and weather_source is not None:
        raise FileNotFoundError(
            f"'weather_source' is {weather_source!r}, but no cached TMY weather file "
            f"{resolved.loc_key}_tmy_<years>_{weather_source}.csv was found in {weather_path}. "
            "Add the file, or unset 'weather_source' to fetch PVGIS weather."
        )

    if weather is None:
        weather, _ = deps.fetch_tmy_weather_data(
            latitude=resolved.lat,
            longitude=resolved.lon,
            sample_year=start_year,
            freq="h",
            timezone=resolved.timezone,
            use_horizon=horizon_profile is None,
        )

    _ensure_weather_horizon_metadata(weather)
    # Both weather loaders return a DatetimeIndex.
    weather_index = cast(pd.DatetimeIndex, weather.index)
    if weather_index.tz is None:
        warn_if_naive_weather_timestamps(
            weather_index, weather.attrs.get("breos_weather_metadata") or {}, "Local weather"
        )
        weather.index = weather_index.tz_localize("UTC")
    weather = remap_tmy_year(weather, start_year)
    if freq == "15min":
        inferred = pd.infer_freq(cast(pd.DatetimeIndex, weather.index)[:10])
        if inferred and "h" in inferred.lower() and "15" not in inferred:
            # The resampler carries the weather metadata over and adds its own
            # resolution and method fields to it.
            weather = deps.resample_to_15min(weather, latitude=resolved.lat, longitude=resolved.lon)
    require_full_year_weather(weather, start_year, freq, resolved.timezone)

    if horizon_profile is not None:
        weather = apply_terrain_horizon_profile(
            weather,
            Location(resolved.lat, resolved.lon, tz=resolved.timezone),
            horizon_profile,
            freq=freq,
            solar_position=solar_position,
        )

    return weather


def build_dc_system_base(cfg: dict[str, Any], resolved: ResolvedAppConfig, weather: pd.DataFrame) -> pd.Series:
    """Build undegraded system-level DC production for one simulation year."""
    return build_pv_production_breakdown(cfg, resolved, weather).dc_after_losses


def build_pv_production_breakdown(
    cfg: dict[str, Any], resolved: ResolvedAppConfig, weather: pd.DataFrame
) -> PVProductionBreakdown:
    """Build undegraded system-level PV production and loss-stage details."""
    location = Location(resolved.lat, resolved.lon, tz=resolved.timezone)
    freq = cfg["resolution"]
    loss_overrides = cfg["pv_loss_overrides"]
    sky_kwargs = {
        "transposition_model": cfg["transposition_model"],
        "albedo": cfg["albedo"],
        "surface_type": cfg["surface_type"],
        "model_perez": cfg["model_perez"],
        "solar_position": cfg["solar_position"],
        "iam_model": cfg["iam_model"],
        "diffuse_iam": cfg["diffuse_iam"],
        "temperature_model": cfg["temperature_model"],
        "bifacial_model": cfg["bifacial_model"],
        "pvrow_height": cfg["pvrow_height"],
        "pvrow_pitch": cfg["pvrow_pitch"],
    }

    if resolved.pv_arrays:
        return calculate_multi_array_production_breakdown(
            weather_data=weather,
            location=location,
            arrays=resolved.pv_arrays,
            freq=freq,
            loss_overrides=loss_overrides,
            gcr=cfg["gcr"],
            **sky_kwargs,
        )

    if resolved.tracking == "fixed":
        breakdown = calculate_pv_production_breakdown(
            weather_data=weather,
            location=location,
            tilt=resolved.tilt,
            surface_azimuth=resolved.azimuth,
            n_modules=cfg["n_modules"],
            pv_params=resolved.pv_params,
            freq=freq,
            loss_overrides=loss_overrides,
            gcr=cfg["gcr"],
            **sky_kwargs,
        )
    else:
        breakdown = calculate_pv_production_tracking_breakdown(
            weather_data=weather,
            location=location,
            n_modules=cfg["n_modules"],
            tracking=resolved.tracking,
            axis_tilt=cfg["axis_tilt"],
            axis_azimuth=resolved.axis_azimuth,
            max_angle=cfg["max_angle"],
            backtrack=cfg["backtrack"],
            gcr=cfg["gcr"],
            cross_axis_tilt=cfg["cross_axis_tilt"],
            dual_axis_max_tilt=cfg["dual_axis_max_tilt"],
            pv_params=resolved.pv_params,
            freq=freq,
            loss_overrides=loss_overrides,
            **sky_kwargs,
        )
    return breakdown


def load_consumption_profile(
    cfg: dict[str, Any], deps: AppRuntimeDependencies, timezone: str | None = None
) -> pd.DataFrame:
    """Load and scale the configured demand profile.

    Profile rows describe household behavior at legal clock time, so the
    location timezone pins them to local wall clock; the simulation aligns
    load and PV by UTC instant.
    """
    return deps.load_profile(
        profile_type=cfg["load_profile"],
        annual_consumption_kwh=cfg["annual_consumption_kwh"],
        start_date=cfg["start_date"],
        freq=cfg["resolution"],
        num_years=1,
        rlp_directory=cfg["rlp_directory"],
        timezone=timezone or "UTC",
        profile_file=cfg["load_profile_file"],
        profile_column=cfg["load_profile_column"],
        profile_unit=cfg["load_profile_unit"],
    )


def prepare_simulation_inputs(
    cfg: dict[str, Any], resolved: ResolvedAppConfig, deps: AppRuntimeDependencies
) -> PreparedSimulationInputs:
    """Prepare weather, PV, demand, and temperature inputs for the App pipeline."""
    freq = cfg["resolution"]
    start_year = int(cfg["start_date"][:4])
    weather = load_weather_for_simulation(
        resolved,
        freq,
        start_year,
        deps,
        horizon_profile=cfg["horizon_profile"],
        solar_position=cfg["solar_position"],
        weather_source=cfg["weather_source"],
    )
    pv_breakdown = build_pv_production_breakdown(cfg, resolved, weather)
    dc_system_base = pv_breakdown.dc_after_losses
    load_data = load_consumption_profile(cfg, deps, timezone=resolved.timezone)
    temperature_series = deps.build_battery_temperature_series(
        cfg["battery_temperature"],
        index=dc_system_base.index,
        weather_df=weather,
        indoor_model=cfg["battery_indoor_model"],
    )
    return PreparedSimulationInputs(
        weather=weather,
        dc_system_base=dc_system_base,
        pv_breakdown=pv_breakdown,
        load_data=load_data,
        temperature_series=temperature_series,
    )


# Config keys that do not reach prepare_simulation_inputs, directly or through
# the ResolvedAppConfig fields it reads (location, tilt, azimuth, tracking,
# pv_arrays, pv_params). Each was traced and is pinned by a test that changes
# it and compares the prepared inputs. They feed dispatch, degradation,
# valuation, emissions or the backend, all of which run after this stage.
# A key not listed here is part of the cache key, so a new key is safe until
# it is shown not to matter.
INPUT_INDEPENDENT_KEYS: frozenset[str] = frozenset(
    {
        # Battery sizing and dispatch.
        "battery_kwh",
        "battery_min_soc",
        "battery_max_soc",
        "battery_max_charge_power_w",
        "battery_max_discharge_power_w",
        "battery_power_limit_c_rate",
        "battery_eol_percentage",
        "battery_rte",
        "enable_resistance_fade",
        "inverter_efficiency",
        "inverter_loading_ratio",
        # Degradation.
        "calendar_model",
        "degradation_engine",
        "blast_model",
        "pv_degradation_rate",
        "projection_years",
        # Prices, valuation and dispatch strategy.
        "cost_preset",
        "costs",
        "inflation_rate",
        "sell_price_inflation",
        "import_price_escalation",
        "om_escalation",
        "replacement_cost_learning",
        "discount_rate",
        "tariff",
        "smart_charging",
        # Emissions and execution.
        "emissions_country",
        "export_emissions_factor_gco2_kwh",
        "execution_backend",
    }
)

# The block's one cached preparation, by (input key, dependencies, prepare).
_PREPARED_INPUTS_CACHE: ContextVar[dict[tuple[Any, ...], PreparedSimulationInputs] | None] = ContextVar(
    "breos_prepared_inputs_cache", default=None
)


@contextmanager
def reuse_prepared_inputs() -> Iterator[None]:
    """Reuse prepared inputs across the App runs inside the block.

    ``breos sweep`` runs many Apps that often differ only in keys the input
    stage never reads, such as a tariff or a battery size. Inside this block,
    :func:`prepare_simulation_inputs_cached` prepares weather, PV, load and
    battery temperature once and hands each run with the same input
    configuration its own deep copy. It holds one preparation, the latest,
    so memory stays at one run's inputs however many configurations the
    block sees; run the Apps grouped by :func:`input_configuration_key` to
    reuse each preparation fully. The cache lives only for the block. Files
    are read once per preparation, so a file changed meanwhile is not re-read,
    and a warning the input stage raises appears only for the run that
    prepared those inputs.
    """
    token = _PREPARED_INPUTS_CACHE.set({})
    try:
        yield
    finally:
        _PREPARED_INPUTS_CACHE.reset(token)


def _input_cache_key(cfg: dict[str, Any]) -> str | None:
    """The resolved config without the input-independent keys, as canonical JSON.

    None when a value is not plain JSON data (an in-memory frame or series),
    which the key could not represent faithfully; that run is not cached.
    """
    relevant = {key: value for key, value in cfg.items() if key not in INPUT_INDEPENDENT_KEYS}
    try:
        return json.dumps(relevant, sort_keys=True)
    except (TypeError, ValueError):
        return None


def input_configuration_key(config: dict[str, Any]) -> str | None:
    """The input configuration a raw App config resolves to, or None if it cannot be cached.

    Two configs with the same key get the same prepared inputs.
    """
    return _input_cache_key(resolve_app_config(config).cfg)


def prepare_simulation_inputs_cached(
    cfg: dict[str, Any],
    resolved: ResolvedAppConfig,
    deps: AppRuntimeDependencies,
    *,
    prepare: Callable[..., PreparedSimulationInputs] = prepare_simulation_inputs,
) -> PreparedSimulationInputs:
    """:func:`prepare_simulation_inputs`, reused inside :func:`reuse_prepared_inputs`.

    ``resolved`` is derived from ``cfg`` alone, so the resolved config
    without :data:`INPUT_INDEPENDENT_KEYS`, with the runtime dependencies,
    identifies the inputs. Outside the block this prepares afresh, as before.
    ``prepare`` is the preparation to reuse; the App runner passes its own
    reference, which tests replace.
    """
    cache = _PREPARED_INPUTS_CACHE.get()
    key = _input_cache_key(cfg) if cache is not None else None
    if cache is None or key is None:
        return prepare(cfg, resolved, deps)
    entry = (key, deps, prepare)
    try:
        hit = entry in cache
    except TypeError:
        # A replaced dependency that cannot be hashed; prepare afresh.
        return prepare(cfg, resolved, deps)
    if not hit:
        cache.clear()
        cache[entry] = prepare(cfg, resolved, deps)
    # A copy per run, so nothing a run does to its inputs reaches the next.
    return deepcopy(cache[entry])
