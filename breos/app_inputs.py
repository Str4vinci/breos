"""Input loading and preparation for App simulations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from pvlib.location import Location

from breos.app_config import ResolvedAppConfig
from breos.solar import (
    calculate_multi_array_production,
    calculate_pv_production_dc,
    calculate_pv_production_dc_tracking,
)
from breos.utils import remap_datetime_index_years


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
    load_data: pd.Series
    temperature_series: pd.Series


def remap_tmy_year(df: pd.DataFrame, target_year: int) -> pd.DataFrame:
    """Remap a TMY DatetimeIndex to target_year."""
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) == 0:
        return df
    was_tz = idx.tz
    idx_utc = idx.tz_convert("UTC") if was_tz is not None else idx.tz_localize("UTC")
    dominant_year = idx_utc.year.value_counts().idxmax()
    offset = target_year - dominant_year
    if offset == 0:
        return df
    remapped = df.copy()
    remapped.index = idx_utc
    remapped = remap_datetime_index_years(remapped, offset)
    new_idx = remapped.index
    new_idx = new_idx.tz_convert(was_tz) if was_tz is not None else new_idx.tz_localize(None)
    remapped.index = new_idx
    return remapped


def load_weather_for_simulation(
    resolved: ResolvedAppConfig,
    freq: str,
    start_year: int,
    deps: AppRuntimeDependencies,
    weather_dir: Path | None = None,
) -> pd.DataFrame:
    """Load TMY weather, falling back to PVGIS fetch.

    When ``weather_dir`` is not given, a ``weather/`` directory in the
    current working directory is scanned first: a file matching the
    location preset key takes precedence over the PVGIS fetch. Remove or
    rename the directory (or its files) to force a fresh fetch.
    """
    weather = None
    weather_path = weather_dir or Path.cwd() / "weather"

    if resolved.loc_key and weather_path.is_dir():
        weather = deps.load_weather(location=resolved.loc_key, data_type="tmy", weather_dir=str(weather_path))

    if weather is None:
        weather, _ = deps.fetch_tmy_weather_data(
            latitude=resolved.lat,
            longitude=resolved.lon,
            sample_year=start_year,
            freq="h",
            timezone=resolved.timezone,
        )

    if weather.index.tz is None:
        weather.index = weather.index.tz_localize("UTC")
    weather = remap_tmy_year(weather, start_year)

    if freq == "15min":
        inferred = pd.infer_freq(weather.index[:10])
        if inferred and "h" in inferred.lower() and "15" not in inferred:
            weather = deps.resample_to_15min(weather, latitude=resolved.lat, longitude=resolved.lon)

    return weather


def build_dc_system_base(cfg: dict[str, Any], resolved: ResolvedAppConfig, weather: pd.DataFrame) -> pd.Series:
    """Build undegraded system-level DC production for one simulation year."""
    location = Location(resolved.lat, resolved.lon, tz=resolved.timezone)
    freq = cfg["resolution"]
    loss_overrides = cfg["pv_loss_overrides"]
    transposition_model = cfg["transposition_model"]

    if resolved.pv_arrays:
        return calculate_multi_array_production(
            weather_data=weather,
            location=location,
            arrays=resolved.pv_arrays,
            freq=freq,
            loss_overrides=loss_overrides,
            transposition_model=transposition_model,
        )

    if resolved.tracking == "fixed":
        dc_1mod = calculate_pv_production_dc(
            weather_data=weather,
            location=location,
            tilt=resolved.tilt,
            surface_azimuth=resolved.azimuth,
            n_modules=1,
            pv_params=resolved.pv_params,
            freq=freq,
            loss_overrides=loss_overrides,
            transposition_model=transposition_model,
        )
    else:
        dc_1mod = calculate_pv_production_dc_tracking(
            weather_data=weather,
            location=location,
            n_modules=1,
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
            transposition_model=transposition_model,
        )
    return dc_1mod * cfg["n_modules"]


def load_consumption_profile(
    cfg: dict[str, Any], deps: AppRuntimeDependencies, timezone: str | None = None
) -> pd.Series:
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
    )


def prepare_simulation_inputs(
    cfg: dict[str, Any], resolved: ResolvedAppConfig, deps: AppRuntimeDependencies
) -> PreparedSimulationInputs:
    """Prepare weather, PV, demand, and temperature inputs for the App pipeline."""
    freq = cfg["resolution"]
    start_year = int(cfg["start_date"][:4])
    weather = load_weather_for_simulation(resolved, freq, start_year, deps)
    dc_system_base = build_dc_system_base(cfg, resolved, weather)
    load_data = load_consumption_profile(cfg, deps, timezone=resolved.timezone)
    temperature_series = deps.build_battery_temperature_series(
        "weather",
        index=dc_system_base.index,
        weather_df=weather,
    )
    return PreparedSimulationInputs(
        weather=weather,
        dc_system_base=dc_system_base,
        load_data=load_data,
        temperature_series=temperature_series,
    )
