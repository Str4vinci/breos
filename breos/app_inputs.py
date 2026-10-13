"""Input loading and preparation for App simulations."""

from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, NamedTuple, TypeVar, cast

import numpy as np
import pandas as pd
from pvlib.location import Location

from breos.app_config import ResolvedAppConfig, SimulationPeriod, default_module_key, resolve_app_config
from breos.electrical import string_cable_loss, string_limited_dc
from breos.inverter import AcCable, InverterConversion, InverterUnits, ac_derate
from breos.pv.horizon import apply_terrain_horizon_profile
from breos.pv.model_options import DEFAULT_SOLAR_POSITION, configured_pv_model_kwargs
from breos.pv.shading import ModuleMeanShading
from breos.pv_modules import get_module
from breos.solar import (
    PVProductionBreakdown,
    _sum_pv_breakdowns,
    calculate_array_breakdowns,
    calculate_multi_array_production_breakdown,
    calculate_pv_production_breakdown,
    calculate_pv_production_tracking_breakdown,
)
from breos.utils import _has_fixed_utc_offset, get_hours_per_step, remap_datetime_index_years
from breos.weather import (
    WEATHER_METADATA_KEY,
    AmbiguousWeatherError,
    _normalised_horizon_metadata,
    build_battery_temperature_series,
    fill_leap_day,
    read_weather_csv,
    warn_if_naive_weather_timestamps,
)

FrameT = TypeVar("FrameT", pd.DataFrame, pd.Series)


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
    # The inverter's AC rating per step as a fraction of its nameplate, from
    # its temperature derating; None for an inverter without one.
    inverter_ac_derate: pd.Series | None = None
    # The efficiency curve's row position per step, from the operating DC
    # voltage; None where the curve is read at one row.
    inverter_curve_row: pd.Series | None = None
    # With string-level energy and several inverters, each inverter's share
    # of the DC and its own conversion; None otherwise.
    inverter_units: InverterUnits | None = None


def unknown_source_metadata(provider: str) -> dict[str, Any]:
    """The metadata recorded for input whose ``provider`` (weather, load-profile) gave none."""
    return {
        "source": "runtime_dependency_or_unknown",
        "note": f"The injected {provider} provider did not expose source metadata.",
    }


def _ensure_weather_horizon_metadata(weather: pd.DataFrame) -> None:
    """Give injected or legacy weather an explicit conservative horizon state."""
    metadata = deepcopy(weather.attrs.get(WEATHER_METADATA_KEY))
    if not isinstance(metadata, dict):
        metadata = unknown_source_metadata("weather")
    metadata["horizon"] = _normalised_horizon_metadata(metadata.get("horizon"))
    weather.attrs[WEATHER_METADATA_KEY] = metadata


def _leap_day_clock(tz: Any, year: int) -> timezone:
    """The fixed UTC offset that ``tz`` keeps in late February of ``year``."""
    noon = pd.Timestamp(year=year, month=2, day=28, hour=12).tz_localize(
        tz, ambiguous=True, nonexistent="shift_forward"
    )
    return timezone(cast(timedelta, noon.utcoffset()))


def remap_tmy_year(df: pd.DataFrame, target_year: int) -> pd.DataFrame:
    """Remap a TMY DatetimeIndex to target_year.

    A leap target year gets a 29 February copied from 28 February, so a
    non-leap TMY covers the leap year the load profile already covers.

    An index at a fixed UTC offset, such as a PVGIS TMY saved at ``+01:00``,
    is shifted on its own clock, the clock :func:`fill_leap_day` reads. Shifted
    in UTC, its local 1 March 00:00 would land on 29 February of a leap year,
    and the fill would be skipped. A naive index is shifted in UTC.

    An index in a named zone, such as ``Europe/Berlin``, is shifted on the
    fixed offset the zone keeps in late February of the TMY's year (summer
    time in southern DST zones), so its local 29 February is dropped or
    filled as for a fixed-offset index,
    before converting back to the zone. The dominant year is read on the
    zone's own clock, so rows at local New Year move by their local year.
    Every other row keeps its instant shifted by whole years, as in UTC, so
    irradiance stays with the sun. The zone then reads the instants with the
    target year's transitions: the hour its spring change skips has no row,
    and the hour its autumn change repeats has two, as on the App's
    simulation calendar. Differing offset rules between the two years can
    leave a target-year boundary hour missing; the full-year check rejects
    that weather. Remapping cannot correct mislabelled input.
    A shift on the zone's wall clock would instead move
    the hours between the two years' transition dates by an hour against the
    sun.
    """
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) == 0:
        return df
    was_tz = idx.tz
    own_clock = was_tz is not None and _has_fixed_utc_offset(was_tz)
    dominant_year: int | None = None
    if own_clock:
        idx_shift = idx
    elif was_tz is not None:
        # On the zone's late-February offset the local 29 February is a whole
        # day, and the shift moves every other instant as a UTC shift would.
        # The year is the zone's own: rows at local New Year can read as the
        # year before on a February offset that differs from January's.
        dominant_year = cast(int, idx.year.value_counts().idxmax())
        idx_shift = idx.tz_convert(_leap_day_clock(was_tz, dominant_year))
    else:
        idx_shift = idx.tz_localize("UTC")
    if dominant_year is None:
        dominant_year = cast(int, idx_shift.year.value_counts().idxmax())
    offset = target_year - dominant_year
    if offset == 0:
        return fill_leap_day(df)
    weather_metadata = deepcopy(df.attrs.get(WEATHER_METADATA_KEY))
    remapped = df.copy()
    remapped.index = idx_shift
    remapped = remap_datetime_index_years(remapped, offset)
    if weather_metadata is not None:
        remapped.attrs[WEATHER_METADATA_KEY] = weather_metadata
    # Fill on the clock where the day was dropped, before the target zone's
    # transitions can put a neighbouring day's row on local 29 February.
    remapped = fill_leap_day(remapped)
    if not own_clock:
        new_idx = cast(pd.DatetimeIndex, remapped.index)
        new_idx = new_idx.tz_convert(was_tz) if was_tz is not None else new_idx.tz_localize(None)
        remapped.index = new_idx
    return remapped


def _weather_source_label(weather: pd.DataFrame) -> str:
    """Name the weather's origin for an error message: its file, else its source."""
    metadata = weather.attrs.get(WEATHER_METADATA_KEY)
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


def require_period_weather(weather: pd.DataFrame, period: SimulationPeriod, freq: str) -> None:
    """Raise unless the weather has a row at every step of the ``period`` window.

    The window runs from local midnight of ``period.start`` to local midnight
    of ``period.end``, exclusive, on the location's civil calendar. A step
    labelled ``t`` covers ``[t, t + step)``, so the window must hold a whole
    number of steps, its first row must be at its start, and its last row one
    step before its end. Missing leading or trailing rows raise, as a gap in
    the middle does in the PV model.
    """
    index = weather.index
    if not isinstance(index, pd.DatetimeIndex) or index.empty:
        raise ValueError(f"Weather from {_weather_source_label(weather)} has no timestamped rows")
    step = pd.Timedelta(hours=get_hours_per_step(freq))
    start, end = period.start_time, period.end_time
    window = f"the period {period.start.isoformat()} to {period.end.isoformat()} ({start} to {end}, end exclusive)"
    if (end - start) % step:
        raise ValueError(
            f"{window} is not a whole number of {freq!r} steps, so it cannot be simulated at that resolution."
        )
    inside = index[(index >= start) & (index < end)]
    first, last = index.min(), index.max()
    leading = inside[0] - start if len(inside) else end - start
    trailing = end - (inside[-1] + step) if len(inside) else pd.Timedelta(0)
    if leading >= step or trailing >= step:
        missing = []
        if leading >= step:
            missing.append(f"the leading {leading} ({int(leading // step)} steps)")
        if trailing >= step:
            missing.append(f"the trailing {trailing} ({int(trailing // step)} steps)")
        raise ValueError(
            f"Weather from {_weather_source_label(weather)} does not cover {window}: after restamping onto "
            f"{period.start.year} it runs from {first} to {last}, so {' and '.join(missing)} of the window "
            f"{'is' if len(missing) == 1 else 'are'} missing. Supply weather for the whole period, or fill the "
            "missing rows explicitly."
        )
    if leading:
        raise ValueError(
            f"{window} does not start on a step of the weather, whose first row in the window is {inside[0]}. "
            f"The window must start and end on a {freq!r} step."
        )


def cut_to_period(frame: FrameT, period: SimulationPeriod) -> FrameT:
    """The rows of ``frame`` inside the ``period`` window, with its attrs kept."""
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("a 'period' needs timestamped weather and load, to place the window on them")
    cut = frame.loc[(index >= period.start_time) & (index < period.end_time)].copy()
    cut.attrs = deepcopy(frame.attrs)
    return cut


def weather_input_frequency(weather: pd.DataFrame) -> str | None:
    """The step of the weather index as a pandas frequency string, or None."""
    index = cast(pd.DatetimeIndex, weather.index)
    frequency = pd.infer_freq(index[:10]) if len(index) >= 3 else None
    if frequency is None and len(index) >= 2:
        frequency = pd.tseries.frequencies.to_offset(index[1] - index[0]).freqstr
    return frequency


def resample_hourly_weather(
    weather: pd.DataFrame,
    freq: str,
    *,
    latitude: float,
    longitude: float,
    resample: Callable[..., pd.DataFrame],
    irradiance_resampling: str = "auto",
    **resample_kwargs: Any,
) -> pd.DataFrame:
    """Resample hourly weather to 15 minutes for a 15-minute study.

    App, Monte Carlo and the optimizer call this, with ``resample`` normally
    :func:`breos.weather.resample_to_15min`. Weather that is not hourly, or a
    study that is not at 15 minutes, is returned unchanged.
    """
    if freq != "15min":
        return weather
    input_frequency = weather_input_frequency(weather)
    if input_frequency and "h" in input_frequency.lower() and "15" not in input_frequency:
        return resample(
            weather,
            latitude=latitude,
            longitude=longitude,
            irradiance_resampling=irradiance_resampling,
            **resample_kwargs,
        )
    return weather


def load_weather_for_simulation(
    resolved: ResolvedAppConfig,
    freq: str,
    start_year: int,
    deps: AppRuntimeDependencies,
    weather_dir: Path | None = None,
    *,
    horizon_profile: Any = None,
    solar_position: str = DEFAULT_SOLAR_POSITION,
    irradiance_resampling: str = "auto",
    weather_source: str | None = None,
    period: SimulationPeriod | None = None,
    weather_file: str | None = None,
) -> pd.DataFrame:
    """Load TMY weather, falling back to PVGIS fetch.

    The weather must cover the whole calendar year of ``start_year``, or with
    a ``period`` the whole window, which the returned weather is cut to.

    ``weather_file`` (the App ``weather_file`` key) names the file to read,
    relative to the working directory, and nothing is scanned or fetched. It
    must hold at most one year of weather.

    Otherwise, when ``weather_dir`` is not given, a ``weather/`` directory in the
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

    if weather_file is not None:
        weather = read_weather_csv(weather_file)
        _require_one_year_weather(weather)
    elif resolved.loc_key and weather_path.is_dir():
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
            f"{resolved.loc_key}_tmy_<years>_{weather_source}.csv (or .csv.gz) was found in {weather_path}. "
            "Add the file, or unset 'weather_source' to fetch PVGIS weather."
        )

    if weather is None:
        weather, _ = deps.fetch_tmy_weather_data(
            latitude=resolved.lat,
            longitude=resolved.lon,
            sample_year=start_year,
            timezone=resolved.timezone,
            use_horizon=horizon_profile is None,
        )

    _ensure_weather_horizon_metadata(weather)
    # Both weather loaders return a DatetimeIndex.
    weather_index = cast(pd.DatetimeIndex, weather.index)
    if weather_index.tz is None:
        warn_if_naive_weather_timestamps(weather_index, weather.attrs.get(WEATHER_METADATA_KEY) or {}, "Local weather")
        weather.index = weather_index.tz_localize("UTC")
    weather = remap_tmy_year(weather, start_year)
    # The resampler carries the weather metadata over and adds its own
    # resolution and method fields to it.
    weather = resample_hourly_weather(
        weather,
        freq,
        latitude=resolved.lat,
        longitude=resolved.lon,
        resample=deps.resample_to_15min,
        irradiance_resampling=irradiance_resampling,
    )
    if period is None:
        require_full_year_weather(weather, start_year, freq, resolved.timezone)
    else:
        require_period_weather(weather, period, freq)

    if horizon_profile is not None:
        weather = apply_terrain_horizon_profile(
            weather,
            Location(resolved.lat, resolved.lon, tz=resolved.timezone),
            horizon_profile,
            freq=freq,
            solar_position=solar_position,
        )

    if period is not None:
        # Cut after the resampler and the horizon, so each kept row is what
        # a full-year run computes for it.
        weather = cut_to_period(weather, period)
    return weather


def _require_one_year_weather(weather: pd.DataFrame) -> None:
    """Raise unless a ``weather_file`` holds timestamped weather spanning at most one year.

    The App restamps the weather onto the simulated year, so a multi-year
    file would put several years of weather on one calendar.
    """
    index = weather.index
    if not isinstance(index, pd.DatetimeIndex) or index.empty:
        raise ValueError(f"Weather from {_weather_source_label(weather)} has no timestamped rows")
    if index.max() - index.min() >= pd.Timedelta(days=366):
        raise ValueError(
            f"Weather from {_weather_source_label(weather)} runs from {index.min()} to {index.max()}, more than "
            "one year. 'weather_file' takes one year of weather, such as a TMY; sample multi-year historical "
            "weather with breos.montecarlo."
        )


def arrays_with_module_shading(resolved: ResolvedAppConfig) -> list[dict[str, Any]]:
    """The resolved arrays, each whose strings carry module shading with their mean as its ``shading``."""
    arrays = [dict(arr) for arr in resolved.pv_arrays or []]
    electrical = resolved.electrical
    if electrical is None:
        return arrays
    for index, arr in enumerate(arrays):
        strings = tuple(
            s.module_shading for s in electrical.strings if s.array == index and s.module_shading is not None
        )
        if strings:
            arr["shading"] = ModuleMeanShading(strings)
    return arrays


def build_pv_production_breakdown(
    cfg: dict[str, Any], resolved: ResolvedAppConfig, weather: pd.DataFrame
) -> PVProductionBreakdown:
    """Build undegraded system-level PV production and loss-stage details."""
    location = Location(resolved.lat, resolved.lon, tz=resolved.timezone)
    freq = cfg["resolution"]
    loss_overrides = cfg["pv_loss_overrides"]
    model_kwargs = configured_pv_model_kwargs(cfg)

    electrical = resolved.electrical
    string_level = electrical is not None and electrical.fidelity == "string"
    cabled = electrical is not None and any(string.has_cable for string in electrical.strings)
    if string_level or cabled:
        assert electrical is not None
        if resolved.pv_arrays:
            arrays = calculate_array_breakdowns(
                weather_data=weather,
                location=location,
                arrays=arrays_with_module_shading(resolved),
                freq=freq,
                loss_overrides=loss_overrides,
                **model_kwargs,
            )
        else:
            arrays = [_single_array_breakdown(cfg, resolved, weather, location, model_kwargs)]
        total = _sum_pv_breakdowns(arrays) if len(arrays) > 1 else arrays[0]
        if string_level:
            energy = string_limited_dc(electrical, arrays)
            parallel, shading = energy.parallel_mismatch, energy.shading
            return replace(
                total,
                dc_after_losses=energy.dc,
                mppt_limit_loss=energy.limit_loss,
                dc_wiring_loss=energy.wiring_loss if cabled else None,
                module_operating=None,
                shading_mismatch_loss=None if shading is None else shading.total,
                string_shading_mismatch=None if shading is None else shading.strings,
                parallel_mismatch_loss=None if parallel is None else parallel.total,
                input_parallel_mismatch=None if parallel is None else parallel.inputs,
                dc_voltage=energy.voltage,
                inverter_dc=energy.inverter_dc if len(energy.inverter_dc) > 1 else None,
                inverter_dc_voltage=energy.inverter_voltage if len(energy.inverter_voltage) > 1 else None,
            )
        wiring_loss = string_cable_loss(electrical, arrays)
        return replace(
            total,
            dc_after_losses=(total.dc_after_losses - wiring_loss).rename("dc_power_W"),
            dc_wiring_loss=wiring_loss,
            module_operating=None,
        )

    if resolved.pv_arrays:
        return calculate_multi_array_production_breakdown(
            weather_data=weather,
            location=location,
            arrays=resolved.pv_arrays,
            freq=freq,
            loss_overrides=loss_overrides,
            **model_kwargs,
        )
    return _single_array_breakdown(cfg, resolved, weather, location, model_kwargs)


def _single_array_breakdown(
    cfg: dict[str, Any],
    resolved: ResolvedAppConfig,
    weather: pd.DataFrame,
    location: Location,
    model_kwargs: dict[str, Any],
) -> PVProductionBreakdown:
    """The breakdown of a system configured without ``pv_arrays``."""
    freq = cfg["resolution"]
    loss_overrides = cfg["pv_loss_overrides"]

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
            **model_kwargs,
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
            cross_axis_tilt=cfg["cross_axis_tilt"],
            dual_axis_max_tilt=cfg["dual_axis_max_tilt"],
            pv_params=resolved.pv_params,
            freq=freq,
            loss_overrides=loss_overrides,
            **model_kwargs,
        )
    return breakdown


def load_consumption_profile(cfg: dict[str, Any], deps: AppRuntimeDependencies, timezone: str) -> pd.DataFrame:
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
        rlp_directory=cfg["rlp_directory"],
        timezone=timezone,
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
    period = resolved.period
    weather = load_weather_for_simulation(
        resolved,
        freq,
        start_year,
        deps,
        horizon_profile=cfg["horizon_profile"],
        solar_position=cfg["solar_position"],
        irradiance_resampling=cfg["irradiance_resampling"],
        weather_source=cfg["weather_source"],
        period=period,
        weather_file=cfg["weather_file"],
    )
    pv_breakdown = build_pv_production_breakdown(cfg, resolved, weather)
    dc_system_base = pv_breakdown.dc_after_losses
    # The full-year profile is scaled to annual_consumption_kwh first, so a
    # window gets its share of the year, not the annual total.
    load_data = load_consumption_profile(cfg, deps, timezone=resolved.timezone)
    if period is not None:
        load_data = cut_to_period(load_data, period)
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
        inverter_ac_derate=build_inverter_ac_derate(resolved, weather, pd.DatetimeIndex(dc_system_base.index)),
        inverter_curve_row=build_inverter_curve_row(resolved, pv_breakdown),
        inverter_units=build_inverter_units(cfg, resolved, weather, pv_breakdown),
    )


class InverterSteps(NamedTuple):
    """The inverter's per-step inputs for one weather year: its AC rating fraction and its curve row, or None."""

    ac_derate: pd.Series | None
    curve_row: pd.Series | None


def build_inverter_curve_row(resolved: ResolvedAppConfig, pv_breakdown: PVProductionBreakdown) -> pd.Series | None:
    """The efficiency curve's row position at each step's operating DC voltage, or None.

    With string-level energy and a curve of several voltage rows, each step
    reads the curve at the inputs' operating voltage, weighted by the power
    each delivers. A step without PV, when only the battery discharges, keeps
    the row of the strings' STC maximum power voltage.
    """
    voltage = pv_breakdown.dc_voltage
    if voltage is None:
        return None
    electrical = resolved.electrical
    conversion = resolved.inverter_conversion
    if electrical is None or conversion is None or len(electrical.inverters) != 1:
        return None
    curve = electrical.inverters[0].efficiency_curve
    if curve is None or curve.voltages_v is None or len(curve.voltages_v) < 2:
        return None
    values = voltage.to_numpy(dtype=float)
    position = np.interp(values, curve.voltages_v, np.arange(len(curve.voltages_v), dtype=float))
    rows = np.where(values > 0, position, conversion.curve_row)
    return pd.Series(rows, index=voltage.index, name="inverter_curve_row")


def build_inverter_ac_derate(
    resolved: ResolvedAppConfig, weather: pd.DataFrame, index: pd.DatetimeIndex
) -> pd.Series | None:
    """The inverter's AC rating per step as a fraction of its nameplate, or None without temperature derating.

    The ambient is ``electrical_environment.inverter_ambient_temp_c`` when
    stated, for an inverter indoors, and otherwise the weather's air
    temperature, for one outdoors.
    """
    electrical = resolved.electrical
    if electrical is None or len(electrical.inverters) != 1:
        return None
    derating = electrical.inverters[0].temperature_derating
    if derating is None:
        return None
    ambient = build_battery_temperature_series(
        electrical.environment.inverter_ambient_temp_c,
        index=pd.DatetimeIndex(index),
        weather_df=weather,
        indoor_model={"enabled": False},
    )
    return pd.Series(ac_derate(derating, ambient.to_numpy(dtype=float)), index=index, name="inverter_ac_derate")


def build_inverter_units(
    cfg: dict[str, Any], resolved: ResolvedAppConfig, weather: pd.DataFrame, pv_breakdown: PVProductionBreakdown
) -> InverterUnits | None:
    """Each inverter's share of the DC and its own conversion, with string-level energy and several inverters.

    None otherwise. Each inverter converts with its own nameplate, its
    ``inverter_efficiency`` (the App's ``inverter_efficiency`` when unset),
    its efficiency curve read at its own inputs' operating voltage, its
    temperature derating, its own AC cable and its night draw. ``[ac_cable]``
    is the run the inverters share (ADR 0007 D10).
    """
    electrical = resolved.electrical
    if electrical is None or len(electrical.inverters) < 2 or pv_breakdown.inverter_dc is None:
        return None
    assert pv_breakdown.inverter_dc_voltage is not None
    index = pd.DatetimeIndex(pv_breakdown.dc_after_losses.index)
    total = pv_breakdown.dc_after_losses.to_numpy(dtype=float)
    modules = [get_module(arr["module"]) for arr in resolved.pv_arrays] or [resolved.pv_params]
    ambient = None
    conversions, shares, derates, rows = [], {}, {}, {}
    for number, inverter in enumerate(electrical.inverters):
        own = None if inverter.ac_cable is None else AcCable(**dict(inverter.ac_cable))
        curve = inverter.efficiency_curve
        voltage = None
        if curve is not None:
            weighted = [
                (s.modules * s.parallel * modules[s.array].Mpp, s.modules * modules[s.array].Vmp)
                for s in electrical.strings
                if s.inverter == number and s.array < len(modules)
            ]
            power = sum(p for p, _ in weighted)
            voltage = sum(p * v for p, v in weighted) / power if power > 0 else None
        conversion = InverterConversion.build(
            curve, voltage_v=voltage, ac_cable=own, night_consumption_w=inverter.night_consumption_w
        )
        conversions.append(conversion)
        dc = pv_breakdown.inverter_dc[number].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            shares[number] = np.where(total > 0, dc / total, 0.0)
        derates[number] = np.ones(len(index))
        if inverter.temperature_derating is not None:
            if ambient is None:
                ambient = build_battery_temperature_series(
                    electrical.environment.inverter_ambient_temp_c,
                    index=index,
                    weather_df=weather,
                    indoor_model={"enabled": False},
                ).to_numpy(dtype=float)
            derates[number] = ac_derate(inverter.temperature_derating, ambient)
        rows[number] = np.full(len(index), conversion.curve_row)
        if curve is not None and curve.voltages_v is not None and len(curve.voltages_v) > 1:
            values = pv_breakdown.inverter_dc_voltage[number].to_numpy(dtype=float)
            position = np.interp(values, curve.voltages_v, np.arange(len(curve.voltages_v), dtype=float))
            rows[number] = np.where(values > 0, position, conversion.curve_row)
    return InverterUnits(
        ac_capacity_w=tuple(float(i.nominal_power_w) for i in electrical.inverters),
        # Unstated efficiencies stay None: the run's inverter_efficiency is not part of
        # the prepared inputs' cache key, so the dispatch supplies it.
        efficiency=tuple(
            None if i.inverter_efficiency is None else float(i.inverter_efficiency) for i in electrical.inverters
        ),
        conversions=tuple(conversions),
        shares=pd.DataFrame(shares, index=index),
        ac_derate=pd.DataFrame(derates, index=index),
        curve_rows=pd.DataFrame(rows, index=index),
        feeder_cable_k_per_w=0.0 if resolved.ac_cable is None else resolved.ac_cable.loss_coefficient_per_w,
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
        "battery_enable_replacement",
        "battery_allow_terminal_replacement",
        "battery_replacement_min_remaining_years",
        "battery_skipped_replacement_action",
        "battery_rte",
        "enable_resistance_fade",
        "inverter_efficiency",
        "inverter_loading_ratio",
        "inverter_ac_rating_kw",
        # Degradation.
        "calendar_model",
        "degradation_engine",
        "blast_model",
        "pv_degradation_rate",
        "projection_years",
        # Prices, valuation and dispatch strategy.
        "cost_preset",
        "costs",
        "currency",
        "inflation_rate",
        "sell_price_inflation",
        "import_price_escalation",
        "om_escalation",
        "replacement_cost_learning",
        "discount_rate",
        "tariff",
        # The no-system reference, with its annual network credit, prices
        # only the household load.
        "reference_tariff",
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


def config_cache_key(cfg: dict[str, Any], independent_keys: frozenset[str]) -> str | None:
    """A resolved config without ``independent_keys``, as canonical JSON, with the parameters of its modules.

    A module is named by its catalogue key, and a registered module's
    parameters can change under the same name, so the key carries them.
    None when a value is not plain JSON data (an in-memory frame or series),
    which the key could not represent faithfully; that run is not cached.
    """
    relevant = {key: value for key, value in cfg.items() if key not in independent_keys}
    try:
        return json.dumps([relevant, _module_parameters(cfg)], sort_keys=True)
    except (TypeError, ValueError):
        return None


def _module_parameters(cfg: dict[str, Any]) -> list[Any]:
    """The parameters of each module the config names, by name, as plain JSON values."""
    default = cfg.get("pv_module") or default_module_key()
    arrays = [arr for arr in cfg.get("pv_arrays") or [] if isinstance(arr, dict)]
    names = {default} | {arr.get("module", default) for arr in arrays}
    out = []
    for name in sorted(n for n in names if isinstance(n, str)):
        try:
            module = get_module(name)
        except (KeyError, ValueError):
            continue
        out.append([name, json.loads(json.dumps(asdict(module), default=str))])
    return out


def _input_cache_key(cfg: dict[str, Any], resolved: ResolvedAppConfig | None = None) -> str | None:
    """The resolved config without the input-independent keys, as canonical JSON, or None.

    ``resolved`` carries what an App took from the module catalogue when it
    was built, which preparation uses even if the catalogue changed since:
    each array's module, the bypass diodes its shading is bound to, and the
    inverter curve's row for steps without PV. The catalogue as it was then
    stands for all of them.
    """
    key = config_cache_key(cfg, INPUT_INDEPENDENT_KEYS)
    if key is None or resolved is None:
        return key
    return json.dumps([key, resolved.module_catalogue], sort_keys=True)


def input_configuration_key(config: dict[str, Any]) -> str | None:
    """The input configuration a raw App config resolves to, or None if it cannot be cached.

    Two configs with the same key get the same prepared inputs.
    """
    resolved = resolve_app_config(config)
    return _input_cache_key(resolved.cfg, resolved)


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
    key = _input_cache_key(cfg, resolved) if cache is not None else None
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
