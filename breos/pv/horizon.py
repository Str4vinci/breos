"""Terrain-horizon profile validation and direct-beam shading."""

from __future__ import annotations

import math
from copy import deepcopy
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd
from pvlib.location import Location

from breos.pv.model_options import DEFAULT_SOLAR_POSITION, resolve_solar_position_method
from breos.utils import get_hours_per_step


def normalise_horizon_profile(profile: Any) -> list[list[float]] | None:
    """Validate and sort ``[azimuth, elevation]`` terrain-horizon pairs.

    Azimuths use pvlib's convention: degrees clockwise from north. The profile
    is circular, so callers do not need to repeat the first point at 360
    degrees. ``360`` is accepted as an alias for ``0`` but cannot be supplied
    together with ``0``.
    """
    if profile is None:
        return None
    if not isinstance(profile, (list, tuple)) or isinstance(profile, (str, bytes)):
        raise TypeError("'horizon_profile' must be a list of [azimuth, elevation] pairs")
    if len(profile) < 2:
        raise ValueError("'horizon_profile' must contain at least two azimuth/elevation pairs")

    points: list[list[float]] = []
    seen_azimuths: set[float] = set()
    for index, point in enumerate(profile):
        if not isinstance(point, (list, tuple)) or isinstance(point, (str, bytes)) or len(point) != 2:
            raise TypeError(f"'horizon_profile[{index}]' must be an [azimuth, elevation] pair")
        azimuth, elevation = point
        if isinstance(azimuth, bool) or not isinstance(azimuth, Real) or not math.isfinite(float(azimuth)):
            raise TypeError(f"'horizon_profile[{index}][0]' azimuth must be a finite number")
        if isinstance(elevation, bool) or not isinstance(elevation, Real) or not math.isfinite(float(elevation)):
            raise TypeError(f"'horizon_profile[{index}][1]' elevation must be a finite number")

        azimuth_value = float(azimuth)
        elevation_value = float(elevation)
        if not 0.0 <= azimuth_value <= 360.0:
            raise ValueError(f"'horizon_profile[{index}][0]' azimuth must be between 0 and 360")
        if not -90.0 <= elevation_value <= 90.0:
            raise ValueError(f"'horizon_profile[{index}][1]' elevation must be between -90 and 90")

        azimuth_value %= 360.0
        if azimuth_value in seen_azimuths:
            raise ValueError(f"'horizon_profile' contains duplicate azimuth {azimuth_value:g}")
        seen_azimuths.add(azimuth_value)
        points.append([azimuth_value, elevation_value])

    points.sort(key=lambda point: point[0])
    return points


def interpolate_horizon_elevation(profile: list[list[float]], solar_azimuth: Any) -> np.ndarray:
    """Linearly interpolate a circular horizon at one or more azimuths."""
    points = np.asarray(profile, dtype=float)
    azimuths = points[:, 0]
    elevations = points[:, 1]
    extended_azimuths = np.concatenate(([azimuths[-1] - 360.0], azimuths, [azimuths[0] + 360.0]))
    extended_elevations = np.concatenate(([elevations[-1]], elevations, [elevations[0]]))
    return np.interp(np.mod(np.asarray(solar_azimuth, dtype=float), 360.0), extended_azimuths, extended_elevations)


def _weather_column(weather: pd.DataFrame, lower: str, upper: str) -> str:
    if lower in weather.columns:
        return lower
    if upper in weather.columns:
        return upper
    raise ValueError(f"weather_data must contain '{lower}' or '{upper}'")


def _solar_position_at_labels(
    location: Location,
    index: pd.DatetimeIndex,
    freq: str,
    solar_position: str,
) -> tuple[pd.DataFrame, str]:
    method = resolve_solar_position_method(solar_position)
    evaluation_index = index
    if method == "mid-interval":
        evaluation_index = index + pd.Timedelta(hours=get_hours_per_step(freq) / 2.0)
    solarpos = location.get_solarposition(times=evaluation_index)
    solarpos.index = index
    return solarpos, method


def apply_terrain_horizon_profile(
    weather: pd.DataFrame,
    location: Location,
    profile: Any,
    *,
    freq: str,
    solar_position: str = DEFAULT_SOLAR_POSITION,
) -> pd.DataFrame:
    """Apply binary far-horizon shading to direct irradiance.

    The provider metadata must explicitly say that no horizon was applied.
    Direct normal irradiance is zeroed while the sun is on or below the
    interpolated terrain line. The corresponding direct-horizontal component
    is removed from GHI; DHI is retained because this v1 profile models far-
    horizon beam obstruction, not diffuse sky-view loss.
    """
    normalised = normalise_horizon_profile(profile)
    if normalised is None:
        return weather
    if not isinstance(weather.index, pd.DatetimeIndex):
        raise ValueError("weather_data must have a DatetimeIndex")

    metadata = deepcopy(weather.attrs.get("breos_weather_metadata"))
    horizon = metadata.get("horizon") if isinstance(metadata, dict) else None
    status = horizon.get("status") if isinstance(horizon, dict) else "unknown"
    if status == "applied":
        provider = horizon.get("provider") or "an upstream provider"
        raise ValueError(
            "Cannot apply 'horizon_profile': weather already has terrain-horizon shading "
            f"from {provider!r}, which would double-count it. Use weather fetched with use_horizon=False."
        )
    if status != "not_applied":
        raise ValueError(
            "Cannot safely apply 'horizon_profile': the weather horizon status is unknown. "
            "Use provenance-marked unshaded weather or remove the legacy local weather file "
            "so BREOS can fetch fresh PVGIS data with use_horizon=False."
        )

    ghi_column = _weather_column(weather, "ghi", "GHI")
    dni_column = _weather_column(weather, "dni", "DNI")
    dhi_column = _weather_column(weather, "dhi", "DHI")
    solarpos, position_method = _solar_position_at_labels(location, weather.index, freq, solar_position)
    if "apparent_elevation" in solarpos:
        solar_elevation = np.asarray(solarpos["apparent_elevation"], dtype=float)
    else:
        solar_elevation = 90.0 - np.asarray(solarpos["apparent_zenith"], dtype=float)
    horizon_elevation = interpolate_horizon_elevation(normalised, solarpos["azimuth"])

    result = weather.copy()
    dni = np.nan_to_num(result[dni_column].to_numpy(dtype=float), nan=0.0)
    ghi = np.nan_to_num(result[ghi_column].to_numpy(dtype=float), nan=0.0)
    dhi = np.nan_to_num(result[dhi_column].to_numpy(dtype=float), nan=0.0)
    shaded = np.isfinite(solar_elevation) & np.isfinite(horizon_elevation) & (solar_elevation <= horizon_elevation)
    shaded &= dni > 0.0

    zenith = np.asarray(solarpos["apparent_zenith"], dtype=float)
    direct_horizontal = dni * np.clip(np.cos(np.radians(zenith)), 0.0, None)
    result.loc[shaded, dni_column] = 0.0
    result.loc[shaded, ghi_column] = np.maximum(dhi[shaded], ghi[shaded] - direct_horizontal[shaded])

    profile_metadata = {
        "type": "azimuth_elevation_pairs",
        "points": normalised,
        "interpolation": "linear_circular",
        "beam_shading": "binary",
        "diffuse_shading": "not_modeled",
        "solar_position": position_method,
        "shaded_timesteps": int(np.count_nonzero(shaded)),
    }
    metadata = deepcopy(metadata)
    metadata["horizon"] = {"status": "applied", "provider": "breos", "profile": profile_metadata}
    result.attrs["breos_weather_metadata"] = metadata
    return result
