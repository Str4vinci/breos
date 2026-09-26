"""
Weather data fetching and processing module.

This module handles:
- Fetching TMY (Typical Meteorological Year) data from PVGIS
- Fetching historical weather data from Open-Meteo
- Converting between hourly and 15-minute resolutions using Makima interpolation
"""

import hashlib
import json
import logging
import os
import re
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pvlib
from pvlib.location import Location
from scipy.interpolate import Akima1DInterpolator

from breos.utils import _datetime_index_seconds, get_hours_per_step, is_leap_year, safe_path_slug

logger = logging.getLogger(__name__)

WEATHER_METADATA_KEY = "breos_weather_metadata"
_WEATHER_METADATA_KEY = WEATHER_METADATA_KEY
_WEATHER_METADATA_SCHEMA_VERSION = 1


def _unknown_horizon_metadata(provider: str | None = None) -> dict[str, str | None]:
    """Return the conservative horizon state for weather of unknown provenance."""
    return {"status": "unknown", "provider": provider, "profile": None}


def _weather_metadata_sidecar_path(filepath: str | os.PathLike[str]) -> Path:
    """Return the stable metadata sidecar path for a weather CSV."""
    return Path(f"{os.fspath(filepath)}.metadata.json")


def _weather_file_sha256(filepath: str | os.PathLike[str]) -> str:
    with open(filepath, "rb") as weather_file:
        return hashlib.file_digest(weather_file, "sha256").hexdigest()


def _json_default(value: Any) -> Any:
    """Convert common scientific scalar types used by provider metadata."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_weather_csv(weather: pd.DataFrame, filepath: str | os.PathLike[str]) -> None:
    """Write a weather CSV and its content-bound provenance sidecar."""
    weather.to_csv(filepath)
    payload = {
        "schema_version": _WEATHER_METADATA_SCHEMA_VERSION,
        "weather_sha256": _weather_file_sha256(filepath),
        _WEATHER_METADATA_KEY: deepcopy(weather.attrs.get(_WEATHER_METADATA_KEY, {})),
    }
    sidecar_path = _weather_metadata_sidecar_path(filepath)
    with open(sidecar_path, "w", encoding="utf-8") as sidecar:
        json.dump(payload, sidecar, indent=2, sort_keys=True, default=_json_default)
        sidecar.write("\n")


def _load_weather_metadata_sidecar(filepath: str | os.PathLike[str], weather_sha256: str) -> dict[str, Any] | None:
    """Load metadata only when its sidecar schema and CSV digest are valid."""
    sidecar_path = _weather_metadata_sidecar_path(filepath)
    if not sidecar_path.is_file():
        return None

    try:
        with open(sidecar_path, encoding="utf-8") as sidecar:
            payload = json.load(sidecar)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable weather metadata sidecar %s: %s", sidecar_path, exc)
        return None

    if not isinstance(payload, dict) or payload.get("schema_version") != _WEATHER_METADATA_SCHEMA_VERSION:
        logger.warning("Ignoring weather metadata sidecar with unsupported schema: %s", sidecar_path)
        return None
    if payload.get("weather_sha256") != weather_sha256:
        logger.warning("Ignoring weather metadata sidecar whose CSV digest does not match: %s", sidecar_path)
        return None

    metadata = payload.get(_WEATHER_METADATA_KEY)
    if not isinstance(metadata, dict):
        logger.warning("Ignoring weather metadata sidecar without a metadata object: %s", sidecar_path)
        return None
    return metadata


def warn_if_naive_weather_timestamps(timestamps, metadata: dict[str, Any], label: str) -> None:
    """Log a warning when weather timestamps carry no timezone BREOS knows of.

    Naive weather timestamps are read as UTC. That is right for files BREOS
    wrote itself, whose metadata records ``timestamp_timezone``, but a file on
    a local clock would shift the sun by its UTC offset.
    """
    if getattr(timestamps, "tz", None) is None and not metadata.get("timestamp_timezone"):
        logger.warning(
            "%s has timestamps without a timezone; BREOS reads them as UTC. If they are local clock "
            "times, the sun is shifted by the UTC offset: write them with their offset "
            "(for example 2025-01-01T00:00+00:00).",
            label,
        )


def weather_metadata(weather: pd.DataFrame) -> dict[str, Any]:
    """Return a detached copy of BREOS weather provenance."""
    metadata = weather.attrs.get(WEATHER_METADATA_KEY, {})
    return deepcopy(metadata) if isinstance(metadata, dict) else {}


def weather_file_metadata(filepath: str | os.PathLike[str]) -> dict[str, Any]:
    """Return validated sidecar metadata plus the bound file path and digest."""
    path = os.path.abspath(os.fspath(filepath))
    sha256 = _weather_file_sha256(path)
    metadata = deepcopy(_load_weather_metadata_sidecar(path, sha256) or {})
    if metadata.get("source") == "OpenMeteo_historical" and metadata.get("radiation_time_basis") == "instant":
        metadata.setdefault("timestamp_label_basis", "instant")
        metadata.setdefault("timestamp_timezone", "GMT")
        metadata.setdefault("irradiance_time_offset_hours", 0.0)
    metadata.update({"path": path, "sha256": sha256})
    return metadata


def read_weather_csv(filepath: str | os.PathLike[str], *, index_col: int | str = 0, utc: bool = True) -> pd.DataFrame:
    """Read a weather CSV and restore its content-bound metadata sidecar."""
    path = os.path.abspath(os.fspath(filepath))
    frame = pd.read_csv(path, index_col=index_col)
    frame.index = pd.to_datetime(frame.index, utc=utc)
    frame.attrs[WEATHER_METADATA_KEY] = weather_file_metadata(path)
    return frame


def _representative_time_offset(
    metadata: dict[str, Any], step: pd.Timedelta, *, require_metadata: bool
) -> pd.Timedelta:
    basis = metadata.get("radiation_time_basis")
    if basis == "instant":
        return pd.Timedelta(hours=float(metadata.get("irradiance_time_offset_hours", 0.0)))
    if basis == "interval_mean":
        label_basis = metadata.get("timestamp_label_basis")
        if label_basis == "left":
            return step / 2
        if label_basis == "right":
            return -step / 2
        raise ValueError("interval-mean weather metadata requires timestamp_label_basis='left' or 'right'")
    if require_metadata:
        raise ValueError(
            "solar_position='weather' requires radiation_time_basis metadata ('instant' or 'interval_mean')"
        )
    return pd.Timedelta(0)


def weather_representative_time_offset(weather: pd.DataFrame, freq: str) -> pd.Timedelta:
    """Return the solar-position offset implied by the weather timestamps."""
    step = pd.Timedelta(hours=get_hours_per_step(freq))
    return _representative_time_offset(weather_metadata(weather), step, require_metadata=True)


def relabel_right_labeled_interval_means(weather: pd.DataFrame) -> pd.DataFrame:
    """Move right-labeled interval means to the start of their source interval."""
    metadata = weather_metadata(weather)
    if metadata.get("radiation_time_basis") != "interval_mean" or metadata.get("timestamp_label_basis") != "right":
        return weather.copy()
    if len(weather.index) < 2:
        raise ValueError("right-labeled interval weather needs at least two timestamps")
    intervals = weather.index[1:] - weather.index[:-1]
    step = intervals[0]
    if not np.all(intervals == step):
        raise ValueError("right-labeled interval weather requires a regular index")
    relabeled = weather.copy()
    relabeled.index = relabeled.index - step
    metadata["timestamp_label_basis"] = "left"
    metadata["source_timestamp_label_basis"] = "right"
    metadata["timestamp_relabel_shift_seconds"] = -float(step.total_seconds())
    relabeled.attrs[WEATHER_METADATA_KEY] = metadata
    return relabeled


def _relabel_weather_date_column(
    weather: pd.DataFrame, metadata: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply interval-label metadata to a frame whose timestamps are in ``date``."""
    indexed = weather.set_index("date")
    indexed.attrs[WEATHER_METADATA_KEY] = deepcopy(metadata)
    relabeled = relabel_right_labeled_interval_means(indexed)
    return relabeled.reset_index(), weather_metadata(relabeled)


# Optional imports for API calls
try:
    import openmeteo_requests
    import requests_cache
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    HAS_OPENMETEO = True
except ImportError:
    HAS_OPENMETEO = False


def parse_weather_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    Parse a weather filename following the convention:
    {location}_{type}_{yearstart}_{yearend}_{source}.csv

    Examples:
        porto_tmy_2005_2023_pvgis-sarah3.csv
        porto_historical_2005_2024_openmeteo.csv
        lisbon_tmy_2014_nsrdb.csv

    Returns:
        Dict with keys: location, type, year_start, year_end, source
        Returns None if filename doesn't match the convention.
    """
    basename = os.path.basename(filename)
    if not basename.endswith(".csv"):
        return None

    name = basename[:-4]  # strip .csv

    # Pattern: location_type_yearstart_yearend_source
    # Source may contain hyphens (e.g., pvgis-sarah3)
    match = re.match(r"^(.+)_(tmy|historical)_(\d{4})_(\d{4})_([\w-]+)$", name)
    if match:
        return {
            "location": match.group(1),
            "type": match.group(2),
            "year_start": match.group(3),
            "year_end": match.group(4),
            "source": match.group(5),
        }

    # Pattern without year_end: location_type_year_source (e.g., lisbon_tmy_2014_nsrdb)
    match = re.match(r"^(.+)_(tmy|historical)_(\d{4})_([\w-]+)$", name)
    if match:
        return {
            "location": match.group(1),
            "type": match.group(2),
            "year_start": match.group(3),
            "year_end": match.group(3),
            "source": match.group(4),
        }

    return None


def load_weather(
    location: str,
    data_type: Optional[str] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    source: Optional[str] = None,
    weather_dir: str = "weather/",
) -> Optional[pd.DataFrame]:
    """
    Smart weather loading: scan local files for matching weather data.

    Searches the weather directory for files matching the naming convention,
    filters by location/type/source, and checks date coverage. If a file
    covers the requested range (e.g., requesting 2008-2010 and a 2005-2024
    file exists), subsets it automatically.

    Args:
        location: Location name (e.g., 'porto', 'lisbon')
        data_type: 'tmy' or 'historical' (None = any)
        start_year: Start year for date coverage check
        end_year: End year for date coverage check
        source: Data source filter (e.g., 'openmeteo', 'pvgis-sarah3')
        weather_dir: Directory to scan for weather files

    Returns:
        DataFrame if a matching file is found, None otherwise.
    """
    if not os.path.isdir(weather_dir):
        return None

    candidates = []
    for fname in os.listdir(weather_dir):
        parsed = parse_weather_filename(fname)
        if parsed is None:
            continue
        if parsed["location"] != location:
            continue
        if data_type is not None and parsed["type"] != data_type:
            continue
        if source is not None and parsed["source"] != source:
            continue
        parsed["filepath"] = os.path.join(weather_dir, fname)
        candidates.append(parsed)

    if not candidates:
        return None

    # If date range is specified, filter by coverage
    if start_year is not None and end_year is not None:
        covered = []
        for c in candidates:
            file_start = int(c["year_start"])
            file_end = int(c["year_end"])
            if file_start <= start_year and file_end >= end_year:
                covered.append(c)
            elif c["type"] == "tmy":
                # TMY files don't need date coverage — they represent a typical year
                covered.append(c)
        candidates = covered if covered else candidates

    # Prefer the first match (could be refined with priority logic)
    best = candidates[0]
    filepath = best["filepath"]

    logger.info("Found local weather file: %s", filepath)

    df = pd.read_csv(filepath, index_col=0, parse_dates=True)

    # Parse datetime index if it didn't work from index_col=0
    if not isinstance(df.index, pd.DatetimeIndex):
        # Try converting the existing index (handles timezone-aware strings)
        try:
            df.index = pd.to_datetime(df.index, utc=True)
        except (ValueError, TypeError):
            # Fall back to looking for named datetime columns
            df = pd.read_csv(filepath)
            for col_name in ["date", "time", "Datetime"]:
                if col_name in df.columns:
                    df[col_name] = pd.to_datetime(df[col_name])
                    df.set_index(col_name, inplace=True)
                    break

    # Subset by year range for historical data
    if best["type"] == "historical" and start_year is not None and end_year is not None:
        file_start = int(best["year_start"])
        file_end = int(best["year_end"])
        if file_start < start_year or file_end > end_year:
            mask = (df.index.year >= start_year) & (df.index.year <= end_year)
            df = df.loc[mask]
            logger.info("Subset to %s-%s (%d rows)", start_year, end_year, len(df))

    path = os.path.abspath(filepath)
    sha256 = _weather_file_sha256(path)
    persisted_metadata = _load_weather_metadata_sidecar(path, sha256)
    metadata = deepcopy(persisted_metadata) if persisted_metadata is not None else {}
    upstream_source = metadata.get("source")
    horizon = metadata.get("horizon")
    if not isinstance(horizon, dict) or horizon.get("status") not in {"applied", "not_applied", "unknown"}:
        horizon = _unknown_horizon_metadata()
    else:
        horizon = deepcopy(horizon)
        horizon.setdefault("provider", None)
        horizon.setdefault("profile", None)

    metadata.update(
        {
            "source": "local_file",
            "path": path,
            "sha256": sha256,
            "parsed_filename": {key: value for key, value in best.items() if key != "filepath"},
            "horizon": horizon,
        }
    )
    if persisted_metadata is not None:
        if upstream_source is not None:
            metadata["upstream_source"] = upstream_source
        metadata["metadata_sidecar"] = str(_weather_metadata_sidecar_path(path))
    df.attrs[_WEATHER_METADATA_KEY] = metadata

    return df


def fill_leap_day(weather: pd.DataFrame) -> pd.DataFrame:
    """Give one year of weather on a leap-year calendar its 29 February.

    A TMY has 8,760 hours, so restamped onto a leap year it has no 29
    February and 1 March follows 28 February directly. The load profile fills
    that day with a copy of 28 February, and so does this, so weather, PV,
    battery temperature and load all cover the leap year's 8,784 hours. The
    day is taken on the index's own clock, which for a PVGIS TMY is the
    location's fixed offset. The weather metadata records the fill.

    A frame that is not in a leap year, or already has 29 February, is
    returned unchanged.
    """
    index = weather.index
    if not isinstance(index, pd.DatetimeIndex) or index.empty:
        return weather
    year = int(index.year.value_counts().idxmax())
    february = index.month == 2
    if not is_leap_year(year) or (february & (index.day == 29)).any():
        return weather
    source = weather[february & (index.day == 28) & (index.year == year)]
    if source.empty:
        return weather

    leap_day = source.copy()
    leap_day.index = source.index + pd.Timedelta(days=1)
    filled = pd.concat([weather, leap_day]).sort_index()
    filled.attrs = deepcopy(weather.attrs)
    metadata = filled.attrs.get(_WEATHER_METADATA_KEY)
    if isinstance(metadata, dict):
        metadata["leap_day"] = {"year": year, "filled_from": f"{year}-02-28"}
    return filled


def fetch_tmy_weather_data(
    latitude: float,
    longitude: float,
    sample_year: Optional[int] = 2025,
    freq: str = "h",
    timezone: Optional[str] = None,
    save_to_file: bool = False,
    use_horizon: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Fetch Typical Meteorological Year (TMY) weather data from PVGIS.

    Args:
        latitude: Latitude of the location
        longitude: Longitude of the location
        sample_year: Year to use for index (default: 2025). Set to None to keep original TMY index.
            For a leap year the TMY is fetched on the preceding year's
            calendar, restamped, and given a 29 February copied from 28
            February (see :func:`fill_leap_day`).
        freq: Frequency for output data ('h' for hourly, '15min' for 15-minute)
        timezone: Timezone string used to determine the location's whole-hour
            UTC offset (offset taken at Jan 1 of sample_year, i.e. standard
            time for northern-hemisphere locations). Auto-detected if None.
            Fractional-hour offsets are rejected because pvlib's PVGIS TMY
            row-roll interface accepts whole hours only.
        save_to_file: Whether to save the data to CSV
        use_horizon: Whether PVGIS should apply its terrain-horizon profile.
            Defaults to True, preserving the historical BREOS behavior.

    Returns:
        Tuple of (tmy_data DataFrame, metadata dict). When sample_year is set,
        the index is fixed-offset local time starting at local midnight of
        Jan 1; rows are rolled (not relabeled) so each timestamp remains the
        correct UTC instant for its irradiance values.

    Raises:
        ValueError: If the selected timezone has a fractional-hour UTC offset.
    """
    roll_utc_offset = None
    coerce_year = sample_year
    if sample_year is not None:
        # pvlib coerces the 8,760 TMY hours onto one calendar year, which
        # must not be a leap year. Fetch a leap year on the preceding year's
        # calendar and restamp it below.
        if is_leap_year(sample_year):
            coerce_year = sample_year - 1

        # Auto-detect timezone if not provided
        if timezone is None:
            from timezonefinder import TimezoneFinder

            tf = TimezoneFinder()
            timezone = tf.timezone_at(lat=latitude, lng=longitude)

        utc_offset = pd.Timestamp(f"{sample_year}-01-01", tz=timezone).utcoffset()
        offset_hours = utc_offset.total_seconds() / 3600
        if not float(offset_hours).is_integer():
            raise ValueError(
                "PVGIS TMY coercion cannot safely roll a fractional-hour timezone "
                f"({timezone}: UTC{offset_hours:+g}). Set sample_year=None to keep the "
                "provider's UTC index, or supply separately prepared local weather."
            )
        roll_utc_offset = int(offset_hours)

    # PVGIS returns UTC-ordered rows. roll_utc_offset/coerce_year make pvlib
    # roll the data so the series starts at local midnight of sample_year
    # while keeping each row's timestamp the correct UTC instant — never
    # relabel the UTC-ordered rows with local-time labels.
    tmy_data, metadata = pvlib.iotools.get_pvgis_tmy(
        latitude,
        longitude,
        outputformat="json",
        usehorizon=use_horizon,
        map_variables=True,
        url="https://re.jrc.ec.europa.eu/api/v5_3/",
        timeout=120,
        roll_utc_offset=roll_utc_offset,
        coerce_year=coerce_year,
    )

    irradiance_offset = float(metadata.get("inputs", {}).get("location", {}).get("irradiance_time_offset", 0.0))
    tmy_data.attrs[_WEATHER_METADATA_KEY] = {
        "source": "PVGIS_TMY",
        "api_metadata": metadata,
        "raw_radiation_variables": ["G(h)", "Gb(n)", "Gd(h)"],
        "stored_radiation_variables": ["ghi", "dni", "dhi"],
        "radiation_time_basis": "instant",
        "timestamp_label_basis": "provider_hour",
        "timestamp_timezone": "UTC",
        "irradiance_time_offset_hours": irradiance_offset,
        "horizon": {
            "status": "applied" if use_horizon else "not_applied",
            "provider": "pvgis",
            "profile": "provider_default" if use_horizon else None,
        },
    }

    if coerce_year != sample_year:
        attrs = deepcopy(tmy_data.attrs)
        tmy_data = tmy_data.copy()
        tmy_data.index = tmy_data.index + pd.DateOffset(years=sample_year - coerce_year)
        tmy_data.attrs = attrs
        tmy_data = fill_leap_day(tmy_data)

    # Resample to 15-min if requested
    if freq in ("15min", "15T", "15m"):
        tmy_data = resample_tmy_to_15min(tmy_data, metadata)
    elif freq not in ("h", "H", "1h", "1H"):
        raise ValueError("freq must be 'h' or '15min'")

    if save_to_file:
        # Encode metadata in filename: {location}_tmy_{year_min}_{year_max}_{db}.csv
        try:
            inputs = metadata.get("inputs", {})
            meta_loc = inputs.get("location", {})
            rad_db = inputs.get("meteo_data", {}).get("radiation_db", "unknown")
            year_min = inputs.get("meteo_data", {}).get("year_min", "unknown")
            year_max = inputs.get("meteo_data", {}).get("year_max", "unknown")
            # Derive location name from coordinates (fallback)
            loc_name = f"lat{meta_loc.get('latitude', latitude):.0f}_lon{meta_loc.get('longitude', longitude):.0f}"
            db_slug = f"pvgis-{rad_db.lower()}" if rad_db != "unknown" else "pvgis"
            filename = f"weather/{loc_name}_tmy_{year_min}_{year_max}_{db_slug}.csv"
        except (KeyError, AttributeError):
            filename = f"weather/tmy_data_{sample_year if sample_year else 'original'}_{freq}.csv"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        save_weather_csv(tmy_data, filename)
        logger.info("Saved TMY data and provenance sidecar to %s", filename)

    return tmy_data, metadata


def fetch_weather_data(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    tilt: float,
    azimuth: float,
    freq: str = "h",
    save_to_file: bool = True,
    location_name: Optional[str] = None,
    output_dir: str = "weather",
    radiation_time_basis: str = "interval_mean",
) -> pd.DataFrame:
    """
    Fetch historical weather data from the Open-Meteo API.

    Radiation can use Open-Meteo's preceding-hour means (the backwards-
    compatible default) or its instantaneous fields. The returned columns keep
    BREOS's established names in either case. Preceding-hour means are
    labelled at the end of their hour, so they run from ``start_date`` 01:00
    to the midnight after ``end_date``. Instantaneous values run from
    ``start_date`` 00:00 to ``end_date`` 23:00.

    Args:
        latitude: Latitude of the location
        longitude: Longitude of the location
        start_date: Start date in format 'YYYY-MM-DD'
        end_date: End date in format 'YYYY-MM-DD'
        tilt: Tilt of the PV panel (degrees)
        azimuth: Azimuth of the PV system (0° S, -90° E, 90° W, 180° N)
        freq: Output frequency ('h' for hourly, '15min' for 15-minute)
        save_to_file: Whether to save the data to CSV
        location_name: Location name for filename (e.g., 'porto'). If None, uses lat/lon.
        output_dir: Directory to save the file (default: 'weather')
        radiation_time_basis: ``"interval_mean"`` for the provider's
            preceding-hour means or ``"instant"`` for values at the label.

    Returns:
        DataFrame with weather variables

    Raises:
        ImportError: If openmeteo_requests is not installed

    Note:
        Responses are cached in a ``.cache.sqlite`` file created in the
        current working directory (30-day expiry). Delete it to force
        fresh API responses.
    """
    if not HAS_OPENMETEO:
        raise ImportError(
            "openmeteo_requests is required for historical weather data. "
            "Install with: uv add openmeteo-requests requests-cache"
        )

    if radiation_time_basis not in {"interval_mean", "instant"}:
        raise ValueError("radiation_time_basis must be 'interval_mean' or 'instant'")
    radiation_suffix = "_instant" if radiation_time_basis == "instant" else ""
    hourly_fields = (
        ("temperature_2m", "temperature_2m"),
        ("wind_speed_10m", "wind_speed_10m"),
        (f"shortwave_radiation{radiation_suffix}", "shortwave_radiation"),
        (f"direct_radiation{radiation_suffix}", "direct_radiation"),
        (f"diffuse_radiation{radiation_suffix}", "diffuse_radiation"),
        (f"direct_normal_irradiance{radiation_suffix}", "direct_normal_irradiance"),
        (f"global_tilted_irradiance{radiation_suffix}", "global_tilted_irradiance"),
        (f"terrestrial_radiation{radiation_suffix}", "terrestrial_radiation"),
    )

    # Setup the Open-Meteo API client with cache and retry. Cache expires
    # after 30 days so we don't serve indefinitely-stale entries if a single
    # bad response was ever written.
    cache_session = requests_cache.CachedSession(".cache", expire_after=timedelta(days=30))
    retries = Retry(total=5, backoff_factor=0.2, status_forcelist=[500, 502, 503, 504])
    cache_session.mount("https://", HTTPAdapter(max_retries=retries))
    cache_session.mount("http://", HTTPAdapter(max_retries=retries))
    openmeteo = openmeteo_requests.Client(session=cache_session)

    # Preceding-hour means are labelled at the end of their hour, so the
    # hours of [start_date, end_date] carry the labels from start_date 01:00
    # through the midnight after end_date. Fetch one extra day to get that
    # last label and trim to the requested hours below.
    right_labelled = radiation_time_basis == "interval_mean"
    first_label = pd.Timestamp(start_date)
    last_label = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    request_end_date = last_label.strftime("%Y-%m-%d") if right_labelled else end_date

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": request_end_date,
        "hourly": [provider_name for provider_name, _output_name in hourly_fields],
        "wind_speed_unit": "ms",
        "timezone": "GMT",
        "tilt": tilt,
        "azimuth": azimuth,
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    # Process hourly data
    hourly = response.Hourly()
    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s"),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        ),
        **{
            output_name: hourly.Variables(index).ValuesAsNumpy()
            for index, (_provider_name, output_name) in enumerate(hourly_fields)
        },
    }

    hourly_dataframe = pd.DataFrame(data=hourly_data)
    hourly_dataframe.set_index("date", inplace=True)
    if right_labelled:
        index = hourly_dataframe.index
        hourly_dataframe = hourly_dataframe[(index > first_label) & (index <= last_label)]
    hourly_dataframe.attrs[_WEATHER_METADATA_KEY] = {
        "source": "OpenMeteo_historical",
        "provider_hourly_fields": [provider_name for provider_name, _output_name in hourly_fields],
        "raw_radiation_variables": [provider_name for provider_name, _output_name in hourly_fields[2:]],
        "stored_radiation_variables": [output_name for _provider_name, output_name in hourly_fields[2:]],
        "radiation_time_basis": radiation_time_basis,
        "timestamp_label_basis": "instant" if radiation_time_basis == "instant" else "right",
        "timestamp_timezone": "GMT",
        "irradiance_time_offset_hours": 0.0 if radiation_time_basis == "instant" else None,
        "horizon": _unknown_horizon_metadata("openmeteo"),
    }

    # Resample to 15-min if requested (pass location for clear-sky scaling)
    if freq in ("15min", "15T", "15m"):
        hourly_dataframe = resample_to_15min(hourly_dataframe, method="makima", latitude=latitude, longitude=longitude)

    if save_to_file:
        start_year = start_date[:4]
        end_year = end_date[:4]
        if location_name:
            loc_slug = safe_path_slug(location_name)
        else:
            loc_slug = f"lat{latitude:.0f}_lon{longitude:.0f}"
        filename = os.path.join(output_dir, f"{loc_slug}_historical_{start_year}_{end_year}_openmeteo.csv")
        os.makedirs(output_dir, exist_ok=True)
        save_weather_csv(hourly_dataframe, filename)
        logger.info("Saved weather data and provenance sidecar to %s", filename)

    return hourly_dataframe


_TMY_RESAMPLED_COLUMNS = ("ghi", "dni", "dhi", "temp_air", "relative_humidity", "wind_speed")


def resample_tmy_to_15min(tmy_data: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """
    Resample PVGIS TMY data from hourly to 15-minute intervals.

    A thin wrapper over :func:`resample_to_15min` with Makima interpolation
    and clear-sky scaling at the PVGIS site, including its elevation. Only
    irradiance, temperature, humidity, and wind columns are kept.

    Args:
        tmy_data: DataFrame with hourly TMY data
        metadata: Metadata dict from PVGIS containing location info

    Returns:
        DataFrame with 15-minute intervals
    """
    loc = metadata["inputs"]["location"]
    columns = [column for column in _TMY_RESAMPLED_COLUMNS if column in tmy_data.columns]
    df_15 = resample_to_15min(
        tmy_data[columns],
        method="makima",
        latitude=loc["latitude"],
        longitude=loc["longitude"],
        altitude=loc["elevation"],
    )
    if "relative_humidity" in df_15:
        df_15["relative_humidity"] = df_15["relative_humidity"].clip(0, 100)

    weather_provenance = df_15.attrs.get(_WEATHER_METADATA_KEY)
    if weather_provenance is not None:
        weather_provenance["irradiance_resampling_method"] = "makima_clear_sky"
    return df_15


def resample_to_15min(
    df_hourly: pd.DataFrame,
    method: str = "makima",
    non_negative_cols: Optional[List[str]] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    preserve_irradiance_energy: bool = False,
    altitude: Optional[float] = None,
) -> pd.DataFrame:
    """
    Resample hourly DataFrame to 15-minute intervals.

    When latitude/longitude are provided, uses clear-sky scaling for irradiance
    columns (GHI, DNI, DHI) to preserve solar physics at sunrise/sunset
    transitions. Otherwise falls back to direct interpolation.

    Supports both TMY column names (ghi, dni, dhi) and Open-Meteo column names
    (shortwave_radiation, direct_normal_irradiance, diffuse_radiation).

    Args:
        df_hourly: DataFrame with hourly DatetimeIndex
        method: Interpolation method ('makima', 'linear', 'cubic')
        non_negative_cols: Columns to clip at zero (auto-detected for solar/wind)
        latitude: Location latitude for clear-sky scaling (optional)
        longitude: Location longitude for clear-sky scaling (optional)
        preserve_irradiance_energy: Renormalize each source hour's four
            irradiance values so their mean equals the source-hour value.
            This is opt-in because it changes established interpolation output.
        altitude: Site elevation in metres for the clear-sky model (optional;
            pvlib looks it up from the coordinates when omitted)

    Returns:
        DataFrame with 15-minute intervals

    Raises:
        ValueError: If DataFrame doesn't have DatetimeIndex
    """
    df_hourly = relabel_right_labeled_interval_means(df_hourly)
    weather_metadata = deepcopy(df_hourly.attrs.get(_WEATHER_METADATA_KEY))

    # Ensure DatetimeIndex
    if not isinstance(df_hourly.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")

    df_hourly = df_hourly.sort_index()

    # Create target 15-min index
    target_index = pd.date_range(
        start=df_hourly.index[0],
        end=df_hourly.index[-1] + pd.Timedelta(minutes=45),
        freq="15min",
    )

    # Convert timestamps to seconds for interpolation
    x_original = _datetime_index_seconds(df_hourly.index)
    x_target = _datetime_index_seconds(target_index)

    # Map column names to irradiance type (supports TMY and Open-Meteo conventions)
    irrad_col_map = {}  # column_name -> clear-sky component ('ghi', 'dni', 'dhi')
    for col in df_hourly.columns:
        col_lower = col.lower()
        if col_lower in ("ghi", "shortwave_radiation", "global_horizontal_irradiance"):
            irrad_col_map[col] = "ghi"
        elif col_lower in ("dni", "direct_normal_irradiance"):
            irrad_col_map[col] = "dni"
        elif col_lower in ("dhi", "diffuse_radiation", "diffuse_horizontal_irradiance"):
            irrad_col_map[col] = "dhi"

    # Use clear-sky scaling if location is provided and we found irradiance columns
    use_clearsky = latitude is not None and longitude is not None and len(irrad_col_map) > 0

    df_15min = pd.DataFrame(index=target_index)
    epsilon = 5.0  # Increased epsilon to avoid divide-by-zero spikes near dawn/dusk

    if use_clearsky:
        site = Location(latitude, longitude, altitude=altitude)
        source_offset = _representative_time_offset(
            weather_metadata or {}, pd.Timedelta(hours=1), require_metadata=False
        )
        target_offset = _representative_time_offset(
            weather_metadata or {}, pd.Timedelta(minutes=15), require_metadata=False
        )
        cs_hourly = site.get_clearsky(df_hourly.index + source_offset)
        cs_15min = site.get_clearsky(target_index + target_offset)

    # Get numeric columns only
    numeric_df = df_hourly.select_dtypes(include=[np.number])

    for col in numeric_df.columns:
        y_original = numeric_df[col].values

        if use_clearsky and col in irrad_col_map:
            # Clear-sky scaling: interpolate clearness index, not raw irradiance
            cs_comp = irrad_col_map[col]
            k_hourly = y_original / (cs_hourly[cs_comp].values + epsilon)

            # Clip K multiplier to physically reasonable max (e.g. 1.5x) to avoid massive dawn/dusk spikes
            k_hourly = np.clip(k_hourly, 0, 1.5)

            if method == "makima":
                interp_k = Akima1DInterpolator(x_original, k_hourly, method="makima")
            else:
                from scipy.interpolate import interp1d

                interp_k = interp1d(x_original, k_hourly, kind=method, fill_value="extrapolate")
            k_15min = interp_k(x_target)
            if method == "makima":
                k_15min[x_target > x_original[-1]] = k_hourly[-1]
            clear_sky = cs_15min[cs_comp].to_numpy(dtype=float)
            reconstructed = k_15min * (clear_sky + epsilon)
            reconstructed[clear_sky <= 0.0] = 0.0
            df_15min[col] = np.clip(reconstructed, 0, None)
        else:
            # Direct interpolation for non-irradiance columns
            if method == "makima":
                interp = Akima1DInterpolator(x_original, y_original, method="makima")
            else:
                from scipy.interpolate import interp1d

                interp = interp1d(x_original, y_original, kind=method, fill_value="extrapolate")
            interpolated = interp(x_target)
            if method == "makima":
                interpolated[x_target > x_original[-1]] = y_original[-1]
            df_15min[col] = interpolated

    # Auto-detect non-negative columns (solar/wind) — applies to columns not
    # already handled by clear-sky scaling
    if non_negative_cols is None:
        non_negative_cols = []
        for col in df_15min.columns:
            if col in irrad_col_map and use_clearsky:
                continue  # already clipped via clear-sky scaling
            if any(x in col.lower() for x in ["irrad", "radiation", "tilted", "terrestrial", "wind", "speed"]):
                non_negative_cols.append(col)

    # Clip negative values
    for col in non_negative_cols:
        if col in df_15min.columns:
            df_15min[col] = np.clip(df_15min[col], 0, None)

    if preserve_irradiance_energy and irrad_col_map:
        if len(df_hourly.index) > 1:
            intervals = df_hourly.index[1:] - df_hourly.index[:-1]
            if not np.all(intervals == pd.Timedelta(hours=1)):
                raise ValueError("preserve_irradiance_energy requires a regular hourly index")
        expected_rows = len(df_hourly) * 4
        if len(df_15min) != expected_rows:
            raise ValueError("preserve_irradiance_energy requires four 15-minute rows per source hour")

        for col in irrad_col_map:
            if col not in df_15min.columns:
                continue
            hourly_values = pd.to_numeric(df_hourly[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            blocks = df_15min[col].to_numpy(dtype=float, copy=True).reshape(len(df_hourly), 4)
            block_sums = blocks.sum(axis=1)
            nonzero = block_sums > 1e-12
            blocks[nonzero] *= (4.0 * hourly_values[nonzero] / block_sums[nonzero])[:, None]
            blocks[~nonzero] = hourly_values[~nonzero, None]
            df_15min[col] = np.clip(blocks.reshape(-1), 0.0, None)

    if weather_metadata is not None:
        weather_metadata["input_resolution"] = "h"
        weather_metadata["output_resolution"] = "15min"
        weather_metadata["irradiance_resampling_method"] = method
        weather_metadata["preserve_irradiance_energy"] = preserve_irradiance_energy
        df_15min.attrs[_WEATHER_METADATA_KEY] = weather_metadata

    return df_15min


def _complete_weather_years(csv_file_path: str) -> Tuple[Dict[int, pd.DataFrame], Dict[str, Any]]:
    """Read a multi-year weather CSV and split it into its complete years.

    Right-labelled interval means are first moved to the start of their
    interval. 29 February is dropped, so a complete year has the step count of
    a non-leap year at the file's own step size. Incomplete years are left out
    with a warning.

    Returns:
        Tuple of (source year -> frame with a ``date`` column, file metadata)
    """
    df = pd.read_csv(csv_file_path)
    try:
        df["date"] = pd.to_datetime(df["date"], format="ISO8601")
    except ValueError:
        try:
            df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y %H:%M")
        except ValueError:
            df["date"] = pd.to_datetime(df["date"], format="mixed")

    metadata = weather_file_metadata(csv_file_path)
    warn_if_naive_weather_timestamps(df["date"].dt, metadata, f"Weather file {csv_file_path}")
    df, metadata = _relabel_weather_date_column(df, metadata)
    df = df[~((df["date"].dt.month == 2) & (df["date"].dt.day == 29))]

    step = df["date"].diff().median()
    if len(df) < 2 or pd.isna(step) or step <= pd.Timedelta(0):
        logger.warning("Weather file %s has no usable timestep, so it has no complete year", csv_file_path)
        return {}, metadata
    expected_rows = int(round(pd.Timedelta(days=365) / step))

    years = df["date"].dt.year
    complete: Dict[int, pd.DataFrame] = {}
    incomplete: List[str] = []
    for year in years.unique():
        year_data = df[years == year].reset_index(drop=True)
        if len(year_data) == expected_rows:
            complete[int(year)] = year_data
        else:
            incomplete.append(f"{year} ({len(year_data)} of {expected_rows} rows)")
    if incomplete:
        hint = ""
        if metadata.get("source_timestamp_label_basis") == "right":
            hint = (
                ". Right-labelled interval means end at the midnight after the last day;"
                " a file without that label is one step short"
            )
        logger.warning("Skipping incomplete weather years in %s: %s%s", csv_file_path, ", ".join(incomplete), hint)
    return complete, metadata


def _remap_weather_year(year_data: pd.DataFrame, source_year: int, target_year: int) -> pd.DataFrame:
    remapped = year_data.copy()
    remapped["date"] = remapped["date"] + pd.DateOffset(years=target_year - source_year)
    return remapped


def select_random_year_and_replace_datetime(
    csv_file_path: str,
    target_year: int = 2025,
    *,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[pd.DataFrame, int]:
    """
    Load weather data, randomly select a complete year, and replace datetime with target year.

    Args:
        csv_file_path: Path to the CSV file
        target_year: Year to replace the selected year's datetime with
        rng: Generator that makes the choice. Pass a seeded one to reproduce
            it; the default draws fresh entropy.

    Returns:
        Tuple of (DataFrame with target year dates, selected_year)

    Raises:
        ValueError: If the file has no complete year
    """
    by_year, metadata = _complete_weather_years(csv_file_path)
    if not by_year:
        raise ValueError(f"No complete years found in weather file: {csv_file_path}")

    rng = np.random.default_rng() if rng is None else rng
    selected_year = int(rng.choice(list(by_year)))

    selected_year_data = _remap_weather_year(by_year[selected_year], selected_year, target_year)
    selected_year_data.attrs[WEATHER_METADATA_KEY] = metadata
    return selected_year_data, selected_year


def preload_weather_by_year(
    csv_file_path: str,
    target_year: int = 2025,
) -> Dict[int, pd.DataFrame]:
    """
    Pre-load weather CSV once and split into per-year DataFrames.

    Each year's dates are remapped to *target_year* so the resulting
    DataFrames can be used directly in simulation (same datetime grid as
    ``select_random_year_and_replace_datetime`` would produce). Incomplete
    years are skipped with a warning.

    Args:
        csv_file_path: Path to the multi-year weather CSV
        target_year: Calendar year to remap all dates to

    Returns:
        Dict mapping original year → DataFrame with target-year dates, indexed by 'date'
    """
    by_year, metadata = _complete_weather_years(csv_file_path)
    result: Dict[int, pd.DataFrame] = {}
    for year, year_data in by_year.items():
        remapped = _remap_weather_year(year_data, year, target_year)
        remapped.attrs[WEATHER_METADATA_KEY] = deepcopy(metadata)
        result[year] = remapped
    return result


def read_epw_file(
    filepath: str, freq: str = "h", latitude: Optional[float] = None, longitude: Optional[float] = None
) -> pd.DataFrame:
    """
    Read an EPW (EnergyPlus Weather) file and return standardized weather DataFrame.

    EPW files can be downloaded from https://climate.onebuilding.org/

    Args:
        filepath: Path to the .epw file
        freq: Output frequency ('h' for hourly, '15min' for 15-minute)
        latitude: Override latitude for clear-sky scaling (auto-detected from EPW if None)
        longitude: Override longitude for clear-sky scaling (auto-detected from EPW if None)

    Returns:
        DataFrame with standardized column names (ghi, dni, dhi, temp_air, wind_speed)
    """
    df, meta = pvlib.iotools.read_epw(filepath)

    # Standardize column names
    rename_map = {
        "ghi": "ghi",
        "dni": "dni",
        "dhi": "dhi",
        "temp_air": "temp_air",
        "wind_speed": "wind_speed",
    }
    available = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df[list(available.keys())].rename(columns=available)

    # Use EPW metadata for coordinates if not provided
    if latitude is None:
        latitude = meta.get("latitude")
    if longitude is None:
        longitude = meta.get("longitude")

    # EPW radiation is energy over the hour ending at the record's hour field
    # (1-24); pvlib labels that hour at its start (0-23). Record the basis
    # before resampling, so the 15-minute clear-sky scaling evaluates each
    # hour at its midpoint and keeps the resampling provenance.
    df.attrs[_WEATHER_METADATA_KEY] = {
        "source": "EPW_file",
        "path": os.path.abspath(filepath),
        "radiation_time_basis": "interval_mean",
        "timestamp_label_basis": "left",
        "horizon": _unknown_horizon_metadata("epw"),
    }

    # Resample to 15-min if requested
    if freq in ("15min", "15T", "15m"):
        df = resample_to_15min(df, method="makima", latitude=latitude, longitude=longitude)

    return df


def extract_ambient_temperature(weather_df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Extract hourly ambient temperature from a weather DataFrame.

    Tries known column names in order of preference:
    - 'temp_air'       — PVGIS TMY (pvlib standard name)
    - 'temperature_2m' — Open-Meteo historical
    - 'temp'           — generic fallback
    - 'air_temperature'— alternative naming

    Returns:
        pd.Series of temperatures, or None if no recognised column found.
    """
    for col in ("temp_air", "temperature_2m", "temp", "air_temperature"):
        if col in weather_df.columns:
            return weather_df[col]
    return None


_TEMPERATURE_FILE_DATE_COLUMNS = ("date", "datetime", "time")
_TEMPERATURE_FILE_VALUE_COLUMNS = ("temp", "temperature", "t_cell", "t_amb")


def _shift_to_calendar_year(series: pd.Series, year: int) -> pd.Series:
    """Restamp a one-year series onto ``year``, keeping month, day, and time.

    29 February is dropped from a leap-year source, as the weather loaders do.
    A non-leap source restamped onto a leap year has no 29 February, which the
    coverage check then reports.
    """
    index = series.index
    series = series[~((index.month == 2) & (index.day == 29))]
    shifted = series.copy()
    shifted.index = series.index + pd.DateOffset(years=year - int(series.index[0].year))
    return shifted


def _align_temperature_to_index(
    source: pd.Series,
    index: pd.DatetimeIndex,
    *,
    label: str,
    align_calendar_year: bool = False,
) -> pd.Series:
    """Place a timestamped temperature series onto the simulation index.

    Each simulation step takes the latest reading at or before it, and only
    within the source's own sampling interval, so hourly readings can drive a
    15-minute simulation but a gap is never bridged. A step with no reading is
    an error, never a default: a reindex across calendar years used to empty
    the whole series and replace it with 25 °C.
    """
    if not isinstance(source.index, pd.DatetimeIndex):
        raise ValueError(f"{label} has no timestamps to align with the simulation index")
    values = pd.to_numeric(source, errors="coerce").astype(float)
    finite = np.isfinite(values.to_numpy())
    if not finite.all():
        first = source.index[int(np.argmin(finite))]
        raise ValueError(f"{label} has {int((~finite).sum())} readings that are not finite numbers (first at {first})")
    # Naive timestamps are read as UTC, as naive weather timestamps are.
    if index.tz is not None:
        stamps = source.index if source.index.tz is not None else source.index.tz_localize("UTC")
        values.index = stamps.tz_convert(index.tz)
    elif source.index.tz is not None:
        values.index = source.index.tz_convert("UTC").tz_localize(None)
    if not values.index.is_unique:
        raise ValueError(f"{label} has duplicate timestamps")
    values = values.sort_index()
    if values.empty:
        raise ValueError(f"{label} has no readings")

    source_years = values.index.year.unique()
    index_years = index.year.unique()
    years_differ = len(source_years) == 1 and len(index_years) == 1 and source_years[0] != index_years[0]
    if align_calendar_year and years_differ:
        values = _shift_to_calendar_year(values, int(index_years[0]))

    stamps = values.index
    step = (stamps[1:] - stamps[:-1]).median() if len(stamps) > 1 else pd.Timedelta(0)
    position = stamps.searchsorted(index, side="right") - 1
    held_from = stamps[np.clip(position, 0, None)]
    lag = index - held_from
    covered = (position >= 0) & ((lag < step) | (lag == pd.Timedelta(0)))
    if not covered.all():
        first = index[int(np.argmin(covered))]
        hint = ""
        if years_differ and not align_calendar_year:
            hint = (
                f" It is from {source_years[0]} and the simulation runs in {index_years[0]};"
                f" restamp it to {index_years[0]}."
            )
        raise ValueError(
            f"{label} does not cover {int((~covered).sum())} of {len(index)} simulation steps "
            f"(first at {first}). It spans {stamps[0]} to {stamps[-1]}; the simulation spans "
            f"{index[0]} to {index[-1]}.{hint}"
        )
    return pd.Series(values.to_numpy()[position], index=index, name=source.name)


def _read_temperature_file(path: str) -> pd.Series:
    """Read a timestamped battery-temperature CSV, or say why it cannot be used."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"battery temperature file not found: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"could not read battery temperature file {path}: {exc}") from exc
    date_col = next((c for c in df.columns if str(c).lower() in _TEMPERATURE_FILE_DATE_COLUMNS), None)
    val_col = next((c for c in df.columns if str(c).lower() in _TEMPERATURE_FILE_VALUE_COLUMNS), None)
    if date_col is None or val_col is None:
        raise ValueError(
            f"battery temperature file {path} needs a timestamp column "
            f"({', '.join(_TEMPERATURE_FILE_DATE_COLUMNS)}) and a temperature column "
            f"({', '.join(_TEMPERATURE_FILE_VALUE_COLUMNS)}); it has {', '.join(map(str, df.columns))}"
        )
    try:
        stamps = pd.to_datetime(df[date_col])
    except (ValueError, TypeError) as exc:
        raise ValueError(f"battery temperature file {path} has unreadable timestamps: {exc}") from exc
    return pd.Series(df[val_col].to_numpy(), index=pd.DatetimeIndex(stamps), name=val_col)


def build_battery_temperature_series(
    temp_config: Any = None,
    index: Optional[pd.DatetimeIndex] = None,
    *,
    start_time: Optional[pd.Timestamp] = None,
    end_time: Optional[pd.Timestamp] = None,
    freq: str = "h",
    default_temp: float = 25.0,
    weather_df: Optional[pd.DataFrame] = None,
    indoor_model: Optional[Dict[str, Any]] = None,
    align_weather_year: bool = False,
) -> pd.Series:
    """Build the battery-temperature series used by degradation models.

    ``temp_config`` accepts the same forms used by internal runners:
    ``None``/``"weather"`` uses weather data, a number is a fixed temperature,
    and a string is treated as a CSV path. The indoor buffering model is applied
    by default and can be disabled with ``indoor_model={"enabled": False}``.

    Temperatures that cannot cover ``index`` raise instead of defaulting: a
    missing file raises ``FileNotFoundError``, and an unreadable file, a file
    without recognised columns, and readings from another calendar year or
    with gaps raise ``ValueError``, as does ``"weather"`` mode when
    ``weather_df`` has no temperature column. ``default_temp`` applies only
    when ``weather_df`` is absent. Pass
    ``align_weather_year=True`` when ``weather_df`` is a representative year
    whose temperatures should be restamped onto the calendar year of ``index``.
    """
    if index is None:
        if start_time is None or end_time is None:
            raise ValueError("Either index or start_time/end_time must be provided.")
        index = pd.date_range(start=start_time, end=end_time, freq=freq)
    else:
        index = pd.DatetimeIndex(index)

    weather_indexed = weather_df
    if weather_indexed is not None and not isinstance(weather_indexed.index, pd.DatetimeIndex):
        date_col = next(
            (c for c in weather_indexed.columns if str(c).lower() in {"date", "datetime", "time"}),
            None,
        )
        if date_col is not None:
            weather_indexed = weather_indexed.copy()
            weather_indexed[date_col] = pd.to_datetime(weather_indexed[date_col])
            weather_indexed = weather_indexed.set_index(date_col)

    if temp_config is None or (isinstance(temp_config, str) and temp_config.lower() == "weather"):
        ambient = extract_ambient_temperature(weather_indexed) if weather_indexed is not None else None
        if ambient is not None:
            ambient = ambient.copy()
            if not isinstance(ambient.index, pd.DatetimeIndex) and len(ambient) == len(index):
                ambient.index = index
            result = _align_temperature_to_index(
                ambient, index, label="weather temperature", align_calendar_year=align_weather_year
            )
        elif weather_indexed is not None:
            raise ValueError(
                "battery temperature 'weather' needs an air temperature column, and the weather has none. "
                "Add one, or set a fixed battery temperature (for example battery_temperature = 25)."
            )
        else:
            result = pd.Series(default_temp, index=index)
    elif isinstance(temp_config, bool):
        raise TypeError("battery temperature must be 'weather', a CSV path, or a number, not a bool")
    elif isinstance(temp_config, (int, float)):
        if not np.isfinite(temp_config):
            raise ValueError(f"fixed battery temperature must be finite, got {temp_config}")
        result = pd.Series(float(temp_config), index=index)
    elif isinstance(temp_config, (str, os.PathLike)):
        path = os.fspath(temp_config)
        result = _align_temperature_to_index(
            _read_temperature_file(path), index, label=f"battery temperature file {path}"
        )
    else:
        raise TypeError(
            f"battery temperature must be 'weather', a CSV path, or a number, got {type(temp_config).__name__}"
        )

    indoor_model = indoor_model or {}
    from breos.constants import (
        DEFAULT_INDOOR_CEILING_C,
        DEFAULT_INDOOR_COUPLING_ALPHA,
        DEFAULT_INDOOR_FLOOR_C,
        DEFAULT_INDOOR_MODEL_ENABLED,
        DEFAULT_INDOOR_SETPOINT_C,
    )

    if indoor_model.get("enabled", DEFAULT_INDOOR_MODEL_ENABLED):
        from breos.battery import apply_indoor_temperature_model

        result = apply_indoor_temperature_model(
            result,
            setpoint_c=indoor_model.get("setpoint_c", DEFAULT_INDOOR_SETPOINT_C),
            coupling_alpha=indoor_model.get("coupling_alpha", DEFAULT_INDOOR_COUPLING_ALPHA),
            floor_c=indoor_model.get("floor_c", DEFAULT_INDOOR_FLOOR_C),
            ceiling_c=indoor_model.get("ceiling_c", DEFAULT_INDOOR_CEILING_C),
        )

    return result
