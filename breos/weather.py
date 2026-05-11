"""
Weather data fetching and processing module.

This module handles:
- Fetching TMY (Typical Meteorological Year) data from PVGIS
- Fetching historical weather data from Open-Meteo
- Converting between hourly and 15-minute resolutions using Makima interpolation
"""

import os
import random
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pvlib
from pvlib.location import Location
from scipy.interpolate import Akima1DInterpolator

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
    match = re.match(r"^([a-z]+)_(tmy|historical)_(\d{4})_(\d{4})_([\w-]+)$", name)
    if match:
        return {
            "location": match.group(1),
            "type": match.group(2),
            "year_start": match.group(3),
            "year_end": match.group(4),
            "source": match.group(5),
        }

    # Pattern without year_end: location_type_year_source (e.g., lisbon_tmy_2014_nsrdb)
    match = re.match(r"^([a-z]+)_(tmy|historical)_(\d{4})_([\w-]+)$", name)
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

    print(f"   Found local weather file: {filepath}")

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
            print(f"   Subset to {start_year}-{end_year} ({len(df)} rows)")

    return df


def fetch_tmy_weather_data(
    latitude: float,
    longitude: float,
    sample_year: Optional[int] = 2025,
    freq: str = "h",
    timezone: Optional[str] = None,
    save_to_file: bool = False,
) -> Tuple[pd.DataFrame, dict]:
    """
    Fetch Typical Meteorological Year (TMY) weather data from PVGIS.

    Args:
        latitude: Latitude of the location
        longitude: Longitude of the location
        sample_year: Year to use for index (default: 2025). Set to None to keep original TMY index.
        freq: Frequency for output data ('h' for hourly, '15min' for 15-minute)
        timezone: Timezone string. Auto-detected if None.
        save_to_file: Whether to save the data to CSV

    Returns:
        Tuple of (tmy_data DataFrame, metadata dict)

    Raises:
        ValueError: If sample_year is a leap year (TMY has 8760 hours)
    """
    tmy_data, metadata = pvlib.iotools.get_pvgis_tmy(
        latitude,
        longitude,
        outputformat="json",
        usehorizon=True,
        map_variables=True,
        url="https://re.jrc.ec.europa.eu/api/v5_3/",
        timeout=120,
    )

    # Reindex to sample year if requested
    if sample_year is not None:
        # Check for leap year
        if sample_year % 4 == 0 and (sample_year % 100 != 0 or sample_year % 400 == 0):
            raise ValueError(f"Sample year {sample_year} is a leap year. TMY has 8760 hours. Use non-leap year.")

        # Auto-detect timezone if not provided
        if timezone is None:
            from timezonefinder import TimezoneFinder

            tf = TimezoneFinder()
            timezone = tf.timezone_at(lat=latitude, lng=longitude)

        new_index = pd.date_range(
            start=f"{sample_year}-01-01 00:00", end=f"{sample_year}-12-31 23:00", freq="h", tz=timezone
        )
        tmy_data.index = new_index

    # Resample to 15-min if requested
    if freq == "15min":
        tmy_data = resample_tmy_to_15min(tmy_data, metadata)
    elif freq != "h":
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
        tmy_data.to_csv(filename)
        print(f"Saved TMY data to {filename}")

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
) -> pd.DataFrame:
    """
    Fetch historical weather data from the Open-Meteo API.

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

    Returns:
        DataFrame with weather variables

    Raises:
        ImportError: If openmeteo_requests is not installed
    """
    if not HAS_OPENMETEO:
        raise ImportError(
            "openmeteo_requests is required for historical weather data. "
            "Install with: uv add openmeteo-requests requests-cache"
        )

    # Setup the Open-Meteo API client with cache and retry
    cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
    retries = Retry(total=5, backoff_factor=0.2, status_forcelist=[500, 502, 503, 504])
    cache_session.mount("https://", HTTPAdapter(max_retries=retries))
    cache_session.mount("http://", HTTPAdapter(max_retries=retries))
    openmeteo = openmeteo_requests.Client(session=cache_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "wind_speed_10m",
            "shortwave_radiation",
            "direct_radiation",
            "diffuse_radiation",
            "direct_normal_irradiance",
            "global_tilted_irradiance",
            "terrestrial_radiation",
        ],
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
        "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
        "wind_speed_10m": hourly.Variables(1).ValuesAsNumpy(),
        "shortwave_radiation": hourly.Variables(2).ValuesAsNumpy(),
        "direct_radiation": hourly.Variables(3).ValuesAsNumpy(),
        "diffuse_radiation": hourly.Variables(4).ValuesAsNumpy(),
        "direct_normal_irradiance": hourly.Variables(5).ValuesAsNumpy(),
        "global_tilted_irradiance": hourly.Variables(6).ValuesAsNumpy(),
        "terrestrial_radiation": hourly.Variables(7).ValuesAsNumpy(),
    }

    hourly_dataframe = pd.DataFrame(data=hourly_data)
    hourly_dataframe.set_index("date", inplace=True)

    # Resample to 15-min if requested (pass location for clear-sky scaling)
    if freq == "15min":
        hourly_dataframe = resample_to_15min(hourly_dataframe, method="makima", latitude=latitude, longitude=longitude)

    if save_to_file:
        start_year = start_date[:4]
        end_year = end_date[:4]
        if location_name:
            loc_slug = location_name.lower()
        else:
            loc_slug = f"lat{latitude:.0f}_lon{longitude:.0f}"
        filename = os.path.join(output_dir, f"{loc_slug}_historical_{start_year}_{end_year}_openmeteo.csv")
        os.makedirs(output_dir, exist_ok=True)
        hourly_dataframe.to_csv(filename)
        print(f"Saved weather data to {filename}")

    return hourly_dataframe


def resample_tmy_to_15min(tmy_data: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """
    Resample TMY data from hourly to 15-minute intervals using Makima interpolation.

    Uses clear-sky scaling for GHI to preserve solar physics.

    Args:
        tmy_data: DataFrame with hourly TMY data
        metadata: Metadata dict from PVGIS containing location info

    Returns:
        DataFrame with 15-minute intervals
    """
    # Location setup for clear-sky model
    loc = metadata["inputs"]["location"]
    site = Location(loc["latitude"], loc["longitude"], altitude=loc["elevation"])

    # Time handling
    df_60 = tmy_data.copy()
    start = df_60.index[0]
    end = df_60.index[-1] + pd.Timedelta(minutes=45)
    index_15 = pd.date_range(start, end, freq="15min", tz=df_60.index.tz)

    # Convert timestamps to unix floats for Scipy
    x_60 = df_60.index.view(np.int64) // 10**9
    x_15 = index_15.view(np.int64) // 10**9

    # Clear-sky scaling for irradiance (GHI, DNI, DHI)
    # Interpolate clearness indices instead of raw irradiance to preserve
    # sunrise/sunset transitions and physical consistency between components.
    cs_60 = site.get_clearsky(df_60.index)
    cs_15 = site.get_clearsky(index_15)

    df_15 = pd.DataFrame(index=index_15)
    epsilon = 5.0  # Increased epsilon to avoid divide-by-zero spikes near dawn/dusk

    for comp in ("ghi", "dni", "dhi"):
        if comp in df_60.columns:
            k_60 = (df_60[comp] / (cs_60[comp] + epsilon)).values
            # Clip K multiplier to physically reasonable max (e.g. 1.5x) to avoid massive dawn/dusk spikes
            k_60 = np.clip(k_60, 0, 1.5)

            makima_k = Akima1DInterpolator(x_60, k_60, method="makima")
            k_15 = makima_k(x_15)
            df_15[comp] = np.clip(k_15 * cs_15[comp], 0, None)

    # Interpolate non-irradiance columns directly with Makima
    met_cols = ["temp_air", "relative_humidity", "wind_speed"]

    for col in met_cols:
        if col in df_60.columns:
            y_60 = df_60[col].values
            makima_generic = Akima1DInterpolator(x_60, y_60, method="makima")
            df_15[col] = makima_generic(x_15)

    # Physical clipping
    if "relative_humidity" in df_15:
        df_15["relative_humidity"] = df_15["relative_humidity"].clip(0, 100)
    if "wind_speed" in df_15:
        df_15["wind_speed"] = np.clip(df_15["wind_speed"], 0, None)

    return df_15


def resample_to_15min(
    df_hourly: pd.DataFrame,
    method: str = "makima",
    non_negative_cols: Optional[List[str]] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
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

    Returns:
        DataFrame with 15-minute intervals

    Raises:
        ValueError: If DataFrame doesn't have DatetimeIndex
    """
    # Ensure DatetimeIndex
    if not isinstance(df_hourly.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")

    df_hourly = df_hourly.sort_index()

    # Create target 15-min index
    target_index = pd.date_range(start=df_hourly.index[0], end=df_hourly.index[-1], freq="15min")

    # Convert timestamps to seconds for interpolation
    x_original = df_hourly.index.astype("int64") // 10**9
    x_target = target_index.astype("int64") // 10**9

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
        site = Location(latitude, longitude)
        cs_hourly = site.get_clearsky(df_hourly.index)
        cs_15min = site.get_clearsky(target_index)

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
            df_15min[col] = np.clip(k_15min * cs_15min[cs_comp].values, 0, None)
        else:
            # Direct interpolation for non-irradiance columns
            if method == "makima":
                interp = Akima1DInterpolator(x_original, y_original, method="makima")
            else:
                from scipy.interpolate import interp1d

                interp = interp1d(x_original, y_original, kind=method, fill_value="extrapolate")
            df_15min[col] = interp(x_target)

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

    return df_15min


def resample_to_hourly(df_15min: pd.DataFrame, agg_method: str = "mean") -> pd.DataFrame:
    """
    Resample 15-minute DataFrame to hourly intervals.

    Args:
        df_15min: DataFrame with 15-minute DatetimeIndex
        agg_method: Aggregation method ('mean', 'sum', 'first', 'last')

    Returns:
        DataFrame with hourly intervals
    """
    if agg_method == "mean":
        return df_15min.resample("h").mean()
    elif agg_method == "sum":
        return df_15min.resample("h").sum()
    elif agg_method == "first":
        return df_15min.resample("h").first()
    elif agg_method == "last":
        return df_15min.resample("h").last()
    else:
        raise ValueError(f"Unknown aggregation method: {agg_method}")


def csv_15min_to_hourly(
    input_file_name: str,
    output_file_name: str,
    datetime_column: str = "Datetime",
    datetime_format: str = "%d/%m/%Y %H:%M",
) -> Optional[pd.DataFrame]:
    """
    Convert 15-minute interval CSV data to hourly data.

    Args:
        input_file_name: Path to input CSV file with 15-minute data
        output_file_name: Path for output CSV file with hourly data
        datetime_column: Name of the datetime column
        datetime_format: Format of the datetime string

    Returns:
        DataFrame with hourly aggregated data, or None on error
    """
    try:
        df = pd.read_csv(input_file_name)
        df[datetime_column] = pd.to_datetime(df[datetime_column], format=datetime_format)
        df.set_index(datetime_column, inplace=True)

        hourly_df = df.resample("h").sum()
        hourly_df = hourly_df.reset_index()

        hourly_df.to_csv(output_file_name, index=False)

        print(f"Successfully converted {input_file_name} to hourly data")
        print(f"Output saved to: {output_file_name}")
        print(f"Original shape: {df.shape} -> Hourly shape: {hourly_df.shape}")

        return hourly_df

    except Exception as e:
        print(f"Error processing file: {e}")
        return None


def csv_hourly_to_15min(
    input_file_name: str,
    output_file_name: str,
    datetime_column: str = "Datetime",
    datetime_format: str = "%d/%m/%Y %H:%M",
    non_negative_cols: Optional[List[str]] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> Optional[pd.DataFrame]:
    """
    Convert hourly CSV data to 15-minute intervals using Makima interpolation.

    Args:
        input_file_name: Path to input CSV file with hourly data
        output_file_name: Path for output CSV file with 15-minute data
        datetime_column: Name of the datetime column
        datetime_format: Format of the datetime string
        non_negative_cols: Columns to force >= 0
        latitude: Location latitude for clear-sky scaling of irradiance (optional)
        longitude: Location longitude for clear-sky scaling of irradiance (optional)

    Returns:
        DataFrame with 15-minute interpolated data, or None on error
    """
    try:
        df = pd.read_csv(input_file_name)
        df[datetime_column] = pd.to_datetime(df[datetime_column], format=datetime_format)
        df.set_index(datetime_column, inplace=True)
        df = df.sort_index()

        # Use the resample function
        df_15min = resample_to_15min(
            df, method="makima", non_negative_cols=non_negative_cols, latitude=latitude, longitude=longitude
        )

        # Reset index for saving
        df_15min = df_15min.reset_index().rename(columns={"index": datetime_column})
        df_15min.to_csv(output_file_name, index=False)

        print(f"Successfully converted {input_file_name} to 15-min data (Makima)")
        print(f"Output saved to: {output_file_name}")
        print(f"Original shape: {df.shape} -> 15-min shape: {df_15min.shape}")

        return df_15min

    except Exception as e:
        print(f"Error processing file: {e}")
        return None


def select_random_year_and_replace_datetime(csv_file_path: str, target_year: int = 2025) -> Tuple[pd.DataFrame, int]:
    """
    Load weather data, randomly select a year, and replace datetime with target year.

    Args:
        csv_file_path: Path to the CSV file
        target_year: Year to replace the selected year's datetime with

    Returns:
        Tuple of (DataFrame with target year dates, selected_year)
    """
    df = pd.read_csv(csv_file_path)

    # Parse datetime with format detection
    try:
        df["date"] = pd.to_datetime(df["date"], format="ISO8601")
    except ValueError:
        try:
            df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y %H:%M")
        except ValueError:
            df["date"] = pd.to_datetime(df["date"], format="mixed")

    # Extract year and get available years
    df["year"] = df["date"].dt.year
    available_years = df["year"].unique()

    # Randomly select a year
    selected_year = random.choice(available_years)

    # Filter data for selected year
    selected_year_data = df[df["year"] == selected_year].copy()

    # Handle leap years by removing Feb 29
    if len(selected_year_data) == 8784:  # Leap year
        feb_29_mask = (selected_year_data["date"].dt.month == 2) & (selected_year_data["date"].dt.day == 29)
        selected_year_data = selected_year_data[~feb_29_mask]

    # Validate
    if len(selected_year_data) != 8760:
        print(f"Warning: Year {selected_year} has {len(selected_year_data)} hours, not 8760")

    # Replace year in datetime
    year_diff = target_year - selected_year
    selected_year_data["date"] = selected_year_data["date"] + pd.DateOffset(years=year_diff)

    # Cleanup
    selected_year_data = selected_year_data.drop("year", axis=1)
    selected_year_data = selected_year_data.reset_index(drop=True)

    return selected_year_data, selected_year


def preload_weather_by_year(
    csv_file_path: str,
    target_year: int = 2025,
) -> Dict[int, pd.DataFrame]:
    """
    Pre-load weather CSV once and split into per-year DataFrames.

    Each year's dates are remapped to *target_year* so the resulting
    DataFrames can be used directly in simulation (same datetime grid as
    ``select_random_year_and_replace_datetime`` would produce).

    Args:
        csv_file_path: Path to the multi-year weather CSV
        target_year: Calendar year to remap all dates to

    Returns:
        Dict mapping original year → DataFrame with target-year dates, indexed by 'date'
    """
    df = pd.read_csv(csv_file_path)

    # Parse datetime once
    try:
        df["date"] = pd.to_datetime(df["date"], format="ISO8601")
    except ValueError:
        try:
            df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y %H:%M")
        except ValueError:
            df["date"] = pd.to_datetime(df["date"], format="mixed")

    df["year"] = df["date"].dt.year
    available_years = df["year"].unique()

    result: Dict[int, pd.DataFrame] = {}
    for yr in available_years:
        yr_data = df[df["year"] == yr].copy()

        # Handle leap years
        if len(yr_data) == 8784:
            feb_29_mask = (yr_data["date"].dt.month == 2) & (yr_data["date"].dt.day == 29)
            yr_data = yr_data[~feb_29_mask]

        if len(yr_data) != 8760:
            continue  # skip incomplete years

        # Remap to target year
        year_diff = target_year - yr
        yr_data["date"] = yr_data["date"] + pd.DateOffset(years=year_diff)
        yr_data = yr_data.drop("year", axis=1).reset_index(drop=True)
        result[int(yr)] = yr_data

    return result


def fetch_tmy_nsrdb(
    latitude: float,
    longitude: float,
    api_key: Optional[str] = None,
    email: Optional[str] = None,
    year: str = "tmy",
    location_name: Optional[str] = None,
    freq: str = "h",
    save_to_file: bool = False,
) -> Tuple[pd.DataFrame, dict]:
    """
    Fetch TMY data from NREL's NSRDB (National Solar Radiation Database) via pvlib.

    Uses pvlib.iotools.get_nsrdb_psm4_tmy (PSM4 API).
    Requires NREL_API_KEY and NREL_EMAIL environment variables, or pass them
    directly. Get a free key at https://developer.nrel.gov/signup/

    Args:
        latitude: Latitude of the location
        longitude: Longitude of the location
        api_key: NREL API key (falls back to NREL_API_KEY env var)
        email: Email for NREL API (falls back to NREL_EMAIL env var)
        year: TMY variant — 'tmy' for TMY, or a specific year string like '2020'
        location_name: Location name for the output filename
        freq: Output frequency ('h' for hourly, '15min' for 15-minute)
        save_to_file: Whether to save the data to CSV

    Returns:
        Tuple of (weather DataFrame, metadata dict)

    Raises:
        ValueError: If API key or email not provided
    """
    api_key = api_key or os.environ.get("NREL_API_KEY")
    email = email or os.environ.get("NREL_EMAIL")

    if not api_key or not email:
        raise ValueError(
            "NREL API key and email are required. Set NREL_API_KEY and NREL_EMAIL "
            "environment variables or pass them directly."
        )

    df, metadata = pvlib.iotools.get_nsrdb_psm4_tmy(
        latitude=latitude,
        longitude=longitude,
        api_key=api_key,
        email=email,
        year=year,
        map_variables=True,
    )

    # Keep only the columns we need (map_variables=True gives pvlib standard names)
    wanted = [c for c in ("ghi", "dni", "dhi", "temp_air", "wind_speed") if c in df.columns]
    df = df[wanted].copy()

    # Resample to 15-min if requested
    if freq == "15min":
        df = resample_to_15min(df, method="makima", latitude=latitude, longitude=longitude)

    if save_to_file and location_name:
        vintage = year if year != "tmy" else "tmy"
        filename = f"weather/{location_name}_tmy_{vintage}_nsrdb.csv"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename)
        print(f"Saved NSRDB data to {filename}")

    return df, metadata


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

    # Resample to 15-min if requested
    if freq == "15min":
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
