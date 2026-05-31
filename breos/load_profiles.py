"""
Load profile management module.

This module handles residential and commercial load profiles,
including loading from CSV files, scaling to annual consumption,
and resampling between hourly and 15-minute intervals.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from breos.utils import count_leap_years, get_hours_per_step, is_leap_year
from breos.weather import resample_to_15min

# Base directory for RLP files
BASE_DIR = Path(__file__).resolve().parent.parent


# Profile type mappings
PROFILE_FILES = {
    "1": "h0SLP_demandlib_1000kwh_hourly.csv",
    "4": "EREDES_2025_BTN_1000kwh_hourly.csv",  # BTN A - Commercial
    "5": "EREDES_2025_BTN_1000kwh_hourly.csv",  # BTN B - Hybrid
    "6": "EREDES_2025_BTN_1000kwh_hourly.csv",  # BTN C - Residential
    "7": "bdew_h0_2025_15min.csv",  # BDEW H0 2025 (Germany)
    "8": "REE_2026_2.0TD_1000kwh_hourly.csv",  # REE 2.0TD (Spain)
}

PROFILE_FILES_15MIN = {
    "1": "h0SLP_demandlib_1000kwh_15min.csv",
    "4": "EREDES_2025_BTN_1000kwh_15min.csv",
    "5": "EREDES_2025_BTN_1000kwh_15min.csv",
    "6": "EREDES_2025_BTN_1000kwh_15min.csv",
    "7": "bdew_h0_2025_15min.csv",
    "8": "REE_2026_2.0TD_1000kwh_15min.csv",
}

PROFILE_NAMES = {
    "1": "H0SLP (demandlib)",
    "4": "E-Redes 2025 - BTN A (Commercial)",
    "5": "E-Redes 2025 - BTN B (Hybrid)",
    "6": "E-Redes 2025 - BTN C (Residential)",
    "7": "BDEW H0 2025 (Germany)",
    "8": "REE 2026 - 2.0TD (Spain)",
}

PROFILE_ALIASES = {
    "crest": "6",
    "eredes_btn_c": "6",
    "bdew_h0": "7",
    "ree_2.0td": "8",
}

# Column mappings for E-Redes profiles
EREDES_COLUMNS = {
    "4": "BTN A - Wh",
    "5": "BTN B - Wh",
    "6": "BTN C - Wh",
}


def load_profile(
    profile_type: str,
    annual_consumption_kwh: float,
    start_date: str = "2025-01-01",
    freq: str = "h",
    num_years: int = 1,
    rlp_directory: Optional[str] = None,
    timezone: Optional[str] = "UTC",
) -> pd.DataFrame:
    """
    Load and scale a residential/commercial load profile.

    This is the main function for loading load profiles. It handles:
    - Multiple profile types (H0SLP, E-Redes, BDEW H0, REE, etc.)
    - Scaling to target annual consumption
    - Multi-year extension
    - 15-minute resolution (using native files or interpolation)

    Args:
        profile_type: Profile type key (see PROFILE_NAMES) or name
        annual_consumption_kwh: Target annual consumption in kWh
        start_date: Start date for the profile (YYYY-MM-DD)
        freq: Time frequency ('h' for hourly, '15min' for 15-minute)
        num_years: Number of years to generate
        rlp_directory: Directory containing RLP files (defaults to 'rlp/')
        timezone: Timezone for the index (default: 'UTC' to match TMY data)

    Returns:
        DataFrame with 'Electrical Consumption [W]' column and DatetimeIndex

    Raises:
        ValueError: If profile_type is not recognized
    """
    profile_type = PROFILE_ALIASES.get(str(profile_type).lower(), str(profile_type))
    if profile_type not in PROFILE_FILES:
        raise ValueError(f"Unknown profile type: {profile_type}. Valid types: {list(PROFILE_FILES.keys())}")

    # Determine RLP directory
    if rlp_directory is None:
        rlp_directory = BASE_DIR / "rlp"
    else:
        rlp_directory = Path(rlp_directory)

    # Check if native 15-min file exists
    use_native_15min = (
        freq in ("15min", "15T")
        and profile_type in PROFILE_FILES_15MIN
        and (rlp_directory / PROFILE_FILES_15MIN[profile_type]).exists()
    )

    if use_native_15min:
        csv_file = rlp_directory / PROFILE_FILES_15MIN[profile_type]
        native_freq = "15min"
    else:
        csv_file = rlp_directory / PROFILE_FILES[profile_type]
        native_freq = "h"

    # Load the profile
    df = _load_profile_csv(csv_file, profile_type)

    # Create datetime index for one year
    start_year = int(start_date[:4])
    hours_in_year = 8760  # Non-leap year
    steps_per_hour = 4 if native_freq == "15min" else 1

    new_index = pd.date_range(start=start_date, periods=hours_in_year * steps_per_hour, freq=native_freq, tz=timezone)

    # Adjust if profile has different length
    if len(df) < len(new_index):
        # Repeat to fill
        repeats = (len(new_index) // len(df)) + 1
        df = pd.concat([df] * repeats, ignore_index=True).iloc[: len(new_index)]
    elif len(df) > len(new_index):
        df = df.iloc[: len(new_index)]

    df.index = new_index
    df.index.name = "DateTime"

    # Scale to target consumption
    scale_to_annual_consumption(df, annual_consumption_kwh)

    # Extend to multiple years if needed
    if num_years > 1:
        df = _extend_to_years(df, start_year, num_years)

    # Resample if needed (hourly to 15-min)
    if freq in ("15min", "15T") and native_freq == "h":
        df = _resample_load_to_15min(df)

    return df


def _load_profile_csv(csv_file: Path, profile_type: str) -> pd.DataFrame:
    """Load a profile CSV file and standardize column names."""
    try:
        if profile_type == "1":
            # H0SLP demandlib format (hourly has 'Electrical Consumption [kW]', 15min has 'h0_dyn' in kW)
            df = pd.read_csv(csv_file, index_col=0)
            if "Electrical Consumption [kW]" in df.columns:
                df["Electrical Consumption [kW]"] *= 1000
                df.rename(columns={"Electrical Consumption [kW]": "Electrical Consumption [W]"}, inplace=True)
            elif "h0_dyn" in df.columns:
                df["h0_dyn"] *= 1000
                df.rename(columns={"h0_dyn": "Electrical Consumption [W]"}, inplace=True)

        elif profile_type in ("4", "5", "6"):
            # E-Redes format
            df = pd.read_csv(csv_file)
            col_name = EREDES_COLUMNS[profile_type]
            if col_name in df.columns:
                df = df[[col_name]].copy()
                df.rename(columns={col_name: "Electrical Consumption [W]"}, inplace=True)
            else:
                # Try to find the column
                for col in df.columns:
                    if "BTN" in col:
                        df = df[[col]].copy()
                        df.rename(columns={col: "Electrical Consumption [W]"}, inplace=True)
                        break
        else:
            df = pd.read_csv(csv_file, index_col=0)
            df.columns = ["Electrical Consumption [W]"]

        return df[["Electrical Consumption [W]"]].copy()

    except Exception as e:
        raise ValueError(f"Error loading profile from {csv_file}: {e}")


def scale_to_annual_consumption(
    load_df: pd.DataFrame, annual_consumption_kwh: float, column: str = "Electrical Consumption [W]"
) -> None:
    """
    Scale load profile to match target annual consumption.

    Modifies the DataFrame in place.

    Args:
        load_df: DataFrame with load data (in W)
        annual_consumption_kwh: Target annual consumption in kWh
        column: Name of the consumption column
    """
    # Calculate current annual consumption in Wh
    current_annual_wh = load_df[column].sum()

    # Determine hours per step
    if isinstance(load_df.index, pd.DatetimeIndex):
        # Infer from index frequency
        if len(load_df) > 1:
            diff = (load_df.index[1] - load_df.index[0]).total_seconds() / 3600
            hours_per_step = diff
        else:
            hours_per_step = 1.0
    else:
        hours_per_step = 1.0  # Assume hourly

    # Current annual in Wh (power * hours_per_step)
    current_annual_wh = load_df[column].sum() * hours_per_step

    # Target in Wh
    target_annual_wh = annual_consumption_kwh * 1000

    # Scale
    if current_annual_wh > 0:
        scaling_factor = target_annual_wh / current_annual_wh
        load_df[column] *= scaling_factor


def _extend_to_years(df: pd.DataFrame, start_year: int, num_years: int) -> pd.DataFrame:
    """
    Extend a 1-year profile to multiple years by repeating data.

    Generates a fresh index for each year to handle leap years correctly
    and avoid duplicates from simple date shifting.
    """
    def _calendar_key(ts: pd.Timestamp, day_override: Optional[int] = None):
        offset = ts.utcoffset()
        offset_seconds = int(offset.total_seconds()) if offset is not None else None
        return (
            ts.month,
            ts.day if day_override is None else day_override,
            ts.hour,
            ts.minute,
            offset_seconds,
        )

    # Build a calendar lookup from the canonical source year. Feb. 29 is excluded
    # so leap years can duplicate Feb. 28 without shifting the rest of the year.
    source_rows = {}
    for ts, row in df.iterrows():
        if ts.month == 2 and ts.day == 29:
            continue
        source_rows[_calendar_key(ts)] = row.to_numpy(copy=True)

    freq = pd.infer_freq(df.index) or "h"
    tz = df.index.tz

    dfs = []

    for i in range(num_years):
        current_year = start_year + i

        # Generate full index for this year
        year_start = f"{current_year}-01-01 00:00"
        year_end = f"{current_year}-12-31 23:45"  # Cover max potential range

        year_index = pd.date_range(start=year_start, end=year_end, freq=freq, tz=tz)
        # Cap at end of year exactly
        year_index = year_index[year_index.year == current_year]

        year_values = []
        for ts in year_index:
            day_override = 28 if (ts.month == 2 and ts.day == 29) else None
            key = _calendar_key(ts, day_override=day_override)
            if key not in source_rows:
                raise KeyError(f"Missing canonical load value for {ts}")
            year_values.append(source_rows[key])
        year_values = np.vstack(year_values)

        # Create DataFrame
        year_df = pd.DataFrame(data=year_values, index=year_index, columns=df.columns)
        dfs.append(year_df)

    return pd.concat(dfs)


def _resample_load_to_15min(df: pd.DataFrame) -> pd.DataFrame:
    """Resample hourly load to 15-minute using interpolation."""
    # For load, we typically want to interpolate (not sum)
    # because the values represent average power in W
    df_15min = resample_to_15min(df, method="makima")

    # Ensure no negative values
    for col in df_15min.columns:
        df_15min[col] = df_15min[col].clip(lower=0)

    return df_15min


def align_load_to_pv(load_df: pd.DataFrame, pv_series: pd.Series, freq: str = "h") -> pd.DataFrame:
    """
    Align load profile DatetimeIndex to match PV production data.

    This handles the common case where load profiles use a generic year (e.g., 2023)
    but PV/TMY data uses a different year (e.g., 1990).

    Args:
        load_df: Load profile DataFrame with DatetimeIndex
        pv_series: PV production Series with DatetimeIndex
        freq: Time frequency

    Returns:
        Load DataFrame with index aligned to PV data's year
    """
    # Get PV time range
    pv_start = pv_series.index[0]
    pv_end = pv_series.index[-1]

    # Create new index matching PV's year
    new_index = pd.date_range(start=pv_start, end=pv_end, freq=freq)

    # Get the load values (ignoring year)
    load_values = load_df.iloc[:, 0].values

    # Adjust length if needed
    if len(load_values) < len(new_index):
        # Repeat to fill
        repeats = (len(new_index) // len(load_values)) + 1
        load_values = np.tile(load_values, repeats)[: len(new_index)]
    elif len(load_values) > len(new_index):
        load_values = load_values[: len(new_index)]

    # Create new DataFrame
    result = pd.DataFrame({load_df.columns[0]: load_values}, index=new_index)
    result.index.name = "DateTime"

    return result
