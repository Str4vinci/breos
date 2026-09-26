"""
Load profile management module.

This module handles residential and commercial load profiles,
including loading from CSV files, scaling to annual consumption,
and resampling between hourly and 15-minute intervals.
"""

from contextlib import nullcontext
from importlib.resources import as_file
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from breos.resources import rlp_resource
from breos.utils import normalise_frequency
from breos.weather import resample_to_15min

# Profile type mappings
PROFILE_FILES = {
    "1": "h0SLP_demandlib_1000kwh_hourly.csv",
    "4": "EREDES_2025_BTN_1000kwh_hourly.csv",
    "5": "EREDES_2025_BTN_1000kwh_hourly.csv",
    "6": "EREDES_2025_BTN_1000kwh_hourly.csv",
    "7": "bdew_h0_2025_15min.csv",
    "8": "REE_2026_2.0TD_1000kwh_hourly.csv",
}

PROFILE_FILES_15MIN = {
    "1": "h0SLP_demandlib_1000kwh_15min.csv",
    "4": "EREDES_2025_BTN_1000kwh_15min.csv",
    "5": "EREDES_2025_BTN_1000kwh_15min.csv",
    "6": "EREDES_2025_BTN_1000kwh_15min.csv",
    "7": "bdew_h0_2025_15min.csv",
    "8": "REE_2026_2.0TD_1000kwh_15min.csv",
}

PROFILE_FILE_NATIVE_FREQ = {
    "7": "15min",
}

PROFILE_NAMES = {
    "1": "H0SLP (demandlib)",
    "4": "E-Redes 2025 - BTN A (external file required)",
    "5": "E-Redes 2025 - BTN B (external file required)",
    "6": "E-Redes 2025 - BTN C (external file required)",
    "7": "BDEW H0 2025 (external file required)",
    "8": "REE 2026 - 2.0TD (external file required)",
}

# Note: "bdew_h0" maps to the bundled demandlib profile "1", which
# implements the BDEW H0 standard shape. Profile "7" is the externally
# published BDEW H0 2025 file and must be supplied via rlp_directory —
# request it explicitly as "7" if that exact dataset is needed.
PROFILE_ALIASES = {
    "default": "1",
    "demandlib_h0": "1",
    "h0": "1",
    "bdew_h0": "1",
    "crest": "1",
    "eredes_btn_c": "6",
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

    This is the main function for loading load profiles. It supports the
    bundled H0SLP demandlib profile, user-supplied external CSVs through
    ``rlp_directory``, scaling to target annual consumption, multi-year
    extension, and hourly or 15-minute output.

    Args:
        profile_type: Profile type key (see PROFILE_NAMES) or name
        annual_consumption_kwh: Target annual consumption in kWh
        start_date: First day of the profile, 1 January of its year (YYYY-01-01).
            Profile rows are stamped from 1 January onward, so any other date
            would shift every season.
        freq: Time frequency ('h' for hourly, '15min' for 15-minute)
        num_years: Number of years to generate
        rlp_directory: Directory containing RLP files. When omitted, BREOS
            uses only redistributable packaged profiles.
        timezone: Timezone for the index. Profile rows are wall-clock local
            behavior (H0 morning/evening peaks), so pass the location's
            timezone to pin them to local time; the simulation aligns load
            and PV by UTC instant. The 'UTC' default keeps the legacy
            UTC-clock convention for callers without a location.

    Returns:
        DataFrame with 'Electrical Consumption [W]' column and DatetimeIndex

    Raises:
        ValueError: If profile_type is not recognized, or start_date is not
            1 January
    """
    freq = normalise_frequency(freq)
    start_ts = pd.Timestamp(start_date)
    if (start_ts.month, start_ts.day) != (1, 1) or start_ts != start_ts.normalize():
        raise ValueError(
            f"start_date must be 1 January at midnight, got {start_date!r}. The profile's first row is "
            "1 January, so a later start would move every season; use "
            f"'{start_ts.year}-01-01'."
        )
    profile_type = PROFILE_ALIASES.get(str(profile_type).lower(), str(profile_type))
    if profile_type not in PROFILE_FILES:
        raise ValueError(f"Unknown profile type: {profile_type}. Valid types: {list(PROFILE_FILES.keys())}")

    packaged = rlp_directory is None
    rlp_path = None if packaged else Path(rlp_directory)

    def _candidate(filename: str):
        return rlp_resource(filename) if packaged else rlp_path / filename

    def _exists(candidate) -> bool:
        return candidate.is_file() if packaged else candidate.exists()

    # Prefer native 15-minute files when requested. If only a native
    # 15-minute external profile is available, load it and downsample later.
    native_candidate = _candidate(PROFILE_FILES_15MIN[profile_type]) if profile_type in PROFILE_FILES_15MIN else None
    use_native_15min = freq == "15min" and profile_type in PROFILE_FILES_15MIN and _exists(native_candidate)

    if use_native_15min:
        csv_resource = native_candidate
        native_freq = "15min"
    else:
        hourly_candidate = _candidate(PROFILE_FILES[profile_type])
        if _exists(hourly_candidate):
            csv_resource = hourly_candidate
            native_freq = PROFILE_FILE_NATIVE_FREQ.get(profile_type, "h")
        elif profile_type in PROFILE_FILES_15MIN and _exists(native_candidate):
            csv_resource = native_candidate
            native_freq = "15min"
        else:
            csv_resource = hourly_candidate
            native_freq = "h"

    if not _exists(csv_resource):
        if packaged:
            profile_name = PROFILE_NAMES.get(profile_type, profile_type)
            raise ValueError(
                f"Profile type {profile_type!r} ({profile_name}) is not bundled with BREOS. "
                "Its upstream redistribution terms are not confirmed for package release. "
                "Pass rlp_directory with a licensed local copy or use profile_type='1'."
            )
        raise FileNotFoundError(f"Load profile file not found: {csv_resource}")

    # Load the profile
    path_context = as_file(csv_resource) if packaged else nullcontext(csv_resource)
    with path_context as csv_file:
        df = _load_profile_csv(Path(csv_file), profile_type, native_freq)

    # Create a naive wall-clock index for one real calendar year; rows describe
    # household behavior at local clock time and are pinned to the timezone
    # afterwards.  A Jan-Dec leap year therefore has 8784 hours.
    start_year = int(start_date[:4])
    steps_per_hour = 4 if native_freq == "15min" else 1
    end_ts = start_ts + pd.DateOffset(years=1)
    new_index = pd.date_range(start=start_ts, end=end_ts, freq=native_freq, inclusive="left")

    df = _fit_profile_to_calendar(df, new_index, steps_per_hour, csv_resource)
    df.index = new_index
    df.index.name = "DateTime"

    # Extend to multiple years if needed (on the naive wall-clock index, so
    # each year is localized at its own DST transition dates below)
    if num_years > 1:
        df = _extend_to_years(df, start_year, num_years)

    # Scale the complete requested calendar before timezone localization. The
    # localization helper preserves this integral while reconciling DST gaps.
    scale_to_annual_consumption(df, annual_consumption_kwh * num_years)

    df = _localize_wall_clock_index(df, timezone, native_freq)

    # Resample if needed (hourly to 15-min)
    if freq == "15min" and native_freq == "h":
        df = _resample_load_to_15min(df)
    elif freq == "h" and native_freq == "15min":
        df = df.resample("h").mean()

    # Interpolation can slightly change the integral. A final normalization is
    # therefore needed for exact returned energy; for native-resolution output
    # this is a no-op apart from floating-point roundoff.
    scale_to_annual_consumption(df, annual_consumption_kwh * num_years)

    return df


def _localize_wall_clock_index(df: pd.DataFrame, timezone: Optional[str], freq: str) -> pd.DataFrame:
    """Pin naive wall-clock profile rows to a timezone's legal time.

    Household behavior follows the legal clock, so each row keeps its
    wall-clock label. In a DST-observing timezone the spring-forward hour
    does not exist (its rows are dropped) and the fall-back hour occurs
    twice (the rows cover the standard-time occurrence; the DST instants
    are forward-filled), keeping the result evenly spaced in absolute time.
    """
    localized = df.copy()
    if timezone is None or timezone == "UTC":
        localized.index = localized.index.tz_localize("UTC")
        return localized

    target_totals = localized.sum()
    idx = localized.index.tz_localize(timezone, nonexistent="shift_forward", ambiguous=False)
    localized.index = idx
    localized = localized[~localized.index.duplicated(keep="last")]
    full = pd.date_range(localized.index[0], localized.index[-1], freq=freq)
    synthesized = ~full.isin(localized.index)
    localized = localized.reindex(full).ffill()

    # Spring-forward removes rows and fall-back creates the same number of
    # absolute-time slots. Reconcile their energy only in the synthesized
    # fall-back slots, preserving ordinary wall-clock profile values exactly.
    if synthesized.any():
        correction = (target_totals - localized.sum()) / int(synthesized.sum())
        localized.loc[synthesized, localized.columns] += correction.to_numpy()
    localized.index.name = "DateTime"
    return localized


def _fit_profile_to_calendar(
    df: pd.DataFrame, new_index: pd.DatetimeIndex, steps_per_hour: int, source
) -> pd.DataFrame:
    """Fit one calendar year of profile rows to the target year's calendar.

    A profile must hold exactly one common or leap year at its native
    resolution. A common-year profile on a leap-year target duplicates
    28 February at the leap-day position, and a leap-year profile on a
    common-year target drops its 29 February, so March onward keeps its
    alignment either way. Any other length raises: repeating or cutting it
    would shift the seasons without a message.
    """
    steps_per_day = 24 * steps_per_hour
    common, leap = 365 * steps_per_day, 366 * steps_per_day
    if len(df) not in (common, leap):
        resolution = "hourly" if steps_per_hour == 1 else "15-minute"
        raise ValueError(
            f"Load profile {source} has {len(df)} data rows; a {resolution} profile must "
            f"have {common} (common year) or {leap} (leap year). Rows are placed on the "
            "calendar by position, so a shorter or longer file would shift the seasons."
        )
    if len(df) == len(new_index):
        return df
    # Day 59 (0-based) is 29 February in a leap year and 1 March otherwise.
    leap_day = 59 * steps_per_day
    if len(df) == common:
        previous_day = df.iloc[leap_day - steps_per_day : leap_day]
        return pd.concat([df.iloc[:leap_day], previous_day, df.iloc[leap_day:]], ignore_index=True)
    return pd.concat([df.iloc[:leap_day], df.iloc[leap_day + steps_per_day :]], ignore_index=True)


def _load_profile_csv(csv_file: Path, profile_type: str, native_freq: str = "h") -> pd.DataFrame:
    """Load a profile CSV file, standardize its column name, and validate it.

    Fully blank rows (such as the trailing ``,,,`` row of E-REDES exports)
    are dropped. The remaining rows must be finite and non-negative, and a
    leading timestamp column, when present, must step at ``native_freq``.
    """
    try:
        # Read without an index column so a blank row is blank in every
        # column, including the timestamp, before it is dropped.
        raw = pd.read_csv(csv_file).dropna(how="all").reset_index(drop=True)
        timestamps = raw.iloc[:, 0]
        if profile_type == "1":
            # H0SLP demandlib format (hourly is stored in W; 15min h0_dyn is kW).
            df = raw.iloc[:, 1:]
            if "Electrical Consumption [W]" in df.columns:
                pass
            elif "Electrical Consumption [kW]" in df.columns:
                # Backwards compatibility for user-supplied files with the
                # historical bundled header.
                df = df.rename(columns={"Electrical Consumption [kW]": "Electrical Consumption [W]"})
                df["Electrical Consumption [W]"] *= 1000
            elif "h0_dyn" in df.columns:
                df = df.rename(columns={"h0_dyn": "Electrical Consumption [W]"})
                df["Electrical Consumption [W]"] *= 1000

        elif profile_type in EREDES_COLUMNS:
            # E-Redes format: one file holds BTN A, B and C, so select by exact name.
            col_name = EREDES_COLUMNS[profile_type]
            if col_name not in raw.columns:
                raise ValueError(f"profile {profile_type} needs the column {col_name!r}; found {list(raw.columns)}")
            df = raw[[col_name]].rename(columns={col_name: "Electrical Consumption [W]"})
        else:
            df = raw.iloc[:, 1:]
            df.columns = ["Electrical Consumption [W]"]

        df = df[["Electrical Consumption [W]"]].copy()
    except Exception as e:
        raise ValueError(f"Error loading profile from {csv_file}: {e}") from e

    _validate_profile_rows(df, timestamps, native_freq, csv_file)
    return df


def _validate_profile_rows(df: pd.DataFrame, timestamps: pd.Series, native_freq: str, csv_file: Path) -> None:
    """Refuse non-numeric, non-finite, negative, or irregularly stamped rows."""
    values = pd.to_numeric(df["Electrical Consumption [W]"], errors="coerce").to_numpy(dtype=float)
    bad = ~np.isfinite(values)
    if bad.any():
        raise ValueError(
            f"Load profile {csv_file} has {int(bad.sum())} missing, non-numeric, or infinite "
            f"values (first at data row {int(np.flatnonzero(bad)[0])})."
        )
    negative = values < 0
    if negative.any():
        raise ValueError(
            f"Load profile {csv_file} has {int(negative.sum())} negative values "
            f"(first at data row {int(np.flatnonzero(negative)[0])}); load must be non-negative."
        )
    df["Electrical Consumption [W]"] = values

    # Rows are placed by position, so a timestamp column is optional. When the
    # file has one, it must step evenly at the profile's resolution; a DST gap
    # or a missing row would otherwise move later rows by one step.
    if pd.api.types.is_numeric_dtype(timestamps):
        return
    stamps = _parse_profile_timestamps(timestamps, csv_file)
    if stamps is None:
        return
    step = pd.Timedelta(pd.tseries.frequencies.to_offset(native_freq))
    irregular = np.flatnonzero(stamps.diff().iloc[1:].to_numpy() != step.to_timedelta64())
    if irregular.size:
        row = int(irregular[0]) + 1
        raise ValueError(
            f"Load profile {csv_file} is not evenly spaced at {native_freq}: data row {row} "
            f"({timestamps.iloc[row]}) follows {timestamps.iloc[row - 1]}."
        )


def _parse_profile_timestamps(timestamps: pd.Series, csv_file: Path) -> Optional[pd.Series]:
    """Parse ISO or day-first (E-REDES) timestamps.

    Returns None when no row parses as a timestamp in either format: the first
    column is then a label, not a time column. Raises when some rows parse and
    others do not, because a damaged time column cannot be checked for gaps.
    """
    best = None
    for kwargs in ({"format": "ISO8601"}, {"format": "%d/%m/%Y %H:%M"}):
        # utc=True compares offset-aware stamps as instants, so a file whose
        # offsets change at DST is evenly spaced; naive stamps are unchanged.
        stamps = pd.to_datetime(timestamps, errors="coerce", utc=True, **kwargs)
        if best is None or stamps.notna().sum() > best.notna().sum():
            best = stamps
    if not best.notna().any():
        return None
    malformed = np.flatnonzero(best.isna().to_numpy())
    if malformed.size:
        row = int(malformed[0])
        raise ValueError(
            f"Load profile {csv_file} has {malformed.size} timestamps that do not parse "
            f"(first at data row {row}: {timestamps.iloc[row]!r})."
        )
    return best


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

    .. warning::
        Values are re-stamped positionally, ignoring timezones — only safe
        when load and PV share the same clock convention (both UTC or both
        the same fixed offset). :func:`breos.battery.simulate_energy_balance`
        performs timezone- and DST-aware alignment internally, so do NOT
        pre-align with this function when passing both series there; the
        optimizer stopped doing so in 0.3.4.

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
