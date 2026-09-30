"""
Load profile management module.

This module handles residential and commercial load profiles,
including loading from CSV files, scaling to annual consumption,
and resampling between hourly and 15-minute intervals.
"""

import hashlib
import os
from contextlib import nullcontext
from dataclasses import dataclass
from importlib.resources import as_file
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import Akima1DInterpolator

from breos.resources import rlp_resource
from breos.utils import _datetime_index_seconds, get_hours_per_step, normalise_frequency

LOAD_COLUMN = "Electrical Consumption [W]"

# Units a profile CSV can declare. W and kW are the mean power over each row's
# interval; Wh and kWh are the energy delivered in it.
PROFILE_UNITS = ("W", "kW", "Wh", "kWh")
_UNIT_TO_W = {"W": 1.0, "kW": 1000.0}
_ENERGY_UNIT_TO_WH = {"Wh": 1.0, "kWh": 1000.0}

CUSTOM_PROFILE = "custom"
LOAD_PROFILE_METADATA_KEY = "breos_load_profile"


@dataclass(frozen=True)
class ProfileSpec:
    """One load-profile family: where its CSV comes from and how to read it.

    ``files`` maps a native resolution (``"h"`` or ``"15min"``) to a filename
    pattern in ``rlp_directory``. For a bundled profile the pattern is also the
    packaged filename. The year belongs to the file, so external patterns
    leave it as ``*``. ``columns`` lists the accepted ``(column, unit)``
    pairs, tried in order; when it is empty the file must have exactly one
    value column, in ``unit``.
    """

    key: str
    name: str
    files: Mapping[str, str]
    columns: tuple[tuple[str, str], ...] = ()
    unit: str = "W"
    bundled: bool = False


def _eredes(key: str, variant: str) -> ProfileSpec:
    return ProfileSpec(
        key=key,
        name=f"E-REDES BTN {variant}",
        files={"h": "EREDES_*_BTN_1000kwh_hourly.csv", "15min": "EREDES_*_BTN_1000kwh_15min.csv"},
        columns=((f"BTN {variant} - Wh", "Wh"),),
    )


PROFILES: dict[str, ProfileSpec] = {
    spec.key: spec
    for spec in (
        ProfileSpec(
            key="demandlib_h0",
            name="H0 standard load profile (demandlib)",
            files={"h": "h0SLP_demandlib_1000kwh_hourly.csv", "15min": "h0SLP_demandlib_1000kwh_15min.csv"},
            # The hourly file is in W and the 15-minute h0_dyn file in kW. The
            # kW header is the one older copies of the hourly file carry.
            columns=(
                ("Electrical Consumption [W]", "W"),
                ("Electrical Consumption [kW]", "kW"),
                ("h0_dyn", "kW"),
            ),
            bundled=True,
        ),
        _eredes("eredes_btn_a", "A"),
        _eredes("eredes_btn_b", "B"),
        _eredes("eredes_btn_c", "C"),
        ProfileSpec(key="bdew_h0", name="BDEW H0 (BDEW publication)", files={"15min": "bdew_h0_*_15min.csv"}),
        ProfileSpec(
            key="ree_2.0td",
            name="REE 2.0TD",
            files={"h": "REE_*_2.0TD_1000kwh_hourly.csv", "15min": "REE_*_2.0TD_1000kwh_15min.csv"},
        ),
        ProfileSpec(key=CUSTOM_PROFILE, name="Custom CSV (load_profile_file)", files={}),
    )
}
PROFILE_KEYS: tuple[str, ...] = tuple(PROFILES)

# Keys removed in 0.7.0. They are not accepted; they only make the error say
# which key replaces them.
_REMOVED_KEYS = {
    "1": "'demandlib_h0'",
    "4": "'eredes_btn_a'",
    "5": "'eredes_btn_b'",
    "6": "'eredes_btn_c'",
    "7": "'bdew_h0'",
    "8": "'ree_2.0td'",
    "h0": "'demandlib_h0' (bundled) or 'bdew_h0' (the BDEW publication)",
    "default": "'demandlib_h0'",
    "crest": "'custom' with load_profile_file for a CREST export ('crest' loaded the demandlib H0 profile)",
}


def resolve_profile_key(profile_type: Any) -> str:
    """Return the canonical key for a load-profile name.

    Keys are case-insensitive. The numeric keys and the aliases ``h0``,
    ``default`` and ``crest`` were removed in 0.7.0 and raise, naming the key
    that replaces them.

    Raises:
        ValueError: If the key is unknown or removed.
    """
    key = str(profile_type).strip().lower()
    if key in PROFILES:
        return key
    if key in _REMOVED_KEYS:
        raise ValueError(
            f"load_profile {profile_type!r} was removed in BREOS 0.7.0; use {_REMOVED_KEYS[key]}. "
            f"Valid keys: {', '.join(PROFILE_KEYS)}."
        )
    raise ValueError(f"Unknown load_profile {profile_type!r}. Valid keys: {', '.join(PROFILE_KEYS)}.")


@dataclass(frozen=True)
class ProfileSource:
    """The CSV a load profile is read from.

    ``native_freq`` is None for an explicitly named file; its resolution is
    then taken from its row count.
    """

    key: str
    source: Any
    packaged: bool
    native_freq: str | None

    @property
    def label(self) -> str:
        """The packaged filename, or the external file's absolute path."""
        if self.packaged:
            return str(self.source.name)
        return str(Path(self.source).resolve())


def _resolution_preference(spec: ProfileSpec, freq: str) -> list[str]:
    # Prefer the requested resolution; otherwise resample from the other one.
    return [resolution for resolution in (freq, "15min" if freq == "h" else "h") if resolution in spec.files]


def resolve_profile_file(
    profile_type: Any,
    freq: str = "h",
    rlp_directory: str | os.PathLike[str] | None = None,
    profile_file: str | os.PathLike[str] | None = None,
) -> ProfileSource:
    """Select the CSV for a load profile.

    An explicit ``profile_file`` wins; a relative path is taken inside
    ``rlp_directory`` when that is set. Otherwise a bundled profile is read
    from the package, or from ``rlp_directory`` when one is given, and an
    external profile's filename pattern must match exactly one file in
    ``rlp_directory``. The requested resolution is preferred; a profile that
    has only the other one is resampled.

    Raises:
        ValueError: If the key is unknown, a ``custom`` profile has no file,
            an external profile has no ``rlp_directory``, or a pattern
            matches several files (they are listed).
        FileNotFoundError: If the named file or a matching file is missing.
    """
    key = resolve_profile_key(profile_type)
    spec = PROFILES[key]
    freq = normalise_frequency(freq)

    if profile_file is not None:
        path = Path(profile_file)
        if not path.is_absolute() and rlp_directory is not None:
            path = Path(rlp_directory) / path
        if not path.is_file():
            raise FileNotFoundError(f"Load profile file not found: {path}")
        return ProfileSource(key, path, packaged=False, native_freq=None)

    if key == CUSTOM_PROFILE:
        raise ValueError("load_profile 'custom' needs load_profile_file, the CSV to read.")

    resolutions = _resolution_preference(spec, freq)
    if rlp_directory is None:
        if not spec.bundled:
            patterns = " or ".join(spec.files[resolution] for resolution in resolutions)
            raise ValueError(
                f"Load profile {key!r} ({spec.name}) is not bundled with BREOS: its upstream redistribution "
                f"terms are not confirmed for package release. Pass rlp_directory holding a licensed copy "
                f"named {patterns}, or load_profile_file, or use 'demandlib_h0'."
            )
        resolution = resolutions[0]
        return ProfileSource(key, rlp_resource(spec.files[resolution]), packaged=True, native_freq=resolution)

    root = Path(rlp_directory)
    for resolution in resolutions:
        pattern = spec.files[resolution]
        matches = sorted(path for path in root.glob(pattern) if path.is_file())
        if len(matches) > 1:
            raise ValueError(
                f"Load profile {key!r}: {len(matches)} files in {root} match {pattern!r}: "
                f"{', '.join(path.name for path in matches)}. Keep one, or set load_profile_file to choose."
            )
        if matches:
            return ProfileSource(key, matches[0], packaged=False, native_freq=resolution)
    patterns = " or ".join(repr(spec.files[resolution]) for resolution in resolutions)
    raise FileNotFoundError(f"Load profile {key!r}: no file in {root} matches {patterns}.")


def validate_profile_options(profile_type: Any, column: Any = None, unit: Any = None) -> None:
    """Check the column and unit options against the profile key.

    Only ``custom`` profiles take them, and ``custom`` needs a unit.

    Raises:
        ValueError: If an option is set for a registered profile, a ``custom``
            profile has no unit, or the unit is not one of ``PROFILE_UNITS``.
    """
    key = resolve_profile_key(profile_type)
    if key != CUSTOM_PROFILE:
        given = [name for name, value in (("load_profile_column", column), ("load_profile_unit", unit)) if value]
        if given:
            raise ValueError(
                f"{' and '.join(given)} apply only to load_profile 'custom'; {key!r} fixes its own column and unit."
            )
        return
    if unit is None:
        raise ValueError(f"load_profile 'custom' needs load_profile_unit, one of: {', '.join(PROFILE_UNITS)}.")
    if unit not in PROFILE_UNITS:
        raise ValueError(f"Unknown load_profile_unit {unit!r}. Valid units: {', '.join(PROFILE_UNITS)}.")
    if column is not None and (not isinstance(column, str) or not column):
        raise ValueError("load_profile_column must be a non-empty column name.")


def load_profile(
    profile_type: str,
    annual_consumption_kwh: float,
    start_date: str = "2025-01-01",
    freq: str = "h",
    rlp_directory: Optional[str] = None,
    timezone: Optional[str] = "UTC",
    *,
    profile_file: str | os.PathLike[str] | None = None,
    profile_column: Optional[str] = None,
    profile_unit: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load and scale a residential/commercial load profile.

    This is the main function for loading load profiles. It supports the
    bundled H0SLP demandlib profile, user-supplied external CSVs through
    ``rlp_directory``, scaling to target annual consumption, and hourly or
    15-minute output.

    Args:
        profile_type: Profile key (see ``PROFILES``), case-insensitive:
            ``demandlib_h0`` (bundled), ``eredes_btn_a``/``_b``/``_c``,
            ``bdew_h0``, ``ree_2.0td``, or ``custom``.
        annual_consumption_kwh: Target annual consumption in kWh
        start_date: First day of the profile, 1 January of its year (YYYY-01-01).
            The bundled H0's source days are matched to the study year's
            weekday, Saturday or Sunday near the same calendar date. Other
            profiles are placed by position. A later start would shift every
            season.
        freq: Time frequency ('h' for hourly, '15min' for 15-minute)
        rlp_directory: Directory containing RLP files. When omitted, BREOS
            uses only redistributable packaged profiles. An external
            profile's filename pattern must match exactly one file in it.
        timezone: Timezone for the index. Profile rows are wall-clock local
            behavior (H0 morning/evening peaks), so pass the location's
            timezone to pin them to local time; the simulation aligns load
            and PV by UTC instant. The 'UTC' default keeps the legacy
            UTC-clock convention for callers without a location.
        profile_file: An explicit CSV to read instead of the registry's
            filename pattern; relative paths are taken inside
            ``rlp_directory`` when it is set. Required for ``custom``. Its
            resolution is taken from its row count.
        profile_column: For ``custom``, the column holding the load. Needed
            only when the file has more than one value column.
        profile_unit: For ``custom``, the unit of that column: ``W`` or
            ``kW`` (mean power over each row) or ``Wh`` or ``kWh`` (energy per
            row). Required for ``custom``.

    Returns:
        DataFrame with 'Electrical Consumption [W]' column and DatetimeIndex.
        ``attrs["breos_load_profile"]`` records the canonical key, the file
        read (packaged filename or absolute path) and its SHA-256, its
        native resolution, and the column and unit taken from it.

    Raises:
        ValueError: If profile_type is not recognized or was removed, the
            column or unit options do not fit it, a filename pattern matches
            several files, or start_date is not 1 January
        FileNotFoundError: If the profile's file is missing
    """
    freq = normalise_frequency(freq)
    start_ts = pd.Timestamp(start_date)
    if (start_ts.month, start_ts.day) != (1, 1) or start_ts != start_ts.normalize():
        raise ValueError(
            f"start_date must be 1 January at midnight, got {start_date!r}. The profile's first row is "
            "1 January, so a later start would move every season; use "
            f"'{start_ts.year}-01-01'."
        )
    validate_profile_options(profile_type, profile_column, profile_unit)
    source = resolve_profile_file(profile_type, freq, rlp_directory, profile_file)
    spec = PROFILES[source.key]
    if source.key == CUSTOM_PROFILE:
        columns = ((profile_column, profile_unit),) if profile_column else ()
        default_unit = str(profile_unit)
    else:
        columns, default_unit = spec.columns, spec.unit

    path_context = as_file(source.source) if source.packaged else nullcontext(source.source)
    with path_context as csv_file:
        csv_path = Path(csv_file)
        df, native_freq, column, unit, source_start = _load_profile_csv(
            csv_path, columns, default_unit, source.native_freq
        )
        sha256 = hashlib.sha256(csv_path.read_bytes()).hexdigest()

    # Create a naive wall-clock index for one real calendar year; rows describe
    # household behavior at local clock time and are pinned to the timezone
    # afterwards. A Jan-Dec leap year therefore has 8784 hours.
    steps_per_hour = 4 if native_freq == "15min" else 1
    end_ts = start_ts + pd.DateOffset(years=1)
    new_index = pd.date_range(start=start_ts, end=end_ts, freq=native_freq, inclusive="left")

    if source.key == "demandlib_h0":
        if (
            source_start is None
            or (source_start.month, source_start.day) != (1, 1)
            or source_start != source_start.normalize()
        ):
            raise ValueError(
                f"The demandlib H0 file {source.label} needs a dated first row at 1 January 00:00 "
                "to align its day types"
            )
        df = _align_h0_day_types(df, new_index, source_start.year, steps_per_hour, source.label)
    else:
        df = _fit_profile_to_calendar(df, new_index, steps_per_hour, source.label)
    df.index = new_index
    df.index.name = "DateTime"

    # Scale the calendar year before timezone localization. The localization
    # helper preserves this integral while reconciling DST gaps.
    scale_to_annual_consumption(df, annual_consumption_kwh)

    df = _localize_wall_clock_index(df, timezone, native_freq)

    # Resample if needed (hourly to 15-min)
    if freq == "15min" and native_freq == "h":
        df = _resample_load_to_15min(df)
    elif freq == "h" and native_freq == "15min":
        df = df.resample("h").mean()

    # Interpolation can slightly change the integral. A final normalization is
    # therefore needed for exact returned energy; for native-resolution output
    # this is a no-op apart from floating-point roundoff.
    scale_to_annual_consumption(df, annual_consumption_kwh)

    df.attrs[LOAD_PROFILE_METADATA_KEY] = {
        "key": source.key,
        "name": spec.name,
        "file": source.label,
        "packaged": source.packaged,
        "sha256": sha256,
        "native_resolution": native_freq,
        "column": column,
        "unit": unit,
    }
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


def _h0_day_type(day: pd.Timestamp) -> int:
    """H0 distinguishes weekdays, Saturdays and Sundays, not individual weekdays."""
    return 0 if day.weekday() < 5 else day.weekday() - 4


def _align_h0_day_types(
    df: pd.DataFrame, target_index: pd.DatetimeIndex, source_year: int, steps_per_hour: int, source: str
) -> pd.DataFrame:
    """Use the nearest source-calendar day of the target's H0 day type.

    The source's daily shape stays near the same month and day. Searches wrap
    around New Year, where late December and early January are both winter.
    For 29 February in a common-year source, 28 February is the anchor.
    The source may be bundled or supplied through ``rlp_directory``.
    """
    steps_per_day = 24 * steps_per_hour
    source_days = pd.date_range(f"{source_year}-01-01", f"{source_year + 1}-01-01", freq="D", inclusive="left")
    if len(df) != len(source_days) * steps_per_day:
        raise ValueError(
            f"Load profile {source} has {len(df)} rows, but its timestamp year {source_year} needs "
            f"{len(source_days) * steps_per_day} at this resolution"
        )
    if target_index[0].year == source_year:
        return df

    source_types = [_h0_day_type(day) for day in source_days]
    chosen: list[int] = []
    for day in target_index[::steps_per_day]:
        anchor = pd.Timestamp(
            year=source_year,
            month=day.month,
            day=28 if day.month == 2 and day.day == 29 and len(source_days) == 365 else day.day,
        )
        anchor_pos = (anchor - source_days[0]).days
        target_type = _h0_day_type(day)
        for distance in range(8):
            offsets = (0,) if distance == 0 else (-distance, distance)
            for offset in offsets:
                pos = (anchor_pos + offset) % len(source_days)
                if source_types[pos] == target_type:
                    chosen.append(pos)
                    break
            else:
                continue
            break
    values = df.to_numpy().reshape(len(source_days), steps_per_day, len(df.columns))
    return pd.DataFrame(values[chosen].reshape(len(target_index), len(df.columns)), columns=df.columns)


_ROWS_PER_YEAR = {8760: "h", 8784: "h", 35040: "15min", 35136: "15min"}


def _load_profile_csv(
    csv_file: Path,
    columns: tuple[tuple[str, str], ...],
    default_unit: str,
    native_freq: Optional[str],
) -> tuple[pd.DataFrame, str, str, str, pd.Timestamp | None]:
    """Read a profile CSV, convert its load column to W, and validate it.

    ``columns`` lists the accepted ``(column, unit)`` pairs in order; when it
    is empty the file must have exactly one value column after its timestamp
    column, read in ``default_unit``. A ``native_freq`` of None is taken from
    the row count. Fully blank rows (such as the trailing ``,,,`` row of
    E-REDES exports) are dropped. The remaining rows must be finite and
    non-negative, and a leading timestamp column, when present, must step at
    the native resolution.

    Returns the frame, its native resolution, and the column and unit read.
    """
    try:
        # Read without an index column so a blank row is blank in every
        # column, including the timestamp, before it is dropped.
        raw = pd.read_csv(csv_file).dropna(how="all").reset_index(drop=True)
    except Exception as e:
        raise ValueError(f"Error loading profile from {csv_file}: {e}") from e
    # A file with a single column holds only values; otherwise the first
    # column is the timestamp (or a row label).
    timestamps = raw.iloc[:, 0] if raw.shape[1] > 1 else None
    if columns:
        found = next(((column, unit) for column, unit in columns if column in raw.columns), None)
        if found is None:
            expected = " or ".join(repr(column) for column, _ in columns)
            raise ValueError(f"Load profile {csv_file} needs the column {expected}; found {list(raw.columns)}.")
        column, unit = found
    else:
        values = list(raw.columns[1:]) if timestamps is not None else list(raw.columns)
        if len(values) != 1:
            raise ValueError(
                f"Load profile {csv_file} has {len(values)} value columns ({values}); "
                "set load_profile_column to the one holding the load."
            )
        column, unit = str(values[0]), default_unit

    if native_freq is None:
        native_freq = _ROWS_PER_YEAR.get(len(raw))
        if native_freq is None:
            raise ValueError(
                f"Load profile {csv_file} has {len(raw)} data rows; one year is 8760 or 8784 hourly rows, "
                "or 35040 or 35136 15-minute rows. Rows are placed on the calendar by position."
            )

    df = raw[[column]].rename(columns={column: LOAD_COLUMN})
    source_start = _validate_profile_rows(df, timestamps, native_freq, csv_file)
    if unit in _ENERGY_UNIT_TO_WH:
        df[LOAD_COLUMN] *= _ENERGY_UNIT_TO_WH[unit] / get_hours_per_step(native_freq)
    else:
        df[LOAD_COLUMN] *= _UNIT_TO_W[unit]
    return df, native_freq, column, unit, source_start


def _validate_profile_rows(
    df: pd.DataFrame, timestamps: Optional[pd.Series], native_freq: str, csv_file: Path
) -> pd.Timestamp | None:
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
    if timestamps is None or pd.api.types.is_numeric_dtype(timestamps):
        return None
    stamps = _parse_profile_timestamps(timestamps, csv_file)
    if stamps is None:
        return None
    step = pd.Timedelta(pd.tseries.frequencies.to_offset(native_freq))
    irregular = np.flatnonzero(stamps.diff().iloc[1:].to_numpy() != step.to_timedelta64())
    if irregular.size:
        row = int(irregular[0]) + 1
        raise ValueError(
            f"Load profile {csv_file} is not evenly spaced at {native_freq}: data row {row} "
            f"({timestamps.iloc[row]}) follows {timestamps.iloc[row - 1]}."
        )
    return stamps.iloc[0]


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


def _resample_load_to_15min(df: pd.DataFrame) -> pd.DataFrame:
    """Resample hourly mean load to 15-minute steps, keeping each hour's mean.

    Each row is the mean power over the hour that starts at its label. The
    values are placed at the middle of their hour, interpolated with Makima
    at the middle of each quarter-hour, clipped at zero, and then scaled per
    hour so the four quarter-hours average exactly to the source value. The
    energy of every hour, and so of the year, is unchanged.
    """
    if len(df.index) > 1 and not np.all(df.index[1:] - df.index[:-1] == pd.Timedelta(hours=1)):
        raise ValueError("Hourly load must have a regular hourly index to resample to 15 minutes")
    target_index = pd.date_range(start=df.index[0], periods=4 * len(df.index), freq="15min", name=df.index.name)
    x_source = _datetime_index_seconds(df.index) + 1800.0
    x_target = _datetime_index_seconds(target_index) + 450.0
    # Makima does not extrapolate: the first and last quarter-hours outside
    # the span of hour midpoints hold the edge hour's value.
    x_target = np.clip(x_target, x_source[0], x_source[-1])

    df_15min = pd.DataFrame(index=target_index)
    for col in df.columns:
        hourly = df[col].to_numpy(dtype=float)
        if len(hourly) > 1:
            quarters = Akima1DInterpolator(x_source, hourly, method="makima")(x_target)
        else:
            quarters = np.repeat(hourly, 4)
        blocks = np.clip(quarters, 0.0, None).reshape(len(hourly), 4)
        block_means = blocks.mean(axis=1)
        scalable = block_means > 0.0
        blocks[scalable] *= (hourly[scalable] / block_means[scalable])[:, None]
        blocks[~scalable] = hourly[~scalable, None]
        df_15min[col] = blocks.reshape(-1)
    return df_15min
