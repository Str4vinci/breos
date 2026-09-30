"""Convert the E-REDES consumption-profile publication to BREOS load-profile files.

E-REDES publishes its yearly profiles as ``Perfil_Consumo_Injecao_E-REDES_<year>.csv``.
BREOS does not bundle them, because their redistribution terms are not
confirmed. This script turns a copy you hold into the two files the
``eredes_btn_a``, ``eredes_btn_b`` and ``eredes_btn_c`` profiles read from
``rlp_directory``::

    python tools/convert_eredes_profiles.py Perfil_Consumo_Injecao_E-REDES_2026.csv --output-dir rlp

It writes ``EREDES_<year>_BTN_1000kwh_15min.csv`` and
``EREDES_<year>_BTN_1000kwh_hourly.csv`` with the columns
``DateTime,BTN A - Wh,BTN B - Wh,BTN C - Wh``. The year is the one the
file's dates cover, not the one in its filename.

The publication is read as follows:

- It is Latin-1 text with four header rows. The BTN A, B and C columns are
  found by their header labels. Blank rows, such as the trailing all-empty
  row, are dropped.
- Each row has a date (``1/Jan/2026`` or ``01/01/2026``), its English weekday
  (``Thu``), the end of its quarter-hour (``00:15`` to ``24:00``) and values
  in kWh per quarter-hour. The dates must cover one complete calendar year,
  and each weekday must match its date.
- The times are Portuguese legal time. On the spring-forward date the hour
  that does not exist is missing (92 rows); on the fall-back date the
  repeated hour is listed twice, with its second, standard-time occurrence
  marked ``a`` (``01:15a``; 100 rows). A file with 96 rows on every date and
  no ``a`` markers is read as civil time instead.
- Every interval end becomes its interval start: ``24:00`` is the next
  midnight, and 15 minutes are subtracted from the instant. The resulting
  starts must step every 15 minutes through the year, with no gap or
  duplicate.

The output is on BREOS's civil clock: 96 quarter-hours on every date, stamped
at their start in local wall-clock time. The repeated fall-back hour keeps its
standard-time occurrence, the one BREOS pins a wall-clock row to when it
localizes a profile. The spring-forward hour, which the legal clock skips, is
linearly interpolated between its neighbours; BREOS drops it again for a
``Europe/Lisbon`` run. Values are converted from kWh to Wh (times 1000). Each
hourly value is the sum of its four quarter-hours, so both files carry the
same days in the same phase. BREOS rescales every profile to the configured
annual consumption, so the small energy change from the clock conversion
does not reach a simulation.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_ENCODING = "latin-1"
HEADER_ROWS = 4
SOURCE_TIMEZONE = "Europe/Lisbon"
VARIANTS = ("A", "B", "C")
OUTPUT_COLUMNS = tuple(f"BTN {variant} - Wh" for variant in VARIANTS)
STEP = pd.Timedelta(minutes=15)
STEPS_PER_DAY = 96

_MONTHS = {
    name: number
    for number, name in enumerate(
        ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"), 1
    )
}
_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_DATE = re.compile(r"^(\d{1,2})/([A-Za-z]{3}|\d{1,2})/(\d{4})$")
_TIME = re.compile(r"^(\d{2}):(\d{2})(a?)$")


def output_names(year: int) -> tuple[str, str]:
    """The 15-minute and hourly filenames BREOS's E-REDES profiles look for."""
    return f"EREDES_{year}_BTN_1000kwh_15min.csv", f"EREDES_{year}_BTN_1000kwh_hourly.csv"


def _fail(source: Path, message: str) -> ValueError:
    return ValueError(f"E-REDES publication {source}: {message}")


def _btn_columns(header: pd.DataFrame, source: Path) -> list[int]:
    columns = []
    for variant in VARIANTS:
        label = f"BTN {variant}"
        found = [int(j) for j in header.columns if (header[j].str.strip() == label).any()]
        if len(found) != 1:
            raise _fail(source, f"needs one {label!r} column in its {HEADER_ROWS} header rows; found {len(found)}")
        columns.append(found[0])
    return columns


def _parse_dates(text: pd.Series, source: Path) -> pd.Series:
    parts = text.str.extract(_DATE)
    months = parts[1].map(
        lambda value: (
            (_MONTHS.get(value.title()) if value.isalpha() else int(value)) if isinstance(value, str) else None
        )
    )
    dates = pd.to_datetime(
        pd.DataFrame({"year": pd.to_numeric(parts[2]), "month": months, "day": pd.to_numeric(parts[0])}),
        errors="coerce",
    )
    bad = np.flatnonzero(dates.isna().to_numpy())
    if bad.size:
        raise _fail(source, f"data row {int(bad[0])} has the date {text.iloc[bad[0]]!r}; expected e.g. 1/Jan/2026")
    return dates


def read_publication(source: str | Path) -> tuple[int, pd.DataFrame]:
    """Read the publication's BTN A/B/C profiles onto BREOS's civil clock.

    Returns the calendar year the dates cover and a frame of kWh per
    quarter-hour, indexed by naive local interval start, with 96 rows on
    every date.

    Raises:
        ValueError: If the file is not the expected layout, its dates do not
            cover one complete calendar year, a weekday does not match its
            date, a value is missing, non-finite or negative, or its
            intervals have a gap or duplicate.
    """
    source = Path(source)
    try:
        raw = pd.read_csv(source, encoding=SOURCE_ENCODING, header=None, dtype=str, keep_default_na=False)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as error:
        raise _fail(source, f"cannot be read as {SOURCE_ENCODING} CSV: {error}") from error
    if raw.shape[1] < 3 + len(VARIANTS) or len(raw) <= HEADER_ROWS:
        raise _fail(source, f"has {raw.shape[1]} columns and {len(raw)} rows; not an E-REDES profile publication")
    columns = _btn_columns(raw.iloc[:HEADER_ROWS], source)
    data = raw.iloc[HEADER_ROWS:].apply(lambda column: column.str.strip())
    data = data[(data != "").any(axis=1)].reset_index(drop=True)

    dates = _parse_dates(data[0], source)
    year = int(dates.iloc[0].year)
    days = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="D", inclusive="left")
    seen = pd.DatetimeIndex(dates.unique())
    if not seen.equals(days):
        raise _fail(
            source,
            f"its dates run from {dates.min().date()} to {dates.max().date()} ({len(seen)} dates); "
            f"it needs every date of one calendar year, which for {year} is {len(days)}",
        )
    weekdays = dates.dt.weekday.map(lambda number: _WEEKDAYS[number])
    wrong = np.flatnonzero((data[1].str.title() != weekdays).to_numpy())
    if wrong.size:
        row = int(wrong[0])
        raise _fail(
            source, f"data row {row} says {data[1].iloc[row]!r} for {dates.iloc[row].date()}, a {weekdays.iloc[row]}"
        )

    times = data[2].str.extract(_TIME)
    hours, minutes = pd.to_numeric(times[0]), pd.to_numeric(times[1])
    bad_time = times[0].isna() | (minutes % 15 != 0) | (minutes >= 60) | (hours > 24) | ((hours == 24) & (minutes != 0))
    if bad_time.any():
        row = int(np.flatnonzero(bad_time.to_numpy())[0])
        raise _fail(source, f"data row {row} has the interval end {data[2].iloc[row]!r}; expected 00:15 to 24:00")
    marked = (times[2] == "a").to_numpy()
    # 24:00 is the next midnight.
    ends = dates + pd.to_timedelta(hours * 60 + minutes, unit="min")

    values = data[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    bad = ~np.isfinite(values) | (values < 0)
    if bad.any():
        row, column = (int(i) for i in np.argwhere(bad)[0])
        raise _fail(
            source,
            f"data row {row} has {data[columns[column]].iloc[row]!r} for BTN {VARIANTS[column]}; values must be finite and non-negative",
        )

    frame = pd.DataFrame(values, columns=list(OUTPUT_COLUMNS))
    rows_per_date = dates.value_counts()
    if not marked.any() and (rows_per_date == STEPS_PER_DAY).all():
        starts = pd.DatetimeIndex(ends - STEP)
        grid = pd.date_range(days[0], periods=len(days) * STEPS_PER_DAY, freq=STEP)
        _check_grid(starts, grid, source, "civil")
        frame.index = starts
        return year, frame

    try:
        local_ends = pd.DatetimeIndex(ends).tz_localize(SOURCE_TIMEZONE, ambiguous=~marked, nonexistent="raise")
    except Exception as error:  # the time zone library's own error types differ
        raise _fail(source, f"has an interval end that is not a {SOURCE_TIMEZONE} legal time: {error}") from error
    starts = local_ends - STEP
    grid = pd.date_range(
        pd.Timestamp(days[0], tz=SOURCE_TIMEZONE),
        pd.Timestamp(f"{year + 1}-01-01", tz=SOURCE_TIMEZONE),
        freq=STEP,
        inclusive="left",
    )
    _check_grid(starts, grid, source, f"{SOURCE_TIMEZONE} legal")

    # The fall-back hour occurs twice on the wall clock: keep its later,
    # standard-time occurrence. The skipped spring-forward hour is
    # interpolated between its neighbours.
    frame.index = starts.tz_localize(None)
    frame = frame[~frame.index.duplicated(keep="last")]
    civil = pd.date_range(days[0], periods=len(days) * STEPS_PER_DAY, freq=STEP)
    frame = frame.reindex(civil).interpolate(method="linear", limit_area="inside")
    return year, frame


def _check_grid(starts: pd.DatetimeIndex, grid: pd.DatetimeIndex, source: Path, clock: str) -> None:
    if starts.equals(grid):
        return
    duplicated = starts[starts.duplicated()]
    if len(duplicated):
        raise _fail(source, f"repeats the {clock}-time interval starting {duplicated[0]}")
    missing = grid.difference(starts)
    if len(missing):
        raise _fail(source, f"has no {clock}-time interval starting {missing[0]}")
    extra = starts.difference(grid)
    if len(extra):
        raise _fail(source, f"has a {clock}-time interval starting {extra[0]}, outside its calendar year")
    raise _fail(source, "lists its intervals out of order")


def to_breos_frames(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The 15-minute and hourly BREOS frames, in Wh per interval, from kWh per quarter-hour."""
    quarter = frame * 1000.0
    hourly_values = quarter.to_numpy().reshape(-1, 4, len(quarter.columns)).sum(axis=1)
    hourly = pd.DataFrame(hourly_values, index=quarter.index[::4], columns=quarter.columns)
    for part in (quarter, hourly):
        part.index.name = "DateTime"
    return quarter, hourly


def convert(source: str | Path, output_dir: str | Path, overwrite: bool = False) -> tuple[Path, Path]:
    """Write the 15-minute and hourly BREOS files for a publication and return their paths.

    Raises:
        FileExistsError: If an output file exists and ``overwrite`` is false.
        ValueError: If the publication does not validate (see
            :func:`read_publication`).
    """
    year, frame = read_publication(source)
    output = Path(output_dir)
    paths = tuple(output / name for name in output_names(year))
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{', '.join(str(path) for path in existing)} exists; pass --overwrite to replace it")
    output.mkdir(parents=True, exist_ok=True)
    for path, part in zip(paths, to_breos_frames(frame)):
        part.to_csv(path, date_format="%Y-%m-%d %H:%M", float_format="%.12g")
    return paths[0], paths[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert an E-REDES consumption-profile publication to BREOS's BTN A/B/C load-profile files."
    )
    parser.add_argument("input", type=Path, help="the publication CSV, Perfil_Consumo_Injecao_E-REDES_<year>.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="directory for the two output files")
    parser.add_argument("--overwrite", action="store_true", help="replace output files that exist")
    args = parser.parse_args(argv)
    try:
        paths = convert(args.input, args.output_dir, overwrite=args.overwrite)
    except (ValueError, FileExistsError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
