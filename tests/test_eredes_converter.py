"""The E-REDES publication converter, on synthetic publications only (#303).

The real E-REDES files are not redistributed with BREOS, so these tests build
publication-shaped files: Latin-1 text, four header rows, English dates and
weekdays, interval-end legal times with the fall-back hour marked ``a``, and a
trailing all-empty row.
"""

import numpy as np
import pandas as pd
import pytest

from breos.load_profiles import load_profile
from tools import convert_eredes_profiles as convert_eredes

YEAR = 2026
HEADER = [
    "Data,,,RESP (MW),Perfis de Consumo e Injeção,,,,",
    ",,,,BTN A,BTN B,BTN C,IP,MP",
    ",,,,,,,,",
    ",,,,,,,,",
]


def _publication_rows(year=YEAR, legal=True):
    """One row per quarter-hour: date, weekday, interval-end label and kWh values."""
    if legal:
        tz = convert_eredes.SOURCE_TIMEZONE
        ends = pd.date_range(
            pd.Timestamp(f"{year}-01-01 00:15", tz=tz), pd.Timestamp(f"{year + 1}-01-01", tz=tz), freq="15min"
        )
        wall = ends.tz_localize(None)
        # The standard-time occurrence of a repeated wall-clock time.
        marked = (ends - pd.Timedelta(hours=1)).tz_localize(None) == wall
    else:
        wall = pd.date_range(f"{year}-01-01 00:15", f"{year + 1}-01-01", freq="15min")
        marked = np.zeros(len(wall), dtype=bool)
    midnight = (wall.hour == 0) & (wall.minute == 0)
    dates = (wall - pd.to_timedelta(midnight.astype(int), unit="D")).normalize()
    labels = np.where(midnight, "24:00", wall.strftime("%H:%M"))
    labels = np.char.add(labels.astype(str), np.where(marked, "a", ""))
    n = len(wall)
    step = np.arange(n)
    return pd.DataFrame(
        {
            "date": [f"{day.day}/{day.strftime('%b')}/{day.year}" for day in dates],
            "weekday": dates.strftime("%a"),
            "time": labels,
            "resp": 5000.0,
            "a": (1 + step % 96) * 1e-4 + (step // 96) * 1e-6,
            "b": 0.02 + step * 1e-8,
            "c": 0.03 + (step % 7) * 1e-4,
            "ip": 0.06,
            "mp": 0.0,
        }
    )


def _write(path, rows, header=HEADER, trailing_blank=True):
    lines = [*header, *(",".join(str(value) for value in row) for row in rows.itertuples(index=False))]
    if trailing_blank:
        lines.append("," * (len(rows.columns) - 1))
    path.write_bytes(("\r\n".join(lines) + "\r\n").encode("latin-1"))
    return path


@pytest.fixture(scope="module")
def legal_rows():
    return _publication_rows()


def test_the_legal_time_publication_becomes_civil_interval_starts(tmp_path, legal_rows):
    source = _write(tmp_path / "Perfil_Consumo_Injecao_E-REDES_2025.csv", legal_rows)
    rows_per_date = legal_rows["date"].value_counts()
    assert (rows_per_date["29/Mar/2026"], rows_per_date["25/Oct/2026"]) == (92, 100)

    quarter_path, hourly_path = convert_eredes.convert(source, tmp_path / "out")

    # The year comes from the dates, not the filename.
    assert (quarter_path.name, hourly_path.name) == convert_eredes.output_names(YEAR)
    assert quarter_path.read_text().splitlines()[0] == "DateTime,BTN A - Wh,BTN B - Wh,BTN C - Wh"
    quarter = pd.read_csv(quarter_path, index_col=0, parse_dates=True)
    hourly = pd.read_csv(hourly_path, index_col=0, parse_dates=True)
    kwh = legal_rows.set_index(legal_rows["date"] + " " + legal_rows["time"])[["a", "b", "c"]]
    kwh.columns = list(convert_eredes.OUTPUT_COLUMNS)

    assert quarter.index.equals(pd.date_range(f"{YEAR}-01-01", periods=35040, freq="15min"))
    assert hourly.index.equals(pd.date_range(f"{YEAR}-01-01", periods=8760, freq="h"))
    # Interval end 00:15 starts at 00:00, and 24:00 on 31 December at 23:45. kWh becomes Wh.
    np.testing.assert_allclose(quarter.iloc[0], 1000 * kwh.loc["1/Jan/2026 00:15"], rtol=1e-11)
    np.testing.assert_allclose(quarter.iloc[-1], 1000 * kwh.loc["31/Dec/2026 24:00"], rtol=1e-11)
    # Spring forward: the interval ending 02:00 legal time started at 00:45;
    # the skipped hour is interpolated between it and 02:00.
    spring = quarter.loc["2026-03-29 00:45":"2026-03-29 02:00"]
    np.testing.assert_allclose(spring.iloc[0], 1000 * kwh.loc["29/Mar/2026 02:00"], rtol=1e-11)
    np.testing.assert_allclose(spring.iloc[-1], 1000 * kwh.loc["29/Mar/2026 02:15"], rtol=1e-11)
    expected = np.linspace(spring.iloc[0].to_numpy(), spring.iloc[-1].to_numpy(), 6)
    np.testing.assert_allclose(spring.to_numpy(), expected, rtol=1e-11)
    # Fall back: the repeated hour keeps its standard-time occurrence.
    for start, end in (("01:00", "01:15a"), ("01:30", "01:45a"), ("01:45", "02:00"), ("00:45", "01:00")):
        np.testing.assert_allclose(quarter.loc[f"2026-10-25 {start}"], 1000 * kwh.loc[f"25/Oct/2026 {end}"], rtol=1e-11)
    # Hourly energy is the sum of its four quarter-hours.
    np.testing.assert_allclose(hourly.to_numpy(), quarter.to_numpy().reshape(-1, 4, 3).sum(axis=1), rtol=1e-11)
    dropped = kwh.loc[[f"25/Oct/2026 {label}" for label in ("01:15", "01:30", "01:45", "01:00a")]].sum()
    added = quarter.loc["2026-03-29 01:00":"2026-03-29 01:45"].sum() / 1000
    np.testing.assert_allclose(quarter.sum() / 1000, kwh.sum() - dropped + added, rtol=1e-10)


def test_converted_files_load_on_their_own_calendar(tmp_path, legal_rows):
    source = _write(tmp_path / "publication.csv", legal_rows)
    convert_eredes.convert(source, tmp_path)
    quarter = pd.read_csv(tmp_path / convert_eredes.output_names(YEAR)[0], index_col=0)

    for freq in ("15min", "h"):
        load = load_profile("eredes_btn_c", 1000, start_date=f"{YEAR}-01-01", freq=freq, rlp_directory=str(tmp_path))
        assert load.attrs["breos_load_profile"]["native_resolution"] == freq
        expected = quarter["BTN C - Wh"].to_numpy()
        if freq == "h":
            expected = expected.reshape(-1, 4).sum(axis=1)
        values = load.iloc[:, 0].to_numpy()
        np.testing.assert_allclose(values / values.sum(), expected / expected.sum(), rtol=1e-10)


@pytest.mark.parametrize("freq", ["15min", "h"])
def test_interpolated_spring_hour_survives_alignment_into_another_year(tmp_path, legal_rows, freq):
    # The converter fills the skipped hour of Sunday 29 March 2026. A 2026
    # Lisbon run drops it again, but Good Friday 30 March 2029 is a
    # Sunday/holiday that takes the whole source date, where 01:00 exists.
    source = _write(tmp_path / "publication.csv", legal_rows)
    quarter_path, hourly_path = convert_eredes.convert(source, tmp_path)
    converted = pd.read_csv(quarter_path if freq == "15min" else hourly_path, index_col=0, parse_dates=True)
    converted = converted["BTN A - Wh"]
    source_day = converted.loc["2026-03-29"].to_numpy()

    load = load_profile(
        "eredes_btn_a",
        3500,
        start_date="2029-01-01",
        freq=freq,
        rlp_directory=str(tmp_path),
        timezone="Europe/Lisbon",
    ).iloc[:, 0]

    civil = load.copy()
    civil.index = civil.index.tz_localize(None)
    civil = civil[~civil.index.duplicated(keep="last")]  # the repeated fall-back hour
    target_day = civil.loc["2029-03-30"].to_numpy()
    assert len(target_day) == len(source_day) == (96 if freq == "15min" else 24)
    np.testing.assert_allclose(target_day / target_day.sum(), source_day / source_day.sum(), rtol=1e-12)
    # The filled hour is kept, still on the straight line between its neighbours.
    if freq == "15min":
        quarters = converted.loc["2026-03-29 00:45":"2026-03-29 02:00"].to_numpy()
        np.testing.assert_allclose(quarters, np.linspace(quarters[0], quarters[-1], 6), rtol=1e-11)
    filled = civil.loc["2029-03-30 01:00":"2029-03-30 01:45"].to_numpy()
    scale = target_day.sum() / source_day.sum()
    expected = converted.loc["2026-03-29 01:00":"2026-03-29 01:45"].to_numpy() * scale
    np.testing.assert_allclose(filled, expected, rtol=1e-12)
    assert (filled > 0).all()

    # The conversion moved the file's annual energy, but the load is scaled to
    # the configured total exactly.
    assert converted.sum() != pytest.approx(1000 * legal_rows["a"].sum(), rel=1e-9)
    hours_per_step = 0.25 if freq == "15min" else 1.0
    assert load.sum() * hours_per_step / 1000 == pytest.approx(3500, rel=1e-12)


def test_a_civil_time_publication_is_read_as_civil_time(tmp_path):
    rows = _publication_rows(year=2024, legal=False)
    assert rows["date"].value_counts().eq(96).all()
    quarter_path, _ = convert_eredes.convert(_write(tmp_path / "civil.csv", rows), tmp_path)

    quarter = pd.read_csv(quarter_path, index_col=0, parse_dates=True)
    assert quarter_path.name == "EREDES_2024_BTN_1000kwh_15min.csv"
    assert quarter.index.equals(pd.date_range("2024-01-01", periods=35136, freq="15min"))
    np.testing.assert_allclose(quarter["BTN A - Wh"].to_numpy(), 1000 * rows["a"].to_numpy(), rtol=1e-11)


def _drop(rows, label):
    return rows[(rows["date"] + " " + rows["time"]) != label].reset_index(drop=True)


def _edit(rows, label, column, value):
    rows = rows.copy()
    rows[column] = rows[column].astype(object)
    rows.loc[(rows["date"] + " " + rows["time"]) == label, column] = value
    return rows


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda rows: _edit(rows, "2/Jan/2026 10:00", "weekday", "Thu"), r"says 'Thu' for 2026-01-02, a Fri"),
        (
            lambda rows: _drop(rows, "2/Jan/2026 10:00"),
            r"has no Europe/Lisbon legal-time interval starting 2026-01-02 09:45",
        ),
        (
            lambda rows: pd.concat([rows.iloc[:200], rows.iloc[199:200], rows.iloc[200:]], ignore_index=True),
            "repeats the Europe/Lisbon legal-time interval",
        ),
        (lambda rows: _edit(rows, "2/Jan/2026 10:00", "b", "-0.1"), r"has '-0.1' for BTN B; values must be finite"),
        (lambda rows: _edit(rows, "2/Jan/2026 10:00", "c", "n/a"), r"has 'n/a' for BTN C"),
        (lambda rows: _edit(rows, "2/Jan/2026 10:00", "date", "2/Jan/26"), r"has the date '2/Jan/26'"),
        (lambda rows: _edit(rows, "2/Jan/2026 10:00", "time", "10:10"), r"has the interval end '10:10'"),
        (lambda rows: rows[~rows["date"].eq("31/Dec/2026")], r"2026-01-01 to 2026-12-30 \(364 dates\).*is 365"),
        (lambda rows: _edit(rows, "31/Dec/2026 24:00", "date", "1/Jan/2027"), "366 dates"),
        (lambda rows: _edit(rows, "29/Mar/2026 02:00", "time", "01:00"), "not a Europe/Lisbon legal time"),
        (lambda rows: _edit(rows, "25/Oct/2026 01:15a", "time", "01:15"), "repeats the Europe/Lisbon legal-time"),
    ],
)
def test_a_damaged_publication_is_refused(tmp_path, legal_rows, change, message):
    source = _write(tmp_path / "damaged.csv", change(legal_rows))

    with pytest.raises(ValueError, match=message):
        convert_eredes.read_publication(source)


def test_a_file_without_the_btn_columns_is_refused(tmp_path, legal_rows):
    header = [HEADER[0], HEADER[1].replace("BTN B", "BTN X"), *HEADER[2:]]
    with pytest.raises(ValueError, match=r"needs one 'BTN B' column"):
        convert_eredes.read_publication(_write(tmp_path / "other.csv", legal_rows, header=header))


def test_the_command_line_refuses_to_overwrite(tmp_path, legal_rows, capsys):
    source = _write(tmp_path / "publication.csv", legal_rows)
    output = tmp_path / "rlp"

    assert convert_eredes.main([str(source), "--output-dir", str(output)]) == 0
    assert capsys.readouterr().out.splitlines() == [str(output / name) for name in convert_eredes.output_names(YEAR)]
    assert convert_eredes.main([str(source), "--output-dir", str(output)]) == 1
    assert "pass --overwrite" in capsys.readouterr().err
    assert convert_eredes.main([str(source), "--output-dir", str(output), "--overwrite"]) == 0
