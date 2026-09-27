"""Tests for load profile helpers."""

import shutil
from importlib.resources import as_file

import numpy as np
import pandas as pd
import pytest

from breos.load_profiles import PROFILE_FILES, _extend_to_years, _resample_load_to_15min, load_profile
from breos.resources import rlp_resource

_LOAD_COLUMN = "Electrical Consumption [W]"


def test_extend_to_years_duplicates_feb_28_for_leap_day_without_shifting_rest():
    idx = pd.date_range("2023-01-01 00:00", periods=8760, freq="h", tz="UTC")
    profile = pd.DataFrame({"Load": np.arange(len(idx), dtype=float)}, index=idx)

    extended = _extend_to_years(profile, start_year=2024, num_years=1)

    feb_28 = extended.loc[pd.Timestamp("2024-02-28 12:00", tz="UTC"), "Load"]
    feb_29 = extended.loc[pd.Timestamp("2024-02-29 12:00", tz="UTC"), "Load"]
    mar_1 = extended.loc[pd.Timestamp("2024-03-01 00:00", tz="UTC"), "Load"]
    source_mar_1 = profile.loc[pd.Timestamp("2023-03-01 00:00", tz="UTC"), "Load"]

    assert len(extended) == 8784
    assert feb_29 == pytest.approx(feb_28)
    assert mar_1 == pytest.approx(source_mar_1)


def test_load_profile_accepts_profile_aliases_at_15min():
    profile = load_profile("bdew_h0", 1000, freq="15min")
    annual_kwh = profile["Electrical Consumption [W]"].sum() * 0.25 / 1000

    assert len(profile) == 35040
    assert annual_kwh == pytest.approx(1000)


@pytest.mark.parametrize("freq", ["30min", "15T", "H"])
def test_load_profile_rejects_unsupported_frequency(freq):
    # 30min used to return the hourly profile unchanged.
    with pytest.raises(ValueError, match="Unsupported frequency"):
        load_profile("1", 1000, freq=freq)


def test_load_profile_pins_rows_to_local_wall_clock_across_dst():
    # Household behavior follows the legal clock, so a localized profile must
    # keep each row's wall-clock label year-round (UTC instants shift by the
    # DST offset) — an instant-spaced index would only shift the start.
    utc_prof = load_profile("1", 5000, freq="h", timezone="UTC").iloc[:, 0]
    loc_prof = load_profile("1", 5000, freq="h", timezone="Europe/Berlin").iloc[:, 0]

    idx = loc_prof.index
    assert len(loc_prof) == 8760
    assert idx.is_unique and idx.is_monotonic_increasing
    assert idx[0] == pd.Timestamp("2025-01-01 00:00", tz="Europe/Berlin")
    # Evenly spaced in absolute time despite the DST transitions
    assert len(idx.to_series().diff().dropna().unique()) == 1
    # Energy preserved (one dropped spring-forward row, one forward-fill)
    assert float(loc_prof.sum()) / 1000.0 == pytest.approx(5000.0, abs=5.0)

    # Same wall-clock pattern as the UTC profile in winter AND summer
    for day in ("2025-01-15", "2025-07-15"):
        np.testing.assert_allclose(loc_prof[day].to_numpy(), utc_prof[day].to_numpy())
    # Local-calendar DST days have 23 and 25 wall-clock hours
    assert len(loc_prof["2025-03-30"]) == 23
    assert len(loc_prof["2025-10-26"]) == 25


def test_load_profile_utc_default_keeps_legacy_convention():
    profile = load_profile("1", 1000, freq="h")

    assert str(profile.index.tz) == "UTC"
    assert profile.index[0] == pd.Timestamp("2025-01-01 00:00", tz="UTC")
    assert len(profile) == 8760


@pytest.mark.parametrize(
    ("freq", "expected_length", "expected_end", "hours_per_step"),
    [
        ("h", 8784, "2024-12-31 23:00", 1.0),
        ("15min", 35136, "2024-12-31 23:45", 0.25),
    ],
)
@pytest.mark.parametrize("timezone", ["UTC", "Europe/Berlin"])
def test_load_profile_uses_real_leap_calendar_and_preserves_energy(
    freq, expected_length, expected_end, hours_per_step, timezone
):
    profile = load_profile(
        "1",
        4321,
        start_date="2024-01-01",
        freq=freq,
        timezone=timezone,
    )
    load = profile["Electrical Consumption [W]"]

    assert len(profile) == expected_length
    assert profile.index[-1] == pd.Timestamp(expected_end, tz=timezone)
    assert profile.index.is_unique and profile.index.is_monotonic_increasing
    assert profile.index.to_series().diff().dropna().nunique() == 1
    assert load.sum() * hours_per_step / 1000 == pytest.approx(4321, abs=1e-9)

    feb28 = load.loc["2024-02-28"].to_numpy()
    feb29 = load.loc["2024-02-29"].to_numpy()
    np.testing.assert_allclose(feb29, feb28, rtol=0, atol=1e-12)


def test_load_profile_leap_day_does_not_shift_march_profile():
    leap = load_profile("1", 1000, start_date="2024-01-01", freq="h", timezone="UTC")
    canonical = load_profile("1", 1000, start_date="2025-01-01", freq="h", timezone="UTC")

    # Compare against a within-profile reference because each returned calendar
    # is independently scaled to the requested annual energy.
    leap_load = leap.iloc[:, 0]
    canonical_load = canonical.iloc[:, 0]
    assert leap_load.loc["2024-03-01 00:00"] / leap_load.iloc[0] == pytest.approx(
        canonical_load.loc["2025-03-01 00:00"] / canonical_load.iloc[0]
    )


def test_non_bundled_profile_requires_external_directory():
    with pytest.raises(ValueError, match="not bundled"):
        load_profile("eredes_btn_c", 1000)


def test_external_native_15min_profile_can_downsample_to_hourly(tmp_path):
    profile_path = tmp_path / "bdew_h0_2025_15min.csv"
    pd.DataFrame({"Electrical Consumption [W]": np.ones(35040)}).to_csv(profile_path)

    profile = load_profile("7", 1000, freq="h", rlp_directory=str(tmp_path))
    annual_kwh = profile["Electrical Consumption [W]"].sum() / 1000

    assert len(profile) == 8760
    assert annual_kwh == pytest.approx(1000)


@pytest.mark.parametrize("start_date", ["2025-07-01", "2025-01-02", "2025-01-01 06:00"])
def test_load_profile_rejects_a_start_that_would_shift_the_seasons(start_date):
    with pytest.raises(ValueError, match=r"start_date must be 1 January at midnight"):
        load_profile("demandlib_h0", 3500, start_date=start_date)


def _write_eredes(path, n_rows, start="2025-01-01", freq="15min", trailing_blank=True):
    stamps = pd.date_range(start, periods=n_rows, freq=freq)
    day = np.arange(n_rows) // (96 if freq == "15min" else 24)
    frame = pd.DataFrame(
        {
            "Datetime": stamps.strftime("%d/%m/%Y %H:%M"),
            "BTN A - Wh": 1.0 + day,
            "BTN B - Wh": 2.0 + day,
            "BTN C - Wh": 1000.0 + day,
        }
    )
    text = frame.to_csv(index=False)
    if trailing_blank:
        text += ",,,\n"
    path.write_text(text)


@pytest.mark.parametrize("timezone", ["UTC", "Europe/Lisbon"])
def test_eredes_file_with_trailing_blank_row_keeps_leap_calendar(tmp_path, timezone):
    # The E-REDES exports carry 35040 intervals plus a trailing ",,," row.
    # Column BTN C is 1000 + day-of-year, so each day is identifiable.
    _write_eredes(tmp_path / "EREDES_2025_BTN_1000kwh_15min.csv", 35040)

    load = load_profile(
        "6", 1000, start_date="2024-01-01", freq="15min", rlp_directory=str(tmp_path), timezone=timezone
    ).iloc[:, 0]
    daily = load.groupby(load.index.date).mean()
    reference = daily.iloc[0] / 1000.0  # scale of the source's first day

    assert len(load) == 35136
    assert np.isfinite(load).all()
    assert daily[pd.Timestamp("2024-02-29").date()] == pytest.approx(daily[pd.Timestamp("2024-02-28").date()])
    # 1 March and 31 December keep the source's own days (day 59 and day 364).
    assert daily[pd.Timestamp("2024-03-01").date()] == pytest.approx(1059 * reference, rel=1e-3)
    assert daily[pd.Timestamp("2024-12-31").date()] == pytest.approx(1364 * reference, rel=1e-3)


def test_eredes_profile_selects_its_own_column_by_exact_name(tmp_path):
    _write_eredes(tmp_path / "EREDES_2025_BTN_1000kwh_hourly.csv", 8760, freq="h")
    text = (tmp_path / "EREDES_2025_BTN_1000kwh_hourly.csv").read_text()
    (tmp_path / "EREDES_2025_BTN_1000kwh_hourly.csv").write_text(text.replace("BTN C - Wh", "Other"))

    with pytest.raises(ValueError, match="needs the column 'BTN C - Wh'"):
        load_profile("6", 1000, rlp_directory=str(tmp_path))
    assert len(load_profile("4", 1000, rlp_directory=str(tmp_path))) == 8760


@pytest.mark.parametrize(("freq", "n_rows"), [("h", 8759), ("h", 8761), ("15min", 35037), ("15min", 35137)])
def test_external_profile_of_the_wrong_length_raises(tmp_path, freq, n_rows):
    name = "EREDES_2025_BTN_1000kwh_15min.csv" if freq == "15min" else "EREDES_2025_BTN_1000kwh_hourly.csv"
    _write_eredes(tmp_path / name, n_rows, freq=freq)

    with pytest.raises(ValueError, match=f"has {n_rows} data rows"):
        load_profile("6", 1000, freq=freq, rlp_directory=str(tmp_path))


def test_leap_year_profile_on_common_year_drops_its_leap_day(tmp_path):
    _write_eredes(tmp_path / "EREDES_2025_BTN_1000kwh_hourly.csv", 8784, start="2024-01-01", freq="h")

    load = load_profile("6", 1000, start_date="2025-01-01", rlp_directory=str(tmp_path)).iloc[:, 0]
    daily = load.groupby(load.index.date).mean()
    reference = daily.iloc[0] / 1000.0

    assert len(load) == 8760
    assert daily[pd.Timestamp("2025-02-28").date()] == pytest.approx(1058 * reference)
    assert daily[pd.Timestamp("2025-03-01").date()] == pytest.approx(1060 * reference)
    assert daily[pd.Timestamp("2025-12-31").date()] == pytest.approx(1365 * reference)


@pytest.mark.parametrize(
    ("value", "message"),
    [("", "missing, non-numeric, or infinite"), ("inf", "missing, non-numeric, or infinite"), ("-1", "negative")],
)
def test_external_profile_rejects_bad_values(tmp_path, value, message):
    values = ["100"] * 8760
    values[1234] = value
    stamps = pd.date_range("2025-01-01", periods=8760, freq="h").strftime("%Y-%m-%d %H:%M:%S")
    lines = ["DateTime,Electrical Consumption [W]", *(f"{t},{v}" for t, v in zip(stamps, values))]
    (tmp_path / "REE_2026_2.0TD_1000kwh_hourly.csv").write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError, match=rf"{message}.*data row 1234"):
        load_profile("8", 1000, rlp_directory=str(tmp_path))


def test_external_profile_rejects_irregular_timestamps(tmp_path):
    # A local-clock file with a DST gap has the right length but moves every
    # later row by one hour.
    stamps = pd.date_range("2025-01-01", periods=8761, freq="h").delete(2000)
    frame = pd.DataFrame({"Electrical Consumption [W]": np.ones(8760)}, index=stamps)
    frame.to_csv(tmp_path / "REE_2026_2.0TD_1000kwh_hourly.csv")

    with pytest.raises(ValueError, match="not evenly spaced at h: data row 2000"):
        load_profile("8", 1000, rlp_directory=str(tmp_path))


def test_external_profile_rejects_a_damaged_timestamp_column(tmp_path):
    # One malformed stamp used to switch the spacing check off, so this file's
    # missing hour went through.
    stamps = pd.date_range("2025-01-01", periods=8761, freq="h").delete(2000).strftime("%Y-%m-%d %H:%M:%S").tolist()
    stamps[5000] = "not a time"
    lines = ["DateTime,Electrical Consumption [W]", *(f"{t},100" for t in stamps)]
    (tmp_path / "REE_2026_2.0TD_1000kwh_hourly.csv").write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError, match="1 timestamps that do not parse.*data row 5000: 'not a time'"):
        load_profile("8", 1000, rlp_directory=str(tmp_path))


def test_external_profile_with_a_label_column_skips_the_spacing_check(tmp_path):
    lines = ["Label,Electrical Consumption [W]", *(f"row{i},100" for i in range(8760))]
    (tmp_path / "REE_2026_2.0TD_1000kwh_hourly.csv").write_text("\n".join(lines) + "\n")

    assert len(load_profile("8", 1000, rlp_directory=str(tmp_path))) == 8760


def test_external_profile_with_dst_offsets_is_evenly_spaced(tmp_path):
    # Offsets change at DST; the instants still step by one hour.
    stamps = pd.date_range("2025-01-01", periods=8760, freq="h", tz="Europe/Lisbon").astype(str)
    lines = ["DateTime,Electrical Consumption [W]", *(f"{t},100" for t in stamps)]
    (tmp_path / "REE_2026_2.0TD_1000kwh_hourly.csv").write_text("\n".join(lines) + "\n")

    assert len(load_profile("8", 1000, rlp_directory=str(tmp_path))) == 8760


def _hourly_only_rlp_directory(tmp_path):
    """An external profile directory with the bundled hourly H0 file and no 15-minute file."""
    with as_file(rlp_resource(PROFILE_FILES["1"])) as source:
        shutil.copy(source, tmp_path / PROFILE_FILES["1"])
    return str(tmp_path)


def _lag_in_steps(resampled, reference):
    """Vertex of the mean squared error over shifts of -1, 0 and +1 step; > 0 means late."""
    early = np.mean((resampled[:-1] - reference[1:]) ** 2)
    aligned = np.mean((resampled - reference) ** 2)
    late = np.mean((resampled[1:] - reference[:-1]) ** 2)
    return (early - late) / (2.0 * (early - 2.0 * aligned + late)), early, aligned, late


def test_hourly_h0_resampled_to_15min_has_no_lag_and_keeps_each_hours_mean():
    # The bundled 15-minute H0 profile, averaged to hours and resampled back,
    # used to run about 22.5 minutes early and miss each hour's mean by 6 W.
    native = load_profile("1", 1000, freq="15min")[_LOAD_COLUMN]
    hourly = native.resample("h").mean().to_frame()

    resampled = _resample_load_to_15min(hourly)[_LOAD_COLUMN]

    assert resampled.index.equals(native.index)
    lag, early, aligned, late = _lag_in_steps(resampled.to_numpy(), native.to_numpy())
    assert abs(lag) < 0.05
    assert aligned < early and aligned < late
    hour_means = resampled.to_numpy().reshape(-1, 4).mean(axis=1)
    np.testing.assert_allclose(hour_means, hourly[_LOAD_COLUMN].to_numpy(), rtol=0, atol=1e-9)
    assert resampled.min() >= 0.0


@pytest.mark.parametrize("timezone", ["UTC", "Europe/Berlin"])
def test_load_profile_from_an_hourly_file_keeps_each_hours_mean_at_15min(tmp_path, timezone):
    directory = _hourly_only_rlp_directory(tmp_path)
    hourly = load_profile("1", 3500, freq="h", rlp_directory=directory, timezone=timezone)[_LOAD_COLUMN]

    quarter = load_profile("1", 3500, freq="15min", rlp_directory=directory, timezone=timezone)[_LOAD_COLUMN]

    assert len(quarter) == 4 * len(hourly)
    assert quarter.index[0] == hourly.index[0]
    np.testing.assert_allclose(quarter.to_numpy().reshape(-1, 4).mean(axis=1), hourly.to_numpy(), rtol=0, atol=1e-9)
    assert quarter.sum() * 0.25 / 1000 == pytest.approx(3500, rel=1e-12)


def test_resample_load_rejects_an_irregular_index():
    idx = pd.DatetimeIndex(["2025-01-01 00:00", "2025-01-01 01:00", "2025-01-01 03:00"], tz="UTC")
    with pytest.raises(ValueError, match="regular hourly index"):
        _resample_load_to_15min(pd.DataFrame({_LOAD_COLUMN: [1.0, 2.0, 3.0]}, index=idx))
