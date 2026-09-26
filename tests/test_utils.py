"""Tests for breos.utils."""

import numpy as np
import pandas as pd
import pytest

from breos.utils import (
    _datetime_index_seconds,
    get_hours_per_step,
    get_steps_per_day,
    get_steps_per_year,
    normalise_frequency,
    remap_datetime_index_years,
    safe_path_slug,
)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("porto", "porto"),
        ("porto_2024", "porto_2024"),
        ("porto-de", "porto-de"),
        ("PORTO", "porto"),
        ("Berlin", "berlin"),
        ("a", "a"),
        ("a" * 64, "a" * 64),
    ],
)
def test_safe_path_slug_accepts_valid(name, expected):
    assert safe_path_slug(name) == expected


@pytest.mark.parametrize(
    "name",
    [
        "",
        "../etc/passwd",
        "/abs/path",
        "porto/sub",
        r"porto\sub",
        "porto..bad",
        ".hidden",
        "_leading",
        "-leading",
        "porto bad",
        "porto.bad",
        "porto\x00null",
        "a" * 65,
    ],
)
def test_safe_path_slug_rejects_unsafe(name):
    with pytest.raises(ValueError):
        safe_path_slug(name)


def test_safe_path_slug_rejects_non_string():
    with pytest.raises(TypeError):
        safe_path_slug(123)


def test_remap_datetime_index_years_drops_invalid_feb_29():
    idx = pd.date_range("2024-02-28 00:00", periods=72, freq="h", tz="UTC")
    series = pd.Series(np.arange(len(idx)), index=idx)

    remapped = remap_datetime_index_years(series, 1)

    assert len(remapped) == 48
    assert not remapped.index.has_duplicates
    assert pd.Timestamp("2025-02-28 00:00", tz="UTC") in remapped.index
    assert pd.Timestamp("2025-03-01 00:00", tz="UTC") in remapped.index


@pytest.mark.parametrize(("freq", "canonical"), [("h", "h"), ("1h", "h"), ("15min", "15min")])
def test_normalise_frequency_returns_the_canonical_pandas_string(freq, canonical):
    assert normalise_frequency(freq) == canonical
    # The canonical string is one pandas itself accepts.
    assert len(pd.date_range("2025-01-01", periods=2, freq=normalise_frequency(freq))) == 2


@pytest.mark.parametrize("freq", ["H", "1H", "15T", "15m", "30min", "60min", "D", "", None, 15])
def test_normalise_frequency_rejects_other_values_and_names_the_accepted_ones(freq):
    with pytest.raises(ValueError, match=r"Unsupported frequency.*'h', '1h', '15min'"):
        normalise_frequency(freq)


@pytest.mark.parametrize("freq", ["h", "1h"])
def test_hourly_frequency_spellings(freq):
    assert get_hours_per_step(freq) == pytest.approx(1.0)
    assert get_steps_per_day(freq) == 24
    assert get_steps_per_year(freq) == 8760


def test_15min_frequency_spelling():
    assert get_hours_per_step("15min") == pytest.approx(0.25)
    assert get_steps_per_day("15min") == 96
    assert get_steps_per_year("15min", leap_year=True) == 35136


@pytest.mark.parametrize("freq", ["H", "1H", "15T", "15m", "30min"])
def test_step_helpers_reject_pandas_2_aliases_and_unsupported_steps(freq):
    with pytest.raises(ValueError, match="Unsupported frequency"):
        get_hours_per_step(freq)
    with pytest.raises(ValueError, match="Unsupported frequency"):
        get_steps_per_day(freq)


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_datetime_index_seconds_is_the_same_at_every_resolution(unit):
    idx = pd.DatetimeIndex(["1969-12-31 23:00", "2025-06-21 00:00", "2025-06-21 00:15"], tz="UTC")

    seconds = _datetime_index_seconds(idx.as_unit(unit))

    np.testing.assert_array_equal(seconds, [-3600.0, 1750464000.0, 1750464900.0])


def test_datetime_index_seconds_keeps_sub_second_offsets_of_a_nanosecond_index():
    # A nanosecond count this large is past float64's exact-integer range, so
    # converting it whole would round away offsets that the split keeps.
    idx = pd.DatetimeIndex(["2025-06-21 00:00:00.5", "2025-06-21 00:00:00.000001"]).as_unit("ns")

    seconds = _datetime_index_seconds(idx)

    assert seconds[0] == 1750464000.5
    assert seconds[1] == pytest.approx(1750464000.000001, abs=1e-7)
    assert seconds[1] > 1750464000.0
