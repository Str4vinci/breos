"""Tests for breos.utils."""

import numpy as np
import pandas as pd
import pytest

from breos.utils import (
    get_hours_per_step,
    get_steps_per_day,
    get_steps_per_year,
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


@pytest.mark.parametrize("freq", ["h", "H", "1h", "1H"])
def test_hourly_frequency_aliases(freq):
    assert get_hours_per_step(freq) == pytest.approx(1.0)
    assert get_steps_per_day(freq) == 24
    assert get_steps_per_year(freq) == 8760


@pytest.mark.parametrize("freq", ["15min", "15T", "15m"])
def test_15min_frequency_aliases(freq):
    assert get_hours_per_step(freq) == pytest.approx(0.25)
    assert get_steps_per_day(freq) == 96
    assert get_steps_per_year(freq, leap_year=True) == 35136
