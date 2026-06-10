"""Tests for load profile helpers."""

import numpy as np
import pandas as pd
import pytest

from breos.load_profiles import _extend_to_years, load_profile


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


def test_load_profile_accepts_aliases_and_15t_frequency():
    profile = load_profile("bdew_h0", 1000, freq="15T")
    annual_kwh = profile["Electrical Consumption [W]"].sum() * 0.25 / 1000

    assert len(profile) == 35040
    assert annual_kwh == pytest.approx(1000)


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
