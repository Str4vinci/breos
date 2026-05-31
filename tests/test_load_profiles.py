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
