"""The year-1 and monthly irradiance keys and performance ratio of an App result."""

import pytest

from tests.offline_app import BASE, simulate_offline

PV_ONLY = {**BASE, "battery_kwh": 0}
SOUTH = {"tilt": 35, "azimuth": 180}
EAST = {"tilt": 35, "azimuth": 90}


def _result(**config):
    return simulate_offline({**PV_ONLY, **config}).result()


def test_year_keys_are_the_sum_of_the_monthly_rows():
    result = _result(n_modules=8)

    for key in ("global_horizontal_kwh_m2", "global_incident_kwh_m2"):
        assert result[key] == pytest.approx(sum(row[key] for row in result["monthly"]), abs=0.1)
    poa_kwh = result["global_incident_kwh_m2"] * result["pv_kwp"]
    assert result["performance_ratio_pct"] == pytest.approx(
        result["usable_ac_system_production_kwh"] / poa_kwh * 100, rel=1e-3
    )
    assert 0 < result["performance_ratio_pct"] < 100


def test_split_array_gives_the_same_incident_irradiance():
    one = _result(pv_arrays=[{"modules": 8, **SOUTH}])
    two = _result(pv_arrays=[{"modules": 4, **SOUTH}, {"modules": 4, **SOUTH}])

    assert two["global_horizontal_kwh_m2"] == one["global_horizontal_kwh_m2"]
    assert two["global_incident_kwh_m2"] == pytest.approx(one["global_incident_kwh_m2"], abs=0.01)
    assert two["performance_ratio_pct"] == pytest.approx(one["performance_ratio_pct"], abs=0.01)


def test_incident_irradiance_is_weighted_by_array_power():
    south = _result(pv_arrays=[{"modules": 8, **SOUTH}])["global_incident_kwh_m2"]
    east = _result(pv_arrays=[{"modules": 8, **EAST}])["global_incident_kwh_m2"]
    mixed = _result(pv_arrays=[{"modules": 2, **SOUTH}, {"modules": 6, **EAST}])["global_incident_kwh_m2"]

    assert south != pytest.approx(east, abs=1.0)
    assert mixed == pytest.approx((2 * south + 6 * east) / 8, abs=0.02)


def test_period_window_reports_its_own_irradiance():
    result = _result(n_modules=8, period={"start": "2025-03-01", "end": "2025-03-05"})

    assert len(result["monthly"]) == 1
    assert result["global_horizontal_kwh_m2"] == pytest.approx(result["monthly"][0]["global_horizontal_kwh_m2"])
    assert result["global_incident_kwh_m2"] > 0
