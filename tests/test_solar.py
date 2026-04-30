"""Tests for the solar module."""

import pandas as pd
import pytest

from breos.solar import (
    calculate_pv_production_dc,
    default_azimuth,
    estimate_optimal_tilt,
)


class TestTiltAndAzimuth:
    def test_optimal_tilt_northern_hemisphere(self):
        tilt = estimate_optimal_tilt(41.0)  # Porto latitude
        assert 20 <= tilt <= 50

    def test_optimal_tilt_equator(self):
        tilt = estimate_optimal_tilt(0.0)
        assert 0 <= tilt <= 20

    def test_azimuth_northern_hemisphere(self):
        assert default_azimuth(41.0) == 180  # South-facing

    def test_azimuth_southern_hemisphere(self):
        assert default_azimuth(-37.0) == 0  # North-facing


class TestPVProduction:
    def test_output_shape(self, synthetic_weather, porto_location, pv_params):
        dc = calculate_pv_production_dc(
            weather_data=synthetic_weather,
            location=porto_location,
            tilt=35,
            surface_azimuth=180,
            n_modules=1,
            pv_params=pv_params,
            freq="h",
        )
        assert isinstance(dc, pd.Series)
        assert len(dc) == len(synthetic_weather)

    def test_all_non_negative(self, dc_production):
        assert (dc_production >= -0.01).all()  # small tolerance for floating point

    def test_more_modules_more_production(self, synthetic_weather, porto_location, pv_params):
        dc_1 = calculate_pv_production_dc(
            weather_data=synthetic_weather,
            location=porto_location,
            tilt=35,
            surface_azimuth=180,
            n_modules=1,
            pv_params=pv_params,
            freq="h",
        )
        dc_5 = calculate_pv_production_dc(
            weather_data=synthetic_weather,
            location=porto_location,
            tilt=35,
            surface_azimuth=180,
            n_modules=5,
            pv_params=pv_params,
            freq="h",
        )
        assert dc_5.sum() == pytest.approx(dc_1.sum() * 5, rel=0.001)

    def test_zero_ghi_zero_production(self, porto_location, pv_params):
        # Night-time weather: all irradiance = 0
        idx = pd.date_range("2023-01-01", periods=24, freq="h", tz="UTC")
        weather = pd.DataFrame(
            {"ghi": 0.0, "dni": 0.0, "dhi": 0.0, "temp_air": 10.0, "wind_speed": 3.0},
            index=idx,
        )
        dc = calculate_pv_production_dc(
            weather_data=weather,
            location=porto_location,
            tilt=35,
            surface_azimuth=180,
            n_modules=1,
            pv_params=pv_params,
            freq="h",
        )
        assert dc.sum() == pytest.approx(0.0, abs=1.0)
