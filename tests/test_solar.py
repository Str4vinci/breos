"""Tests for the solar module."""

import pandas as pd
import pytest

from breos.solar import (
    calculate_multi_array_production,
    calculate_pv_production_dc,
    calculate_pv_production_dc_tracking,
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

    def test_loss_overrides_change_production(self, synthetic_weather, porto_location, pv_params):
        kwargs = dict(
            weather_data=synthetic_weather,
            location=porto_location,
            tilt=35,
            surface_azimuth=180,
            n_modules=1,
            pv_params=pv_params,
            freq="h",
        )
        base = calculate_pv_production_dc(**kwargs)
        no_shading = calculate_pv_production_dc(**kwargs, loss_overrides={"shading": 0.0})

        # Removing the 3% shading loss scales production by 1/0.97
        assert no_shading.sum() == pytest.approx(base.sum() / 0.97, rel=1e-6)

    def test_loss_overrides_reject_unknown_component(self, synthetic_weather, porto_location, pv_params):
        with pytest.raises(ValueError, match="Unknown loss component"):
            calculate_pv_production_dc(
                weather_data=synthetic_weather,
                location=porto_location,
                tilt=35,
                surface_azimuth=180,
                n_modules=1,
                pv_params=pv_params,
                freq="h",
                loss_overrides={"typo_component": 1.0},
            )

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


class TestTracking:
    def _fixed(self, weather, loc, pv_params, tilt=35):
        return calculate_pv_production_dc(
            weather_data=weather,
            location=loc,
            tilt=tilt,
            surface_azimuth=180,
            n_modules=1,
            pv_params=pv_params,
            freq="h",
        )

    def _single(self, weather, loc, pv_params, **kw):
        return calculate_pv_production_dc_tracking(
            weather_data=weather,
            location=loc,
            n_modules=1,
            tracking="single_axis",
            pv_params=pv_params,
            freq="h",
            **kw,
        )

    def _dual(self, weather, loc, pv_params, **kw):
        return calculate_pv_production_dc_tracking(
            weather_data=weather,
            location=loc,
            n_modules=1,
            tracking="dual_axis",
            pv_params=pv_params,
            freq="h",
            **kw,
        )

    def test_single_axis_output_shape(self, synthetic_weather, porto_location, pv_params):
        dc = self._single(synthetic_weather, porto_location, pv_params)
        assert isinstance(dc, pd.Series)
        assert len(dc) == len(synthetic_weather)

    def test_single_axis_non_negative(self, synthetic_weather, porto_location, pv_params):
        dc = self._single(synthetic_weather, porto_location, pv_params)
        assert (dc.fillna(0) >= -0.01).all()

    def test_dual_axis_output_shape(self, synthetic_weather, porto_location, pv_params):
        dc = self._dual(synthetic_weather, porto_location, pv_params)
        assert isinstance(dc, pd.Series)
        assert len(dc) == len(synthetic_weather)

    def test_tracking_beats_fixed(self, synthetic_weather, porto_location, pv_params):
        """Single-axis tracker should produce more annual energy than optimal fixed tilt."""
        fixed = self._fixed(synthetic_weather, porto_location, pv_params).sum()
        # No backtracking, no row shading penalty for a fair upper-bound comparison
        single = self._single(synthetic_weather, porto_location, pv_params, backtrack=False, max_angle=90).sum()
        assert single > fixed

    def test_dual_geq_single_geq_fixed(self, synthetic_weather, porto_location, pv_params):
        """Energy hierarchy: dual_axis >= single_axis >= fixed (no-backtrack, full range)."""
        fixed = self._fixed(synthetic_weather, porto_location, pv_params).sum()
        single = self._single(synthetic_weather, porto_location, pv_params, backtrack=False, max_angle=90).sum()
        dual = self._dual(synthetic_weather, porto_location, pv_params).sum()
        assert dual >= single >= fixed

    def test_backtracking_reduces_low_sun_output(self, synthetic_weather, porto_location, pv_params):
        """Backtracking sacrifices some low-sun output to avoid row-to-row shading."""
        no_bt = self._single(synthetic_weather, porto_location, pv_params, backtrack=False, gcr=0.6, max_angle=60).sum()
        bt = self._single(synthetic_weather, porto_location, pv_params, backtrack=True, gcr=0.6, max_angle=60).sum()
        # With high GCR (0.6) backtracking is non-trivially restrictive
        assert bt < no_bt

    def test_invalid_tracking_mode(self, synthetic_weather, porto_location, pv_params):
        with pytest.raises(ValueError, match="tracking must be"):
            calculate_pv_production_dc_tracking(
                weather_data=synthetic_weather,
                location=porto_location,
                n_modules=1,
                tracking="trinity_axis",
                pv_params=pv_params,
                freq="h",
            )

    def test_zero_ghi_zero_production_tracking(self, porto_location, pv_params):
        idx = pd.date_range("2023-01-01", periods=24, freq="h", tz="UTC")
        weather = pd.DataFrame(
            {"ghi": 0.0, "dni": 0.0, "dhi": 0.0, "temp_air": 10.0, "wind_speed": 3.0},
            index=idx,
        )
        for mode in ("single_axis", "dual_axis"):
            dc = calculate_pv_production_dc_tracking(
                weather_data=weather,
                location=porto_location,
                n_modules=1,
                tracking=mode,
                pv_params=pv_params,
                freq="h",
            )
            assert dc.fillna(0).sum() == pytest.approx(0.0, abs=1.0)


class TestMultiArrayTracking:
    def test_mixed_arrays(self, synthetic_weather, porto_location):
        arrays = [
            {"modules": 100, "tilt": 30, "azimuth": 180},
            {"modules": 100, "tracking": "single_axis", "axis_azimuth": 180, "gcr": 0.35},
            {"modules": 100, "tracking": "dual_axis"},
        ]
        total = calculate_multi_array_production(
            weather_data=synthetic_weather,
            location=porto_location,
            arrays=arrays,
            freq="h",
        )
        assert isinstance(total, pd.Series)
        assert len(total) == len(synthetic_weather)
        assert total.sum() > 0

    def test_unknown_tracking_mode_raises(self, synthetic_weather, porto_location):
        arrays = [{"modules": 100, "tracking": "quad_axis"}]
        with pytest.raises(ValueError, match="unknown tracking mode"):
            calculate_multi_array_production(
                weather_data=synthetic_weather,
                location=porto_location,
                arrays=arrays,
                freq="h",
            )
