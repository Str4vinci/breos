"""Tests for the solar module."""

import pandas as pd
import pytest

import breos.solar as solar
from breos.solar import (
    PEREZ_MODELS,
    SURFACE_TYPES,
    TRANSPOSITION_MODELS,
    PVModuleParams,
    calculate_multi_array_production,
    calculate_pv_production_dc,
    calculate_pv_production_dc_tracking,
    dc_to_ac,
    default_azimuth,
    estimate_optimal_tilt,
)


def _module_params(**overrides):
    """Generic_400W-style datasheet values for PVModuleParams tests."""
    params = dict(
        Mpp=400,
        Vmp=41.0,
        Imp=9.76,
        Voc=49.3,
        Isc=10.30,
        T_Pmax_pct=-0.35,
        T_Voc_pct=-0.265,
        T_Isc_pct=0.05,
        N_Cells=144,
    )
    params.update(overrides)
    return PVModuleParams(**params)


class TestPVModuleParams:
    def test_gamma_pmp_defaults_to_power_coefficient(self):
        params = _module_params()
        assert params.gamma_pmp == params.T_Pmax_pct

    def test_gamma_pmp_override_is_respected(self):
        # A user-supplied gamma_pmp must not be silently replaced by T_Pmax_pct.
        params = _module_params(gamma_pmp=-0.30)
        assert params.gamma_pmp == -0.30


class TestDcToAc:
    def _dc(self, watts, periods=4):
        idx = pd.date_range("2023-06-01", periods=periods, freq="h", tz="UTC")
        return pd.Series([float(watts)] * periods, index=idx)

    def test_clips_at_ac_nameplate(self):
        # AC nameplate = pv_peak / loading ratio = 8000 W. pvlib's pdc0 is a
        # DC-input limit (pac0 = eta * pdc0), so passing the nameplate as pdc0
        # used to clip ~4% low, at eta * nameplate = 7680 W.
        ac = dc_to_ac(self._dc(20000.0), pv_peak_power_w=10000.0, inverter_loading_ratio=1.25, inverter_efficiency=0.96)
        assert ac.max() == pytest.approx(8000.0)

    def test_full_dc_limit_reaches_nameplate_exactly(self):
        # At pdc == pdc0 (= nameplate / eta) the pvwatts curve outputs pac0.
        ac = dc_to_ac(
            self._dc(8000.0 / 0.96), pv_peak_power_w=10000.0, inverter_loading_ratio=1.25, inverter_efficiency=0.96
        )
        assert ac.iloc[0] == pytest.approx(8000.0)

    def test_part_load_stays_below_input_and_nameplate(self):
        ac = dc_to_ac(self._dc(4000.0), pv_peak_power_w=10000.0, inverter_loading_ratio=1.25, inverter_efficiency=0.96)
        assert (ac < 4000.0).all()
        assert (ac <= 8000.0).all()


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

    def test_default_path_uses_local_cec_fit(self, synthetic_weather, porto_location, pv_params, monkeypatch):
        calls = 0
        original_fit = solar.fit_cec_params

        def wrapped_fit(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original_fit(*args, **kwargs)

        monkeypatch.setattr(solar, "fit_cec_params", wrapped_fit)
        solar._cec_param_cache.clear()
        try:
            dc = calculate_pv_production_dc(
                weather_data=synthetic_weather.iloc[:48],
                location=porto_location,
                tilt=35,
                surface_azimuth=180,
                n_modules=1,
                pv_params=pv_params,
                freq="h",
            )
        finally:
            solar._cec_param_cache.clear()

        assert calls == 1
        assert dc.sum() > 0

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


class TestTranspositionModel:
    def _dc(self, weather, loc, pv_params, **kw):
        return calculate_pv_production_dc(
            weather_data=weather,
            location=loc,
            tilt=35,
            surface_azimuth=180,
            n_modules=1,
            pv_params=pv_params,
            freq="h",
            **kw,
        )

    def test_default_matches_explicit_isotropic(self, synthetic_weather, porto_location, pv_params):
        # The default must reproduce the prior isotropic result bit-for-bit.
        default = self._dc(synthetic_weather, porto_location, pv_params)
        isotropic = self._dc(synthetic_weather, porto_location, pv_params, transposition_model="isotropic")
        pd.testing.assert_series_equal(default, isotropic)

    def test_all_models_run_and_are_non_negative(self, synthetic_weather, porto_location, pv_params):
        for model in TRANSPOSITION_MODELS:
            dc = self._dc(synthetic_weather, porto_location, pv_params, transposition_model=model)
            assert (dc >= -0.01).all(), model
            assert dc.sum() > 0, model

    def test_anisotropic_differs_from_isotropic(self, synthetic_weather, porto_location, pv_params):
        # Anisotropic models raise clear-day POA, so annual energy should exceed isotropic.
        isotropic = self._dc(synthetic_weather, porto_location, pv_params, transposition_model="isotropic").sum()
        perez = self._dc(synthetic_weather, porto_location, pv_params, transposition_model="perez").sum()
        assert perez > isotropic

    def test_case_insensitive(self, synthetic_weather, porto_location, pv_params):
        lower = self._dc(synthetic_weather, porto_location, pv_params, transposition_model="haydavies")
        upper = self._dc(synthetic_weather, porto_location, pv_params, transposition_model="HayDavies")
        pd.testing.assert_series_equal(lower, upper)

    def test_invalid_model_raises(self, synthetic_weather, porto_location, pv_params):
        with pytest.raises(ValueError, match="Unknown transposition model"):
            self._dc(synthetic_weather, porto_location, pv_params, transposition_model="bogus")

    def test_per_array_override(self, synthetic_weather, porto_location):
        # A per-array transposition_model overrides the function-level default.
        arrays = [{"modules": 50, "tilt": 30, "azimuth": 180, "transposition_model": "perez"}]
        default_iso = calculate_multi_array_production(
            weather_data=synthetic_weather,
            location=porto_location,
            arrays=[{"modules": 50, "tilt": 30, "azimuth": 180}],
            freq="h",
            transposition_model="isotropic",
        )
        per_array = calculate_multi_array_production(
            weather_data=synthetic_weather,
            location=porto_location,
            arrays=arrays,
            freq="h",
            transposition_model="isotropic",
        )
        assert per_array.sum() != pytest.approx(default_iso.sum())


class TestSolarPosition:
    def _dc(self, weather, loc, pv_params, **kw):
        return calculate_pv_production_dc(
            weather_data=weather,
            location=loc,
            tilt=35,
            surface_azimuth=180,
            n_modules=1,
            pv_params=pv_params,
            freq="h",
            **kw,
        )

    def test_default_matches_explicit_interval_start(self, synthetic_weather, porto_location, pv_params):
        # The default must reproduce the prior behaviour bit-for-bit.
        default = self._dc(synthetic_weather, porto_location, pv_params)
        explicit = self._dc(synthetic_weather, porto_location, pv_params, solar_position="interval-start")
        pd.testing.assert_series_equal(default, explicit)

    def test_mid_interval_shifts_output_but_keeps_index(self, synthetic_weather, porto_location, pv_params):
        start = self._dc(synthetic_weather, porto_location, pv_params, solar_position="interval-start")
        mid = self._dc(synthetic_weather, porto_location, pv_params, solar_position="mid-interval")
        # Same label grid, different sun geometry per step.
        assert mid.index.equals(start.index)
        assert not mid.equals(start)
        # A half-hour shift redistributes energy within the day; annual totals stay close.
        assert mid.sum() == pytest.approx(start.sum(), rel=0.05)

    def test_mid_interval_moves_energy_toward_morning_for_east_array(
        self, synthetic_weather, porto_location, pv_params
    ):
        # For an east-facing array the sun evaluated half a step later has moved
        # off the panel normal by evening and onto it in the morning; the split
        # between pre- and post-noon energy must therefore change.
        def split(sp):
            dc = calculate_pv_production_dc(
                weather_data=synthetic_weather,
                location=porto_location,
                tilt=35,
                surface_azimuth=90,
                n_modules=1,
                pv_params=pv_params,
                freq="h",
                solar_position=sp,
            )
            morning = dc[dc.index.hour < 12].sum()
            return morning / dc.sum()

        assert split("mid-interval") != pytest.approx(split("interval-start"), rel=1e-3)

    def test_case_insensitive(self, synthetic_weather, porto_location, pv_params):
        lower = self._dc(synthetic_weather, porto_location, pv_params, solar_position="mid-interval")
        upper = self._dc(synthetic_weather, porto_location, pv_params, solar_position="Mid-Interval")
        pd.testing.assert_series_equal(lower, upper)

    def test_invalid_method_raises(self, synthetic_weather, porto_location, pv_params):
        with pytest.raises(ValueError, match="Unknown solar position method"):
            self._dc(synthetic_weather, porto_location, pv_params, solar_position="midpoint")

    def test_tracking_accepts_mid_interval(self, synthetic_weather, porto_location, pv_params):
        dc = calculate_pv_production_dc_tracking(
            weather_data=synthetic_weather,
            location=porto_location,
            n_modules=1,
            tracking="single_axis",
            pv_params=pv_params,
            freq="h",
            solar_position="mid-interval",
        )
        assert (dc >= -0.01).all()
        assert dc.sum() > 0


class TestGroundReflectance:
    def _dc(self, weather, loc, pv_params, **kw):
        return calculate_pv_production_dc(
            weather_data=weather,
            location=loc,
            tilt=35,
            surface_azimuth=180,
            n_modules=1,
            pv_params=pv_params,
            freq="h",
            **kw,
        )

    def test_default_albedo_unchanged(self, synthetic_weather, porto_location, pv_params):
        # Not passing albedo must reproduce pvlib's 0.25 default exactly.
        base = self._dc(synthetic_weather, porto_location, pv_params)
        explicit = self._dc(synthetic_weather, porto_location, pv_params, albedo=0.25)
        pd.testing.assert_series_equal(base, explicit)

    def test_higher_albedo_raises_yield(self, synthetic_weather, porto_location, pv_params):
        base = self._dc(synthetic_weather, porto_location, pv_params).sum()
        snowy = self._dc(synthetic_weather, porto_location, pv_params, albedo=0.65).sum()
        assert snowy > base

    def test_surface_type_matches_equivalent_albedo(self, synthetic_weather, porto_location, pv_params):
        # pvlib maps surface_type="snow" to albedo 0.65.
        by_type = self._dc(synthetic_weather, porto_location, pv_params, surface_type="snow")
        by_value = self._dc(synthetic_weather, porto_location, pv_params, albedo=0.65)
        pd.testing.assert_series_equal(by_type, by_value)

    def test_albedo_and_surface_type_conflict(self, synthetic_weather, porto_location, pv_params):
        with pytest.raises(ValueError, match="either 'albedo' or 'surface_type'"):
            self._dc(synthetic_weather, porto_location, pv_params, albedo=0.3, surface_type="snow")

    def test_invalid_surface_type(self, synthetic_weather, porto_location, pv_params):
        with pytest.raises(ValueError, match="Unknown surface_type"):
            self._dc(synthetic_weather, porto_location, pv_params, surface_type="lava")

    def test_albedo_out_of_range(self, synthetic_weather, porto_location, pv_params):
        with pytest.raises(ValueError, match="albedo must be between 0 and 1"):
            self._dc(synthetic_weather, porto_location, pv_params, albedo=1.5)

    def test_all_surface_types_resolve(self, synthetic_weather, porto_location, pv_params):
        for surface_type in SURFACE_TYPES:
            dc = self._dc(synthetic_weather, porto_location, pv_params, surface_type=surface_type)
            assert dc.sum() > 0, surface_type


class TestPerezCoefficients:
    def _perez(self, weather, loc, pv_params, model_perez):
        return calculate_pv_production_dc(
            weather_data=weather,
            location=loc,
            tilt=35,
            surface_azimuth=180,
            n_modules=1,
            pv_params=pv_params,
            freq="h",
            transposition_model="perez",
            model_perez=model_perez,
        )

    def test_coefficient_set_changes_result(self, synthetic_weather, porto_location, pv_params):
        default = self._perez(synthetic_weather, porto_location, pv_params, "allsitescomposite1990")
        france = self._perez(synthetic_weather, porto_location, pv_params, "france1988")
        assert france.sum() != pytest.approx(default.sum())

    def test_all_perez_sets_resolve(self, synthetic_weather, porto_location, pv_params):
        for model_perez in PEREZ_MODELS:
            dc = self._perez(synthetic_weather, porto_location, pv_params, model_perez)
            assert dc.sum() > 0, model_perez

    def test_invalid_perez_model(self, synthetic_weather, porto_location, pv_params):
        with pytest.raises(ValueError, match="Unknown Perez coefficient model"):
            self._perez(synthetic_weather, porto_location, pv_params, "not_a_set")
