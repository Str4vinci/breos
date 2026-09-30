"""Tests for the emissions module."""

import numpy as np
import pytest

from breos.emissions import EmissionsParams, calculate_co2_projection


def _last_year(params, pv_kwh, export_kwh, grid_shift_kwh=None):
    shift = None if grid_shift_kwh is None else np.array([grid_shift_kwh])
    return calculate_co2_projection(np.array([pv_kwh]), np.array([export_kwh]), params, shift).iloc[-1]


class TestEmissionsParams:
    def test_average_intensity_uses_average_when_marginal_is_available(self):
        params = EmissionsParams(
            average_grid_carbon_intensity_gco2_kwh=100.0,
            marginal_grid_carbon_intensity_gco2_kwh=350.0,
        )

        assert params.average_intensity_gco2_kwh == pytest.approx(100.0)
        assert params.avoided_intensity_gco2_kwh == pytest.approx(350.0)
        assert params.avoided_intensity_type == "marginal"


class TestCO2Projection:
    def test_basic(self):
        params = EmissionsParams(average_grid_carbon_intensity_gco2_kwh=100.0)
        year = _last_year(params, 10000.0, 4000.0)
        # 10000 kWh * 100 gCO2/kWh = 1_000_000 g = 1000 kg
        assert year["CO2_Avoided_Total_kg"] == pytest.approx(1000.0)
        # 6000 * 100 / 1000 = 600 kg
        assert year["CO2_Avoided_SelfConsumed_kg"] == pytest.approx(600.0)

    def test_zero_production(self):
        params = EmissionsParams(average_grid_carbon_intensity_gco2_kwh=110.52)
        year = _last_year(params, 0.0, 0.0)
        assert year["CO2_Avoided_Total_kg"] == 0.0
        assert year["CO2_Avoided_SelfConsumed_kg"] == 0.0

    def test_portugal_intensity(self, emissions_params):
        year = _last_year(emissions_params, 1000.0, 500.0)
        # 1000 * 127.91 / 1000 = 127.91 kg
        assert year["CO2_Avoided_Total_kg"] == pytest.approx(127.91)

    def test_avoided_co2_uses_marginal_intensity_when_available(self):
        params = EmissionsParams(
            average_grid_carbon_intensity_gco2_kwh=100.0,
            marginal_grid_carbon_intensity_gco2_kwh=350.0,
        )
        year = _last_year(params, 10.0, 6.0)

        assert year["CO2_Avoided_Total_kg"] == pytest.approx(3.5)
        assert year["CO2_Avoided_SelfConsumed_kg"] == pytest.approx(1.4)
        assert year["CO2_Avoided_CI_Type"] == "marginal"
        assert year["Average_Grid_CI_gCO2_kWh"] == pytest.approx(100.0)

    def test_separate_export_factor_sums_exactly(self):
        params = EmissionsParams(
            average_grid_carbon_intensity_gco2_kwh=100.0,
            export_displacement_carbon_intensity_gco2_kwh=25.0,
        )
        year = _last_year(params, 10.0, 6.0)

        assert year["CO2_Avoided_SelfConsumed_kg"] == pytest.approx(0.4)
        assert year["CO2_Avoided_Export_kg"] == pytest.approx(0.15)
        assert year["CO2_Avoided_Total_kg"] == pytest.approx(
            year["CO2_Avoided_SelfConsumed_kg"] + year["CO2_Avoided_Export_kg"]
        )

    def test_export_factor_falls_back_to_avoided_grid_factor(self):
        params = EmissionsParams(marginal_grid_carbon_intensity_gco2_kwh=300.0)
        year = _last_year(params, 10.0, 6.0)
        assert year["CO2_Avoided_Export_kg"] == pytest.approx(1.8)

    def test_grid_shift_counts_against_self_consumption(self):
        params = EmissionsParams(average_grid_carbon_intensity_gco2_kwh=100.0)
        year = _last_year(params, 10.0, 6.0, grid_shift_kwh=-1.0)
        # (10 - 6 - 1) kWh * 100 g/kWh
        assert year["CO2_Avoided_SelfConsumed_kg"] == pytest.approx(0.3)

    def test_the_duplicate_grid_intensity_column_is_gone(self):
        params = EmissionsParams(average_grid_carbon_intensity_gco2_kwh=100.0)
        projection = calculate_co2_projection(np.array([10.0]), np.array([6.0]), params)
        assert "Grid_CI_gCO2_kWh" not in projection.columns
        assert projection["CO2_Avoided_CI_gCO2_kWh"].tolist() == [100.0]
