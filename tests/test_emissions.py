"""Tests for the emissions module."""

import numpy as np
import pytest

from breos.emissions import EmissionsParams, calculate_co2_projection, calculate_co2_savings


class TestCO2Savings:
    def test_basic(self):
        params = EmissionsParams(average_grid_carbon_intensity_gco2_kwh=100.0)
        result = calculate_co2_savings(
            total_pv_kwh=10000,
            self_consumed_kwh=6000,
            emissions_params=params,
        )
        # 10000 kWh * 100 gCO2/kWh = 1_000_000 g = 1000 kg
        assert result["CO2_Avoided_Total_kg"] == pytest.approx(1000.0)
        # 6000 * 100 / 1000 = 600 kg
        assert result["CO2_Avoided_SelfConsumed_kg"] == pytest.approx(600.0)
        assert result["CO2_Avoided_Total_tCO2"] == pytest.approx(1.0)

    def test_zero_production(self):
        params = EmissionsParams(average_grid_carbon_intensity_gco2_kwh=110.52)
        result = calculate_co2_savings(0.0, 0.0, params)
        assert result["CO2_Avoided_Total_kg"] == 0.0
        assert result["CO2_Avoided_SelfConsumed_kg"] == 0.0

    def test_portugal_intensity(self, emissions_params):
        result = calculate_co2_savings(1000.0, 500.0, emissions_params)
        # 1000 * 127.91 / 1000 = 127.91 kg
        assert result["CO2_Avoided_Total_kg"] == pytest.approx(127.91)

    def test_average_intensity_uses_average_when_marginal_is_available(self):
        params = EmissionsParams(
            average_grid_carbon_intensity_gco2_kwh=100.0,
            marginal_grid_carbon_intensity_gco2_kwh=350.0,
        )

        assert params.average_intensity_gco2_kwh == pytest.approx(100.0)

    def test_avoided_co2_uses_marginal_intensity_when_available(self):
        params = EmissionsParams(
            average_grid_carbon_intensity_gco2_kwh=100.0,
            marginal_grid_carbon_intensity_gco2_kwh=350.0,
        )

        result = calculate_co2_savings(
            total_pv_kwh=10.0,
            self_consumed_kwh=4.0,
            emissions_params=params,
        )

        assert result["CO2_Avoided_Total_kg"] == pytest.approx(3.5)
        assert result["CO2_Avoided_SelfConsumed_kg"] == pytest.approx(1.4)
        assert result["CO2_Avoided_Intensity_Type"] == "marginal"
        assert result["Average_Grid_Carbon_Intensity_gCO2_kWh"] == pytest.approx(100.0)

    def test_separate_export_factor_sums_exactly(self):
        params = EmissionsParams(
            average_grid_carbon_intensity_gco2_kwh=100.0,
            export_displacement_carbon_intensity_gco2_kwh=25.0,
        )
        result = calculate_co2_savings(10.0, 4.0, params)

        assert result["CO2_Avoided_SelfConsumed_kg"] == pytest.approx(0.4)
        assert result["CO2_Avoided_Export_kg"] == pytest.approx(0.15)
        assert result["CO2_Avoided_Total_kg"] == pytest.approx(
            result["CO2_Avoided_SelfConsumed_kg"] + result["CO2_Avoided_Export_kg"]
        )

    def test_export_factor_falls_back_to_avoided_grid_factor(self):
        params = EmissionsParams(marginal_grid_carbon_intensity_gco2_kwh=300.0)
        result = calculate_co2_savings(10.0, 4.0, params)
        assert result["CO2_Avoided_Export_kg"] == pytest.approx(1.8)


class TestCO2Projection:
    def test_shape(self):
        params = EmissionsParams(average_grid_carbon_intensity_gco2_kwh=100.0)
        yearly_pv = np.array([5000, 4975, 4950])
        yearly_export = np.array([2000, 1990, 1980])
        proj = calculate_co2_projection(yearly_pv, yearly_export, params)
        assert len(proj) == 3
        assert "CO2_Avoided_Total_kg" in proj.columns
        assert "CO2_Avoided_Total_Cumulative_kg" in proj.columns

    def test_cumulative_increasing(self):
        params = EmissionsParams(average_grid_carbon_intensity_gco2_kwh=100.0)
        yearly_pv = np.array([5000, 5000, 5000, 5000])
        yearly_export = np.array([2000, 2000, 2000, 2000])
        proj = calculate_co2_projection(yearly_pv, yearly_export, params)
        cumulative = proj["CO2_Avoided_Total_Cumulative_kg"].values
        assert all(cumulative[i] <= cumulative[i + 1] for i in range(len(cumulative) - 1))
