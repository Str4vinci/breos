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
