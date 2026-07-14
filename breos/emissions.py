"""
Emissions module for CO2 savings calculations.

This module handles:
- Grid carbon intensity parameters per country (average and marginal)
- CO2 emissions avoided by PV production (total and self-consumed)
- Multi-year CO2 savings projections
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class EmissionsParams:
    """Parameters for CO2 emissions calculations.

    Either average or marginal grid carbon intensity must be provided. When
    a marginal intensity is given, it is used for avoided-emissions accounting
    (the more accurate signal for grid CO2 displacement); otherwise the
    average intensity is used.
    """

    average_grid_carbon_intensity_gco2_kwh: Optional[float] = None
    marginal_grid_carbon_intensity_gco2_kwh: Optional[float] = None
    export_displacement_carbon_intensity_gco2_kwh: Optional[float] = None
    source: str = ""  # Data source citation for average intensity
    marginal_source: str = ""  # Data source citation for marginal intensity
    year: int = 2024  # Reference year for the data
    country: str = ""  # Country name

    @property
    def average_intensity_gco2_kwh(self) -> float:
        if self.average_grid_carbon_intensity_gco2_kwh is not None:
            return float(self.average_grid_carbon_intensity_gco2_kwh)
        if self.marginal_grid_carbon_intensity_gco2_kwh is not None:
            return float(self.marginal_grid_carbon_intensity_gco2_kwh)
        raise ValueError("EmissionsParams requires an average or marginal grid carbon intensity")

    @property
    def avoided_intensity_gco2_kwh(self) -> float:
        if self.marginal_grid_carbon_intensity_gco2_kwh is not None:
            return float(self.marginal_grid_carbon_intensity_gco2_kwh)
        return self.average_intensity_gco2_kwh

    @property
    def avoided_intensity_type(self) -> str:
        return "marginal" if self.marginal_grid_carbon_intensity_gco2_kwh is not None else "average"

    @property
    def export_displacement_intensity_gco2_kwh(self) -> float:
        """Intensity displaced by exports, falling back to avoided grid CI."""
        if self.export_displacement_carbon_intensity_gco2_kwh is not None:
            return float(self.export_displacement_carbon_intensity_gco2_kwh)
        return self.avoided_intensity_gco2_kwh


def calculate_co2_savings(
    total_pv_kwh: float,
    self_consumed_kwh: float,
    emissions_params: EmissionsParams,
) -> Dict[str, float]:
    """
    Calculate CO2 emissions avoided by PV production.

    Args:
        total_pv_kwh: Total PV production in kWh
        self_consumed_kwh: Explicit PV-origin AC delivered to load, including
            direct PV and later PV-origin battery discharge.
        emissions_params: Emissions parameters with grid carbon intensity

    Returns:
        Dict with CO2 avoided metrics in kg and tonnes.
    """
    ci = emissions_params.avoided_intensity_gco2_kwh
    export_ci = emissions_params.export_displacement_intensity_gco2_kwh
    exported_kwh = max(0.0, total_pv_kwh - self_consumed_kwh)
    co2_self_kg = self_consumed_kwh * ci / 1000
    co2_export_kg = exported_kwh * export_ci / 1000
    co2_total_kg = co2_self_kg + co2_export_kg

    return {
        "CO2_Avoided_Total_kg": co2_total_kg,
        "CO2_Avoided_SelfConsumed_kg": co2_self_kg,
        "CO2_Avoided_Export_kg": co2_export_kg,
        "CO2_Avoided_Total_tCO2": co2_total_kg / 1000,
        "CO2_Avoided_SelfConsumed_tCO2": co2_self_kg / 1000,
        "CO2_Avoided_Export_tCO2": co2_export_kg / 1000,
        "Grid_Carbon_Intensity_gCO2_kWh": ci,
        "CO2_Avoided_Intensity_gCO2_kWh": ci,
        "CO2_Avoided_Intensity_Type": emissions_params.avoided_intensity_type,
        "Average_Grid_Carbon_Intensity_gCO2_kWh": emissions_params.average_intensity_gco2_kwh,
        "Marginal_Grid_Carbon_Intensity_gCO2_kWh": emissions_params.marginal_grid_carbon_intensity_gco2_kwh,
        "Export_Displacement_Carbon_Intensity_gCO2_kWh": export_ci,
    }


def calculate_co2_projection(
    yearly_pv_kwh: np.ndarray,
    yearly_export_kwh: np.ndarray,
    emissions_params: EmissionsParams,
) -> pd.DataFrame:
    """
    Calculate multi-year CO2 savings projection.

    Args:
        yearly_pv_kwh: Array of PV production per year (kWh)
        yearly_export_kwh: Array of grid export per year (kWh)
        emissions_params: Emissions parameters

    Returns:
        DataFrame with yearly and cumulative CO2 avoided columns.
    """
    ci = emissions_params.avoided_intensity_gco2_kwh
    export_ci = emissions_params.export_displacement_intensity_gco2_kwh
    n_years = len(yearly_pv_kwh)

    yearly_self_consumed = yearly_pv_kwh - yearly_export_kwh
    co2_self = yearly_self_consumed * ci / 1000
    co2_export = yearly_export_kwh * export_ci / 1000
    co2_total = co2_self + co2_export

    proj = pd.DataFrame(
        {
            "Year": range(1, n_years + 1),
            "CO2_Avoided_Total_kg": co2_total,
            "CO2_Avoided_SelfConsumed_kg": co2_self,
            "CO2_Avoided_Export_kg": co2_export,
            "CO2_Avoided_Total_Cumulative_kg": np.cumsum(co2_total),
            "CO2_Avoided_SelfConsumed_Cumulative_kg": np.cumsum(co2_self),
            "CO2_Avoided_Export_Cumulative_kg": np.cumsum(co2_export),
            "Grid_CI_gCO2_kWh": ci,
            "CO2_Avoided_CI_gCO2_kWh": ci,
            "CO2_Avoided_CI_Type": emissions_params.avoided_intensity_type,
            "Average_Grid_CI_gCO2_kWh": emissions_params.average_intensity_gco2_kwh,
            "Marginal_Grid_CI_gCO2_kWh": emissions_params.marginal_grid_carbon_intensity_gco2_kwh,
            "Export_Displacement_CI_gCO2_kWh": export_ci,
        }
    )

    return proj
