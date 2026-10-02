"""
Emissions module for CO2 savings calculations.

This module handles:
- Grid carbon intensity parameters per country (average and marginal)
- Multi-year projections of the CO2 emissions avoided by PV production
  (self-consumed and exported)
"""

from dataclasses import dataclass
from typing import Optional

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


def calculate_co2_projection(
    yearly_pv_kwh: np.ndarray,
    yearly_export_kwh: np.ndarray,
    emissions_params: EmissionsParams,
    yearly_grid_shift_kwh: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Calculate multi-year CO2 savings projection.

    Avoided emissions use net exchange: the load the system
    covered without importing, times the grid factor, plus PV export times
    the export factor. Grid energy shifted through the battery is imported,
    so it earns nothing, and its round-trip loss counts against the system.

    Args:
        yearly_pv_kwh: Array of PV production per year (kWh)
        yearly_export_kwh: Array of grid export per year (kWh)
        emissions_params: Emissions parameters
        yearly_grid_shift_kwh: Per year, grid-origin battery AC to load minus
            the grid AC imported to charge the battery; zero or negative.
            None without grid charging.

    Returns:
        DataFrame with yearly and cumulative CO2 avoided columns.
    """
    ci = emissions_params.avoided_intensity_gco2_kwh
    export_ci = emissions_params.export_displacement_intensity_gco2_kwh
    n_years = len(yearly_pv_kwh)

    yearly_self_consumed = yearly_pv_kwh - yearly_export_kwh
    if yearly_grid_shift_kwh is not None:
        yearly_self_consumed = yearly_self_consumed + yearly_grid_shift_kwh
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
            "CO2_Avoided_CI_gCO2_kWh": ci,
            "CO2_Avoided_CI_Type": emissions_params.avoided_intensity_type,
            "Average_Grid_CI_gCO2_kWh": emissions_params.average_intensity_gco2_kwh,
            "Marginal_Grid_CI_gCO2_kWh": emissions_params.marginal_grid_carbon_intensity_gco2_kwh,
            "Export_Displacement_CI_gCO2_kWh": export_ci,
        }
    )

    return proj
