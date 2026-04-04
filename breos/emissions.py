"""
Emissions module for CO2 savings calculations.

This module handles:
- Grid carbon intensity parameters per country
- CO2 emissions avoided by PV production (total and self-consumed)
- Multi-year CO2 savings projections
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class EmissionsParams:
    """Parameters for CO2 emissions calculations."""

    grid_carbon_intensity_gco2_kwh: float  # gCO2eq/kWh of grid electricity
    source: str = ""  # Data source citation
    year: int = 2024  # Reference year for the data
    country: str = ""  # Country name


def calculate_co2_savings(
    total_pv_kwh: float,
    self_consumed_kwh: float,
    emissions_params: EmissionsParams,
) -> Dict[str, float]:
    """
    Calculate CO2 emissions avoided by PV production.

    Args:
        total_pv_kwh: Total PV production in kWh
        self_consumed_kwh: Self-consumed PV in kWh (PV_Production - Sell_To_Grid)
        emissions_params: Emissions parameters with grid carbon intensity

    Returns:
        Dict with CO2 avoided metrics in kg and tonnes.
    """
    ci = emissions_params.grid_carbon_intensity_gco2_kwh

    co2_total_kg = total_pv_kwh * ci / 1000
    co2_self_kg = self_consumed_kwh * ci / 1000

    return {
        "CO2_Avoided_Total_kg": co2_total_kg,
        "CO2_Avoided_SelfConsumed_kg": co2_self_kg,
        "CO2_Avoided_Total_tCO2": co2_total_kg / 1000,
        "CO2_Avoided_SelfConsumed_tCO2": co2_self_kg / 1000,
        "Grid_Carbon_Intensity_gCO2_kWh": ci,
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
    ci = emissions_params.grid_carbon_intensity_gco2_kwh
    n_years = len(yearly_pv_kwh)

    yearly_self_consumed = yearly_pv_kwh - yearly_export_kwh
    co2_total = yearly_pv_kwh * ci / 1000
    co2_self = yearly_self_consumed * ci / 1000

    proj = pd.DataFrame(
        {
            "Year": range(1, n_years + 1),
            "CO2_Avoided_Total_kg": co2_total,
            "CO2_Avoided_SelfConsumed_kg": co2_self,
            "CO2_Avoided_Total_Cumulative_kg": np.cumsum(co2_total),
            "CO2_Avoided_SelfConsumed_Cumulative_kg": np.cumsum(co2_self),
            "Grid_CI_gCO2_kWh": ci,
        }
    )

    return proj
