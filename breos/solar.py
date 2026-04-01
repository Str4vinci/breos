"""
Solar PV production calculations module.

This module provides functions for calculating photovoltaic power production
using pvlib, with support for both hourly and 15-minute time resolutions.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import warnings
import math

import numpy as np
import pandas as pd
import pvlib
from pvlib.location import Location
from pvlib.pvsystem import PVSystem

from breos.utils import get_hours_per_step

# Module-level cache for CEC model parameters (depends only on module specs, not weather)
_cec_param_cache: Dict[tuple, tuple] = {}


@dataclass
class PVModuleParams:
    """Parameters for a PV module."""
    Mpp: float # W (STC power)
    Vmp: float # V
    Imp: float # A
    Voc: float # V
    Isc: float # A

    T_Pmax_pct: float # %/°C
    T_Voc_pct: float # %/°C
    T_Isc_pct: float # %/°C

    N_Cells: int # Number of cells (eg 6*24 or 144)

    Name: Optional[str] = None  # Metadata: specific module model name
    Module_Efficiency: Optional[float] = None  # Metadata: module efficiency % (not used in calc)
    celltype: str = 'monoSi'

    alpha_sc_abs: Optional[float] = None  # A/°C - if provided, overrides T_Isc_pct conversion
    beta_voc_abs: Optional[float] = None  # V/°C - if provided, overrides T_Voc_pct conversion
    gamma_pmp:  Optional[float] = None

    def __post_init__(self):
        # 1. HANDLE CURRENT (alpha_sc)
        if self.alpha_sc_abs is not None:
                # User provided absolute A/C directly
            self.alpha_sc = self.alpha_sc_abs
        else:
            # Convert from %/C
            self.alpha_sc = (self.T_Isc_pct * self.Isc) / 100
        # 2. HANDLE VOLTAGE (beta_voc)
        if self.beta_voc_abs is not None:
            # User provided absolute V/C directly
            self.beta_voc = self.beta_voc_abs
        else:
                # Convert from %/C
            self.beta_voc = (self.T_Voc_pct * self.Voc) / 100
        
        # 3. HANDLE POWER (gamma_pmp)
        # Power is almost always used as %/C in pvlib models, passed as unitless decimal or %
        self.gamma_pmp = self.T_Pmax_pct

def calculate_pv_production_dc(
    weather_data: pd.DataFrame,
    location: Location,
    slope: float,
    surface_azimuth: float,
    n_modules: int,
    pv_params: Optional[PVModuleParams] = None,
    freq: str = 'h',
    degradation_rate: float = 0.0,
    current_year: Optional[int] = None,
    start_year: Optional[int] = None,
    verbose: bool = False
) -> pd.Series:
    """
    Calculate PV DC production from weather data.
    
    Returns DC power BEFORE inverter conversion. Use dc_to_ac() to convert
    to AC power for grid export or AC loads.
    
    For DC-coupled battery systems:
    - Use DC output directly for battery charging (apply only charge efficiency)
    - Use dc_to_ac() for power going to AC loads or grid
    
    Args:
        weather_data: DataFrame with weather variables (must include ghi/dni/dhi or shortwave_radiation)
        location: pvlib Location object
        slope: Panel tilt angle (degrees)
        surface_azimuth: Panel azimuth (degrees, 180=South)
        n_modules: Number of PV modules
        pv_params: PV module parameters (uses defaults if None)
        freq: Time frequency ('h' or '15min')
        degradation_rate: Annual degradation rate (0.005 = 0.5%/year)
        current_year: Current simulation year (for age-based degradation)
        start_year: Year system was installed (for age calculation)
        verbose: Whether to print production summary
        
    Returns:
        pd.Series with DC power production in Watts (before inverter)
    """
    if pv_params is None:
        from breos.pv_modules import get_module
        pv_params = get_module("Generic_400W")

    # Determine time range from weather data
    if isinstance(weather_data.index, pd.DatetimeIndex):
        start_h = weather_data.index[0]
        end_h = weather_data.index[-1]
    else:
        raise ValueError("weather_data must have a DatetimeIndex")
    
    # Get solar position
    times = pd.date_range(start=start_h, end=end_h, freq=freq)
    solarpos = location.get_solarposition(times=times)
    
    # Align weather data with times
    weather_aligned = weather_data.reindex(times, method='nearest')
    
    # Calculate angle of incidence and IAM
    aoi = pvlib.irradiance.aoi(slope, surface_azimuth, solarpos.apparent_zenith, solarpos.azimuth)
    iam = pvlib.iam.ashrae(aoi)
    
    # Extract irradiance components (handle different column naming conventions)
    dni, ghi, dhi = _extract_irradiance(weather_aligned)
    temp_air, wind_speed = _extract_met_data(weather_aligned)
    
    # Create PV system
    pv_system = PVSystem(
        surface_tilt=slope,
        surface_azimuth=surface_azimuth,
        modules_per_string=n_modules,
        strings_per_inverter=1
    )
    
    # Calculate POA irradiance
    poa_irradiance = pv_system.get_irradiance(
        solarpos.zenith,
        solarpos.azimuth,
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        model='isotropic'
    )
    
    # Effective irradiance (with IAM)
    effective_irradiance = poa_irradiance['poa_direct'].values * iam + poa_irradiance['poa_diffuse'].values
    
    # Cell temperature
    temp_cell = pvlib.temperature.faiman(
        poa_irradiance['poa_global'].values,
        temp_air,
        wind_speed
    )
    
    # Calculate CEC model parameters (cached — only depends on module specs)
    _cache_key = (
        pv_params.celltype, pv_params.Vmp, pv_params.Imp,
        pv_params.Voc, pv_params.Isc, pv_params.alpha_sc,
        pv_params.beta_voc, pv_params.gamma_pmp, pv_params.N_Cells,
    )
    if _cache_key in _cec_param_cache:
        I_L_ref, I_o_ref, R_s, R_sh_ref, a_ref, Adjust = _cec_param_cache[_cache_key]
    else:
        I_L_ref, I_o_ref, R_s, R_sh_ref, a_ref, Adjust = pvlib.ivtools.sdm.fit_cec_sam(
            celltype=pv_params.celltype,
            v_mp=pv_params.Vmp,
            i_mp=pv_params.Imp,
            v_oc=pv_params.Voc,
            i_sc=pv_params.Isc,
            alpha_sc=pv_params.alpha_sc,
            beta_voc=pv_params.beta_voc,
            gamma_pmp=pv_params.gamma_pmp,
            cells_in_series=pv_params.N_Cells,
        )
        _cec_param_cache[_cache_key] = (I_L_ref, I_o_ref, R_s, R_sh_ref, a_ref, Adjust)

    # Suppress RuntimeWarnings from pvlib during low/zero irradiance conditions (harmless)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # Calculate module parameters at operating conditions
        cec_params = pvlib.pvsystem.calcparams_cec(
            effective_irradiance,
            temp_cell,
            pv_params.alpha_sc,
            a_ref,
            I_L_ref,
            I_o_ref,
            R_sh_ref,
            R_s,
            Adjust
        )
        
        # Maximum power point
        mpp = pvlib.pvsystem.max_power_point(*cec_params, method='newton')
    
    # Calculate losses
    if current_year is not None and start_year is not None:
        years_operating = current_year - start_year + 0.5
        age_degradation_factor = 100 * (1 - (1 - degradation_rate) ** years_operating)
    else:
        age_degradation_factor = 0.0
    
    total_losses_percent = pvlib.pvsystem.pvwatts_losses(
        soiling=2,
        shading=3,
        snow=0,
        mismatch=2,
        wiring=2,
        connections=0.5,
        lid=1.5,
        nameplate_rating=1,
        age=age_degradation_factor,
        availability=3
    )
    
    # Scale power by number of modules
    dc_scaled = pv_system.scale_voltage_current_power(mpp)
    dc_power = dc_scaled.p_mp * (1 - total_losses_percent / 100)
    
    # Create output series with datetime index
    dc_power = pd.Series(dc_power, index=times, name='dc_power_W')
    
    if verbose:
        hours_per_step = get_hours_per_step(freq)
        total_kwh = dc_power.sum() * hours_per_step / 1000
        print(f"Total PV DC production for slope {slope} deg: {total_kwh:.1f} kWh")

    return dc_power


def dc_to_ac(
    dc_power: pd.Series,
    pv_peak_power_w: float,
    inverter_loading_ratio: float = 1.25,
    inverter_efficiency: float = 0.96
) -> pd.Series:
    """
    Convert DC power to AC power through inverter.

    Applies inverter efficiency and clipping based on inverter size.
    Use this for:
    - Calculating actual AC production for plots/reports
    - Power going directly to AC loads (no battery)
    - Grid export

    Args:
        dc_power: DC power in Watts (from calculate_pv_production)
        pv_peak_power_w: Total PV system peak power in Watts (n_modules * Mpp)
        inverter_loading_ratio: DC/AC ratio for inverter sizing (default 1.25)
        inverter_efficiency: Nominal inverter efficiency (default 0.96)
        
    Returns:
        pd.Series with AC power in Watts
    """
    inv_size = pv_peak_power_w / inverter_loading_ratio
    
    ac_power = pvlib.inverter.pvwatts(
        pdc=dc_power,
        pdc0=inv_size,
        eta_inv_nom=inverter_efficiency,
        eta_inv_ref=0.9637
    )
    
    return pd.Series(ac_power, index=dc_power.index, name='ac_power_W')


def calculate_pv_production_tmy(
    tmy_data: pd.DataFrame,
    location: Location,
    slope: float,
    surface_azimuth: float,
    n_modules: int,
    pv_params: Optional[PVModuleParams] = None,
    freq: str = 'h',
    verbose: bool = True
) -> pd.Series:
    """
    Calculate PV DC production from TMY data.
    
    This is a convenience wrapper around calculate_pv_production_dc for TMY data.
    
    Args:
        tmy_data: DataFrame with TMY weather data (ghi, dni, dhi, temp_air, wind_speed)
        location: pvlib Location object
        slope: Panel tilt angle (degrees)
        surface_azimuth: Panel azimuth (degrees, 180=South)
        n_modules: Number of PV modules
        pv_params: PV module parameters
        freq: Time frequency ('h' or '15min')
        verbose: Whether to print production summary
        
    Returns:
        pd.Series with DC power production in Watts
    """
    return calculate_pv_production_dc(
        weather_data=tmy_data,
        location=location,
        slope=slope,
        surface_azimuth=surface_azimuth,
        n_modules=n_modules,
        pv_params=pv_params,
        freq=freq,
        degradation_rate=0.0,  # TMY doesn't include degradation
        verbose=verbose
    )


def calculate_pv_production_ac(
    weather_data: pd.DataFrame,
    location: Location,
    slope: float,
    surface_azimuth: float,
    n_modules: int,
    pv_params: Optional[PVModuleParams] = None,
    freq: str = 'h',
    degradation_rate: float = 0.0,
    current_year: Optional[int] = None,
    start_year: Optional[int] = None,
    inverter_loading_ratio: float = 1.25,
    inverter_efficiency: float = 0.96,
    verbose: bool = False
) -> pd.Series:
    """
    Calculate PV AC production from weather data.

    Calculates DC production then converts to AC through inverter.
    Use this for display/reporting purposes or AC-coupled systems.

    Args:
        weather_data: DataFrame with weather variables
        location: pvlib Location object
        slope: Panel tilt angle (degrees)
        surface_azimuth: Panel azimuth (degrees, 180=South)
        n_modules: Number of PV modules
        pv_params: PV module parameters (uses defaults if None)
        freq: Time frequency ('h' or '15min')
        degradation_rate: Annual degradation rate (0.005 = 0.5%/year)
        current_year: Current simulation year (for age-based degradation)
        start_year: Year system was installed (for age calculation)
        inverter_loading_ratio: DC/AC ratio for inverter sizing
        inverter_efficiency: Nominal inverter efficiency
        verbose: Whether to print production summary
        
    Returns:
        pd.Series with AC power production in Watts
    """
    if pv_params is None:
        from breos.pv_modules import get_module
        pv_params = get_module("Generic_400W")

    dc_power = calculate_pv_production_dc(
        weather_data=weather_data,
        location=location,
        slope=slope,
        surface_azimuth=surface_azimuth,
        n_modules=n_modules,
        pv_params=pv_params,
        freq=freq,
        degradation_rate=degradation_rate,
        current_year=current_year,
        start_year=start_year,
        verbose=False
    )
    
    pv_peak_power_w = n_modules * pv_params.Mpp
    ac_power = dc_to_ac(dc_power, pv_peak_power_w, inverter_loading_ratio, inverter_efficiency)
    
    if verbose:
        hours_per_step = get_hours_per_step(freq)
        total_kwh = ac_power.sum() * hours_per_step / 1000
        print(f"Total PV AC production for slope {slope} deg: {total_kwh:.1f} kWh")

    return ac_power


def _extract_irradiance(weather_df: pd.DataFrame):
    """Extract DNI, GHI, DHI from weather DataFrame with flexible column names."""
    # Try different column naming conventions
    dni_cols = ['dni', 'DNI', 'direct_normal_irradiance']
    ghi_cols = ['ghi', 'GHI', 'shortwave_radiation', 'global_horizontal_irradiance']
    dhi_cols = ['dhi', 'DHI', 'diffuse_radiation', 'diffuse_horizontal_irradiance']
    
    dni = _get_column(weather_df, dni_cols)
    ghi = _get_column(weather_df, ghi_cols)
    dhi = _get_column(weather_df, dhi_cols)
    
    return dni, ghi, dhi


def _extract_met_data(weather_df: pd.DataFrame):
    """Extract temperature and wind speed from weather DataFrame."""
    temp_cols = ['temp_air', 'temperature_2m', 'temp', 'air_temperature']
    wind_cols = ['wind_speed', 'wind_speed_10m', 'ws', 'WS10m']
    
    temp_air = _get_column(weather_df, temp_cols, default=25.0)
    wind_speed = _get_column(weather_df, wind_cols, default=1.0)
    
    return temp_air, wind_speed


def _get_column(df: pd.DataFrame, possible_names: list, default=None):
    """Get column from DataFrame trying multiple possible names."""
    for name in possible_names:
        if name in df.columns:
            return df[name].values
    
    if default is not None:
        return np.full(len(df), default)
    
    raise KeyError(f"Could not find column. Tried: {possible_names}")


def estimate_optimal_tilt(latitude: float) -> float:
    """
    Estimate optimal fixed tilt angle based on latitude.
    
    Simple rule of thumb: tilt ≈ latitude * 0.76 + 3.1
    Taken from https://solarpaneltilt.com and
    https://iea-pvps.org/wp-content/uploads/2020/01/Photovoltaic_Module_Energy_Yield_Measurements_Existing_Approaches_and_Best_Practice_by_Task_13.pdf
    
    Args:
        latitude: Site latitude in degrees
        
    Returns:
        Estimated optimal tilt angle in degrees
    """

    lat = abs(latitude)

    if lat < 25:
        return lat * 0.87
    elif lat <= 50:
        return lat * 0.76 + 3.1
    else:
        return 45.0


def default_azimuth(latitude: float) -> float:
    """
    Return the optimal default azimuth based on hemisphere.
    
    In the northern hemisphere, panels should face South (180°).
    In the southern hemisphere, panels should face North (0°).
    
    Args:
        latitude: Site latitude in degrees (negative = southern hemisphere)
        
    Returns:
        Default azimuth angle in degrees (180.0 or 0.0)
    """
    return 180.0 if latitude >= 0 else 0.0


def zeb_sizer(houseload: pd.DataFrame, ac_loss: pd.Series, current_n_modules: int, freq: str = 'h') -> Dict[str, float]:
    """
    Size a Zero Energy Building (ZEB) PV system.

    Args:
        houseload: DataFrame with electrical consumption in Watts
        ac_loss: Series with PV production in Watts
        current_n_modules: Current number of PV modules
        freq: Time frequency ('h' or '15min')

    Returns:
        Dict with sizing results including:
            - yearly_pv_production_Wh: Current annual PV production
            - yearly_consumption_Wh: Annual consumption
            - pv_to_load_ratio: Current PV-to-load ratio
            - is_zeb: Whether current system achieves ZEB
            - panels_needed_for_zeb: Number of panels needed for ratio=1.0
    """
    hours_per_step = get_hours_per_step(freq)
    yearly_pv_production = ac_loss.sum() * hours_per_step
    total_yearly_consumption = houseload.iloc[:, 0].sum() * hours_per_step

    ratio = yearly_pv_production / total_yearly_consumption

    # Calculate panels needed for ZEB (ratio = 1.0)
    if ratio >= 1.0:
        panels_needed_for_zeb = current_n_modules
    else:
        # Need to scale up: panels_needed = current_panels / ratio
        panels_needed_for_zeb = math.ceil(current_n_modules / ratio)

    return {
        'yearly_pv_production_Wh': yearly_pv_production,
        'yearly_consumption_Wh': total_yearly_consumption,
        'pv_to_load_ratio': ratio,
        'is_zeb': ratio >= 1.0,
        'panels_needed_for_zeb': panels_needed_for_zeb
    }

def calculate_multi_array_production(
    weather_data: pd.DataFrame,
    location: Location,
    arrays: List[Dict[str, Any]],
    freq: str = 'h',
    degradation_rate: float = 0.0,
    current_year: Optional[int] = None,
    start_year: Optional[int] = None,
    verbose: bool = False
) -> pd.Series:
    """
    Calculate combined DC production from multiple PV arrays.
    
    Args:
        weather_data: DataFrame with weather variables
        location: pvlib Location object
        arrays: List of dicts, each containing:
            - 'modules': Number of modules (int)
            - 'slope': Tilt angle (float)
            - 'azimuth': Azimuth angle (float)
            - 'module': Module model name (str) [Optional]
        freq: Time frequency ('h' or '15min')
        degradation_rate: Annual degradation rate
        current_year: Current simulation year
        start_year: Installation year
        verbose: Print summary
        
    Returns:
        pd.Series with total DC power (watts)
    """
    # Import locally to avoid circular dependencies (if solar imported by pv_modules)
    try:
        from breos.pv_modules import get_module
    except ImportError:
        raise ImportError("breos.pv_modules is required for multi-array production")

    total_dc = None
    
    for i, arr in enumerate(arrays):
        n_mod = arr.get('modules', 0)
        if n_mod <= 0:
            continue
            
        slope = arr.get('slope', 35)
        azimuth = arr.get('azimuth', default_azimuth(location.latitude))
        mod_name = arr.get('module', 'Generic_400W')
        
        pv_params = get_module(mod_name)
        
        if verbose:
            print(f"   Array {i+1}: {n_mod}x {mod_name}, Slope={slope}, Azimuth={azimuth}")
            
        dc = calculate_pv_production_dc(
            weather_data=weather_data,
            location=location,
            slope=slope,
            surface_azimuth=azimuth,
            n_modules=n_mod,
            pv_params=pv_params,
            freq=freq,
            degradation_rate=degradation_rate,
            current_year=current_year,
            start_year=start_year,
            verbose=False
        )
        
        if total_dc is None:
            total_dc = dc.fillna(0)
        else:
            total_dc = total_dc + dc.fillna(0)
            
    if total_dc is None:
        # Return zeros if no valid arrays
        times = weather_data.index
        return pd.Series(0.0, index=weather_data.index)
        
    if verbose:
        hours_per_step = get_hours_per_step(freq)
        total_kwh = total_dc.sum() * hours_per_step / 1000
        print(f"   Total Multi-Array Production: {total_kwh:,.1f} kWh")
        
    return total_dc
