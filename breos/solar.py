"""
Solar PV production calculations module.

This module provides functions for calculating photovoltaic power production
using pvlib, with support for both hourly and 15-minute time resolutions.
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pvlib
from pvlib.location import Location

from breos.cec_fit import fit_cec_params
from breos.utils import get_hours_per_step

# Module-level cache for CEC model parameters (depends only on module specs, not weather)
_cec_param_cache: Dict[tuple, tuple] = {}

# System loss components (percent) applied to every DC production calculation
# via pvlib's pvwatts_losses. Combined multiplicatively they total ~14.1%
# (age-based degradation is added separately per simulation year).
DEFAULT_PVWATTS_LOSSES: Dict[str, float] = {
    "soiling": 2.0,
    "shading": 3.0,
    "snow": 0.0,
    "mismatch": 2.0,
    "wiring": 2.0,
    "connections": 0.5,
    "lid": 1.5,
    "nameplate_rating": 1.0,
    "availability": 3.0,
}


@dataclass
class PVModuleParams:
    """Parameters for a PV module."""

    Mpp: float  # W (STC power)
    Vmp: float  # V
    Imp: float  # A
    Voc: float  # V
    Isc: float  # A

    T_Pmax_pct: float  # %/°C
    T_Voc_pct: float  # %/°C
    T_Isc_pct: float  # %/°C

    N_Cells: int  # Number of cells (eg 6*24 or 144)

    Name: Optional[str] = None  # Metadata: specific module model name
    Module_Efficiency: Optional[float] = None  # Metadata: module efficiency fraction, e.g. 0.213 (not used in calc)
    celltype: str = "monoSi"

    alpha_sc_abs: Optional[float] = None  # A/°C - if provided, overrides T_Isc_pct conversion
    beta_voc_abs: Optional[float] = None  # V/°C - if provided, overrides T_Voc_pct conversion
    gamma_pmp: Optional[float] = None

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


def _prepare_solarpos_and_weather(weather_data: pd.DataFrame, location: Location, freq: str):
    """Build time index, solar position, and aligned weather frame."""
    if not isinstance(weather_data.index, pd.DatetimeIndex):
        raise ValueError("weather_data must have a DatetimeIndex")

    times = pd.date_range(start=weather_data.index[0], end=weather_data.index[-1], freq=freq)
    solarpos = location.get_solarposition(times=times)
    weather_aligned = weather_data.reindex(times, method="nearest")
    return times, solarpos, weather_aligned


def _compute_effective_irradiance_and_cell_temp(
    weather_aligned: pd.DataFrame,
    solarpos: pd.DataFrame,
    surface_tilt,
    surface_azimuth,
):
    """Compute effective POA irradiance (with IAM) and cell temperature.

    surface_tilt / surface_azimuth may be scalars (fixed) or per-timestep arrays/Series
    (tracking). Uses pvlib.irradiance.get_total_irradiance which is array-aware.
    """
    dni, ghi, dhi = _extract_irradiance(weather_aligned)
    temp_air, wind_speed = _extract_met_data(weather_aligned)

    aoi = pvlib.irradiance.aoi(surface_tilt, surface_azimuth, solarpos.apparent_zenith, solarpos.azimuth)
    iam = pvlib.iam.ashrae(aoi)

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        solar_zenith=solarpos.zenith,
        solar_azimuth=solarpos.azimuth,
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        model="isotropic",
    )

    poa_direct = np.nan_to_num(poa["poa_direct"].values, nan=0.0)
    poa_diffuse = np.nan_to_num(poa["poa_diffuse"].values, nan=0.0)
    poa_global = np.nan_to_num(poa["poa_global"].values, nan=0.0)
    iam_clean = np.nan_to_num(np.asarray(iam, dtype=float), nan=0.0)

    effective_irradiance = poa_direct * iam_clean + poa_diffuse
    temp_cell = pvlib.temperature.faiman(poa_global, temp_air, wind_speed)
    return effective_irradiance, temp_cell


def _get_cec_params(pv_params: "PVModuleParams"):
    """Fetch (and cache) CEC single-diode model params for a module."""
    key = (
        pv_params.celltype,
        pv_params.Vmp,
        pv_params.Imp,
        pv_params.Voc,
        pv_params.Isc,
        pv_params.alpha_sc,
        pv_params.beta_voc,
        pv_params.gamma_pmp,
        pv_params.N_Cells,
    )
    if key in _cec_param_cache:
        return _cec_param_cache[key]

    cec = fit_cec_params(
        celltype=pv_params.celltype,
        Vmp=pv_params.Vmp,
        Imp=pv_params.Imp,
        Voc=pv_params.Voc,
        Isc=pv_params.Isc,
        alpha_sc=pv_params.alpha_sc,
        beta_voc=pv_params.beta_voc,
        gamma_pmp=pv_params.gamma_pmp,
        cells_in_series=pv_params.N_Cells,
    )
    _cec_param_cache[key] = cec
    return cec


def _dc_from_poa(
    effective_irradiance: np.ndarray,
    temp_cell: np.ndarray,
    pv_params: "PVModuleParams",
    n_modules: int,
    times: pd.DatetimeIndex,
    degradation_rate: float = 0.0,
    current_year: Optional[int] = None,
    start_year: Optional[int] = None,
    loss_overrides: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """Run CEC single-diode + pvwatts loss model and scale to array.

    Shared between fixed-tilt and tracking DC paths. System losses default
    to DEFAULT_PVWATTS_LOSSES; ``loss_overrides`` replaces individual
    components (percent).
    """
    loss_components = dict(DEFAULT_PVWATTS_LOSSES)
    if loss_overrides:
        unknown = set(loss_overrides) - set(DEFAULT_PVWATTS_LOSSES)
        if unknown:
            valid = ", ".join(sorted(DEFAULT_PVWATTS_LOSSES))
            raise ValueError(f"Unknown loss component(s) {sorted(unknown)}. Valid components: {valid}")
        loss_components.update(loss_overrides)

    I_L_ref, I_o_ref, R_s, R_sh_ref, a_ref, Adjust = _get_cec_params(pv_params)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        cec = pvlib.pvsystem.calcparams_cec(
            effective_irradiance, temp_cell, pv_params.alpha_sc, a_ref, I_L_ref, I_o_ref, R_sh_ref, R_s, Adjust
        )
        mpp = pvlib.pvsystem.max_power_point(*cec, method="newton")

    if current_year is not None and start_year is not None:
        years_operating = current_year - start_year + 0.5
        age_degradation_factor = 100 * (1 - (1 - degradation_rate) ** years_operating)
    else:
        age_degradation_factor = 0.0

    total_losses_percent = pvlib.pvsystem.pvwatts_losses(
        age=age_degradation_factor,
        **loss_components,
    )

    p_mp = mpp["p_mp"] if isinstance(mpp, dict) else mpp.p_mp
    dc_power = np.asarray(p_mp) * n_modules * (1 - total_losses_percent / 100)
    return pd.Series(dc_power, index=times, name="dc_power_W")


def calculate_pv_production_dc(
    weather_data: pd.DataFrame,
    location: Location,
    tilt: float,
    surface_azimuth: float,
    n_modules: int,
    pv_params: Optional[PVModuleParams] = None,
    freq: str = "h",
    degradation_rate: float = 0.0,
    current_year: Optional[int] = None,
    start_year: Optional[int] = None,
    verbose: bool = False,
    loss_overrides: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """
    Calculate PV DC production from weather data (fixed-tilt array).

    Returns DC power BEFORE inverter conversion. Use dc_to_ac() to convert
    to AC power for grid export or AC loads.

    For DC-coupled battery systems:
    - Use DC output directly for battery charging (apply only charge efficiency)
    - Use dc_to_ac() for power going to AC loads or grid

    System losses: DEFAULT_PVWATTS_LOSSES (~14.1% combined: soiling 2,
    shading 3, mismatch 2, wiring 2, connections 0.5, LID 1.5, nameplate 1,
    availability 3) are always applied via pvlib's pvwatts_losses; inverter
    conversion is NOT included here. Pass ``loss_overrides`` to change
    individual components (percent), e.g. ``{"shading": 0.0}``.

    Args:
        weather_data: DataFrame with weather variables (must include ghi/dni/dhi or shortwave_radiation)
        location: pvlib Location object
        tilt: Panel tilt angle (degrees)
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

    times, solarpos, weather_aligned = _prepare_solarpos_and_weather(weather_data, location, freq)
    effective_irradiance, temp_cell = _compute_effective_irradiance_and_cell_temp(
        weather_aligned, solarpos, surface_tilt=tilt, surface_azimuth=surface_azimuth
    )
    dc_power = _dc_from_poa(
        effective_irradiance,
        temp_cell,
        pv_params,
        n_modules,
        times,
        degradation_rate=degradation_rate,
        current_year=current_year,
        start_year=start_year,
        loss_overrides=loss_overrides,
    )

    if verbose:
        total_kwh = dc_power.sum() * get_hours_per_step(freq) / 1000
        print(f"Total PV DC production for tilt {tilt} deg: {total_kwh:.1f} kWh")

    return dc_power


def calculate_pv_production_dc_tracking(
    weather_data: pd.DataFrame,
    location: Location,
    n_modules: int,
    tracking: str = "single_axis",
    axis_tilt: float = 0.0,
    axis_azimuth: float = 180.0,
    max_angle: float = 60.0,
    backtrack: bool = True,
    gcr: float = 0.35,
    cross_axis_tilt: float = 0.0,
    dual_axis_max_tilt: float = 90.0,
    pv_params: Optional[PVModuleParams] = None,
    freq: str = "h",
    degradation_rate: float = 0.0,
    current_year: Optional[int] = None,
    start_year: Optional[int] = None,
    verbose: bool = False,
    loss_overrides: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """
    Calculate PV DC production for a tracking array (single- or dual-axis).

    Single-axis (horizontal or tilted) trackers are the dominant configuration in
    utility-scale PV. Dual-axis trackers gain slightly more energy but at higher
    cost; they are common in CPV and high-latitude installations.

    Args:
        weather_data: DataFrame with weather variables.
        location: pvlib Location object.
        n_modules: Number of PV modules.
        tracking: ``"single_axis"`` or ``"dual_axis"``.
        axis_tilt: Tilt of the rotation axis (degrees). Single-axis only.
            ``0`` = horizontal (HSAT); typical utility installations are 0.
        axis_azimuth: Compass direction of the rotation axis (degrees).
            ``180`` = N-S axis (panels rotate east→west across the day). Single-axis only.
        max_angle: Maximum rotation from horizontal (degrees, ±). Typical ±60°.
        backtrack: Whether the tracker backtracks to avoid row-to-row shading at low sun.
        gcr: Ground coverage ratio (array area / land area). Typical 0.3–0.4 for utility.
        cross_axis_tilt: Tilt of the axis perpendicular to the rotation axis (terrain slope).
        dual_axis_max_tilt: Maximum panel tilt for dual-axis (degrees). ``90`` = unlimited.
        pv_params: PV module parameters (uses defaults if None).
        freq: Time frequency (``"h"`` or ``"15min"``).
        degradation_rate: Annual degradation rate.
        current_year: Current simulation year (for age-based degradation).
        start_year: Year system was installed.
        verbose: Whether to print production summary.

    Returns:
        pd.Series with DC power production in Watts (before inverter).
    """
    if tracking not in ("single_axis", "dual_axis"):
        raise ValueError(f"tracking must be 'single_axis' or 'dual_axis', got {tracking!r}")

    if pv_params is None:
        from breos.pv_modules import get_module

        pv_params = get_module("Generic_400W")

    times, solarpos, weather_aligned = _prepare_solarpos_and_weather(weather_data, location, freq)

    if tracking == "single_axis":
        tracker = pvlib.tracking.singleaxis(
            apparent_zenith=solarpos.apparent_zenith,
            solar_azimuth=solarpos.azimuth,
            axis_tilt=axis_tilt,
            axis_azimuth=axis_azimuth,
            max_angle=max_angle,
            backtrack=backtrack,
            gcr=gcr,
            cross_axis_tilt=cross_axis_tilt,
        )
        # singleaxis returns NaN when sun is below horizon — stow to axis orientation
        surface_tilt = tracker["surface_tilt"].fillna(axis_tilt).values
        surface_azimuth = tracker["surface_azimuth"].fillna(axis_azimuth).values
    else:
        # Dual-axis: panel normal points at sun. Clip below horizon.
        zenith = solarpos.apparent_zenith.values
        sun_azimuth = solarpos.azimuth.values
        surface_tilt = np.clip(zenith, 0.0, dual_axis_max_tilt)
        surface_azimuth = sun_azimuth
        # When sun is below horizon, stow flat facing south/north (axis_azimuth fallback)
        below_horizon = zenith >= 90.0
        surface_tilt = np.where(below_horizon, 0.0, surface_tilt)
        surface_azimuth = np.where(below_horizon, axis_azimuth, surface_azimuth)

    effective_irradiance, temp_cell = _compute_effective_irradiance_and_cell_temp(
        weather_aligned, solarpos, surface_tilt=surface_tilt, surface_azimuth=surface_azimuth
    )
    dc_power = _dc_from_poa(
        effective_irradiance,
        temp_cell,
        pv_params,
        n_modules,
        times,
        degradation_rate=degradation_rate,
        current_year=current_year,
        start_year=start_year,
        loss_overrides=loss_overrides,
    )

    if verbose:
        total_kwh = dc_power.sum() * get_hours_per_step(freq) / 1000
        print(f"Total PV DC production ({tracking}): {total_kwh:.1f} kWh")

    return dc_power


def dc_to_ac(
    dc_power: pd.Series, pv_peak_power_w: float, inverter_loading_ratio: float = 1.25, inverter_efficiency: float = 0.96
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

    ac_power = pvlib.inverter.pvwatts(pdc=dc_power, pdc0=inv_size, eta_inv_nom=inverter_efficiency, eta_inv_ref=0.9637)

    return pd.Series(ac_power, index=dc_power.index, name="ac_power_W")


def calculate_pv_production_tmy(
    tmy_data: pd.DataFrame,
    location: Location,
    tilt: float,
    surface_azimuth: float,
    n_modules: int,
    pv_params: Optional[PVModuleParams] = None,
    freq: str = "h",
    verbose: bool = True,
) -> pd.Series:
    """
    Calculate PV DC production from TMY data.

    This is a convenience wrapper around calculate_pv_production_dc for TMY data.

    Args:
        tmy_data: DataFrame with TMY weather data (ghi, dni, dhi, temp_air, wind_speed)
        location: pvlib Location object
        tilt: Panel tilt angle (degrees)
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
        tilt=tilt,
        surface_azimuth=surface_azimuth,
        n_modules=n_modules,
        pv_params=pv_params,
        freq=freq,
        degradation_rate=0.0,  # TMY doesn't include degradation
        verbose=verbose,
    )


def calculate_pv_production_ac(
    weather_data: pd.DataFrame,
    location: Location,
    tilt: float,
    surface_azimuth: float,
    n_modules: int,
    pv_params: Optional[PVModuleParams] = None,
    freq: str = "h",
    degradation_rate: float = 0.0,
    current_year: Optional[int] = None,
    start_year: Optional[int] = None,
    inverter_loading_ratio: float = 1.25,
    inverter_efficiency: float = 0.96,
    verbose: bool = False,
) -> pd.Series:
    """
    Calculate PV AC production from weather data.

    Calculates DC production then converts to AC through inverter.
    Use this for display/reporting purposes or AC-coupled systems.

    Args:
        weather_data: DataFrame with weather variables
        location: pvlib Location object
        tilt: Panel tilt angle (degrees)
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
        tilt=tilt,
        surface_azimuth=surface_azimuth,
        n_modules=n_modules,
        pv_params=pv_params,
        freq=freq,
        degradation_rate=degradation_rate,
        current_year=current_year,
        start_year=start_year,
        verbose=False,
    )

    pv_peak_power_w = n_modules * pv_params.Mpp
    ac_power = dc_to_ac(dc_power, pv_peak_power_w, inverter_loading_ratio, inverter_efficiency)

    if verbose:
        hours_per_step = get_hours_per_step(freq)
        total_kwh = ac_power.sum() * hours_per_step / 1000
        print(f"Total PV AC production for tilt {tilt} deg: {total_kwh:.1f} kWh")

    return ac_power


def _extract_irradiance(weather_df: pd.DataFrame):
    """Extract DNI, GHI, DHI from weather DataFrame with flexible column names."""
    # Try different column naming conventions
    dni_cols = ["dni", "DNI", "direct_normal_irradiance"]
    ghi_cols = ["ghi", "GHI", "shortwave_radiation", "global_horizontal_irradiance"]
    dhi_cols = ["dhi", "DHI", "diffuse_radiation", "diffuse_horizontal_irradiance"]

    dni = _get_column(weather_df, dni_cols)
    ghi = _get_column(weather_df, ghi_cols)
    dhi = _get_column(weather_df, dhi_cols)

    return dni, ghi, dhi


def _extract_met_data(weather_df: pd.DataFrame):
    """Extract temperature and wind speed from weather DataFrame."""
    temp_cols = ["temp_air", "temperature_2m", "temp", "air_temperature"]
    wind_cols = ["wind_speed", "wind_speed_10m", "ws", "WS10m"]

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


def zeb_sizer(houseload: pd.DataFrame, ac_loss: pd.Series, current_n_modules: int, freq: str = "h") -> Dict[str, float]:
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
        "yearly_pv_production_Wh": yearly_pv_production,
        "yearly_consumption_Wh": total_yearly_consumption,
        "pv_to_load_ratio": ratio,
        "is_zeb": ratio >= 1.0,
        "panels_needed_for_zeb": panels_needed_for_zeb,
    }


def calculate_multi_array_production(
    weather_data: pd.DataFrame,
    location: Location,
    arrays: List[Dict[str, Any]],
    freq: str = "h",
    degradation_rate: float = 0.0,
    current_year: Optional[int] = None,
    start_year: Optional[int] = None,
    verbose: bool = False,
    loss_overrides: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """
    Calculate combined DC production from multiple PV arrays.

    Each array is either fixed-tilt or tracking. Mixed configurations are supported.

    Args:
        weather_data: DataFrame with weather variables
        location: pvlib Location object
        arrays: List of array dictionaries. Each entry requires ``modules``.
            Common keys include ``module`` and ``tracking``. Fixed-tilt arrays
            use ``tilt`` and ``azimuth``. Single-axis arrays can also set
            ``axis_tilt``, ``axis_azimuth``, ``max_angle``, ``backtrack``,
            ``gcr``, and ``cross_axis_tilt``. Dual-axis arrays can set
            ``dual_axis_max_tilt``.
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
        n_mod = arr.get("modules", 0)
        if n_mod <= 0:
            continue

        mod_name = arr.get("module", "Generic_400W")
        pv_params = get_module(mod_name)
        tracking = arr.get("tracking", "fixed")

        if tracking == "fixed":
            tilt = arr.get("tilt", 35)
            azimuth = arr.get("azimuth", default_azimuth(location.latitude))

            if verbose:
                print(f"   Array {i + 1}: {n_mod}x {mod_name}, fixed Tilt={tilt}, Azimuth={azimuth}")

            dc = calculate_pv_production_dc(
                weather_data=weather_data,
                location=location,
                tilt=tilt,
                surface_azimuth=azimuth,
                n_modules=n_mod,
                pv_params=pv_params,
                freq=freq,
                degradation_rate=degradation_rate,
                current_year=current_year,
                start_year=start_year,
                verbose=False,
                loss_overrides=loss_overrides,
            )
        elif tracking in ("single_axis", "dual_axis"):
            if verbose:
                if tracking == "single_axis":
                    print(
                        f"   Array {i + 1}: {n_mod}x {mod_name}, single-axis "
                        f"axis_azimuth={arr.get('axis_azimuth', 180.0)}, "
                        f"gcr={arr.get('gcr', 0.35)}, max_angle=±{arr.get('max_angle', 60.0)}"
                    )
                else:
                    print(f"   Array {i + 1}: {n_mod}x {mod_name}, dual-axis")

            dc = calculate_pv_production_dc_tracking(
                weather_data=weather_data,
                location=location,
                n_modules=n_mod,
                tracking=tracking,
                axis_tilt=arr.get("axis_tilt", 0.0),
                axis_azimuth=arr.get("axis_azimuth", default_azimuth(location.latitude)),
                max_angle=arr.get("max_angle", 60.0),
                backtrack=arr.get("backtrack", True),
                gcr=arr.get("gcr", 0.35),
                cross_axis_tilt=arr.get("cross_axis_tilt", 0.0),
                dual_axis_max_tilt=arr.get("dual_axis_max_tilt", 90.0),
                pv_params=pv_params,
                freq=freq,
                degradation_rate=degradation_rate,
                current_year=current_year,
                start_year=start_year,
                verbose=False,
                loss_overrides=loss_overrides,
            )
        else:
            raise ValueError(
                f"Array {i + 1}: unknown tracking mode {tracking!r}. Use 'fixed', 'single_axis', or 'dual_axis'."
            )

        if total_dc is None:
            total_dc = dc.fillna(0)
        else:
            total_dc = total_dc + dc.fillna(0)

    if total_dc is None:
        # Return zeros if no valid arrays
        return pd.Series(0.0, index=weather_data.index)

    if verbose:
        hours_per_step = get_hours_per_step(freq)
        total_kwh = total_dc.sum() * hours_per_step / 1000
        print(f"   Total Multi-Array Production: {total_kwh:,.1f} kWh")

    return total_dc
