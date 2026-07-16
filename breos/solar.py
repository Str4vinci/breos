"""
Solar PV production calculations module.

This module provides functions for calculating photovoltaic power production
using pvlib, with support for both hourly and 15-minute time resolutions.
"""

import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pvlib
from pvlib.albedo import SURFACE_ALBEDOS
from pvlib.location import Location

from breos.cec_fit import fit_cec_params
from breos.inverter import calculate_dc_ac_power
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


def resolve_pvwatts_losses(
    loss_overrides: Optional[Dict[str, float]] = None,
    *,
    age_degradation_percent: float = 0.0,
) -> dict[str, Any]:
    """Return resolved PVWatts loss components and their combined percentage.

    ``loss_overrides`` replaces named BREOS default components. Age-based
    degradation is reported separately because App applies annual degradation
    outside the static PVWatts component stack.
    """
    components = dict(DEFAULT_PVWATTS_LOSSES)
    if loss_overrides:
        unknown = set(loss_overrides) - set(DEFAULT_PVWATTS_LOSSES)
        if unknown:
            valid = ", ".join(sorted(DEFAULT_PVWATTS_LOSSES))
            raise ValueError(f"Unknown loss component(s) {sorted(unknown)}. Valid components: {valid}")
        components.update(loss_overrides)

    combined_percent = pvlib.pvsystem.pvwatts_losses(
        age=age_degradation_percent,
        **components,
    )
    return {
        "components_pct": components,
        "age_degradation_pct": float(age_degradation_percent),
        "combined_pct": float(combined_percent),
    }


# Sky-diffusion (transposition) models for projecting GHI/DHI/DNI onto the
# plane of array, as supported by pvlib.irradiance.get_total_irradiance.
# ``isotropic`` is the simple, robust baseline (and the default); the
# anisotropic models are more accurate on clear days but need extra inputs
# (extraterrestrial DNI and, for the Perez variants, relative airmass).
TRANSPOSITION_MODELS = (
    "isotropic",
    "klucher",
    "haydavies",
    "reindl",
    "king",
    "perez",
    "perez-driesse",
)
DEFAULT_TRANSPOSITION_MODEL = "isotropic"

# Perez sky-diffusion coefficient sets accepted by pvlib's perez model. Only
# used when ``transposition_model == "perez"``; the default matches pvlib.
PEREZ_MODELS = (
    "allsitescomposite1990",
    "allsitescomposite1988",
    "sandiacomposite1988",
    "usacomposite1988",
    "france1988",
    "phoenix1988",
    "elmonte1988",
    "osage1988",
    "albuquerque1988",
    "capecanaveral1988",
    "albany1988",
)
DEFAULT_PEREZ_MODEL = "allsitescomposite1990"

# Named ground-cover types pvlib maps to a ground reflectance (albedo); an
# alternative to supplying a numeric ``albedo`` directly.
SURFACE_TYPES = tuple(sorted(SURFACE_ALBEDOS))

# Where within each timestep the solar position is evaluated.
# ``interval-start`` evaluates at the timestamp itself (the default, and the
# only prior behaviour). ``mid-interval`` evaluates half a step later, which
# is the PVWatts/SAM convention for interval-averaged irradiance: an hourly
# value labelled 07:00 that represents the 07:00-08:00 average pairs with the
# 07:30 sun position. Use it when the weather source reports interval
# averages (e.g. ERA5); keep the default for instantaneous samples.
SOLAR_POSITION_METHODS = (
    "interval-start",
    "mid-interval",
)
DEFAULT_SOLAR_POSITION = "interval-start"

# Whether the incidence-angle modifier is applied to the diffuse POA
# components. ``none`` applies IAM to beam only, with diffuse passing at 1.0
# — the default and the only prior behaviour, a known ~0.5-1% systematic
# overestimate. ``marion`` additionally weighs the sky- and ground-diffuse
# components with the same ashrae IAM integrated over their view factors
# (Marion 2017, via pvlib's ``iam.marion_diffuse``).
DIFFUSE_IAM_METHODS = (
    "none",
    "marion",
)
DEFAULT_DIFFUSE_IAM = "none"
_MARION_DIFFUSE_GRID_STEP_DEG = 0.5
_marion_diffuse_grid_cache: Dict[tuple[float, float, float], tuple[np.ndarray, Dict[str, np.ndarray]]] = {}

# Cell-temperature model and mounting presets. ``faiman`` is pvlib's Faiman
# (2008) model with its open-rack default coefficients (u0=25, u1=6.84) —
# the default and the only prior behaviour. The ``pvsyst-*`` presets use
# pvlib's PVsyst cell model with its documented mounting parameter sets:
# free-standing coefficients run cool for roof-mounted systems, so rooftop
# studies should pick the mounting-appropriate preset (``semi-integrated``
# for close roof mounts with a rear air gap, ``insulated`` for fully
# building-integrated modules with no rear ventilation).
TEMPERATURE_MODELS = (
    "faiman",
    "pvsyst-freestanding",
    "pvsyst-semi-integrated",
    "pvsyst-insulated",
)
DEFAULT_TEMPERATURE_MODEL = "faiman"

# pvsyst-* preset -> key into pvlib's TEMPERATURE_MODEL_PARAMETERS["pvsyst"].
_PVSYST_MOUNTING = {
    "pvsyst-freestanding": "freestanding",
    "pvsyst-semi-integrated": "semi_integrated",
    "pvsyst-insulated": "insulated",
}


def _resolve_transposition_model(model: str) -> str:
    """Normalise and validate a sky-diffusion transposition model name."""
    normalised = str(model).strip().lower()
    if normalised not in TRANSPOSITION_MODELS:
        valid = ", ".join(TRANSPOSITION_MODELS)
        raise ValueError(f"Unknown transposition model {model!r}. Valid models: {valid}")
    return normalised


def _resolve_solar_position_method(method: str) -> str:
    """Normalise and validate a solar-position evaluation method name."""
    normalised = str(method).strip().lower()
    if normalised not in SOLAR_POSITION_METHODS:
        valid = ", ".join(SOLAR_POSITION_METHODS)
        raise ValueError(f"Unknown solar position method {method!r}. Valid methods: {valid}")
    return normalised


def _resolve_diffuse_iam_method(method: str) -> str:
    """Normalise and validate a diffuse-IAM method name."""
    normalised = str(method).strip().lower()
    if normalised not in DIFFUSE_IAM_METHODS:
        valid = ", ".join(DIFFUSE_IAM_METHODS)
        raise ValueError(f"Unknown diffuse IAM method {method!r}. Valid methods: {valid}")
    return normalised


def _marion_diffuse_ashrae(surface_tilt):
    """Return Marion diffuse IAM for ashrae, interpolating large tilt arrays.

    pvlib's exact Marion integration is fast for fixed tilt but expensive for
    tracker arrays with thousands of distinct angles. The integrated
    sky/ground multipliers are smooth over tilt, so a cached 0.5 degree grid
    keeps tracker runs tractable without changing the scalar fixed-tilt path.
    """
    tilt_array = np.asarray(surface_tilt, dtype=float)
    if tilt_array.ndim == 0 or tilt_array.size <= 16:
        return pvlib.iam.marion_diffuse("ashrae", surface_tilt)

    finite = tilt_array[np.isfinite(tilt_array)]
    if finite.size == 0:
        zeros = np.zeros_like(tilt_array, dtype=float)
        return {"sky": zeros, "ground": zeros}

    step = _MARION_DIFFUSE_GRID_STEP_DEG
    lo = math.floor(float(finite.min()) / step) * step
    hi = math.ceil(float(finite.max()) / step) * step
    key = (lo, hi, step)

    if key not in _marion_diffuse_grid_cache:
        grid = np.arange(lo, hi + step / 2.0, step)
        values = {"sky": [], "ground": []}
        for tilt in grid:
            exact = pvlib.iam.marion_diffuse("ashrae", float(tilt))
            values["sky"].append(float(exact["sky"]))
            values["ground"].append(float(exact["ground"]))
        _marion_diffuse_grid_cache[key] = (
            grid,
            {region: np.asarray(region_values) for region, region_values in values.items()},
        )

    grid, values = _marion_diffuse_grid_cache[key]
    interp_tilt = np.nan_to_num(tilt_array, nan=lo)
    return {region: np.interp(interp_tilt, grid, region_values) for region, region_values in values.items()}


def _resolve_temperature_model(model: str) -> str:
    """Normalise and validate a cell-temperature model / mounting preset name."""
    normalised = str(model).strip().lower()
    if normalised not in TEMPERATURE_MODELS:
        valid = ", ".join(TEMPERATURE_MODELS)
        raise ValueError(f"Unknown temperature model {model!r}. Valid models: {valid}")
    return normalised


def _resolve_perez_model(model_perez: str) -> str:
    """Validate the Perez coefficient set name."""
    if model_perez not in PEREZ_MODELS:
        valid = ", ".join(PEREZ_MODELS)
        raise ValueError(f"Unknown Perez coefficient model {model_perez!r}. Valid models: {valid}")
    return model_perez


def _resolve_ground_reflectance(albedo, surface_type):
    """Validate the ground-reflectance inputs and return ``(albedo, surface_type)``.

    Accepts either a numeric ``albedo`` (0-1) or a named ``surface_type`` from
    ``SURFACE_TYPES`` (which pvlib maps to an albedo), but not both.
    """
    if albedo is not None and surface_type is not None:
        raise ValueError("Set either 'albedo' or 'surface_type', not both.")
    if surface_type is not None and surface_type not in SURFACE_ALBEDOS:
        valid = ", ".join(SURFACE_TYPES)
        raise ValueError(f"Unknown surface_type {surface_type!r}. Valid types: {valid}")
    if albedo is not None and not 0.0 <= albedo <= 1.0:
        raise ValueError(f"albedo must be between 0 and 1, got {albedo!r}")
    return albedo, surface_type


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
        if self.gamma_pmp is None:
            self.gamma_pmp = self.T_Pmax_pct


@dataclass(frozen=True)
class PVProductionBreakdown:
    """Intermediate PV model stages for loss-waterfall reporting.

    All series are DC power in watts, indexed like the production series.
    ``dc_after_losses`` is the same output returned by
    :func:`calculate_pv_production_dc` for the same inputs.
    """

    horizontal_reference_dc: pd.Series
    poa_global_dc: pd.Series
    effective_irradiance_dc: pd.Series
    module_dc: pd.Series
    dc_after_static_losses: pd.Series
    dc_after_losses: pd.Series
    pvwatts_component_losses: Dict[str, pd.Series]
    pvwatts_components_pct: Dict[str, float]
    pvwatts_combined_pct: float
    age_degradation_pct: float
    age_degradation_loss: pd.Series


@dataclass(frozen=True)
class _IrradianceModelResult:
    """Irradiance and cell-temperature arrays used by the PV model."""

    ghi: np.ndarray
    poa_global: np.ndarray
    effective_irradiance: np.ndarray
    temp_cell: np.ndarray


def _prepare_solarpos_and_weather(
    weather_data: pd.DataFrame,
    location: Location,
    freq: str,
    solar_position: str = DEFAULT_SOLAR_POSITION,
):
    """Build time index, solar position, and aligned weather frame.

    ``solar_position="mid-interval"`` evaluates the sun half a timestep after
    each label (the PVWatts/SAM convention for interval-averaged irradiance)
    while keeping the returned frame indexed at the labels, so downstream
    alignment is unchanged.
    """
    if not isinstance(weather_data.index, pd.DatetimeIndex):
        raise ValueError("weather_data must have a DatetimeIndex")
    method = _resolve_solar_position_method(solar_position)

    times = pd.date_range(start=weather_data.index[0], end=weather_data.index[-1], freq=freq)
    if method == "mid-interval":
        half_step = pd.Timedelta(hours=get_hours_per_step(freq) / 2.0)
        solarpos = location.get_solarposition(times=times + half_step)
        solarpos.index = times
    else:
        solarpos = location.get_solarposition(times=times)
    weather_aligned = weather_data.reindex(times, method="nearest")
    return times, solarpos, weather_aligned


def _compute_irradiance_and_cell_temp_detail(
    weather_aligned: pd.DataFrame,
    solarpos: pd.DataFrame,
    surface_tilt,
    surface_azimuth,
    transposition_model: str = DEFAULT_TRANSPOSITION_MODEL,
    albedo: Optional[float] = None,
    surface_type: Optional[str] = None,
    model_perez: str = DEFAULT_PEREZ_MODEL,
    diffuse_iam: str = DEFAULT_DIFFUSE_IAM,
    temperature_model: str = DEFAULT_TEMPERATURE_MODEL,
) -> _IrradianceModelResult:
    """Compute GHI, POA, effective irradiance, and cell temperature.

    surface_tilt / surface_azimuth may be scalars (fixed) or per-timestep arrays/Series
    (tracking). Uses pvlib.irradiance.get_total_irradiance which is array-aware.

    ``transposition_model`` selects the sky-diffusion model (see
    ``TRANSPOSITION_MODELS``); the default ``"isotropic"`` reproduces prior
    behaviour bit-for-bit. ``albedo`` (0-1) or ``surface_type`` (see
    ``SURFACE_TYPES``) sets the ground reflectance for the ground-diffuse
    component; when neither is given pvlib's 0.25 default applies.
    ``model_perez`` selects the Perez coefficient set (only used by the
    ``"perez"`` model). ``diffuse_iam`` selects whether IAM is also applied
    to the diffuse components (see ``DIFFUSE_IAM_METHODS``); the default
    ``"none"`` reproduces prior behaviour bit-for-bit. ``temperature_model``
    selects the cell-temperature model / mounting preset (see
    ``TEMPERATURE_MODELS``); the default ``"faiman"`` reproduces prior
    behaviour bit-for-bit, and the pvsyst presets use pvlib's default
    ``module_efficiency``/``alpha_absorption``.
    """
    model = _resolve_transposition_model(transposition_model)
    albedo, surface_type = _resolve_ground_reflectance(albedo, surface_type)
    model_perez = _resolve_perez_model(model_perez)
    diffuse_iam = _resolve_diffuse_iam_method(diffuse_iam)
    temperature_model = _resolve_temperature_model(temperature_model)
    dni, ghi, dhi = _extract_irradiance(weather_aligned)
    temp_air, wind_speed = _extract_met_data(weather_aligned)

    aoi = pvlib.irradiance.aoi(surface_tilt, surface_azimuth, solarpos.apparent_zenith, solarpos.azimuth)
    iam = pvlib.iam.ashrae(aoi)

    # Hay-Davies, Reindl, and the Perez variants need extraterrestrial DNI; the
    # Perez variants additionally need relative airmass. Isotropic, Klucher, and
    # King ignore both, and passing them does not change the isotropic result,
    # so computing them unconditionally keeps the call site simple.
    dni_extra = pvlib.irradiance.get_extra_radiation(solarpos.index)
    airmass = pvlib.atmosphere.get_relative_airmass(solarpos.apparent_zenith)

    # Only forward ground-reflectance overrides when set, so the default path
    # keeps pvlib's built-in albedo and stays bit-for-bit identical.
    ground_kwargs: Dict[str, Any] = {}
    if albedo is not None:
        ground_kwargs["albedo"] = albedo
    if surface_type is not None:
        ground_kwargs["surface_type"] = surface_type

    # apparent_zenith (refraction-corrected) everywhere, matching pvlib's
    # ModelChain; aoi above uses it too. Mixing true and apparent zenith in
    # one transposition call was an inconsistency fixed in 0.3.4.
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        solar_zenith=solarpos.apparent_zenith,
        solar_azimuth=solarpos.azimuth,
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        dni_extra=dni_extra,
        airmass=airmass,
        model=model,
        model_perez=model_perez,
        **ground_kwargs,
    )

    poa_direct = np.nan_to_num(poa["poa_direct"].values, nan=0.0)
    poa_diffuse = np.nan_to_num(poa["poa_diffuse"].values, nan=0.0)
    poa_global = np.nan_to_num(poa["poa_global"].values, nan=0.0)
    iam_clean = np.nan_to_num(np.asarray(iam, dtype=float), nan=0.0)

    if diffuse_iam == "marion":
        # Marion (2017) view-factor-integrated IAM on the diffuse components,
        # using the same ashrae model as the beam IAM above. Transposition
        # folds any horizon-brightening term into poa_sky_diffuse, so the sky
        # multiplier covers it too.
        poa_sky = np.nan_to_num(poa["poa_sky_diffuse"].values, nan=0.0)
        poa_ground = np.nan_to_num(poa["poa_ground_diffuse"].values, nan=0.0)
        multipliers = _marion_diffuse_ashrae(surface_tilt)
        sky_mult = np.nan_to_num(np.asarray(multipliers["sky"], dtype=float), nan=0.0)
        ground_mult = np.nan_to_num(np.asarray(multipliers["ground"], dtype=float), nan=0.0)
        effective_irradiance = poa_direct * iam_clean + poa_sky * sky_mult + poa_ground * ground_mult
    else:
        effective_irradiance = poa_direct * iam_clean + poa_diffuse

    if temperature_model == "faiman":
        temp_cell = pvlib.temperature.faiman(poa_global, temp_air, wind_speed)
    else:
        params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["pvsyst"][_PVSYST_MOUNTING[temperature_model]]
        temp_cell = pvlib.temperature.pvsyst_cell(poa_global, temp_air, wind_speed, **params)
    return _IrradianceModelResult(
        ghi=np.nan_to_num(np.asarray(ghi, dtype=float), nan=0.0),
        poa_global=poa_global,
        effective_irradiance=effective_irradiance,
        temp_cell=np.nan_to_num(np.asarray(temp_cell, dtype=float), nan=25.0),
    )


def _compute_effective_irradiance_and_cell_temp(
    weather_aligned: pd.DataFrame,
    solarpos: pd.DataFrame,
    surface_tilt,
    surface_azimuth,
    transposition_model: str = DEFAULT_TRANSPOSITION_MODEL,
    albedo: Optional[float] = None,
    surface_type: Optional[str] = None,
    model_perez: str = DEFAULT_PEREZ_MODEL,
    diffuse_iam: str = DEFAULT_DIFFUSE_IAM,
    temperature_model: str = DEFAULT_TEMPERATURE_MODEL,
):
    """Compute effective POA irradiance (with IAM) and cell temperature."""
    detail = _compute_irradiance_and_cell_temp_detail(
        weather_aligned,
        solarpos,
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        transposition_model=transposition_model,
        albedo=albedo,
        surface_type=surface_type,
        model_perez=model_perez,
        diffuse_iam=diffuse_iam,
        temperature_model=temperature_model,
    )
    return detail.effective_irradiance, detail.temp_cell


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


def _age_degradation_percent(
    degradation_rate: float = 0.0,
    current_year: Optional[int] = None,
    start_year: Optional[int] = None,
) -> float:
    """Return the age-based PVWatts degradation percentage for this year."""
    if current_year is not None and start_year is not None:
        years_operating = current_year - start_year + 0.5
        return float(100 * (1 - (1 - degradation_rate) ** years_operating))
    return 0.0


def _module_dc_before_losses(
    effective_irradiance: np.ndarray,
    temp_cell: np.ndarray,
    pv_params: "PVModuleParams",
    n_modules: int,
    times: pd.DatetimeIndex,
    name: str,
) -> pd.Series:
    """Run the CEC single-diode model before system-level PVWatts losses."""
    I_L_ref, I_o_ref, R_s, R_sh_ref, a_ref, Adjust = _get_cec_params(pv_params)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        cec = pvlib.pvsystem.calcparams_cec(
            effective_irradiance, temp_cell, pv_params.alpha_sc, a_ref, I_L_ref, I_o_ref, R_sh_ref, R_s, Adjust
        )
        mpp = pvlib.pvsystem.max_power_point(*cec, method="newton")

    p_mp = mpp["p_mp"] if isinstance(mpp, dict) else mpp.p_mp
    return pd.Series(np.asarray(p_mp) * n_modules, index=times, name=name)


def _apply_pvwatts_loss_series(
    dc_power: pd.Series,
    loss_overrides: Optional[Dict[str, float]] = None,
    *,
    age_degradation_percent: float = 0.0,
) -> tuple[pd.Series, pd.Series, Dict[str, pd.Series], dict[str, Any]]:
    """Apply PVWatts losses sequentially and return component series."""
    loss_info = resolve_pvwatts_losses(loss_overrides)
    remaining = dc_power.copy()
    component_losses: Dict[str, pd.Series] = {}

    for name, pct in loss_info["components_pct"].items():
        loss = remaining * (float(pct) / 100.0)
        component_losses[name] = loss.rename(f"{name}_loss_W")
        remaining = remaining - loss

    dc_after_static = remaining.rename("dc_after_static_losses_W")
    age_loss = (dc_after_static * (float(age_degradation_percent) / 100.0)).rename("age_degradation_loss_W")
    dc_after_losses = (dc_after_static - age_loss).rename("dc_power_W")

    return (
        dc_after_static,
        dc_after_losses,
        component_losses,
        {
            **loss_info,
            "age_degradation_pct": float(age_degradation_percent),
        },
    )


def _scale_reference_dc(
    base_dc: pd.Series,
    numerator: np.ndarray,
    denominator: np.ndarray,
    name: str,
) -> pd.Series:
    """Scale a DC reference series by an irradiance ratio, avoiding night noise."""
    numerator_arr = np.nan_to_num(np.asarray(numerator, dtype=float), nan=0.0)
    denominator_arr = np.nan_to_num(np.asarray(denominator, dtype=float), nan=0.0)
    ratio = np.divide(
        numerator_arr,
        denominator_arr,
        out=np.zeros_like(numerator_arr),
        where=np.abs(denominator_arr) > 1e-6,
    )
    return (base_dc * ratio).rename(name)


def _build_pv_production_breakdown(
    weather_aligned: pd.DataFrame,
    solarpos: pd.DataFrame,
    surface_tilt,
    surface_azimuth,
    pv_params: "PVModuleParams",
    n_modules: int,
    times: pd.DatetimeIndex,
    degradation_rate: float = 0.0,
    current_year: Optional[int] = None,
    start_year: Optional[int] = None,
    loss_overrides: Optional[Dict[str, float]] = None,
    transposition_model: str = DEFAULT_TRANSPOSITION_MODEL,
    albedo: Optional[float] = None,
    surface_type: Optional[str] = None,
    model_perez: str = DEFAULT_PEREZ_MODEL,
    diffuse_iam: str = DEFAULT_DIFFUSE_IAM,
    temperature_model: str = DEFAULT_TEMPERATURE_MODEL,
) -> PVProductionBreakdown:
    """Build the full fixed/tracking PV production breakdown."""
    detail = _compute_irradiance_and_cell_temp_detail(
        weather_aligned,
        solarpos,
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        transposition_model=transposition_model,
        albedo=albedo,
        surface_type=surface_type,
        model_perez=model_perez,
        diffuse_iam=diffuse_iam,
        temperature_model=temperature_model,
    )
    module_dc = _module_dc_before_losses(
        detail.effective_irradiance,
        detail.temp_cell,
        pv_params,
        n_modules,
        times,
        name="module_dc_W",
    )
    gamma_per_c = float(pv_params.gamma_pmp) / 100.0
    temperature_factor = 1.0 + gamma_per_c * (detail.temp_cell - 25.0)
    safe_temperature_factor = np.where(np.abs(temperature_factor) > 1e-6, temperature_factor, 1.0)
    effective_irradiance_dc = (module_dc / pd.Series(safe_temperature_factor, index=times)).rename(
        "effective_irradiance_dc_W"
    )
    poa_global_dc = _scale_reference_dc(
        effective_irradiance_dc,
        detail.poa_global,
        detail.effective_irradiance,
        name="poa_global_dc_W",
    )
    horizontal_reference_dc = _scale_reference_dc(
        effective_irradiance_dc,
        detail.ghi,
        detail.effective_irradiance,
        name="horizontal_reference_dc_W",
    )
    age_pct = _age_degradation_percent(
        degradation_rate=degradation_rate,
        current_year=current_year,
        start_year=start_year,
    )
    dc_after_static, dc_after_losses, component_losses, loss_info = _apply_pvwatts_loss_series(
        module_dc,
        loss_overrides,
        age_degradation_percent=age_pct,
    )

    return PVProductionBreakdown(
        horizontal_reference_dc=horizontal_reference_dc,
        poa_global_dc=poa_global_dc,
        effective_irradiance_dc=effective_irradiance_dc,
        module_dc=module_dc,
        dc_after_static_losses=dc_after_static,
        dc_after_losses=dc_after_losses,
        pvwatts_component_losses=component_losses,
        pvwatts_components_pct=loss_info["components_pct"],
        pvwatts_combined_pct=loss_info["combined_pct"],
        age_degradation_pct=age_pct,
        age_degradation_loss=(dc_after_static - dc_after_losses).rename("age_degradation_loss_W"),
    )


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

    total_losses_percent = resolve_pvwatts_losses(
        loss_overrides,
        age_degradation_percent=age_degradation_factor,
    )["combined_pct"]

    p_mp = mpp["p_mp"] if isinstance(mpp, dict) else mpp.p_mp
    dc_power = np.asarray(p_mp) * n_modules * (1 - total_losses_percent / 100)
    return pd.Series(dc_power, index=times, name="dc_power_W")


def calculate_pv_production_breakdown(
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
    transposition_model: str = DEFAULT_TRANSPOSITION_MODEL,
    albedo: Optional[float] = None,
    surface_type: Optional[str] = None,
    model_perez: str = DEFAULT_PEREZ_MODEL,
    solar_position: str = DEFAULT_SOLAR_POSITION,
    diffuse_iam: str = DEFAULT_DIFFUSE_IAM,
    temperature_model: str = DEFAULT_TEMPERATURE_MODEL,
) -> PVProductionBreakdown:
    """Calculate fixed-tilt PV production with intermediate loss stages."""
    if pv_params is None:
        from breos.pv_modules import get_module

        pv_params = get_module("Generic_400W")

    times, solarpos, weather_aligned = _prepare_solarpos_and_weather(
        weather_data, location, freq, solar_position=solar_position
    )
    breakdown = _build_pv_production_breakdown(
        weather_aligned,
        solarpos,
        surface_tilt=tilt,
        surface_azimuth=surface_azimuth,
        pv_params=pv_params,
        n_modules=n_modules,
        times=times,
        degradation_rate=degradation_rate,
        current_year=current_year,
        start_year=start_year,
        loss_overrides=loss_overrides,
        transposition_model=transposition_model,
        albedo=albedo,
        surface_type=surface_type,
        model_perez=model_perez,
        diffuse_iam=diffuse_iam,
        temperature_model=temperature_model,
    )

    if verbose:
        total_kwh = breakdown.dc_after_losses.sum() * get_hours_per_step(freq) / 1000
        print(f"Total PV DC production for tilt {tilt} deg: {total_kwh:.1f} kWh")

    return breakdown


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
    transposition_model: str = DEFAULT_TRANSPOSITION_MODEL,
    albedo: Optional[float] = None,
    surface_type: Optional[str] = None,
    model_perez: str = DEFAULT_PEREZ_MODEL,
    solar_position: str = DEFAULT_SOLAR_POSITION,
    diffuse_iam: str = DEFAULT_DIFFUSE_IAM,
    temperature_model: str = DEFAULT_TEMPERATURE_MODEL,
) -> pd.Series:
    """
    Calculate PV DC production from weather data (fixed-tilt array).

    Returns DC power BEFORE inverter conversion. Use dc_to_ac() to convert
    to AC power for grid export or AC loads.

    For DC-coupled battery systems:
    - Use DC output directly for battery charging (apply only charge efficiency)
    - Use dc_to_ac() for power going directly to AC loads or grid

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
        loss_overrides: Per-component PVWatts loss overrides (percent)
        transposition_model: Sky-diffusion model for POA transposition
            (one of ``TRANSPOSITION_MODELS``); defaults to ``"isotropic"``.
        albedo: Ground reflectance (0-1) for the ground-diffuse component;
            ``None`` uses pvlib's 0.25 default. Mutually exclusive with
            ``surface_type``.
        surface_type: Named ground cover (one of ``SURFACE_TYPES``) mapped to
            an albedo by pvlib; an alternative to ``albedo``.
        model_perez: Perez coefficient set (one of ``PEREZ_MODELS``); only
            used when ``transposition_model`` is ``"perez"``.
        solar_position: Where within each timestep the sun position is
            evaluated (one of ``SOLAR_POSITION_METHODS``). ``"mid-interval"``
            matches PVWatts/SAM for interval-averaged irradiance; the default
            ``"interval-start"`` reproduces prior behaviour bit-for-bit.
        diffuse_iam: Whether IAM is also applied to the diffuse POA
            components (one of ``DIFFUSE_IAM_METHODS``). ``"marion"``
            weighs sky- and ground-diffuse with the view-factor-integrated
            ashrae IAM; the default ``"none"`` (beam-only IAM) reproduces
            prior behaviour bit-for-bit.
        temperature_model: Cell-temperature model / mounting preset (one of
            ``TEMPERATURE_MODELS``). The ``pvsyst-*`` presets use pvlib's
            PVsyst cell model with its documented mounting coefficients;
            the default ``"faiman"`` (open rack) reproduces prior behaviour
            bit-for-bit.

    Returns:
        pd.Series with DC power production in Watts (before inverter)
    """
    return calculate_pv_production_breakdown(
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
        verbose=verbose,
        loss_overrides=loss_overrides,
        transposition_model=transposition_model,
        albedo=albedo,
        surface_type=surface_type,
        model_perez=model_perez,
        solar_position=solar_position,
        diffuse_iam=diffuse_iam,
        temperature_model=temperature_model,
    ).dc_after_losses


def calculate_pv_production_tracking_breakdown(
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
    transposition_model: str = DEFAULT_TRANSPOSITION_MODEL,
    albedo: Optional[float] = None,
    surface_type: Optional[str] = None,
    model_perez: str = DEFAULT_PEREZ_MODEL,
    solar_position: str = DEFAULT_SOLAR_POSITION,
    diffuse_iam: str = DEFAULT_DIFFUSE_IAM,
    temperature_model: str = DEFAULT_TEMPERATURE_MODEL,
) -> PVProductionBreakdown:
    """Calculate tracking-array PV production with intermediate loss stages.

    Single-axis (horizontal or tilted) trackers are the dominant configuration in
    utility-scale PV. Dual-axis trackers gain slightly more energy but at higher
    cost; they are common in CPV and high-latitude installations.
    """
    if tracking not in ("single_axis", "dual_axis"):
        raise ValueError(f"tracking must be 'single_axis' or 'dual_axis', got {tracking!r}")

    if pv_params is None:
        from breos.pv_modules import get_module

        pv_params = get_module("Generic_400W")

    times, solarpos, weather_aligned = _prepare_solarpos_and_weather(
        weather_data, location, freq, solar_position=solar_position
    )

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

    breakdown = _build_pv_production_breakdown(
        weather_aligned,
        solarpos,
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        pv_params=pv_params,
        n_modules=n_modules,
        times=times,
        degradation_rate=degradation_rate,
        current_year=current_year,
        start_year=start_year,
        loss_overrides=loss_overrides,
        transposition_model=transposition_model,
        albedo=albedo,
        surface_type=surface_type,
        model_perez=model_perez,
        diffuse_iam=diffuse_iam,
        temperature_model=temperature_model,
    )

    if verbose:
        total_kwh = breakdown.dc_after_losses.sum() * get_hours_per_step(freq) / 1000
        print(f"Total PV DC production ({tracking}): {total_kwh:.1f} kWh")

    return breakdown


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
    transposition_model: str = DEFAULT_TRANSPOSITION_MODEL,
    albedo: Optional[float] = None,
    surface_type: Optional[str] = None,
    model_perez: str = DEFAULT_PEREZ_MODEL,
    solar_position: str = DEFAULT_SOLAR_POSITION,
    diffuse_iam: str = DEFAULT_DIFFUSE_IAM,
    temperature_model: str = DEFAULT_TEMPERATURE_MODEL,
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
        dual_axis_max_tilt: Maximum panel tilt for dual-axis. ``90`` = unlimited.
        pv_params: PV module parameters (uses defaults if None).
        freq: Time frequency (``"h"`` or ``"15min"``).
        degradation_rate: Annual degradation rate.
        current_year: Current simulation year (for age-based degradation).
        start_year: Year system was installed.
        verbose: Whether to print production summary.
        loss_overrides: Per-component PVWatts loss overrides (percent).
        transposition_model: Sky-diffusion model for POA transposition
            (one of ``TRANSPOSITION_MODELS``); defaults to ``"isotropic"``.
        albedo: Ground reflectance (0-1) for the ground-diffuse component;
            ``None`` uses pvlib's 0.25 default. Mutually exclusive with
            ``surface_type``.
        surface_type: Named ground cover (one of ``SURFACE_TYPES``) mapped to
            an albedo by pvlib; an alternative to ``albedo``.
        model_perez: Perez coefficient set (one of ``PEREZ_MODELS``); only
            used when ``transposition_model`` is ``"perez"``.
        solar_position: Where within each timestep the sun position is
            evaluated (one of ``SOLAR_POSITION_METHODS``); also drives the
            tracker rotation angles.
        diffuse_iam: Whether IAM is also applied to the diffuse POA
            components (one of ``DIFFUSE_IAM_METHODS``); the default
            ``"none"`` reproduces prior behaviour bit-for-bit.
        temperature_model: Cell-temperature model / mounting preset (one of
            ``TEMPERATURE_MODELS``); the default ``"faiman"`` (open rack)
            reproduces prior behaviour bit-for-bit.

    Returns:
        pd.Series with DC power production in Watts (before inverter).
    """
    return calculate_pv_production_tracking_breakdown(
        weather_data=weather_data,
        location=location,
        n_modules=n_modules,
        tracking=tracking,
        axis_tilt=axis_tilt,
        axis_azimuth=axis_azimuth,
        max_angle=max_angle,
        backtrack=backtrack,
        gcr=gcr,
        cross_axis_tilt=cross_axis_tilt,
        dual_axis_max_tilt=dual_axis_max_tilt,
        pv_params=pv_params,
        freq=freq,
        degradation_rate=degradation_rate,
        current_year=current_year,
        start_year=start_year,
        verbose=verbose,
        loss_overrides=loss_overrides,
        transposition_model=transposition_model,
        albedo=albedo,
        surface_type=surface_type,
        model_perez=model_perez,
        solar_position=solar_position,
        diffuse_iam=diffuse_iam,
        temperature_model=temperature_model,
    ).dc_after_losses


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

    ac_power = dc_power.map(lambda value: calculate_dc_ac_power(value, inv_size, inverter_efficiency).ac_power_w)

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
    transposition_model: str = DEFAULT_TRANSPOSITION_MODEL,
    albedo: Optional[float] = None,
    surface_type: Optional[str] = None,
    model_perez: str = DEFAULT_PEREZ_MODEL,
    solar_position: str = DEFAULT_SOLAR_POSITION,
    diffuse_iam: str = DEFAULT_DIFFUSE_IAM,
    temperature_model: str = DEFAULT_TEMPERATURE_MODEL,
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
        transposition_model=transposition_model,
        albedo=albedo,
        surface_type=surface_type,
        model_perez=model_perez,
        solar_position=solar_position,
        diffuse_iam=diffuse_iam,
        temperature_model=temperature_model,
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
    transposition_model: str = DEFAULT_TRANSPOSITION_MODEL,
    albedo: Optional[float] = None,
    surface_type: Optional[str] = None,
    model_perez: str = DEFAULT_PEREZ_MODEL,
    solar_position: str = DEFAULT_SOLAR_POSITION,
    diffuse_iam: str = DEFAULT_DIFFUSE_IAM,
    temperature_model: str = DEFAULT_TEMPERATURE_MODEL,
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
        transposition_model=transposition_model,
        albedo=albedo,
        surface_type=surface_type,
        model_perez=model_perez,
        solar_position=solar_position,
        diffuse_iam=diffuse_iam,
        temperature_model=temperature_model,
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


def _sum_pv_breakdowns(breakdowns: list[PVProductionBreakdown]) -> PVProductionBreakdown:
    """Sum per-array PVProductionBreakdown objects into one system total."""
    if not breakdowns:
        raise ValueError("At least one PV production breakdown is required")

    def _sum_attr(name: str) -> pd.Series:
        total = getattr(breakdowns[0], name).copy()
        for breakdown in breakdowns[1:]:
            total = total.add(getattr(breakdown, name), fill_value=0.0)
        return total.rename(getattr(breakdowns[0], name).name)

    component_losses: Dict[str, pd.Series] = {}
    component_names = breakdowns[0].pvwatts_component_losses.keys()
    for component in component_names:
        total = breakdowns[0].pvwatts_component_losses[component].copy()
        for breakdown in breakdowns[1:]:
            total = total.add(breakdown.pvwatts_component_losses[component], fill_value=0.0)
        component_losses[component] = total.rename(f"{component}_loss_W")

    return PVProductionBreakdown(
        horizontal_reference_dc=_sum_attr("horizontal_reference_dc"),
        poa_global_dc=_sum_attr("poa_global_dc"),
        effective_irradiance_dc=_sum_attr("effective_irradiance_dc"),
        module_dc=_sum_attr("module_dc"),
        dc_after_static_losses=_sum_attr("dc_after_static_losses"),
        dc_after_losses=_sum_attr("dc_after_losses"),
        pvwatts_component_losses=component_losses,
        pvwatts_components_pct=breakdowns[0].pvwatts_components_pct,
        pvwatts_combined_pct=breakdowns[0].pvwatts_combined_pct,
        age_degradation_pct=breakdowns[0].age_degradation_pct,
        age_degradation_loss=_sum_attr("age_degradation_loss"),
    )


def calculate_multi_array_production_breakdown(
    weather_data: pd.DataFrame,
    location: Location,
    arrays: List[Dict[str, Any]],
    freq: str = "h",
    degradation_rate: float = 0.0,
    current_year: Optional[int] = None,
    start_year: Optional[int] = None,
    verbose: bool = False,
    loss_overrides: Optional[Dict[str, float]] = None,
    transposition_model: str = DEFAULT_TRANSPOSITION_MODEL,
    albedo: Optional[float] = None,
    surface_type: Optional[str] = None,
    model_perez: str = DEFAULT_PEREZ_MODEL,
    solar_position: str = DEFAULT_SOLAR_POSITION,
    diffuse_iam: str = DEFAULT_DIFFUSE_IAM,
    temperature_model: str = DEFAULT_TEMPERATURE_MODEL,
) -> PVProductionBreakdown:
    """Calculate combined DC production breakdown from multiple PV arrays.

    Each array is either fixed-tilt or tracking. Mixed configurations are supported.
    """
    # Import locally to avoid circular dependencies (if solar imported by pv_modules)
    try:
        from breos.pv_modules import get_module
    except ImportError:
        raise ImportError("breos.pv_modules is required for multi-array production")

    breakdowns: list[PVProductionBreakdown] = []

    for i, arr in enumerate(arrays):
        n_mod = arr.get("modules", 0)
        if n_mod <= 0:
            continue

        mod_name = arr.get("module", "Generic_400W")
        pv_params = get_module(mod_name)
        tracking = arr.get("tracking", "fixed")
        arr_transposition = arr.get("transposition_model", transposition_model)
        arr_albedo = arr.get("albedo", albedo)
        arr_surface_type = arr.get("surface_type", surface_type)
        arr_model_perez = arr.get("model_perez", model_perez)

        if tracking == "fixed":
            tilt = arr.get("tilt", 35)
            azimuth = arr.get("azimuth", default_azimuth(location.latitude))

            if verbose:
                print(f"   Array {i + 1}: {n_mod}x {mod_name}, fixed Tilt={tilt}, Azimuth={azimuth}")

            breakdown = calculate_pv_production_breakdown(
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
                transposition_model=arr_transposition,
                albedo=arr_albedo,
                surface_type=arr_surface_type,
                model_perez=arr_model_perez,
                solar_position=solar_position,
                diffuse_iam=diffuse_iam,
                temperature_model=temperature_model,
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

            breakdown = calculate_pv_production_tracking_breakdown(
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
                transposition_model=arr_transposition,
                albedo=arr_albedo,
                surface_type=arr_surface_type,
                model_perez=arr_model_perez,
                solar_position=solar_position,
                diffuse_iam=diffuse_iam,
                temperature_model=temperature_model,
            )
        else:
            raise ValueError(
                f"Array {i + 1}: unknown tracking mode {tracking!r}. Use 'fixed', 'single_axis', or 'dual_axis'."
            )

        breakdowns.append(breakdown)

    if not breakdowns:
        zeros = pd.Series(0.0, index=weather_data.index, name="dc_power_W")
        static_loss_info = resolve_pvwatts_losses(loss_overrides)
        return PVProductionBreakdown(
            horizontal_reference_dc=zeros.rename("horizontal_reference_dc_W"),
            poa_global_dc=zeros.rename("poa_global_dc_W"),
            effective_irradiance_dc=zeros.rename("effective_irradiance_dc_W"),
            module_dc=zeros.rename("module_dc_W"),
            dc_after_static_losses=zeros.rename("dc_after_static_losses_W"),
            dc_after_losses=zeros,
            pvwatts_component_losses={
                name: zeros.rename(f"{name}_loss_W") for name in static_loss_info["components_pct"]
            },
            pvwatts_components_pct=static_loss_info["components_pct"],
            pvwatts_combined_pct=static_loss_info["combined_pct"],
            age_degradation_pct=0.0,
            age_degradation_loss=zeros.rename("age_degradation_loss_W"),
        )

    total = _sum_pv_breakdowns(breakdowns)

    if verbose:
        hours_per_step = get_hours_per_step(freq)
        total_kwh = total.dc_after_losses.sum() * hours_per_step / 1000
        print(f"   Total Multi-Array Production: {total_kwh:,.1f} kWh")

    return total


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
    transposition_model: str = DEFAULT_TRANSPOSITION_MODEL,
    albedo: Optional[float] = None,
    surface_type: Optional[str] = None,
    model_perez: str = DEFAULT_PEREZ_MODEL,
    solar_position: str = DEFAULT_SOLAR_POSITION,
    diffuse_iam: str = DEFAULT_DIFFUSE_IAM,
    temperature_model: str = DEFAULT_TEMPERATURE_MODEL,
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
            ``dual_axis_max_tilt``. Any array may also set
            ``transposition_model``, ``albedo``/``surface_type``, or
            ``model_perez`` to override the function-level defaults.
        freq: Time frequency ('h' or '15min')
        degradation_rate: Annual degradation rate
        current_year: Current simulation year
        start_year: Installation year
        verbose: Print summary
        loss_overrides: Per-component PVWatts loss overrides (percent)
        transposition_model: Default sky-diffusion model for arrays that do
            not set their own (one of ``TRANSPOSITION_MODELS``).
        albedo: Default ground reflectance (0-1); arrays may override with
            their own ``albedo`` or ``surface_type``.
        surface_type: Default named ground cover (one of ``SURFACE_TYPES``);
            mutually exclusive with ``albedo``.
        model_perez: Default Perez coefficient set (one of ``PEREZ_MODELS``).
        diffuse_iam: Whether IAM is also applied to the diffuse POA
            components (one of ``DIFFUSE_IAM_METHODS``); function-level for
            all arrays, like ``solar_position``.
        temperature_model: Cell-temperature model / mounting preset (one of
            ``TEMPERATURE_MODELS``); function-level for all arrays, like
            ``solar_position``.

    Returns:
        pd.Series with total DC power (watts)
    """
    return calculate_multi_array_production_breakdown(
        weather_data=weather_data,
        location=location,
        arrays=arrays,
        freq=freq,
        degradation_rate=degradation_rate,
        current_year=current_year,
        start_year=start_year,
        verbose=verbose,
        loss_overrides=loss_overrides,
        transposition_model=transposition_model,
        albedo=albedo,
        surface_type=surface_type,
        model_perez=model_perez,
        solar_position=solar_position,
        diffuse_iam=diffuse_iam,
        temperature_model=temperature_model,
    ).dc_after_losses
