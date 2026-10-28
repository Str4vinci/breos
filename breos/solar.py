"""
Solar PV production calculations module.

This module provides functions for calculating photovoltaic power production
using pvlib, with support for both hourly and 15-minute time resolutions.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd
import pvlib
from pvlib.albedo import SURFACE_ALBEDOS
from pvlib.location import Location

from breos.cec_fit import fit_cec_params
from breos.inverter import _calculate_dc_ac_power_arrays
from breos.pv import _solver
from breos.pv.iam import calculate_front_effective_irradiance
from breos.pv.model_options import (
    BIFACIAL_MODELS as BIFACIAL_MODELS,
)
from breos.pv.model_options import (
    DEFAULT_BIFACIAL_MODEL,
    DEFAULT_DIFFUSE_IAM,
    DEFAULT_IAM_MODEL,
    DEFAULT_PEREZ_MODEL,
    DEFAULT_SOLAR_POSITION,
    DEFAULT_TEMPERATURE_MODEL,
    DEFAULT_TRANSPOSITION_MODEL,
    PER_ARRAY_MODEL_OPTION_KEYS,
    PV_MODEL_CONFIG_KEYS,
    PVModelOptions,
    resolve_pv_model_options,
    resolve_solar_position_method,
    solar_position_at_labels,
    solar_position_time_offset,
    solar_position_warning_scope,
)
from breos.pv.model_options import (
    DIFFUSE_IAM_METHODS as DIFFUSE_IAM_METHODS,
)
from breos.pv.model_options import (
    IAM_MODELS as IAM_MODELS,
)
from breos.pv.model_options import (
    PEREZ_MODELS as PEREZ_MODELS,
)
from breos.pv.model_options import (
    SOLAR_POSITION_METHODS as SOLAR_POSITION_METHODS,
)
from breos.pv.model_options import (
    SURFACE_TYPES as SURFACE_TYPES,
)
from breos.pv.model_options import (
    TEMPERATURE_MODELS as TEMPERATURE_MODELS,
)
from breos.pv.model_options import (
    TRANSPOSITION_MODELS as TRANSPOSITION_MODELS,
)
from breos.pv.shading import CIRCUMSOLAR_MODELS, ArrayShading, ModuleMeanShading, ShadingTable, as_shading_table
from breos.pv.temperature import calculate_cell_temperature
from breos.pv_modules import PVModuleParams, get_module
from breos.utils import IRRADIANCE_COLUMN_ALIASES, MET_COLUMN_ALIASES, find_irradiance_column, find_met_column

# The module an entry point models when it is given none.
_DEFAULT_MODULE = "Generic_400W"
_TRACKERS = ("single_axis", "dual_axis")

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

# ``PV_MODEL_CONFIG_KEYS`` is the PV model-option block every public
# ``calculate_pv_production_*`` entry point accepts. Each of those functions
# still declares these explicitly — the signature is the documentation, and
# ``**kwargs`` would silently swallow a misspelled option — but they hand the
# block to the engine with ``_model_option_kwargs`` instead of re-listing it.
# On a tracker ``gcr`` is also the row spacing the backtracking uses.


def _model_option_kwargs(caller_locals: Dict[str, Any]) -> Dict[str, Any]:
    """Pick the shared model-option block out of a caller's ``locals()``.

    Call this from a wrapper that only forwards its options, passing
    ``locals()`` before any of the named options is rebound, so the result is
    the arguments as the caller received them rather than anything derived
    later in the body. Missing a key is a ``KeyError`` at the call, which is
    what keeps the tuple and the signatures from drifting apart silently.
    """
    return {key: caller_locals[key] for key in PV_MODEL_CONFIG_KEYS}


def resolve_pvwatts_losses(
    loss_overrides: Optional[Dict[str, float]] = None,
    *,
    age_degradation_percent: float = 0.0,
) -> dict[str, Any]:
    """Return resolved PVWatts loss components and their combined percentage.

    ``loss_overrides`` replaces named BREOS default components. Age-based
    degradation is reported separately because App applies annual degradation
    outside the static PVWatts component stack. ``age_degradation_percent`` is
    already a loss percentage, not an age: for a module ``age`` full years old
    at the start of the simulated year it is
    ``100 * (1 - (1 - degradation_rate) ** age)``.
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


@dataclass(frozen=True)
class ModuleOperatingPoint:
    """One module's single-diode parameters and maximum power point at each step.

    The five parameters are those of :func:`pvlib.pvsystem.calcparams_cec`, in
    A, ohm and V; ``v_mp`` and ``i_mp`` are the maximum power point in V and A.
    Every array is indexed like the production series.
    """

    photocurrent: np.ndarray
    saturation_current: np.ndarray
    resistance_series: np.ndarray
    resistance_shunt: np.ndarray
    n_ns_vth: np.ndarray
    v_mp: np.ndarray
    i_mp: np.ndarray


@dataclass(frozen=True)
class IrradianceParts:
    """One array's front irradiance by how near shading reaches it, for module-level shading.

    ``beam`` is the direct irradiance after the incidence-angle modifier and
    ``circumsolar`` the sky diffuse from around the sun, both of which a
    module's beam shaded fraction reduces; ``sky`` the rest of the sky
    diffuse, which its sky view factor scales and which can be negative where
    a Perez model's horizon term is; ``other`` the ground-reflected and rear
    irradiance, which shading does not reach. All are unshaded, in W/m², with
    the cell temperature and the sun's position at each step.
    """

    beam: np.ndarray
    circumsolar: np.ndarray
    sky: np.ndarray
    other: np.ndarray
    temp_cell: np.ndarray
    sun_azimuth: np.ndarray
    sun_elevation: np.ndarray
    pv_params: PVModuleParams

    def irradiance(self, beam_shaded: np.ndarray, sky_view: Any) -> np.ndarray:
        """A module's effective irradiance at each step, from its beam shaded fraction and sky view factor.

        The sky diffuse it keeps is clipped at 0, as the array's is.
        """
        sky = np.clip(self.circumsolar * (1.0 - beam_shaded) + self.sky * sky_view, 0.0, None)
        return self.beam * (1.0 - beam_shaded) + sky + self.other

    def diode_parameters(self, effective_irradiance: np.ndarray, rows: Any = None) -> tuple[np.ndarray, ...]:
        """One module's single-diode parameters at ``effective_irradiance`` and the array's cell temperature.

        ``effective_irradiance`` has the steps ``rows`` (all steps when None)
        as its last axis.
        """
        I_L_ref, I_o_ref, R_s, R_sh_ref, a_ref, Adjust = _get_cec_params(self.pv_params)
        temp_cell = self.temp_cell if rows is None else self.temp_cell[rows]
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            cec = pvlib.pvsystem.calcparams_cec(
                effective_irradiance, temp_cell, self.pv_params.alpha_sc, a_ref, I_L_ref, I_o_ref, R_sh_ref, R_s, Adjust
            )
        cec = (_solver.floor_photocurrent(cec[0], I_L_ref), *cec[1:])
        shape = np.shape(effective_irradiance)
        return tuple(np.broadcast_to(np.asarray(value, dtype=float), shape) for value in cec)


@dataclass(frozen=True)
class PVProductionBreakdown:
    """Intermediate PV model stages for loss-waterfall reporting.

    All series are DC power in watts, indexed like the production series.
    ``dc_after_losses`` is the same output returned by
    :func:`calculate_pv_production_dc` for the same inputs.
    ``age_degradation_pct`` is the start-of-year module-age loss applied after
    ``dc_after_static_losses``; it is 0 in the installation year.
    """

    horizontal_reference_dc: pd.Series
    poa_global_dc: pd.Series
    front_effective_irradiance_dc: pd.Series
    rear_gain_dc: pd.Series
    effective_irradiance_dc: pd.Series
    module_dc: pd.Series
    dc_after_static_losses: pd.Series
    dc_after_losses: pd.Series
    pvwatts_component_losses: Dict[str, pd.Series]
    pvwatts_components_pct: Dict[str, float]
    pvwatts_combined_pct: float
    age_degradation_pct: float
    age_degradation_loss: pd.Series
    # One array's module operating point; None on a sum of arrays.
    module_operating: Optional[ModuleOperatingPoint] = None
    # DC lost to the inverter's input limits with string-level energy
    # (``electrical_fidelity = "string"``), already taken out of
    # ``dc_after_losses``; None otherwise.
    mppt_limit_loss: Optional[pd.Series] = None
    # DC lost in the strings' cables when the topology gives them, already
    # taken out of ``dc_after_losses``; None otherwise.
    dc_wiring_loss: Optional[pd.Series] = None
    # The plane-of-array stage after near shading when the array has a
    # shading table; None otherwise. ``poa_global_dc`` stays unshaded.
    poa_shaded_dc: Optional[pd.Series] = None
    # One array's irradiance by how near shading reaches it, when its strings
    # carry module shading; None otherwise.
    irradiance_parts: Optional[IrradianceParts] = None
    # DC lost to the electrical effect of uneven shading on the strings'
    # modules, already taken out of ``dc_after_losses``; None without module
    # shading. ``string_shading_mismatch`` holds each such string's own, by
    # its position in ``pv_strings``.
    shading_mismatch_loss: Optional[pd.Series] = None
    string_shading_mismatch: Optional[Dict[int, pd.Series]] = None
    # DC that strings on different curves lose by sharing an MPPT input's
    # voltage, already taken out of ``dc_after_losses``; None when no input
    # holds such strings. ``input_parallel_mismatch`` holds each input's, by
    # its ``(inverter, mppt)`` indices.
    parallel_mismatch_loss: Optional[pd.Series] = None
    input_parallel_mismatch: Optional[Dict[Tuple[int, int], pd.Series]] = None
    # The inverter inputs' operating DC voltage with string-level energy,
    # weighted by the power each input delivers, in V; 0 where none does.
    # None otherwise.
    dc_voltage: Optional[pd.Series] = None
    # With string-level energy and several inverters, each inverter's share
    # of ``dc_after_losses`` and its inputs' operating DC voltage, in the
    # order of ``inverters``; None otherwise.
    inverter_dc: Optional[Tuple[pd.Series, ...]] = None
    inverter_dc_voltage: Optional[Tuple[pd.Series, ...]] = None
    solver_diagnostics: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class _IrradianceModelResult:
    """Irradiance and cell-temperature arrays used by the PV model."""

    ghi: np.ndarray
    poa_global: np.ndarray
    front_effective_irradiance: np.ndarray
    rear_effective_irradiance: np.ndarray
    effective_irradiance: np.ndarray
    temp_cell: np.ndarray
    # The plane-of-array and effective irradiance before near shading; None
    # without it.
    poa_global_unshaded: Optional[np.ndarray] = None
    effective_irradiance_unshaded: Optional[np.ndarray] = None
    # The irradiance by how shading reaches it, for module-level shading.
    parts: Optional[IrradianceParts] = None


class _SolarGrid(NamedTuple):
    """The simulation grid, the sun's position at each step, and the weather on that grid.

    Every array on one site shares it, so the sun is computed once per
    weather frame.
    """

    times: pd.DatetimeIndex
    solarpos: pd.DataFrame
    weather: pd.DataFrame


def _prepare_solarpos_and_weather(
    weather_data: pd.DataFrame,
    location: Location,
    freq: str,
    solar_position: str = DEFAULT_SOLAR_POSITION,
) -> _SolarGrid:
    """Build the time grid, the solar position, and the aligned weather frame.

    ``solar_position="mid-interval"`` evaluates the sun half a timestep after
    each label (the PVWatts/SAM convention for interval-averaged irradiance)
    while keeping the returned frame indexed at the labels, so downstream
    alignment is unchanged.
    """
    times = _simulation_grid(weather_data, freq, solar_position)
    solarpos, _method = solar_position_at_labels(location, times, weather_data, freq, solar_position)
    return _SolarGrid(times, solarpos, weather_data.set_axis(times))


def _simulation_grid(weather_data: pd.DataFrame, freq: str, solar_position: str) -> pd.DatetimeIndex:
    """Check the weather and the solar-position method, and return the simulation grid."""
    if not isinstance(weather_data.index, pd.DatetimeIndex):
        raise ValueError("weather_data must have a DatetimeIndex")
    # A misspelt method is reported before any problem with the weather grid.
    resolve_solar_position_method(solar_position)
    return _require_weather_grid(weather_data.index, freq)


def _require_weather_grid(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """Return the simulation grid, which must be the weather's own timestamps.

    PV is computed at each weather timestamp, so the weather must step evenly
    at ``freq`` from its first timestamp to its last. A gap, or weather at
    another resolution, raises instead of being filled from the nearest row.
    """
    if len(index) == 0:
        raise ValueError("weather_data has no rows")
    times = pd.date_range(start=index[0], periods=len(index), freq=freq)
    mismatch = np.flatnonzero(index.as_unit("ns").asi8 != times.as_unit("ns").asi8)  # type: ignore[attr-defined]  # pandas-stubs omits asi8
    if mismatch.size:
        row = int(mismatch[0])
        step = index[row] - index[row - 1] if row else None
        raise ValueError(
            f"weather_data must step evenly at freq={freq!r}: row {row} is {index[row]}, "
            f"{step} after the previous row, where {times[row]} was expected. Resample or fill "
            "the weather explicitly before computing PV."
        )
    return times


def _compute_irradiance_and_cell_temp_detail(
    weather_aligned: pd.DataFrame,
    solarpos: pd.DataFrame,
    surface_tilt,
    surface_azimuth,
    pv_params: PVModuleParams,
    model_options: PVModelOptions,
    shading: Optional[ArrayShading] = None,
) -> _IrradianceModelResult:
    """Compute GHI, POA, effective irradiance, and cell temperature.

    surface_tilt / surface_azimuth may be scalars (fixed) or per-timestep arrays/Series
    (tracking). Uses pvlib.irradiance.get_total_irradiance which is array-aware.

    ``model_options`` must come from
    :func:`breos.pv.model_options.resolve_pv_model_options`; every model name
    and geometry value it carries is taken as already validated. Its BREOS
    defaults (isotropic transposition, beam-only IAM, faiman cell temperature,
    no rear-side model, pvlib's own 0.25 albedo) reproduce pre-0.4 behaviour
    bit-for-bit — see ``breos.pv.model_options`` for what each choice means.

    The cell-temperature model is driven by front-side ``poa_global`` plus the
    bifaciality-weighted rear irradiance, so rear gain contributes heat as well
    as power; with ``bifacial_model="none"`` the rear term is zero and the
    result is unchanged.

    ``shading`` reduces the front plane-of-array irradiance before the
    incidence-angle modifier: the beam by the table's shaded fraction at the
    sun's position, the sky diffuse by the sky view factor, and, with a
    transposition model that separates it, the circumsolar diffuse by the
    beam's shaded fraction. The ground-reflected and rear irradiance are not
    shaded.
    """
    dni, ghi, dhi = _extract_irradiance(weather_aligned)
    temp_air, wind_speed = _extract_met_data(weather_aligned)

    aoi = pvlib.irradiance.aoi(surface_tilt, surface_azimuth, solarpos.apparent_zenith, solarpos.azimuth)
    # Hay-Davies, Reindl, and the Perez variants need extraterrestrial DNI; the
    # Perez variants additionally need relative airmass. Isotropic, Klucher, and
    # King ignore both, and passing them does not change the isotropic result,
    # so computing them unconditionally keeps the call site simple.
    dni_extra = pvlib.irradiance.get_extra_radiation(solarpos.index)
    airmass = pvlib.atmosphere.get_relative_airmass(solarpos.apparent_zenith)

    # Only forward ground-reflectance overrides when set, so the default path
    # keeps pvlib's built-in albedo and stays bit-for-bit identical.
    ground_kwargs: Dict[str, Any] = {}
    if model_options.albedo is not None:
        ground_kwargs["albedo"] = model_options.albedo
    if model_options.surface_type is not None:
        ground_kwargs["surface_type"] = model_options.surface_type

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
        model=model_options.transposition_model,
        model_perez=model_options.model_perez,
        **ground_kwargs,
    )

    poa_global_unshaded = None
    front_unshaded = None
    components = None
    if shading is not None:
        poa_global_unshaded = np.nan_to_num(poa["poa_global"].values, nan=0.0)
        front_unshaded = calculate_front_effective_irradiance(
            poa,
            aoi,
            surface_tilt,
            model_options.diffuse_iam,
            model_options.iam_model,
        )
        poa, components = _shaded_poa(
            poa,
            shading,
            solarpos,
            surface_tilt,
            surface_azimuth,
            dni=dni,
            dhi=dhi,
            dni_extra=dni_extra,
            airmass=airmass,
            model_options=model_options,
        )
    poa_global = np.nan_to_num(poa["poa_global"].values, nan=0.0)
    front_effective_irradiance = calculate_front_effective_irradiance(
        poa,
        aoi,
        surface_tilt,
        model_options.diffuse_iam,
        model_options.iam_model,
    )

    if model_options.bifacial_model == "infinite_sheds":
        tilt = np.asarray(surface_tilt, dtype=float)
        azimuth = np.asarray(surface_azimuth, dtype=float)
        back_tilt = 180.0 - tilt
        back_azimuth = np.mod(azimuth + 180.0, 360.0)
        # infinite_sheds only accepts 'isotropic' or 'haydavies', so the rear
        # side matches the front only when the front is one of those; every
        # other front model falls back to isotropic rather than failing.
        rear_transposition = "haydavies" if model_options.transposition_model == "haydavies" else "isotropic"
        # Rear ground reflectance must be a number here (infinite_sheds takes
        # no surface_type), so resolve the named type to pvlib's albedo value.
        rear_albedo = (
            float(model_options.albedo)
            if model_options.albedo is not None
            else float(SURFACE_ALBEDOS.get(model_options.surface_type, 0.25))
        )
        # resolve_pv_model_options guarantees the row geometry and bifaciality
        # are set whenever a rear-side model is selected.
        rear = pvlib.bifacial.infinite_sheds.get_irradiance_poa(
            surface_tilt=back_tilt,
            surface_azimuth=back_azimuth,
            solar_zenith=np.asarray(solarpos.apparent_zenith, dtype=float),
            solar_azimuth=np.asarray(solarpos.azimuth, dtype=float),
            gcr=float(model_options.gcr),
            height=float(model_options.pvrow_height),  # type: ignore[arg-type]
            pitch=float(model_options.pvrow_pitch),  # type: ignore[arg-type]
            ghi=np.asarray(ghi, dtype=float),
            dhi=np.asarray(dhi, dtype=float),
            dni=np.asarray(dni, dtype=float),
            albedo=rear_albedo,
            model=rear_transposition,
            dni_extra=np.asarray(dni_extra, dtype=float),
            vectorize=tilt.ndim > 0 and tilt.size > 1,
        )
        rear_poa = np.clip(np.nan_to_num(np.asarray(rear["poa_global"], dtype=float), nan=0.0), 0.0, None)
        rear_effective_irradiance = float(model_options.bifaciality) * rear_poa  # type: ignore[arg-type]
        effective_irradiance = front_effective_irradiance + rear_effective_irradiance
    else:
        rear_effective_irradiance = np.zeros_like(front_effective_irradiance)
        effective_irradiance = front_effective_irradiance

    # Rear irradiance heats the module as well as powering it, so it belongs in
    # the thermal balance too. This matches pvlib's bifacial convention, where
    # infinite_sheds returns poa_global = poa_front + poa_back * bifaciality and
    # that composed series is what feeds the temperature model. Driving the
    # temperature model from the front side alone credits the rear gain with
    # power but no heat and overstates net bifacial gain. On the non-bifacial
    # path rear_effective_irradiance is all zeros, so bifacial_model="none"
    # stays bit-for-bit identical.
    thermal_irradiance = poa_global + rear_effective_irradiance
    temp_cell = calculate_cell_temperature(
        thermal_irradiance,
        temp_air,
        wind_speed,
        model_options.temperature_model,
        module_efficiency=pv_params.Module_Efficiency,
        noct=pv_params.NOCT,
    )
    parts = None
    if isinstance(shading, ModuleMeanShading):
        assert components is not None
        direct, circumsolar, sky_rest, ground = components

        def effective(direct_part, sky_part, ground_part) -> np.ndarray:
            frame = pd.DataFrame(
                {
                    "poa_direct": direct_part,
                    "poa_diffuse": sky_part + ground_part,
                    "poa_sky_diffuse": sky_part,
                    "poa_ground_diffuse": ground_part,
                }
            )
            return calculate_front_effective_irradiance(
                frame, aoi, surface_tilt, model_options.diffuse_iam, model_options.iam_model
            )

        zeros = np.zeros_like(direct)
        parts = IrradianceParts(
            beam=effective(direct, zeros, zeros),
            circumsolar=effective(zeros, circumsolar, zeros),
            sky=effective(zeros, sky_rest, zeros),
            other=effective(zeros, zeros, ground) + rear_effective_irradiance,
            temp_cell=np.asarray(temp_cell, dtype=float),
            sun_azimuth=np.asarray(solarpos.azimuth, dtype=float),
            sun_elevation=90.0 - np.asarray(solarpos.apparent_zenith, dtype=float),
            pv_params=pv_params,
        )
    return _IrradianceModelResult(
        ghi=np.nan_to_num(np.asarray(ghi, dtype=float), nan=0.0),
        poa_global=poa_global,
        front_effective_irradiance=front_effective_irradiance,
        rear_effective_irradiance=rear_effective_irradiance,
        effective_irradiance=effective_irradiance,
        temp_cell=np.asarray(temp_cell, dtype=float),
        poa_global_unshaded=poa_global_unshaded,
        effective_irradiance_unshaded=(None if front_unshaded is None else front_unshaded + rear_effective_irradiance),
        parts=parts,
    )


def _shaded_poa(
    poa: pd.DataFrame,
    shading: ArrayShading,
    solarpos: pd.DataFrame,
    surface_tilt,
    surface_azimuth,
    *,
    dni,
    dhi,
    dni_extra,
    airmass,
    model_options: PVModelOptions,
) -> tuple[pd.DataFrame, tuple[np.ndarray, ...]]:
    """The plane-of-array components after near shading from ``shading``, and the unshaded parts.

    The parts are the direct, circumsolar, other sky diffuse and
    ground-reflected irradiance before shading.
    """
    elevation = 90.0 - np.asarray(solarpos.apparent_zenith, dtype=float)
    beam_shaded = shading.beam_fraction(np.asarray(solarpos.azimuth, dtype=float), elevation)
    direct = np.nan_to_num(poa["poa_direct"].to_numpy(dtype=float), nan=0.0)
    sky = np.nan_to_num(poa["poa_sky_diffuse"].to_numpy(dtype=float), nan=0.0)
    ground = np.nan_to_num(poa["poa_ground_diffuse"].to_numpy(dtype=float), nan=0.0)
    circumsolar = np.zeros_like(sky)
    model = model_options.transposition_model
    if model in CIRCUMSOLAR_MODELS:
        # The same call get_total_irradiance makes, with its parts.
        common = {
            "surface_tilt": surface_tilt,
            "surface_azimuth": surface_azimuth,
            "dhi": dhi,
            "dni": dni,
            "dni_extra": dni_extra,
            "solar_zenith": solarpos.apparent_zenith,
            "solar_azimuth": solarpos.azimuth,
        }
        if model == "haydavies":
            parts = pvlib.irradiance.haydavies(**common, return_components=True)
        elif model == "perez":
            parts = pvlib.irradiance.perez(
                **common, airmass=airmass, model=model_options.model_perez, return_components=True
            )
        else:
            parts = pvlib.irradiance.perez_driesse(**common, airmass=airmass, return_components=True)
        # Not capped at the sky total: Perez's isotropic and horizon terms
        # can be negative, so its circumsolar part can exceed their sum.
        circumsolar = np.clip(np.nan_to_num(np.asarray(parts["poa_circumsolar"], dtype=float), nan=0.0), 0.0, None)
    shaded_direct = direct * (1.0 - beam_shaded)
    shaded_sky = np.clip((sky - circumsolar) * shading.sky_view_factor + circumsolar * (1.0 - beam_shaded), 0.0, None)
    frame = pd.DataFrame(
        {
            "poa_global": shaded_direct + shaded_sky + ground,
            "poa_direct": shaded_direct,
            "poa_diffuse": shaded_sky + ground,
            "poa_sky_diffuse": shaded_sky,
            "poa_ground_diffuse": ground,
        },
        index=poa.index,
    )
    return frame, (direct, circumsolar, sky - circumsolar, ground)


def _get_cec_params(pv_params: PVModuleParams):
    """Fetch (and cache) CEC single-diode model params for a module."""
    key = (
        pv_params.celltype,
        pv_params.Vmp,
        pv_params.Imp,
        pv_params.Voc,
        pv_params.Isc,
        pv_params.alpha_sc,
        pv_params.beta_voc,
        pv_params.gamma_pmp_effective,
        pv_params.cells_in_series,
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
        gamma_pmp=pv_params.gamma_pmp_effective,
        cells_in_series=pv_params.cells_in_series,
    )
    _cec_param_cache[key] = cec
    return cec


def _age_degradation_percent(
    degradation_rate: float = 0.0,
    current_year: Optional[int] = None,
    start_year: Optional[int] = None,
) -> float:
    """Return the age-based PVWatts degradation percentage for ``current_year``.

    Age is counted at the start of the simulated year: the modules are
    ``current_year - start_year`` full years old, so the installation year has
    no age loss and year ``n`` of operation is degraded by ``n - 1`` years.
    Degradation compounds, so the DC output after static losses is scaled by
    ``(1 - degradation_rate) ** (current_year - start_year)``. App, Monte Carlo,
    the optimizer and the economics projection use the same convention.
    Without both years there is no age loss.
    """
    if current_year is None or start_year is None:
        return 0.0
    years_operating = current_year - start_year
    if years_operating < 0:
        raise ValueError(f"current_year ({current_year}) must not be earlier than start_year ({start_year})")
    return float(100 * (1 - (1 - degradation_rate) ** years_operating))


def _module_dc_before_losses(
    effective_irradiance: np.ndarray,
    temp_cell: np.ndarray,
    pv_params: PVModuleParams,
    n_modules: int,
    times: pd.DatetimeIndex,
    name: str,
) -> tuple[pd.Series, ModuleOperatingPoint]:
    """Run the CEC single-diode model before system-level PVWatts losses.

    Returns the array's DC power and one module's operating point.
    """
    I_L_ref, I_o_ref, R_s, R_sh_ref, a_ref, Adjust = _get_cec_params(pv_params)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        cec = pvlib.pvsystem.calcparams_cec(
            effective_irradiance, temp_cell, pv_params.alpha_sc, a_ref, I_L_ref, I_o_ref, R_sh_ref, R_s, Adjust
        )
        cec = (_solver.floor_photocurrent(cec[0], I_L_ref), *cec[1:])
        mpp = _solver.mpp(*cec, stage="module_mpp", steps=times)

    def column(key: str) -> np.ndarray:
        return np.asarray(mpp[key] if isinstance(mpp, dict) else getattr(mpp, key), dtype=float)

    shape = np.shape(column("p_mp"))
    photocurrent, saturation_current, resistance_series, resistance_shunt, n_ns_vth = (
        np.broadcast_to(np.asarray(value, dtype=float), shape) for value in cec
    )
    operating = ModuleOperatingPoint(
        photocurrent=photocurrent,
        saturation_current=saturation_current,
        resistance_series=resistance_series,
        resistance_shunt=resistance_shunt,
        n_ns_vth=n_ns_vth,
        v_mp=column("v_mp"),
        i_mp=column("i_mp"),
    )
    return pd.Series(column("p_mp") * n_modules, index=times, name=name), operating


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


@dataclass(frozen=True, eq=False)
class _PVArray:
    """One array as the PV model sees it: its modules, how it faces the sky, and its model options.

    ``tracking`` is ``"fixed"``, facing ``tilt`` and ``azimuth``, or a
    ``"single_axis"`` or ``"dual_axis"`` tracker with the axis geometry; a
    tracker's row spacing is ``model_kwargs["gcr"]``. ``model_kwargs`` holds
    every key of ``PV_MODEL_CONFIG_KEYS`` but ``solar_position``, which
    belongs to the grid. ``shading`` is the array's near-shading table.
    """

    n_modules: int
    pv_params: PVModuleParams
    model_kwargs: Mapping[str, Any]
    tracking: str = "fixed"
    tilt: float = 35.0
    azimuth: float = 180.0
    axis_tilt: float = 0.0
    axis_azimuth: float = 180.0
    max_angle: float = 60.0
    backtrack: bool = True
    cross_axis_tilt: float = 0.0
    dual_axis_max_tilt: float = 90.0
    shading: Any = None

    def __post_init__(self) -> None:
        if self.tracking not in ("fixed", *_TRACKERS):
            raise ValueError(f"unknown tracking mode {self.tracking!r}. Use 'fixed', 'single_axis', or 'dual_axis'.")

    def orientation(self, solarpos: pd.DataFrame) -> tuple[Any, Any]:
        """Surface tilt and azimuth: scalars for a fixed array, one per step for a tracker."""
        if self.tracking == "fixed":
            return self.tilt, self.azimuth
        if self.tracking == "single_axis":
            tracker = pvlib.tracking.singleaxis(
                apparent_zenith=solarpos.apparent_zenith,
                solar_azimuth=solarpos.azimuth,
                axis_tilt=self.axis_tilt,
                axis_azimuth=self.axis_azimuth,
                max_angle=self.max_angle,
                backtrack=self.backtrack,
                gcr=self.model_kwargs["gcr"],
                cross_axis_tilt=self.cross_axis_tilt,
            )
            # singleaxis returns NaN when sun is below horizon — stow to axis orientation
            return (
                tracker["surface_tilt"].fillna(self.axis_tilt).values,
                tracker["surface_azimuth"].fillna(self.axis_azimuth).values,
            )

        # Dual-axis: panel normal points at sun. Clip below horizon.
        zenith = solarpos.apparent_zenith.to_numpy(dtype=float)
        sun_azimuth = solarpos.azimuth.to_numpy(dtype=float)
        surface_tilt = np.clip(zenith, 0.0, self.dual_axis_max_tilt)
        # When sun is below horizon, stow flat facing south/north (axis_azimuth fallback)
        below_horizon = zenith >= 90.0
        surface_tilt = np.where(below_horizon, 0.0, surface_tilt)
        surface_azimuth = np.where(below_horizon, self.axis_azimuth, sun_azimuth)
        return surface_tilt, surface_azimuth


def _array_breakdown(
    grid: _SolarGrid,
    array: _PVArray,
    *,
    degradation_rate: float = 0.0,
    current_year: Optional[int] = None,
    start_year: Optional[int] = None,
    loss_overrides: Optional[Dict[str, float]] = None,
) -> PVProductionBreakdown:
    """The PV model: one array's production on ``grid``, with every loss stage.

    Every public PV production function, the App and the optimizer compute
    through this. Module age is counted at the start of ``current_year``.
    """
    times, solarpos, weather_aligned = grid
    pv_params = array.pv_params
    surface_tilt, surface_azimuth = array.orientation(solarpos)
    model_options = resolve_pv_model_options(bifaciality=pv_params.bifaciality, **array.model_kwargs)
    detail = _compute_irradiance_and_cell_temp_detail(
        weather_aligned,
        solarpos,
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        pv_params=pv_params,
        model_options=model_options,
        shading=as_shading_table(array.shading),
    )
    n_modules = array.n_modules
    module_dc, module_operating = _module_dc_before_losses(
        detail.effective_irradiance,
        detail.temp_cell,
        pv_params,
        n_modules,
        times,
        name="module_dc_W",
    )
    gamma_per_c = pv_params.gamma_pmp_effective / 100.0
    temperature_factor = 1.0 + gamma_per_c * (detail.temp_cell - 25.0)
    safe_temperature_factor = np.where(np.abs(temperature_factor) > 1e-6, temperature_factor, 1.0)
    effective_irradiance_dc = (module_dc / pd.Series(safe_temperature_factor, index=times)).rename(
        "effective_irradiance_dc_W"
    )
    front_effective_irradiance_dc = _scale_reference_dc(
        effective_irradiance_dc,
        detail.front_effective_irradiance,
        detail.effective_irradiance,
        name="front_effective_irradiance_dc_W",
    )
    rear_gain_dc = (effective_irradiance_dc - front_effective_irradiance_dc).rename("rear_gain_dc_W")
    # The stages before near shading scale from the unshaded irradiance, so a
    # fully shaded step still shows what the shading removed.
    reference_dc, reference_irradiance = effective_irradiance_dc, detail.effective_irradiance
    poa_shaded_dc = None
    if detail.effective_irradiance_unshaded is not None:
        reference_irradiance = detail.effective_irradiance_unshaded
        unshaded_dc, _ = _module_dc_before_losses(
            reference_irradiance, detail.temp_cell, pv_params, n_modules, times, name="module_dc_W"
        )
        reference_dc = unshaded_dc / pd.Series(safe_temperature_factor, index=times)
        poa_shaded_dc = _scale_reference_dc(
            effective_irradiance_dc,
            detail.poa_global,
            detail.effective_irradiance,
            name="poa_shaded_dc_W",
        )
    poa_global_dc = _scale_reference_dc(
        reference_dc,
        detail.poa_global if detail.poa_global_unshaded is None else detail.poa_global_unshaded,
        reference_irradiance,
        name="poa_global_dc_W",
    )
    horizontal_reference_dc = _scale_reference_dc(
        reference_dc,
        detail.ghi,
        reference_irradiance,
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
        front_effective_irradiance_dc=front_effective_irradiance_dc,
        rear_gain_dc=rear_gain_dc,
        effective_irradiance_dc=effective_irradiance_dc,
        module_dc=module_dc,
        dc_after_static_losses=dc_after_static,
        dc_after_losses=dc_after_losses,
        pvwatts_component_losses=component_losses,
        pvwatts_components_pct=loss_info["components_pct"],
        pvwatts_combined_pct=loss_info["combined_pct"],
        age_degradation_pct=age_pct,
        age_degradation_loss=(dc_after_static - dc_after_losses).rename("age_degradation_loss_W"),
        module_operating=module_operating,
        poa_shaded_dc=poa_shaded_dc,
        irradiance_parts=detail.parts,
    )


def _one_array_breakdown(
    weather_data: pd.DataFrame,
    location: Location,
    freq: str,
    model_kwargs: Mapping[str, Any],
    *,
    n_modules: int,
    pv_params: Optional[PVModuleParams],
    degradation_rate: float,
    current_year: Optional[int],
    start_year: Optional[int],
    loss_overrides: Optional[Dict[str, float]],
    **mount: Any,
) -> PVProductionBreakdown:
    """One array's breakdown from a public entry point's arguments.

    ``model_kwargs`` is the model-option block as the entry point received
    it, and ``mount`` the :class:`_PVArray` orientation fields.
    """
    options = dict(model_kwargs)
    grid = _prepare_solarpos_and_weather(weather_data, location, freq, options.pop("solar_position"))
    array = _PVArray(n_modules, pv_params if pv_params is not None else get_module(_DEFAULT_MODULE), options, **mount)
    return _array_breakdown(
        grid,
        array,
        degradation_rate=degradation_rate,
        current_year=current_year,
        start_year=start_year,
        loss_overrides=loss_overrides,
    )


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
    loss_overrides: Optional[Dict[str, float]] = None,
    transposition_model: str = DEFAULT_TRANSPOSITION_MODEL,
    albedo: Optional[float] = None,
    surface_type: Optional[str] = None,
    model_perez: str = DEFAULT_PEREZ_MODEL,
    solar_position: str = DEFAULT_SOLAR_POSITION,
    iam_model: str = DEFAULT_IAM_MODEL,
    diffuse_iam: str = DEFAULT_DIFFUSE_IAM,
    temperature_model: str = DEFAULT_TEMPERATURE_MODEL,
    bifacial_model: str = DEFAULT_BIFACIAL_MODEL,
    gcr: float = 0.35,
    pvrow_height: Optional[float] = None,
    pvrow_pitch: Optional[float] = None,
    shading: Optional[ShadingTable | Dict[str, Any]] = None,
) -> PVProductionBreakdown:
    """Calculate fixed-tilt PV production with intermediate loss stages.

    Module age is counted at the start of ``current_year``: ``current_year -
    start_year`` full years of compound ``degradation_rate``, and none in the
    installation year. See :func:`calculate_pv_production_dc` for the
    parameters. ``shading`` is the array's near-shading table, a
    :class:`breos.pv.shading.ShadingTable` or its mapping; it reduces the
    front plane-of-array irradiance before the incidence-angle modifier.

    .. deprecated:: 0.8.0
       Use :func:`calculate_array_breakdowns` with one array; this function
       will be removed in 0.9.0.
    """
    warnings.warn(
        "breos.solar.calculate_pv_production_breakdown is deprecated and will be removed in BREOS 0.9.0; "
        "use calculate_array_breakdowns with one array",
        DeprecationWarning,
        stacklevel=2,
    )
    return _one_array_breakdown(
        weather_data,
        location,
        freq,
        _model_option_kwargs(locals()),
        n_modules=n_modules,
        pv_params=pv_params,
        degradation_rate=degradation_rate,
        current_year=current_year,
        start_year=start_year,
        loss_overrides=loss_overrides,
        tilt=tilt,
        azimuth=surface_azimuth,
        shading=shading,
    )


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
    loss_overrides: Optional[Dict[str, float]] = None,
    transposition_model: str = DEFAULT_TRANSPOSITION_MODEL,
    albedo: Optional[float] = None,
    surface_type: Optional[str] = None,
    model_perez: str = DEFAULT_PEREZ_MODEL,
    solar_position: str = DEFAULT_SOLAR_POSITION,
    iam_model: str = DEFAULT_IAM_MODEL,
    diffuse_iam: str = DEFAULT_DIFFUSE_IAM,
    temperature_model: str = DEFAULT_TEMPERATURE_MODEL,
    bifacial_model: str = DEFAULT_BIFACIAL_MODEL,
    gcr: float = 0.35,
    pvrow_height: Optional[float] = None,
    pvrow_pitch: Optional[float] = None,
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
        degradation_rate: Annual compound PV degradation rate (0.005 = 0.5%/year).
        current_year: Calendar year being simulated. Module age is counted at
            the start of this year, ``current_year - start_year`` full years,
            and DC output after static losses is scaled by
            ``(1 - degradation_rate) ** age``. ``None`` means no age loss.
        start_year: Installation year, the first year of operation. At
            ``current_year == start_year`` the modules are new and have no age
            loss. ``None`` means no age loss.
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
            evaluated (one of ``SOLAR_POSITION_METHODS``). The default
            ``"auto"`` reads the time basis from the weather metadata, as
            ``"weather"`` does, and takes the sun at the labels, with a
            warning, when the metadata does not state one.
            ``"mid-interval"`` matches PVWatts/SAM for interval-averaged
            irradiance; ``"interval-start"`` reproduces the results of
            BREOS before 0.8.0 bit-for-bit.
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
        bifacial_model: Rear-irradiance model (one of ``BIFACIAL_MODELS``).
            ``"none"`` preserves front-only production; ``"infinite_sheds"``
            requires module bifaciality plus ``gcr``, ``pvrow_height``, and
            ``pvrow_pitch``.

    Returns:
        pd.Series with DC power production in Watts (before inverter)
    """
    return _one_array_breakdown(
        weather_data,
        location,
        freq,
        _model_option_kwargs(locals()),
        n_modules=n_modules,
        pv_params=pv_params,
        degradation_rate=degradation_rate,
        current_year=current_year,
        start_year=start_year,
        loss_overrides=loss_overrides,
        tilt=tilt,
        azimuth=surface_azimuth,
    ).dc_after_losses


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
    loss_overrides: Optional[Dict[str, float]] = None,
    transposition_model: str = DEFAULT_TRANSPOSITION_MODEL,
    albedo: Optional[float] = None,
    surface_type: Optional[str] = None,
    model_perez: str = DEFAULT_PEREZ_MODEL,
    solar_position: str = DEFAULT_SOLAR_POSITION,
    iam_model: str = DEFAULT_IAM_MODEL,
    diffuse_iam: str = DEFAULT_DIFFUSE_IAM,
    temperature_model: str = DEFAULT_TEMPERATURE_MODEL,
    bifacial_model: str = DEFAULT_BIFACIAL_MODEL,
    pvrow_height: Optional[float] = None,
    pvrow_pitch: Optional[float] = None,
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
        degradation_rate: Annual compound PV degradation rate (0.005 = 0.5%/year).
        current_year: Calendar year being simulated. Module age is counted at
            the start of this year, ``current_year - start_year`` full years,
            and DC output after static losses is scaled by
            ``(1 - degradation_rate) ** age``. ``None`` means no age loss.
        start_year: Installation year, the first year of operation. At
            ``current_year == start_year`` the modules are new and have no age
            loss. ``None`` means no age loss.
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
        bifacial_model: Rear-irradiance model (one of ``BIFACIAL_MODELS``).
            ``"infinite_sheds"`` requires module bifaciality and row height/pitch.

    Returns:
        pd.Series with DC power production in Watts (before inverter).
    """
    if tracking not in _TRACKERS:
        raise ValueError(f"tracking must be 'single_axis' or 'dual_axis', got {tracking!r}")
    return _one_array_breakdown(
        weather_data,
        location,
        freq,
        _model_option_kwargs(locals()),
        n_modules=n_modules,
        pv_params=pv_params,
        degradation_rate=degradation_rate,
        current_year=current_year,
        start_year=start_year,
        loss_overrides=loss_overrides,
        tracking=tracking,
        axis_tilt=axis_tilt,
        axis_azimuth=axis_azimuth,
        max_angle=max_angle,
        backtrack=backtrack,
        cross_axis_tilt=cross_axis_tilt,
        dual_axis_max_tilt=dual_axis_max_tilt,
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

    # A missing DC value converts to 0 W, as calculate_dc_ac_power returns for it.
    dc_values = np.nan_to_num(dc_power.to_numpy(dtype=np.float64), nan=0.0, posinf=np.inf, neginf=-np.inf)
    # With no AC rating an infinite DC input gives an infinite AC output, as in
    # the scalar path; only the unused loss array meets inf - inf.
    with np.errstate(invalid="ignore"):
        ac_power, _, _ = _calculate_dc_ac_power_arrays(dc_values, inv_size, inverter_efficiency)

    return pd.Series(ac_power, index=dc_power.index, name="ac_power_W")


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
    transposition_model: str = DEFAULT_TRANSPOSITION_MODEL,
    albedo: Optional[float] = None,
    surface_type: Optional[str] = None,
    model_perez: str = DEFAULT_PEREZ_MODEL,
    solar_position: str = DEFAULT_SOLAR_POSITION,
    iam_model: str = DEFAULT_IAM_MODEL,
    diffuse_iam: str = DEFAULT_DIFFUSE_IAM,
    temperature_model: str = DEFAULT_TEMPERATURE_MODEL,
    bifacial_model: str = DEFAULT_BIFACIAL_MODEL,
    gcr: float = 0.35,
    pvrow_height: Optional[float] = None,
    pvrow_pitch: Optional[float] = None,
    loss_overrides: Optional[Dict[str, float]] = None,
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
        degradation_rate: Annual compound PV degradation rate (0.005 = 0.5%/year).
        current_year: Calendar year being simulated. Module age is counted at
            the start of this year, ``current_year - start_year`` full years,
            and DC output after static losses is scaled by
            ``(1 - degradation_rate) ** age``. ``None`` means no age loss.
        start_year: Installation year, the first year of operation. At
            ``current_year == start_year`` the modules are new and have no age
            loss. ``None`` means no age loss.
        inverter_loading_ratio: DC/AC ratio for inverter sizing
        inverter_efficiency: Nominal inverter efficiency
        loss_overrides: Per-component PVWatts loss overrides (percent), as
            for :func:`calculate_pv_production_dc`

    Returns:
        pd.Series with AC power production in Watts
    """
    model_kwargs = _model_option_kwargs(locals())

    if pv_params is None:
        pv_params = get_module(_DEFAULT_MODULE)

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
        loss_overrides=loss_overrides,
        **model_kwargs,
    )

    pv_peak_power_w = n_modules * pv_params.Mpp
    ac_power = dc_to_ac(dc_power, pv_peak_power_w, inverter_loading_ratio, inverter_efficiency)

    return ac_power


def _extract_irradiance(weather_df: pd.DataFrame):
    """Extract DNI, GHI, DHI under any name in ``IRRADIANCE_COLUMN_ALIASES``."""
    values = []
    for component in ("dni", "ghi", "dhi"):
        column = find_irradiance_column(weather_df.columns, component)
        if column is None:
            raise KeyError(f"Could not find column. Tried: {list(IRRADIANCE_COLUMN_ALIASES[component])}")
        values.append(weather_df[column].values)
    return tuple(values)


def _extract_met_data(weather_df: pd.DataFrame):
    """Extract air temperature and wind speed from a weather DataFrame.

    Both drive the cell temperature, so a missing column or a value that is
    not finite raises instead of defaulting. To run with fixed values, add them as columns, for example
    ``weather["temp_air"] = 25.0`` and ``weather["wind_speed"] = 1.0``.
    """
    met = []
    for key, quantity, example in (
        ("temp_air", "air temperature", 'weather["temp_air"] = 25.0'),
        ("wind_speed", "wind speed", 'weather["wind_speed"] = 1.0'),
    ):
        name = find_met_column(weather_df.columns, key)
        if name is None:
            raise ValueError(
                f"weather has no {quantity} column (tried {list(MET_COLUMN_ALIASES[key])}). The cell "
                f"temperature needs it; to run with a fixed value, add it explicitly, e.g. {example}."
            )
        values = pd.to_numeric(weather_df[name], errors="coerce").to_numpy(dtype=float)
        bad = ~np.isfinite(values)
        if bad.any():
            raise ValueError(
                f"weather column {name!r} has {int(bad.sum())} values that are not finite numbers "
                f"(first at {weather_df.index[int(np.flatnonzero(bad)[0])]}). Fill them explicitly before computing PV."
            )
        met.append(values)
    return tuple(met)


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


def _sum_pv_breakdowns(breakdowns: list[PVProductionBreakdown]) -> PVProductionBreakdown:
    """Sum per-array PVProductionBreakdown objects into one system total."""
    if not breakdowns:
        raise ValueError("At least one PV production breakdown is required")

    def _sum(series: list[pd.Series]) -> pd.Series:
        total = series[0].copy()
        for other in series[1:]:
            total = total.add(other, fill_value=0.0)
        total.name = series[0].name
        return total

    def _sum_attr(name: str) -> pd.Series:
        return _sum([getattr(breakdown, name) for breakdown in breakdowns])

    component_losses = {
        component: _sum([breakdown.pvwatts_component_losses[component] for breakdown in breakdowns])
        for component in breakdowns[0].pvwatts_component_losses
    }

    return PVProductionBreakdown(
        horizontal_reference_dc=_sum_attr("horizontal_reference_dc"),
        poa_global_dc=_sum_attr("poa_global_dc"),
        front_effective_irradiance_dc=_sum_attr("front_effective_irradiance_dc"),
        rear_gain_dc=_sum_attr("rear_gain_dc"),
        effective_irradiance_dc=_sum_attr("effective_irradiance_dc"),
        module_dc=_sum_attr("module_dc"),
        dc_after_static_losses=_sum_attr("dc_after_static_losses"),
        dc_after_losses=_sum_attr("dc_after_losses"),
        pvwatts_component_losses=component_losses,
        pvwatts_components_pct=breakdowns[0].pvwatts_components_pct,
        pvwatts_combined_pct=breakdowns[0].pvwatts_combined_pct,
        age_degradation_pct=breakdowns[0].age_degradation_pct,
        age_degradation_loss=_sum_attr("age_degradation_loss"),
        poa_shaded_dc=(_sum_attr("poa_shaded_dc") if all(b.poa_shaded_dc is not None for b in breakdowns) else None),
    )


def _configured_array(index: int, arr: Mapping[str, Any], location: Location, defaults: Mapping[str, Any]) -> _PVArray:
    """The model of one ``arrays`` entry, as :func:`calculate_multi_array_production` reads it.

    ``defaults`` is the function-level model-option block without
    ``solar_position``; an entry may override the ``PER_ARRAY_MODEL_OPTION_KEYS``.
    """
    options = {
        key: arr.get(key, defaults[key]) if key in PER_ARRAY_MODEL_OPTION_KEYS else defaults[key] for key in defaults
    }
    pv_params = get_module(arr.get("module", _DEFAULT_MODULE))
    tracking = arr.get("tracking", "fixed")
    shading = arr.get("shading")
    if tracking == "fixed":
        return _PVArray(
            arr["modules"],
            pv_params,
            options,
            tilt=arr.get("tilt", 35),
            azimuth=arr.get("azimuth", default_azimuth(location.latitude)),
            shading=shading,
        )
    if tracking in _TRACKERS:
        return _PVArray(
            arr["modules"],
            pv_params,
            options,
            tracking=tracking,
            axis_tilt=arr.get("axis_tilt", 0.0),
            axis_azimuth=arr.get("axis_azimuth", default_azimuth(location.latitude)),
            max_angle=arr.get("max_angle", 60.0),
            backtrack=arr.get("backtrack", True),
            cross_axis_tilt=arr.get("cross_axis_tilt", 0.0),
            dual_axis_max_tilt=arr.get("dual_axis_max_tilt", 90.0),
            shading=shading,
        )
    raise ValueError(
        f"Array {index + 1}: unknown tracking mode {tracking!r}. Use 'fixed', 'single_axis', or 'dual_axis'."
    )


def _breakdowns_on_one_grid(
    weather_data: pd.DataFrame,
    location: Location,
    freq: str,
    solar_position: str,
    models: List[_PVArray],
    **aging: Any,
) -> list[PVProductionBreakdown]:
    """Each array's breakdown, in order, with the sun computed once for all of them."""
    grid = _prepare_solarpos_and_weather(weather_data, location, freq, solar_position)
    return [_array_breakdown(grid, model, **aging) for model in models]


@solar_position_warning_scope()
def calculate_array_breakdowns(
    weather_data: pd.DataFrame,
    location: Location,
    arrays: List[Dict[str, Any]],
    freq: str = "h",
    degradation_rate: float = 0.0,
    current_year: Optional[int] = None,
    start_year: Optional[int] = None,
    loss_overrides: Optional[Dict[str, float]] = None,
    transposition_model: str = DEFAULT_TRANSPOSITION_MODEL,
    albedo: Optional[float] = None,
    surface_type: Optional[str] = None,
    model_perez: str = DEFAULT_PEREZ_MODEL,
    solar_position: str = DEFAULT_SOLAR_POSITION,
    iam_model: str = DEFAULT_IAM_MODEL,
    diffuse_iam: str = DEFAULT_DIFFUSE_IAM,
    temperature_model: str = DEFAULT_TEMPERATURE_MODEL,
    bifacial_model: str = DEFAULT_BIFACIAL_MODEL,
    gcr: float = 0.35,
    pvrow_height: Optional[float] = None,
    pvrow_pitch: Optional[float] = None,
) -> list[PVProductionBreakdown]:
    """Calculate one DC production breakdown per PV array, in the order given.

    Takes the parameters of :func:`calculate_multi_array_production` and
    returns each array's production with its loss stages instead of their sum.
    Each breakdown carries its module operating point, so the list is what
    :func:`breos.electrical.string_limited_dc` and
    :func:`breos.electrical.string_cable_loss` take. Every array must have at
    least one module, so the result lines up with ``arrays``. The sun's
    position is computed once for all arrays.
    """
    options = _model_option_kwargs(locals())
    solar_position = options.pop("solar_position")
    for i, arr in enumerate(arrays):
        if arr.get("modules", 0) < 1:
            raise ValueError(f"Array {i + 1}: modules must be at least 1, got {arr.get('modules', 0)}")
    models = [_configured_array(i, arr, location, options) for i, arr in enumerate(arrays)]
    return _breakdowns_on_one_grid(
        weather_data,
        location,
        freq,
        solar_position,
        models,
        degradation_rate=degradation_rate,
        current_year=current_year,
        start_year=start_year,
        loss_overrides=loss_overrides,
    )


def calculate_multi_array_production(
    weather_data: pd.DataFrame,
    location: Location,
    arrays: List[Dict[str, Any]],
    freq: str = "h",
    degradation_rate: float = 0.0,
    current_year: Optional[int] = None,
    start_year: Optional[int] = None,
    loss_overrides: Optional[Dict[str, float]] = None,
    transposition_model: str = DEFAULT_TRANSPOSITION_MODEL,
    albedo: Optional[float] = None,
    surface_type: Optional[str] = None,
    model_perez: str = DEFAULT_PEREZ_MODEL,
    solar_position: str = DEFAULT_SOLAR_POSITION,
    iam_model: str = DEFAULT_IAM_MODEL,
    diffuse_iam: str = DEFAULT_DIFFUSE_IAM,
    temperature_model: str = DEFAULT_TEMPERATURE_MODEL,
    bifacial_model: str = DEFAULT_BIFACIAL_MODEL,
    gcr: float = 0.35,
    pvrow_height: Optional[float] = None,
    pvrow_pitch: Optional[float] = None,
) -> pd.Series:
    """
    Calculate combined DC production from multiple PV arrays.

    Each array is either fixed-tilt or tracking. Mixed configurations are supported.
    An array with ``modules = 0`` contributes nothing, and a negative module
    count raises ``ValueError``. When every array is empty, the result is zero
    on the same time grid a non-empty array would use. The sun's position is
    computed once for all arrays.

    Args:
        weather_data: DataFrame with weather variables
        location: pvlib Location object
        arrays: List of array dictionaries. Each entry requires ``modules``.
            Common keys include ``module`` and ``tracking``. Fixed-tilt arrays
            use ``tilt`` and ``azimuth``. Single-axis arrays can also set
            ``axis_tilt``, ``axis_azimuth``, ``max_angle``, ``backtrack``,
            ``gcr``, and ``cross_axis_tilt``. Dual-axis arrays can set
            ``dual_axis_max_tilt``. Any array may also set
            ``transposition_model``, ``albedo``/``surface_type``,
            ``model_perez`` or ``temperature_model`` to override the
            function-level defaults, and ``shading``, the array's near-shading
            table, a :class:`breos.pv.shading.ShadingTable` or its mapping; it
            reduces the front plane-of-array irradiance before the
            incidence-angle modifier.
        freq: Time frequency ('h' or '15min')
        degradation_rate: Annual compound PV degradation rate (0.005 = 0.5%/year).
        current_year: Calendar year being simulated. Module age is counted at
            the start of this year, ``current_year - start_year`` full years,
            and DC output after static losses is scaled by
            ``(1 - degradation_rate) ** age``. ``None`` means no age loss.
        start_year: Installation year, the first year of operation. At
            ``current_year == start_year`` the modules are new and have no age
            loss. ``None`` means no age loss.
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
        temperature_model: Default cell-temperature model / mounting preset
            (one of ``TEMPERATURE_MODELS``) for arrays that do not set their
            own. The thermal model encodes the mounting, so it may differ by
            array: a rack and a flush roof on one site run about 10 °C apart.
        bifacial_model: Default rear-irradiance model for arrays that do not
            override it. ``"infinite_sheds"`` requires bifacial module metadata
            plus GCR, row height, and row pitch.

    Returns:
        pd.Series with total DC power (watts)
    """
    options = _model_option_kwargs(locals())
    solar_position = options.pop("solar_position")
    models = []
    for i, arr in enumerate(arrays):
        n_mod = arr.get("modules", 0)
        if n_mod < 0:
            raise ValueError(f"Array {i + 1}: modules must be zero or positive, got {n_mod}")
        # An empty array contributes nothing.
        if n_mod > 0:
            models.append(_configured_array(i, arr, location, options))

    if not models:
        # Zero on the grid a non-empty array is computed on, so an empty
        # system lines up with the rest of the run. The inputs are checked as
        # for one: "weather" still needs its time basis.
        times = _simulation_grid(weather_data, freq, solar_position)
        solar_position_time_offset(solar_position, weather_data, freq)
        resolve_pvwatts_losses(loss_overrides)
        return pd.Series(0.0, index=times, name="dc_power_W")
    breakdowns = _breakdowns_on_one_grid(
        weather_data,
        location,
        freq,
        solar_position,
        models,
        degradation_rate=degradation_rate,
        current_year=current_year,
        start_year=start_year,
        loss_overrides=loss_overrides,
    )
    return _sum_pv_breakdowns(breakdowns).dc_after_losses
