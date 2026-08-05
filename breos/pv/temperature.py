"""Focused cell-temperature kernels used by :mod:`breos.solar`.

The models are driven by plane-of-array irradiance (pre-IAM), which is what
pvlib's thermal models expect: they describe the heat actually absorbed by the
module, not the fraction of it that reaches the cell as usable photons. For a
bifacial system the caller composes front-side ``poa_global`` with the
bifaciality-weighted rear irradiance first, so rear gain contributes heat as
well as power; a front-only system passes ``poa_global`` unchanged.
"""

from __future__ import annotations

import numpy as np
import pvlib

# pvsyst-* preset -> key into pvlib's TEMPERATURE_MODEL_PARAMETERS["pvsyst"].
_PVSYST_MOUNTING = {
    "pvsyst-freestanding": "freestanding",
    "pvsyst-semi-integrated": "semi_integrated",
    "pvsyst-insulated": "insulated",
}

# sapm-* preset -> key into pvlib's TEMPERATURE_MODEL_PARAMETERS["sapm"].
# The names intentionally retain the construction detail so callers do not
# mistake the glass/glass and glass/polymer coefficients for equivalents.
_SAPM_MOUNTING = {
    "sapm-open-rack-glass-glass": "open_rack_glass_glass",
    "sapm-close-mount-glass-glass": "close_mount_glass_glass",
    "sapm-open-rack-glass-polymer": "open_rack_glass_polymer",
    "sapm-insulated-back-glass-polymer": "insulated_back_glass_polymer",
}

# Module efficiency assumed by the PVsyst heat balance when a module carries no
# sourced ``Module_Efficiency``.
#
# This is a physical quantity, not a tuning constant: pvlib defines it as
# "DC power / (POA irradiance x module area)", and it sets the share of absorbed
# energy that leaves as electricity instead of heat. PVsyst's own U-values --
# the ones pvlib ships in TEMPERATURE_MODEL_PARAMETERS["pvsyst"] -- were fitted
# against the real per-timestep efficiency, so supplying a realistic value is
# what agrees with that calibration.
#
# pvlib's own default is 0.1, a legacy placeholder that claims 90% of absorbed
# energy becomes heat. No crystalline-silicon module has been near that in
# decades, and inheriting it silently biases every pvsyst-* run cool-side-down
# by roughly 2.5 C at 800 W/m2 (~0.9% DC). 0.20 sits mid-band for modern c-Si
# (~19-22%); anywhere in that band moves cell temperature by at most ~0.5 C
# (~0.2% DC), so the exact figure matters far less than not using 0.1.
#
# Modules that do carry sourced metadata use it instead -- this only covers
# catalog entries and user-supplied PVModuleParams that never specified one.
DEFAULT_MODULE_EFFICIENCY = 0.20


def _valid_efficiency(module_efficiency) -> bool:
    """Return whether a supplied module efficiency is a usable fraction."""
    return (
        not isinstance(module_efficiency, (bool, np.bool_))
        and isinstance(module_efficiency, (int, float, np.number))
        and np.isfinite(module_efficiency)
        and 0 < module_efficiency <= 1
    )


def _valid_noct(noct) -> bool:
    """Return whether a supplied datasheet NOCT is a plausible finite °C value."""
    return (
        not isinstance(noct, (bool, np.bool_))
        and isinstance(noct, (int, float, np.number))
        and np.isfinite(noct)
        and 0 < noct <= 100
    )


def validate_temperature_inputs(temperature_model: str, module_efficiency=None, noct=None) -> None:
    """Validate model-specific thermal metadata without supplying defaults.

    PVsyst tolerates a missing module efficiency because a representative c-Si
    value is defensible for it (see :data:`DEFAULT_MODULE_EFFICIENCY`) -- but a
    *supplied* value must still be physical. SAM's NOCT model has no such
    fallback: NOCT varies far too much between constructions to stand in for,
    so both it and an efficiency fraction are required before it can run.
    """
    if (
        temperature_model in _PVSYST_MOUNTING
        and module_efficiency is not None
        and not _valid_efficiency(module_efficiency)
    ):
        raise ValueError(
            "PV module Module_Efficiency must be a finite fraction in (0, 1] when supplied to a PVsyst temperature model"
        )
    if temperature_model != "noct-sam":
        return
    if module_efficiency is None:
        raise ValueError("temperature_model='noct-sam' requires PV module Module_Efficiency metadata")
    if not _valid_efficiency(module_efficiency):
        raise ValueError(
            "PV module Module_Efficiency must be a finite fraction in (0, 1] for temperature_model='noct-sam'"
        )
    if noct is None:
        raise ValueError("temperature_model='noct-sam' requires PV module NOCT metadata")
    if not _valid_noct(noct):
        raise ValueError("PV module NOCT must be a finite temperature in (0, 100] °C for temperature_model='noct-sam'")


def calculate_cell_temperature(
    poa_global,
    temp_air,
    wind_speed,
    temperature_model: str,
    *,
    module_efficiency=None,
    noct=None,
) -> np.ndarray:
    """Run one already-resolved BREOS cell-temperature model.

    ``temperature_model`` must already be one of ``TEMPERATURE_MODELS``; an
    unrecognised name raises ``KeyError`` here rather than being validated,
    because :func:`breos.pv.model_options.resolve_temperature_model` owns that
    check. PVsyst's heat balance uses a sourced ``module_efficiency`` when the
    module carries one and :data:`DEFAULT_MODULE_EFFICIENCY` otherwise, so every
    module is modelled with a realistic conversion efficiency rather than
    inheriting pvlib's legacy 0.1 for whichever catalog entries happen to lack
    the metadata. ``noct-sam`` refuses to run unless both of its datasheet
    inputs are present, rather than inventing thermal metadata.
    """
    if temperature_model == "faiman":
        # pvlib's open-rack defaults (u0=25, u1=6.84); BREOS's historical path.
        return pvlib.temperature.faiman(poa_global, temp_air, wind_speed)

    if temperature_model in _PVSYST_MOUNTING:
        validate_temperature_inputs(temperature_model, module_efficiency, noct)
        params = dict(pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["pvsyst"][_PVSYST_MOUNTING[temperature_model]])
        # Always forward an explicit efficiency. Omitting the keyword would fall
        # through to pvlib's 0.1, which would model two catalog modules under
        # visibly different physics for the same preset purely by which one has
        # datasheet metadata.
        params["module_efficiency"] = module_efficiency if module_efficiency is not None else DEFAULT_MODULE_EFFICIENCY
        return pvlib.temperature.pvsyst_cell(poa_global, temp_air, wind_speed, **params)

    if temperature_model in _SAPM_MOUNTING:
        params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"][_SAPM_MOUNTING[temperature_model]]
        return pvlib.temperature.sapm_cell(poa_global, temp_air, wind_speed, **params)

    validate_temperature_inputs(temperature_model, module_efficiency, noct)
    return pvlib.temperature.noct_sam(
        poa_global,
        temp_air,
        wind_speed,
        noct=noct,
        module_efficiency=module_efficiency,
    )
