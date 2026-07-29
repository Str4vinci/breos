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

    PVsyst accepts missing module efficiency for backwards compatibility with
    its previous pvlib-default path, but uses a supplied catalog value. SAM's
    NOCT model has no defensible fallback: both a datasheet NOCT and a module
    efficiency fraction are required before it can run.
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
    check. A sourced ``module_efficiency`` is passed to PVsyst's heat-balance
    model; when metadata is absent the historical pvlib-default path is kept.
    ``noct-sam`` refuses to run unless both required datasheet inputs are
    present, rather than inventing thermal metadata.
    """
    if temperature_model == "faiman":
        # pvlib's open-rack defaults (u0=25, u1=6.84); BREOS's historical path.
        return pvlib.temperature.faiman(poa_global, temp_air, wind_speed)

    if temperature_model in _PVSYST_MOUNTING:
        validate_temperature_inputs(temperature_model, module_efficiency, noct)
        params = dict(pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["pvsyst"][_PVSYST_MOUNTING[temperature_model]])
        # Only forward a sourced value. Leaving the keyword out preserves
        # pvlib's documented 0.1 default for legacy/catalog entries without
        # module-efficiency metadata.
        if module_efficiency is not None:
            params["module_efficiency"] = module_efficiency
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
