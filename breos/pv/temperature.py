"""Focused cell-temperature kernels used by :mod:`breos.solar`.

Both models are driven by plane-of-array irradiance (pre-IAM), which is what
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


def calculate_cell_temperature(poa_global, temp_air, wind_speed, temperature_model: str) -> np.ndarray:
    """Run one already-resolved BREOS cell-temperature model.

    ``temperature_model`` must already be one of ``TEMPERATURE_MODELS``; an
    unrecognised name raises ``KeyError`` here rather than being validated,
    because :func:`breos.pv.model_options.resolve_temperature_model` owns that
    check. The pvsyst presets deliberately take pvlib's default
    ``module_efficiency``/``alpha_absorption`` — feeding the catalog's real
    module efficiency in is a separate, behaviour-changing 0.5.0 slice.
    """
    if temperature_model == "faiman":
        # pvlib's open-rack defaults (u0=25, u1=6.84); BREOS's historical path.
        return pvlib.temperature.faiman(poa_global, temp_air, wind_speed)

    params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["pvsyst"][_PVSYST_MOUNTING[temperature_model]]
    return pvlib.temperature.pvsyst_cell(poa_global, temp_air, wind_speed, **params)
