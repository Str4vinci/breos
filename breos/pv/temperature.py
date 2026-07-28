"""Focused cell-temperature kernels used by :mod:`breos.solar`."""

from __future__ import annotations

import numpy as np
import pvlib

_PVSYST_MOUNTING = {
    "pvsyst-freestanding": "freestanding",
    "pvsyst-semi-integrated": "semi_integrated",
    "pvsyst-insulated": "insulated",
}


def calculate_cell_temperature(poa_global, temp_air, wind_speed, temperature_model: str) -> np.ndarray:
    """Run one already-resolved BREOS cell-temperature model."""
    if temperature_model == "faiman":
        return pvlib.temperature.faiman(poa_global, temp_air, wind_speed)

    params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["pvsyst"][_PVSYST_MOUNTING[temperature_model]]
    return pvlib.temperature.pvsyst_cell(poa_global, temp_air, wind_speed, **params)
