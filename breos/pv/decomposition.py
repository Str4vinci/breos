"""Rebuild DNI and DHI from GHI-only weather (irradiance decomposition).

BREOS's PV model needs all three irradiance components. Weather that carries
only global horizontal irradiance (a pyranometer, an inverter logger, some
APIs) can still be run by estimating the beam/diffuse split with one of
pvlib's decomposition models. This is opt-in and fills a gap only: weather
that already has DNI and DHI is returned unchanged, so a configured model
never overrides a measured or provider-supplied split.

The sun position comes from
:func:`breos.pv.model_options.solar_position_at_labels`, the same evaluation
the transposition and terrain-shading stages use, so the three agree on where
the sun is within each step.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import pvlib
from pvlib.location import Location

from breos.pv.model_options import (
    DEFAULT_SOLAR_POSITION,
    is_known_model,
    normalise_model_name,
    solar_position_at_labels,
)
from breos.utils import find_irradiance_column
from breos.weather import WEATHER_METADATA_KEY

# pvlib GHI -> DNI/DHI decomposition models. ``erbs_driesse`` is the
# recommended choice: Erbs's widely validated diffuse-fraction correlation,
# reformulated by Driesse et al. (2024) to be continuous, without the
# breakpoints that put kinks into the original. ``dirint`` and ``disc`` are
# the NREL quasi-physical models; ``dirindex`` adds a clear-sky reference;
# ``boland``, ``orgill_hollands`` and ``louche`` are further diffuse-fraction
# correlations.
DECOMPOSITION_MODELS = (
    "erbs_driesse",
    "erbs",
    "dirint",
    "disc",
    "dirindex",
    "boland",
    "orgill_hollands",
    "louche",
)
DEFAULT_DECOMPOSITION_MODEL = "erbs_driesse"

# Models that estimate the diffuse fraction; the rest estimate DNI directly.
# Either way BREOS keeps only the DNI and takes DHI from closure,
# GHI = DNI cos(zenith) + DHI, so every model yields a split that adds up.
_DIFFUSE_FRACTION_MODELS = ("erbs_driesse", "erbs", "boland", "orgill_hollands", "louche")


def resolve_decomposition_model(model: str) -> str:
    """Normalise and validate a decomposition model name."""
    if not is_known_model(model, DECOMPOSITION_MODELS):
        valid = ", ".join(DECOMPOSITION_MODELS)
        raise ValueError(f"Unknown irradiance decomposition model {model!r}. Valid models: {valid}")
    return normalise_model_name(model)


def _weather_step(index: pd.DatetimeIndex) -> str:
    """The weather's own step as a frequency string, for the sun-position offset."""
    if len(index) < 2:
        raise ValueError("irradiance decomposition needs at least two timestamped weather rows")
    return pd.tseries.frequencies.to_offset(index[1] - index[0]).freqstr


def _dni(model: str, ghi: pd.Series, sun: pd.DataFrame, location: Location) -> pd.Series:
    """Run one pvlib model and return its DNI estimate."""
    times = ghi.index
    zenith = sun["zenith"]  # pvlib's decomposition models take the true (unrefracted) zenith
    if model in _DIFFUSE_FRACTION_MODELS:
        return getattr(pvlib.irradiance, model)(ghi, zenith, times)["dni"]
    # Pressure from the site altitude rather than a weather column: weather
    # sources disagree on its unit (Pa vs hPa), and these models are only
    # weakly sensitive to it.
    pressure = pvlib.atmosphere.alt2pres(location.altitude)
    if model == "disc":
        return pvlib.irradiance.disc(ghi, zenith, times, pressure=pressure)["dni"]
    if model == "dirint":
        return pvlib.irradiance.dirint(ghi, zenith, times, pressure=pressure)
    # dirindex scales DIRINT by a clear-sky reference.
    clearsky = location.get_clearsky(times, model="ineichen", solar_position=sun)
    return pvlib.irradiance.dirindex(ghi, clearsky["ghi"], clearsky["dni"], zenith, times, pressure=pressure)


def decompose_ghi(
    weather: pd.DataFrame,
    location: Location,
    model: str = DEFAULT_DECOMPOSITION_MODEL,
    *,
    solar_position: str = DEFAULT_SOLAR_POSITION,
) -> pd.DataFrame:
    """Return ``weather`` with DNI and DHI estimated from GHI, when both are missing.

    Weather that already holds DNI and DHI (under any name in
    :data:`breos.utils.IRRADIANCE_COLUMN_ALIASES`) is returned unchanged:
    the model fills a gap and never replaces a supplied split. Weather with
    only one of the two is refused rather than half-modelled, as is weather
    without GHI.

    The estimate is written to ``dni`` and ``dhi`` columns of a copy, both
    non-negative and zero at night, and the copy's weather metadata records
    the model under ``"irradiance_decomposition"``. ``solar_position`` must
    match the PV model's setting so the decomposition and the transposition
    use the same sun.

    Args:
        weather: Weather with a ``DatetimeIndex`` at an even step and a GHI
            column (``ghi``, ``shortwave_radiation`` or
            ``global_horizontal_irradiance``).
        location: Site location; its altitude sets the air pressure used by
            ``disc``, ``dirint`` and ``dirindex``.
        model: One of :data:`DECOMPOSITION_MODELS`.
        solar_position: One of
            :data:`breos.pv.model_options.SOLAR_POSITION_METHODS`.

    Returns:
        The weather, with ``dni`` and ``dhi`` added when they were missing.
    """
    model = resolve_decomposition_model(model)
    columns = {component: find_irradiance_column(weather.columns, component) for component in ("ghi", "dni", "dhi")}
    if columns["dni"] is not None and columns["dhi"] is not None:
        return weather
    if columns["dni"] is not None or columns["dhi"] is not None:
        present = "DNI" if columns["dni"] is not None else "DHI"
        raise ValueError(
            f"Weather has GHI and {present} but not the other component. Irradiance decomposition only fills "
            "weather that has GHI alone; derive the missing component before running BREOS."
        )
    if columns["ghi"] is None:
        raise ValueError(
            "Irradiance decomposition needs a GHI column (ghi, shortwave_radiation or global_horizontal_irradiance)"
        )
    if not isinstance(weather.index, pd.DatetimeIndex):
        raise ValueError("Irradiance decomposition needs weather with a DatetimeIndex")

    index = weather.index
    sun, method = solar_position_at_labels(location, index, weather, _weather_step(index), solar_position)
    zenith = sun["zenith"]
    ghi = weather[columns["ghi"]].astype(float).clip(lower=0.0)

    dni_values = np.asarray(_dni(model, ghi, sun, location), dtype=float)
    ghi_values = ghi.to_numpy(dtype=float)
    cos_zenith = np.cos(np.radians(zenith.to_numpy(dtype=float)))
    # No beam with the sun at or below the horizon, and no non-finite values:
    # dirindex divides by a clear-sky DIRINT that reaches zero at low sun.
    daylight = cos_zenith > 0.0
    dni_values = np.where(daylight & np.isfinite(dni_values), np.maximum(dni_values, 0.0), 0.0)
    # Cap the horizontal beam at GHI so the diffuse part is never negative,
    # then take DHI by closure.
    beam_horizontal = np.minimum(dni_values * np.where(daylight, cos_zenith, 0.0), np.nan_to_num(ghi_values))
    dni_values = np.divide(beam_horizontal, cos_zenith, out=np.zeros_like(dni_values), where=daylight)
    dhi_values = ghi_values - beam_horizontal
    dni = pd.Series(dni_values, index=index)
    dhi = pd.Series(np.maximum(np.nan_to_num(dhi_values, nan=0.0), 0.0), index=index)
    # A missing GHI stays missing rather than becoming a zero split.
    missing = ghi.isna()
    dni[missing] = np.nan
    dhi[missing] = np.nan

    result = weather.copy()
    result["dni"] = dni
    result["dhi"] = dhi
    existing = weather.attrs.get(WEATHER_METADATA_KEY)
    metadata: dict[str, Any] = deepcopy(existing) if isinstance(existing, dict) else {}
    metadata["irradiance_decomposition"] = {
        "model": model,
        "derived_variables": ["dni", "dhi"],
        "source_variable": str(columns["ghi"]),
        "solar_position": method,
        "pvlib_version": pvlib.__version__,
    }
    result.attrs = deepcopy(weather.attrs)
    result.attrs[WEATHER_METADATA_KEY] = metadata
    return result
