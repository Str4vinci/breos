"""Focused incidence-angle modifier kernels used by :mod:`breos.solar`.

The beam IAM is always pvlib's ``ashrae``, which is BREOS's historical
behaviour; the diffuse components are only weighted when the caller opted in
via ``diffuse_iam="marion"``. Inputs are assumed already validated by
:mod:`breos.pv.model_options`.
"""

from __future__ import annotations

import math

import numpy as np
import pvlib

# Tilt resolution of the cached Marion multiplier grid. The integrated sky and
# ground multipliers vary smoothly and slowly over tilt, so 0.5 degrees is far
# finer than the ~0.5-1% effect the diffuse IAM itself corrects for.
_MARION_DIFFUSE_GRID_STEP_DEG = 0.5

# (lo, hi, step) -> (tilt grid, {"sky"/"ground": multipliers}). Keyed by the
# rounded-out tilt span so repeated years or Monte Carlo runs over the same
# tracker geometry reuse one grid. Unbounded, but each entry is a few thousand
# floats and the number of distinct spans in a session is tiny.
_marion_diffuse_grid_cache: dict[tuple[float, float, float], tuple[np.ndarray, dict[str, np.ndarray]]] = {}


def _marion_diffuse_ashrae(surface_tilt):
    """Return Marion diffuse IAM for ashrae, interpolating large tilt arrays.

    pvlib's exact Marion integration is fast for fixed tilt but expensive for
    tracker arrays with thousands of distinct angles. The integrated
    sky/ground multipliers are smooth over tilt, so a cached 0.5 degree grid
    keeps tracker runs tractable without changing the scalar fixed-tilt path.
    """
    tilt_array = np.asarray(surface_tilt, dtype=float)
    # Small inputs (scalar fixed tilt, or a handful of angles) go straight to
    # pvlib so the common case stays bit-for-bit exact.
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
    # NaN tilts (below-horizon tracker steps) are pinned to the grid floor;
    # their POA components are zeroed by the caller, so the value is unused.
    interp_tilt = np.nan_to_num(tilt_array, nan=lo)
    return {region: np.interp(interp_tilt, grid, region_values) for region, region_values in values.items()}


def calculate_front_effective_irradiance(poa, aoi, surface_tilt, diffuse_iam: str) -> np.ndarray:
    """Apply the ashrae beam IAM and, optionally, the Marion diffuse IAM.

    ``poa`` is the frame returned by ``pvlib.irradiance.get_total_irradiance``
    and ``diffuse_iam`` must already be one of ``DIFFUSE_IAM_METHODS``. Returns
    front-side effective irradiance in W/m2, with NaNs zeroed so downstream
    single-diode and thermal calls never see them.
    """
    iam = pvlib.iam.ashrae(aoi)
    poa_direct = np.nan_to_num(poa["poa_direct"].values, nan=0.0)
    poa_diffuse = np.nan_to_num(poa["poa_diffuse"].values, nan=0.0)
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
        return poa_direct * iam_clean + poa_sky * sky_mult + poa_ground * ground_mult

    return poa_direct * iam_clean + poa_diffuse
