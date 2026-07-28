"""Focused incidence-angle modifier kernels used by :mod:`breos.solar`."""

from __future__ import annotations

import math

import numpy as np
import pvlib

_MARION_DIFFUSE_GRID_STEP_DEG = 0.5
_marion_diffuse_grid_cache: dict[tuple[float, float, float], tuple[np.ndarray, dict[str, np.ndarray]]] = {}


def _marion_diffuse_ashrae(surface_tilt):
    """Return Marion diffuse IAM for Ashrae, interpolating large tilt arrays."""
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


def calculate_front_effective_irradiance(poa, aoi, surface_tilt, diffuse_iam: str) -> np.ndarray:
    """Apply BREOS's historical Ashrae beam and optional Marion diffuse IAM."""
    iam = pvlib.iam.ashrae(aoi)
    poa_direct = np.nan_to_num(poa["poa_direct"].values, nan=0.0)
    poa_diffuse = np.nan_to_num(poa["poa_diffuse"].values, nan=0.0)
    iam_clean = np.nan_to_num(np.asarray(iam, dtype=float), nan=0.0)

    if diffuse_iam == "marion":
        poa_sky = np.nan_to_num(poa["poa_sky_diffuse"].values, nan=0.0)
        poa_ground = np.nan_to_num(poa["poa_ground_diffuse"].values, nan=0.0)
        multipliers = _marion_diffuse_ashrae(surface_tilt)
        sky_mult = np.nan_to_num(np.asarray(multipliers["sky"], dtype=float), nan=0.0)
        ground_mult = np.nan_to_num(np.asarray(multipliers["ground"], dtype=float), nan=0.0)
        return poa_direct * iam_clean + poa_sky * sky_mult + poa_ground * ground_mult

    return poa_direct * iam_clean + poa_diffuse
