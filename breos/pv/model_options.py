"""Resolved options for BREOS's internal PV model stage."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from pvlib.albedo import SURFACE_ALBEDOS

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

SURFACE_TYPES = tuple(sorted(SURFACE_ALBEDOS))

SOLAR_POSITION_METHODS = (
    "interval-start",
    "mid-interval",
)
DEFAULT_SOLAR_POSITION = "interval-start"

DIFFUSE_IAM_METHODS = (
    "none",
    "marion",
)
DEFAULT_DIFFUSE_IAM = "none"

BIFACIAL_MODELS = (
    "none",
    "infinite_sheds",
)
DEFAULT_BIFACIAL_MODEL = "none"

TEMPERATURE_MODELS = (
    "faiman",
    "pvsyst-freestanding",
    "pvsyst-semi-integrated",
    "pvsyst-insulated",
)
DEFAULT_TEMPERATURE_MODEL = "faiman"


@dataclass(frozen=True)
class PVModelOptions:
    """Validated choices consumed by the internal irradiance/PV kernels."""

    transposition_model: str
    albedo: float | None
    surface_type: str | None
    model_perez: str
    diffuse_iam: str
    temperature_model: str
    bifacial_model: str
    bifaciality: float | None
    gcr: float
    pvrow_height: float | None
    pvrow_pitch: float | None


def resolve_transposition_model(model: str) -> str:
    """Normalise and validate a sky-diffusion transposition model name."""
    normalised = str(model).strip().lower()
    if normalised not in TRANSPOSITION_MODELS:
        valid = ", ".join(TRANSPOSITION_MODELS)
        raise ValueError(f"Unknown transposition model {model!r}. Valid models: {valid}")
    return normalised


def resolve_solar_position_method(method: str) -> str:
    """Normalise and validate a solar-position evaluation method name."""
    normalised = str(method).strip().lower()
    if normalised not in SOLAR_POSITION_METHODS:
        valid = ", ".join(SOLAR_POSITION_METHODS)
        raise ValueError(f"Unknown solar position method {method!r}. Valid methods: {valid}")
    return normalised


def resolve_diffuse_iam_method(method: str) -> str:
    """Normalise and validate a diffuse-IAM method name."""
    normalised = str(method).strip().lower()
    if normalised not in DIFFUSE_IAM_METHODS:
        valid = ", ".join(DIFFUSE_IAM_METHODS)
        raise ValueError(f"Unknown diffuse IAM method {method!r}. Valid methods: {valid}")
    return normalised


def resolve_temperature_model(model: str) -> str:
    """Normalise and validate a cell-temperature model / mounting preset."""
    normalised = str(model).strip().lower()
    if normalised not in TEMPERATURE_MODELS:
        valid = ", ".join(TEMPERATURE_MODELS)
        raise ValueError(f"Unknown temperature model {model!r}. Valid models: {valid}")
    return normalised


def resolve_bifacial_model(model: str) -> str:
    """Normalise and validate a rear-irradiance model name."""
    normalised = str(model).strip().lower()
    if normalised not in BIFACIAL_MODELS:
        valid = ", ".join(BIFACIAL_MODELS)
        raise ValueError(f"Unknown bifacial model {model!r}. Valid models: {valid}")
    return normalised


def validate_bifacial_inputs(
    model: str,
    bifaciality: float | None,
    gcr: float,
    pvrow_height: float | None,
    pvrow_pitch: float | None,
) -> str:
    """Validate opt-in bifacial metadata and row geometry."""
    model = resolve_bifacial_model(model)
    if model == "none":
        return model
    if bifaciality is None:
        raise ValueError("bifacial_model='infinite_sheds' requires PV module bifaciality metadata")

    geometry = {
        "gcr": gcr,
        "pvrow_height": pvrow_height,
        "pvrow_pitch": pvrow_pitch,
    }
    for name, value in geometry.items():
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise TypeError(f"{name} must be a finite number for bifacial modeling")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a finite number for bifacial modeling")
    if not 0.0 < float(gcr) <= 1.0:
        raise ValueError("gcr must be between 0 (exclusive) and 1 (inclusive) for bifacial modeling")
    if float(pvrow_height) <= 0.0:
        raise ValueError("pvrow_height must be > 0 for bifacial modeling")
    if float(pvrow_pitch) <= 0.0:
        raise ValueError("pvrow_pitch must be > 0 for bifacial modeling")
    return model


def resolve_perez_model(model_perez: str) -> str:
    """Validate the Perez coefficient set name."""
    if model_perez not in PEREZ_MODELS:
        valid = ", ".join(PEREZ_MODELS)
        raise ValueError(f"Unknown Perez coefficient model {model_perez!r}. Valid models: {valid}")
    return model_perez


def resolve_ground_reflectance(albedo, surface_type):
    """Validate ground-reflectance inputs and return their resolved pair."""
    if albedo is not None and surface_type is not None:
        raise ValueError("Set either 'albedo' or 'surface_type', not both.")
    if surface_type is not None and surface_type not in SURFACE_ALBEDOS:
        valid = ", ".join(SURFACE_TYPES)
        raise ValueError(f"Unknown surface_type {surface_type!r}. Valid types: {valid}")
    if albedo is not None and not 0.0 <= albedo <= 1.0:
        raise ValueError(f"albedo must be between 0 and 1, got {albedo!r}")
    return albedo, surface_type


def resolve_pv_model_options(
    *,
    transposition_model: str = DEFAULT_TRANSPOSITION_MODEL,
    albedo: float | None = None,
    surface_type: str | None = None,
    model_perez: str = DEFAULT_PEREZ_MODEL,
    diffuse_iam: str = DEFAULT_DIFFUSE_IAM,
    temperature_model: str = DEFAULT_TEMPERATURE_MODEL,
    bifacial_model: str = DEFAULT_BIFACIAL_MODEL,
    bifaciality: float | None = None,
    gcr: float = 0.35,
    pvrow_height: float | None = None,
    pvrow_pitch: float | None = None,
) -> PVModelOptions:
    """Resolve and validate one complete set of PV-kernel choices."""
    transposition_model = resolve_transposition_model(transposition_model)
    albedo, surface_type = resolve_ground_reflectance(albedo, surface_type)
    model_perez = resolve_perez_model(model_perez)
    diffuse_iam = resolve_diffuse_iam_method(diffuse_iam)
    temperature_model = resolve_temperature_model(temperature_model)
    bifacial_model = validate_bifacial_inputs(
        bifacial_model,
        bifaciality,
        gcr,
        pvrow_height,
        pvrow_pitch,
    )
    return PVModelOptions(
        transposition_model=transposition_model,
        albedo=albedo,
        surface_type=surface_type,
        model_perez=model_perez,
        diffuse_iam=diffuse_iam,
        temperature_model=temperature_model,
        bifacial_model=bifacial_model,
        bifaciality=bifaciality,
        gcr=gcr,
        pvrow_height=pvrow_height,
        pvrow_pitch=pvrow_pitch,
    )
