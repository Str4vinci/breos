"""Resolved options for BREOS's internal PV model stage.

Every name a user can pass for an irradiance, optics, thermal, or bifacial
choice is enumerated, documented, and validated here, so ``breos.solar`` and
``breos.app_config`` share one definition of what is selectable and what each
selection means. The kernels in this package assume they are handed an
already-resolved :class:`PVModelOptions` and do no validation of their own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from pvlib.albedo import SURFACE_ALBEDOS

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
#
# Applied in breos.solar._prepare_solarpos_and_weather, which shifts the
# solar-position times before transposition, so it is resolved separately
# from PVModelOptions rather than carried on it.
SOLAR_POSITION_METHODS = (
    "interval-start",
    "mid-interval",
)
DEFAULT_SOLAR_POSITION = "interval-start"

# Beam incidence-angle modifier. ``ashrae`` is BREOS's historical model and
# remains the default. The alternatives expose pvlib's physical optics and
# Martin-Ruiz empirical model without adding BREOS-owned fitting parameters.
IAM_MODELS = (
    "ashrae",
    "physical",
    "martin_ruiz",
)
DEFAULT_IAM_MODEL = "ashrae"

# Whether the incidence-angle modifier is applied to the diffuse POA
# components. ``none`` applies IAM to beam only, with diffuse passing at 1.0
# — the default and the only prior behaviour, a known ~0.5-1% systematic
# overestimate. ``marion`` additionally weighs the sky- and ground-diffuse
# components with the selected beam IAM integrated over their view factors
# (Marion 2017, via pvlib's ``iam.marion_diffuse``).
DIFFUSE_IAM_METHODS = (
    "none",
    "marion",
)
DEFAULT_DIFFUSE_IAM = "none"

# Rear-side irradiance is opt-in. ``none`` preserves the historical front-only
# model exactly; ``infinite_sheds`` uses pvlib's row-geometry model for the back
# surface while leaving BREOS's existing front-side transposition unchanged.
BIFACIAL_MODELS = (
    "none",
    "infinite_sheds",
)
DEFAULT_BIFACIAL_MODEL = "none"

# Cell-temperature model and mounting presets. ``faiman`` is pvlib's Faiman
# (2008) model with its open-rack default coefficients (u0=25, u1=6.84) —
# the default and the only prior behaviour. The ``pvsyst-*`` presets use
# PVsyst's documented mounting parameter sets. ``sapm-*`` names retain the
# construction and mounting combinations defined by pvlib/Sandia. ``noct-sam``
# is available only when the selected module carries sourced NOCT and
# efficiency metadata; it never guesses those inputs.
TEMPERATURE_MODELS = (
    "faiman",
    "pvsyst-freestanding",
    "pvsyst-semi-integrated",
    "pvsyst-insulated",
    "sapm-open-rack-glass-glass",
    "sapm-close-mount-glass-glass",
    "sapm-open-rack-glass-polymer",
    "sapm-insulated-back-glass-polymer",
    "noct-sam",
)
DEFAULT_TEMPERATURE_MODEL = "faiman"


# --------------------------------------------------------------------------
# Shared predicates
#
# These carry the *rules* only, never the phrasing. ``breos.app_config`` and
# the resolvers below check the same conditions but must report them
# differently: config validation speaks in config keys ("'pv_arrays[0].gcr'
# must be ...") and treats ``None`` as "not set" for per-array overrides,
# while the resolvers speak in argument names ("gcr must be ...") and treat
# ``None`` as a value. Folding the messages together would flatten one of
# those behaviours, so each caller formats its own error and only the
# predicate is shared. Adding a selectable model in a later slice means
# adding one tuple above and using ``is_known_model`` against it — not a
# third copy of the membership rule.
# --------------------------------------------------------------------------


def normalise_model_name(value) -> str:
    """Canonicalise a user-supplied model name for a membership check.

    Model names are matched case- and whitespace-insensitively, so config
    files and keyword arguments accept ``"Perez"`` and ``" perez "`` alike.
    Non-strings are stringified rather than rejected, which is what makes
    ``resolve_*`` report an unknown-name error (listing the valid names) for
    e.g. an integer instead of a bare ``TypeError``.
    """
    return str(value).strip().lower()


def is_known_model(value, valid: tuple[str, ...]) -> bool:
    """Return whether *value* names one of *valid* after normalisation."""
    return normalise_model_name(value) in valid


def is_valid_albedo(value) -> bool:
    """Return whether *value* is a ground reflectance in [0, 1].

    Deliberately does not type-check: a non-numeric ``value`` raises
    ``TypeError`` from the comparison, which is the long-standing behaviour of
    the keyword-argument path. Callers that owe the user a friendlier message
    (``breos.app_config``) check the type themselves first.
    """
    return 0.0 <= value <= 1.0


def is_valid_gcr(value) -> bool:
    """Return whether *value* is a ground coverage ratio in (0, 1].

    Zero is excluded because a zero-coverage array has no rows to model, and
    values above 1 would mean the modules cover more than the ground beneath
    them. Callers pass an already-finite number.
    """
    return 0.0 < float(value) <= 1.0


@dataclass(frozen=True)
class PVModelOptions:
    """Validated choices consumed by the internal irradiance/PV kernels.

    Build one of these with :func:`resolve_pv_model_options` — never by hand.
    Constructing it directly bypasses every check in this module, and the
    kernels downstream trust their fields without re-validating. Frozen so a
    resolved set cannot drift between the transposition, IAM, thermal, and
    bifacial stages of a single production run.

    ``albedo`` and ``surface_type`` are mutually exclusive and both may be
    ``None``, in which case pvlib's own 0.25 default applies. ``bifaciality``
    comes from the PV module's metadata rather than from user config, and the
    ``gcr``/``pvrow_height``/``pvrow_pitch`` row geometry is only required
    (and only validated) when ``bifacial_model`` is not ``"none"``.
    """

    transposition_model: str
    albedo: float | None
    surface_type: str | None
    model_perez: str
    iam_model: str
    diffuse_iam: str
    temperature_model: str
    bifacial_model: str
    bifaciality: float | None
    gcr: float
    pvrow_height: float | None
    pvrow_pitch: float | None


def resolve_transposition_model(model: str) -> str:
    """Normalise and validate a sky-diffusion transposition model name."""
    if not is_known_model(model, TRANSPOSITION_MODELS):
        valid = ", ".join(TRANSPOSITION_MODELS)
        raise ValueError(f"Unknown transposition model {model!r}. Valid models: {valid}")
    return normalise_model_name(model)


def resolve_solar_position_method(method: str) -> str:
    """Normalise and validate a solar-position evaluation method name."""
    if not is_known_model(method, SOLAR_POSITION_METHODS):
        valid = ", ".join(SOLAR_POSITION_METHODS)
        raise ValueError(f"Unknown solar position method {method!r}. Valid methods: {valid}")
    return normalise_model_name(method)


def resolve_iam_model(model: str) -> str:
    """Normalise and validate a beam incidence-angle modifier model."""
    if not is_known_model(model, IAM_MODELS):
        valid = ", ".join(IAM_MODELS)
        raise ValueError(f"Unknown IAM model {model!r}. Valid models: {valid}")
    return normalise_model_name(model)


def resolve_diffuse_iam_method(method: str) -> str:
    """Normalise and validate a diffuse-IAM method name."""
    if not is_known_model(method, DIFFUSE_IAM_METHODS):
        valid = ", ".join(DIFFUSE_IAM_METHODS)
        raise ValueError(f"Unknown diffuse IAM method {method!r}. Valid methods: {valid}")
    return normalise_model_name(method)


def resolve_temperature_model(model: str) -> str:
    """Normalise and validate a cell-temperature model / mounting preset."""
    if not is_known_model(model, TEMPERATURE_MODELS):
        valid = ", ".join(TEMPERATURE_MODELS)
        raise ValueError(f"Unknown temperature model {model!r}. Valid models: {valid}")
    return normalise_model_name(model)


def resolve_bifacial_model(model: str) -> str:
    """Normalise and validate a rear-irradiance model name."""
    if not is_known_model(model, BIFACIAL_MODELS):
        valid = ", ".join(BIFACIAL_MODELS)
        raise ValueError(f"Unknown bifacial model {model!r}. Valid models: {valid}")
    return normalise_model_name(model)


def validate_bifacial_inputs(
    model: str,
    bifaciality: float | None,
    gcr: float,
    pvrow_height: float | None,
    pvrow_pitch: float | None,
) -> str:
    """Validate opt-in bifacial metadata and row geometry, returning the model.

    Row geometry is only meaningful once a rear-side model is selected, so the
    ``none`` path returns early and leaves ``gcr``/``pvrow_*`` unchecked — the
    fixed-tilt and tracking paths carry their own ``gcr`` default that has
    nothing to do with bifacial modeling. ``bifaciality`` is module metadata,
    not user config: a ``None`` here means the selected PV module was never
    characterised for rear-side gain, which is a configuration error rather
    than a reason to silently fall back to front-only production.
    """
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
    if not is_valid_gcr(gcr):
        raise ValueError("gcr must be between 0 (exclusive) and 1 (inclusive) for bifacial modeling")
    if float(pvrow_height) <= 0.0:
        raise ValueError("pvrow_height must be > 0 for bifacial modeling")
    if float(pvrow_pitch) <= 0.0:
        raise ValueError("pvrow_pitch must be > 0 for bifacial modeling")
    return model


def resolve_perez_model(model_perez: str) -> str:
    """Validate the Perez coefficient set name."""
    if not is_known_model(model_perez, PEREZ_MODELS):
        valid = ", ".join(PEREZ_MODELS)
        raise ValueError(f"Unknown Perez coefficient model {model_perez!r}. Valid models: {valid}")
    return normalise_model_name(model_perez)


def resolve_ground_reflectance(albedo, surface_type):
    """Validate the ground-reflectance inputs and return ``(albedo, surface_type)``.

    Accepts either a numeric ``albedo`` (0-1) or a named ``surface_type`` from
    ``SURFACE_TYPES`` (which pvlib maps to an albedo), but not both. Both may
    be ``None``; the pair is passed through unchanged for the transposition
    call to turn into pvlib's ``albedo``/``surface_type`` keyword.
    """
    if albedo is not None and surface_type is not None:
        raise ValueError("Set either 'albedo' or 'surface_type', not both.")
    if surface_type is not None and surface_type not in SURFACE_ALBEDOS:
        valid = ", ".join(SURFACE_TYPES)
        raise ValueError(f"Unknown surface_type {surface_type!r}. Valid types: {valid}")
    if albedo is not None and not is_valid_albedo(albedo):
        raise ValueError(f"albedo must be between 0 and 1, got {albedo!r}")
    return albedo, surface_type


def resolve_pv_model_options(
    *,
    transposition_model: str = DEFAULT_TRANSPOSITION_MODEL,
    albedo: float | None = None,
    surface_type: str | None = None,
    model_perez: str = DEFAULT_PEREZ_MODEL,
    iam_model: str = DEFAULT_IAM_MODEL,
    diffuse_iam: str = DEFAULT_DIFFUSE_IAM,
    temperature_model: str = DEFAULT_TEMPERATURE_MODEL,
    bifacial_model: str = DEFAULT_BIFACIAL_MODEL,
    bifaciality: float | None = None,
    gcr: float = 0.35,
    pvrow_height: float | None = None,
    pvrow_pitch: float | None = None,
) -> PVModelOptions:
    """Resolve and validate one complete set of PV-kernel choices.

    The only supported way to build a :class:`PVModelOptions`. Call it once
    per production run, before any irradiance work, so a bad model name fails
    before the expensive transposition rather than midway through it.

    Raises ``ValueError`` (or ``TypeError`` for non-numeric bifacial geometry)
    naming the offending option and its valid values.
    """
    transposition_model = resolve_transposition_model(transposition_model)
    albedo, surface_type = resolve_ground_reflectance(albedo, surface_type)
    model_perez = resolve_perez_model(model_perez)
    iam_model = resolve_iam_model(iam_model)
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
        iam_model=iam_model,
        diffuse_iam=diffuse_iam,
        temperature_model=temperature_model,
        bifacial_model=bifacial_model,
        bifaciality=bifaciality,
        gcr=gcr,
        pvrow_height=pvrow_height,
        pvrow_pitch=pvrow_pitch,
    )
