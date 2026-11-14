"""Resolved options for BREOS's internal PV model stage.

Every name a user can pass for an irradiance, optics, thermal, or bifacial
choice is enumerated, documented, and validated here, so ``breos.solar`` and
``breos.app_config`` share one definition of what is selectable and what each
selection means. The kernels in this package assume they are handed an
already-resolved :class:`PVModelOptions` and do no validation of their own.
"""

from __future__ import annotations

import math
import os
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterator, Mapping

import numpy as np
import pandas as pd
from pvlib.albedo import SURFACE_ALBEDOS
from pvlib.location import Location

from breos.utils import get_hours_per_step
from breos.weather import weather_metadata, weather_representative_time_offset

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
# ``interval-start`` evaluates at the timestamp itself (the only behaviour
# before 0.8.0). ``mid-interval`` evaluates half a step later, which is the
# PVWatts/SAM convention for interval-averaged irradiance: an hourly value
# labelled 07:00 that represents the 07:00-08:00 average pairs with the 07:30
# sun position. ``weather`` reads the provider convention from content-bound
# weather metadata. It supports instantaneous samples with an explicit
# provider offset and interval means with either left or right labels, and
# raises when the metadata does not state its radiation time basis.
# ``auto`` (the default) is ``weather`` when the metadata states its radiation
# time basis and ``interval-start`` when it does not, with one warning per run.
#
# Resolved by solar_position_at_labels, which both transposition
# (breos.solar._prepare_solarpos_and_weather) and terrain shading
# (breos.pv.horizon) call, so the two cannot disagree about where the sun is.
# It depends on the weather, so it is resolved separately from
# PVModelOptions rather than carried on it.
SOLAR_POSITION_METHODS = (
    "auto",
    "interval-start",
    "mid-interval",
    "weather",
)
DEFAULT_SOLAR_POSITION = "auto"

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

# The model options every PV production entry point accepts, with their
# defaults. Keep them with the resolvers that define the options. Callers
# must not maintain their own subsets.
PV_MODEL_DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {
        "transposition_model": DEFAULT_TRANSPOSITION_MODEL,
        "albedo": None,
        "surface_type": None,
        "model_perez": DEFAULT_PEREZ_MODEL,
        "solar_position": DEFAULT_SOLAR_POSITION,
        "iam_model": DEFAULT_IAM_MODEL,
        "diffuse_iam": DEFAULT_DIFFUSE_IAM,
        "temperature_model": DEFAULT_TEMPERATURE_MODEL,
        "bifacial_model": DEFAULT_BIFACIAL_MODEL,
        "gcr": 0.35,
        "pvrow_height": None,
        "pvrow_pitch": None,
    }
)
PV_MODEL_CONFIG_KEYS = tuple(PV_MODEL_DEFAULTS)

# The model options one entry of App's ``pv_arrays``, or of the ``arrays`` of
# ``breos.solar.calculate_multi_array_production``, may override.
# ``iam_model``, ``diffuse_iam`` and ``solar_position`` are deliberately
# absent: they are function-level for every array. That asymmetry is
# intentional and documented on the public function. Sky and ground
# geometry differ between arrays on one building, and so does the thermal
# model: it encodes the mounting, and a rack and a flush roof on one site run
# about 10 °C apart. The solar-position convention and the angle-of-incidence
# model describe the simulation, not the array. App echoes a ``pv_arrays``
# entry's options in this order.
PER_ARRAY_MODEL_OPTION_KEYS = (
    "temperature_model",
    "gcr",
    "transposition_model",
    "albedo",
    "surface_type",
    "model_perez",
    "bifacial_model",
    "pvrow_height",
    "pvrow_pitch",
)


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
    ``_resolve_choice`` report an unknown-name error (listing the valid names) for
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


def _resolve_choice(value: Any, valid: tuple[str, ...], label: str) -> str:
    """Normalise ``value`` and check it is one of ``valid``; ``label`` names it in the error.

    The error lists the valid names: "Unknown IAM model 'x'. Valid models: ...".
    """
    if not is_known_model(value, valid):
        kind = label.rsplit(" ", 1)[-1]
        raise ValueError(f"Unknown {label} {value!r}. Valid {kind}s: {', '.join(valid)}")
    return normalise_model_name(value)


def resolve_solar_position_method(method: str) -> str:
    """Normalise and validate a solar-position evaluation method name."""
    return _resolve_choice(method, SOLAR_POSITION_METHODS, "solar position method")


def resolve_solar_position_for_weather(method: str, weather: pd.DataFrame) -> str:
    """Return the solar-position method ``method`` applies to ``weather``.

    ``auto`` resolves to ``weather`` when the weather metadata states its
    ``radiation_time_basis`` and to ``interval-start`` when it does not; the
    other methods resolve to themselves. Nothing is checked or warned here:
    malformed metadata raises when the ``weather`` offset is computed.
    """
    method = resolve_solar_position_method(method)
    if method != "auto":
        return method
    return "weather" if weather_metadata(weather).get("radiation_time_basis") is not None else "interval-start"


@dataclass
class _WarningScope:
    """An open ``auto`` fallback warning scope: the process that opened it, and whether it has warned."""

    pid: int
    warned: bool = False


# The fallback warning scope of ``solar_position = "auto"``: None outside a
# scope, where every fallback warns. A scope belongs to the process that
# opened it: a process forked inside one inherits the variable, and treats
# the inherited scope as absent, so its own runs warn as their own.
_AUTO_FALLBACK_SCOPE: ContextVar[_WarningScope | None] = ContextVar("solar_position_auto_fallback_scope", default=None)

AUTO_FALLBACK_WARNING = (
    "solar_position = 'auto': the weather does not state its time basis (no 'radiation_time_basis' in its "
    "metadata), so the sun position is taken at the step labels, as with 'interval-start'. To silence this "
    "warning, set solar_position explicitly ('interval-start', 'mid-interval' or 'weather'), or give the "
    "weather file a metadata sidecar that states its radiation_time_basis."
)


def _open_scope() -> _WarningScope | None:
    """The scope this process has open, or None."""
    scope = _AUTO_FALLBACK_SCOPE.get()
    return scope if scope is not None and scope.pid == os.getpid() else None


@contextmanager
def solar_position_warning_scope() -> Iterator[None]:
    """Warn at most once about an ``auto`` fallback inside the block.

    A run computes the sun position several times: for each PV array, for
    terrain shading, for each Monte Carlo weather year and forecast file.
    The App, Monte Carlo and optimizer wrap a run in this scope so that the
    fallback warning appears once per run. A scope opened inside another
    of the same process is part of it. Also usable as a decorator.
    """
    if _open_scope() is not None:
        yield
        return
    token = _AUTO_FALLBACK_SCOPE.set(_WarningScope(os.getpid()))
    try:
        yield
    finally:
        _AUTO_FALLBACK_SCOPE.reset(token)


def warn_solar_position_fallback(requested: str, resolved: str) -> None:
    """Warn that ``auto`` fell back to the step labels, once per warning scope."""
    if normalise_model_name(requested) != "auto" or resolved != "interval-start":
        return
    scope = _open_scope()
    if scope is not None:
        if scope.warned:
            return
        scope.warned = True
    warnings.warn(AUTO_FALLBACK_WARNING, UserWarning, stacklevel=3)


def solar_position_time_offset(method: str, weather: pd.DataFrame, freq: str) -> pd.Timedelta:
    """Return how far after each weather label the sun position is evaluated.

    ``interval-start`` gives zero and ``mid-interval`` half a step. ``weather``
    reads the offset from the weather metadata and raises when the metadata
    does not state its radiation time basis. ``auto`` gives the offset of the
    method it resolves to (:func:`resolve_solar_position_for_weather`).
    """
    method = resolve_solar_position_for_weather(method, weather)
    if method == "mid-interval":
        return pd.Timedelta(hours=get_hours_per_step(freq) / 2.0)
    if method == "weather":
        return weather_representative_time_offset(weather, freq)
    return pd.Timedelta(0)


# The sun positions computed last, by everything they depend on. The sun is
# most of a PV run's time, and the optimizer, the terrain horizon and sweeps
# ask for the same grid again and again. At most about 20 MB: 45 hourly
# years or 11 quarter-hourly ones, the oldest dropped first.
_SOLAR_POSITIONS: OrderedDict[tuple, pd.DataFrame] = OrderedDict()
_SOLAR_POSITIONS_MAX_ROWS = 400_000
_PVLIB_SOLAR_POSITION = Location.get_solarposition


def _solar_position_key(location: Any, times: pd.DatetimeIndex, offset: pd.Timedelta) -> tuple | None:
    """What pvlib's sun position at ``times + offset`` depends on, or None if it is not kept.

    Only a plain pvlib ``Location`` with pvlib's own method is kept: a
    subclass or a replaced method may compute something else for the site.
    pvlib reads the instants in UTC, so the time zone is not part of the key.
    """
    if (
        type(location) is not Location
        or Location.get_solarposition is not _PVLIB_SOLAR_POSITION
        or "get_solarposition" in vars(location)
    ):
        return None
    return (
        location.latitude,
        location.longitude,
        location.altitude,
        offset.value,
        times.as_unit("ns").asi8.tobytes(),  # type: ignore[attr-defined]  # pandas-stubs omits asi8
    )


def solar_position_record(method: str, weather: pd.DataFrame, freq: str) -> dict[str, Any]:
    """The solar-position method asked for, the one ``weather`` resolves it to, and its offset.

    The provenance record of a run: ``requested`` and ``resolved`` method
    names, and ``offset_minutes`` after each label. It does not warn.
    """
    requested = resolve_solar_position_method(method)
    resolved = resolve_solar_position_for_weather(requested, weather)
    offset = solar_position_time_offset(resolved, weather, freq)
    return {"requested": requested, "resolved": resolved, "offset_minutes": offset.total_seconds() / 60.0}


def solar_position_at_labels(
    location: Any, times: pd.DatetimeIndex, weather: pd.DataFrame, freq: str, solar_position: str
) -> tuple[pd.DataFrame, str]:
    """Return the sun position for each label in ``times`` and the resolved method.

    The sun is evaluated :func:`solar_position_time_offset` after each label,
    and the result is indexed at the labels. The last positions computed are
    kept, so a repeated request returns a copy of the same frame.
    An ``auto`` that falls back to the labels warns (:func:`warn_solar_position_fallback`).
    """
    method = resolve_solar_position_for_weather(solar_position, weather)
    offset = solar_position_time_offset(method, weather, freq)
    warn_solar_position_fallback(solar_position, method)
    key = _solar_position_key(location, times, offset)
    if key is not None and key in _SOLAR_POSITIONS:
        _SOLAR_POSITIONS.move_to_end(key)
        solarpos = _SOLAR_POSITIONS[key]
    else:
        solarpos = location.get_solarposition(times=times + offset)
        if key is not None and len(times) <= _SOLAR_POSITIONS_MAX_ROWS:
            _SOLAR_POSITIONS[key] = solarpos
            while sum(len(frame) for frame in _SOLAR_POSITIONS.values()) > _SOLAR_POSITIONS_MAX_ROWS:
                _SOLAR_POSITIONS.popitem(last=False)
    solarpos = solarpos.copy()
    solarpos.index = times
    return solarpos, method


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
    model = _resolve_choice(model, BIFACIAL_MODELS, "bifacial model")
    if model == "none":
        return model
    if bifaciality is None:
        raise ValueError("bifacial_model='infinite_sheds' requires PV module bifaciality metadata")

    geometry = {
        "gcr": gcr,
        "pvrow_height": pvrow_height,
        "pvrow_pitch": pvrow_pitch,
    }
    checked: dict[str, float] = {}
    for name, value in geometry.items():
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise TypeError(f"{name} must be a finite number for bifacial modeling")
        checked[name] = float(value)
        if not math.isfinite(checked[name]):
            raise ValueError(f"{name} must be a finite number for bifacial modeling")
    if not is_valid_gcr(gcr):
        raise ValueError("gcr must be between 0 (exclusive) and 1 (inclusive) for bifacial modeling")
    if checked["pvrow_height"] <= 0.0:
        raise ValueError("pvrow_height must be > 0 for bifacial modeling")
    if checked["pvrow_pitch"] <= 0.0:
        raise ValueError("pvrow_pitch must be > 0 for bifacial modeling")
    return model


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
    transposition_model = _resolve_choice(transposition_model, TRANSPOSITION_MODELS, "transposition model")
    albedo, surface_type = resolve_ground_reflectance(albedo, surface_type)
    model_perez = _resolve_choice(model_perez, PEREZ_MODELS, "Perez coefficient model")
    iam_model = _resolve_choice(iam_model, IAM_MODELS, "IAM model")
    diffuse_iam = _resolve_choice(diffuse_iam, DIFFUSE_IAM_METHODS, "diffuse IAM method")
    temperature_model = _resolve_choice(temperature_model, TEMPERATURE_MODELS, "temperature model")
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


def configured_pv_model_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the whole PV model-option block of a resolved application config.

    A key the config does not set takes its ``PV_MODEL_DEFAULTS`` value. This
    function is the single config-to-call mapping used by App and optimization.
    """
    return {key: config.get(key, default) for key, default in PV_MODEL_DEFAULTS.items()}
