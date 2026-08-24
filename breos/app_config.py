"""Configuration and resource resolution for the public App facade."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from numbers import Real
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from breos.degradation.profiles import ENABLED_BLAST_MODEL_KEYS, apply_battery_profile_defaults
from breos.economics import COST_CONFIG_KEY_TO_PARAM, CostParams, calculate_costs
from breos.emissions import EmissionsParams
from breos.pv.horizon import normalise_horizon_profile
from breos.pv.model_options import is_known_model, is_valid_albedo, is_valid_gcr, normalise_model_name
from breos.pv.temperature import validate_temperature_inputs
from breos.pv_modules import MODULES, PVModuleParams, get_module
from breos.resources import load_config_json
from breos.solar import (
    BIFACIAL_MODELS,
    DEFAULT_BIFACIAL_MODEL,
    DEFAULT_DIFFUSE_IAM,
    DEFAULT_IAM_MODEL,
    DEFAULT_PEREZ_MODEL,
    DEFAULT_SOLAR_POSITION,
    DEFAULT_TEMPERATURE_MODEL,
    DEFAULT_TRANSPOSITION_MODEL,
    DIFFUSE_IAM_METHODS,
    IAM_MODELS,
    PEREZ_MODELS,
    SOLAR_POSITION_METHODS,
    SURFACE_TYPES,
    TEMPERATURE_MODELS,
    TRANSPOSITION_MODELS,
    estimate_optimal_tilt,
)
from breos.solar import default_azimuth as default_azimuth_fn

_NO_DEFAULT = object()


@dataclass(frozen=True)
class AppConfigField:
    """Declarative metadata for one public App configuration key.

    Scientific constraints intentionally remain in the focused validators
    below. This registry owns the mechanical contract that had previously
    drifted across defaults, the allowed-key list, CLI arguments, and CLI
    override propagation.
    """

    default: Any = _NO_DEFAULT
    default_order: int | None = None
    cli_flags: tuple[str, ...] = ()
    cli_type: Callable[[str], Any] | None = None
    cli_choices: tuple[str, ...] | None = None
    cli_action: str | None = None
    cli_help: str | None = None
    cli_normalizer: Callable[[Any], Any] | None = None

    @property
    def has_default(self) -> bool:
        return self.default is not _NO_DEFAULT


def _lower(value: str) -> str | None:
    return value.lower() if value else None


def _upper(value: str) -> str | None:
    return value.upper() if value else None


def _underscored(value: str) -> str | None:
    return value.replace("-", "_") if value else None


APP_CONFIG_FIELDS: dict[str, AppConfigField] = {
    # CLI-exposed fields are kept in parser display order. Required inputs have
    # no default; argparse still leaves them optional so --config can supply
    # them, and the existing App validators remain the source of required-key
    # errors.
    "location": AppConfigField(
        cli_flags=("--location",),
        cli_help="Location preset key, for example 'porto'.",
        cli_normalizer=_lower,
    ),
    "n_modules": AppConfigField(cli_flags=("--n-modules",), cli_type=int, cli_help="Number of PV modules."),
    "annual_consumption_kwh": AppConfigField(
        cli_flags=("--annual-consumption-kwh",),
        cli_type=float,
        cli_help="Annual electricity demand in kWh.",
    ),
    "battery_kwh": AppConfigField(
        default=0.0,
        default_order=0,
        cli_flags=("--battery-kwh",),
        cli_type=float,
        cli_help="Battery capacity in kWh.",
    ),
    "battery_max_charge_power_w": AppConfigField(
        default=None,
        default_order=42,
        cli_flags=("--battery-max-charge-power-w",),
        cli_type=float,
        cli_help="Maximum DC power entering the battery charge path in W (default: unlimited).",
    ),
    "battery_max_discharge_power_w": AppConfigField(
        default=None,
        default_order=43,
        cli_flags=("--battery-max-discharge-power-w",),
        cli_type=float,
        cli_help="Maximum battery AC power delivered to load in W (default: unlimited).",
    ),
    "cost_preset": AppConfigField(
        default=None,
        default_order=28,
        cli_flags=("--cost-preset",),
        cli_help="Cost preset key, for example 'residential-pt'.",
        cli_normalizer=_underscored,
    ),
    "emissions_country": AppConfigField(
        default=None,
        default_order=32,
        cli_flags=("--emissions-country",),
        cli_help="Country code for emissions, for example 'pt'.",
        cli_normalizer=_upper,
    ),
    "pv_module": AppConfigField(
        default=None, default_order=2, cli_flags=("--pv-module",), cli_help="PV module catalogue key."
    ),
    "load_profile": AppConfigField(
        default="1", default_order=3, cli_flags=("--load-profile",), cli_help="Load profile type."
    ),
    "rlp_directory": AppConfigField(
        default=None,
        default_order=4,
        cli_flags=("--rlp-directory",),
        cli_type=Path,
        cli_help="Directory containing licensed external RLP CSV files.",
        cli_normalizer=str,
    ),
    "tilt": AppConfigField(
        default=None,
        default_order=5,
        cli_flags=("--tilt",),
        cli_type=float,
        cli_help="PV tilt angle in degrees.",
    ),
    "azimuth": AppConfigField(
        default=None,
        default_order=6,
        cli_flags=("--azimuth",),
        cli_type=float,
        cli_help="PV surface azimuth in degrees.",
    ),
    "transposition_model": AppConfigField(
        default=DEFAULT_TRANSPOSITION_MODEL,
        default_order=15,
        cli_flags=("--transposition-model", "--sky-model"),
        cli_choices=tuple(TRANSPOSITION_MODELS),
        cli_help="Sky-diffusion model for POA transposition (default: isotropic).",
    ),
    "albedo": AppConfigField(
        default=None,
        default_order=16,
        cli_flags=("--albedo",),
        cli_type=float,
        cli_help="Ground reflectance 0-1 (default: pvlib 0.25). Excludes --surface-type.",
    ),
    "surface_type": AppConfigField(
        default=None,
        default_order=17,
        cli_flags=("--surface-type",),
        cli_choices=tuple(SURFACE_TYPES),
        cli_help="Named ground cover mapped to an albedo (alternative to --albedo).",
    ),
    "model_perez": AppConfigField(
        default=DEFAULT_PEREZ_MODEL,
        default_order=18,
        cli_flags=("--perez-model",),
        cli_choices=tuple(PEREZ_MODELS),
        cli_help="Perez coefficient set (only used with --transposition-model perez).",
    ),
    "solar_position": AppConfigField(
        default=DEFAULT_SOLAR_POSITION,
        default_order=19,
        cli_flags=("--solar-position",),
        cli_choices=tuple(SOLAR_POSITION_METHODS),
        cli_help=(
            "Where within each timestep the sun position is evaluated. 'mid-interval' matches "
            "PVWatts/SAM for interval-averaged weather (default: interval-start)."
        ),
    ),
    "iam_model": AppConfigField(
        default=DEFAULT_IAM_MODEL,
        default_order=20,
        cli_flags=("--iam-model",),
        cli_choices=tuple(IAM_MODELS),
        cli_help="Beam incidence-angle modifier (default: ashrae, historical compatibility).",
    ),
    "diffuse_iam": AppConfigField(
        default=DEFAULT_DIFFUSE_IAM,
        default_order=21,
        cli_flags=("--diffuse-iam",),
        cli_choices=tuple(DIFFUSE_IAM_METHODS),
        cli_help=(
            "Whether IAM is also applied to the diffuse POA components. 'marion' weighs sky- and "
            "ground-diffuse with the view-factor-integrated selected IAM model (default: none, beam-only)."
        ),
    ),
    "temperature_model": AppConfigField(
        default=DEFAULT_TEMPERATURE_MODEL,
        default_order=22,
        cli_flags=("--temperature-model",),
        cli_choices=tuple(TEMPERATURE_MODELS),
        cli_help=(
            "Cell-temperature model / mounting preset. The pvsyst-* and sapm-* presets use documented "
            "mounting coefficients; noct-sam additionally requires sourced module NOCT and efficiency "
            "metadata (not yet bundled). Default: faiman, open rack."
        ),
    ),
    "bifacial_model": AppConfigField(
        default=DEFAULT_BIFACIAL_MODEL,
        default_order=23,
        cli_flags=("--bifacial-model",),
        cli_choices=tuple(BIFACIAL_MODELS),
        cli_help=(
            "Rear-irradiance model (default: none; infinite_sheds requires bifacial module metadata and row geometry)."
        ),
    ),
    "pvrow_height": AppConfigField(
        default=None,
        default_order=24,
        cli_flags=("--pvrow-height",),
        cli_type=float,
        cli_help="PV row center height above ground; use the same unit as --pvrow-pitch.",
    ),
    "pvrow_pitch": AppConfigField(
        default=None,
        default_order=25,
        cli_flags=("--pvrow-pitch",),
        cli_type=float,
        cli_help="Distance between PV rows; use the same unit as --pvrow-height.",
    ),
    "gcr": AppConfigField(
        default=0.35,
        default_order=12,
        cli_flags=("--gcr",),
        cli_type=float,
        cli_help="PV row ground coverage ratio (default: 0.35).",
    ),
    "resolution": AppConfigField(
        default="h",
        default_order=26,
        cli_flags=("--resolution",),
        cli_choices=("h", "15min"),
        cli_help="Simulation time resolution.",
    ),
    "projection_years": AppConfigField(
        default=20,
        default_order=27,
        cli_flags=("--projection-years",),
        cli_type=int,
        cli_help="Economic projection horizon.",
    ),
    "inflation_rate": AppConfigField(
        default=0.02,
        default_order=29,
        cli_flags=("--inflation-rate",),
        cli_type=float,
        cli_help="Annual electricity price inflation.",
    ),
    "sell_price_inflation": AppConfigField(
        default=0.0,
        default_order=30,
        cli_flags=("--sell-price-inflation",),
        cli_type=float,
        cli_help="Annual inflation of the grid export (sell) price. Default 0.",
    ),
    "export_emissions_factor_gco2_kwh": AppConfigField(
        default=None,
        default_order=33,
        cli_flags=("--export-emissions-factor-gco2-kwh",),
        cli_type=float,
        cli_help="Exported-generation displacement factor in gCO2/kWh (default: grid avoided factor).",
    ),
    "discount_rate": AppConfigField(
        default=0.03,
        default_order=31,
        cli_flags=("--discount-rate",),
        cli_type=float,
        cli_help="Discount rate for NPV calculations.",
    ),
    "pv_degradation_rate": AppConfigField(
        default=0.005,
        default_order=34,
        cli_flags=("--pv-degradation-rate",),
        cli_type=float,
        cli_help="Annual PV degradation rate.",
    ),
    "calendar_model": AppConfigField(
        default="naumann_lam_field_calibrated",
        default_order=35,
        cli_flags=("--calendar-model",),
        cli_help="Battery calendar aging model.",
    ),
    "degradation_engine": AppConfigField(
        default="native",
        default_order=36,
        cli_flags=("--degradation-engine",),
        cli_choices=("native", "blast"),
        cli_help="Battery degradation engine (default: native Naumann/Lam).",
    ),
    "blast_model": AppConfigField(
        default=None,
        default_order=37,
        cli_flags=("--blast-model",),
        cli_help="Stable BLAST battery-model key; requires --degradation-engine blast.",
    ),
    "dc_coupled": AppConfigField(
        default=True,
        default_order=45,
        cli_flags=("--dc-coupled",),
        cli_action="store_true",
        cli_help="Use the supported DC-coupled/hybrid battery model.",
    ),
    "inverter_efficiency": AppConfigField(
        default=0.96,
        default_order=46,
        cli_flags=("--inverter-efficiency",),
        cli_type=float,
        cli_help="Inverter efficiency.",
    ),
    "inverter_loading_ratio": AppConfigField(
        default=1.25,
        default_order=47,
        cli_flags=("--inverter-loading-ratio",),
        cli_type=float,
        cli_help="DC/AC oversizing ratio.",
    ),
    "start_date": AppConfigField(
        default="2023-01-01",
        default_order=49,
        cli_flags=("--start-date",),
        cli_help="Simulation start date, YYYY-MM-DD.",
    ),
    # Config-file/API-only fields.
    "costs": AppConfigField(),
    "pv_arrays": AppConfigField(default=None, default_order=1),
    "tracking": AppConfigField(default="fixed", default_order=7),
    "axis_tilt": AppConfigField(default=0.0, default_order=8),
    "axis_azimuth": AppConfigField(default=None, default_order=9),
    "max_angle": AppConfigField(default=60.0, default_order=10),
    "backtrack": AppConfigField(default=True, default_order=11),
    "cross_axis_tilt": AppConfigField(default=0.0, default_order=13),
    "dual_axis_max_tilt": AppConfigField(default=90.0, default_order=14),
    "battery_min_soc": AppConfigField(default=0.10, default_order=38),
    "battery_max_soc": AppConfigField(default=0.90, default_order=39),
    "battery_eol_percentage": AppConfigField(default=0.70, default_order=40),
    "battery_rte": AppConfigField(default=None, default_order=41),
    "enable_resistance_fade": AppConfigField(default=False, default_order=44),
    "pv_loss_overrides": AppConfigField(default=None, default_order=48),
    "horizon_profile": AppConfigField(default=None, default_order=50),
    "battery_temperature": AppConfigField(default="weather", default_order=51),
    "battery_indoor_model": AppConfigField(default=None, default_order=52),
    # Runner sections are accepted by App resolution so each workflow can use
    # the same base config validation. The CLI validates their own structure.
    "montecarlo": AppConfigField(),
    "sweep": AppConfigField(),
    # Kept solely to preserve the existing actionable legacy-selector error.
    "battery_type": AppConfigField(),
}

_DEFAULT_FIELDS = sorted(
    ((name, field) for name, field in APP_CONFIG_FIELDS.items() if field.has_default),
    key=lambda item: item[1].default_order if item[1].default_order is not None else math.inf,
)
DEFAULTS: dict[str, Any] = {name: field.default for name, field in _DEFAULT_FIELDS}

# Everything at the top level must be registered so typos (e.g.
# ``batery_kwh``) fail loudly instead of being silently dropped by defaults.
ALLOWED_CONFIG_KEYS: frozenset[str] = frozenset(APP_CONFIG_FIELDS)

# Cost override keys deliberately use the existing preset-catalog vocabulary;
# the canonical translation to CostParams lives in ``breos.economics`` so the
# App and lower-level construction helper cannot drift.
COST_OVERRIDE_KEYS: frozenset[str] = frozenset(COST_CONFIG_KEY_TO_PARAM)


@dataclass(frozen=True)
class ResolvedAppConfig:
    """Config values resolved to runtime objects used by the App pipeline."""

    cfg: dict[str, Any]
    lat: float
    lon: float
    timezone: str
    loc_key: str | None
    pv_arrays: list[dict[str, Any]]
    pv_params: PVModuleParams
    avg_module_power_w: float
    system_kwp: float
    tilt: float
    azimuth: float
    tracking: str
    axis_azimuth: float
    cost_params: CostParams
    emissions_params: EmissionsParams | None


def load_json(name: str) -> dict[str, Any]:
    """Load a packaged App configuration resource."""
    return load_config_json(name)


def merge_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Apply user values over BLAST profile defaults over global defaults."""
    return apply_battery_profile_defaults(DEFAULTS, config)


def default_module_key() -> str:
    """Return the catalog key used when a config names no PV module.

    ``DEFAULTS["pv_module"]`` is ``None`` rather than a key, so the real
    default is the catalog's insertion order. Resolving it here keeps the
    validation, resolution, and results layers from each re-deriving it and
    silently disagreeing if the catalog is reordered.
    """
    return next(iter(MODULES))


def _validate_sky_settings(
    transposition_model: Any,
    albedo: Any,
    surface_type: Any,
    model_perez: Any,
    where: str = "",
) -> None:
    """Validate the sky-diffusion settings shared by the top level and arrays.

    ``where`` prefixes the key name in error messages (e.g. ``pv_arrays[0]``);
    ``None`` values are treated as "not set" and skipped, so per-array overrides
    only validate the keys they actually provide.

    The rules come from :mod:`breos.pv.model_options`; only the config-key
    phrasing and the None-means-unset handling are this layer's own.
    """
    prefix = f"{where}." if where else ""
    if transposition_model is not None and not is_known_model(transposition_model, TRANSPOSITION_MODELS):
        valid = ", ".join(TRANSPOSITION_MODELS)
        raise ValueError(f"'{prefix}transposition_model' must be one of: {valid}")
    if albedo is not None and surface_type is not None:
        raise ValueError(f"Set either '{prefix}albedo' or '{prefix}surface_type', not both.")
    if albedo is not None and (not isinstance(albedo, (int, float)) or not is_valid_albedo(albedo)):
        raise ValueError(f"'{prefix}albedo' must be a number between 0 and 1")
    if surface_type is not None and surface_type not in SURFACE_TYPES:
        valid = ", ".join(SURFACE_TYPES)
        raise ValueError(f"'{prefix}surface_type' must be one of: {valid}")
    if model_perez is not None and model_perez not in PEREZ_MODELS:
        valid = ", ".join(PEREZ_MODELS)
        raise ValueError(f"'{prefix}model_perez' must be one of: {valid}")


def _validate_gcr(gcr: Any, prefix: str = "") -> None:
    """Validate a ground coverage ratio, whichever model consumes it.

    ``gcr`` has two consumers and neither is a shading calculation: the
    ``infinite_sheds`` rear-side view factors, and — on the tracking path —
    the backtracking rotation schedule that pvlib's ``singleaxis`` derives
    from it. The second is why this is checked even with no rear-side model
    active. pvlib does not reject a nonsensical ratio; it quietly computes a
    different rotation, so a mistyped ``3.5`` returns roughly half the annual
    energy with no error anywhere. ``prefix`` already carries its trailing
    dot, matching the other config-key messages.
    """
    if not is_valid_gcr(_finite_real(gcr, f"{prefix}gcr")):
        raise ValueError(f"'{prefix}gcr' must be between 0 (exclusive) and 1 (inclusive)")


def _validate_bifacial_settings(
    model: Any,
    module: Any,
    gcr: Any,
    pvrow_height: Any,
    pvrow_pitch: Any,
    where: str = "",
) -> None:
    """Validate opt-in bifacial module metadata and row geometry.

    Shares its predicates with :mod:`breos.pv.model_options` but reports them
    against config keys. ``pvrow_*`` geometry is required only for an active
    rear-side model; ``gcr`` is also checked independently after the existing
    validators because tracking can consume it without bifacial modeling.
    """
    prefix = f"{where}." if where else ""
    normalised = normalise_model_name(model)
    if not is_known_model(model, BIFACIAL_MODELS):
        valid = ", ".join(BIFACIAL_MODELS)
        raise ValueError(f"'{prefix}bifacial_model' must be one of: {valid}")

    for key, value in (("pvrow_height", pvrow_height), ("pvrow_pitch", pvrow_pitch)):
        if value is not None and _finite_real(value, f"{prefix}{key}") <= 0:
            raise ValueError(f"'{prefix}{key}' must be > 0 when configured")

    if normalised == "none":
        return

    module_key = module or default_module_key()
    if module_key in MODULES and MODULES[module_key].bifaciality is None:
        raise ValueError(
            f"'{prefix}bifacial_model=infinite_sheds' requires bifaciality metadata for PV module {module_key!r}"
        )
    if pvrow_height is None:
        raise ValueError(f"'{prefix}pvrow_height' is required when bifacial_model='infinite_sheds'")
    if pvrow_pitch is None:
        raise ValueError(f"'{prefix}pvrow_pitch' is required when bifacial_model='infinite_sheds'")
    # Deliberately last, so an active rear-side model still reports its
    # missing metadata and geometry before quibbling about the ratio.
    _validate_gcr(gcr, prefix)


def validate_config(cfg: dict[str, Any]) -> None:
    """Validate user-facing App config before resolving derived values."""
    has_arrays = _validate_structure_and_location(cfg)
    _validate_pv_and_inverter(cfg, has_arrays)
    _validate_time_and_weather(cfg)
    _validate_economics(cfg)
    _validate_battery_and_degradation(cfg)
    _validate_reachable_gcr(cfg, has_arrays)


def _validate_reachable_gcr(cfg: dict[str, Any], has_arrays: bool) -> None:
    """Check every ``gcr`` that can reach the model, once everything else passes.

    Runs last, and deliberately so: an out-of-range ``gcr`` used to be caught
    only under an active bifacial model, so checking it earlier would change
    which error an already-broken config reports. Running it here makes the
    check purely additive — a config failing on some other key keeps failing
    on that key, and ``gcr`` is only ever the *new* reason a config is
    rejected.

    Arrays that set no ``gcr`` inherit the top-level value, which is also the
    function-level default handed to the multi-array entry point, so the
    top-level value is checked even when every array overrides it. Bifacial
    arrays have already had their effective ``gcr`` checked in the loop above;
    re-checking an explicit override here is harmless and covers the
    non-bifacial tracking path that nothing else reaches.
    """
    _validate_gcr(cfg["gcr"])
    if has_arrays:
        for index, array in enumerate(cfg["pv_arrays"]):
            if "gcr" in array:
                _validate_gcr(array["gcr"], f"pv_arrays[{index}].")


def _validate_structure_and_location(cfg: dict[str, Any]) -> bool:
    """Validate top-level keys, required inputs, and location structure."""
    if "battery_type" in cfg:
        raise ValueError(
            "'battery_type' is an ambiguous legacy selector and is not supported by App. "
            "Use degradation_engine='native' (default), or set degradation_engine='blast' with blast_model='<key>'."
        )
    unknown = set(cfg) - ALLOWED_CONFIG_KEYS
    if unknown:
        available = ", ".join(sorted(ALLOWED_CONFIG_KEYS))
        raise ValueError(f"Unknown config key(s): {', '.join(sorted(unknown))}. Available: {available}")

    for key in ("location", "annual_consumption_kwh"):
        if key not in cfg:
            raise ValueError(f"Missing required config key: '{key}'")

    has_arrays = bool(cfg.get("pv_arrays"))
    if not has_arrays and "n_modules" not in cfg:
        raise ValueError("Missing required config key: 'n_modules'")

    loc = cfg["location"]
    if isinstance(loc, dict):
        for field in ("latitude", "longitude", "timezone"):
            if field not in loc:
                raise ValueError(f"Custom location must include '{field}'")
        lat = _finite_real(loc["latitude"], "location.latitude")
        lon = _finite_real(loc["longitude"], "location.longitude")
        if not -90 <= lat <= 90:
            raise ValueError("'location.latitude' must be between -90 and 90")
        if not -180 <= lon <= 180:
            raise ValueError("'location.longitude' must be between -180 and 180")
        if not isinstance(loc["timezone"], str):
            raise TypeError("'location.timezone' must be an IANA timezone string")
        try:
            ZoneInfo(loc["timezone"])
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Unknown IANA timezone: {loc['timezone']!r}") from exc
    elif not isinstance(loc, str):
        raise TypeError("'location' must be a string key or a dict with latitude/longitude/timezone")

    return has_arrays


def _validate_pv_and_inverter(cfg: dict[str, Any], has_arrays: bool) -> None:
    """Validate PV sizing, array geometry, sky models, and inverter inputs."""
    if not has_arrays and (not _is_int(cfg["n_modules"]) or cfg["n_modules"] < 1):
        raise ValueError("'n_modules' must be >= 1")
    if has_arrays:
        if not isinstance(cfg["pv_arrays"], list):
            raise TypeError("'pv_arrays' must be a list")
        for i, arr in enumerate(cfg["pv_arrays"]):
            if not isinstance(arr, dict):
                raise TypeError(f"'pv_arrays[{i}]' must be a dict")
            modules = arr.get("modules", 0)
            if not _is_int(modules) or modules < 1:
                raise ValueError(f"'pv_arrays[{i}].modules' must be >= 1")
            module = arr.get("module", cfg.get("pv_module"))
            if module is not None and module not in MODULES:
                available = ", ".join(sorted(MODULES))
                raise ValueError(f"Unknown PV module {module!r} in pv_arrays[{i}]. Available: {available}")
            tilt = arr.get("tilt", cfg.get("tilt"))
            azimuth = arr.get("azimuth", cfg.get("azimuth"))
            if tilt is not None and not 0 <= _finite_real(tilt, f"pv_arrays[{i}].tilt") <= 90:
                raise ValueError(f"'pv_arrays[{i}].tilt' must be between 0 and 90")
            if azimuth is not None and not 0 <= _finite_real(azimuth, f"pv_arrays[{i}].azimuth") <= 360:
                raise ValueError(f"'pv_arrays[{i}].azimuth' must be between 0 and 360")
            _validate_sky_settings(
                arr.get("transposition_model"),
                arr.get("albedo"),
                arr.get("surface_type"),
                arr.get("model_perez"),
                where=f"pv_arrays[{i}]",
            )
            _validate_bifacial_settings(
                arr.get("bifacial_model", cfg["bifacial_model"]),
                module or default_module_key(),
                arr.get("gcr", cfg["gcr"]),
                arr.get("pvrow_height", cfg["pvrow_height"]),
                arr.get("pvrow_pitch", cfg["pvrow_pitch"]),
                where=f"pv_arrays[{i}]",
            )
    if _finite_real(cfg["annual_consumption_kwh"], "annual_consumption_kwh") <= 0:
        raise ValueError("'annual_consumption_kwh' must be > 0")
    if _finite_real(cfg["battery_kwh"], "battery_kwh") < 0:
        raise ValueError("'battery_kwh' must be >= 0")
    if cfg.get("pv_module") is not None and cfg["pv_module"] not in MODULES:
        available = ", ".join(sorted(MODULES))
        raise ValueError(f"Unknown PV module {cfg['pv_module']!r}. Available: {available}")
    tilt = cfg.get("tilt")
    if tilt is not None and not 0 <= _finite_real(tilt, "tilt") <= 90:
        raise ValueError("'tilt' must be between 0 and 90")
    azimuth = cfg.get("azimuth")
    if azimuth is not None and not 0 <= _finite_real(azimuth, "azimuth") <= 360:
        raise ValueError("'azimuth' must be between 0 and 360")
    if not has_arrays:
        _validate_bifacial_settings(
            cfg["bifacial_model"],
            cfg.get("pv_module") or default_module_key(),
            cfg["gcr"],
            cfg["pvrow_height"],
            cfg["pvrow_pitch"],
        )
    if not 0 < _finite_real(cfg["inverter_efficiency"], "inverter_efficiency") <= 1:
        raise ValueError("'inverter_efficiency' must be between 0 (exclusive) and 1 (inclusive)")
    if _finite_real(cfg["inverter_loading_ratio"], "inverter_loading_ratio") <= 0:
        raise ValueError("'inverter_loading_ratio' must be > 0")


def _validate_time_and_weather(cfg: dict[str, Any]) -> None:
    """Validate simulation horizon, resolution, and solar-model selections."""
    if not _is_int(cfg["projection_years"]) or cfg["projection_years"] < 1:
        raise ValueError("'projection_years' must be >= 1")
    if not 0 <= _finite_real(cfg["pv_degradation_rate"], "pv_degradation_rate") < 1:
        raise ValueError("'pv_degradation_rate' must be between 0 (inclusive) and 1 (exclusive)")
    if cfg["resolution"] not in ("h", "15min"):
        raise ValueError("'resolution' must be 'h' or '15min'")
    cfg["horizon_profile"] = normalise_horizon_profile(cfg["horizon_profile"])
    _validate_sky_settings(cfg["transposition_model"], cfg["albedo"], cfg["surface_type"], cfg["model_perez"])
    if not is_known_model(cfg["solar_position"], SOLAR_POSITION_METHODS):
        valid = ", ".join(SOLAR_POSITION_METHODS)
        raise ValueError(f"'solar_position' must be one of: {valid}")
    if not is_known_model(cfg["iam_model"], IAM_MODELS):
        valid = ", ".join(IAM_MODELS)
        raise ValueError(f"'iam_model' must be one of: {valid}")
    if not is_known_model(cfg["diffuse_iam"], DIFFUSE_IAM_METHODS):
        valid = ", ".join(DIFFUSE_IAM_METHODS)
        raise ValueError(f"'diffuse_iam' must be one of: {valid}")
    if not is_known_model(cfg["temperature_model"], TEMPERATURE_MODELS):
        valid = ", ".join(TEMPERATURE_MODELS)
        raise ValueError(f"'temperature_model' must be one of: {valid}")
    cfg["bifacial_model"] = normalise_model_name(cfg["bifacial_model"])
    overrides = cfg.get("pv_loss_overrides")
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise TypeError("'pv_loss_overrides' must be a dict of loss component percentages")
        for name, value in overrides.items():
            if not isinstance(value, (int, float)) or not 0 <= value <= 100:
                raise ValueError(f"'pv_loss_overrides[{name!r}]' must be a percentage between 0 and 100")


def _validate_economics(cfg: dict[str, Any]) -> None:
    """Validate financial and optional export-emissions inputs."""
    for key in ("inflation_rate", "discount_rate"):
        if _finite_real(cfg[key], key) <= -1:
            raise ValueError(f"'{key}' must be greater than -1")
    if not -1 < _finite_real(cfg["sell_price_inflation"], "sell_price_inflation") < 1:
        raise ValueError("'sell_price_inflation' must be between -1 and 1 (exclusive)")
    if "costs" in cfg:
        overrides = cfg["costs"]
        if not isinstance(overrides, dict):
            raise TypeError("'costs' must be a table/dict of cost overrides")
        unknown = set(overrides) - COST_OVERRIDE_KEYS
        if unknown:
            available = ", ".join(f"costs.{key}" for key in sorted(COST_OVERRIDE_KEYS))
            if len(unknown) == 1:
                unknown_text = f"Unknown key 'costs.{next(iter(unknown))}'"
            else:
                unknown_text = "Unknown keys " + ", ".join(f"'costs.{key}'" for key in sorted(unknown))
            raise ValueError(f"{unknown_text}. Available: {available}")
        for key, value in overrides.items():
            if _finite_real(value, f"costs.{key}") < 0:
                raise ValueError(f"'costs.{key}' must be >= 0")
    if cfg["export_emissions_factor_gco2_kwh"] is not None:
        if _finite_real(cfg["export_emissions_factor_gco2_kwh"], "export_emissions_factor_gco2_kwh") < 0:
            raise ValueError("'export_emissions_factor_gco2_kwh' must be >= 0 when configured")


def _validate_battery_and_degradation(cfg: dict[str, Any]) -> None:
    """Validate battery dispatch and explicit degradation-engine selection."""
    min_soc = _finite_real(cfg["battery_min_soc"], "battery_min_soc")
    max_soc = _finite_real(cfg["battery_max_soc"], "battery_max_soc")
    if not 0 <= min_soc < max_soc <= 1:
        raise ValueError("'battery_min_soc' and 'battery_max_soc' must satisfy 0 <= min < max <= 1")
    if not 0 < _finite_real(cfg["battery_eol_percentage"], "battery_eol_percentage") < 1:
        raise ValueError("'battery_eol_percentage' must be between 0 and 1 (exclusive)")
    if cfg["battery_rte"] is not None and not 0 < _finite_real(cfg["battery_rte"], "battery_rte") <= 1:
        raise ValueError("'battery_rte' must be between 0 (exclusive) and 1 (inclusive)")
    for key in ("battery_max_charge_power_w", "battery_max_discharge_power_w"):
        if cfg[key] is not None and _finite_real(cfg[key], key) < 0:
            raise ValueError(f"'{key}' must be >= 0 when configured")
    battery_temperature = cfg["battery_temperature"]
    if isinstance(battery_temperature, Real) and not isinstance(battery_temperature, bool):
        _finite_real(battery_temperature, "battery_temperature")
    elif not isinstance(battery_temperature, str):
        raise TypeError("'battery_temperature' must be 'weather', a CSV path, or a finite temperature")
    indoor_model = cfg["battery_indoor_model"]
    if indoor_model is not None:
        if not isinstance(indoor_model, dict):
            raise TypeError("'battery_indoor_model' must be a mapping when configured")
        unknown = set(indoor_model) - {"enabled", "setpoint_c", "coupling_alpha", "floor_c", "ceiling_c"}
        if unknown:
            raise ValueError(f"Unknown battery_indoor_model key(s): {', '.join(sorted(unknown))}")
        if "enabled" in indoor_model and not isinstance(indoor_model["enabled"], bool):
            raise TypeError("'battery_indoor_model.enabled' must be a boolean")
        for key in ("setpoint_c", "coupling_alpha", "floor_c", "ceiling_c"):
            if key in indoor_model:
                _finite_real(indoor_model[key], f"battery_indoor_model.{key}")
        coupling = indoor_model.get("coupling_alpha")
        if coupling is not None and not 0 <= float(coupling) <= 1:
            raise ValueError("'battery_indoor_model.coupling_alpha' must be between 0 and 1")
        floor = indoor_model.get("floor_c")
        ceiling = indoor_model.get("ceiling_c")
        if floor is not None and ceiling is not None and float(floor) > float(ceiling):
            raise ValueError("'battery_indoor_model.floor_c' must not exceed 'battery_indoor_model.ceiling_c'")
    if not isinstance(cfg["dc_coupled"], bool):
        raise TypeError("'dc_coupled' must be a boolean")
    if not cfg["dc_coupled"]:
        raise NotImplementedError("BREOS 0.3.x supports DC-coupled/hybrid battery dispatch only")
    valid_calendar_models = {
        "naumann",
        "naumann_lam",
        "naumann_lam_field_calibrated",
        "naumann_lam_field_calibrated_v1",
        "naumann_lam_field_calibrated_v2",
    }
    calendar_model = str(cfg["calendar_model"]).strip().lower().replace("-", "_")
    if calendar_model not in valid_calendar_models:
        raise ValueError(f"'calendar_model' must be one of: {', '.join(sorted(valid_calendar_models))}")
    if not isinstance(cfg["start_date"], str):
        raise TypeError("'start_date' must be an ISO date string (YYYY-MM-DD)")
    try:
        date.fromisoformat(cfg["start_date"])
    except ValueError as exc:
        raise ValueError("'start_date' must be a valid ISO date (YYYY-MM-DD)") from exc

    if not isinstance(cfg["enable_resistance_fade"], bool):
        raise TypeError("'enable_resistance_fade' must be a boolean")

    degradation_engine = str(cfg["degradation_engine"]).strip().lower()
    if degradation_engine not in ("native", "blast"):
        raise ValueError("'degradation_engine' must be one of: native, blast")
    cfg["degradation_engine"] = degradation_engine

    if degradation_engine == "blast":
        if cfg["battery_kwh"] <= 0:
            raise ValueError("'degradation_engine=blast' requires 'battery_kwh' > 0")
        if "montecarlo" in cfg:
            raise ValueError("'degradation_engine=blast' is not supported with Monte Carlo yet")
        if cfg["enable_resistance_fade"]:
            raise ValueError("'degradation_engine=blast' cannot be combined with 'enable_resistance_fade'")
        if cfg["blast_model"] not in ENABLED_BLAST_MODEL_KEYS:
            available = ", ".join(ENABLED_BLAST_MODEL_KEYS)
            raise ValueError(f"Unknown blast_model {cfg['blast_model']!r}. Available: {available}")
    elif cfg["blast_model"] is not None:
        raise ValueError("'blast_model' requires 'degradation_engine=blast'; native degradation remains the default")


def _is_int(value: Any) -> bool:
    """Return whether *value* is an integer, excluding booleans."""
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_real(value: Any, key: str) -> float:
    """Return a finite float or raise an actionable public config error."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"'{key}' must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"'{key}' must be a finite number")
    return result


def resolve_location(cfg: dict[str, Any]) -> tuple[float, float, str, str | None]:
    """Resolve a location preset or custom coordinate dict."""
    loc = cfg["location"]
    if isinstance(loc, str):
        locations = load_json("locations.json")
        if loc not in locations:
            available = ", ".join(sorted(locations))
            raise ValueError(f"Unknown location '{loc}'. Available: {available}")
        loc_data = locations[loc]
        return loc_data["latitude"], loc_data["longitude"], loc_data["timezone"], loc
    return loc["latitude"], loc["longitude"], loc["timezone"], None


def normalise_pv_arrays(arrays: list[dict[str, Any]] | None, cfg: dict[str, Any], lat: float) -> list[dict[str, Any]]:
    """Apply App-level PV defaults to each configured PV array."""
    if not arrays:
        return []

    default_module = cfg.get("pv_module") or default_module_key()
    default_tilt = cfg.get("tilt") if cfg.get("tilt") is not None else estimate_optimal_tilt(lat)
    default_azimuth = cfg.get("azimuth") if cfg.get("azimuth") is not None else default_azimuth_fn(lat)

    passthrough_keys = (
        "tracking",
        "axis_tilt",
        "axis_azimuth",
        "max_angle",
        "backtrack",
        "gcr",
        "cross_axis_tilt",
        "dual_axis_max_tilt",
        "transposition_model",
        "albedo",
        "surface_type",
        "model_perez",
        "bifacial_model",
        "pvrow_height",
        "pvrow_pitch",
    )

    normalized: list[dict[str, Any]] = []
    for arr in arrays:
        entry = {
            "modules": int(arr["modules"]),
            "module": arr.get("module") or default_module,
            "tilt": float(arr.get("tilt", default_tilt)),
            "azimuth": float(arr.get("azimuth", default_azimuth)),
        }
        for key in passthrough_keys:
            if key in arr:
                entry[key] = arr[key]
        normalized.append(entry)
    return normalized


def resolve_pv_system(
    cfg: dict[str, Any], lat: float
) -> tuple[list[dict[str, Any]], PVModuleParams, int, float, float, float, float]:
    """Resolve PV module, array, tilt, azimuth, and system sizing details.

    Returns the resolved module count rather than writing it back into ``cfg``;
    the caller materialises it so the dict wrapped by the frozen
    :class:`ResolvedAppConfig` is built once and not mutated in place.
    """
    pv_arrays = normalise_pv_arrays(cfg.get("pv_arrays"), cfg, lat)
    if pv_arrays:
        n_modules = sum(arr["modules"] for arr in pv_arrays)
        total_power_w = sum(arr["modules"] * get_module(arr["module"]).Mpp for arr in pv_arrays)
        avg_module_power_w = total_power_w / n_modules
        system_kwp = total_power_w / 1000
        module_name = pv_arrays[0]["module"]
    else:
        n_modules = cfg["n_modules"]
        module_name = cfg["pv_module"]

    if module_name is None:
        module_name = default_module_key()
    pv_params = get_module(module_name)

    if not pv_arrays:
        avg_module_power_w = pv_params.Mpp
        system_kwp = n_modules * pv_params.Mpp / 1000

    tilt = cfg["tilt"] if cfg["tilt"] is not None else estimate_optimal_tilt(lat)
    azimuth = cfg["azimuth"] if cfg["azimuth"] is not None else default_azimuth_fn(lat)
    return pv_arrays, pv_params, n_modules, avg_module_power_w, system_kwp, tilt, azimuth


def validate_temperature_module_metadata(
    temperature_model: str,
    pv_arrays: list[dict[str, Any]],
    pv_params: PVModuleParams,
) -> None:
    """Validate any module metadata required by the selected thermal model.

    Array configurations may name different modules, so SAM NOCT needs each
    one checked during App config resolution rather than failing after weather
    loading. The thermal kernel repeats this validation for direct solar calls.
    """
    model = normalise_model_name(temperature_model)
    modules = [get_module(arr["module"]) for arr in pv_arrays] if pv_arrays else [pv_params]
    for module in modules:
        validate_temperature_inputs(model, module.Module_Efficiency, module.NOCT)


def resolve_tracking(cfg: dict[str, Any], lat: float) -> tuple[str, float]:
    """Resolve tracker mode and orientation defaults."""
    tracking = cfg["tracking"]
    if tracking not in ("fixed", "single_axis", "dual_axis"):
        raise ValueError(f"tracking must be 'fixed', 'single_axis', or 'dual_axis', got {tracking!r}")
    axis_azimuth = cfg["axis_azimuth"] if cfg["axis_azimuth"] is not None else default_azimuth_fn(lat)
    return tracking, axis_azimuth


def resolve_costs(cfg: dict[str, Any]) -> CostParams:
    """Build CostParams from packaged presets, overrides, and financial defaults.

    Preset keys override the :class:`CostParams` dataclass defaults; a key
    missing from a preset falls back to the same default used when no
    preset is configured, so the two paths cannot diverge.
    """
    params: dict[str, Any] = {}

    if cfg.get("cost_preset"):
        costs_db = load_json("costs.json")
        preset_key = cfg["cost_preset"]
        if preset_key not in costs_db:
            available = ", ".join(sorted(costs_db))
            raise ValueError(f"Unknown cost preset '{preset_key}'. Available: {available}")
        preset = costs_db[preset_key]
        for config_key, param_key in COST_CONFIG_KEY_TO_PARAM.items():
            if config_key in preset:
                params[param_key] = preset[config_key]

    # Explicit values are the final layer: user overrides > named preset >
    # CostParams defaults. Validation has already guaranteed this is a known,
    # finite, non-negative table.
    for config_key, value in cfg.get("costs", {}).items():
        params[COST_CONFIG_KEY_TO_PARAM[config_key]] = value

    params["dc_ac_ratio"] = cfg["inverter_loading_ratio"]
    params.setdefault("inflation_rate", cfg["inflation_rate"])
    params.setdefault("sell_price_inflation", cfg["sell_price_inflation"])
    params.setdefault("discount_rate", cfg["discount_rate"])
    params["pv_degradation_rate"] = cfg["pv_degradation_rate"]

    return CostParams(**params)


def resolve_emissions(cfg: dict[str, Any]) -> EmissionsParams | None:
    """Resolve optional emissions preset."""
    if not cfg["emissions_country"]:
        return None

    emissions_db = load_json("emissions.json")
    key = cfg["emissions_country"]
    if key not in emissions_db:
        available = ", ".join(sorted(emissions_db))
        raise ValueError(f"Unknown emissions country '{key}'. Available: {available}")
    params = dict(emissions_db[key])
    if cfg["export_emissions_factor_gco2_kwh"] is not None:
        params["export_displacement_carbon_intensity_gco2_kwh"] = cfg["export_emissions_factor_gco2_kwh"]
    return EmissionsParams(**params)


def build_costs_dict(cfg: dict[str, Any], resolved: ResolvedAppConfig) -> dict[str, float]:
    """Build the cost-analysis input dictionary for the resolved system."""
    return calculate_costs(
        n_modules=cfg["n_modules"],
        module_power_w=resolved.avg_module_power_w,
        battery_capacity_wh=cfg["battery_kwh"] * 1000,
        cost_params=resolved.cost_params,
    )


def resolve_app_config(config: dict[str, Any]) -> ResolvedAppConfig:
    """Merge, validate, and resolve App configuration."""
    cfg = merge_defaults(config)
    validate_config(cfg)

    lat, lon, timezone, loc_key = resolve_location(cfg)
    pv_arrays, pv_params, n_modules, avg_module_power_w, system_kwp, tilt, azimuth = resolve_pv_system(cfg, lat)
    validate_temperature_module_metadata(cfg["temperature_model"], pv_arrays, pv_params)
    tracking, axis_azimuth = resolve_tracking(cfg, lat)

    # Materialise the resolved module count (derived from pv_arrays when set)
    # into a fresh dict rather than mutating the merged config in place.
    cfg = {**cfg, "n_modules": n_modules}

    return ResolvedAppConfig(
        cfg=cfg,
        lat=lat,
        lon=lon,
        timezone=timezone,
        loc_key=loc_key,
        pv_arrays=pv_arrays,
        pv_params=pv_params,
        avg_module_power_w=avg_module_power_w,
        system_kwp=system_kwp,
        tilt=tilt,
        azimuth=azimuth,
        tracking=tracking,
        axis_azimuth=axis_azimuth,
        cost_params=resolve_costs(cfg),
        emissions_params=resolve_emissions(cfg),
    )
