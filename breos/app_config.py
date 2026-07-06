"""Configuration and resource resolution for the public App facade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from breos.economics import CostParams, calculate_costs
from breos.emissions import EmissionsParams
from breos.pv_modules import MODULES, PVModuleParams, get_module
from breos.resources import load_config_json
from breos.solar import (
    DEFAULT_DIFFUSE_IAM,
    DEFAULT_PEREZ_MODEL,
    DEFAULT_SOLAR_POSITION,
    DEFAULT_TRANSPOSITION_MODEL,
    DIFFUSE_IAM_METHODS,
    PEREZ_MODELS,
    SOLAR_POSITION_METHODS,
    SURFACE_TYPES,
    TRANSPOSITION_MODELS,
    estimate_optimal_tilt,
)
from breos.solar import default_azimuth as default_azimuth_fn

DEFAULTS: dict[str, Any] = {
    "battery_kwh": 0.0,
    "pv_arrays": None,
    "pv_module": None,
    "load_profile": "1",
    "rlp_directory": None,
    "tilt": None,
    "azimuth": None,
    "tracking": "fixed",
    "axis_tilt": 0.0,
    "axis_azimuth": None,
    "max_angle": 60.0,
    "backtrack": True,
    "gcr": 0.35,
    "cross_axis_tilt": 0.0,
    "dual_axis_max_tilt": 90.0,
    "transposition_model": DEFAULT_TRANSPOSITION_MODEL,
    "albedo": None,
    "surface_type": None,
    "model_perez": DEFAULT_PEREZ_MODEL,
    "solar_position": DEFAULT_SOLAR_POSITION,
    "diffuse_iam": DEFAULT_DIFFUSE_IAM,
    "resolution": "h",
    "projection_years": 20,
    "cost_preset": None,
    "inflation_rate": 0.02,
    "sell_price_inflation": 0.0,
    "discount_rate": 0.03,
    "emissions_country": None,
    "pv_degradation_rate": 0.005,
    "calendar_model": "naumann_lam_field_calibrated",
    "battery_min_soc": 0.10,
    "battery_max_soc": 0.90,
    "battery_eol_percentage": 0.70,
    "battery_rte": None,
    "dc_coupled": True,
    "inverter_efficiency": 0.96,
    "inverter_loading_ratio": 1.25,
    "pv_loss_overrides": None,
    "start_date": "2023-01-01",
}

# Required inputs are not in DEFAULTS (they have no default); ``montecarlo`` is
# an optional config-file section consumed by the Monte Carlo runner/CLI, which
# validates the same dict through resolve_app_config. Everything else at the top
# level must be a known key so typos (e.g. ``batery_kwh``) fail loudly instead
# of being silently dropped by merge_defaults.
ALLOWED_CONFIG_KEYS: frozenset[str] = frozenset(DEFAULTS) | {
    "location",
    "annual_consumption_kwh",
    "n_modules",
    "montecarlo",
}


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
    """Return a mutable config with App defaults applied."""
    return {**DEFAULTS, **config}


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
    """
    prefix = f"{where}." if where else ""
    if transposition_model is not None and str(transposition_model).strip().lower() not in TRANSPOSITION_MODELS:
        valid = ", ".join(TRANSPOSITION_MODELS)
        raise ValueError(f"'{prefix}transposition_model' must be one of: {valid}")
    if albedo is not None and surface_type is not None:
        raise ValueError(f"Set either '{prefix}albedo' or '{prefix}surface_type', not both.")
    if albedo is not None and (not isinstance(albedo, (int, float)) or not 0 <= albedo <= 1):
        raise ValueError(f"'{prefix}albedo' must be a number between 0 and 1")
    if surface_type is not None and surface_type not in SURFACE_TYPES:
        valid = ", ".join(SURFACE_TYPES)
        raise ValueError(f"'{prefix}surface_type' must be one of: {valid}")
    if model_perez is not None and model_perez not in PEREZ_MODELS:
        valid = ", ".join(PEREZ_MODELS)
        raise ValueError(f"'{prefix}model_perez' must be one of: {valid}")


def validate_config(cfg: dict[str, Any]) -> None:
    """Validate user-facing App config before resolving derived values."""
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
    elif not isinstance(loc, str):
        raise TypeError("'location' must be a string key or a dict with latitude/longitude/timezone")

    if not has_arrays and cfg["n_modules"] < 1:
        raise ValueError("'n_modules' must be >= 1")
    if has_arrays:
        if not isinstance(cfg["pv_arrays"], list):
            raise TypeError("'pv_arrays' must be a list")
        for i, arr in enumerate(cfg["pv_arrays"]):
            if not isinstance(arr, dict):
                raise TypeError(f"'pv_arrays[{i}]' must be a dict")
            modules = arr.get("modules", 0)
            if modules < 1:
                raise ValueError(f"'pv_arrays[{i}].modules' must be >= 1")
            tilt = arr.get("tilt", cfg.get("tilt"))
            azimuth = arr.get("azimuth", cfg.get("azimuth"))
            if tilt is not None and not 0 <= tilt <= 90:
                raise ValueError(f"'pv_arrays[{i}].tilt' must be between 0 and 90")
            if azimuth is not None and not 0 <= azimuth <= 360:
                raise ValueError(f"'pv_arrays[{i}].azimuth' must be between 0 and 360")
            _validate_sky_settings(
                arr.get("transposition_model"),
                arr.get("albedo"),
                arr.get("surface_type"),
                arr.get("model_perez"),
                where=f"pv_arrays[{i}]",
            )
    if cfg["annual_consumption_kwh"] <= 0:
        raise ValueError("'annual_consumption_kwh' must be > 0")
    if cfg["battery_kwh"] < 0:
        raise ValueError("'battery_kwh' must be >= 0")
    tilt = cfg.get("tilt")
    if tilt is not None and not 0 <= tilt <= 90:
        raise ValueError("'tilt' must be between 0 and 90")
    azimuth = cfg.get("azimuth")
    if azimuth is not None and not 0 <= azimuth <= 360:
        raise ValueError("'azimuth' must be between 0 and 360")
    if not 0 < cfg["inverter_efficiency"] <= 1:
        raise ValueError("'inverter_efficiency' must be between 0 (exclusive) and 1 (inclusive)")
    if cfg["inverter_loading_ratio"] <= 0:
        raise ValueError("'inverter_loading_ratio' must be > 0")
    if cfg["projection_years"] < 1:
        raise ValueError("'projection_years' must be >= 1")
    if not 0 <= cfg["pv_degradation_rate"] < 1:
        raise ValueError("'pv_degradation_rate' must be between 0 (inclusive) and 1 (exclusive)")
    if cfg["resolution"] not in ("h", "15min"):
        raise ValueError("'resolution' must be 'h' or '15min'")
    _validate_sky_settings(cfg["transposition_model"], cfg["albedo"], cfg["surface_type"], cfg["model_perez"])
    if str(cfg["solar_position"]).strip().lower() not in SOLAR_POSITION_METHODS:
        valid = ", ".join(SOLAR_POSITION_METHODS)
        raise ValueError(f"'solar_position' must be one of: {valid}")
    if str(cfg["diffuse_iam"]).strip().lower() not in DIFFUSE_IAM_METHODS:
        valid = ", ".join(DIFFUSE_IAM_METHODS)
        raise ValueError(f"'diffuse_iam' must be one of: {valid}")
    overrides = cfg.get("pv_loss_overrides")
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise TypeError("'pv_loss_overrides' must be a dict of loss component percentages")
        for name, value in overrides.items():
            if not isinstance(value, (int, float)) or not 0 <= value <= 100:
                raise ValueError(f"'pv_loss_overrides[{name!r}]' must be a percentage between 0 and 100")
    if not -1 < cfg["sell_price_inflation"] < 1:
        raise ValueError("'sell_price_inflation' must be between -1 and 1 (exclusive)")
    if not 0 <= cfg["battery_min_soc"] < cfg["battery_max_soc"] <= 1:
        raise ValueError("'battery_min_soc' and 'battery_max_soc' must satisfy 0 <= min < max <= 1")
    if not 0 < cfg["battery_eol_percentage"] < 1:
        raise ValueError("'battery_eol_percentage' must be between 0 and 1 (exclusive)")
    if cfg["battery_rte"] is not None and not 0 < cfg["battery_rte"] <= 1:
        raise ValueError("'battery_rte' must be between 0 (exclusive) and 1 (inclusive)")


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

    default_module = cfg.get("pv_module") or next(iter(MODULES))
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
        module_name = next(iter(MODULES))
    pv_params = get_module(module_name)

    if not pv_arrays:
        avg_module_power_w = pv_params.Mpp
        system_kwp = n_modules * pv_params.Mpp / 1000

    tilt = cfg["tilt"] if cfg["tilt"] is not None else estimate_optimal_tilt(lat)
    azimuth = cfg["azimuth"] if cfg["azimuth"] is not None else default_azimuth_fn(lat)
    return pv_arrays, pv_params, n_modules, avg_module_power_w, system_kwp, tilt, azimuth


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
    defaults = CostParams()

    if cfg.get("cost_preset"):
        costs_db = load_json("costs.json")
        preset_key = cfg["cost_preset"]
        if preset_key not in costs_db:
            available = ", ".join(sorted(costs_db))
            raise ValueError(f"Unknown cost preset '{preset_key}'. Available: {available}")
        preset = costs_db[preset_key]

        params["electricity_cost"] = preset.get("electricity_cost", defaults.electricity_cost)
        params["electricity_sold_cost"] = preset.get("electricity_sold_cost", defaults.electricity_sold_cost)
        params["daily_power_cost"] = preset.get("daily_power_cost", defaults.daily_power_cost)
        params["module_cost_per_w"] = preset.get("module_cost_per_w", defaults.module_cost_per_w)
        params["battery_cost_per_kwh"] = preset.get("storage_cost_per_kwh", defaults.battery_cost_per_kwh)
        params["inverter_cost_per_kw"] = preset.get("inverter_cost_per_kw_hybrid", defaults.inverter_cost_per_kw)
        params["inverter_cost_per_kw_nobatt"] = preset.get(
            "inverter_cost_per_kw_simple", defaults.inverter_cost_per_kw_nobatt
        )
        params["installation_cost_per_module"] = preset.get(
            "installation_cost_per_module", defaults.installation_cost_per_module
        )
        params["battery_installation_cost"] = preset.get(
            "installation_cost_battery", defaults.battery_installation_cost
        )
        params["maintenance_cost_per_panel"] = preset.get(
            "maintenance_cost_per_panel", defaults.maintenance_cost_per_panel
        )
        params["maintenance_cost_fixed"] = preset.get("maintenance_cost", defaults.maintenance_cost_fixed)
        params["other_cost_per_module"] = preset.get("other_cost_per_module", defaults.other_cost_per_module)
        params["other_cost_fixed"] = preset.get("other_costs", defaults.other_cost_fixed)

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
    return EmissionsParams(**emissions_db[key])


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
