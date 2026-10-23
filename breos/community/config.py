"""Internal ACC configuration, separate from App's registry (ADR 0006).

PV participants use App's resolver unchanged. Assets use it to resolve their
PV equipment with a validation-only positive load, then replace that load by
zero: the community runner must build their zero-load series. Load-only
households reuse App's focused validators without inventing PV equipment.
Vehicles, partial-year runs and community storage are not supported here.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from datetime import date
from types import MappingProxyType
from typing import Any, Literal, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from breos.app_config import (
    ALLOWED_CONFIG_KEYS,
    ResolvedAppConfig,
    _check_fields,
    _normalise_config_values,
    _validate_battery_and_degradation,
    _validate_economics,
    _validate_load_profile,
    _validate_time_and_weather,
    check_cost_currency,
    merge_defaults,
    normalize_config_keys,
    resolve_app_config,
    resolve_costs,
    resolve_location,
    resolve_reference_tariff_spec,
    resolve_tariff_spec,
)
from breos.config_schema import boolean, choice, number
from breos.economics import CostParams
from breos.tariffs import PriceList, ReferenceTariffSpec, ScheduleDefinition, TariffSpec, run_currency

# K3: location includes the timezone; weather is selected once. Equipment's
# PV degradation is site-specific, but all monetary escalators are shared.
SHARED_APP_KEYS = frozenset(
    {
        "location",
        "weather_file",
        "weather_source",
        "start_date",
        "resolution",
        "projection_years",
        "currency",
        "execution_backend",
        "discount_rate",
        "inflation_rate",
        "import_price_escalation",
        "om_escalation",
        "replacement_cost_learning",
        "sell_price_inflation",
    }
)
_PRICE_KEYS = ("shared_energy_price", "shared_network_charge", "surplus_price")
_SCHEDULE_KEYS = frozenset({"schedule", "custom_schedule", "boundary_policy", "study_date"})
_COMMUNITY_KEYS = SHARED_APP_KEYS | _SCHEDULE_KEYS | {*_PRICE_KEYS, "allocation", "pool_participant_export"}
_LOAD_KEYS = frozenset(
    {
        "annual_consumption_kwh",
        "load_profile",
        "load_profile_file",
        "load_profile_column",
        "load_profile_unit",
        "rlp_directory",
    }
)
_BATTERY_KEYS = frozenset(key for key in ALLOWED_CONFIG_KEYS if key.startswith("battery_")) | {
    "battery",
    "calendar_model",
    "degradation_engine",
    "blast_model",
    "enable_resistance_fade",
    "terminal_value",
}
_HOUSEHOLD_KEYS = _LOAD_KEYS | {"costs", "cost_preset", "tariff", "reference_tariff"}


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(value[key]) for key in sorted(value)})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(child) for child in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(child) for child in value]
    return value


def _table(value: Any, where: str) -> dict[str, Any]:
    """Normalise this table's field names only, never identity/period maps."""
    if not isinstance(value, Mapping):
        raise TypeError(f"'{where}' must be a table/mapping")
    result = {}
    for key, child in value.items():
        if not isinstance(key, str):
            raise TypeError(f"'{where}' must use string keys")
        name = key.replace("-", "_")
        if name in result:
            raise ValueError(f"'{where}': duplicate key '{name}' after key normalisation")
        result[name] = child
    return result


def _unknown(table: Mapping[str, Any], allowed: frozenset[str], where: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ValueError(f"'{where}': unknown key(s): {', '.join(unknown)}")


@dataclass(frozen=True)
class SharedSettings:
    """K3's common clock, weather, currency and economics; App defaults apply."""

    location: str | Mapping[str, Any]
    weather_file: str | None
    weather_source: str | None
    start_date: str
    resolution: Literal["15min"]
    projection_years: int
    currency: str
    execution_backend: str
    discount_rate: float
    inflation_rate: float
    import_price_escalation: float | None
    om_escalation: float | None
    replacement_cost_learning: float
    sell_price_inflation: float

    def app_payload(self) -> dict[str, Any]:
        """A fresh shared layer for an App resolver or the load-only builder."""
        return {field.name: _thaw(getattr(self, field.name)) for field in fields(self)}


@dataclass(frozen=True)
class CommunitySettings:
    shared: SharedSettings
    allocation: Literal["fixed", "proportional"]
    coefficients: Mapping[str, float] | None
    pool_participant_export: bool
    shared_energy_price: PriceList
    shared_network_charge: PriceList
    surplus_price: PriceList
    schedule: ScheduleDefinition | None
    boundary_policy: str
    study_date: date | None


@dataclass(frozen=True)
class HouseholdConfig:
    """A load-only site, with no synthetic PV system or battery dispatch.

    ``cfg`` contains shared and load/billing keys only. The runner builds the
    15-minute load from it, and uses that load as import with zero export.
    """

    cfg: Mapping[str, Any]
    tariff: TariffSpec | None
    reference_tariff: ReferenceTariffSpec | None
    cost_params: CostParams
    timezone: str


@dataclass(frozen=True)
class ParticipantConfig:
    id: str
    coefficient: float | None
    site: ResolvedAppConfig | HouseholdConfig


@dataclass(frozen=True)
class AssetConfig:
    id: str
    ownership: Mapping[str, float] | Literal["external"]
    site: ResolvedAppConfig


@dataclass(frozen=True)
class CommunityConfig:
    community: CommunitySettings
    participants: tuple[ParticipantConfig, ...]
    assets: tuple[AssetConfig, ...]


def _shared_settings(raw: Mapping[str, Any]) -> SharedSettings:
    if "location" not in raw:
        raise ValueError("'community' needs location")
    if raw.get("resolution") != "15min":
        raise ValueError("'community.resolution' must be '15min'; ACC requires 15-minute intervals")
    payload = normalize_config_keys({key: _thaw(raw[key]) for key in SHARED_APP_KEYS if key in raw})
    cfg = _normalise_config_values(merge_defaults(payload))
    _check_fields(cfg)
    location = cfg["location"]
    if isinstance(location, dict):
        for name, low, high in (("latitude", -90, 90), ("longitude", -180, 180)):
            if name not in location:
                raise ValueError(f"'community.location' needs {name}")
            number(minimum=low, maximum=high)(location[name], f"community.location.{name}")
        try:
            timezone = location["timezone"]
            if not isinstance(timezone, str):
                raise TypeError("'community.location.timezone' must be an IANA timezone string")
            ZoneInfo(timezone)
        except (KeyError, ZoneInfoNotFoundError) as exc:
            raise ValueError("'community.location.timezone' must name an IANA timezone") from exc
    elif not isinstance(location, str):
        raise TypeError("'community.location' must be a preset or a location table")
    resolve_location(cfg)
    _validate_time_and_weather(cfg)
    _validate_economics(cfg)
    _validate_battery_and_degradation(cfg)  # App owns date/backend validation too.
    cfg["currency"] = run_currency(cfg["currency"], None)
    return SharedSettings(**{key: _freeze(cfg[key]) for key in SHARED_APP_KEYS})


def _prices(raw: Mapping[str, Any], shared: SharedSettings) -> tuple[list[PriceList], ReferenceTariffSpec]:
    prices = []
    last_spec = None
    cfg = shared.app_payload()
    timezone = resolve_location(cfg)[2]
    for key in _PRICE_KEYS:
        if key not in raw:
            raise ValueError(f"'community' needs {key}; an explicit 0 is valid")
        value = raw[key]
        if not isinstance(value, Mapping):
            value = {"all": number(minimum=0)(value, f"community.{key}")}
        tariff = {
            "currency": shared.currency,
            "import_prices": _thaw(value),
            "fixed_charge_per_day": 0.0,
            **{name: _thaw(raw[name]) for name in _SCHEDULE_KEYS if name in raw},
        }
        try:
            spec = resolve_reference_tariff_spec({**cfg, "reference_tariff": tariff}, timezone, shared.currency)
        except (TypeError, ValueError) as exc:
            raise type(exc)(str(exc).replace("reference_tariff", f"community.{key}")) from exc
        assert spec is not None
        prices.append(_freeze(spec.prices.import_prices))
        last_spec = spec
    assert last_spec is not None
    return prices, last_spec


def _sites(value: Any, where: str, seen: set[str]) -> dict[str, dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"'{where}' must be an array of site tables")
    sites = {}
    for position, entry in enumerate(value):
        label = f"{where}[{position}]"
        if isinstance(entry, Mapping) and isinstance(entry.get("id"), str):
            label = f"site '{entry['id']}'"
        row = _table(entry, label)
        site_id = row.pop("id", None)
        if not isinstance(site_id, str) or re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", site_id) is None:
            raise ValueError(f"'{where}[{position}].id': {site_id!r} must match ^[a-z0-9][a-z0-9_-]{{0,63}}$")
        if site_id in seen:
            raise ValueError(f"site '{site_id}': ids must be unique across participants and assets")
        seen.add(site_id)
        sites[site_id] = row
    return {key: sites[key] for key in sorted(sites)}


def _site_payload(row: dict[str, Any], *, asset: bool) -> dict[str, Any]:
    # Community metadata is already removed. Only this App payload is allowed
    # through App's recursive normalisation (ownership ids must stay verbatim).
    payload = normalize_config_keys({key: _thaw(value) for key, value in row.items()})
    repeated = sorted(set(payload) & SHARED_APP_KEYS)
    if repeated:
        raise ValueError(f"shared key(s) {', '.join(repeated)} belong in [community], even when values agree")
    if "vehicle" in payload:
        raise ValueError("[vehicle] is not supported in community sites yet; its ACC semantics need maintainer review")
    for key in ("period", "montecarlo", "sweep"):
        if key in payload:
            raise ValueError(f"'{key}' is not supported in community sites")
    reference = payload.get("reference_tariff")
    if isinstance(reference, Mapping) and "import_price_escalation" in reference:
        raise ValueError(
            "reference_tariff.import_price_escalation is shared; set import_price_escalation in [community]"
        )
    if asset:
        forbidden = sorted(set(payload) & (_BATTERY_KEYS | _LOAD_KEYS | {"load", "smart_charging", "reference_tariff"}))
        if forbidden:
            raise ValueError(
                f"PV-only assets refuse battery, load, smart-charging and reference keys: {', '.join(forbidden)}"
            )
        tariff = payload.get("tariff")
        if tariff is not None:
            if not isinstance(tariff, Mapping) or not {"import_prices", "fixed_charge_per_day"} <= set(tariff):
                raise ValueError("asset tariff needs explicit import_prices and fixed_charge_per_day; 0 is valid")
        else:
            costs = payload.get("costs")
            if not isinstance(costs, Mapping) or not {"electricity_cost", "daily_power_cost"} <= set(costs):
                raise ValueError("asset needs explicit costs.electricity_cost and costs.daily_power_cost; 0 is valid")
    _unknown(payload, ALLOWED_CONFIG_KEYS, "App payload")
    return payload


def _household(payload: dict[str, Any], shared: SharedSettings) -> HouseholdConfig:
    _unknown(payload, _HOUSEHOLD_KEYS, "load-only household (no PV)")
    if "annual_consumption_kwh" not in payload:
        raise ValueError("load-only household needs annual_consumption_kwh")
    cfg = _normalise_config_values(merge_defaults({**shared.app_payload(), **payload}))
    _check_fields(cfg)
    _validate_load_profile(cfg)
    _validate_economics(cfg)
    timezone = resolve_location(cfg)[2]
    tariff = resolve_tariff_spec(cfg, timezone)
    run_currency(shared.currency, tariff.prices.currency if tariff is not None else None)
    reference = resolve_reference_tariff_spec(cfg, timezone, shared.currency)
    # A load-only household prices no PV equipment; only the flat supplier
    # prices would otherwise leak EUR defaults into another currency.
    if cfg.get("cost_preset"):
        check_cost_currency(
            shared.currency,
            given=cfg.get("costs") or {},
            preset=cfg["cost_preset"],
            tariff=tariff is not None,
            battery=False,
        )
    elif shared.currency != "EUR" and tariff is None:
        missing = {"electricity_cost", "daily_power_cost"} - set(cfg.get("costs") or {})
        if missing:
            raise ValueError(
                f"load-only household in {shared.currency} needs explicit costs: {', '.join(sorted(missing))}"
            )
    kept = SHARED_APP_KEYS | _HOUSEHOLD_KEYS
    return HouseholdConfig(
        _freeze({key: value for key, value in cfg.items() if key in kept}),
        tariff,
        reference,
        resolve_costs(cfg),
        timezone,
    )


def _ownership(value: Any, participants: set[str]) -> Mapping[str, float] | Literal["external"]:
    if value == "external":
        return "external"
    if not isinstance(value, Mapping) or not value:
        raise ValueError("ownership must be participant shares or 'external'")
    if any(not isinstance(key, str) or key not in participants for key in value):
        raise ValueError("ownership shares must name existing participant ids verbatim")
    shares = {key: number(minimum=0)(value[key], f"ownership.{key}") for key in sorted(value)}
    if not math.isclose(sum(shares.values()), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("ownership shares must sum to 1 (atol=1e-12)")
    return _freeze(shares)


def parse_community_config(config: Mapping[str, Any]) -> CommunityConfig:
    """Parse TOML-shaped community/participants/assets into frozen dataclasses.

    Site collections and all identity maps are canonicalised in sorted id
    order. Agreements are required, with no inferred prices or ownership.
    Fixed shares use settlement's absolute sum tolerance of 1e-12.
    Asset fixed charges are always explicit: state 0 for no supply contract.
    App's resolved dataclasses are retained for PV sites; as in App, their
    nested runtime dictionaries are owned copies, not immutable value objects.
    """
    root = _table(config, "config")
    _unknown(root, frozenset({"community", "participants", "assets"}), "config")
    raw = _table(root.get("community"), "community")
    _unknown(raw, _COMMUNITY_KEYS, "community")
    if "allocation" not in raw:
        raise ValueError("'community' needs allocation")
    allocation = choice(("fixed", "proportional"), case_insensitive=False)(raw["allocation"], "community.allocation")
    shared = _shared_settings(raw)
    prices, price_spec = _prices(raw, shared)
    pool = boolean(raw.get("pool_participant_export", True), "community.pool_participant_export")
    seen: set[str] = set()
    participant_rows = _sites(root.get("participants"), "participants", seen)
    if not participant_rows:
        raise ValueError("community needs at least one participant")
    asset_rows = _sites(root.get("assets", []), "assets", seen)
    coefficients: dict[str, float] = {}
    participants = []
    for site_id, original in participant_rows.items():
        try:
            row = dict(original)
            if allocation == "fixed":
                if "coefficient" not in row:
                    raise ValueError("fixed allocation needs coefficient")
                coefficient = number(minimum=0)(row.pop("coefficient"), "coefficient")
                coefficients[site_id] = coefficient
            else:
                if "coefficient" in row:
                    raise ValueError("proportional allocation does not take coefficient")
                coefficient = None
            payload = _site_payload(row, asset=False)
            site: ResolvedAppConfig | HouseholdConfig
            if "n_modules" in payload or "pv_arrays" in payload:
                site = resolve_app_config({**shared.app_payload(), **payload})
            else:
                site = _household(payload, shared)
            participants.append(ParticipantConfig(site_id, coefficient, site))
        except (TypeError, ValueError, OSError) as exc:
            raise type(exc)(f"participant '{site_id}': {exc}") from exc
    if allocation == "fixed" and not math.isclose(sum(coefficients.values()), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"participant coefficients ({', '.join(coefficients)}) must sum to 1 (atol=1e-12)")
    assets = []
    for site_id, original in asset_rows.items():
        try:
            row = dict(original)
            ownership = _ownership(row.pop("ownership", None), set(participant_rows))
            payload = _site_payload(row, asset=True)
            site = resolve_app_config({**shared.app_payload(), **payload, "annual_consumption_kwh": 1.0})
            site = replace(site, cfg={**site.cfg, "annual_consumption_kwh": 0.0, "load_profile": None})
            assets.append(AssetConfig(site_id, ownership, site))
        except (TypeError, ValueError, OSError) as exc:
            raise type(exc)(f"asset '{site_id}': {exc}") from exc
    schedule = price_spec.schedule
    assert schedule is None or isinstance(schedule, ScheduleDefinition)
    community = CommunitySettings(
        shared,
        cast(Literal["fixed", "proportional"], allocation),
        _freeze(coefficients) if allocation == "fixed" else None,
        pool,
        prices[0],
        prices[1],
        prices[2],
        schedule,
        price_spec.boundary_policy,
        price_spec.study_date,
    )
    return CommunityConfig(community, tuple(participants), tuple(assets))
