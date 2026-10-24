"""Public community accounts and JSON-safe run provenance."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
from types import MappingProxyType
from typing import Any, Literal

import pandas as pd

from breos.app_config import ResolvedAppConfig, resolve_location
from breos.app_results import _infinity_as_text
from breos.battery import LEDGER_SCHEMA_VERSION
from breos.community.config import _thaw
from breos.community.money import MONEY_SCHEMA_VERSION, AssetAccount, CommunityMoney, ParticipantAccount
from breos.community.runner import CommunityRun, SiteRun
from breos.economics import projection_rates_record
from breos.electrical import electrical_result
from breos.tariffs import reference_tariff_provenance, tariff_provenance
from breos.utils import package_version


@dataclass(frozen=True)
class CommunityResult:
    """Unrounded community accounts, keyed by stable sorted site ids.

    Site ``energy`` tables are Wh; site ``yearly`` bills are at year-1
    prices and ``cashflows`` contain nominal and discounted money. Community
    ``yearly`` combines nominal accounts, energy and closure diagnostics.
    Whole-asset accounts are already split into participant accounts: do not
    add them to ``totals`` a second time. External assets remain separate.

    ``shared_coverage_of_import`` is coverage of household grid purchases,
    bounded by 100%. ``shared_coverage_of_load`` divides by gross household
    load and can exceed 100% when metered import includes battery grid
    charging. Sharing offsets metered import, including grid charging and
    inverter draw. Both ratios are zero for a zero denominator.

    First-year interval money and residual frames accompany the settlement
    returned by :meth:`breos.Community.settlement`. Every year's energy totals
    and maximum absolute interval residuals remain available. DataFrames and
    provenance are readable/mutable, as App's result dict is; the dataclass
    freezes field assignment, not pandas contents. No summary or serializer
    is added: App's plain result dict has no such methods either.
    """

    currency: str
    participants: Mapping[str, ParticipantAccount]
    assets: Mapping[str, AssetAccount]
    yearly: pd.DataFrame
    totals: Mapping[str, float]
    ownership: Mapping[str, Mapping[str, float] | Literal["external"]]
    settlement_totals: Mapping[int, pd.DataFrame]
    interval_money: pd.DataFrame
    interval_closure: pd.DataFrame
    provenance: dict[str, Any]

    @property
    def shared_coverage_of_import(self) -> pd.Series:
        """Yearly grid-purchase coverage, at most 100%; zero for zero import."""
        return self.yearly["Shared_Coverage_Of_Metered_Import"].copy()

    @property
    def shared_coverage_of_load(self) -> pd.Series:
        """Yearly shared use / gross load; can exceed 100%; zero for zero load."""
        return self.yearly["Shared_Coverage_Of_Load"].copy()


def _json_record(value: Any) -> Any:
    # Match App's public provenance scalar convention, including "inf".
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_record(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {key: _json_record(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_record(child) for child in value]
    return json.loads(json.dumps(_infinity_as_text(_thaw(value)), default=str))


def _site_provenance(site: SiteRun, account: ParticipantAccount | AssetAccount) -> dict[str, Any]:
    resolved = site.config
    cfg = _thaw(resolved.cfg)
    lat, lon, timezone, loc_key = resolve_location(cfg)
    cfg["location"] = {"preset": loc_key, "latitude": lat, "longitude": lon, "timezone": timezone}
    if isinstance(resolved, ResolvedAppConfig):
        cfg.update(tilt=resolved.tilt, azimuth=resolved.azimuth, axis_azimuth=resolved.axis_azimuth)
        cfg["pv_module"] = cfg["pv_module"] or resolved.pv_module_key
    record = {
        **_json_record(site.provenance),
        "breos_version": package_version(),
        "currency": cfg["currency"],
        "resolved_config": _json_record(cfg),
        "resolution": cfg["resolution"],
        "timezone": timezone,
        "start_date": cfg["start_date"],
        "economics": projection_rates_record(cfg),
    }
    year = int(cfg["start_date"][:4])
    if site.tariff is not None:
        record["tariff"] = tariff_provenance(site.tariff, calendar_year=year)
    if site.reference_tariff is not None:
        record["reference_tariff"] = reference_tariff_provenance(
            site.reference_tariff,
            calendar_year=year,
            import_price_escalation=record["economics"]["import_price_escalation"],
        )
    if isinstance(resolved, ResolvedAppConfig):
        record["ledger_schema_version"] = LEDGER_SCHEMA_VERSION
        record["weather"].setdefault("latitude", lat)
        record["weather"].setdefault("longitude", lon)
        record["electrical"] = electrical_result(resolved.electrical)
        record["vehicle"] = None
    if isinstance(account, ParticipantAccount) and account.terminal_health is not None:
        record["terminal_value"] = _json_record(account.terminal_health.provenance)
    return record


def build_result(run: CommunityRun, money: CommunityMoney) -> CommunityResult:
    """Wrap priced accounts and capture provenance before releasing physics."""
    settings = run.config.community
    table = settings.shared.app_payload()
    table.update(
        allocation=settings.allocation,
        coefficients=_thaw(settings.coefficients),
        pool_participant_export=settings.pool_participant_export,
        shared_energy_price=_thaw(settings.shared_energy_price),
        shared_network_charge=_thaw(settings.shared_network_charge),
        surplus_price=_thaw(settings.surplus_price),
        schedule=_json_record(settings.schedule),
        boundary_policy=settings.boundary_policy,
        study_date=settings.study_date,
    )
    ownership = {asset.id: asset.ownership for asset in run.config.assets}
    accounts: dict[str, ParticipantAccount | AssetAccount] = {**money.participants, **money.assets}
    provenance = {
        **deepcopy(dict(run.provenance)),
        "breos_version": package_version(),
        "money_schema_version": MONEY_SCHEMA_VERSION,
        "community": _json_record(table),
        "participant_ids": list(money.participants),
        "asset_ids": list(money.assets),
        "allocation": settings.allocation,
        "ownership": _json_record(ownership),
        "sites": {key: _site_provenance(run.sites[key], accounts[key]) for key in sorted(run.sites)},
        "retention": "first_year_frames_and_all_annual_totals",
    }
    return CommunityResult(
        money.currency,
        money.participants,
        money.assets,
        money.yearly,
        money.totals,
        MappingProxyType(ownership),
        money.settlement_totals,
        money.ledgers[1],
        money.interval_closure[1],
        provenance,
    )
