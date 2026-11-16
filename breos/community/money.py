"""Internal ACC accounts at year-1 prices and App's valuation conventions.

Positive costs are outflows; positive revenues are inflows. Interval ledgers
exclude annual fixed charges, annual network credits and equipment cashflows.
Those are booked in annual accounts, never distributed over arbitrary steps.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from breos.app_config import ResolvedAppConfig, build_costs_dict
from breos.community.config import CommunityConfig
from breos.community.runner import CommunityRun, SiteRun, iter_settlements
from breos.economics import (
    TerminalHealthCredit,
    discount_cashflows,
    price_year_rows,
    resolve_escalation_rates,
    value_year_rows,
)
from breos.projection import price_reference_year_rows, price_tariff_year_rows_by_step, value_projection
from breos.tariffs import ReferenceTariffSpec, TariffPrices

MONEY_SCHEMA_VERSION = "1.0"
INTERVAL_MONEY_COLUMNS = (
    "Supplier_Import_Cost",
    "Own_Export_Revenue",
    "Shared_Energy_Payment",
    "Shared_Network_Charge",
    "Shared_Energy_Receipt",
    "Surplus_Revenue",
    "Net_Variable_Cost",
)
_TRANSFERS = {
    "Shared_Energy_Payment": ("Cost_Shared_Energy", "import"),
    "Shared_Network_Charge": ("Cost_Shared_Network", "import"),
    "Shared_Energy_Receipt": ("Revenue_Shared_Energy", "import"),
    "Surplus_Revenue": ("Revenue_Surplus", "sell"),
}


@dataclass(frozen=True)
class IntervalMoney:
    ledger: pd.DataFrame
    closure: pd.DataFrame


@dataclass(frozen=True)
class AssetAccount:
    """Whole asset: Wh energy, year-1 bills, nominal cashflows and total NPC.

    Equipment CAPEX is at t=0. No no-system baseline or savings NPV exists.
    """

    costs: Mapping[str, float]
    energy: pd.DataFrame
    yearly: pd.DataFrame
    cashflows: pd.DataFrame
    net_present_cost: float


@dataclass(frozen=True)
class ParticipantAccount:
    """Own site plus owned assets, compared with that site alone.

    ``yearly`` contains bills at year-1 prices; ``cashflows`` contains nominal
    annual components and discounted totals. ``benefit_parts`` are NPVs;
    ``Benefit_*`` cashflow columns give their annual nominal counterparts.
    Terminal battery credit is a separate sensitivity, excluded from benefit.
    """

    costs: Mapping[str, float]
    owned_asset_investment: float
    energy: pd.DataFrame
    yearly: pd.DataFrame
    cashflows: pd.DataFrame
    alone_cashflows: pd.DataFrame
    npv_savings: float
    npv_alone: float
    community_benefit: float
    benefit_parts: Mapping[str, float]
    benefit_residual: float
    terminal_health: TerminalHealthCredit | None
    terminal_adjusted_npv: float | None


@dataclass(frozen=True)
class CommunityMoney:
    """Priced accounts, without retaining the physics run or later step arrays.

    ``yearly`` and ``totals`` sum participant accounts including their owned
    asset shares once. ``assets`` separately reports every whole asset,
    external ones included. All monetary values are unrounded in ``currency``.
    """

    config: CommunityConfig
    currency: str
    participants: Mapping[str, ParticipantAccount]
    assets: Mapping[str, AssetAccount]
    yearly: pd.DataFrame
    totals: Mapping[str, float]
    settlements: Mapping[int, pd.DataFrame]
    settlement_totals: Mapping[int, pd.DataFrame]
    ledgers: Mapping[int, pd.DataFrame]
    interval_closure: Mapping[int, pd.DataFrame]


def _community_prices(run: CommunityRun, index: pd.DatetimeIndex) -> tuple[np.ndarray, ...]:
    settings = run.config.community
    timezone = next(iter(run.sites.values())).config.timezone
    out = []
    for values in (settings.shared_energy_price, settings.shared_network_charge, settings.surplus_price):
        spec = ReferenceTariffSpec(
            prices=TariffPrices(settings.shared.currency, values, {"all": 0.0}),
            schedule=settings.schedule,
            boundary_policy=settings.boundary_policy,
            study_date=settings.study_date,
        )
        out.append(np.asarray(spec.resolve(index, timezone).import_price_per_kwh, dtype=float))
    return tuple(out)


def price_settlement(run: CommunityRun, year: int, frame: pd.DataFrame) -> IntervalMoney:
    """Price one settlement in year-1 currency/kWh, with interval closure.

    Supplier prices follow each site's calendar; agreements follow the
    community schedule. Assets receive their own attributed injection income.
    Ownership is applied in the accounts, not to injection attribution.
    """
    index = pd.DatetimeIndex(frame.index.get_level_values("interval").unique())
    shared, network, surplus = _community_prices(run, index)
    ids = sorted(run.sites)
    values = np.zeros((len(index), len(ids), len(INTERVAL_MONEY_COLUMNS)))
    external = {asset.id for asset in run.config.assets if asset.ownership == "external"}
    for position, key in enumerate(ids):
        site = run.sites[key]
        rows = frame.xs(key, level="site_id")
        import_price = site.tariff.import_price_per_kwh if site.tariff else site.config.cost_params.electricity_cost
        export_price = (
            site.tariff.export_price_per_kwh if site.tariff else site.config.cost_params.electricity_sold_cost
        )
        for column, energy, price in (
            (0, "Supplier_Import", import_price),
            (1, "Own_Export", export_price),
            (2, "Shared_Used", shared),
            (3, "Shared_Used", network),
            (4, "Attributed_Shared_Used", shared),
            (5, "Attributed_Surplus", surplus),
        ):
            values[:, position, column] = rows[energy].to_numpy() * price / 1000.0
    values[:, :, 6] = (
        values[:, :, 0] - values[:, :, 1] + values[:, :, 2] + values[:, :, 3] - values[:, :, 4] - values[:, :, 5]
    )
    site_rows = frame.index.get_level_values("site_type") != "community"
    ledger = pd.DataFrame(
        values.reshape(-1, len(INTERVAL_MONEY_COLUMNS)), index=frame.index[site_rows], columns=INTERVAL_MONEY_COLUMNS
    )
    owned = np.array([key not in external for key in ids])
    payments = values[:, :, 2].sum(axis=1)
    receipts = values[:, :, 4].sum(axis=1)
    community_cost = values[:, owned, 6].sum(axis=1)
    boundary = (
        (values[:, owned, 0] - values[:, owned, 1]).sum(axis=1)
        + values[:, :, 3].sum(axis=1)
        - values[:, :, 5].sum(axis=1)
        + (values[:, ~owned, 4] + values[:, ~owned, 5]).sum(axis=1)
    )
    closure = pd.DataFrame(
        {
            "Shared_Payments": payments,
            "Shared_Receipts": receipts,
            "Shared_Residual": payments - receipts,
            "Community_Cost": community_cost,
            "Boundary_Cost": boundary,
            "Boundary_Residual": community_cost - boundary,
        },
        index=index,
    )
    for result in (ledger, closure):
        result.attrs.update(
            money_schema_version=MONEY_SCHEMA_VERSION,
            currency=run.config.community.shared.currency,
            price_basis="year_1",
            project_year=year,
        )
    return IntervalMoney(ledger, closure)


def _costs(site: SiteRun) -> dict[str, float]:
    if isinstance(site.config, ResolvedAppConfig):
        return build_costs_dict(site.config.cfg, site.config)
    # The load-only reference has no equipment, CAPEX, O&M or replacements.
    params = site.config.cost_params
    return {
        "electricity_cost": params.electricity_cost,
        "electricity_sold_cost": params.electricity_sold_cost,
        "daily_power_cost": params.daily_power_cost,
        "total_initial_cost": 0.0,
        "annual_operation_cost": 0.0,
        "replacement_cost_each": 0.0,
    }


def _physical_rows(site: SiteRun) -> pd.DataFrame:
    if site.projection is not None:
        return site.projection.yearly_df.copy()
    load = float(site.load_w.sum()) / 1000.0 * 0.25
    return pd.DataFrame(
        {
            "Year": np.arange(1, len(site.meters) + 1),
            "Load_kWh": load,
            "Import_kWh": load,
            "Export_kWh": 0.0,
            "PV_Production_kWh": 0.0,
            "PV_Degradation_Factor": 1.0,
            "Simulated_Hours": len(site.index) * 0.25,
            "Replacements": 0,
        }
    )


def _bill_row(
    site: SiteRun,
    year: int,
    rows: pd.DataFrame,
    costs: dict[str, float],
    supplier_wh: np.ndarray,
    own_export_wh: np.ndarray,
    *,
    asset: bool,
) -> pd.DataFrame:
    """Reuse App's weighted-power arithmetic, including its flat-price order."""
    row = rows.iloc[[year - 1]].copy()
    supplier_w, own_export_w = supplier_wh * 4.0, own_export_wh * 4.0
    row["Import_kWh"] = float(np.sum(supplier_w)) / 1000.0 * 0.25
    row["Export_kWh"] = float(np.sum(own_export_w)) / 1000.0 * 0.25
    if site.tariff is not None:
        grid_charge = (
            site.projection.priced_flows[year - 1]["Grid_AC_To_Battery"]
            if site.projection is not None and site.projection.priced_flows
            else np.zeros(len(site.index))
        )
        row = price_tariff_year_rows_by_step(
            row,
            [
                {
                    "Import_From_Grid": supplier_w,
                    "PV_AC_Export": own_export_w,
                    "Houseload": site.load_w.to_numpy(),
                    "Grid_AC_To_Battery": grid_charge,
                }
            ],
            site.tariff,
            "15min",
        )
    else:
        row = row.drop(columns=[key for key in ("Import_Cost", "Export_Revenue", "Fixed_Charge") if key in row])
    if site.reference_tariff is not None and not asset:
        row = price_reference_year_rows(row, site.load_w.to_numpy(), site.reference_tariff, "15min")
    row = price_year_rows(row, costs)
    if asset:
        # These zeros are only the valuer's mechanical input; no asset
        # baseline or reference NPV is exposed or added to the community.
        row["Baseline_Import_Cost"] = 0.0
        row["Baseline_Fixed_Charge"] = 0.0
        row = row.drop(columns=[key for key in row if str(key).startswith("Baseline_Network_")])
    return row


def _value(
    rows: pd.DataFrame, costs: dict[str, float], cfg: Mapping[str, Any], *, load_only: bool = False
) -> pd.DataFrame:
    flows = value_year_rows(
        rows,
        costs,
        inflation_rate=cfg["inflation_rate"],
        sell_price_inflation=cfg["sell_price_inflation"],
        import_price_escalation=cfg.get("import_price_escalation"),
        om_escalation=cfg.get("om_escalation"),
        replacement_cost_learning=cfg.get("replacement_cost_learning", 0.0),
    )
    if load_only:
        # App's no-system bill sums import and fixed charges BEFORE escalating.
        # Preserve that arithmetic for households, including their alone bill.
        rate = resolve_escalation_rates(cfg["inflation_rate"], cfg.get("import_price_escalation"))[
            "import_price_escalation"
        ]
        flows["Cost_System_Annual"] = (rows["Import_Cost"] + rows["Fixed_Charge"]) * (1 + rate) ** (rows["Year"] - 1)
        if "Cost_Network_Credit" in flows:
            flows["Cost_System_Annual"] -= flows["Cost_Network_Credit"]
    return flows


def _discount(flows: pd.DataFrame, investment: float, cfg: Mapping[str, Any]) -> pd.DataFrame:
    return discount_cashflows(
        flows, total_investment=investment, discount_rate=cfg["discount_rate"], currency=cfg["currency"]
    )


def _add_transfers(flows: pd.DataFrame, rows: pd.DataFrame, cfg: Mapping[str, Any]) -> None:
    rates = resolve_escalation_rates(
        cfg["inflation_rate"], cfg.get("import_price_escalation"), cfg.get("om_escalation")
    )
    import_factor = (1 + rates["import_price_escalation"]) ** (rows["Year"] - 1)
    sell_factor = (1 + cfg["sell_price_inflation"]) ** (rows["Year"] - 1)
    for source, (target, rate) in _TRANSFERS.items():
        flows[target] = rows[source] * (import_factor if rate == "import" else sell_factor)
    flows["Network_Credit_Change"] = rows["Network_Credit_Change"] * import_factor
    flows["Cost_System_Annual"] = (
        flows["Cost_System_Annual"]
        + flows["Cost_Shared_Energy"]
        + flows["Cost_Shared_Network"]
        - flows["Revenue_Shared_Energy"]
        - flows["Revenue_Surplus"]
    )


def _alone(site: SiteRun, costs: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame, TerminalHealthCredit | None]:
    if isinstance(site.config, ResolvedAppConfig):
        assert site.projection is not None
        value = value_projection(site.config.cfg, site.config, site.projection)
        return value.yearly_df, value.cost_projection, value.terminal_health
    rows = _physical_rows(site)
    billed = pd.concat(
        [
            _bill_row(site, year, rows, costs, np.asarray(meter.import_wh), np.asarray(meter.export_wh), asset=False)
            for year, meter in enumerate(site.meters, 1)
        ],
        ignore_index=True,
    )
    return billed, _discount(_value(billed, costs, site.config.cfg, load_only=True), 0.0, site.config.cfg), None


def _column(frame: pd.DataFrame, name: str) -> np.ndarray:
    return frame[name].to_numpy(dtype=float) if name in frame else np.zeros(len(frame))


def price_community(run: CommunityRun, *, retain_year_frames: bool = False) -> CommunityMoney:
    """Price the community, each participant alone and the no-system baseline.

    Shared payments/receipts and network charges escalate at the shared import
    rate; surplus and own export at the sell rate. Equipment follows App's
    year-end cashflows, replacement instants and pack-only terminal credit.
    External assets' own expenses remain outside participant accounts; their
    injection receipts cross the boundary and are reported explicitly.

    Only first-year settlement/money frames are kept by default. Per-year
    energy, prices and closure totals remain. Interval residuals follow the
    same first-year/all-years retention option as the dense frames.
    Later dense frames are released after pricing; this result holds no
    reference to the physics run, so its raw arrays can also be released.
    """
    ids = sorted(run.sites)
    config = run.config
    shared = config.community.shared
    cfg = shared.app_payload()
    years = shared.projection_years
    costs = {key: _costs(run.sites[key]) for key in ids}
    physical = {key: _physical_rows(run.sites[key]) for key in ids}
    references = {p.id: _alone(run.sites[p.id], costs[p.id]) for p in sorted(config.participants, key=lambda p: p.id)}
    asset_ids = {asset.id for asset in config.assets}
    billed: dict[str, list[pd.DataFrame]] = {key: [] for key in ids}
    energy: dict[str, list[dict[str, float]]] = {key: [] for key in ids}
    settlements, ledgers, residuals = {}, {}, {}
    shared_maxima, boundary_maxima = [], []
    for year, frame in iter_settlements(run):
        invoice = price_settlement(run, year, frame)
        if year == 1 or retain_year_frames:
            settlements[year], ledgers[year] = frame, invoice.ledger
            residuals[year] = invoice.closure[["Shared_Residual", "Boundary_Residual"]].copy()
        shared_maxima.append(float(invoice.closure["Shared_Residual"].abs().max()))
        boundary_maxima.append(float(invoice.closure["Boundary_Residual"].abs().max()))
        for key in ids:
            site = run.sites[key]
            site_frame = frame.xs(key, level="site_id")
            row = _bill_row(
                site,
                year,
                physical[key],
                costs[key],
                site_frame["Supplier_Import"].to_numpy(),
                site_frame["Own_Export"].to_numpy(),
                asset=key in asset_ids,
            )
            money = invoice.ledger.xs(key, level="site_id")
            for name in _TRANSFERS:
                row[name] = float(money[name].sum())
            old_credit = (
                float(references[key][0].iloc[year - 1].get("Network_Credit", 0.0)) if key in references else 0.0
            )
            row["Network_Credit_Change"] = (
                float(row.iloc[0].get("Network_Credit", 0.0)) - old_credit if key in references else 0.0
            )
            billed[key].append(row)
            sums = {str(name): float(site_frame[name].sum()) for name in site_frame.columns if name != "Coefficient"}
            energy[key].append(
                {
                    "Year": year,
                    **sums,
                    "Load_Wh": float(physical[key].iloc[year - 1]["Load_kWh"]) * 1000,
                    "Own_PV_Production_Wh": float(physical[key].iloc[year - 1]["PV_Production_kWh"]) * 1000,
                }
            )
        del invoice, frame, site_frame, money
    rows = {key: pd.concat(billed[key], ignore_index=True) for key in ids}
    assets = {}
    for key in sorted(asset_ids):
        flows = _value(rows[key], costs[key], cfg)
        _add_transfers(flows, rows[key], cfg)
        cashflows = _discount(flows, costs[key]["total_initial_cost"], cfg)
        # Assets have no reference/no-system account. Keep only cost fields.
        cashflows = cashflows.drop(
            columns=[name for name in cashflows if str(name).startswith(("Cost_No_Sys", "Savings_"))]
        )
        cashflows.attrs.pop("final_npv_savings", None)
        cashflows.attrs.pop("payback_year", None)
        assets[key] = AssetAccount(
            MappingProxyType(costs[key]),
            pd.DataFrame(energy[key]),
            rows[key].drop(columns=[name for name in rows[key] if str(name).startswith("Baseline_")]),
            cashflows,
            float(cashflows["Cost_System_Cumulative_NPV"].iloc[-1]),
        )
    participants = {}
    discounts = (1 + shared.discount_rate) ** -np.arange(1, years + 1, dtype=float)
    import_rate = resolve_escalation_rates(shared.inflation_rate, shared.import_price_escalation)[
        "import_price_escalation"
    ]
    import_factors = (1 + import_rate) ** np.arange(years)
    sell_factors = (1 + shared.sell_price_inflation) ** np.arange(years)
    ownership = {asset.id: asset.ownership for asset in config.assets}
    for key in sorted(references):
        flows = _value(rows[key], costs[key], cfg, load_only=run.sites[key].projection is None)
        _add_transfers(flows, rows[key], cfg)
        owned_investment, owned_npc = 0.0, 0.0
        owned_annual = np.zeros(years)
        for asset_id in sorted(assets):
            shares = ownership[asset_id]
            share = 0.0 if shares == "external" else shares.get(key, 0.0)
            owned_investment += share * costs[asset_id]["total_initial_cost"]
            owned_npc += share * assets[asset_id].net_present_cost
            owned_annual += share * assets[asset_id].cashflows["Cost_System_Annual"].to_numpy()
            for name in (
                "Cost_Import",
                "Cost_Daily",
                "Cost_Operation",
                "Cost_Network_Credit",
                "Revenue_Shared_Energy",
                "Revenue_Surplus",
            ):
                target = f"Asset_{name}"
                flows[target] = _column(flows, target) + share * _column(assets[asset_id].cashflows, name)
        flows["Asset_Cost_Annual"] = owned_annual
        flows["Cost_System_Annual"] += owned_annual
        cashflows = _discount(flows, costs[key]["total_initial_cost"] + owned_investment, cfg)
        alone_rows, alone_flows, terminal = references[key]
        npv = float(cashflows.attrs["final_npv_savings"])
        npv_alone = float(alone_flows.attrs["final_npv_savings"])
        benefit = npv - npv_alone
        annual_parts = {
            "supplier_import_saving": (alone_rows["Import_Cost"].to_numpy() - rows[key]["Import_Cost"].to_numpy())
            * import_factors,
            "own_export_revenue_change": (
                rows[key]["Export_Revenue"].to_numpy() - alone_rows["Export_Revenue"].to_numpy()
            )
            * sell_factors,
            "shared_energy_payment": -flows["Cost_Shared_Energy"].to_numpy(),
            "shared_network_charge": -flows["Cost_Shared_Network"].to_numpy(),
            "shared_energy_receipt": flows["Revenue_Shared_Energy"].to_numpy(),
            "surplus_receipt": flows["Revenue_Surplus"].to_numpy(),
            "network_credit_change": flows["Network_Credit_Change"].to_numpy(),
            "asset_ownership": -owned_annual,
        }
        parts = {name: float(np.sum(values * discounts)) for name, values in annual_parts.items()}
        parts["asset_ownership"] = -owned_npc
        annual_change = alone_flows["Cost_System_Annual"].to_numpy() - cashflows["Cost_System_Annual"].to_numpy()
        for name, values in annual_parts.items():
            cashflows[f"Benefit_{name}"] = values
        cashflows["Benefit_Residual"] = annual_change - sum(annual_parts.values(), np.zeros(years))
        terminal = replace(terminal, adjusted_npv=npv + terminal.npv) if terminal is not None else None
        participants[key] = ParticipantAccount(
            MappingProxyType(costs[key]),
            owned_investment,
            pd.DataFrame(energy[key]),
            rows[key],
            cashflows,
            alone_flows,
            npv,
            npv_alone,
            benefit,
            MappingProxyType(parts),
            benefit - sum(parts.values()),
            terminal,
            terminal.adjusted_npv if terminal is not None else None,
        )
    all_accounts: dict[str, ParticipantAccount | AssetAccount] = {**participants, **assets}

    def total(column: str, keys: list[str], *, alone: bool = False) -> np.ndarray:
        return sum(
            (
                _column(participants[key].alone_cashflows if alone else all_accounts[key].cashflows, column)
                for key in keys
            ),
            np.zeros(years),
        )

    pids = sorted(participants)
    aids = sorted(assets)
    external_ids = [key for key in aids if ownership[key] == "external"]

    def owned_asset_total(column: str) -> np.ndarray:
        return total(f"Asset_{column}", pids)

    payments = total("Cost_Shared_Energy", pids)
    receipts = total("Revenue_Shared_Energy", sorted(all_accounts))
    network = total("Cost_Shared_Network", pids)
    surplus = total("Revenue_Surplus", sorted(all_accounts))
    external_receipts = total("Revenue_Shared_Energy", external_ids) + total("Revenue_Surplus", external_ids)
    supplier = sum(
        (total(name, pids) + owned_asset_total(name)) * sign
        for name, sign in (("Cost_Import", 1), ("Cost_Daily", 1), ("Revenue_Export", -1), ("Cost_Network_Credit", -1))
    )
    operation = total("Cost_Operation", pids) + owned_asset_total("Cost_Operation")
    replacements = total("Cost_Replacement", pids)
    boundary = supplier + operation + replacements + network - surplus + external_receipts
    annual = pd.DataFrame(
        {
            "Year": np.arange(1, years + 1),
            "Cost_System_Annual": total("Cost_System_Annual", pids),
            "Cost_Alone_Annual": total("Cost_System_Annual", pids, alone=True),
            "Cost_No_Sys_Annual": total("Cost_No_Sys_Annual", pids),
            "Shared_Energy_Payment": payments,
            "Shared_Energy_Receipt": receipts,
            "Shared_Residual": payments - receipts,
            "Shared_Network_Charge": network,
            "Surplus_Revenue": surplus,
            "External_Owner_Receipts": external_receipts,
            "Supplier_Net_Cost": supplier,
            "Equipment_Operation_Cost": operation,
            "Replacement_Cost": replacements,
            "Network_Credit_Change": total("Network_Credit_Change", pids),
            "Boundary_Cost": boundary,
            "Boundary_Residual": total("Cost_System_Annual", pids) - boundary,
            "Max_Interval_Shared_Residual": shared_maxima,
            "Max_Interval_Boundary_Residual": boundary_maxima,
        }
    )
    for source in ("Energy_For_Sharing", "Shared_Used_Total", "Community_Surplus"):
        annual[f"{source}_Wh"] = [
            run.settlement_totals[y].loc[("community", ""), source] for y in run.settlement_totals
        ]
    load_wh = sum((participants[key].energy["Load_Wh"].to_numpy() for key in pids), np.zeros(years))
    import_wh = sum((participants[key].energy["Metered_Import"].to_numpy() for key in pids), np.zeros(years))
    used = annual["Shared_Used_Total_Wh"].to_numpy()
    annual["Shared_Coverage_Of_Load"] = np.divide(used, load_wh, out=np.zeros(years), where=load_wh > 0)
    annual["Shared_Coverage_Of_Metered_Import"] = np.divide(used, import_wh, out=np.zeros(years), where=import_wh > 0)
    npv = sum(participants[key].npv_savings for key in pids)
    alone_npv = sum(participants[key].npv_alone for key in pids)
    totals = {
        "npv_savings": npv,
        "npv_alone": alone_npv,
        "community_benefit": npv - alone_npv,
        "terminal_health_credit_npv": sum(
            account.terminal_health.npv for account in participants.values() if account.terminal_health is not None
        ),
        "initial_investment": sum(participants[key].cashflows.attrs["total_investment"] for key in pids),
        "no_system_cost_npv": sum(
            float(participants[key].cashflows["Cost_No_Sys_Cumulative_NPV"].iloc[-1]) for key in pids
        ),
        "alone_cost_npv": sum(
            float(participants[key].alone_cashflows["Cost_System_Cumulative_NPV"].iloc[-1]) for key in pids
        ),
        "community_cost_npv": sum(
            float(participants[key].cashflows["Cost_System_Cumulative_NPV"].iloc[-1]) for key in pids
        ),
        "external_asset_net_present_cost": sum(assets[key].net_present_cost for key in external_ids),
    }
    totals["terminal_adjusted_npv"] = npv + totals["terminal_health_credit_npv"]
    annual.attrs.update(currency=shared.currency, money_schema_version=MONEY_SCHEMA_VERSION, price_basis="nominal_year")
    return CommunityMoney(
        config,
        shared.currency,
        MappingProxyType(participants),
        MappingProxyType(assets),
        annual,
        MappingProxyType(totals),
        MappingProxyType(settlements),
        run.settlement_totals,
        MappingProxyType(ledgers),
        MappingProxyType(residuals),
    )
