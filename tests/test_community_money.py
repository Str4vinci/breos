"""ADR 0006 S4, account closure and App's exact pricing conventions."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from breos import App
from breos.community.config import HouseholdConfig, parse_community_config
from breos.community.money import price_community, price_settlement
from breos.community.runner import CommunityRun, SiteRun, iter_settlements, run_community
from breos.community.settlement import MeterBalances, settle_meter_balances
from breos.projection import CarryState, ProjectionRun
from tests.offline_app import CREDIT, FLAT_REFERENCE, TOU
from tests.test_community_runner import offline_deps  # noqa: F401
from tests.test_multi_inverter import TWO_INVERTERS

ZERO_PV_COSTS = {
    "electricity_cost": 0.2,
    "daily_power_cost": 0,
    "module_cost_per_w": 0,
    "inverter_cost_per_kw_simple": 0,
    "installation_cost_per_module": 0,
    "other_cost_per_module": 0,
    "maintenance_cost_per_panel": 0,
}


def tiny_run(*, ownership="external", years=1, pool=True, credit=False, scheduled=False, economics=None, hour=0):
    """K4 arithmetic only: one symbolic quarter-hour per project year."""
    community = {
        "location": "porto",
        "start_date": "2026-01-01",
        "resolution": "15min",
        "projection_years": years,
        "allocation": "fixed",
        "pool_participant_export": pool,
        "shared_energy_price": 0.1,
        "shared_network_charge": 0.02,
        "surplus_price": 0.04,
    }
    if scheduled:
        community.update(schedule=TOU["schedule"], shared_energy_price={"peak": 0.2, "off_peak": 0.1})
    community.update(economics or {})
    participants = [
        {
            "id": key,
            "coefficient": coefficient,
            "annual_consumption_kwh": 1000,
            "costs": {"electricity_cost": 0.2, "electricity_sold_cost": 0.07, "daily_power_cost": 0},
        }
        for key, coefficient in zip(("a", "b", "c"), (0.5, 0.3, 0.2))
    ]
    if credit:
        for participant in participants:
            participant.pop("costs")
            participant["tariff"] = {
                **deepcopy(TOU),
                "import_prices": {"all": 0.2},
                "export_prices": {"all": 0.07},
                "fixed_charge_per_day": 0,
                "annual_network_credit": {
                    "amount_per_year": 1,
                    "network_fixed_per_year": 0,
                    "network_import_prices": {"all": 0.1},
                },
            }
    raw = {
        "community": community,
        "participants": participants,
        "assets": [{"id": "roof", "n_modules": 1, "ownership": ownership, "costs": deepcopy(ZERO_PV_COSTS)}],
    }
    config = parse_community_config(raw)
    index = pd.date_range(f"2026-01-01 {hour:02d}:00", periods=1, freq="15min", tz="Europe/Lisbon")
    amounts = {"a": (1000.0, 0.0), "b": (3000.0, 0.0), "c": (4000.0, 0.0), "roof": (0.0, 6000.0)}
    sites = {}
    definitions = {**{p.id: p.site for p in config.participants}, **{a.id: a.site for a in config.assets}}
    for key, definition in definitions.items():
        imported, exported = amounts[key]
        meters = tuple(MeterBalances(np.array([imported]), np.array([exported])) for _ in range(years))
        load = pd.Series(imported * 4, index=index, name="Houseload")
        projection = None
        if key == "roof":
            rows = pd.DataFrame(
                {
                    "Year": np.arange(1, years + 1),
                    "Load_kWh": 0.0,
                    "Import_kWh": imported / 1000,
                    "Export_kWh": exported / 1000,
                    "PV_Production_kWh": exported / 1000,
                    "PV_Degradation_Factor": 1.0,
                    "Simulated_Hours": 0.25,
                    "Replacements": 0,
                }
            )
            projection = ProjectionRun(rows, CarryState(), 0, None, [])
        tariff = definition.tariff.resolve(index, definition.timezone) if definition.tariff else None
        sites[key] = SiteRun(definition, index, projection, meters, load, tariff, None, {})
    frames, totals = {}, {}
    for year in range(1, years + 1):
        frame = settle_meter_balances(
            {p.id: sites[p.id].meters[year - 1] for p in config.participants},
            {a.id: sites[a.id].meters[year - 1] for a in config.assets},
            index=index,
            allocation=config.community.allocation,
            coefficients=config.community.coefficients,
            pool_participant_export=pool,
        )
        frame.attrs["project_year"] = year
        frames[year] = frame
        totals[year] = frame.drop(columns="Coefficient").groupby(level=["site_type", "site_id"]).sum()
    return CommunityRun(config, sites, frames, totals, {})


def assert_closure(result, *, atol=1e-9):
    for closure in result.interval_closure.values():
        np.testing.assert_allclose(closure, 0, rtol=0, atol=atol)
    np.testing.assert_allclose(
        result.yearly[
            ["Shared_Residual", "Boundary_Residual", "Max_Interval_Shared_Residual", "Max_Interval_Boundary_Residual"]
        ],
        0,
        rtol=0,
        atol=atol,
    )
    for participant in result.participants.values():
        assert abs(participant.benefit_residual) <= atol
        np.testing.assert_allclose(participant.cashflows["Benefit_Residual"], 0, rtol=0, atol=atol)
        assert participant.community_benefit == participant.npv_savings - participant.npv_alone


def test_s4_exact_bills_and_transfers():
    run = tiny_run()
    interval = price_settlement(run, 1, run.settlements[1])
    result = price_community(run)
    b = result.participants["b"]
    assert b.cashflows["Cost_System_Annual"].iloc[0] == pytest.approx(0.456, rel=0, abs=1e-12)
    assert b.alone_cashflows["Cost_System_Annual"].iloc[0] == pytest.approx(0.60, rel=0, abs=1e-12)
    assert result.yearly["Shared_Energy_Payment"].iloc[0] == pytest.approx(0.40, rel=0, abs=1e-12)
    assert result.assets["roof"].cashflows["Revenue_Shared_Energy"].iloc[0] == pytest.approx(0.40, rel=0, abs=1e-12)
    assert result.assets["roof"].cashflows["Revenue_Surplus"].iloc[0] == pytest.approx(0.08, rel=0, abs=1e-12)
    assert result.yearly["Shared_Network_Charge"].iloc[0] == pytest.approx(0.08, rel=0, abs=1e-12)
    assert interval.ledger.xs("roof", level="site_id")["Own_Export_Revenue"].iloc[0] == 0
    assert_closure(result)
    assert not any(str(name).startswith("Cost_No_Sys") for name in result.assets["roof"].cashflows)
    assert not any(str(name).startswith("Baseline_") for name in result.assets["roof"].yearly)
    assert b.cashflows["Benefit_supplier_import_saving"].iloc[0] == pytest.approx(0.36)
    assert b.cashflows["Benefit_shared_energy_payment"].iloc[0] == pytest.approx(-0.18)
    assert b.cashflows["Benefit_shared_network_charge"].iloc[0] == pytest.approx(-0.036)


@pytest.mark.parametrize("ownership", ["external", {"a": 0.25, "b": 0.75}])
def test_ownership_splits_investment_om_imports_fixed_charges_and_income(ownership):
    run = tiny_run(ownership=ownership, years=3)
    roof = run.sites["roof"]
    cfg = deepcopy(roof.config.cfg)
    cfg["costs"].update(module_cost_per_w=1, maintenance_cost=10, daily_power_cost=2)
    from breos.app_config import resolve_costs

    definition = replace(roof.config, cfg=cfg, cost_params=resolve_costs(cfg))
    meters = (MeterBalances([10], [0]), roof.meters[1], roof.meters[2])
    sites = {**run.sites, "roof": replace(roof, config=definition, meters=meters)}
    # Force every frame to rebuild from the changed meter arrays.
    frame = settle_meter_balances(
        {p.id: sites[p.id].meters[0] for p in run.config.participants},
        {"roof": meters[0]},
        index=roof.index,
        allocation="fixed",
        coefficients=run.config.community.coefficients,
    )
    totals = {
        1: frame.drop(columns="Coefficient").groupby(level=["site_type", "site_id"]).sum(),
        **{k: v for k, v in run.settlement_totals.items() if k != 1},
    }
    modified = replace(run, sites=sites, settlements={1: frame}, settlement_totals=totals)
    result = price_community(modified)
    asset = result.assets["roof"]
    assert asset.yearly["Import_Cost"].iloc[0] == pytest.approx(0.002)
    assert asset.yearly["Fixed_Charge"].iloc[0] == pytest.approx(2 * 0.25 / 24)
    for key, participant in result.participants.items():
        share = 0 if ownership == "external" else ownership.get(key, 0)
        assert participant.owned_asset_investment == share * asset.costs["total_initial_cost"]
        assert participant.benefit_parts["asset_ownership"] == -share * asset.net_present_cost
        for name in ("Cost_Import", "Cost_Operation", "Cost_Daily", "Revenue_Shared_Energy", "Revenue_Surplus"):
            np.testing.assert_allclose(
                participant.cashflows[f"Asset_{name}"], share * asset.cashflows[name], rtol=0, atol=1e-12
            )
    assert_closure(result)


def test_network_credit_supplier_cap_and_sharing_line():
    result = price_community(tiny_run(credit=True))
    b = result.participants["b"]
    assert b.yearly["Network_Credit"].iloc[0] == pytest.approx(0.12)
    assert b.alone_cashflows["Cost_Network_Credit"].iloc[0] == pytest.approx(0.30)
    assert b.yearly["Network_Credit_Change"].iloc[0] == pytest.approx(-0.18)
    assert b.yearly["Shared_Network_Charge"].iloc[0] == pytest.approx(0.036)
    assert b.cashflows["Cost_No_Sys_Network_Credit"].iloc[0] == pytest.approx(0.30)
    assert_closure(result)


@pytest.mark.parametrize("hour,price_factor", [(0, 1), (12, 2)])
def test_separate_import_and_sell_escalation_and_scheduled_prices(hour, price_factor):
    run = tiny_run(
        years=3,
        ownership={"a": 1},
        scheduled=True,
        economics={"import_price_escalation": 0.1, "sell_price_inflation": 0.2, "om_escalation": 0.3},
        hour=hour,
    )
    result = price_community(run)
    np.testing.assert_allclose(
        result.participants["b"].cashflows["Cost_Shared_Energy"],
        price_factor * 0.18 * np.array([1, 1.1, 1.21]),
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.assets["roof"].cashflows["Revenue_Shared_Energy"],
        price_factor * 0.4 * np.array([1, 1.1, 1.21]),
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.assets["roof"].cashflows["Revenue_Surplus"], 0.08 * np.array([1, 1.2, 1.44]), rtol=0, atol=1e-12
    )
    assert_closure(result)


@pytest.mark.parametrize("equipment", ["pv", "dc", "ac"])
@pytest.mark.parametrize("billing", ["flat", "tariff", "reference", "credit"])
def test_participant_alone_equals_app_exactly(equipment, billing, offline_deps):
    participant = {"id": "home", "n_modules": 8, "annual_consumption_kwh": 4000}
    if equipment != "pv":
        participant.update(
            battery_kwh=5,
            battery_temperature=21,
            battery_eol_percentage=0.99999,
            battery_allow_terminal_replacement=False,
            battery_replacement_min_remaining_years=0.1,
            battery_replacement_cost=4000,
            terminal_value={"basis": "battery_health_fraction"},
        )
    if equipment == "ac":
        participant.update(
            battery_coupling="ac",
            battery_inverter_ac_rating_kw=2.5,
            battery_inverter_efficiency=0.95,
            costs={"battery_inverter_cost_per_kw": 120},
        )
    if billing != "flat":
        participant["tariff"] = deepcopy(TOU)
    if billing == "reference":
        participant["reference_tariff"] = deepcopy(FLAT_REFERENCE)
    if billing == "credit":
        participant["tariff"]["annual_network_credit"] = deepcopy(CREDIT)
    raw = {
        "community": {
            "location": "porto",
            "start_date": "2026-01-01",
            "resolution": "15min",
            "projection_years": 3,
            "allocation": "proportional",
            "pool_participant_export": False,
            "shared_energy_price": 0.1,
            "shared_network_charge": 0.02,
            "surplus_price": 0.04,
            "inflation_rate": 0.03,
            "import_price_escalation": 0.04,
            "om_escalation": 0.01,
            "sell_price_inflation": 0.02,
            "replacement_cost_learning": 0.05,
            "discount_rate": 0.06,
        },
        "participants": [participant],
    }
    parsed = parse_community_config(raw)
    run = run_community(parsed, deps=offline_deps)
    result = price_community(run)
    app = App({**parsed.community.shared.app_payload(), **{k: v for k, v in participant.items() if k != "id"}})
    app.simulate()
    account = result.participants["home"]
    expected = app._artifacts.cost_projection
    assert account.npv_savings == expected.attrs["final_npv_savings"]
    assert account.npv_alone == expected.attrs["final_npv_savings"]
    assert account.community_benefit == 0
    for column in (
        "Cost_System_Annual",
        "Cost_System_Annual_NPV",
        "Cost_System_Cumulative_NPV",
        "Cost_No_Sys_Annual",
        "Savings_Cumulative_NPV",
        "Cost_Replacement",
        "Replacement_Time_Years",
    ):
        np.testing.assert_array_equal(
            account.cashflows[column].to_numpy().view(np.uint64), expected[column].to_numpy(dtype=float).view(np.uint64)
        )
    if equipment != "pv":
        assert account.costs["replacement_cost_each"] == 4000
        assert account.terminal_health.nominal == app._artifacts.terminal_health.nominal
        assert account.terminal_adjusted_npv == app._artifacts.terminal_health.adjusted_npv
        assert (
            account.terminal_health.provenance["replacement_cost_each_t0_prices"] == 2500
        )  # pack, not bundled replacement
    assert_closure(result, atol=1e-7)


@pytest.mark.parametrize("billing", ["flat", "tariff", "reference", "credit"])
def test_load_only_uses_apps_no_system_billing_exactly(billing, offline_deps):
    participant = {"id": "load", "annual_consumption_kwh": 3500}
    if billing != "flat":
        participant["tariff"] = deepcopy(TOU)
    if billing == "credit":
        participant["tariff"]["annual_network_credit"] = deepcopy(CREDIT)
    if billing == "reference":
        participant["reference_tariff"] = deepcopy(FLAT_REFERENCE)
    raw = {
        "community": {
            "location": "porto",
            "start_date": "2026-01-01",
            "resolution": "15min",
            "projection_years": 2,
            "allocation": "proportional",
            "pool_participant_export": False,
            "shared_energy_price": 0,
            "shared_network_charge": 0,
            "surplus_price": 0,
            "import_price_escalation": 0.04,
            "discount_rate": 0.06,
        },
        "participants": [participant],
    }
    parsed = parse_community_config(raw)
    result = price_community(run_community(parsed, deps=offline_deps))
    account = result.participants["load"]
    assert isinstance(parsed.participants[0].site, HouseholdConfig)
    # App requires PV: compare its no-system bills for the same prepared load.
    app_payload = {
        **parsed.community.shared.app_payload(),
        **{k: v for k, v in participant.items() if k not in ("id", "reference_tariff")},
        "n_modules": 8,
    }
    app = App(app_payload)
    app.simulate()
    baseline = app._artifacts.cost_projection
    np.testing.assert_array_equal(account.cashflows["Cost_System_Annual"], baseline["Cost_No_Sys_Annual"])
    assert account.owned_asset_investment == account.costs["total_initial_cost"] == 0
    if billing == "reference":
        reference_app = App({**app_payload, "reference_tariff": deepcopy(FLAT_REFERENCE)})
        reference_app.simulate()
        baseline = reference_app._artifacts.cost_projection
    np.testing.assert_array_equal(account.cashflows["Cost_No_Sys_Annual_NPV"], baseline["Cost_No_Sys_Annual_NPV"])
    expected = float(
        (baseline["Cost_No_Sys_Annual_NPV"].cumsum() - account.cashflows["Cost_System_Annual_NPV"].cumsum()).iloc[-1]
    )
    assert account.npv_savings == expected
    assert account.community_benefit == 0
    assert_closure(result)


def test_retention_lazy_reconstruction_and_reordering(offline_deps):
    raw = {
        "community": {
            "location": "porto",
            "start_date": "2026-01-01",
            "resolution": "15min",
            "projection_years": 3,
            "allocation": "proportional",
            "shared_energy_price": 0.1,
            "shared_network_charge": 0.02,
            "surplus_price": 0.04,
        },
        "participants": [{"id": "z", "annual_consumption_kwh": 3500}, {"id": "a", "annual_consumption_kwh": 2500}],
        "assets": [{"id": "roof", "n_modules": 8, "ownership": {"a": 0.7, "z": 0.3}, "costs": deepcopy(ZERO_PV_COSTS)}],
    }
    parsed = parse_community_config(raw)
    run = run_community(parsed, deps=offline_deps)
    assert list(run.settlements) == [1]
    assert list(run.settlement_totals) == [1, 2, 3]
    for year, frame in iter_settlements(run):
        pd.testing.assert_frame_equal(
            frame.drop(columns="Coefficient").groupby(level=["site_type", "site_id"]).sum(),
            run.settlement_totals[year],
            check_exact=True,
        )
    result = price_community(run)
    assert list(result.settlements) == list(result.ledgers) == [1]
    assert list(result.interval_closure) == [1]
    full = price_community(run, retain_year_frames=True)
    assert list(full.settlements) == list(full.ledgers) == [1, 2, 3]
    assert list(full.interval_closure) == [1, 2, 3]
    pd.testing.assert_frame_equal(result.yearly, full.yearly, check_exact=True)
    raw["participants"].reverse()
    raw["assets"][0]["ownership"] = {"z": 0.3, "a": 0.7}
    reordered = price_community(run_community(parse_community_config(raw), deps=offline_deps))
    assert reordered.totals == result.totals
    pd.testing.assert_frame_equal(reordered.yearly, result.yearly, check_exact=True)
    for key in result.participants:
        assert reordered.participants[key].benefit_parts == result.participants[key].benefit_parts
    assert_closure(result, atol=1e-7)


@pytest.mark.parametrize("pool", [False, True])
def test_participant_export_sold_once_and_assets_pay_tariff_imports(pool, offline_deps, trina):
    raw = {
        "community": {
            "location": "porto",
            "start_date": "2026-01-01",
            "resolution": "15min",
            "projection_years": 2,
            "allocation": "proportional",
            "pool_participant_export": pool,
            "shared_energy_price": 0.10,
            "shared_network_charge": 0.02,
            "surplus_price": 0.04,
        },
        "participants": [
            {"id": "producer", "n_modules": 20, "annual_consumption_kwh": 1000, "tariff": deepcopy(TOU)},
            {"id": "load", "annual_consumption_kwh": 4000},
        ],
        "assets": [
            {
                "id": "roof",
                **{
                    key: deepcopy(TWO_INVERTERS[key])
                    for key in ("pv_arrays", "inverters", "pv_strings", "electrical_environment", "electrical_fidelity")
                },
                "ownership": {"producer": 0.25, "load": 0.75},
                "tariff": deepcopy(TOU),
            }
        ],
    }
    run = run_community(parse_community_config(raw), deps=offline_deps)
    result = price_community(run, retain_year_frames=True)
    producer = result.participants["producer"]
    assert producer.alone_cashflows["Revenue_Export"].sum() > 0
    if pool:
        assert (producer.cashflows["Revenue_Export"] == 0).all()
        assert producer.cashflows["Revenue_Shared_Energy"].sum() > 0
        assert producer.cashflows["Revenue_Surplus"].sum() > 0
    else:
        np.testing.assert_array_equal(producer.cashflows["Revenue_Export"], producer.alone_cashflows["Revenue_Export"])
        assert (producer.cashflows[["Revenue_Shared_Energy", "Revenue_Surplus"]] == 0).all().all()
    asset = result.assets["roof"]
    assert asset.energy["Metered_Import"].sum() > 0
    assert asset.cashflows["Cost_Import"].sum() > 0
    assert asset.yearly["Fixed_Charge"].iloc[0] == 365 * TOU["fixed_charge_per_day"]
    for year, ledger in result.ledgers.items():
        # Interval and annual weighted tariff totals agree to floating precision.
        first_prices = ledger.xs("roof", level="site_id")["Supplier_Import_Cost"].sum()
        assert asset.yearly["Import_Cost"].iloc[year - 1] == pytest.approx(first_prices, rel=0, abs=1e-10)
    assert_closure(result, atol=1e-7)


def test_network_credit_includes_supplier_fixed_part_but_excludes_shared_network(offline_deps):
    raw = {
        "community": {
            "location": "porto",
            "start_date": "2026-01-01",
            "resolution": "15min",
            "projection_years": 2,
            "allocation": "proportional",
            "shared_energy_price": 0.1,
            "shared_network_charge": 100,
            "surplus_price": 0.04,
        },
        "participants": [
            {
                "id": "home",
                "annual_consumption_kwh": 300,
                "tariff": {**deepcopy(TOU), "annual_network_credit": deepcopy(CREDIT)},
                "reference_tariff": {
                    **deepcopy(FLAT_REFERENCE),
                    "annual_network_credit": deepcopy(CREDIT),
                    "schedule": TOU["schedule"],
                },
            }
        ],
        "assets": [{"id": "roof", "n_modules": 60, "ownership": "external", "costs": deepcopy(ZERO_PV_COSTS)}],
    }
    run = run_community(parse_community_config(raw), deps=offline_deps)
    result = price_community(run)
    account = result.participants["home"]
    assert account.cashflows["Cost_Shared_Network"].iloc[0] > CREDIT["amount_per_year"]
    supplier_wh = result.settlements[1].xs("home", level="site_id")["Supplier_Import"].to_numpy()
    tariff = run.sites["home"].tariff
    expected_charge = (
        float(np.sum(supplier_wh / 1000 * tariff.network_price_per_kwh)) + CREDIT["network_fixed_per_year"]
    )
    assert account.yearly["Network_Charge"].iloc[0] == pytest.approx(expected_charge, abs=1e-9)
    assert account.yearly["Network_Credit"].iloc[0] == pytest.approx(
        min(CREDIT["amount_per_year"], expected_charge), abs=1e-9
    )
    np.testing.assert_array_equal(
        account.cashflows["Cost_No_Sys_Network_Credit"], account.alone_cashflows["Cost_No_Sys_Network_Credit"]
    )
    assert account.cashflows["Network_Credit_Change"].iloc[0] < 0
    np.testing.assert_allclose(
        account.cashflows["Network_Credit_Change"],
        account.cashflows["Cost_Network_Credit"] - account.alone_cashflows["Cost_Network_Credit"],
        rtol=0,
        atol=1e-9,
    )
    assert_closure(result)
