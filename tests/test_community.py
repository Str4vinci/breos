"""The public ACC rollout gate: accounts, provenance and offline example."""

import json
import tomllib
import weakref
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import breos
from breos import App, Community
from breos.app_config import ALLOWED_CONFIG_KEYS, DEFAULTS
from breos.community import facade
from breos.community.results import CommunityResult
from tests.community_conservation import assert_settlement_closure
from tests.offline_app import FLAT_REFERENCE, TOU, offline, simulate_offline
from tests.test_community_runner import assert_bits

EXAMPLE = Path(__file__).resolve().parents[1] / "configs/examples/community-portugal.toml"


@pytest.fixture
def config():
    with EXAMPLE.open("rb") as stream:
        payload = tomllib.load(stream)
    payload["community"]["projection_years"] = 2
    return payload


def simulate(config):
    fetch, weather = offline()
    with fetch, weather:
        community = Community(config)
        community.simulate()
    return community


@pytest.mark.parametrize("method", ["result", "settlement"])
def test_lifecycle_and_public_surface(config, method):
    community = Community(config)
    assert breos.Community is Community and "Community" in breos.__all__
    assert "CommunityResult" not in breos.__all__  # App has no exported result type either.
    with pytest.raises(RuntimeError, match=rf"^Call simulate\(\) before {method}\(\)\.$"):
        getattr(community, method)()
    for unsupported in ("revalue", "montecarlo", "optimize", "summary", "to_dict"):
        assert not hasattr(community, unsupported)
    with pytest.raises(TypeError, match="table/mapping"):
        Community(EXAMPLE)


def test_example_offline_closure_and_result_fields(config, monkeypatch):
    original, defaults, allowed = deepcopy(config), deepcopy(DEFAULTS), ALLOWED_CONFIG_KEYS.copy()
    runs = []
    runner = facade.run_community

    def record(*args, **kwargs):
        run = runner(*args, **kwargs)
        runs.append(weakref.ref(run))
        return run

    monkeypatch.setattr(facade, "run_community", record)
    community = simulate(config)
    result, ledger = community.result(), community.settlement()
    assert isinstance(result, CommunityResult)
    assert community.result() is result
    assert runs[0]() is None, "the facade must release full-horizon physical arrays"
    assert config == original and DEFAULTS == defaults and ALLOWED_CONFIG_KEYS == allowed
    assert list(result.participants) == ["apt-1", "apt-2", "apt-3"]
    assert list(result.assets) == ["shared-roof"]
    assert result.ownership["shared-roof"] == {"apt-1": 0.5, "apt-2": 0.3, "apt-3": 0.2}
    assert result.currency == "EUR"
    assert list(result.settlement_totals) == [1, 2]
    assert ledger.attrs["project_year"] == 1
    assert len(ledger.index.get_level_values("interval").unique()) == 365 * 96
    assert_settlement_closure(ledger, coefficients={"apt-1": 0.4, "apt-2": 0.35, "apt-3": 0.25})
    assert result.yearly["Shared_Used_Total_Wh"].min() > 0
    for name in (
        "Shared_Residual",
        "Boundary_Residual",
        "Max_Interval_Shared_Residual",
        "Max_Interval_Boundary_Residual",
    ):
        np.testing.assert_allclose(result.yearly[name], 0, rtol=0, atol=1e-7)
    np.testing.assert_allclose(result.interval_closure, 0, rtol=0, atol=1e-9)
    for account in result.participants.values():
        assert account.community_benefit == account.npv_savings - account.npv_alone
        assert account.benefit_residual == pytest.approx(0, abs=1e-7)
        assert account.cashflows["Benefit_Residual"].abs().max() < 1e-7
        assert {
            "Load_Wh",
            "Own_PV_Production_Wh",
            "Metered_Import",
            "Metered_Export",
            "Shared_Allocated",
            "Shared_Used",
            "Participant_Surplus",
            "Supplier_Import",
        } <= set(account.energy)
    load = sum(account.energy["Load_Wh"] for account in result.participants.values())
    imported = sum(account.energy["Metered_Import"] for account in result.participants.values())
    pd.testing.assert_series_equal(
        result.shared_coverage_of_load, result.yearly["Shared_Used_Total_Wh"] / load, check_names=False
    )
    pd.testing.assert_series_equal(
        result.shared_coverage_of_import, result.yearly["Shared_Used_Total_Wh"] / imported, check_names=False
    )
    assert not hasattr(result, "coverage")
    assert not hasattr(result, "summary") and not hasattr(result, "to_dict")
    ledger.iloc[:, :] = 0
    ledger.attrs["project_year"] = 99
    assert community.settlement()["Shared_Used"].sum() > 0
    assert community.settlement().attrs["project_year"] == 1


def test_asset_credit_warning_preserves_settlement_cap_and_owned_money(config):
    config["community"]["projection_years"] = 1
    asset = config["assets"][0]
    asset.pop("costs")
    asset["tariff"] = deepcopy(TOU)
    baseline = simulate(config)
    asset["tariff"]["annual_network_credit"] = {
        "amount_per_year": 100,
        "network_fixed_per_year": 20,
        "network_import_prices": {"peak": 0.05, "off_peak": 0.02},
    }
    with pytest.warns(
        UserWarning, match="shared-roof.*producer-only.*rarely qualifies.*BREOS does not judge eligibility"
    ) as caught:
        credited = simulate(config)
    assert len(caught) == 1
    pd.testing.assert_frame_equal(credited.settlement(), baseline.settlement(), check_exact=True)
    before, after = baseline.result(), credited.result()
    roof = after.assets["shared-roof"]
    assert roof.energy["Metered_Import"].iloc[0] == 0
    assert roof.yearly["Network_Credit"].iloc[0] == roof.yearly["Network_Charge"].iloc[0] == 20
    for column in ("Cost_Import", "Cost_Daily", "Cost_Operation", "Revenue_Shared_Energy", "Revenue_Surplus"):
        pd.testing.assert_series_equal(
            roof.cashflows[column], before.assets["shared-roof"].cashflows[column], check_exact=True
        )
    for participant_id, share in after.ownership["shared-roof"].items():
        old, new = before.participants[participant_id], after.participants[participant_id]
        assert new.cashflows["Asset_Cost_Network_Credit"].iloc[0] == share * 20
        assert old.cashflows["Cost_System_Annual"].iloc[0] - new.cashflows["Cost_System_Annual"].iloc[
            0
        ] == pytest.approx(share * 20)
        discount_rate = credited._resolved.community.shared.discount_rate
        assert new.npv_savings - old.npv_savings == pytest.approx(share * 20 / (1 + discount_rate))
    np.testing.assert_allclose(after.interval_closure, 0, rtol=0, atol=1e-9)
    np.testing.assert_allclose(after.yearly[["Shared_Residual", "Boundary_Residual"]], 0, rtol=0, atol=1e-9)


@pytest.mark.parametrize("billing", ["flat", "tariff", "fixed_target", "daily_persistence"])
def test_provenance_matches_app_and_is_sorted_json_safe(config, billing):
    config["participants"][0]["reference_tariff"] = deepcopy(FLAT_REFERENCE)
    if billing == "flat":
        config["participants"][0].pop("tariff")
    else:
        config["participants"][1].pop("costs")
        config["participants"][1]["tariff"] = deepcopy(TOU)
        config["participants"][1]["reference_tariff"] = deepcopy(FLAT_REFERENCE)
    if billing == "tariff":
        config["community"].update(schedule=TOU["schedule"], shared_energy_price={"peak": 0.12, "off_peak": 0.08})
    if billing in ("fixed_target", "daily_persistence"):
        config["participants"][0]["smart_charging"] = {
            "mode": billing,
            "charge_periods": ["off_peak"],
            "discharge_periods": ["peak"],
            "grid_charge_efficiency": 0.95,
        }
        if billing == "fixed_target":
            config["participants"][0]["smart_charging"]["target_usable_fraction"] = 0.5
        else:
            config["participants"][0]["smart_charging"].update(target_levels=3, soc_states=3)
    config["participants"].reverse()
    community = simulate(config)
    provenance = community.result().provenance
    assert json.loads(json.dumps(provenance, allow_nan=False)) == provenance
    assert provenance["site_ids"] == ["apt-1", "apt-2", "apt-3", "shared-roof"]
    assert provenance["participant_ids"] == ["apt-1", "apt-2", "apt-3"]
    assert list(provenance["sites"]) == provenance["site_ids"]
    assert provenance["allocation"] == provenance["community"]["allocation"] == "fixed"
    assert provenance["settlement_schema_version"] == "1.0"
    assert provenance["community"]["coefficients"] == {"apt-1": 0.4, "apt-2": 0.35, "apt-3": 0.25}
    assert provenance["ownership"] == {"shared-roof": {"apt-1": 0.5, "apt-2": 0.3, "apt-3": 0.2}}
    if billing == "tariff":
        assert provenance["community"]["schedule"]["schedule"]["identifier"] == TOU["schedule"]
        assert provenance["community"]["shared_energy_price"] == {"peak": 0.12, "off_peak": 0.08}
    site = next(p for p in config["participants"] if p["id"] == "apt-1")
    payload = {key: value for key, value in site.items() if key not in ("id", "coefficient")}
    shared = community._resolved.community.shared.app_payload()
    fetch, weather = offline()
    with fetch, weather:
        app = App({**shared, **payload})
        app.simulate()
    actual = provenance["sites"]["apt-1"]
    for key, value in app.result()["provenance"].items():
        assert actual[key] == value, key
    household = provenance["sites"]["apt-2"]
    assert household["load_profile"]["key"] == "demandlib_h0"
    assert household["load_profile"]["packaged"] is True
    assert household["execution"] == {"effective": "load_only"}
    assert household["resolved_config"]["annual_consumption_kwh"] == 3000
    assert "weather" not in household
    if billing != "flat":
        assert household["tariff"]["import_prices"] == TOU["import_prices"]
        assert household["reference_tariff"]["import_prices"] == FLAT_REFERENCE["import_prices"]
        assert household["tariff"]["schedule_hash"]
    assert provenance["sites"]["shared-roof"]["load_profile"]["source"] == "community_zero_load"


def test_rerun_replaces_results_and_input_mutation_does_not_change_run(config, monkeypatch):
    community = Community(config)
    config["participants"].clear()
    fetch, weather = offline()
    with fetch, weather:
        assert community.simulate() is None
        first = community.result()
        community.simulate()
    assert community.result() is not first
    pd.testing.assert_frame_equal(first.yearly, community.result().yearly, check_exact=True)
    last = community.result()
    ledger = community.settlement()

    def fail(*_args, **_kwargs):
        raise ValueError("unavailable input")

    monkeypatch.setattr(facade, "run_community", fail)
    with pytest.raises(ValueError, match="unavailable input"):
        community.simulate()
    assert community.result() is last
    pd.testing.assert_frame_equal(ledger, community.settlement(), check_exact=True)


def test_azores_missing_midnight_settlement_and_no_sharing_app_parity():
    raw = {
        "community": {
            "location": {"latitude": 37.74, "longitude": -25.67, "timezone": "Atlantic/Azores"},
            "start_date": "2026-01-01",
            "resolution": "15min",
            "projection_years": 1,
            "allocation": "proportional",
            "shared_energy_price": 0.10,
            "shared_network_charge": 0.02,
            "surplus_price": 0.04,
        },
        "participants": [
            {
                "id": "pv-home",
                "n_modules": 8,
                "annual_consumption_kwh": 3500,
                "tariff": {
                    "currency": "EUR",
                    "custom_schedule": {
                        "identifier": "azores_daily",
                        "version": "1",
                        "timezone": "Atlantic/Azores",
                        "cycle": "daily",
                        "periods": ["off_peak", "peak"],
                        "rules": [
                            {
                                "days": "all",
                                "season": "all",
                                "intervals": {
                                    "off_peak": [["00:00", "08:00"], ["22:00", "24:00"]],
                                    "peak": [["08:00", "22:00"]],
                                },
                            }
                        ],
                    },
                    "import_prices": {"off_peak": 0.11, "peak": 0.28},
                    "export_prices": {"all": 0.05},
                    "fixed_charge_per_day": 0.25,
                },
            },
            {
                "id": "load-home",
                "annual_consumption_kwh": 2500,
                "costs": {"electricity_cost": 0.20, "daily_power_cost": 0.25},
            },
        ],
        "assets": [
            {
                "id": "roof",
                "n_modules": 12,
                "ownership": "external",
                "costs": {"electricity_cost": 0.20, "daily_power_cost": 0},
            }
        ],
    }
    community = simulate(raw)  # Uses tests.offline_app.offline for every site.
    ledger, result = community.settlement(), community.result()
    assert_settlement_closure(ledger)
    instants = ledger.index.get_level_values("interval").unique()
    # The offline weather covers the UTC year (Azores local 2025-12-31 23:00
    # onwards), which App accepts; local 29 March jumps from 00:00 to 01:00.
    assert len(instants) == 365 * 96
    assert ledger.index.is_unique
    assert len(ledger) == len(instants) * 4  # Three sites plus the community row.
    np.testing.assert_array_equal(ledger.groupby(level=["site_type", "site_id"]).size(), len(instants))
    assert (instants[1:] - instants[:-1] == pd.Timedelta(minutes=15)).all()
    local = instants.tz_convert("Atlantic/Azores")
    spring_day = local[(local.month == 3) & (local.day == 29)]
    assert len(spring_day) == 23 * 4 and spring_day[0].hour == 1
    assert result.yearly["Shared_Used_Total_Wh"].iloc[0] > 0
    np.testing.assert_allclose(result.interval_closure, 0, rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        result.yearly[
            ["Shared_Residual", "Boundary_Residual", "Max_Interval_Shared_Residual", "Max_Interval_Boundary_Residual"]
        ],
        0,
        rtol=0,
        atol=1e-9,
    )

    no_sharing = deepcopy(raw)
    no_sharing["community"]["pool_participant_export"] = False
    no_sharing["assets"] = []
    alone = simulate(no_sharing)
    pv = raw["participants"][0]
    app = simulate_offline(
        {
            **alone._resolved.community.shared.app_payload(),
            **{key: value for key, value in pv.items() if key != "id"},
        }
    )
    meter = alone.settlement().xs("pv-home", level="site_id")
    assert_bits(meter["Supplier_Import"], app.timeseries()["Import_From_Grid"] * 0.25)
    assert_bits(meter["Own_Export"], app.timeseries()["PV_AC_Export"] * 0.25)
    account, expected = alone.result().participants["pv-home"], app._artifacts.cost_projection
    assert account.npv_savings == expected.attrs["final_npv_savings"]
    assert account.community_benefit == 0
    for column in ("Cost_System_Annual", "Cost_No_Sys_Annual", "Savings_Cumulative_NPV"):
        assert_bits(account.cashflows[column], expected[column])
