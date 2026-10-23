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
from tests.offline_app import FLAT_REFERENCE, TOU, offline

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
