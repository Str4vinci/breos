"""ADR 0005 PR 4: three households, complete baseline pricing and vehicle value."""

import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from breos.app import _revalued_config
from breos.app_config import build_costs_dict, resolve_app_config
from breos.projection import build_battery_config, build_pv_only_battery_config
from breos.vehicle import VehicleConfig
from breos.vehicle.economics import vehicle_pack_value, vehicle_wear
from breos.vehicle.households import VehicleHouseholds, _absence_unserved, value_vehicle_households
from tests.backend_parity import BACKENDS
from tests.energy_conservation import assert_energy_conservation, assert_vehicle_conservation
from tests.offline_app import (
    BASE,
    CREDIT,
    FLAT_REFERENCE,
    TOU,
    TOU_REFERENCE,
    assert_fields_close,
    result_fields,
    revalue_offline,
    simulate_offline,
)
from tests.test_ac_several_inverters import APP_AC
from tests.test_currency import EUR_COSTS, USD_COSTS
from tests.test_multi_inverter import TWO_INVERTERS
from tests.test_vehicle_config import AWAY, VEHICLE

CONFIG = {**BASE, "projection_years": 2, "vehicle": VEHICLE}


@pytest.mark.parametrize(
    "study_unserved,reference_unserved,unpriced_comparable",
    [(1.0, 1.0, False), (1.0, 2.0, False), (0.5e-9, 0.0, True), (1.1e-9, 0.0, False)],
)
@pytest.mark.parametrize("penalty", [None, 0.0, 0.6])
def test_comparability_cannot_offset_a_worse_absence_with_a_better_one(
    app, study_unserved, reference_unserved, unpriced_comparable, penalty
):
    households = app._artifacts.vehicle_households
    schedule = households.schedule
    departures = np.flatnonzero(~schedule.plugged & np.r_[True, schedule.plugged[:-1]])
    runs = {}
    for key, run in households.runs.items():
        flows = [dict(year) for year in run.priced_flows]
        for year in flows:
            year["Vehicle_Drive_Unserved"] = np.zeros(len(schedule.index))
        pos, amount = (departures[0], study_unserved) if key == "R2" else (departures[1], reference_unserved)
        flows[0]["Vehicle_Drive_Unserved"][pos] = amount * 1000 / schedule.hours_per_step
        rows = run.yearly_df.copy()
        rows["Vehicle_Drive_Unserved_kWh"] = [amount, 0.0]
        runs[key] = replace(run, priced_flows=tuple(flows), yearly_df=rows)
    vehicle = {**VEHICLE}
    if penalty is not None:
        vehicle["outside_charging_price"] = penalty
    resolved = resolve_app_config({**CONFIG, "vehicle": vehicle})
    _, result = value_vehicle_households(resolved.cfg, resolved, VehicleHouseholds(runs, schedule), None, None)
    for name in ("household_adjusted_npv", "vehicle_benefit"):
        assert result[name]["comparable"] == (penalty is not None or unpriced_comparable)
        if penalty is None and not unpriced_comparable:
            assert schedule.index[departures[0]].isoformat() in result[name]["reason"]


@pytest.mark.parametrize("freq,hours", [("h", 1.0), ("15min", 0.25)])
def test_absence_comparison_joins_year_tail_and_head_and_labels_initial_partial_trip(app, freq, hours):
    original = app._artifacts.vehicle_households
    schedule = replace(
        original.schedule,
        index=pd.date_range(original.schedule.index[0], periods=8, freq=freq),
        hours_per_step=hours,
        plugged=[False, False, True, False, False, True, False, False],
        steps_to_departure=np.zeros(8),
        drive_first_wh=np.zeros(8),
        park_hours=np.zeros(8),
        drive_second_wh=np.zeros(8),
        initial_absence_departure=original.schedule.index[0] - pd.Timedelta(hours=12),
    )
    runs = {}
    for key, run in original.runs.items():
        flows = [{"Vehicle_Drive_Unserved": np.zeros(8)} for _ in range(2)]
        first, second = (0.6, 0.4) if key == "R2" else (0.3, 0.7)
        flows[0]["Vehicle_Drive_Unserved"][-1] = first * 1000 / hours
        flows[1]["Vehicle_Drive_Unserved"][0] = second * 1000 / hours
        runs[key] = replace(run, priced_flows=tuple(flows))
    labels, totals = _absence_unserved(VehicleHouseholds(runs, schedule))
    assert schedule.initial_absence_departure.isoformat() in labels[0]
    assert len(labels) == 5
    for amounts in totals.values():
        np.testing.assert_allclose(amounts, [0, 0, 1, 0, 0], rtol=0, atol=1e-12)


@pytest.fixture(scope="module")
def app():
    return simulate_offline(CONFIG)


def test_vh7_replacement_aware_wear_and_pack_only_credit():
    cfg = VehicleConfig(**{**VEHICLE, "battery_kwh": 40})
    kwargs = dict(horizon=20, inflation=0.02, learning=0.0, discount=0.03)
    r1 = vehicle_pack_value(cfg, [], 0.78, **kwargs)
    r2 = vehicle_pack_value(cfg, [{"year": 13, "time_years": 12.5, "action": "replaced"}], 0.88, **kwargs)
    assert r1["replacement_outlays_npv"] == 0.0
    assert r2["replacement_outlays_npv"] == pytest.approx(4248.915318, abs=1e-6, rel=0)
    assert r1["terminal_pack_credit_npv"] == pytest.approx(1053.098298, abs=1e-6, rel=0)
    assert r2["terminal_pack_credit_npv"] == pytest.approx(2369.471170, abs=1e-6, rel=0)
    wear = vehicle_wear(r2, r1)
    assert wear["wear_npv"] == pytest.approx(2932.542446, abs=1e-6, rel=0)
    assert wear["replacement_outlays_difference_npv"] == pytest.approx(4248.915318, abs=1e-6, rel=0)
    invoice = vehicle_pack_value(
        replace(cfg, battery_replacement_cost=6000), [{"time_years": 12.5, "action": "replaced"}], 0.88, **kwargs
    )
    assert vehicle_wear(r1, r2)["wear_npv"] == -wear["wear_npv"]
    assert invoice["terminal_pack_credit_npv"] == r2["terminal_pack_credit_npv"]
    assert invoice["replacement_outlays_npv"] == pytest.approx(r2["replacement_outlays_npv"] * 1.25)


@pytest.mark.parametrize("health", [None, 0.0, 0.6, 0.7, 1.0, 1.2])
def test_terminal_credit_clipped_and_outlays_booked_individually(health):
    cfg = VehicleConfig(**{**VEHICLE, "battery_kwh": 40, "battery_replacement_cost": 6000})
    events = [{"time_years": time, "action": "replaced"} for time in [0.25, 0.75]]
    value = vehicle_pack_value(cfg, events, health, horizon=2, inflation=0.02, learning=0.01, discount=0.03)
    assert value["replacement_outlays_npv"] == pytest.approx(
        sum(6000 * (1.02 * 0.99 / 1.03) ** t for t in [0.25, 0.75])
    )
    fraction = 0.0 if health is None else np.clip((health - 0.7) / 0.3, 0.0, 1.0)
    assert value["terminal_pack_credit_npv"] == pytest.approx(4800 * (1.02 * 0.99 / 1.03) ** 2 * fraction)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("freq", ["h", "15min"])
@pytest.mark.parametrize("several", [False, True])
def test_full_app_three_households_schema_and_energy_closure(trina, backend, freq, several):
    config = {
        **(TWO_INVERTERS if several else BASE),
        **(APP_AC if several else {"battery_kwh": 0}),
        "projection_years": 1,
        "resolution": freq,
        "execution_backend": backend,
        "vehicle": {**VEHICLE, "away_charging": {**AWAY, "price": 0.2}},
    }
    app = simulate_offline(config)
    result, artifacts = app.result(), app._artifacts
    households = artifacts.vehicle_households
    assert set(households.runs) == {"R0", "R1", "R2"}
    for key, run in households.runs.items():
        frame = run.first_year_results_df
        battery = (
            build_battery_config(app._resolved.cfg, app._resolved, initial_soh=100.0)
            if several and key != "R0"
            else build_pv_only_battery_config(app._resolved.cfg, app._resolved)
        )
        assert_energy_conservation(frame, battery)
        assert_vehicle_conservation(
            frame, app._resolved.vehicle, 1.0 if freq == "h" else 0.25, schedule=households.schedule
        )
    baseline = households.runs["R0"].first_year_results_df
    assert (baseline["PV_DC"] == 0.0).all()
    assert baseline["Inverter_Night_Consumption"].sum() == 0.0
    assert set(result["vehicle"]) == {
        "mode",
        "reference_mode",
        "currency",
        "households",
        "household_adjusted_npv",
        "vehicle_benefit",
    }
    assert result["provenance"]["vehicle"]["schedule_hash"] == households.schedule.schedule_hash()
    assert result["provenance"]["vehicle"]["config"]["allow_terminal_replacement"] is True
    assert result["result_schema_version"] == "1"
    assert result["yearly"][0]["vehicle"]["drive_served_kwh"] > 0
    assert result["vehicle"]["households"]["R2"]["totals"]["away_purchased_kwh"] > 0
    assert result["self_consumption_kwh"] == pytest.approx(
        sum(row["self_consumption_kwh"] for row in result["monthly"]), abs=0.08
    )
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "system",
    [
        {"battery_kwh": 0, "n_modules": 2},
        {"battery_kwh": 20, "n_modules": 20},
        {"battery_kwh": 5, "battery_rte": 0.7, "battery_temperature": 40},
    ],
)
def test_q6_r0_does_not_depend_on_system(app, system):
    changed = simulate_offline({**CONFIG, **system})
    assert changed.result()["vehicle"]["households"]["R0"] == app.result()["vehicle"]["households"]["R0"]
    for left, right in zip(
        app._artifacts.vehicle_households.runs["R0"].priced_flows,
        changed._artifacts.vehicle_households.runs["R0"].priced_flows,
        strict=True,
    ):
        np.testing.assert_array_equal(left["Import_From_Grid"], right["Import_From_Grid"])


@pytest.mark.parametrize(
    "tariff,reference",
    [
        (None, None),
        (TOU, None),
        (TOU, TOU_REFERENCE),
        ({**TOU, "annual_network_credit": CREDIT}, None),
        ({**TOU, "annual_network_credit": CREDIT}, {**TOU_REFERENCE, "annual_network_credit": CREDIT}),
        (None, FLAT_REFERENCE),
    ],
)
def test_r0_baseline_reaches_every_pricing_path(tariff, reference):
    app = simulate_offline({**CONFIG, "tariff": tariff, "reference_tariff": reference})
    artifacts = app._artifacts
    run = artifacts.vehicle_households.runs["R0"]
    prices = artifacts.resolved_reference_tariff or artifacts.resolved_tariff
    for year, ((_, row), flows) in enumerate(zip(artifacts.yearly_df.iterrows(), run.priced_flows, strict=True)):
        imported = flows["Import_From_Grid"]
        expected = (
            float(np.sum(imported * np.asarray(prices.import_price_per_kwh)) / 1000)
            if prices
            else float(imported.sum() / 1000 * artifacts.costs["electricity_cost"])
        )
        assert row["Baseline_Import_Cost"] == pytest.approx(expected)
        assert row["Baseline_Import_kWh"] == run.yearly_df.iloc[year]["Import_kWh"]
        assert row["Baseline_Import_kWh"] > row["Load_kWh"]
        if prices and prices.prices.annual_network_credit:
            credit = prices.prices.annual_network_credit
            charge = (
                float(np.sum(imported * np.asarray(prices.network_price_per_kwh)) / 1000)
                + credit.network_fixed_per_year
            )
            assert row["Baseline_Network_Charge"] == pytest.approx(charge)
            assert row["Baseline_Network_Credit"] == pytest.approx(min(credit.amount_per_year, charge))


@pytest.mark.parametrize("name", ["charger_cost", "battery_cost_per_kwh"])
def test_economics_defensively_refuses_missing_vehicle_price(name):
    resolved = resolve_app_config(CONFIG)
    object.__setattr__(resolved.vehicle, name, None)
    with pytest.raises(ValueError, match=f"vehicle.{name}"):
        build_costs_dict(resolved.cfg, resolved)


def test_charger_premium_counted_once_and_wear_does_not_change_system_npv(app):
    free = revalue_offline(app, {"vehicle": {"charger_cost": 0}})
    original = app.result()
    assert free["total_investment"] == original["total_investment"] - VEHICLE["charger_cost"]
    assert free["npv_savings"] == original["npv_savings"] + VEHICLE["charger_cost"]
    for name in ["vehicle_benefit", "household_adjusted_npv"]:
        assert free["vehicle"][name]["npv"] == pytest.approx(original["vehicle"][name]["npv"] + VEHICLE["charger_cost"])
    repriced = revalue_offline(app, {"vehicle": {"battery_cost_per_kwh": 240}})
    assert repriced["npv_savings"] == original["npv_savings"]
    assert repriced["vehicle"]["vehicle_benefit"]["wear_npv"] == pytest.approx(
        2 * original["vehicle"]["vehicle_benefit"]["wear_npv"]
    )


@pytest.mark.parametrize("dispatch_controls", [False, True])
@pytest.mark.parametrize("change", ["prices", "schedule", "reference", "remove_reference", "remove_tariff", "network"])
def test_revalue_retains_three_households_and_matches_fresh(dispatch_controls, change):
    vehicle = {**VEHICLE, "away_charging": {**AWAY, "price": 0.3}}
    if dispatch_controls:
        vehicle.update(charge_periods=["off_peak"], discharge_periods=["peak"])
    config = {**CONFIG, "projection_years": 1, "vehicle": vehicle, "tariff": TOU, "reference_tariff": FLAT_REFERENCE}
    changes = {
        "prices": {
            "tariff": {"import_prices": {"peak": 0.35, "off_peak": 0.07}},
            "vehicle": {"battery_cost_per_kwh": 200, "away_charging": {"price": 0.5}},
        },
        "schedule": {
            "tariff": {
                "custom_schedule": {
                    "identifier": "changed",
                    "version": "1",
                    "timezone": "Europe/Lisbon",
                    "cycle": "daily",
                    "periods": ["peak", "off_peak"],
                    "rules": [
                        {
                            "days": "all",
                            "season": "all",
                            "intervals": {
                                "off_peak": [["00:00", "07:00"], ["23:00", "24:00"]],
                                "peak": [["07:00", "23:00"]],
                            },
                        }
                    ],
                },
                "schedule": None,
            }
        },
        "reference": {"reference_tariff": TOU_REFERENCE},
        "remove_reference": {"reference_tariff": None},
        "remove_tariff": {"tariff": None},
        "network": {
            "tariff": {"annual_network_credit": CREDIT},
            "reference_tariff": {"annual_network_credit": {**CREDIT, "network_import_prices": {"all": 0.1}}},
        },
    }[change]
    app = simulate_offline(config)
    if dispatch_controls and change == "remove_tariff":
        with pytest.raises(ValueError, match="needs a "):
            revalue_offline(app, changes)
        return
    revalued = revalue_offline(app, changes)
    fresh = simulate_offline(_revalued_config(config, changes)).result()
    assert_fields_close(result_fields(revalued), result_fields(fresh))
    assert revalued["provenance"]["revaluation"]["method"] == (
        "resimulated"
        if dispatch_controls and change == "schedule"
        else "repriced_by_step"
        if change == "schedule"
        else "repriced"
    )


@pytest.mark.parametrize("mode,reference", [("uncontrolled", "none"), ("v1g", "none"), ("v1g", "uncontrolled")])
def test_optional_reference_household(mode, reference):
    app = simulate_offline(
        {**CONFIG, "projection_years": 1, "vehicle": {**VEHICLE, "mode": mode, "reference_mode": reference}}
    )
    result = app.result()["vehicle"]
    assert ("R1" in result["households"]) == (reference != "none")
    assert (result["vehicle_benefit"] is None) == (reference == "none")
    assert "v2h_benefit" not in result
    if reference != "none":
        assert result["vehicle_benefit"]["study_mode"] == mode
        assert result["vehicle_benefit"]["reference_mode"] == reference
    assert "v2h_benefit_npv" not in app.result()


SMART = {
    "mode": "fixed_target",
    "target_usable_fraction": 0.95,
    "charge_periods": ["off_peak"],
    "discharge_periods": ["peak"],
    "grid_charge_efficiency": 0.95,
    "grid_import_limit_w": 1200,
}


@pytest.mark.parametrize("penalty", [None, 0.0, 0.6])
def test_comparability_and_priced_unserved_driving_in_every_household(penalty):
    config = {
        **CONFIG,
        "projection_years": 1,
        "battery_kwh": 100,
        "tariff": TOU,
        "smart_charging": SMART,
        "vehicle": {**VEHICLE, "battery_kwh": 10, "outside_charging_price": penalty},
    }
    if penalty is None:
        del config["vehicle"]["outside_charging_price"]
    app = simulate_offline(config)
    result = app.result()["vehicle"]
    for key, household in result["households"].items():
        unserved = household["totals"]["drive_unserved_kwh"]
        assert unserved > 0.0
        assert household["unserved_driving_cost_npv"] == pytest.approx(unserved * (penalty or 0.0) / 1.03)
    for comparison in ["household_adjusted_npv", "vehicle_benefit"]:
        assert result[comparison]["comparable"] == (penalty is not None)
        assert (result[comparison]["reason"] is None) == (penalty is not None)
    if penalty is None:
        priced = revalue_offline(app, {"vehicle": {"outside_charging_price": 0.6}})
        assert priced["vehicle"]["household_adjusted_npv"]["comparable"] is True
        assert priced["vehicle"]["vehicle_benefit"]["comparable"] is True
        assert priced["npv_savings"] == app.result()["npv_savings"]
        assert priced["vehicle"]["households"]["R2"]["totals"] == result["households"]["R2"]["totals"]
    adjusted = result["household_adjusted_npv"]
    assert adjusted["npv"] == pytest.approx(
        adjusted["system_npv"]
        - adjusted["away_cost_difference_npv"]
        - adjusted["unserved_cost_difference_npv"]
        - adjusted["wear_npv"]
    )


def test_away_purchases_priced_separately_and_in_emissions():
    config = {
        **CONFIG,
        "vehicle": {**VEHICLE, "away_charging": {**AWAY, "price": 0.3}},
        "import_price_escalation": 0.04,
    }
    app = simulate_offline(config)
    artifacts, result = app._artifacts, app.result()
    r0 = artifacts.vehicle_households.runs["R0"].yearly_df
    rows = artifacts.yearly_df
    params = app._resolved.emissions_params
    for n, (_, row) in enumerate(rows.iterrows()):
        away = row["Vehicle_Away_Charge_Purchased_kWh"]
        assert row["Vehicle_Away_Charge_Cost"] == away * 0.3
        unattributed = (
            row["Battery_AC_To_Load_kWh"]
            - row["PV_Origin_Battery_AC_Load_kWh"]
            - row["Grid_Origin_Battery_AC_Load_kWh"]
        )
        expected = (
            (
                r0.iloc[n]["Import_kWh"]
                + r0.iloc[n]["Vehicle_Away_Charge_Purchased_kWh"]
                - row["Import_kWh"]
                - away
                - unattributed
            )
            * params.avoided_intensity_gco2_kwh
            / 1000
        )
        assert artifacts.cost_projection.iloc[n]["CO2_Avoided_SelfConsumed_kg"] == pytest.approx(expected)
    for household in result["vehicle"]["households"].values():
        expected = sum(
            year["away_purchased_kwh"] * 0.3 * 1.04 ** (year["year"] - 1) / 1.03 ** year["year"]
            for year in household["yearly"]
        )
        assert household["away_purchases_npv"] == pytest.approx(expected)
    free = revalue_offline(app, {"vehicle": {"away_charging": {"price": 0}}})
    assert result["vehicle"]["households"]["R0"]["totals"]["avoided_co2_kg"] == 0.0
    assert result["vehicle"]["households"]["R2"]["totals"]["avoided_co2_kg"] == pytest.approx(
        artifacts.cost_projection["CO2_Avoided_Total_kg"].sum()
    )
    assert free["npv_savings"] == result["npv_savings"]
    assert free["vehicle"]["households"]["R2"]["away_purchases_npv"] == 0


def test_currency_restatement_includes_every_vehicle_amount():
    vehicle = {
        **VEHICLE,
        "outside_charging_price": 0.6,
        "battery_replacement_cost": 6000,
        "away_charging": {**AWAY, "price": 0.3},
    }
    config = {**CONFIG, "projection_years": 1, "cost_preset": None, "costs": EUR_COSTS, "vehicle": vehicle}
    app = simulate_offline(config)
    changes = {
        "currency": "USD",
        "costs": USD_COSTS,
        "vehicle": {
            name: vehicle[name] * 1.1
            for name in ["charger_cost", "battery_cost_per_kwh", "outside_charging_price", "battery_replacement_cost"]
        },
    }
    changes["vehicle"]["away_charging"] = {"price": 0.33}
    for name in [
        "charger_cost",
        "battery_cost_per_kwh",
        "outside_charging_price",
        "battery_replacement_cost",
        "away_charging",
    ]:
        missing = deepcopy(changes)
        del missing["vehicle"][name]
        with pytest.raises(ValueError, match="vehicle."):
            revalue_offline(app, missing)
    revalued = revalue_offline(app, changes)
    fresh = simulate_offline(_revalued_config(config, changes)).result()
    assert_fields_close(result_fields(revalued), result_fields(fresh))
    assert revalued["vehicle"]["vehicle_benefit"]["npv"] == pytest.approx(
        app.result()["vehicle"]["vehicle_benefit"]["npv"] * 1.1
    )
    for physical in [{"battery_kwh": 70}, {"allow_terminal_replacement": False}]:
        with pytest.raises(ValueError, match="vehicle prices only"):
            revalue_offline(app, {"vehicle": physical})


@pytest.mark.parametrize("terminal", [False, True])
def test_vehicle_terminal_policy_applies_to_all_households(terminal):
    app = simulate_offline(
        {
            **CONFIG,
            "projection_years": 1,
            "vehicle": {**VEHICLE, "eol_percentage": 0.99999999, "allow_terminal_replacement": terminal},
        }
    )
    for household in app.result()["vehicle"]["households"].values():
        event = household["replacement_history"][-1]
        assert event["time_years"] == 1.0
        assert event["action"] == ("replaced" if terminal else "kept")
        assert event["reason"] == ("end_of_life" if terminal else "terminal_period")
        if terminal:
            assert household["final_soh_fraction"] == 1.0


def test_stationary_tariff_controls_revalue_with_vehicle():
    config = {**CONFIG, "projection_years": 1, "tariff": TOU, "smart_charging": SMART}
    app = simulate_offline(config)
    changes = {"tariff": {"import_prices": {"peak": 0.33, "off_peak": 0.08}, "fixed_charge_per_day": 0.5}}
    result = revalue_offline(app, changes)
    assert_fields_close(
        result_fields(result), result_fields(simulate_offline(_revalued_config(config, changes)).result())
    )
    assert result["provenance"]["revaluation"]["method"] == "repriced"


def test_period_reports_vehicle_energy_without_lifetime_comparisons():
    window = {
        **CONFIG,
        "period": {"start": "2025-06-01", "end": "2025-06-03"},
        "vehicle": {**VEHICLE, "eol_percentage": 0.99999999},
    }
    window.pop("projection_years")
    app = simulate_offline(window)
    result = app.result()
    assert result["npv_savings"] is None
    assert result["vehicle"]["household_adjusted_npv"] is None
    assert result["vehicle"]["vehicle_benefit"] is None
    assert result["vehicle"]["households"]["R2"]["totals"]["drive_served_kwh"] > 0
    for household in result["vehicle"]["households"].values():
        assert household["replacement_history"]
        assert household["terminal_pack_credit_npv"] is None
        assert all(
            event["outlay_npv"] is None and event["outlay_nominal"] is None
            for event in household["replacement_history"]
        )


def test_no_vehicle_provenance_is_null():
    result = simulate_offline({**BASE, "projection_years": 1}).result()
    assert result["provenance"]["vehicle"] is None
    assert "vehicle" not in result


@pytest.mark.parametrize("freq", ["h", "15min"])
def test_app_vehicle_backend_parity(freq):
    pytest.importorskip("numba", reason="the compiled backend needs the breos[fast] extra")
    kwargs = {
        **CONFIG,
        "projection_years": 1,
        "resolution": freq,
        "vehicle": {**VEHICLE, "away_charging": {**AWAY, "price": 0.2}},
        "tariff": TOU,
    }
    python = simulate_offline({**kwargs, "execution_backend": "python"})
    numba = simulate_offline({**kwargs, "execution_backend": "numba"})
    assert python.result()["vehicle"] == numba.result()["vehicle"]
    assert python.result()["npv_savings"] == numba.result()["npv_savings"]


def test_revalue_adds_tariff_to_price_blind_vehicle_flows(app):
    changes = {"tariff": TOU, "reference_tariff": TOU_REFERENCE}
    result = revalue_offline(app, changes)
    fresh = simulate_offline(_revalued_config(CONFIG, changes)).result()
    assert_fields_close(result_fields(result), result_fields(fresh))
    assert result["provenance"]["revaluation"]["method"] == "repriced_by_step"


@pytest.mark.filterwarnings("ignore::breos.degradation.validation.BlastExperimentalRangeWarning")
def test_app_allows_vehicle_blast():
    app = simulate_offline(
        {
            **CONFIG,
            "projection_years": 1,
            "vehicle": {**VEHICLE, "degradation_engine": "blast", "blast_model": "nmc_gr_50ah_b1"},
        }
    )
    for run in app._artifacts.vehicle_households.runs.values():
        assert run.carry.vehicle.degradation_state["degradation_engine"] == "blast"
        assert 0.0 < run.carry.vehicle.soh_fraction < 1.0
