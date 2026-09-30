"""The economics prices battery replacements; the physics only reports them (ADR 0003 E4, #183)."""

from dataclasses import fields
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from breos import App
from breos.battery import (
    BatteryConfig,
    frame_replaced_capacity_wh,
    simulate_energy_balance,
    simulate_energy_balance_summary,
)
from breos.economics import (
    CostParams,
    calculate_costs,
    price_year_rows,
    replacement_event_cost,
    replacement_total_t0,
)
from tools.generate_app_golden import SCENARIOS, _fake_fetch


def test_battery_config_has_no_price():
    assert "replacement_cost" not in {field.name for field in fields(BatteryConfig)}


@pytest.mark.parametrize("configured", [None, "auto", "calculate", " Calculate "])
def test_a_replacement_is_priced_per_kwh_unless_configured(configured):
    assert replacement_event_cost(6.5, 420.0, configured) == 6.5 * 420.0


def test_a_configured_replacement_price_wins():
    assert replacement_event_cost(6.5, 420.0, 1234.0) == 1234.0


@pytest.mark.parametrize("configured", [-1.0, float("nan"), True, "cheap"])
def test_a_bad_replacement_price_is_refused(configured):
    with pytest.raises(ValueError, match="battery.replacement_cost must be a non-negative number"):
        replacement_event_cost(6.5, 420.0, configured)


def test_calculate_costs_carries_the_replacement_price():
    params = CostParams(battery_cost_per_kwh=420.0)
    assert calculate_costs(4, 500.0, 5000.0, params)["replacement_cost_each"] == 5.0 * 420.0
    assert calculate_costs(4, 500.0, 5000.0, params, replacement_cost_each=99.0)["replacement_cost_each"] == 99.0


def test_year_rows_price_each_replacement():
    rows = pd.DataFrame({"Year": [1, 2, 3], "Replacements": [0, 1, 3]})
    priced = price_year_rows(
        rows.assign(Import_kWh=0.0, Export_kWh=0.0, Load_kWh=0.0),
        {"electricity_cost": 0.2, "electricity_sold_cost": 0.05, "daily_power_cost": 0.0, "replacement_cost_each": 0.1},
    )
    # One event at a time, as the physics layer once summed its per-event cost.
    assert priced["Replacement_Cost"].tolist() == [0.0, 0.1, 0.1 + 0.1 + 0.1]


def test_the_price_sits_after_the_count_as_before():
    rows = pd.DataFrame({"Year": [1], "Replacements": [1], "Replaced_Capacity_kWh": [5.0], "Import_kWh": [0.0]})
    priced = price_year_rows(
        rows.assign(Export_kWh=0.0, Load_kWh=0.0),
        {"electricity_cost": 0.2, "electricity_sold_cost": 0.05, "daily_power_cost": 0.0, "replacement_cost_each": 9.0},
    )
    assert list(priced.columns[:4]) == ["Year", "Replacements", "Replacement_Cost", "Replaced_Capacity_kWh"]


def test_replacement_counts_must_be_whole():
    costs = {
        "electricity_cost": 0.2,
        "electricity_sold_cost": 0.05,
        "daily_power_cost": 0.0,
        "replacement_cost_each": 9.0,
    }
    rows = pd.DataFrame({"Year": [1, 2], "Import_kWh": 0.0, "Export_kWh": 0.0, "Load_kWh": 0.0})
    # A missing count, as a CSV round trip leaves it, is no replacement.
    assert price_year_rows(rows.assign(Replacements=[np.nan, 1.0]), costs)["Replacement_Cost"].tolist() == [0.0, 9.0]
    with pytest.raises(ValueError, match="whole non-negative count"):
        price_year_rows(rows.assign(Replacements=[0.0, 1.7]), costs)


def test_replacements_without_a_price_are_refused():
    rows = pd.DataFrame({"Year": [1], "Replacements": [1], "Import_kWh": [0.0], "Export_kWh": [0.0], "Load_kWh": [0.0]})
    costs = {"electricity_cost": 0.2, "electricity_sold_cost": 0.05, "daily_power_cost": 0.0}
    with pytest.raises(ValueError, match="replacement_cost_each"):
        price_year_rows(rows, costs)
    # Without a replacement there is nothing to price.
    assert price_year_rows(rows.assign(Replacements=0), costs)["Replacement_Cost"].tolist() == [0.0]


def test_the_t0_total_is_added_year_by_year():
    # Built-in sum() compensates from Python 3.12 and would give 1.0 here; the
    # projection loop added the years one at a time, on every Python.
    assert replacement_total_t0([0.1] * 10) == 0.9999999999999999


def test_stored_replacement_money_is_kept():
    # Ledger schema < 3.0 stored the money per year; the rows keep it rather
    # than re-pricing, and need no replacement_cost_each.
    costs = {
        "total_initial_cost": 1000.0,
        "electricity_cost": 0.2,
        "electricity_sold_cost": 0.05,
        "daily_power_cost": 0.0,
        "annual_operation_cost": 0.0,
    }
    rows = pd.DataFrame(
        {"Year": [1], "Replacements": [1], "Replacement_Cost": [640.0], "Import_kWh": [0.0], "Export_kWh": [0.0]}
    )
    assert price_year_rows(rows.assign(Load_kWh=0.0), costs)["Replacement_Cost"].tolist() == [640.0]


def test_the_physics_reports_the_swapped_capacity():
    index = pd.date_range("2025-01-01", periods=24 * 20, freq="h", tz="UTC")
    pv = pd.Series(np.where((index.hour >= 10) & (index.hour < 15), 4000.0, 0.0), index=index)
    load = pd.DataFrame({"Load": np.where(index.hour >= 18, 2000.0, 200.0)}, index=index)
    config = BatteryConfig(nominal_energy_wh=4000.0, eol_percentage=0.9999, initial_soh=99.995)
    results, _total_pv, summary, n_replacements, _degradation = simulate_energy_balance(
        pv_dc=pv, houseload=load, battery_config=config, freq="h", temperature_series=pd.Series(35.0, index=index)
    )
    assert n_replacements > 0
    swapped = results["Battery_Replaced_Capacity_Wh"]
    assert swapped[results["Battery_Replaced"]].eq(4000.0).all()
    assert swapped[~results["Battery_Replaced"]].eq(0.0).all()
    assert summary["Replaced_Capacity_kWh"].iloc[0] == 4.0 * n_replacements


@pytest.fixture(scope="module")
def priced_runs():
    config = SCENARIOS["native_h_replacement"]
    runs = {}
    with (
        mock.patch("breos.app.fetch_tmy_weather_data", _fake_fetch),
        mock.patch("breos.app.load_weather", lambda **_kwargs: None),
    ):
        for storage_cost in (500.0, 250.0):
            app = App({**config, "costs": {**config.get("costs", {}), "storage_cost_per_kwh": storage_cost}})
            app.simulate()
            runs[storage_cost] = app.result()
    return runs


def test_a_storage_price_moves_only_the_money(priced_runs):
    full, half = priced_runs[500.0], priced_runs[250.0]
    # The same simulation: a price does not reach the physics.
    assert full["yearly"] == half["yearly"]
    assert full["battery_replacements"] == half["battery_replacements"] > 0
    assert half["battery_replacement_cost_t0_prices"] == full["battery_replacement_cost_t0_prices"] / 2
    assert [row["cost_replacement"] for row in half["financial"][1:]] == pytest.approx(
        [row["cost_replacement"] / 2 for row in full["financial"][1:]], abs=0.01
    )


def test_both_dispatch_paths_add_the_swapped_capacity_alike():
    # A non-integer capacity and several swaps, where a pandas column sum and
    # the day loop's one-at-a-time total can part in the last bit.
    index = pd.date_range("2025-01-01", periods=24 * 20, freq="h", tz="UTC")
    common = dict(
        pv_dc=pd.Series(np.where((index.hour >= 10) & (index.hour < 15), 4000.0, 0.0), index=index),
        houseload=pd.DataFrame({"Load": np.where(index.hour >= 18, 2000.0, 200.0)}, index=index),
        battery_config=BatteryConfig(nominal_energy_wh=3192.29, eol_percentage=0.9999, initial_soh=99.995),
        freq="h",
        temperature_series=pd.Series(35.0, index=index),
    )
    results, _total_pv, _summary, n_replacements, _degradation = simulate_energy_balance(**common)
    summary = simulate_energy_balance_summary(**common)
    assert n_replacements >= 6
    assert frame_replaced_capacity_wh(results) == summary.replaced_capacity_wh
