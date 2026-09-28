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
    cost_analysis_projection,
    price_year_rows,
    replacement_event_cost,
    replacement_total_t0,
)
from tools import recalculate_economics
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


def test_replacements_without_a_price_are_refused():
    rows = pd.DataFrame({"Year": [1], "Replacements": [1], "Import_kWh": [0.0], "Export_kWh": [0.0], "Load_kWh": [0.0]})
    costs = {"electricity_cost": 0.2, "electricity_sold_cost": 0.05, "daily_power_cost": 0.0}
    with pytest.raises(ValueError, match="replacement_cost_each"):
        price_year_rows(rows, costs)
    # Without a replacement there is nothing to price.
    assert price_year_rows(rows.assign(Replacements=0), costs)["Replacement_Cost"].tolist() == [0.0]


def test_the_frame_path_prices_the_marked_swaps():
    index = pd.date_range("2025-01-01", periods=48, freq="h")
    replaced = np.zeros(48, dtype=bool)
    replaced[30] = True
    frame = pd.DataFrame(
        {
            "Datetime": index,
            "PV_AC_To_Load": 100.0,
            "PV_Origin_Battery_AC_To_Load": 0.0,
            "PV_AC_Export": 0.0,
            "Houseload": 500.0,
            "Import_From_Grid": 400.0,
            "Battery_Replaced": replaced,
        }
    )
    costs = {
        "total_initial_cost": 1000.0,
        "electricity_cost": 0.2,
        "electricity_sold_cost": 0.05,
        "daily_power_cost": 0.0,
        "annual_operation_cost": 0.0,
        "replacement_cost_each": 750.0,
    }
    projection = cost_analysis_projection(frame, costs, num_years=2, inflation_rate=0.0, discount_rate=0.0)
    assert projection["Cost_Replacement"].tolist() == [750.0, 0.0]
    assert projection.attrs["total_replacement_cost"] == 750.0


def test_the_t0_total_is_added_year_by_year():
    # Built-in sum() compensates from Python 3.12 and would give 1.0 here; the
    # projection loop added the years one at a time, on every Python.
    assert replacement_total_t0([0.1] * 10) == 0.9999999999999999


def _frame_with_swaps(n_swaps, **extra):
    index = pd.date_range("2025-01-01", periods=48, freq="h")
    replaced = np.zeros(48, dtype=bool)
    replaced[: 8 * n_swaps : 8] = True
    frame = pd.DataFrame(
        {
            "Datetime": index,
            "PV_AC_To_Load": 100.0,
            "PV_Origin_Battery_AC_To_Load": 0.0,
            "PV_AC_Export": 0.0,
            "Houseload": 500.0,
            "Import_From_Grid": 400.0,
            "Battery_Replaced": replaced,
            **extra,
        }
    )
    return frame, replaced


_FRAME_COSTS = {
    "total_initial_cost": 1000.0,
    "electricity_cost": 0.2,
    "electricity_sold_cost": 0.05,
    "daily_power_cost": 0.0,
    "annual_operation_cost": 0.0,
}


def test_the_frame_path_group_sums_the_swaps():
    # Six swaps at 0.1: one at a time gives 0.6, the group sum the projection
    # has always taken of the per-step money gives 0.6000000000000001.
    frame, replaced = _frame_with_swaps(6)
    projection = cost_analysis_projection(
        frame, {**_FRAME_COSTS, "replacement_cost_each": 0.1}, num_years=1, inflation_rate=0.0, discount_rate=0.0
    )
    per_step = pd.DataFrame({"Replacement_Cost": np.where(replaced, 0.1, 0.0)})
    expected = float(per_step.groupby(frame["Datetime"].dt.year).sum()["Replacement_Cost"].iloc[0])
    assert expected != 0.6  # what adding 0.1 six times gives
    assert projection["Cost_Replacement"].tolist() == [expected]


def test_stored_replacement_money_is_kept_on_both_paths():
    # Ledger schema < 3.0 stored the money per step and per year; both paths
    # keep it rather than re-pricing, and need no replacement_cost_each.
    frame, replaced = _frame_with_swaps(1, Replacement_Cost=0.0)
    frame.loc[replaced, "Replacement_Cost"] = 640.0
    projection = cost_analysis_projection(frame, _FRAME_COSTS, num_years=1, inflation_rate=0.0, discount_rate=0.0)
    assert projection["Cost_Replacement"].tolist() == [640.0]
    rows = pd.DataFrame(
        {"Year": [1], "Replacements": [1], "Replacement_Cost": [640.0], "Import_kWh": [0.0], "Export_kWh": [0.0]}
    )
    assert price_year_rows(rows.assign(Load_kWh=0.0), _FRAME_COSTS)["Replacement_Cost"].tolist() == [640.0]


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


def test_recalculate_economics_prices_capacity_only_outputs(tmp_path):
    frame, replaced = _frame_with_swaps(1, Battery_Replaced_Capacity_Wh=0.0)
    frame.loc[replaced, "Battery_Replaced_Capacity_Wh"] = 5000.0
    frame["Datetime"] = frame["Datetime"].dt.strftime("%d/%m/%Y %H:%M")
    frame.to_csv(tmp_path / "hourly_results.csv", index=False)
    pd.DataFrame(
        {
            "Year": [1, 2],
            "Cost_System_Annual": [100.0, 100.0],
            "Cost_System_Cumulative": [1100.0, 1200.0],
            "Cost_No_Sys_Annual": [300.0, 306.0],
        }
    ).to_csv(tmp_path / "cost_projection.csv", index=False)
    assert recalculate_economics.swapped_pack_kwh(frame) == 5.0
    assert recalculate_economics.recalculate_dir(tmp_path).startswith("OK")
    projection = pd.read_csv(tmp_path / "cost_projection.csv")
    price = recalculate_economics.NEW_PRICES["pt"]["storage_cost_per_kwh"]
    assert projection["Cost_Replacement"].iloc[0] == pytest.approx(
        5.0 * price * 1.02 ** projection["Replacement_Time_Years"].iloc[0]
    )
