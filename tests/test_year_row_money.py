"""Year rows carry money at year-1 prices; the fixed charge follows the simulated duration (ADR 0003 E5, E7)."""

import numpy as np
import pandas as pd
import pytest

from breos import App
from breos.economics import YEAR_ROW_MONEY_COLUMNS, cost_analysis_projection, price_year_rows

COSTS = {
    "total_initial_cost": 6000.0,
    "electricity_cost": 0.21,
    "electricity_sold_cost": 0.05,
    "daily_power_cost": 0.30,
    "annual_operation_cost": 40.0,
}


def _rows(hours):
    n = len(hours)
    return pd.DataFrame(
        {
            "Year": np.arange(1, n + 1),
            "Load_kWh": [4000.0] * n,
            "PV_Production_kWh": [3000.0] * n,
            "Import_kWh": [2100.0] * n,
            "Export_kWh": [900.0] * n,
            "PV_Degradation_Factor": [1.0] * n,
            "Replacement_Cost": [0.0] * n,
            "Replacement_Year_Fraction": [np.nan] * n,
            "Simulated_Hours": hours,
        }
    )


def test_flat_money_columns_are_energy_times_price():
    priced = price_year_rows(_rows([8760.0]), COSTS)

    assert priced["Import_Cost"].iloc[0] == 2100.0 * 0.21
    assert priced["Export_Revenue"].iloc[0] == 900.0 * 0.05
    assert priced["Baseline_Import_Cost"].iloc[0] == 4000.0 * 0.21
    assert priced["Fixed_Charge"].iloc[0] == 365.0 * 0.30


@pytest.mark.parametrize(("hours", "days"), [(8760.0, 365), (8784.0, 366), (24 * 31.0, 31)])
def test_fixed_charge_is_billed_on_the_simulated_duration(hours, days):
    projection = cost_analysis_projection(_rows([hours]), COSTS, num_years=1, inflation_rate=0.0)

    assert projection["Cost_Daily"].iloc[0] == pytest.approx(days * 0.30, rel=1e-15)
    # The fee is charged with and without the system, so it cancels in savings.
    assert projection["Cost_No_Sys_Annual"].iloc[0] == pytest.approx(4000.0 * 0.21 + days * 0.30)


def test_rows_without_a_duration_are_billed_as_365_days():
    priced = price_year_rows(_rows([8784.0]).drop(columns="Simulated_Hours"), COSTS)
    assert priced["Fixed_Charge"].iloc[0] == 365 * 0.30


def test_money_columns_given_by_the_caller_are_kept():
    rows = _rows([8760.0]).assign(Import_Cost=[123.0])

    priced = price_year_rows(rows, COSTS)
    projection = cost_analysis_projection(rows, COSTS, num_years=1, inflation_rate=0.0)

    assert priced["Import_Cost"].iloc[0] == 123.0
    assert projection["Cost_Import"].iloc[0] == 123.0


@pytest.mark.usefixtures("_patch_weather")
def test_app_rows_and_financial_components_reconcile():
    app = App(
        {"location": "porto", "n_modules": 8, "annual_consumption_kwh": 4000, "battery_kwh": 5, "projection_years": 3}
    )
    app.simulate()
    result = app.result()

    for row in result["financial"][1:]:
        assert {"cost_import", "revenue_export", "cost_operation", "cost_fixed_charge", "cost_replacement"} <= set(row)
    first = result["financial"][1]
    year1 = result["yearly"][0]
    assert first["cost_import"] == pytest.approx(
        year1["import_kwh"] * app._resolved.cost_params.electricity_cost, abs=0.02
    )
    assert set(YEAR_ROW_MONEY_COLUMNS) == {"Import_Cost", "Export_Revenue", "Fixed_Charge", "Baseline_Import_Cost"}
