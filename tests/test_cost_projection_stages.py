"""cost_analysis_projection is valuation, discounting, emissions and file output in turn (#183)."""

import pandas as pd
import pytest

from breos.economics import (
    add_co2_projection,
    cost_analysis_projection,
    discount_cashflows,
    price_year_rows,
    value_year_rows,
    write_cost_projection,
)
from breos.emissions import EmissionsParams

COSTS = {
    "total_initial_cost": 8000.0,
    "electricity_cost": 0.22,
    "electricity_sold_cost": 0.06,
    "daily_power_cost": 0.3,
    "annual_operation_cost": 40.0,
    "replacement_cost_each": 2100.0,
}
EMISSIONS = EmissionsParams(
    average_grid_carbon_intensity_gco2_kwh=187.3, export_displacement_carbon_intensity_gco2_kwh=143.1
)
RATES = {
    "inflation_rate": 0.021,
    "sell_price_inflation": 0.01,
    "om_escalation": 0.03,
    "replacement_cost_learning": 0.02,
}


def _rows():
    return pd.DataFrame(
        {
            "Year": [1, 2, 3],
            "Load_kWh": [4000.0, 4010.0, 3990.0],
            "Import_kWh": [1500.0, 1520.0, 1540.0],
            "Export_kWh": [900.0, 880.0, 870.0],
            "PV_Production_kWh": [3400.0, 3383.0, 3366.1],
            "PV_Degradation_Factor": [1.0, 0.995, 0.990025],
            "Replacements": [0, 1, 0],
            "Replacement_Year_Fraction": [float("nan"), 0.4, float("nan")],
            "Simulated_Hours": [8760.0, 8760.0, 8784.0],
        }
    )


def test_the_stages_compose_to_the_projection(tmp_path):
    whole = cost_analysis_projection(
        None,
        COSTS,
        num_years=3,
        discount_rate=0.04,
        yearly_summary_df=_rows(),
        emissions_params=EMISSIONS,
        results_directory=str(tmp_path / "whole"),
        **RATES,
    )
    rows = price_year_rows(_rows(), COSTS)
    staged = discount_cashflows(
        value_year_rows(rows, COSTS, **RATES), total_investment=COSTS["total_initial_cost"], discount_rate=0.04
    )
    add_co2_projection(staged, rows, EMISSIONS)
    pd.testing.assert_frame_equal(staged, whole, check_exact=True)
    assert staged.attrs == whole.attrs
    path = write_cost_projection(staged, str(tmp_path / "staged"))
    assert (tmp_path / "staged" / "cost_projection.csv").read_bytes() == (
        tmp_path / "whole" / "cost_projection.csv"
    ).read_bytes()
    assert path.endswith("cost_projection.csv")


def test_valuation_does_not_discount():
    flows = value_year_rows(price_year_rows(_rows(), COSTS), COSTS, **RATES)
    assert not any(column.endswith("_NPV") for column in flows.columns)
    assert flows["Cost_Replacement"].iloc[1] == pytest.approx(2100.0 * (1.021 * 0.98) ** 1.4)
    assert flows.attrs["total_replacement_cost"] == 2100.0


def test_the_projection_carries_the_lcoe_and_co2_every_runner_reports():
    projection = cost_analysis_projection(
        None, COSTS, num_years=3, discount_rate=0.04, yearly_summary_df=_rows(), emissions_params=EMISSIONS, **RATES
    )
    assert projection.attrs["lcoe_per_kwh"] > 0
    total = projection["CO2_Avoided_SelfConsumed_kg"] + projection["CO2_Avoided_Export_kg"]
    pd.testing.assert_series_equal(projection["CO2_Avoided_Total_kg"], total, check_names=False)
    assert projection.attrs["lifetime_co2_avoided_export_kg"] == projection["CO2_Avoided_Export_Cumulative_kg"].iloc[-1]
