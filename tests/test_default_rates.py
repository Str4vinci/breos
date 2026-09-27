"""One default discount and inflation rate for every entry point (ADR 0003 E6)."""

import inspect

import pandas as pd
import pytest

from breos.app_config import DEFAULTS, resolve_app_config
from breos.economics import (
    DEFAULT_DISCOUNT_RATE,
    DEFAULT_INFLATION_RATE,
    CostParams,
    calculate_lcoe,
    calculate_lcoe_from_projection,
    cost_analysis_projection,
    cost_params_from_config,
)


def test_the_defaults_are_three_and_two_percent():
    assert (DEFAULT_DISCOUNT_RATE, DEFAULT_INFLATION_RATE) == (0.03, 0.02)


def test_every_entry_point_defaults_to_the_one_set():
    params = CostParams()
    assert (params.discount_rate, params.inflation_rate) == (0.03, 0.02)
    from_config = cost_params_from_config({}, {})
    assert (from_config.discount_rate, from_config.inflation_rate) == (0.03, 0.02)
    assert (DEFAULTS["discount_rate"], DEFAULTS["inflation_rate"]) == (0.03, 0.02)
    projection = inspect.signature(cost_analysis_projection).parameters
    assert (projection["discount_rate"].default, projection["inflation_rate"].default) == (0.03, 0.02)
    assert inspect.signature(calculate_lcoe_from_projection).parameters["discount_rate"].default == 0.03
    assert inspect.signature(calculate_lcoe).parameters["discount_rate"].default == 0.03


def test_the_packaged_financials_preset_is_gone():
    from importlib.resources import files

    assert not files("breos.data.configs").joinpath("financials.json").is_file()


def test_an_explicit_zero_is_used_as_given():
    params = cost_params_from_config({}, {"discount_rate": 0.0, "inflation_rate": 0.0})
    assert (params.discount_rate, params.inflation_rate) == (0.0, 0.0)
    resolved = resolve_app_config(
        {
            "location": "porto",
            "n_modules": 4,
            "annual_consumption_kwh": 3000,
            "discount_rate": 0.0,
            "inflation_rate": 0.0,
        }
    )
    assert (resolved.cfg["discount_rate"], resolved.cfg["inflation_rate"]) == (0.0, 0.0)
    assert (resolved.cost_params.discount_rate, resolved.cost_params.inflation_rate) == (0.0, 0.0)


def _projection(**rates):
    yearly = pd.DataFrame(
        {
            "Year": [1, 2, 3],
            "Load_kWh": [4000.0] * 3,
            "PV_Production_kWh": [3000.0] * 3,
            "Export_kWh": [1000.0] * 3,
            "Import_kWh": [2000.0] * 3,
            "PV_Degradation_Factor": [1.0, 0.995, 0.990025],
            "Replacement_Cost": [0.0] * 3,
            "Replacement_Year_Fraction": [float("nan")] * 3,
        }
    )
    costs = {
        "total_initial_cost": 5000.0,
        "electricity_cost": 0.2,
        "electricity_sold_cost": 0.05,
        "daily_power_cost": 0.0,
        "annual_operation_cost": 50.0,
    }
    return cost_analysis_projection(None, costs, num_years=3, yearly_summary_df=yearly, **rates)


def test_a_zero_discount_rate_leaves_cashflows_undiscounted():
    zero = _projection(discount_rate=0.0, inflation_rate=0.0)
    default = _projection()

    assert zero["Savings_Cumulative_NPV"].iloc[-1] == pytest.approx(zero["Savings_Cumulative"].iloc[-1])
    assert default["Savings_Cumulative_NPV"].iloc[-1] != pytest.approx(zero["Savings_Cumulative_NPV"].iloc[-1])
    assert calculate_lcoe_from_projection(
        zero, total_investment=5000.0, discount_rate=0.0
    ) < calculate_lcoe_from_projection(zero, total_investment=5000.0)
