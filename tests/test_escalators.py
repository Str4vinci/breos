"""Separate escalators for import, O&M and replacement learning (ADR 0003 E1, E2)."""

import numpy as np
import pandas as pd
import pytest

from breos import App, optimization
from breos.economics import cost_analysis_projection, projection_rates_record

COSTS = {
    "total_initial_cost": 6000.0,
    "electricity_cost": 0.21,
    "electricity_sold_cost": 0.05,
    "daily_power_cost": 0.30,
    "annual_operation_cost": 40.0,
}
YEARS = 5


def _rows():
    return pd.DataFrame(
        {
            "Year": np.arange(1, YEARS + 1),
            "Load_kWh": [4000.0] * YEARS,
            "PV_Production_kWh": [3000.0] * YEARS,
            "Import_kWh": [2100.0] * YEARS,
            "Export_kWh": [900.0] * YEARS,
            "PV_Degradation_Factor": [1.0] * YEARS,
            "Replacement_Cost": [0.0, 0.0, 1500.0, 0.0, 0.0],
            "Replacement_Year_Fraction": [np.nan, np.nan, 0.5, np.nan, np.nan],
            "Simulated_Hours": [8760.0] * YEARS,
        }
    )


def _project(**rates):
    return cost_analysis_projection(None, COSTS, num_years=YEARS, yearly_summary_df=_rows(), **rates)


def test_omitted_escalators_reproduce_the_single_inflation_rate():
    default = _project(inflation_rate=0.025)
    explicit = _project(inflation_rate=0.025, import_price_escalation=0.025, om_escalation=0.025)

    pd.testing.assert_frame_equal(default, explicit, check_exact=True)


def test_each_escalator_moves_only_its_flow():
    base = _project(inflation_rate=0.02)
    importing = _project(inflation_rate=0.02, import_price_escalation=0.06)
    om = _project(inflation_rate=0.02, om_escalation=0.06)

    for column in ("Cost_Import", "Cost_Daily", "Cost_No_Sys_Annual"):
        assert importing[column].iloc[-1] > base[column].iloc[-1]
        pd.testing.assert_series_equal(om[column], base[column])
    assert importing["Cost_Import"].iloc[-1] == pytest.approx(2100 * 0.21 * 1.06**4)
    assert om["Cost_Operation"].iloc[-1] == pytest.approx(40 * 1.06**4)
    pd.testing.assert_series_equal(importing["Cost_Operation"], base["Cost_Operation"])
    pd.testing.assert_series_equal(importing["Revenue_Export"], base["Revenue_Export"])
    pd.testing.assert_series_equal(importing["Cost_Replacement"], base["Cost_Replacement"])


def test_replacement_learning_lowers_replacements_only():
    base = _project(inflation_rate=0.02)
    learning = _project(inflation_rate=0.02, replacement_cost_learning=0.05)

    t = base["Replacement_Time_Years"].iloc[2]
    assert learning["Cost_Replacement"].iloc[2] == pytest.approx(1500 * 1.02**t * 0.95**t)
    assert learning["Cost_Replacement"].iloc[2] < base["Cost_Replacement"].iloc[2]
    pd.testing.assert_series_equal(learning["Cost_Import"], base["Cost_Import"])


def test_provenance_records_the_rates_used():
    record = projection_rates_record(
        {"inflation_rate": 0.02, "sell_price_inflation": 0.01, "discount_rate": 0.05, "om_escalation": 0.03}
    )

    assert record["import_price_escalation"] == 0.02  # inherited
    assert record["om_escalation"] == 0.03
    assert record["export_price_escalation"] == 0.01
    assert record["replacement_cost_learning"] == 0.0
    assert record["real_discount_rate"] == pytest.approx(1.05 / 1.02 - 1)
    assert record["basis"] == "nominal"


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"import_price_escalation": -1.0}, "'import_price_escalation' must be greater than -1"),
        ({"om_escalation": -2}, "'om_escalation' must be greater than -1"),
        ({"replacement_cost_learning": 1.0}, "'replacement_cost_learning' must be at least 0 and below 1"),
        ({"replacement_cost_learning": -0.1}, "'replacement_cost_learning' must be at least 0 and below 1"),
    ],
)
def test_escalator_config_is_checked(config, message):
    with pytest.raises(ValueError, match=message):
        App({"location": "porto", "n_modules": 6, "annual_consumption_kwh": 3000, **config})


@pytest.mark.usefixtures("_patch_weather")
def test_app_threads_the_escalators_through():
    base = {
        "location": "porto",
        "n_modules": 8,
        "annual_consumption_kwh": 4000,
        "battery_kwh": 5,
        "projection_years": 5,
    }
    default, faster = App(base), App({**base, "import_price_escalation": 0.05})
    default.simulate()
    faster.simulate()

    assert faster.result()["npv_savings"] > default.result()["npv_savings"]
    assert faster.result()["provenance"]["economics"]["import_price_escalation"] == 0.05
    assert default.result()["provenance"]["economics"]["import_price_escalation"] == 0.02


def _optimizer_case(monkeypatch, financials):
    index = pd.date_range("2026-01-05", periods=48, freq="h", tz="Europe/Lisbon")
    pv = pd.Series(np.where((index.hour >= 10) & (index.hour < 16), 1800.0, 0.0), index=index)
    monkeypatch.setattr(optimization, "calculate_pv_production_dc", lambda **kwargs: pv.copy())
    config = {
        "location": {"latitude": 41.15, "longitude": -8.61, "timezone": "Europe/Lisbon", "altitude": 100},
        "simulation": {"resolution": "h", "years_projection": 3},
        "battery": {"temperature": 20.0},
        "financials": financials,
    }
    return optimization.evaluate_projected_design(
        pd.DataFrame({"temp_air": 20.0}, index=index),
        pd.DataFrame({"Load": 1000.0}, index=index),
        config,
        n_modules=4,
        battery_kwh=0.0,
        tilt=30.0,
        azimuth=180.0,
    )


def test_optimizer_prices_and_records_its_financials_escalators(monkeypatch):
    base = _optimizer_case(monkeypatch, {"inflation_rate": 0.02})
    faster = _optimizer_case(monkeypatch, {"inflation_rate": 0.02, "import_price_escalation": 0.08})

    pd.testing.assert_series_equal(faster.financial["Cost_Operation"], base.financial["Cost_Operation"])
    assert faster.financial["Cost_Import"].iloc[-1] == pytest.approx(base.financial["Cost_Import"].iloc[0] * 1.08**2)
    assert base.provenance["economics"]["import_price_escalation"] == 0.02
    assert faster.provenance["economics"]["import_price_escalation"] == 0.08
