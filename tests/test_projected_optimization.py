"""Projected optimization uses the canonical yearly simulation ledger."""

import numpy as np
import pandas as pd
import pytest

from breos.optimization import (
    ProjectedDesignResult,
    _evaluate_projected_design_metrics,
    _summarize_projected_lifetime_metrics,
    evaluate_projected_design,
)
from breos.pv_modules import get_module


def test_lifetime_grid_independence_uses_total_import_and_load():
    yearly = pd.DataFrame(
        {
            "Year": [1, 2],
            "Load_kWh": [100.0, 200.0],
            "Import_kWh": [20.0, 100.0],
            "PV_Production_kWh": [80.0, 120.0],
            "Grid_Independence_%": [80.0, 50.0],
        }
    )

    metrics = _summarize_projected_lifetime_metrics(yearly)

    assert metrics["Projected_Grid_Independence_%"] == pytest.approx(60.0)
    assert metrics["Projected_Grid_Independence_Mean_%"] == pytest.approx(65.0)
    assert metrics["Projected_ZEB_Ratio"] == pytest.approx(2.0 / 3.0)


def test_projected_evaluator_carries_physical_and_degradation_state(monkeypatch):
    idx = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
    base_dc = pd.Series([1000.0, 500.0], index=idx)
    tmy = pd.DataFrame({"temp_air": [20.0, 20.0], "ghi": [500.0, 250.0]}, index=idx)
    load = pd.DataFrame({"Load": [500.0, 500.0]}, index=idx)
    temperature = pd.Series([20.0, 20.0], index=idx)
    calls = []

    def fake_balance(**kwargs):
        calls.append(kwargs)
        year = len(calls)
        results = pd.DataFrame(
            {
                "PV_AC_To_Load": [300.0, 300.0],
                "Battery_AC_To_Load_PV": [50.0, 50.0],
                "PV_AC_Export": [50.0, 50.0],
                "Houseload": [500.0, 500.0],
                "Import_From_Grid": [150.0 * year, 150.0 * year],
                "Sell_To_Grid": [50.0, 50.0],
                "Battery_Energy_End": [100.0, 100.0 + year],
                "Battery_PV_Origin_Energy_End": [50.0, 50.0 + year],
            },
            index=idx,
        )
        degradation = pd.DataFrame(
            {
                "Cumulative_FEC": [10.0 * year],
                "Cumulative_Calendar_Seconds": [1000.0 * year],
                "Cumulative_Cycle_Degradation": [0.01 * year],
                "Cumulative_Calendar_Degradation": [0.02 * year],
                "Resistance_Growth": [0.03 * year],
                "SOH": [95.0 if year == 1 else 99.0],
            }
        )
        replacements = 1 if year == 2 else 0
        replacement_cost = 1000.0 if replacements else 0.0
        return results, 0.0, pd.DataFrame(), replacement_cost, replacements, degradation, {"year": year}

    captured = {}

    def fake_projection(**kwargs):
        captured["yearly"] = kwargs["yearly_summary_df"].copy()
        projection = pd.DataFrame(
            {
                "Year": [1, 2],
                "Savings_Cumulative_NPV": [-100.0, 50.0],
                "CO2_Avoided_Total_Cumulative_kg": [10.0, 21.0],
                "CO2_Avoided_SelfConsumed_Cumulative_kg": [8.0, 17.0],
            }
        )
        projection.attrs["payback_year"] = 2
        return projection

    monkeypatch.setattr("breos.optimization.simulate_energy_balance", fake_balance)
    monkeypatch.setattr("breos.optimization.cost_analysis_projection", fake_projection)
    monkeypatch.setattr("breos.optimization.calculate_lcoe_from_projection", lambda *_args, **_kwargs: 0.123)

    metrics = _evaluate_projected_design_metrics(
        base_dc_power=base_dc,
        tmy_data=tmy,
        houseload=load,
        temperature_series=temperature,
        pv_params=get_module("Suntech_STP550S_STC"),
        batt_spec={
            "min_soc": 0.1,
            "max_soc": 0.9,
            "enable_replacement": True,
            "enable_resistance_fade": True,
        },
        costs_cfg={"storage_cost_per_kwh": 500.0},
        fin_cfg={"project_lifespan": 2},
        freq="h",
        years_projection=2,
        degradation_rate=0.10,
        n_modules=2,
        battery_kwh=2.0,
        inverter_efficiency=0.96,
        inverter_ac_capacity_w=1000.0,
        return_tables=True,
    )

    assert np.array_equal(calls[0]["pv_dc"].to_numpy(), base_dc.to_numpy())
    assert np.allclose(calls[1]["pv_dc"].to_numpy(), base_dc.to_numpy() * 0.9)
    assert calls[1]["initial_fec"] == pytest.approx(10.0)
    assert calls[1]["initial_calendar_seconds"] == pytest.approx(1000.0)
    assert calls[1]["initial_resistance_growth"] == pytest.approx(0.03)
    assert calls[1]["initial_cumulative_cycle_deg"] == pytest.approx(0.01)
    assert calls[1]["initial_cumulative_cal_deg"] == pytest.approx(0.02)
    assert calls[1]["initial_energy_wh"] == pytest.approx(101.0)
    assert calls[1]["initial_pv_origin_energy_wh"] == pytest.approx(51.0)
    assert calls[1]["battery_config"].replacement_cost == pytest.approx(1000.0)
    assert captured["yearly"]["Import_kWh"].tolist() == pytest.approx([0.3, 0.6])
    assert captured["yearly"]["Replacement_Cost"].tolist() == pytest.approx([0.0, 1000.0])
    assert metrics["Projected_NPV_Eur"] == pytest.approx(50.0)
    assert metrics["Projected_Total_Replacements"] == 1
    assert metrics["Projected_Replacement_Cost_Eur"] == pytest.approx(1000.0)
    assert metrics["Projected_Final_SOH_%"] == pytest.approx(99.0)
    assert metrics["Projected_Breakeven_Year"] == pytest.approx(2.0)
    assert metrics["Projected_LCOE_Eur_kWh"] == pytest.approx(0.123)
    assert metrics["Projected_CO2_Avoided_Total_kg"] == pytest.approx(21.0)
    assert metrics["Projected_CO2_Avoided_SelfConsumed_kg"] == pytest.approx(17.0)
    assert metrics["_yearly_summary_df"]["Battery_Cumulative_FEC"].tolist() == pytest.approx([10.0, 20.0])
    assert metrics["_yearly_summary_df"]["Battery_Cumulative_Calendar_Degradation"].tolist() == pytest.approx(
        [0.02, 0.04]
    )


def test_public_projected_design_evaluator_returns_plot_source_tables(monkeypatch):
    idx = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
    weather = pd.DataFrame({"temp_air": [20.0, 20.0]}, index=idx)
    load = pd.DataFrame({"Load": [500.0, 500.0]}, index=idx)
    yearly = pd.DataFrame({"Year": [1], "Grid_Independence_%": [60.0]})
    financial = pd.DataFrame({"Year": [1], "Savings_Cumulative_NPV": [100.0]})

    monkeypatch.setattr(
        "breos.optimization.calculate_pv_production_dc",
        lambda **_kwargs: pd.Series([100.0, 200.0], index=idx),
    )
    monkeypatch.setattr(
        "breos.optimization._temperature_series_from_config",
        lambda *_args, **_kwargs: pd.Series([20.0, 20.0], index=idx),
    )
    monkeypatch.setattr(
        "breos.optimization._evaluate_projected_design_metrics",
        lambda **_kwargs: {
            "Projected_Grid_Independence_%": 60.0,
            "Projected_NPV_Eur": 100.0,
            "_yearly_summary_df": yearly,
            "_cost_projection_df": financial,
        },
    )

    result = evaluate_projected_design(
        weather,
        load,
        {
            "location": {"latitude": 41.15, "longitude": -8.63},
            "pv": {"module": "Suntech_STP550S_STC"},
            "simulation": {"resolution": "h", "years_projection": 1},
            "financials": {"project_lifespan": 1},
            "costs": {"dc_ac_ratio": 1.25},
        },
        n_modules=9,
        battery_kwh=5.0,
        tilt=25.0,
        azimuth=185.0,
    )

    assert isinstance(result, ProjectedDesignResult)
    assert result.metrics["Modules"] == 9
    assert result.metrics["Projected_NPV_Eur"] == pytest.approx(100.0)
    pd.testing.assert_frame_equal(result.yearly, yearly)
    pd.testing.assert_frame_equal(result.financial, financial)
