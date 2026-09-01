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
                "Cumulative_FEC_All_Packs": [4.0 * year],
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
    assert metrics["_yearly_summary_df"]["Battery_Annual_FEC"].tolist() == pytest.approx([4.0, 8.0])
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


def _synthetic_projection_inputs(days: int = 30):
    """A short, hard-cycled year that is cheap enough to project twice."""
    idx = pd.date_range("2025-01-01 00:00", periods=days * 24, freq="h", tz="UTC")
    daily_pv = [0.0] * 8 + [3000.0] * 8 + [0.0] * 8
    daily_load = [900.0] * 8 + [0.0] * 8 + [900.0] * 8
    base_dc = pd.Series(daily_pv * days, index=idx)
    houseload = pd.DataFrame({"Load": daily_load * days}, index=idx)
    temperature = pd.Series(25.0, index=idx)
    return idx, base_dc, houseload, temperature


def _project(batt_spec, *, battery_kwh, years=2, days=30):
    idx, base_dc, houseload, temperature = _synthetic_projection_inputs(days)
    return _evaluate_projected_design_metrics(
        base_dc_power=base_dc,
        tmy_data=pd.DataFrame({"temp_air": 20.0, "ghi": 500.0}, index=idx),
        houseload=houseload,
        temperature_series=temperature,
        pv_params=get_module("Suntech_STP550S_STC"),
        batt_spec=batt_spec,
        costs_cfg={"storage_cost_per_kwh": 500.0},
        fin_cfg={"project_lifespan": years},
        freq="h",
        years_projection=years,
        degradation_rate=0.005,
        n_modules=6,
        battery_kwh=battery_kwh,
        inverter_efficiency=0.96,
        inverter_ac_capacity_w=2640.0,
        return_tables=True,
    )


def test_annual_fec_spans_a_replacement_inside_one_projected_year():
    # The pack starts a hair above end of life, so it retires on the first
    # simulated day and a fresh pack carries the rest of year 1. Cumulative
    # FEC only ever reports the pack currently installed, so the retired
    # pack's part-year cycles survive only if annual FEC is accumulated
    # separately.
    metrics = _project(
        {
            "min_soc": 0.1,
            "max_soc": 0.9,
            "initial_soh": 70.000001,
            "eol_percentage": 0.70,
            "enable_replacement": True,
        },
        battery_kwh=5.0,
    )
    yearly = metrics["_yearly_summary_df"]

    assert yearly["Replacements"].tolist() == [1, 0]
    year1 = yearly.iloc[0]
    # Both sides of the replacement are present: the annual figure exceeds
    # what the post-replacement pack alone accumulated, and the excess is the
    # retired pack's own part-year count.
    assert year1["Battery_Annual_FEC"] > year1["Battery_Cumulative_FEC"] > 0.0
    retired_pack_fec = year1["Battery_Annual_FEC"] - year1["Battery_Cumulative_FEC"]
    assert retired_pack_fec > 0.0

    # Year 2 has no replacement, so the annual figure is exactly the
    # difference of the cumulative series across the year boundary.
    year2 = yearly.iloc[1]
    assert year2["Battery_Annual_FEC"] == pytest.approx(
        year2["Battery_Cumulative_FEC"] - year1["Battery_Cumulative_FEC"], abs=1e-12
    )


def test_annual_fec_is_zero_without_a_battery():
    metrics = _project({"min_soc": 0.1, "max_soc": 0.9, "enable_replacement": True}, battery_kwh=0.0)
    yearly = metrics["_yearly_summary_df"]

    assert "Battery_Annual_FEC" in yearly.columns
    assert yearly["Battery_Annual_FEC"].tolist() == [0.0, 0.0]
    assert yearly["Battery_Cumulative_FEC"].tolist() == [0.0, 0.0]
    assert metrics["Projected_Total_Replacements"] == 0


def test_annual_fec_differences_the_cumulative_series_without_a_replacement():
    metrics = _project(
        {"min_soc": 0.1, "max_soc": 0.9, "eol_percentage": 0.70, "enable_replacement": True},
        battery_kwh=5.0,
    )
    yearly = metrics["_yearly_summary_df"]

    assert metrics["Projected_Total_Replacements"] == 0
    assert (yearly["Battery_Annual_FEC"] > 0.0).all()
    cumulative = np.concatenate([[0.0], yearly["Battery_Cumulative_FEC"].to_numpy()])
    np.testing.assert_allclose(yearly["Battery_Annual_FEC"].to_numpy(), np.diff(cumulative), atol=1e-12)
