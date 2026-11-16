"""Projected optimization uses the canonical yearly simulation ledger."""

import numpy as np
import pandas as pd
import pytest

from breos.battery import SimulationSummary
from breos.economics import DEFAULT_DISCOUNT_RATE, DEFAULT_INFLATION_RATE
from breos.optimization import (
    _evaluate_projected_design_metrics,
    _summarize_projected_lifetime_metrics,
    evaluate_projected_design,
)
from breos.projection import _ROW_SUM_COLUMNS
from breos.pv_modules import get_module

# The rates resolve_optimization_config fills in; the private evaluator reads
# them from the resolved financials rather than defaulting them again.
_RATES = {
    "inflation_rate": DEFAULT_INFLATION_RATE,
    "sell_price_inflation": 0.0,
    "discount_rate": DEFAULT_DISCOUNT_RATE,
}


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
    load = pd.DataFrame({"Load": [500.0, 500.0]}, index=idx)
    temperature = pd.Series([20.0, 20.0], index=idx)
    calls = []

    def fake_summary(**kwargs):
        calls.append(kwargs)
        year = len(calls)
        results = pd.DataFrame(
            {
                # Every column the year row reads, zero unless set below.
                **{column: [0.0, 0.0] for column in _ROW_SUM_COLUMNS},
                "PV_AC_To_Load": [300.0, 300.0],
                "PV_Origin_Battery_AC_To_Load": [50.0, 50.0],
                "PV_AC_Export": [50.0, 50.0],
                "Houseload": [500.0, 500.0],
                "Import_From_Grid": [150.0 * year, 150.0 * year],
                "Battery_Energy_End": [100.0, 100.0 + year],
                "Battery_PV_Origin_Energy_End": [50.0, 50.0 + year],
                "Battery_Grid_Origin_Energy_End": [0.0, 0.0],
            },
            index=idx,
        )
        replacements = 1 if year == 2 else 0
        return SimulationSummary(
            n_steps=len(results),
            hours_per_step=1.0,
            column_sums={column: float(results[column].sum()) for column in results.columns},
            total_pv_wh=0.0,
            summary_row={},
            final_soh_percent=95.0 if year == 1 else 99.0,
            n_replacements=replacements,
            replaced_capacity_wh=2000.0 * replacements,
            opening_energy_wh=0.0,
            opening_pv_origin_energy_wh=0.0,
            opening_grid_origin_energy_wh=0.0,
            carried_energy_wh=float(results["Battery_Energy_End"].iloc[-1]),
            carried_pv_origin_energy_wh=float(results["Battery_PV_Origin_Energy_End"].iloc[-1]),
            carried_grid_origin_energy_wh=0.0,
            has_degradation_rows=True,
            fec_cum=10.0 * year,
            cumulative_calendar_seconds=1000.0 * year,
            cumulative_cycle_degradation=0.01 * year,
            cumulative_calendar_degradation=0.02 * year,
            resistance_growth=0.03 * year,
            replacement_steps=(1,) if replacements else (),
            fec_all_packs=4.0 * year,
            final_degradation_state={"year": year},
        )

    captured = {}

    def fake_projection(**kwargs):
        captured["yearly"] = kwargs["yearly_summary_df"].copy()
        captured["costs"] = dict(kwargs["costs"])
        projection = pd.DataFrame(
            {
                "Year": [1, 2],
                "Savings_Cumulative_NPV": [-100.0, 50.0],
                "CO2_Avoided_Total_Cumulative_kg": [10.0, 21.0],
                "CO2_Avoided_SelfConsumed_Cumulative_kg": [8.0, 17.0],
            }
        )
        projection.attrs["payback_year"] = 2
        projection.attrs["lcoe_per_kwh"] = 0.123
        projection.attrs["total_replacement_cost"] = 1000.0
        return projection

    monkeypatch.setattr("breos.projection.simulate_energy_balance_summary", fake_summary)
    monkeypatch.setattr("breos.optimization.cost_analysis_projection", fake_projection)

    metrics = _evaluate_projected_design_metrics(
        base_dc_power=base_dc,
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
        fin_cfg={**_RATES, "project_lifespan": 2},
        freq="h",
        years_projection=2,
        degradation_rate=0.10,
        n_modules=2,
        battery_kwh=2.0,
        inverter_efficiency=0.96,
        inverter_ac_capacity_w=1000.0,
        return_tables=True,
    )

    assert np.array_equal(calls[0]["aligned"].pv_dc_w, base_dc.to_numpy())
    assert np.array_equal(calls[1]["aligned"].pv_dc_w, base_dc.to_numpy() * 0.9)
    assert np.array_equal(calls[1]["aligned"].temperature_c, temperature.to_numpy())
    assert calls[1]["initial_fec"] == pytest.approx(10.0)
    assert calls[1]["initial_calendar_seconds"] == pytest.approx(1000.0)
    assert calls[1]["initial_resistance_growth"] == pytest.approx(0.03)
    assert calls[1]["initial_cumulative_cycle_deg"] == pytest.approx(0.01)
    assert calls[1]["initial_cumulative_cal_deg"] == pytest.approx(0.02)
    assert calls[1]["initial_energy_wh"] == pytest.approx(101.0)
    assert calls[1]["initial_pv_origin_energy_wh"] == pytest.approx(51.0)
    assert captured["yearly"]["Import_kWh"].tolist() == pytest.approx([0.3, 0.6])
    # The physics reports the swap and its capacity; the economics prices it (ADR 0003 E4).
    assert captured["yearly"]["Replacements"].tolist() == [0, 1]
    assert captured["yearly"]["Replaced_Capacity_kWh"].tolist() == pytest.approx([0.0, 2.0])
    assert captured["costs"]["replacement_cost_each"] == pytest.approx(1000.0)
    assert metrics["Projected_NPV"] == pytest.approx(50.0)
    assert metrics["Projected_Total_Replacements"] == 1
    assert metrics["Projected_Replacement_Cost_T0_Prices"] == pytest.approx(1000.0)
    assert metrics["Projected_Final_SOH_%"] == pytest.approx(99.0)
    assert metrics["Projected_Payback_Year"] == pytest.approx(2.0)
    assert "Projected_Breakeven_Year" not in metrics
    assert "Projected_Breakeven_Year_Interpolated" not in metrics
    assert metrics["Projected_LCOE_per_kWh"] == pytest.approx(0.123)
    assert metrics["Projected_CO2_Avoided_Total_kg"] == pytest.approx(21.0)
    assert metrics["Projected_CO2_Avoided_SelfConsumed_kg"] == pytest.approx(17.0)
    assert metrics["_yearly_summary_df"]["Battery_Cumulative_FEC"].tolist() == pytest.approx([10.0, 20.0])
    assert metrics["_yearly_summary_df"]["Battery_Annual_FEC"].tolist() == pytest.approx([4.0, 8.0])
    assert metrics["_yearly_summary_df"]["Battery_Cumulative_Calendar_Degradation"].tolist() == pytest.approx(
        [0.02, 0.04]
    )


class TestZeroModuleDesign:
    """The battery-only corner of an exhaustive sizing lattice.

    The projected evaluator used to require at least one module, so the grid
    could not measure the no-PV slice and had to assume it dominated. It is a
    real coordinate, and this is the boundary the relaxed contract opens.
    """

    @staticmethod
    def _config():
        return {
            "location": {"latitude": 41.15, "longitude": -8.63},
            "pv": {"module": "Suntech_STP550S_STC"},
            "simulation": {"resolution": "h", "years_projection": 1},
            "financials": {"project_lifespan": 1},
            "costs": {"dc_ac_ratio": 1.25},
        }

    @staticmethod
    def _inputs(periods: int = 48):
        idx = pd.date_range("2025-01-01", periods=periods, freq="h", tz="UTC")
        weather = pd.DataFrame({"temp_air": np.full(periods, 20.0)}, index=idx)
        load = pd.DataFrame({"Load": np.full(periods, 500.0)}, index=idx)
        return idx, weather, load

    def test_a_negative_module_count_is_still_rejected(self):
        _idx, weather, load = self._inputs(periods=2)
        with pytest.raises(ValueError, match="n_modules must be non-negative"):
            evaluate_projected_design(
                weather,
                load,
                self._config(),
                n_modules=-1,
                battery_kwh=5.0,
                tilt=35.0,
                azimuth=180.0,
            )

    def test_zero_modules_evaluates_as_a_battery_only_design(self, monkeypatch):
        """No PV production, no inverter rating, and an undefined LCOE.

        Every kWh is imported, so the projection has no generation to spread
        its cost over and LCOE is infinite rather than a divide-by-zero.
        """
        idx, weather, load = self._inputs()
        seen = {}

        def _no_pv(**kwargs):
            seen["n_modules"] = kwargs["n_modules"]
            return pd.Series(np.zeros(len(idx)), index=idx)

        monkeypatch.setattr("breos.optimization.calculate_pv_production_dc", _no_pv)

        result = evaluate_projected_design(
            weather,
            load,
            self._config(),
            n_modules=0,
            battery_kwh=5.0,
            tilt=35.0,
            azimuth=180.0,
        )

        assert seen["n_modules"] == 0
        assert result.metrics["Modules"] == 0
        assert result.metrics["Projected_PV_Production_Year1_kWh"] == pytest.approx(0.0)
        assert result.metrics["Projected_PV_DC_Year1_kWh"] == pytest.approx(0.0)
        assert np.isinf(result.metrics["Projected_LCOE_per_kWh"])
        # A design that generates nothing cannot displace any import.
        assert result.metrics["Projected_Grid_Independence_%"] == pytest.approx(0.0)
        assert result.yearly["Import_kWh"].iloc[0] == pytest.approx(result.yearly["Load_kWh"].iloc[0], rel=1e-9)
        # The cost projection comes back as the plot-source financial table.
        assert result.financial["Savings_Cumulative_NPV"].size == 1

    def test_zero_modules_gives_the_inverter_no_rating(self, monkeypatch):
        """The rating is sized from the array, so the no-PV corner has none."""
        idx, weather, load = self._inputs(periods=2)
        seen = {}

        monkeypatch.setattr(
            "breos.optimization.calculate_pv_production_dc",
            lambda **_kwargs: pd.Series(np.zeros(len(idx)), index=idx),
        )

        def _capture(**kwargs):
            seen["inverter_ac_capacity_w"] = kwargs["inverter_ac_capacity_w"]
            return {
                "Projected_Grid_Independence_%": 0.0,
                "Projected_NPV": 0.0,
                "_yearly_summary_df": pd.DataFrame({"Year": [1]}),
                "_cost_projection_df": pd.DataFrame({"Year": [1]}),
            }

        monkeypatch.setattr("breos.optimization._evaluate_projected_design_metrics", _capture)
        evaluate_projected_design(
            weather,
            load,
            self._config(),
            n_modules=0,
            battery_kwh=5.0,
            tilt=35.0,
            azimuth=180.0,
        )
        assert seen["inverter_ac_capacity_w"] == pytest.approx(0.0)
