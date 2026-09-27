"""One projection loop, carry state and year row for App and Monte Carlo (#179)."""

import numpy as np
import pandas as pd
import pytest

from breos.app_config import resolve_app_config
from breos.battery import align_simulation_inputs
from breos.projection import CarryState, ProjectionYear, run_projection, value_projection

YEARS = 3


def _inputs():
    """One hourly year whose evening discharge runs past midnight."""
    idx = pd.date_range("2025-01-01 00:00", periods=8760, freq="h", tz="UTC")
    hour = idx.hour.to_numpy()
    pv = pd.Series(np.where((hour >= 9) & (hour < 16), 2600.0, 0.0), index=idx)
    load = pd.DataFrame({"Load": np.where((hour >= 19) | (hour < 3), 700.0, 250.0)}, index=idx)
    return pv, load, pd.Series(20.0, index=idx)


def _resolved(**overrides):
    return resolve_app_config(
        {
            "location": "porto",
            "n_modules": 6,
            "annual_consumption_kwh": 4000,
            "battery_kwh": 5.0,
            "cost_preset": "residential_pt",
            "pv_degradation_rate": 0.005,
            **overrides,
        }
    )


def _run(resolved, *, summary, has_battery=True):
    pv, load, temperature = _inputs()
    # As Monte Carlo aligns them: a PV-only year has no battery temperature.
    aligned = align_simulation_inputs(pv, load, temperature if has_battery else None, freq="h")
    rate = resolved.cfg["pv_degradation_rate"]

    def year_inputs(year_idx):
        factor = (1 - rate) ** year_idx
        if summary:
            return ProjectionYear(
                pv_degradation_factor=factor, aligned=aligned.scaled(pv_factor=factor, load_factor=1.0)
            )
        return ProjectionYear(
            pv_degradation_factor=factor, pv_dc=pv * factor, houseload=load, temperature_series=temperature
        )

    return run_projection(
        resolved.cfg, resolved, YEARS, year_inputs, has_battery=has_battery, execution_backend="python"
    )


@pytest.mark.parametrize(
    ("overrides", "has_battery"),
    [({}, True), ({"battery_eol_percentage": 0.99, "enable_resistance_fade": True}, True), ({"battery_kwh": 0}, False)],
    ids=["battery", "replacement-and-fade", "pv-only"],
)
def test_frames_and_summaries_give_the_same_year_rows(overrides, has_battery):
    resolved = _resolved(**overrides)

    frames = _run(resolved, summary=False, has_battery=has_battery)
    summaries = _run(resolved, summary=True, has_battery=has_battery)

    pd.testing.assert_frame_equal(frames.yearly_df, summaries.yearly_df, check_exact=True)
    assert frames.carry == summaries.carry
    assert frames.total_replacements == summaries.total_replacements
    assert frames.total_replacement_cost == summaries.total_replacement_cost
    assert frames.first_year_results_df is not None and summaries.first_year_results_df is None
    if "battery_eol_percentage" in overrides:
        assert frames.total_replacements > 0
        assert frames.yearly_df["Battery_Resistance_Growth"].iloc[-1] > 0


def test_year_rows_carry_the_battery_across_years():
    run = _run(_resolved(), summary=True)
    rows = run.yearly_df

    assert list(rows["Year"]) == [1, 2, 3]
    assert rows["Battery_SOH_%"].is_monotonic_decreasing
    assert rows["Battery_Cumulative_FEC"].is_monotonic_increasing
    assert rows["Battery_SOH_%"].iloc[-1] == run.carry.soh_pct
    assert rows["Battery_Carried_Energy_Wh"].iloc[-1] == run.carry.energy_wh
    np.testing.assert_allclose(
        rows["PV_Production_kWh"],
        rows["Direct_PV_AC_Load_kWh"] + rows["PV_Origin_Battery_AC_Load_kWh"] + rows["Export_kWh"],
    )


def test_carry_state_starts_year_one_at_the_battery_initial_state():
    first = CarryState().simulation_kwargs()

    assert "initial_energy_wh" not in first
    assert first["initial_degradation_state"] is None
    later = CarryState(energy_wh=1234.0, pv_origin_energy_wh=None).simulation_kwargs()
    assert (later["initial_energy_wh"], later["initial_pv_origin_energy_wh"]) == (1234.0, 0.0)


def test_value_projection_prices_the_year_rows():
    resolved = _resolved()
    run = _run(resolved, summary=True)

    value = value_projection(resolved.cfg, resolved, run)
    costs, projection, lcoe = value.costs, value.cost_projection, value.lcoe
    assert {"Import_Cost", "Export_Revenue", "Fixed_Charge", "Baseline_Import_Cost"} <= set(value.yearly_df)

    assert list(projection["Year"]) == [1, 2, 3]
    assert costs["total_initial_cost"] > 0
    assert np.isfinite(lcoe) and lcoe > 0
