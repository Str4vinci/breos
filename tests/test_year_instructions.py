"""Instructions per project year, and a planner asked as each year begins (ADR 0002 A16)."""

import math

import numpy as np
import pandas as pd
import pytest

from breos.app_config import resolve_app_config
from breos.battery import align_simulation_inputs, resistance_to_efficiency
from breos.dispatch_instructions import DispatchInstructions
from breos.projection import ProjectionYear, YearStart, run_projection

YEARS = 3


def _inputs():
    """One hourly year with a sunny midday and an evening load the battery serves."""
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


def _night_target(fraction):
    """Grid charging to ``fraction`` of the usable window from 03:00 to 06:00, discharge at every other hour."""
    hour = pd.date_range("2025-01-01 00:00", periods=8760, freq="h", tz="UTC").hour.to_numpy()
    charge = (hour >= 3) & (hour < 6)
    return DispatchInstructions(
        discharge_allowed=~charge,
        reserve_fraction=np.zeros(8760),
        grid_target_fraction=np.where(charge, fraction, np.nan),
        grid_charge_efficiency=0.95,
        grid_import_limit_w=5000.0,
    )


def _run(resolved, instructions, *, summary=False, years=YEARS):
    pv, load, temperature = _inputs()
    aligned = align_simulation_inputs(pv, load, temperature, freq="h")
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
        resolved.cfg,
        resolved,
        years,
        year_inputs,
        has_battery=True,
        execution_backend="python",
        instructions=instructions,
    )


def _assert_same_run(left, right):
    pd.testing.assert_frame_equal(left.yearly_df, right.yearly_df, check_exact=True)
    if left.first_year_results_df is not None:
        pd.testing.assert_frame_equal(left.first_year_results_df, right.first_year_results_df, check_exact=True)
    assert left.carry == right.carry


@pytest.mark.parametrize("summary", [False, True], ids=["frames", "summaries"])
def test_one_set_per_year_repeated_is_the_one_set_every_year_replays(summary):
    resolved = _resolved(enable_resistance_fade=True)
    single = _night_target(0.6)

    once = _run(resolved, single, summary=summary)
    per_year = _run(resolved, [single] * YEARS, summary=summary)

    _assert_same_run(once, per_year)
    assert once.year_instructions == per_year.year_instructions == (single,) * YEARS


def test_a_planner_returning_one_set_is_the_one_set_every_year_replays():
    resolved = _resolved(enable_resistance_fade=True)
    single = _night_target(0.6)

    _assert_same_run(_run(resolved, single), _run(resolved, lambda start: single))


def test_per_year_instructions_change_only_the_years_they_reach():
    resolved = _resolved()
    low, high = _night_target(0.2), _night_target(0.9)

    base = _run(resolved, [low, low, low])
    second = _run(resolved, [low, high, low])

    rows, changed = base.yearly_df, second.yearly_df
    # Year one ran the same set from the same state.
    pd.testing.assert_series_equal(rows.iloc[0], changed.iloc[0], check_exact=True)
    pd.testing.assert_frame_equal(base.first_year_results_df, second.first_year_results_df, check_exact=True)
    # Year two charged to the higher target.
    assert changed["Grid_AC_To_Battery_kWh"].iloc[1] > rows["Grid_AC_To_Battery_kWh"].iloc[1]
    # Year three ran the first set again; only the state year two left differs.
    assert changed["Grid_AC_To_Battery_kWh"].iloc[2] == pytest.approx(rows["Grid_AC_To_Battery_kWh"].iloc[2], rel=0.05)
    assert [item.instruction_hash() for item in second.year_instructions] == [
        low.instruction_hash(),
        high.instruction_hash(),
        low.instruction_hash(),
    ]


def test_a_planner_sees_the_state_each_year_opens_in():
    resolved = _resolved(enable_resistance_fade=True, battery_rte=0.9)
    starts: list[YearStart] = []

    def planner(start):
        starts.append(start)
        return _night_target(0.3 + 0.2 * start.year_idx)

    run = _run(resolved, planner)
    rows = run.yearly_df

    assert [start.year_idx for start in starts] == [0, 1, 2]
    one_way = math.sqrt(0.9)
    first = starts[0].battery_state
    assert first.soh_fraction == 1.0 and first.resistance_growth == 0.0
    assert (first.charge_efficiency, first.discharge_efficiency) == (one_way, one_way)
    assert first.energy_wh == pytest.approx(5000.0 * resolved.cfg["battery_max_soc"], rel=1e-15)
    for start in starts[1:]:
        previous = rows.iloc[start.year_idx - 1]
        state = start.battery_state
        assert state.energy_wh == previous["Battery_Carried_Energy_Wh"]
        assert state.soh_fraction == pytest.approx(previous["Battery_SOH_%"] / 100.0, rel=1e-12)
        assert state.soh_fraction < 1.0
        assert state.resistance_growth == previous["Battery_Resistance_Growth"]
        expected = resistance_to_efficiency(state.resistance_growth, one_way, one_way)
        assert (state.charge_efficiency, state.discharge_efficiency) == expected
        assert start.battery_config.initial_soh == previous["Battery_SOH_%"]
        assert start.year.pv_degradation_factor == rows["PV_Degradation_Factor"].iloc[start.year_idx]
    assert [item.instruction_hash() for item in run.year_instructions] == [
        _night_target(fraction).instruction_hash() for fraction in (0.3, 0.5, 0.7)
    ]


def test_per_year_instructions_need_one_set_per_year():
    resolved = _resolved()
    with pytest.raises(ValueError, match="hold 2 sets; the projection has 3 years"):
        _run(resolved, [_night_target(0.5)] * 2)
    with pytest.raises(TypeError, match="one per project year"):
        _run(resolved, [_night_target(0.5), None, _night_target(0.5)])


def test_a_planner_runs_on_per_step_years_and_must_return_instructions():
    resolved = _resolved()
    with pytest.raises(ValueError, match="per-step projection years"):
        _run(resolved, lambda start: _night_target(0.5), summary=True)
    with pytest.raises(TypeError, match="must return DispatchInstructions"):
        _run(resolved, lambda start: None)
