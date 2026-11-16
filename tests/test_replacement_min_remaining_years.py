"""The opt-in minimum service time of a replacement battery (#400).

A pack that reaches end of life a few months before the project ends is
replaced at full price and serves only those months. ``BatteryConfig.
replacement_min_remaining_years``, the App and Monte Carlo key
``battery_replacement_min_remaining_years`` and the optimizer's ``[battery]
replacement_min_remaining_years`` skip a swap that would leave the new pack
less than that many project years to serve. Time is counted as
``Replacement_Time_Years`` books a swap, to the end of the project, so each
projection year knows how many years follow it. The default 0 changes
nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from breos.battery import BatteryConfig, align_simulation_inputs, simulate_energy_balance
from breos.economics import replacement_booking_time
from breos.optimization import evaluate_projected_design
from breos.projection import CarryState, ProjectionYear, project_years
from tests.test_terminal_replacement import (
    EVERY_PERIOD_EOL,
    _app,
    _assert_same_run,
    _design_inputs,
    _inputs,
    _replaced_steps,
    _require_numba,
    _run,
)

# Three hourly days stand for one project year, as in the terminal-guard tests.
YEAR_STEPS = 72


# -- one span ---------------------------------------------------------------------


def test_the_minimum_is_compared_in_steps_and_keeps_an_exact_match():
    # Four hourly days, one span-year: the closes leave 72, 48, 24 and 0 steps.
    default = _run(96, eol_percentage=EVERY_PERIOD_EOL)
    quarter = _run(96, eol_percentage=EVERY_PERIOD_EOL, replacement_min_remaining_years=0.25)

    assert _replaced_steps(default) == [23, 47, 71, 95]
    # 24 steps left is exactly a quarter of the span: that swap still happens.
    assert _replaced_steps(quarter) == [23, 47, 71]
    pd.testing.assert_frame_equal(default[0].iloc[:-1], quarter[0].iloc[:-1], check_exact=True)


def test_the_years_after_the_span_count_toward_what_is_left():
    one_year = {"eol_percentage": EVERY_PERIOD_EOL, "replacement_min_remaining_years": 1.0}
    assert _replaced_steps(_run(YEAR_STEPS, **one_year)) == []
    _assert_same_run(
        _run(YEAR_STEPS, **one_year, replacement_years_after_span=1),
        _run(YEAR_STEPS, eol_percentage=EVERY_PERIOD_EOL),
    )
    # 1.5 years with one year after the span: half the span must be left.
    half = _run(YEAR_STEPS, eol_percentage=EVERY_PERIOD_EOL, replacement_min_remaining_years=1.5)
    assert _replaced_steps(half) == []
    after = _run(
        YEAR_STEPS,
        eol_percentage=EVERY_PERIOD_EOL,
        replacement_min_remaining_years=1.5,
        replacement_years_after_span=1,
    )
    assert _replaced_steps(after) == [23]


def test_with_a_zero_minimum_the_years_after_the_span_change_nothing():
    _assert_same_run(
        _run(YEAR_STEPS, eol_percentage=EVERY_PERIOD_EOL),
        _run(YEAR_STEPS, eol_percentage=EVERY_PERIOD_EOL, replacement_years_after_span=5),
    )


@pytest.mark.parametrize("freq", ["h", "15min"])
def test_python_and_numba_skip_the_same_swaps(freq):
    _require_numba()
    n_steps = 4 * (24 if freq == "h" else 96)
    config = {"eol_percentage": EVERY_PERIOD_EOL, "replacement_min_remaining_years": 0.6}
    python = _run(n_steps, freq, **config)
    numba = _run(n_steps, freq, backend="numba", **config)

    _assert_same_run(python, numba)
    assert len(_replaced_steps(numba)) == 1


# -- the projection ------------------------------------------------------------------


def _project(
    minimum: float,
    *,
    years: int = 3,
    eol: float = EVERY_PERIOD_EOL,
    allow: bool = True,
    aligned: bool = False,
    backend: str = "python",
):
    pv, load, temperature = _inputs(YEAR_STEPS)
    year_aligned = align_simulation_inputs(pv, load, temperature, freq="h") if aligned else None

    def year_inputs(_year_idx):
        if year_aligned is not None:
            return ProjectionYear(1.0, aligned=year_aligned)
        return ProjectionYear(1.0, pv_dc=pv, houseload=load, temperature_series=temperature)

    def battery(soh_pct):
        return BatteryConfig(
            nominal_energy_wh=5000.0,
            initial_soh=soh_pct,
            eol_percentage=eol,
            allow_terminal_replacement=allow,
            replacement_min_remaining_years=minimum,
        )

    return project_years(
        years, year_inputs, battery_config=battery, freq="h", has_battery=True, execution_backend=backend
    )


def _late_last_year_eol() -> tuple[float, list[float]]:
    """A threshold first crossed at the close of the last year's second day, and the unreplaced SOH by close."""
    soh: list[float] = []
    pv, load, temperature = _inputs(YEAR_STEPS)
    carry = CarryState()
    for year in range(3):
        run = simulate_energy_balance(
            pv_dc=pv,
            houseload=load,
            temperature_series=temperature,
            battery_config=BatteryConfig(nominal_energy_wh=5000.0, initial_soh=carry.soh_pct, enable_replacement=False),
            freq="h",
            return_degradation_state=True,
            finalize_degradation=year == 2,
            **carry.simulation_kwargs(),
        )
        carry = carry.after_frames(run[0], run[4], run[5], has_battery=True)
        soh.extend(run[4]["SOH"].tolist())
    return (soh[-3] + soh[-2]) / 200.0, soh


def test_a_late_last_year_end_of_life_is_not_replaced_with_a_one_year_minimum():
    eol, soh = _late_last_year_eol()
    default = _project(0.0, eol=eol)
    terminal_guard = _project(0.0, eol=eol, allow=False)
    guarded = _project(1.0, eol=eol)

    # The swap closes the last year's second day: not the final period, so the
    # terminal guard keeps it.
    assert default.yearly_df["Replacement_Steps"].tolist() == ["", "", "47"]
    pd.testing.assert_frame_equal(default.yearly_df, terminal_guard.yearly_df, check_exact=True)
    assert guarded.yearly_df["Replacements"].tolist() == [0, 0, 0]
    assert guarded.total_replacements == 0
    pd.testing.assert_frame_equal(default.yearly_df.iloc[:2], guarded.yearly_df.iloc[:2], check_exact=True)
    # The old pack keeps ageing below its end-of-life threshold and is reported so.
    assert guarded.carry.soh_pct == soh[-1] < 100.0 * eol
    assert np.isnan(guarded.yearly_df["Replacement_Year_Fraction"].iloc[-1])


@pytest.mark.parametrize("minimum", [0.25, 0.5, 1.0, 1.5, 2.0, 2.9, 3.0, 4.0])
def test_a_swap_is_kept_exactly_when_its_booked_time_leaves_the_minimum(minimum):
    # Every close crosses end of life, so the default swaps at all nine closes.
    default = _project(0.0)
    guarded = _project(minimum)
    assert default.yearly_df["Replacements"].tolist() == [3, 3, 3]

    kept: list[list[str]] = []
    for year, steps in enumerate(default.yearly_df["Replacement_Steps"]):
        booked = [
            float(replacement_booking_time([year + 1], [(int(step) + 1) / YEAR_STEPS], [True])[0])
            for step in steps.split(";")
        ]
        kept.append([step for step, time in zip(steps.split(";"), booked, strict=True) if 3 - time >= minimum])
    assert guarded.yearly_df["Replacement_Steps"].tolist() == [";".join(steps) for steps in kept]


def test_the_close_of_a_year_with_exactly_the_minimum_left_still_swaps():
    run = _project(1.0)
    assert run.yearly_df["Replacement_Steps"].tolist() == ["23;47;71", "23;47;71", ""]
    assert run.carry.soh_pct < 100.0


def test_frames_summaries_and_numba_agree():
    frames = _project(1.5)
    summary = _project(1.5, aligned=True)
    pd.testing.assert_frame_equal(frames.yearly_df, summary.yearly_df, check_exact=True)
    assert frames.carry.soh_pct == summary.carry.soh_pct
    assert frames.yearly_df["Replacements"].tolist() == [3, 1, 0]
    _require_numba()
    numba = _project(1.5, backend="numba")
    pd.testing.assert_frame_equal(frames.yearly_df, numba.yearly_df, check_exact=True)
    assert frames.carry.soh_pct == numba.carry.soh_pct


# -- App, Monte Carlo and the optimizer ------------------------------------------------


def test_app_skips_the_last_years_swaps_and_records_the_setting():
    default = _app()
    guarded = _app(battery_replacement_min_remaining_years=1, terminal_value={"basis": "battery_health_fraction"})
    default_result, guarded_result = default.result(), guarded.result()

    assert default_result["provenance"]["resolved_config"]["battery_replacement_min_remaining_years"] == 0.0
    assert guarded_result["provenance"]["resolved_config"]["battery_replacement_min_remaining_years"] == 1.0
    policy = guarded_result["provenance"]["terminal_value"]["replacement_policy"]
    assert policy == {
        "enable_replacement": True,
        "allow_terminal_replacement": True,
        "replacement_min_remaining_years": 1.0,
        "skipped_replacement_action": "keep",
    }
    # The close of year 1 leaves exactly one year, so it still swaps.
    assert default._artifacts.yearly_df["Replacements"].tolist() == [365, 365]
    assert guarded._artifacts.yearly_df["Replacements"].tolist() == [365, 0]
    assert guarded_result["battery_soh_end_pct"] < 100.0 * EVERY_PERIOD_EOL
    pd.testing.assert_frame_equal(
        default._artifacts.cost_projection.iloc[:1], guarded._artifacts.cost_projection.iloc[:1], check_exact=True
    )


def test_the_projected_optimizer_skips_the_swaps_and_records_the_setting(monkeypatch):
    weather, load, pv = _design_inputs()
    monkeypatch.setattr("breos.optimization.calculate_pv_production_dc", lambda **_kwargs: pv)
    result = evaluate_projected_design(
        weather,
        load,
        {
            "location": {"latitude": 41.15, "longitude": -8.63},
            "simulation": {"resolution": "h", "years_projection": 2},
            "financials": {"project_lifespan": 2},
            "battery": {"eol_percentage": EVERY_PERIOD_EOL, "replacement_min_remaining_years": 1},
        },
        n_modules=4,
        battery_kwh=5.0,
        tilt=35.0,
        azimuth=180.0,
    )
    treatment = result.provenance["battery_replacement_treatment"]
    assert treatment["replacement_min_remaining_years"] == 1.0
    assert treatment["allow_terminal_replacement"] is True
    assert result.yearly["Replacements"].tolist() == [3, 0]
