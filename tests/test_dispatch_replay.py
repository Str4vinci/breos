"""The dispatch-replay oracle: instructions through the production App run (plan step 7)."""

import math

import numpy as np
import pandas as pd
import pytest

from breos.app import App
from breos.dispatch_instructions import DispatchInstructions
from breos.smart_charging import resolve_instructions
from tools.oracles.replay import PLANNED_FLOWS, Tolerance, prepare_replay, replay_instructions

BASE = {"location": "porto", "n_modules": 8, "annual_consumption_kwh": 4000, "battery_kwh": 5.0, "projection_years": 3}
TOU = {
    "schedule": "pt_mainland_2026_daily_bi",
    "currency": "EUR",
    "import_prices": {"peak": 0.28, "off_peak": 0.11},
    "export_prices": {"all": 0.05},
    "fixed_charge_per_day": 0.25,
}
FIXED = {
    "mode": "fixed_target",
    "target_usable_fraction": 0.5,
    "charge_periods": ["off_peak"],
    "discharge_periods": ["peak"],
    "grid_charge_efficiency": 0.95,
    "grid_import_limit_w": 5000,
}
CONFIG = {**BASE, "tariff": TOU, "smart_charging": FIXED}

pytestmark = pytest.mark.usefixtures("_patch_weather")


def _fixed_target(case):
    return resolve_instructions(case.resolved.smart_charging, case.tariff)


def _app_artifacts(config):
    app = App(config)
    app.simulate()
    return app._artifacts


def test_replaying_the_fixed_target_instructions_reproduces_the_app_run():
    app = _app_artifacts(CONFIG)
    case = prepare_replay(CONFIG)
    instructions = _fixed_target(case)
    assert instructions == app.instructions
    frame = app.first_year_results_df
    planned = {name: frame[column] for name, column in PLANNED_FLOWS.items()}
    replay = replay_instructions(case, instructions, planned=planned)
    run = replay.artifacts

    pd.testing.assert_frame_equal(run.first_year_results_df, frame, check_exact=True)
    pd.testing.assert_frame_equal(run.yearly_df, app.yearly_df, check_exact=True)
    pd.testing.assert_frame_equal(run.cost_projection, app.cost_projection, check_exact=True)
    pd.testing.assert_frame_equal(run.projection.period_energy, app.projection.period_energy, check_exact=True)
    assert run.projection.carry == app.projection.carry
    assert (replay.plan_matched, replay.mismatched_steps) == (True, ())
    assert all(not difference.any() for difference in replay.planned_minus_delivered_wh.values())
    year_one = app.yearly_df.iloc[0]
    assert replay.first_year_step_cost.sum() == pytest.approx(
        year_one["Import_Cost"] - year_one["Export_Revenue"], rel=1e-12
    )
    assert replay.instruction_hash == instructions.instruction_hash()
    # The table did not make these instructions, so the run claims no smart-charging provenance.
    assert (app.smart_charging is not None, run.smart_charging) == (True, None)


def test_the_given_instructions_drive_the_dispatch():
    # The config asks for fixed-target charging; the replay dispatches greedily.
    case = prepare_replay(CONFIG)
    replay = replay_instructions(case, DispatchInstructions.noop(len(case.index)))
    frame = replay.artifacts.first_year_results_df
    smart = _app_artifacts(CONFIG).first_year_results_df
    greedy = _app_artifacts({**BASE, "tariff": TOU}).first_year_results_df

    assert smart["Grid_AC_To_Battery"].sum() > 0.0
    assert frame["Grid_AC_To_Battery"].sum() == 0.0
    pd.testing.assert_frame_equal(frame, greedy, check_exact=True)


@pytest.mark.filterwarnings("ignore:'projection_years'")
def test_a_period_window_replays_once_and_bills_its_civil_days():
    # March 2025 in Lisbon: 31 civil days, one of them 23 hours long.
    config = {**CONFIG, "start_date": "2025-01-01", "period": {"start": "2025-03-01", "end": "2025-04-01"}}
    app = _app_artifacts(config)
    case = prepare_replay(config)
    replay = replay_instructions(case, _fixed_target(case))
    yearly = replay.artifacts.yearly_df

    pd.testing.assert_frame_equal(yearly, app.yearly_df, check_exact=True)
    assert len(yearly) == 1 and replay.artifacts.cost_projection is None
    assert len(case.index) == 31 * 24 - 1
    assert yearly["Fixed_Charge"].iloc[0] == pytest.approx(31 * TOU["fixed_charge_per_day"], rel=1e-12)


def test_a_plan_that_asks_past_a_power_limit_does_not_match():
    case = prepare_replay({**CONFIG, "battery_max_charge_power_w": 1000.0})
    instructions = _fixed_target(case)
    unplanned = replay_instructions(case, instructions)
    frame = unplanned.artifacts.first_year_results_df
    delivered = frame["Grid_AC_To_Battery"].to_numpy()
    step = int(np.flatnonzero(delivered > 0)[0])
    # The battery takes at most 1 kW of DC charge input, so the grid supplies
    # at most 1 kW over the 0.95 AC-to-DC conversion.
    assert frame["Battery_Charge_Input"].iloc[step] <= 1000.0 * (1 + 1e-12)
    assert delivered[step] <= 1000.0 / 0.95 * (1 + 1e-12)

    # The plan asks the grid for 3 kW there.
    planned = delivered.copy()
    planned[step] = 3000.0
    replay = replay_instructions(case, instructions, planned={"grid_charge_ac_w": planned})
    assert (replay.mismatched_steps, replay.plan_matched) == ((step,), False)
    assert replay.planned_minus_delivered_wh["grid_charge_ac_w"][step] == pytest.approx(3000.0 - delivered[step])
    # A plan never drives the dispatch: only the report differs.
    pd.testing.assert_frame_equal(replay.artifacts.yearly_df, unplanned.artifacts.yearly_df, check_exact=True)


def test_a_planned_power_is_compared_as_energy_over_the_step():
    # At 15 minutes, 400 W too much is 100 Wh and 150 W is 37.5 Wh.
    case = prepare_replay({**CONFIG, "resolution": "15min", "projection_years": 1})
    instructions = _fixed_target(case)
    frame = replay_instructions(case, instructions).artifacts.first_year_results_df
    delivered = frame["Grid_AC_To_Battery"].to_numpy()
    first, second = np.flatnonzero(delivered > 0)[:2]
    below = int(np.flatnonzero(delivered > 400.0)[-1])
    planned = delivered.copy()
    planned[first] += 400.0
    planned[second] += 150.0
    # A plan below what was delivered misses too.
    planned[below] -= 400.0
    replay = replay_instructions(
        case,
        instructions,
        planned={"grid_charge_ac_w": planned, "battery_energy_wh": frame["Battery_Energy_End"]},
        tolerance={"grid_charge_ac_w": Tolerance(atol_wh=50.0)},
    )
    difference = replay.planned_minus_delivered_wh["grid_charge_ac_w"]
    assert (difference[first], difference[second], difference[below]) == (
        pytest.approx(100.0),
        pytest.approx(37.5),
        pytest.approx(-100.0),
    )
    assert replay.mismatched_steps == (first, below)
    assert replay.tolerances == {"grid_charge_ac_w": Tolerance(atol_wh=50.0), "battery_energy_wh": Tolerance()}


def test_a_relative_tolerance_scales_with_the_delivered_energy():
    case = prepare_replay({**CONFIG, "projection_years": 1})
    instructions = _fixed_target(case)
    energy = replay_instructions(case, instructions).artifacts.first_year_results_df["Battery_Energy_End"].to_numpy()
    step = int(np.argmax(energy))
    planned = energy.copy()
    planned[step] *= 1.01
    loose = replay_instructions(
        case, instructions, planned={"battery_energy_wh": planned}, tolerance=Tolerance(rtol=0.02)
    )
    tight = replay_instructions(
        case, instructions, planned={"battery_energy_wh": planned}, tolerance=Tolerance(rtol=0.005)
    )
    assert (loose.plan_matched, tight.mismatched_steps) == (True, (step,))


def test_a_replay_rejects_what_it_cannot_price_or_check():
    with pytest.raises(ValueError, match=r"no \[tariff\] table"):
        prepare_replay(BASE)
    with pytest.raises(ValueError, match="needs a battery"):
        prepare_replay({**BASE, "tariff": TOU, "battery_kwh": 0.0})
    case = prepare_replay(CONFIG)
    instructions = _fixed_target(case)
    n = len(instructions)
    with pytest.raises(ValueError, match="Unknown planned flow"):
        replay_instructions(case, instructions, planned={"export_w": np.zeros(n)})
    with pytest.raises(ValueError, match="finite, non-negative"):
        replay_instructions(case, instructions, planned={"discharge_ac_w": np.full(n, math.nan)})
    with pytest.raises(ValueError, match="Tolerance for unknown flow"):
        replay_instructions(case, instructions, tolerance={"export_w": Tolerance()})
    with pytest.raises(ValueError, match="'rtol' must be finite"):
        Tolerance(rtol=-1.0)


def test_replay_runs_the_requested_backend_without_changing_the_case():
    pytest.importorskip("numba")
    case = prepare_replay(CONFIG)
    instructions = _fixed_target(case)

    python = replay_instructions(case, instructions)
    numba = replay_instructions(case, instructions, execution_backend="numba")

    assert python.artifacts.execution["execution_backend"] == "python"
    assert numba.artifacts.execution["execution_backend"] == "numba"
    # The override is per call: the prepared case keeps its own backend.
    assert case.resolved.cfg["execution_backend"] == "python"
    pd.testing.assert_frame_equal(
        python.artifacts.first_year_results_df, numba.artifacts.first_year_results_df, check_exact=True
    )
