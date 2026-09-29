"""The dispatch-replay oracle: instructions through the production projection (plan step 7)."""

import math

import numpy as np
import pandas as pd
import pytest

from breos.app import App
from breos.smart_charging import resolve_instructions
from tools.oracles.replay import PLANNED_FLOWS, prepare_replay, replay_instructions

BASE = {"location": "porto", "n_modules": 8, "annual_consumption_kwh": 4000, "battery_kwh": 5.0, "projection_years": 3}
TOU = {
    "schedule": "pt_mainland_2026_daily_bi",
    "currency": "EUR",
    "import_prices": {"peak": 0.28, "off_peak": 0.11},
    "export_prices": {"all": 0.05},
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


def _record_app_artifacts(monkeypatch):
    import breos.app as app_module

    artifacts = []
    run_app = app_module.run_app_simulation

    def record(*args):
        artifacts.append(run_app(*args))
        return artifacts[-1]

    monkeypatch.setattr(app_module, "run_app_simulation", record)
    return artifacts


def _fixed_target(case):
    return resolve_instructions(case.resolved.smart_charging, case.tariff)


def test_replaying_the_fixed_target_instructions_reproduces_the_app_run(monkeypatch):
    artifacts = _record_app_artifacts(monkeypatch)
    App(CONFIG).simulate()
    app = artifacts[0]

    case = prepare_replay(CONFIG)
    instructions = _fixed_target(case)
    assert instructions == app.instructions
    frame = app.first_year_results_df
    planned = {name: frame[column] for name, column in PLANNED_FLOWS.items()}
    replay = replay_instructions(case, instructions, planned=planned)

    pd.testing.assert_frame_equal(replay.run.first_year_results_df, frame, check_exact=True)
    pd.testing.assert_frame_equal(replay.value.yearly_df, app.yearly_df, check_exact=True)
    pd.testing.assert_frame_equal(replay.value.cost_projection, app.cost_projection, check_exact=True)
    assert replay.run.carry == app.projection.carry
    assert (replay.requests_feasible, replay.corrected_steps) == (True, ())
    assert all(not difference.any() for difference in replay.planned_minus_delivered_wh.values())
    year_one = app.yearly_df.iloc[0]
    assert replay.first_year_step_cost.sum() == pytest.approx(
        year_one["Import_Cost"] - year_one["Export_Revenue"], rel=1e-12
    )
    assert replay.instruction_hash == instructions.instruction_hash()


def test_an_infeasible_request_is_reported_as_corrected():
    case = prepare_replay({**CONFIG, "battery_max_charge_power_w": 1000.0})
    instructions = _fixed_target(case)
    feasible = replay_instructions(case, instructions)
    delivered = feasible.run.first_year_results_df["Grid_AC_To_Battery"].to_numpy()
    step = int(np.flatnonzero(delivered > 0)[0])

    # The plan asks the grid for 3 kW where the battery can take at most 1 kW.
    planned = delivered.copy()
    planned[step] = 3000.0
    replay = replay_instructions(case, instructions, planned={"grid_charge_ac_w": planned})
    assert replay.corrected_steps == (step,)
    assert replay.requests_feasible is False
    hours = 1.0
    assert replay.planned_minus_delivered_wh["grid_charge_ac_w"][step] == pytest.approx(
        (3000.0 - delivered[step]) * hours
    )
    # The request changes nothing in the physics: only the report differs.
    pd.testing.assert_frame_equal(replay.value.yearly_df, feasible.value.yearly_df, check_exact=True)


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
