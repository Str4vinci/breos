"""The daily-target oracle: one grid-charge target per day, planned and replayed (plan step 7)."""

import json

import numpy as np
import pandas as pd
import pytest

from breos._daily_targets import daily_target_instructions
from tools.oracles.daily_target_dp import (
    DP_DAYS_SCHEMA,
    DP_ORACLE_SCHEMA,
    daily_target_problem,
    main,
    report,
    run_daily_target_oracle,
)
from tools.oracles.replay import DEFAULT_TOLERANCE, prepare_replay

BASE = {"location": "porto", "n_modules": 8, "annual_consumption_kwh": 4000, "battery_kwh": 5.0, "projection_years": 1}
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


def _config(days, **overrides):
    """A [period] window of ``days`` civil days from 1 March 2025 in Lisbon."""
    end = (pd.Timestamp("2025-03-01") + pd.Timedelta(days=days)).date().isoformat()
    return {
        **BASE,
        "tariff": TOU,
        "smart_charging": FIXED,
        "start_date": "2025-01-01",
        "period": {"start": "2025-03-01", "end": end},
        **overrides,
    }


pytestmark = [
    pytest.mark.usefixtures("_patch_weather"),
    pytest.mark.filterwarnings("ignore:'projection_years'"),
]

# Few levels and states keep the dynamic program to a fraction of a second.
PLANNER = {"target_levels": 6, "soc_states": 9}


def test_the_oracle_places_its_daily_targets_and_prices_the_replay():
    case = prepare_replay(_config(5))
    result = run_daily_target_oracle(case, **PLANNER)
    problem = result.problem

    assert problem.day_starts == case.tariff.day_starts and problem.n_days == 5
    assert result.instructions == daily_target_instructions(
        case.configured_instructions(), problem.day_starts, result.plan.targets
    )
    assert set(result.plan.targets) <= set(np.linspace(0.0, 1.0, 6))
    # The planned flows are the plan's own model: its stage cost, step by step.
    assert result.planned_step_cost.sum() == pytest.approx(result.plan.stage_cost, rel=1e-12)
    # Health moves a little during the week, so production nearly delivers the plan.
    assert result.replay.first_year_step_cost.sum() == pytest.approx(result.plan.stage_cost, rel=1e-3)
    # Choosing the target day by day beats the configured 50 % every day here.
    assert result.replay.first_year_step_cost.sum() < result.fixed_target.first_year_step_cost.sum()
    assert result.fixed_target.instruction_hash == case.configured_instructions().instruction_hash()

    summary = report(result, case)
    assert summary["schema"] == DP_ORACLE_SCHEMA
    assert summary["replay_minus_fixed_target"] == pytest.approx(
        summary["replay"]["first_year_cost"] - summary["fixed_target"]["first_year_cost"]
    )
    assert sum(summary["planner"]["days_by_target"].values()) == 5
    json.dumps(summary, allow_nan=False)


def test_the_plan_is_what_production_delivers_while_health_holds():
    # One civil day is one degradation window: production changes health only
    # when it closes, so the planner's fixed health is exact and every
    # planned flow is delivered to within the replay's default 1e-7 Wh.
    case = prepare_replay(_config(1))
    result = run_daily_target_oracle(case, **PLANNER, tolerance=DEFAULT_TOLERANCE)
    assert result.replay.plan_matched, result.replay.mismatched_steps
    assert result.replay.first_year_step_cost.sum() == pytest.approx(result.plan.stage_cost, rel=1e-12)


def test_the_oracle_needs_a_fixed_target_table():
    case = prepare_replay({key: value for key, value in _config(2).items() if key != "smart_charging"})
    with pytest.raises(ValueError, match="mode = 'fixed_target'"):
        daily_target_problem(case)


def test_the_command_line_writes_a_tagged_summary_and_one_row_per_day(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(_config(3)), encoding="utf-8")
    output, days = tmp_path / "dp.json", tmp_path / "days.csv"
    assert main(["--config", str(config), "--output", str(output), "--csv", str(days), "--target-levels", "3"]) == 0

    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["schema"] == DP_ORACLE_SCHEMA and summary["n_days"] == 3
    assert summary["planner"]["target_levels"] == [0.0, 0.5, 1.0]
    assert days.read_text(encoding="utf-8").splitlines()[0] == f"# schema: {DP_DAYS_SCHEMA}"
    frame = pd.read_csv(days, comment="#")
    assert len(frame) == 3
    assert frame["replayed_cost"].sum() == pytest.approx(summary["replay"]["first_year_cost"], rel=1e-9)
