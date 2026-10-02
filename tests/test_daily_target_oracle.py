"""The daily-target oracle: one grid-charge target per day, planned and replayed (plan step 7)."""

import json

import numpy as np
import pandas as pd
import pytest

from breos._daily_targets import _DayEvaluator, daily_target_instructions, target_grid
from tools.oracles.daily_target_dp import (
    DP_DAYS_SCHEMA,
    DP_ORACLE_SCHEMA,
    DP_YEAR_DAYS_SCHEMA,
    daily_target_problem,
    fixed_health_state,
    main,
    report,
    run_daily_target_oracle,
)
from tools.oracles.replay import DEFAULT_TOLERANCE, prepare_replay, replay_instructions

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


def test_the_planned_flows_use_the_planners_own_day_state():
    # A field the planner's state gains must reach the planned flows too.
    case = prepare_replay(_config(1, battery_max_charge_power_w=1500.0))
    problem = daily_target_problem(case)
    assert fixed_health_state(problem) == _DayEvaluator(problem, target_grid(1), "python").state


# Three whole years, so each year opens at the health the last one left.
YEARLY = {**BASE, "tariff": TOU, "smart_charging": FIXED, "start_date": "2025-01-01", "projection_years": 3}
YEARLY_PLANNER = {"target_levels": 3, "soc_states": 3}


@pytest.fixture(scope="module")
def yearly():
    from tests.conftest import _build_synthetic_weather

    weather = _build_synthetic_weather()
    weather.attrs["breos_weather_metadata"] = {"source": "PVGIS_TMY"}
    tmy = {"inputs": {"location": {"latitude": 41.15, "longitude": -8.63, "elevation": 0}}}
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("breos.app.fetch_tmy_weather_data", lambda *args, **kwargs: (weather.copy(), tmy))
        monkeypatch.setattr("breos.app.load_weather", lambda **kw: None)
        case = prepare_replay(YEARLY)
        result = run_daily_target_oracle(case, planning="yearly", **YEARLY_PLANNER)
        first_year = run_daily_target_oracle(case, **YEARLY_PLANNER)
        replayed = replay_instructions(case, [plan.instructions for plan in result.year_plans])
    return case, result, first_year, replayed


def test_yearly_replanning_plans_each_year_at_the_state_the_replay_reached(yearly):
    case, result, _first_year, _replayed = yearly
    plans, rows = result.year_plans, result.replay.artifacts.yearly_df

    assert [plan.year for plan in plans] == [1, 2, 3] and result.planning == "yearly"
    assert plans[0].problem.health()[0] == 1.0
    for plan in plans[1:]:
        previous = rows.iloc[plan.year - 2]
        soh, eff_charge, eff_discharge = plan.problem.health()
        assert soh == pytest.approx(previous["Battery_SOH_%"] / 100.0, rel=1e-12) and soh < 1.0
        assert plan.problem.battery_config.initial_soh == previous["Battery_SOH_%"]
        assert plan.start_energy_wh == previous["Battery_Carried_Energy_Wh"]
        assert plan.plan.start_energy_wh == plan.start_energy_wh
        assert plan.terminal_energy_wh <= plan.start_energy_wh
        assert plan.pv_degradation_factor == rows["PV_Degradation_Factor"].iloc[plan.year - 1] < 1.0
        assert (eff_charge, eff_discharge) == (
            plan.problem.battery_config.charge_efficiency,
            plan.problem.battery_config.discharge_efficiency,
        )
    # Each year is planned on its own degraded PV.
    assert plans[1].problem.pv_dc_w.sum() < plans[0].problem.pv_dc_w.sum()
    assert len({plan.inputs_sha256() for plan in plans}) == 3
    # The replay dispatched every year on that year's plan.
    assert result.replay.year_instruction_hashes == tuple(plan.instructions.instruction_hash() for plan in plans)


def test_yearly_replanning_replays_as_the_same_schedule_given_per_year(yearly):
    _case, result, _first_year, replayed = yearly
    # Handing the chosen sets back as one set per year reproduces the planner's run bit for bit.
    pd.testing.assert_frame_equal(result.replay.artifacts.yearly_df, replayed.artifacts.yearly_df, check_exact=True)
    assert replayed.instruction_hash == result.replay.instruction_hash
    assert replayed.year_instruction_hashes == result.replay.year_instruction_hashes


def test_yearly_replanning_plans_year_one_as_the_first_year_mode(yearly):
    _case, result, first_year, _replayed = yearly
    np.testing.assert_array_equal(result.plan.targets, first_year.plan.targets)
    assert result.instructions == first_year.instructions
    assert result.plan.objective == first_year.plan.objective
    # A first-year plan replays its year-one set every year.
    assert set(first_year.replay.year_instruction_hashes) == {first_year.instructions.instruction_hash()}
    pd.testing.assert_frame_equal(
        result.replay.artifacts.first_year_results_df,
        first_year.replay.artifacts.first_year_results_df,
        check_exact=True,
    )
    assert result.replay.plan_matched == first_year.replay.plan_matched


def test_the_yearly_report_records_the_planning_mode_and_each_years_inputs(yearly):
    case, result, first_year, _replayed = yearly
    summary = report(result, case)
    json.dumps(summary, allow_nan=False)

    assert summary["planner"]["planning"] == "yearly"
    years = summary["years"]
    assert [year["year"] for year in years] == [1, 2, 3]
    rows = result.replay.artifacts.yearly_df
    for year, plan in zip(years, result.year_plans, strict=True):
        assert year["opening_state"]["soh_fraction"] == plan.problem.health()[0]
        assert year["inputs"]["sha256"] == plan.inputs_sha256()
        assert year["replayed_cost"] == pytest.approx(
            rows["Import_Cost"].iloc[plan.year - 1] - rows["Export_Revenue"].iloc[plan.year - 1], rel=1e-12
        )
    lifetime = summary["lifetime"]
    assert len(lifetime["replay"]["year_costs"]) == 3 and lifetime["replay"]["npv_savings"] is not None
    assert "not an NPV bound" in lifetime["basis"]
    assert report(first_year, case)["years"] == [] and report(first_year, case)["planner"]["planning"] == "first_year"


def test_the_command_line_writes_one_row_per_year_and_day_under_yearly_planning(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({**YEARLY, "projection_years": 2}), encoding="utf-8")
    output, days = tmp_path / "dp.json", tmp_path / "days.csv"
    argv = ["--config", str(config), "--output", str(output), "--csv", str(days), "--planning", "yearly"]
    assert main([*argv, "--target-levels", "2", "--soc-states", "2"]) == 0

    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["planner"]["planning"] == "yearly" and summary["planner"]["soc_states"] == 2
    assert days.read_text(encoding="utf-8").splitlines()[0] == f"# schema: {DP_YEAR_DAYS_SCHEMA}"
    frame = pd.read_csv(days, comment="#")
    assert list(frame["year"].unique()) == [1, 2] and len(frame) == 2 * 365


def test_the_oracle_refuses_an_unknown_planning_mode():
    case = prepare_replay(_config(1))
    with pytest.raises(ValueError, match="'planning' must be one of first_year, yearly"):
        run_daily_target_oracle(case, planning="rolling")
