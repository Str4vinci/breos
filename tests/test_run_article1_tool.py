"""Static coverage for the Article 1 workflow wrapper."""

from pathlib import Path

import pytest

from tools.run_article1 import commands_for_stage


def _script_names(commands: list[list[str]]) -> list[str]:
    return [Path(command[1]).name for command in commands]


def test_deterministic_stage_runs_preflight_and_every_non_monte_carlo_analysis():
    commands = commands_for_stage("deterministic", Path("inputs"), Path("outputs"), n_procs=6)

    assert _script_names(commands) == [
        "preflight_article1_inputs.py",
        "reproduce_article1_montecarlo.py",
        "reproduce_article1.py",
        "reproduce_article1.py",
        "reproduce_article1.py",
        "reproduce_article1.py",
        "reproduce_article1.py",
        "reproduce_article1.py",
        "reproduce_article1_context.py",
        "reproduce_article1_context.py",
    ]
    assert "--validate-only" in commands[1]
    preflight = commands[0]
    assert "--validation-directory" not in preflight
    assert "--copy-validation-to" not in preflight
    assert not any("external-validation" in argument for argument in preflight)
    assert any("battery-cost-sensitivity" in argument for command in commands for argument in command)
    assert any(argument == "6" for command in commands for argument in command)


def test_analysis_stage_does_not_repeat_fixed_candidates():
    commands = commands_for_stage("analysis", Path("inputs"), Path("outputs"), n_procs=6)

    assert _script_names(commands).count("reproduce_article1.py") == 5
    assert not any("base-v1" in argument for command in commands for argument in command)


def test_all_stage_runs_c2_first_and_finishes_with_bundle_verification():
    commands = commands_for_stage("all", Path("inputs"), Path("outputs"), n_procs=4)

    assert _script_names(commands)[-3:] == [
        "reproduce_article1_montecarlo.py",
        "reproduce_article1_montecarlo.py",
        "verify_article1_bundle.py",
    ]
    first_monte_carlo = commands[-3]
    assert first_monte_carlo[first_monte_carlo.index("--case") + 1] == "C2"
    assert "--runs" not in first_monte_carlo


def test_monte_carlo_run_override_is_explicit():
    commands = commands_for_stage("monte-carlo", Path("inputs"), Path("outputs"), n_procs=2, runs=25)

    monte_carlo = commands[2:]
    assert len(monte_carlo) == 2
    for command in monte_carlo:
        assert command[command.index("--runs") + 1] == "25"


def test_unknown_stage_fails_before_running_anything():
    with pytest.raises(ValueError, match="Unknown stage"):
        commands_for_stage("unknown", Path("inputs"), Path("outputs"), n_procs=1)
