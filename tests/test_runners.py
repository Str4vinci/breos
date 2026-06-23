"""Tests for workflow runner module boundaries."""

from breos.runners import SimulationArtifacts, run_app_simulation
from breos.runners.app import SimulationArtifacts as AppSimulationArtifacts
from breos.runners.app import run_app_simulation as run_app_runner


def test_app_runner_exports_are_available_from_runner_package():
    assert run_app_simulation is run_app_runner
    assert SimulationArtifacts is AppSimulationArtifacts
