"""The backend choice must reach every simulating path, and no other.

These are propagation and configuration tests, not numerical ones. Parity lives
in ``test_numba_dispatch_parity.py``; what is checked here is that the choice
travels explicitly, defaults to the reference implementation everywhere, is
validated in one place, and is recorded wherever results are written.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest

from breos import App
from breos.app_config import resolve_app_config
from breos.execution import (
    DEFAULT_EXECUTION_BACKEND,
    EXECUTION_BACKENDS,
    aggregate_jit_cache_states,
    backend_provenance,
    validate_execution_backend,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_CONFIG = {
    "location": "porto",
    "n_modules": 4,
    "annual_consumption_kwh": 5000,
    "battery_kwh": 5,
    "projection_years": 2,
}


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), PROJECT_ROOT / "tools" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_reference_implementation_is_the_default_everywhere():
    """Nothing selects the compiled path without being asked."""
    assert DEFAULT_EXECUTION_BACKEND == "python"
    assert resolve_app_config(BASE_CONFIG).cfg["execution_backend"] == "python"

    from breos import optimization

    for name in ("optimize_battery_size", "evaluate_projected_design", "optimize_system_multi_objective"):
        signature = inspect.signature(getattr(optimization, name))
        parameter = signature.parameters["execution_backend"]
        assert parameter.default == DEFAULT_EXECUTION_BACKEND, f"{name} does not default to the reference path"


def test_optimization_takes_the_backend_as_an_argument_not_from_config():
    """Candidate scoring is the hottest loop, so its backend must be explicit.

    A backend read out of a nested config dict would be invisible at the call
    site and impossible to attribute afterwards. Every optimization entry point
    therefore names it as a parameter.
    """
    from breos import optimization

    source = inspect.getsource(optimization)
    assert 'config.get("execution_backend"' not in source
    assert 'config["execution_backend"]' not in source


@pytest.mark.parametrize("backend", EXECUTION_BACKENDS)
def test_app_records_the_backend_and_its_toolchain(backend):
    if backend == "numba":
        pytest.importorskip("numba", reason="the compiled backend needs the breos[fast] extra")
    app = App({**BASE_CONFIG, "execution_backend": backend})
    app.simulate()

    execution = app.result()["provenance"]["execution"]
    assert execution["execution_backend"] == backend
    for key in ("python", "numpy", "pandas"):
        assert key in execution
    if backend == "numba":
        # A bit-identity claim is unverifiable after the fact without these.
        assert "numba" in execution and "llvmlite" in execution
        assert execution["jit_cache"] in {"warm", "cold", "unknown"}
    else:
        assert "jit_cache" not in execution


def test_app_and_monte_carlo_record_the_same_execution_keys():
    """Two provenance blocks built by two code paths would drift."""
    app = App({**BASE_CONFIG, "execution_backend": "python"})
    app.simulate()
    assert app.result()["provenance"]["execution"] == backend_provenance("python")


@pytest.mark.parametrize("bad", ["", "NUMBA", "cython", None, 0])
def test_unknown_backend_names_are_rejected_once_and_the_same_way(bad):
    with pytest.raises(ValueError, match="execution_backend must be one of"):
        validate_execution_backend(bad)
    with pytest.raises(ValueError, match="execution_backend must be one of"):
        resolve_app_config({**BASE_CONFIG, "execution_backend": bad})


def test_jit_cache_aggregation_never_raises():
    """Provenance bookkeeping must not be able to fail a completed run."""
    for states in ([], ["unknown"], ["warm", "unknown"], ["warm"], ["warm", "cold"]):
        assert aggregate_jit_cache_states(states) in {"warm", "cold", "unknown"}


def test_article_runner_forwards_the_backend_only_to_stages_that_simulate():
    """A stage that never enters the dispatch loop must not claim a backend.

    ``preflight_article1_inputs.py`` inventories inputs, the ``--validate-only``
    Monte Carlo call checks configurations without simulating,
    ``reproduce_article1_context.py`` builds orientation and weather-comparison
    tables, and ``verify_article1_bundle.py`` checks finished outputs. Passing
    the flag to any of them would record a claim about code that never ran.
    """
    run_article1 = _load_tool("run_article1.py")
    commands = run_article1.commands_for_stage(
        "all",
        PROJECT_ROOT / "dev/article1-inputs",
        PROJECT_ROOT / "results/article1",
        n_procs=2,
        runs=None,
        execution_backend="numba",
    )

    non_simulating = (
        "preflight_article1_inputs.py",
        "reproduce_article1_context.py",
        "verify_article1_bundle.py",
    )
    simulating = 0
    for command in commands:
        joined = " ".join(command)
        carries_backend = "--execution-backend" in command
        if any(tool in joined for tool in non_simulating) or "--validate-only" in command:
            assert not carries_backend, f"backend forwarded to a non-simulating stage: {joined}"
            continue
        assert carries_backend, f"simulating stage did not receive the backend: {joined}"
        assert command[command.index("--execution-backend") + 1] == "numba"
        simulating += 1

    assert simulating == 8, f"expected eight simulating commands, found {simulating}"


def test_article_tools_default_to_the_reference_backend():
    for name in ("run_article1.py", "reproduce_article1.py", "reproduce_article1_montecarlo.py"):
        module = _load_tool(name)
        parser_source = inspect.getsource(module.main)
        assert "--execution-backend" in parser_source, f"{name} does not expose the backend"
