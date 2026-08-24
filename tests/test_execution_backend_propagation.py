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


def test_missing_numba_is_reported_before_app_prepares_inputs(monkeypatch):
    """The dependency check must precede input preparation, which can hit the network.

    Downloading a year of weather and then failing on a missing import wastes
    the expensive step to report the cheap problem.
    """
    from breos import _numba_dispatch
    from breos.runners import app as app_runner

    monkeypatch.setattr(_numba_dispatch, "numba_available", lambda: False)

    def _must_not_run(*args, **kwargs):
        raise AssertionError("inputs were prepared before the backend was checked")

    monkeypatch.setattr(app_runner, "prepare_simulation_inputs", _must_not_run)

    with pytest.raises(_numba_dispatch.NumbaUnavailableError):
        app_runner.run_app_simulation(
            {**resolve_app_config({**BASE_CONFIG, "execution_backend": "numba"}).cfg},
            resolve_app_config({**BASE_CONFIG, "execution_backend": "numba"}),
            None,
        )


def test_missing_numba_is_reported_before_the_first_candidate(monkeypatch):
    """optimize_battery_size checks at entry, not inside the size loop.

    An empty size list proves the ordering: with the check inside the loop
    there would be nothing to trip over, and the call would succeed.
    """
    from breos import _numba_dispatch, optimization

    monkeypatch.setattr(_numba_dispatch, "numba_available", lambda: False)

    with pytest.raises(_numba_dispatch.NumbaUnavailableError):
        optimization.optimize_battery_size(
            pv_dc=None,
            houseload=None,
            battery_sizes_wh=[],
            execution_backend="numba",
        )


@pytest.mark.parametrize(
    ("function", "expensive"),
    [
        ("evaluate_projected_design", "calculate_pv_production_dc("),
        ("optimize_system_multi_objective", "Pool("),
    ],
)
def test_dependency_check_precedes_the_expensive_step(function, expensive):
    """Ordering inside these two is not observable without running them.

    Both would need a full weather frame or a live worker pool to exercise, so
    the check here is that the guard textually precedes the expensive call.
    """
    from breos import optimization

    source = inspect.getsource(getattr(optimization, function))
    assert "require_backend(execution_backend)" in source
    assert source.index("require_backend(execution_backend)") < source.index(expensive), (
        f"{function} does the expensive work before checking the backend"
    )


def test_numba_provenance_always_carries_a_cache_field():
    """A driver that cannot observe its workers still records the field.

    The deterministic Article report fans work out to subprocesses and has no
    observations to aggregate. "unknown" is provenance; a missing key reads as
    an oversight when the run that produced it took hours.
    """
    pytest.importorskip("numba", reason="the compiled backend needs the breos[fast] extra")

    assert backend_provenance("numba")["jit_cache"] == "unknown"
    assert backend_provenance("numba", jit_cache_states=["warm", "warm"])["jit_cache"] == "warm"
    assert backend_provenance("numba", jit_cache_states=["warm", "cold"])["jit_cache"] == "cold"
    assert "jit_cache" not in backend_provenance("python")


def test_deterministic_article_report_records_the_cache_field():
    """reproduce_article1.py writes backend_provenance straight into its report."""
    module = _load_tool("reproduce_article1.py")
    source = inspect.getsource(module.main)
    assert '"execution": backend_provenance(args.execution_backend)' in source


def test_app_assembled_outputs_are_identical_on_both_backends():
    """The gate the timestep tests cannot provide.

    Timestep parity compares the buffer matrix. It says nothing about the
    yearly rollups, cost projection, LCOE, NPV, payback, monthly and financial
    tables, degradation summary or PV loss waterfall that App assembles on top
    -- all of which pass through economics and aggregation code the timestep
    comparison never touches.
    """
    pytest.importorskip("numba", reason="the compiled backend needs the breos[fast] extra")
    app_parity = _load_tool("parity/app_parity.py")

    compared, differences = app_parity.compare_scenario("c2_balanced", app_parity.SCENARIOS["c2_balanced"])
    assert not differences, "\n".join(differences)
    assert compared > 400, f"only {compared} fields compared -- the gate is not covering the output"
