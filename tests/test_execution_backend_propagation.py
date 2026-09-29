"""The backend choice must reach every simulating path, and no other.

These are propagation and configuration tests, not numerical ones. Parity lives
in ``test_numba_dispatch_parity.py``; what is checked here is that the choice
travels explicitly, defaults to the reference implementation everywhere, is
validated in one place, and is recorded wherever results are written.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from breos import App
from breos.app_config import resolve_app_config
from breos.execution import (
    DEFAULT_EXECUTION_BACKEND,
    EXECUTION_BACKENDS,
    PV_ONLY_DISPATCH_PATH,
    aggregate_jit_cache_states,
    backend_provenance,
    is_pv_only_dispatch,
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

# One projected year keeps each optimizer call to a single simulated year.
OPTIMIZATION_CONFIG = {
    "location": {"latitude": 41.15, "longitude": -8.61, "timezone": "UTC"},
    "constraints": {"budget": 100000.0, "max_area_m2": 100.0, "max_modules": 8, "max_battery_kwh": 5.0},
    "simulation": {"resolution": "h", "years_projection": 1},
    "mode": {"fixed_azimuth": 180},
    "battery": {"temperature": 20.0},
}
DESIGN = {"n_modules": 6, "battery_kwh": 5.0, "tilt": 30.0, "azimuth": 180.0}


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), PROJECT_ROOT / "tools" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _must_not_run(what: str):
    def _fail(*args, **kwargs):
        raise AssertionError(f"{what} before the backend was checked")

    return _fail


@pytest.fixture(scope="module")
def _optimization_inputs():
    from tests.conftest import _build_synthetic_weather

    weather = _build_synthetic_weather(2023)
    return weather, pd.DataFrame({"Load": 400.0}, index=weather.index)


@pytest.fixture
def _dispatch_backends(monkeypatch):
    """Record the backend each simulated span resolves its dispatch kernel for.

    The reference kernel stands in for the compiled one, so this runs without
    the breos[fast] extra: what is observed is which name reaches the kernel,
    not what the compiled code does with it.
    """
    from breos import _numba_dispatch, battery

    reference = battery._resolve_dispatch_day("python")
    seen: list[str] = []

    def _resolve(execution_backend):
        seen.append(execution_backend)
        return reference

    monkeypatch.setattr(battery, "_resolve_dispatch_day", _resolve)
    monkeypatch.setattr(_numba_dispatch, "require_numba_dispatch_day", lambda: reference)
    return seen


def _run_app(inputs, request, **backend):
    request.getfixturevalue("_patch_weather")
    App({**BASE_CONFIG, **backend}).simulate()


def _run_optimize_battery_size(inputs, request, **backend):
    from breos.optimization import optimize_battery_size

    weather, load = inputs
    optimize_battery_size(
        pv_dc=weather["ghi"] * 2.0, houseload=load, battery_sizes_wh=[5000.0], verbose=False, **backend
    )


def _run_evaluate_projected_design(inputs, request, **backend):
    from breos.optimization import evaluate_projected_design

    evaluate_projected_design(*inputs, OPTIMIZATION_CONFIG, **DESIGN, **backend)


def _run_solar_design_problem(inputs, request, **backend):
    pytest.importorskip("pymoo")
    from breos.optimization import SolarDesignProblem

    problem = SolarDesignProblem(*inputs, OPTIMIZATION_CONFIG, "results/_test_run/backend_propagation", **backend)
    problem._evaluate(np.array([DESIGN["n_modules"], DESIGN["battery_kwh"], DESIGN["tilt"]], dtype=float), {})


def _run_optimize_system_multi_objective(inputs, request, **backend):
    pytest.importorskip("pymoo")
    from breos.optimization import optimize_system_multi_objective

    optimize_system_multi_objective(*inputs, OPTIMIZATION_CONFIG, pop_size=2, n_gen=1, seed=1, **backend)


def test_the_reference_implementation_is_the_default_everywhere():
    """Nothing selects the compiled path without being asked.

    That each entry point also defaults to it is observed at the kernel, in
    the default cases of the propagation test below.
    """
    assert DEFAULT_EXECUTION_BACKEND == "python"
    assert resolve_app_config(BASE_CONFIG).cfg["execution_backend"] == "python"


@pytest.mark.parametrize("backend", [None, "numba"], ids=["default", "numba"])
@pytest.mark.parametrize(
    "run",
    [
        _run_app,
        _run_optimize_battery_size,
        _run_evaluate_projected_design,
        _run_solar_design_problem,
        _run_optimize_system_multi_objective,
    ],
    ids=lambda run: run.__name__.removeprefix("_run_"),
)
def test_the_chosen_backend_reaches_the_dispatch_kernel(
    run, backend, request, _optimization_inputs, _dispatch_backends
):
    """Every simulating entry point hands its backend to the kernel.

    Without an explicit choice, the reference kernel is what arrives. With
    one, a choice dropped on the way would fall back to that default and
    arrive as python, so the compiled case is the one that proves the
    argument travels.
    """
    run(_optimization_inputs, request, **({} if backend is None else {"execution_backend": backend}))

    assert _dispatch_backends, "no simulated span resolved a dispatch kernel"
    assert set(_dispatch_backends) == {backend or DEFAULT_EXECUTION_BACKEND}


@pytest.mark.parametrize(
    "run",
    [_run_evaluate_projected_design, _run_solar_design_problem, _run_optimize_system_multi_objective],
    ids=lambda run: run.__name__.removeprefix("_run_"),
)
def test_optimization_refuses_a_backend_named_in_its_config(run, request, _optimization_inputs, monkeypatch):
    """Candidate scoring is the hottest loop, so its backend must be explicit.

    A backend read out of a nested config dict would be invisible at the call
    site and impossible to attribute afterwards. The config therefore cannot
    carry one: every entry point that takes a config refuses the key, rather
    than reading or ignoring it, so the argument is the only way in.
    """
    monkeypatch.setitem(OPTIMIZATION_CONFIG, "execution_backend", "numba")

    with pytest.raises(ValueError, match="Pass execution_backend to the function"):
        run(_optimization_inputs, request)


@pytest.mark.parametrize("backend", EXECUTION_BACKENDS)
def test_app_records_the_backend_and_its_toolchain(backend, _patch_weather):
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


def test_app_and_monte_carlo_record_the_same_execution_keys(_patch_weather):
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


def test_missing_numba_is_reported_before_app_prepares_inputs(monkeypatch):
    """The dependency check must precede input preparation, which can hit the network.

    Downloading a year of weather and then failing on a missing import wastes
    the expensive step to report the cheap problem.
    """
    from breos import _numba_dispatch
    from breos.runners import app as app_runner

    monkeypatch.setattr(_numba_dispatch, "numba_available", lambda: False)
    monkeypatch.setattr(app_runner, "prepare_simulation_inputs", _must_not_run("inputs were prepared"))

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


def test_missing_numba_is_reported_before_the_pv_model_runs(monkeypatch, request, _optimization_inputs):
    """evaluate_projected_design checks before a year of irradiance is computed."""
    from breos import _numba_dispatch, optimization

    monkeypatch.setattr(_numba_dispatch, "numba_available", lambda: False)
    monkeypatch.setattr(optimization, "calculate_pv_production_dc", _must_not_run("the PV model ran"))

    with pytest.raises(_numba_dispatch.NumbaUnavailableError):
        _run_evaluate_projected_design(_optimization_inputs, request, execution_backend="numba")


def test_missing_numba_is_reported_before_the_worker_pool_exists(monkeypatch, request, _optimization_inputs):
    """optimize_system_multi_objective checks before it starts workers or scores a candidate.

    Two workers are asked for so a pool would be created if the check came
    late; a missing dependency must not surface from inside one.
    """
    import multiprocessing

    from breos import _numba_dispatch, optimization

    pytest.importorskip("pymoo")
    monkeypatch.setattr(_numba_dispatch, "numba_available", lambda: False)
    monkeypatch.setattr(multiprocessing, "Pool", _must_not_run("a worker pool was created"))
    monkeypatch.setattr(optimization, "calculate_pv_production_dc", _must_not_run("a candidate was scored"))

    with pytest.raises(_numba_dispatch.NumbaUnavailableError):
        optimization.optimize_system_multi_objective(
            *_optimization_inputs, OPTIMIZATION_CONFIG, pop_size=2, n_gen=1, n_procs=2, execution_backend="numba"
        )


def test_numba_provenance_always_carries_a_cache_field():
    """A driver that cannot observe its workers still records the field.

    A driver that fans work out to subprocesses has no observations to
    aggregate. "unknown" is provenance; a missing
    key reads as an oversight when the run that produced it took hours.
    """
    pytest.importorskip("numba", reason="the compiled backend needs the breos[fast] extra")

    assert backend_provenance("numba")["jit_cache"] == "unknown"
    assert backend_provenance("numba", jit_cache_states=["warm", "warm"])["jit_cache"] == "warm"
    assert backend_provenance("numba", jit_cache_states=["warm", "cold"])["jit_cache"] == "cold"
    assert "jit_cache" not in backend_provenance("python")


def test_pv_only_dispatch_path_agrees_with_the_dispatch_itself():
    """One predicate, so provenance cannot claim a path the run did not take.

    A battery too small to move energy, or with no room between its SOC
    limits, takes the vectorized PV-only path. Provenance said otherwise while
    App, Monte Carlo and the dispatch each spelled the question themselves.
    """
    assert is_pv_only_dispatch(0.0, 0.9, 0.1)
    assert is_pv_only_dispatch(0.5, 0.9, 0.1)
    assert is_pv_only_dispatch(10_000.0, 0.9, 0.9)
    assert not is_pv_only_dispatch(10_000.0, 0.9, 0.1)

    assert backend_provenance("python", pv_only=True)["dispatch_path"] == PV_ONLY_DISPATCH_PATH
    assert "dispatch_path" not in backend_provenance("python")


def test_app_records_the_pv_only_path_for_a_battery_too_small_to_dispatch(_patch_weather):
    """The disagreement this centralisation removes, at the App boundary.

    Half a watt-hour is below what the dispatch will move, so the run takes
    the vectorized PV-only path. Provenance used to say a battery was
    dispatched -- and on the compiled backend, that a kernel had run.
    """
    app = App({**BASE_CONFIG, "battery_kwh": 0.0005})
    app.simulate()
    assert app.result()["provenance"]["execution"]["dispatch_path"] == PV_ONLY_DISPATCH_PATH


def test_app_assembled_outputs_are_identical_on_both_backends(_patch_weather):
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
