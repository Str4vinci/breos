"""Worker processes run their native thread pools on one thread; the parent keeps its own."""

from __future__ import annotations

import multiprocessing

import numpy as np
import pytest
from threadpoolctl import threadpool_info, threadpool_limits

import breos.montecarlo as montecarlo_module
from breos.battery import weighted_column_sums
from breos.execution import limit_worker_threads, single_thread_blas
from breos.montecarlo import MonteCarloSettings, run_montecarlo

# Long enough for OpenBLAS to split a dot product across threads: one year
# at 15-minute steps.
STEPS_15MIN = 35_040


def _blas_threads() -> set[int]:
    return {library["num_threads"] for library in threadpool_info() if library["user_api"] == "blas"}


@pytest.fixture
def parent_blas_threads():
    """Give the test process two BLAS threads, so one thread in a worker is the limit at work."""
    if not _blas_threads():
        pytest.skip("threadpoolctl finds no BLAS library to limit")
    with threadpool_limits(limits=2, user_api="blas"):
        if _blas_threads() != {2}:
            pytest.skip("the BLAS library cannot run two threads here")
        yield 2


def _worker_blas_threads(_: int) -> set[int]:
    return _blas_threads()


_RUN_TRAJECTORY = montecarlo_module._run_trajectory_index


def _recording_trajectory(run_idx: int):
    """A Monte Carlo trajectory that also reports its worker's BLAS threads."""
    run_idx, metrics, trajectory, jit_cache_state = _RUN_TRAJECTORY(run_idx)
    return run_idx, {**metrics, "blas_threads": sorted(_blas_threads())}, trajectory, jit_cache_state


def test_a_limited_worker_sees_one_blas_thread(parent_blas_threads):
    with multiprocessing.Pool(2, initializer=limit_worker_threads) as pool:
        seen = pool.map(_worker_blas_threads, range(4))

    assert seen == [{1}] * 4
    assert _blas_threads() == {parent_blas_threads}


@pytest.mark.skipif(
    multiprocessing.get_start_method() != "fork",
    reason="the patched trajectory reaches the workers only through fork",
)
def test_montecarlo_workers_see_one_blas_thread(parent_blas_threads, tmp_path, monkeypatch, write_multiyear_weather):
    monkeypatch.setattr(montecarlo_module, "_run_trajectory_index", _recording_trajectory)
    weather = write_multiyear_weather(tmp_path / "multi.csv")
    config = {
        "location": "porto",
        "n_modules": 8,
        "annual_consumption_kwh": 4000,
        "battery_kwh": 5.0,
        "cost_preset": "residential_pt",
        "emissions_country": "PT",
        "resolution": "h",
        "projection_years": 1,
    }
    settings = MonteCarloSettings(weather_file=str(weather), n_runs=2, years_per_run=1, seed=42, n_procs=2)

    runs = run_montecarlo(config, settings).runs

    assert runs["blas_threads"].tolist() == [[1], [1]]
    assert _blas_threads() == {parent_blas_threads}


def test_single_thread_blas_restores_the_callers_threads(parent_blas_threads):
    with single_thread_blas():
        assert _blas_threads() == {1}
    assert _blas_threads() == {parent_blas_threads}


def test_tariff_sums_do_not_depend_on_the_blas_thread_count(parent_blas_threads):
    rng = np.random.default_rng(7)
    columns = {"Grid_Import": rng.random(STEPS_15MIN) * 3000.0}
    weights = {"Import_Cost": ("Grid_Import", rng.random(STEPS_15MIN) * 0.3)}

    with threadpool_limits(limits=1, user_api="blas"):
        one_thread = float(np.dot(columns["Grid_Import"], weights["Import_Cost"][1]))

    assert weighted_column_sums(columns, weights) == {"Import_Cost": one_thread}
