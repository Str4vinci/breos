"""Backend selection, dependency checking, and run provenance."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from breos.battery import EXECUTION_BACKENDS, _dispatch_day_python, _resolve_dispatch_day
from breos.montecarlo import MonteCarloSettings, _aggregate_jit_cache_states, run_montecarlo


def _write_multiyear_weather(path, years=(2021, 2022)):
    frames = []
    for year in years:
        idx = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
        idx = idx[~((idx.month == 2) & (idx.day == 29))]
        hour = idx.hour.to_numpy()
        daylight = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None)
        ghi = 700.0 * daylight
        frames.append(
            pd.DataFrame(
                {
                    "date": idx,
                    "temperature_2m": 15.0 + 8.0 * daylight,
                    "wind_speed_10m": 2.0,
                    "shortwave_radiation": ghi,
                    "direct_normal_irradiance": 0.8 * ghi,
                    "diffuse_radiation": 0.2 * ghi,
                }
            )
        )
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    return path


def _base_config():
    return {
        "location": "porto",
        "n_modules": 6,
        "annual_consumption_kwh": 4000,
        "battery_kwh": 5.0,
        "cost_preset": "residential_pt",
        "emissions_country": "PT",
        "resolution": "h",
        "projection_years": 2,
    }


def test_python_is_the_default_and_the_reference():
    assert EXECUTION_BACKENDS == ("python", "numba")
    assert MonteCarloSettings(weather_file="x").execution_backend == "python"
    assert _resolve_dispatch_day("python") is _dispatch_day_python


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="execution_backend must be one of"):
        _resolve_dispatch_day("cuda")


def test_montecarlo_rejects_unknown_backend_before_loading_inputs(tmp_path):
    settings = MonteCarloSettings(weather_file=str(tmp_path / "missing.csv"), n_runs=1, execution_backend="cuda")
    with pytest.raises(ValueError, match="execution_backend must be one of"):
        run_montecarlo(_base_config(), settings)


def test_missing_numba_fails_before_any_trajectory_runs(tmp_path, monkeypatch):
    import breos._numba_dispatch as dispatch
    import breos.montecarlo as mc_module

    weather = _write_multiyear_weather(tmp_path / "multi.csv")
    monkeypatch.setattr(dispatch, "numba_available", lambda: False)

    def _must_not_run(*args, **kwargs):
        raise AssertionError("a trajectory started despite the missing dependency")

    monkeypatch.setattr(mc_module, "simulate_energy_balance_summary", _must_not_run)

    settings = MonteCarloSettings(weather_file=str(weather), n_runs=4, execution_backend="numba")
    with pytest.raises(dispatch.NumbaUnavailableError, match=r"breos\[fast\]"):
        run_montecarlo(_base_config(), settings)


def test_provenance_records_the_python_backend_and_versions(tmp_path):
    weather = _write_multiyear_weather(tmp_path / "multi.csv")
    settings = MonteCarloSettings(weather_file=str(weather), n_runs=1, years_per_run=1, seed=3)
    result = run_montecarlo(_base_config(), settings)

    execution = result.provenance["execution"]
    assert execution["execution_backend"] == "python"
    assert execution["numpy"] == np.__version__
    assert execution["pandas"] == pd.__version__
    assert execution["python"].count(".") == 2
    # No compiler is involved on the reference path, so none is claimed.
    assert "numba" not in execution
    assert "jit_cache" not in execution


def test_provenance_records_the_compiler_versions_and_cache_state(tmp_path):
    pytest.importorskip("numba", reason="the compiled backend needs the breos[fast] extra")
    import llvmlite
    import numba

    weather = _write_multiyear_weather(tmp_path / "multi.csv")
    settings = MonteCarloSettings(
        weather_file=str(weather), n_runs=1, years_per_run=1, seed=3, execution_backend="numba"
    )
    result = run_montecarlo(_base_config(), settings)

    execution = result.provenance["execution"]
    assert execution["execution_backend"] == "numba"
    # A bit-identity claim is scoped to a toolchain, so the toolchain is
    # recorded for every run, not only for benchmarks.
    assert execution["numba"] == numba.__version__
    assert execution["llvmlite"] == llvmlite.__version__
    assert execution["jit_cache"] in {"warm", "cold"}


def test_jit_cache_state_ignores_unverified_cache_files(tmp_path, monkeypatch):
    import breos._numba_dispatch as dispatch

    monkeypatch.setenv("NUMBA_CACHE_DIR", str(tmp_path))
    (tmp_path / "_numba_dispatch.stale.nbi").write_bytes(b"not a valid Numba cache index")
    dispatch.reset_jit_cache_observation()

    assert dispatch.jit_cache_state() is None


def _call_numba_cache_probe(dispatch):
    dispatch._dispatch_day_numba(
        SimpleNamespace(matrix=np.zeros((37, 1))),
        np.zeros(1),
        np.zeros(1),
        np.full(1, 25.0),
        0,
        1,
        battery_config=SimpleNamespace(
            nominal_energy_wh=0.0,
            max_soc=0.9,
            min_soc=0.1,
            inverter_efficiency=0.96,
            thermal_resistance_kw=0.05,
        ),
        has_battery=False,
        battery_soh_decimal=1.0,
        Battery_SOH=100.0,
        Battery_Energy_Wh=0.0,
        Battery_PV_Origin_Energy_Wh=0.0,
        eff_charge=0.95,
        eff_discharge=0.95,
        hours_per_step=1.0,
        standby_loss_per_step_wh=0.0,
        cap_wh=np.inf,
        cap_charge_wh=np.inf,
        cap_discharge_wh=np.inf,
    )


def test_jit_cache_state_reports_miss_then_in_memory_reuse(monkeypatch):
    import breos._numba_dispatch as dispatch

    class _Kernel:
        stats = SimpleNamespace(cache_hits={}, cache_misses={})
        signatures = []

        def __call__(self, *args):
            if not self.signatures:
                self.stats.cache_misses["signature"] = 1
                self.signatures.append(("compiled",))
            return 0.0, 0.0, 0.0, 0.0

    monkeypatch.setattr(dispatch, "_KERNEL", _Kernel())
    dispatch.reset_jit_cache_observation()
    _call_numba_cache_probe(dispatch)
    assert dispatch.observed_jit_cache_state() == "cold"

    dispatch.reset_jit_cache_observation()
    _call_numba_cache_probe(dispatch)
    assert dispatch.observed_jit_cache_state() == "warm"


def test_montecarlo_provenance_uses_worker_observations_across_repeated_studies(monkeypatch):
    import breos._numba_dispatch as dispatch
    import breos.montecarlo as mc_module

    class _Kernel:
        stats = SimpleNamespace(cache_hits={}, cache_misses={})
        signatures = []

        def __call__(self, *args):
            if not self.signatures:
                self.stats.cache_misses["signature"] = 1
                self.signatures.append(("compiled",))
            return 0.0, 0.0, 0.0, 0.0

    monkeypatch.setattr(dispatch, "_KERNEL", _Kernel())
    monkeypatch.setattr(
        mc_module,
        "_precompute_year_caches",
        lambda *args: ({2021: pd.Series([0.0])}, {2021: pd.Series([25.0])}),
    )
    # A one-column frame, which is what load_consumption_profile really
    # returns; the study now aligns it before any trajectory runs.
    monkeypatch.setattr(
        mc_module,
        "load_consumption_profile",
        lambda *args, **kwargs: pd.DataFrame({"Load": [0.0]}),
    )

    def _simulate_without_inputs(*args, **kwargs):
        _call_numba_cache_probe(dispatch)
        return {}, pd.DataFrame()

    monkeypatch.setattr(mc_module, "_simulate_trajectory", _simulate_without_inputs)
    settings = MonteCarloSettings(
        weather_file="unused.csv",
        n_runs=2,
        years_per_run=1,
        execution_backend="numba",
    )

    first = run_montecarlo(_base_config(), settings)
    second = run_montecarlo(_base_config(), settings)

    assert first.provenance["execution"]["jit_cache"] == "cold"
    assert second.provenance["execution"]["jit_cache"] == "warm"


@pytest.mark.parametrize(
    ("states", "expected"),
    [
        (["warm"], "warm"),
        (["warm", "warm"], "warm"),
        (["warm", "cold", "warm"], "cold"),
    ],
)
def test_jit_cache_worker_observations_are_aggregated(states, expected):
    assert _aggregate_jit_cache_states(states) == expected


@pytest.mark.parametrize("states", [[], ["unknown"], ["warm", "unknown"], ["cold", "unknown"]])
def test_unclassifiable_jit_cache_observations_degrade_to_unknown(states):
    """Provenance bookkeeping must never fail a study that produced results.

    An empty list means nothing was observed, which is not evidence of a warm
    cache; a single "unknown" from any worker makes the study-level claim
    untrustworthy. Both report "unknown" rather than raising.
    """
    assert _aggregate_jit_cache_states(states) == "unknown"


def test_both_backends_agree_on_a_seeded_study(tmp_path):
    pytest.importorskip("numba", reason="the compiled backend needs the breos[fast] extra")
    weather = _write_multiyear_weather(tmp_path / "multi.csv")

    def run(backend):
        return run_montecarlo(
            _base_config(),
            MonteCarloSettings(
                weather_file=str(weather),
                n_runs=3,
                years_per_run=2,
                seed=11,
                collect_yearly=True,
                execution_backend=backend,
            ),
        )

    reference = run("python")
    compiled = run("numba")

    pd.testing.assert_frame_equal(reference.runs, compiled.runs, check_exact=True)
    pd.testing.assert_frame_equal(reference.yearly, compiled.yearly, check_exact=True)
    assert reference.summary == compiled.summary


_CACHE_PROBE = """
import json, sys
sys.path.insert(0, {root!r})
from breos._numba_dispatch import _build_kernel
import numpy as np

kernel = _build_kernel()
matrix = np.zeros((37, 96))
kernel(
    matrix, np.zeros(96), np.zeros(96), np.full(96, 25.0), 0, 96,
    0.0, 0.0, False, 0.0, 1.0, 100.0, 0.9, 0.1, 0.0, 0.95, 0.95, 0.96,
    np.inf, np.inf, np.inf, True, 0.05, 0.25, 2.0,
)
print(json.dumps({{
    "hits": int(sum(kernel.stats.cache_hits.values())),
    "misses": int(sum(kernel.stats.cache_misses.values())),
}}))
"""


def _run_cache_probe(cache_dir):
    """Compile-and-call the dispatch kernel in a fresh process, return its counters."""
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = str(Path(__file__).resolve().parents[1])
    completed = subprocess.run(
        [sys.executable, "-c", _CACHE_PROBE.format(root=root)],
        env={**os.environ, "NUMBA_CACHE_DIR": str(cache_dir)},
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _dispatch_cache_data_files(cache_dir):
    from pathlib import Path

    return sorted(Path(cache_dir).glob("**/*_dispatch_day_kernel*.nbc"))


def test_dispatch_kernel_cache_survives_a_new_process(tmp_path):
    """The on-disk cache must work across processes, not just within one.

    This is a regression test for a defect, not a nicety. When the kernels were
    defined inside a factory, ``_dispatch_day_kernel`` closed over the three
    helper dispatchers; Numba's cache index key for a closure includes the cell
    contents, which are not stable across processes, so every process missed
    the cache and appended another data file. Under multiprocessing that meant
    every worker recompiled, and the cache directory grew without bound.

    Separate processes are the whole point -- an in-process check would pass
    against the broken version, because in-memory reuse always worked.
    """
    pytest.importorskip("numba", reason="the compiled backend needs the breos[fast] extra")
    cache_dir = tmp_path / "numba-cache"
    cache_dir.mkdir()

    first = _run_cache_probe(cache_dir)
    assert first["misses"] == 1, "the first process in a clean cache directory must compile"
    assert first["hits"] == 0

    second = _run_cache_probe(cache_dir)
    assert second["hits"] == 1, "a second process must load the compiled kernel from disk"
    assert second["misses"] == 0


def test_dispatch_kernel_cache_does_not_accumulate_data_files(tmp_path):
    """Repeated processes must reuse one cache entry rather than append new ones."""
    pytest.importorskip("numba", reason="the compiled backend needs the breos[fast] extra")
    cache_dir = tmp_path / "numba-cache"
    cache_dir.mkdir()

    for _ in range(3):
        _run_cache_probe(cache_dir)

    data_files = _dispatch_cache_data_files(cache_dir)
    assert len(data_files) == 1, f"cache entries accumulated across processes: {[p.name for p in data_files]}"
