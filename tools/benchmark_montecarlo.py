"""Benchmark the Monte Carlo execution backends.

Compilation is measured separately from simulation, and every timed run is
preceded by a warm-up, because ``cache=True`` means a first call can carry LLVM
compile time that later calls do not. Reporting a cold first call as the
steady-state cost would understate the gain; reporting it as warm would
overstate it.

Usage:
    python tools/benchmark_montecarlo.py --runs 8 --procs 1 10
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def machine_info() -> dict[str, object]:
    info: dict[str, object] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    try:
        model = [
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ]
        if model:
            info["cpu_model"] = model[0]
    except OSError:
        pass
    try:
        import llvmlite
        import numba

        info["numba"] = numba.__version__
        info["llvmlite"] = llvmlite.__version__
    except ImportError:
        info["numba"] = "not installed"
    return info


_COMPILE_PROBE = """
import json, sys, time
sys.path.insert(0, {root!r})
from breos._numba_dispatch import _build_kernel
import numpy as np
start = time.perf_counter()
kernel = _build_kernel()
hits_before = sum(kernel.stats.cache_hits.values())
misses_before = sum(kernel.stats.cache_misses.values())
matrix = np.zeros((37, 96))
pv = np.zeros(96); load = np.zeros(96); temp = np.full(96, 25.0)
kernel(matrix, pv, load, temp, 0, 96, 0.0, 0.0, False, 0.0, 1.0, 100.0, 0.9, 0.1,
       0.0, 0.95, 0.95, 0.96, np.inf, np.inf, np.inf, True, 0.05, 0.25, 2.0)
hits_after = sum(kernel.stats.cache_hits.values())
misses_after = sum(kernel.stats.cache_misses.values())
state = "cold" if misses_after > misses_before else "warm" if hits_after > hits_before else "unknown"
print(json.dumps({{"cache": state, "seconds": time.perf_counter() - start}}))
"""


def measure_compilation() -> dict[str, float | str]:
    """Time the first kernel call with a cold cache and again with a warm one."""
    results: dict[str, float | str] = {}
    with tempfile.TemporaryDirectory() as cache_dir:
        env = {**os.environ, "NUMBA_CACHE_DIR": cache_dir}
        probe = _COMPILE_PROBE.format(root=str(PROJECT_ROOT))
        for label in ("cold", "warm"):
            out = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True, check=True)
            payload = json.loads(out.stdout.strip().splitlines()[-1])
            results[f"{label}_cache_first_call_s"] = round(payload["seconds"], 3)
            results[f"{label}_cache_state_observed"] = payload["cache"]
    return results


def build_study(weather: Path, runs: int, procs: int, backend: str, years: int):
    from breos.montecarlo import MonteCarloSettings

    return MonteCarloSettings(
        weather_file=str(weather),
        n_runs=runs,
        years_per_run=years,
        seed=42,
        load_uncertainty=0.05,
        load_distribution="uniform",
        min_load_scale=0.95,
        max_load_scale=1.05,
        preserve_irradiance_energy=True,
        collect_yearly=False,
        n_procs=procs,
        execution_backend=backend,
    )


def config(n_modules: int, battery_kwh: float, years: int) -> dict:
    return {
        "location": "porto",
        "n_modules": n_modules,
        "annual_consumption_kwh": 5000,
        "battery_kwh": battery_kwh,
        "cost_preset": "residential_pt",
        "emissions_country": "PT",
        "resolution": "15min",
        "projection_years": years,
        "battery_max_charge_power_w": 4352.0,
        "battery_min_soc": 0.10,
        "battery_max_soc": 0.90,
        "battery_eol_percentage": 0.70,
        "battery_rte": 0.95,
        "battery_temperature": 25.0,
        "inverter_efficiency": 0.96,
        "inverter_loading_ratio": 1.25,
        "calendar_model": "naumann_lam_field_calibrated",
        "pv_degradation_rate": 0.005,
    }


def _time_setup() -> dict[str, float]:
    """Instrument the one-off year-cache build so it can be reported apart."""
    import breos.montecarlo as mc_module

    holder = {"seconds": 0.0}
    original = getattr(mc_module, "_precompute_year_caches_original", mc_module._precompute_year_caches)
    mc_module._precompute_year_caches_original = original

    def timed(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            holder["seconds"] = time.perf_counter() - start

    mc_module._precompute_year_caches = timed
    return holder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weather-file", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=8, help="Trajectories per timed measurement")
    parser.add_argument("--years", type=int, default=20)
    parser.add_argument("--procs", type=int, nargs="+", default=[1])
    parser.add_argument("--n-modules", type=int, default=10)
    parser.add_argument("--battery-kwh", type=float, default=10.0)
    parser.add_argument("--backends", nargs="+", default=["python", "numba"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from breos.montecarlo import run_montecarlo

    cfg = config(args.n_modules, args.battery_kwh, args.years)
    report: dict[str, object] = {"machine": machine_info(), "measurements": []}

    if "numba" in args.backends:
        print("measuring compilation...", flush=True)
        report["compilation"] = measure_compilation()
        for key, value in report["compilation"].items():
            print(f"  {key}: {value}")

    print("\nwarming up...", flush=True)
    for backend in args.backends:
        run_montecarlo(cfg, build_study(args.weather_file, 1, 1, backend, args.years))

    baseline: dict[int, float] = {}
    for procs in args.procs:
        for backend in args.backends:
            settings = build_study(args.weather_file, args.runs, procs, backend, args.years)
            setup_holder = _time_setup()
            start = time.perf_counter()
            result = run_montecarlo(cfg, settings)
            wall = time.perf_counter() - start
            # The PV and weather year caches are built once per study and cost
            # the same on both backends. Reporting only total wall time would
            # dilute the dispatch speedup by a fixed setup that neither backend
            # is responsible for, so both numbers are kept.
            setup = setup_holder["seconds"]
            simulation = wall - setup
            per_trajectory = simulation / args.runs
            throughput = args.runs / simulation
            entry = {
                "backend": backend,
                "workers": procs,
                "trajectories": args.runs,
                "years_per_trajectory": args.years,
                "wall_s": round(wall, 3),
                "setup_s": round(setup, 3),
                "simulation_s": round(simulation, 3),
                "s_per_trajectory": round(per_trajectory, 4),
                "trajectories_per_s": round(throughput, 4),
                "execution_provenance": result.provenance["execution"],
            }
            if backend == "python":
                baseline[procs] = simulation
            elif procs in baseline:
                entry["speedup_vs_python"] = round(baseline[procs] / simulation, 2)
            report["measurements"].append(entry)
            speed = f"  speedup {entry['speedup_vs_python']}x" if "speedup_vs_python" in entry else ""
            print(
                f"{backend:>7} | {procs:>2} worker(s) | wall {wall:8.2f} s "
                f"(setup {setup:5.2f} s, simulation {simulation:8.2f} s) | "
                f"{per_trajectory:7.3f} s/trajectory | {throughput:6.3f} traj/s{speed}",
                flush=True,
            )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
