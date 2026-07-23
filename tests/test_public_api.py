"""Tests for the documented top-level BREOS API surface."""

import os
import subprocess
import sys

import breos


def test_top_level_all_is_narrow_release_surface():
    stable_api = set(breos.__all__)

    expected = {
        "App",
        "__version__",
        "BatteryConfig",
        "CostParams",
        "EmissionsParams",
        "InverterConfig",
        "InverterConversionResult",
        "OptimizationResult",
        "PVModuleParams",
        "fetch_tmy_weather_data",
        "load_profile",
        "calculate_pv_production_dc",
        "calculate_multi_array_production",
        "simulate_energy_balance",
        "calculate_costs",
        "cost_analysis_projection",
        "calculate_co2_savings",
        "optimize_tilt",
        "optimize_battery_size",
        "optimize_system_multi_objective",
        "export_results",
        "load_results",
    }
    intentionally_excluded = {
        "R_GAS",
        "plot_co2_savings",
        "PolysunDegradationConfig",
        "compute_dod_histogram",
        "build_battery_temperature_series",
        "remap_datetime_index_years",
    }

    assert expected.issubset(stable_api)
    assert stable_api.isdisjoint(intentionally_excluded)


def test_existing_top_level_attributes_remain_importable():
    assert breos.R_GAS > 0
    assert callable(breos.build_battery_temperature_series)
    assert callable(breos.compute_dod_histogram)


def test_top_level_plotting_compatibility_is_lazy(tmp_path):
    """Core imports stay quiet while historical plotting attributes still work."""

    code = """
import sys

import breos

assert "breos.plotting" not in sys.modules
assert "matplotlib" not in sys.modules
assert "plot_co2_savings" in dir(breos)
assert callable(breos.plot_co2_savings)
assert "breos.plotting" in sys.modules
assert "matplotlib" in sys.modules
"""
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(tmp_path)
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
