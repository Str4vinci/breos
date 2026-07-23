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

    core_import_code = """
import sys

import breos

assert "breos.plotting" not in sys.modules
assert "matplotlib" not in sys.modules
assert "plot_co2_savings" in dir(breos)
"""
    plotting_import_code = """
import sys

import breos

assert "breos.plotting" not in sys.modules
assert "matplotlib" not in sys.modules
assert callable(breos.plot_co2_savings)
assert "breos.plotting" in sys.modules
assert "matplotlib" in sys.modules
"""
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(tmp_path)
    core_import = subprocess.run(
        [sys.executable, "-c", core_import_code],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert core_import.returncode == 0, core_import.stderr
    assert core_import.stderr == ""

    plotting_import = subprocess.run(
        [sys.executable, "-c", plotting_import_code],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    # A fresh Matplotlib installation may announce font-cache creation on
    # stderr. The quiet-import contract applies before plotting is requested.
    assert plotting_import.returncode == 0, plotting_import.stderr
