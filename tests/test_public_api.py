"""Tests for the documented top-level BREOS API surface."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _subprocess_env(**overrides):
    env = os.environ.copy()
    pythonpath = [str(_REPO_ROOT), env.get("PYTHONPATH", "")]
    env["PYTHONPATH"] = os.pathsep.join(path for path in pythonpath if path)
    env.update(overrides)
    return env


def test_importing_breos_does_not_import_installed_numba(tmp_path):
    pytest.importorskip("numba")
    code = """
import sys
import breos
import breos.app
import breos.cli
import breos.montecarlo
import breos.optimization
assert "numba" not in sys.modules, sorted(name for name in sys.modules if name.startswith("numba"))
"""
    env = _subprocess_env(MPLCONFIGDIR=str(tmp_path))
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env, text=True, capture_output=True, check=False
    )

    assert completed.returncode == 0, completed.stderr


def test_top_level_namespace_matches_stable_release_surface():
    code = """
import types
import breos
from breos import cli as cli_module

expected = {
    "App",
    "__version__",
    "BatteryConfig",
    "CostParams",
    "EmissionsParams",
    "InverterConversionResult",
    "calculate_dc_ac_power",
    "OptimizationResult",
    "PVModuleParams",
    "fetch_tmy_weather_data",
    "load_profile",
    "calculate_pv_production_dc",
    "calculate_multi_array_production",
    "simulate_energy_balance",
    "calculate_costs",
    "cost_analysis_projection",
    "optimize_system_multi_objective",
    "export_results",
    "load_results",
    "repair_series",
    "InputRepairReport",
}
public_non_module_names = {
    name
    for name, value in vars(breos).items()
    if not isinstance(value, types.ModuleType) and (not name.startswith("_") or name in breos.__all__)
}
assert public_non_module_names == set(breos.__all__)
assert expected <= public_non_module_names
assert breos.App is not None
assert "__version__" in public_non_module_names
assert not hasattr(breos, "plot_co2_savings")
assert isinstance(breos.app, types.ModuleType)
assert isinstance(breos.cli, types.ModuleType)
assert breos.cli is cli_module
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], env=_subprocess_env(), text=True, capture_output=True, check=False
    )

    assert completed.returncode == 0, completed.stderr


def test_core_import_works_without_optional_plotting_or_numba(tmp_path):
    code = """
import importlib.abc
import sys

class BlockOptionalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "matplotlib" or fullname.startswith("matplotlib.") or fullname == "numba" or fullname.startswith("numba."):
            raise ModuleNotFoundError(f"blocked optional dependency: {fullname}")
        return None

sys.meta_path.insert(0, BlockOptionalImports())
import breos
assert callable(breos.App)
assert "matplotlib" not in sys.modules
assert "numba" not in sys.modules
"""
    env = _subprocess_env(MPLCONFIGDIR=str(tmp_path))
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env, text=True, capture_output=True, check=False
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""


def test_plotting_import_preserves_matplotlib_backend(tmp_path):
    code = """
import matplotlib

matplotlib.use("svg")
before = matplotlib.get_backend()

import breos.plotting

assert matplotlib.get_backend() == before
"""
    env = _subprocess_env(MPLCONFIGDIR=str(tmp_path))
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env, text=True, capture_output=True, check=False
    )

    assert completed.returncode == 0, completed.stderr


def test_keyword_only_boundaries_are_pinned():
    """A positional call past a removed argument raises instead of rebinding."""
    import inspect

    from breos.economics import cost_analysis_projection
    from breos.optimization import optimize_system_multi_objective

    def positional(func):
        # A *args would swallow a positional argument past the boundary.
        assert all(
            parameter.kind is not inspect.Parameter.VAR_POSITIONAL
            for parameter in inspect.signature(func).parameters.values()
        )
        return [
            name
            for name, parameter in inspect.signature(func).parameters.items()
            if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD and name != "self"
        ]

    assert positional(cost_analysis_projection) == [
        "yearly_summary_df",
        "costs",
        "num_years",
        "inflation_rate",
        "sell_price_inflation",
        "discount_rate",
    ]
    assert positional(optimize_system_multi_objective) == ["tmy_data", "houseload", "config"]

    pytest.importorskip("pymoo")
    from breos.optimization import SolarDesignProblem

    assert positional(SolarDesignProblem.__init__) == ["tmy_data", "houseload", "config"]
