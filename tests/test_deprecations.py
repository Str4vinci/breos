"""Compatibility tests for APIs scheduled for removal in BREOS 0.6.0."""

import inspect
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import breos
from breos import battery, io, optimization, plotting, polysun_degradation, solar, utils, weather

DEPRECATED_CALLABLES = {
    battery: {"compute_halfcycle_energy_throughput", "k_c_rate_Q", "k_doc_Q", "update_battery_soc"},
    polysun_degradation: {
        "PolysunDegradationConfig",
        "compute_dod_histogram",
        "compute_miner_damage",
        "predict_polysun_lifetime",
        "simulate_polysun_degradation",
        "woehler_cycles_to_failure",
    },
    plotting: {
        "plot_degradation_methodology_comparison",
        "plot_lifetime_prediction_comparison",
        "plot_loo_cv_summary",
        "plot_loo_param_stability",
        "plot_loo_predictions",
        "plot_optimization_results_2d",
        "plot_optimization_results_3d",
        "plot_smart_charging_sweep",
        "plot_temperature_sensitivity_comparison",
    },
    io: {"export_monthly_summary", "export_yearly_summary", "save_simulation_report"},
    weather: {"csv_15min_to_hourly", "csv_hourly_to_15min", "fetch_tmy_nsrdb", "resample_to_hourly"},
    solar: {"calculate_pv_production_tmy", "zeb_sizer"},
    optimization: {"optimize_tilt_brent", "size_for_zeb"},
    utils: {"count_leap_years", "number_of_cores"},
}


def _source_tree_env(tmp_path):
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(tmp_path)
    project_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [project_root, env.get("PYTHONPATH")]))
    return env


def test_importing_core_package_does_not_emit_deprecation_warning(tmp_path):
    code = "import warnings; warnings.simplefilter('error', DeprecationWarning); import breos"
    env = _source_tree_env(tmp_path)

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_every_scheduled_callable_records_its_removal_release():
    for module, names in DEPRECATED_CALLABLES.items():
        for name in names:
            assert getattr(module, name).__breos_deprecated_removal__ == "0.6.0"


def test_function_warning_preserves_behavior_and_points_to_replacement():
    with pytest.warns(
        DeprecationWarning,
        match=r"breos\.utils\.count_leap_years.*BREOS 0\.6\.0.*breos\.utils\.is_leap_year",
    ):
        result = utils.count_leap_years(2024, 5)

    assert result == 2
    assert list(inspect.signature(utils.count_leap_years).parameters) == ["start_year", "num_years"]


def test_deprecated_dataclass_warns_only_when_instantiated():
    config_class = polysun_degradation.PolysunDegradationConfig

    with pytest.warns(DeprecationWarning, match=r"PolysunDegradationConfig.*BREOS 0\.6\.0"):
        config = config_class(n_bins=12)

    assert config.n_bins == 12
    assert "n_bins" in inspect.signature(config_class).parameters


def test_polysun_entrypoint_emits_one_warning_at_the_user_call_site():
    with pytest.warns(DeprecationWarning):
        config = polysun_degradation.PolysunDegradationConfig(n_bins=4)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        result = polysun_degradation.simulate_polysun_degradation(
            np.array([0.2, 0.8, 0.2]),
            config,
            n_years=1,
        )

    assert len(caught) == 1
    assert Path(caught[0].filename) == Path(__file__)
    assert len(result) == 1


def test_deprecated_plot_warns_before_preserving_argument_validation():
    with pytest.warns(DeprecationWarning, match=r"plot_loo_cv_summary.*BREOS 0\.6\.0"):
        with pytest.raises(TypeError):
            plotting.plot_loo_cv_summary()


def test_numba_module_warns_on_direct_import(tmp_path):
    code = "import warnings; warnings.simplefilter('always', DeprecationWarning); import breos.numba_kernels"
    env = _source_tree_env(tmp_path)

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "breos.numba_kernels is deprecated" in result.stderr
    assert "removed in BREOS 0.6.0" in result.stderr


def test_top_level_compatibility_aliases_are_unchanged():
    assert breos.save_simulation_report is io.save_simulation_report
    assert breos.fetch_tmy_nsrdb is weather.fetch_tmy_nsrdb
    assert breos.optimize_tilt_brent is optimization.optimize_tilt_brent
    assert breos.calculate_pv_production_tmy is solar.calculate_pv_production_tmy
    assert breos.compute_dod_histogram is polysun_degradation.compute_dod_histogram
