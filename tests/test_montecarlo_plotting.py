"""Tests for Monte Carlo plotting helpers."""

import numpy as np
import pandas as pd
import pytest


def test_plot_montecarlo_simulation_accepts_breos_summary_schema(tmp_path):
    pytest.importorskip("matplotlib")

    from breos.plotting import plot_montecarlo_simulation

    runs = pd.DataFrame(
        {
            "run": [1, 2, 3],
            "npv_savings": [1200.0, 1800.0, -200.0],
            "payback_year": [10.0, 12.0, np.nan],
            "lcoe_per_kwh": [0.21, 0.19, 0.24],
            "final_soh_pct": [74.0, 73.5, 74.5],
            "mean_grid_independence_pct": [55.0, 58.0, 52.0],
            "total_replacements": [0, 0, 0],
        }
    )

    plot_montecarlo_simulation(runs, str(tmp_path), verbose=False)

    expected = [
        "plots/breakeven_histogram.png",
        "plots/breakeven_cdf.png",
        "plots/breakeven_summary_bar.png",
        "plots/montecarlo_npv_distribution.png",
        "plots/montecarlo_grid_independence_distribution.png",
        "plots/montecarlo_final_soh_distribution.png",
        "plots/montecarlo_lcoe_distribution.png",
    ]
    for rel_path in expected:
        path = tmp_path / rel_path
        assert path.exists()
        assert path.stat().st_size > 0


def test_plot_montecarlo_simulation_rejects_other_tables(tmp_path):
    pytest.importorskip("matplotlib")

    from breos.plotting import plot_montecarlo_simulation

    # The legacy run-year table that the removed overlay plots read.
    run_years = pd.DataFrame({"run_number": [1, 1], "year": [1, 2], "cumulative_system_cost": [900.0, 950.0]})

    with pytest.raises(ValueError, match="one-row-per-run"):
        plot_montecarlo_simulation(run_years, str(tmp_path), verbose=False)
    assert not (tmp_path / "plots").exists()
