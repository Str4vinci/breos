"""Tests for Monte Carlo plotting helpers."""

import pandas as pd
import pytest


def test_plot_montecarlo_simulation_rejects_other_tables(tmp_path):
    pytest.importorskip("matplotlib")

    from breos.plotting import plot_montecarlo_simulation

    # The legacy run-year table that the removed overlay plots read.
    run_years = pd.DataFrame({"run_number": [1, 1], "year": [1, 2], "cumulative_system_cost": [900.0, 950.0]})

    with pytest.raises(ValueError, match="one-row-per-run"):
        plot_montecarlo_simulation(run_years, str(tmp_path), verbose=False)
    assert not (tmp_path / "plots").exists()
