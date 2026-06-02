"""Tests for public optimization helpers."""

import pandas as pd
import pytest

from breos.optimization import optimize_battery_size


def test_optimize_battery_size_uses_current_energy_balance_api():
    idx = pd.date_range("2025-01-01 00:00", periods=24, freq="h", tz="UTC")
    pv_dc = pd.Series([0.0] * 8 + [800.0] * 8 + [0.0] * 8, index=idx)
    houseload = pd.DataFrame({"Load": [300.0] * 24}, index=idx)

    result = optimize_battery_size(
        pv_dc=pv_dc,
        houseload=houseload,
        battery_sizes_wh=[0.0, 1000.0, 3000.0],
        objective="max_grid_independence",
        verbose=False,
    )

    assert result.optimal_value in {0.0, 1000.0, 3000.0}
    assert result.iterations == 3
    assert set(result.details["all_results"].columns) >= {
        "battery_size_wh",
        "grid_independence",
        "self_consumption",
    }


def test_optimize_battery_size_rejects_unknown_objective():
    idx = pd.date_range("2025-01-01 00:00", periods=2, freq="h", tz="UTC")
    pv_dc = pd.Series([0.0, 0.0], index=idx)
    houseload = pd.DataFrame({"Load": [100.0, 100.0]}, index=idx)

    with pytest.raises(ValueError, match="objective must be"):
        optimize_battery_size(
            pv_dc=pv_dc,
            houseload=houseload,
            battery_sizes_wh=[0.0],
            objective="max_magic",
            verbose=False,
        )
