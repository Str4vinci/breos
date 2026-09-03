"""Numerical checks for the recovered validation drivers."""

import numpy as np
import pytest

from tools.validation.recovery.hkust.drivers.hkust_validate import _finite_metric
from tools.validation.recovery.orientation_diversity.drivers.orientation_screen import (
    _metric_row as orientation_metric,
)
from tools.validation.recovery.pcoe.drivers.pcoe_validate import (
    _metric_row as pcoe_metric,
)
from tools.validation.recovery.pcoe.drivers.pcoe_validate import _power_consistency_row
from tools.validation.recovery.reunion_microgrid.drivers.reunion_validate import _metrics as reunion_metric
from tools.validation.recovery.sandia_task13.drivers.sandia_thermal_validate import (
    _metric_row as sandia_metric,
)


@pytest.mark.parametrize(
    "row",
    [
        sandia_metric("model", 200, np.array([1.0, 2.0, 4.0]), np.array([2.0, 2.0, 5.0])),
        pcoe_metric("current", "model", 200, np.array([1.0, 2.0, 4.0]), np.array([2.0, 2.0, 5.0])),
    ],
)
def test_thermal_driver_metrics_use_prediction_minus_measurement(row):
    assert row["n"] == 3
    assert row["bias_C"] == pytest.approx(2 / 3)
    assert row["mae_C"] == pytest.approx(2 / 3)
    assert row["rmse_C"] == pytest.approx(np.sqrt(2 / 3))
    assert -1.0 <= row["r"] <= 1.0


def test_pcoe_power_consistency_counts_large_absolute_and_relative_errors():
    row = _power_consistency_row(
        "all",
        actual=np.array([100.0, 200.0, 300.0]),
        derived=np.array([111.0, 198.0, 312.0]),
    )

    assert row["bias_derived_minus_Pdc_W"] == pytest.approx(7.0)
    assert row["mae_W"] == pytest.approx(25 / 3)
    assert row["rmse_W"] == pytest.approx(np.sqrt(269 / 3))
    assert row["abs_error_gt_10W"] == 2
    assert row["relative_error_gt_1pct"] == 2


def test_reunion_metrics_drop_nonfinite_pairs_and_report_constant_correlation():
    row = reunion_metric(
        "thermal",
        "model",
        actual=np.array([1.0, np.nan, 3.0]),
        predicted=np.array([2.0, 100.0, 2.0]),
        unit="C",
    )

    assert row["n"] == 2
    assert row["bias_C"] == 0.0
    assert row["mae_C"] == 1.0
    assert row["rmse_C"] == 1.0
    assert row["r"] is None


def test_orientation_metric_reports_a_perfect_fit_and_missing_second_coefficient():
    row = orientation_metric(
        "PV-1",
        "common_radiation",
        "radiation",
        coefficients=np.array([2.0]),
        actual=np.array([1.0, 2.0, 4.0]),
        predicted=np.array([1.0, 2.0, 4.0]),
    )

    assert row["bias_W"] == 0.0
    assert row["rmse_W"] == 0.0
    assert row["r"] == pytest.approx(1.0)
    assert row["r2"] == pytest.approx(1.0)
    assert np.isnan(row["coefficient_2"])


def test_hkust_metric_handles_empty_and_constant_populations():
    assert _finite_metric(np.array([]), np.array([])) == {
        "n": 0,
        "bias_W": None,
        "mae_W": None,
        "rmse_W": None,
        "r": None,
        "r2": None,
    }

    constant = _finite_metric(np.array([2.0, 2.0]), np.array([3.0, 3.0]))
    assert constant["bias_W"] == 1.0
    assert constant["r"] is None
    assert constant["r2"] is None
