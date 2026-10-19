"""Shared execution-backend parametrization and bit-for-bit result comparison."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

# The Python reference always runs; the compiled backend only where numba is installed.
BACKENDS = [
    "python",
    pytest.param(
        "numba", marks=pytest.mark.skipif(not importlib.util.find_spec("numba"), reason="numba not installed")
    ),
]


def assert_frames_bit_equal(left: pd.DataFrame, right: pd.DataFrame, context: str = "") -> None:
    """Assert every result column but ``Datetime`` is bit-identical, naming the first differing step."""
    for column in left.columns:
        if column == "Datetime":
            continue
        a, b = left[column].to_numpy(), right[column].to_numpy()
        assert np.array_equal(a, b), f"{context}column {column} differs at step {int(np.flatnonzero(a != b)[0])}"
