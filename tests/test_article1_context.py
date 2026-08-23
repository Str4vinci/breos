"""Static coverage for Article 1 contextual source-data tooling."""

import pytest

from tools.reproduce_article1_context import _inclusive_values


def test_article1_context_grid_includes_both_bounds():
    assert _inclusive_values(10.0, 20.0, 5.0).tolist() == [10.0, 15.0, 20.0]


def test_article1_context_grid_rejects_non_positive_step():
    with pytest.raises(ValueError, match="positive"):
        _inclusive_values(10.0, 20.0, 0.0)
