"""Static coverage for Article 1 contextual source-data tooling."""

import tomllib

import pytest

from tools.reproduce_article1_context import DEFAULT_CONFIG, _inclusive_values, _pv_module_provenance


def test_article1_context_grid_includes_both_bounds():
    assert _inclusive_values(10.0, 20.0, 5.0).tolist() == [10.0, 15.0, 20.0]


def test_article1_context_grid_rejects_non_positive_step():
    with pytest.raises(ValueError, match="positive"):
        _inclusive_values(10.0, 20.0, 0.0)


def test_article1_context_records_resolved_pv_module():
    config = tomllib.loads(DEFAULT_CONFIG.read_text())

    module = _pv_module_provenance(config)

    assert module["parameters"]["T_Pmax_pct"] == -0.34
    assert module["width_m"] == 1.134
    assert module["length_m"] == 2.278
