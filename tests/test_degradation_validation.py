"""Focused tests for BLAST validity warning collection."""

from __future__ import annotations

import warnings

import numpy as np

from breos.degradation.validation import (
    AGING_HORIZON_CATEGORY,
    EXPERIMENTAL_RANGE_CATEGORY,
    BlastExperimentalRangeWarning,
    BlastWarningCollector,
)


def test_unsourced_aging_horizon_does_not_emit_or_record_warning():
    collector = BlastWarningCollector("lfp_gr_250ah_prismatic")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        collector.check_aging_horizon(1_000_000.0)

    assert caught == []
    assert collector.records(AGING_HORIZON_CATEGORY) == []


def test_restored_collector_deduplicates_and_returns_defensive_copies():
    collector = BlastWarningCollector("lfp_gr_250ah_prismatic")
    t_secs = np.array([0.0, 86400.0])
    soc = np.array([0.5, 0.5])
    temperature_c = np.array([55.0, 55.0])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        collector.check_experimental_range(t_secs, soc, temperature_c)
        restored = BlastWarningCollector.from_snapshot(
            "lfp_gr_250ah_prismatic",
            collector.records(),
        )
        restored.check_experimental_range(t_secs, soc, temperature_c)

    copied_records = restored.records(EXPERIMENTAL_RANGE_CATEGORY)
    copied_records[0]["observed"][0] = -999.0

    assert len([item for item in caught if item.category is BlastExperimentalRangeWarning]) == 1
    assert restored.records(EXPERIMENTAL_RANGE_CATEGORY)[0]["observed"] == [55.0, 55.0]
