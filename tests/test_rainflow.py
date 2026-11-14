"""Rainflow counting in breos._rainflow.

The ASTM example pins the counts the standard publishes. The comparison with
the `rainflow` package, a dev-only dependency kept as an independent oracle,
pins the conventions the standard leaves open.
"""

from collections import defaultdict

import numpy as np
import pytest

from breos._rainflow import extract_cycles, reversals
from breos.degradation.native import _detect_cycles_rainflow_arrays

# ASTM E1049-85 (2011), section 5.4.4 and its figure: the load history and the
# range counts the standard's worked example lists.
ASTM_SERIES = [-2, 1, -3, 5, -1, 3, -4, 4, -2]
ASTM_RANGE_COUNTS = {3: 0.5, 4: 1.5, 6: 0.5, 8: 1.0, 9: 0.5}


def test_astm_e1049_worked_example():
    counts: dict[float, float] = defaultdict(float)
    for rng, _mean, count, _start, _end in extract_cycles(ASTM_SERIES):
        counts[rng] += count
    assert dict(counts) == ASTM_RANGE_COUNTS


def test_astm_e1049_cycles_in_closing_order():
    assert list(extract_cycles(ASTM_SERIES)) == [
        (3, -0.5, 0.5, 0, 1),
        (4, -1.0, 0.5, 1, 2),
        (4, 1.0, 1.0, 4, 5),
        (8, 1.0, 0.5, 2, 3),
        (9, 0.5, 0.5, 3, 6),
        (8, 0.0, 0.5, 6, 7),
        (6, 1.0, 0.5, 7, 8),
    ]


def test_plateau_reversal_is_labelled_at_its_last_sample():
    indices, values = reversals([0.5, 0.5, 0.2, 0.2, 0.2, 0.8, 0.8, 0.4, 0.4])
    assert indices.tolist() == [0, 4, 6, 8]
    assert values.tolist() == [0.5, 0.2, 0.8, 0.4]


@pytest.mark.parametrize(
    "series",
    [[], [0.4], [0.4, 0.9], [0.3, 0.3]],
)
def test_series_shorter_than_three_samples_has_no_cycles(series):
    assert list(extract_cycles(series)) == []


def test_constant_series_closes_one_zero_range_half_cycle():
    assert list(extract_cycles([0.5] * 10)) == [(0.0, 0.5, 0.5, 0, 9)]


def test_rejects_multidimensional_input():
    with pytest.raises(ValueError, match="one-dimensional"):
        list(extract_cycles(np.zeros((3, 3))))


def _oracle_series():
    rng = np.random.default_rng(20261104)
    for n in range(7):
        for _ in range(50):
            yield rng.integers(0, 3, n).astype(float)
    for case in range(1500):
        n = int(rng.integers(3, 400))
        kind = case % 5
        if kind == 0:
            yield rng.random(n)
        elif kind == 1:
            yield np.cumsum(rng.normal(size=n))
        elif kind == 2:
            yield rng.integers(0, 5, n).astype(float)  # many ties and plateaus
        elif kind == 3:
            yield np.round(np.clip(np.cumsum(rng.normal(0.0, 0.1, n)) + 0.5, 0.0, 1.0), 2)
        else:
            yield np.repeat(rng.random(n // 4 + 1), rng.integers(1, 6, n // 4 + 1))[:n]


def test_matches_rainflow_package_exactly():
    rainflow = pytest.importorskip("rainflow")
    for series in _oracle_series():
        assert list(extract_cycles(series)) == list(rainflow.extract_cycles(series))
        indices, values = reversals(series)
        expected = list(rainflow.reversals(series))
        assert indices.tolist() == [index for index, _ in expected]
        assert values.tolist() == [value for _, value in expected]


def test_battery_cycle_records_match_rainflow_package():
    rainflow = pytest.importorskip("rainflow")
    rng = np.random.default_rng(7)
    # Twenty 15-minute days of a daily charge/discharge swing with dispatch noise.
    steps = np.arange(20 * 96)
    soc = np.clip(0.5 + 0.4 * np.sin(2 * np.pi * steps / 96) + rng.normal(0.0, 0.03, steps.size), 0.05, 0.95)
    ticks_per_second = 1_000_000.0
    time_ticks = np.arange(soc.size, dtype=np.int64) * int(900 * ticks_per_second)

    observed = _detect_cycles_rainflow_arrays(soc, time_ticks, ticks_per_second)

    expected = []
    for rng_pct, mean_pct, count, i_start, i_end in rainflow.extract_cycles(soc * 100.0):
        doc = rng_pct / 100.0
        if doc < 0.01:
            continue
        hours = (time_ticks[i_end] - time_ticks[i_start]) / ticks_per_second / 3600.0
        expected.append(
            {
                "doc": doc,
                "mean_soc": mean_pct / 100.0,
                "count": count,
                "mean_c_rate": doc / hours if hours > 0 else 0.0,
                "start_idx": i_start,
                "end_idx": i_end,
            }
        )
    assert observed == expected
    assert len(observed) > 100
