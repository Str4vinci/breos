"""Rainflow cycle counting after ASTM E1049-85 (2011), section 5.4.4.

The battery degradation helpers count SOC cycles over a whole series with
:func:`extract_cycles`. The energy balance counts them day by day with the
incremental counter in :mod:`breos.degradation.protocol`, which gives the same
cycles; the tests hold the two together.

The output reproduces ``rainflow.extract_cycles`` from the `rainflow` package
(3.2.0) value for value, including its conventions where the standard leaves
a choice: a plateau's reversal is placed at the plateau's last sample, the
first and last samples are always reversals, and a series of fewer than three
samples has no cycles. Reversals are found with array operations, so only the
turning points, not every sample, go through the Python counting loop.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator

import numpy as np
from numpy.typing import ArrayLike

#: One counted cycle: ``(range, mean, count, start_index, end_index)``, where
#: ``count`` is 1.0 for a full cycle and 0.5 for a half cycle.
Cycle = tuple[float, float, float, int, int]


def reversals(series: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Return the indices and values of the reversal points of a 1-D series.

    A reversal is a sample where the first difference changes sign; a run of
    equal samples counts as one point, labelled by its last sample. The first
    and last samples are reversals whenever the series has at least three
    samples. As in ``rainflow.reversals``, a two-sample series yields only its
    first sample and a shorter one none, so neither has cycles.
    """
    values = np.asarray(series, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"rainflow series must be one-dimensional, got shape {values.shape}")
    n = values.size
    if n < 3:
        first = 1 if n == 2 else 0
        return np.zeros(first, dtype=np.intp), values[:first].copy()

    # The last index of each run of equal samples, and the run's value.
    run_ends = np.flatnonzero(np.append(values[1:] != values[:-1], True))
    run_values = values[run_ends]
    steps = np.diff(run_values)
    # A run between two others is a reversal when the direction changes
    # across it. The first and last runs are the series' ends, handled below.
    with np.errstate(invalid="ignore"):
        turns = np.flatnonzero(steps[:-1] * steps[1:] < 0) + 1
    indices = np.concatenate(([0], run_ends[turns], [n - 1])).astype(np.intp)
    points = np.concatenate((values[:1], run_values[turns], values[-1:]))
    return indices, points


def extract_cycles(series: ArrayLike) -> Iterator[Cycle]:
    """Yield the rainflow cycles of a 1-D series in the order they close.

    Each item is ``(range, mean, count, start_index, end_index)``: the cycle's
    range and mean in the series' unit, 1.0 for a full cycle or 0.5 for a
    half cycle, and the indices of the two reversals that bound it. The
    residue left when the series ends is emitted as half cycles.
    """
    indices, values = reversals(series)
    points: deque[tuple[int, float]] = deque()

    for point in zip(indices.tolist(), values.tolist(), strict=True):
        points.append(point)
        while len(points) >= 3:
            x1, x2, x3 = points[-3][1], points[-2][1], points[-1][1]
            if abs(x3 - x2) < abs(x2 - x1):
                break
            if len(points) == 3:
                # The range holds the starting point: half a cycle, and the
                # start is dropped.
                yield _cycle(points[0], points[1], 0.5)
                points.popleft()
            else:
                # A full cycle: drop its peak and valley, keep the newest point.
                yield _cycle(points[-3], points[-2], 1.0)
                last = points.pop()
                points.pop()
                points.pop()
                points.append(last)

    while len(points) > 1:
        yield _cycle(points[0], points[1], 0.5)
        points.popleft()


def _cycle(start: tuple[int, float], end: tuple[int, float], count: float) -> Cycle:
    (i_start, x_start), (i_end, x_end) = start, end
    return abs(x_start - x_end), 0.5 * (x_start + x_end), count, i_start, i_end
