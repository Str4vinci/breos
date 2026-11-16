"""Rainflow cycle counting after ASTM E1049-85 (2011), section 5.4.4.

:class:`RainflowCounter` is BREOS's one implementation of the counting rules.
It takes samples in pieces and keeps the open turning points between them, so
the energy balance can count day by day and carry the count across periods
(see :class:`breos.degradation.protocol._NativeRainflowCounter`). The
whole-series :func:`extract_cycles` feeds it one series; the battery
degradation helpers and the vendored BLAST models count with it. The compiled
day scan in :mod:`breos.degradation._rainflow_kernel` repeats the rules for the
numba backend, and the tests hold it to this module.

The output reproduces ``rainflow.extract_cycles`` from the `rainflow` package
(3.2.0) value for value, including its conventions where the standard leaves
a choice: a plateau's reversal is placed at the plateau's last sample, the
first and last samples are always reversals, and a series of fewer than three
samples has no cycles. Reversals are found with array operations, so only the
turning points, not every sample, go through the Python counting loop.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator

import numpy as np
from numpy.typing import ArrayLike

#: One counted cycle: ``(range, mean, count, start_index, end_index)``, where
#: ``count`` is 1.0 for a full cycle and 0.5 for a half cycle.
Cycle = tuple[float, float, float, int, int]

#: A turning point: ``(value, sample_index)``.
Point = tuple[float, int]

#: A counted cycle as the counter emits it: its two bounding points, in
#: series order, and 1.0 for a full cycle or 0.5 for a half cycle.
RawCycle = tuple[Point, Point, float]


def _scan(scan: np.ndarray, last_delta: float | None) -> tuple[np.ndarray, int, float] | None:
    """Find the reversals among consecutive samples.

    ``last_delta`` is the step into ``scan[0]``'s plateau, None if there is
    none. A sample equal to the one before it extends a plateau, which is
    labelled at its last sample, and a reversal is a sign change of
    consecutive non-flat steps. Returns None if no two samples differ, else
    the positions of the reversals in ``scan``, the position of the last
    sample before the last move, and that move's step.
    """
    moved = np.flatnonzero(scan[1:] != scan[:-1])
    if not moved.size:
        return None
    # Python floats never raise; numpy may, under the caller's error settings.
    with np.errstate(all="ignore"):
        deltas = scan[moved + 1] - scan[moved]
        previous_deltas = np.empty_like(deltas)
        previous_deltas[0] = np.nan if last_delta is None else last_delta
        previous_deltas[1:] = deltas[:-1]
        turns = moved[previous_deltas * deltas < 0]
    return turns, int(moved[-1]), float(deltas[-1])


class RainflowCounter:
    """Incremental rainflow counter over a series fed in pieces.

    :meth:`extend` takes the next samples and returns the cycles they close;
    :meth:`finish` closes the series and returns the residue as half cycles.
    Sample indices count from 0 across all pieces. The first piece needs at
    least two samples. Ranges are compared on ``value * scale``; a scale of
    1.0 compares the values themselves.

    The state is the stack of open turning points (``points``) and the
    scanner's position: ``current``, the newest sample; ``previous``, the last
    sample before the newest move; ``last_delta``, the step into
    ``current``'s plateau; ``moved``, whether any two samples so far differ;
    and ``next_index``. Callers read and set these to carry a count.
    """

    __slots__ = ("scale", "points", "current", "previous", "last_delta", "moved", "next_index")

    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale
        self.points: deque[Point] = deque()
        self.current: Point | None = None
        self.previous: Point | None = None
        self.last_delta: float | None = None
        self.moved = False
        self.next_index = 0

    def reset(self) -> None:
        self.points.clear()
        self.current = None
        self.previous = None
        self.last_delta = None
        self.moved = False
        self.next_index = 0

    def extend(self, values: np.ndarray) -> list[RawCycle]:
        """Feed the next samples, a 1-D array, and return the cycles they close."""
        values = np.asarray(values, dtype=float)
        if self.current is None:
            if values.size < 2:
                raise ValueError("the first samples given to a rainflow counter must number at least two")
            # The first sample is always a reversal. The step from it to the
            # second is taken on Python floats, which never raise.
            first = (float(values[0]), 0)
            self.points.append(first)
            self.previous = first
            self.current = (float(values[1]), 1)
            self.last_delta = self.current[0] - first[0]
            self.moved = self.last_delta != 0.0
            self.next_index = 2
            scan = values[1:]
        else:
            if values.size == 0:
                return []
            scan = np.empty(values.size + 1)
            scan[0] = self.current[0]
            scan[1:] = values

        n_new = scan.size - 1
        if n_new == 0:
            return []
        # scan[0] is the current sample; scan[k] for k >= 1 is sample offset + k.
        offset = self.next_index - 1
        cycles: list[RawCycle] = []
        found = _scan(scan, self.last_delta)
        if found is not None:
            turns, last_still, self.last_delta = found
            if turns.size:
                cycles = self._push(zip(scan[turns].tolist(), (turns + offset).tolist(), strict=True))
            self.previous = (float(scan[last_still]), offset + last_still)
            self.moved = True
        self.current = (float(scan[n_new]), offset + n_new)
        self.next_index = offset + n_new + 1
        return cycles

    def finish(self) -> list[RawCycle]:
        """Close the series: the newest sample is a reversal, and the residue's ranges are half cycles."""
        if self.current is None:
            return []
        cycles = self._push((self.current,))
        points = self.points
        while len(points) > 1:
            cycles.append((points[0], points[1], 0.5))
            points.popleft()
        self.reset()
        return cycles

    def _push(self, reversals: Iterable[Point]) -> list[RawCycle]:
        """Add reversals to the stack and count the cycles they close (ASTM E1049-85, 5.4.4)."""
        points = self.points
        scale = self.scale
        cycles: list[RawCycle] = []
        for point in reversals:
            points.append(point)
            while len(points) >= 3:
                x1 = points[-3][0] * scale
                x2 = points[-2][0] * scale
                x3 = points[-1][0] * scale
                if abs(x3 - x2) < abs(x2 - x1):
                    break
                if len(points) == 3:
                    # The range holds the starting point: half a cycle, and
                    # the start is dropped.
                    cycles.append((points[0], points[1], 0.5))
                    points.popleft()
                else:
                    # A full cycle: drop its peak and valley, keep the newest point.
                    cycles.append((points[-3], points[-2], 1.0))
                    last = points.pop()
                    points.pop()
                    points.pop()
                    points.append(last)
        return cycles


def _as_series(series: ArrayLike) -> np.ndarray:
    values = np.asarray(series, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"rainflow series must be one-dimensional, got shape {values.shape}")
    return values


def reversals(series: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Return the indices and values of the reversal points of a 1-D series.

    A reversal is a sample where the first difference changes sign; a run of
    equal samples counts as one point, labelled by its last sample. The first
    and last samples are reversals whenever the series has at least three
    samples. As in ``rainflow.reversals``, a two-sample series yields only its
    first sample and a shorter one none, so neither has cycles.
    """
    values = _as_series(series)
    n = values.size
    if n < 3:
        first = 1 if n == 2 else 0
        return np.zeros(first, dtype=np.intp), values[:first].copy()
    # The counter's own scan, from the second sample, as RainflowCounter.extend starts.
    found = _scan(values[1:], float(values[1]) - float(values[0]))
    turns = found[0] + 1 if found is not None else np.zeros(0, dtype=np.intp)
    indices = np.concatenate(([0], turns, [n - 1])).astype(np.intp)
    return indices, values[indices]


def extract_cycles(series: ArrayLike) -> Iterator[Cycle]:
    """Yield the rainflow cycles of a 1-D series in the order they close.

    Each item is ``(range, mean, count, start_index, end_index)``: the cycle's
    range and mean in the series' unit, 1.0 for a full cycle or 0.5 for a
    half cycle, and the indices of the two reversals that bound it. The
    residue left when the series ends is emitted as half cycles.
    """
    values = _as_series(series)
    if values.size < 3:
        return
    counter = RainflowCounter()
    for cycles in (counter.extend(values), counter.finish()):
        for (x_start, i_start), (x_end, i_end), count in cycles:
            yield abs(x_start - x_end), 0.5 * (x_start + x_end), count, i_start, i_end
