"""One day of the native rainflow count on arrays, for the compiled backend.

:class:`breos.degradation.protocol._NativeRainflowCounter` keeps its residue
on a deque of tuples and finds reversals with numpy; that suits the Python
backend. With ``execution_backend = "numba"``,
:class:`~breos.degradation.protocol._CompiledRainflowCounter` holds the same
state in the arrays below and runs :func:`rainflow_day` compiled by
:mod:`breos._numba_dispatch_kernels`, once per day.

Every operation repeats the counter's arithmetic in the same order on float64
values: the same subtraction for a delta, the same product for the sign test,
the same percent conversions for a range and the same cycle formulas. Under
``fastmath=False`` the compiled function therefore returns the same bits as
the counter. Python floats and compiled code never raise on IEEE overflow,
underflow or NaN, so neither does this.

State layout (owned by the counter, mutated in place):

``points_f[k] = (soc, elapsed_seconds)`` and ``points_i[k] = sample index``
    The residue, oldest first; ``istate[_N_POINTS]`` entries are in use.
``fstate``, ``istate``
    The scanner's current and previous points, the last non-flat delta, and
    the flags that say which of them exist (see the ``_F_*`` and ``_I_*``
    positions).

Emitted cycles go to ``out_f[n] = (doc, mean_soc, count, mean_c_rate)`` and
``out_i[n] = (start_idx, end_idx)``; the function returns how many. The
caller sizes every array for ``istate[_N_POINTS] + len(soc) + 1`` rows, the
most a day can push or emit.
"""

from __future__ import annotations

import numpy as np

# fstate positions
_F_CURRENT_SOC = 0
_F_CURRENT_TIME = 1
_F_PREVIOUS_SOC = 2
_F_PREVIOUS_TIME = 3
_F_LAST_DELTA = 4
FSTATE_SIZE = 5

# istate positions
_I_N_POINTS = 0
_I_HAS_CURRENT = 1
_I_CURRENT_INDEX = 2
_I_HAS_PREVIOUS = 3
_I_PREVIOUS_INDEX = 4
_I_HAS_LAST_DELTA = 5
_I_HAS_NONFLAT_PAIR = 6
_I_NEXT_INDEX = 7
ISTATE_SIZE = 8


def _emit_cycle(
    points_f: np.ndarray,
    points_i: np.ndarray,
    first: int,
    second: int,
    count: float,
    out_f: np.ndarray,
    out_i: np.ndarray,
    n_cycles: int,
    min_doc_fraction: float,
) -> int:
    """Write the cycle between two residue points if it is deep enough; return the new cycle count.

    The formulas are those of ``_NativeRainflowCounter._format_cycle``.
    """
    soc1 = points_f[first, 0]
    soc2 = points_f[second, 0]
    doc = abs(soc1 * 100.0 - soc2 * 100.0) / 100.0
    if not doc >= min_doc_fraction:
        return n_cycles
    duration_seconds = abs(points_f[second, 1] - points_f[first, 1])
    duration_hours = duration_seconds / 3600.0
    out_f[n_cycles, 0] = doc
    out_f[n_cycles, 1] = 0.5 * (soc1 * 100.0 + soc2 * 100.0) / 100.0
    out_f[n_cycles, 2] = count
    out_f[n_cycles, 3] = doc / duration_hours if duration_hours > 0 else 0.0
    out_i[n_cycles, 0] = points_i[first]
    out_i[n_cycles, 1] = points_i[second]
    return n_cycles + 1


def _push_reversal(
    points_f: np.ndarray,
    points_i: np.ndarray,
    n_points: int,
    soc: float,
    elapsed_seconds: float,
    index: int,
    out_f: np.ndarray,
    out_i: np.ndarray,
    n_cycles: int,
    min_doc_fraction: float,
) -> tuple[int, int]:
    """``_NativeRainflowCounter._push_reversal`` on the residue arrays; return ``(n_points, n_cycles)``."""
    points_f[n_points, 0] = soc
    points_f[n_points, 1] = elapsed_seconds
    points_i[n_points] = index
    n_points += 1
    while n_points >= 3:
        x1 = points_f[n_points - 3, 0] * 100.0
        x2 = points_f[n_points - 2, 0] * 100.0
        x3 = points_f[n_points - 1, 0] * 100.0
        x_range = abs(x3 - x2)
        y_range = abs(x2 - x1)
        if x_range < y_range:
            break
        if n_points == 3:
            n_cycles = _emit_cycle(points_f, points_i, 0, 1, 0.5, out_f, out_i, n_cycles, min_doc_fraction)
            # popleft
            for k in range(n_points - 1):
                points_f[k, 0] = points_f[k + 1, 0]
                points_f[k, 1] = points_f[k + 1, 1]
                points_i[k] = points_i[k + 1]
            n_points -= 1
        else:
            n_cycles = _emit_cycle(
                points_f, points_i, n_points - 3, n_points - 2, 1.0, out_f, out_i, n_cycles, min_doc_fraction
            )
            # Drop the two points that closed the cycle; the newest point stays.
            points_f[n_points - 3, 0] = points_f[n_points - 1, 0]
            points_f[n_points - 3, 1] = points_f[n_points - 1, 1]
            points_i[n_points - 3] = points_i[n_points - 1]
            n_points -= 2
    return n_points, n_cycles


def rainflow_day(
    points_f: np.ndarray,
    points_i: np.ndarray,
    fstate: np.ndarray,
    istate: np.ndarray,
    soc: np.ndarray,
    step_seconds: float,
    start_soc: float,
    out_f: np.ndarray,
    out_i: np.ndarray,
    min_doc_fraction: float,
) -> int:
    """Advance the count over one non-empty day of SOC samples; return the number of cycles emitted.

    Mirrors the scan of ``_NativeRainflowCounter.step`` (without its
    validation and without closing the residue, which the counter does).
    """
    n_points = istate[_I_N_POINTS]
    n_cycles = 0
    start = 0
    if istate[_I_HAS_CURRENT] == 0:
        # The anchor is the boundary SOC before the first sample; it joins the
        # residue directly, without a cycle check.
        points_f[n_points, 0] = start_soc
        points_f[n_points, 1] = 0.0
        points_i[n_points] = 0
        n_points += 1
        fstate[_F_PREVIOUS_SOC] = start_soc
        fstate[_F_PREVIOUS_TIME] = 0.0
        istate[_I_PREVIOUS_INDEX] = 0
        istate[_I_HAS_PREVIOUS] = 1
        first = soc[0]
        fstate[_F_CURRENT_SOC] = first
        fstate[_F_CURRENT_TIME] = step_seconds
        istate[_I_CURRENT_INDEX] = 1
        istate[_I_HAS_CURRENT] = 1
        last_delta = first - start_soc
        fstate[_F_LAST_DELTA] = last_delta
        istate[_I_HAS_LAST_DELTA] = 1
        istate[_I_HAS_NONFLAT_PAIR] = 1 if last_delta != 0.0 else 0
        istate[_I_NEXT_INDEX] = 2
        start = 1

    n_samples = soc.shape[0] - start
    if n_samples > 0:
        first_index = istate[_I_NEXT_INDEX]
        current_soc = fstate[_F_CURRENT_SOC]
        current_time = fstate[_F_CURRENT_TIME]
        current_index = istate[_I_CURRENT_INDEX]
        has_last_delta = istate[_I_HAS_LAST_DELTA] != 0
        last_delta = fstate[_F_LAST_DELTA]
        last_moved = -1
        # Position p holds values[p] (p = 0 is the current point); values[p + 1]
        # is soc[start + p]. A sample equal to the one before it extends a
        # plateau, and a reversal is a sign change of consecutive non-flat deltas.
        before = current_soc
        for p in range(n_samples):
            value = soc[start + p]
            if value != before:
                delta = value - before
                if has_last_delta and last_delta * delta < 0:
                    if p == 0:
                        point_soc = current_soc
                        point_time = current_time
                        point_index = current_index
                    else:
                        point_index = first_index + p - 1
                        point_soc = before
                        point_time = point_index * step_seconds
                    n_points, n_cycles = _push_reversal(
                        points_f,
                        points_i,
                        n_points,
                        point_soc,
                        point_time,
                        point_index,
                        out_f,
                        out_i,
                        n_cycles,
                        min_doc_fraction,
                    )
                last_delta = delta
                has_last_delta = True
                last_moved = p
            before = value
        if last_moved >= 0:
            if last_moved == 0:
                fstate[_F_PREVIOUS_SOC] = current_soc
                fstate[_F_PREVIOUS_TIME] = current_time
                istate[_I_PREVIOUS_INDEX] = current_index
            else:
                previous_index = first_index + last_moved - 1
                fstate[_F_PREVIOUS_SOC] = soc[start + last_moved - 1]
                fstate[_F_PREVIOUS_TIME] = previous_index * step_seconds
                istate[_I_PREVIOUS_INDEX] = previous_index
            istate[_I_HAS_PREVIOUS] = 1
            fstate[_F_LAST_DELTA] = last_delta
            istate[_I_HAS_LAST_DELTA] = 1
            istate[_I_HAS_NONFLAT_PAIR] = 1
        last_index = first_index + n_samples - 1
        fstate[_F_CURRENT_SOC] = soc[start + n_samples - 1]
        fstate[_F_CURRENT_TIME] = last_index * step_seconds
        istate[_I_CURRENT_INDEX] = last_index
        istate[_I_NEXT_INDEX] = first_index + n_samples

    istate[_I_N_POINTS] = n_points
    return n_cycles
