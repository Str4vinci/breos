"""Front metrics, ported verbatim from phd/dev/pvbat/a3_validation.py (the non-TOU A3 study)."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd




def uniform_levels(lower: float, upper: float, step: float) -> np.ndarray:
    """Return an inclusive uniform grid, rejecting incompatible bounds."""
    if step <= 0 or upper < lower:
        raise ValueError("uniform grid requires step > 0 and upper >= lower")
    count = (upper - lower) / step
    if not np.isclose(count, round(count), atol=1e-10):
        raise ValueError("grid step must land exactly on the upper bound")
    return lower + np.arange(int(round(count)) + 1, dtype=float) * step


def levels_from_spec(spec: Mapping[str, Mapping[str, float]]) -> Dict[str, np.ndarray]:
    levels = {}
    for name in DESIGN_COLUMNS:
        item = spec[name]
        levels[name] = uniform_levels(
            float(item["min"]), float(item["max"]), float(item["step"])
        )
    return levels


def cartesian_designs(levels: Mapping[str, Sequence[float]]) -> np.ndarray:
    """Generate the deterministic lexicographic Cartesian design matrix."""
    return np.asarray(list(product(*(levels[name] for name in DESIGN_COLUMNS))), dtype=float)


def pareto_mask(objectives: np.ndarray) -> np.ndarray:
    """Return non-dominated mask for finite minimization objectives."""
    f = np.asarray(objectives, dtype=float)
    if f.ndim != 2:
        raise ValueError("objectives must be a two-dimensional array")
    finite = np.all(np.isfinite(f), axis=1)
    candidates = np.flatnonzero(finite)
    mask = np.zeros(len(f), dtype=bool)
    if f.shape[1] == 2:
        # Sweep equal-x groups. A group's minimum y is non-dominated only when
        # it strictly improves on every lower-x group; duplicate minima remain
        # non-dominated under the strict Pareto definition.
        order = candidates[np.lexsort((f[candidates, 1], f[candidates, 0]))]
        best_y = np.inf
        pos = 0
        while pos < len(order):
            end = pos + 1
            x = f[order[pos], 0]
            while end < len(order) and f[order[end], 0] == x:
                end += 1
            group = order[pos:end]
            group_min = f[group[0], 1]
            if group_min < best_y:
                mask[group[f[group, 1] == group_min]] = True
                best_y = group_min
            pos = end
        return mask
    # Safe generic fallback for uncommon higher-dimensional validation use.
    mask[candidates] = True
    for i in candidates:
        if np.any(np.all(f[candidates] <= f[i], axis=1) & np.any(f[candidates] < f[i], axis=1)):
            mask[i] = False
    return mask


@dataclass(frozen=True)
class Normalization:
    ideal: np.ndarray
    nadir: np.ndarray
    reference: np.ndarray

    def transform(self, objectives: np.ndarray) -> np.ndarray:
        scale = np.where(self.nadir > self.ideal, self.nadir - self.ideal, 1.0)
        return (np.asarray(objectives, dtype=float) - self.ideal) / scale


def frozen_normalization(exact_front: np.ndarray, margin: float = 0.10) -> Normalization:
    f = np.asarray(exact_front, dtype=float)
    if f.ndim != 2 or len(f) == 0:
        raise ValueError("exact_front must be non-empty")
    ideal = np.min(f, axis=0)
    nadir = np.max(f, axis=0)
    return Normalization(ideal=ideal, nadir=nadir, reference=np.full(f.shape[1], 1.0 + margin))


def hypervolume_2d(front: np.ndarray, reference: Sequence[float]) -> float:
    """Exact dominated hypervolume for a two-objective minimization front."""
    f = np.asarray(front, dtype=float)
    ref = np.asarray(reference, dtype=float)
    if f.size == 0:
        return 0.0
    if f.shape[1] != 2 or ref.shape != (2,):
        raise ValueError("hypervolume_2d requires two objectives")
    f = f[np.all(f < ref, axis=1)]
    f = np.unique(f[pareto_mask(f)], axis=0)
    f = f[np.argsort(f[:, 0], kind="mergesort")]
    area = 0.0
    previous_y = ref[1]
    for x, y in f:
        if y < previous_y:
            area += max(0.0, ref[0] - x) * (previous_y - y)
            previous_y = y
    return float(area)


def igd_plus(reference_front: np.ndarray, approximation: np.ndarray) -> float:
    """IGD+ for minimization in a shared normalized objective space."""
    ref = np.asarray(reference_front, dtype=float)
    approx = np.asarray(approximation, dtype=float)
    if len(ref) == 0 or len(approx) == 0:
        return float("inf")
    distances = []
    for point in ref:
        distances.append(np.min(np.linalg.norm(np.maximum(approx - point, 0.0), axis=1)))
    return float(np.mean(distances))


def chord_knee(front_normalized: np.ndarray) -> Tuple[int, float]:
    """Pre-registered knee: maximum distance to normalized extremes' chord."""
    f = np.asarray(front_normalized, dtype=float)
    if len(f) == 0:
        raise ValueError("front cannot be empty")
    if len(f) == 1:
        return 0, 0.0
    a = f[np.argmin(f[:, 0])]
    b = f[np.argmin(f[:, 1])]
    direction = b - a
    denominator = np.linalg.norm(direction)
    if denominator <= 1e-15:
        return 0, 0.0
    offsets = f - a
    distances = np.abs(direction[0] * offsets[:, 1] - direction[1] * offsets[:, 0]) / denominator
    idx = int(np.argmax(distances))
    return idx, float(distances[idx])


def recovery_metrics(
    exact_front: np.ndarray,
    approximation: np.ndarray,
    *, margin: float = 0.10,
    tolerance: float = 0.01,
) -> Dict[str, object]:
    """Compare an approximate front against the exact front."""
    exact = np.unique(np.asarray(exact_front, dtype=float), axis=0)
    approx = np.unique(np.asarray(approximation, dtype=float), axis=0)
    exact = exact[pareto_mask(exact)]
    approx = approx[pareto_mask(approx)] if len(approx) else approx
    norm = frozen_normalization(exact, margin)
    en = norm.transform(exact)
    an = norm.transform(approx) if len(approx) else np.empty((0, 2))
    exact_hv = hypervolume_2d(en, norm.reference)
    approx_hv = hypervolume_2d(an, norm.reference)
    nearest = np.full(len(en), np.inf)
    if len(an):
        nearest = np.min(np.linalg.norm(en[:, None, :] - an[None, :, :], axis=2), axis=1)
    recovered = nearest <= tolerance
    knee_idx, knee_distance = chord_knee(en)
    extreme_indices = [int(np.argmin(en[:, j])) for j in range(2)]
    missed_runs = []
    run = 0
    for value in recovered[np.argsort(en[:, 0])]:
        run = 0 if value else run + 1
        missed_runs.append(run)
    return {
        "normalization_ideal": norm.ideal.tolist(),
        "normalization_nadir": norm.nadir.tolist(),
        "hypervolume_reference_normalized": norm.reference.tolist(),
        "exact_hypervolume": exact_hv,
        "approx_hypervolume": approx_hv,
        "hypervolume_ratio": approx_hv / exact_hv if exact_hv > 0 else 1.0,
        "igd_plus": igd_plus(en, an),
        "pareto_recovery_fraction": float(np.mean(recovered)),
        "pareto_recovered_count": int(np.sum(recovered)),
        "pareto_exact_count": int(len(exact)),
        "extrema_recovered": [bool(recovered[i]) for i in extreme_indices],
        "knee_exact_index": knee_idx,
        "knee_distance_to_chord": knee_distance,
        "knee_recovered": bool(recovered[knee_idx]),
        "maximum_regret_normalized": float(np.max(nearest)) if len(nearest) else 0.0,
        "largest_consecutive_missed_section": int(max(missed_runs, default=0)),
        "tolerance_normalized": tolerance,
    }


