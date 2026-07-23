"""BLAST degradation engine adapter.

This module is intentionally integration-neutral: it wraps vendored BLAST model
instances for incremental stepping, but it does not alter BREOS dispatch,
configuration, or replacement behavior.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from breos.degradation.blast import models
from breos.degradation.profiles import BATTERY_MODEL_REGISTRY, BLAST_STATE_SCHEMA_VERSION, CORE_BLAST_MODEL_KEYS
from breos.degradation.validation import (
    BlastAgingHorizonWarning,
    BlastExperimentalRangeWarning,
    BlastWarningCollector,
)

BLAST_MODEL_CLASSES = {key: getattr(models, profile.class_name) for key, profile in BATTERY_MODEL_REGISTRY.items()}

# Backwards-compatible internal name used by the replayed Phase 1 tests.
P1_BLAST_MODEL_KEYS = CORE_BLAST_MODEL_KEYS


class BlastNumericalError(RuntimeError):
    """A BLAST model produced a non-finite state; results are unusable."""


def _as_1d_float_array(name: str, values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def normalize_step_inputs(t_secs: Any, soc: Any, temperature_c: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and normalize one BLAST update chunk."""

    t_secs_array = _as_1d_float_array("t_secs", t_secs)
    soc_array = _as_1d_float_array("soc", soc)
    temperature_array = _as_1d_float_array("temperature_c", temperature_c)

    if temperature_array.size == 1 and t_secs_array.size > 1:
        temperature_array = np.full(t_secs_array.shape, temperature_array.item())

    if not (t_secs_array.size == soc_array.size == temperature_array.size):
        raise ValueError("t_secs, soc, and temperature_c must have the same length")
    if t_secs_array.size < 2:
        raise ValueError("BLAST update chunks need at least two time points")
    if not np.all(np.diff(t_secs_array) > 0):
        raise ValueError("t_secs must be strictly increasing")
    if np.any((soc_array < -1e-12) | (soc_array > 1 + 1e-12)):
        raise ValueError("soc must be normalized to the range [0, 1]")

    return t_secs_array, soc_array, temperature_array


def build_endpoint_day(
    step_seconds: float,
    soc_samples: Any,
    temperature_c_samples: Any,
    start_soc: float,
    start_temperature_c: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a full-day endpoint grid from buffered post-step samples.

    BREOS buffers post-step samples. BLAST derives elapsed time and cycle
    throughput from first differences, so the prior anchor must be prepended.
    """

    if not np.isfinite(step_seconds) or step_seconds <= 0:
        raise ValueError("step_seconds must be a positive finite value")

    soc_post = _as_1d_float_array("soc_samples", soc_samples)
    temperature_post = _as_1d_float_array("temperature_c_samples", temperature_c_samples)
    if temperature_post.size == 1 and soc_post.size > 1:
        temperature_post = np.full(soc_post.shape, temperature_post.item())
    if temperature_post.size != soc_post.size:
        raise ValueError("soc_samples and temperature_c_samples must have the same length")
    if soc_post.size == 0:
        raise ValueError("daily endpoint construction needs at least one sample")

    t_secs = np.arange(soc_post.size + 1, dtype=float) * float(step_seconds)
    soc = np.concatenate(([float(start_soc)], soc_post))
    temperature_c = np.concatenate(([float(start_temperature_c)], temperature_post))
    return normalize_step_inputs(t_secs, soc, temperature_c)


class BlastEngine:
    """Thin wrapper around one vendored BLAST model instance."""

    _ARRAY_GROUPS = ("states", "outputs", "stressors", "rates")

    def __init__(self, blast_model_key: str, **model_kwargs: Any):
        if blast_model_key not in BLAST_MODEL_CLASSES:
            available = ", ".join(BLAST_MODEL_CLASSES)
            raise KeyError(f"Unknown BLAST model key {blast_model_key!r}. Available: {available}")
        self.blast_model_key = blast_model_key
        self._model_kwargs = dict(model_kwargs)
        self.model = self._new_model()
        self._warning_collector = BlastWarningCollector(blast_model_key)

    def _new_model(self):
        return BLAST_MODEL_CLASSES[self.blast_model_key](**self._model_kwargs)

    def step(self, t_secs_day: Any, soc_abs_day: Any, t_cell_day_c: Any) -> float:
        """Update by one time chunk and return current SoH fraction."""

        t_secs, soc, temperature_c = normalize_step_inputs(t_secs_day, soc_abs_day, t_cell_day_c)
        self._warning_collector.check_experimental_range(t_secs, soc, temperature_c)
        self.model.update_battery_state(t_secs, soc, temperature_c)
        self._warning_collector.check_aging_horizon(float(self.model.stressors["t_days"][-1]))
        self._check_finite_update()
        return self.soh()

    def _check_finite_update(self) -> None:
        """Raise ``BlastNumericalError`` if the update produced a non-finite value.

        Inspects the newest entry of every ``states`` and ``outputs`` array, not
        only capacity ``q``: a state-domain overshoot can corrupt an internal
        loss term one or more steps before it surfaces in ``q``. Only the most
        recent entry is read, which the update just applied always populates, so
        the intentional initial NaN sentinels in ``stressors`` and ``rates`` are
        never inspected.
        """

        offending = [
            f"{group_name}.{field}"
            for group_name in ("states", "outputs")
            for field, values in getattr(self.model, group_name).items()
            if not np.isfinite(values[-1])
        ]
        if offending:
            elapsed_days = float(self.model.stressors["t_days"][-1])
            raise BlastNumericalError(
                f"BLAST model {self.blast_model_key!r} produced non-finite values in "
                f"{', '.join(offending)} after {elapsed_days:.0f} simulated days. This "
                "indicates a state-update instability; the engine state is corrupt and "
                "the simulation cannot continue."
            )

    def warning_records(self, category: str | None = None) -> list[dict[str, Any]]:
        """Return JSON-safe, deduplicated warnings accumulated by this run."""
        return self._warning_collector.records(category)

    def soh(self) -> float:
        """Return current BLAST capacity fraction."""

        return float(self.model.outputs["q"][-1])

    def reset(self) -> None:
        """Reset to a fresh beginning-of-life model instance."""

        self.model = self._new_model()

    def state_snapshot(self, *, serializable: bool = True) -> dict[str, Any]:
        """Copy the model state required for cross-year threading."""

        snapshot: dict[str, Any] = {
            "schema_version": BLAST_STATE_SCHEMA_VERSION,
            "blast_model_key": self.blast_model_key,
            "model_kwargs": deepcopy(self._model_kwargs),
            "warnings": self.warning_records(),
        }
        for group_name in self._ARRAY_GROUPS:
            group = getattr(self.model, group_name)
            snapshot[group_name] = {
                key: (value.tolist() if serializable else value.copy()) for key, value in group.items()
            }
        return snapshot

    @classmethod
    def from_snapshot(cls, blast_model_key: str, snapshot: dict[str, Any]) -> "BlastEngine":
        """Rebuild an engine from a previous ``state_snapshot`` result."""

        snapshot_key = snapshot.get("blast_model_key")
        if snapshot_key is not None and snapshot_key != blast_model_key:
            raise ValueError(f"Snapshot blast_model_key {snapshot_key!r} does not match {blast_model_key!r}")
        schema_version = snapshot.get("schema_version", BLAST_STATE_SCHEMA_VERSION)
        if schema_version != BLAST_STATE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported BLAST state schema {schema_version!r}; expected {BLAST_STATE_SCHEMA_VERSION!r}"
            )

        engine = cls(blast_model_key, **snapshot.get("model_kwargs", {}))
        engine._warning_collector = BlastWarningCollector.from_snapshot(
            blast_model_key,
            snapshot.get("warnings", []),
        )
        for group_name in cls._ARRAY_GROUPS:
            group = snapshot[group_name]
            setattr(
                engine.model,
                group_name,
                {key: np.asarray(value, dtype=float) for key, value in group.items()},
            )
        return engine
