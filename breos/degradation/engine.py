"""BLAST degradation engine adapter.

This module is intentionally integration-neutral: it wraps vendored BLAST model
instances for incremental stepping, but it does not alter BREOS dispatch,
configuration, or replacement behavior.
"""

from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Any

import numpy as np

from breos.degradation.blast import models
from breos.degradation.profiles import BATTERY_MODEL_REGISTRY, BLAST_STATE_SCHEMA_VERSION, CORE_BLAST_MODEL_KEYS

BLAST_MODEL_CLASSES = {key: getattr(models, profile.class_name) for key, profile in BATTERY_MODEL_REGISTRY.items()}

# Backwards-compatible internal name used by the replayed Phase 1 tests.
P1_BLAST_MODEL_KEYS = CORE_BLAST_MODEL_KEYS


class BlastNumericalError(RuntimeError):
    """A BLAST model produced a non-finite state; results are unusable."""


class BlastExperimentalRangeWarning(UserWarning):
    """A BLAST input falls outside conditions represented by its test data."""


class BlastAgingHorizonWarning(UserWarning):
    """A BLAST simulation extends beyond a sourced aging-data horizon."""


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
        self._warning_records: dict[str, dict[str, Any]] = {}

    def _new_model(self):
        return BLAST_MODEL_CLASSES[self.blast_model_key](**self._model_kwargs)

    def step(self, t_secs_day: Any, soc_abs_day: Any, t_cell_day_c: Any) -> float:
        """Update by one time chunk and return current SoH fraction."""

        t_secs, soc, temperature_c = normalize_step_inputs(t_secs_day, soc_abs_day, t_cell_day_c)
        self._check_experimental_range(t_secs, soc, temperature_c)
        self.model.update_battery_state(t_secs, soc, temperature_c)
        self._check_aging_horizon()
        soh = self.soh()
        if not np.isfinite(soh):
            elapsed_days = float(self.model.stressors["t_days"][-1])
            raise BlastNumericalError(
                f"BLAST model {self.blast_model_key!r} produced a non-finite SoH "
                f"after {elapsed_days:.0f} simulated days. This indicates a "
                "state-update instability; the engine state is corrupt and the "
                "simulation cannot continue."
            )
        return soh

    def _record_warning(
        self,
        code: str,
        message: str,
        category: str,
        warning_class: type[Warning],
        **details: Any,
    ) -> None:
        if code in self._warning_records:
            return
        self._warning_records[code] = {
            "code": code,
            "message": message,
            "category": category,
            **details,
        }
        warnings.warn(message, warning_class, stacklevel=3)

    def _check_experimental_range(self, t_secs: np.ndarray, soc: np.ndarray, temperature_c: np.ndarray) -> None:
        limits = BATTERY_MODEL_REGISTRY[self.blast_model_key].experimental_range
        checks = [
            (
                "temperature_c",
                float(np.min(temperature_c)),
                float(np.max(temperature_c)),
                limits["cycling_temperature_c"],
            ),
            ("soc", float(np.min(soc)), float(np.max(soc)), limits["soc"]),
        ]
        observed_dod = float(np.ptp(soc))
        if observed_dod > 1e-12:
            checks.append(("dod", observed_dod, observed_dod, limits["dod"]))
        for field, observed_min, observed_max, supported in checks:
            supported_min = float(min(supported))
            supported_max = float(max(supported))
            if observed_min < supported_min - 1e-12 or observed_max > supported_max + 1e-12:
                message = (
                    f"BLAST model {self.blast_model_key!r} received {field} range "
                    f"[{observed_min:.6g}, {observed_max:.6g}] outside experimental range "
                    f"[{supported_min:.6g}, {supported_max:.6g}]."
                )
                self._record_warning(
                    f"experimental_range.{field}",
                    message,
                    "experimental_range",
                    BlastExperimentalRangeWarning,
                    field=field,
                    observed=[observed_min, observed_max],
                    supported=[supported_min, supported_max],
                )

        elapsed_hours = np.diff(t_secs) / 3600.0
        rates = np.diff(soc) / elapsed_hours
        rate_checks = (
            ("c_rate_charge", float(max(0.0, np.max(rates))), float(limits["max_c_rate_charge"])),
            ("c_rate_discharge", float(max(0.0, -np.min(rates))), float(limits["max_c_rate_discharge"])),
        )
        for field, observed, supported in rate_checks:
            if observed > supported + 1e-12:
                message = (
                    f"BLAST model {self.blast_model_key!r} received {field} {observed:.6g} C "
                    f"above experimental maximum {supported:.6g} C."
                )
                self._record_warning(
                    f"experimental_range.{field}",
                    message,
                    "experimental_range",
                    BlastExperimentalRangeWarning,
                    field=field,
                    observed=observed,
                    supported=supported,
                )

    def _check_aging_horizon(self) -> None:
        horizon_days = BATTERY_MODEL_REGISTRY[self.blast_model_key].aging_horizon_days
        if horizon_days is None:
            return
        simulated_days = float(self.model.stressors["t_days"][-1])
        if simulated_days > horizon_days + 1e-12:
            message = (
                f"BLAST model {self.blast_model_key!r} is at {simulated_days:.6g} simulated days, "
                f"beyond its sourced {horizon_days:.6g}-day aging-data horizon."
            )
            self._record_warning(
                "aging_horizon.days",
                message,
                "aging_horizon",
                BlastAgingHorizonWarning,
                observed_days=simulated_days,
                supported_days=horizon_days,
            )

    def warning_records(self, category: str | None = None) -> list[dict[str, Any]]:
        """Return JSON-safe, deduplicated warnings accumulated by this run."""
        return [
            deepcopy(record)
            for record in self._warning_records.values()
            if category is None or record["category"] == category
        ]

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
        engine._warning_records = {record["code"]: deepcopy(record) for record in snapshot.get("warnings", [])}
        for group_name in cls._ARRAY_GROUPS:
            group = snapshot[group_name]
            setattr(
                engine.model,
                group_name,
                {key: np.asarray(value, dtype=float) for key, value in group.items()},
            )
        return engine
