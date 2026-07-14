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

BLAST_MODEL_CLASSES = {
    "lfp_gr_250ah_prismatic": models.Lfp_Gr_250AhPrismatic,
    "nca_gr_panasonic_3ah": models.Nca_Gr_Panasonic3Ah_Battery,
    "lmo_gr_nissanleaf_66ah_2nd": models.Lmo_Gr_NissanLeaf66Ah_2ndLife_Battery,
    "nmc811_grsi_lgm50_5ah": models.Nmc811_GrSi_LGM50_5Ah_Battery,
    "nmc811_grsi_lgmj1_4ah": models.Nmc811_GrSi_LGMJ1_4Ah_Battery,
    "nmc_gr_50ah_b1": models.NMC_Gr_50Ah_B1,
    "nmc_gr_50ah_b2": models.NMC_Gr_50Ah_B2,
    "nmc_gr_75ah_a": models.NMC_Gr_75Ah_A,
    "nmc111_gr_sanyo_2ah": models.Nmc111_Gr_Sanyo2Ah_Battery,
    "nmc_lto_10ah": models.Nmc_Lto_10Ah_Battery,
    "lfp_gr_sonymurata_3ah": models.Lfp_Gr_SonyMurata3Ah_Battery,
    "nca_grsi_sonymurata_2p5ah": models.NCA_GrSi_SonyMurata2p5Ah_Battery,
    "nmc111_gr_kokam_75ah": models.Nmc111_Gr_Kokam75Ah_Battery,
    "nmc622_gr_denso_50ah": models.Nmc622_Gr_DENSO50Ah_Battery,
}

P1_BLAST_MODEL_KEYS = ("lfp_gr_250ah_prismatic", "nca_gr_panasonic_3ah")


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

    def _new_model(self):
        return BLAST_MODEL_CLASSES[self.blast_model_key](**self._model_kwargs)

    def step(self, t_secs_day: Any, soc_abs_day: Any, t_cell_day_c: Any) -> float:
        """Update by one time chunk and return current SoH fraction."""

        t_secs, soc, temperature_c = normalize_step_inputs(t_secs_day, soc_abs_day, t_cell_day_c)
        self.model.update_battery_state(t_secs, soc, temperature_c)
        return self.soh()

    def soh(self) -> float:
        """Return current BLAST capacity fraction."""

        return float(self.model.outputs["q"][-1])

    def reset(self) -> None:
        """Reset to a fresh beginning-of-life model instance."""

        self.model = self._new_model()

    def state_snapshot(self, *, serializable: bool = True) -> dict[str, Any]:
        """Copy the model state required for cross-year threading."""

        snapshot: dict[str, Any] = {
            "blast_model_key": self.blast_model_key,
            "model_kwargs": deepcopy(self._model_kwargs),
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

        engine = cls(blast_model_key, **snapshot.get("model_kwargs", {}))
        for group_name in cls._ARRAY_GROUPS:
            group = snapshot[group_name]
            setattr(
                engine.model,
                group_name,
                {key: np.asarray(value, dtype=float) for key, value in group.items()},
            )
        return engine
