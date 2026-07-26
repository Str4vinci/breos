"""Construction of public degradation result blocks.

This module owns the small, stable schema exposed by :class:`breos.App`.
Keeping it separate from the simulation runner makes the engine-specific
precision and provenance policy explicit without moving any degradation
physics or continuation-state handling.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal

from breos.degradation.protocol import resolve_degradation_provenance, warning_records_from_snapshot

DegradationEngineName = Literal["blast", "native"]


def build_degradation_summary(
    *,
    engine: DegradationEngineName,
    model_key: str,
    final_soh_pct: float,
    replacement_events: list[dict[str, int]],
    initial_soh_pct: float = 100.0,
    model_profile: Mapping[str, Any] | None = None,
    warning_records: Iterable[Mapping[str, Any]] = (),
    state_schema_version: str | None = None,
) -> dict[str, Any]:
    """Build the public, JSON-serializable degradation summary.

    ``model_profile``, warnings, and state schema are meaningful only for the
    explicit BLAST engine. Native output deliberately retains its smaller,
    backwards-compatible schema and two-decimal SOH precision.
    """
    if engine == "native":
        return {
            "engine": "native",
            "model_key": model_key,
            "initial_soh_pct": initial_soh_pct,
            "final_soh_pct": round(float(final_soh_pct), 2),
            "replacement_events": replacement_events,
        }

    if model_profile is None:
        raise ValueError("BLAST degradation results require model_profile")
    if state_schema_version is None:
        raise ValueError("BLAST degradation results require state_schema_version")

    warnings = list(warning_records)
    return {
        "engine": "blast",
        "model_key": model_key,
        "model_profile": dict(model_profile),
        "initial_soh_pct": initial_soh_pct,
        "final_soh_pct": round(float(final_soh_pct), 1),
        "replacement_events": replacement_events,
        "calibration_basis": "cell-model",
        "pack_calibrated": False,
        "experimental_range_warnings": [
            warning for warning in warnings if warning.get("category") == "experimental_range"
        ],
        "aging_horizon_extrapolation_warnings": [
            warning for warning in warnings if warning.get("category") == "aging_horizon"
        ],
        "state_schema_version": state_schema_version,
    }


def build_degradation_summary_from_state(
    *,
    engine: DegradationEngineName,
    model_key: str,
    final_soh_pct: float,
    replacement_events: list[dict[str, int]],
    state: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build a result block from lifecycle state without runner branching."""

    provenance = resolve_degradation_provenance(engine, model_key)
    return build_degradation_summary(
        engine=provenance.engine,
        model_key=provenance.model_key,
        model_profile=provenance.model_profile,
        final_soh_pct=final_soh_pct,
        replacement_events=replacement_events,
        warning_records=warning_records_from_snapshot(engine, state),
        state_schema_version=provenance.state_schema_version,
    )
