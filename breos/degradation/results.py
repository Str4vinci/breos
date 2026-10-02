"""Construction of public degradation result blocks.

This module owns the small, stable schema exposed by :class:`breos.App`.
Keeping it separate from the simulation runner makes the engine-specific
precision and provenance policy explicit without moving any degradation
physics or continuation-state handling.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from breos.degradation.profiles import BLAST_STATE_SCHEMA_VERSION, get_battery_model_profile
from breos.degradation.protocol import DegradationEngineName


def build_degradation_summary_from_state(
    *,
    engine: DegradationEngineName,
    model_key: str,
    final_soh_pct: float,
    replacement_events: list[dict[str, int]],
    state: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the public, JSON-serializable degradation summary.

    The model profile, warnings, and state schema are meaningful only for the
    explicit BLAST engine, which reads its warnings from the carried lifecycle
    ``state``. Native output deliberately retains its smaller,
    backwards-compatible schema and two-decimal SOH precision.
    """
    if engine == "native":
        return {
            "engine": "native",
            "model_key": model_key,
            "initial_soh_pct": 100.0,
            "final_soh_pct": round(float(final_soh_pct), 2),
            "replacement_events": replacement_events,
        }
    if engine != "blast":
        raise ValueError("degradation engine must be 'native' or 'blast'")

    warnings: list[Mapping[str, Any]] = []
    if state is not None:
        warnings = list(state.get("blast_engine", state).get("warnings", []))
    return {
        "engine": "blast",
        "model_key": model_key,
        "model_profile": get_battery_model_profile(model_key).as_dict(),
        "initial_soh_pct": 100.0,
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
        "state_schema_version": BLAST_STATE_SCHEMA_VERSION,
    }
