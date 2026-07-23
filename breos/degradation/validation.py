"""BLAST experimental-validity checks and run-level warning collection."""

from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Any, Iterable, Mapping

import numpy as np

from breos.degradation.profiles import BATTERY_MODEL_REGISTRY

EXPERIMENTAL_RANGE_CATEGORY = "experimental_range"
AGING_HORIZON_CATEGORY = "aging_horizon"
AGING_HORIZON_CODE = "aging_horizon.days"


class BlastExperimentalRangeWarning(UserWarning):
    """A BLAST input falls outside conditions represented by its test data."""


class BlastAgingHorizonWarning(UserWarning):
    """A BLAST simulation extends beyond a sourced aging-data horizon."""


def experimental_range_code(field: str) -> str:
    """Return the stable warning code for one experimental-range field."""

    return f"{EXPERIMENTAL_RANGE_CATEGORY}.{field}"


class BlastWarningCollector:
    """Evaluate BLAST validity limits and retain deduplicated JSON-safe warnings.

    Warning history belongs to the simulation run rather than the current cell.
    Callers can therefore reset the scientific model after a battery replacement
    without resetting this collector.
    """

    def __init__(
        self,
        blast_model_key: str,
        records: Iterable[Mapping[str, Any]] = (),
    ) -> None:
        self.blast_model_key = blast_model_key
        self._profile = BATTERY_MODEL_REGISTRY[blast_model_key]
        self._records = {str(record["code"]): deepcopy(dict(record)) for record in records}

    @classmethod
    def from_snapshot(
        cls,
        blast_model_key: str,
        records: Iterable[Mapping[str, Any]],
    ) -> BlastWarningCollector:
        """Restore warning history from a serialized engine snapshot."""

        return cls(blast_model_key, records)

    def _record(
        self,
        code: str,
        message: str,
        category: str,
        warning_class: type[Warning],
        **details: Any,
    ) -> None:
        if code in self._records:
            return
        self._records[code] = {
            "code": code,
            "message": message,
            "category": category,
            **details,
        }
        warnings.warn(message, warning_class, stacklevel=3)

    def check_experimental_range(
        self,
        t_secs: np.ndarray,
        soc: np.ndarray,
        temperature_c: np.ndarray,
    ) -> None:
        """Record any input conditions outside the model's sourced test range."""

        limits = self._profile.experimental_range
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
                self._record(
                    experimental_range_code(field),
                    message,
                    EXPERIMENTAL_RANGE_CATEGORY,
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
                self._record(
                    experimental_range_code(field),
                    message,
                    EXPERIMENTAL_RANGE_CATEGORY,
                    BlastExperimentalRangeWarning,
                    field=field,
                    observed=observed,
                    supported=supported,
                )

    def check_aging_horizon(self, simulated_days: float) -> None:
        """Record extrapolation beyond a sourced numeric aging-data horizon."""

        horizon_days = self._profile.aging_horizon_days
        if horizon_days is None or simulated_days <= horizon_days + 1e-12:
            return
        message = (
            f"BLAST model {self.blast_model_key!r} is at {simulated_days:.6g} simulated days, "
            f"beyond its sourced {horizon_days:.6g}-day aging-data horizon."
        )
        self._record(
            AGING_HORIZON_CODE,
            message,
            AGING_HORIZON_CATEGORY,
            BlastAgingHorizonWarning,
            observed_days=simulated_days,
            supported_days=horizon_days,
        )

    def records(self, category: str | None = None) -> list[dict[str, Any]]:
        """Return copied, JSON-safe warning records in first-observed order."""

        return [
            deepcopy(record) for record in self._records.values() if category is None or record["category"] == category
        ]
