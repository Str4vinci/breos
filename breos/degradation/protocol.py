"""Narrow internal lifecycle contract for battery degradation engines.

The adapters in this module intentionally stop at the degradation boundary.
They do not own dispatch, battery inventory, replacement energy, resistance
feedback, or the public :class:`breos.App` facade.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from breos.degradation.profiles import BLAST_STATE_SCHEMA_VERSION, get_battery_model_profile

if TYPE_CHECKING:
    from breos.degradation.engine import BlastEngine

DegradationEngineName = Literal["native", "blast"]


@dataclass(frozen=True)
class DegradationDay:
    """Inputs shared by the native and BLAST daily lifecycle steps."""

    soc: pd.Series
    temperature_c: np.ndarray
    step_seconds: float
    start_soc: float
    start_temperature_c: float

    @property
    def mean_soc(self) -> float:
        return float(self.soc.mean())

    @property
    def mean_temperature_c(self) -> float:
        return float(np.mean(self.temperature_c))


@dataclass(frozen=True)
class DegradationStep:
    """Normalized result of one degradation lifecycle step."""

    soh_fraction: float
    fec: float
    calendar_seconds: float
    cycle_degradation: float
    calendar_degradation: float
    engine_degradation: float = 0.0


@dataclass(frozen=True)
class DegradationProvenance:
    """Engine identity used by the public degradation-result builder."""

    engine: DegradationEngineName
    model_key: str
    model_profile: Mapping[str, Any] | None = None
    state_schema_version: str | None = None


@runtime_checkable
class DegradationLifecycle(Protocol):
    """Internal lifecycle operations required by the energy-balance runner."""

    def step(self, day: DegradationDay) -> DegradationStep: ...

    def soh(self) -> float: ...

    def reset(self) -> None: ...

    def snapshot(self, *, day_start_soc: float, day_start_temperature_c: float) -> dict[str, Any]: ...

    def warnings(self) -> list[dict[str, Any]]: ...

    def provenance(self) -> DegradationProvenance: ...

    def tracking_fields(self, step: DegradationStep) -> dict[str, Any]: ...


CycleStep = Callable[..., tuple[float, float, float]]
CalendarStep = Callable[..., tuple[float, float, float]]


class NativeDegradationAdapter:
    """Lifecycle adapter for BREOS' native Naumann/Lam state handling."""

    def __init__(
        self,
        *,
        model_key: str,
        initial_soh_fraction: float,
        initial_fec: float,
        initial_calendar_seconds: float,
        initial_cumulative_cycle_degradation: float,
        initial_cumulative_calendar_degradation: float,
        nominal_energy_wh: float,
        battery_type: str,
        k0_fraction: float,
        activation_energy: float,
        soc_exponent: float,
        time_exponent: float,
        cycle_step: CycleStep,
        calendar_step: CalendarStep,
        debug: bool = False,
    ) -> None:
        self.model_key = model_key
        self._soh = float(initial_soh_fraction)
        self._fec = float(initial_fec)
        self._calendar_seconds = float(initial_calendar_seconds)
        self._cumulative_cycle_degradation = float(initial_cumulative_cycle_degradation)
        self._cumulative_calendar_degradation = float(initial_cumulative_calendar_degradation)
        self._nominal_energy_wh = float(nominal_energy_wh)
        self._battery_type = battery_type
        self._k0_fraction = float(k0_fraction)
        self._activation_energy = float(activation_energy)
        self._soc_exponent = float(soc_exponent)
        self._time_exponent = float(time_exponent)
        self._cycle_step = cycle_step
        self._calendar_step = calendar_step
        self._debug = debug

    def step(self, day: DegradationDay) -> DegradationStep:
        soh_after_cycle, cycle_degradation, self._fec = self._cycle_step(
            self._soh,
            day.soc,
            self._nominal_energy_wh,
            fec_cum=self._fec,
            battery_type=self._battery_type,
            debug=self._debug,
        )
        self._soh, calendar_degradation, self._calendar_seconds = self._calendar_step(
            soh_after_cycle,
            k0_frac=self._k0_fraction,
            Ea=self._activation_energy,
            n=self._soc_exponent,
            cal_b=self._time_exponent,
            T_cell_C=day.mean_temperature_c,
            cumulative_cal_seconds=self._calendar_seconds,
            dt_days=1.0,
            mean_soc_absolute=day.mean_soc,
            debug=self._debug,
        )
        self._cumulative_cycle_degradation += cycle_degradation
        self._cumulative_calendar_degradation += calendar_degradation
        return DegradationStep(
            soh_fraction=self._soh,
            fec=self._fec,
            calendar_seconds=self._calendar_seconds,
            cycle_degradation=cycle_degradation,
            calendar_degradation=calendar_degradation,
        )

    def soh(self) -> float:
        return self._soh

    def reset(self) -> None:
        self._soh = 1.0
        self._fec = 0.0
        self._calendar_seconds = 0.0
        self._cumulative_cycle_degradation = 0.0
        self._cumulative_calendar_degradation = 0.0

    def snapshot(self, *, day_start_soc: float, day_start_temperature_c: float) -> dict[str, Any]:
        del day_start_soc, day_start_temperature_c
        return {
            "degradation_engine": "native",
            "fec_cum": self._fec,
            "cumulative_calendar_seconds": self._calendar_seconds,
            "cumulative_cycle_degradation": self._cumulative_cycle_degradation,
            "cumulative_calendar_degradation": self._cumulative_calendar_degradation,
        }

    def warnings(self) -> list[dict[str, Any]]:
        return []

    def provenance(self) -> DegradationProvenance:
        return resolve_degradation_provenance("native", self.model_key)

    def tracking_fields(self, step: DegradationStep) -> dict[str, Any]:
        del step
        return {}


class BlastDegradationAdapter:
    """Lifecycle adapter around the vendored BLAST engine wrapper."""

    def __init__(
        self,
        model_key: str,
        *,
        initial_state: Mapping[str, Any] | None = None,
        initial_fec: float = 0.0,
        initial_calendar_seconds: float = 0.0,
        initial_cumulative_cycle_degradation: float = 0.0,
        initial_cumulative_calendar_degradation: float = 0.0,
    ) -> None:
        from breos.degradation.engine import BlastEngine, build_endpoint_day

        self.model_key = model_key
        self._build_endpoint_day = build_endpoint_day
        state = dict(initial_state or {})
        engine_snapshot = state.get("blast_engine", state)
        self._engine: BlastEngine = (
            BlastEngine.from_snapshot(model_key, engine_snapshot) if engine_snapshot else BlastEngine(model_key)
        )
        self._soh = self._engine.soh()
        self._fec = float(initial_fec)
        self._calendar_seconds = float(initial_calendar_seconds)
        self._cumulative_cycle_degradation = float(initial_cumulative_cycle_degradation)
        self._cumulative_calendar_degradation = float(initial_cumulative_calendar_degradation)

    def step(self, day: DegradationDay) -> DegradationStep:
        previous_soh = self._soh
        t_secs, soc, temperature_c = self._build_endpoint_day(
            day.step_seconds,
            day.soc.to_numpy(),
            day.temperature_c,
            start_soc=day.start_soc,
            start_temperature_c=day.start_temperature_c,
        )
        # BLAST capacity extrapolations can dip below zero for cells aged far
        # past their data; a dead battery has zero usable capacity.
        self._soh = max(0.0, self._engine.step(t_secs, soc, temperature_c))
        self._fec = float(self._engine.model.stressors["efc"][-1])
        self._calendar_seconds += 86400.0
        return DegradationStep(
            soh_fraction=self._soh,
            fec=self._fec,
            calendar_seconds=self._calendar_seconds,
            cycle_degradation=0.0,
            calendar_degradation=0.0,
            engine_degradation=max(0.0, previous_soh - self._soh),
        )

    def soh(self) -> float:
        return self._soh

    def reset(self) -> None:
        self._engine.reset()
        self._soh = 1.0
        self._fec = 0.0
        self._calendar_seconds = 0.0
        self._cumulative_cycle_degradation = 0.0
        self._cumulative_calendar_degradation = 0.0

    def snapshot(self, *, day_start_soc: float, day_start_temperature_c: float) -> dict[str, Any]:
        return {
            "degradation_engine": "blast",
            "fec_cum": self._fec,
            "cumulative_calendar_seconds": self._calendar_seconds,
            "cumulative_cycle_degradation": self._cumulative_cycle_degradation,
            "cumulative_calendar_degradation": self._cumulative_calendar_degradation,
            "blast_model": self.model_key,
            "blast_engine": self._engine.state_snapshot(),
            "day_start_soc_absolute": float(day_start_soc),
            "day_start_temperature_c": float(day_start_temperature_c),
        }

    def warnings(self) -> list[dict[str, Any]]:
        return self._engine.warning_records()

    def provenance(self) -> DegradationProvenance:
        return resolve_degradation_provenance("blast", self.model_key)

    def tracking_fields(self, step: DegradationStep) -> dict[str, Any]:
        return {
            "BLAST_Model": self.model_key,
            "BLAST_Degradation": step.engine_degradation,
        }


def resolve_degradation_provenance(engine: str, model_key: str) -> DegradationProvenance:
    """Resolve result provenance without exposing the lifecycle adapters."""

    if engine == "native":
        return DegradationProvenance(engine="native", model_key=model_key)
    if engine == "blast":
        profile = get_battery_model_profile(model_key)
        return DegradationProvenance(
            engine="blast",
            model_key=model_key,
            model_profile=profile.as_dict(),
            state_schema_version=BLAST_STATE_SCHEMA_VERSION,
        )
    raise ValueError("degradation engine must be 'native' or 'blast'")


def warning_records_from_snapshot(engine: str, state: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Read lifecycle warnings from the current schema without runner branching."""

    if engine == "native" or state is None:
        return []
    engine_state = state.get("blast_engine", state)
    return list(engine_state.get("warnings", []))
