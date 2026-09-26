"""Narrow internal lifecycle contract for battery degradation engines.

The adapters in this module intentionally stop at the degradation boundary.
They do not own dispatch, battery inventory, replacement energy, resistance
feedback, or the public :class:`breos.App` facade.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import numpy as np

from breos.degradation.profiles import BLAST_STATE_SCHEMA_VERSION, get_battery_model_profile

if TYPE_CHECKING:
    from breos.degradation.engine import BlastEngine

DegradationEngineName = Literal["native", "blast"]
_NATIVE_RAINFLOW_STATE_VERSION = 1


class _NativeRainflowCounter:
    """Incremental ASTM rainflow counter with a residual turning-point stack.

    The state mirrors ``rainflow.reversals`` and ``rainflow.extract_cycles``:
    confirmed reversals can close cycles as samples arrive, while the final
    half cycles remain pending until the caller closes the whole simulation.
    Only the unresolved turning points and two reversal-scanner points are
    retained, so daily stepping does not rescan the run history.
    """

    def __init__(self, state: Mapping[str, Any] | None = None) -> None:
        self._points: deque[tuple[float, float, int]] = deque()
        self._previous: tuple[float, float, int] | None = None
        self._current: tuple[float, float, int] | None = None
        self._last_delta: float | None = None
        self._has_nonflat_pair = False
        self._next_index = 0
        self._step_seconds: float | None = None
        if state:
            self._restore(state)

    @staticmethod
    def _point(value: Any) -> tuple[float, float, int] | None:
        if value is None:
            return None
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError("native rainflow points must contain SOC, elapsed seconds, and sample index")
        soc, elapsed_seconds, index = value
        return float(soc), float(elapsed_seconds), int(index)

    def _restore(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != _NATIVE_RAINFLOW_STATE_VERSION:
            raise ValueError(
                "Unsupported native rainflow state schema "
                f"{state.get('schema_version')!r}; expected {_NATIVE_RAINFLOW_STATE_VERSION!r}"
            )
        try:
            self._points = deque(self._point(point) for point in state.get("residue", []))  # type: ignore[arg-type]
            self._previous = self._point(state.get("previous"))
            self._current = self._point(state.get("current"))
            last_delta = state.get("last_delta")
            self._last_delta = None if last_delta is None else float(last_delta)
            self._has_nonflat_pair = bool(state.get("has_nonflat_pair", False))
            self._next_index = int(state.get("next_index", 0))
            step_seconds = state.get("step_seconds")
            self._step_seconds = None if step_seconds is None else float(step_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid native rainflow state") from exc
        if self._next_index < 0 or any(point is None for point in self._points):
            raise ValueError("Invalid native rainflow state")
        if self._step_seconds is not None and (not np.isfinite(self._step_seconds) or self._step_seconds <= 0):
            raise ValueError("Invalid native rainflow step size")

    def snapshot(self) -> dict[str, Any]:
        def serialize(point: tuple[float, float, int] | None) -> list[float | int] | None:
            return None if point is None else [point[0], point[1], point[2]]

        return {
            "schema_version": _NATIVE_RAINFLOW_STATE_VERSION,
            "residue": [serialize(point) for point in self._points],
            "previous": serialize(self._previous),
            "current": serialize(self._current),
            "last_delta": self._last_delta,
            "has_nonflat_pair": self._has_nonflat_pair,
            "next_index": self._next_index,
            "step_seconds": self._step_seconds,
        }

    def reset(self) -> None:
        self._points.clear()
        self._previous = None
        self._current = None
        self._last_delta = None
        self._has_nonflat_pair = False
        self._next_index = 0
        self._step_seconds = None

    def finish(self) -> list[dict[str, float | int]]:
        """Emit terminal half cycles from the current residue and reset it."""
        cycles: list[dict[str, float | int]] = []
        if self._current is not None and self._has_nonflat_pair:
            cycles.extend(self._push_reversal(self._current, min_doc_fraction=0.01))
            while len(self._points) > 1:
                cycle = self._format_cycle(self._points[0], self._points[1], 0.5)
                self._points.popleft()
                if cycle["doc"] >= 0.01:
                    cycles.append(cycle)
        self.reset()
        return cycles

    def _format_cycle(
        self,
        point1: tuple[float, float, int],
        point2: tuple[float, float, int],
        count: float,
    ) -> dict[str, float | int]:
        soc1, time1, index1 = point1
        soc2, time2, index2 = point2
        # Match _detect_cycles_rainflow_arrays, which passes percent SOC to
        # rainflow before converting the range back to a fraction. Computing
        # the range directly on fractions can put a 1% cycle just below the
        # minimum through floating-point rounding.
        doc = abs(soc1 * 100.0 - soc2 * 100.0) / 100.0
        duration_seconds = abs(time2 - time1)
        duration_hours = duration_seconds / 3600.0
        return {
            "doc": doc,
            "mean_soc": 0.5 * (soc1 * 100.0 + soc2 * 100.0) / 100.0,
            "count": count,
            "mean_c_rate": doc / duration_hours if duration_hours > 0 else 0.0,
            "start_idx": index1,
            "end_idx": index2,
        }

    def _push_reversal(
        self,
        point: tuple[float, float, int],
        *,
        min_doc_fraction: float,
    ) -> list[dict[str, float | int]]:
        cycles: list[dict[str, float | int]] = []
        self._points.append(point)
        while len(self._points) >= 3:
            x1 = self._points[-3][0] * 100.0
            x2 = self._points[-2][0] * 100.0
            x3 = self._points[-1][0] * 100.0
            x_range = abs(x3 - x2)
            y_range = abs(x2 - x1)
            if x_range < y_range:
                break
            if len(self._points) == 3:
                cycle = self._format_cycle(self._points[0], self._points[1], 0.5)
                self._points.popleft()
            else:
                cycle = self._format_cycle(self._points[-3], self._points[-2], 1.0)
                last = self._points.pop()
                self._points.pop()
                self._points.pop()
                self._points.append(last)
            if cycle["doc"] >= min_doc_fraction:
                cycles.append(cycle)
        return cycles

    def step(self, day: "DegradationDay") -> list[dict[str, float | int]]:
        if day.soc.size == 0:
            return self.finish() if day.finalize_cycles else []
        if len(day.soc) != len(day.time_ticks):
            raise ValueError("SOC and time arrays must have the same length")
        if day.step_seconds <= 0 or not np.isfinite(day.step_seconds):
            raise ValueError("step_seconds must be positive and finite")
        if self._step_seconds is None:
            self._step_seconds = float(day.step_seconds)
        elif self._step_seconds != day.step_seconds:
            raise ValueError("native rainflow step size changed across a continued span")
        if self._current is not None and not np.isclose(day.start_soc, self._current[0], rtol=0.0, atol=1e-9):
            raise ValueError("native rainflow carry SOC does not match the next span's starting SOC")

        cycles: list[dict[str, float | int]] = []
        if self._current is None:
            anchor = (float(day.start_soc), 0.0, 0)
            self._points.append(anchor)
            first_sample = (float(day.soc[0]), day.step_seconds, 1)
            self._previous = anchor
            self._current = first_sample
            self._last_delta = first_sample[0] - anchor[0]
            self._has_nonflat_pair = self._last_delta != 0.0
            self._next_index = 2
            samples = zip(day.soc[1:], day.time_ticks[1:], strict=True)
        else:
            samples = zip(day.soc, day.time_ticks, strict=True)

        for soc, _ in samples:
            assert self._current is not None
            next_point = (float(soc), self._next_index * day.step_seconds, self._next_index)
            self._next_index += 1
            if next_point[0] == self._current[0]:
                # rainflow.reversals labels a plateau at its final sample.
                # Keep the timestamp and index current even without a turn.
                self._current = next_point
                continue
            next_delta = next_point[0] - self._current[0]
            if self._last_delta is not None and self._last_delta * next_delta < 0:
                cycles.extend(
                    self._push_reversal(
                        self._current,
                        min_doc_fraction=0.01,
                    )
                )
            self._previous = self._current
            self._current = next_point
            self._last_delta = next_delta
            self._has_nonflat_pair = True

        if day.finalize_cycles:
            cycles.extend(self.finish())
        return cycles


@dataclass(frozen=True)
class DegradationDay:
    """Inputs shared by the native and BLAST daily lifecycle steps."""

    soc: np.ndarray
    time_ticks: np.ndarray
    ticks_per_second: float
    temperature_c: np.ndarray
    step_seconds: float
    start_soc: float
    start_temperature_c: float
    finalize_cycles: bool = True

    @property
    def mean_soc(self) -> float:
        return float(np.mean(self.soc))

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
    cycle_records: tuple[Mapping[str, float | int], ...] = ()


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

    def finalize_cycles(self) -> DegradationStep: ...

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
        initial_rainflow_state: Mapping[str, Any] | None = None,
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
        self._rainflow = _NativeRainflowCounter(initial_rainflow_state)
        self._debug = debug

    def step(self, day: DegradationDay) -> DegradationStep:
        cycles = self._rainflow.step(day)
        soh_after_cycle, cycle_degradation, self._fec = self._cycle_step(
            self._soh,
            cycles,
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
            dt_days=(len(day.soc) * day.step_seconds) / 86400.0,
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
            cycle_records=tuple(cycles),
        )

    def finalize_cycles(self) -> DegradationStep:
        """Charge pending terminal half cycles without adding calendar aging."""
        cycles = self._rainflow.finish()
        self._soh, cycle_degradation, self._fec = self._cycle_step(
            self._soh,
            cycles,
            self._nominal_energy_wh,
            fec_cum=self._fec,
            battery_type=self._battery_type,
            debug=self._debug,
        )
        self._cumulative_cycle_degradation += cycle_degradation
        return DegradationStep(
            soh_fraction=self._soh,
            fec=self._fec,
            calendar_seconds=self._calendar_seconds,
            cycle_degradation=cycle_degradation,
            calendar_degradation=0.0,
            cycle_records=tuple(cycles),
        )

    def soh(self) -> float:
        return self._soh

    def reset(self) -> None:
        self._soh = 1.0
        self._fec = 0.0
        self._calendar_seconds = 0.0
        self._cumulative_cycle_degradation = 0.0
        self._cumulative_calendar_degradation = 0.0
        self._rainflow.reset()

    def snapshot(self, *, day_start_soc: float, day_start_temperature_c: float) -> dict[str, Any]:
        return {
            "degradation_engine": "native",
            "soh_fraction": self._soh,
            "fec_cum": self._fec,
            "cumulative_calendar_seconds": self._calendar_seconds,
            "cumulative_cycle_degradation": self._cumulative_cycle_degradation,
            "cumulative_calendar_degradation": self._cumulative_calendar_degradation,
            "native_rainflow_state": self._rainflow.snapshot(),
            "day_start_soc_absolute": float(day_start_soc),
            "day_start_temperature_c": float(day_start_temperature_c),
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
            day.soc,
            day.temperature_c,
            start_soc=day.start_soc,
            start_temperature_c=day.start_temperature_c,
        )
        # BLAST capacity extrapolations can dip below zero for cells aged far
        # past their data; a dead battery has zero usable capacity.
        self._soh = max(0.0, self._engine.step(t_secs, soc, temperature_c))
        self._fec = float(self._engine.model.stressors["efc"][-1])
        self._calendar_seconds += len(day.soc) * day.step_seconds
        return DegradationStep(
            soh_fraction=self._soh,
            fec=self._fec,
            calendar_seconds=self._calendar_seconds,
            cycle_degradation=0.0,
            calendar_degradation=0.0,
            engine_degradation=max(0.0, previous_soh - self._soh),
        )

    def finalize_cycles(self) -> DegradationStep:
        """BLAST owns its rainflow state and has no adapter-side residue."""
        return DegradationStep(
            soh_fraction=self._soh,
            fec=self._fec,
            calendar_seconds=self._calendar_seconds,
            cycle_degradation=0.0,
            calendar_degradation=0.0,
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
