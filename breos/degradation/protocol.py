"""Narrow internal lifecycle contract for battery degradation engines.

The adapters in this module intentionally stop at the degradation boundary.
They do not own dispatch, battery inventory, replacement energy, resistance
feedback, or the public :class:`breos.App` facade.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

import numpy as np

from breos._rainflow import Point, RainflowCounter, RawCycle
from breos.degradation import _rainflow_kernel as _rk

if TYPE_CHECKING:
    from breos.degradation.engine import BlastEngine

DegradationEngineName = Literal["native", "blast"]
_NATIVE_RAINFLOW_STATE_VERSION = 1
# Cycles shallower than this SOC fraction do not age the native model.
_MIN_CYCLE_DOC = 0.01


class _NativeRainflowCounter:
    """The energy balance's rainflow count, day by day, on :class:`breos._rainflow.RainflowCounter`.

    A run's first day starts from an anchor at the starting SOC, sample 0;
    sample k is k steps later. Ranges are compared on percent SOC, as
    ``_detect_cycles_rainflow_arrays`` counts them, and cycles shallower than
    1 % are dropped. The final half cycles remain pending until the caller
    closes the whole simulation, and the open count round-trips through
    :meth:`snapshot`, so daily stepping does not rescan the run history.
    """

    def __init__(self, state: Mapping[str, Any] | None = None) -> None:
        self._counter = RainflowCounter(scale=100.0)
        self._step_seconds: float | None = None
        if state:
            self._restore(state)

    @staticmethod
    def _point(value: Any) -> Point | None:
        if value is None:
            return None
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError("native rainflow points must contain SOC, elapsed seconds, and sample index")
        soc, elapsed_seconds, index = value
        float(elapsed_seconds)  # Checked, not kept: a point's time is its index times the step.
        return float(soc), int(index)

    def _restore(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != _NATIVE_RAINFLOW_STATE_VERSION:
            raise ValueError(
                "Unsupported native rainflow state schema "
                f"{state.get('schema_version')!r}; expected {_NATIVE_RAINFLOW_STATE_VERSION!r}"
            )
        counter = self._counter
        try:
            residue = [self._point(point) for point in state.get("residue", [])]
            if any(point is None for point in residue):
                raise ValueError("native rainflow residue points must not be null")
            counter.points = deque(point for point in residue if point is not None)
            counter.previous = self._point(state.get("previous"))
            counter.current = self._point(state.get("current"))
            last_delta = state.get("last_delta")
            counter.last_delta = None if last_delta is None else float(last_delta)
            counter.moved = bool(state.get("has_nonflat_pair", False))
            counter.next_index = int(state.get("next_index", 0))
            step_seconds = state.get("step_seconds")
            self._step_seconds = None if step_seconds is None else float(step_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid native rainflow state") from exc
        if counter.next_index < 0:
            raise ValueError("Invalid native rainflow state")
        if self._step_seconds is None:
            if counter.points or counter.current is not None or counter.previous is not None:
                raise ValueError("Invalid native rainflow state")
        elif not np.isfinite(self._step_seconds) or self._step_seconds <= 0:
            raise ValueError("Invalid native rainflow step size")

    def snapshot(self) -> dict[str, Any]:
        counter = self._counter
        step_seconds = self._step_seconds

        def serialize(point: Point | None) -> list[float | int] | None:
            if point is None:
                return None
            assert step_seconds is not None
            return [point[0], point[1] * step_seconds, point[1]]

        return {
            "schema_version": _NATIVE_RAINFLOW_STATE_VERSION,
            "residue": [serialize(point) for point in counter.points],
            "previous": serialize(counter.previous),
            "current": serialize(counter.current),
            "last_delta": counter.last_delta,
            "has_nonflat_pair": counter.moved,
            "next_index": counter.next_index,
            "step_seconds": step_seconds,
        }

    def reset(self) -> None:
        self._counter.reset()
        self._step_seconds = None

    def finish(self) -> list[dict[str, float | int]]:
        """Emit terminal half cycles from the current residue and reset it."""
        cycles = self._format(self._counter.finish())
        self.reset()
        return cycles

    def _format(self, cycles: list[RawCycle]) -> list[dict[str, float | int]]:
        """The cycles of at least 1 % depth as records, with depth and mean as SOC fractions."""
        step_seconds = self._step_seconds
        records: list[dict[str, float | int]] = []
        for (soc1, index1), (soc2, index2), count in cycles:
            # Match _detect_cycles_rainflow_arrays, which counts on percent SOC
            # and converts the range back to a fraction. Computing the range
            # directly on fractions can put a 1% cycle just below the minimum
            # through floating-point rounding.
            doc = abs(soc1 * 100.0 - soc2 * 100.0) / 100.0
            if not doc >= _MIN_CYCLE_DOC:
                continue
            assert step_seconds is not None
            duration_hours = abs(index2 * step_seconds - index1 * step_seconds) / 3600.0
            records.append(
                {
                    "doc": doc,
                    "mean_soc": 0.5 * (soc1 * 100.0 + soc2 * 100.0) / 100.0,
                    "count": count,
                    "mean_c_rate": doc / duration_hours if duration_hours > 0 else 0.0,
                    "start_idx": index1,
                    "end_idx": index2,
                }
            )
        return records

    def _check_day(self, day: "DegradationDay", current_soc: float | None) -> None:
        """Reject a day that cannot continue the count from ``current_soc`` (None before the first sample)."""
        if len(day.soc) != len(day.time_ticks):
            raise ValueError("SOC and time arrays must have the same length")
        if day.step_seconds <= 0 or not np.isfinite(day.step_seconds):
            raise ValueError("step_seconds must be positive and finite")
        if self._step_seconds is None:
            self._step_seconds = float(day.step_seconds)
        elif self._step_seconds != day.step_seconds:
            raise ValueError("native rainflow step size changed across a continued span")
        # Scalar np.isclose costs ~17 us; abs() <= atol is the same test for finite
        # SOC (rtol=0), and anything non-finite falls through to np.isclose. The
        # difference is taken on Python floats, which, like np.isclose, never
        # raise under numpy's floating-point error settings.
        if current_soc is not None and not (
            abs(float(day.start_soc) - current_soc) <= 1e-9
            or np.isclose(day.start_soc, current_soc, rtol=0.0, atol=1e-9)
        ):
            raise ValueError("native rainflow carry SOC does not match the next span's starting SOC")

    def step(self, day: "DegradationDay") -> list[dict[str, float | int]]:
        if day.soc.size == 0:
            return self.finish() if day.finalize_cycles else []
        current = self._counter.current
        self._check_day(day, None if current is None else current[0])
        if current is None:
            # The anchor at the starting SOC is sample 0.
            cycles = self._format(self._counter.extend(np.concatenate(([float(day.start_soc)], day.soc))))
        else:
            cycles = self._format(self._counter.extend(day.soc))
        if day.finalize_cycles:
            cycles.extend(self.finish())
        return cycles


class _CompiledRainflowCounter(_NativeRainflowCounter):
    """:class:`_NativeRainflowCounter` with the daily scan compiled, for ``execution_backend = "numba"``.

    The residue and scanner state live in arrays that
    :func:`breos.degradation._rainflow_kernel.rainflow_day` advances in place;
    the shared counter's state is rebuilt from them only to snapshot or to
    close the residue, which stay in Python. Both counters return the same
    cycles, bit for bit, and the same snapshots.
    """

    def __init__(self, state: Mapping[str, Any] | None = None, *, kernel: Callable[..., int]) -> None:
        self._kernel = kernel
        self._capacity = 0
        self._points_f = np.empty((0, 2))
        self._points_i = np.empty(0, dtype=np.int64)
        self._out_f = np.empty((0, 4))
        self._out_i = np.empty((0, 2), dtype=np.int64)
        self._fstate = np.empty(0)
        self._istate = np.empty(0, dtype=np.int64)
        super().__init__(state)
        self._load()

    def _ensure_capacity(self, rows: int) -> None:
        if rows <= self._capacity:
            return
        capacity = max(rows, 2 * self._capacity, 256)
        points_f = np.empty((capacity, 2))
        points_i = np.empty(capacity, dtype=np.int64)
        if self._capacity:
            points_f[: self._capacity] = self._points_f
            points_i[: self._capacity] = self._points_i
        self._points_f = points_f
        self._points_i = points_i
        self._out_f = np.empty((capacity, 4))
        self._out_i = np.empty((capacity, 2), dtype=np.int64)
        self._capacity = capacity

    def _load(self) -> None:
        """Copy the shared counter's state into the arrays."""
        counter = self._counter
        step_seconds = 0.0 if self._step_seconds is None else self._step_seconds
        self._ensure_capacity(len(counter.points) + 1)
        for row, (soc, index) in enumerate(counter.points):
            self._points_f[row, 0] = soc
            self._points_f[row, 1] = index * step_seconds
            self._points_i[row] = index
        fstate = np.zeros(_rk.FSTATE_SIZE)
        istate = np.zeros(_rk.ISTATE_SIZE, dtype=np.int64)
        istate[_rk._I_N_POINTS] = len(counter.points)
        if counter.current is not None:
            soc, index = counter.current
            istate[_rk._I_HAS_CURRENT] = 1
            fstate[_rk._F_CURRENT_SOC], fstate[_rk._F_CURRENT_TIME], istate[_rk._I_CURRENT_INDEX] = (
                soc,
                index * step_seconds,
                index,
            )
        if counter.previous is not None:
            soc, index = counter.previous
            istate[_rk._I_HAS_PREVIOUS] = 1
            fstate[_rk._F_PREVIOUS_SOC], fstate[_rk._F_PREVIOUS_TIME], istate[_rk._I_PREVIOUS_INDEX] = (
                soc,
                index * step_seconds,
                index,
            )
        if counter.last_delta is not None:
            istate[_rk._I_HAS_LAST_DELTA] = 1
            fstate[_rk._F_LAST_DELTA] = counter.last_delta
        istate[_rk._I_HAS_NONFLAT_PAIR] = int(counter.moved)
        istate[_rk._I_NEXT_INDEX] = counter.next_index
        self._fstate = fstate
        self._istate = istate

    def _store(self) -> None:
        """Rebuild the shared counter's state from the arrays."""
        counter = self._counter
        fstate = self._fstate.tolist()
        istate = self._istate.tolist()
        n_points = istate[_rk._I_N_POINTS]
        counter.points = deque(
            zip(self._points_f[:n_points, 0].tolist(), self._points_i[:n_points].tolist(), strict=True)
        )
        counter.current = (
            (fstate[_rk._F_CURRENT_SOC], istate[_rk._I_CURRENT_INDEX]) if istate[_rk._I_HAS_CURRENT] else None
        )
        counter.previous = (
            (fstate[_rk._F_PREVIOUS_SOC], istate[_rk._I_PREVIOUS_INDEX]) if istate[_rk._I_HAS_PREVIOUS] else None
        )
        counter.last_delta = fstate[_rk._F_LAST_DELTA] if istate[_rk._I_HAS_LAST_DELTA] else None
        counter.moved = bool(istate[_rk._I_HAS_NONFLAT_PAIR])
        counter.next_index = istate[_rk._I_NEXT_INDEX]

    def snapshot(self) -> dict[str, Any]:
        self._store()
        return super().snapshot()

    def reset(self) -> None:
        super().reset()
        self._load()

    def finish(self) -> list[dict[str, float | int]]:
        self._store()
        # The parent's finish ends with self.reset(), which reloads the arrays.
        return super().finish()

    def step(self, day: "DegradationDay") -> list[dict[str, float | int]]:
        if day.soc.size == 0:
            return self.finish() if day.finalize_cycles else []
        has_current = self._istate[_rk._I_HAS_CURRENT]
        self._check_day(day, float(self._fstate[_rk._F_CURRENT_SOC]) if has_current else None)
        soc = np.ascontiguousarray(day.soc, dtype=np.float64)
        # A day pushes at most one point per sample plus the anchor, and every
        # emitted cycle removes a point.
        self._ensure_capacity(int(self._istate[_rk._I_N_POINTS]) + len(soc) + 1)
        n_cycles = self._kernel(
            self._points_f,
            self._points_i,
            self._fstate,
            self._istate,
            soc,
            float(day.step_seconds),
            float(day.start_soc),
            self._out_f,
            self._out_i,
            _MIN_CYCLE_DOC,
        )
        cycles: list[dict[str, float | int]] = []
        if n_cycles:
            for (doc, mean_soc, count, mean_c_rate), (start_idx, end_idx) in zip(
                self._out_f[:n_cycles].tolist(), self._out_i[:n_cycles].tolist(), strict=True
            ):
                cycles.append(
                    {
                        "doc": doc,
                        "mean_soc": mean_soc,
                        "count": count,
                        "mean_c_rate": mean_c_rate,
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                    }
                )
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
    # The caller has usually taken these means already; passing them in
    # avoids a second np.mean of the same arrays every day.
    soc_mean: float | None = None
    temperature_mean_c: float | None = None

    @property
    def mean_soc(self) -> float:
        return float(np.mean(self.soc)) if self.soc_mean is None else self.soc_mean

    @property
    def mean_temperature_c(self) -> float:
        return float(np.mean(self.temperature_c)) if self.temperature_mean_c is None else self.temperature_mean_c


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


class DegradationLifecycle(Protocol):
    """Internal lifecycle operations required by the energy-balance runner."""

    def step(self, day: DegradationDay) -> DegradationStep: ...

    def finalize_cycles(self) -> DegradationStep: ...

    def soh(self) -> float: ...

    def reset(self) -> None: ...

    def snapshot(self, *, day_start_soc: float, day_start_temperature_c: float) -> dict[str, Any]: ...

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
        k0_fraction: float,
        activation_energy: float,
        soc_exponent: float,
        time_exponent: float,
        cycle_step: CycleStep,
        calendar_step: CalendarStep,
        initial_rainflow_state: Mapping[str, Any] | None = None,
        rainflow_kernel: Callable[..., int] | None = None,
    ) -> None:
        """``rainflow_kernel`` is the compiled day scan for the numba backend; None counts in numpy and Python."""
        self.model_key = model_key
        self._soh = float(initial_soh_fraction)
        self._fec = float(initial_fec)
        self._calendar_seconds = float(initial_calendar_seconds)
        self._k0_fraction = float(k0_fraction)
        self._activation_energy = float(activation_energy)
        self._soc_exponent = float(soc_exponent)
        self._time_exponent = float(time_exponent)
        self._cycle_step = cycle_step
        self._calendar_step = calendar_step
        self._rainflow: _NativeRainflowCounter = (
            _NativeRainflowCounter(initial_rainflow_state)
            if rainflow_kernel is None
            else _CompiledRainflowCounter(initial_rainflow_state, kernel=rainflow_kernel)
        )

    def step(self, day: DegradationDay) -> DegradationStep:
        cycles = self._rainflow.step(day)
        soh_after_cycle, cycle_degradation, self._fec = self._cycle_step(
            self._soh,
            cycles,
            fec_cum=self._fec,
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
        )
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
            fec_cum=self._fec,
        )
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
        self._rainflow.reset()

    def snapshot(self, *, day_start_soc: float, day_start_temperature_c: float) -> dict[str, Any]:
        return {
            "degradation_engine": "native",
            "soh_fraction": self._soh,
            "fec_cum": self._fec,
            "cumulative_calendar_seconds": self._calendar_seconds,
            "native_rainflow_state": self._rainflow.snapshot(),
            "day_start_soc_absolute": float(day_start_soc),
            "day_start_temperature_c": float(day_start_temperature_c),
        }

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
    ) -> None:
        from breos.degradation.engine import BlastEngine, build_endpoint_day

        self.model_key = model_key
        self._build_endpoint_day = build_endpoint_day
        state = dict(initial_state or {})
        engine_snapshot = state.get("blast_engine", state)
        self._engine: BlastEngine = (
            BlastEngine.from_snapshot(model_key, engine_snapshot) if engine_snapshot else BlastEngine(model_key)
        )
        # Floored as each step floors it, so a pack continued at zero health
        # cannot start a span below zero.
        self._soh = max(0.0, self._engine.soh())
        self._fec = float(initial_fec)
        self._calendar_seconds = float(initial_calendar_seconds)

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

    def snapshot(self, *, day_start_soc: float, day_start_temperature_c: float) -> dict[str, Any]:
        return {
            "degradation_engine": "blast",
            "fec_cum": self._fec,
            "cumulative_calendar_seconds": self._calendar_seconds,
            "blast_model": self.model_key,
            "blast_engine": self._engine.state_snapshot(),
            "day_start_soc_absolute": float(day_start_soc),
            "day_start_temperature_c": float(day_start_temperature_c),
        }

    def tracking_fields(self, step: DegradationStep) -> dict[str, Any]:
        return {
            "BLAST_Model": self.model_key,
            "BLAST_Degradation": step.engine_degradation,
        }
