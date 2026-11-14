"""Contract tests for the internal degradation lifecycle adapters."""

import numpy as np
import pandas as pd
import pytest

from breos.degradation.native import _detect_cycles_rainflow_arrays
from breos.degradation.protocol import (
    DegradationDay,
    NativeDegradationAdapter,
    _CompiledRainflowCounter,
    _NativeRainflowCounter,
)


def _day(*, temperature_c: float = 25.0) -> DegradationDay:
    index = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
    return DegradationDay(
        soc=np.asarray([0.9, 0.1]),
        time_ticks=index.asi8,
        ticks_per_second=1_000_000.0,
        temperature_c=np.asarray([temperature_c, temperature_c]),
        step_seconds=3600.0,
        start_soc=0.9,
        start_temperature_c=temperature_c,
    )


def test_native_adapter_implements_lifecycle_contract_and_snapshot_shape():
    def cycle_step(soh, cycles, *, fec_cum, **kwargs):
        assert len(cycles) == 1
        assert cycles[0]["doc"] == pytest.approx(0.8)
        return soh - 0.01, 0.01, fec_cum + 0.8

    def calendar_step(soh, *, cumulative_cal_seconds, **kwargs):
        return soh - 0.02, 0.02, cumulative_cal_seconds + 86400.0

    adapter = NativeDegradationAdapter(
        model_key="naumann_lam_field_calibrated",
        initial_soh_fraction=1.0,
        initial_fec=2.0,
        initial_calendar_seconds=86400.0,
        k0_fraction=1.0,
        activation_energy=1.0,
        soc_exponent=1.0,
        time_exponent=1.0,
        cycle_step=cycle_step,
        calendar_step=calendar_step,
    )

    step = adapter.step(_day())

    assert step.soh_fraction == pytest.approx(0.97)
    assert step.fec == pytest.approx(2.8)
    assert step.calendar_seconds == pytest.approx(172800.0)
    assert step.cycle_degradation == pytest.approx(0.01)
    assert step.calendar_degradation == pytest.approx(0.02)
    assert adapter.tracking_fields(step) == {}
    snapshot = adapter.snapshot(day_start_soc=0.1, day_start_temperature_c=25.0)
    assert snapshot["degradation_engine"] == "native"
    assert snapshot["soh_fraction"] == pytest.approx(0.97)
    assert snapshot["fec_cum"] == pytest.approx(2.8)
    assert snapshot["cumulative_calendar_seconds"] == pytest.approx(172800.0)
    # The cycle/calendar split is a per-step increment; the energy loop owns the running total.
    assert "cumulative_cycle_degradation" not in snapshot
    assert "cumulative_calendar_degradation" not in snapshot
    assert snapshot["native_rainflow_state"]["residue"] == []
    assert snapshot["day_start_soc_absolute"] == pytest.approx(0.1)

    adapter.reset()
    assert adapter.soh() == pytest.approx(1.0)
    reset = adapter.snapshot(day_start_soc=0.1, day_start_temperature_c=25.0)
    assert reset["fec_cum"] == 0.0
    assert reset["cumulative_calendar_seconds"] == 0.0
    assert reset["native_rainflow_state"]["current"] is None


def test_native_rainflow_stream_matches_nested_cross_midnight_whole_trace():
    trace = np.asarray([0.5, 0.2, 0.8, 0.4, 0.7, 0.3, 0.6, 0.25, 0.55, 0.1, 0.5])
    ticks_per_second = 1_000_000.0
    time_ticks = np.arange(len(trace), dtype=np.int64) * int(3600 * ticks_per_second)

    def cycle_step(soh, cycles, *, fec_cum, **kwargs):
        increment = sum(float(cycle["doc"]) * float(cycle["count"]) for cycle in cycles)
        return soh, 0.0, fec_cum + increment

    def calendar_step(soh, *, cumulative_cal_seconds, dt_days, **kwargs):
        return soh, 0.0, cumulative_cal_seconds + dt_days * 86400.0

    adapter_kwargs = {
        "model_key": "naumann_lam_field_calibrated",
        "initial_soh_fraction": 1.0,
        "initial_fec": 0.0,
        "initial_calendar_seconds": 0.0,
        "k0_fraction": 1.0,
        "activation_energy": 1.0,
        "soc_exponent": 1.0,
        "time_exponent": 1.0,
        "cycle_step": cycle_step,
        "calendar_step": calendar_step,
    }
    first = NativeDegradationAdapter(**adapter_kwargs)
    first_step = first.step(
        DegradationDay(
            soc=trace[1:6],
            time_ticks=time_ticks[1:6],
            ticks_per_second=ticks_per_second,
            temperature_c=np.full(5, 25.0),
            step_seconds=3600.0,
            start_soc=trace[0],
            start_temperature_c=25.0,
            finalize_cycles=False,
        )
    )
    snapshot = first.snapshot(day_start_soc=trace[5], day_start_temperature_c=25.0)
    continued = NativeDegradationAdapter(
        **{
            **adapter_kwargs,
            "initial_fec": snapshot["fec_cum"],
            "initial_rainflow_state": snapshot["native_rainflow_state"],
        }
    )
    second_step = continued.step(
        DegradationDay(
            soc=trace[6:],
            # App can replay the source-year timestamps on a later projection
            # year; cycle duration must continue from the carried timestep.
            time_ticks=time_ticks[1:6],
            ticks_per_second=ticks_per_second,
            temperature_c=np.full(5, 25.0),
            step_seconds=3600.0,
            start_soc=trace[5],
            start_temperature_c=25.0,
            finalize_cycles=True,
        )
    )

    observed = [*first_step.cycle_records, *second_step.cycle_records]
    expected = _detect_cycles_rainflow_arrays(trace, time_ticks, ticks_per_second)
    assert len(observed) == len(expected)
    assert [_cycle_fields(cycle) for cycle in observed] == [_cycle_fields(cycle) for cycle in expected]
    assert any(cycle["start_idx"] <= 5 and cycle["end_idx"] >= 6 for cycle in observed)
    assert second_step.fec == pytest.approx(sum(cycle["doc"] * cycle["count"] for cycle in expected))


_CYCLE_KEYS = ("start_idx", "end_idx", "doc", "mean_soc", "count", "mean_c_rate")


def _cycle_fields(cycle):
    return tuple(cycle[key] for key in _CYCLE_KEYS)


def _soc_trace(rng, kind, n):
    if kind == "walk":
        x = np.cumsum(rng.normal(0.0, 0.05, n))
        return 0.1 + 0.8 * (x - x.min()) / max(np.ptp(x), 1e-9)
    if kind == "daily":
        t = np.arange(n) / 24.0
        return 0.5 + 0.4 * np.sin(2 * np.pi * t) * rng.uniform(0.3, 1.0, n) + rng.normal(0.0, 0.01, n)
    # Dispatch-like: quantised moves that sit on both SOC limits.
    x = np.empty(n)
    x[0] = rng.uniform(0.1, 0.9)
    for i in range(1, n):
        x[i] = np.clip(round(x[i - 1] + rng.choice([-0.2, -0.1, 0.0, 0.0, 0.1, 0.2]), 3), 0.1, 0.9)
    return x


@pytest.mark.parametrize("kind", ["walk", "daily", "plateau"])
def test_native_rainflow_matches_the_reference_counter_bit_for_bit_across_restored_windows(kind):
    # The simulation counts cycles one window at a time and carries the open
    # turning points through a snapshot; the reference counts the whole trace
    # with the rainflow package. Every cycle must match exactly, and no
    # window may emit a cycle its own samples have not closed.
    rng = np.random.default_rng(20261019)
    ticks_per_second = 1_000_000.0
    for _ in range(40):
        n = int(rng.integers(30, 24 * 15))
        trace = np.asarray(_soc_trace(rng, kind, n), dtype=float)
        ticks = np.arange(n, dtype=np.int64) * int(3600 * ticks_per_second)
        cuts = sorted(set(rng.integers(2, n - 1, size=int(rng.integers(0, 10))).tolist()))
        bounds = [1, *cuts, n]
        counter = _NativeRainflowCounter()
        observed = []
        for start, stop in zip(bounds[:-1], bounds[1:]):
            counter = _NativeRainflowCounter(counter.snapshot())
            emitted = counter.step(
                DegradationDay(
                    soc=trace[start:stop],
                    time_ticks=ticks[start:stop],
                    ticks_per_second=ticks_per_second,
                    temperature_c=np.full(stop - start, 25.0),
                    step_seconds=3600.0,
                    start_soc=trace[start - 1],
                    start_temperature_c=25.0,
                    finalize_cycles=stop == n,
                )
            )
            assert all(cycle["end_idx"] < stop for cycle in emitted)
            observed.extend(emitted)

        expected = _detect_cycles_rainflow_arrays(trace, ticks, ticks_per_second)
        assert [_cycle_fields(cycle) for cycle in observed] == [_cycle_fields(cycle) for cycle in expected]


def test_native_rainflow_uses_plateau_end_and_counts_exact_one_percent_cycle():
    trace = np.asarray([0.5, 0.05, 0.05, 0.06, 0.05, 0.5])
    ticks = np.arange(len(trace), dtype=np.int64) * 3_600_000_000
    counter = _NativeRainflowCounter()
    observed = []
    for start, stop, final in ((0, 2, False), (2, 5, True)):
        observed.extend(
            counter.step(
                DegradationDay(
                    soc=trace[start + 1 : stop + 1],
                    time_ticks=ticks[start + 1 : stop + 1],
                    ticks_per_second=1_000_000.0,
                    temperature_c=np.full(stop - start, 25.0),
                    step_seconds=3600.0,
                    start_soc=trace[start],
                    start_temperature_c=25.0,
                    finalize_cycles=final,
                )
            )
        )

    expected = _detect_cycles_rainflow_arrays(trace, ticks, 1_000_000.0)
    assert [(cycle["start_idx"], cycle["end_idx"]) for cycle in observed] == [
        (cycle["start_idx"], cycle["end_idx"]) for cycle in expected
    ]
    assert observed[0]["doc"] == pytest.approx(0.01)


class _PerSampleRainflowCounter(_NativeRainflowCounter):
    """The counter's scan as a plain per-sample loop, the reference for its numpy scan."""

    def step(self, day):
        if day.soc.size == 0:
            return self.finish() if day.finalize_cycles else []
        if self._step_seconds is None:
            self._step_seconds = float(day.step_seconds)
        counter = self._counter
        closed = []
        if counter.current is None:
            anchor = (float(day.start_soc), 0)
            counter.points.append(anchor)
            counter.previous = anchor
            counter.current = (float(day.soc[0]), 1)
            counter.last_delta = counter.current[0] - anchor[0]
            counter.moved = counter.last_delta != 0.0
            counter.next_index = 2
            samples = day.soc[1:]
        else:
            samples = day.soc
        for soc in samples:
            next_point = (float(soc), counter.next_index)
            counter.next_index += 1
            if next_point[0] == counter.current[0]:
                counter.current = next_point
                continue
            next_delta = next_point[0] - counter.current[0]
            if counter.last_delta is not None and counter.last_delta * next_delta < 0:
                closed.extend(counter._push((counter.current,)))
            counter.previous = counter.current
            counter.current = next_point
            counter.last_delta = next_delta
            counter.moved = True
        cycles = self._format(closed)
        if day.finalize_cycles:
            cycles.extend(self.finish())
        return cycles


def _edge_trace(rng, kind, n):
    if kind == "walk":
        return np.clip(0.5 + np.cumsum(rng.normal(0, 0.05, n)), 0.05, 0.95)
    if kind == "plateau":
        return np.round(np.clip(0.5 + np.cumsum(rng.normal(0, 0.05, n)), 0.1, 0.9), 2)
    if kind == "ties":
        return rng.choice([0.2, 0.5, 0.8, 0.5000000000000001], n)
    if kind == "flat":
        return np.full(n, 0.6)
    if kind == "tiny":
        # Steps near 1e-200 whose products underflow to zero.
        return rng.choice([-2, -1, 0, 1, 2], n) * 1e-200
    if kind == "nonfinite":
        return rng.choice([0.2, 0.5, np.inf, -np.inf, np.nan, 1e308, -1e308], n)
    if kind == "damped":
        # Each swing smaller than the last: nothing closes and the residue keeps
        # every reversal, past the compiled counter's first array capacity.
        k = np.arange(n)
        return 0.5 + np.where(k % 2 == 0, 1.0, -1.0) * 0.45 * 0.999**k
    t = np.arange(n)
    return np.clip(0.5 + 0.45 * np.sin(2 * np.pi * t / 96 + rng.uniform(0, 6)) + rng.normal(0, 0.01, n), 0.1, 0.95)


def _numpy_counter(state=None):
    return _NativeRainflowCounter(state)


def _interpreted_kernel_counter(state=None):
    # The kernel's own source run by Python: pins its logic without Numba.
    # Interpreted, it computes on numpy scalars, which honour np.errstate;
    # compiled code ignores it, as the Python-float loop does.
    from breos.degradation._rainflow_kernel import rainflow_day

    def kernel(*args):
        with np.errstate(all="ignore"):
            return rainflow_day(*args)

    return _CompiledRainflowCounter(state, kernel=kernel)


def _compiled_kernel_counter(state=None):
    pytest.importorskip("numba")
    from breos._numba_dispatch import require_numba_rainflow_day

    return _CompiledRainflowCounter(state, kernel=require_numba_rainflow_day())


@pytest.mark.parametrize(
    "make_counter",
    [_numpy_counter, _interpreted_kernel_counter, _compiled_kernel_counter],
    ids=["numpy", "kernel-python", "kernel-numba"],
)
@pytest.mark.parametrize("raise_on_error", [False, True])
@pytest.mark.parametrize("kind", ["walk", "plateau", "ties", "flat", "tiny", "nonfinite", "damped", "battery"])
def test_native_rainflow_scan_matches_the_per_sample_loop_on_edge_cases(kind, raise_on_error, make_counter):
    # Plateaus, exact ties, steps that underflow when multiplied, flat days and
    # one to 96 samples a day, with restores mid-stream: every cycle and every
    # snapshot must equal the per-sample loop's, by repr, so a numpy scalar
    # leaking into the carried state fails too. The loop's Python floats never
    # raise, so the scan must not either under numpy's strictest error settings.
    rng = np.random.default_rng(20261103)
    for _ in range(80):
        steps_per_day = int(rng.choice([1, 2, 24, 96]))
        days = int(rng.integers(1, 8))
        trace = _edge_trace(rng, kind, steps_per_day * days)
        start = float(trace[0]) if rng.random() < 0.5 else float(np.round(rng.uniform(0.1, 0.9), 2))
        reference, counter = _PerSampleRainflowCounter(), make_counter()
        for d in range(days):
            soc = trace[d * steps_per_day : (d + 1) * steps_per_day]
            day = DegradationDay(
                soc=soc,
                time_ticks=np.arange(len(soc), dtype=np.int64),
                ticks_per_second=1.0,
                temperature_c=np.zeros(len(soc)),
                step_seconds=86400 / steps_per_day,
                start_soc=start,
                start_temperature_c=20.0,
                finalize_cycles=d == days - 1 and rng.random() < 0.7,
            )
            carried = reference._counter.current
            if carried is not None and not np.isclose(start, carried[0], rtol=0.0, atol=1e-9):
                # A carry the next day does not start from (NaN never does) is refused.
                with pytest.raises(ValueError, match="carry SOC"):
                    counter.step(day)
                break
            with np.errstate(all="raise" if raise_on_error else "ignore"):
                emitted = counter.step(day)
            assert repr(emitted) == repr(reference.step(day))
            assert repr(counter.snapshot()) == repr(reference.snapshot())
            if rng.random() < 0.2:
                counter = make_counter(counter.snapshot())
                reference = _PerSampleRainflowCounter(reference.snapshot())
            start = float(soc[-1]) if reference._counter.current is not None else start


def test_numba_backend_counts_native_rainflow_with_the_compiled_scan(monkeypatch):
    # The fingerprints cannot see a silent fallback to the numpy scan: it
    # returns the same bits. Pin the selection itself, down to the counter a
    # real battery and vehicle run build.
    pytest.importorskip("numba")
    from breos import battery
    from breos._numba_dispatch_kernels import _rainflow_day_kernel
    from breos.vehicle import lifecycle as vehicle_lifecycle

    assert battery._resolve_rainflow_kernel("python") is None
    assert battery._resolve_rainflow_kernel("numba") is _rainflow_day_kernel

    built = []
    real_battery, real_vehicle = battery._build_degradation_lifecycle, vehicle_lifecycle._build_vehicle_lifecycle

    def spy_battery(*args, **kwargs):
        result = real_battery(*args, **kwargs)
        built.append(("battery", type(result[0]._rainflow)))
        return result

    def spy_vehicle(*args, **kwargs):
        result = real_vehicle(*args, **kwargs)
        built.append(("vehicle", type(result._rainflow)))
        return result

    monkeypatch.setattr(battery, "_build_degradation_lifecycle", spy_battery)
    monkeypatch.setattr(vehicle_lifecycle, "_build_vehicle_lifecycle", spy_vehicle)
    from breos import App

    vehicle = {
        "mode": "v2h",
        "battery_kwh": 60,
        "departure_soc": 0.8,
        "charger_ac_rating_kw": 7.4,
        "charger_efficiency": 0.94,
        "battery_cost_per_kwh": 120,
        "charger_cost": 0,
        "schedule": {
            "weekday": {"arrive": "18:00", "depart": "08:00", "trip_kwh": 9},
            "weekend": {"arrive": "14:00", "depart": "10:00", "trip_kwh": 12},
        },
    }
    for backend, expected in (("numba", _CompiledRainflowCounter), ("python", _NativeRainflowCounter)):
        built.clear()
        App(
            {
                "location": "porto",
                "n_modules": 4,
                "annual_consumption_kwh": 3000,
                "battery_kwh": 5.0,
                "start_date": "2025-01-01",
                "projection_years": 1,
                "execution_backend": backend,
                "vehicle": vehicle,
            }
        ).simulate()
        assert built and {kind for kind, _ in built} == {"battery", "vehicle"}
        assert all(counter is expected for _, counter in built), (backend, [(k, c.__name__) for k, c in built])
