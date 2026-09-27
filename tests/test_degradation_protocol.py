"""Contract tests for the internal degradation lifecycle adapters."""

import numpy as np
import pandas as pd
import pytest

from breos.battery import _detect_cycles_rainflow_arrays
from breos.degradation.protocol import (
    BlastDegradationAdapter,
    DegradationDay,
    DegradationLifecycle,
    NativeDegradationAdapter,
    _NativeRainflowCounter,
)
from breos.degradation.validation import BlastExperimentalRangeWarning


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
    def cycle_step(soh, cycles, nominal_energy_wh, *, fec_cum, **kwargs):
        assert len(cycles) == 1
        assert cycles[0]["doc"] == pytest.approx(0.8)
        assert nominal_energy_wh == 5000.0
        return soh - 0.01, 0.01, fec_cum + 0.8

    def calendar_step(soh, *, cumulative_cal_seconds, **kwargs):
        return soh - 0.02, 0.02, cumulative_cal_seconds + 86400.0

    adapter = NativeDegradationAdapter(
        model_key="naumann_lam_field_calibrated",
        initial_soh_fraction=1.0,
        initial_fec=2.0,
        initial_calendar_seconds=86400.0,
        initial_cumulative_cycle_degradation=0.1,
        initial_cumulative_calendar_degradation=0.2,
        nominal_energy_wh=5000.0,
        battery_type="lfp",
        k0_fraction=1.0,
        activation_energy=1.0,
        soc_exponent=1.0,
        time_exponent=1.0,
        cycle_step=cycle_step,
        calendar_step=calendar_step,
    )

    assert isinstance(adapter, DegradationLifecycle)
    step = adapter.step(_day())

    assert step.soh_fraction == pytest.approx(0.97)
    assert step.fec == pytest.approx(2.8)
    assert step.calendar_seconds == pytest.approx(172800.0)
    assert step.cycle_degradation == pytest.approx(0.01)
    assert step.calendar_degradation == pytest.approx(0.02)
    assert adapter.warnings() == []
    assert adapter.tracking_fields(step) == {}
    assert adapter.provenance().engine == "native"
    snapshot = adapter.snapshot(day_start_soc=0.1, day_start_temperature_c=25.0)
    assert snapshot["degradation_engine"] == "native"
    assert snapshot["soh_fraction"] == pytest.approx(0.97)
    assert snapshot["fec_cum"] == pytest.approx(2.8)
    assert snapshot["cumulative_calendar_seconds"] == pytest.approx(172800.0)
    assert snapshot["cumulative_cycle_degradation"] == pytest.approx(0.11)
    assert snapshot["cumulative_calendar_degradation"] == pytest.approx(0.22)
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

    def cycle_step(soh, cycles, nominal_energy_wh, *, fec_cum, **kwargs):
        increment = sum(float(cycle["doc"]) * float(cycle["count"]) for cycle in cycles)
        return soh, 0.0, fec_cum + increment

    def calendar_step(soh, *, cumulative_cal_seconds, dt_days, **kwargs):
        return soh, 0.0, cumulative_cal_seconds + dt_days * 86400.0

    adapter_kwargs = {
        "model_key": "naumann_lam_field_calibrated",
        "initial_soh_fraction": 1.0,
        "initial_fec": 0.0,
        "initial_calendar_seconds": 0.0,
        "initial_cumulative_cycle_degradation": 0.0,
        "initial_cumulative_calendar_degradation": 0.0,
        "nominal_energy_wh": 5000.0,
        "battery_type": "lfp",
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
    for actual, reference in zip(observed, expected, strict=True):
        assert (actual["start_idx"], actual["end_idx"]) == (reference["start_idx"], reference["end_idx"])
        np.testing.assert_allclose(
            [actual[key] for key in ("doc", "mean_soc", "count", "mean_c_rate")],
            [reference[key] for key in ("doc", "mean_soc", "count", "mean_c_rate")],
            rtol=0.0,
            atol=1e-12,
        )
    assert any(cycle["start_idx"] <= 5 and cycle["end_idx"] >= 6 for cycle in observed)
    assert second_step.fec == pytest.approx(sum(cycle["doc"] * cycle["count"] for cycle in expected))


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


def test_blast_adapter_implements_lifecycle_restore_warning_and_reset_contract():
    adapter = BlastDegradationAdapter("lfp_gr_250ah_prismatic")
    assert isinstance(adapter, DegradationLifecycle)

    with pytest.warns(BlastExperimentalRangeWarning):
        step = adapter.step(_day(temperature_c=55.0))

    assert step.soh_fraction < 1.0
    assert step.cycle_degradation == 0.0
    assert step.calendar_degradation == 0.0
    assert adapter.tracking_fields(step) == {
        "BLAST_Model": "lfp_gr_250ah_prismatic",
        "BLAST_Degradation": step.engine_degradation,
    }
    assert adapter.provenance().engine == "blast"
    assert adapter.provenance().state_schema_version == "1.0"
    assert [record["category"] for record in adapter.warnings()] == ["experimental_range"]

    snapshot = adapter.snapshot(day_start_soc=0.1, day_start_temperature_c=55.0)
    restored = BlastDegradationAdapter(
        "lfp_gr_250ah_prismatic",
        initial_state=snapshot,
        initial_fec=snapshot["fec_cum"],
        initial_calendar_seconds=snapshot["cumulative_calendar_seconds"],
        initial_cumulative_cycle_degradation=snapshot["cumulative_cycle_degradation"],
        initial_cumulative_calendar_degradation=snapshot["cumulative_calendar_degradation"],
    )
    assert restored.soh() == pytest.approx(adapter.soh())
    assert restored.warnings() == adapter.warnings()

    restored.reset()
    assert restored.soh() == pytest.approx(1.0)
    assert restored.warnings() == adapter.warnings()
    reset = restored.snapshot(day_start_soc=0.9, day_start_temperature_c=25.0)
    assert reset["blast_engine"]["outputs"]["q"][-1] == pytest.approx(1.0)
    assert reset["blast_engine"]["stressors"]["efc"][-1] == pytest.approx(0.0)
