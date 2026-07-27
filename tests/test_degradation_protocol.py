"""Contract tests for the internal degradation lifecycle adapters."""

import numpy as np
import pandas as pd
import pytest

from breos.degradation.protocol import (
    BlastDegradationAdapter,
    DegradationDay,
    DegradationLifecycle,
    NativeDegradationAdapter,
)
from breos.degradation.validation import BlastExperimentalRangeWarning


def _day(*, temperature_c: float = 25.0) -> DegradationDay:
    index = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
    return DegradationDay(
        soc=pd.Series([0.9, 0.1], index=index),
        temperature_c=np.asarray([temperature_c, temperature_c]),
        step_seconds=3600.0,
        start_soc=0.9,
        start_temperature_c=temperature_c,
    )


def test_native_adapter_implements_lifecycle_contract_and_snapshot_shape():
    def cycle_step(soh, soc, nominal_energy_wh, *, fec_cum, **kwargs):
        assert len(soc) == 2
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
    assert adapter.snapshot(day_start_soc=0.1, day_start_temperature_c=25.0) == {
        "degradation_engine": "native",
        "fec_cum": pytest.approx(2.8),
        "cumulative_calendar_seconds": pytest.approx(172800.0),
        "cumulative_cycle_degradation": pytest.approx(0.11),
        "cumulative_calendar_degradation": pytest.approx(0.22),
    }

    adapter.reset()
    assert adapter.soh() == pytest.approx(1.0)
    reset = adapter.snapshot(day_start_soc=0.1, day_start_temperature_c=25.0)
    assert reset["fec_cum"] == 0.0
    assert reset["cumulative_calendar_seconds"] == 0.0


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
