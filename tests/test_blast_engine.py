"""Unit tests for the BLAST degradation engine adapter."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from breos.degradation.engine import (
    BLAST_MODEL_CLASSES,
    P1_BLAST_MODEL_KEYS,
    BlastAgingHorizonWarning,
    BlastEngine,
    BlastExperimentalRangeWarning,
    BlastNumericalError,
    build_endpoint_day,
)


def _daily_profile() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_secs = np.arange(25, dtype=float) * 3600.0
    soc = 0.55 + 0.05 * np.sin(np.linspace(0, 2 * np.pi, 25))
    temperature_c = np.full(25, 25.0)
    return t_secs, soc, temperature_c


def test_build_endpoint_day_spans_full_day():
    soc_samples = np.linspace(0.5, 0.6, 24)
    t_secs, soc, temperature_c = build_endpoint_day(
        step_seconds=3600,
        soc_samples=soc_samples,
        temperature_c_samples=25.0,
        start_soc=0.5,
        start_temperature_c=25.0,
    )

    assert len(t_secs) == 25
    assert t_secs[0] == 0
    assert t_secs[-1] == 86400
    assert len(soc) == 25
    assert len(temperature_c) == 25


def test_build_endpoint_day_rejects_mismatched_samples():
    with pytest.raises(ValueError, match="same length"):
        build_endpoint_day(
            step_seconds=3600,
            soc_samples=[0.5, 0.6],
            temperature_c_samples=[25.0, 26.0, 27.0],
            start_soc=0.5,
            start_temperature_c=25.0,
        )


def test_p1_models_step_and_snapshot():
    t_secs, soc, temperature_c = _daily_profile()

    for model_key in P1_BLAST_MODEL_KEYS:
        engine = BlastEngine(model_key)
        soh_day_1 = engine.step(t_secs, soc, temperature_c)
        assert np.isfinite(soh_day_1)
        assert 0.0 < soh_day_1 <= 1.0

        snapshot = engine.state_snapshot()
        restored = BlastEngine.from_snapshot(model_key, snapshot)
        soh_day_2 = restored.step(t_secs, soc, temperature_c)

        assert np.isfinite(soh_day_2)
        assert soh_day_2 <= soh_day_1
        assert restored.model.stressors["t_days"][-1] == pytest.approx(2.0)


def test_all_models_snapshot_restore_continuity():
    t_secs, soc, temperature_c = _daily_profile()

    for model_key in BLAST_MODEL_CLASSES:
        engine = BlastEngine(model_key)
        engine.step(t_secs, soc, temperature_c)

        restored = BlastEngine.from_snapshot(model_key, engine.state_snapshot())
        soh_restored = restored.step(t_secs, soc, temperature_c)
        soh_direct = engine.step(t_secs, soc, temperature_c)

        assert soh_restored == pytest.approx(soh_direct, abs=1e-12)
        assert restored.model.stressors["t_days"][-1] == pytest.approx(2.0)


def test_snapshot_restore_preserves_mid_swing_boundary_efc():
    hours = np.arange(49, dtype=float)
    soc = 0.5 + 0.3 * np.sin((hours - 3.0) * 2 * np.pi / 24.0)
    temperature_c = np.full(hours.shape, 25.0)

    day_1 = build_endpoint_day(
        step_seconds=3600,
        soc_samples=soc[1:25],
        temperature_c_samples=temperature_c[1:25],
        start_soc=soc[0],
        start_temperature_c=temperature_c[0],
    )
    day_2 = build_endpoint_day(
        step_seconds=3600,
        soc_samples=soc[25:49],
        temperature_c_samples=temperature_c[25:49],
        start_soc=soc[24],
        start_temperature_c=temperature_c[24],
    )
    assert abs(soc[25] - soc[24]) > 0.01

    continuous = BlastEngine("lfp_gr_250ah_prismatic")
    continuous.step(*day_1)
    q_after_day_1 = continuous.soh()
    continuous_soh = continuous.step(*day_2)

    split = BlastEngine("lfp_gr_250ah_prismatic")
    split.step(*day_1)
    restored = BlastEngine.from_snapshot("lfp_gr_250ah_prismatic", split.state_snapshot())
    split_soh = restored.step(*day_2)

    expected_efc = (np.abs(np.diff(soc[:25])).sum() / 2.0) + (np.abs(np.diff(soc[24:49])).sum() / 2.0) * q_after_day_1

    assert split_soh == pytest.approx(continuous_soh, abs=1e-12)
    assert restored.model.stressors["efc"][-1] == pytest.approx(
        continuous.model.stressors["efc"][-1],
        abs=1e-12,
    )
    assert continuous.model.stressors["efc"][-1] == pytest.approx(expected_efc, abs=1e-12)


def test_adapter_matches_vendored_simulate_battery_life_fixed_soc_storage():
    days = 100
    t_day = np.array([0.0, 86400.0])
    soc_day = np.array([0.55, 0.55])
    temperature_day_c = np.array([25.0, 25.0])
    input_day = {
        "Time_s": t_day,
        "SOC": soc_day,
        "Temperature_C": temperature_day_c,
    }

    for model_key, model_cls in BLAST_MODEL_CLASSES.items():
        engine = BlastEngine(model_key)
        adapter_soh = []
        for _ in range(days):
            adapter_soh.append(engine.step(t_day, soc_day, temperature_day_c))

        standalone = model_cls()
        for _ in range(days):
            standalone.simulate_battery_life(input_day)

        np.testing.assert_allclose(
            adapter_soh,
            standalone.outputs["q"][1:],
            rtol=0,
            atol=1e-6,
            err_msg=model_key,
        )


def test_experimental_range_warnings_deduplicate_across_snapshot_continuation():
    profile = (np.array([0.0, 86400.0]), np.array([0.5, 0.5]), np.array([55.0, 55.0]))
    engine = BlastEngine("lfp_gr_250ah_prismatic")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        engine.step(*profile)
        restored = BlastEngine.from_snapshot("lfp_gr_250ah_prismatic", json.loads(json.dumps(engine.state_snapshot())))
        restored.step(*profile)

    records = restored.warning_records("experimental_range")
    assert len([record for record in records if record["field"] == "temperature_c"]) == 1
    assert {record["field"] for record in records} == {"temperature_c"}
    assert len([item for item in caught if item.category is BlastExperimentalRangeWarning]) == 1
    # Constant-SOC storage has no cycling DOD; restoring emits no duplicate temperature warning.


def test_nonzero_shallow_dod_emits_experimental_range_warning():
    engine = BlastEngine("lfp_gr_250ah_prismatic")

    with pytest.warns(BlastExperimentalRangeWarning, match="dod range"):
        engine.step(
            np.array([0.0, 3600.0]),
            np.array([0.5, 0.6]),
            np.array([25.0, 25.0]),
        )

    records = engine.warning_records("experimental_range")
    assert [record["field"] for record in records] == ["dod"]
    assert records[0]["observed"] == [pytest.approx(0.1), pytest.approx(0.1)]
    assert records[0]["supported"] == [0.8, 1.0]


def test_sourced_aging_horizon_warns_once_and_survives_snapshot():
    profile = (
        np.array([0.0, 43200.0, 86400.0]),
        np.array([0.1, 0.9, 0.1]),
        np.array([25.0, 25.0, 25.0]),
    )
    engine = BlastEngine("nca_gr_panasonic_3ah")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(301):
            engine.step(*profile)
        restored = BlastEngine.from_snapshot("nca_gr_panasonic_3ah", engine.state_snapshot())
        restored.step(*profile)

    records = restored.warning_records("aging_horizon")
    assert len(records) == 1
    assert records[0]["supported_days"] == 300.0
    assert len([item for item in caught if item.category is BlastAgingHorizonWarning]) == 1


def test_all_models_stay_finite_when_daily_stressors_soften():
    """Deep-cycling days followed by light low-SOC days must not NaN.

    Regression: negative-exponent power states (nmc_lto_10ah) and shrinking
    sigmoid asymptotes (nca_grsi_sonymurata_2p5ah) overshoot their trajectory
    domain when day-to-day stressors soften abruptly, which real dispatch
    profiles produce routinely (e.g. the first calm winter day).
    """
    hours = np.arange(25, dtype=float)
    t_secs = hours * 3600.0
    temperature_c = np.full(25, 20.0)
    deep = np.interp(hours, [0, 10, 14, 17, 23, 24], [0.1, 0.1, 0.9, 0.9, 0.1, 0.1])
    light = np.interp(hours, [0, 10, 14, 17, 23, 24], [0.1, 0.1, 0.5, 0.5, 0.1, 0.1])

    for key in BLAST_MODEL_CLASSES:
        engine = BlastEngine(key)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for soc in (deep, deep, light, light):
                soh = engine.step(t_secs, soc, temperature_c)
        assert np.isfinite(soh), key
        for group_name in ("states", "outputs"):
            group = getattr(engine.model, group_name)
            for state_name, values in group.items():
                assert np.all(np.isfinite(values)), f"{key}.{group_name}.{state_name}"


def test_step_raises_on_non_finite_soh():
    engine = BlastEngine("lfp_gr_250ah_prismatic")
    t_secs, soc, temperature_c = _daily_profile()

    def corrupting_update(*_args):
        engine.model.outputs["q"] = np.append(engine.model.outputs["q"], np.nan)

    engine.model.update_battery_state = corrupting_update
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(BlastNumericalError, match="non-finite SoH"):
            engine.step(t_secs, soc, temperature_c)
