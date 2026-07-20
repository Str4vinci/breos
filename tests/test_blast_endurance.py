"""All-model, multi-year BLAST endurance test (slow).

Steps a committed, deterministic synthetic "field year"
(``tests/fixtures/blast/synthetic_field_year.json``, produced by
``tools/generate_synthetic_field_year.py``) daily for 20 repeated years through
every ``BlastEngine`` model.

The fixture combines seasonal depth-of-discharge variation, winter idle spells,
partial cycles, irregular day-to-day depth changes, and temperature seasonality
bounded to [5, 35] C. Those softening day-to-day transitions drive BLAST states
past their trajectory-inversion domain, which:

- previously (unguarded, pre-``ba964c9``) returned NaN for nca_grsi around
  year 9 — caught here by the per-step finiteness assertions;
- with the ``ba964c9`` sigmoid clamp, snapped an accumulated loss *down* to
  ``y_inf - y0`` (a negative increment), recovering capacity — caught here by
  the nca_grsi SOH-monotonicity and loss-state assertions.

The recorded regression bounds below are read off the committed fixture with
the corrected model. They are deterministic degradation endpoints, not
clip-satisfiable [0, 1] guards: a spurious recovery, NaN, or clamp regression
moves a model off its endpoint and fails the test.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from breos.degradation.engine import BLAST_MODEL_CLASSES, BlastEngine

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "blast" / "synthetic_field_year.json"
REPEAT_YEARS = 20

# Final SOH fraction after 20 repeated field years, per model, on the committed
# fixture. Tolerance is tight enough that a recovery/NaN/clamp regression fails,
# loose enough to absorb platform floating-point drift over ~7300 daily steps.
EXPECTED_FINAL_SOH = {
    "lfp_gr_250ah_prismatic": 0.880216,
    "nca_gr_panasonic_3ah": 0.774276,
    "lmo_gr_nissanleaf_66ah_2nd": 0.425957,
    "nmc811_grsi_lgm50_5ah": 0.596662,
    "nmc811_grsi_lgmj1_4ah": 0.038059,
    "nmc_gr_50ah_b1": 0.741789,
    "nmc_gr_50ah_b2": 0.869392,
    "nmc_gr_75ah_a": 0.823992,
    "nmc111_gr_sanyo_2ah": 0.460749,
    "nmc_lto_10ah": 0.982370,
    "lfp_gr_sonymurata_3ah": 0.835267,
    "nca_grsi_sonymurata_2p5ah": 0.685712,
    "nmc111_gr_kokam_75ah": 0.887791,
    "nmc622_gr_denso_50ah": 0.809839,
}
SOH_ABS_TOL = 3e-3


def _load_field_year() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fixture = json.loads(FIXTURE_PATH.read_text())
    assert fixture["schema"] == "blast-breos-synthetic-field-year-v1"
    t_secs = np.asarray(fixture["t_secs"], dtype=float)
    soc_days = np.asarray(fixture["soc_days"], dtype=float)
    temperature_days = np.asarray(fixture["temperature_days"], dtype=float)
    assert soc_days.shape == (fixture["days"], fixture["hours"])
    assert temperature_days.shape == soc_days.shape
    # Temperature seasonality is bounded to the intended [5, 35] C window.
    assert temperature_days.min() >= 5.0 - 1e-9
    assert temperature_days.max() <= 35.0 + 1e-9
    return t_secs, soc_days, temperature_days


@pytest.mark.slow
def test_all_blast_models_endure_twenty_field_years():
    t_secs, soc_days, temperature_days = _load_field_year()
    days = soc_days.shape[0]

    assert set(EXPECTED_FINAL_SOH) == set(BLAST_MODEL_CLASSES)

    for model_key in BLAST_MODEL_CLASSES:
        engine = BlastEngine(model_key)
        soh_history = np.empty(days * REPEAT_YEARS)
        step = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _year in range(REPEAT_YEARS):
                for day in range(days):
                    soh = engine.step(t_secs, soc_days[day], temperature_days[day])
                    # Returned SOH and every newest state/output value are finite.
                    assert np.isfinite(soh), f"{model_key} step {step} returned non-finite SOH"
                    for group_name in ("states", "outputs"):
                        for field, values in getattr(engine.model, group_name).items():
                            assert np.isfinite(values[-1]), f"{model_key}.{group_name}.{field} step {step}"
                    soh_history[step] = soh
                    step += 1

        final_soh = soh_history[-1]
        expected = EXPECTED_FINAL_SOH[model_key]
        # Observed SOH stays within the recorded fixture regression bounds.
        assert final_soh == pytest.approx(expected, abs=SOH_ABS_TOL), model_key
        assert soh_history.min() >= expected - SOH_ABS_TOL, model_key
        # No spurious capacity creation beyond initial full health.
        assert soh_history.max() <= 1.0 + 1e-9, model_key


@pytest.mark.slow
def test_nca_grsi_soh_and_sigmoid_states_never_recover():
    """nca_grsi SOH must never recover and its sigmoid losses never decrease.

    This is the direct ba964c9 regression: the shrinking-asymptote guard first
    activates around year 9 on this fixture, and the pre-fix clamp turned the
    zeroed increment into a negative one, recovering capacity.
    """
    t_secs, soc_days, temperature_days = _load_field_year()
    days = soc_days.shape[0]

    engine = BlastEngine("nca_grsi_sonymurata_2p5ah")
    soh_history: list[float] = []
    q_loss_t: list[float] = []
    q_loss_efc: list[float] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _year in range(REPEAT_YEARS):
            for day in range(days):
                soh_history.append(engine.step(t_secs, soc_days[day], temperature_days[day]))
                q_loss_t.append(float(engine.model.states["qLoss_t"][-1]))
                q_loss_efc.append(float(engine.model.states["qLoss_EFC"][-1]))

    soh = np.asarray(soh_history)
    d_soh = np.diff(soh)
    # SOH is monotonically non-increasing (never recovers) within FP tolerance.
    assert d_soh.max() <= 1e-9, f"nca_grsi SOH recovered by {d_soh.max():.3e}"
    # The sigmoid loss states are monotonically non-decreasing (never snap down).
    assert np.diff(np.asarray(q_loss_t)).min() >= -1e-12
    assert np.diff(np.asarray(q_loss_efc)).min() >= -1e-12
    # The fixture actually exercises meaningful degradation over the horizon.
    assert soh[-1] == pytest.approx(EXPECTED_FINAL_SOH["nca_grsi_sonymurata_2p5ah"], abs=SOH_ABS_TOL)
    assert soh[0] - soh[-1] > 0.25
