"""Parity against fixtures generated from unmodified BLAST-Lite v1.1.0."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from breos.degradation.engine import BLAST_MODEL_CLASSES, BlastEngine

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "blast" / "blast_parity_multicondition.json"
UPSTREAM_COMMIT = "d789e00bca60f628de640745c18eb724b07358bd"


@pytest.fixture(scope="module")
def parity_fixture():
    with FIXTURE_PATH.open(encoding="utf-8") as fixture_file:
        return json.load(fixture_file)


def test_fixture_records_exact_upstream_and_generation_environment(parity_fixture):
    assert parity_fixture["metadata"] == {
        "source": f"local untransformed BLAST-Lite checkout (blast/ at commit {UPSTREAM_COMMIT})",
        "numpy_version": "1.26.4",
        "python_version": "3.11.15",
        "days_per_condition": 60,
    }


@pytest.mark.parametrize("model_key", tuple(BLAST_MODEL_CLASSES))
def test_all_model_parameters_match_upstream(model_key, parity_fixture):
    expected = parity_fixture["parameters"][model_key]
    model = BLAST_MODEL_CLASSES[model_key]()

    assert model.cap == expected["cap"]
    assert model._params_life == expected["params_life"]
    assert model.experimental_range == expected["experimental_range"]


@pytest.mark.filterwarnings("ignore::breos.degradation.engine.BlastExperimentalRangeWarning")
@pytest.mark.parametrize("model_key", tuple(BLAST_MODEL_CLASSES))
@pytest.mark.parametrize(
    "condition_name",
    ("hot_storage", "cold_storage", "deep_cycle", "shallow_fast_cycle", "tvar_deep_cycle"),
)
def test_adapter_matches_upstream_across_conditions(model_key, condition_name, parity_fixture):
    condition = parity_fixture["conditions"][condition_name]
    profile = condition["profile"]
    engine = BlastEngine(model_key)
    q = []
    efc = []

    for _ in range(profile["days"]):
        q.append(engine.step(profile["time_s"], profile["soc"], profile["temperature_c"]))
        efc.append(float(engine.model.stressors["efc"][-1]))

    expected_q = condition["trajectories"][model_key]["q"]
    expected_efc = condition["trajectories"][model_key]["efc"]
    np.testing.assert_allclose(q, expected_q, rtol=0, atol=1e-12)
    np.testing.assert_allclose(efc, expected_efc, rtol=0, atol=1e-12)

    for output_key, expected in condition["final_outputs"][model_key].items():
        assert float(engine.model.outputs[output_key][-1]) == pytest.approx(expected, rel=0, abs=1e-12)
