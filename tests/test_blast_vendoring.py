"""Smoke and parity checks for the vendored BLAST-Lite Phase 0 source."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "blast" / "blast_golden_soh_100d.json"
MULTICONDITION_FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "blast" / "blast_parity_multicondition.json"
)
VENDORED_ROOT = Path(__file__).resolve().parents[1] / "breos" / "degradation" / "blast"

MODEL_CLASSES = {
    "lfp_gr_250ah_prismatic": (
        "breos.degradation.blast.models",
        "Lfp_Gr_250AhPrismatic",
    ),
    "nca_gr_panasonic_3ah": (
        "breos.degradation.blast.models",
        "Nca_Gr_Panasonic3Ah_Battery",
    ),
    "lmo_gr_nissanleaf_66ah_2nd": (
        "breos.degradation.blast.models",
        "Lmo_Gr_NissanLeaf66Ah_2ndLife_Battery",
    ),
    "nmc811_grsi_lgm50_5ah": (
        "breos.degradation.blast.models",
        "Nmc811_GrSi_LGM50_5Ah_Battery",
    ),
    "nmc811_grsi_lgmj1_4ah": (
        "breos.degradation.blast.models",
        "Nmc811_GrSi_LGMJ1_4Ah_Battery",
    ),
    "nmc_gr_50ah_b1": (
        "breos.degradation.blast.models",
        "NMC_Gr_50Ah_B1",
    ),
    "nmc_gr_50ah_b2": (
        "breos.degradation.blast.models",
        "NMC_Gr_50Ah_B2",
    ),
    "nmc_gr_75ah_a": (
        "breos.degradation.blast.models",
        "NMC_Gr_75Ah_A",
    ),
    "nmc111_gr_sanyo_2ah": (
        "breos.degradation.blast.models",
        "Nmc111_Gr_Sanyo2Ah_Battery",
    ),
    "nmc_lto_10ah": (
        "breos.degradation.blast.models",
        "Nmc_Lto_10Ah_Battery",
    ),
    "lfp_gr_sonymurata_3ah": (
        "breos.degradation.blast.models",
        "Lfp_Gr_SonyMurata3Ah_Battery",
    ),
    "nca_grsi_sonymurata_2p5ah": (
        "breos.degradation.blast.models",
        "NCA_GrSi_SonyMurata2p5Ah_Battery",
    ),
    "nmc111_gr_kokam_75ah": (
        "breos.degradation.blast.models",
        "Nmc111_Gr_Kokam75Ah_Battery",
    ),
    "nmc622_gr_denso_50ah": (
        "breos.degradation.blast.models",
        "Nmc622_Gr_DENSO50Ah_Battery",
    ),
}


def _load_class(module_name: str, class_name: str):
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def _run_model(model_cls, fixture: dict) -> dict[str, list[float]]:
    t_secs = np.asarray(fixture["profile"]["time_s"], dtype=float)
    soc = np.asarray(fixture["profile"]["soc"], dtype=float)
    temperature_c = np.asarray(fixture["profile"]["temperature_c"], dtype=float)
    days = fixture["profile"]["days"]

    model = model_cls()
    soh: list[float] = []
    t_days: list[float] = []
    efc: list[float] = []

    for _ in range(days):
        model.update_battery_state(t_secs.copy(), soc.copy(), temperature_c.copy())
        soh.append(float(model.outputs["q"][-1]))
        t_days.append(float(model.stressors["t_days"][-1]))
        efc.append(float(model.stressors["efc"][-1]))

    return {"soh": soh, "t_days": t_days, "efc": efc}


def test_vendored_blast_imports_all_models():
    from breos.degradation.blast.models import available_models

    assert set(available_models()) == {
        class_name for _, class_name in MODEL_CLASSES.values()
    }


def test_vendored_blast_has_no_forbidden_heavy_imports_or_trapz():
    forbidden_imports = {
        "geopy",
        "h5pyd",
        "matplotlib",
        "matplotlib.pyplot",
        "pandas",
    }

    for path in VENDORED_ROOT.rglob("*.py"):
        source = path.read_text()
        assert "np.trapz" not in source
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = {alias.name for alias in node.names}
                assert imported.isdisjoint(forbidden_imports), path
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                assert node.module not in forbidden_imports, path


def test_vendored_blast_matches_golden_soh_fixture():
    fixture = json.loads(FIXTURE_PATH.read_text())
    assert fixture["schema"] == "blast-lite-breos-golden-soh-v1"
    assert set(fixture["trajectories"]) == set(MODEL_CLASSES)

    for model_key, (module_name, class_name) in MODEL_CLASSES.items():
        model_cls = _load_class(module_name, class_name)
        actual = _run_model(model_cls, fixture)
        expected = fixture["trajectories"][model_key]

        np.testing.assert_allclose(actual["soh"], expected["soh"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            actual["t_days"], expected["t_days"], rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(actual["efc"], expected["efc"], rtol=0, atol=1e-12)


def test_vendored_blast_parameters_match_untransformed_source():
    """Every model's literal parameters equal the untransformed-source dump."""

    fixture = json.loads(MULTICONDITION_FIXTURE_PATH.read_text())
    assert fixture["schema"] == "blast-lite-breos-parity-multicondition-v1"
    assert set(fixture["parameters"]) == set(MODEL_CLASSES)

    for model_key, (module_name, class_name) in MODEL_CLASSES.items():
        model = _load_class(module_name, class_name)()
        expected = fixture["parameters"][model_key]

        assert float(model.cap) == expected["cap"], model_key
        actual_params = {k: float(v) for k, v in model._params_life.items()}
        assert actual_params == expected["params_life"], model_key
        actual_range = json.loads(json.dumps(model.experimental_range, default=float))
        assert actual_range == expected["experimental_range"], model_key


def test_vendored_blast_matches_multicondition_parity_fixture():
    """Transform neutrality beyond the single-point golden fixture.

    The conditions activate hot/cold calendar terms, deep-DOD and higher-rate
    cycling terms, and intraday-varying temperature (the trapezoid-integration
    path rewritten from ``np.trapz``). Trajectories were generated from the
    untransformed source under numpy 1.x.
    """

    fixture = json.loads(MULTICONDITION_FIXTURE_PATH.read_text())
    days = fixture["metadata"]["days_per_condition"]

    for condition_name, condition in fixture["conditions"].items():
        profile = condition["profile"]
        t_secs = np.asarray(profile["time_s"], dtype=float)
        soc = np.asarray(profile["soc"], dtype=float)
        temperature_c = np.asarray(profile["temperature_c"], dtype=float)

        for model_key, (module_name, class_name) in MODEL_CLASSES.items():
            model = _load_class(module_name, class_name)()
            q, efc = [], []
            for _ in range(days):
                model.update_battery_state(
                    t_secs.copy(), soc.copy(), temperature_c.copy()
                )
                q.append(float(model.outputs["q"][-1]))
                efc.append(float(model.stressors["efc"][-1]))

            expected = condition["trajectories"][model_key]
            err = f"{condition_name}/{model_key}"
            np.testing.assert_allclose(q, expected["q"], rtol=0, atol=1e-12, err_msg=err)
            np.testing.assert_allclose(
                efc, expected["efc"], rtol=0, atol=1e-12, err_msg=err
            )

            expected_final = condition["final_outputs"][model_key]
            actual_final = {
                out_key: float(series[-1]) for out_key, series in model.outputs.items()
            }
            assert set(actual_final) == set(expected_final), err
            for out_key, expected_value in expected_final.items():
                assert actual_final[out_key] == pytest.approx(
                    expected_value, abs=1e-12
                ), f"{err}:{out_key}"
