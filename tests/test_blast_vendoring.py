"""Smoke and parity checks for the vendored BLAST-Lite Phase 0 source."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

import numpy as np

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "blast" / "blast_golden_soh_100d.json"
REPO_ROOT = Path(__file__).resolve().parents[1]
VENDORED_ROOT = REPO_ROOT / "breos" / "degradation" / "blast"
VENDORED_MANIFEST_PATH = VENDORED_ROOT / "VENDORED.md"

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

    assert set(available_models()) == {class_name for _, class_name in MODEL_CLASSES.values()}


def test_vendored_manifest_pins_provenance_and_current_result_hashes():
    manifest = VENDORED_MANIFEST_PATH.read_text()
    assert "d789e00bca60f628de640745c18eb724b07358bd" in manifest
    assert "b12e8f377a4b8d93901b54300acbbe2a1f987b95" in manifest

    result_entries = re.findall(
        r"^\| `[^`]+` \| `[0-9a-f]{64}` \| `([^`]+)` \| `([0-9a-f]{64})` \|",
        manifest,
        flags=re.MULTILINE,
    )
    assert len(result_entries) == 23
    assert {path for path, _ in result_entries} >= {
        "breos/degradation/blast/LICENSE",
        "breos/degradation/blast/NOTICE",
        "tests/fixtures/blast/blast_golden_soh_100d.json",
    }

    for relative_path, expected_sha256 in result_entries:
        result_path = REPO_ROOT / relative_path
        assert result_path.is_file(), relative_path
        assert hashlib.sha256(result_path.read_bytes()).hexdigest() == expected_sha256, relative_path


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
        np.testing.assert_allclose(actual["t_days"], expected["t_days"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(actual["efc"], expected["efc"], rtol=0, atol=1e-12)
