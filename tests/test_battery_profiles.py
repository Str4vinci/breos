"""Public discovery, metadata, and configuration semantics for BLAST models."""

import json

import pytest

import breos
from breos.app_config import resolve_app_config
from breos.degradation.engine import BLAST_MODEL_CLASSES, BlastEngine
from breos.degradation.profiles import (
    BATTERY_MODEL_REGISTRY,
    BLAST_STATE_SCHEMA_VERSION,
    CORE_BLAST_MODEL_KEYS,
    get_battery_model_profile,
    merge_battery_config_layers,
)


def _base_config(**overrides):
    return {
        "location": "porto",
        "n_modules": 10,
        "annual_consumption_kwh": 4000,
        "battery_kwh": 5.0,
        **overrides,
    }


def test_registry_is_the_single_catalog_for_all_vendored_models():
    assert len(BATTERY_MODEL_REGISTRY) == 14
    assert tuple(BLAST_MODEL_CLASSES) == tuple(BATTERY_MODEL_REGISTRY)
    assert CORE_BLAST_MODEL_KEYS == ("lfp_gr_250ah_prismatic", "nca_gr_panasonic_3ah")

    for key, profile in BATTERY_MODEL_REGISTRY.items():
        model = BLAST_MODEL_CLASSES[key]()
        assert profile.nominal_capacity_ah == model.cap
        assert profile.experimental_range["cycling_temperature_c"] == model.experimental_range["cycling_temperature"]
        assert profile.experimental_range["dod"] == model.experimental_range["dod"]
        assert profile.experimental_range["soc"] == model.experimental_range["soc"]
        assert profile.experimental_range["max_c_rate_charge"] == model.experimental_range["max_rate_charge"]
        assert profile.experimental_range["max_c_rate_discharge"] == model.experimental_range["max_rate_discharge"]
        assert profile.output_keys == tuple(model.outputs)
        assert profile.citations
        assert profile.operating_defaults == {}


def test_python_discovery_is_public_and_json_serializable():
    models = breos.list_battery_models()
    assert len(models) == 14
    assert get_battery_model_profile("lfp_gr_250ah_prismatic").supports_capacity is True
    assert get_battery_model_profile("nmc111_gr_sanyo_2ah").supports_resistance is True
    assert models[0]["calibration_basis"] == "cell-model"
    assert models[0]["pack_calibrated"] is False
    json.dumps(models)


def test_config_precedence_is_user_then_profile_then_global():
    resolved = merge_battery_config_layers(
        {"battery_min_soc": 0.1, "battery_max_soc": 0.9, "source": "global"},
        {"battery_min_soc": 0.2, "source": "profile"},
        {"battery_min_soc": 0.3},
    )
    assert resolved == {"battery_min_soc": 0.3, "battery_max_soc": 0.9, "source": "profile"}


def test_native_is_default_and_blast_is_explicit_opt_in():
    native = resolve_app_config(_base_config()).cfg
    assert native["degradation_engine"] == "native"
    assert native["blast_model"] is None

    blast = resolve_app_config(_base_config(degradation_engine="blast", blast_model="lfp_gr_250ah_prismatic")).cfg
    assert blast["degradation_engine"] == "blast"
    assert blast["blast_model"] == "lfp_gr_250ah_prismatic"


def test_config_rejects_ambiguous_or_incomplete_model_selection():
    with pytest.raises(ValueError, match="ambiguous legacy selector"):
        resolve_app_config(_base_config(battery_type="lfp"))
    with pytest.raises(ValueError, match="requires degradation_engine='blast'"):
        resolve_app_config(_base_config(blast_model="lfp_gr_250ah_prismatic"))
    with pytest.raises(ValueError, match="Available"):
        resolve_app_config(_base_config(degradation_engine="blast", blast_model="nmc622_gr_denso_50ah"))


def test_snapshot_schema_survives_json_round_trip_and_rejects_unknown_versions():
    engine = BlastEngine("lfp_gr_250ah_prismatic")
    snapshot = json.loads(json.dumps(engine.state_snapshot()))
    assert snapshot["schema_version"] == BLAST_STATE_SCHEMA_VERSION
    restored = BlastEngine.from_snapshot("lfp_gr_250ah_prismatic", snapshot)
    assert restored.soh() == pytest.approx(1.0)

    snapshot["schema_version"] = "999"
    with pytest.raises(ValueError, match="Unsupported BLAST state schema"):
        BlastEngine.from_snapshot("lfp_gr_250ah_prismatic", snapshot)
