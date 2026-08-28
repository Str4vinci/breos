"""Static checks for forthcoming publication input and result-bundle verification tools."""

import hashlib
import json
from pathlib import Path

import pytest

from tools.preflight_article1_inputs import EXPECTED_SHA256, _checked_input, _checked_weather_metadata
from tools.verify_article1_bundle import (
    EXPECTED_HISTORICAL_WEATHER_SHA256,
    EXPECTED_MONTE_CARLO_YEARLY_COLUMNS,
    BundleAudit,
    _report_source_paths,
)

ARTICLE1_INTERVAL_MEAN_WEATHER_SHA256 = "71c26d072c09faf16dab37230cfe8b2d430bd39344333227d00c7be4e76a188a"
ARTICLE1_INTERVAL_MEAN_SIDECAR = (
    Path(__file__).parents[1]
    / "validation/article1/input-metadata/porto_historical_2005_2024_openmeteo.csv.metadata.json"
)


def test_article1_tools_pin_interval_mean_openmeteo_weather():
    assert EXPECTED_SHA256["historical_weather"] == ARTICLE1_INTERVAL_MEAN_WEATHER_SHA256
    assert EXPECTED_HISTORICAL_WEATHER_SHA256 == ARTICLE1_INTERVAL_MEAN_WEATHER_SHA256

    sidecar = json.loads(ARTICLE1_INTERVAL_MEAN_SIDECAR.read_text())
    metadata = sidecar["breos_weather_metadata"]
    assert sidecar["weather_sha256"] == ARTICLE1_INTERVAL_MEAN_WEATHER_SHA256
    assert metadata["radiation_time_basis"] == "interval_mean"
    assert metadata["timestamp_label_basis"] == "right"
    assert metadata["timestamp_timezone"] == "GMT"


def test_article1_input_preflight_accepts_only_pinned_hash(tmp_path, monkeypatch):
    path = tmp_path / "input.csv"
    path.write_text("value\n1\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setitem(EXPECTED_SHA256, "test_input", digest)

    record = _checked_input("test_input", path)

    assert record["sha256"] == digest
    assert record["size_bytes"] == path.stat().st_size


def test_article1_input_preflight_rejects_hash_drift(tmp_path, monkeypatch):
    path = tmp_path / "input.csv"
    path.write_text("changed\n")
    monkeypatch.setitem(EXPECTED_SHA256, "test_input", "0" * 64)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _checked_input("test_input", path)


def test_article1_preflight_requires_canonical_weather_timing(tmp_path):
    weather = tmp_path / "weather.csv"
    weather.write_text("date,shortwave_radiation\n2025-01-01,0\n")
    digest = hashlib.sha256(weather.read_bytes()).hexdigest()
    sidecar = tmp_path / "weather.csv.metadata.json"
    sidecar.write_text(
        "{\n"
        '  "schema_version": 1,\n'
        f'  "weather_sha256": "{digest}",\n'
        '  "breos_weather_metadata": {\n'
        '    "radiation_time_basis": "instant",\n'
        '    "timestamp_label_basis": "instant"\n'
        "  }\n"
        "}\n"
    )

    with pytest.raises(ValueError, match="wrong radiation time basis"):
        _checked_weather_metadata(
            weather,
            radiation_time_basis="interval_mean",
            timestamp_label_basis="right",
        )


def test_article1_bundle_audit_reports_missing_files(tmp_path):
    audit = BundleAudit(tmp_path)

    audit.require_file("missing.csv")

    assert audit.errors == ["missing file: missing.csv"]


def test_article1_bundle_checks_actual_montecarlo_weather_treatment(tmp_path):
    config = {}
    payload = {
        "breos_source": {"commit": "abc", "tracked_worktree_dirty": False},
        "resolved_pv_module": {
            "parameters": {"T_Pmax_pct": -0.34, "T_Voc_pct": -0.26},
            "width_m": 1.134,
            "length_m": 2.278,
        },
        "resolved_config": config,
        "resolved_config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest(),
        "effective_runtime_pv_model_options": {
            "transposition_model": "perez",
            "model_perez": "allsitescomposite1990",
            "iam_model": "ashrae",
            "diffuse_iam": "marion",
            "albedo": 0.25,
            "temperature_model": "faiman",
            "bifacial_model": "none",
            "solar_position": "weather",
        },
        "weather_metadata": {
            "radiation_time_basis": "interval_mean",
            "provider_hourly_fields": ["shortwave_radiation"],
            "timestamp_timezone": "GMT",
            "timestamp_label_basis": "right",
        },
        "effective_runtime_weather": {
            "solar_position_offset_minutes": 7.5,
            "metadata": {
                "radiation_time_basis": "interval_mean",
                "timestamp_label_basis": "left",
                "source_timestamp_label_basis": "right",
                "preserve_irradiance_energy": True,
            },
        },
    }
    audit = BundleAudit(tmp_path)

    audit._verify_common_provenance(payload, "monte-carlo-v1/c1/provenance.json")

    assert audit.errors == []


def test_article1_bundle_does_not_require_private_comparison_files(tmp_path):
    audit = BundleAudit(tmp_path)

    audit.verify()

    assert not any("external-validation" in error for error in audit.errors)


def test_article1_bundle_accepts_exact_monte_carlo_yearly_schema(tmp_path):
    path = tmp_path / "monte-carlo-v1/c1/yearly.csv"
    path.parent.mkdir(parents=True)
    path.write_text(",".join(EXPECTED_MONTE_CARLO_YEARLY_COLUMNS) + "\n")
    audit = BundleAudit(tmp_path)

    audit.verify_monte_carlo_yearly_schema("c1")

    assert audit.errors == []


@pytest.mark.parametrize(
    ("columns", "expected_error"),
    [
        (EXPECTED_MONTE_CARLO_YEARLY_COLUMNS[:-1], "missing ['Marginal_Grid_CI_gCO2_kWh']"),
        ((*EXPECTED_MONTE_CARLO_YEARLY_COLUMNS, "Extra"), "unexpected ['Extra']"),
        (
            (
                EXPECTED_MONTE_CARLO_YEARLY_COLUMNS[1],
                EXPECTED_MONTE_CARLO_YEARLY_COLUMNS[0],
                *EXPECTED_MONTE_CARLO_YEARLY_COLUMNS[2:],
            ),
            "column order differs",
        ),
    ],
)
def test_article1_bundle_rejects_monte_carlo_yearly_schema_drift(tmp_path, columns, expected_error):
    path = tmp_path / "monte-carlo-v1/c1/yearly.csv"
    path.parent.mkdir(parents=True)
    path.write_text(",".join(columns) + "\n")
    audit = BundleAudit(tmp_path)

    audit.verify_monte_carlo_yearly_schema("c1")

    assert any(expected_error in error for error in audit.errors)


def test_article1_source_compatibility_uses_the_generator_for_each_report():
    assert "tools/reproduce_article1.py" in _report_source_paths("base-v1/reproduction.json")
    assert "tools/reproduce_article1_context.py" in _report_source_paths("orientation/provenance.json")
    assert "tools/reproduce_article1_montecarlo.py" in _report_source_paths("monte-carlo-v1/c2/provenance.json")


def test_article1_bundle_accepts_unaffected_ancestor_outputs(tmp_path, monkeypatch):
    audit = BundleAudit(tmp_path)
    audit.reports = [
        (
            tmp_path / "base-v1/reproduction.json",
            {
                "breos_source": {"commit": "old"},
                "dependency_versions": {"numpy": "1"},
            },
        ),
        (
            tmp_path / "monte-carlo-v1/c2/provenance.json",
            {
                "breos_source": {"commit": "new"},
                "dependency_versions": {"numpy": "1"},
            },
        ),
    ]
    monkeypatch.setattr(audit, "_is_ancestor", lambda commit, target: commit in {"old", "new"} and target == "new")
    monkeypatch.setattr(audit, "_numerical_source_changed", lambda commit, target: False)
    monkeypatch.setattr(audit, "_source_object", lambda commit, path: f"unchanged:{path}")

    audit._verify_source_compatibility({"breos_source": {"commit": "new"}})

    assert audit.errors == []
    assert audit.source_commits == {"old", "new"}


def test_article1_bundle_accepts_reports_with_compatible_dependency_subsets(tmp_path, monkeypatch):
    audit = BundleAudit(tmp_path)
    audit.reports = [
        (
            tmp_path / "base-v1/reproduction.json",
            {
                "breos_source": {"commit": "old"},
                "dependency_versions": {"numpy": "1", "pymoo": "2"},
            },
        ),
        (
            tmp_path / "orientation/provenance.json",
            {
                "breos_source": {"commit": "new"},
                "dependency_versions": {"numpy": "1"},
            },
        ),
    ]
    monkeypatch.setattr(audit, "_is_ancestor", lambda commit, target: True)
    monkeypatch.setattr(audit, "_numerical_source_changed", lambda commit, target: False)
    monkeypatch.setattr(audit, "_source_object", lambda commit, path: f"same:{path}")

    audit._verify_source_compatibility({"breos_source": {"commit": "new"}})

    assert audit.errors == []


def test_article1_bundle_rejects_conflicting_dependency_versions(tmp_path, monkeypatch):
    audit = BundleAudit(tmp_path)
    audit.reports = [
        (
            tmp_path / "base-v1/reproduction.json",
            {
                "breos_source": {"commit": "old"},
                "dependency_versions": {"numpy": "1"},
            },
        ),
        (
            tmp_path / "orientation/provenance.json",
            {
                "breos_source": {"commit": "new"},
                "dependency_versions": {"numpy": "2"},
            },
        ),
    ]
    monkeypatch.setattr(audit, "_is_ancestor", lambda commit, target: True)
    monkeypatch.setattr(audit, "_numerical_source_changed", lambda commit, target: False)
    monkeypatch.setattr(audit, "_source_object", lambda commit, path: f"same:{path}")

    audit._verify_source_compatibility({"breos_source": {"commit": "new"}})

    assert any("conflicting dependency versions" in error for error in audit.errors)


def test_article1_bundle_rejects_changed_generator_for_ancestor_output(tmp_path, monkeypatch):
    audit = BundleAudit(tmp_path)
    audit.reports = [
        (
            tmp_path / "base-v1/reproduction.json",
            {"breos_source": {"commit": "old"}},
        )
    ]
    monkeypatch.setattr(audit, "_is_ancestor", lambda commit, target: True)
    monkeypatch.setattr(audit, "_numerical_source_changed", lambda commit, target: False)
    monkeypatch.setattr(
        audit,
        "_source_object",
        lambda commit, path: f"{commit}:{path}" if path == "tools/reproduce_article1.py" else f"same:{path}",
    )

    audit._verify_source_compatibility({"breos_source": {"commit": "new"}})

    assert any("tools/reproduce_article1.py differs" in error for error in audit.errors)


def test_article1_bundle_rejects_changed_numerical_source(tmp_path, monkeypatch):
    audit = BundleAudit(tmp_path)
    audit.reports = [
        (
            tmp_path / "base-v1/reproduction.json",
            {"breos_source": {"commit": "old"}},
        )
    ]
    monkeypatch.setattr(audit, "_is_ancestor", lambda commit, target: True)
    monkeypatch.setattr(audit, "_numerical_source_changed", lambda commit, target: True)
    monkeypatch.setattr(audit, "_source_object", lambda commit, path: f"same:{path}")

    audit._verify_source_compatibility({"breos_source": {"commit": "new"}})

    assert any("numerical source differs" in error for error in audit.errors)
