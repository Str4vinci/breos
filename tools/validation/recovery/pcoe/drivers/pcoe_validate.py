"""Reproduce the PCoE PV system test-bed validation.

The run validates the BREOS thermal component against measured module
temperature and checks the internal consistency of the recorded electrical
channels. It does not invent a module or inverter nameplate for a full PV
power run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

# Raw inputs live outside the repository because of size and licensing. Point
# BREOS_VALIDATION_DATA at the directory holding the downloaded datasets; the
# README records which archive belongs where.
DATA_ROOT = Path(os.environ.get("BREOS_VALIDATION_DATA", "datasets")).expanduser()
SOURCE_FILE = DATA_ROOT / "pcoe_pv_testbed" / "pcoe_pv_system_testbed.csv"
OUTPUT = Path(
    os.environ.get(
        "BREOS_VALIDATION_OUTPUT",
        "results/validation_pcoe_recovered",
    )
).expanduser()
DEFAULT_BREOS_ROOT = Path("/tmp/breos-article1-0.6.0")
BREOS_ROOT = Path(os.environ.get("BREOS_VALIDATION_ROOT", DEFAULT_BREOS_ROOT))
sys.path.insert(0, str(BREOS_ROOT))

from breos.inverter import calculate_dc_ac_power  # noqa: E402
from breos.pv.temperature import calculate_cell_temperature  # noqa: E402

THERMAL_COLUMNS = ["Gpoa", "Tamb", "WS", "Tmod"]
ELECTRICAL_COLUMNS = ["Vdc", "Idc", "Pac", "Pdc"]
THRESHOLDS = (0, 50, 200, 400)
THERMAL_MODELS = {
    "breos_faiman_default": "faiman",
    "breos_pvsyst_freestanding": "pvsyst-freestanding",
}
OUTPUT_FILES = (
    "thermal_metrics.csv",
    "thermal_sensitivity_metrics.csv",
    "monthly_bias.csv",
    "electrical_checks.json",
    "input_manifest.sha256",
    "run_config.json",
    "dataset_facts.json",
    "provenance.json",
    "README.md",
)
ZENODO_MD5 = "35386ee96c1a57d10f8d73dfacee858b"


def _git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(BREOS_ROOT), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _dependency_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_row(
    analysis: str,
    model: str,
    threshold: int,
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float | int | str]:
    error = predicted - actual
    return {
        "analysis": analysis,
        "model": model,
        "gpoa_threshold_W_m2": threshold,
        "n": int(actual.size),
        "bias_C": float(np.mean(error)),
        "mae_C": float(np.mean(np.abs(error))),
        "rmse_C": float(np.sqrt(np.mean(error**2))),
        "r": float(np.corrcoef(actual, predicted)[0, 1]),
    }


def _metric_rows(
    analysis: str,
    threshold: int,
    actual: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> list[dict[str, float | int | str]]:
    return [_metric_row(analysis, model, threshold, actual, prediction) for model, prediction in predictions.items()]


def _power_consistency_row(
    population: str,
    actual: np.ndarray,
    derived: np.ndarray,
) -> dict[str, float | int | str]:
    error = derived - actual
    return {
        "population": population,
        "n": int(actual.size),
        "bias_derived_minus_Pdc_W": float(np.mean(error)),
        "mae_W": float(np.mean(np.abs(error))),
        "rmse_W": float(np.sqrt(np.mean(error**2))),
        "r": float(np.corrcoef(actual, derived)[0, 1]),
        "abs_error_gt_10W": int(np.sum(np.abs(error) > 10)),
        "relative_error_gt_1pct": int(np.sum(np.abs(error) / np.maximum(np.abs(actual), 1e-12) > 0.01)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PCoE PV system test-bed validation.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace the existing generated files in the result directory",
    )
    args = parser.parse_args()

    existing = [OUTPUT / name for name in OUTPUT_FILES if (OUTPUT / name).exists()]
    if existing and not args.force:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"Refusing to replace existing result files: {names}. Use --force to replace them.")

    if not SOURCE_FILE.is_file():
        raise FileNotFoundError(SOURCE_FILE)

    data = pd.read_csv(SOURCE_FILE)
    expected_columns = [
        "time_stamp",
        *THERMAL_COLUMNS,
        "RH",
        "Waplha",
        "Rain",
        "AzS",
        "Als",
        "kt",
        "ET_GHI",
        *ELECTRICAL_COLUMNS,
    ]
    if set(data.columns) != set(expected_columns):
        raise ValueError(f"Unexpected columns: {list(data.columns)}")
    data.index = pd.to_datetime(
        data.pop("time_stamp"),
        format="%d-%m-%Y %H:%M:%S",
        errors="coerce",
    )
    if data.index.isna().any():
        raise ValueError("Unparseable timestamps in the PCoE file")
    data = data.sort_index()
    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    missing_counts = {column: int(data[column].isna().sum()) for column in data.columns}
    deltas = data.index.to_series().diff().dropna().dt.total_seconds().div(60)
    delta_counts = {str(int(minutes)): int(count) for minutes, count in deltas.value_counts().sort_index().items()}

    valid = data[THERMAL_COLUMNS].dropna()
    valid = valid[(valid["Gpoa"] >= 0) & (valid["WS"] >= 0)].copy()
    valid["delta_minutes"] = valid.index.to_series().diff().dt.total_seconds().div(60).to_numpy()
    valid["gpoa_change_W_m2"] = valid["Gpoa"].diff().abs()
    valid["stable"] = (valid["delta_minutes"] == 15) & (valid["gpoa_change_W_m2"] <= 25)
    valid["consecutive_15min"] = valid["delta_minutes"] == 15

    predictions: dict[str, np.ndarray] = {}
    for label, model in THERMAL_MODELS.items():
        predictions[label] = np.asarray(
            calculate_cell_temperature(
                valid["Gpoa"],
                valid["Tamb"],
                valid["WS"],
                model,
            ),
            dtype=float,
        )
        valid[f"{label}_prediction"] = predictions[label]
        valid[f"{label}_error"] = predictions[label] - valid["Tmod"].to_numpy(dtype=float)

    actual_temperature = valid["Tmod"].to_numpy(dtype=float)
    thermal_rows: list[dict[str, float | int | str]] = []
    for threshold in THRESHOLDS:
        threshold_mask = valid["Gpoa"].to_numpy(dtype=float) >= threshold
        thermal_rows.extend(
            _metric_rows(
                "all_valid",
                threshold,
                actual_temperature[threshold_mask],
                {label: prediction[threshold_mask] for label, prediction in predictions.items()},
            )
        )

    sensitivity_rows: list[dict[str, float | int | str]] = []
    previous_inputs = valid[["Gpoa", "Tamb", "WS"]].shift(1)
    for label, model in THERMAL_MODELS.items():
        lag_prediction = np.asarray(
            calculate_cell_temperature(
                previous_inputs["Gpoa"],
                previous_inputs["Tamb"],
                previous_inputs["WS"],
                model,
            ),
            dtype=float,
        )
        valid[f"{label}_lag_prediction"] = lag_prediction
        valid[f"{label}_lag_error"] = lag_prediction - actual_temperature

    for threshold in THRESHOLDS:
        threshold_mask = valid["Gpoa"].to_numpy(dtype=float) >= threshold
        stable_mask = threshold_mask & valid["stable"].to_numpy(dtype=bool)
        sensitivity_rows.extend(
            _metric_rows(
                "stable_current_inputs",
                threshold,
                actual_temperature[stable_mask],
                {label: prediction[stable_mask] for label, prediction in predictions.items()},
            )
        )
        lag_mask = threshold_mask & valid["consecutive_15min"].to_numpy(dtype=bool)
        for label in THERMAL_MODELS:
            lag_prediction = valid[f"{label}_lag_prediction"].to_numpy(dtype=float)
            finite_mask = lag_mask & np.isfinite(lag_prediction)
            sensitivity_rows.append(
                _metric_row(
                    "equilibrium_inputs_from_previous_15min",
                    label,
                    threshold,
                    actual_temperature[finite_mask],
                    lag_prediction[finite_mask],
                )
            )

    monthly_rows: list[dict[str, float | int | str]] = []
    primary = valid[valid["Gpoa"] >= 200]
    for month, month_data in primary.groupby(primary.index.month):
        for label in THERMAL_MODELS:
            error = month_data[f"{label}_error"].to_numpy(dtype=float)
            monthly_rows.append(
                {
                    "model": label,
                    "month": int(month),
                    "month_name": month_data.index[0].strftime("%B"),
                    "n": int(error.size),
                    "bias_C": float(np.mean(error)),
                    "mae_C": float(np.mean(np.abs(error))),
                    "rmse_C": float(np.sqrt(np.mean(error**2))),
                }
            )

    pdc = data["Pdc"].to_numpy(dtype=float)
    pdc_from_channels = data["Vdc"].to_numpy(dtype=float) * data["Idc"].to_numpy(dtype=float)
    pdc_checks = [
        _power_consistency_row("all_rows", pdc, pdc_from_channels),
        _power_consistency_row("Pdc_gt_100W", pdc[pdc > 100], pdc_from_channels[pdc > 100]),
    ]

    operating = data["Pdc"].to_numpy(dtype=float) > 100
    ac_dc_ratio = data.loc[operating, "Pac"].to_numpy(dtype=float) / pdc[operating]
    night = data["Gpoa"].to_numpy(dtype=float) == 0
    electrical_checks = {
        "pdc_vs_vdc_times_idc": pdc_checks,
        "pac_over_pdc_when_Pdc_gt_100W": {
            "n": int(ac_dc_ratio.size),
            "median": float(np.median(ac_dc_ratio)),
            "mean": float(np.mean(ac_dc_ratio)),
            "p05": float(np.quantile(ac_dc_ratio, 0.05)),
            "p95": float(np.quantile(ac_dc_ratio, 0.95)),
            "min": float(np.min(ac_dc_ratio)),
            "max": float(np.max(ac_dc_ratio)),
            "Pac_gt_Pdc": int(np.sum(data.loc[operating, "Pac"].to_numpy() > pdc[operating])),
        },
        "nighttime_Pac_when_Gpoa_eq_0": {
            "n": int(np.sum(night)),
            "median_W": float(np.median(data.loc[night, "Pac"])),
            "p05_W": float(np.quantile(data.loc[night, "Pac"], 0.05)),
            "p95_W": float(np.quantile(data.loc[night, "Pac"], 0.95)),
            "Pdc_gt_0W": int(np.sum(data.loc[night, "Pdc"] > 0)),
        },
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(thermal_rows).to_csv(
        OUTPUT / "thermal_metrics.csv",
        index=False,
        float_format="%.9f",
    )
    pd.DataFrame(sensitivity_rows).to_csv(
        OUTPUT / "thermal_sensitivity_metrics.csv",
        index=False,
        float_format="%.9f",
    )
    pd.DataFrame(monthly_rows).to_csv(
        OUTPUT / "monthly_bias.csv",
        index=False,
        float_format="%.9f",
    )
    (OUTPUT / "electrical_checks.json").write_text(json.dumps(electrical_checks, indent=2) + "\n", encoding="utf-8")

    input_manifest = OUTPUT / "input_manifest.sha256"
    input_manifest.write_text(f"{_sha256(SOURCE_FILE)}  {SOURCE_FILE.name}\n", encoding="utf-8")

    config = {
        "source_file": str(SOURCE_FILE),
        "breos_root": str(BREOS_ROOT),
        "timestamp_format": "%d-%m-%Y %H:%M:%S",
        "thermal_columns": THERMAL_COLUMNS,
        "electrical_columns": ELECTRICAL_COLUMNS,
        "valid_filter": {
            "non_null": THERMAL_COLUMNS,
            "Gpoa_min_W_m2": 0,
            "WS_min_m_s": 0,
        },
        "thresholds_W_m2": list(THRESHOLDS),
        "stable_period": {
            "consecutive_interval_minutes": 15,
            "max_absolute_Gpoa_change_W_m2": 25,
        },
        "thermal_models": THERMAL_MODELS,
        "primary_threshold_W_m2": 200,
        "electrical_consistency": "Pdc compared with Vdc multiplied by Idc; Pac/Pdc is descriptive because inverter nameplate metadata is absent",
    }
    config_path = OUTPUT / "run_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    facts = {
        "source_file": str(SOURCE_FILE),
        "source_filename_on_zenodo": "PCoE-Dataset PV system test-bed.csv",
        "source_record": "https://zenodo.org/records/15779578",
        "source_doi": "10.5281/zenodo.15779578",
        "source_md5_local": _md5(SOURCE_FILE),
        "source_md5_zenodo": ZENODO_MD5,
        "rows": int(len(data)),
        "columns": list(data.columns),
        "start": data.index.min().isoformat(),
        "end": data.index.max().isoformat(),
        "duplicate_timestamps": int(data.index.duplicated().sum()),
        "timestamp_delta_minutes_counts": delta_counts,
        "missing_counts": missing_counts,
        "valid_thermal_rows": int(len(valid)),
        "stable_period_rows": int(valid["stable"].sum()),
        "consecutive_15min_rows": int(valid["consecutive_15min"].sum()),
        "observed_ranges": {
            column: {
                "min": float(data[column].min()),
                "max": float(data[column].max()),
            }
            for column in [*THERMAL_COLUMNS, *ELECTRICAL_COLUMNS]
        },
        "unit_basis": "column names, local dataset mapping, and observed magnitudes; no separate local data dictionary was included",
        "breos_root": str(BREOS_ROOT),
        "breos_commit": _git_value("rev-parse", "HEAD"),
        "breos_worktree_status": _git_value("status", "--porcelain"),
        "configuration_hash": _sha256(config_path),
        "dependencies": {
            "python": platform.python_version(),
            "breos": _dependency_version("breos"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pvlib": _dependency_version("pvlib"),
        },
        "reference_temperature_column": "Tmod",
        "reference_temperature_description": "measured module temperature; sensor placement is not documented locally",
    }
    (OUTPUT / "dataset_facts.json").write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8")

    provenance = {
        "driver": str(Path(__file__).resolve()),
        "driver_sha256": _sha256(Path(__file__).resolve()),
        "input_manifest": str(input_manifest),
        "configuration_file": str(config_path),
        "configuration_sha256": _sha256(config_path),
        "input_hashes_include": "the downloaded PCoE CSV",
        "output_hashes_exclude": "provenance.json itself",
        "output_hashes": {
            name: _sha256(OUTPUT / name)
            for name in OUTPUT_FILES
            if name != "provenance.json" and (OUTPUT / name).exists()
        },
    }
    (OUTPUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "dataset_facts": facts,
                "thermal_metrics": thermal_rows,
                "thermal_sensitivity_metrics": sensitivity_rows,
                "monthly_bias": monthly_rows,
                "electrical_checks": electrical_checks,
                "provenance": provenance,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
