"""Reproduce the Sandia/IEA Task 13 BREOS thermal validation.

This deliberately validates the thermal component only. The dataset supplies
measured POA irradiance, ambient temperature, wind speed, and back-of-module
temperature, so it can test the Faiman response without inventing site
geometry or missing electrical module parameters.
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
import pvlib

# Raw inputs live outside the repository because of size and licensing. Point
# BREOS_VALIDATION_DATA at the directory holding the downloaded datasets; the
# README records which archive belongs where.
DATA_ROOT = Path(os.environ.get("BREOS_VALIDATION_DATA", "datasets")).expanduser()
SOURCE = DATA_ROOT / "sandia_iea_pvps_task13"
OUTPUT = Path(
    os.environ.get(
        "BREOS_VALIDATION_OUTPUT",
        "results/validation_sandia_task13_recovered",
    )
).expanduser()
DEFAULT_BREOS_ROOT = Path("/tmp/breos-article1-0.6.0")
BREOS_ROOT = Path(os.environ.get("BREOS_VALIDATION_ROOT", DEFAULT_BREOS_ROOT))
sys.path.insert(0, str(BREOS_ROOT))

from breos.pv.temperature import calculate_cell_temperature  # noqa: E402

REQUIRED_COLUMNS = ["Gpoa", "AIR_TEMP", "WIND_SPEED", "Tbom"]
THRESHOLDS = (0, 50, 200, 400)
SANDIA_U0 = 29.84
SANDIA_U1 = 3.44
OUTPUT_FILES = (
    "thermal_metrics.csv",
    "thermal_sensitivity_metrics.csv",
    "monthly_bias.csv",
    "input_manifest.sha256",
    "run_config.json",
    "dataset_facts.json",
    "provenance.json",
    "README.md",
)


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


def _metric_row(
    model: str,
    threshold: int,
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float | int | str]:
    error = predicted - actual
    return {
        "model": model,
        "gpoa_threshold_W_m2": threshold,
        "n": int(actual.size),
        "bias_C": float(np.mean(error)),
        "mae_C": float(np.mean(np.abs(error))),
        "rmse_C": float(np.sqrt(np.mean(error**2))),
        "r": float(np.corrcoef(actual, predicted)[0, 1]),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dependency_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _metric_rows(
    analysis: str,
    threshold: int,
    actual: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> list[dict[str, float | int | str]]:
    return [
        _metric_row(model, threshold, actual, prediction) | {"analysis": analysis}
        for model, prediction in predictions.items()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Sandia/IEA Task 13 BREOS thermal validation.")
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

    csv_files = sorted(SOURCE.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {SOURCE}")
    input_files = sorted(path for path in SOURCE.iterdir() if path.is_file())

    frames: list[pd.DataFrame] = []
    for path in csv_files:
        frame = pd.read_csv(path, sep=";")
        frame.columns = [str(column).strip() for column in frame.columns]
        timestamp = pd.to_datetime(
            frame["Date"].astype(str) + " " + frame["Time"].astype(str),
            dayfirst=True,
            errors="coerce",
        )
        if timestamp.isna().any():
            raise ValueError(f"Unparseable timestamps in {path}")
        frame.index = timestamp
        frames.append(frame)

    data = pd.concat(frames).sort_index()
    for column in REQUIRED_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    missing_counts = {column: int(data[column].isna().sum()) for column in REQUIRED_COLUMNS}
    deltas = data.index.to_series().diff().dropna().dt.total_seconds().div(60)
    delta_counts = {str(int(minutes)): int(count) for minutes, count in deltas.value_counts().sort_index().items()}

    valid = data[REQUIRED_COLUMNS].dropna()
    valid = valid[(valid["Gpoa"] >= 0) & (valid["WIND_SPEED"] >= 0)].copy()
    valid["delta_minutes"] = valid.index.to_series().diff().dt.total_seconds().div(60).to_numpy()
    valid["gpoa_change_W_m2"] = valid["Gpoa"].diff().abs()
    valid["stable"] = (valid["delta_minutes"] == 5) & (valid["gpoa_change_W_m2"] <= 25)
    valid["consecutive_5min"] = valid["delta_minutes"] == 5

    default_prediction = np.asarray(
        calculate_cell_temperature(
            valid["Gpoa"],
            valid["AIR_TEMP"],
            valid["WIND_SPEED"],
            "faiman",
        ),
        dtype=float,
    )
    sandia_prediction = np.asarray(
        pvlib.temperature.faiman(
            valid["Gpoa"],
            valid["AIR_TEMP"],
            valid["WIND_SPEED"],
            u0=SANDIA_U0,
            u1=SANDIA_U1,
        ),
        dtype=float,
    )
    actual_all = valid["Tbom"].to_numpy(dtype=float)
    valid["default_prediction"] = default_prediction
    valid["sandia_prediction"] = sandia_prediction
    valid["default_error"] = default_prediction - actual_all

    metric_rows: list[dict[str, float | int | str]] = []
    for threshold in THRESHOLDS:
        mask = valid["Gpoa"].to_numpy(dtype=float) >= threshold
        metric_rows.extend(
            _metric_rows(
                "all_valid",
                threshold,
                actual_all[mask],
                {
                    "breos_faiman_default": default_prediction[mask],
                    "faiman_sandia_u0_u1": sandia_prediction[mask],
                },
            )
        )

    sensitivity_rows: list[dict[str, float | int | str]] = []
    for threshold in THRESHOLDS:
        threshold_mask = valid["Gpoa"].to_numpy(dtype=float) >= threshold
        stable_mask = threshold_mask & valid["stable"].to_numpy(dtype=bool)
        sensitivity_rows.extend(
            _metric_rows(
                "stable_current_inputs",
                threshold,
                actual_all[stable_mask],
                {
                    "breos_faiman_default": default_prediction[stable_mask],
                    "faiman_sandia_u0_u1": sandia_prediction[stable_mask],
                },
            )
        )

        previous = valid[["Gpoa", "AIR_TEMP", "WIND_SPEED"]].shift(1)
        lag_prediction = np.asarray(
            pvlib.temperature.faiman(
                previous["Gpoa"],
                previous["AIR_TEMP"],
                previous["WIND_SPEED"],
            ),
            dtype=float,
        )
        lag_mask = threshold_mask & valid["consecutive_5min"].to_numpy(dtype=bool)
        lag_mask &= np.isfinite(lag_prediction)
        sensitivity_rows.extend(
            _metric_rows(
                "equilibrium_inputs_from_previous_5min",
                threshold,
                actual_all[lag_mask],
                {"breos_faiman_default": lag_prediction[lag_mask]},
            )
        )

    monthly_rows: list[dict[str, float | int | str]] = []
    primary = valid[valid["Gpoa"] >= 200]
    for month, month_data in primary.groupby(primary.index.month):
        error = month_data["default_error"].to_numpy(dtype=float)
        monthly_rows.append(
            {
                "month": int(month),
                "month_name": month_data.index[0].strftime("%B"),
                "n": int(error.size),
                "bias_C": float(np.mean(error)),
                "mae_C": float(np.mean(np.abs(error))),
                "rmse_C": float(np.sqrt(np.mean(error**2))),
            }
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(
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

    manifest_lines = [f"{_sha256(path)}  {path.relative_to(SOURCE).as_posix()}" for path in input_files]
    (OUTPUT / "input_manifest.sha256").write_text(
        "\n".join(manifest_lines) + "\n",
        encoding="utf-8",
    )

    config = {
        "source_directory": str(SOURCE),
        "breos_root": str(BREOS_ROOT),
        "csv_separator": ";",
        "timestamp_dayfirst": True,
        "required_columns": REQUIRED_COLUMNS,
        "valid_filter": {
            "non_null": REQUIRED_COLUMNS,
            "Gpoa_min_W_m2": 0,
            "WIND_SPEED_min_m_s": 0,
        },
        "thresholds_W_m2": list(THRESHOLDS),
        "stable_period": {
            "consecutive_interval_minutes": 5,
            "max_absolute_Gpoa_change_W_m2": 25,
        },
        "faiman_default": {"u0": 25.0, "u1": 6.84},
        "faiman_workbook_sensitivity": {"u0": SANDIA_U0, "u1": SANDIA_U1},
        "primary_threshold_W_m2": 200,
    }
    config_path = OUTPUT / "run_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    facts = {
        "source_directory": str(SOURCE),
        "source_files": [path.relative_to(SOURCE).as_posix() for path in input_files],
        "source_file_sizes_bytes": {path.relative_to(SOURCE).as_posix(): path.stat().st_size for path in input_files},
        "rows": int(len(data)),
        "start": data.index.min().isoformat(),
        "end": data.index.max().isoformat(),
        "duplicate_timestamps": int(data.index.duplicated().sum()),
        "required_columns": REQUIRED_COLUMNS,
        "missing_counts": missing_counts,
        "timestamp_delta_minutes_counts": delta_counts,
        "valid_thermal_rows": int(len(valid)),
        "valid_filter": "non-null Gpoa, AIR_TEMP, WIND_SPEED, Tbom; Gpoa >= 0; WIND_SPEED >= 0",
        "breos_root": str(BREOS_ROOT),
        "breos_commit": _git_value("rev-parse", "HEAD"),
        "breos_worktree_status": _git_value("status", "--porcelain"),
        "pvlib_version": pvlib.__version__,
        "default_faiman_parameters": {"u0": 25.0, "u1": 6.84},
        "sandia_workbook_faiman_parameters": {
            "u0": SANDIA_U0,
            "u1": SANDIA_U1,
        },
        "reference_column": "Tbom",
        "reference_description": "back-of-module temperature; not a direct cell-temperature measurement",
        "stable_period_rows": int(valid["stable"].sum()),
        "consecutive_5min_rows": int(valid["consecutive_5min"].sum()),
        "configuration_hash": _sha256(config_path),
        "dependencies": {
            "python": platform.python_version(),
            "breos": _dependency_version("breos"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pvlib": pvlib.__version__,
        },
    }
    (OUTPUT / "dataset_facts.json").write_text(
        json.dumps(facts, indent=2) + "\n",
        encoding="utf-8",
    )

    provenance = {
        "driver": str(Path(__file__).resolve()),
        "driver_sha256": _sha256(Path(__file__).resolve()),
        "input_manifest": str(OUTPUT / "input_manifest.sha256"),
        "configuration_file": str(config_path),
        "configuration_sha256": _sha256(config_path),
        "input_hashes_include": "all regular files directly under the source directory",
        "output_hashes_exclude": "provenance.json itself",
        "output_hashes": {
            name: _sha256(OUTPUT / name)
            for name in OUTPUT_FILES
            if name != "provenance.json" and (OUTPUT / name).exists()
        },
    }
    (OUTPUT / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "dataset_facts": facts,
                "metrics": metric_rows,
                "sensitivity_metrics": sensitivity_rows,
                "monthly_bias": monthly_rows,
                "provenance": provenance,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
