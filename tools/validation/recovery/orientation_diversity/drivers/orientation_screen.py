"""Run the defensible orientation screen for the rooftop PV workbook.

The workbook has no date, timezone, site location, or panel geometry. The
driver therefore does not call the BREOS weather-to-PV path. It checks the
panel aggregation and measures the held-out predictive value of the supplied
incidence-angle cosine values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
from importlib import metadata
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
import pandas as pd

SOURCE_DIR = Path("/home/leo/Downloads/datasets/orientation_diversity_pv")
WORKBOOK = SOURCE_DIR / "PV_Data.xlsx"
SOURCE_README = SOURCE_DIR / "README.txt"
OUTPUT = Path(
    os.environ.get(
        "BREOS_VALIDATION_OUTPUT",
        "/home/leo/code/breos/results/validation_orientation_diversity_recovered_20260902",
    )
)
BREOS_ROOT = Path(os.environ.get("BREOS_VALIDATION_ROOT", "/tmp/breos-article1-0.6.0"))
OUTPUT_FILES = (
    "orientation_screen_metrics.csv",
    "data_integrity.json",
    "input_manifest.sha256",
    "run_config.json",
    "dataset_facts.json",
    "provenance.json",
    "README.md",
)
XML_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
DATA_WIDTH = 21
TRAIN_DAY_MAX = 206
TEST_DAY_MIN = 207
SOLAR_RADIATION_TEST_MIN = 20.0
PANEL_IDS = (1, 2, 3, 4, 5)


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


def _column_index(cell_reference: str) -> int:
    letters = re.match(r"[A-Z]+", cell_reference)
    if letters is None:
        raise ValueError(f"Invalid cell reference: {cell_reference}")
    index = 0
    for letter in letters.group(0):
        index = index * 26 + ord(letter) - ord("A") + 1
    return index - 1


def _read_sheet_rows(
    archive: ZipFile,
    sheet_name: str,
    shared_strings: list[str],
    width: int,
) -> list[list[str | None]]:
    root = ET.fromstring(archive.read(sheet_name))
    rows: list[list[str | None]] = []
    for row in root.findall(".//m:sheetData/m:row", XML_NS):
        values: list[str | None] = [None] * width
        for cell in row.findall("m:c", XML_NS):
            value = cell.find("m:v", XML_NS)
            cell_value = None if value is None else value.text
            if cell.attrib.get("t") == "s" and cell_value is not None:
                cell_value = shared_strings[int(cell_value)]
            if cell.attrib.get("t") == "inlineStr":
                text = cell.find(".//m:t", XML_NS)
                cell_value = None if text is None else text.text
            column = _column_index(cell.attrib["r"])
            if column >= width:
                raise ValueError(f"Cell exceeds expected width: {cell.attrib['r']}")
            values[column] = cell_value
        rows.append(values)
    return rows


def _metric_row(
    panel: str,
    model: str,
    features: str,
    coefficients: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float | int | str]:
    error = predicted - actual
    total_sum_squares = np.sum((actual - actual.mean()) ** 2)
    return {
        "panel": panel,
        "model": model,
        "features": features,
        "coefficient_1": float(coefficients[0]),
        "coefficient_2": (float(coefficients[1]) if coefficients.size > 1 else np.nan),
        "n": int(actual.size),
        "bias_W": float(np.mean(error)),
        "mae_W": float(np.mean(np.abs(error))),
        "rmse_W": float(np.sqrt(np.mean(error**2))),
        "r": float(np.corrcoef(actual, predicted)[0, 1]),
        "r2": float(1 - np.sum(error**2) / total_sum_squares),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the orientation-diversity data and response screen.")
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
    if not WORKBOOK.is_file() or not SOURCE_README.is_file():
        raise FileNotFoundError("The workbook and local README are both required")

    with ZipFile(WORKBOOK) as archive:
        shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared_strings = [
            "".join(text.text or "" for text in item.findall(".//m:t", XML_NS))
            for item in shared_root.findall("m:si", XML_NS)
        ]
        description_rows = _read_sheet_rows(
            archive,
            "xl/worksheets/sheet2.xml",
            shared_strings,
            DATA_WIDTH,
        )
        data_rows = _read_sheet_rows(
            archive,
            "xl/worksheets/sheet1.xml",
            shared_strings,
            DATA_WIDTH,
        )

    headers = description_rows[1]
    if any(header is None for header in headers):
        raise ValueError("The description-sheet header row contains blanks")
    data = pd.DataFrame(data_rows, columns=headers)
    numeric_columns = [column for column in headers if column != "Weather Condition"]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data[numeric_columns].isna().any().any():
        raise ValueError("The workbook contains missing numeric values")

    panel_power_columns = [f"PV-{panel} Power" for panel in PANEL_IDS]
    panel_sum = data[panel_power_columns].sum(axis=1)
    total_difference = data["Total PV Power"] - panel_sum
    radiation = data["Solar Radiation"].to_numpy(dtype=float)
    train_mask = data["Day"].to_numpy(dtype=int) <= TRAIN_DAY_MAX
    test_mask = (data["Day"].to_numpy(dtype=int) >= TEST_DAY_MIN) & (radiation > SOLAR_RADIATION_TEST_MIN)

    screen_rows: list[dict[str, float | int | str]] = []
    total_predictions: dict[str, np.ndarray] = {
        "common_radiation": np.zeros(int(test_mask.sum()), dtype=float),
        "radiation_plus_aoi": np.zeros(int(test_mask.sum()), dtype=float),
    }
    for panel in PANEL_IDS:
        actual = data[f"PV-{panel} Power"].to_numpy(dtype=float)
        aoi = data[f"PV-{panel} Cosine Value of the Solar Incidence Angle"].to_numpy(dtype=float)
        aoi = np.clip(aoi, 0.0, None)
        feature_sets = {
            "common_radiation": (
                radiation[:, None],
                "Solar Radiation",
            ),
            "radiation_plus_aoi": (
                np.column_stack((radiation, radiation * aoi)),
                "Solar Radiation, Solar Radiation x AOI cosine",
            ),
        }
        for model, (features, feature_names) in feature_sets.items():
            coefficients = np.linalg.lstsq(features[train_mask], actual[train_mask], rcond=None)[0]
            predicted = features[test_mask] @ coefficients
            screen_rows.append(
                _metric_row(
                    f"PV-{panel}",
                    model,
                    feature_names,
                    coefficients,
                    actual[test_mask],
                    predicted,
                )
            )
            total_predictions[model] += predicted

    total_actual = data["Total PV Power"].to_numpy(dtype=float)[test_mask]
    screen_rows.extend(
        [
            _metric_row(
                "Total",
                model,
                "sum of panel predictions",
                np.array([np.nan]),
                total_actual,
                predicted,
            )
            for model, predicted in total_predictions.items()
        ]
    )

    panel_facts = []
    for panel in PANEL_IDS:
        power = data[f"PV-{panel} Power"]
        aoi = data[f"PV-{panel} Cosine Value of the Solar Incidence Angle"]
        panel_facts.append(
            {
                "panel": f"PV-{panel}",
                "power_min": float(power.min()),
                "power_max": float(power.max()),
                "positive_power_rows": int((power > 0).sum()),
                "aoi_min": float(aoi.min()),
                "aoi_max": float(aoi.max()),
                "power_r_solar_radiation": float(power.corr(data["Solar Radiation"])),
                "power_r_radiation_times_aoi": float(
                    power.corr(data["Solar Radiation"] * data[f"PV-{panel} Cosine Value of the Solar Incidence Angle"])
                ),
                "positive_power_rows_when_radiation_zero": int(((data["Solar Radiation"] == 0) & (power > 0)).sum()),
            }
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(screen_rows).to_csv(
        OUTPUT / "orientation_screen_metrics.csv",
        index=False,
        float_format="%.9f",
    )
    integrity = {
        "total_power_identity_max_abs_W": float(np.abs(total_difference).max()),
        "total_power_identity_nonzero_rows": int((total_difference != 0).sum()),
        "numeric_missing_values": int(data[numeric_columns].isna().sum().sum()),
        "day_hour_duplicate_rows": int(data.duplicated(["Day", "Hour"]).sum()),
        "day_count": int(data["Day"].nunique()),
        "hour_count": int(data["Hour"].nunique()),
        "rows_per_day_min": int(data.groupby("Day").size().min()),
        "rows_per_day_max": int(data.groupby("Day").size().max()),
        "panel_facts": panel_facts,
    }
    (OUTPUT / "data_integrity.json").write_text(json.dumps(integrity, indent=2) + "\n", encoding="utf-8")

    input_files = [WORKBOOK, SOURCE_README]
    (OUTPUT / "input_manifest.sha256").write_text(
        "\n".join(f"{_sha256(path)}  {path.name}" for path in input_files) + "\n",
        encoding="utf-8",
    )
    config = {
        "source_workbook": str(WORKBOOK),
        "source_readme": str(SOURCE_README),
        "data_sheet": "xl/worksheets/sheet1.xml",
        "header_sheet": "xl/worksheets/sheet2.xml row 2",
        "data_width": DATA_WIDTH,
        "train_day_max": TRAIN_DAY_MAX,
        "test_day_min": TEST_DAY_MIN,
        "solar_radiation_test_min_W_m2": SOLAR_RADIATION_TEST_MIN,
        "baseline_features": ["Solar Radiation"],
        "orientation_features": [
            "Solar Radiation",
            "Solar Radiation x max(AOI cosine, 0)",
        ],
        "fit": "ordinary least squares through the origin on days 1 through 206",
        "breos_pv_simulation": "not run because date, timezone, site geometry, and module parameters are absent",
    }
    config_path = OUTPUT / "run_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    facts = {
        "source_record": "https://zenodo.org/records/19245713",
        "source_doi": "10.5281/zenodo.19245713",
        "source_license": "CC BY 4.0, as stated on the source record",
        "source_files": [path.name for path in input_files],
        "workbook_sheet1_rows": int(len(data_rows)),
        "workbook_sheet1_columns": int(len(headers)),
        "columns": headers,
        "days": {"min": int(data["Day"].min()), "max": int(data["Day"].max())},
        "hours": {"min": int(data["Hour"].min()), "max": int(data["Hour"].max())},
        "test_rows": int(test_mask.sum()),
        "train_rows": int(train_mask.sum()),
        "has_calendar_dates": False,
        "has_timezone": False,
        "has_site_coordinates": False,
        "has_panel_tilt_azimuth": False,
        "has_ghi_dni_dhi": False,
        "breos_pv_simulation": "not run",
        "breos_pv_simulation_reason": "The current BREOS production path needs a DatetimeIndex, location, panel geometry, irradiance components, and module parameters. The workbook supplies none of those as verified inputs.",
        "breos_root": str(BREOS_ROOT),
        "breos_commit": _git_value("rev-parse", "HEAD"),
        "breos_worktree_status": _git_value("status", "--porcelain"),
        "configuration_hash": _sha256(config_path),
        "dependencies": {
            "python": platform.python_version(),
            "breos": _dependency_version("breos"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    (OUTPUT / "dataset_facts.json").write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8")

    provenance = {
        "driver": str(Path(__file__).resolve()),
        "driver_sha256": _sha256(Path(__file__).resolve()),
        "input_manifest": str(OUTPUT / "input_manifest.sha256"),
        "configuration_file": str(config_path),
        "configuration_sha256": _sha256(config_path),
        "input_hashes_include": "PV_Data.xlsx and README.txt",
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
                "facts": facts,
                "integrity": integrity,
                "screen_metrics": screen_rows,
                "provenance": provenance,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
