"""Reproduce the BREOS checks against the Reunion Island microgrid data.

The collection contains measured meteo, PV-regulator, inverter, battery, and
house-load channels.  This runner keeps the defensible parts separate:

* module-temperature models are compared with the measured under-panel
  temperature;
* the BREOS lumped battery-temperature helper is compared with measured
  battery temperature using the documented current sign convention; and
* inverter/load and submeter relationships are reported as telemetry checks.

There is no SOC, nominal battery capacity, module electrical datasheet, or
inverter nameplate in the downloaded files, so the runner does not invent a
full dispatch or PV electrical model validation.
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
from typing import Any

import numpy as np
import pandas as pd

SOURCE = Path("/home/leo/Downloads/datasets/reunion_island_microgrid")
OUTPUT = Path("/home/leo/code/breos/results/validation_reunion_microgrid_20260829")
DEFAULT_BREOS_ROOT = Path("/tmp/breos-article1-0.6.0")
BREOS_ROOT = Path(os.environ.get("BREOS_VALIDATION_ROOT", DEFAULT_BREOS_ROOT))
sys.path.insert(0, str(BREOS_ROOT))

from breos.battery import compute_cell_temperature  # noqa: E402
from breos.constants import (  # noqa: E402
    DEFAULT_CHARGE_EFFICIENCY,
    DEFAULT_DISCHARGE_EFFICIENCY,
    DEFAULT_THERMAL_RESISTANCE_KW,
)
from breos.pv.temperature import calculate_cell_temperature  # noqa: E402

METEO_FILE = "Meteo_dataset_.csv"
PLANT_FILE = "Solar_plant_dataset.csv"
LOAD_FILES = {
    "house_1": ("Demand_house_1_dataset.csv", False, "Input4", ["Input1", "Input2", "Input3"]),
    "house_2": ("Demand_house_2_dataset.csv", True, "Input5", ["Input1", "Input2", "Input3", "Input4"]),
    "house_3": ("Demand_house_3_dataset.csv", False, "Input1", ["Input2", "Input3", "Input4"]),
}
THERMAL_THRESHOLDS = (0, 50, 200, 400)
PRIMARY_THRESHOLD = 200
OUTPUT_FILES = (
    "thermal_metrics.csv",
    "thermal_sensitivity_metrics.csv",
    "thermal_monthly.csv",
    "battery_thermal_metrics.csv",
    "electrical_checks.json",
    "load_checks.csv",
    "input_manifest.sha256",
    "run_config.json",
    "dataset_facts.json",
    "provenance.json",
    "README.md",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
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


def _dependency_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _timestamp_step_counts(index: pd.DatetimeIndex) -> dict[str, int]:
    deltas = index.sort_values().to_series().diff().dropna().dt.total_seconds()
    return {str(int(seconds)): int(count) for seconds, count in deltas.value_counts().sort_index().items()}


def _timestamp_facts(index: pd.DatetimeIndex) -> dict[str, Any]:
    counts = index.value_counts()
    duplicate_groups = counts[counts > 1]
    return {
        "rows": int(len(index)),
        "start": index.min().isoformat(),
        "end": index.max().isoformat(),
        "duplicate_timestamp_groups": int(len(duplicate_groups)),
        "duplicate_rows_in_groups": int(duplicate_groups.sum()),
        "duplicate_group_sizes": {
            str(int(size)): int(count) for size, count in duplicate_groups.value_counts().sort_index().items()
        },
        "step_seconds_counts": _timestamp_step_counts(index),
        "gaps_over_60_seconds": int((index.sort_values().to_series().diff() > pd.Timedelta(seconds=60)).sum()),
    }


def _read_meteo() -> tuple[pd.DataFrame, dict[str, Any]]:
    path = SOURCE / METEO_FILE
    raw = pd.read_csv(path, sep=";", skiprows=2)
    timestamp = pd.to_datetime(raw.pop("Date"), errors="coerce")
    if timestamp.isna().any():
        raise ValueError(f"Unparseable timestamps in {path}")
    raw_index = pd.DatetimeIndex(timestamp)
    numeric = raw.apply(pd.to_numeric, errors="coerce")
    numeric.index = raw_index
    grouped = numeric.groupby(level=0).mean().sort_index()
    facts = _timestamp_facts(raw_index)
    facts.update(
        {
            "file": METEO_FILE,
            "rows_after_exact_timestamp_mean": int(len(grouped)),
            "missing_counts_raw": {column: int(numeric[column].isna().sum()) for column in numeric.columns},
            "columns": list(numeric.columns),
            "duplicate_timestamp_start": (
                raw_index[raw_index.duplicated(keep=False)].min().isoformat()
                if raw_index.duplicated(keep=False).any()
                else None
            ),
            "duplicate_timestamp_end": (
                raw_index[raw_index.duplicated(keep=False)].max().isoformat()
                if raw_index.duplicated(keep=False).any()
                else None
            ),
            "line_count_including_header_and_metadata": int(len(raw) + 3),
        }
    )
    return grouped, facts


def _read_plant() -> tuple[pd.DataFrame, dict[str, Any]]:
    path = SOURCE / PLANT_FILE
    raw = pd.read_csv(path, skiprows=1)
    timestamp = pd.to_datetime(raw.pop("Date"), errors="coerce")
    if timestamp.isna().any():
        raise ValueError(f"Unparseable timestamps in {path}")
    raw_index = pd.DatetimeIndex(timestamp)
    numeric = raw.apply(pd.to_numeric, errors="coerce")
    numeric.index = raw_index
    grouped = numeric.groupby(level=0).mean().sort_index()
    facts = _timestamp_facts(raw_index)
    facts.update(
        {
            "file": PLANT_FILE,
            "rows_after_exact_timestamp_mean": int(len(grouped)),
            "missing_counts_raw": {column: int(numeric[column].isna().sum()) for column in numeric.columns},
            "columns": list(numeric.columns),
        }
    )
    return grouped, facts


def _read_loads() -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    loads: dict[str, pd.DataFrame] = {}
    facts: dict[str, Any] = {}
    for house, (filename, dayfirst, main, components) in LOAD_FILES.items():
        path = SOURCE / filename
        raw = pd.read_csv(path, skiprows=1)
        timestamp = pd.to_datetime(raw.pop("Date"), dayfirst=dayfirst, errors="coerce")
        if timestamp.isna().any():
            raise ValueError(f"Unparseable timestamps in {path}")
        raw_index = pd.DatetimeIndex(timestamp)
        numeric = raw.apply(pd.to_numeric, errors="coerce")
        numeric.index = raw_index
        grouped = numeric.groupby(level=0).mean().sort_index()
        loads[house] = grouped
        file_facts = _timestamp_facts(raw_index)
        file_facts.update(
            {
                "file": filename,
                "rows_after_exact_timestamp_mean": int(len(grouped)),
                "missing_counts_raw": {column: int(numeric[column].isna().sum()) for column in numeric.columns},
                "columns": list(numeric.columns),
                "main_channel": main,
                "component_channels": components,
                "timestamp_dayfirst": dayfirst,
            }
        )
        facts[house] = file_facts
    return loads, facts


def _correlation(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    if actual.size < 2 or np.std(actual) == 0 or np.std(predicted) == 0:
        return None
    return float(np.corrcoef(actual, predicted)[0, 1])


def _metrics(
    analysis: str,
    model: str,
    actual: np.ndarray | pd.Series,
    predicted: np.ndarray | pd.Series,
    unit: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    valid = np.isfinite(actual_array) & np.isfinite(predicted_array)
    actual_array = actual_array[valid]
    predicted_array = predicted_array[valid]
    error = predicted_array - actual_array
    row: dict[str, Any] = {
        "analysis": analysis,
        "model": model,
        "n": int(actual_array.size),
        f"bias_{unit}": float(np.mean(error)),
        f"mae_{unit}": float(np.mean(np.abs(error))),
        f"rmse_{unit}": float(np.sqrt(np.mean(error**2))),
        "r": _correlation(actual_array, predicted_array),
    }
    if extra:
        row.update(extra)
    return row


def _thermal_checks(
    meteo: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    # The plant and load files are one-minute series.  Mean the meteo data to
    # that resolution after averaging exact duplicate timestamps.  This gives
    # July equal time weight despite its three readings sharing one timestamp.
    minute = meteo.resample("min").mean().dropna(subset=["Gincl", "Tpv", "Ta", "Ws"])
    minute = minute[(minute["Gincl"] >= 0) & (minute["Ws"] >= 0)].copy()
    minute["poa_global_W_m2"] = minute["Gincl"] * 1000.0
    actual = minute["Tpv"].to_numpy(dtype=float)
    default = np.asarray(
        calculate_cell_temperature(minute["poa_global_W_m2"], minute["Ta"], minute["Ws"], "faiman"),
        dtype=float,
    )
    pvsyst = np.asarray(
        calculate_cell_temperature(
            minute["poa_global_W_m2"],
            minute["Ta"],
            minute["Ws"],
            "pvsyst-freestanding",
            module_efficiency=0.20,
        ),
        dtype=float,
    )
    predictions = {
        "breos_faiman_default": default,
        "breos_pvsyst_freestanding_sensitivity": pvsyst,
    }
    thermal_rows: list[dict[str, Any]] = []
    for threshold in THERMAL_THRESHOLDS:
        mask = minute["poa_global_W_m2"].to_numpy() >= threshold
        for model, prediction in predictions.items():
            thermal_rows.append(
                _metrics(
                    "all_valid",
                    model,
                    actual[mask],
                    prediction[mask],
                    "C",
                    {"poa_threshold_W_m2": threshold},
                )
            )

    delta_minutes = minute.index.to_series().diff().dt.total_seconds().div(60)
    stable = (delta_minutes == 1) & (minute["poa_global_W_m2"].diff().abs() <= 25)
    sensitivity_rows: list[dict[str, Any]] = []
    for threshold in THERMAL_THRESHOLDS:
        threshold_mask = minute["poa_global_W_m2"] >= threshold
        stable_mask = (threshold_mask & stable).to_numpy()
        for model, prediction in predictions.items():
            sensitivity_rows.append(
                _metrics(
                    "stable_current_inputs",
                    model,
                    actual[stable_mask],
                    prediction[stable_mask],
                    "C",
                    {"poa_threshold_W_m2": threshold},
                )
            )

        previous = minute.reindex(minute.index - pd.Timedelta(minutes=5))
        previous.index = minute.index
        lag_prediction = np.asarray(
            calculate_cell_temperature(
                previous["Gincl"] * 1000.0,
                previous["Ta"],
                previous["Ws"],
                "faiman",
            ),
            dtype=float,
        )
        lag_mask = (threshold_mask & previous[["Gincl", "Ta", "Ws"]].notna().all(axis=1)).to_numpy()
        sensitivity_rows.append(
            _metrics(
                "equilibrium_inputs_from_previous_5min",
                "breos_faiman_default",
                actual[lag_mask],
                lag_prediction[lag_mask],
                "C",
                {"poa_threshold_W_m2": threshold},
            )
        )

    minute["default_error_C"] = default - actual
    monthly_rows: list[dict[str, Any]] = []
    primary = minute[minute["poa_global_W_m2"] >= PRIMARY_THRESHOLD]
    for month, month_data in primary.groupby(primary.index.month):
        error = month_data["default_error_C"].to_numpy(dtype=float)
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
    facts = {
        "analysis_resolution": "1-minute mean",
        "rows_after_required_filter": int(len(minute)),
        "start": minute.index.min().isoformat(),
        "end": minute.index.max().isoformat(),
        "stable_rows": int(stable.sum()),
        "stable_rows_at_primary_threshold": int((stable & (minute["poa_global_W_m2"] >= PRIMARY_THRESHOLD)).sum()),
        "primary_rows": int(len(primary)),
        "primary_threshold_W_m2": PRIMARY_THRESHOLD,
    }
    return thermal_rows, sensitivity_rows, monthly_rows, facts


def _battery_checks(
    meteo: pd.DataFrame,
    plant: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    meteo_minute = meteo.resample("min").mean()
    common = plant[["Ubat", "Ibat", "Tbat"]].join(meteo_minute[["Ta"]], how="inner").dropna()
    common["battery_terminal_power_W"] = common["Ubat"] * common["Ibat"]
    # The dataset README says positive battery current is discharge and
    # negative current is charge.  Split the signed terminal power accordingly.
    common["charge_power_W"] = (-common["battery_terminal_power_W"]).clip(lower=0)
    common["discharge_power_W"] = common["battery_terminal_power_W"].clip(lower=0)
    hourly = (
        common.resample("h")
        .agg(
            {
                "Ta": "mean",
                "Tbat": "mean",
                "battery_terminal_power_W": "mean",
                "charge_power_W": "mean",
                "discharge_power_W": "mean",
            }
        )
        .dropna()
    )
    prediction = np.fromiter(
        (
            compute_cell_temperature(
                float(row.Ta),
                float(row.charge_power_W),
                float(row.discharge_power_W),
                DEFAULT_CHARGE_EFFICIENCY,
                DEFAULT_DISCHARGE_EFFICIENCY,
                DEFAULT_THERMAL_RESISTANCE_KW,
            )
            for row in hourly.itertuples()
        ),
        dtype=float,
        count=len(hourly),
    )
    rows: list[dict[str, Any]] = []
    for analysis, mask in (
        ("all_common_hourly", np.ones(len(hourly), dtype=bool)),
        ("absolute_battery_power_at_least_100_W", hourly["battery_terminal_power_W"].abs().to_numpy() >= 100),
    ):
        rows.append(
            _metrics(
                analysis,
                "breos_lumped_battery_temperature_default",
                hourly["Tbat"].to_numpy()[mask],
                prediction[mask],
                "C",
            )
        )
    facts = {
        "minute_common_rows": int(len(common)),
        "hourly_rows": int(len(hourly)),
        "start": common.index.min().isoformat(),
        "end": common.index.max().isoformat(),
        "positive_current_rows": int((common["Ibat"] > 0).sum()),
        "negative_current_rows": int((common["Ibat"] < 0).sum()),
        "zero_current_rows": int((common["Ibat"] == 0).sum()),
    }
    return rows, facts


def _load_checks(
    loads: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for house, (_, _, main, components) in LOAD_FILES.items():
        frame = loads[house]
        required = frame[[main, *components]]
        valid = required.notna().all(axis=1)
        measured_main = required.loc[valid, main].to_numpy(dtype=float)
        component_sum = required.loc[valid, components].sum(axis=1).to_numpy(dtype=float)
        row = _metrics(
            "submeter_sum_minus_main",
            f"{house}_component_sum",
            measured_main,
            component_sum,
            "W",
            {
                "house": house,
                "main_channel": main,
                "component_channels": ",".join(components),
                "source_rows_after_timestamp_mean": int(len(frame)),
                "valid_rows": int(valid.sum()),
                "main_missing_rows": int(frame[main].isna().sum()),
                "within_1W_fraction": float(np.mean(np.abs(component_sum - measured_main) <= 1)),
                "within_5W_fraction": float(np.mean(np.abs(component_sum - measured_main) <= 5)),
                "main_mean_W": float(frame[main].mean()),
                "main_median_W": float(frame[main].median()),
                "main_sample_energy_kWh": float(frame[main].sum() / 60000.0),
            },
        )
        rows.append(row)
    return rows


def _quantiles(series: pd.Series) -> dict[str, float | None]:
    values = series.dropna()
    if values.empty:
        return {"min": None, "p01": None, "median": None, "p99": None, "max": None}
    q = values.quantile([0, 0.01, 0.5, 0.99, 1])
    return {
        "min": float(q.loc[0.0]),
        "p01": float(q.loc[0.01]),
        "median": float(q.loc[0.5]),
        "p99": float(q.loc[0.99]),
        "max": float(q.loc[1.0]),
    }


def _electrical_checks(
    plant: pd.DataFrame,
    loads: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    mains = pd.concat(
        [
            loads["house_1"]["Input4"].rename("house_1"),
            loads["house_2"]["Input5"].rename("house_2"),
            loads["house_3"]["Input1"].rename("house_3"),
        ],
        axis=1,
        sort=False,
    ).sort_index()
    mains["total_house_load_W"] = mains.sum(axis=1, min_count=1)
    all_houses = mains.dropna(subset=["house_1", "house_2", "house_3"])
    common = plant.join(all_houses[["total_house_load_W"]], how="inner").dropna(subset=["Pout", "total_house_load_W"])
    load_metric = _metrics(
        "inverter_output_minus_three_house_mains",
        "measured_Pout_vs_measured_house_load",
        common["total_house_load_W"].to_numpy(dtype=float),
        (common["Pout"] * 1000.0).to_numpy(dtype=float),
        "W",
    )
    pdc = plant["Ppv1"] + plant["Ppv2"]
    pdc_valid = pdc.dropna()
    pout_valid = plant["Pout"].dropna()
    pbat = (plant["Ubat"] * plant["Ibat"]).dropna()
    return {
        "plant_rows_after_timestamp_mean": int(len(plant)),
        "plant_start": plant.index.min().isoformat(),
        "plant_end": plant.index.max().isoformat(),
        "plant_missing_counts": {column: int(plant[column].isna().sum()) for column in plant.columns},
        "plant_step_seconds_counts": _timestamp_step_counts(plant.index),
        "pout_kW_quantiles": _quantiles(plant["Pout"]),
        "pv_regulator_total_kW_quantiles": _quantiles(pdc),
        "battery_terminal_power_kW_quantiles": _quantiles(pbat / 1000.0),
        "battery_current_positive_discharge_fraction": float((plant["Ibat"] > 0).mean()),
        "battery_current_negative_charge_fraction": float((plant["Ibat"] < 0).mean()),
        "observed_sample_energy_kWh": {
            "Pout": float(pout_valid.sum() / 60.0),
            "Ppv1_plus_Ppv2": float(pdc_valid.sum() / 60.0),
            "three_house_mains_common_rows": float(all_houses["total_house_load_W"].sum() / 1000.0 / 60.0),
        },
        "three_house_mains_common_rows": int(len(all_houses)),
        "inverter_load_consistency": load_metric,
        "boundary_note": (
            "Pout is inverter output while Ppv1/Ppv2 are solar-regulator outputs; "
            "the files do not establish a direct same-boundary PV conversion target."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Reunion microgrid BREOS validation checks.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing generated result files",
    )
    args = parser.parse_args()

    existing = [OUTPUT / name for name in OUTPUT_FILES if (OUTPUT / name).exists()]
    if existing and not args.force:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"Refusing to replace existing result files: {names}. Use --force to replace them.")

    input_files = sorted(path for path in SOURCE.iterdir() if path.is_file())
    if not input_files:
        raise FileNotFoundError(f"No files found in {SOURCE}")

    meteo, meteo_facts = _read_meteo()
    plant, plant_facts = _read_plant()
    loads, load_facts = _read_loads()
    thermal_rows, thermal_sensitivity_rows, monthly_rows, thermal_facts = _thermal_checks(meteo)
    battery_rows, battery_facts = _battery_checks(meteo, plant)
    load_rows = _load_checks(loads)
    electrical = _electrical_checks(plant, loads)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(thermal_rows).to_csv(OUTPUT / "thermal_metrics.csv", index=False, float_format="%.9f")
    pd.DataFrame(thermal_sensitivity_rows).to_csv(
        OUTPUT / "thermal_sensitivity_metrics.csv", index=False, float_format="%.9f"
    )
    pd.DataFrame(monthly_rows).to_csv(OUTPUT / "thermal_monthly.csv", index=False, float_format="%.9f")
    pd.DataFrame(battery_rows).to_csv(OUTPUT / "battery_thermal_metrics.csv", index=False, float_format="%.9f")
    _write_json(OUTPUT / "electrical_checks.json", electrical)
    pd.DataFrame(load_rows).to_csv(OUTPUT / "load_checks.csv", index=False, float_format="%.9f")

    manifest_lines = [f"{_sha256(path)}  {path.relative_to(SOURCE).as_posix()}" for path in input_files]
    (OUTPUT / "input_manifest.sha256").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    config = {
        "source_directory": str(SOURCE),
        "breos_root": str(BREOS_ROOT),
        "meteo_file": METEO_FILE,
        "plant_file": PLANT_FILE,
        "load_files": {
            house: {
                "file": filename,
                "timestamp_dayfirst": dayfirst,
                "main_channel": main,
                "component_channels": components,
            }
            for house, (filename, dayfirst, main, components) in LOAD_FILES.items()
        },
        "meteo_units": {
            "Gincl": "kW/m2 in source; multiplied by 1000 for BREOS W/m2",
            "Ghi": "kW/m2 in source; not used in the thermal comparison",
            "Tpv": "degC under-panel surface temperature",
            "Ta": "degC",
            "Ws": "m/s",
        },
        "timestamp_handling": {
            "exact_duplicate_policy": "mean all numeric channels sharing a timestamp",
            "meteo_resampling": "mean to 1 minute after exact-timestamp grouping",
            "plant_and_load_resampling": "mean exact duplicate timestamps; otherwise native 1-minute samples",
            "timezone": "naive source timestamps retained; no timezone conversion",
        },
        "thermal_models": {
            "faiman_default": {"u0": 25.0, "u1": 6.84},
            "pvsyst_freestanding_sensitivity": {"module_efficiency": 0.20},
        },
        "thermal_thresholds_W_m2": list(THERMAL_THRESHOLDS),
        "primary_threshold_W_m2": PRIMARY_THRESHOLD,
        "stable_period": {
            "consecutive_interval_minutes": 1,
            "maximum_absolute_Gincl_change_W_m2": 25,
        },
        "battery_temperature_model": {
            "analysis_resolution": "hourly means of one-minute telemetry",
            "current_sign": "positive discharge, negative charge as stated in source README",
            "charge_efficiency": float(DEFAULT_CHARGE_EFFICIENCY),
            "discharge_efficiency": float(DEFAULT_DISCHARGE_EFFICIENCY),
            "thermal_resistance_K_per_W": float(DEFAULT_THERMAL_RESISTANCE_KW),
        },
        "not_attempted": [
            "full PV electrical model",
            "battery SOC/dispatch simulation",
            "battery degradation validation",
        ],
    }
    config_path = OUTPUT / "run_config.json"
    _write_json(config_path, config)

    facts = {
        "source_directory": str(SOURCE),
        "source_files": [path.relative_to(SOURCE).as_posix() for path in input_files],
        "source_file_sizes_bytes": {path.relative_to(SOURCE).as_posix(): path.stat().st_size for path in input_files},
        "meteo": meteo_facts,
        "plant": plant_facts,
        "loads": load_facts,
        "thermal": thermal_facts,
        "battery_thermal": battery_facts,
        "electrical": {
            "three_house_main_channels": {house: values[2] for house, values in LOAD_FILES.items()},
            "pv_regulator_channels": ["Ppv1", "Ppv2"],
            "inverter_channel": "Pout",
            "battery_channels": ["Ubat", "Ibat", "Tbat"],
        },
        "breos_root": str(BREOS_ROOT),
        "breos_commit": _git_value("rev-parse", "HEAD"),
        "breos_worktree_status": _git_value("status", "--porcelain"),
        "dependencies": {
            "python": platform.python_version(),
            "breos": _dependency_version("breos"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "configuration_hash": _sha256(config_path),
    }
    _write_json(OUTPUT / "dataset_facts.json", facts)

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
    _write_json(OUTPUT / "provenance.json", provenance)

    print(
        json.dumps(
            {
                "thermal_primary": [row for row in thermal_rows if row["poa_threshold_W_m2"] == PRIMARY_THRESHOLD],
                "battery_thermal": battery_rows,
                "load_checks": load_rows,
                "electrical_checks": electrical,
                "provenance": provenance,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
