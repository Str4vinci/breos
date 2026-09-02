"""Run an exploratory calibrated forward check against the HKUST dataset.

The local Brick file supplies equipment and location metadata but no usable
tilt or azimuth fields. The driver therefore fits an effective fixed
orientation from pre-2023 data, fits one output scale factor from the same
period, and evaluates 2023 without refitting either parameter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pvlib
from pvlib.location import Location

DEFAULT_BREOS_ROOT = Path("/tmp/breos-article1-0.6.0")
BREOS_ROOT = Path(os.environ.get("BREOS_VALIDATION_ROOT", DEFAULT_BREOS_ROOT))
sys.path.insert(0, str(BREOS_ROOT))

from breos.solar import PVModuleParams, calculate_pv_production_dc, dc_to_ac  # noqa: E402

SOURCE_DIR = Path("/home/leo/Downloads/datasets/hkust_rooftop_60stations")
DATASET_ROOT = SOURCE_DIR / "Dataset"
METEO_ROOT = DATASET_ROOT / "Time series dataset" / "Meteorological dataset"
PV_ROOT = DATASET_ROOT / "Time series dataset" / "PV generation dataset"
METADATA_FILE = DATASET_ROOT / "Metadata" / "PV generation system metadata.ttl"
DEFAULT_OUTPUT = Path(
    "/home/leo/code/breos/results/validation_hkust_timing-corrected-exploratory_20260830"
)
TIMEZONE = "Asia/Hong_Kong"
LOCATION = Location(22.3363, 114.2634, tz=TIMEZONE)
DAYLIGHT_GHI_THRESHOLD = 200.0
TRAIN_LAST_YEAR = 2022
TEST_YEAR = 2023
ORIENTATION_TILTS = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0)
ORIENTATION_AZIMUTHS = tuple(float(value) for value in range(0, 360, 45))
NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
OUTPUT_FILES = (
    "site_metrics.csv",
    "orientation_fit.csv",
    "data_quality.csv",
    "metadata_mapping.csv",
    "optimizer_consistency.csv",
    "aggregate_metrics.json",
    "optimizer_consistency_summary.json",
    "input_manifest.sha256",
    "run_config.json",
    "dataset_facts.json",
    "timing_contract.json",
    "provenance.json",
    "README.md",
)

PV_MODEL_OPTIONS = {
    "loss_overrides": {
        "soiling": 2.0,
        "shading": 3.0,
        "snow": 0.0,
        "mismatch": 2.0,
        "wiring": 2.0,
        "connections": 0.5,
        "lid": 1.5,
        "nameplate_rating": 1.0,
        "availability": 3.0,
    },
    "transposition_model": "perez",
    "albedo": 0.25,
    "surface_type": None,
    "model_perez": "allsitescomposite1990",
    "solar_position": "mid-interval",
    "iam_model": "ashrae",
    "diffuse_iam": "marion",
    "temperature_model": "faiman",
    "bifacial_model": "none",
    "gcr": 0.35,
    "pvrow_height": None,
    "pvrow_pitch": None,
}

SOURCE_LABEL_BASIS_BY_FREQUENCY = {
    "h": "interval-end",
    "15min": "interval-start",
}

WEATHER_SPECS = {
    "ghi": ("Irradiance", "Irradiance_*.csv", "Irradiance (W/m2)"),
    "temp_air": ("Temperature", "Temperature_*.csv", "Temp (Degree Celsius)"),
    "wind_speed": ("Wind", "Wind_*.csv", "Wind Speed (m/s)"),
}
CEC_MAP = {
    "JKM365N_6TL3": "Jinko_Solar_Co___Ltd_JKM365M_72_V",
    "JKM365N_6TL3_V": "Jinko_Solar_Co___Ltd_JKM365M_72_V",
    "JKM370N_6TL3": "Jinko_Solar_Co___Ltd_JKM370M_72_V",
    "JKM390M_6RL3_TV": "Jinko_Solar_Co___Ltd_JKM390M_72HL_V",
    "JKM395M_6RL3_V": "Jinko_Solar_Co___Ltd_JKM395M_72HL_V",
    "JKM410N_6RL3": "Jinko_Solar_Co___Ltd_JKM410M_72HL_V",
    "JKM415N_6RL3_V": "Jinko_Solar_Co___Ltd_JKM410M_72HL_V",
    "SMF310M_6X10UW": "Jinko_Solar_Co___Ltd_JKM310M_72B",
}


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _number_value(block: str, property_name: str) -> float | None:
    match = re.search(
        rf"{re.escape(property_name)}\s+\[.*?brick:value\s+({NUMBER_PATTERN})",
        block,
        flags=re.DOTALL,
    )
    return float(match.group(1)) if match else None


def _quoted_value(block: str, property_name: str) -> str | None:
    match = re.search(
        rf'{re.escape(property_name)}\s+\[\s*brick:value\s+"([^"]*)"',
        block,
        flags=re.DOTALL,
    )
    return match.group(1) if match else None


def _parts(block: str) -> list[str]:
    match = re.search(r"brick:hasPart\s+(.*?);", block, flags=re.DOTALL)
    if not match:
        return []
    return re.findall(r"pvsystem:([^\s,;]+)", match.group(1))


def _parse_metadata(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    systems: dict[str, dict[str, Any]] = {}
    inverters: dict[str, dict[str, Any]] = {}
    modules: dict[str, dict[str, Any]] = {}

    for block in re.split(r"(?m)(?=^pvsystem:)", path.read_text(encoding="utf-8")):
        name_match = re.match(r"pvsystem:([^\s]+)", block)
        type_match = re.search(r"\ba\s+brick:([^\s;]+)", block)
        if not name_match or not type_match:
            continue
        name = name_match.group(1)
        entity_type = type_match.group(1)
        if entity_type == "PV_Generation_System":
            systems[name] = {
                "parts": _parts(block),
                "latitude": _number_from_text(block, "brick:latitude"),
                "longitude": _number_from_text(block, "brick:longitude"),
                "rated_kw": _number_value(block, "ext:ratedPowerOutput"),
                "connection_date": _quoted_value(block, "ext:connectionDate"),
            }
        elif entity_type == "Inverter":
            inverters[name] = {
                "parts": _parts(block),
                "brand": _quoted_value(block, "ext:brand"),
                "model": _quoted_value(block, "ext:module"),
            }
        elif entity_type == "PV_Panel":
            modules[name] = {
                "brand": _quoted_value(block, "ext:brand"),
                "model": _quoted_value(block, "ext:module"),
                "count": _number_value(block, "ext:number"),
                "rated_w": _number_value(block, "ext:ratedPowerOutput"),
                "efficiency": _number_value(block, "ext:ratedModuleConversionEfficiency"),
                "specific_type": _quoted_value(block, "ext:specificType"),
            }
    return {"systems": systems, "inverters": inverters, "modules": modules}


def _number_from_text(text: str, property_name: str) -> float | None:
    match = re.search(rf"{re.escape(property_name)}\s+({NUMBER_PATTERN})", text)
    return float(match.group(1)) if match else None


def _normalise_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _system_name_for_stem(stem: str, systems: dict[str, dict[str, Any]]) -> str:
    aliases = {_normalise_name("Indoor Sports Centre"): "Indoor_Sport_Center"}
    if _normalise_name(stem) in aliases:
        return aliases[_normalise_name(stem)]
    matches = [name for name in systems if _normalise_name(name) == _normalise_name(stem)]
    if len(matches) != 1:
        raise ValueError(f"Could not map {stem!r} to one metadata system: {matches}")
    return matches[0]


def _parse_inverter_kw(model: str | None) -> float | None:
    if not model:
        return None
    solar_edge = re.search(r"SE(\d+)(?:[_ ](\d))?K", model.upper())
    if solar_edge:
        integer = float(solar_edge.group(1))
        decimal = float(solar_edge.group(2) or 0) / 10
        return integer + decimal
    sma_large = re.search(r"(\d{4,5})TL", model.upper())
    if sma_large:
        return float(sma_large.group(1)) / 1000
    sma_stp = re.search(r"STP[_]?(\d+)(?:[_](\d+))?", model.upper())
    if sma_stp:
        integer = float(sma_stp.group(1))
        suffix = sma_stp.group(2)
        if suffix == "40" and integer == 50:
            return integer
        if suffix and len(suffix) == 1:
            return integer + float(suffix) / 10
        return integer
    return None


def _build_system_info(
    system_name: str,
    metadata: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    system = metadata["systems"][system_name]
    inverters = metadata["inverters"]
    modules = metadata["modules"]
    inverter_entities = [inverters[name] for name in system["parts"]]
    groups: list[dict[str, Any]] = []
    for inverter_name, inverter in zip(system["parts"], inverter_entities, strict=True):
        for module_name in inverter["parts"]:
            module = dict(modules[module_name])
            module.update({"entity": module_name, "inverter_entity": inverter_name})
            groups.append(module)
    inverter_models = [entity["model"] for entity in inverter_entities]
    parsed_inverter_ratings = [_parse_inverter_kw(model) for model in inverter_models]
    if all(value is not None for value in parsed_inverter_ratings):
        inverter_ac_kw = float(sum(value for value in parsed_inverter_ratings if value is not None))
        rating_source = "parsed inverter model names"
    elif system["rated_kw"] is not None:
        inverter_ac_kw = float(system["rated_kw"])
        rating_source = "system ratedPowerOutput fallback"
    else:
        inverter_ac_kw = None
        rating_source = "unavailable"
    return {
        "system_entity": system_name,
        "latitude": system["latitude"],
        "longitude": system["longitude"],
        "system_rated_kw": system["rated_kw"],
        "connection_date": system["connection_date"],
        "inverter_models": inverter_models,
        "inverter_ac_kw": inverter_ac_kw,
        "inverter_rating_source": rating_source,
        "module_groups": groups,
        "module_count": int(sum(group["count"] or 0 for group in groups)),
        "module_dc_kw": float(sum((group["count"] or 0) * (group["rated_w"] or 0) for group in groups) / 1000),
    }


def _site_paths() -> list[tuple[Path, bool]]:
    paths: list[tuple[Path, bool]] = []
    with_optimizer = PV_ROOT / "PV stations with panel level optimizer" / "Site level dataset"
    without_optimizer = PV_ROOT / "PV stations without panel level optimizer" / "Site level dataset"
    paths.extend((path, True) for path in sorted(with_optimizer.glob("*.csv")))
    paths.extend((path, False) for path in sorted(without_optimizer.glob("*.csv")))
    return paths


def _inverter_paths() -> list[Path]:
    directory = PV_ROOT / "PV stations with panel level optimizer" / "Inverter level dataset"
    return sorted(directory.glob("*.csv"))


def _load_weather_1min() -> tuple[pd.DataFrame, dict[str, int]]:
    series: dict[str, pd.Series] = {}
    missing_counts: dict[str, int] = {}
    for name, (subdirectory, pattern, column) in WEATHER_SPECS.items():
        chunks: list[pd.Series] = []
        for path in sorted((METEO_ROOT / subdirectory).glob(pattern)):
            frame = pd.read_csv(path, usecols=["Time", column], na_values=["NA"])
            times = pd.to_datetime(frame["Time"], errors="coerce")
            if times.isna().any():
                raise ValueError(f"Unparseable weather timestamps in {path}")
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            indexed = frame.set_index(times.dt.tz_localize(TIMEZONE))[column]
            chunks.append(indexed)
        combined = pd.concat(chunks).sort_index()
        if combined.index.has_duplicates:
            raise ValueError(f"Duplicate timestamps in weather channel {name}")
        series[name] = combined
        missing_counts[name] = int(combined.isna().sum())
    weather = pd.concat(series, axis=1)
    if weather.index.has_duplicates:
        raise ValueError("Duplicate timestamps in combined weather data")
    return weather, missing_counts


def _prepare_weather(weather_1min: pd.DataFrame, freq: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    # The two site populations use different conventions. Hourly inverter
    # rows are interval means labelled at the end. The 15-minute optimizer
    # site power matches the following three five-minute inverter samples and
    # is therefore left-labelled. Normalize both to internal interval-start
    # labels before applying midpoint solar geometry.
    offset = pd.tseries.frequencies.to_offset(freq)
    label_basis = SOURCE_LABEL_BASIS_BY_FREQUENCY[freq]
    if label_basis == "interval-end":
        weather = weather_1min.resample(freq, closed="right", label="right").mean()
        weather.index = weather.index - offset
    else:
        weather = weather_1min.resample(freq, closed="left", label="left").mean()
    weather["ghi"] = weather["ghi"].clip(lower=0)
    representative_times = weather.index + offset / 2
    solar_position = LOCATION.get_solarposition(representative_times)
    solar_position.index = weather.index
    erbs = pvlib.irradiance.erbs(weather["ghi"], solar_position["zenith"], weather.index)
    weather["dni"] = erbs["dni"].clip(lower=0)
    weather["dhi"] = erbs["dhi"].clip(lower=0)
    return weather[["ghi", "dni", "dhi", "temp_air", "wind_speed"]], solar_position


def _read_site(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(path, usecols=["Time", "generation(kWh)", "power(W)"], na_values=["NA"])
    times = pd.to_datetime(frame.pop("Time"), errors="coerce")
    if times.isna().any():
        raise ValueError(f"Unparseable site timestamps in {path}")
    frame.index = times.dt.tz_localize(TIMEZONE)
    duplicate_count = int(frame.index.duplicated().sum())
    if duplicate_count:
        frame = frame[~frame.index.duplicated(keep="first")]
    frame = frame.sort_index()
    for column in frame:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    deltas = frame.index.to_series().diff().dropna().dt.total_seconds().div(60)
    median_minutes = float(deltas.median()) if not deltas.empty else None
    if median_minutes is None:
        freq = None
    elif median_minutes <= 15.0:
        freq = "15min"
    elif median_minutes <= 60.0:
        freq = "h"
    else:
        freq = None
    source_period_start = str(frame.index[0]) if len(frame) else None
    source_period_end = str(frame.index[-1]) if len(frame) else None
    label_basis = SOURCE_LABEL_BASIS_BY_FREQUENCY.get(freq) if freq is not None else None
    if label_basis == "interval-end":
        frame.index = frame.index - pd.tseries.frequencies.to_offset(freq)
    return frame, {
        "raw_rows": int(len(frame) + duplicate_count),
        "rows_after_duplicate_removal": int(len(frame)),
        "duplicate_timestamps": duplicate_count,
        "median_step_minutes": median_minutes,
        "freq": freq,
        "source_period_start": source_period_start,
        "source_period_end": source_period_end,
        "source_label_basis": label_basis,
        "internal_label_basis": "interval-start",
        "internal_label_shift_minutes": (
            -median_minutes if median_minutes is not None and label_basis == "interval-end" else 0.0
        ),
    }


def _hours_per_step(freq: str) -> float:
    return 0.25 if freq == "15min" else 1.0


def _finite_metric(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int | None]:
    if actual.size == 0:
        return {"n": 0, "bias_W": None, "mae_W": None, "rmse_W": None, "r": None, "r2": None}
    error = predicted - actual
    actual_centered = actual - actual.mean()
    if np.std(actual) == 0 or np.std(predicted) == 0:
        correlation = None
    else:
        correlation = float(np.corrcoef(actual, predicted)[0, 1])
    denominator = float(np.sum(actual_centered**2))
    return {
        "n": int(actual.size),
        "bias_W": float(error.mean()),
        "mae_W": float(np.abs(error).mean()),
        "rmse_W": float(np.sqrt(np.mean(error**2))),
        "r": correlation,
        "r2": float(1 - np.sum(error**2) / denominator) if denominator else None,
    }


def _orientation_search(
    target: pd.Series,
    weather: pd.DataFrame,
    solar_position: pd.DataFrame,
    valid_mask: pd.Series,
    stride: int,
    freq: str,
) -> dict[str, float | int]:
    train = pd.DataFrame({"target": target, "valid": valid_mask}).join(weather, how="left")
    train = train[train["valid"] & (train["ghi"] >= DAYLIGHT_GHI_THRESHOLD)].dropna()
    sampled = train.iloc[::stride]
    best: dict[str, float | int] | None = None
    for tilt in ORIENTATION_TILTS:
        for azimuth in ORIENTATION_AZIMUTHS:
            positions = solar_position.loc[sampled.index]
            representative_times = sampled.index + pd.tseries.frequencies.to_offset(freq) / 2
            dni_extra = pvlib.irradiance.get_extra_radiation(representative_times)
            dni_extra.index = sampled.index
            airmass = pvlib.atmosphere.get_relative_airmass(positions["apparent_zenith"])
            poa = pvlib.irradiance.get_total_irradiance(
                surface_tilt=tilt,
                surface_azimuth=azimuth,
                solar_zenith=positions["apparent_zenith"],
                solar_azimuth=positions["azimuth"],
                dni=sampled["dni"],
                ghi=sampled["ghi"],
                dhi=sampled["dhi"],
                dni_extra=dni_extra,
                airmass=airmass,
                model=PV_MODEL_OPTIONS["transposition_model"],
                model_perez=PV_MODEL_OPTIONS["model_perez"],
                albedo=PV_MODEL_OPTIONS["albedo"],
            )["poa_global"].to_numpy(dtype=float)
            actual = sampled["target"].to_numpy(dtype=float)
            finite = np.isfinite(poa) & np.isfinite(actual) & (poa > 0)
            if not finite.any():
                continue
            poa = poa[finite]
            actual = actual[finite]
            scale = float(np.dot(poa, actual) / np.dot(poa, poa))
            residual = poa * scale - actual
            candidate = {
                "tilt_deg": float(tilt),
                "azimuth_deg": float(azimuth),
                "poa_scale_W_per_Wm2": scale,
                "train_poa_rmse_W": float(np.sqrt(np.mean(residual**2))),
                "train_poa_n": int(actual.size),
                "orientation_fit_stride": int(stride),
            }
            if best is None or candidate["train_poa_rmse_W"] < best["train_poa_rmse_W"]:
                best = candidate
    if best is None:
        raise ValueError("No valid daylight rows were available for orientation fitting")
    return best


def _module_params(cec_db: pd.DataFrame, cec_column: str, cache: dict[str, PVModuleParams]) -> PVModuleParams:
    if cec_column in cache:
        return cache[cec_column]
    row = cec_db[cec_column]
    params = PVModuleParams(
        Mpp=float(row["STC"]),
        Vmp=float(row["V_mp_ref"]),
        Imp=float(row["I_mp_ref"]),
        Voc=float(row["V_oc_ref"]),
        Isc=float(row["I_sc_ref"]),
        T_Pmax_pct=float(row["gamma_r"]),
        T_Voc_pct=float(row["beta_oc"]) / float(row["V_oc_ref"]) * 100,
        T_Isc_pct=float(row["alpha_sc"]) / float(row["I_sc_ref"]) * 100,
        N_Cells=int(round(float(row["N_s"]))),
        Module_Efficiency=float(row["STC"]) / (float(row["A_c"]) * 1000),
        celltype="monoSi",
        Name=f"CEC analog {cec_column}",
        alpha_sc_abs=float(row["alpha_sc"]),
        beta_voc_abs=float(row["beta_oc"]),
        gamma_pmp=float(row["gamma_r"]),
    )
    cache[cec_column] = params
    return params


def _run_breos_model(
    weather: pd.DataFrame,
    info: dict[str, Any],
    orientation: dict[str, float | int],
    cec_db: pd.DataFrame,
    module_cache: dict[str, PVModuleParams],
    freq: str,
) -> tuple[pd.Series, dict[str, Any]]:
    dc_power: pd.Series | None = None
    dc_peak_w = 0.0
    mappings: list[str] = []
    for group in info["module_groups"]:
        model_name = str(group["model"])
        if model_name not in CEC_MAP:
            raise KeyError(f"No CEC analog mapping for {model_name}")
        cec_column = CEC_MAP[model_name]
        params = _module_params(cec_db, cec_column, module_cache)
        group_dc = calculate_pv_production_dc(
            weather_data=weather,
            location=LOCATION,
            tilt=float(orientation["tilt_deg"]),
            surface_azimuth=float(orientation["azimuth_deg"]),
            n_modules=int(group["count"]),
            pv_params=params,
            freq=freq,
            degradation_rate=0.0,
            verbose=False,
            **PV_MODEL_OPTIONS,
        )
        dc_power = group_dc if dc_power is None else dc_power.add(group_dc, fill_value=0.0)
        dc_peak_w += int(group["count"]) * params.Mpp
        mappings.append(f"{model_name}->{cec_column}")
    if dc_power is None or dc_peak_w <= 0:
        raise ValueError("Metadata has no usable module groups")
    inverter_ac_kw = info["inverter_ac_kw"]
    if inverter_ac_kw is None or inverter_ac_kw <= 0:
        raise ValueError("Metadata has no usable inverter rating")
    loading_ratio = dc_peak_w / (float(inverter_ac_kw) * 1000)
    ac_power = dc_to_ac(
        dc_power,
        pv_peak_power_w=dc_peak_w,
        inverter_loading_ratio=loading_ratio,
        inverter_efficiency=0.96,
    )
    return ac_power, {
        "model_dc_peak_w": float(dc_peak_w),
        "inverter_ac_kw": float(inverter_ac_kw),
        "dc_ac_ratio": float(loading_ratio),
        "module_mappings": mappings,
    }


def _physical_mask(target: pd.Series, info: dict[str, Any]) -> tuple[pd.Series, float | None]:
    capacities = [
        float(value)
        for value in (info["inverter_ac_kw"], info["system_rated_kw"])
        if value is not None and value > 0
    ]
    capacity_kw = max(capacities) if capacities else None
    upper_w = float(capacity_kw) * 1000 * 1.10 if capacity_kw else None
    valid = target.notna() & (target >= 0)
    if upper_w is not None:
        valid &= target <= upper_w
    return valid, upper_w


def _site_mapping_row(path: Path, with_optimizer: bool, info: dict[str, Any], freq: str | None) -> dict[str, Any]:
    groups = info["module_groups"]
    models = ";".join(str(group["model"]) for group in groups)
    cec_mappings = ";".join(f"{group['model']}->{CEC_MAP.get(str(group['model']), 'missing')}" for group in groups)
    qualities = []
    for group in groups:
        model = str(group["model"])
        qualities.append("direct CEC analog" if model in {"JKM390M_6RL3_TV", "JKM395M_6RL3_V"} else "approximate CEC analog")
    return {
        "site": path.stem,
        "source_file": str(path.relative_to(SOURCE_DIR)),
        "panel_level_optimizer": with_optimizer,
        "system_entity": info["system_entity"],
        "frequency": freq,
        "connection_date": info["connection_date"],
        "metadata_latitude": info["latitude"],
        "metadata_longitude": info["longitude"],
        "system_rated_kw": info["system_rated_kw"],
        "inverter_ac_kw": info["inverter_ac_kw"],
        "inverter_rating_source": info["inverter_rating_source"],
        "inverter_models": ";".join(str(value) for value in info["inverter_models"]),
        "module_count": info["module_count"],
        "module_dc_kw": info["module_dc_kw"],
        "module_models": models,
        "cec_analog_mappings": cec_mappings,
        "module_mapping_quality": ";".join(qualities),
        "tilt_azimuth_in_local_ttl": False,
    }


def _score_site(
    path: Path,
    with_optimizer: bool,
    info: dict[str, Any],
    weather_by_freq: dict[str, pd.DataFrame],
    solar_position_by_freq: dict[str, pd.DataFrame],
    cec_db: pd.DataFrame,
    module_cache: dict[str, PVModuleParams],
    inverter_files: list[Path],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any] | None,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    target_frame, raw_facts = _read_site(path)
    freq = raw_facts["freq"]
    quality = _site_mapping_row(path, with_optimizer, info, freq)
    quality.update(raw_facts)
    if freq not in weather_by_freq:
        quality["status"] = "unsupported_frequency"
        row = {"site": path.stem, "status": "unsupported_frequency", "panel_level_optimizer": with_optimizer}
        consistency = _optimizer_consistency(path, with_optimizer, inverter_files, info)
        return row, quality, {}, consistency, np.array([]), np.array([]), np.array([])

    weather = weather_by_freq[freq]
    solar_position = solar_position_by_freq[freq]
    target = target_frame["power(W)"]
    in_weather = (target.index >= weather.index[0]) & (target.index <= weather.index[-1])
    target = target[in_weather]
    weather_at_target = weather.reindex(target.index)
    physical, upper_w = _physical_mask(target, info)
    weather_complete = weather_at_target.notna().all(axis=1)
    valid_for_fit = physical & weather_complete
    train_mask = valid_for_fit & (target.index.year <= TRAIN_LAST_YEAR)
    test_mask = valid_for_fit & (target.index.year == TEST_YEAR)
    quality.update(
        {
            "weather_period_start": str(weather.index[0]),
            "weather_period_end": str(weather.index[-1]),
            "rows_outside_weather_period": int((~in_weather).sum()),
            "weather_matched_rows": int(in_weather.sum()),
            "weather_complete_rows": int(weather_complete.sum()),
            "missing_power_rows": int(target.isna().sum()),
            "negative_power_rows": int((target < 0).fillna(False).sum()),
            "above_physical_upper_rows": int((target > upper_w).fillna(False).sum()) if upper_w else 0,
            "physical_upper_W": upper_w,
            "train_daylight_rows": int((train_mask & (weather_at_target["ghi"] >= DAYLIGHT_GHI_THRESHOLD)).sum()),
            "test_daylight_rows": int((test_mask & (weather_at_target["ghi"] >= DAYLIGHT_GHI_THRESHOLD)).sum()),
        }
    )
    if train_mask.sum() < 100 or test_mask.sum() < 100:
        quality["status"] = "insufficient_holdout_rows"
        row = {
            "site": path.stem,
            "status": "insufficient_holdout_rows",
            "panel_level_optimizer": with_optimizer,
            "frequency": freq,
            "train_rows": int(train_mask.sum()),
            "test_rows": int(test_mask.sum()),
        }
        consistency = _optimizer_consistency(path, with_optimizer, inverter_files, info)
        return row, quality, {}, consistency, np.array([]), np.array([]), np.array([])

    stride = 4 if freq == "15min" else 1
    orientation = _orientation_search(target, weather, solar_position, train_mask, stride, freq)
    model_weather = weather.loc[
        target.index.min().floor(freq) : target.index.max().floor(freq)
    ]
    model, model_facts = _run_breos_model(
        model_weather,
        info,
        orientation,
        cec_db,
        module_cache,
        freq,
    )
    pair = pd.DataFrame({"target": target}).join(weather_at_target).join(model.rename("model"), how="left")
    pair["physical"] = physical
    pair["train"] = train_mask
    pair["test"] = test_mask
    pair["daylight"] = pair["ghi"] >= DAYLIGHT_GHI_THRESHOLD
    pair = pair[pair["physical"] & pair["model"].notna() & pair["ghi"].notna()]
    train_daylight = pair[pair["train"] & pair["daylight"]]
    test_daylight = pair[pair["test"] & pair["daylight"]]
    test_all = pair[pair["test"]]
    model_train = train_daylight["model"].to_numpy(dtype=float)
    target_train = train_daylight["target"].to_numpy(dtype=float)
    if model_train.size == 0 or np.dot(model_train, model_train) <= 0:
        raise ValueError("BREOS model has no usable training output")
    calibration_gain = float(np.dot(model_train, target_train) / np.dot(model_train, model_train))
    test_actual = test_daylight["target"].to_numpy(dtype=float)
    test_raw = test_daylight["model"].to_numpy(dtype=float)
    test_predicted = test_raw * calibration_gain
    train_predicted = train_daylight["model"].to_numpy(dtype=float) * calibration_gain
    raw_metrics = _finite_metric(test_actual, test_raw)
    calibrated_metrics = _finite_metric(test_actual, test_predicted)
    train_metrics = _finite_metric(target_train, train_predicted)
    step_hours = _hours_per_step(freq)
    test_energy_actual = float(test_all["target"].sum() * step_hours / 1000)
    test_energy_raw = float(test_all["model"].sum() * step_hours / 1000)
    test_energy_predicted = test_energy_raw * calibration_gain
    test_energy_bias_pct = (
        (test_energy_predicted - test_energy_actual) / test_energy_actual * 100 if test_energy_actual else None
    )
    expected_test = weather.loc["2023"].index
    quality.update(
        {
            "status": "scored",
            "matched_model_rows": int(len(pair)),
            "test_all_rows": int(len(test_all)),
            "test_coverage_fraction": float(len(test_all) / len(expected_test)) if len(expected_test) else None,
        }
    )
    site_row: dict[str, Any] = {
        "site": path.stem,
        "status": "scored",
        "panel_level_optimizer": with_optimizer,
        "frequency": freq,
        "system_entity": info["system_entity"],
        "connection_date": info["connection_date"],
        "module_count": info["module_count"],
        "module_dc_kw": info["module_dc_kw"],
        "system_rated_kw": info["system_rated_kw"],
        "inverter_ac_kw": info["inverter_ac_kw"],
        "model_dc_peak_w": model_facts["model_dc_peak_w"],
        "dc_ac_ratio": model_facts["dc_ac_ratio"],
        "tilt_deg": orientation["tilt_deg"],
        "azimuth_deg": orientation["azimuth_deg"],
        "orientation_train_poa_rmse_W": orientation["train_poa_rmse_W"],
        "orientation_train_poa_n": orientation["train_poa_n"],
        "calibration_gain": calibration_gain,
        "train_daylight_n": train_metrics["n"],
        "train_daylight_bias_W": train_metrics["bias_W"],
        "train_daylight_mae_W": train_metrics["mae_W"],
        "train_daylight_rmse_W": train_metrics["rmse_W"],
        "train_daylight_r": train_metrics["r"],
        "train_daylight_r2": train_metrics["r2"],
        "test_daylight_n": calibrated_metrics["n"],
        "test_daylight_bias_W": calibrated_metrics["bias_W"],
        "test_daylight_mae_W": calibrated_metrics["mae_W"],
        "test_daylight_rmse_W": calibrated_metrics["rmse_W"],
        "test_daylight_nrmse_pct_inverter": calibrated_metrics["rmse_W"] / (info["inverter_ac_kw"] * 10)
        if calibrated_metrics["rmse_W"] is not None and info["inverter_ac_kw"]
        else None,
        "test_daylight_r": calibrated_metrics["r"],
        "test_daylight_r2": calibrated_metrics["r2"],
        "test_raw_daylight_rmse_W": raw_metrics["rmse_W"],
        "test_raw_daylight_r": raw_metrics["r"],
        "test_all_n": int(len(test_all)),
        "test_measured_energy_kwh": test_energy_actual,
        "test_raw_model_energy_kwh": test_energy_raw,
        "test_calibrated_model_energy_kwh": test_energy_predicted,
        "test_calibrated_energy_bias_pct": test_energy_bias_pct,
        "module_mappings": ";".join(model_facts["module_mappings"]),
    }
    orientation_row = {
        "site": path.stem,
        "system_entity": info["system_entity"],
        "panel_level_optimizer": with_optimizer,
        "frequency": freq,
        **orientation,
        "calibration_gain": calibration_gain,
    }
    quality["module_mappings"] = ";".join(model_facts["module_mappings"])
    consistency = _optimizer_consistency(path, with_optimizer, inverter_files, info)
    return site_row, quality, orientation_row, consistency, test_actual, test_predicted, test_raw


def _inverter_files_for_site(site_stem: str, paths: list[Path]) -> list[Path]:
    matches = []
    for path in paths:
        base = re.sub(r"_Inverter(?:_\d+)?$", "", path.stem)
        if _normalise_name(base) == _normalise_name(site_stem):
            matches.append(path)
    if _normalise_name(site_stem) == _normalise_name("Indoor Sports Centre"):
        matches = [
            path
            for path in paths
            if _normalise_name(re.sub(r"_Inverter(?:_\d+)?$", "", path.stem))
            == _normalise_name("Indoor Sports Centre")
        ]
    return matches


def _read_power_series(path: Path, column: str) -> pd.Series:
    frame = pd.read_csv(path, usecols=["Time", column], na_values=["NA"])
    times = pd.to_datetime(frame.pop("Time"), errors="coerce")
    if times.isna().any():
        raise ValueError(f"Unparseable timestamps in {path}")
    frame.index = times.dt.tz_localize(TIMEZONE)
    frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[~frame.index.duplicated(keep="first")].sort_index()
    return frame[column]


def _optimizer_consistency(
    site_path: Path,
    with_optimizer: bool,
    inverter_paths: list[Path],
    info: dict[str, Any],
) -> dict[str, Any] | None:
    if not with_optimizer:
        return None
    matching = _inverter_files_for_site(site_path.stem, inverter_paths)
    if not matching:
        return {
            "site": site_path.stem,
            "status": "no_matching_inverter_files",
            "inverter_file_count": 0,
        }
    site = _read_power_series(site_path, "power(W)")
    inverter_series = [
        _read_power_series(path, "totalActivePower(W)")
        .resample("15min", closed="left", label="left")
        .mean()
        for path in matching
    ]
    inverter = pd.concat(inverter_series, axis=1, sort=False).sum(axis=1, min_count=len(inverter_series))
    common = pd.concat({"site": site, "inverter": inverter}, axis=1, sort=False).dropna()
    common = common[(common["site"] >= 0) & (common["inverter"] >= 0)]
    capacity_kw = info["inverter_ac_kw"] or info["system_rated_kw"]
    if capacity_kw:
        common = common[common[["site", "inverter"]].le(float(capacity_kw) * 1000 * 1.10).all(axis=1)]
    actual = common["site"].to_numpy(dtype=float)
    derived = common["inverter"].to_numpy(dtype=float)
    metrics = _finite_metric(actual, derived)
    daily = common.groupby(common.index.normalize()).sum() * 0.25 / 1000
    daily_error = daily["inverter"] - daily["site"]
    relative = np.abs(daily_error) / np.maximum(np.minimum(np.abs(daily["inverter"]), np.abs(daily["site"])), 1e-12)
    return {
        "site": site_path.stem,
        "status": "scored",
        "inverter_file_count": len(matching),
        "inverter_files": ";".join(path.name for path in matching),
        "common_15min_rows": metrics["n"],
        "bias_inverter_minus_site_W": metrics["bias_W"],
        "mae_W": metrics["mae_W"],
        "rmse_W": metrics["rmse_W"],
        "r": metrics["r"],
        "daily_rows": int(len(daily)),
        "daily_mean_abs_difference_kwh": float(np.abs(daily_error).mean()) if len(daily_error) else None,
        "daily_max_abs_difference_kwh": float(np.abs(daily_error).max()) if len(daily_error) else None,
        "daily_difference_lt_0_1_kwh_pct": float((np.abs(daily_error) < 0.1).mean() * 100) if len(daily_error) else None,
        "daily_difference_lt_1_kwh_pct": float((np.abs(daily_error) < 1.0).mean() * 100) if len(daily_error) else None,
        "daily_mean_relative_difference_pct": float(relative.mean() * 100) if len(relative) else None,
    }


def _aggregate_metrics(
    site_rows: list[dict[str, Any]],
    population_actual: dict[str, list[np.ndarray]],
    population_predicted: dict[str, list[np.ndarray]],
    population_raw: dict[str, list[np.ndarray]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "test_year": TEST_YEAR,
        "daylight_ghi_threshold_W_m2": DAYLIGHT_GHI_THRESHOLD,
        "sites_total": len(site_rows),
        "sites_scored": int(sum(row.get("status") == "scored" for row in site_rows)),
    }
    for population, arrays in population_actual.items():
        if not arrays:
            continue
        actual = np.concatenate(arrays)
        predicted = np.concatenate(population_predicted[population])
        metrics = _finite_metric(actual, predicted)
        scoped = [row for row in site_rows if row.get("status") == "scored" and row.get("panel_level_optimizer") == (population == "with_optimizer")]
        result[population] = {
            "pooled_daylight_metrics": metrics,
            "pooled_daylight_raw_metrics": _finite_metric(
                actual, np.concatenate(population_raw[population])
            ),
            "macro_mean_test_rmse_W": float(np.mean([row["test_daylight_rmse_W"] for row in scoped])) if scoped else None,
            "macro_mean_test_raw_rmse_W": float(np.mean([row["test_raw_daylight_rmse_W"] for row in scoped])) if scoped else None,
            "macro_mean_test_r": float(np.mean([row["test_daylight_r"] for row in scoped if row["test_daylight_r"] is not None])) if scoped else None,
            "sites_scored": len(scoped),
        }
    all_actual = [array for arrays in population_actual.values() for array in arrays]
    all_predicted = [array for arrays in population_predicted.values() for array in arrays]
    all_raw = [array for arrays in population_raw.values() for array in arrays]
    if all_actual:
        result["all_scored_sites"] = {
            "pooled_daylight_metrics": _finite_metric(np.concatenate(all_actual), np.concatenate(all_predicted)),
            "pooled_daylight_raw_metrics": _finite_metric(np.concatenate(all_actual), np.concatenate(all_raw)),
        }
        scored = [row for row in site_rows if row.get("status") == "scored"]
        result["all_scored_sites"]["macro_mean_test_rmse_W"] = float(np.mean([row["test_daylight_rmse_W"] for row in scored]))
        result["all_scored_sites"]["macro_mean_test_raw_rmse_W"] = float(np.mean([row["test_raw_daylight_rmse_W"] for row in scored]))
        result["all_scored_sites"]["macro_mean_test_r"] = float(np.mean([row["test_daylight_r"] for row in scored if row["test_daylight_r"] is not None]))
    return result


def _input_files() -> list[Path]:
    return sorted(path for path in SOURCE_DIR.rglob("*") if path.is_file())


def _timing_contract() -> dict[str, Any]:
    index = pd.date_range("2022-01-01 00:00", periods=121, freq="min", tz=TIMEZONE)
    values = np.arange(121, dtype=float)
    synthetic = pd.DataFrame(
        {"ghi": values, "temp_air": values + 10.0, "wind_speed": values + 1.0},
        index=index,
    )
    prepared, solar_position = _prepare_weather(synthetic, "h")
    label = pd.Timestamp("2022-01-01 00:00", tz=TIMEZONE)
    expected_weather_mean = float(np.arange(1, 61, dtype=float).mean())
    observed_weather_mean = float(prepared.loc[label, "ghi"])
    midpoint = label + pd.Timedelta(minutes=30)
    expected_solar = LOCATION.get_solarposition(pd.DatetimeIndex([midpoint])).iloc[0]
    observed_zenith = float(solar_position.loc[label, "apparent_zenith"])
    expected_zenith = float(expected_solar["apparent_zenith"])
    if observed_weather_mean != expected_weather_mean:
        raise AssertionError("Right-labelled weather aggregation contract failed")
    if not np.isclose(observed_zenith, expected_zenith):
        raise AssertionError("Midpoint solar-position contract failed")

    hourly_path = sorted(
        (
            PV_ROOT
            / "PV stations without panel level optimizer"
            / "Site level dataset"
        ).glob("*.csv")
    )[0]
    hourly = pd.read_csv(
        hourly_path,
        usecols=["generation(kWh)", "power(W)"],
        na_values=["NA"],
    )
    hourly_generation = pd.to_numeric(hourly["generation(kWh)"], errors="coerce")
    hourly_power = pd.to_numeric(hourly["power(W)"], errors="coerce")
    finite_hourly = hourly_generation.notna() & hourly_power.notna()
    hourly_residual = hourly_generation[finite_hourly] - hourly_power[finite_hourly] / 1000

    optimizer_site_path = (
        PV_ROOT
        / "PV stations with panel level optimizer"
        / "Site level dataset"
        / "Indoor Sports Centre.csv"
    )
    optimizer_inverter_path = (
        PV_ROOT
        / "PV stations with panel level optimizer"
        / "Inverter level dataset"
        / "Indoor Sports Centre_Inverter.csv"
    )
    optimizer_site = _read_power_series(optimizer_site_path, "power(W)")
    optimizer_inverter = _read_power_series(
        optimizer_inverter_path, "totalActivePower(W)"
    )

    def alignment_metrics(candidate: pd.Series) -> dict[str, float | int]:
        pair = pd.concat(
            {"site": optimizer_site, "candidate": candidate},
            axis=1,
            sort=False,
        ).dropna()
        pair = pair[(pair >= 0).all(axis=1)]
        error = pair["candidate"] - pair["site"]
        return {
            "n": int(len(pair)),
            "mae_W": float(error.abs().mean()),
            "rmse_W": float(np.sqrt(np.mean(error**2))),
            "r": float(pair.corr().iloc[0, 1]),
        }

    optimizer_alignment = {
        "source_site_file": str(optimizer_site_path.relative_to(SOURCE_DIR)),
        "source_inverter_file": str(optimizer_inverter_path.relative_to(SOURCE_DIR)),
        "preceding_interval_right_label": alignment_metrics(
            optimizer_inverter.resample("15min", closed="right", label="right").mean()
        ),
        "following_interval_left_label": alignment_metrics(
            optimizer_inverter.resample("15min", closed="left", label="left").mean()
        ),
    }
    if not (
        optimizer_alignment["following_interval_left_label"]["rmse_W"]
        < optimizer_alignment["preceding_interval_right_label"]["rmse_W"]
    ):
        raise AssertionError("Optimizer site label-basis diagnostic failed")

    population_rows: list[dict[str, Any]] = []
    all_inverter_paths = _inverter_paths()
    for site_path, with_optimizer in _site_paths():
        if not with_optimizer:
            continue
        matching = _inverter_files_for_site(site_path.stem, all_inverter_paths)
        if not matching:
            continue
        site_power = _read_power_series(site_path, "power(W)")
        components = [
            _read_power_series(path, "totalActivePower(W)") for path in matching
        ]
        summed = pd.concat(components, axis=1, sort=False).sum(
            axis=1,
            min_count=len(components),
        )

        def population_metrics(candidate: pd.Series) -> dict[str, float | int]:
            pair = pd.concat(
                {"site": site_power, "candidate": candidate},
                axis=1,
                sort=False,
            ).dropna()
            pair = pair[(pair >= 0).all(axis=1)]
            error = pair["candidate"] - pair["site"]
            return {
                "n": int(len(pair)),
                "rmse_W": float(np.sqrt(np.mean(error**2))),
                "r": float(pair.corr().iloc[0, 1]),
            }

        right_metrics = population_metrics(
            summed.resample("15min", closed="right", label="right").mean()
        )
        left_metrics = population_metrics(
            summed.resample("15min", closed="left", label="left").mean()
        )
        population_rows.append(
            {
                "site": site_path.stem,
                "right_label": right_metrics,
                "left_label": left_metrics,
                "left_lower_rmse": left_metrics["rmse_W"] < right_metrics["rmse_W"],
                "left_higher_r": left_metrics["r"] > right_metrics["r"],
            }
        )

    left_lower_rmse_sites = sum(row["left_lower_rmse"] for row in population_rows)
    left_higher_r_sites = sum(row["left_higher_r"] for row in population_rows)
    if left_lower_rmse_sites != len(population_rows) or left_higher_r_sites != len(
        population_rows
    ):
        raise AssertionError("Optimizer population label-basis diagnostic failed")
    population_summary = {
        "sites": len(population_rows),
        "left_lower_rmse_sites": int(left_lower_rmse_sites),
        "left_higher_r_sites": int(left_higher_r_sites),
        "median_right_rmse_W": float(
            np.median([row["right_label"]["rmse_W"] for row in population_rows])
        ),
        "median_left_rmse_W": float(
            np.median([row["left_label"]["rmse_W"] for row in population_rows])
        ),
        "median_right_r": float(
            np.median([row["right_label"]["r"] for row in population_rows])
        ),
        "median_left_r": float(
            np.median([row["left_label"]["r"] for row in population_rows])
        ),
        "site_results": population_rows,
    }

    return {
        "status": "passed",
        "source_pv_label_basis_by_frequency": SOURCE_LABEL_BASIS_BY_FREQUENCY,
        "weather_interval_by_frequency": {
            "h": "right-closed and right-labelled before internal shift",
            "15min": "left-closed and left-labelled",
        },
        "internal_label_basis": "interval-start",
        "solar_position_basis": "interval midpoint",
        "synthetic_hour_internal_label": str(label),
        "synthetic_expected_weather_mean": expected_weather_mean,
        "synthetic_observed_weather_mean": observed_weather_mean,
        "synthetic_midpoint": str(midpoint),
        "synthetic_expected_apparent_zenith": expected_zenith,
        "synthetic_observed_apparent_zenith": observed_zenith,
        "hourly_energy_identity_check": {
            "source_file": str(hourly_path.relative_to(SOURCE_DIR)),
            "finite_rows": int(finite_hourly.sum()),
            "max_abs_generation_minus_power_times_one_hour_kWh": float(
                hourly_residual.abs().max()
            ),
            "mean_abs_generation_minus_power_times_one_hour_kWh": float(
                hourly_residual.abs().mean()
            ),
        },
        "optimizer_15min_label_diagnostic": optimizer_alignment,
        "optimizer_15min_population_label_diagnostic": population_summary,
    }


def _write_readme(output: Path, aggregate: dict[str, Any]) -> None:
    all_sites = aggregate.get("all_scored_sites", {})
    calibrated = all_sites.get("pooled_daylight_metrics", {})
    raw = all_sites.get("pooled_daylight_raw_metrics", {})
    text = f"""# HKUST rooftop exploratory holdout check

This package corrects the interval-label defect in the 2026-08-29 run. The
hourly non-optimizer rows are interval means labelled at the interval end.
They are shifted back one hour. For all 37 optimizer sites, 15-minute site
power matches left-labelled five-minute inverter aggregation better than
right-labelled aggregation by both RMSE and correlation. Those rows are not
shifted. Weather is aggregated with the matching convention. Solar position
is evaluated at each interval midpoint.

The old package at `results/validation_hkust_20260829` paired each PV row with
weather from the following interval. Its reported metrics are invalid and it
is preserved only as failure evidence.

## Result

- Scored sites: {aggregate.get('sites_scored')} of {aggregate.get('sites_total')}
- Test year: {aggregate.get('test_year')}
- Measured energy: {aggregate.get('test_measured_energy_kwh'):.3f} kWh
- Raw model energy: {aggregate.get('test_raw_model_energy_kwh'):.3f} kWh
- Raw energy bias: {aggregate.get('test_raw_energy_bias_pct'):.3f}%
- Calibrated model energy: {aggregate.get('test_calibrated_model_energy_kwh'):.3f} kWh
- Calibrated energy bias: {aggregate.get('test_calibrated_energy_bias_pct'):.3f}%
- Pooled daylight raw RMSE: {raw.get('rmse_W'):.3f} W
- Pooled daylight raw correlation: {raw.get('r'):.6f}
- Pooled daylight calibrated RMSE: {calibrated.get('rmse_W'):.3f} W
- Pooled daylight calibrated correlation: {calibrated.get('r'):.6f}

## Interpretation boundary

This is an exploratory calibrated holdout check, not publication-grade
external validation of the full PV chain. The downloaded Brick metadata has
no tilt or azimuth fields. The driver fits one effective orientation and one
scale factor per site on 2021 to 2022 data, then scores 2023 without refitting.
Several named modules also require approximate CEC catalogue analogues. DNI
and DHI are derived from the single campus GHI station with Erbs rather than
measured locally at each roof.

The PV chain is explicit in `run_config.json`: Perez transposition, the
allsitescomposite1990 coefficients, Ashrae beam IAM, Marion diffuse IAM,
Faiman temperature, albedo 0.25, midpoint solar position, no bifacial model,
and the recorded static PVWatts loss components.

## Reproduction

Run the copied driver with a new output directory. It refuses to replace any
generated files:

```bash
MPLCONFIGDIR=/tmp/hkust-mpl \\
  /home/leo/code/breos/.venv/bin/python \\
  drivers/hkust_validate.py --output-dir /path/to/new-empty-directory
```

`provenance.json` records the clean BREOS commit, dependencies, driver hash,
configuration hash, complete input manifest hash, and output hashes.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the HKUST rooftop PV validation.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()

    existing = [output / name for name in OUTPUT_FILES if (output / name).exists()]
    if existing:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"Refusing to replace existing result files: {names}")
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(SOURCE_DIR)
    if not METADATA_FILE.is_file():
        raise FileNotFoundError(METADATA_FILE)
    if not BREOS_ROOT.is_dir():
        raise FileNotFoundError(BREOS_ROOT)

    metadata_graph = _parse_metadata(METADATA_FILE)
    site_paths = _site_paths()
    inverter_paths = _inverter_paths()
    if len(site_paths) != 60:
        raise ValueError(f"Expected 60 site files, found {len(site_paths)}")
    if len(inverter_paths) != 44:
        raise ValueError(f"Expected 44 inverter files, found {len(inverter_paths)}")
    cec_db = pvlib.pvsystem.retrieve_sam("CECMod")
    weather_1min, weather_missing_1min = _load_weather_1min()
    timing_contract = _timing_contract()
    weather_by_freq: dict[str, pd.DataFrame] = {}
    solar_position_by_freq: dict[str, pd.DataFrame] = {}
    weather_missing_resampled: dict[str, dict[str, int]] = {}
    for freq in ("15min", "h"):
        weather_by_freq[freq], solar_position_by_freq[freq] = _prepare_weather(weather_1min, freq)
        weather_missing_resampled[freq] = {
            column: int(weather_by_freq[freq][column].isna().sum()) for column in weather_by_freq[freq]
        }

    site_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    orientation_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    consistency_rows: list[dict[str, Any]] = []
    population_actual: dict[str, list[np.ndarray]] = {"with_optimizer": [], "without_optimizer": []}
    population_predicted: dict[str, list[np.ndarray]] = {"with_optimizer": [], "without_optimizer": []}
    population_raw: dict[str, list[np.ndarray]] = {"with_optimizer": [], "without_optimizer": []}
    module_cache: dict[str, PVModuleParams] = {}
    seen_systems: set[str] = set()

    for path, with_optimizer in site_paths:
        system_name = _system_name_for_stem(path.stem, metadata_graph["systems"])
        if system_name in seen_systems:
            raise ValueError(f"Duplicate site mapping for {system_name}")
        seen_systems.add(system_name)
        info = _build_system_info(system_name, metadata_graph)
        target_frame, raw_facts = _read_site(path)
        mapping_rows.append(_site_mapping_row(path, with_optimizer, info, raw_facts["freq"]))
        try:
            site_row, quality, orientation, consistency, actual, predicted, raw = _score_site(
                path,
                with_optimizer,
                info,
                weather_by_freq,
                solar_position_by_freq,
                cec_db,
                module_cache,
                _inverter_files_for_site(path.stem, inverter_paths),
            )
        except Exception as exc:
            site_row = {
                "site": path.stem,
                "status": "model_error",
                "panel_level_optimizer": with_optimizer,
                "error": f"{type(exc).__name__}: {exc}",
            }
            quality = {
                "site": path.stem,
                "panel_level_optimizer": with_optimizer,
                "status": "model_error",
                "error": f"{type(exc).__name__}: {exc}",
                **raw_facts,
            }
            orientation = {}
            consistency = None
            actual = np.array([])
            predicted = np.array([])
            raw = np.array([])
        site_rows.append(site_row)
        quality_rows.append(quality)
        if orientation:
            orientation_rows.append(orientation)
        if consistency:
            consistency_rows.append(consistency)
        if actual.size:
            population = "with_optimizer" if with_optimizer else "without_optimizer"
            population_actual[population].append(actual)
            population_predicted[population].append(predicted)
            population_raw[population].append(raw)

    if seen_systems != set(metadata_graph["systems"]):
        missing = sorted(set(metadata_graph["systems"]) - seen_systems)
        raise ValueError(f"Site files did not cover metadata systems: {missing}")

    output.mkdir(parents=True, exist_ok=False)
    driver_dir = output / "drivers"
    driver_dir.mkdir()
    copied_driver = driver_dir / Path(__file__).name
    copied_driver.write_bytes(Path(__file__).read_bytes())
    pd.DataFrame(site_rows).to_csv(output / "site_metrics.csv", index=False, float_format="%.9g")
    pd.DataFrame(orientation_rows).to_csv(output / "orientation_fit.csv", index=False, float_format="%.9g")
    pd.DataFrame(quality_rows).to_csv(output / "data_quality.csv", index=False, float_format="%.9g")
    pd.DataFrame(mapping_rows).to_csv(output / "metadata_mapping.csv", index=False, float_format="%.9g")
    pd.DataFrame(consistency_rows).to_csv(output / "optimizer_consistency.csv", index=False, float_format="%.9g")

    optimizer_consistency_summary = {
        "sites_with_optimizer": len(consistency_rows),
        "sites_scored": int(sum(row.get("status") == "scored" for row in consistency_rows)),
        "common_15min_rows": int(sum(row.get("common_15min_rows", 0) or 0 for row in consistency_rows)),
        "daily_rows": int(sum(row.get("daily_rows", 0) or 0 for row in consistency_rows)),
        "weighted_daily_difference_lt_0_1_kwh_pct": None,
        "weighted_daily_difference_lt_1_kwh_pct": None,
        "mean_site_daily_mean_abs_difference_kwh": None,
    }
    scored_consistency = [row for row in consistency_rows if row.get("status") == "scored"]
    if scored_consistency:
        daily_total = sum(row.get("daily_rows", 0) or 0 for row in scored_consistency)
        optimizer_consistency_summary["weighted_daily_difference_lt_0_1_kwh_pct"] = float(
            sum((row.get("daily_difference_lt_0_1_kwh_pct") or 0) * (row.get("daily_rows", 0) or 0) for row in scored_consistency)
            / daily_total
        ) if daily_total else None
        optimizer_consistency_summary["weighted_daily_difference_lt_1_kwh_pct"] = float(
            sum((row.get("daily_difference_lt_1_kwh_pct") or 0) * (row.get("daily_rows", 0) or 0) for row in scored_consistency)
            / daily_total
        ) if daily_total else None
        optimizer_consistency_summary["mean_site_daily_mean_abs_difference_kwh"] = float(
            np.mean([row["daily_mean_abs_difference_kwh"] for row in scored_consistency if row["daily_mean_abs_difference_kwh"] is not None])
        )
    _write_json(output / "optimizer_consistency_summary.json", optimizer_consistency_summary)

    aggregate = _aggregate_metrics(site_rows, population_actual, population_predicted, population_raw)
    aggregate["test_measured_energy_kwh"] = float(
        sum(row.get("test_measured_energy_kwh", 0) or 0 for row in site_rows if row.get("status") == "scored")
    )
    aggregate["test_raw_model_energy_kwh"] = float(
        sum(row.get("test_raw_model_energy_kwh", 0) or 0 for row in site_rows if row.get("status") == "scored")
    )
    aggregate["test_calibrated_model_energy_kwh"] = float(
        sum(row.get("test_calibrated_model_energy_kwh", 0) or 0 for row in site_rows if row.get("status") == "scored")
    )
    aggregate["test_raw_energy_bias_pct"] = (
        (aggregate["test_raw_model_energy_kwh"] - aggregate["test_measured_energy_kwh"])
        / aggregate["test_measured_energy_kwh"]
        * 100
        if aggregate["test_measured_energy_kwh"]
        else None
    )
    aggregate["test_calibrated_energy_bias_pct"] = (
        (aggregate["test_calibrated_model_energy_kwh"] - aggregate["test_measured_energy_kwh"])
        / aggregate["test_measured_energy_kwh"]
        * 100
        if aggregate["test_measured_energy_kwh"]
        else None
    )
    _write_json(output / "aggregate_metrics.json", aggregate)

    source_files = _input_files()
    with (output / "input_manifest.sha256").open("w", encoding="utf-8") as handle:
        for path in source_files:
            handle.write(f"{_sha256(path)}  {path.relative_to(SOURCE_DIR).as_posix()}\n")

    metadata_text = METADATA_FILE.read_text(encoding="utf-8")
    dataset_facts = {
        "source_record": "https://doi.org/10.5061/dryad.m37pvmd99",
        "source_paper": "https://doi.org/10.1038/s41597-025-04397-y",
        "source_period": "2021-2023",
        "source_location": "HKUST campus, Sai Kung District, Hong Kong",
        "source_location_coordinates": {"latitude": 22.3363, "longitude": 114.2634},
        "source_claimed_site_count": 60,
        "source_claimed_module_count": 6085,
        "local_site_file_count": len(site_paths),
        "local_site_files_with_panel_optimizer": int(sum(with_optimizer for _, with_optimizer in site_paths)),
        "local_site_files_without_panel_optimizer": int(sum(not with_optimizer for _, with_optimizer in site_paths)),
        "local_inverter_file_count": len(inverter_paths),
        "local_weather_file_count": len(_input_files()) - len(site_paths) - len(inverter_paths) - 2,
        "local_metadata_system_count": len(metadata_graph["systems"]),
        "local_metadata_inverter_count": len(metadata_graph["inverters"]),
        "local_metadata_module_entity_count": len(metadata_graph["modules"]),
        "local_metadata_module_count": int(sum(row["module_count"] for row in mapping_rows)),
        "local_metadata_system_rated_kw_sum": float(sum(row["system_rated_kw"] or 0 for row in mapping_rows)),
        "local_metadata_module_dc_kw_sum": float(sum(row["module_dc_kw"] or 0 for row in mapping_rows)),
        "local_ttl_contains_tilt_or_azimuth_text": bool(re.search(r"tilt|azimuth", metadata_text, flags=re.IGNORECASE)),
        "local_ttl_geometry_fields_used": False,
        "local_weather_rows_1min": int(len(weather_1min)),
        "local_weather_period_start": str(weather_1min.index[0]),
        "local_weather_period_end": str(weather_1min.index[-1]),
        "weather_missing_rows_1min": weather_missing_1min,
        "weather_missing_rows_resampled": weather_missing_resampled,
        "irradiance_treatment": "DNI and DHI derived from measured GHI with pvlib.irradiance.erbs; they are not measured channels.",
        "timezone_treatment": "Source timestamps are naive and assumed Asia/Hong_Kong local time. Hourly non-optimizer PV rows are right-labelled and shifted back one hour. The 15-minute optimizer site power is left-labelled and is not shifted. Weather is aggregated with the corresponding convention, and solar geometry is evaluated at each interval midpoint.",
        "model": "BREOS calculate_pv_production_dc with explicitly recorded PV options, then dc_to_ac with 0.96 inverter efficiency.",
        "effective_pv_model_options": PV_MODEL_OPTIONS,
        "orientation_treatment": "Effective fixed tilt and azimuth selected by a Perez POA grid search using midpoint solar positions on 2021-2022 daylight rows; no test-year refit.",
        "scale_treatment": "One non-negative-through-origin output scale factor fitted on 2021-2022 daylight rows per site; no test-year refit.",
        "model_target": "site-level power(W), treated as inverter AC power",
        "breos_root": str(BREOS_ROOT),
        "breos_commit": _git_value("rev-parse", "HEAD"),
        "breos_worktree_status": _git_value("status", "--short"),
        "dependencies": {
            "python": platform_version(),
            "breos": _dependency_version("breos"),
            "numpy": _dependency_version("numpy"),
            "pandas": _dependency_version("pandas"),
            "pvlib": _dependency_version("pvlib"),
        },
    }
    _write_json(output / "dataset_facts.json", dataset_facts)
    _write_json(output / "timing_contract.json", timing_contract)

    run_config = {
        "source_dir": str(SOURCE_DIR),
        "site_file_count": len(site_paths),
        "breos_root": str(BREOS_ROOT),
        "timezone": TIMEZONE,
        "location": {"latitude": LOCATION.latitude, "longitude": LOCATION.longitude},
        "weather_resampling": {
            "h": "right-closed and right-labelled mean, then shift one hour to interval-start labels",
            "15min": "left-closed and left-labelled mean; labels already denote interval starts",
        },
        "source_pv_label_basis_by_frequency": SOURCE_LABEL_BASIS_BY_FREQUENCY,
        "internal_label_basis": "interval-start",
        "solar_position_basis": "interval midpoint",
        "irradiance_decomposition": "pvlib.irradiance.erbs from GHI and midpoint pvlib solar zenith",
        "effective_pv_model_options": PV_MODEL_OPTIONS,
        "inverter_efficiency": 0.96,
        "training_period": f"<= {TRAIN_LAST_YEAR}",
        "test_period": str(TEST_YEAR),
        "daylight_ghi_threshold_W_m2": DAYLIGHT_GHI_THRESHOLD,
        "orientation_tilts_deg": list(ORIENTATION_TILTS),
        "orientation_azimuths_deg": list(ORIENTATION_AZIMUTHS),
        "orientation_fit_stride": {"15min": 4, "h": 1},
        "module_mapping": CEC_MAP,
        "module_mapping_note": "CEC analogs are used because the downloaded Brick file names modules but does not provide a complete CEC parameter set; Sunman and JKM415 mappings are approximate.",
        "physical_target_filter": "0 <= power(W) <= 1.10 * max(parsed inverter AC rating, system ratedPowerOutput)",
        "overwrite_guard": "refuse an existing non-empty generated result directory",
        "interpretation": "exploratory calibrated holdout check because authoritative array geometry is absent from the local metadata and several module mappings use CEC analogues",
    }
    _write_json(output / "run_config.json", run_config)
    configuration_hash = _sha256(output / "run_config.json")
    _write_readme(output, aggregate)

    output_hashes = {
        name: _sha256(output / name)
        for name in OUTPUT_FILES
        if name not in {"provenance.json", "README.md"} and (output / name).exists()
    }
    output_hashes["README.md"] = _sha256(output / "README.md")
    provenance = {
        "driver": str(copied_driver),
        "driver_sha256": _sha256(copied_driver),
        "input_manifest": str(output / "input_manifest.sha256"),
        "input_manifest_sha256": _sha256(output / "input_manifest.sha256"),
        "configuration_file": str(output / "run_config.json"),
        "configuration_sha256": configuration_hash,
        "configuration_sha256_method": "SHA-256 of the exact UTF-8 run_config.json bytes",
        "input_hashes_include": "all files under the downloaded HKUST dataset directory",
        "output_hashes_exclude": "provenance.json itself",
        "output_hashes": output_hashes,
    }
    _write_json(output / "provenance.json", provenance)

    print(json.dumps({"aggregate_metrics": aggregate, "optimizer_consistency": optimizer_consistency_summary}, indent=2, sort_keys=True, allow_nan=False))


def platform_version() -> str:
    return platform.python_version()


if __name__ == "__main__":
    main()
