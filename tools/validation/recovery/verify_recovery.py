"""Verify recovered validation outputs against the recorded historical results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _close(label: str, actual: float, expected: float, tolerance: float) -> None:
    difference = abs(actual - expected)
    if difference > tolerance:
        raise AssertionError(
            f"{label}: got {actual!r}, expected {expected!r} within {tolerance}; difference={difference}"
        )
    print(f"PASS {label}: {actual:.12g}")


def _one(frame: pd.DataFrame, **conditions: Any) -> pd.Series:
    selected = frame
    for column, value in conditions.items():
        selected = selected[selected[column] == value]
    if len(selected) != 1:
        raise AssertionError(f"Expected one row for {conditions}, found {len(selected)}")
    return selected.iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    root = args.output_root

    sandia = pd.read_csv(root / "validation_sandia_task13_recovered_20260902" / "thermal_metrics.csv")
    row = _one(sandia, model="breos_faiman_default", gpoa_threshold_W_m2=200)
    if int(row["n"]) != 26_023:
        raise AssertionError(f"Sandia n: {row['n']}")
    _close("Sandia bias C", float(row["bias_C"]), -0.038, 0.001)
    _close("Sandia RMSE C", float(row["rmse_C"]), 2.993, 0.001)
    _close("Sandia r", float(row["r"]), 0.970, 0.001)

    pcoe = pd.read_csv(root / "validation_pcoe_recovered_20260902" / "thermal_metrics.csv")
    row = _one(pcoe, model="breos_faiman_default", gpoa_threshold_W_m2=200)
    if int(row["n"]) != 12_708:
        raise AssertionError(f"PCoE n: {row['n']}")
    _close("PCoE bias C", float(row["bias_C"]), 2.694, 0.001)
    _close("PCoE RMSE C", float(row["rmse_C"]), 3.821, 0.001)

    reunion = pd.read_csv(root / "validation_reunion_microgrid_recovered_20260902" / "thermal_metrics.csv")
    row = _one(
        reunion,
        model="breos_faiman_default",
        poa_threshold_W_m2=200,
    )
    if int(row["n"]) != 108_986:
        raise AssertionError(f"Reunion thermal n: {row['n']}")
    _close("Reunion Faiman bias C", float(row["bias_C"]), 8.882, 0.001)
    _close("Reunion Faiman RMSE C", float(row["rmse_C"]), 11.162, 0.001)
    battery = pd.read_csv(root / "validation_reunion_microgrid_recovered_20260902" / "battery_thermal_metrics.csv")
    row = battery.iloc[0]
    if int(row["n"]) != 5_655:
        raise AssertionError(f"Reunion battery n: {row['n']}")
    _close("Reunion battery bias C", float(row["bias_C"]), -3.292, 0.001)

    orientation = pd.read_csv(
        root / "validation_orientation_diversity_recovered_20260902" / "orientation_screen_metrics.csv"
    )
    baseline = _one(orientation, panel="Total", model="common_radiation")
    enhanced = _one(orientation, panel="Total", model="radiation_plus_aoi")
    _close("Orientation baseline RMSE W", float(baseline["rmse_W"]), 159.126, 0.001)
    _close("Orientation AOI RMSE W", float(enhanced["rmse_W"]), 131.821, 0.001)
    _close("Orientation AOI R2", float(enhanced["r2"]), 0.883, 0.001)

    dkasc_dir = root / "validation_dkasc_recovered_20260902" / "results"
    leg_a = pd.read_csv(dkasc_dir / "transposition_leg_a.csv")
    leg_a_row = leg_a[
        (leg_a["transposition"] == "perez")
        & (leg_a["perez_set"] == "allsitescomposite1990")
        & (leg_a["albedo"] == "default")
    ].iloc[0]
    _close(
        "DKASC Leg A Perez bias pct",
        float(leg_a_row["bias_%"]),
        -0.4787011936222662,
        1e-9,
    )
    _close("DKASC Leg A Perez r", float(leg_a_row["r"]), 0.9931215534958022, 1e-9)
    leg_b = pd.read_csv(dkasc_dir / "transposition_leg_b.csv")
    leg_b_row = leg_b[leg_b["transposition"] == "perez"].iloc[0]
    _close(
        "DKASC Leg B Perez ratio error pct",
        float(leg_b_row["ratio_err_%"]),
        0.5949768499216912,
        1e-9,
    )

    hkust = json.loads(
        (
            root / "validation_hkust_timing-corrected-exploratory-v4_recovered_20260902" / "aggregate_metrics.json"
        ).read_text(encoding="utf-8")
    )
    if int(hkust["sites_scored"]) != 56:
        raise AssertionError(f"HKUST sites scored: {hkust['sites_scored']}")
    _close("HKUST raw energy bias pct", hkust["test_raw_energy_bias_pct"], 27.122, 0.001)
    pooled = hkust["all_scored_sites"]["pooled_daylight_raw_metrics"]
    _close("HKUST raw daylight RMSE W", pooled["rmse_W"], 10_635.397, 0.001)
    _close("HKUST raw daylight r", pooled["r"], 0.911280, 0.000001)

    print("All recovered local-data validations reproduce their recorded checkpoints.")


if __name__ == "__main__":
    main()
