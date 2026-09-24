#!/usr/bin/env python3
"""Task 5(f): does a capacity-proportional power limit defer charge or destroy it?

Batch B1 measured that no design changes its replacement year under a 1 C or
0.5 C limit, but published no mechanism. The untested explanation is timing
rather than energy: a 1 kW limit fills a 1 kWh battery in an hour while the PV
surplus window is many hours long, so the limit defers the charge without
reducing the energy stored, leaving depth-of-cycle and the SOH trajectory
nearly unchanged.

That prediction is falsifiable. If it holds, annual battery throughput and FEC
must be flat across the three power models even for designs whose NPV moves.
This replays the two designs Batch B1 identified -- the 1 kWh design whose NPV
does move, and the 6 kWh design whose limits never bind -- under all three
power models and reports throughput and FEC beside the NPV delta.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import shlex
import sys
import tomllib
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

_spec = importlib.util.spec_from_file_location("repro", PROJECT_ROOT / "tools/reproduce_article1.py")
repro = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(repro)

import breos  # noqa: E402
from breos.optimization import evaluate_projected_design  # noqa: E402

GI = "Projected_Grid_Independence_%"
NPV = "Projected_NPV_Eur"

# The two designs Batch B1 singled out, at the tilt/azimuth it reported them at.
# The 1 kWh design is the test: its NPV moves, so its dispatch demonstrably
# changes. The 6 kWh design is the control: 1 C on 6 kWh is 6 kW against a
# 5-module array peaking near 2.75 kW DC, so no limit can bind.
DESIGNS = {
    "6mod_1kWh_30_200": dict(n_modules=6, battery_kwh=1.0, tilt=30.0, azimuth=200.0),
    "5mod_6kWh_40_195": dict(n_modules=5, battery_kwh=6.0, tilt=40.0, azimuth=195.0),
}

ARMS = {
    "P0_constant": "validation/article1/article1-projected-optimization.toml",
    "P1_1c": "validation/article1/revision-0.6.1/article1-power-1c.toml",
    "P2_0p5c": "validation/article1/revision-0.6.1/article1-power-0p5c.toml",
}

CHARGE = "Battery_Charge_Throughput_kWh"
DISCHARGE = "Battery_Discharge_Throughput_kWh"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rlp-directory", type=Path, required=True)
    ap.add_argument("--execution-backend", default="numba")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    configs, hashes = {}, {}
    for arm, rel in ARMS.items():
        raw = (PROJECT_ROOT / rel).read_bytes()
        configs[arm] = tomllib.loads(raw.decode("utf-8"))
        hashes[arm] = hashlib.sha256(raw).hexdigest()

    weather, load, weather_path, rlp_path = repro._load_inputs(configs["P0_constant"], args.rlp_directory)

    yearly_rows, summary_rows = [], []
    for design_label, design in DESIGNS.items():
        for arm in ARMS:
            r = evaluate_projected_design(
                weather, load, configs[arm], execution_backend=args.execution_backend, **design
            )
            y = r.yearly.copy()
            if "Battery_Annual_FEC" not in y:
                raise RuntimeError("projected results do not expose exact Battery_Annual_FEC")
            # This is the all-pack annual ledger. Unlike differencing the
            # installed pack's cumulative counter, it retains the retired
            # pack's part-year contribution when a replacement occurs.
            y["Annual_FEC"] = y["Battery_Annual_FEC"]
            y["FEC_Reset_Year"] = y["Replacements"] > 0
            y.insert(0, "Arm", arm)
            y.insert(0, "Design", design_label)
            yearly_rows.append(y)

            rep = y.index[y["Replacements"] > 0]
            summary_rows.append(
                {
                    "Design": design_label,
                    "Arm": arm,
                    "Charge_Throughput_Total_kWh": float(y[CHARGE].sum()),
                    "Discharge_Throughput_Total_kWh": float(y[DISCHARGE].sum()),
                    "Charge_Throughput_Year1_kWh": float(y[CHARGE].iloc[0]),
                    "FEC_Lifetime": float(y["Annual_FEC"].sum()),
                    "FEC_Annual_Mean": float(y["Annual_FEC"].mean()),
                    "FEC_Year1": float(y["Annual_FEC"].iloc[0]),
                    # Whether a power limit can move a replacement year is a
                    # race between the SOH it saves and one year of ageing.
                    "SOH_Year_Before_Replacement_%": (
                        None if len(rep) == 0 else float(y.loc[rep[0] - 1, "Battery_SOH_%"])
                    ),
                    "SOH_Annual_Decrement_pp": (
                        float((y["Battery_SOH_%"].iloc[0] - y.loc[rep[0] - 1, "Battery_SOH_%"]) / (rep[0] - 1))
                        if len(rep) and rep[0] >= 2
                        else float("nan")
                    ),
                    "SOC_Normalized_Mean_%": float(y["Battery_SOC_Normalized_Mean_%"].mean()),
                    "SOC_Absolute_Mean_%": float(y["Battery_SOC_Absolute_Mean_%"].mean()),
                    "Replacement_Year": None if len(rep) == 0 else int(y.loc[rep[0], "Year"]),
                    "Total_Replacements": int(r.metrics["Projected_Total_Replacements"]),
                    "Final_SOH_%": float(r.metrics["Projected_Final_SOH_%"]),
                    GI: float(r.metrics[GI]),
                    NPV: float(r.metrics[NPV]),
                }
            )

    yearly = pd.concat(yearly_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    # Everything is judged against the constant-power arm, so express each
    # C-rate arm as a relative change from it.
    base = summary[summary["Arm"] == "P0_constant"].set_index("Design")
    for col, out in (
        ("Charge_Throughput_Total_kWh", "Charge_Throughput_Delta_%"),
        ("FEC_Lifetime", "FEC_Delta_%"),
        ("SOC_Normalized_Mean_%", "SOC_Normalized_Delta_pp"),
    ):
        ref = summary["Design"].map(base[col])
        summary[out] = (summary[col] - ref) / ref * 100.0 if out.endswith("_%") else summary[col] - ref
    summary["NPV_Delta_Eur"] = summary[NPV] - summary["Design"].map(base[NPV])
    # The decisive ratio: SOH the limit saves against one year of ageing. A
    # replacement year can only move when the former approaches the latter.
    summary["SOH_Saved_pp"] = summary["SOH_Year_Before_Replacement_%"] - summary["Design"].map(
        base["SOH_Year_Before_Replacement_%"]
    )
    summary["SOH_Saved_Vs_One_Year"] = summary["SOH_Saved_pp"] / summary["SOH_Annual_Decrement_pp"]
    summary["GI_Delta_pp"] = summary[GI] - summary["Design"].map(base[GI])

    args.output.mkdir(parents=True, exist_ok=True)
    yearly_path = args.output / "task5f_throughput_by_year.csv"
    summary_path = args.output / "task5f_throughput_summary.csv"
    yearly.to_csv(yearly_path, index=False)
    summary.to_csv(summary_path, index=False)

    show = [
        "Design",
        "Arm",
        "Charge_Throughput_Total_kWh",
        "Charge_Throughput_Delta_%",
        "FEC_Lifetime",
        "FEC_Annual_Mean",
        "FEC_Delta_%",
        "Replacement_Year",
        "SOH_Saved_pp",
        "SOH_Annual_Decrement_pp",
        "SOH_Saved_Vs_One_Year",
        "NPV_Delta_Eur",
    ]
    print(summary[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    provenance = {
        "breos_version": breos.__version__,
        "breos_source": repro._git_revision(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "arms": {arm: {"config": rel, "config_sha256": hashes[arm]} for arm, rel in ARMS.items()},
        "designs": DESIGNS,
        "hypothesis": (
            "Timing, not energy: the limit defers charge without reducing stored energy. "
            "Predicts throughput and FEC flat across arms even where NPV moves."
        ),
        "weather": str(weather_path.relative_to(PROJECT_ROOT)),
        "weather_uncompressed_sha256": repro._sha256(weather_path, decompress_gzip=weather_path.suffix == ".gz"),
        "external_rlp_sha256": repro._sha256(rlp_path) if rlp_path else None,
        "execution_backend": args.execution_backend,
        "outputs": {
            yearly_path.name: repro._sha256(yearly_path),
            summary_path.name: repro._sha256(summary_path),
        },
    }
    (args.output / "task5f_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"\nWrote {yearly_path}\nWrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
