#!/usr/bin/env python3
"""Task 5(d) and 5(e): representatives and small-battery counts across power models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

GI = "Projected_Grid_Independence_%"
NPV = "Projected_NPV_Eur"
PB = "Projected_Breakeven_Year_Exact"
CORE = ("max_npv", "knee", "max_gi")
FRONTS = (("base-v1", "Field v1"), ("field-v2", "Field v2"), ("laboratory", "Laboratory"))


def _replacement_year(bundle: Path, rep: str) -> int | None:
    path = bundle / "representatives" / rep / "yearly_summary.csv"
    if not path.is_file():
        return None
    y = pd.read_csv(path)
    hit = y.index[y["Replacements"] > 0]
    return None if len(hit) == 0 else int(y.loc[hit[0], "Year"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--constant-root", type=Path, required=True, help="P0 bundles (Batch A task1/)")
    ap.add_argument("--task5-root", type=Path, required=True, help="Holds 1c/ and 0p5c/")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    arms = {
        "P0_constant_4352W": args.constant_root,
        "P1_1C_symmetric": args.task5_root / "1c",
        "P2_0p5C_symmetric": args.task5_root / "0p5c",
    }

    rep_rows, stat_rows = [], []
    for power_model, root in arms.items():
        for slug, label in FRONTS:
            bundle = root / slug
            reps = pd.read_csv(bundle / "pareto_representatives.csv")
            front = pd.read_csv(bundle / "pareto_results.csv")
            prov = json.loads((bundle / "reproduction.json").read_text())

            for _, r in reps.iterrows():
                rep_rows.append(
                    {
                        "Power_Model": power_model,
                        "Degradation_Model": label,
                        "Representative": r["Representative"],
                        "Is_Core_Representative": r["Representative"] in CORE,
                        "Modules": int(r["Modules"]),
                        "Battery_kWh": r["Battery_kWh"],
                        "Tilt_deg": r["Tilt"],
                        "Azimuth_deg": r["Azimuth"],
                        "Grid_Independence_%": r[GI],
                        "NPV_Eur": r[NPV],
                        "Payback_Year": r.get(PB),
                        "Replacement_Year": _replacement_year(bundle, r["Representative"]),
                        "Total_Replacements": r["Projected_Total_Replacements"],
                    }
                )

            b = front[front["Battery_kWh"] > 0]
            small = front[(front["Battery_kWh"] > 0) & (front["Battery_kWh"] < 4.0)]
            stat_rows.append(
                {
                    "Power_Model": power_model,
                    "Degradation_Model": label,
                    "Front_Size": len(front),
                    "PV_Only_Points": int((front["Battery_kWh"] == 0).sum()),
                    "Battery_Points": len(b),
                    "Designs_Below_4kWh": len(small),
                    "Smallest_Battery_On_Front_kWh": float(b["Battery_kWh"].min()) if len(b) else float("nan"),
                    "Best_Battery_NPV_Eur": float(b[NPV].max()) if len(b) else float("nan"),
                    "Battery_Points_NPV_gt_5400": int((b[NPV] > 5400).sum()),
                    "Max_NPV_Eur": float(front[NPV].max()),
                    "Max_GI_%": float(front[GI].max()),
                    "NSGA2_Generations_Run": prov["optimization"]["iterations"],
                    "Pareto_SHA256": prov["optimization"]["pareto_sha256"],
                }
            )

    reps_df = pd.DataFrame(rep_rows)
    stats_df = pd.DataFrame(stat_rows)
    reps_df.to_csv(args.output / "task5d_representatives.csv", index=False)
    stats_df.to_csv(args.output / "task5e_front_statistics.csv", index=False)

    print("=== Task 5(d): the three core representatives, all nine combinations ===")
    core = reps_df[reps_df["Is_Core_Representative"]]
    print(
        core[
            [
                "Power_Model",
                "Degradation_Model",
                "Representative",
                "Modules",
                "Battery_kWh",
                "Tilt_deg",
                "Azimuth_deg",
                "Grid_Independence_%",
                "NPV_Eur",
                "Payback_Year",
                "Replacement_Year",
            ]
        ].to_string(index=False)
    )
    print("\n=== Task 5(e): front composition ===")
    print(stats_df.drop(columns=["Pareto_SHA256"]).to_string(index=False))
    print(f"\nWrote {args.output / 'task5d_representatives.csv'}")
    print(f"Wrote {args.output / 'task5e_front_statistics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
