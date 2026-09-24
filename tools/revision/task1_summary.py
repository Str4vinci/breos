#!/usr/bin/env python3
"""Task 1: assemble the comparison CSVs and the Markdown summary."""

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
MODELS = (("base-v1", "Field v1"), ("field-v2", "Field v2"), ("laboratory", "Laboratory"))
MANUSCRIPT = {
    "base-v1": {"Modules": 6, "Battery_kWh": 0.0, GI: 40.71, NPV: 5451.83},
    "field-v2": {"Modules": 5, "Battery_kWh": 6.0, GI: 60.71, NPV: 5498.69},
    "laboratory": {"Modules": 8, "Battery_kWh": 8.0, GI: 75.64, NPV: 6217.10},
}


def _replacement_year(root: Path, slug: str, rep: str) -> int | None:
    path = root / slug / "representatives" / rep / "yearly_summary.csv"
    if not path.is_file():
        return None
    y = pd.read_csv(path)
    hit = y.index[y["Replacements"] > 0]
    return None if len(hit) == 0 else int(y.loc[hit[0], "Year"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--front-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rep_rows, front_rows, check_rows = [], [], []
    for slug, label in MODELS:
        reps = pd.read_csv(args.front_root / slug / "pareto_representatives.csv")
        front = pd.read_csv(args.front_root / slug / "pareto_results.csv")
        prov = json.loads((args.front_root / slug / "reproduction.json").read_text())

        for _, r in reps.iterrows():
            rep_rows.append(
                {
                    "Model": label,
                    "Bundle": slug,
                    "Representative": r["Representative"],
                    "Modules": int(r["Modules"]),
                    "Battery_kWh": r["Battery_kWh"],
                    "Tilt_deg": r["Tilt"],
                    "Azimuth_deg": r["Azimuth"],
                    "Grid_Independence_%": r[GI],
                    "NPV_Eur": r[NPV],
                    "Payback_Year": r.get(PB),
                    "Replacement_Year": _replacement_year(args.front_root, slug, r["Representative"]),
                    "Total_Replacements": r["Projected_Total_Replacements"],
                }
            )

        b = front[front["Battery_kWh"] > 0]
        front_rows.append(
            {
                "Model": label,
                "Bundle": slug,
                "Calendar_Model": prov["resolved_config"]["battery"]["calendar_model"],
                "NSGA2_Generations_Run": prov["optimization"]["iterations"],
                "NSGA2_Generation_Cap": prov["optimization"]["settings"]["n_gen"],
                "Front_Size": len(front),
                "PV_Only_Points": int((front["Battery_kWh"] == 0).sum()),
                "Battery_Points": len(b),
                "Battery_Points_NPV_gt_5400": int((b[NPV] > 5400).sum()),
                "Best_Battery_NPV_Eur": float(b[NPV].max()) if len(b) else float("nan"),
                "Max_NPV_Eur": float(front[NPV].max()),
                "Max_GI_%": float(front[GI].max()),
                "Points_With_A_Replacement": int((front["Projected_Total_Replacements"] > 0).sum()),
                "Pareto_SHA256": prov["optimization"]["pareto_sha256"],
            }
        )

        mn = reps[reps["Representative"] == "max_npv"].iloc[0]
        exp = MANUSCRIPT[slug]
        check_rows.append(
            {
                "Model": label,
                "Manuscript_Modules": exp["Modules"],
                "Reproduced_Modules": int(mn["Modules"]),
                "Manuscript_Battery_kWh": exp["Battery_kWh"],
                "Reproduced_Battery_kWh": mn["Battery_kWh"],
                "Manuscript_GI_%": exp[GI],
                "Reproduced_GI_%": round(float(mn[GI]), 2),
                "GI_Delta_pp": round(float(mn[GI]) - exp[GI], 4),
                "Manuscript_NPV_Eur": exp[NPV],
                "Reproduced_NPV_Eur": round(float(mn[NPV]), 2),
                "NPV_Delta_Eur": round(float(mn[NPV]) - exp[NPV], 4),
            }
        )

    pd.DataFrame(rep_rows).to_csv(args.output / "task1a_representatives.csv", index=False)
    pd.DataFrame(front_rows).to_csv(args.output / "task1a_front_statistics.csv", index=False)
    pd.DataFrame(check_rows).to_csv(args.output / "task1a_manuscript_check.csv", index=False)
    for name in ("task1a_representatives", "task1a_front_statistics", "task1a_manuscript_check"):
        print(f"Wrote {args.output / (name + '.csv')}")
    print()
    print(pd.DataFrame(check_rows).to_string(index=False))
    print()
    print(pd.DataFrame(front_rows).drop(columns=["Pareto_SHA256"]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
