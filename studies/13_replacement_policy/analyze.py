"""Study 13 analysis: best replacement policy in BREOS's rule family vs the default.

Default = eol 0.70, no minimum remaining years. For every (site, model, cost, battery): the default
NPV, the best policy and its gain, never replacing, the best threshold-only policy (rem 0) and
the best remaining-years-only policy (eol 0.70).

Usage: uv run --with tabulate python analyze.py -> output/summary.md, output/best.csv
"""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
NPV, REP = "Projected_NPV", "Projected_Total_Replacements"
DEFAULT = "eol0.70_rem0_keep"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    d = pd.concat((pd.read_pickle(f) for f in sorted(glob.glob(str(HERE / "data/policy/*.pkl")))), ignore_index=True)
    rows = []
    for key, g in d.groupby(["site", "model", "cost", "battery_kwh"]):
        s = g.set_index("policy")
        best = s[NPV].idxmax()
        thr = s[s.index.str.endswith("_rem0_keep")][NPV].idxmax()
        rem = s[s.index.str.startswith("eol0.70_")][NPV].idxmax()
        rows.append(dict(zip(["site", "model", "cost", "battery_kwh"], key)) | {
            "default_npv": s.at[DEFAULT, NPV], "default_replacements": s.at[DEFAULT, REP],
            "best_policy": best, "best_npv": s.at[best, NPV], "best_replacements": s.at[best, REP],
            "gain_best": s.at[best, NPV] - s.at[DEFAULT, NPV],
            "gain_never": s.at["never", NPV] - s.at[DEFAULT, NPV],
            "best_threshold_only": thr, "gain_threshold_only": s.at[thr, NPV] - s.at[DEFAULT, NPV],
            "best_remaining_only": rem, "gain_remaining_only": s.at[rem, NPV] - s.at[DEFAULT, NPV],
            "final_soh_best": s.at[best, "Projected_Final_SOH_%"],
            "gi_loss_best_pp": s.at[DEFAULT, "Projected_Grid_Independence_%"] - s.at[best, "Projected_Grid_Independence_%"],
        })
    b = pd.DataFrame(rows)
    b.to_csv(OUT / "best.csv", index=False)
    agg = b.groupby(["model", "cost", "battery_kwh"]).agg(
        gain_best_median=("gain_best", "median"), gain_best_max=("gain_best", "max"),
        gain_never_median=("gain_never", "median"), gain_threshold_median=("gain_threshold_only", "median"),
        gain_remaining_median=("gain_remaining_only", "median"),
        never_is_best_pct=("best_policy", lambda x: 100 * (x == "never").mean()),
        gi_loss_pp_median=("gi_loss_best_pp", "median"),
    ).round(0)
    lines = ["# Study 13 summary\n", "Gains in EUR of projected NPV over the default rule (TMY weather).\n",
             agg.to_markdown(), "", "## Best policy per case\n",
             b[["site", "model", "cost", "battery_kwh", "default_replacements", "best_policy", "best_replacements",
                "gain_best", "gain_never"]].round(0).to_markdown(index=False), ""]
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines[:4]))


if __name__ == "__main__":
    main()
