"""Study 12 analysis: spread of projected results across orders of the same 19 weather years.

Usage: uv run --with tabulate python analyze.py -> output/summary.md, output/orders.csv
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
NPV, REP = "Projected_NPV", "Projected_Total_Replacements"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    d = pd.concat((pd.read_pickle(f) for f in sorted(glob.glob(str(HERE / "data/order/*.pkl")))), ignore_index=True)
    d["design"] = d.modules.astype(str) + "x" + d.battery_kwh.map("{:g}kWh".format)
    rows = []
    for (site, variant, design), g in d.groupby(["site", "variant", "design"]):
        named = g.set_index("order")
        rnd = g[g.order.str.startswith("random_")]
        rows.append({
            "site": site, "variant": variant, "design": design,
            "chronological": named.at["chronological", NPV], "reversed": named.at["reversed", NPV],
            "worst_first": named.at["worst_first", NPV], "best_first": named.at["best_first", NPV],
            "tmy_repeated": named.at["tmy_repeated", NPV],
            "random_min": rnd[NPV].min(), "random_p5": rnd[NPV].quantile(0.05), "random_median": rnd[NPV].median(),
            "random_p95": rnd[NPV].quantile(0.95), "random_max": rnd[NPV].max(), "random_std": rnd[NPV].std(),
            "range_all_orders": g[g.order != "tmy_repeated"][NPV].max() - g[g.order != "tmy_repeated"][NPV].min(),
            "replacements_seen": ",".join(map(str, sorted(g[g.order != "tmy_repeated"][REP].unique()))),
            "replacement_share": g[g.order.str.startswith("random_")][REP].value_counts(normalize=True).round(3).to_dict(),
        })
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "orders.csv", index=False)
    cols = ["site", "variant", "design", "chronological", "worst_first", "best_first", "random_p5", "random_median",
            "random_p95", "random_std", "range_all_orders", "tmy_repeated", "replacements_seen"]
    lines = ["# Study 12 summary\n", "NPV in EUR over 19 years; the same 19 years in every order.\n",
             s[cols].round(0).to_markdown(index=False), "",
             "## Range of NPV across orders, by design and variant (median over sites)\n",
             s.groupby(["variant", "design"])[["range_all_orders", "random_std"]].median().round(0).to_markdown(), ""]
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines[-3:]))


if __name__ == "__main__":
    main()
