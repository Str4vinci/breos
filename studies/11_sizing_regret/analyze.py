"""Study 11 analysis: what does choosing a size on the TMY cost in actual years?

Decisions, each made on the TMY grid and then scored in every actual year w (the same design,
weather w repeated over the 20 years):
- max_npv: the NPV-best design.
- target_XX: the NPV-best design whose projected grid independence reaches XX % (50..90).
For each decision: the design chosen on the TMY, the design that is best in year w, the NPV
regret NPV_w(best_w) - NPV_w(tmy pick), and whether the TMY pick misses the target in year w.
Robust pick: the design with the best mean over all 19 years (same rule), compared with the TMY pick.
Front: the TMY front's designs scored in year w, hypervolume against year w's own front
(objectives: grid dependence, -NPV; normalised per site/variant over all years, 10 % margin).

Usage: uv run python analyze.py -> output/*.csv, output/*.png, output/summary.md
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _shared.sizing import pareto_mask  # noqa: E402

OUT = HERE / "output"
TARGETS = (50, 60, 70, 80, 90)
GI = "Projected_Grid_Independence_%"
NPV = "Projected_NPV"


def load() -> pd.DataFrame:
    d = pd.concat((pd.read_pickle(f) for f in sorted(glob.glob(str(HERE / "data/grid/*.pkl")))), ignore_index=True)
    d["design"] = list(zip(d.modules, d.battery_kwh))
    return d


def pick(g: pd.DataFrame, target: int | None, score: pd.Series | None = None):
    """NPV-best design (optionally meeting a GI target); score overrides NPV (robust pick)."""
    s = g.set_index("design")
    npv = s[NPV] if score is None else score
    ok = s.index if target is None else s.index[s[GI] >= target]
    return None if len(ok) == 0 else npv.loc[ok].idxmax()


def hv2(F: np.ndarray, ref=(1.0, 1.0)) -> float:
    F = F[np.all(F <= ref, axis=1)]
    if len(F) == 0:
        return 0.0
    F = F[pareto_mask(F)]
    F = F[np.argsort(F[:, 0])]
    hv, prev = 0.0, ref[1]
    for x, y in F:
        hv += (ref[0] - x) * (prev - y)
        prev = y
    return hv


def main() -> None:
    OUT.mkdir(exist_ok=True)
    d = load()
    rows, front_rows, robust_rows = [], [], []
    for (site, variant), sv in d.groupby(["site", "variant"]):
        by_w = {w: g for w, g in sv.groupby("weather")}
        if "tmy" not in by_w:
            continue
        years = sorted(w for w in by_w if w != "tmy")
        tmy = by_w["tmy"]
        for target in (None, *TARGETS):
            name = "max_npv" if target is None else f"target_{target}"
            d_t = pick(tmy, target)
            # Robust: best mean NPV over years among designs that meet the target in every year? Use mean GI.
            stack = pd.concat([by_w[w].set_index("design")[[NPV, GI]] for w in years], keys=years)
            mean = stack.groupby(level=1).mean()
            ok = mean.index if target is None else mean.index[mean[GI] >= target]
            d_r = mean.loc[ok, NPV].idxmax() if len(ok) else None
            for w in years:
                g = by_w[w].set_index("design")
                d_w = pick(by_w[w], target)
                if d_t is None or d_w is None:
                    continue
                rows.append({
                    "site": site, "variant": variant, "decision": name, "year": int(w),
                    "tmy_pick": d_t, "year_pick": d_w, "same": d_t == d_w,
                    "tmy_pick_gi": g.at[d_t, GI], "misses_target": target is not None and g.at[d_t, GI] < target,
                    # The TMY pick meets the target in year w: NPV left on the table.
                    "regret_eur": np.nan if (target is not None and g.at[d_t, GI] < target) else g.at[d_w, NPV] - g.at[d_t, NPV],
                    # It misses: the shortfall, and the NPV the year's own pick costs to meet it.
                    "shortfall_pp": max(0.0, target - g.at[d_t, GI]) if target is not None else 0.0,
                    "cost_to_meet_eur": (g.at[d_t, NPV] - g.at[d_w, NPV]) if (target is not None and g.at[d_t, GI] < target) else np.nan,
                    "tmy_pick_npv": g.at[d_t, NPV], "tmy_pick_npv_on_tmy": tmy.set_index("design").at[d_t, NPV],
                    "tmy_pick_replacements": g.at[d_t, "Projected_Total_Replacements"],
                })
            if d_t is not None and d_r is not None:
                robust_rows.append({
                    "site": site, "variant": variant, "decision": name, "tmy_pick": d_t, "robust_pick": d_r,
                    "same": d_t == d_r, "mean_npv_loss_eur": mean.at[d_r, NPV] - mean.at[d_t, NPV],
                    "tmy_pick_mean_gi": mean.at[d_t, GI],
                })
        # Fronts.
        allF = np.column_stack([100 - sv[GI].to_numpy(), -sv[NPV].to_numpy()])
        lo, hi = allF.min(0), allF.max(0)
        span = hi - lo
        norm = lambda F: (F - (lo - 0.1 * span)) / (1.2 * span)  # noqa: E731
        tF = np.column_stack([100 - tmy[GI], -tmy[NPV]])
        t_front = set(np.asarray(tmy.design)[pareto_mask(tF)])
        for w in years:
            g = by_w[w]
            F = np.column_stack([100 - g[GI], -g[NPV]])
            mask_t = g.design.isin(t_front).to_numpy()
            front_rows.append({
                "site": site, "variant": variant, "year": int(w),
                "hv_ratio": hv2(norm(F[mask_t])) / hv2(norm(F)),
                "tmy_front_size": len(t_front), "year_front_size": int(pareto_mask(F).sum()),
            })
    r, f, rb = pd.DataFrame(rows), pd.DataFrame(front_rows), pd.DataFrame(robust_rows)
    r.to_csv(OUT / "decisions.csv", index=False)
    f.to_csv(OUT / "fronts.csv", index=False)
    rb.to_csv(OUT / "robust.csv", index=False)

    lines = ["# Study 11 summary\n"]
    s = r.groupby(["variant", "decision"]).agg(
        same_pct=("same", lambda x: 100 * x.mean()), regret_median=("regret_eur", "median"),
        regret_p90=("regret_eur", lambda x: x.quantile(0.9)), regret_max=("regret_eur", "max"),
        miss_pct=("misses_target", lambda x: 100 * x.mean()),
        shortfall_pp_median_when_missed=("shortfall_pp", lambda x: x[x > 0].median()),
        shortfall_pp_max=("shortfall_pp", "max"),
        cost_to_meet_median=("cost_to_meet_eur", "median"), cost_to_meet_max=("cost_to_meet_eur", "max"),
    ).round(1)
    lines += ["## TMY pick scored in actual years (all sites)\n", s.to_markdown(), ""]
    s2 = r.groupby(["variant", "decision", "site"]).agg(
        same_pct=("same", lambda x: 100 * x.mean()), regret_median=("regret_eur", "median"),
        regret_max=("regret_eur", "max"), miss_pct=("misses_target", lambda x: 100 * x.mean()),
        shortfall_pp_max=("shortfall_pp", "max"), cost_to_meet_max=("cost_to_meet_eur", "max"),
    ).round(1)
    lines += ["## Per site\n", s2.to_markdown(), ""]
    lines += ["## Robust pick (best mean over the 19 years) vs TMY pick\n",
              rb.groupby(["variant", "decision"]).agg(same_pct=("same", lambda x: 100 * x.mean()),
                                                     loss_median=("mean_npv_loss_eur", "median"),
                                                     loss_max=("mean_npv_loss_eur", "max")).round(1).to_markdown(), ""]
    lines += ["## TMY front scored in actual years: hypervolume ratio\n",
              f.groupby(["variant", "site"]).hv_ratio.describe()[["min", "25%", "50%", "max"]].round(4).to_markdown(), ""]
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines[:6]))


if __name__ == "__main__":
    main()
