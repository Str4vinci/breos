#!/usr/bin/env python3
"""Task 1(b): plot the three full degradation-model Pareto fronts on one axis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

GI = "Projected_Grid_Independence_%"
NPV = "Projected_NPV_Eur"

# dataviz categorical slots 1-3, validated all-pairs light mode
SERIES = (
    ("base-v1", "Field v1", "#2a78d6", "o"),
    ("field-v2", "Field v2", "#eb6834", "s"),
    ("laboratory", "Laboratory", "#1baf7a", "^"),
)
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#b8b7b2"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--front-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.edgecolor": MUTED,
            "axes.linewidth": 0.7,
            "figure.dpi": 150,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(6.3, 4.2), constrained_layout=True)

    ax.axhline(0.0, color=MUTED, lw=0.8, ls="--", zorder=1)
    ax.annotate(
        "NPV = 0",
        xy=(0.995, 0.0),
        xycoords=("axes fraction", "data"),
        ha="right",
        va="bottom",
        fontsize=7,
        color=INK_2,
    )

    summary = []
    for slug, label, color, marker in SERIES:
        front = pd.read_csv(args.front_root / slug / "pareto_results.csv").sort_values(GI)
        reps = pd.read_csv(args.front_root / slug / "pareto_representatives.csv")

        ax.plot(front[GI], front[NPV], color=color, lw=1.6, alpha=0.85, zorder=3, label=label)
        ax.scatter(
            front[GI],
            front[NPV],
            s=13,
            facecolor=color,
            edgecolor="#fcfcfb",
            linewidth=0.6,
            marker=marker,
            zorder=4,
        )

        mn = reps[reps["Representative"] == "max_npv"].iloc[0]
        ax.scatter(
            [mn[GI]],
            [mn[NPV]],
            s=95,
            facecolor="none",
            edgecolor=color,
            linewidth=1.8,
            marker=marker,
            zorder=6,
        )
        summary.append((label, color, front, mn))

    # direct labels on each front (relief for the contrast WARN on slot 3)
    for label, color, front, mn in summary:
        tip = front.iloc[-1]
        ax.annotate(
            label,
            xy=(tip[GI], tip[NPV]),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8,
            color=INK_2,
        )
        ax.annotate(
            f"max NPV\n{int(mn['Modules'])} mod, {mn['Battery_kWh']:.0f} kWh",
            xy=(mn[GI], mn[NPV]),
            xytext=(0, 13),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color=INK_2,
        )

    ax.set_xlabel("Projected grid independence (%)")
    ax.set_ylabel("Projected NPV (EUR)")
    ax.grid(True, color=MUTED, lw=0.5, alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK_2, length=3, width=0.7)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK)
    # explicit padding: the leftmost representative label needs room to its left
    lo = min(f[GI].min() for _l, _c, f, _m in summary)
    hi = max(f[GI].max() for _l, _c, f, _m in summary)
    ax.set_xlim(lo - 4.0, hi + 4.5)
    leg = ax.legend(loc="lower left", frameon=False, handlelength=1.8, labelcolor=INK)
    leg.set_zorder(7)

    for ext in ("pdf", "svg", "png"):
        path = args.output / f"task1b_fronts_by_degradation_model.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
