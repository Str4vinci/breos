"""Study 14 analysis: HV ratio and IGD+ of NSGA-II fronts with two vs three objectives.

Usage: uv run --with tabulate python analyze_3obj.py -> output/summary.md, output/replays.csv
"""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"


def q(p):
    return lambda x: x.quantile(p)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    d = pd.concat((pd.read_pickle(f) for f in sorted(glob.glob(str(HERE / "data/runs/*/*/replay_3obj.pkl")))), ignore_index=True)
    d.to_csv(OUT / "replays.csv", index=False)
    agg = d.groupby(["objectives", "budget"]).agg(
        runs=("seed", "size"), n_eval=("n_eval", "median"), exact_front=("exact_front_size", "median"),
        hv_final_p5=("hv_ratio_final", q(0.05)), hv_final_median=("hv_ratio_final", "median"),
        hv_all_p5=("hv_ratio_all_evaluated", q(0.05)), hv_all_median=("hv_ratio_all_evaluated", "median"),
        hv_all_min=("hv_ratio_all_evaluated", "min"),
        front_final=("front_size_final", "median"), front_all=("front_size_all_evaluated", "median"),
    ).round(4)
    per_case = d.groupby(["space", "scenario", "objectives", "budget"]).agg(
        hv_final_median=("hv_ratio_final", "median"), hv_all_min=("hv_ratio_all_evaluated", "min"),
        hv_all_median=("hv_ratio_all_evaluated", "median"), exact_front=("exact_front_size", "first"),
    ).round(4)
    lines = ["# Study 14 summary\n", "HV ratio = fraction of the exact front's hypervolume (normalised, 10 % margin).\n",
             agg.to_markdown(), "", "## Per case\n", per_case.to_markdown(), ""]
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines[:4]))


if __name__ == "__main__":
    main()
