"""Study 15 analysis: peak memory and wall time by config family, resolution and horizon.

Usage: uv run --with tabulate python analyze.py -> output/summary.md
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    d = pd.DataFrame(json.loads(ln) for ln in (HERE / "data/results.jsonl").read_text().splitlines())
    parts = d.id.str.split(":", expand=True)
    d["family"], d["name"] = parts[0], parts[1]
    d["res"] = d.id.str.extract(r":(h|15min)(?::|$)")
    d["years"] = d.id.str.extract(r":(\d+)y$").astype(float)
    d["total_s"] = d[[c for c in ("import_s", "build_s", "simulate_s", "result_s") if c in d]].sum(axis=1)
    d.to_csv(OUT / "results.csv", index=False)
    fails = d[~d.ok.astype(bool)][["id", "error", "peak_rss_mb"]]
    top = d.sort_values("peak_rss_mb", ascending=False).head(15)[["id", "peak_rss_mb", "total_s", "timeseries_mb"]]
    ex = d[d.family == "ex"].pivot_table(index="name", columns=["res", "years"], values="peak_rss_mb").round(0)
    ext = d[d.family == "ex"].pivot_table(index="name", columns=["res", "years"], values="simulate_s").round(1)
    lines = ["# Study 15 summary\n", "## Failures\n", fails.to_markdown(index=False), "",
             "## Largest peak memory (MB)\n", top.round(1).to_markdown(index=False), "",
             "## Examples: peak RSS (MB)\n", ex.to_markdown(), "", "## Examples: simulate time (s)\n", ext.to_markdown(), "",
             "## Scaling and Monte Carlo\n",
             d[d.family.isin(["scale", "mc"])][["id", "peak_rss_mb", "simulate_s", "timeseries_mb"]].round(1).to_markdown(index=False), ""]
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines[:6]))


if __name__ == "__main__":
    main()
