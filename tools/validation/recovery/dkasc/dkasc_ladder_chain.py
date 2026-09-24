"""DKASC loss ladder rerun under the chain options the case study uses.

dkasc_analysis.py runs its loss ladder with Hay-Davies and no diffuse IAM. The
case-study chain uses Perez (allsitescomposite1990) with the Marion diffuse
IAM. This script reruns the same ladder three ways so the shift can be split:

* ``haydavies``: the published configuration, as a reproduction check.
* ``perez``: Perez transposition only.
* ``perez+marion``: Perez plus the Marion diffuse IAM, the case-study chain.

Day screening is unchanged: it is built once from Hay-Davies with availability
and shading removed, exactly as in dkasc_analysis.py, so every configuration is
scored on the same days.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from dkasc_analysis import FIXED, REFERENCE_MODEL
from dkasc_validate import (
    LOSS_CASES,
    day_mask,
    load,
    metrics,
    run_chain,
    screen_array_days,
    screen_weather_days,
    weather_frame,
)

CONFIGS = {
    "haydavies": {"transposition_model": "haydavies", "diffuse_iam": "none"},
    "perez": {"transposition_model": "perez", "model_perez": "allsitescomposite1990", "diffuse_iam": "none"},
    "perez+marion": {"transposition_model": "perez", "model_perez": "allsitescomposite1990", "diffuse_iam": "marion"},
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--ladder-years", default="2016:2016", help="YYYY:YYYY slice used for the loss ladder")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)

    full = load(args.data)
    wfull = weather_frame(full, "closure")
    lo, hi = args.ladder_years.split(":")
    d = full.loc[lo:hi]
    w = wfull.loc[d.index]
    wok = screen_weather_days(d)
    ok = {}
    for label in FIXED:
        r = run_chain(d, label, REFERENCE_MODEL, LOSS_CASES["no-availability-no-shading"], weather=w)
        ok[label] = screen_array_days(d, label, r["ac_model_kW"], wok)
    print(f"=== screening over {d.index[0].date()} -> {d.index[-1].date()} ({REFERENCE_MODEL}) ===")
    for label in FIXED:
        print(f"  {label:4s} normal on {int(ok[label].sum())} days")

    rows = []
    for config, kwargs in CONFIGS.items():
        for label in FIXED:
            m = day_mask(d.index, ok[label])
            for case, overrides in LOSS_CASES.items():
                r = run_chain(d, label, loss_overrides=overrides, weather=w, **kwargs)
                mm = metrics(r["ac_model_kW"][m], d[f"P_{label}"][m])
                rows.append(
                    {
                        "config": config,
                        "array": label,
                        "orientation": "west" if label == "16D" else "north",
                        "losses": case,
                        **mm,
                    }
                )
    ladder = pd.DataFrame(rows)
    ladder.to_csv(args.outdir / "loss_ladder_chain.csv", index=False)

    wide = ladder.pivot_table(index=["array", "losses"], columns="config", values="bias_%", sort=False)[list(CONFIGS)]
    wide["perez_shift_pp"] = wide["perez"] - wide["haydavies"]
    wide["marion_shift_pp"] = wide["perez+marion"] - wide["perez"]
    wide["total_shift_pp"] = wide["perez+marion"] - wide["haydavies"]
    wide.to_csv(args.outdir / "loss_ladder_chain_bias_wide.csv")
    print("\n=== AC energy bias % by configuration ===")
    print(wide.round(3).to_string())


if __name__ == "__main__":
    main()
