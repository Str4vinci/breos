"""Full BREOS-vs-NIST analysis: loss ladder, thermal models, regime breakdown.

See nist_validate.py for the measurement-reference rationale. This script adds
the parts that make the comparison interpretable rather than just a number:

* Outage screening. 2016 contains multi-day array outages and snow-covered
  days on which any irradiance-driven model necessarily overpredicts. They are
  identified from the measured/model daily ratio and reported separately
  instead of being absorbed into a loss coefficient.
* A loss ladder, so the effective system loss the array actually exhibits can
  be read off directly and compared with the BREOS default stack.
* Cell-temperature validation against the seven module backsheet RTDs, which
  is normally impossible in a whole-system yield comparison.
* A clear / mixed / overcast split, because that is where the Esposende
  residual concentrates.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
from nist_validate import (
    LOSS_CASES,
    MODULE,
    N_MODULES,
    SHUNTS,
    load,
    metrics,
    run_chain,
)

LAT, LON, ALT, TILT, AZIM = 39.1319, -77.2141, 138.0, 20.0, 180.0
EST = "Etc/GMT+5"  # NIST publishes Local Standard Time; no DST transition exists
GROUND_RTDS = [f"RTD_C_{i}" for i in range(1, 8)]
OUTAGE_RATIO = 0.90  # measured/model daily ratio below this = not normal operation


def clear_sky_poa(index: pd.DatetimeIndex) -> pd.Series:
    loc = pvlib.location.Location(LAT, LON, tz=EST, altitude=ALT)
    # Mid-interval solar position: the NIST channels are 1-minute averages
    # stamped by the Campbell logger at interval end.
    mid = index - pd.Timedelta(seconds=30)
    sp = loc.get_solarposition(mid)
    cs = loc.get_clearsky(mid, model="ineichen")
    poa = pvlib.irradiance.get_total_irradiance(
        TILT,
        AZIM,
        sp["apparent_zenith"],
        sp["azimuth"],
        cs["dni"],
        cs["ghi"],
        cs["dhi"],
        model="haydavies",
        dni_extra=pvlib.irradiance.get_extra_radiation(mid),
    )["poa_global"]
    return pd.Series(np.nan_to_num(poa.to_numpy()), index=index)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    d = load(args.data)
    d["ClearSkyPOA"] = clear_sky_poa(d.index)

    # --- outage screening, from the loss-free chain so the rule cannot be
    # --- tuned by the loss assumption under test
    base = run_chain(d, "RefCell1_Wm2", "faiman", LOSS_CASES["no-availability-no-shading"], 0.0)
    dm = base["ac_model_kW"].resample("D").sum() / 60.0
    dd = d["MeterAC_kW"].resample("D").sum() / 60.0
    ratio = (dd / dm).replace([np.inf, -np.inf], np.nan)
    normal_days = set(ratio[ratio >= OUTAGE_RATIO].index.date)
    normal = d.index.map(lambda t: t.date() in normal_days).to_numpy(dtype=bool)

    outage = ratio[ratio < OUTAGE_RATIO]
    lost = (dm[outage.index] - dd[outage.index]).sum()
    print(
        f"outage/snow screening: {len(outage)} of {len(ratio)} days excluded, "
        f"{lost:,.0f} kWh ({lost / dm.sum() * 100:.2f} % of modelled annual)"
    )
    outage.round(3).to_csv(args.outdir / "excluded_days.csv", header=["measured_over_model"])

    # --- loss ladder, thermal models, POA sensors -------------------------
    rows = []
    for temp_model in ("faiman", "pvsyst-freestanding", "sapm-open-rack-glass-glass"):
        for poa_col in ("RefCell1_Wm2", "SEWSPOAIrrad_Wm2"):
            for case, overrides in LOSS_CASES.items():
                r = run_chain(d, poa_col, temp_model, overrides, 0.0)
                for label, mask in (("all days", slice(None)), ("normal days", normal)):
                    m = metrics(r["ac_model_kW"][mask], d["MeterAC_kW"][mask])
                    rows.append({"temp_model": temp_model, "poa": poa_col, "losses": case, "subset": label, **m})
    res = pd.DataFrame(rows)
    res.to_csv(args.outdir / "sweep.csv", index=False)

    pd.set_option("display.width", 250)
    print("\n=== loss ladder (faiman, RefCell1 POA) ===")
    sub = res[(res.temp_model == "faiman") & (res.poa == "RefCell1_Wm2")]
    print(
        sub[["losses", "subset", "model_kWh", "meas_kWh", "bias_%", "daily_nRMSE_%", "daily_r"]]
        .round(3)
        .to_string(index=False)
    )

    print("\n=== thermal model, normal days, no-availability-no-shading ===")
    sub = res[(res.subset == "normal days") & (res.losses == "no-availability-no-shading")]
    print(sub[["temp_model", "poa", "bias_%", "daily_nRMSE_%", "daily_r"]].round(3).to_string(index=False))

    # --- cell temperature vs measured backsheet ---------------------------
    rtd = d[GROUND_RTDS].mean(axis=1)
    hot = (d["RefCell1_Wm2"] > 400) & rtd.notna()
    print("\n=== cell temperature vs module backsheet RTD mean (POA > 400 W/m2) ===")
    for temp_model in ("faiman", "pvsyst-freestanding", "sapm-open-rack-glass-glass"):
        r = run_chain(d, "RefCell1_Wm2", temp_model, LOSS_CASES["module-only"], 0.0)
        err = r["temp_cell"][hot] - rtd[hot]
        print(
            f"  {temp_model:28s} bias={err.mean():+6.2f} C  RMSE={np.sqrt((err**2).mean()):5.2f} C  r={r['temp_cell'][hot].corr(rtd[hot]):.4f}"
        )

    # --- regime breakdown --------------------------------------------------
    dpoa = d["RefCell1_Wm2"].resample("D").sum()
    dcs = d["ClearSkyPOA"].resample("D").sum()
    kt = (dpoa / dcs).reindex(dm.index)
    r = run_chain(d, "RefCell1_Wm2", "faiman", LOSS_CASES["no-availability-no-shading"], 0.0)
    am = r["ac_model_kW"].resample("D").sum() / 60.0
    regimes = {
        "clear (kt>0.85)": kt > 0.85,
        "mixed (0.5-0.85)": (kt > 0.5) & (kt <= 0.85),
        "overcast (kt<=0.5)": kt <= 0.5,
    }
    print("\n=== regime breakdown (normal days only, no-availability-no-shading) ===")
    keep = pd.Series([dd.index[i].date() in normal_days for i in range(len(dd))], index=dd.index)
    out = []
    for name, sel in regimes.items():
        s = sel & keep
        out.append(
            {
                "regime": name,
                "days": int(s.sum()),
                "model_kWh": am[s].sum(),
                "meas_kWh": dd[s].sum(),
                "bias_%": (am[s].sum() / dd[s].sum() - 1) * 100,
            }
        )
    reg = pd.DataFrame(out)
    print(reg.round(2).to_string(index=False))
    reg.to_csv(args.outdir / "regimes.csv", index=False)

    # --- monthly -----------------------------------------------------------
    mo = pd.DataFrame({"model": am, "meas": dd})[keep].resample("MS").sum()
    mo["bias_%"] = (mo.model / mo.meas - 1) * 100
    print("\n=== monthly (normal days, no-availability-no-shading) ===")
    print(mo.round(2).to_string())
    mo.to_csv(args.outdir / "monthly.csv")
    print(f"\nwrote outputs to {args.outdir}")


if __name__ == "__main__":
    main()
