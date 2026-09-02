"""Loss ladder, residual structure and the tracking path, for DKASC Alice Springs.

dkasc_transposition.py answers the question this dataset was chosen for. This
script covers the rest of the comparison, in the shape of the NIST run so the
two sites can be read against each other:

* A loss ladder per array, so the effective system loss each array actually
  exhibits can be read off and compared with NIST's measured 5.35 % and BREOS's
  14.1 % default.
* Degradation. Several of these arrays date from 2008, and the annual bias
  trend measures how fast they have aged -- a lever NIST's single year could
  not touch. It also separates ordinary ageing from the 16A fault.
* A clear / mixed / overcast split, because that is where the Esposende
  residual concentrates. Alice Springs is a desert site, so its clear-sky
  population is far larger than either NIST's or Esposende's.
* The measurement floor. DKASC has no redundant co-located irradiance sensor,
  so unlike NIST the floor cannot be measured by differencing two pyranometers;
  it is bounded here from the meter cross-check and from the year-to-year
  scatter of the pair ratio instead.
* The dual-axis tracker at site 1A, as a separate check of the tracking path.

Everything is reported on screened days. Screening is built once from a fixed
reference configuration so no result under test can move the day set.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from breos.inverter import _calculate_dc_ac_power_arrays
from breos.solar import calculate_pv_production_tracking_breakdown

from dkasc_arrays import ARRAYS
from dkasc_validate import (
    LOSS_CASES, STEP_H, clear_sky_ghi, day_mask, load, location, metrics,
    run_chain, screen_array_days, screen_weather_days, weather_frame,
)

FIXED = ("16A", "16D", "12", "13")
REFERENCE_MODEL = "haydavies"   # fixed for screening; never the variable under test


def effective_loss_pct(module_only_bias_pct: float) -> float:
    """The single loss stack that would zero out a module-only bias.

    If the loss-free chain runs B per cent high, the stack that reconciles it
    is B / (1 + B): this is the array's own effective system loss, the quantity
    BREOS's 14.1 % default is trying to represent.
    """
    b = module_only_bias_pct / 100.0
    return b / (1.0 + b) * 100.0


def loss_ladder(d: pd.DataFrame, w: pd.DataFrame, ok: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for label in FIXED:
        m = day_mask(d.index, ok[label])
        for case, overrides in LOSS_CASES.items():
            r = run_chain(d, label, REFERENCE_MODEL, overrides, weather=w)
            mm = metrics(r["ac_model_kW"][m], d[f"P_{label}"][m])
            rows.append({"array": label, "orientation": "west" if label == "16D" else "north",
                         "losses": case, **mm})
    return pd.DataFrame(rows)


def degradation(d: pd.DataFrame, w: pd.DataFrame) -> pd.DataFrame:
    """Annual loss-free bias per array, and the trend implied by it.

    A model run with no degradation term drifts high at exactly the rate the
    array ages, so the slope of the annual bias is a measurement of the
    degradation rate rather than an assumption about it.
    """
    rows = []
    for year, sub in d.groupby(d.index.year):
        ws = w.loc[sub.index]
        wok = screen_weather_days(sub)
        for label in FIXED:
            if sub[f"P_{label}"].notna().mean() < 0.5:
                continue
            r = run_chain(sub, label, REFERENCE_MODEL,
                          LOSS_CASES["module-only"], weather=ws)
            ok = screen_array_days(sub, label, r["ac_model_kW"], wok)
            m = day_mask(sub.index, ok)
            mm = metrics(r["ac_model_kW"][m], sub[f"P_{label}"][m])
            rows.append({"year": year, "array": label, "days": int(ok.sum()),
                         "bias_module_only_%": mm["bias_%"]})
    t = pd.DataFrame(rows)
    fits = []
    for label, g in t.groupby("array"):
        g = g.sort_values("year")
        slope = np.polyfit(g.year, g["bias_module_only_%"], 1)[0]
        # d(bias)/dt in points per year maps to a degradation rate of
        # slope / (1 + bias) -- to first order, slope per cent of output a year.
        fits.append({"array": label, "years": f"{g.year.min()}-{g.year.max()}",
                     "bias_first_%": g["bias_module_only_%"].iloc[0],
                     "bias_last_%": g["bias_module_only_%"].iloc[-1],
                     "implied_degradation_%/yr": slope / (1 + g["bias_module_only_%"].mean() / 100)})
    return t, pd.DataFrame(fits)


def regimes(d: pd.DataFrame, w: pd.DataFrame, ok: dict[str, pd.Series]) -> pd.DataFrame:
    cs = clear_sky_ghi(d.index)
    kt = (d["ghi"].resample("D").sum() / cs.resample("D").sum())
    bands = {"clear (kt>0.85)": kt > 0.85,
             "mixed (0.5-0.85)": (kt > 0.5) & (kt <= 0.85),
             "overcast (kt<=0.5)": kt <= 0.5}
    rows = []
    for label in FIXED:
        r = run_chain(d, label, REFERENCE_MODEL,
                      LOSS_CASES["no-availability-no-shading"], weather=w)
        dm = r["ac_model_kW"].resample("D").sum() * STEP_H
        dd = d[f"P_{label}"].resample("D").sum() * STEP_H
        for name, sel in bands.items():
            s = sel & ok[label]
            if not s.any():
                continue
            rows.append({"array": label, "regime": name, "days": int(s.sum()),
                         "model_kWh": dm[s].sum(), "meas_kWh": dd[s].sum(),
                         "bias_%": (dm[s].sum() / dd[s].sum() - 1) * 100})
    return pd.DataFrame(rows)


def monthly(d: pd.DataFrame, w: pd.DataFrame, ok: dict[str, pd.Series]) -> pd.DataFrame:
    out = {}
    for label in FIXED:
        r = run_chain(d, label, REFERENCE_MODEL,
                      LOSS_CASES["no-availability-no-shading"], weather=w)
        m = day_mask(d.index, ok[label])
        dm = r["ac_model_kW"][m].resample("MS").sum() * STEP_H
        dd = d[f"P_{label}"][m].resample("MS").sum() * STEP_H
        out[label] = (dm / dd - 1) * 100
    return pd.DataFrame(out)


def tracker(d: pd.DataFrame, w: pd.DataFrame) -> pd.DataFrame:
    """Site 1A: two DEGERenergie 5000NT dual-axis trackers, 60 Trina TSM-175DC01.

    A separate check of the tracking path, which the fixed arrays never
    exercise. Because a dual-axis tracker holds the plane normal to the sun,
    its POA is dominated by the beam component and the sky-diffuse model
    matters far less than it does for the fixed west-facing array -- so this is
    a test of the tracking geometry, not a second transposition test.

    ``dual_axis_max_tilt`` is swept because the real tracker cannot lie flat:
    a mechanical elevation limit shows up as a summer-midday shortfall, and 90
    degrees is BREOS's unconstrained default rather than a DEGER specification.
    """
    arr = ARRAYS["1A"]
    if d["P_1A"].notna().mean() < 0.5:
        return pd.DataFrame()

    wok = screen_weather_days(d)
    rows = []
    for model in ("isotropic", "haydavies", "perez"):
        for max_tilt in (90.0, 75.0, 60.0):
            bd = calculate_pv_production_tracking_breakdown(
                w, location(), n_modules=arr.n_modules, tracking="dual_axis",
                dual_axis_max_tilt=max_tilt, pv_params=arr.module, freq="5min",
                loss_overrides=LOSS_CASES["no-availability-no-shading"],
                transposition_model=model,
            )
            ac_w, _, _ = _calculate_dc_ac_power_arrays(
                bd.dc_after_losses.to_numpy(), arr.inverter_ac_kw * 1000.0, arr.inverter_eta
            )
            ac = pd.Series(ac_w / 1000.0, index=bd.dc_after_losses.index)
            ok = screen_array_days(d, "1A", ac, wok)
            m = day_mask(d.index, ok)
            rows.append({"transposition": model, "dual_axis_max_tilt": max_tilt,
                         "days": int(ok.sum()),
                         **metrics(ac[m], d["P_1A"][m])})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--ladder-years", default=None,
                    help="YYYY:YYYY slice used for the loss ladder, regimes and tracker")
    ap.add_argument("--tracker-years", default=None,
                    help="YYYY:YYYY slice for the tracker; site 1A's record only "
                         "begins 2013-08, so it usually needs a later window than "
                         "the fixed arrays and is skipped if under half-covered")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)

    full = load(args.data)
    wfull = weather_frame(full, "closure")

    print("=== degradation: annual loss-free bias per array ===")
    per_year, fits = degradation(full, wfull)
    per_year.to_csv(args.outdir / "degradation_by_year.csv", index=False)
    print(per_year.pivot(index="year", columns="array",
                         values="bias_module_only_%").round(2).to_string())
    print()
    print(fits.round(3).to_string(index=False))
    fits.to_csv(args.outdir / "degradation_fit.csv", index=False)

    if args.ladder_years:
        lo, hi = args.ladder_years.split(":")
        d = full.loc[lo:hi]
    else:
        d = full
    w = wfull.loc[d.index]
    wok = screen_weather_days(d)
    ok = {}
    for label in FIXED:
        r = run_chain(d, label, REFERENCE_MODEL,
                      LOSS_CASES["no-availability-no-shading"], weather=w)
        ok[label] = screen_array_days(d, label, r["ac_model_kW"], wok)
    print(f"\n=== screening over {d.index[0].date()} -> {d.index[-1].date()} ===")
    print(f"  weather record usable on {int(wok.sum())} of {len(wok)} days")
    for label in FIXED:
        print(f"  {label:4s} normal on {int(ok[label].sum())} days "
              f"({int((wok & ~ok[label]).sum())} array-abnormal)")

    print("\n=== loss ladder (Hay-Davies, DNI from the measured GHI/DHI closure) ===")
    ladder = loss_ladder(d, w, ok)
    ladder.to_csv(args.outdir / "loss_ladder.csv", index=False)
    print(ladder[["array", "orientation", "losses", "model_kWh", "meas_kWh",
                  "bias_%", "daily_nRMSE_%", "daily_r"]].round(3).to_string(index=False))

    print("\n=== effective system loss implied by each array ===")
    mo = ladder[ladder.losses == "module-only"]
    eff = pd.DataFrame({
        "array": mo.array.to_numpy(),
        "module_only_bias_%": mo["bias_%"].to_numpy(),
        "effective_system_loss_%": [effective_loss_pct(b) for b in mo["bias_%"]],
    })
    print(eff.round(2).to_string(index=False))
    eff.to_csv(args.outdir / "effective_loss.csv", index=False)
    print("  BREOS default DEFAULT_PVWATTS_LOSSES stacks to 14.1 %; "
          "NIST Gaithersburg measured 5.35 %.")

    print("\n=== regime breakdown (no-availability-no-shading) ===")
    reg = regimes(d, w, ok)
    reg.to_csv(args.outdir / "regimes.csv", index=False)
    print(reg.round(2).to_string(index=False))

    print("\n=== dual-axis tracker, site 1A (no-availability-no-shading) ===")
    if args.tracker_years:
        lo, hi = args.tracker_years.split(":")
        td = full.loc[lo:hi]
        tw = wfull.loc[td.index]
    else:
        td, tw = d, w
    trk = tracker(td, tw)
    if trk.empty:
        print("  source 91 not present or not covered in this window")
    else:
        trk.to_csv(args.outdir / "tracker.csv", index=False)
        print(trk[["transposition", "dual_axis_max_tilt", "days", "model_kWh",
                   "meas_kWh", "bias_%", "daily_nRMSE_%", "daily_r"]]
              .round(3).to_string(index=False))

    print("\n=== monthly bias, % (no-availability-no-shading) ===")
    mo_t = monthly(d, w, ok)
    mo_t.to_csv(args.outdir / "monthly.csv")
    print(mo_t.round(2).to_string())
    print(f"\nwrote outputs to {args.outdir}")


if __name__ == "__main__":
    main()
