"""Transposition validation -- the test the NIST run could not perform.

NIST left decomposition and transposition completely untested, because its only
irradiance in engineering units was already plane-of-array. Perez-versus-
isotropic is worth roughly 5 % of annual energy at Esposende, so it is the
largest open modelling lever in BREOS. DKASC closes it two independent ways.

**Leg A -- irradiance only.** The site's tilted pyranometer sits at 20 deg /
azimuth 0, co-planar with every fixed array. Modelled POA is compared straight
against it. No PV model, no module parameters, no loss stack: whatever error
appears is transposition error plus the instrument's own uncertainty. This leg
tests the sun-facing orientation only.

**Leg B -- the controlled orientation pair.** Arrays 16A and 16D are physically
identical 1.98 kW BP 3165J arrays with the same inverter model, commissioned
the same day, differing only in azimuth: 0 (north, sun-facing in the southern
hemisphere) and 270 (west). One irradiance record has to reproduce both.

Leg B's headline number is the **north/west ratio error**, not the two absolute
biases. Module parameters, inverter efficiency, the loss stack, array age and
site soiling are common to the pair and cancel in the ratio; what survives is
the model's handling of a 90-degree-off-axis plane. A west-facing array sees a
far larger share of its annual energy as sky diffuse, so it is the orientation
that discriminates between the sky models. **A model that fits only the
sun-facing array is not validated.**

DNI is taken from the geometric closure DNI = (GHI - DHI) / cos(zenith), which
given two measured components is an identity rather than a model, so Leg A and
Leg B isolate transposition. The decomposition sweep at the end re-runs the
same comparison with DNI estimated from GHI alone (DISC, DIRINT, Erbs) to
report separately what a GHI-only weather source costs.

No model is selected by fitting. The array geometry is DKASC's published
20 deg / azimuth 0 and 20 deg / azimuth 270 throughout; dkasc_facts.py reports
the independent data-driven fit of the pyranometer's orientation (21 deg / 359)
as a check on that geometry, and it is never used in place of it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from dkasc_arrays import ARRAYS, GTI_SENSOR_AZIMUTH, GTI_SENSOR_TILT
from dkasc_validate import (
    LOSS_CASES,
    STEP_H,
    TRANSPOSITION_MODELS,
    day_mask,
    irradiance_metrics,
    load,
    modelled_poa,
    run_chain,
    screen_array_days,
    screen_weather_days,
    solar_position,
    weather_frame,
)

# The window over which 16A and 16D are genuinely the identical pair the test
# assumes. Chosen from measurement-side evidence only -- see pair_drift() -- and
# never from which transposition model it favours: from 2015 array 16A departs
# from 16D *and* from the two independent north-facing arrays on the site, which
# is an array fault, not a modelling error. Running the pair test through that
# fault would report a 5-9 % transposition failure that does not exist.
PAIR_WINDOW = ("2009", "2014")

PEREZ_SETS = (
    "allsitescomposite1990",
    "allsitescomposite1988",
    "sandiacomposite1988",
    "usacomposite1988",
    "france1988",
    "phoenix1988",
    "elmonte1988",
    "osage1988",
    "albuquerque1988",
    "capecanaveral1988",
    "albany1988",
)
DNI_METHODS = ("closure", "disc", "dirint", "erbs")
# Desert ground. pvlib's default is 0.25; 0.30 brackets red sand. The
# ground-reflected term is albedo * GHI * (1 - cos(tilt)) / 2, which is
# identical for 16A and 16D because both are at 20 deg, so albedo shifts the
# pair together and cannot manufacture a north/west difference.
ALBEDOS = (None, 0.20, 0.30)


def daytime(d: pd.DataFrame, solpos: pd.DataFrame, zenith_max: float = 85.0) -> pd.Series:
    """Samples with the sun up and enough signal for a ratio to mean anything."""
    return (solpos["apparent_zenith"] < zenith_max) & (d["ghi"] > 20.0)


def leg_a(d: pd.DataFrame, w: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Modelled POA against the co-planar tilted pyranometer."""
    rows = []
    for model in TRANSPOSITION_MODELS:
        poa = modelled_poa(w, GTI_SENSOR_TILT, GTI_SENSOR_AZIMUTH, model)
        rows.append(
            {
                "transposition": model,
                "perez_set": "-",
                "albedo": "default",
                **irradiance_metrics(poa, d["gti_meas"], mask),
            }
        )
    for pset in PEREZ_SETS:
        poa = modelled_poa(w, GTI_SENSOR_TILT, GTI_SENSOR_AZIMUTH, "perez", model_perez=pset)
        rows.append(
            {
                "transposition": "perez",
                "perez_set": pset,
                "albedo": "default",
                **irradiance_metrics(poa, d["gti_meas"], mask),
            }
        )
    for alb in ALBEDOS:
        if alb is None:
            continue
        for model in ("isotropic", "haydavies", "perez"):
            poa = modelled_poa(w, GTI_SENSOR_TILT, GTI_SENSOR_AZIMUTH, model, albedo=alb)
            rows.append(
                {
                    "transposition": model,
                    "perez_set": "-",
                    "albedo": alb,
                    **irradiance_metrics(poa, d["gti_meas"], mask),
                }
            )
    return pd.DataFrame(rows)


def normal_days(d: pd.DataFrame, w: pd.DataFrame, labels) -> pd.Series:
    """Days on which both the weather record and every named array are healthy.

    The screen is built once from a fixed reference configuration (Hay-Davies,
    loss-free) and reused for every model under test, so the choice of
    transposition model can never move the set of days it is scored on.
    """
    ok = screen_weather_days(d)
    for label in labels:
        r = run_chain(d, label, "haydavies", LOSS_CASES["no-availability-no-shading"], weather=w)
        ok &= screen_array_days(d, label, r["ac_model_kW"], ok)
    return ok


def pair_drift_measured(d: pd.DataFrame) -> pd.DataFrame:
    """Quarterly measured energy ratios -- no model anywhere in this table.

    This is the justification for PAIR_WINDOW, and it is deliberately
    model-free: choosing the window from modelled biases would risk selecting
    the years that happen to suit a transposition model. Arrays 12 and 13 are
    independent north-facing arrays of different manufacture, scaled by DC
    rating, so 12/16A and 13/16A measure 16A against the rest of the site.

    16A holds steady against both from 2009 to 2014 and then steps down about
    10 % from 2015Q4 onward. Whatever happened to 16A, it happened to 16A
    alone, and the pair stops being a controlled pair at that point.
    """
    e = pd.DataFrame({c: d[f"P_{c}"].resample("QS").sum() * STEP_H for c in ("16A", "16D", "12", "13")})
    e = e[e.min(axis=1) > 50.0]
    a = ARRAYS
    out = pd.DataFrame(
        {
            "16D/16A": e["16D"] / e["16A"],
            "12/16A_per_kW": e["12"] / e["16A"] / (a["12"].dc_kw / a["16A"].dc_kw),
            "13/16A_per_kW": e["13"] / e["16A"] / (a["13"].dc_kw / a["16A"].dc_kw),
            "13/12": e["13"] / e["12"],
        }
    )
    out.index = out.index.to_period("Q")
    return out


def pair_drift(d: pd.DataFrame, w: pd.DataFrame) -> pd.DataFrame:
    """Year by year: is 16A still the same array as 16D?

    This is the evidence for PAIR_WINDOW and it is reported in full rather than
    summarised, because the window is a selection and the reader is entitled to
    see what was selected away. Arrays 12 and 13 are independent north-facing
    arrays of different manufacture; they are here as a control. While 16A's
    bias tracks theirs, the pair is sound. When it walks away from all three at
    once, 16A is the thing that changed.
    """
    rows = []
    for year, sub in d.groupby(d.index.year):
        if sub["P_16A"].notna().mean() < 0.5:
            continue
        ws = w.loc[sub.index]
        ok = normal_days(sub, ws, ("16A", "16D", "12", "13"))
        m = day_mask(sub.index, ok)
        row = {"year": year, "days": int(ok.sum())}
        meas, model = {}, {}
        for label in ("16A", "16D", "12", "13"):
            r = run_chain(sub, label, "haydavies", LOSS_CASES["no-availability-no-shading"], weather=ws)
            meas[label] = sub[f"P_{label}"][m].sum() * STEP_H
            model[label] = r["ac_model_kW"][m].sum() * STEP_H
            row[f"bias_{label}_%"] = (model[label] / meas[label] - 1.0) * 100.0
        wn_meas = meas["16D"] / meas["16A"]
        wn_model = model["16D"] / model["16A"]
        row["W/N_meas"] = wn_meas
        row["W/N_model"] = wn_model
        row["ratio_err_%"] = (wn_model / wn_meas - 1.0) * 100.0
        rows.append(row)
    return pd.DataFrame(rows)


def leg_b(
    d: pd.DataFrame,
    w: pd.DataFrame,
    loss_case: str,
    ok_days: pd.Series,
    models=TRANSPOSITION_MODELS,
    perez_sets=PEREZ_SETS,
) -> pd.DataFrame:
    """Full chain for the identical north and west arrays, per transposition model."""
    overrides = LOSS_CASES[loss_case]
    m = day_mask(d.index, ok_days)
    meas = {label: d[f"P_{label}"][m].sum() * STEP_H for label in ("16A", "16D")}
    wn_meas = meas["16D"] / meas["16A"]

    cases = [(model, perez_sets[0]) for model in models]
    cases += [("perez", pset) for pset in perez_sets[1:]]
    rows = []
    for model, pset in cases:
        out = {}
        for label in ("16A", "16D"):
            r = run_chain(d, label, transposition_model=model, loss_overrides=overrides, model_perez=pset, weather=w)
            out[label] = r["ac_model_kW"][m].sum() * STEP_H
        wn_model = out["16D"] / out["16A"]
        rows.append(
            {
                "transposition": model,
                "perez_set": pset if model == "perez" else "-",
                "bias_16A_north_%": (out["16A"] / meas["16A"] - 1.0) * 100.0,
                "bias_16D_west_%": (out["16D"] / meas["16D"] - 1.0) * 100.0,
                "spread_pp": ((out["16D"] / meas["16D"]) - (out["16A"] / meas["16A"])) * 100.0,
                "W/N_model": wn_model,
                "W/N_meas": wn_meas,
                "ratio_err_%": (wn_model / wn_meas - 1.0) * 100.0,
            }
        )
    return pd.DataFrame(rows)


def decomposition_sweep(d: pd.DataFrame, mask: pd.Series, solpos: pd.DataFrame) -> pd.DataFrame:
    """What a GHI-only weather source costs, on top of transposition."""
    rows = []
    for dni_method in DNI_METHODS:
        w = weather_frame(d, dni_method, solpos)
        for model in ("isotropic", "haydavies", "perez"):
            poa = modelled_poa(w, GTI_SENSOR_TILT, GTI_SENSOR_AZIMUTH, model)
            m = irradiance_metrics(poa, d["gti_meas"], mask)
            rows.append(
                {
                    "dni": dni_method,
                    "transposition": model,
                    "POA_bias_%": m["bias_%"],
                    "POA_RMSE_Wm2": m["RMSE_Wm2"],
                    "r": m["r"],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--gti-data",
        type=Path,
        required=True,
        help="frame covering a year with good tilted-pyranometer coverage (Leg A)",
    )
    ap.add_argument(
        "--pair-data",
        type=Path,
        required=True,
        help="multi-year frame covering the 16A/16D pair (Leg B and the drift check)",
    )
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--loss-case", default="no-availability-no-shading")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)

    # ---- Leg A: irradiance only, no PV model in the way --------------------
    d = load(args.gti_data)
    solpos = solar_position(d.index)
    w = weather_frame(d, "closure", solpos)
    ok = screen_weather_days(d)
    mask = daytime(d, solpos) & day_mask(d.index, ok)
    print(
        f"LEG A: {args.gti_data.name}, {int(ok.sum())} of {len(ok)} days with a usable "
        f"weather record, {mask.sum():,} daytime samples"
    )

    print("\n=== LEG A: modelled POA vs the co-planar tilted pyranometer (20 deg / azimuth 0) ===")
    a = leg_a(d, w, mask)
    a.to_csv(args.outdir / "transposition_leg_a.csv", index=False)
    print(
        a[a.albedo == "default"][
            ["transposition", "perez_set", "model_kWh_m2", "meas_kWh_m2", "bias_%", "RMSE_Wm2", "r"]
        ]
        .round(3)
        .to_string(index=False)
    )
    print("\n-- albedo sensitivity (the ground term is common to both arrays at 20 deg,")
    print("   so albedo shifts the pair together and cannot fake a north/west difference) --")
    print(a[a.albedo != "default"][["transposition", "albedo", "bias_%", "RMSE_Wm2"]].round(3).to_string(index=False))

    print("\n=== decomposition: DNI from GHI alone vs the measured GHI/DHI closure ===")
    dec = decomposition_sweep(d, mask, solpos)
    dec.to_csv(args.outdir / "transposition_decomposition.csv", index=False)
    print(dec.round(3).to_string(index=False))

    del d, w, solpos, mask

    # ---- the pair: is 16A still the same array as 16D? ---------------------
    pair = load(args.pair_data)
    wp = weather_frame(pair, "closure")

    print("\n=== 16A against the rest of the site: measured energy ratios only, no model ===")
    meas_drift = pair_drift_measured(pair)
    meas_drift.to_csv(args.outdir / "transposition_pair_drift_measured.csv")
    print(meas_drift.round(4).to_string())
    print("\n16A holds against both independent north arrays through 2014 and then")
    print("steps down about 10 %. PAIR_WINDOW is chosen from this table, which")
    print("contains no model output and so cannot favour any transposition model.")

    print("\n=== 16A/16D pair health, year by year (Hay-Davies, loss-free reference) ===")
    print("arrays 12 and 13 are independent north-facing arrays, included as a control")
    drift = pair_drift(pair, wp)
    drift.to_csv(args.outdir / "transposition_pair_drift.csv", index=False)
    print(
        drift[
            [
                "year",
                "days",
                "bias_16A_%",
                "bias_16D_%",
                "bias_12_%",
                "bias_13_%",
                "W/N_meas",
                "W/N_model",
                "ratio_err_%",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    # ---- Leg B, over the window in which the pair is sound -----------------
    lo, hi = PAIR_WINDOW
    sub = pair.loc[lo:hi]
    ws = wp.loc[sub.index]
    ok_days = normal_days(sub, ws, ("16A", "16D"))
    print(
        f"\n=== LEG B: identical north (16A) and west (16D) arrays, {lo}-{hi}, "
        f"{int(ok_days.sum())} normal days, losses='{args.loss_case}' ==="
    )
    b = leg_b(sub, ws, args.loss_case, ok_days)
    b.to_csv(args.outdir / "transposition_leg_b.csv", index=False)
    print(b.round(4).to_string(index=False))
    print("\nratio_err_% is the headline: module parameters, inverter efficiency, the")
    print("loss stack, array age and site soiling are common to the pair and divide out")
    print("of the west/north ratio, leaving the model's handling of the off-axis plane.")
    print(f"\nwrote outputs to {args.outdir}")


if __name__ == "__main__":
    main()
