"""Validate the BREOS PV chain against the NIST Gaithersburg Ground array, 2016.

The array is driven from *measured* plane-of-array irradiance rather than from
GHI, because NIST publishes its pyranometers as raw millivolts with no
sensitivity constant; the only irradiance in engineering units is the silicon
reference cell, which is plane-of-array. That constraint is a virtue here: it
removes decomposition, transposition and weather-station-distance error from
the comparison, leaving the module, thermal, loss and inverter stages -- the
part of the chain a whole-system yield comparison cannot isolate.

Measurement references (chosen in preference to the inverter's own metering,
which integrates to negative monthly energy in April and October 2016):

* DC  -- sum of the seven combiner-box shunt channels, 100 % coverage.
* AC  -- the independent revenue-grade AC meter, cross-checked against its own
         cumulative kWh counter (agreement within 0.1 % in every month).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from breos.inverter import _calculate_dc_ac_power_arrays
from breos.pv.temperature import calculate_cell_temperature
from breos.solar import (
    PVModuleParams,
    _apply_pvwatts_loss_series,
    _module_dc_before_losses,
)

# Sharp NU-U235F2, from the CEC module database entry of the same name. NIST
# Technical documentation gives the array as 1152 of these at 20 deg tilt,
# azimuth 180, one PV Powered PVP260kW inverter.
MODULE = PVModuleParams(
    Mpp=235.2, Vmp=30.0, Imp=7.84, Voc=37.0, Isc=8.6,
    T_Pmax_pct=-0.458,          # CEC gamma_r
    T_Voc_pct=-0.12173 / 37.0 * 100,
    T_Isc_pct=0.003784 / 8.6 * 100,
    N_Cells=60,
    Name="Sharp NU-U235F2",
    Module_Efficiency=235.2 / (1.573 * 1000.0),
    NOCT=45.4,
    celltype="monoSi",
)
N_MODULES = 1152
INVERTER_AC_W = 260_000.0
# CEC database Paco/Pdco for PV_Powered__PVP260KW__480V_.
INVERTER_ETA = 260_000.0 / 269_829.8125
SHUNTS = [f"ShuntPDC_kW_{i}" for i in range(1, 8)]

# Named loss configurations. "breos-default" is DEFAULT_PVWATTS_LOSSES, i.e.
# what a BREOS user gets without touching anything. "no-availability" removes
# the 3 % availability allowance, which double-counts here because the measured
# series already reflects whatever downtime actually occurred. "module-only"
# strips the whole stack to isolate the CEC and thermal models.
LOSS_CASES = {
    "breos-default": None,
    "no-availability": {"availability": 0.0},
    "no-availability-no-shading": {"availability": 0.0, "shading": 0.0},
    "module-only": {k: 0.0 for k in (
        "soiling", "shading", "snow", "mismatch", "wiring",
        "connections", "lid", "nameplate_rating", "availability")},
}


def load(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, index_col=0, parse_dates=True)
    d["ShuntDC_kW"] = d[SHUNTS].sum(axis=1)
    d = d.rename(columns={"Meter_c5": "MeterAC_kW"})
    # Wind drops out for 5.6 % of minutes; interpolate over short gaps and fall
    # back to the annual median rather than dropping otherwise-good minutes.
    wind = d["WindSpeedAve_ms"].interpolate(limit=30)
    d["wind"] = wind.fillna(wind.median())
    return d


def run_chain(d: pd.DataFrame, poa_col: str, temp_model: str,
              loss_overrides, degradation_pct: float) -> pd.DataFrame:
    poa = d[poa_col].to_numpy(dtype=float)
    temp_air = d["AmbTemp_C"].to_numpy(dtype=float)
    wind = d["wind"].to_numpy(dtype=float)

    temp_cell = calculate_cell_temperature(
        poa, temp_air, wind, temp_model,
        module_efficiency=MODULE.Module_Efficiency, noct=MODULE.NOCT,
    )
    dc_raw = _module_dc_before_losses(
        np.clip(poa, 0.0, None), temp_cell, MODULE, N_MODULES, d.index, "dc_raw_W"
    )
    _, dc_after, _, _ = _apply_pvwatts_loss_series(
        dc_raw, loss_overrides, age_degradation_percent=degradation_pct
    )
    ac_w, _, _ = _calculate_dc_ac_power_arrays(
        dc_after.to_numpy(), INVERTER_AC_W, INVERTER_ETA
    )
    return pd.DataFrame(
        {
            "temp_cell": temp_cell,
            "dc_model_kW": dc_raw.to_numpy() / 1000.0,
            "dc_model_after_losses_kW": dc_after.to_numpy() / 1000.0,
            "ac_model_kW": ac_w / 1000.0,
        },
        index=d.index,
    )


def metrics(model_kw: pd.Series, meas_kw: pd.Series) -> dict:
    e_model, e_meas = model_kw.sum() / 60.0, meas_kw.sum() / 60.0
    dm = model_kw.resample("D").sum() / 60.0
    dd = meas_kw.resample("D").sum() / 60.0
    ok = dd > 1.0
    return {
        "model_kWh": e_model,
        "meas_kWh": e_meas,
        "bias_%": (e_model / e_meas - 1.0) * 100.0,
        "daily_RMSE_kWh": float(np.sqrt(((dm[ok] - dd[ok]) ** 2).mean())),
        "daily_nRMSE_%": float(np.sqrt(((dm[ok] - dd[ok]) ** 2).mean()) / dd[ok].mean() * 100),
        "daily_r": float(dm[ok].corr(dd[ok])),
        "hourly_r": float((model_kw.resample("h").mean()).corr(meas_kw.resample("h").mean())),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--poa-col", default="RefCell1_Wm2")
    ap.add_argument("--temp-model", default="faiman")
    ap.add_argument("--degradation-pct", type=float, default=0.0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    d = load(args.data)
    rows = []
    for case, overrides in LOSS_CASES.items():
        r = run_chain(d, args.poa_col, args.temp_model, overrides, args.degradation_pct)
        dc = metrics(r["dc_model_after_losses_kW"], d["ShuntDC_kW"])
        ac = metrics(r["ac_model_kW"], d["MeterAC_kW"])
        rows.append({"case": case, "stage": "DC (vs combiner shunts)", **dc})
        rows.append({"case": case, "stage": "AC (vs revenue meter)", **ac})

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
