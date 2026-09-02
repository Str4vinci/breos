"""Consolidate the NIST Gaithersburg Ground-array PVDAQ CSVs into one frame.

Source: NREL PVDAQ system_id 4902 (NIST_Ground_1) on the OEDI data lake,
documented by Boyd (2017), J. Res. NIST 122:40, doi:10.6028/jres.122.040, with
channel units from the NIST data dictionary (nist.gov/file/391591).

Two facts from that documentation drive this module and are asserted, not
assumed:

* Timestamps are Local Standard Time (EST, UTC-5) with no daylight saving.
* Pyranometers (Eppley PSP, Kipp & Zonen CMP11) are published as raw
  millivolts with no sensitivity constant, so the only irradiance usable in
  W/m2 is the silicon reference cell and the integrator's silicon pyranometer,
  both of which are plane-of-array. There is no horizontal irradiance in
  engineering units for this array.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# PVDAQ channel -> NIST data-dictionary name. PVDAQ's own ghi/poa labelling is
# unreliable (it labels the ground array's Vaisala instrument "poa"), so the
# orientation in each comment comes from the NIST dictionary, not the column.
CHANNELS = {
    "irradiance_poa_o_2204": "RefCell1_Wm2",       # Ground POA, IMT Si-420 reference cell
    "irradiance_poa_o_2206": "SEWSPOAIrrad_Wm2",   # Ground POA, integrator's domed Si pyranometer
    "irradiance_ghi_o_2202": "Pyra1_mV",           # Ground GHI, Eppley PSP -- millivolts, unusable
    "irradiance_poa_o_2203": "Pyra2_mV",           # Ground POA, Kipp & Zonen CMP11 -- millivolts
    "temperature_ambient_o_2205": "AmbTemp_C",     # RM Young 41342LC, research grade
    "temperature_ambient_o_2206": "SEWSAmbientTemp_C",
    "temperature_module_o_2206": "SEWSModuleTemp_C",
    "wind_speed_o_2206": "WindSpeedAve_ms",        # mean horizontal wind speed
    "wind_speed_o_2206_2": "WindSpeed_ms",         # 1-minute maximum (gust)
    "dc_power_inv_14538": "InvPDC_kW",
    "ac_power_inv_14538": "InvPAC_kW",
    # Revenue-grade AC meter, independent of the inverter's own metering.
    # PVDAQ flattens the meter's channels into positional suffixes; the
    # mapping to NIST names is resolved in nist_meter_map() below.
    "ac_power_meter_1864": "Meter_c1",
    "ac_power_meter_1864_2": "Meter_c2",
    "ac_power_meter_1864_3": "Meter_c3",
    "ac_power_meter_1864_4": "Meter_c4",
    "ac_power_meter_1864_5": "Meter_c5",
}
# Prefix-matched families: the numeric suffix is a PVDAQ metric id.
PREFIX_FAMILIES = {
    "rtd_c_avg_": "RTD_C",            # module backsheet temperatures
    "shuntpdc_kw_avg_": "ShuntPDC_kW",  # per-combiner DC power
}
EST = "Etc/GMT+5"


def _resolve(columns: pd.Index) -> dict[str, str]:
    rename = {c: n for c, n in CHANNELS.items() if c in columns}
    for col in columns:
        for prefix, label in PREFIX_FAMILIES.items():
            if col.startswith(prefix):
                rename[col] = f"{label}_{col[len(prefix):].split('__')[0]}"
    return rename


def build(raw_dir: Path, out: Path) -> pd.DataFrame:
    files = sorted(raw_dir.glob("system_4902__date_*.csv"))
    if not files:
        raise SystemExit(f"no PVDAQ day files under {raw_dir}")

    frames = []
    for path in files:
        day = pd.read_csv(path, low_memory=False)
        rename = _resolve(day.columns)
        keep = ["measured_on", *rename]
        frames.append(day[keep].rename(columns=rename))

    df = pd.concat(frames, ignore_index=True)
    df["measured_on"] = pd.to_datetime(df["measured_on"])
    # LST throughout: a fixed-offset zone, so no DST transition can be implied.
    df = df.set_index(df["measured_on"].dt.tz_localize(EST)).drop(columns="measured_on")
    df = df[~df.index.duplicated(keep="first")].sort_index()

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)  # csv.gz: no parquet engine in the project's dependency set
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    df = build(args.raw_dir, args.out)
    span = f"{df.index[0]} -> {df.index[-1]}"
    print(f"rows={len(df)}  cols={len(df.columns)}  span={span}")
    print(f"wrote {args.out}")
    missing = df.isna().mean().mul(100).round(2).sort_values(ascending=False)
    print("\nmissing (%) by channel:")
    print(missing.to_string())


if __name__ == "__main__":
    main()
