"""Consolidate the DKASC Alice Springs 5-minute CSVs into one aligned frame.

Source: Desert Knowledge Australia Solar Centre, Alice Springs, downloaded from
https://dkasolarcentre.com.au/download?location=alice-springs as one CSV per
"source id". Each per-array CSV already carries a copy of the weather-station
channels, so the weather file is used as the authoritative met record and the
array files contribute only their metering channels.

Four facts are established from the data itself in dkasc_facts.py and are
asserted here rather than assumed:

* Timestamps are Australian Central Standard Time (UTC+9:30) with no daylight
  saving. Fitting measured GHI against clear-sky GHI puts ACST at 120.3 W/m2
  RMSE against 137.1 (UTC+9), 142.1 (UTC+10) and 199.4 (UTC+10:30, i.e. ACDT).
* The stamp labels the *centre* of the 5-minute averaging window, not its start
  or end. The same fit, restricted to low-sun clear samples where the signal is
  steepest, minimises at -0.25 min; interval-start would give +2.5 and
  interval-end -2.5. Solar position is therefore evaluated *at* the stamp.
* Channels are in engineering units already -- W/m2, degC, m/s, kW, kWh. Unlike
  the NIST array there are no millivolt channels, but every irradiance and wind
  channel carries occasional out-of-range spikes (GHI to -986 and +2726 W/m2,
  wind to -1742 m/s) which are screened here, not modelled.
* ``Active_Power`` (kW) and ``Active_Energy_Delivered_Received`` (cumulative
  kWh) are independent channels off the same Class 0.5 meter. Their annual
  totals agree within 0.3-1.0 % for every array from 2009 to 2020, which is the
  cross-check that qualifies the power channel as the AC reference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# ACST, as a fixed-offset zone so no DST transition can ever be implied. The
# POSIX sign convention in the Etc/ zones is inverted, hence GMT-9 for UTC+9.
ACST_OFFSET = pd.Timedelta(hours=9, minutes=30)

# Source id -> (raw filename stem, short label). Confirmed two ways: against the
# column names inside the site-wide Alice_Springs_2025.csv export, which spell
# out "<id>_DKA_<meter>_<phase>", and against the per-source pages on
# dkasolarcentre.com.au. See dkasc_arrays.py for the array each meter carries.
SOURCES = {
    100: ("100-Site_DKA-M1_A-Phase", "16A"),
    81: ("81-Site_DKA-M2_A-Phase", "16D"),
    84: ("84-Site_DKA-M5_B-Phase", "12"),
    92: ("92-Site_DKA-M6_B-Phase", "13"),
    91: ("91-Site_DKA-M9_B-Phase", "1A"),
    # 59 is NOT site 38. Site 38 (Q CELLS 5.9 kW mono, 2017) sits on M19
    # *B*-phase; 59 is M19 C-phase, which carries an array this run did not
    # identify. The data agrees: source 59 reads zero until 2018, so whatever
    # is on it was energised then. It is built here but never analysed.
    59: ("59-Site_DKA-M19_C-Phase", "M19C"),
    # 214 is site 32, Canadian Solar 5.3 kW poly, fixed, 2016 -- the "II"
    # generation on M18 B-phase, the earlier one being a 2013 Canadian Solar
    # array. Its record starts 2016-11-09, matching that commissioning.
    214: ("214-Site_DKA-M18_B-Phase_II", "M18B2"),
    96: ("96-Site_DKA-MasterMeter1", "master"),
}
WEATHER_STEM = "101-Site_DKA-WeatherStation"

WEATHER_CHANNELS = {
    "Global_Horizontal_Radiation": "ghi",
    "Diffuse_Horizontal_Radiation": "dhi",
    "Radiation_Global_Tilted": "gti_meas",  # POA pyranometer, 20 deg / azimuth 0
    "Radiation_Diffuse_Tilted": "dti_meas",
    "Weather_Temperature_Celsius": "temp_air",
    "Wind_Speed": "wind_speed",
    "Weather_Relative_Humidity": "rh",
    "Weather_Daily_Rainfall": "rain_mm",
}
# Physically admissible ranges. Anything outside becomes NaN: these are logger
# spikes and sentinel values, and letting them through would corrupt an annual
# total far more than the missing sample does.
LIMITS = {
    "ghi": (-10.0, 1600.0),
    "dhi": (-10.0, 1000.0),
    "gti_meas": (-10.0, 1800.0),
    "dti_meas": (-10.0, 1200.0),
    "temp_air": (-15.0, 55.0),
    "wind_speed": (0.0, 60.0),
    "rh": (0.0, 100.0),
    "rain_mm": (0.0, 500.0),
}
POWER_LIMIT_FACTOR = 1.6  # kW, relative to array rating; above this is a spike


def _read(path: Path, usecols: list[str]) -> pd.DataFrame:
    """Read one DKASC CSV, coercing the timestamp and dropping corrupt rows.

    The 214 file carries one truncated line whose fields have run together
    (``12"2025-08-23 05:10:00"``), which makes pandas type the energy column as
    object; coercion drops that row and leaves the column numeric.
    """
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    ts = pd.to_datetime(df["timestamp"], errors="coerce", format="mixed")
    df = df.drop(columns="timestamp")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    ok = ts.notna().to_numpy()
    df.index = pd.DatetimeIndex(ts)
    return df[ok].sort_index()


def _screen(s: pd.Series, lo: float, hi: float) -> pd.Series:
    return s.where((s >= lo) & (s <= hi))


def build(raw_dir: Path, out: Path, start: int, end: int, sources: list[int]) -> pd.DataFrame:
    wpath = raw_dir / f"{WEATHER_STEM}.csv"
    if not wpath.exists():
        raise SystemExit(f"missing weather file {wpath}")

    w = _read(wpath, ["timestamp", *WEATHER_CHANNELS]).rename(columns=WEATHER_CHANNELS)
    w = w.loc[str(start) : str(end)]
    for col, (lo, hi) in LIMITS.items():
        w[col] = _screen(w[col], lo, hi)
    # Small negative irradiance at night is thermal offset in the pyranometer,
    # not a fault; clip it away only after the spike screen above.
    for col in ("ghi", "dhi", "gti_meas", "dti_meas"):
        w[col] = w[col].clip(lower=0.0)

    from dkasc_arrays import ARRAYS  # local import: keeps the array table in one file

    for sid in sources:
        stem, label = SOURCES[sid]
        path = raw_dir / f"{stem}.csv"
        if not path.exists():
            print(f"  ! skipping source {sid}: {path.name} not present")
            continue
        a = _read(path, ["timestamp", "Active_Power", "Active_Energy_Delivered_Received"])
        a = a.loc[str(start) : str(end)]
        rating = ARRAYS[label].dc_kw if label in ARRAYS else 300.0
        p = _screen(a["Active_Power"], -0.1, rating * POWER_LIMIT_FACTOR).clip(lower=0.0)
        w[f"P_{label}"] = p.reindex(w.index)
        w[f"E_{label}"] = a["Active_Energy_Delivered_Received"].reindex(w.index)

    w = w[~w.index.duplicated(keep="first")]
    # Reindex onto an exact 5-minute grid so every array shares one clock and
    # gaps are explicit rather than implied by a missing row.
    grid = pd.date_range(w.index[0].floor("5min"), w.index[-1].ceil("5min"), freq="5min")
    w = w.reindex(grid)
    w.index.name = "timestamp_acst"

    out.parent.mkdir(parents=True, exist_ok=True)
    w.to_csv(out)  # csv.gz: no parquet engine in the project's dependency set
    return w


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start-year", type=int, default=2016)
    ap.add_argument("--end-year", type=int, default=2016)
    ap.add_argument("--sources", type=int, nargs="*", default=sorted(SOURCES))
    args = ap.parse_args()

    df = build(args.raw_dir, args.out, args.start_year, args.end_year, args.sources)
    print(f"rows={len(df)}  cols={len(df.columns)}  span={df.index[0]} -> {df.index[-1]} (ACST)")
    print(f"wrote {args.out}")
    miss = df.isna().mean().mul(100).round(2).sort_values(ascending=False)
    print("\nmissing (%) by channel:")
    print(miss.to_string())


if __name__ == "__main__":
    main()
