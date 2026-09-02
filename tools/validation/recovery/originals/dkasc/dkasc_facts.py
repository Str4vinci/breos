"""Establish the DKASC record's conventions from the data, before modelling.

Every check here exists because the equivalent question cost real time on the
NIST run, where the published column labels were wrong in places and two
irradiance channels turned out to be raw millivolts. Nothing in this file
assumes a convention from a column name or a download page; each fact is
measured against the sun or against a second channel.

Run it first. Its output is the evidence behind the assertions in
dkasc_build.py and dkasc_validate.py.

    1. Timezone           -- fit measured GHI against clear-sky GHI at each
                             candidate UTC offset. ACST (+9:30) wins by a
                             factor of two in RMSE; ACDT (+10:30) is excluded
                             outright, so no daylight saving is applied.
    2. Interval labelling -- the same fit swept over sub-interval offsets, on
                             low-sun clear samples where the signal is steepest.
                             Interval-start would minimise at +2.5 min and
                             interval-end at -2.5 min.
    3. Sensor orientation -- fit the tilted pyranometer's tilt and azimuth from
                             the data, as an independent check on DKASC's
                             published 20 deg / azimuth 0 for the fixed arrays.
                             The published geometry is what the validation uses;
                             this fit only confirms it.
    4. Metering           -- integrate Active_Power and difference the
                             cumulative Active_Energy counter. Two independent
                             channels off the same meter; the year they stop
                             agreeing is the year the record stops being usable.
    5. Channel ranges     -- min/max per channel per year, to expose sentinel
                             values and spikes before they reach an annual total.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
from pvlib.location import Location

from dkasc_arrays import SITE_ALT, SITE_LAT, SITE_LON
from dkasc_build import LIMITS, SOURCES, WEATHER_CHANNELS, WEATHER_STEM, _read

CANDIDATE_OFFSETS = {
    "UTC+9:00 (AWST+1)": 9.0,
    "UTC+9:30 (ACST)": 9.5,
    "UTC+10:00 (AEST)": 10.0,
    "UTC+10:30 (ACDT, i.e. ACST with DST)": 10.5,
}


def _clearsky(index_utc: pd.DatetimeIndex) -> np.ndarray:
    loc = Location(SITE_LAT, SITE_LON, tz="UTC", altitude=SITE_ALT)
    return loc.get_clearsky(index_utc, model="ineichen")["ghi"].to_numpy()


def _scaled_rmse(measured: np.ndarray, modelled: np.ndarray) -> float:
    """RMSE after least-squares scaling the model onto the measurement.

    Ineichen clear-sky carries several per cent of level error at this site,
    which is far larger than any timing effect and would otherwise swamp the
    comparison. Removing a single gain leaves the *shape* of the daily curve,
    which is what a timing error distorts.
    """
    gain = (measured * modelled).sum() / (modelled * modelled).sum()
    return float(np.sqrt(((measured - gain * modelled) ** 2).mean()))


def _clear_sample_mask(w: pd.DataFrame) -> np.ndarray:
    """Select bright samples on clear days, using measurements only.

    This mask must not be built from a clear-sky series computed at one of the
    candidate offsets. Doing so lets every offset select the samples that suit
    it, and the comparison stops meaning anything: an earlier draft of this
    check did exactly that and ranked UTC+9:00 ahead of UTC+9:30. Daily
    clearness picks the clear days, because a daily total barely moves under a
    one-hour shift, and the per-sample cut is on measured GHI alone.
    """
    utc = w.index.tz_localize("UTC") - pd.Timedelta(hours=9, minutes=30)
    daily_clearness = (w["ghi"].resample("D").sum()
                       / pd.Series(_clearsky(utc), index=w.index).resample("D").sum())
    clear_days = set(daily_clearness[daily_clearness > 0.70].index.date)
    on_clear_day = np.array([t.date() in clear_days for t in w.index])
    return on_clear_day & (w["ghi"].to_numpy() > 200.0)


def timezone_and_labelling(w: pd.DataFrame) -> None:
    """Facts 1 and 2: which offset, and where in the interval the stamp sits."""
    ghi = w["ghi"].to_numpy()
    clear = _clear_sample_mask(w)

    print("\n=== 1. timezone ===")
    print(f"bright samples on clear days (selected from measurements only): {clear.sum():,}")
    print("RMSE of measured GHI against clear-sky GHI, on that one fixed sample set,")
    print("after least-squares scaling each candidate to remove the clear-sky model's")
    print("own level error, so the statistic measures shape and not calibration.")
    print("(best sub-interval shift searched within +/-5 min in each case)")
    for name, off in CANDIDATE_OFFSETS.items():
        utc = w.index.tz_localize("UTC") - pd.Timedelta(hours=off)
        best = min(
            _scaled_rmse(ghi[clear], _clearsky(utc + pd.Timedelta(minutes=s))[clear])
            for s in np.arange(-5, 5.5, 0.5)
        )
        print(f"  {name:38s} {best:8.2f} W/m2")

    print("\n=== 2. interval labelling ===")
    utc = w.index.tz_localize("UTC") - pd.Timedelta(hours=9, minutes=30)
    loc = Location(SITE_LAT, SITE_LON, tz="UTC", altitude=SITE_ALT)
    zen = loc.get_solarposition(utc)["apparent_zenith"].to_numpy()
    # Low sun is the steepest part of the day, where a 2.5-minute error shows up
    # most strongly. The clear-day selection stays measurement-based.
    sel = clear & (zen > 55) & (zen < 82)
    rows = []
    for sh in np.arange(-6, 6.01, 0.25):
        m = _clearsky(utc + pd.Timedelta(minutes=sh))
        rows.append((sh, _scaled_rmse(ghi[sel], m[sel])))
    r = pd.DataFrame(rows, columns=["shift_min", "RMSE"])
    best = r.shift_min[r.RMSE.idxmin()]
    print(f"  low-sun clear samples: {sel.sum():,}")
    print(f"  best shift {best:+.2f} min   "
          f"(interval-start would be +2.50, interval-end -2.50, centre 0.00)")
    print(f"  RMSE at -2.50 / 0.00 / +2.50 min: "
          f"{r.set_index('shift_min').RMSE.reindex([-2.5, 0.0, 2.5]).round(2).to_list()} W/m2")


def sensor_orientation(w: pd.DataFrame) -> None:
    """Fact 3: recover the tilted pyranometer's orientation from the data."""
    print("\n=== 3. tilted-pyranometer orientation (check on the published 20 deg / azimuth 0) ===")
    utc = w.index.tz_localize("UTC") - pd.Timedelta(hours=9, minutes=30)
    loc = Location(SITE_LAT, SITE_LON, tz="UTC", altitude=SITE_ALT)
    sp = loc.get_solarposition(utc)
    zen, az = sp["apparent_zenith"].to_numpy(), sp["azimuth"].to_numpy()
    extra = pvlib.irradiance.get_extra_radiation(utc).to_numpy()
    cosz = np.cos(np.radians(zen))
    ghi, dhi, gti = (w[c].to_numpy() for c in ("ghi", "dhi", "gti_meas"))
    dni = np.clip(np.divide(ghi - dhi, cosz, out=np.zeros(len(w)), where=cosz > 0.05), 0, 1400)
    ok = np.isfinite(ghi) & np.isfinite(dhi) & np.isfinite(gti) & (zen < 80) & (ghi > 100)
    print(f"  usable samples: {ok.sum():,}")

    def rmse(tilt: float, azim: float, sel: np.ndarray) -> float:
        poa = pvlib.irradiance.get_total_irradiance(
            tilt, azim % 360, zen[sel], az[sel], dni[sel], ghi[sel], dhi[sel],
            model="haydavies", dni_extra=extra[sel], albedo=0.25)["poa_global"]
        # pvlib returns plain ndarrays, not a DataFrame, when every input is an
        # array, so this must not assume a pandas result.
        return float(np.sqrt(((np.asarray(poa) - gti[sel]) ** 2).mean()))

    # Coarse pass on every 4th sample, then refine on all of them. The coarse
    # grid only has to land in the right basin; the refinement sets the answer.
    idx = np.flatnonzero(ok)
    coarse = idx[::4]
    best = min(((rmse(t, a, coarse), t, a)
                for t in np.arange(0.0, 46.0, 5.0)
                for a in np.arange(0.0, 360.0, 10.0)), key=lambda r: r[0])
    _, t0, a0 = best
    best = min(((rmse(t, a, idx), t, a)
                for t in np.arange(max(0.0, t0 - 5), t0 + 5.01, 1.0)
                for a in np.arange(a0 - 10, a0 + 10.01, 2.0)), key=lambda r: r[0])
    _, t0, a0 = best
    best = min(((rmse(t, a, idx), t, a)
                for t in np.arange(max(0.0, t0 - 1), t0 + 1.01, 0.5)
                for a in np.arange(a0 - 2, a0 + 2.01, 1.0)), key=lambda r: r[0])
    best = (best[0], best[1], best[2] % 360)
    print(f"  data-driven fit: tilt {best[1]:.1f} deg, azimuth {best[2]:.1f} deg  "
          f"(RMSE {best[0]:.2f} W/m2)")
    print("  DKASC publishes 20 deg / azimuth 0 (solar north) for the fixed arrays.")
    print("  Azimuth is degrees clockwise from north, so 0 is the sun-facing")
    print("  orientation in the southern hemisphere; a hemisphere sign error")
    print("  would place this fit near 180, not near 0.")


def metering_cross_check(raw_dir: Path, sources: list[int]) -> None:
    """Fact 4: does the power channel integrate to the meter's own counter?"""
    print("\n=== 4. metering cross-check: integrated Active_Power vs cumulative counter ===")
    print("ratio of the two per array-year; the year they diverge is the year the")
    print("5-minute power record starts dropping samples the counter still accrues\n")
    table = {}
    for sid in sources:
        stem, label = SOURCES[sid]
        path = raw_dir / f"{stem}.csv"
        if not path.exists():
            continue
        a = _read(path, ["timestamp", "Active_Power", "Active_Energy_Delivered_Received"])
        p, e = a["Active_Power"].clip(lower=0), a["Active_Energy_Delivered_Received"]
        by_power = p.groupby(p.index.year).sum() / 12.0
        by_count = e.groupby(e.index.year).apply(
            lambda s: s.dropna().iloc[-1] - s.dropna().iloc[0] if s.notna().any() else np.nan)
        table[label] = (by_power / by_count).round(4)
    print(pd.DataFrame(table).to_string())


def channel_ranges(w: pd.DataFrame, raw: pd.DataFrame) -> None:
    """Fact 5: units and out-of-range values, from the data not the column name."""
    print("\n=== 5. channel ranges before and after screening ===")
    rows = []
    for col, (lo, hi) in LIMITS.items():
        r, s = raw[col], w[col]
        rows.append({"channel": col, "raw_min": r.min(), "raw_max": r.max(),
                     "screened_min": s.min(), "screened_max": s.max(),
                     "rejected": int(r.notna().sum() - s.notna().sum()),
                     "limits": f"[{lo}, {hi}]"})
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    print("\nEvery channel is already in engineering units -- W/m2, degC, m/s -- so")
    print("unlike the NIST array there is no millivolt channel to convert. What the")
    print("raw columns do carry is out-of-range spikes and sentinel values, which")
    print("would corrupt an annual total far more than the missing sample does.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--year", type=int, default=2016,
                    help="year used for the timing and orientation fits")
    ap.add_argument("--fit-years", type=int, default=3,
                    help="years ending at --year pooled for the interval-labelling fit")
    ap.add_argument("--sections", default="1,2,3,4,5",
                    help="comma-separated subset to run; the timing fits (1,2) are "
                         "the slow ones")
    args = ap.parse_args()
    pd.set_option("display.width", 220)

    raw = _read(args.raw_dir / f"{WEATHER_STEM}.csv",
                ["timestamp", *WEATHER_CHANNELS]).rename(columns=WEATHER_CHANNELS)
    w = raw.copy()
    for col, (lo, hi) in LIMITS.items():
        w[col] = w[col].where((w[col] >= lo) & (w[col] <= hi))
    for col in ("ghi", "dhi", "gti_meas", "dti_meas"):
        w[col] = w[col].clip(lower=0.0)

    span = w.loc[str(args.year - args.fit_years + 1):str(args.year)]
    one = w.loc[str(args.year)]
    print(f"weather record: {len(raw):,} rows, {raw.index[0]} -> {raw.index[-1]}")
    print(f"timing fits use {args.year - args.fit_years + 1}-{args.year}; "
          f"orientation fit uses {args.year}")

    want = {s.strip() for s in args.sections.split(",")}
    if {"1", "2"} & want:
        timezone_and_labelling(span)
    if "3" in want:
        sensor_orientation(one)
    if "4" in want:
        metering_cross_check(args.raw_dir, sorted(SOURCES))
    if "5" in want:
        channel_ranges(one, raw.loc[str(args.year)])


if __name__ == "__main__":
    main()
