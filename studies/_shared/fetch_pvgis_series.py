"""Fetch PVGIS v5.3 hourly radiation series and write them as BREOS multi-year weather files.

PVGIS seriescalc returns plane-of-array components. At tilt 0 the plane is horizontal, so
  ghi = Gb(h) + Gd(h),  dhi = Gd(h),  dni = Gb(h) / sin(solar_elevation),
where the elevation is the one PVGIS reports for the same timestamp. PVGIS stamps each value at the
satellite slot (HH:10 for SARAH3, HH:30 for ERA5); the rows are labelled at HH:00 and the minute
offset is recorded as ``irradiance_time_offset_hours`` with ``radiation_time_basis = "instant"``,
the convention BREOS's PVGIS TMY fetcher uses.

Usage: uv run python fetch_pvgis_series.py porto berlin --db PVGIS-SARAH3 --start 2005 --end 2023
Writes data/<site>_pvgis_<db>_<start>_<end>.csv (+ .metadata.json sidecar).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

from breos.resources import load_config_json
from breos.weather import WEATHER_METADATA_KEY, save_weather_csv

DATA = Path(__file__).parent / "data"


def fetch(lat: float, lon: float, db: str, start: int, end: int, use_horizon: bool = True) -> pd.DataFrame:
    for attempt in range(5):
        try:
            raw, meta = pvlib.iotools.get_pvgis_hourly(
                lat, lon, start=start, end=end, raddatabase=db, components=True, surface_tilt=0,
                surface_azimuth=180, usehorizon=use_horizon, map_variables=True, timeout=300,
                url="https://re.jrc.ec.europa.eu/api/v5_3/",
            )
            break
        except Exception as exc:
            print(f"  retry {attempt}: {exc!s:.150}", file=sys.stderr, flush=True)
            time.sleep(15 * (attempt + 1))
    else:
        raise RuntimeError("PVGIS fetch failed")
    minutes = sorted(set(raw.index.minute))
    if len(minutes) != 1:
        raise ValueError(f"mixed timestamp minutes {minutes}")
    offset_h = minutes[0] / 60
    beam_h = raw["poa_direct"].clip(lower=0)
    dhi = raw["poa_sky_diffuse"].clip(lower=0)
    sin_el = np.sin(np.radians(raw["solar_elevation"]))
    dni = np.where(raw["solar_elevation"] > 1.0, beam_h / sin_el.where(sin_el > 0, np.nan), 0.0)
    dni = np.nan_to_num(np.clip(dni, 0, 1200))
    out = pd.DataFrame(
        {
            "ghi": (beam_h + dhi).round(2).to_numpy(),
            "dni": np.round(dni, 2),
            "dhi": dhi.round(2).to_numpy(),
            "temp_air": raw["temp_air"].to_numpy(),
            "wind_speed": raw["wind_speed"].to_numpy(),
        },
        index=raw.index.floor("h"),
    )
    out.index.name = "date"
    out.attrs[WEATHER_METADATA_KEY] = {
        "source": f"PVGIS_hourly_{db}",
        "api_metadata": meta,
        "raw_radiation_variables": ["Gb(i)@tilt0", "Gd(i)@tilt0"],
        "stored_radiation_variables": ["ghi", "dni", "dhi"],
        "derivation": "ghi=Gb(h)+Gd(h); dhi=Gd(h); dni=Gb(h)/sin(elevation) for elevation>1 deg, capped 1200",
        "radiation_time_basis": "instant",
        "timestamp_label_basis": "provider_hour",
        "timestamp_timezone": "UTC",
        "irradiance_time_offset_hours": offset_h,
        "horizon": {"status": "applied" if use_horizon else "not_applied", "provider": "pvgis",
                    "profile": "provider_default" if use_horizon else None},
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sites", nargs="+", help="preset keys, or name=lat,lon")
    ap.add_argument("--db", default="PVGIS-SARAH3")
    ap.add_argument("--start", type=int, default=2005)
    ap.add_argument("--end", type=int, default=2023)
    ap.add_argument("--no-horizon", action="store_true")
    args = ap.parse_args()
    locations = load_config_json("locations.json")
    DATA.mkdir(exist_ok=True)
    for site in args.sites:
        if "=" in site:
            site, coords = site.split("=")
            lat, lon = map(float, coords.split(","))
        else:
            lat, lon = locations[site]["latitude"], locations[site]["longitude"]
        slug = args.db.lower().replace("pvgis-", "")
        out = DATA / f"{site}_pvgis_{slug}_{args.start}_{args.end}.csv"
        if out.exists():
            print(f"{out.name} exists")
            continue
        df = fetch(lat, lon, args.db, args.start, args.end, not args.no_horizon)
        frame = df.reset_index()
        frame.attrs = df.attrs
        save_weather_csv(df, out)
        print(f"saved {out.name}: {len(df)} rows, offset {df.attrs[WEATHER_METADATA_KEY]['irradiance_time_offset_hours']:.3f} h, "
              f"mean annual GHI {df['ghi'].sum() / 1000 / (args.end - args.start + 1):.0f} kWh/m2")


if __name__ == "__main__":
    main()
