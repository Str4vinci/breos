#!/usr/bin/env python3
"""Fetch validation weather and independent reference results.

For each location in ``validation/locations.json`` this fetches, per source:

- ``weather``:  PVGIS TMY hourly weather (ghi/dni/dhi/temp_air/wind_speed) —
  the input BREOS runs on. Saved under ``validation/data/weather/`` using the
  repo weather-file naming convention.
- ``pvgis``:    PVGIS v5.3 PVcalc monthly/annual PV output for the same site
  and mounting — the JRC's independent PV model (Huld et al. 2010), computed
  on the full multi-year satellite record.
- ``pvwatts``:  NREL PVWatts v8 monthly/annual AC output for the same system.
  Uses ``NREL_API_KEY`` if set, otherwise ``DEMO_KEY`` (rate-limited but
  sufficient for the handful of validation sites).

Each source is fetched independently and failures are recorded in the
reference JSON instead of aborting, so partial refreshes work — e.g. run
``--sources pvwatts`` later from a network that can reach
``developer.nrel.gov``. Existing reference JSONs are merged, not overwritten.

Usage:
    uv run python validation/fetch_references.py                    # everything
    uv run python validation/fetch_references.py porto berlin      # subset
    uv run python validation/fetch_references.py --sources pvwatts
    uv run python validation/fetch_references.py --force           # refetch weather
"""

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from breos.solar import resolve_pvwatts_losses  # noqa: E402
from breos.utils import safe_path_slug  # noqa: E402
from validation.common import REFERENCES_DIR, WEATHER_DIR, find_weather_file, load_spec  # noqa: E402

PVGIS_PVCALC_URL = "https://re.jrc.ec.europa.eu/api/v5_3/PVcalc"
PVWATTS_URL = "https://developer.nrel.gov/api/pvwatts/v8.json"

# PVWatts station-based international data can sit far from the site; beyond
# this distance the reference is recorded but flagged untrusted (excluded
# from CI band assertions).
PVWATTS_TRUST_DISTANCE_M = 200_000


def _today() -> str:
    return datetime.date.today().isoformat()


def resolve_loss_percentages(system: dict) -> dict:
    """Derive the loss numbers that make the three models comparable.

    BREOS applies the PVWatts component stack (~14.1%) on DC, then the
    inverter curve (nominal ``inverter_efficiency``). PVWatts takes the same
    DC-side ``losses`` plus a separate ``inv_eff``. PVGIS takes one combined
    system-loss figure that includes the inverter, so it gets
    ``1 - (1 - losses) * inverter_efficiency``.
    """
    combined = resolve_pvwatts_losses()["combined_pct"]
    inv_eff = system["inverter_efficiency"]
    pvgis_loss = (1.0 - (1.0 - combined / 100.0) * inv_eff) * 100.0
    return {
        "breos_losses_pct": round(combined, 2),
        "pvgis_loss_pct": round(pvgis_loss, 2),
    }


def fetch_weather(key: str, loc: dict, force: bool = False) -> dict:
    """Fetch a PVGIS TMY for the location and save it as CSV. Returns metadata."""
    existing = find_weather_file(key)
    if existing is not None and not force:
        print(f"  weather: already present ({existing.name})")
        return {"file": existing.name, "fetched": None}

    from breos.weather import fetch_tmy_weather_data

    print("  weather: fetching PVGIS TMY...")
    tmy_data, metadata = fetch_tmy_weather_data(
        latitude=loc["latitude"],
        longitude=loc["longitude"],
        sample_year=2025,
        freq="h",
        timezone=loc["timezone"],
        save_to_file=False,
    )

    inputs = metadata.get("inputs", {})
    meteo = inputs.get("meteo_data", {})
    rad_db = str(meteo.get("radiation_db", "unknown"))
    year_min = meteo.get("year_min", "unknown")
    year_max = meteo.get("year_max", "unknown")
    db_slug = f"pvgis-{rad_db.lower().replace('pvgis-', '')}"

    # Keep the repo light: only the columns BREOS reads, rounded to physical
    # precision, gzipped. The committed file is the deterministic input the
    # drift test recomputes from, so precision is fixed here, once.
    tmy_data = tmy_data[[c for c in ("ghi", "dni", "dhi", "temp_air", "wind_speed") if c in tmy_data.columns]]
    tmy_data = tmy_data.round({"ghi": 1, "dni": 1, "dhi": 1, "temp_air": 2, "wind_speed": 2})

    WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    filename = WEATHER_DIR / f"{safe_path_slug(key)}_tmy_{year_min}_{year_max}_{db_slug}.csv.gz"
    tmy_data.to_csv(filename)
    print(f"  weather: saved {filename.name} ({len(tmy_data):,} rows)")

    return {
        "file": filename.name,
        "radiation_db": rad_db,
        "year_min": year_min,
        "year_max": year_max,
        "fetched": _today(),
    }


def fetch_pvgis_pvcalc(loc: dict, system: dict, losses: dict) -> dict:
    """Fetch PVGIS PVcalc monthly/annual output (the JRC's own PV model)."""
    # PVGIS aspect convention: 0 = south, 90 = west, -90 = east.
    aspect = (loc["azimuth"] % 360) - 180
    params = {
        "lat": loc["latitude"],
        "lon": loc["longitude"],
        "peakpower": system["peak_power_kw"],
        "loss": losses["pvgis_loss_pct"],
        "angle": loc["tilt"],
        "aspect": aspect,
        "mountingplace": system["mounting"],
        "outputformat": "json",
    }
    print("  pvgis: fetching PVcalc...")
    resp = requests.get(PVGIS_PVCALC_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    meteo = data["inputs"]["meteo_data"]
    monthly_rows = sorted(data["outputs"]["monthly"]["fixed"], key=lambda r: r["month"])
    monthly_kwh = [round(float(r["E_m"]), 2) for r in monthly_rows]
    poa_monthly = [round(float(r["H(i)_m"]), 2) for r in monthly_rows]
    totals = data["outputs"]["totals"]["fixed"]

    print(f"  pvgis: E_y = {totals['E_y']:.0f} kWh ({meteo['radiation_db']}, {meteo['year_min']}-{meteo['year_max']})")
    return {
        "fetched": _today(),
        "params": params,
        "radiation_db": meteo["radiation_db"],
        "years": f"{meteo['year_min']}-{meteo['year_max']}",
        "monthly_kwh": monthly_kwh,
        "annual_kwh": round(float(totals["E_y"]), 2),
        "poa_monthly_kwh_m2": poa_monthly,
        "poa_annual_kwh_m2": round(float(sum(poa_monthly)), 2),
    }


def fetch_pvwatts(loc: dict, system: dict, losses: dict) -> dict:
    """Fetch PVWatts v8 monthly/annual AC output. Tries nsrdb, then intl."""
    api_key = os.environ.get("NREL_API_KEY", "DEMO_KEY")
    base_params = {
        "lat": loc["latitude"],
        "lon": loc["longitude"],
        "system_capacity": system["peak_power_kw"],
        "module_type": 0,  # standard crystalline
        "array_type": 0,  # fixed open rack — matches BREOS's free-standing Faiman default
        "tilt": loc["tilt"],
        "azimuth": loc["azimuth"],  # PVWatts uses the pvlib convention (180 = south)
        "losses": losses["breos_losses_pct"],
        "dc_ac_ratio": system["dc_ac_ratio"],
        "inv_eff": system["inverter_efficiency"] * 100.0,
        "albedo": system["albedo"],
        "radius": 0,  # use the closest station regardless of distance; we record it
    }

    last_error = None
    for dataset in ("nsrdb", "intl"):
        params = dict(base_params, dataset=dataset, api_key=api_key)
        print(f"  pvwatts: trying dataset={dataset}...")
        try:
            resp = requests.get(PVWATTS_URL, params=params, timeout=60)
            data = resp.json()
        except requests.exceptions.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            print(f"  pvwatts: unreachable ({type(exc).__name__})")
            break  # network problem, not a dataset problem
        except ValueError as exc:
            last_error = f"invalid JSON response: {exc}"
            continue

        errors = data.get("errors") or []
        if errors:
            last_error = "; ".join(str(e) for e in errors)
            continue

        outputs = data["outputs"]
        station = data.get("station_info", {}) or {}
        distance = station.get("distance")
        trusted = dataset == "nsrdb" or (distance is not None and distance <= PVWATTS_TRUST_DISTANCE_M)
        stored_params = {k: v for k, v in params.items() if k != "api_key"}

        print(f"  pvwatts: ac_annual = {outputs['ac_annual']:.0f} kWh (dataset={dataset}, station {distance} m)")
        return {
            "fetched": _today(),
            "params": stored_params,
            "dataset": dataset,
            "station_distance_m": distance,
            "station": station.get("location") or station.get("solar_resource_file"),
            "monthly_kwh": [round(float(v), 2) for v in outputs["ac_monthly"]],
            "annual_kwh": round(float(outputs["ac_annual"]), 2),
            "solrad_annual_kwh_m2_day": round(float(outputs["solrad_annual"]), 3),
            "trusted": bool(trusted),
        }

    print(f"  pvwatts: FAILED ({last_error})")
    return {"fetched": _today(), "error": last_error}


def main():
    parser = argparse.ArgumentParser(description="Fetch validation weather and reference results")
    parser.add_argument("locations", nargs="*", help="Location keys (default: all)")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["weather", "pvgis", "pvwatts"],
        default=["weather", "pvgis", "pvwatts"],
        help="Which sources to fetch (default: all)",
    )
    parser.add_argument("--force", action="store_true", help="Refetch weather even if a file exists")
    args = parser.parse_args()

    system, locations = load_spec()
    losses = resolve_loss_percentages(system)

    keys = args.locations or list(locations)
    unknown = [k for k in keys if k not in locations]
    if unknown:
        print(f"Unknown location(s): {', '.join(unknown)}. Available: {', '.join(locations)}")
        sys.exit(1)

    REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
    failures = []

    for key in keys:
        loc = locations[key]
        print(f"\n=== {key} ({loc['name']}) ===")

        ref_path = REFERENCES_DIR / f"{key}.json"
        if ref_path.exists():
            with open(ref_path) as f:
                ref = json.load(f)
        else:
            ref = {}

        ref.update(
            {
                "location_key": key,
                "name": loc["name"],
                "latitude": loc["latitude"],
                "longitude": loc["longitude"],
                "tilt": loc["tilt"],
                "azimuth_pvlib": loc["azimuth"],
                "system": {**system, **losses},
            }
        )

        if "weather" in args.sources:
            try:
                weather_meta = fetch_weather(key, loc, force=args.force)
                if weather_meta.get("fetched"):
                    ref["weather"] = weather_meta
                elif "weather" not in ref:
                    ref["weather"] = weather_meta
            except Exception as exc:
                print(f"  weather: FAILED ({exc})")
                failures.append((key, "weather", str(exc)))

        if "pvgis" in args.sources:
            try:
                ref["pvgis_pvcalc"] = fetch_pvgis_pvcalc(loc, system, losses)
            except Exception as exc:
                print(f"  pvgis: FAILED ({exc})")
                failures.append((key, "pvgis", str(exc)))

        if "pvwatts" in args.sources:
            ref["pvwatts_v8"] = fetch_pvwatts(loc, system, losses)
            if "error" in ref["pvwatts_v8"]:
                failures.append((key, "pvwatts", ref["pvwatts_v8"]["error"]))

        with open(ref_path, "w") as f:
            json.dump(ref, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"  saved {ref_path.relative_to(PROJECT_ROOT)}")

    if failures:
        print("\nIncomplete fetches (recorded in the reference JSONs):")
        for key, source, err in failures:
            print(f"  {key}/{source}: {err}")
    else:
        print("\nAll fetches completed.")


if __name__ == "__main__":
    main()
