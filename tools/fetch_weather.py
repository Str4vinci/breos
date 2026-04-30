#!/usr/bin/env python3
"""
Weather Data Fetcher
====================
Fetch and save TMY or historical weather data for a location.

Usage:
    python tools/fetch_weather.py tmy --location porto
    python tools/fetch_weather.py historical --location porto --start 2005 --end 2024
    python tools/fetch_weather.py tmy --location porto --force

Locations are resolved from configs/base/locations.json.
Run with --list to see available locations.
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

LOCATIONS_PATH = PROJECT_ROOT / "configs" / "base" / "locations.json"
WEATHER_DIR = PROJECT_ROOT / "weather"


def load_locations() -> dict:
    with open(LOCATIONS_PATH) as f:
        return json.load(f)


def resolve_location(name: str) -> dict:
    locations = load_locations()
    if name not in locations:
        available = ", ".join(locations.keys())
        print(f"ERROR: Location '{name}' not found.")
        print(f"Available: {available}")
        print(f"\nTo add a new location, edit {LOCATIONS_PATH}")
        print("or use: python tools/fetch_historical_weather.py <config.json> --add-locations")
        sys.exit(1)
    return locations[name]


def fetch_tmy(args):
    from breos.weather import fetch_tmy_weather_data, parse_weather_filename

    loc = resolve_location(args.location)
    print(f"Location: {args.location} ({loc['latitude']}, {loc['longitude']})")

    # Check if TMY already exists locally
    WEATHER_DIR.mkdir(exist_ok=True)
    existing = []
    for fname in os.listdir(WEATHER_DIR):
        parsed = parse_weather_filename(fname)
        if parsed and parsed["location"] == args.location and parsed["type"] == "tmy":
            existing.append(fname)

    if existing and not args.force:
        print(f"TMY already exists: {existing[0]}")
        print("Use --force to overwrite or fetch a new one.")
        return

    print("Fetching TMY from PVGIS...")
    tmy_data, metadata = fetch_tmy_weather_data(
        latitude=loc["latitude"],
        longitude=loc["longitude"],
        freq="h",
        save_to_file=False,
    )

    # Build proper filename from metadata
    inputs = metadata.get("inputs", {})
    rad_db = inputs.get("meteo_data", {}).get("radiation_db", "unknown")
    year_min = inputs.get("meteo_data", {}).get("year_min", "unknown")
    year_max = inputs.get("meteo_data", {}).get("year_max", "unknown")
    db_slug = f"pvgis-{rad_db.lower().replace('pvgis-', '')}"
    filename = WEATHER_DIR / f"{args.location}_tmy_{year_min}_{year_max}_{db_slug}.csv"

    tmy_data.to_csv(filename)
    size_mb = filename.stat().st_size / 1e6
    print(f"Saved {filename.name}  ({len(tmy_data):,} rows, {size_mb:.1f} MB)")


def fetch_historical(args):
    from breos.weather import fetch_weather_data

    loc = resolve_location(args.location)
    print(f"Location: {args.location} ({loc['latitude']}, {loc['longitude']})")
    print(f"Period: {args.start}-{args.end}")

    filename = WEATHER_DIR / f"{args.location}_historical_{args.start}_{args.end}_openmeteo.csv"

    if filename.exists() and not args.force:
        print(f"Already exists: {filename.name}")
        print("Use --force to overwrite.")
        return

    WEATHER_DIR.mkdir(exist_ok=True)
    print("Fetching from Open-Meteo Archive API...")
    df = fetch_weather_data(
        latitude=loc["latitude"],
        longitude=loc["longitude"],
        start_date=f"{args.start}-01-01",
        end_date=f"{args.end}-12-31",
        tilt=0,
        azimuth=0,
        freq="h",
        save_to_file=False,
    )

    df.to_csv(filename)
    size_mb = filename.stat().st_size / 1e6
    print(f"Saved {filename.name}  ({len(df):,} rows, {size_mb:.1f} MB)")


def list_locations(args):
    locations = load_locations()
    print(f"Available locations ({LOCATIONS_PATH}):\n")
    for key, loc in locations.items():
        print(f"  {key:15s}  {loc.get('name', '')} ({loc['latitude']}, {loc['longitude']})")


def main():
    parser = argparse.ArgumentParser(description="Fetch and save weather data for a location")
    sub = parser.add_subparsers(dest="command")

    # tmy subcommand
    tmy_p = sub.add_parser("tmy", help="Fetch TMY from PVGIS")
    tmy_p.add_argument("--location", required=True, help="Location key (e.g., 'porto')")
    tmy_p.add_argument("--force", action="store_true", help="Overwrite existing file")

    # historical subcommand
    hist_p = sub.add_parser("historical", help="Fetch historical data from Open-Meteo")
    hist_p.add_argument("--location", required=True, help="Location key (e.g., 'porto')")
    hist_p.add_argument("--start", type=int, required=True, help="Start year (e.g., 2005)")
    hist_p.add_argument("--end", type=int, required=True, help="End year (e.g., 2024)")
    hist_p.add_argument("--force", action="store_true", help="Overwrite existing file")

    # list subcommand
    sub.add_parser("list", help="List available locations")

    args = parser.parse_args()

    if args.command == "tmy":
        fetch_tmy(args)
    elif args.command == "historical":
        fetch_historical(args)
    elif args.command == "list":
        list_locations(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
