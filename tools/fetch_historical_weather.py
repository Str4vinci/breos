#!/usr/bin/env python3
"""
Historical Weather Data Fetcher
================================
Batch-fetch historical weather data from Open-Meteo's Archive API for multiple cities.

Reads a JSON config specifying cities (inline or from locations.json) and a date range,
then calls pvbat.weather.fetch_weather_data() for each city. Data is saved to weather/.

Usage:
    python tools/fetch_historical_weather.py configs/tools/fetch_berlin_lisbon.json
    python tools/fetch_historical_weather.py configs/tools/fetch_berlin_lisbon.json --force
    python tools/fetch_historical_weather.py configs/tools/fetch_berlin_lisbon.json --add-locations
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from breos.weather import fetch_weather_data

LOCATIONS_PATH = PROJECT_ROOT / "configs" / "base" / "locations.json"
WEATHER_DIR = PROJECT_ROOT / "weather"


def load_locations() -> dict:
    with open(LOCATIONS_PATH) as f:
        return json.load(f)


def resolve_cities(config: dict) -> dict[str, dict]:
    """Resolve city definitions: inline dicts are used directly,
    string 'locations' means look up the key in locations.json."""
    locations = None
    resolved = {}

    for key, value in config["cities"].items():
        if isinstance(value, dict):
            resolved[key] = value
        elif value == "locations":
            if locations is None:
                locations = load_locations()
            if key not in locations:
                print(f"ERROR: City '{key}' not found in {LOCATIONS_PATH}")
                sys.exit(1)
            resolved[key] = locations[key]
        else:
            print(f"ERROR: Invalid city spec for '{key}': {value}")
            sys.exit(1)

    return resolved


def add_cities_to_locations(cities: dict[str, dict], config: dict) -> None:
    """Add inline-defined cities to locations.json if they aren't already there."""
    locations = load_locations()
    added = []

    for key, value in config["cities"].items():
        if isinstance(value, dict) and key not in locations:
            locations[key] = value
            added.append(key)

    if added:
        with open(LOCATIONS_PATH, "w") as f:
            json.dump(locations, f, indent=4)
            f.write("\n")
        print(f"Added to {LOCATIONS_PATH}: {', '.join(added)}")
    else:
        print("No new cities to add to locations.json")


def fetch_city(city_key: str, city_info: dict, start_year: int, end_year: int, source: str, force: bool) -> bool:
    """Fetch weather data for a single city. Returns True if data was fetched."""
    outfile = WEATHER_DIR / f"{city_key}_historical_{start_year}_{end_year}_{source}.csv"

    if outfile.exists() and not force:
        print(f"SKIP: {outfile.name} already exists (use --force to overwrite)")
        return False

    print(f"Fetching {city_key} ({city_info['name']}) from {start_year}-01-01 to {end_year}-12-31 ...")

    df = fetch_weather_data(
        latitude=city_info["latitude"],
        longitude=city_info["longitude"],
        start_date=f"{start_year}-01-01",
        end_date=f"{end_year}-12-31",
        tilt=0,
        azimuth=0,
        freq="h",
        save_to_file=False,
    )

    WEATHER_DIR.mkdir(exist_ok=True)
    df.to_csv(outfile)
    print(f"Saved {outfile.name}  ({len(df):,} rows, {outfile.stat().st_size / 1e6:.1f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Fetch historical weather data for cities")
    parser.add_argument("config", help="Path to JSON config file")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument(
        "--add-locations", action="store_true", help="Add inline-defined cities to configs/base/locations.json"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    print(f"Config: {config.get('description', args.config)}")

    start_year = config["start_year"]
    end_year = config["end_year"]
    delay = config.get("delay_seconds", 5)
    source = config.get("source", "openmeteo")

    if args.add_locations:
        add_cities_to_locations({}, config)

    cities = resolve_cities(config)
    print(f"Cities: {', '.join(cities.keys())}  |  Range: {start_year}-{end_year}  |  Source: {source}\n")

    fetched_count = 0
    city_keys = list(cities.keys())

    for i, key in enumerate(city_keys):
        city_source = cities[key].get("source", source) if isinstance(cities[key], dict) else source
        was_fetched = fetch_city(key, cities[key], start_year, end_year, city_source, args.force)
        if was_fetched:
            fetched_count += 1
            if i < len(city_keys) - 1:
                print(f"Waiting {delay}s before next request ...\n")
                time.sleep(delay)
        else:
            print()

    print(f"Done. Fetched {fetched_count}/{len(cities)} cities.")


if __name__ == "__main__":
    main()
