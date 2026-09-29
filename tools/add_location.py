#!/usr/bin/env python3
"""
Add a geocoded location to the packaged location presets.

Looks an address up with Nominatim, finds its time zone, and writes the entry
to breos/data/configs/locations.json in this checkout. The ``location`` config
key and tools/fetch_weather.py find it when BREOS runs from this checkout, for
example after ``uv sync``, not from an installed wheel. Run
tools/generate_option_docs.py afterwards to update the options page. For a
one-off site, set ``location`` to a latitude/longitude/timezone table instead.
Needs geopy and timezonefinder: ``pip install geopy timezonefinder``.

Usage:
    python tools/add_location.py "Porto, Portugal"
"""

import argparse
import json
import sys
from pathlib import Path

from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder

PROJECT_ROOT = Path(__file__).parent.parent
LOCATIONS_FILE = PROJECT_ROOT / "breos" / "data" / "configs" / "locations.json"


def get_location_data(address):
    geolocator = Nominatim(user_agent="breos-dev-tool")
    location = geolocator.geocode(address)
    return location


def get_timezone(lat, lng):
    tf = TimezoneFinder()
    return tf.timezone_at(lat=lat, lng=lng)


def save_location(slug, data):
    if not LOCATIONS_FILE.exists():
        print(f"Error: {LOCATIONS_FILE} not found.")
        return False

    try:
        with open(LOCATIONS_FILE, "r") as f:
            locations = json.load(f)
    except Exception as e:
        print(f"Error reading {LOCATIONS_FILE}: {e}")
        return False

    if slug in locations:
        overwrite = input(f"Location '{slug}' already exists. Overwrite? (y/N): ").lower()
        if overwrite != "y":
            print("Aborted.")
            return False

    locations[slug] = data

    try:
        with open(LOCATIONS_FILE, "w") as f:
            json.dump(locations, f, indent=4)
            f.write("\n")
        print(f"Successfully added '{slug}' to {LOCATIONS_FILE}")
        return True
    except Exception as e:
        print(f"Error writing to {LOCATIONS_FILE}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Add a new location to the simulation config.")
    parser.add_argument("address", nargs="?", help="The address to search for.")
    args = parser.parse_args()

    address = args.address
    if not address:
        address = input("Enter address to search: ")

    if not address:
        print("No address provided. Exiting.")
        return

    print(f"Searching for '{address}'...")
    location = get_location_data(address)

    if not location:
        print("Location not found.")
        return

    print("\nFound Location:")
    print(f"Address: {location.address}")
    print(f"Latitude: {location.latitude}")
    print(f"Longitude: {location.longitude}")
    print(f"OSM Link: https://www.openstreetmap.org/?mlat={location.latitude}&mlon={location.longitude}&zoom=12")

    confirm = input("\nIs this correct? (y/N): ").lower()
    if confirm != "y":
        print("Aborted.")
        return

    print("Finding timezone...")
    timezone = get_timezone(location.latitude, location.longitude)
    print(f"Timezone: {timezone}")

    default_slug = location.address.split(",")[0].lower().replace(" ", "_")
    slug = input(f"Enter key name for config (default: {default_slug}): ").strip()
    if not slug:
        slug = default_slug

    new_entry = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "timezone": timezone,
        "name": location.address,
    }

    save_location(slug, new_entry)


if __name__ == "__main__":
    main()
