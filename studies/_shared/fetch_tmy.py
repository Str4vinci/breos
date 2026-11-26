"""Fetch PVGIS v5.3 TMYs through BREOS's own fetcher (horizon on, sample year 2025).

Usage: uv run python fetch_tmy.py   -> data/<site>_tmy_pvgis.csv (+ sidecar)
"""

from __future__ import annotations

from pathlib import Path

from breos.weather import fetch_tmy_weather_data, save_weather_csv

DATA = Path(__file__).parent / "data"
SITES = {
    "porto": (41.1579, -8.6291, "Europe/Lisbon"),
    "lisbon": (38.7223, -9.1393, "Europe/Lisbon"),
    "madrid": (40.4168, -3.7038, "Europe/Madrid"),
    "athens": (37.9838, 23.7275, "Europe/Athens"),
    "erlangen": (49.5897, 11.0078, "Europe/Berlin"),
    "berlin": (52.52, 13.405, "Europe/Berlin"),
    "stockholm": (59.3293, 18.0686, "Europe/Stockholm"),
}

if __name__ == "__main__":
    for site, (lat, lon, tz) in SITES.items():
        out = DATA / f"{site}_tmy_pvgis.csv"
        if out.exists():
            continue
        tmy, meta = fetch_tmy_weather_data(lat, lon, sample_year=2025, timezone=tz)
        save_weather_csv(tmy, out)
        months = meta.get("months_selected") or meta.get("outputs", {}).get("months_selected")
        print(site, len(tmy), "months:", [m.get("year") for m in months] if months else "?")
