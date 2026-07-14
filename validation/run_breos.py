#!/usr/bin/env python3
"""Run the BREOS PV chain on the checked-in validation weather.

For every location in ``validation/locations.json`` with a weather file under
``validation/data/weather/``, computes hourly AC production with
``breos.solar.calculate_pv_production_ac`` for three model configs
(``isotropic`` — the shipped default; ``perez`` — what the external
references effectively use; and ``perez_mid`` — perez plus mid-interval
solar position, the full PVWatts/SAM convention) and writes monthly/annual
energies to ``validation/results/breos_results.json``.

``--write-baseline`` additionally snapshots the results to
``validation/baselines/breos_baseline.json``, the file
``tests/test_validation_drift.py`` guards against. Only regenerate the
baseline on *intentional* model changes, and say so in the changelog.

Usage:
    uv run python validation/run_breos.py
    uv run python validation/run_breos.py --write-baseline
"""

import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pvlib.location import Location  # noqa: E402

import breos  # noqa: E402
from breos.pv_modules import get_module  # noqa: E402
from breos.solar import calculate_pv_production_ac  # noqa: E402
from validation.common import (  # noqa: E402
    BASELINE_PATH,
    MODEL_CONFIGS,
    RESULTS_DIR,
    RESULTS_PATH,
    find_weather_file,
    load_spec,
    load_validation_weather,
)


def run_location(key: str, loc: dict, system: dict, weather_file: Path) -> dict:
    weather = load_validation_weather(weather_file)
    location = Location(loc["latitude"], loc["longitude"], tz=loc["timezone"])
    pv_params = get_module(system["module"])

    models = {}
    for name, model_kwargs in MODEL_CONFIGS.items():
        ac = calculate_pv_production_ac(
            weather_data=weather,
            location=location,
            tilt=loc["tilt"],
            surface_azimuth=loc["azimuth"],
            n_modules=system["n_modules"],
            pv_params=pv_params,
            freq="h",
            inverter_loading_ratio=system["dc_ac_ratio"],
            inverter_efficiency=system["inverter_efficiency"],
            albedo=system["albedo"],
            **model_kwargs,
        )
        monthly = ac.groupby(ac.index.month).sum() / 1000.0  # Wh -> kWh at hourly steps
        monthly_kwh = [round(float(monthly.get(m, 0.0)), 3) for m in range(1, 13)]
        annual_kwh = round(float(ac.sum()) / 1000.0, 3)
        models[name] = {"monthly_kwh": monthly_kwh, "annual_kwh": annual_kwh}
        print(f"  {name:>10}: {annual_kwh:8.1f} kWh/yr")

    return {"weather_file": weather_file.name, "models": models}


def main():
    parser = argparse.ArgumentParser(description="Run BREOS on the checked-in validation weather")
    parser.add_argument("locations", nargs="*", help="Location keys (default: all with weather)")
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Snapshot results as the drift-test baseline (intentional model changes only)",
    )
    args = parser.parse_args()

    system, locations = load_spec()
    keys = args.locations or list(locations)

    results = {
        "breos_version": breos.__version__,
        "generated": datetime.date.today().isoformat(),
        "system": system,
        "locations": {},
    }

    skipped = []
    for key in keys:
        weather_file = find_weather_file(key)
        if weather_file is None:
            skipped.append(key)
            continue
        print(f"=== {key} ===")
        results["locations"][key] = run_location(key, locations[key], system, weather_file)

    if skipped:
        print(f"\nSkipped (no weather file — run fetch_references.py first): {', '.join(skipped)}")
    if not results["locations"]:
        print("Nothing to run.")
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(f"\nWrote {RESULTS_PATH.relative_to(PROJECT_ROOT)}")

    if args.write_baseline:
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(RESULTS_PATH, BASELINE_PATH)
        print(f"Wrote {BASELINE_PATH.relative_to(PROJECT_ROOT)} (drift-test baseline)")


if __name__ == "__main__":
    main()
