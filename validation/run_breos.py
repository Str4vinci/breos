#!/usr/bin/env python3
"""Run the BREOS PV chain on the checked-in validation weather.

For every location in ``validation/locations.json`` with a weather file under
``validation/data/weather/``, computes hourly AC production with
``breos.solar.calculate_pv_production_ac`` for the focused model variants in
``validation.common.MODEL_CONFIGS`` and writes monthly/annual energies,
annual front-side POA (kWh/m²), and DC nameplate (kW) to
``validation/results/breos_results.json``. The variants cover the shipped
isotropic default, Perez, timestamp and IAM choices, roof temperature, and
front-only/rear-gain comparisons including equal-nameplate module counts.
POA is an independent pvlib Perez transposition of the same weather, computed
once per location and shared across variants to normalize the PR comparison.

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

import breos  # noqa: E402
from validation.common import (  # noqa: E402
    BASELINE_PATH,
    MODEL_CONFIGS,
    RESULTS_DIR,
    RESULTS_PATH,
    compute_annual_poa,
    compute_model_result,
    find_weather_file,
    load_spec,
    load_validation_weather,
)


def run_location(key: str, loc: dict, system: dict, weather_file: Path) -> dict:
    weather = load_validation_weather(weather_file)
    annual_poa = compute_annual_poa(weather, loc, system)
    models = {}
    for name, model_kwargs in MODEL_CONFIGS.items():
        result = compute_model_result(weather, loc, system, model_kwargs, annual_poa)
        annual_kwh = round(result["annual_kwh"], 3)
        models[name] = {
            **result,
            "monthly_kwh": [round(v, 3) for v in result["monthly_kwh"]],
            "annual_kwh": annual_kwh,
        }
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
