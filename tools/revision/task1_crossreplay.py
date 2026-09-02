#!/usr/bin/env python3
"""Task 1(d): replay each degradation model's max-NPV design under both models.

Varies only ``battery.calendar_model``; the design variables (modules, battery,
tilt, azimuth) stay bound to the front that produced them.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import platform
import shlex
import sys
import tomllib
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

_spec = importlib.util.spec_from_file_location("repro", PROJECT_ROOT / "tools/reproduce_article1.py")
repro = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(repro)

import breos  # noqa: E402
from breos.optimization import evaluate_projected_design  # noqa: E402

MODELS = {
    "v1": "naumann_lam_field_calibrated_v1",
    "v2": "naumann_lam_field_calibrated_v2",
    "lab": "naumann_lam",
}
FRONTS = (("v1", "base-v1"), ("v2", "field-v2"), ("lab", "laboratory"))


def _replacement_year(yearly: pd.DataFrame) -> int | None:
    hit = yearly.index[yearly["Replacements"] > 0]
    return None if len(hit) == 0 else int(yearly.loc[hit[0], "Year"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=repro.DEFAULT_CONFIG)
    ap.add_argument("--rlp-directory", type=Path, required=True)
    ap.add_argument("--front-root", type=Path, required=True)
    ap.add_argument("--execution-backend", default="numba")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    config_bytes = args.config.read_bytes()
    base = tomllib.loads(config_bytes.decode("utf-8"))
    weather, load, weather_path, rlp_path = repro._load_inputs(base, args.rlp_directory)

    designs: dict[str, dict] = {}
    for tag, slug in FRONTS:
        reps = pd.read_csv(args.front_root / slug / "pareto_representatives.csv")
        row = reps[reps["Representative"] == "max_npv"].iloc[0]
        designs[tag] = {
            "modules": int(row["Modules"]),
            "battery_kwh": float(row["Battery_kWh"]),
            "tilt": float(row["Tilt"]),
            "azimuth": float(row["Azimuth"]),
        }

    rows = []
    for design_tag, design in designs.items():
        for model_tag, model_name in MODELS.items():
            cfg = copy.deepcopy(base)
            cfg["battery"]["calendar_model"] = model_name
            result = evaluate_projected_design(
                weather,
                load,
                cfg,
                n_modules=design["modules"],
                battery_kwh=design["battery_kwh"],
                tilt=design["tilt"],
                azimuth=design["azimuth"],
                execution_backend=args.execution_backend,
            )
            m = result.metrics
            rows.append(
                {
                    "Design_From": design_tag,
                    "Replayed_Under": model_tag,
                    "Calendar_Model": model_name,
                    "Modules": design["modules"],
                    "Battery_kWh": design["battery_kwh"],
                    "Tilt": design["tilt"],
                    "Azimuth": design["azimuth"],
                    "Projected_Grid_Independence_%": m["Projected_Grid_Independence_%"],
                    "Projected_NPV_Eur": m["Projected_NPV_Eur"],
                    "Replacement_Year": _replacement_year(result.yearly),
                    "Projected_Total_Replacements": m["Projected_Total_Replacements"],
                    "Projected_Replacement_Cost_Eur": m["Projected_Replacement_Cost_Eur"],
                    "Projected_Breakeven_Year_Exact": m.get("Projected_Breakeven_Year_Exact"),
                    "Projected_Final_SOH_%": m["Projected_Final_SOH_%"],
                }
            )

    args.output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    csv_path = args.output / "task1d_cross_replay.csv"
    table.to_csv(csv_path, index=False)
    print(table.to_string(index=False))

    provenance = {
        "breos_version": breos.__version__,
        "breos_source": repro._git_revision(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "config": str(args.config.relative_to(PROJECT_ROOT)),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "config_diff_vs_base": {"battery.calendar_model": sorted(MODELS.values())},
        "weather": str(weather_path.relative_to(PROJECT_ROOT)),
        "weather_file_sha256": repro._sha256(weather_path),
        "weather_uncompressed_sha256": repro._sha256(weather_path, decompress_gzip=weather_path.suffix == ".gz"),
        "external_rlp_filename": rlp_path.name if rlp_path else None,
        "external_rlp_sha256": repro._sha256(rlp_path) if rlp_path else None,
        "execution_backend": args.execution_backend,
        "designs": designs,
        "output_csv": csv_path.name,
        "output_csv_sha256": repro._sha256(csv_path),
    }
    (args.output / "task1d_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"\nWrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
