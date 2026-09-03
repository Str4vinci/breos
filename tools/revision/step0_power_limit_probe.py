#!/usr/bin/env python3
"""Step 0: is the battery power limit constant, and how much does it bias the front?

Emits two tables:
  * the realised rainflow C-rate the cycle-ageing term consumes, per capacity;
  * the effect of removing the constant charge cap entirely.
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

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

_spec = importlib.util.spec_from_file_location("repro", PROJECT_ROOT / "tools/reproduce_article1.py")
repro = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(repro)

import breos  # noqa: E402
import breos.battery as B  # noqa: E402
from breos.constants import A_Q, B_Q  # noqa: E402
from breos.optimization import evaluate_projected_design  # noqa: E402

GI = "Projected_Grid_Independence_%"
NPV = "Projected_NPV_Eur"
CAPACITIES = (2.0, 5.0, 9.0, 13.0, 20.0, 30.0)
DESIGN = {"n_modules": 9, "tilt": 35.0, "azimuth": 190.0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=repro.DEFAULT_CONFIG)
    ap.add_argument("--rlp-directory", type=Path, required=True)
    ap.add_argument("--execution-backend", default="numba")
    ap.add_argument("--years", type=int, default=1)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    config_bytes = args.config.read_bytes()
    base = tomllib.loads(config_bytes.decode("utf-8"))
    weather, load, weather_path, rlp_path = repro._load_inputs(base, args.rlp_directory)
    cfg = copy.deepcopy(base)
    cfg["simulation"]["years_projection"] = args.years
    cap_w = float(base["battery"]["max_charge_power_w"])

    original = B._detect_cycles_rainflow_arrays
    buffer: list[dict] = []

    def spy(*a, **k):
        out = original(*a, **k)
        buffer.extend(out)
        return out

    crate_rows, cap_rows = [], []
    for kwh in CAPACITIES:
        B._detect_cycles_rainflow_arrays = spy
        buffer.clear()
        capped = evaluate_projected_design(
            weather, load, cfg, battery_kwh=kwh, execution_backend=args.execution_backend, **DESIGN
        )
        B._detect_cycles_rainflow_arrays = original

        c = np.array([x["mean_c_rate"] for x in buffer])
        w = np.array([x["doc"] * x.get("count", 1.0) for x in buffer])
        crate_rows.append(
            {
                "Battery_kWh": kwh,
                "Nominal_C_At_Cap": cap_w / 1000.0 / kwh,
                "Detected_Cycles": len(buffer),
                "FEC_Weighted_Mean_C_Rate": float((c * w).sum() / w.sum()),
                "C_Rate_p95": float(np.percentile(c, 95)),
                "C_Rate_Max": float(c.max()),
                "FEC_Weighted_kC": float((np.clip(A_Q * c + B_Q, 0, None) * w).sum() / w.sum()),
                "Year1_SOH_%": capped.metrics["Projected_Final_SOH_%"],
            }
        )

        uncapped_cfg = copy.deepcopy(cfg)
        uncapped_cfg["battery"].pop("max_charge_power_w", None)
        uncapped = evaluate_projected_design(
            weather, load, uncapped_cfg, battery_kwh=kwh, execution_backend=args.execution_backend, **DESIGN
        )
        cap_rows.append(
            {
                "Battery_kWh": kwh,
                "GI_With_Constant_Cap_%": capped.metrics[GI],
                "GI_Uncapped_%": uncapped.metrics[GI],
                "GI_Delta_pp": uncapped.metrics[GI] - capped.metrics[GI],
                "NPV_With_Constant_Cap_Eur": capped.metrics[NPV],
                "NPV_Uncapped_Eur": uncapped.metrics[NPV],
                "NPV_Delta_Eur": uncapped.metrics[NPV] - capped.metrics[NPV],
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    crate = pd.DataFrame(crate_rows)
    binding = pd.DataFrame(cap_rows)
    crate_path = args.output / "step0_realised_c_rate.csv"
    bind_path = args.output / "step0_charge_cap_binding.csv"
    crate.to_csv(crate_path, index=False)
    binding.to_csv(bind_path, index=False)
    print(crate.to_string(index=False))
    print()
    print(binding.to_string(index=False))

    provenance = {
        "breos_version": breos.__version__,
        "breos_source": repro._git_revision(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "config": str(args.config.relative_to(PROJECT_ROOT)),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "config_diff_vs_base": {
            "simulation.years_projection": args.years,
            "battery.max_charge_power_w": f"{cap_w} (kept) vs removed (uncapped arm)",
        },
        "design": DESIGN,
        "constant_charge_cap_w": cap_w,
        "max_discharge_power_w_configured": base["battery"].get("max_discharge_power_w", None),
        "cycle_ageing_coefficients": {"A_Q": A_Q, "B_Q": B_Q},
        "weather": str(weather_path.relative_to(PROJECT_ROOT)),
        "weather_uncompressed_sha256": repro._sha256(weather_path, decompress_gzip=weather_path.suffix == ".gz"),
        "external_rlp_sha256": repro._sha256(rlp_path) if rlp_path else None,
        "execution_backend": args.execution_backend,
        "outputs": {crate_path.name: repro._sha256(crate_path), bind_path.name: repro._sha256(bind_path)},
    }
    (args.output / "step0_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"\nWrote {crate_path}\nWrote {bind_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
