#!/usr/bin/env python3
"""Task 6(b,c): replay C2, C3 and C4 across the bootstrap parameter distribution.

Stage one enumerated the complete bootstrap distribution of the v1 calibration
as 126 multinomially weighted parameter sets. This replays the three reference
candidates under every one of them and reports the weighted distribution of the
replacement year and of NPV, which is what reviewer question 6(c) asks for and
what 6(d) needs in order to say whether the field-versus-laboratory design
difference is separable.

The four calendar parameters reach the dispatch through exactly one function,
``breos.battery._get_degradation_params``, which maps a model name onto the
tuple the native degradation adapter consumes. Substituting that function is
therefore enough to run a whole design under a bootstrap parameter set without
inventing a configuration surface for parameters that are not user-facing.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import shlex
import sys
import tomllib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

_spec = importlib.util.spec_from_file_location("repro", PROJECT_ROOT / "tools/reproduce_article1.py")
repro = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(repro)

import breos  # noqa: E402
import breos.battery as battery_module  # noqa: E402
from breos.optimization import evaluate_projected_design  # noqa: E402

GI = "Projected_Grid_Independence_%"
NPV = "Projected_NPV_Eur"
PARAM_NAMES = ("k0_frac", "Ea", "cal_b", "n")
CANDIDATES = ("C2", "C3", "C4")

_STATE: dict = {}


def _init(config, rlp_directory, backend):
    weather, load, _, _ = repro._load_inputs(config, Path(rlp_directory))
    _STATE.update(config=config, weather=weather, load=load, backend=backend)
    _STATE["original_params"] = battery_module._get_degradation_params


def _run(task):
    label, design, params = task
    # _get_degradation_params returns (k0_frac, Ea, time_exponent, soc_exponent),
    # which is the order the v1 branch returns its four constants in, so cal_b
    # is the time exponent and n the SOC exponent.
    tup = (params["k0_frac"], params["Ea"], params["cal_b"], params["n"])
    battery_module._get_degradation_params = lambda _model, _t=tup: _t
    try:
        r = evaluate_projected_design(
            _STATE["weather"],
            _STATE["load"],
            _STATE["config"],
            execution_backend=_STATE["backend"],
            **design,
        )
    finally:
        battery_module._get_degradation_params = _STATE["original_params"]

    rep = r.yearly.index[r.yearly["Replacements"] > 0]
    return {
        "multiset_id": params["multiset_id"],
        "weight": params["weight"],
        "Candidate": label,
        **{k: v for k, v in design.items()},
        **{name: params[name] for name in PARAM_NAMES},
        "Replacement_Year": None if len(rep) == 0 else int(r.yearly.loc[rep[0], "Year"]),
        "Total_Replacements": int(r.metrics["Projected_Total_Replacements"]),
        "Final_SOH_%": float(r.metrics["Projected_Final_SOH_%"]),
        GI: float(r.metrics[GI]),
        NPV: float(r.metrics[NPV]),
    }


def weighted_quantile(values, weights, q: float) -> float:
    order = sorted(range(len(values)), key=lambda i: values[i])
    cumulative, total = 0.0, sum(weights)
    for i in order:
        cumulative += weights[i] / total
        if cumulative >= q:
            return float(values[i])
    return float(values[order[-1]])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=repro.DEFAULT_CONFIG)
    ap.add_argument("--bootstrap-params", type=Path, required=True)
    ap.add_argument("--rlp-directory", type=Path, required=True)
    ap.add_argument("--n-procs", type=int, default=16)
    ap.add_argument("--execution-backend", default="numba")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    config_bytes = args.config.read_bytes()
    config = tomllib.loads(config_bytes.decode("utf-8"))
    model = str(config.get("battery", {}).get("calendar_model", ""))
    if "field_calibrated" not in model:
        raise SystemExit(f"expected a field-calibrated v1 config, got calendar_model={model!r}")

    # The candidates are read from the config rather than restated here, so the
    # replayed geometry is the manuscript's by construction.
    by_label = {c["label"]: c for c in config["reference_candidates"]}
    designs = {
        label: dict(
            n_modules=int(by_label[label]["modules"]),
            battery_kwh=float(by_label[label]["battery_kwh"]),
            tilt=float(by_label[label]["tilt"]),
            azimuth=float(by_label[label]["azimuth"]),
        )
        for label in CANDIDATES
    }
    for label, d in designs.items():
        print(f"{label}: {d}")

    boot = pd.read_csv(args.bootstrap_params)
    tasks = [(label, designs[label], row) for row in boot.to_dict("records") for label in CANDIDATES]
    print(f"{len(boot)} parameter sets x {len(CANDIDATES)} candidates = {len(tasks)} replays")

    with ProcessPoolExecutor(
        max_workers=args.n_procs,
        initializer=_init,
        initargs=(config, str(args.rlp_directory), args.execution_backend),
    ) as pool:
        rows = list(pool.map(_run, tasks, chunksize=4))

    table = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    replay_path = args.output / "task6_candidate_replays.csv"
    table.to_csv(replay_path, index=False)

    summary_rows = []
    for label in CANDIDATES:
        sub = table[table["Candidate"] == label]
        w = sub["weight"].tolist()
        years = sub["Replacement_Year"].fillna(-1).tolist()
        npvs = sub[NPV].tolist()
        # Replacement year is discrete, so its distribution is reported as the
        # weight on each observed year rather than as an interval.
        year_weight = sub.groupby(sub["Replacement_Year"].fillna(-1))["weight"].sum()
        summary_rows.append(
            {
                "Candidate": label,
                "Battery_kWh": sub["battery_kwh"].iloc[0],
                "Repl_Year_Modal": int(year_weight.idxmax()),
                "Repl_Year_Modal_Weight": float(year_weight.max()),
                "Repl_Year_Distinct": int(sub["Replacement_Year"].nunique(dropna=False)),
                "Repl_Year_Min": int(min(years)),
                "Repl_Year_Max": int(max(years)),
                "Repl_Year_p2.5": weighted_quantile(years, w, 0.025),
                "Repl_Year_p97.5": weighted_quantile(years, w, 0.975),
                "NPV_Median": weighted_quantile(npvs, w, 0.50),
                "NPV_p2.5": weighted_quantile(npvs, w, 0.025),
                "NPV_p97.5": weighted_quantile(npvs, w, 0.975),
                "NPV_Range_Eur": max(npvs) - min(npvs),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary_path = args.output / "task6_candidate_distribution.csv"
    summary.to_csv(summary_path, index=False)
    print()
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    provenance = {
        "schema": "breos-task6-replay-v1",
        "breos_version": breos.__version__,
        "breos_source": repro._git_revision(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "config": str(args.config.relative_to(PROJECT_ROOT)),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "bootstrap_params": str(args.bootstrap_params),
        "bootstrap_params_sha256": repro._sha256(args.bootstrap_params),
        "parameter_injection": "breos.battery._get_degradation_params substituted per replay",
        "candidates": designs,
        "n_parameter_sets": int(len(boot)),
        "n_replays": len(tasks),
        "execution_backend": args.execution_backend,
        "outputs": {
            replay_path.name: repro._sha256(replay_path),
            summary_path.name: repro._sha256(summary_path),
        },
    }
    (args.output / "task6_replay_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"\nWrote {replay_path}\nWrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
