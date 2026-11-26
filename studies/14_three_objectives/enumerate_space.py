"""Exactly enumerate a scenario's discrete design space.

Feasible designs (area and budget) run the production SolarDesignProblem
evaluation. Infeasible designs get their exact G from CAPEX and area; their
objectives are never read by NSGA-II's feasibility-first ranking, so they are
stored as NaN.

Usage: python enumerate_space.py SCENARIO SPACE [--workers N]
Output: runs/<space>/<scenario>/exact.pkl + manifest.json
"""

from __future__ import annotations

import argparse
import os
import time

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_var] = "1"

from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import (  # noqa: E402
    BASE_CONFIG,
    RUNS,
    weather_file,
    build_problem,
    capex_and_area,
    dump_json,
    evaluate,
    full_grid,
    scenario_config,
    sha256,
)

_PROBLEM = None
KEEP = (
    "Projected_NPV",
    "Projected_Grid_Independence_%",
    "Projected_Initial_Cost",
    "Projected_Total_Replacements",
    "Projected_ZEB_Ratio",
    "Projected_CO2_Avoided_Total_kg",
    "Projected_CO2_Avoided_SelfConsumed_kg",
    "Projected_LCOE_per_kWh",
    "Projected_Payback_Year_Interpolated",
)


def _init(scenario: str, space: str) -> None:
    global _PROBLEM
    _PROBLEM = build_problem(scenario, space)


def _work(item: tuple[int, np.ndarray]) -> tuple[int, np.ndarray, list[dict], float]:
    i, chunk = item
    t0 = time.perf_counter()
    rows = []
    for x in chunk:
        out = evaluate(_PROBLEM, x)
        row = {"f0": out["F"][0], "f1": out["F"][1], "g0": out["G"][0], "g1": out["G"][1]}
        row.update({k: out[k] for k in KEEP})
        rows.append(row)
    return i, chunk, rows, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("space")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=25)
    args = ap.parse_args()

    out_dir = RUNS / args.space / args.scenario
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / "exact.pkl"
    if final.exists():
        print(f"{final} exists; skipping")
        return
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(exist_ok=True)

    problem = build_problem(args.scenario, args.space)
    X = full_grid(problem)
    capex, area = capex_and_area(problem, X[:, 0], X[:, 1])
    g0 = capex - problem.budget_limit
    g1 = area - problem.area_limit
    feasible = (g0 <= 0) & (g1 <= 0)
    todo = X[feasible]
    chunks = [todo[i : i + args.chunk] for i in range(0, len(todo), args.chunk)]
    done = {int(p.stem.split("_")[1]) for p in parts_dir.glob("part_*.pkl")}
    pending = [(i, c) for i, c in enumerate(chunks) if i not in done]
    print(f"{args.scenario}/{args.space}: grid {len(X)}, feasible {len(todo)}, chunks {len(chunks)}, pending {len(pending)}")

    t0 = time.perf_counter()
    cpu = 0.0
    n_done = 0
    with Pool(args.workers, initializer=_init, initargs=(args.scenario, args.space)) as pool:
        for i, chunk, rows, dt in pool.imap_unordered(_work, pending):
            frame = pd.DataFrame(chunk, columns=[f"x{k}" for k in range(chunk.shape[1])])
            frame = pd.concat([frame, pd.DataFrame(rows)], axis=1)
            frame.to_pickle(parts_dir / f"part_{i:06d}.pkl")
            cpu += dt
            n_done += len(chunk)
            if n_done % (args.chunk * args.workers * 4) < args.chunk:
                el = time.perf_counter() - t0
                rate = n_done / el
                left = (len(todo) - len(done) * args.chunk - n_done) / rate
                print(f"  {n_done} designs, {el:.0f}s, {rate:.1f}/s, ~{left / 60:.0f} min left", flush=True)
    wall = time.perf_counter() - t0

    parts = pd.concat([pd.read_pickle(p) for p in sorted(parts_dir.glob("part_*.pkl"))], ignore_index=True)
    xcols = [f"x{k}" for k in range(X.shape[1])]
    grid = pd.DataFrame(X, columns=xcols)
    grid["g0_capex"] = g0
    grid["g1_area"] = g1
    grid["feasible"] = feasible
    merged = grid.merge(parts, on=xcols, how="left", validate="one_to_one")
    assert merged.loc[feasible, "f0"].notna().all(), "missing feasible evaluations"
    # The simulated G must equal the analytic G (sanity on the CAPEX shortcut).
    sim = merged.loc[feasible]
    assert np.allclose(sim["g0"], sim["g0_capex"], rtol=0, atol=1e-6), "CAPEX shortcut mismatch"
    assert np.allclose(sim["g1"], sim["g1_area"], rtol=0, atol=1e-9), "area shortcut mismatch"
    merged["g0"] = merged["g0"].fillna(merged["g0_capex"])
    merged["g1"] = merged["g1"].fillna(merged["g1_area"])
    merged = merged.drop(columns=["g0_capex", "g1_area"])
    merged.to_pickle(final)

    mpath = out_dir / "manifest.json"
    dump_json(
        mpath,
        {
            "scenario": args.scenario,
            "space": args.space,
            "config": scenario_config(args.scenario, args.space),
            "base_config_sha256": sha256(BASE_CONFIG),
            "weather_sha256": sha256(weather_file(args.space)),
            "grid_size": int(len(X)),
            "feasible": int(feasible.sum()),
            "workers": args.workers,
            "wall_s_this_session": wall,
            "cpu_s_this_session": cpu,
            "designs_this_session": n_done,
            "mean_eval_s": cpu / n_done if n_done else None,
            "breos_git": os.popen("git -C /home/leonardo/pythonprojects/breos rev-parse HEAD").read().strip(),
        },
    )
    print(f"done: wall {wall:.0f}s, cpu/eval {cpu / max(n_done, 1):.3f}s -> {final}")


if __name__ == "__main__":
    main()
