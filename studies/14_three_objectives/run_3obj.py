"""Study 14: would BREOS's NSGA-II setup still find the front with a third objective?

The shipped optimizer has two objectives (grid dependence, -NPV). This replays the same
NSGA-II (operators, repair, termination from breos.optimization) on exact lookup tables
(enumerate_space.py) with a third objective added, and compares what it returns with the
exact 3D front. Third objectives: -CO2 avoided over the project (an emissions-minded user)
and initial cost (a budget-minded user). The two-objective problem runs on the same seeds
and budgets as the control.

Fronts scored: the final population's non-dominated set (what 0.8.1 returns by default) and
the front of every evaluated design (the opt-in pareto_front = "all_evaluated").
Metric: hypervolume ratio against the exact front, normalised to the exact front's ideal and
nadir with a 10 % margin; IGD+ in the same space.

Usage: uv run python run_3obj.py SPACE SCENARIO[,SCENARIO] [--seeds 1-50] [--workers N]
Output: data/runs/<space>/<scenario>/replay_3obj.pkl
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time

import numpy as np
import pandas as pd

from common import RUNS, build_problem, design_key
from metrics import igd_plus, pareto_mask
from nsga_runs import BUDGETS, load_exact, parse_seeds

OBJECTIVE_SETS = {
    "2obj": ["f0", "f1"],
    "3obj_co2": ["f0", "f1", "f_co2"],
    "3obj_capex": ["f0", "f1", "f_capex"],
}


def with_objectives(exact: pd.DataFrame) -> pd.DataFrame:
    exact = exact.copy()
    exact["f_co2"] = -exact["Projected_CO2_Avoided_Total_kg"]
    exact["f_capex"] = exact["Projected_Initial_Cost"]
    return exact


def lookup_problem(real, exact: pd.DataFrame, fcols: list[str]):
    from pymoo.core.problem import Problem

    xcols = [c for c in exact.columns if c.startswith("x")]
    table = {design_key(r): i for i, r in enumerate(exact[xcols].to_numpy())}
    f = exact[fcols].to_numpy(dtype=float).copy()
    f[~exact["feasible"].to_numpy()] = 1e12
    g = exact[["g0", "g1"]].to_numpy(dtype=float)

    class LookupProblem(Problem):
        def __init__(self):
            super().__init__(n_var=real.n_var, n_obj=len(fcols), n_ieq_constr=real.n_ieq_constr, xl=real.xl, xu=real.xu)
            self.gene_steps = real.gene_steps
            self.layout_gene = real.layout_gene
            self.layouts = real.layouts
            self.single_orientation_genes = real.single_orientation_genes
            self.log: list[int] = []

        def _evaluate(self, x, out, *args, **kwargs):
            idx = np.array([table[design_key(row)] for row in x])
            self.log.extend(idx.tolist())
            out["F"] = f[idx]
            out["G"] = g[idx]

    return LookupProblem(), table


def run(real, exact, fcols, budget, seed):
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.optimize import minimize

    from breos.optimization import DiscreteGridRepair, _build_multi_objective_termination

    pop, n_off, n_gen, early_stop = BUDGETS[budget]
    problem, table = lookup_problem(real, exact, fcols)
    algorithm = NSGA2(
        pop_size=pop, sampling=FloatRandomSampling(), crossover=SBX(prob=0.9, eta=15), mutation=PM(eta=20),
        repair=DiscreteGridRepair(), eliminate_duplicates=True, n_offsprings=n_off,
    )
    termination, _ = _build_multi_objective_termination(n_gen, early_stop)
    res = minimize(problem, algorithm, termination, seed=seed, verbose=False)
    final = np.array([table[design_key(x)] for x in np.atleast_2d(res.X)], dtype=int) if res.X is not None else np.array([], int)
    return final, np.unique(problem.log), int(res.algorithm.n_gen) - 1, int(res.algorithm.evaluator.n_eval)


def front_rows(exact, fcols, idx):
    idx = np.asarray([i for i in idx if exact["feasible"].iat[i]], dtype=int)
    if len(idx) == 0:
        return idx
    F = exact[fcols].to_numpy()[idx]
    return idx[pareto_mask(F)]


def hv(points: np.ndarray, ref: np.ndarray) -> float:
    from pymoo.indicators.hv import HV

    return float(HV(ref_point=ref)(points)) if len(points) else 0.0


_S: dict = {}


def _init(space, scenario):
    _S["real"] = build_problem(scenario, space)
    exact = with_objectives(load_exact(scenario, space))
    _S["exact"] = exact
    _S["norm"] = {}
    for name, fcols in OBJECTIVE_SETS.items():
        feas = np.flatnonzero(exact["feasible"].to_numpy())
        ef = exact[fcols].to_numpy()[front_rows(exact, fcols, feas)]
        lo, hi = ef.min(0), ef.max(0)
        span = np.where(hi > lo, hi - lo, 1.0)
        ideal, nadir = lo - 0.1 * span, hi + 0.1 * span
        tr = lambda F, ideal=ideal, nadir=nadir: (F - ideal) / (nadir - ideal)  # noqa: E731
        _S["norm"][name] = (tr, tr(ef), hv(tr(ef), np.ones(len(fcols))))


def _job(args):
    space, scenario, name, budget, seed = args
    exact, fcols = _S["exact"], OBJECTIVE_SETS[name]
    tr, exact_front, exact_hv = _S["norm"][name]
    t = time.perf_counter()
    final, evaluated, gens, n_eval = run(_S["real"], exact, fcols, budget, seed)
    out = {"space": space, "scenario": scenario, "objectives": name, "budget": budget, "seed": seed,
           "generations": gens, "n_eval": n_eval, "unique_evaluated": len(evaluated),
           "exact_front_size": len(exact_front), "s": time.perf_counter() - t}
    ref = np.ones(len(fcols))
    for label, idx in (("final", final), ("all_evaluated", evaluated)):
        rows = front_rows(exact, fcols, idx)
        F = tr(exact[fcols].to_numpy()[rows]) if len(rows) else np.empty((0, len(fcols)))
        F = F[np.all(F <= ref, axis=1)] if len(F) else F
        out[f"hv_ratio_{label}"] = hv(F, ref) / exact_hv
        out[f"igd_plus_{label}"] = igd_plus(exact_front, F) if len(F) else np.nan
        out[f"front_size_{label}"] = len(rows)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("space")
    ap.add_argument("scenarios")
    ap.add_argument("--seeds", default="1-50")
    ap.add_argument("--budgets", default="prod,prod_noes,long,big")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    for scenario in a.scenarios.split(","):
        path = RUNS / a.space / scenario / "replay_3obj.pkl"
        if path.exists():
            print(f"skip {path}")
            continue
        if not (RUNS / a.space / scenario / "exact.pkl").exists():
            # A Pool whose initializer raises respawns workers forever; fail fast instead.
            print(f"no exact table for {a.space}/{scenario}; skipped", flush=True)
            continue
        t = time.time()
        jobs = [(a.space, scenario, n, b, s) for n in OBJECTIVE_SETS for b in a.budgets.split(",") for s in parse_seeds(a.seeds)]
        with mp.Pool(a.workers, initializer=_init, initargs=(a.space, scenario)) as pool:
            rows = list(pool.imap_unordered(_job, jobs, chunksize=2))
        df = pd.DataFrame(rows)
        df.to_pickle(path)
        print(f"{time.strftime('%H:%M:%S')} {a.space}/{scenario}: {len(jobs)} runs in {time.time() - t:.0f}s", flush=True)
        print(df.groupby(["objectives", "budget"])[["hv_ratio_final", "hv_ratio_all_evaluated", "n_eval"]].median().round(4), flush=True)


if __name__ == "__main__":
    main()
