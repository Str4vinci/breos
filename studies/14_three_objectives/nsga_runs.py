"""Production NSGA-II on a scenario, scored from the exact enumeration table.

The algorithm is assembled exactly as breos.optimization.
optimize_system_multi_objective assembles it (same sampling, SBX, PM,
DiscreteGridRepair, duplicate elimination, termination). Only the evaluation
differs: each candidate's F and G are read from runs/<space>/<scenario>/exact.pkl
instead of re-simulated. `validate_lookup.py` checks this is bit-identical to
the real optimizer for the same seed.

Usage: python nsga_runs.py SCENARIO SPACE BUDGET --seeds 1-100 [--workers 12]
Output: runs/<space>/<scenario>/nsga_<budget>.pkl
"""

from __future__ import annotations

import argparse
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from common import RUNS, build_problem, design_key

# name -> (pop_size, n_offsprings, n_gen, early_stop)
PROD_EARLY_STOP = {"ftol": 0.0025, "period": 10, "min_gen": 20, "n_skip": 0}
BUDGETS = {
    "prod": (40, 20, 40, PROD_EARLY_STOP),  # the shipped config, as users run it
    "prod_noes": (40, 20, 40, False),
    "long": (40, 20, 200, False),
    "big": (100, 50, 200, False),
}


def load_exact(scenario: str, space: str) -> pd.DataFrame:
    return pd.read_pickle(RUNS / space / scenario / "exact.pkl")


def make_lookup_problem(real_problem, exact: pd.DataFrame):
    from pymoo.core.problem import Problem

    xcols = [c for c in exact.columns if c.startswith("x")]
    keys = [design_key(r) for r in exact[xcols].to_numpy()]
    # Infeasible designs carry NaN objectives in the table; NSGA-II ranks them
    # by constraint violation only, so any finite placeholder is never read.
    f = exact[["f0", "f1"]].to_numpy(dtype=float).copy()
    f[~exact["feasible"].to_numpy()] = 1e12
    g = exact[["g0", "g1"]].to_numpy(dtype=float)
    table = {k: i for i, k in enumerate(keys)}

    class LookupProblem(Problem):
        def __init__(self):
            super().__init__(n_var=real_problem.n_var, n_obj=2, n_ieq_constr=real_problem.n_ieq_constr,
                             xl=real_problem.xl, xu=real_problem.xu)
            # Attributes DiscreteGridRepair reads.
            self.gene_steps = real_problem.gene_steps
            self.layout_gene = real_problem.layout_gene
            self.layouts = real_problem.layouts
            self.single_orientation_genes = real_problem.single_orientation_genes
            self.log: list[int] = []

        def _evaluate(self, x, out, *args, **kwargs):
            idx = np.array([table[design_key(row)] for row in x])
            self.log.extend(idx.tolist())
            out["F"] = f[idx]
            out["G"] = g[idx]

    return LookupProblem()


def run_one(real_problem, exact: pd.DataFrame, budget: str, seed: int, mutation_eta: float = 20,
            sampling: str = "random") -> dict:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.callback import Callback
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.lhs import LHS
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.optimize import minimize

    from breos.optimization import DiscreteGridRepair, _build_multi_objective_termination

    pop_size, n_off, n_gen, early_stop = BUDGETS[budget]
    problem = make_lookup_problem(real_problem, exact)
    trace: list[tuple[int, int, np.ndarray]] = []  # (gen, n_eval, population-opt row indices)

    class Trace(Callback):
        def notify(self, algorithm):
            opt_idx = [table_index(ind.X) for ind in algorithm.opt if ind.feasible[0]]
            trace.append((algorithm.n_gen, algorithm.evaluator.n_eval, np.array(opt_idx, dtype=int)))

    xcols = [c for c in exact.columns if c.startswith("x")]
    index = {design_key(r): i for i, r in enumerate(exact[xcols].to_numpy())}

    def table_index(x):
        return index[design_key(x)]

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling={"random": FloatRandomSampling, "lhs": LHS}[sampling](),  # random is production
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=mutation_eta),  # 20 is the production value
        repair=DiscreteGridRepair(),
        eliminate_duplicates=True,
        n_offsprings=n_off,
    )
    termination, _ = _build_multi_objective_termination(n_gen, early_stop)
    t0 = time.perf_counter()
    res = minimize(problem, algorithm, termination, seed=seed, callback=Trace(), verbose=False)
    final_idx = np.array([table_index(x) for x in np.atleast_2d(res.X)], dtype=int) if res.X is not None else np.array([], int)
    return {
        "seed": seed,
        "budget": budget,
        "final": final_idx,  # what the optimizer returns (result.opt)
        "evaluated": np.array(problem.log, dtype=int),  # every evaluation, in order
        "trace": trace,
        "generations": max(0, int(res.algorithm.n_gen) - 1),
        "n_eval": int(res.algorithm.evaluator.n_eval),
        "search_s": time.perf_counter() - t0,
    }


_STATE: dict = {}


def _init(scenario, space):
    _STATE["problem"] = build_problem(scenario, space)
    _STATE["exact"] = load_exact(scenario, space)


def _job(args):
    budget, seed = args
    return run_one(_STATE["problem"], _STATE["exact"], budget, seed)


def parse_seeds(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("space")
    ap.add_argument("budgets", nargs="+", choices=list(BUDGETS))
    ap.add_argument("--seeds", default="1-100")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)
    with Pool(args.workers, initializer=_init, initargs=(args.scenario, args.space), maxtasksperchild=4) as pool:  # bounds a ~300 MB-per-seed leak
        for budget in args.budgets:
            t0 = time.perf_counter()
            runs = pool.map(_job, [(budget, s) for s in seeds], chunksize=1)
            path = RUNS / args.space / args.scenario / f"nsga_{budget}.pkl"
            pd.to_pickle(runs, path)
            print(f"{args.scenario}/{args.space}/{budget}: {len(runs)} seeds in {time.perf_counter() - t0:.0f}s -> {path}")


if __name__ == "__main__":
    main()
