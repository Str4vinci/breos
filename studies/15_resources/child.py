"""One measured run in a fresh process. argv[1] = JSON spec; prints one JSON line.

Phases timed: import, build (App/Community construction), simulate, result (+ timeseries for App).
Peak memory = ru_maxrss of this process (kB on Linux), so each spec runs in its own process.
"""

import json
import resource
import sys
import time
import warnings

warnings.simplefilter("ignore")
spec = json.loads(sys.argv[1])
out = {"id": spec["id"]}
t = time.perf_counter()
import breos  # noqa: E402,F401

out["import_s"] = time.perf_counter() - t
cfg = spec["config"]
try:
    if spec["kind"] == "app":
        t = time.perf_counter()
        app = breos.App(cfg)
        out["build_s"] = time.perf_counter() - t
        t = time.perf_counter()
        app.simulate()
        out["simulate_s"] = time.perf_counter() - t
        t = time.perf_counter()
        app.result()
        ts = app.timeseries()
        out["result_s"] = time.perf_counter() - t
        out["timeseries_rows"] = len(ts)
        out["timeseries_mb"] = float(ts.memory_usage(deep=True).sum() / 2**20)
    elif spec["kind"] == "community":
        from breos.community import Community

        t = time.perf_counter()
        c = Community(cfg)
        out["build_s"] = time.perf_counter() - t
        t = time.perf_counter()
        c.simulate()
        out["simulate_s"] = time.perf_counter() - t
    elif spec["kind"] == "montecarlo":
        from breos.montecarlo import MonteCarloSettings, run_montecarlo

        t = time.perf_counter()
        run_montecarlo(cfg, MonteCarloSettings(**spec["settings"]))
        out["simulate_s"] = time.perf_counter() - t
    out["ok"] = True
except Exception as exc:  # recorded, not raised: a failing config is a finding
    out["ok"] = False
    out["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
out["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(json.dumps(out))
