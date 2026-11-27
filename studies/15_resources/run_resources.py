"""Study 15: peak memory and wall time across the config space.

Specs: every App example in configs/examples (and the community example) at hourly and 15-min,
over 1, 10 and 25 projection years; Monte Carlo with 10/100/1000 runs; module-count scaling
(10/100/1000 modules, one array) at both resolutions over 1 and 20 years. Weather: the local
PVGIS TMY of the preset location where long-run data has one, otherwise BREOS's own fetch
(cached under data/). Each spec runs in a fresh process (child.py) with WORKERS in parallel,
so wall times include some contention; peak memory is per process.

Usage: uv run python run_resources.py [--workers 3] -> data/results.jsonl (resumable)
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _shared import sizing as S  # noqa: E402

EX = S.REPO / "configs" / "examples"
DATA = HERE / "data"
SKIP = {"sweep.toml", "tariff-comparison.toml", "montecarlo.toml", "community-portugal.toml"}
TMY = {"porto": "porto", "lisbon": "lisbon", "berlin": "berlin", "madrid": "madrid"}
TIMEOUT_S = 3600


def _with_weather(cfg: dict) -> dict:
    loc = cfg.get("location")
    if isinstance(loc, str) and loc in TMY and "weather_file" not in cfg:
        cfg["weather_file"] = str(S.DATA / f"{TMY[loc]}_tmy_pvgis.csv")
    return cfg


def specs() -> list[dict]:
    out = []
    for path in sorted(EX.glob("*.toml")):
        if path.name in SKIP:
            continue
        base = tomllib.loads(path.read_text())
        for res in ("h", "15min"):
            for years in (1, 10, 25):
                cfg = _with_weather(copy.deepcopy(base))
                cfg.update(resolution=res, projection_years=years)
                out.append({"id": f"ex:{path.stem}:{res}:{years}y", "kind": "app", "config": cfg})
    com = tomllib.loads((EX / "community-portugal.toml").read_text())
    for res in ("15min",):  # ACC requires 15-minute intervals
        for years in (1, 10, 25):
            cfg = copy.deepcopy(com)
            cfg["community"].update(resolution=res, projection_years=years)
            out.append({"id": f"community:{res}:{years}y", "kind": "community", "config": cfg})
    mc = tomllib.loads((EX / "montecarlo.toml").read_text())
    settings = mc.pop("montecarlo")
    for res in ("h", "15min"):
        for n in (10, 100, 1000):
            if res == "15min" and n == 1000:
                continue
            cfg = {**copy.deepcopy(mc), "resolution": res}  # Monte Carlo takes its weather from settings
            st = {**settings, "weather_file": str(S.DATA / "porto_pvgis_sarah3_2005_2023.csv"), "n_runs": n, "n_procs": 1}
            out.append({"id": f"mc:{res}:{n}runs", "kind": "montecarlo", "config": cfg, "settings": st})
    for res in ("h", "15min"):
        for n in (10, 100, 1000):
            for years in (1, 20):
                cfg = {
                    "location": "porto", "weather_file": str(S.DATA / "porto_tmy_pvgis.csv"),
                    "annual_consumption_kwh": 400 * n, "battery_kwh": n / 2, "n_modules": n,
                    "pv_module": "Suntech_STP550S_STC", "cost_preset": "residential_pt",
                    "resolution": res, "projection_years": years,
                }
                out.append({"id": f"scale:{n}mod:{res}:{years}y", "kind": "app", "config": cfg})
    return out


def run(spec: dict) -> dict:
    try:
        p = subprocess.run(
            [sys.executable, str(HERE / "child.py"), json.dumps(spec, default=str)],
            capture_output=True, text=True, timeout=TIMEOUT_S, cwd=DATA,
        )
        lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
        if lines:
            return json.loads(lines[-1])
        return {"id": spec["id"], "ok": False, "error": f"exit {p.returncode}: {p.stderr[-300:]}"}
    except subprocess.TimeoutExpired:
        return {"id": spec["id"], "ok": False, "error": f"timeout {TIMEOUT_S}s"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    res_path = DATA / "results.jsonl"
    done = {json.loads(ln)["id"] for ln in res_path.read_text().splitlines()} if res_path.exists() else set()
    todo = [s for s in specs() if s["id"] not in done]
    # One warm-up per kind so a first-time weather fetch or numba compile is not timed.
    print(f"{len(todo)} specs to run", flush=True)
    with ThreadPoolExecutor(a.workers) as pool, res_path.open("a") as fh:
        for r in pool.map(run, todo):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(r["id"], "ok" if r.get("ok") else r.get("error"), round(r.get("peak_rss_mb", 0)), "MB", flush=True)


if __name__ == "__main__":
    main()
