"""Run the gallery cases with whatever Python and packages this interpreter has (study 16).

Usage: <env>/bin/python run_gallery.py OUT_DIR [CASE ...]   (default: every case but montecarlo)
Uses tools/regenerate_gallery_results.py's own run_case, so the outputs have the stored layout.
"""

import argparse
import importlib.util
import json
import platform
import sys
import time
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("regen", REPO / "tools" / "regenerate_gallery_results.py")
regen = importlib.util.module_from_spec(spec)
sys.modules["regen"] = regen
spec.loader.exec_module(regen)  # sets the numpy X86_V3 limit before numpy loads

out = Path(sys.argv[1])
names = sys.argv[2:] or [n for n in regen.CASES if n != "montecarlo"]
options = argparse.Namespace(check=False, cross_machine=False, mc_weather=None, procs=4, cases=names, list=False)
out.mkdir(parents=True, exist_ok=True)
versions = {"python": platform.python_version()}
for pkg in ("numpy", "pandas", "pvlib", "scipy", "numba", "pymoo"):
    try:
        versions[pkg] = metadata.version(pkg)
    except metadata.PackageNotFoundError:
        versions[pkg] = None
(out / "versions.json").write_text(json.dumps(versions, indent=1))
for name in names:
    if (out / name / "manifest.json").exists():
        continue
    t = time.perf_counter()
    try:
        regen.run_case(regen.CASES[name], out / name, options)
        print(f"{name}: {time.perf_counter() - t:.0f}s", flush=True)
    except Exception as exc:
        (out / f"{name}.error.txt").write_text(f"{type(exc).__name__}: {exc}")
        print(f"{name}: FAILED {type(exc).__name__}: {str(exc)[:200]}", flush=True)
