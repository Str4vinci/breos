"""Study 16: compare each environment's gallery outputs with the stored results.

For every (env, case): identical at the tool's 1e-9 check, accepted by --cross-machine, and our
own magnitudes: numeric values compared, the largest relative and absolute differences, the field
that moved most, and any change in a discrete field (replacements, payback year, counts).

Usage: uv run --with tabulate python compare.py -> output/summary.md, output/fields.csv
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
spec = importlib.util.spec_from_file_location("regen", REPO / "tools" / "regenerate_gallery_results.py")
regen = importlib.util.module_from_spec(spec)
sys.modules["regen"] = regen
spec.loader.exec_module(regen)
STORED = REPO / "docs" / "examples" / "_results"
OUT = HERE / "output"


def leaves(tree, path=""):
    if isinstance(tree, dict):
        for k, v in tree.items():
            if k not in regen.VOLATILE_KEYS:
                yield from leaves(v, f"{path}.{k}")
    elif isinstance(tree, list):
        for i, v in enumerate(tree):
            yield from leaves(v, f"{path}[{i}]")
    else:
        yield path, tree


def numeric_pairs(stored: Path, fresh: Path):
    if stored.suffix == ".json":
        a = dict(leaves(json.loads(stored.read_text())))
        b = dict(leaves(json.loads(fresh.read_text())))
        for k in a.keys() & b.keys():
            x, y = a[k], b[k]
            if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
                field = k.rsplit(".", 1)[-1].split("[")[0]
                yield k, field, float(x), float(y)
    else:
        a, b = pd.read_csv(stored), pd.read_csv(fresh)
        if list(a.columns) != list(b.columns) or len(a) != len(b):
            return
        for c in a.columns:
            if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
                for i, (x, y) in enumerate(zip(a[c].to_numpy(float), b[c].to_numpy(float))):
                    yield f"{stored.name}[{i},{c}]", c, x, y


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rows, fields = [], []
    for env in sorted((HERE / "data" / "out").iterdir()):
        versions = json.loads((env / "versions.json").read_text())
        for case in sorted(p for p in env.iterdir() if p.is_dir()):
            manifest = json.loads((STORED / case.name / "manifest.json").read_text())
            strict, cross, n, nz = [], [], 0, 0
            worst = (0.0, 0.0, "")
            discrete = []
            for name in manifest["files"]:
                s, f = STORED / case.name / name, case / name
                if not f.exists():
                    strict.append(f"{name} missing")
                    continue
                strict += regen.compare_file(s, f, False)
                cross += regen.compare_file(s, f, True)
                for path, field, x, y in numeric_pairs(s, f):
                    n += 1
                    if np.isnan(x) and np.isnan(y):
                        continue
                    d = abs(x - y)
                    if d > 0:
                        nz += 1
                        rel = d / max(abs(x), 1e-12)
                        fields.append({"env": env.name, "case": case.name, "path": path, "field": field,
                                       "stored": x, "fresh": y, "abs": d, "rel": rel})
                        if rel > worst[0]:
                            worst = (rel, d, path)
                        if field in regen.DISCRETE_FIELDS:
                            discrete.append(f"{path}: {x:g} -> {y:g}")
            rows.append({"env": env.name, **{k: versions.get(k) for k in ("python", "numpy", "pandas", "pvlib", "scipy", "numba")},
                         "case": case.name, "numbers": n, "changed": nz, "strict_ok": not strict, "cross_machine_ok": not cross,
                         "max_rel": worst[0], "max_abs": worst[1], "worst_path": worst[2], "discrete_changes": "; ".join(discrete[:5])})
        for err in env.glob("*.error.txt"):
            rows.append({"env": env.name, "case": err.name.split(".")[0], "error": err.read_text()[:200]})
    r, f = pd.DataFrame(rows), pd.DataFrame(fields)
    r.to_csv(OUT / "cases.csv", index=False)
    f.to_csv(OUT / "fields.csv", index=False)
    piv = r.pivot_table(index="env", values=["strict_ok", "cross_machine_ok"], aggfunc="mean").round(2)
    piv["max_rel"] = r.groupby("env").max_rel.max()
    piv["cases_with_discrete_change"] = r.groupby("env").discrete_changes.apply(lambda x: int((x.fillna("") != "").sum()))
    lines = ["# Study 16 summary\n", "Share of gallery cases identical (1e-9) and accepted by --cross-machine, per environment.\n",
             piv.to_markdown(), "", "## Per case\n",
             r.drop(columns=[c for c in ("python", "numpy", "pandas", "pvlib", "scipy", "numba") if c in r]).to_markdown(index=False), ""]
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines[:4]))


if __name__ == "__main__":
    main()
