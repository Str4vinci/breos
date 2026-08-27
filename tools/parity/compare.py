"""Compare two harness dumps field by field, exact first.

Prints one line per differing field with its maximum absolute and relative
difference and the count of differing elements. Exits non-zero if anything
differs, so it can gate a parity claim.
"""

from __future__ import annotations

import sys

import numpy as np


def compare(left_path: str, right_path: str, left_label: str = "left", right_label: str = "right") -> int:
    left = np.load(left_path)
    right = np.load(right_path)

    only_left = sorted(set(left.files) - set(right.files))
    only_right = sorted(set(right.files) - set(left.files))
    for key in only_left:
        print(f"MISSING in {right_label}: {key}")
    for key in only_right:
        print(f"MISSING in {left_label}: {key}")

    shared = sorted(set(left.files) & set(right.files))
    differing = []
    for key in shared:
        a, b = left[key], right[key]
        if a.shape != b.shape:
            print(f"SHAPE {key}: {a.shape} vs {b.shape}")
            differing.append(key)
            continue
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        if same.all():
            continue
        delta = np.abs(np.where(np.isnan(a) | np.isnan(b), 0.0, a - b))
        scale = np.maximum(np.abs(a), np.abs(b))
        rel = np.where(scale > 0.0, delta / np.where(scale > 0.0, scale, 1.0), 0.0)
        differing.append(key)
        print(f"DIFF {key}: n={int((~same).sum())}/{a.size} max_abs={delta.max():.6e} max_rel={rel.max():.6e}")

    total = len(shared)
    if not differing and not only_left and not only_right:
        print(f"EXACT MATCH across {total} fields ({left_label} vs {right_label})")
        return 0
    print(f"\n{len(differing)}/{total} fields differ ({left_label} vs {right_label})")
    return 1


if __name__ == "__main__":
    sys.exit(compare(*sys.argv[1:]))
