"""Validation gate: App's assembled outputs must be identical on both backends.

The timestep parity tests compare the physics -- the buffer matrix, step by
step. They say nothing about what App *builds* out of that: the yearly rollups,
the cost projection, LCOE, NPV, payback, the monthly and financial tables, the
degradation summary and the PV loss waterfall. Those are the numbers a reader
of the Article actually sees, and they pass through economics and aggregation
code that the timestep comparison never touches.

So this walks the whole result dict, flattens every leaf to a comparable value,
and requires exact equality field by field. Floats are compared bitwise via
their raw bytes, not with a tolerance: the claim being gated is bit identity,
and "close" would let a reassociated sum through.

The ``provenance.execution`` block is excluded by design -- it *must* differ,
because it records which backend ran. Everything else must not.

Usage:
    python tools/parity/app_parity.py                # every scenario
    python tools/parity/app_parity.py c2             # one scenario
"""

from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Configurations shaped like the Article's cases: one with no battery and a
# trivial dispatch branch, one balanced case with a binding charge cap, one
# larger system, and one at 15-minute resolution where the day loop runs 96
# steps instead of 24.
SCENARIOS: dict[str, dict] = {
    "c1_no_battery": {
        "location": "porto",
        "n_modules": 6,
        "annual_consumption_kwh": 5000,
        "battery_kwh": 0,
        "projection_years": 5,
        "resolution": "h",
    },
    "c2_balanced": {
        "location": "porto",
        "n_modules": 9,
        "annual_consumption_kwh": 5000,
        "battery_kwh": 5,
        "projection_years": 5,
        "resolution": "h",
        "battery_max_charge_power_w": 4352.0,
    },
    "large_system": {
        "location": "porto",
        "n_modules": 16,
        "annual_consumption_kwh": 9000,
        "battery_kwh": 12,
        "projection_years": 8,
        "resolution": "h",
    },
    "quarter_hourly": {
        "location": "porto",
        "n_modules": 9,
        "annual_consumption_kwh": 5000,
        "battery_kwh": 5,
        "projection_years": 3,
        "resolution": "15min",
    },
}

# The two places that record *which backend ran*. They are required to differ,
# and nothing else is. Keep this list minimal: every entry added here is a
# field the gate stops checking.
EXCLUDED_PREFIXES = (
    "provenance.execution",
    "provenance.resolved_config.execution_backend",
)


def _bits(value: float) -> int:
    """Return the raw bit pattern of a float, so NaN compares equal to NaN."""
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def flatten(value, prefix: str = "") -> dict[str, object]:
    """Flatten a nested result into ``{dotted.path: leaf}``."""
    flat: dict[str, object] = {}
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            flat.update(flatten(value[key], f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, (list, tuple)):
        flat[f"{prefix}.__len__"] = len(value)
        for index, item in enumerate(value):
            flat.update(flatten(item, f"{prefix}[{index}]"))
    else:
        flat[prefix] = value
    return flat


def _differs(left, right) -> str | None:
    """Return a description of how two leaves differ, or None if identical."""
    if isinstance(left, float) and isinstance(right, float):
        if _bits(left) == _bits(right):
            return None
        if math.isnan(left) and math.isnan(right):
            return None
        return f"{left!r} != {right!r} (abs diff {abs(left - right):.6g})"
    if type(left) is not type(right):
        return f"type {type(left).__name__} != {type(right).__name__}"
    if left != right:
        return f"{left!r} != {right!r}"
    return None


def compare_scenario(name: str, config: dict) -> tuple[int, list[str]]:
    """Run one config on both backends and return (fields compared, differences)."""
    from breos import App

    results = {}
    for backend in ("python", "numba"):
        app = App({**config, "execution_backend": backend})
        app.simulate()
        results[backend] = flatten(app.result())

    reference, compiled = results["python"], results["numba"]

    def excluded(key: str) -> bool:
        return key.startswith(EXCLUDED_PREFIXES)

    differences = []
    # A key present on one side only is a difference in the output contract,
    # not just in a value -- but the excluded prefixes legitimately carry
    # different keys, since a numba record has compiler versions a Python one
    # does not.
    for key in sorted(set(reference) - set(compiled)):
        if not excluded(key):
            differences.append(f"{name}: MISSING under numba: {key}")
    for key in sorted(set(compiled) - set(reference)):
        if not excluded(key):
            differences.append(f"{name}: MISSING under python: {key}")

    compared = 0
    for key in sorted(set(reference) & set(compiled)):
        if excluded(key):
            continue
        compared += 1
        detail = _differs(reference[key], compiled[key])
        if detail is not None:
            differences.append(f"{name}: {key}: {detail}")

    # The exclusion must be earning its keep: if the backends stopped being
    # recorded, this gate would silently compare two identical Python runs.
    if reference.get("provenance.execution.execution_backend") == compiled.get(
        "provenance.execution.execution_backend"
    ):
        differences.append(f"{name}: both runs recorded the same backend -- the comparison proves nothing")

    return compared, differences


def main() -> int:
    selected = sys.argv[1:] or list(SCENARIOS)
    unknown = [name for name in selected if name not in SCENARIOS]
    if unknown:
        print(f"unknown scenario(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"available: {', '.join(SCENARIOS)}", file=sys.stderr)
        return 2

    total_fields = 0
    all_differences: list[str] = []
    for name in selected:
        compared, differences = compare_scenario(name, SCENARIOS[name])
        total_fields += compared
        all_differences.extend(differences)
        status = "EXACT" if not differences else f"{len(differences)} DIFFER"
        print(f"{name:>16}: {compared:>5} fields  {status}", flush=True)

    print()
    if all_differences:
        for line in all_differences:
            print(line)
        print(f"\nFAIL: {len(all_differences)} difference(s) across {total_fields} compared fields")
        return 1
    print(f"PASS: {total_fields} fields identical across {len(selected)} scenario(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
