"""Write, or check, the App-level golden baseline.

The baseline pins what ``App.result()`` returns for a handful of fixed
configurations, field by field, so a refactor of the dispatch or the year loop
can be checked against the numbers the App reported before it. It records
every float by its bit pattern, as ``tools/parity/app_parity.py`` compares
them, next to its ``repr`` for a human reader.

The runs never touch the network. Weather is a deterministic synthetic year
(the test suite's own builder), passed in where the App would fetch PVGIS.

``tests/test_app_golden.py`` compares against the committed file with a
relative tolerance of 1e-9, because BREOS does not promise bit identity
across platforms or library versions. ``--check`` compares bit for bit, which
is the right test on the machine that wrote the file.

Usage:
    python tools/generate_app_golden.py            # rewrite the baseline
    python tools/generate_app_golden.py --check    # compare bit for bit
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.parity.app_parity import flatten  # noqa: E402

GOLDEN_PATH = PROJECT_ROOT / "tests" / "fixtures" / "app_golden" / "app_golden.json"
SCHEMA = "breos-app-golden-v1"

# Fields that record where and how a run happened rather than what it found.
EXCLUDED_PREFIXES = ("provenance.execution", "provenance.breos_version")

_COMMON = {
    "location": "porto",
    "n_modules": 8,
    "annual_consumption_kwh": 4000,
    "start_date": "2025-01-01",
    "projection_years": 3,
    "cost_preset": "residential_pt",
    "emissions_country": "PT",
}

# Native and BLAST aging, hourly and 15-minute, with and without a pack
# replacement inside the projection. A raised end-of-life threshold brings
# one or two replacements into the three projected years.
SCENARIOS: dict[str, dict[str, Any]] = {
    "native_h_pv_only": {**_COMMON, "battery_kwh": 0, "resolution": "h"},
    "native_h_battery": {**_COMMON, "battery_kwh": 5, "resolution": "h"},
    "native_h_replacement": {**_COMMON, "battery_kwh": 5, "resolution": "h", "battery_eol_percentage": 0.93},
    "native_15min_replacement": {
        **_COMMON,
        "battery_kwh": 5,
        "resolution": "15min",
        "battery_eol_percentage": 0.93,
    },
    "blast_h_battery": {
        **_COMMON,
        "battery_kwh": 5,
        "resolution": "h",
        "degradation_engine": "blast",
        "blast_model": "nmc_gr_50ah_b1",
    },
    "blast_15min_replacement": {
        **_COMMON,
        "battery_kwh": 5,
        "resolution": "15min",
        "degradation_engine": "blast",
        "blast_model": "nmc_gr_50ah_b1",
        "battery_eol_percentage": 0.95,
    },
}


def synthetic_weather(year: int = 2025) -> pd.DataFrame:
    """Return the test suite's synthetic hourly year: sinusoidal sun and temperature, UTC."""
    index = pd.date_range(start=f"{year}-01-01", periods=8760, freq="h", tz="UTC")
    hour_of_year = np.arange(8760, dtype=float)
    day_of_year = hour_of_year / 24.0
    hour_of_day = hour_of_year % 24
    solar_angle = np.clip(np.sin((hour_of_day - 6) / 12 * np.pi), 0, 1)
    seasonal = 0.6 + 0.4 * np.sin((day_of_year - 80) / 365 * 2 * np.pi)
    ghi = solar_angle * seasonal * 800
    temp_air = 15 + 8 * np.sin((day_of_year - 80) / 365 * 2 * np.pi) + 4 * np.sin((hour_of_day - 14) / 24 * 2 * np.pi)
    return pd.DataFrame(
        {"ghi": ghi, "dni": ghi * 0.7, "dhi": ghi * 0.3, "temp_air": temp_air, "wind_speed": np.full(8760, 3.0)},
        index=index,
    )


def _fake_fetch(*_args, **kwargs):
    weather = synthetic_weather(int(kwargs.get("sample_year") or 2025))
    weather.attrs["breos_weather_metadata"] = {
        "source": "synthetic_golden",
        "horizon": {"status": "not_applied", "provider": "pvgis", "profile": None},
    }
    return weather, {"inputs": {"location": {"latitude": 41.1579, "longitude": -8.6291, "elevation": 0}}}


def run_scenario(name: str) -> dict[str, Any]:
    """Run one scenario offline and return its flattened result, volatile fields removed."""
    from breos import App

    with (
        mock.patch("breos.app.fetch_tmy_weather_data", _fake_fetch),
        mock.patch("breos.app.load_weather", lambda **_kwargs: None),
    ):
        app = App({**SCENARIOS[name], "execution_backend": "python"})
        app.simulate()
        result = app.result()
    return {key: value for key, value in flatten(result).items() if not key.startswith(EXCLUDED_PREFIXES)}


def _bits(value: float) -> str:
    return f"0x{struct.unpack('<Q', struct.pack('<d', value))[0]:016x}"


def encode(value: Any) -> Any:
    """Encode one leaf for JSON; floats keep their exact bits."""
    if isinstance(value, (float, np.floating)):
        return {"float": repr(float(value)), "bits": _bits(float(value))}
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if value is None or isinstance(value, str):
        return value
    return {"repr": repr(value)}


def decode_float(entry: dict[str, str]) -> float:
    return struct.unpack("<d", struct.pack("<Q", int(entry["bits"], 16)))[0]


def compare(name: str, actual: dict[str, Any], expected: dict[str, Any], *, rel: float = 0.0) -> list[str]:
    """Return a line per field that differs; ``rel=0`` compares floats bit for bit."""
    differences = [f"{name}: missing {key}" for key in sorted(set(expected) - set(actual))]
    differences += [f"{name}: unexpected {key}" for key in sorted(set(actual) - set(expected))]
    for key in sorted(set(actual) & set(expected)):
        got, want = encode(actual[key]), expected[key]
        if isinstance(want, dict) and "bits" in want:
            if not (isinstance(got, dict) and "bits" in got):
                differences.append(f"{name}: {key}: expected a float, got {got!r}")
                continue
            if got["bits"] == want["bits"]:
                continue
            new, old = float(got["float"]), decode_float(want)
            if rel and (math.isclose(new, old, rel_tol=rel, abs_tol=rel) or (math.isnan(new) and math.isnan(old))):
                continue
            differences.append(f"{name}: {key}: {new!r} != {old!r}")
        elif got != want:
            differences.append(f"{name}: {key}: {got!r} != {want!r}")
    return differences


def _environment() -> dict[str, str]:
    import pvlib

    try:
        commit = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {
        "breos_commit": commit,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pvlib": pvlib.__version__,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare bit for bit instead of writing")
    args = parser.parse_args(argv)

    results = {name: run_scenario(name) for name in SCENARIOS}
    if args.check:
        golden = json.loads(GOLDEN_PATH.read_text())
        differences = [
            line for name in SCENARIOS for line in compare(name, results[name], golden["scenarios"].get(name, {}))
        ]
        for line in differences:
            print(line)
        print(f"{'FAIL' if differences else 'PASS'}: {len(differences)} difference(s)")
        return 1 if differences else 0

    payload = {
        "schema": SCHEMA,
        "generated_with": _environment(),
        "scenarios": {name: {key: encode(value) for key, value in results[name].items()} for name in SCENARIOS},
    }
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {GOLDEN_PATH.relative_to(PROJECT_ROOT)}: {sum(len(r) for r in results.values())} fields")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
