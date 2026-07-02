"""Shared paths and helpers for the validation suite.

Used by the validation scripts and by ``tests/test_validation_drift.py`` so
that the baseline generator and the drift test load weather identically.
"""

import json
from pathlib import Path

import pandas as pd

VALIDATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VALIDATION_DIR.parent
WEATHER_DIR = VALIDATION_DIR / "data" / "weather"
REFERENCES_DIR = VALIDATION_DIR / "data" / "references"
BASELINE_PATH = VALIDATION_DIR / "baselines" / "breos_baseline.json"
RESULTS_DIR = VALIDATION_DIR / "results"
RESULTS_PATH = RESULTS_DIR / "breos_results.json"
REPORT_PATH = VALIDATION_DIR / "REPORT.md"

MONTH_LABELS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def load_spec():
    """Return (system_spec, locations) from validation/locations.json."""
    with open(VALIDATION_DIR / "locations.json") as f:
        cfg = json.load(f)
    return cfg["system"], cfg["locations"]


def find_weather_file(location_key: str):
    """Return the checked-in TMY weather file for a location, or None."""
    if not WEATHER_DIR.is_dir():
        return None
    matches = sorted(WEATHER_DIR.glob(f"{location_key}_tmy_*.csv*"))
    return matches[0] if matches else None


def load_validation_weather(filepath) -> pd.DataFrame:
    """Load a checked-in validation weather CSV with a tz-aware index.

    Timestamps are converted to UTC (same instants, so solar position is
    unaffected); monthly aggregation therefore uses UTC month boundaries,
    which only moves zero-production night hours between months.
    """
    df = pd.read_csv(filepath, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def load_reference(location_key: str):
    """Return the reference JSON for a location, or None if not fetched."""
    path = REFERENCES_DIR / f"{location_key}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)
