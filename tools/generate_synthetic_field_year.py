"""Generate the committed synthetic "field year" endurance fixture.

The fixture is a deterministic hourly state-of-charge and cell-temperature
profile for one calendar year (365 days x 25 hourly endpoints). It is stepped
daily for 20 repeated years through every ``BlastEngine`` model by
``tests/test_blast_endurance.py``.

The profile deliberately combines the day-to-day stressor softening that drives
BLAST states past their trajectory-inversion domain:

- seasonal depth-of-discharge variation (deeper in summer, shallow in winter),
- winter idle spells (near-flat storage days),
- partial cycles,
- irregular day-to-day depth changes (seeded jitter),
- temperature seasonality bounded between 5 and 35 C.

Regenerate with::

    uv run python tools/generate_synthetic_field_year.py

The generator is pinned (fixed seed and constants); regenerating reproduces the
committed JSON byte-for-byte. Changing it would move the empirically validated
endurance bounds asserted by the test, so only regenerate deliberately.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SEED = 20260720
TEMPERATURE_AMPLITUDE_C = 15.0
TEMPERATURE_MEAN_C = 22.0
DAYS = 365
HOURS = 25  # hourly endpoints spanning a full day (0..24h)

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "blast" / "synthetic_field_year.json"


def build_field_year() -> dict[str, object]:
    """Build the deterministic one-year hourly SOC and temperature profile."""
    rng = np.random.default_rng(SEED)
    hours = np.arange(HOURS, dtype=float)
    t_secs = hours * 3600.0
    soc_days = np.empty((DAYS, HOURS))
    temperature_days = np.empty((DAYS, HOURS))

    for day in range(DAYS):
        # +1 at mid-summer (day ~172), -1 at mid-winter.
        season = float(np.sin(2 * np.pi * (day - 80) / 365.0))

        # Temperature seasonality with a small diurnal swing, clamped to [5, 35].
        t_mean = float(np.clip(TEMPERATURE_MEAN_C + TEMPERATURE_AMPLITUDE_C * season, 5.0, 35.0))
        temperature_days[day, :] = np.clip(t_mean + 1.5 * np.sin(2 * np.pi * hours / 24.0), 5.0, 35.0)

        # Seasonal depth-of-discharge, deeper in summer, plus day-to-day jitter.
        base_dod = 0.35 + 0.35 * (season + 1.0) / 2.0
        dod = float(np.clip(base_dod + rng.uniform(-0.12, 0.12), 0.1, 0.85))

        winter = season < -0.4
        idle = winter and rng.random() < 0.5  # winter idle / storage spells
        partial = rng.random() < 0.25  # partial cycles

        if idle:
            soc = np.full(HOURS, 0.5 + 0.02 * np.sin(2 * np.pi * hours / 24.0))
        else:
            amplitude = dod / 2.0 * (0.5 if partial else 1.0)
            phase = rng.uniform(0.0, 2 * np.pi)
            soc = 0.5 + amplitude * np.sin(2 * np.pi * (hours - 6) / 24.0 + phase)

        soc_days[day, :] = np.clip(soc, 0.02, 0.98)

    return {
        "schema": "blast-breos-synthetic-field-year-v1",
        "seed": SEED,
        "temperature_amplitude_c": TEMPERATURE_AMPLITUDE_C,
        "temperature_mean_c": TEMPERATURE_MEAN_C,
        "days": DAYS,
        "hours": HOURS,
        "t_secs": t_secs.tolist(),
        "soc_days": soc_days.tolist(),
        "temperature_days": temperature_days.tolist(),
    }


def main() -> None:
    fixture = build_field_year()
    FIXTURE_PATH.write_text(json.dumps(fixture, separators=(",", ":"), sort_keys=True) + "\n")
    print(f"Wrote {FIXTURE_PATH} ({FIXTURE_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
