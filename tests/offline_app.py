"""Offline App runs and the tariff tables shared by the pricing and revaluation tests.

The runs read the synthetic weather of ``tools/generate_app_golden.py``, so
they need no network and their numbers match the golden scenarios.
"""

import math
from unittest import mock

import pytest

from breos import App, optimization
from tools.generate_app_golden import EXCLUDED_PREFIXES, SCENARIOS, _fake_fetch, flatten

# The hourly native-aging golden scenario, with a pack replacement inside its three years.
BASE = {**SCENARIOS["native_h_replacement"], "execution_backend": "python"}
# Portugal's 2026 daily bi-hourly cycle at illustrative prices.
TOU = {
    "schedule": "pt_mainland_2026_daily_bi",
    "currency": "EUR",
    "import_prices": {"peak": 0.28, "off_peak": 0.11},
    "export_prices": {"all": 0.05},
    "fixed_charge_per_day": 0.25,
}
# No-system reference tariffs (#339): flat, on the bundled daily bi-hourly
# schedule, and on a custom schedule with quarterly month seasons.
FLAT_REFERENCE = {"currency": "EUR", "import_prices": {"all": 0.30}, "fixed_charge_per_day": 0.40}
TOU_REFERENCE = {
    "schedule": "pt_mainland_2026_daily_bi",
    "currency": "EUR",
    "import_prices": {"peak": 0.35, "off_peak": 0.09},
    "fixed_charge_per_day": 0.20,
}
CUSTOM_SCHEDULE = {
    "identifier": "reference_custom_daily",
    "version": "1",
    "timezone": "Europe/Lisbon",
    "cycle": "daily",
    "periods": ["peak", "off_peak"],
    "rules": [
        {
            "days": "all",
            "season": "all",
            "intervals": {"off_peak": [["00:00", "08:00"], ["22:00", "24:00"]], "peak": [["08:00", "22:00"]]},
        }
    ],
}
QUARTERS = {"q1": [1, 2, 3], "q2": [4, 5, 6], "q3": [7, 8, 9], "q4": [10, 11, 12]}
MONTH_REFERENCE = {
    "currency": "EUR",
    "custom_schedule": {**CUSTOM_SCHEDULE, "identifier": "reference_quarterly", "seasons": QUARTERS},
    "import_prices": {
        "q1": {"peak": 0.35, "off_peak": 0.09},
        "q2": {"peak": 0.30, "off_peak": 0.12},
        "q3": {"peak": 0.25, "off_peak": 0.08},
        "q4": {"peak": 0.40, "off_peak": 0.15},
    },
    "fixed_charge_per_day": 0.20,
}
# An annual network credit (#375) for the daily bi-hourly schedule.
CREDIT = {
    "amount_per_year": 146.58,
    "network_fixed_per_year": 39.70,
    "network_import_prices": {"peak": 0.0888, "off_peak": 0.0311},
}


def offline():
    """The weather fetch and file load patched to the golden synthetic weather."""
    return (
        mock.patch("breos.app.fetch_tmy_weather_data", _fake_fetch),
        mock.patch("breos.app.load_weather", lambda **_kwargs: None),
    )


def simulate_offline(config) -> App:
    fetch, weather = offline()
    with fetch, weather:
        app = App(config)
        app.simulate()
    return app


def revalue_offline(app, changes):
    fetch, weather = offline()
    with fetch, weather:
        return app.revalue(changes)


def result_fields(result):
    """The flattened result without its volatile fields or the revaluation record."""
    return {
        key: value
        for key, value in flatten(result).items()
        if not key.startswith(EXCLUDED_PREFIXES) and not key.startswith("provenance.revaluation")
    }


def assert_fields_close(revalued, fresh, rel=1e-12):
    assert revalued.keys() == fresh.keys()
    for key, value in fresh.items():
        if isinstance(value, float) and math.isfinite(value):
            assert revalued[key] == pytest.approx(value, rel=rel, abs=1e-9), key
        else:
            assert revalued[key] == value, key


def evaluate_pv_design(weather, load, config):
    """The projected evaluation of 4 modules, no battery, at 30 degrees facing south."""
    return optimization.evaluate_projected_design(
        weather, load, config, n_modules=4, battery_kwh=0.0, tilt=30.0, azimuth=180.0
    )
