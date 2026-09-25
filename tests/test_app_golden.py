"""App results match the committed pre-0.7 golden baseline (#184).

Regenerate with ``python tools/generate_app_golden.py`` only when a change to
reported numbers is intended, and record that change in the changelog.
"""

import json

import pytest

from tools.generate_app_golden import GOLDEN_PATH, SCENARIOS, SCHEMA, compare, encode, run_scenario

_GOLDEN = json.loads(GOLDEN_PATH.read_text())


def test_golden_file_covers_every_scenario():
    assert _GOLDEN["schema"] == SCHEMA
    assert set(_GOLDEN["scenarios"]) == set(SCENARIOS)


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_app_result_matches_golden(name):
    # BREOS promises bit identity between backends on one machine, not across
    # platforms, so the committed floats are compared to 1e-9 here.
    # ``tools/generate_app_golden.py --check`` compares them bit for bit.
    differences = compare(name, run_scenario(name), _GOLDEN["scenarios"][name], rel=1e-9)

    assert not differences, "\n".join(differences[:20])


def test_compare_reports_a_changed_float():
    expected = {"npv_savings_eur": encode(100.0), "battery_replacements": 1}

    assert compare("x", {"npv_savings_eur": 100.0, "battery_replacements": 1}, expected) == []
    assert compare("x", {"npv_savings_eur": 100.0 + 1e-6, "battery_replacements": 1}, expected, rel=1e-9) == [
        "x: npv_savings_eur: 100.000001 != 100.0"
    ]
    assert compare("x", {"npv_savings_eur": 100.0}, expected) == ["x: missing battery_replacements"]
