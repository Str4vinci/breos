"""App results match the committed pre-0.7 golden baseline (#184).

Regenerate with ``python tools/generate_app_golden.py`` only when a change to
reported numbers is intended, and record that change in the changelog.
"""

import json

import pytest

from tools.generate_app_golden import (
    GOLDEN_DIR,
    SCENARIOS,
    SCHEMA,
    compare,
    encode,
    golden_path,
    load_golden,
    run_scenario,
)


def test_golden_files_cover_every_scenario():
    assert {path.stem for path in GOLDEN_DIR.glob("*.json")} == set(SCENARIOS)
    for name in SCENARIOS:
        assert json.loads(golden_path(name).read_text())["schema"] == SCHEMA


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_app_result_matches_golden(name):
    # BREOS promises bit identity between backends on one machine, not across
    # platforms, so the committed floats are compared to 1e-9 here.
    # ``tools/generate_app_golden.py --check`` compares them bit for bit.
    differences = compare(name, run_scenario(name), load_golden(name), rel=1e-9)

    assert not differences, "\n".join(differences[:20])


def test_compare_reports_a_changed_float():
    expected = {"npv_savings_eur": encode(100.0), "battery_replacements": 1}

    assert compare("x", {"npv_savings_eur": 100.0, "battery_replacements": 1}, expected) == []
    assert compare("x", {"npv_savings_eur": 100.0 + 1e-6, "battery_replacements": 1}, expected, rel=1e-9) == [
        "x: npv_savings_eur: 100.000001 != 100.0"
    ]
    assert compare("x", {"npv_savings_eur": 100.0}, expected) == ["x: missing battery_replacements"]
    assert compare("x", {"npv_savings_eur": 100, "battery_replacements": 1}, expected) == [
        "x: npv_savings_eur: expected float 100.0, got 100"
    ]


def test_compare_is_bit_exact_without_a_tolerance():
    expected = {"residual_kwh": encode(0.0)}

    assert compare("x", {"residual_kwh": -0.0}, expected) == ["x: residual_kwh: -0.0 != 0.0"]
    assert compare("x", {"residual_kwh": -0.0}, expected, rel=1e-9) == []
