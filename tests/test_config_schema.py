"""Nested config tables are checked by one schema (#181)."""

import math

import pytest

from breos.app import App
from breos.config_schema import TableSpec, anything, boolean, choice, list_of, mapping_of, number, table, text

BASE = {"location": "porto", "n_modules": 6, "annual_consumption_kwh": 3000}


def _band(checked, where):
    if checked.get("low", -math.inf) > checked.get("high", math.inf):
        raise ValueError(f"'{where}.low' must not exceed '{where}.high'")


SPEC = TableSpec(
    "demo",
    keys={
        "rate": number(minimum=0, maximum=1),
        "enabled": boolean,
        "mode": choice(("flat", "time_of_use")),
        "name": text,
        "low": number(),
        "high": number(),
        "periods": list_of(choice(("peak", "off_peak")), min_length=1),
        "prices": mapping_of(text, number(minimum=0)),
        "nested": table(TableSpec("inner", keys={"x": number()}, required=frozenset({"x"}))),
        "free": anything,
    },
    required=frozenset({"mode"}),
    check=_band,
)


def test_a_valid_table_is_returned_normalised_and_the_input_is_unchanged():
    raw = {"mode": "Time_Of_Use", "rate": 1, "prices": {"peak": 2}, "nested": {"x": 3}, "periods": ["PEAK"]}

    checked = SPEC.validate(raw)

    assert checked == {
        "mode": "time_of_use",
        "rate": 1.0,
        "prices": {"peak": 2.0},
        "nested": {"x": 3.0},
        "periods": ["peak"],
    }
    assert raw["mode"] == "Time_Of_Use" and raw["rate"] == 1


@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        (0.5, TypeError, r"'demo' must be a table/dict"),
        ({"mode": "flat", "rte": 1}, ValueError, r"Unknown key 'demo\.rte'\. Available: demo\.enabled, "),
        ({"mode": "flat", "a": 1, "b": 2}, ValueError, r"Unknown keys 'demo\.a', 'demo\.b'"),
        ({}, ValueError, r"'demo' needs demo\.mode"),
        ({"mode": "flat", "rate": 1.5}, ValueError, r"'demo\.rate' must be between 0 and 1"),
        ({"mode": "flat", "rate": True}, TypeError, r"'demo\.rate' must be a finite number"),
        ({"mode": "flat", "rate": float("nan")}, ValueError, r"'demo\.rate' must be a finite number"),
        ({"mode": "flat", "enabled": "no"}, TypeError, r"'demo\.enabled' must be true or false"),
        ({"mode": "tou"}, ValueError, r"'demo\.mode' must be one of: flat, time_of_use; got 'tou'"),
        ({"mode": "flat", "name": ""}, TypeError, r"'demo\.name' must be a non-empty string"),
        ({"mode": "flat", "low": 5, "high": 1}, ValueError, r"'demo\.low' must not exceed 'demo\.high'"),
        ({"mode": "flat", "periods": []}, ValueError, r"'demo\.periods' needs at least 1 entry"),
        ({"mode": "flat", "periods": ["peak", "noon"]}, ValueError, r"'demo\.periods\[1\]' must be one of"),
        ({"mode": "flat", "prices": {"peak": -1}}, ValueError, r"'demo\.prices\.peak' must be >= 0"),
        ({"mode": "flat", "nested": {}}, ValueError, r"'demo\.nested' needs demo\.nested\.x"),
    ],
)
def test_table_errors_name_the_dotted_key(value, error, message):
    with pytest.raises(error, match=message):
        SPEC.validate(value)


def test_where_renames_a_table_inside_a_list():
    with pytest.raises(ValueError, match=r"Unknown key 'items\[2\]\.bad'"):
        SPEC.validate({"mode": "flat", "bad": 1}, "items[2]")


@pytest.mark.parametrize(
    ("config", "error", "message"),
    [
        ({"costs": {"storage_cost_per_kwh": -1}}, ValueError, r"'costs\.storage_cost_per_kwh' must be >= 0"),
        ({"battery_indoor_model": {"floor_c": 30, "ceiling_c": 10}}, ValueError, r"floor_c' must not exceed"),
        (
            {"battery_indoor_model": {"enabled": "yes"}},
            TypeError,
            r"'battery_indoor_model\.enabled' must be true or false",
        ),
        ({"battery_indoor_model": {"setpoint": 20}}, ValueError, r"Unknown key 'battery_indoor_model\.setpoint'"),
        ({"pv_arrays": [{"modules": 4}, {"modules": 2, "tlt": 10}]}, ValueError, r"Unknown key 'pv_arrays\[1\]\.tlt'"),
    ],
)
def test_app_nested_tables_use_the_schema(config, error, message):
    with pytest.raises(error, match=message):
        App({**BASE, **config})
