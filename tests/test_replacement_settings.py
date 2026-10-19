"""The replacement-policy settings at each layer they cross.

Each policy is a ``BatteryConfig`` field, an App and Monte Carlo key
``battery_<field>`` and an optimizer ``[battery] <field>`` key (#304, #377,
#400, #404). Every layer has the same contract: a default, a normalized
value that reaches the battery, and a refusal that names the key. One row
per setting holds that layer's values and messages.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

from breos.app_config import resolve_app_config
from breos.battery import BatteryConfig
from breos.optimization import _build_battery_config_from_spec
from breos.optimization_config import resolve_optimization_config
from breos.projection import build_battery_config

APP_BASE = {"location": "porto", "n_modules": 8, "annual_consumption_kwh": 4000, "battery_kwh": 5}
OPTIMIZER_BASE = {"location": {"latitude": 41.15, "longitude": -8.61}}


@dataclass(frozen=True)
class Layer:
    """A setting at one layer: its default, a value and what it is stored as, and refused values."""

    default: object
    value: object
    stored: object
    refused: tuple[tuple[object, type[Exception]], ...]
    message: str


@dataclass(frozen=True)
class Setting:
    field: str
    battery: Layer | None = None
    app: Layer | None = None
    optimizer: Layer | None = None


SETTINGS = (
    Setting(
        "allow_terminal_replacement",
        battery=Layer(
            True,
            np.bool_(False),
            False,
            tuple((value, ValueError) for value in ("false", "true", 0, 1, 1.0, None)),
            "allow_terminal_replacement must be a bool",
        ),
        app=Layer(
            True,
            False,
            False,
            tuple((value, TypeError) for value in ("false", 0, 1, None)),
            "'battery_allow_terminal_replacement' must be a boolean",
        ),
        optimizer=Layer(
            True,
            False,
            False,
            tuple((value, TypeError) for value in ("false", 0, None)),
            "'battery.allow_terminal_replacement' must be true or false",
        ),
    ),
    Setting(
        "replacement_min_remaining_years",
        battery=Layer(
            0.0,
            1,
            1.0,
            tuple((value, ValueError) for value in (-0.5, math.nan, math.inf, True, None)),
            "replacement_min_remaining_years must be",
        ),
        app=Layer(
            0.0,
            1,
            1.0,
            (
                (-0.5, ValueError),
                (math.nan, ValueError),
                (math.inf, ValueError),
                (True, TypeError),
                ("1", TypeError),
            ),
            "'battery_replacement_min_remaining_years' must be",
        ),
        optimizer=Layer(
            0.0,
            1,
            1.0,
            ((-0.5, ValueError), (math.nan, ValueError), (True, TypeError)),
            "'battery.replacement_min_remaining_years' must be",
        ),
    ),
    Setting(
        "skipped_replacement_action",
        battery=Layer(
            "keep",
            "retire",
            "retire",
            tuple((value, ValueError) for value in ("replace", "Retire", None, True)),
            "skipped_replacement_action must be one of: keep, retire",
        ),
        app=Layer(
            "keep",
            "retire",
            "retire",
            (("replace", ValueError),),
            "'battery_skipped_replacement_action' must be one of: keep, retire",
        ),
        optimizer=Layer(
            "keep",
            "retire",
            "retire",
            (("sell", ValueError),),
            "battery.skipped_replacement_action",
        ),
    ),
    Setting(
        "enable_replacement",
        app=Layer(
            True,
            False,
            False,
            (("false", TypeError),),
            "'battery_enable_replacement' must be a boolean",
        ),
    ),
    # The years that follow a span are set by the projection, never by a user key.
    Setting(
        "replacement_years_after_span",
        battery=Layer(
            0,
            np.int64(2),
            2,
            tuple((value, ValueError) for value in (-1, 1.0, True, "1", None)),
            "replacement_years_after_span must be a non-negative integer",
        ),
    ),
)


def _rows(layer: str):
    return [
        pytest.param(setting.field, getattr(setting, layer), id=setting.field)
        for setting in SETTINGS
        if getattr(setting, layer) is not None
    ]


def _refusals(layer: str):
    return [
        pytest.param(setting.field, value, error, getattr(setting, layer).message, id=f"{setting.field}-{value!r}")
        for setting in SETTINGS
        if getattr(setting, layer) is not None
        for value, error in getattr(setting, layer).refused
    ]


def _assert_stored(actual, expected) -> None:
    # The type is part of the contract: a bool stays a bool, years are a float or an int.
    assert actual == expected
    assert type(actual) is type(expected)


# -- BatteryConfig ---------------------------------------------------------------------


@pytest.mark.parametrize(("field", "layer"), _rows("battery"))
def test_the_battery_config_default_and_its_normalized_value(field, layer):
    _assert_stored(getattr(BatteryConfig(nominal_energy_wh=5000.0), field), layer.default)
    _assert_stored(getattr(BatteryConfig(nominal_energy_wh=5000.0, **{field: layer.value}), field), layer.stored)


@pytest.mark.parametrize(("field", "value", "error", "message"), _refusals("battery"))
def test_the_battery_config_refuses_a_bad_value(field, value, error, message):
    with pytest.raises(error, match=message):
        BatteryConfig(nominal_energy_wh=5000.0, **{field: value})


# -- App and Monte Carlo key -------------------------------------------------------------


@pytest.mark.parametrize(("field", "layer"), _rows("app"))
def test_the_app_key_resolves_its_default_and_reaches_the_battery(field, layer):
    key = f"battery_{field}"
    for config, expected in ((APP_BASE, layer.default), ({**APP_BASE, key: layer.value}, layer.stored)):
        resolved = resolve_app_config(config)
        _assert_stored(resolved.cfg[key], expected)
        _assert_stored(getattr(build_battery_config(resolved.cfg, resolved, initial_soh=100.0), field), expected)


@pytest.mark.parametrize(("field", "value", "error", "message"), _refusals("app"))
def test_the_app_key_refuses_a_bad_value(field, value, error, message):
    with pytest.raises(error, match=message):
        resolve_app_config({**APP_BASE, f"battery_{field}": value})


# -- optimizer [battery] key -------------------------------------------------------------


@pytest.mark.parametrize(("field", "layer"), _rows("optimizer"))
def test_the_optimizer_key_resolves_its_default_and_is_forwarded(field, layer):
    _assert_stored(resolve_optimization_config(OPTIMIZER_BASE)["battery"][field], layer.default)
    resolved = resolve_optimization_config({**OPTIMIZER_BASE, "battery": {field: layer.value}})
    _assert_stored(resolved["battery"][field], layer.stored)
    battery = _build_battery_config_from_spec({field: layer.value}, nominal_energy_wh=5000.0)
    _assert_stored(getattr(battery, field), layer.stored)


@pytest.mark.parametrize(("field", "value", "error", "message"), _refusals("optimizer"))
def test_the_optimizer_key_refuses_a_bad_value(field, value, error, message):
    with pytest.raises(error, match=message):
        resolve_optimization_config({**OPTIMIZER_BASE, "battery": {field: value}})
