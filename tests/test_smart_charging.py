"""The [smart_charging] table, the fixed-target controller and its instruction contract (ADR 0002, #178)."""

import json
import math
import pickle
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from breos.app import App
from breos.app_config import resolve_tariff_spec
from breos.dispatch_instructions import DispatchInstructions
from breos.smart_charging import SmartChargingSpec, resolve_instructions, smart_charging_provenance

LISBON = "Europe/Lisbon"
BASE = {"location": "porto", "n_modules": 8, "annual_consumption_kwh": 4000, "battery_kwh": 5.0, "projection_years": 3}
TOU = {
    "schedule": "pt_mainland_2026_daily_bi",
    "currency": "EUR",
    "import_prices": {"peak": 0.28, "off_peak": 0.11},
    "export_prices": {"all": 0.05},
}
FIXED = {
    "mode": "fixed_target",
    "target_usable_fraction": 0.5,
    "charge_periods": ["off_peak"],
    "discharge_periods": ["peak"],
    "grid_charge_efficiency": 0.95,
    "grid_import_limit_w": 5000,
}
TRI = {
    "schedule": "pt_mainland_2026_daily_tri",
    "currency": "EUR",
    "import_prices": {"off_peak": 0.10, "mid_peak": 0.18, "peak": 0.30},
    "export_prices": {"all": 0.04},
}
TRI_SPEC = SmartChargingSpec(
    mode="fixed_target",
    target_usable_fraction=0.6,
    charge_periods=("off_peak",),
    discharge_periods=("peak",),
    grid_charge_efficiency=0.9,
)


def _app(smart_charging, **extra):
    return App({**BASE, "tariff": TOU, **extra, "smart_charging": smart_charging})


def _without(table, *keys):
    return {key: value for key, value in table.items() if key not in keys}


def _tri_tariff(index):
    spec = resolve_tariff_spec({"tariff": TRI, "costs": None, "resolution": "15min"}, LISBON)
    return spec.resolve(index, LISBON)


def _instructions(**overrides):
    fields = {
        "discharge_allowed": [True, False, False],
        "reserve_fraction": [0.0, 0.0, 0.2],
        "grid_target_fraction": [np.nan, 0.5, np.nan],
        "grid_charge_efficiency": 0.95,
        "grid_import_limit_w": 4000.0,
    }
    return DispatchInstructions(**{**fields, **overrides})


# --- [smart_charging] validation --------------------------------------------


@pytest.mark.parametrize(
    ("table", "extra", "error", "message"),
    [
        ([], {}, TypeError, r"'smart_charging' must be a table/dict"),
        (_without(FIXED, "mode"), {}, ValueError, r"'smart_charging' needs smart_charging\.mode$"),
        (
            {**FIXED, "mode": "daily_persistence"},
            {},
            ValueError,
            r"'smart_charging\.mode' must be one of: disabled, fixed_target",
        ),
        (
            {**FIXED, "target_soc": 0.5},
            {},
            ValueError,
            r"Unknown key 'smart_charging\.target_soc'\. Available: smart_charging\.charge_periods",
        ),
        (
            {**FIXED, "target_usable_fraction": 1.2},
            {},
            ValueError,
            r"'smart_charging\.target_usable_fraction' must be between 0 and 1$",
        ),
        (
            {**FIXED, "target_usable_fraction": -0.1},
            {},
            ValueError,
            r"'smart_charging\.target_usable_fraction' must be between 0 and 1$",
        ),
        (
            {**FIXED, "target_usable_fraction": "half"},
            {},
            TypeError,
            r"'smart_charging\.target_usable_fraction' must be a finite number",
        ),
        (
            {**FIXED, "charge_periods": []},
            {},
            ValueError,
            r"'smart_charging\.charge_periods' needs at least 1 entry",
        ),
        (
            {**FIXED, "discharge_periods": []},
            {},
            ValueError,
            r"'smart_charging\.discharge_periods' needs at least 1 entry",
        ),
        (
            {**FIXED, "charge_periods": "off_peak"},
            {},
            TypeError,
            r"'smart_charging\.charge_periods' must be a list",
        ),
        (
            {**FIXED, "discharge_periods": [3]},
            {},
            TypeError,
            r"'smart_charging\.discharge_periods\[0\]' must be a non-empty string",
        ),
        (
            {**FIXED, "charge_periods": ["night"]},
            {},
            ValueError,
            r"'smart_charging\.charge_periods' has period\(s\) night that schedule 'pt_mainland_2026_daily_bi' "
            r"does not have\. Its periods: off_peak, peak\.",
        ),
        (
            {**FIXED, "discharge_periods": ["peak", "mid_peak"]},
            {},
            ValueError,
            r"'smart_charging\.discharge_periods' has period\(s\) mid_peak that schedule",
        ),
        (
            {**FIXED, "discharge_periods": ["peak", "off_peak"]},
            {},
            ValueError,
            r"'smart_charging\.charge_periods' and 'smart_charging\.discharge_periods' share off_peak; "
            r"every step either charges or discharges \(ADR 0002 A8\)",
        ),
        (
            _without(FIXED, "grid_charge_efficiency"),
            {},
            ValueError,
            r"'smart_charging' needs smart_charging\.grid_charge_efficiency for mode = 'fixed_target'",
        ),
        (
            _without(FIXED, "target_usable_fraction", "charge_periods"),
            {},
            ValueError,
            r"'smart_charging' needs smart_charging\.target_usable_fraction, smart_charging\.charge_periods for",
        ),
        (
            {**FIXED, "grid_charge_efficiency": 0.0},
            {},
            ValueError,
            r"'smart_charging\.grid_charge_efficiency' must be between 0 and 1 \(exclusive of the lower bound\)",
        ),
        (
            {**FIXED, "grid_charge_efficiency": 1.05},
            {},
            ValueError,
            r"'smart_charging\.grid_charge_efficiency' must be between 0 and 1 \(exclusive of the lower bound\)",
        ),
        (
            {**FIXED, "grid_import_limit_w": 0},
            {},
            ValueError,
            r"'smart_charging\.grid_import_limit_w' must be > 0",
        ),
        (
            {**FIXED, "grid_import_limit_w": math.inf},
            {},
            ValueError,
            r"'smart_charging\.grid_import_limit_w' must be a finite number",
        ),
        (
            {"mode": "disabled", "charge_periods": ["off_peak"], "grid_charge_efficiency": 0.9},
            {},
            ValueError,
            r"'smart_charging\.mode' = 'disabled' takes no other keys; remove smart_charging\.charge_periods, "
            r"smart_charging\.grid_charge_efficiency",
        ),
        (
            FIXED,
            {"tariff": None},
            ValueError,
            r"'smart_charging\.mode' = 'fixed_target' needs a \[tariff\]",
        ),
        (
            FIXED,
            {"battery_kwh": 0.0},
            ValueError,
            r"'smart_charging\.mode' = 'fixed_target' needs a battery; set battery_kwh > 0",
        ),
    ],
)
def test_smart_charging_config_is_checked_at_construction(table, extra, error, message):
    with pytest.raises(error, match=message):
        _app(table, **extra)


def test_a_valid_fixed_target_table_resolves_to_a_spec():
    table = {**FIXED, "mode": "Fixed_Target", "charge_periods": ["off_peak", "off_peak"]}

    resolved = _app(table)._resolved

    assert resolved.smart_charging == SmartChargingSpec(
        mode="fixed_target",
        target_usable_fraction=0.5,
        charge_periods=("off_peak",),
        discharge_periods=("peak",),
        grid_charge_efficiency=0.95,
        grid_import_limit_w=5000.0,
    )
    assert _app(_without(FIXED, "grid_import_limit_w"))._resolved.smart_charging.grid_import_limit_w is None
    assert _app({**FIXED, "grid_import_limit_w": None})._resolved.smart_charging.grid_import_limit_w is None


def test_disabled_needs_neither_a_tariff_nor_a_battery():
    resolved = App({**BASE, "battery_kwh": 0.0, "smart_charging": {"mode": "disabled"}})._resolved

    assert resolved.smart_charging == SmartChargingSpec(mode="disabled")
    assert App(BASE)._resolved.smart_charging is None


# --- App behaviour until the dispatch step lands ------------------------------


def test_fixed_target_is_refused_before_simulation(monkeypatch):
    app = _app(FIXED)
    monkeypatch.setattr("breos.app.run_app_simulation", lambda *a, **k: pytest.fail("simulation started"))

    with pytest.raises(ValueError, match="smart_charging fixed_target is not supported yet"):
        app.simulate()


@pytest.mark.usefixtures("_patch_weather")
def test_disabled_smart_charging_matches_no_table():
    omitted = App({**BASE, "tariff": TOU})
    disabled = _app({"mode": "disabled"})
    omitted.simulate()
    disabled.simulate()

    first, second = deepcopy(omitted.result()), deepcopy(disabled.result())
    # The configuration echo records what was given; everything else is identical.
    assert first["provenance"]["resolved_config"].pop("smart_charging") is None
    assert second["provenance"]["resolved_config"].pop("smart_charging") == {"mode": "disabled"}
    assert json.dumps(first, sort_keys=True, default=str) == json.dumps(second, sort_keys=True, default=str)


# --- The fixed-target controller ---------------------------------------------


def test_fixed_target_follows_the_resolved_tri_hourly_periods():
    # 12 January 2026, a winter weekday, in 15-minute steps.
    index = pd.date_range("2026-01-12", periods=96, freq="15min", tz=LISBON)
    tariff = _tri_tariff(index)
    labels = np.asarray(tariff.period_labels)

    instructions = resolve_instructions(TRI_SPEC, tariff)

    assert len(instructions) == 96
    assert set(labels) == {"off_peak", "mid_peak", "peak"}
    np.testing.assert_array_equal(instructions.discharge_allowed, labels == "peak")
    np.testing.assert_array_equal(instructions.reserve_fraction, np.zeros(96))
    np.testing.assert_array_equal(instructions.grid_target_fraction, np.where(labels == "off_peak", 0.6, np.nan))
    # 08:45 is mid-peak and 09:00 peak (Diretiva ERSE n.º 1/2026); mid-peak
    # neither discharges nor grid-charges.
    assert labels[35] == "mid_peak" and not instructions.discharge_allowed[35]
    assert np.isnan(instructions.grid_target_fraction[35])
    assert labels[36] == "peak" and instructions.discharge_allowed[36]
    assert labels[8] == "off_peak" and instructions.grid_target_fraction[8] == 0.6
    assert instructions.grid_charge_efficiency == 0.9
    assert instructions.grid_import_limit_w == math.inf


def test_fixed_target_stays_aligned_across_a_dst_day():
    # 29 March 2026 has 23 local hours; a UTC index keeps every instant.
    index = pd.date_range("2026-03-28", "2026-03-31", freq="15min", tz="UTC", inclusive="left")
    tariff = _tri_tariff(index)
    spec = SmartChargingSpec(
        mode="fixed_target",
        target_usable_fraction=1.0,
        charge_periods=("off_peak",),
        discharge_periods=("mid_peak", "peak"),
        grid_charge_efficiency=1.0,
        grid_import_limit_w=3000.0,
    )

    instructions = resolve_instructions(spec, tariff)

    labels = np.asarray(tariff.period_labels)
    assert len(instructions) == len(index)
    np.testing.assert_array_equal(instructions.discharge_allowed, labels != "off_peak")
    np.testing.assert_array_equal(np.isnan(instructions.grid_target_fraction), labels != "off_peak")
    assert instructions.grid_import_limit_w == 3000.0


def test_disabled_gives_no_instructions():
    assert resolve_instructions(SmartChargingSpec(mode="disabled"), None) is None


def test_the_controller_checks_its_tariff():
    index = pd.date_range("2026-01-12", periods=4, freq="15min", tz=LISBON)
    with pytest.raises(ValueError, match="needs a resolved tariff"):
        resolve_instructions(TRI_SPEC, None)
    spec = SmartChargingSpec(
        mode="fixed_target",
        target_usable_fraction=0.5,
        charge_periods=("night",),
        discharge_periods=("peak",),
        grid_charge_efficiency=0.9,
    )
    with pytest.raises(ValueError, match=r"'smart_charging\.charge_periods' has period\(s\) night"):
        resolve_instructions(spec, _tri_tariff(index))


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"mode": "greedy"}, r"'smart_charging\.mode' must be one of"),
        ({"mode": "disabled", "target_usable_fraction": 0.5}, "takes no other settings"),
        ({"mode": "fixed_target", "charge_periods": ("off_peak",)}, r"needs smart_charging\.target_usable_fraction"),
        (
            {
                "mode": "fixed_target",
                "target_usable_fraction": 0.5,
                "charge_periods": ("peak",),
                "discharge_periods": ("peak",),
                "grid_charge_efficiency": 0.9,
            },
            "share peak",
        ),
    ],
)
def test_a_spec_built_directly_stays_coherent(fields, message):
    with pytest.raises(ValueError, match=message):
        SmartChargingSpec(**fields)


def test_provenance_records_parameters_and_hashes():
    index = pd.date_range("2026-01-12", periods=96, freq="15min", tz=LISBON)
    tariff = _tri_tariff(index)
    instructions = resolve_instructions(TRI_SPEC, tariff)

    record = smart_charging_provenance(TRI_SPEC, instructions, tariff)

    assert record == {
        "mode": "fixed_target",
        "target_usable_fraction": 0.6,
        "charge_periods": ["off_peak"],
        "discharge_periods": ["peak"],
        "grid_charge_efficiency": 0.9,
        "grid_import_limit_w": None,
        "instruction_hash": instructions.instruction_hash(),
        "schedule_hash": tariff.schedule_hash,
        "terminal_convention": "physical_carry",
    }
    json.dumps(record, allow_nan=False)
    with pytest.raises(ValueError, match="96 steps but the tariff has 4"):
        smart_charging_provenance(TRI_SPEC, instructions, _tri_tariff(index[:4]))


# --- DispatchInstructions ----------------------------------------------------


def test_instructions_are_frozen_contiguous_copies():
    source = np.array([0.0, 0.0, 0.2])
    instructions = _instructions(reserve_fraction=source)
    source[2] = 0.9

    assert instructions.reserve_fraction[2] == 0.2
    for name, dtype in (
        ("discharge_allowed", np.bool_),
        ("reserve_fraction", np.float64),
        ("grid_target_fraction", np.float64),
    ):
        array = getattr(instructions, name)
        assert array.dtype == dtype and array.ndim == 1
        assert array.flags.c_contiguous and not array.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            array[0] = array[0]
    with pytest.raises(AttributeError):
        instructions.grid_charge_efficiency = 0.5
    assert len(instructions) == 3
    assert isinstance(instructions.grid_import_limit_w, float)


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"discharge_allowed": [1, 0, 0]}, TypeError, "'discharge_allowed' must be a boolean array"),
        ({"discharge_allowed": [[True, False, False]]}, ValueError, "'discharge_allowed' must be 1-D"),
        ({"reserve_fraction": ["a", "b", "c"]}, TypeError, "'reserve_fraction' must be a 1-D array"),
        ({"reserve_fraction": [0.0, 0.0]}, ValueError, "'reserve_fraction' has 2 steps but 'discharge_allowed' has 3"),
        ({"grid_target_fraction": [np.nan] * 4}, ValueError, "'grid_target_fraction' has 4 steps"),
        ({"reserve_fraction": [0.0, np.nan, 0.0]}, ValueError, "'reserve_fraction' must be finite"),
        ({"reserve_fraction": [0.0, 1.5, 0.0]}, ValueError, "'reserve_fraction' must be between 0 and 1"),
        ({"reserve_fraction": [-0.1, 0.0, 0.0]}, ValueError, "'reserve_fraction' must be between 0 and 1"),
        ({"grid_target_fraction": [np.nan, np.inf, np.nan]}, ValueError, "'grid_target_fraction' must be finite"),
        ({"grid_target_fraction": [np.nan, 1.1, np.nan]}, ValueError, "'grid_target_fraction' must be between"),
        ({"grid_target_fraction": [0.5, 0.5, np.nan]}, ValueError, r"Step 0 both allows discharge.*A8"),
        ({"grid_charge_efficiency": 0.0}, ValueError, "'grid_charge_efficiency' must be between 0"),
        ({"grid_charge_efficiency": 1.01}, ValueError, "'grid_charge_efficiency' must be between 0"),
        ({"grid_charge_efficiency": math.nan}, ValueError, "'grid_charge_efficiency' must be between 0"),
        ({"grid_charge_efficiency": True}, TypeError, "'grid_charge_efficiency' must be a number"),
        ({"grid_import_limit_w": 0.0}, ValueError, "'grid_import_limit_w' must be greater than 0"),
        ({"grid_import_limit_w": -5.0}, ValueError, "'grid_import_limit_w' must be greater than 0"),
        ({"grid_import_limit_w": math.nan}, ValueError, "'grid_import_limit_w' must be greater than 0"),
        ({"grid_import_limit_w": "5 kW"}, TypeError, "'grid_import_limit_w' must be a number"),
    ],
)
def test_instructions_reject_invalid_steps(overrides, error, message):
    with pytest.raises(error, match=message):
        _instructions(**overrides)


def test_instructions_accept_their_bounds():
    instructions = _instructions(
        reserve_fraction=[1.0, 0.0, 0.0],
        grid_target_fraction=[np.nan, 1.0, 0.0],
        grid_charge_efficiency=1,
        grid_import_limit_w=math.inf,
    )
    assert instructions.grid_charge_efficiency == 1.0
    assert instructions.grid_import_limit_w == math.inf


def test_noop_changes_nothing():
    noop = DispatchInstructions.noop(np.int64(4))

    np.testing.assert_array_equal(noop.discharge_allowed, np.ones(4, dtype=bool))
    np.testing.assert_array_equal(noop.reserve_fraction, np.zeros(4))
    assert np.isnan(noop.grid_target_fraction).all()
    assert (noop.grid_charge_efficiency, noop.grid_import_limit_w) == (1.0, math.inf)
    assert len(DispatchInstructions.noop(0)) == 0
    with pytest.raises(ValueError, match="non-negative integer"):
        DispatchInstructions.noop(-1)


def test_instruction_hash_is_stable():
    first, second = _instructions(), _instructions()

    assert first.instruction_hash() == second.instruction_hash()
    assert len(first.instruction_hash()) == 64
    assert first == second and hash(first) == hash(second)
    # Pinned: the hash must not move between runs, processes or platforms.
    assert DispatchInstructions.noop(3).instruction_hash() == (
        "74802b4fde0ea77449ea784bad6040cd744851cebb957fb0760871907d2d3cdf"
    )
    # How a NaN or a zero was made does not change the instructions.
    other_nan = np.frombuffer(np.uint64(0x7FF8000000000001).tobytes(), dtype=np.float64)[0]
    assert np.isnan(other_nan)
    assert _instructions(grid_target_fraction=[other_nan, 0.5, -np.nan]) == first
    assert _instructions(reserve_fraction=[-0.0, 0.0, 0.2]) == first


@pytest.mark.parametrize(
    "overrides",
    [
        {"discharge_allowed": [False, False, False]},
        {"reserve_fraction": [0.0, 0.0, 0.3]},
        {"grid_target_fraction": [np.nan, 0.6, np.nan]},
        {"grid_target_fraction": [np.nan, np.nan, np.nan]},
        {"grid_charge_efficiency": 0.9},
        {"grid_import_limit_w": math.inf},
    ],
)
def test_instruction_hash_sees_every_field(overrides):
    assert _instructions(**overrides).instruction_hash() != _instructions().instruction_hash()


def test_instructions_survive_pickling():
    instructions = _instructions()

    restored = pickle.loads(pickle.dumps(instructions))

    assert restored == instructions
    assert restored.instruction_hash() == instructions.instruction_hash()
    assert not restored.grid_target_fraction.flags.writeable
    assert not restored.discharge_allowed.flags.writeable
