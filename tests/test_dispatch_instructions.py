"""The per-step instruction contract of the dispatch step (ADR 0002, #178)."""

import math
import pickle

import numpy as np
import pytest

from breos.dispatch_instructions import DispatchInstructions


def _instructions(**overrides):
    fields = {
        "discharge_allowed": [True, False, False],
        "reserve_fraction": [0.0, 0.0, 0.2],
        "grid_target_fraction": [np.nan, 0.5, np.nan],
        "grid_charge_efficiency": 0.95,
        "grid_import_limit_w": 4000.0,
    }
    return DispatchInstructions(**{**fields, **overrides})


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
        (
            {"grid_target_fraction": [0.5, 0.5, np.nan]},
            ValueError,
            r"Step 0 both allows discharge.*reserve_fraction must be >= grid_target_fraction",
        ),
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


@pytest.mark.parametrize("reserve", [0.5, 0.7])
def test_overlap_is_valid_only_with_a_floor_at_or_above_the_target(reserve):
    instructions = _instructions(reserve_fraction=[reserve, 0.0, 0.2], grid_target_fraction=[0.5, 0.5, np.nan])
    assert instructions.reserve_fraction[0] >= instructions.grid_target_fraction[0]
    assert instructions.instruction_hash() != _instructions().instruction_hash()
    with pytest.raises(ValueError, match="reserve_fraction must be >= grid_target_fraction"):
        _instructions(reserve_fraction=[np.nextafter(0.5, 0.0), 0.0, 0.2], grid_target_fraction=[0.5, 0.5, np.nan])


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
