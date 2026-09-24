"""Parity and branch coverage for the optional compiled dispatch backend.

Two things are asserted here, and both matter. The first is that the compiled
backend reproduces the Python reference exactly, column by column and timestep
by timestep. The second is that each scenario actually reaches the branch it
was built for: a parity test over a scenario that never exercises the
discharge cap passes without testing anything, so every branch case asserts
its own precondition before asserting parity.

Typical sizing studies do not cover this ground. PV-only designs have no
battery, and battery designs often reach their power limits only through a
symmetric ``power_limit_c_rate``, which caps the stored energy rather than
exercising the absolute caps and the binding behaviour the scenarios below
are built to vary.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import breos.battery as battery_module
from breos._numba_dispatch import _build_kernel
from breos.battery import (
    _STATE_ROW_INDEX,
    BatteryConfig,
    simulate_energy_balance,
    simulate_energy_balance_summary,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "parity"))

from harness import FREQ, SCENARIOS, build  # noqa: E402

numba = pytest.importorskip("numba", reason="the compiled backend needs the breos[fast] extra")


def _run(name: str, backend: str):
    pv, load, temp, cfg, sim_kwargs = build(name)
    return simulate_energy_balance(
        pv_dc=pv,
        houseload=load,
        battery_config=BatteryConfig(**cfg),
        freq=FREQ,
        temperature_series=temp,
        return_degradation_state=True,
        execution_backend=backend,
        **sim_kwargs,
    )


def _assert_identical(name: str, python_out, numba_out) -> None:
    py_df, py_total, py_summary, py_cost, py_reps, py_deg = python_out[:6]
    nb_df, nb_total, nb_summary, nb_cost, nb_reps, nb_deg = numba_out[:6]

    assert list(py_df.columns) == list(nb_df.columns)
    for column in py_df.columns:
        if column == "Datetime":
            continue
        left = py_df[column].to_numpy()
        right = nb_df[column].to_numpy()
        # Exact, not approximate: a tolerance here would hide the operation
        # that caused a difference, which is the thing worth knowing.
        assert np.array_equal(left, right), (
            f"{name}: column {column} differs at "
            f"{int(np.flatnonzero(left != right)[0])} "
            f"(max abs {np.max(np.abs(left - right)):.6e})"
        )

    assert py_total == nb_total
    assert py_cost == nb_cost
    assert py_reps == nb_reps
    for column in py_deg.columns:
        if column == "Datetime":
            continue
        assert np.array_equal(py_deg[column].to_numpy(), nb_deg[column].to_numpy()), f"{name}: degradation {column}"
    for column in py_summary.columns:
        assert py_summary[column].iloc[0] == nb_summary[column].iloc[0], f"{name}: summary {column}"
    assert python_out[6] == numba_out[6]


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_numba_matches_python_exactly(scenario):
    _assert_identical(scenario, _run(scenario, "python"), _run(scenario, "numba"))


def test_single_day_matches():
    """Validation step 2: one compiled day against one reference day."""
    _assert_identical("one_day", _run("one_day", "python"), _run("one_day", "numba"))


def test_numba_pv_only_summary_uses_common_vectorized_path(monkeypatch):
    pv, load, temp, cfg, sim_kwargs = build("no_battery")
    vectorized_calls = []
    reduced_buffer_calls = []
    original_dispatch = battery_module._dispatch_no_battery_vectorized
    original_buffers = battery_module._PvOnlySummaryBuffers

    def recording_dispatch(*args, **kwargs):
        vectorized_calls.append(len(args[1]))
        return original_dispatch(*args, **kwargs)

    def recording_buffers(n_steps):
        reduced_buffer_calls.append(n_steps)
        return original_buffers(n_steps)

    monkeypatch.setattr(battery_module, "_dispatch_no_battery_vectorized", recording_dispatch)
    monkeypatch.setattr(battery_module, "_PvOnlySummaryBuffers", recording_buffers)
    result = simulate_energy_balance_summary(
        pv_dc=pv,
        houseload=load,
        battery_config=BatteryConfig(**cfg),
        freq=FREQ,
        temperature_series=temp,
        execution_backend="numba",
        **sim_kwargs,
    )

    assert vectorized_calls == [35040]
    assert reduced_buffer_calls == [35040]
    assert result.n_steps == 35040


def test_trailing_partial_day_matches():
    """A day window that does not close must still dispatch identically."""
    python_out = _run("partial_day", "python")
    assert len(python_out[0]) % 96 != 0, "scenario no longer has a trailing partial day"
    # One closed day only: the stub contributes no degradation row.
    assert len(python_out[5]) == 1
    _assert_identical("partial_day", python_out, _run("partial_day", "numba"))


def test_discharge_power_limit_binds_and_matches():
    df = _run("discharge_limited", "python")[0]
    cap_ac_wh = 900.0 * 0.25
    delivered = df["Battery_AC_To_Load"].to_numpy() * 0.25
    assert delivered.max() > 0.0
    # The cap is solved on the shared inverter operating point, so the binding
    # steps sit just under it rather than exactly on it.
    assert np.isclose(delivered.max(), cap_ac_wh, rtol=1e-6), delivered.max()
    assert (delivered > cap_ac_wh * 0.999).sum() > 50, "discharge cap barely binds"
    _assert_identical("discharge_limited", _run("discharge_limited", "python"), _run("discharge_limited", "numba"))


def test_charge_power_limit_binds_and_matches():
    df = _run("charge_limited", "python")[0]
    cap_wh = 700.0 * 0.25
    charged = df["Battery_Charge_Input"].to_numpy() * 0.25
    assert np.isclose(charged.max(), cap_wh, rtol=1e-9), charged.max()
    assert (charged >= cap_wh - 1e-9).sum() > 50, "charge cap barely binds"
    _assert_identical("charge_limited", _run("charge_limited", "python"), _run("charge_limited", "numba"))


def test_state_of_charge_saturates_at_both_bounds():
    df = _run("saturating", "python")[0]
    soc = df["Battery_SOC_Normalized"].to_numpy()
    assert (soc >= 1.0).sum() > 20, "upper SOC bound never reached"
    assert (soc <= 0.0).sum() > 20, "lower SOC bound never reached"
    assert df["PV_DC_Curtailed"].to_numpy().max() > 0.0, "inverter never clips"
    _assert_identical("saturating", _run("saturating", "python"), _run("saturating", "numba"))


def test_capacity_window_loss_occurs_and_matches():
    df = _run("baseline", "python")[0]
    assert df["Capacity_Window_Loss"].to_numpy().max() > 0.0, "capacity window never derates"
    _assert_identical("baseline", _run("baseline", "python"), _run("baseline", "numba"))


def test_both_inverter_loss_channels_are_populated():
    df = _run("baseline", "python")[0]
    direct = df["PV_Direct_Inverter_Loss"].to_numpy()
    battery = df["Battery_Inverter_Loss"].to_numpy()
    assert direct.max() > 0.0, "no direct-solar inverter loss"
    assert battery.max() > 0.0, "no battery inverter loss"
    # They are separate channels, not one number reported twice.
    assert not np.array_equal(direct, battery)
    assert np.allclose(df["Inverter_Loss"].to_numpy(), direct + battery)


def test_replacement_at_threshold_matches():
    python_out = _run("replacement", "python")
    df, deg = python_out[0], python_out[5]
    assert python_out[4] > 0, "scenario never replaces the pack"
    replaced_at = np.flatnonzero(df["Battery_Replaced"].to_numpy())
    assert replaced_at.size == python_out[4]
    # A replacement fires on the day the pack crosses end of life, and the
    # closing step's recorded state is rewritten to the fresh pack.
    for step in replaced_at:
        assert df["Battery_SOH"].to_numpy()[step] == 100.0
        assert df["Battery_Replacement_Energy_Added"].to_numpy()[step] > 0.0
    # The daily record is written after the replacement check, so a
    # replacement day reports the fresh pack at 100%. The minimum recorded SOH
    # is therefore the last value that did *not* trigger, and it sitting just
    # above the 99.5% threshold is what makes this a decision-boundary case.
    eol_percent = 99.5
    lowest_surviving = deg["SOH"].to_numpy().min()
    assert eol_percent < lowest_surviving < eol_percent + 0.05, lowest_surviving
    _assert_identical("replacement", python_out, _run("replacement", "numba"))


def test_carried_state_between_years_matches():
    python_out = _run("carried_state", "python")
    df = python_out[0]
    assert df["Battery_Energy_Beginning"].iloc[0] > 0.0
    assert df["Battery_PV_Origin_Energy_Beginning"].iloc[0] > 0.0
    _assert_identical("carried_state", python_out, _run("carried_state", "numba"))


def test_cycle_counting_boundary_is_unaffected_by_backend():
    """Degradation is Python-only, so its inputs must arrive bit-identical."""
    py_deg = _run("baseline", "python")[5]
    nb_deg = _run("baseline", "numba")[5]
    for column in ("Cumulative_FEC", "Cumulative_Cycle_Degradation", "Cumulative_Calendar_Degradation", "SOH"):
        assert np.array_equal(py_deg[column].to_numpy(), nb_deg[column].to_numpy()), column
    assert py_deg["Cumulative_FEC"].iloc[-1] > 0.0


def test_summary_path_matches_detailed_path_under_numba():
    pv, load, temp, cfg, sim_kwargs = build("baseline")
    common = dict(
        pv_dc=pv,
        houseload=load,
        battery_config=BatteryConfig(**cfg),
        freq=FREQ,
        temperature_series=temp,
        **sim_kwargs,
    )
    df = simulate_energy_balance(**common, execution_backend="numba")[0]
    summary = simulate_energy_balance_summary(**common, execution_backend="numba")
    for column in df.columns:
        if column == "Datetime":
            continue
        assert float(df[column].sum()) == summary.column_sums[column], column


# --- Temperature range coverage -------------------------------------------
#
# The indoor-temperature model is a Series-to-Series transform applied before
# the loop, so the kernel cannot tell a remapped series from raw weather or a
# pinned constant. What it does branch on is the *value*: lfp_capacity_factor
# switches at 25 C and at 0 C and saturates at 0.5, and compute_cell_temperature
# runs every step. These cases pin bit-identity across the realistic range,
# including a fixed 25 C control and the clamp boundaries of the indoor model.


def _temperature_variant(values: np.ndarray):
    pv, load, _temp, cfg, sim_kwargs = build("baseline")
    series = pd.Series(values, index=pv.index)
    common = dict(
        pv_dc=pv,
        houseload=load,
        battery_config=BatteryConfig(**cfg),
        freq=FREQ,
        temperature_series=series,
        return_degradation_state=True,
        **sim_kwargs,
    )
    return (
        simulate_energy_balance(**common, execution_backend="python"),
        simulate_energy_balance(**common, execution_backend="numba"),
    )


def _hours(index) -> np.ndarray:
    return index.hour.to_numpy() + index.minute.to_numpy() / 60.0


@pytest.mark.parametrize(
    "label, make",
    [
        # The fixed-temperature control pins 25 C; the harness's varying series
        # never exercises it.
        ("pinned 25C", lambda idx: np.full(len(idx), 25.0)),
        ("exactly at 0C boundary", lambda idx: np.full(len(idx), 0.0)),
        ("just below 25C boundary", lambda idx: np.full(len(idx), np.nextafter(25.0, 0.0))),
        ("sub-zero winter", lambda idx: -15.0 + 4.0 * np.cos(_hours(idx) / 24.0 * 2 * np.pi)),
        # Cold enough to saturate the derating floor at 0.5.
        ("extreme cold at derating floor", lambda idx: np.full(len(idx), -60.0)),
        ("above the indoor ceiling", lambda idx: np.full(len(idx), 40.0)),
        ("indoor floor boundary", lambda idx: np.full(len(idx), 15.0)),
        ("indoor ceiling boundary", lambda idx: np.full(len(idx), 35.0)),
        ("hard intraday swing", lambda idx: 20.0 * np.sin(_hours(idx) / 24.0 * 2 * np.pi * 3)),
    ],
)
def test_temperature_range_parity(label, make):
    pv, _load, _temp, _cfg, _sim = build("baseline")
    python_out, numba_out = _temperature_variant(make(pv.index))
    _assert_identical(f"temperature::{label}", python_out, numba_out)


def test_indoor_temperature_model_output_parity():
    """The remapped distribution, end to end, not just a synthetic curve."""
    from breos.battery import apply_indoor_temperature_model

    pv, _load, _temp, _cfg, _sim = build("baseline")
    idx = pv.index
    # The model is clamp(0.3 * outdoor + 15.4, 15, 35), so the ceiling needs an
    # outdoor value near 65 C. That is not a plausible ambient temperature, but
    # the point here is to land the clamp exactly on both bounds, where a min or
    # max could order differently between backends.
    outdoor = pd.Series(
        -25.0 + 100.0 * ((idx.dayofyear.to_numpy() % 180) / 180.0) + 9.0 * np.sin(_hours(idx) / 24.0 * 2 * np.pi),
        index=idx,
    )
    indoor = apply_indoor_temperature_model(outdoor)
    assert (indoor.to_numpy() == 15.0).any(), "indoor floor never reached"
    assert (indoor.to_numpy() == 35.0).any(), "indoor ceiling never reached"

    python_out, numba_out = _temperature_variant(indoor.to_numpy())
    _assert_identical("temperature::indoor model", python_out, numba_out)


def test_derating_floor_is_actually_reached_at_extreme_cold():
    """Confirm the -60 C case is not passing vacuously."""
    from breos.battery import lfp_capacity_factor

    assert lfp_capacity_factor(-60.0) == 0.5
    python_out, _ = _temperature_variant(np.full(35040, -60.0))
    warm_out, _ = _temperature_variant(np.full(35040, 25.0))
    # A saturated derating window stores materially less than the 25 C case.
    assert python_out[0]["Battery_Energy"].max() < warm_out[0]["Battery_Energy"].max() * 0.6


_FOLDED_SQUARE_SEED = 20260915
_FOLDED_SQUARE_ATTEMPTS = 4_000_000


def _reference_square(value: float) -> float:
    """The spelling ``calculate_dc_ac_power`` uses; CPython routes it to libm ``pow``."""
    return value**2


def _folded_square(value: float) -> float:
    """The spelling LLVM folds a constant exponent down to."""
    return value * value


def _pvwatts_ac_power(dc_power: float, ac_rating: float, efficiency: float, square) -> float:
    """Mirror of the reference part-load curve with the square left pluggable.

    The square is the only thing that differs between the two spellings, so
    running a candidate input through both says whether the disagreement
    reaches the AC power the parity assertion compares or is swallowed on the
    way by the curve and the clamps.
    """
    from breos.inverter import (
        PVWATTS_CURVE_CONSTANT,
        PVWATTS_CURVE_LINEAR,
        PVWATTS_CURVE_QUADRATIC,
        PVWATTS_REFERENCE_EFFICIENCY,
    )

    pdc0 = ac_rating / efficiency
    dc_used = min(dc_power, pdc0)
    zeta = dc_used / pdc0
    return max(
        0.0,
        min(
            dc_used,
            ac_rating,
            (efficiency / PVWATTS_REFERENCE_EFFICIENCY)
            * pdc0
            * (PVWATTS_CURVE_QUADRATIC * square(zeta) + PVWATTS_CURVE_LINEAR * zeta + PVWATTS_CURVE_CONSTANT),
        ),
    )


def _find_folded_square_discriminator(ac_rating: float, efficiency: float) -> float | None:
    """Search for a ``dc_power`` whose zeta separates libm ``pow`` from ``x * x``.

    Which inputs separate the two is a property of the platform's libm, so a
    pinned literal only discriminates on the machine it was found on. The
    input this test used to carry separates them under glibc and not under
    Apple's libm, where all three spellings agree. Both libms do have
    separating inputs -- on the order of one zeta in a thousand -- so search
    for one on the machine running the test rather than ship one.

    A candidate clears three bars. The reference spelling ``zeta ** 2`` has to
    agree with the explicit ``math.pow`` the kernel calls, or the backends
    would part company for a reason this guard is not about. The folded square
    has to disagree with it, which is the fold being observable at all. And
    the disagreement has to survive into the AC power, which costs about two
    orders of magnitude: roughly one candidate in 10^5 gets through.
    """
    pdc0 = ac_rating / efficiency
    rng = random.Random(_FOLDED_SQUARE_SEED)
    for _ in range(_FOLDED_SQUARE_ATTEMPTS):
        # Below pdc0 so zeta stays on the curve, and clear of the low end where
        # the curve is clamped flat at zero and discriminates nothing.
        dc_power = rng.uniform(0.05 * pdc0, pdc0)
        zeta = min(dc_power, pdc0) / pdc0
        if _reference_square(zeta) != math.pow(zeta, 2.0) or _reference_square(zeta) == _folded_square(zeta):
            continue
        if _pvwatts_ac_power(dc_power, ac_rating, efficiency, _reference_square) != _pvwatts_ac_power(
            dc_power, ac_rating, efficiency, _folded_square
        ):
            return dc_power
    return None


def test_zeta_squared_must_use_libm_pow_not_the_folded_square():
    """Pin the operation the bit-identity claim depends on.

    CPython evaluates ``zeta ** 2`` as a libm ``pow`` call. LLVM rewrites a
    constant-exponent ``pow`` into ``x * x``, and for some inputs libm's
    ``pow`` and the correctly rounded square differ by one ULP. Simplifying
    the kernel back to ``zeta ** 2`` would reintroduce that difference, so
    this test drives the kernel with an input where it shows up.

    BREOS promises no bit identity across platforms or libm versions. It does
    promise it between the Python and numba backends on one machine, and that
    is the claim this defends. A trajectory-level test will not do it: the bad
    zeta rate is around 2.5e-6 and the outer clamps absorb most of what lands,
    so whole simulated years stay bit-identical with the guard removed.
    """
    from breos.inverter import calculate_dc_ac_power

    ac_rating, efficiency = 5400.0, 0.96
    dc_power = _find_folded_square_discriminator(ac_rating, efficiency)
    if dc_power is None:
        pytest.skip(
            f"no input among {_FOLDED_SQUARE_ATTEMPTS} candidates separates this libm's pow from the "
            "folded square; glibc and Apple libm both have such inputs, so this is worth looking into "
            "rather than accepting"
        )

    # The premise: on this input the two spellings genuinely disagree, and the
    # disagreement reaches the value asserted below.
    assert _pvwatts_ac_power(dc_power, ac_rating, efficiency, _reference_square) != _pvwatts_ac_power(
        dc_power, ac_rating, efficiency, _folded_square
    ), "input no longer distinguishes libm pow from the folded square"

    kernel = _build_kernel()
    matrix = np.zeros((37, 1))
    pv = np.array([dc_power * 4.0])  # Wh at a 15-minute step
    load = np.zeros(1)
    temp = np.full(1, 25.0)
    kernel(
        matrix,
        pv,
        load,
        temp,
        0,
        1,
        0.0,
        0.0,
        False,
        0.0,
        1.0,
        100.0,
        0.9,
        0.1,
        0.0,
        0.95,
        0.95,
        efficiency,
        np.inf,
        np.inf,
        ac_rating,
        False,
        0.05,
        0.25,
        2.0,
        1.0,
        np.inf,
    )
    reference = calculate_dc_ac_power(dc_power, ac_rating, efficiency)
    assert matrix[_STATE_ROW_INDEX["pv_production"], 0] * 0.25 == pytest.approx(
        dc_power - reference.clipping_loss_dc_w - reference.conversion_loss_w, abs=0.0, rel=0.0
    )
