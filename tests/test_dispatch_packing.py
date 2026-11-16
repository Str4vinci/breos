"""A span packs the dispatch arguments once and restates each day; the results are the per-day packing's.

Without a vehicle or a daily controller, ``_simulate_core`` packs
``_day_arguments`` on the span's first day and then rewrites only the day's
bounds and starting state with ``_restate_day``. That is safe only while
every other argument really is fixed over the span, so this holds the
restated packing to a fresh one and a whole simulation to the per-day path.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from breos import battery
from breos._dispatch import _day_arguments, _restate_day
from breos.battery import BatteryConfig, _ResultBuffers, simulate_energy_balance
from breos.dispatch_instructions import DispatchInstructions

N_STEPS = 96 * 4


def _instructions(n: int, *, efficiency):
    hours = np.arange(n) % 96
    return DispatchInstructions(
        discharge_allowed=hours >= 40,
        reserve_fraction=np.where(hours < 24, 0.3, 0.1),
        grid_target_fraction=np.where(hours < 16, 0.6, np.nan),
        grid_charge_efficiency=efficiency,
        grid_import_limit_w=3000.0,
    )


def _config(coupling: str) -> BatteryConfig:
    extra = {"coupling": "ac", "battery_inverter_efficiency": 0.96} if coupling == "ac" else {}
    return BatteryConfig(nominal_energy_wh=5000.0, inverter_ac_capacity_w=4000.0, **extra)


def _span(coupling: str):
    rng = np.random.default_rng(7)
    pv = np.clip(rng.normal(1500.0, 900.0, N_STEPS), 0.0, None)
    load = rng.uniform(100.0, 2500.0, N_STEPS)
    temp = rng.uniform(-5.0, 35.0, N_STEPS)
    chain = None
    if coupling == "ac":
        chain = (pv * 0.24, pv * 0.01, np.zeros(N_STEPS), np.zeros(N_STEPS))
    fixed = dict(
        battery_config=_config(coupling),
        hours_per_step=0.25,
        standby_loss_per_step_wh=0.5,
        cap_wh=1000.0,
        cap_charge_wh=800.0,
        cap_discharge_wh=900.0,
        cap_stored_wh=math.inf,
        instructions=_instructions(N_STEPS, efficiency=None if coupling == "ac" else 0.92),
        ac_derate=np.ones(N_STEPS),
        curve_rows=np.zeros(N_STEPS),
        pv_chain=chain,
    )
    return _ResultBuffers(N_STEPS), pv, load, temp, fixed


def _state(energy, pv_origin, grid_origin, soh, eff_charge, eff_discharge):
    return dict(
        Battery_Energy_Wh=energy,
        Battery_PV_Origin_Energy_Wh=pv_origin,
        Battery_Grid_Origin_Energy_Wh=grid_origin,
        battery_soh_decimal=soh,
        eff_charge=eff_charge,
        eff_discharge=eff_discharge,
    )


@pytest.mark.parametrize("coupling", ["dc", "ac"])
def test_a_restated_packing_is_the_fresh_packing_of_that_day(coupling):
    out, pv, load, temp, fixed = _span(coupling)
    packed = list(_day_arguments(out, pv, load, temp, 0, 96, **fixed, **_state(2500.0, 2000.0, 500.0, 1.0, 0.95, 0.95)))
    # numpy scalars, as aging hands them over, and an int energy all coerce as a fresh packing does.
    day = (96, 192, np.float64(1234.5), 1000, np.float64(234.5), np.float64(0.987654321), 0.9431, np.float64(0.9512))
    _restate_day(packed, *day)
    fresh = _day_arguments(out, pv, load, temp, 96, 192, **fixed, **_state(*day[2:]))

    assert len(packed) == len(fresh)
    for position, (restated, expected) in enumerate(zip(packed, fresh)):
        assert type(restated) is type(expected), position
        if isinstance(expected, np.ndarray):
            assert restated is expected or np.array_equal(restated, expected, equal_nan=True), position
        elif isinstance(expected, float) and math.isnan(expected):
            assert math.isnan(restated), position
        else:
            assert restated == expected, position


def _simulate(backend: str, coupling: str, instructions: bool) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=96 * 9, freq="15min", tz="UTC")
    hours = (np.arange(len(index)) % 96) / 4.0
    pv = np.clip(np.sin((hours - 6.0) / 12.0 * np.pi), 0.0, None) * 3500.0
    return simulate_energy_balance(
        pv_dc=pd.Series(pv, index=index),
        houseload=pd.DataFrame({"Load": 600.0 + 400.0 * np.cos(hours / 24.0 * 2 * np.pi)}, index=index),
        battery_config=_config(coupling),
        freq="15min",
        temperature_series=pd.Series(np.linspace(-2.0, 30.0, len(index)), index=index),
        initial_energy_wh=1000.0,
        dispatch_instructions=(
            _instructions(len(index), efficiency=None if coupling == "ac" else 0.92) if instructions else None
        ),
        execution_backend=backend,
    )[0]


@pytest.mark.parametrize("backend", ["python", "numba"])
@pytest.mark.parametrize(("coupling", "instructions"), [("dc", False), ("dc", True), ("ac", True)])
def test_the_packed_span_matches_the_per_day_packing(monkeypatch, backend, coupling, instructions):
    if backend == "numba":
        pytest.importorskip("numba", reason="the compiled backend needs the breos[fast] extra")
    restated = []
    real_restate = battery._restate_day

    def counting_restate(arguments, *day):
        restated.append(day[0])
        real_restate(arguments, *day)

    monkeypatch.setattr(battery, "_restate_day", counting_restate)
    packed = _simulate(backend, coupling, instructions)
    # Every day after the first was restated, so the packed path really ran.
    assert restated == [96 * day for day in range(1, 9)]

    monkeypatch.setattr(battery, "_resolve_packed_day", lambda dispatch_day: None)
    per_day = _simulate(backend, coupling, instructions)
    # The days age, so the restated health and efficiency differ from the first day's.
    assert per_day["Battery_SOH"].iloc[-1] < per_day["Battery_SOH"].iloc[0]
    pd.testing.assert_frame_equal(packed, per_day, check_exact=True)
