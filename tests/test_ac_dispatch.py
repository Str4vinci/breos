"""The AC-coupled dispatch step (ADR 0004 C3-C8): reference cases AC1-AC7 and invariants P1-P6."""

import importlib.util
import math
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from breos._daily_targets import _DayEvaluator
from breos.battery import BatteryConfig, _simulate_core, simulate_energy_balance
from breos.dispatch_instructions import DispatchInstructions
from tests.energy_conservation import assert_energy_conservation, assert_origin_reconciliation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "parity"))
from harness import AC_SCENARIOS, RESOLUTIONS, build, build_ac  # noqa: E402

_BACKENDS = [
    "python",
    pytest.param(
        "numba", marks=pytest.mark.skipif(not importlib.util.find_spec("numba"), reason="numba not installed")
    ),
]

# The reference cases' pack (ADR 0004, "Reference cases").
REFERENCE = dict(
    nominal_energy_wh=10000.0,
    min_soc=0.0,
    max_soc=1.0,
    charge_efficiency=0.95,
    discharge_efficiency=0.95,
    inverter_efficiency=0.96,
    standby_loss_wh=0.0,
    thermal_resistance_k_per_w=0.0,
    enable_replacement=False,
)
ETA = 0.96  # both inverters' nominal efficiency
EFF = 0.95  # the pack's charge and discharge efficiency


# --- an independent PVWatts reference ----------------------------------------


def _pvwatts(dc, rating, efficiency):
    """AC out of the PVWatts curve for DC in, with the AC ``rating``; flat when the rating is infinite."""
    if dc <= 0.0:
        return 0.0
    if math.isinf(rating):
        return dc * efficiency
    pdc0 = rating / efficiency
    used = min(dc, pdc0)
    zeta = used / pdc0
    return min(used, rating, efficiency / 0.9637 * pdc0 * (-0.0162 * zeta * zeta + 0.9858 * zeta - 0.0059))


def _pvwatts_input(ac, rating, efficiency):
    """The DC :func:`_pvwatts` needs for ``ac``, by bisection to the last bit."""
    if math.isinf(rating):
        return ac / efficiency
    lower, upper = 0.0, rating / efficiency
    for _ in range(200):
        middle = (lower + upper) / 2.0
        if _pvwatts(middle, rating, efficiency) < ac:
            lower = middle
        else:
            upper = middle
    return upper


def _charge(ac, rating):
    """Battery inverter AC to DC: the curve run with input limit ``rating`` (C4)."""
    return _pvwatts(ac, rating * ETA, ETA)


def _charge_input(dc, rating):
    return _pvwatts_input(dc, rating * ETA, ETA)


# --- running a reference case ------------------------------------------------


def _run(pv, load, initial_energy_wh, *, coupling="ac", freq="h", instructions=None, backend="python", **config):
    index = pd.date_range("2025-06-01", periods=len(pv), freq=freq, tz="UTC")
    extra = {"coupling": "ac", "battery_inverter_efficiency": ETA} if coupling == "ac" else {}
    battery = BatteryConfig(**{**REFERENCE, **extra, **config})
    frame = simulate_energy_balance(
        pv_dc=pd.Series(pv, index=index, dtype=float),
        houseload=pd.DataFrame({"Load": load}, index=index, dtype=float),
        battery_config=battery,
        freq=freq,
        temperature_series=pd.Series(25.0, index=index),
        initial_energy_wh=initial_energy_wh,
        dispatch_instructions=instructions,
        execution_backend=backend,
    )[0]
    assert_energy_conservation(frame, battery)
    assert_origin_reconciliation(frame, 1.0 if freq == "h" else 0.25)
    return frame


def _charge_target(n, *, discharge, reserve, efficiency=None, limit_w=math.inf, target=0.4):
    return DispatchInstructions(
        discharge_allowed=np.full(n, discharge),
        reserve_fraction=np.full(n, reserve),
        grid_target_fraction=np.full(n, target),
        grid_charge_efficiency=efficiency,
        grid_import_limit_w=limit_w,
    )


def _check(frame, expected, step=0):
    for column, value in expected.items():
        np.testing.assert_allclose(frame[column].iloc[step], value, rtol=1e-9, atol=1e-9, err_msg=column)


# --- reference cases ---------------------------------------------------------


@pytest.mark.parametrize("backend", _BACKENDS)
def test_ac1_pv_through_the_battery_and_back(backend):
    frame = _run([3000, 0], [1000, 1500], 0.0, backend=backend)
    _check(
        frame,
        {
            "PV_Direct_Inverter_Loss": 120.0,
            "PV_AC_To_Load": 1000.0,
            "PV_DC_To_Battery": 0.0,
            "PV_AC_To_Battery": 1880.0,
            "PV_Charge_Conversion_Loss": 75.2,
            "Battery_Charge_Input": 1804.8,
            "Battery_Charge_Stored": 1714.56,
            "PV_Origin_Battery_Charge_Stored": 1714.56,
            "Battery_Energy": 1714.56,
        },
    )
    draw = 1500.0 / ETA / EFF
    _check(
        frame,
        {
            "Battery_AC_To_Load": 1500.0,
            "Battery_Discharge_DC": draw,
            "Battery_Inverter_Loss": 62.5,
            "PV_Origin_Battery_AC_To_Load": 1500.0,
            "Battery_Energy": 1714.56 - draw,
        },
        step=1,
    )
    dc = _run([3000, 0], [1000, 1500], 0.0, coupling="dc", backend=backend)
    # From the same surplus, AC coupling stores eta * eta_b of what DC coupling stores.
    np.testing.assert_allclose(
        frame["Battery_Charge_Stored"].iloc[0], dc["Battery_Charge_Stored"].iloc[0] * ETA * ETA, rtol=1e-12
    )


@pytest.mark.parametrize("battery_inverter_w", [None, 1500.0])
def test_ac2_clipping_is_not_recovered(battery_inverter_w):
    frame = _run([3000], [0], 5000.0, inverter_ac_capacity_w=2000.0, battery_inverter_ac_capacity_w=battery_inverter_w)
    into_battery = 2000.0 if battery_inverter_w is None else 1500.0
    _check(
        frame,
        {
            "PV_DC_Curtailed": 3000.0 - 2000.0 / ETA,
            "PV_Direct_Inverter_Loss": 2000.0 / ETA - 2000.0,
            "PV_AC_To_Battery": into_battery,
            "PV_Charge_Conversion_Loss": into_battery * (1 - ETA),
            "PV_AC_Export": 2000.0 - into_battery,
            "Battery_Charge_Stored": into_battery * ETA * EFF,
            "Battery_Energy": 5000.0 + into_battery * ETA * EFF,
        },
    )
    dc = _run([3000], [0], 5000.0, coupling="dc", inverter_ac_capacity_w=2000.0)
    assert dc["PV_DC_Curtailed"].iloc[0] == 0.0
    assert dc["PV_DC_To_Battery"].iloc[0] == 3000.0


def test_ac3_night_grid_charge_matches_dc_coupling():
    ac = _run([0], [500], 2000.0, instructions=_charge_target(1, discharge=False, reserve=0.0))
    dc = _run(
        [0], [500], 2000.0, coupling="dc", instructions=_charge_target(1, discharge=False, reserve=0.0, efficiency=ETA)
    )
    grid_ac = 2000.0 / EFF / ETA
    expected = {
        "Grid_AC_To_Battery": grid_ac,
        "Grid_Charge_Conversion_Loss": grid_ac * (1 - ETA),
        "Battery_Charge_Input": 2000.0 / EFF,
        "Battery_Charge_Stored": 2000.0,
        "Import_From_Grid": 500.0 + grid_ac,
        "Battery_Energy": 4000.0,
    }
    _check(ac, expected)
    _check(dc, expected)


def test_ac4_saturated_pv_inverter_and_grid_charge():
    instructions = _charge_target(1, discharge=False, reserve=0.0)
    ac = _run([2500], [2000], 2000.0, instructions=instructions, inverter_ac_capacity_w=2000.0)
    _check(
        ac,
        {
            "PV_AC_To_Load": 2000.0,
            "PV_DC_Curtailed": 2500.0 - 2000.0 / ETA,
            "Grid_AC_To_Battery": 2000.0 / EFF / ETA,
            "Import_From_Grid": 2000.0 / EFF / ETA,
            "Battery_Energy": 4000.0,
        },
    )
    dc = _run(
        [2500],
        [2000],
        2000.0,
        coupling="dc",
        instructions=_charge_target(1, discharge=False, reserve=0.0, efficiency=ETA),
        inverter_ac_capacity_w=2000.0,
    )
    _check(dc, {"PV_DC_To_Battery": 2500.0 - 2000.0 / ETA, "Grid_AC_To_Battery": 0.0, "Import_From_Grid": 0.0})


@pytest.mark.parametrize("backend", _BACKENDS)
def test_ac5_battery_inverter_rating_limits_discharge(backend):
    frame = _run([0], [3000], 5000.0, battery_inverter_ac_capacity_w=2000.0, backend=backend)
    dc = 2000.0 / ETA
    expected = {
        "Battery_AC_To_Load": 2000.0,
        "Battery_Inverter_Loss": dc - 2000.0,
        "Battery_Discharge_DC": dc / EFF,
        "Battery_Discharge_Loss": dc / EFF - dc,
        "Import_From_Grid": 1000.0,
        "Battery_Energy": 5000.0 - dc / EFF,
    }
    _check(frame, expected)
    # A hybrid inverter of the same rating and no PV gives the same numbers.
    _check(_run([0], [3000], 5000.0, coupling="dc", inverter_ac_capacity_w=2000.0, backend=backend), expected)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_ac6_light_load_follows_the_part_load_curve(backend):
    frame = _run([0], [300], 5000.0, battery_inverter_ac_capacity_w=3000.0, backend=backend)
    dc = _pvwatts_input(300.0, 3000.0, ETA)
    assert dc == pytest.approx(324.751952, abs=1e-6)
    _check(
        frame,
        {
            "Battery_AC_To_Load": 300.0,
            "Battery_Inverter_Loss": dc - 300.0,
            "Battery_Discharge_DC": dc / EFF,
            "Battery_Energy": 5000.0 - dc / EFF,
        },
    )
    assert frame["Battery_AC_To_Load"].iloc[0] <= 300.0


def _ac7_expected(rating):
    """C7 for the AC7 step, from the independent curve above."""
    pv_ac = _pvwatts(2000.0, 4000.0, ETA)
    pv_to_battery = pv_ac - 1000.0
    dc_needed = 2000.0 / EFF
    required = _charge_input(dc_needed, rating) - pv_to_battery if dc_needed < rating * ETA else math.inf
    grid_ac = min(required, rating - pv_to_battery)
    total = pv_to_battery + grid_ac
    dc_total = _charge(total, rating)
    grid_dc = dc_total * grid_ac / total
    pv_dc = dc_total - grid_dc
    return {
        "PV_Direct_Inverter_Loss": 2000.0 - pv_ac,
        "PV_AC_To_Load": 1000.0,
        "PV_AC_To_Battery": pv_to_battery,
        "PV_AC_Export": 0.0,
        "Grid_AC_To_Battery": grid_ac,
        "PV_Charge_Conversion_Loss": pv_to_battery - pv_dc,
        "Grid_DC_To_Battery": grid_dc,
        "Grid_Charge_Conversion_Loss": grid_ac - grid_dc,
        "Battery_Charge_Input": dc_total,
        "Battery_Charge_Stored": dc_total * EFF,
        "PV_Origin_Battery_Charge_Stored": pv_dc * EFF,
        "Grid_Origin_Battery_Charge_Stored": grid_dc * EFF,
        "Import_From_Grid": grid_ac,
        "Battery_Energy": 4000.0 if required <= rating - pv_to_battery else 2000.0 + dc_total * EFF,
    }


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize(
    ("rating", "table"),
    [
        (3000.0, {"Grid_AC_To_Battery": 1263.949329, "PV_Charge_Conversion_Loss": 34.941257, "Battery_Energy": 4000}),
        (2000.0, {"Grid_AC_To_Battery": 1075.950939, "PV_Charge_Conversion_Loss": 36.961962, "Battery_Energy": 3824}),
    ],
)
def test_ac7_pv_and_grid_charge_in_one_step(backend, rating, table):
    instructions = _charge_target(1, discharge=True, reserve=0.4)
    frame = _run(
        [2000],
        [1000],
        2000.0,
        instructions=instructions,
        inverter_ac_capacity_w=4000.0,
        battery_inverter_ac_capacity_w=rating,
        backend=backend,
    )
    expected = _ac7_expected(rating)
    _check(frame, expected)
    for column, value in table.items():
        assert frame[column].iloc[0] == pytest.approx(value, abs=1e-6), column
    if rating == 3000.0:
        # The target is reached: hold_target lands on it exactly.
        assert frame["Battery_Energy"].iloc[0] == 4000.0
    else:
        # The rating binds: the battery inverter runs at full input and stops short.
        assert frame["PV_AC_To_Battery"].iloc[0] + frame["Grid_AC_To_Battery"].iloc[0] == pytest.approx(2000.0)
        assert frame["Battery_Energy"].iloc[0] == pytest.approx(3824.0)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_ac7_at_15_minutes_takes_the_rating_per_step(backend):
    instructions = _charge_target(1, discharge=True, reserve=0.4)
    hourly = _run(
        [2000],
        [1000],
        2000.0,
        instructions=instructions,
        inverter_ac_capacity_w=4000.0,
        battery_inverter_ac_capacity_w=3000.0,
        backend=backend,
    )
    quarter = _run(
        [8000],
        [4000],
        2000.0,
        freq="15min",
        instructions=instructions,
        inverter_ac_capacity_w=16000.0,
        battery_inverter_ac_capacity_w=12000.0,
        backend=backend,
    )
    for column in _ac7_expected(3000.0):
        if column == "Battery_Energy":
            assert quarter[column].iloc[0] == hourly[column].iloc[0]
        else:
            np.testing.assert_allclose(quarter[column].iloc[0], 4.0 * hourly[column].iloc[0], rtol=1e-12, atol=1e-9)


# --- the battery's own limits (flat inverters, so every value is by hand) --------


@pytest.mark.parametrize("backend", _BACKENDS)
def test_the_charge_input_limit_binds_pv_charging(backend):
    frame = _run([3000], [0], 2000.0, max_charge_power_w=1000.0, backend=backend)
    _check(
        frame,
        {
            "PV_AC_To_Battery": 1000.0 / ETA,
            "Battery_Charge_Input": 1000.0,
            "Battery_Charge_Stored": 1000.0 * EFF,
            "PV_AC_Export": 3000.0 * ETA - 1000.0 / ETA,
        },
    )


@pytest.mark.parametrize("backend", _BACKENDS)
def test_the_charge_input_limit_is_shared_by_pv_and_grid(backend):
    # PV's 460 W of surplus AC comes first; the grid gets what is left of 1 kW DC.
    instructions = _charge_target(1, discharge=False, reserve=0.0, target=0.9)
    frame = _run([1000], [500], 2000.0, instructions=instructions, max_charge_power_w=1000.0, backend=backend)
    surplus = 1000.0 * ETA - 500.0
    _check(
        frame,
        {
            "PV_AC_To_Battery": surplus,
            "Grid_AC_To_Battery": 1000.0 / ETA - surplus,
            "Battery_Charge_Input": 1000.0,
            "PV_Origin_Battery_Charge_Stored": surplus * ETA * EFF,
            "Battery_Energy": 2000.0 + 1000.0 * EFF,
        },
    )


@pytest.mark.parametrize("backend", _BACKENDS)
def test_the_discharge_limit_binds_battery_ac(backend):
    frame = _run([0], [3000], 5000.0, max_discharge_power_w=800.0, backend=backend)
    _check(
        frame,
        {
            "Battery_AC_To_Load": 800.0,
            "Battery_Discharge_DC": 800.0 / ETA / EFF,
            "Import_From_Grid": 2200.0,
        },
    )


@pytest.mark.parametrize("backend", _BACKENDS)
def test_the_c_rate_limits_stored_energy_both_ways(backend):
    # 0.1 C on 10 kWh: at most 1000 Wh stored or released in the hour.
    charging = _run([5000], [0], 2000.0, power_limit_c_rate=0.1, backend=backend)
    _check(
        charging,
        {"Battery_Charge_Stored": 1000.0, "Battery_Charge_Input": 1000.0 / EFF, "PV_AC_To_Battery": 1000.0 / EFF / ETA},
    )
    discharging = _run([0], [3000], 5000.0, power_limit_c_rate=0.1, backend=backend)
    _check(
        discharging,
        {"Battery_Discharge_DC": 1000.0, "Battery_AC_To_Load": 1000.0 * EFF * ETA, "Battery_Energy": 4000.0},
    )


@pytest.mark.parametrize("backend", _BACKENDS)
def test_a_binding_discharge_floor_lands_exactly(backend):
    # A14: reserve and target share 0.5, so hold_target lands on 5000 Wh.
    instructions = _charge_target(1, discharge=True, reserve=0.5, target=0.5)
    frame = _run([0], [3000], 5100.0, instructions=instructions, backend=backend)
    assert frame["Battery_Energy"].iloc[0] == 5000.0
    _check(frame, {"Battery_Discharge_DC": 100.0, "Battery_AC_To_Load": 100.0 * EFF * ETA})


def test_a_dc_coupled_run_refuses_a_missing_efficiency_even_without_dispatch():
    instructions = _charge_target(24, discharge=False, reserve=0.0)
    pv, load, temp, _cfg, _sim = build("baseline", "h")
    pv, load, temp = pv.iloc[:24], load.iloc[:24], temp.iloc[:24]
    kwargs = dict(pv_dc=pv, houseload=load, freq="h", temperature_series=temp, dispatch_instructions=instructions)
    for battery, extra in (
        (BatteryConfig(nominal_energy_wh=0.0), {}),
        (BatteryConfig(nominal_energy_wh=10000.0), {"battery_retired": True, "initial_energy_wh": 0.0}),
    ):
        with pytest.raises(ValueError, match="DC-coupled battery needs a grid_charge_efficiency"):
            simulate_energy_balance(battery_config=battery, **kwargs, **extra)


# --- the site limit and the PV inverter's night draw ------------------------


def test_night_draw_counts_against_the_site_limit_before_grid_charge():
    instructions = _charge_target(1, discharge=False, reserve=0.0, limit_w=1000.0, target=0.9)
    frame = _run([0], [400], 2000.0, instructions=instructions, inverter_conversion={"night_consumption_w": 10.0})
    _check(
        frame,
        {
            "Inverter_Night_Consumption": 10.0,
            "Grid_AC_To_Battery": 590.0,
            "Import_From_Grid": 1000.0,
        },
    )


def test_night_draw_is_independent_of_the_battery():
    # M3: the PV inverter draws on PV-less steps even while the battery discharges.
    frame = _run([0, 0], [500, 500], 5000.0, inverter_conversion={"night_consumption_w": 12.0})
    assert (frame["Battery_AC_To_Load"] > 0.0).all()
    assert (frame["Inverter_Night_Consumption"] == 12.0).all()
    np.testing.assert_allclose(frame["Import_From_Grid"], 12.0, rtol=0, atol=1e-9)


# --- invariants on year-long cases ------------------------------------------


def _year(name, freq, backend, **overrides):
    pv, load, temp, cfg, sim = build_ac(name, freq)
    cfg = {**cfg, **overrides}
    config = BatteryConfig(**cfg)
    frame = simulate_energy_balance(
        pv_dc=pv,
        houseload=load,
        battery_config=config,
        freq=freq,
        temperature_series=temp,
        execution_backend=backend,
        **sim,
    )[0]
    return frame, config, (pv, load, temp, cfg, sim)


def _pv_only(inputs, freq, backend):
    pv, load, temp, cfg, _sim = inputs
    pv_cfg = {
        key: value
        for key, value in cfg.items()
        if key not in ("battery_inverter_ac_capacity_w", "battery_inverter_efficiency")
    }
    pv_cfg.update(nominal_energy_wh=0.0, coupling="dc")
    return simulate_energy_balance(
        pv_dc=pv,
        houseload=load,
        battery_config=BatteryConfig(**pv_cfg),
        freq=freq,
        temperature_series=temp,
        execution_backend=backend,
    )[0]


PV_SIDE = ("PV_DC", "PV_DC_Curtailed", "PV_DC_To_Inverter", "PV_Direct_Inverter_Loss", "PV_Production")


@pytest.mark.filterwarnings("ignore::breos.degradation.validation.BlastExperimentalRangeWarning")
@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("freq", RESOLUTIONS)
@pytest.mark.parametrize("scenario", AC_SCENARIOS)
def test_p1_p6_the_pv_side_is_the_pv_only_run_and_every_identity_holds(scenario, freq, backend):
    frame, config, inputs = _year(scenario, freq, backend)
    pv_only = _pv_only(inputs, freq, backend)
    for column in PV_SIDE:
        np.testing.assert_array_equal(frame[column].to_numpy(), pv_only[column].to_numpy(), err_msg=column)
    np.testing.assert_allclose(
        frame["PV_AC_To_Load"] + frame["PV_AC_Export"] + frame["PV_AC_To_Battery"],
        pv_only["PV_AC_To_Load"] + pv_only["PV_AC_Export"],
        rtol=1e-9,
        atol=1e-9,
    )
    np.testing.assert_array_equal(frame["Inverter_Night_Consumption"], pv_only["Inverter_Night_Consumption"])
    assert_energy_conservation(frame, config)
    assert_origin_reconciliation(frame, 1.0 if freq == "h" else 0.25)


CONVERSION = {
    "curve_ac": [0.1, 0.5, 1.0],
    "curve_eff": [[0.93, 0.965, 0.96], [0.94, 0.972, 0.968]],
    "curve_row": 0.4,
    "ac_cable_k_per_w": 4e-7,
    "night_consumption_w": 9.0,
}


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("freq", RESOLUTIONS)
def test_p1_holds_with_the_whole_pv_conversion(freq, backend):
    # M2: the efficiency curve, the AC cable, ac_output_scale and night draw
    # all come from the PV-only conversion.
    frame, config, inputs = _year("ac_baseline", freq, backend, inverter_conversion=CONVERSION, ac_output_scale=0.97)
    pv_only = _pv_only(inputs, freq, backend)
    for column in (*PV_SIDE, "Inverter_Night_Consumption"):
        np.testing.assert_array_equal(frame[column].to_numpy(), pv_only[column].to_numpy(), err_msg=column)
    assert frame["Inverter_Night_Consumption"].sum() > 0.0
    assert_energy_conservation(frame, config)


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("freq", RESOLUTIONS)
def test_p1_holds_with_a_per_step_curve_row_and_derating(freq, backend):
    pv, load, temp, cfg, _sim = build_ac("ac_baseline", freq)
    cfg = {**cfg, "inverter_conversion": CONVERSION}
    rows = pd.Series(np.linspace(0.0, 1.0, len(pv)), index=pv.index)
    derate = pd.Series(np.where(np.arange(len(pv)) % 7 == 0, 0.8, 1.0), index=pv.index)
    common = dict(
        pv_dc=pv,
        houseload=load,
        freq=freq,
        temperature_series=temp,
        execution_backend=backend,
        inverter_ac_derate=derate,
        inverter_curve_row=rows,
    )
    frame = simulate_energy_balance(battery_config=BatteryConfig(**cfg), **common)[0]
    pv_cfg = {
        k: v for k, v in cfg.items() if k not in ("battery_inverter_ac_capacity_w", "battery_inverter_efficiency")
    }
    pv_only = simulate_energy_balance(
        battery_config=BatteryConfig(**{**pv_cfg, "nominal_energy_wh": 0.0, "coupling": "dc"}), **common
    )[0]
    for column in PV_SIDE:
        np.testing.assert_array_equal(frame[column].to_numpy(), pv_only[column].to_numpy(), err_msg=column)
    assert frame["PV_DC_Curtailed"].sum() > 0.0


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("freq", RESOLUTIONS)
def test_p3_a_battery_that_cannot_move_gives_the_pv_only_flows(freq, backend):
    frame, _config, inputs = _year("ac_baseline", freq, backend, battery_inverter_ac_capacity_w=0.0)
    pv_only = _pv_only(inputs, freq, backend)
    for column in (*PV_SIDE, "PV_AC_To_Load", "PV_AC_Export", "Import_From_Grid", "Inverter_Loss"):
        np.testing.assert_array_equal(frame[column].to_numpy(), pv_only[column].to_numpy(), err_msg=column)
    assert (frame["PV_AC_To_Battery"] == 0.0).all() and (frame["Battery_AC_To_Load"] == 0.0).all()


@pytest.mark.parametrize("backend", _BACKENDS)
def test_p3_a_full_battery_held_by_its_instructions_gives_the_pv_only_flows(backend):
    pv, load, _temp, cfg, _sim = build("baseline", "h")
    cfg = {
        **cfg,
        "coupling": "ac",
        "battery_inverter_efficiency": 0.95,
        "battery_inverter_ac_capacity_w": 3000.0,
        "min_soc": 0.0,
        "max_soc": 1.0,
        "standby_loss_wh": 0.0,
        "enable_replacement": False,
        "enable_resistance_fade": False,
    }
    hold = DispatchInstructions(
        discharge_allowed=np.zeros(len(pv), dtype=bool),
        reserve_fraction=np.zeros(len(pv)),
        grid_target_fraction=np.full(len(pv), np.nan),
        grid_charge_efficiency=None,
        grid_import_limit_w=math.inf,
    )
    temperature = pd.Series(25.0, index=pv.index)
    frame = simulate_energy_balance(
        pv_dc=pv,
        houseload=load,
        battery_config=BatteryConfig(**cfg),
        freq="h",
        temperature_series=temperature,
        initial_energy_wh=cfg["nominal_energy_wh"],
        dispatch_instructions=hold,
        execution_backend=backend,
    )[0]
    pv_only = _pv_only((pv, load, temperature, cfg, {}), "h", backend)
    flows = ("PV_AC_To_Load", "PV_AC_Export", "Import_From_Grid", "PV_Direct_Inverter_Loss", "PV_DC_Curtailed")
    for column in flows:
        np.testing.assert_array_equal(frame[column].to_numpy(), pv_only[column].to_numpy(), err_msg=column)
    assert (frame["PV_AC_To_Battery"] == 0.0).all()


@pytest.mark.parametrize("backend", _BACKENDS)
def test_p4_a_retired_pack_steps_are_the_pv_only_run(backend):
    frame, _config, inputs = _year(
        "ac_replacement", "h", backend, enable_replacement=False, skipped_replacement_action="retire"
    )
    pv_only = _pv_only(inputs, "h", backend)
    retired = frame["Battery_SOH"].to_numpy() < 100.0 * 0.995
    retired &= (frame["Battery_Energy"] == 0.0).to_numpy() & (frame["Battery_Energy_Beginning"] == 0.0).to_numpy()
    assert retired.sum() > 1000, "the pack must retire early in the year"
    for column in (*PV_SIDE, "PV_AC_To_Load", "PV_AC_Export", "Import_From_Grid"):
        np.testing.assert_array_equal(
            frame[column].to_numpy()[retired], pv_only[column].to_numpy()[retired], err_msg=column
        )


# --- instructions, planners and controllers -----------------------------------


def test_grid_charge_efficiency_none_has_its_own_hash():
    def make(efficiency):
        return _charge_target(4, discharge=False, reserve=0.0, efficiency=efficiency)

    none = make(None)
    assert none.grid_charge_efficiency is None
    assert none == make(None)
    assert none.instruction_hash() not in {make(value).instruction_hash() for value in (1.0, 0.95, 1e-300)}
    assert pickle.loads(pickle.dumps(none)) == none


def test_a_dc_coupled_battery_refuses_instructions_without_a_grid_charge_efficiency():
    with pytest.raises(ValueError, match="DC-coupled battery needs a grid_charge_efficiency"):
        _run([0], [500], 2000.0, coupling="dc", instructions=_charge_target(1, discharge=False, reserve=0.0))


def test_an_ac_coupled_battery_refuses_a_grid_charge_efficiency_it_would_ignore():
    instructions = _charge_target(1, discharge=False, reserve=0.0, efficiency=0.96)
    with pytest.raises(ValueError, match="take grid_charge_efficiency = None"):
        _run([0], [500], 2000.0, instructions=instructions)
    # Without a grid target the scalar is never read, as in the no-op set.
    discharge_only = _charge_target(1, discharge=True, reserve=0.0, efficiency=1.0, target=np.nan)
    _run([0], [500], 2000.0, instructions=discharge_only)


@pytest.mark.parametrize("keyword", ["day_controller", "instruction_planner"])
def test_planners_and_controllers_refuse_an_ac_coupled_battery_for_now(keyword):
    pv, load, temp, cfg, _sim = build_ac("ac_baseline", "h")
    with pytest.raises(NotImplementedError, match="do not plan for it yet"):
        _simulate_core(
            pv_dc=pv,
            houseload=load,
            battery_config=BatteryConfig(**cfg),
            freq="h",
            temperature_series=temp,
            **{keyword: object()},
        )


def test_the_daily_target_evaluator_refuses_an_ac_coupled_battery():
    problem = SimpleNamespace(
        battery_config=BatteryConfig(**{**REFERENCE, "coupling": "ac", "battery_inverter_efficiency": ETA})
    )
    with pytest.raises(NotImplementedError, match="does not plan for an AC-coupled battery yet"):
        _DayEvaluator(problem, np.array([0.5]), "python")  # type: ignore[arg-type]
