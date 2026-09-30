"""Grid charging and dispatch instructions in the canonical step (ADR 0002 A6-A8, #178)."""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from breos._dispatch import lfp_capacity_factor
from breos.battery import BatteryConfig, simulate_energy_balance, simulate_energy_balance_summary
from breos.dispatch_instructions import DispatchInstructions
from tests.energy_conservation import assert_energy_conservation, assert_origin_reconciliation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "parity"))
from harness import INSTRUCTION_SCENARIOS, RESOLUTIONS, build, build_instructed  # noqa: E402

_BACKENDS = [
    "python",
    pytest.param(
        "numba", marks=pytest.mark.skipif(not importlib.util.find_spec("numba"), reason="numba not installed")
    ),
]
EFF = 0.95


def _simulate(pv, load, temperature, config, instructions=None, backend="python", **kwargs):
    results, *_ = simulate_energy_balance(
        pv_dc=pv,
        houseload=load,
        battery_config=config,
        freq=kwargs.pop("freq", "h"),
        temperature_series=temperature,
        dispatch_instructions=instructions,
        execution_backend=backend,
        **kwargs,
    )
    return results


def _night(hours=6, load_w=300.0, temperature_c=25.0):
    index = pd.date_range("2025-03-01", periods=hours, freq="h", tz="UTC")
    return (
        pd.Series(0.0, index=index),
        pd.DataFrame({"Load": load_w}, index=index),
        pd.Series(temperature_c, index=index),
    )


def _charge_everywhere(n, target, *, efficiency=EFF, limit_w=math.inf):
    return DispatchInstructions(
        discharge_allowed=np.zeros(n, dtype=bool),
        reserve_fraction=np.zeros(n),
        grid_target_fraction=np.full(n, target),
        grid_charge_efficiency=efficiency,
        grid_import_limit_w=limit_w,
    )


def _pack(**overrides):
    settings = {
        "nominal_energy_wh": 10000.0,
        "min_soc": 0.1,
        "max_soc": 0.9,
        "standby_loss_wh": 0.0,
        "thermal_resistance_k_per_w": 0.0,
        "enable_replacement": False,
    }
    return BatteryConfig(**{**settings, **overrides})


def _window(config, temperature_c=25.0):
    f_cap = lfp_capacity_factor(temperature_c)
    return config.nominal_energy_wh * config.min_soc * f_cap, config.nominal_energy_wh * config.max_soc * f_cap


# --- no-op instructions ---------------------------------------------------


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("scenario", ["baseline", "replacement", "carried_state", "c_rate_limited", "blast"])
@pytest.mark.filterwarnings("ignore::breos.degradation.validation.BlastExperimentalRangeWarning")
def test_noop_instructions_are_the_greedy_step_bit_for_bit(scenario, backend):
    pv, load, temp, cfg, sim = build(scenario, "h")
    noop = DispatchInstructions(
        discharge_allowed=np.ones(len(pv), dtype=bool),
        reserve_fraction=np.zeros(len(pv)),
        grid_target_fraction=np.full(len(pv), np.nan),
        grid_charge_efficiency=0.8,
        grid_import_limit_w=10.0,
    )
    greedy = _simulate(pv, load, temp, BatteryConfig(**cfg), None, backend, **sim)
    instructed = _simulate(pv, load, temp, BatteryConfig(**cfg), noop, backend, **sim)
    pd.testing.assert_frame_equal(greedy, instructed, check_exact=True)


# --- conservation and origins under grid charging --------------------------


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("freq", RESOLUTIONS)
@pytest.mark.parametrize("scenario", INSTRUCTION_SCENARIOS)
@pytest.mark.filterwarnings("ignore::breos.degradation.validation.BlastExperimentalRangeWarning")
def test_instructed_runs_conserve_energy_and_reconcile_origins(scenario, freq, backend):
    pv, load, temp, cfg, sim = build_instructed(scenario, freq)
    config = BatteryConfig(**cfg)
    results = _simulate(pv, load, temp, config, sim.pop("dispatch_instructions"), backend, freq=freq, **sim)

    assert_energy_conservation(results, config)
    assert_origin_reconciliation(results, 0.25 if freq == "15min" else 1.0)
    if scenario != "noop_instructions":
        assert results["Grid_Origin_Battery_Charge_Stored"].sum() > 0.0
        assert results["Grid_Origin_Battery_AC_To_Load"].sum() > 0.0
    if scenario == "fixed_target_replacement":
        assert results["Battery_Replaced"].any()
        assert results["Grid_Origin_Replacement_Energy_Removed"].sum() > 0.0


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("scenario", ["fixed_target_limited", "reserve_floor", "fixed_target_blast"])
@pytest.mark.filterwarnings("ignore::breos.degradation.validation.BlastExperimentalRangeWarning")
def test_the_summary_path_applies_the_same_instructions(scenario, backend):
    # Monte Carlo and the optimizer take the summary path, with its reduced
    # buffers; it must dispatch exactly as the detailed one does.
    pv, load, temp, cfg, sim = build_instructed(scenario, "h")
    kwargs = {**sim, "execution_backend": backend}
    detailed = simulate_energy_balance(
        pv_dc=pv, houseload=load, battery_config=BatteryConfig(**cfg), freq="h", temperature_series=temp, **kwargs
    )[0]
    summary = simulate_energy_balance_summary(
        pv_dc=pv, houseload=load, battery_config=BatteryConfig(**cfg), freq="h", temperature_series=temp, **kwargs
    )

    assert summary.column_sums["Grid_AC_To_Battery"] > 0.0
    for column in ("Grid_AC_To_Battery", "Grid_Charge_Conversion_Loss", "Import_From_Grid", "Battery_Discharge_DC"):
        assert summary.column_sums[column] == detailed[column].sum(), column
    assert summary.carried_grid_origin_energy_wh == detailed["Battery_Grid_Origin_Energy_End"].iloc[-1]


def test_noop_instructions_are_shared_per_length():
    assert DispatchInstructions.noop(24) is DispatchInstructions.noop(24)
    assert DispatchInstructions.noop(24) is not DispatchInstructions.noop(48)
    assert not DispatchInstructions.noop(24).grid_target_fraction.flags.writeable


# --- the grid-charge sub-step ---------------------------------------------


def test_grid_charge_reaches_the_target_through_both_efficiencies():
    pv, load, temperature = _night()
    config = _pack(max_charge_power_w=10000.0)
    emin, emax = _window(config)
    results = _simulate(pv, load, temperature, config, _charge_everywhere(6, 0.5), initial_energy_wh=emin)

    target = emin + 0.5 * (emax - emin)
    assert results["Battery_Energy_End"].iloc[-1] == pytest.approx(target, rel=1e-12)
    first = results.iloc[0]
    assert first["Grid_DC_To_Battery"] == pytest.approx(first["Grid_AC_To_Battery"] * EFF, rel=1e-12)
    assert first["Battery_Charge_Stored"] == pytest.approx(first["Grid_DC_To_Battery"] * config.charge_efficiency)
    assert first["Import_From_Grid"] == pytest.approx(300.0 + first["Grid_AC_To_Battery"])
    # Grid charge is charge input, so the cell loss sees it (A6).
    assert first["Battery_Charge_Input"] == first["Grid_DC_To_Battery"]
    assert first["Battery_Charge_Loss"] > 0.0
    # It adds to the grid origin only.
    assert results["Battery_PV_Origin_Energy_End"].iloc[-1] == 0.0
    assert results["Battery_Grid_Origin_Energy_End"].iloc[-1] == pytest.approx(target - emin, rel=1e-12)
    # Once there, it holds: no further import for the battery.
    assert results["Grid_AC_To_Battery"].iloc[-1] == 0.0


def test_the_target_is_a_fraction_of_the_window_that_moves_with_temperature():
    """A7: the same fraction names less energy on a cold night."""
    reached = {}
    for temperature_c in (25.0, 5.0):
        pv, load, temperature = _night(temperature_c=temperature_c)
        config = _pack()
        emin, emax = _window(config, temperature_c)
        results = _simulate(pv, load, temperature, config, _charge_everywhere(6, 0.5), initial_energy_wh=emin)
        reached[temperature_c] = results["Battery_Energy_End"].iloc[-1]
        assert reached[temperature_c] == pytest.approx(emin + 0.5 * (emax - emin), rel=1e-12)
    assert reached[5.0] < reached[25.0]


@pytest.mark.parametrize(
    ("config_kwargs", "limit_w", "bound"),
    [
        # The battery's own charge-input limit: DC after conversion.
        ({"max_charge_power_w": 800.0}, math.inf, ("Grid_DC_To_Battery", 800.0)),
        # The site limit counts the load's own import too.
        ({}, 1000.0, ("Import_From_Grid", 1000.0)),
        # The inverter's AC rating bounds the grid-charge AC input.
        ({"inverter_ac_capacity_w": 600.0}, math.inf, ("Grid_AC_To_Battery", 600.0)),
    ],
    ids=["charge-power", "site-import", "inverter"],
)
def test_grid_charge_respects_every_shared_limit(config_kwargs, limit_w, bound):
    pv, load, temperature = _night()
    config = _pack(**config_kwargs)
    emin, _ = _window(config)
    results = _simulate(
        pv, load, temperature, config, _charge_everywhere(6, 1.0, limit_w=limit_w), initial_energy_wh=emin
    )
    column, limit = bound
    assert results[column].max() == pytest.approx(limit, rel=1e-12)
    assert (results[column] <= limit * (1 + 1e-12)).all()


def test_pv_keeps_priority_on_the_inverter_and_charge_limits():
    """A6: a grid target never takes a shared limit from PV in the same step."""
    index = pd.date_range("2025-06-01 10:00", periods=1, freq="h", tz="UTC")
    pv = pd.Series(2000.0, index=index)
    load = pd.DataFrame({"Load": 2400.0}, index=index)
    temperature = pd.Series(25.0, index=index)
    config = _pack(inverter_ac_capacity_w=3000.0, max_charge_power_w=5000.0)
    emin, _ = _window(config)
    greedy = _simulate(pv, load, temperature, config, None, initial_energy_wh=emin)
    charged = _simulate(pv, load, temperature, config, _charge_everywhere(1, 1.0), initial_energy_wh=emin)

    for column in ("PV_AC_To_Load", "PV_DC_To_Battery", "PV_DC_To_Inverter", "PV_AC_Export"):
        assert charged[column].iloc[0] == greedy[column].iloc[0], column
    # The grid gets what PV left of the inverter's 3 kW.
    step = charged.iloc[0]
    assert step["Grid_AC_To_Battery"] == pytest.approx(3000.0 - step["PV_AC_To_Load"], rel=1e-12)


def test_no_grid_charge_while_pv_is_exported():
    index = pd.date_range("2025-06-01 08:00", periods=8, freq="h", tz="UTC")
    pv = pd.Series(4000.0, index=index)
    load = pd.DataFrame({"Load": 300.0}, index=index)
    temperature = pd.Series(25.0, index=index)
    # A slow charger, so PV surplus is exported while the pack is below target.
    config = _pack(max_charge_power_w=500.0)
    emin, _ = _window(config)
    results = _simulate(pv, load, temperature, config, _charge_everywhere(8, 1.0), initial_energy_wh=emin)

    assert (results["PV_AC_Export"] > 0.0).all()
    assert (results["Grid_AC_To_Battery"] == 0.0).all()


def test_discharge_gate_and_reserve_floor():
    pv, load, temperature = _night(hours=12, load_w=800.0)
    config = _pack()
    emin, emax = _window(config)
    start = emax
    blocked = DispatchInstructions(
        discharge_allowed=np.zeros(12, dtype=bool),
        reserve_fraction=np.zeros(12),
        grid_target_fraction=np.full(12, np.nan),
        grid_charge_efficiency=EFF,
        grid_import_limit_w=math.inf,
    )
    held = _simulate(pv, load, temperature, config, blocked, initial_energy_wh=start)
    assert (held["Battery_Discharge_DC"] == 0.0).all()
    assert (held["Import_From_Grid"] == 800.0).all()

    reserve = DispatchInstructions(
        discharge_allowed=np.ones(12, dtype=bool),
        reserve_fraction=np.full(12, 0.6),
        grid_target_fraction=np.full(12, np.nan),
        grid_charge_efficiency=EFF,
        grid_import_limit_w=math.inf,
    )
    floored = _simulate(pv, load, temperature, config, reserve, initial_energy_wh=start)
    floor = emin + 0.6 * (emax - emin)
    assert floored["Battery_Discharge_DC"].sum() > 0.0
    assert floored["Battery_Energy_End"].min() == pytest.approx(floor, rel=1e-12)
    assert floored["Import_From_Grid"].iloc[-1] == 800.0


def test_grid_charge_heats_the_cell():
    pv, load, temperature = _night()
    config = _pack(thermal_resistance_k_per_w=0.01)
    emin, _ = _window(config)
    results = _simulate(pv, load, temperature, config, _charge_everywhere(6, 0.8), initial_energy_wh=emin)

    charging = results["Grid_DC_To_Battery"] > 0.0
    assert charging.any()
    assert (results.loc[charging, "T_cell"] > 25.0).all()


def test_only_pv_origin_discharge_counts_as_pv():
    """Grid energy shifted to the evening is not PV self-consumption."""
    pv, load, temp, cfg, sim = build_instructed("fixed_target", "h")
    results = _simulate(pv, load, temp, BatteryConfig(**cfg), sim.pop("dispatch_instructions"), **sim)

    unattributed = (
        results["Battery_AC_To_Load"]
        - results["PV_Origin_Battery_AC_To_Load"]
        - results["Grid_Origin_Battery_AC_To_Load"]
    )
    assert (unattributed > -1e-9).all()
    assert results["Grid_Origin_Battery_AC_To_Load"].sum() > 0.0
    # Grid-origin delivery shows up nowhere in the PV-origin column.
    assert results["PV_Origin_Battery_AC_To_Load"].sum() < results["Battery_AC_To_Load"].sum()


@pytest.mark.parametrize("scenario", ["fixed_target", "fixed_target_blast"])
@pytest.mark.filterwarnings("ignore::breos.degradation.validation.BlastExperimentalRangeWarning")
def test_grid_charging_is_aged_like_pv_charging(scenario):
    """Native and BLAST degradation both see grid charge as throughput."""
    pv, load, temp, cfg, sim = build_instructed(scenario, "h")
    config = BatteryConfig(**cfg)
    instructions = sim.pop("dispatch_instructions")
    *_, greedy_degradation = simulate_energy_balance(
        pv_dc=pv, houseload=load, battery_config=config, freq="h", temperature_series=temp, **sim
    )
    *_, charged_degradation = simulate_energy_balance(
        pv_dc=pv,
        houseload=load,
        battery_config=config,
        freq="h",
        temperature_series=temp,
        dispatch_instructions=instructions,
        **sim,
    )
    assert (
        charged_degradation["Cumulative_FEC_All_Packs"].iloc[-1]
        > greedy_degradation["Cumulative_FEC_All_Packs"].iloc[-1]
    )
    assert charged_degradation["SOH"].iloc[-1] < greedy_degradation["SOH"].iloc[-1]


def test_instructions_must_cover_the_simulation():
    pv, load, temperature = _night()
    with pytest.raises(ValueError, match="dispatch_instructions cover 5 steps; the simulation has 6"):
        _simulate(pv, load, temperature, _pack(), _charge_everywhere(5, 0.5))
