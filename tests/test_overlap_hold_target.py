"""Opt-in overlapping periods retain the grid target (ADR 0002 A14, #347)."""

import dataclasses
import hashlib
import math

import numpy as np
import pandas as pd
import pytest

from breos._dispatch import _dispatch_dc_step, lfp_capacity_factor
from breos.app import App
from breos.battery import BatteryConfig, _simulate_detailed_run, simulate_energy_balance
from breos.dispatch_instructions import DispatchInstructions
from breos.smart_charging import FixedTargetDayController, SmartChargingSpec, period_layout, resolve_instructions
from tests.energy_conservation import assert_energy_conservation, assert_origin_reconciliation
from tests.test_controller_seam import SPEC, _project, _Recording, _scenario, _steps_per_day
from tests.test_smart_charging import BASE, FIXED, TOU
from tools.parity.harness import build_instructed

HOLD = {**FIXED, "overlap_policy": "hold_target", "discharge_periods": ["off_peak", "peak"]}


def test_overlap_requires_explicit_opt_in():
    for table in (
        {key: value for key, value in HOLD.items() if key != "overlap_policy"},
        {**HOLD, "overlap_policy": "reject"},
    ):
        with pytest.raises(ValueError, match="share off_peak.*hold_target"):
            App({**BASE, "tariff": TOU, "smart_charging": table})
    app = App({**BASE, "tariff": TOU, "smart_charging": HOLD})
    assert app._resolved.smart_charging.overlap_policy == "hold_target"
    with pytest.raises(ValueError, match="overlap_policy.*must be one of"):
        App({**BASE, "tariff": TOU, "smart_charging": {**HOLD, "overlap_policy": "net"}})


@pytest.mark.parametrize("mode", ["disabled", "discharge_only", "daily_persistence"])
def test_hold_target_is_refused_where_it_cannot_hold_a_target(mode):
    fields = {"mode": mode, "overlap_policy": "hold_target"}
    reason = (
        "planner replaces grid targets but keeps reserves fixed"
        if mode == "daily_persistence"
        else "no grid-charge target"
    )
    with pytest.raises(ValueError, match=reason):
        SmartChargingSpec(**fields)
    with pytest.raises(ValueError, match=reason):
        App({**BASE, "tariff": TOU, "smart_charging": fields})


@pytest.mark.parametrize("target", [0.0, 0.5, 1.0])
def test_only_overlap_steps_gain_a_floor_equal_to_the_target(target):
    spec = SmartChargingSpec(
        mode="fixed_target",
        overlap_policy="hold_target",
        target_usable_fraction=target,
        charge_periods=("off_peak", "mid_peak"),
        discharge_periods=("mid_peak", "peak"),
        grid_charge_efficiency=0.95,
    )
    instructions = period_layout(spec, ["off_peak", "mid_peak", "peak", "other"], target)
    np.testing.assert_array_equal(instructions.discharge_allowed, [False, True, True, False])
    np.testing.assert_array_equal(instructions.reserve_fraction, [0.0, target, 0.0, 0.0])
    np.testing.assert_array_equal(instructions.grid_target_fraction, [target, target, np.nan, np.nan])
    without_overlap = dataclasses.replace(spec, discharge_periods=("peak",))
    assert period_layout(without_overlap, ["off_peak", "peak"], target) == period_layout(
        dataclasses.replace(without_overlap, overlap_policy="reject"), ["off_peak", "peak"], target
    )


@pytest.fixture(scope="module", params=["python", "numba"])
def step(request):
    if request.param == "python":
        return _dispatch_dc_step
    numba = pytest.importorskip("numba")
    import breos._numba_dispatch_kernels  # noqa: F401 -- register the shared helpers

    return numba.njit(fastmath=False)(_dispatch_dc_step)


def _step(
    step,
    energy,
    floor,
    *,
    pv=0.0,
    load=1e6,
    eff_discharge=0.8611525595091509,
    emin=604.219051517796,
    emax=6042.190515177959,
    inverter_cap=1e9,
):
    return step(
        pv,
        load,
        energy,
        emin,
        emax,
        0.95,
        eff_discharge,
        0.96,
        1e9,
        1e9,
        inverter_cap,
        1.0,
        1e9,
        2.0,
        True,
        floor,
        floor,
        0.95,
        1e9,
    )


def test_binding_floor_has_no_next_step_grid_charge_dust(step):
    floor = 668.0481574491401
    discharged = _step(step, 3755.3464593806407, floor)
    assert discharged[0] == floor
    assert discharged[7] > 0.0 and discharged[-2] == 0.0
    held = _step(step, discharged[0], floor)
    assert held[0] == floor and held[7] == held[-2] == 0.0


def test_binding_grid_target_has_no_next_step_charge_or_discharge_dust(step):
    rng = np.random.default_rng(347)
    for _ in range(100):
        emax = rng.uniform(100.0, 10_000.0)
        emin = 0.1 * emax
        floor = emin + rng.uniform(0.1, 0.9) * (emax - emin)
        charged = _step(step, emin, floor, emin=emin, emax=emax)
        assert charged[0] == floor and charged[-2] > 0.0 and charged[7] == 0.0
        held = _step(step, charged[0], floor, emin=emin, emax=emax)
        assert held[0] == floor and held[7] == held[-2] == 0.0


def test_pv_surplus_can_charge_above_the_overlap_target_and_export(step):
    result = _step(step, 3000.0, 2500.0, pv=10_000.0, load=500.0, emin=500.0, emax=5000.0, inverter_cap=10_000.0)
    assert result[0] == 5000.0 and result[1] > 0.0 and result[5] > 0.0
    assert result[7] == result[-2] == 0.0


@pytest.mark.parametrize("freq", ["h", "15min"])
@pytest.mark.parametrize("backend", ["python", "numba"])
def test_floor_and_target_use_exactly_the_same_temperature_and_health_mapping(freq, backend):
    if backend == "numba":
        pytest.importorskip("numba")
    index = pd.date_range("2026-01-01", periods=3, freq=freq, tz="UTC")
    temperatures = [10.0, -10.0, 25.0]
    fraction = 0.63
    config = BatteryConfig(
        nominal_energy_wh=4739.0,
        initial_soh=87.0,
        standby_loss_wh=0.0,
        thermal_resistance_k_per_w=0.0,
        enable_replacement=False,
        inverter_ac_capacity_w=20_000.0,
    )
    instructions = DispatchInstructions(
        discharge_allowed=np.ones(3, dtype=bool),
        reserve_fraction=np.full(3, fraction),
        grid_target_fraction=np.full(3, fraction),
        grid_charge_efficiency=0.95,
        grid_import_limit_w=math.inf,
    )
    frame = simulate_energy_balance(
        pv_dc=pd.Series(0.0, index=index),
        houseload=pd.DataFrame({"Load": 1e6}, index=index),
        temperature_series=pd.Series(temperatures, index=index),
        battery_config=config,
        freq=freq,
        execution_backend=backend,
        dispatch_instructions=instructions,
    )[0]
    bounds = []
    for temperature in temperatures:
        capacity = config.nominal_energy_wh * 0.87
        emin = capacity * config.min_soc * lfp_capacity_factor(temperature)
        emax = capacity * config.max_soc * lfp_capacity_factor(temperature)
        bounds.append(emin + fraction * (emax - emin))
    np.testing.assert_array_equal(frame["Battery_Energy_End"], bounds)
    assert (frame["Battery_Discharge_DC"].iloc[:2] > 0.0).all()
    assert frame["Grid_AC_To_Battery"].iloc[2] > 0.0
    assert_energy_conservation(frame, config)
    assert_origin_reconciliation(frame, 1.0 if freq == "h" else 0.25)


def _overlap_scenario(start, freq, days=5):
    scenario = _scenario(start, days * _steps_per_day(freq), freq, "Europe/Berlin")
    spec = dataclasses.replace(
        SPEC,
        overlap_policy="hold_target",
        discharge_periods=("off_peak", "peak", "mid"),
        charge_periods=("off_peak", "mid"),
        target_usable_fraction=0.55,
    )
    return dataclasses.replace(scenario, instructions=resolve_instructions(spec, scenario.tariff))


@pytest.mark.filterwarnings("ignore::breos.degradation.validation.BlastExperimentalRangeWarning")
@pytest.mark.parametrize("freq", ["h", "15min"])
@pytest.mark.parametrize("start", ["2024-03-28T10:00Z", "2024-10-25T13:00Z"])
@pytest.mark.parametrize("engine", ["native", "blast"])
def test_overlap_reconciles_with_aging_replacement_dst_and_backend_parity(freq, start, engine):
    pytest.importorskip("numba")
    scenario = _overlap_scenario(start, freq)
    battery = BatteryConfig(
        nominal_energy_wh=5000.0,
        max_charge_power_w=2500.0,
        max_discharge_power_w=2500.0,
        inverter_ac_capacity_w=4000.0,
        enable_resistance_fade=engine == "native",
        initial_soh=70.01 if engine == "native" else 100.0,
    )
    common = dict(
        pv_dc=scenario.pv,
        houseload=scenario.load,
        temperature_series=scenario.temperature,
        freq=freq,
        battery_config=battery,
        finalize_degradation=True,
        initial_energy_wh=1000.0,
        initial_pv_origin_energy_wh=300.0,
        initial_grid_origin_energy_wh=200.0,
        degradation_engine=engine,
        **({"blast_model": "nmc_gr_50ah_b1"} if engine == "blast" else {}),
    )
    reference = _simulate_detailed_run(
        **common, execution_backend="python", dispatch_instructions=scenario.instructions
    )
    frame = reference.results_df
    hours = 1.0 if freq == "h" else 0.25
    assert_energy_conservation(frame, battery)
    assert_origin_reconciliation(frame, hours)
    grid = frame["Grid_AC_To_Battery"]
    discharge = frame["Battery_Discharge_DC"]
    assert grid.sum() > 0.0 and discharge.sum() > 0.0 and frame["PV_AC_Export"].sum() > 0.0
    assert not ((frame["Battery_Charge_Input"] > 0.0) & (discharge > 0.0)).any()
    assert not ((grid > 0.0) & ((discharge > 0.0) | (frame["PV_AC_Export"] > 0.0))).any()
    overlap = scenario.instructions.discharge_allowed & np.isfinite(scenario.instructions.grid_target_fraction)
    discharged_on_overlap = overlap & (discharge > 0.0)
    assert discharged_on_overlap.any()
    assert (frame.loc[discharged_on_overlap, "Battery_SOC_Normalized"] >= 0.55 - 1e-14).all()
    assert (frame.loc[overlap, "Battery_SOC_Normalized"] > 0.55 + 0.01).any(), "PV must charge above the target"
    if engine == "native":
        assert reference.n_replacements == 1
    for backend, controller in (("numba", False), ("python", True), ("numba", True)):
        run = _simulate_detailed_run(
            **common,
            execution_backend=backend,
            **(
                dict(day_controller=FixedTargetDayController(scenario.instructions), controller_tariff=scenario.tariff)
                if controller
                else dict(dispatch_instructions=scenario.instructions)
            ),
        )
        pd.testing.assert_frame_equal(frame, run.results_df, check_exact=True)
        pd.testing.assert_frame_equal(reference.degradation_df, run.degradation_df, check_exact=True)
        pd.testing.assert_frame_equal(reference.summary_df, run.summary_df, check_exact=True)
        if controller:
            assert run.controller_instructions == scenario.instructions


@pytest.mark.parametrize("freq", ["h", "15min"])
@pytest.mark.parametrize("backend", ["python", "numba"])
def test_overlap_controller_carries_exact_instructions_across_the_year_seam(freq, backend):
    if backend == "numba":
        pytest.importorskip("numba")
    scenario = _overlap_scenario("2023-01-01T00:00Z", freq, days=365)
    recording = _Recording(FixedTargetDayController(scenario.instructions))
    static, static_years = _project(scenario, backend, 2)
    adapted, adapted_years = _project(scenario, backend, 2, controller=recording)
    pd.testing.assert_frame_equal(static.yearly_df, adapted.yearly_df, check_exact=True)
    assert dataclasses.replace(adapted.carry, controller_carry=None) == static.carry
    for (first, _), (second, _) in zip(static_years, adapted_years, strict=True):
        pd.testing.assert_frame_equal(first, second, check_exact=True)
        assert_origin_reconciliation(second, 1.0 if freq == "h" else 0.25)
    executed = adapted.controller_instructions
    np.testing.assert_array_equal(executed.reserve_fraction, np.tile(scenario.instructions.reserve_fraction, 2))
    np.testing.assert_array_equal(executed.grid_target_fraction, np.tile(scenario.instructions.grid_target_fraction, 2))
    assert adapted_years[0][1].active_day_decision is not None, "Berlin's civil day must span the replay seam"


# Captured before adding overlap support: every numeric ledger column, in
# order, at both resolutions and with each aging engine.
@pytest.mark.filterwarnings("ignore::breos.degradation.validation.BlastExperimentalRangeWarning")
@pytest.mark.parametrize(
    ("name", "freq", "digest"),
    [
        ("fixed_target", "h", "40da4a37357944f0f5d4bcb304ab8680ee59f3773d0efa9a149278174e9b412a"),
        ("fixed_target", "15min", "5ebb36809c4cd411e9493c080c3aa607146f9f4e68a30f928519cf01a2725261"),
        ("fixed_target_blast", "h", "8f3513ed5fae3e95b8b87e9dd60568df0fcf7ef18314ad408b1cbe5cd370688c"),
        ("fixed_target_blast", "15min", "81cdd6efb46399168f1aa28847fb4d7cc73dd0478293649c5907bda658c6a272"),
    ],
)
def test_default_reject_preserves_the_existing_ledger_bits(name, freq, digest):
    pv, load, temp, config, kwargs = build_instructed(name, freq)
    frame = simulate_energy_balance(
        pv_dc=pv,
        houseload=load,
        temperature_series=temp,
        battery_config=BatteryConfig(**config),
        freq=freq,
        execution_backend="python",
        **kwargs,
    )[0]
    assert hashlib.sha256(frame.drop(columns="Datetime").to_numpy(dtype=np.float64).tobytes()).hexdigest() == digest


@pytest.mark.usefixtures("_patch_weather")
def test_app_executes_and_records_hold_target():
    app = App({**BASE, "tariff": TOU, "smart_charging": HOLD})
    app.simulate()
    result = app.result()
    assert result["provenance"]["smart_charging"]["overlap_policy"] == "hold_target"
    assert result["result_schema_version"] == "2.7"
    instructions = app._artifacts.instructions
    overlap = instructions.discharge_allowed & np.isfinite(instructions.grid_target_fraction)
    np.testing.assert_array_equal(instructions.reserve_fraction[overlap], instructions.grid_target_fraction[overlap])
    assert result["smart_charging"]["yearly"][0]["grid_charge_ac_kwh"] > 0.0


def test_monte_carlo_accepts_hold_target(tmp_path, write_multiyear_weather):
    from breos.montecarlo import MonteCarloSettings, run_montecarlo

    weather = write_multiyear_weather(tmp_path / "multi.csv")
    settings = MonteCarloSettings(weather_file=str(weather), n_runs=2, years_per_run=2, seed=347, collect_yearly=True)
    result = run_montecarlo({**BASE, "tariff": TOU, "smart_charging": HOLD}, settings)
    assert result.provenance["smart_charging"]["overlap_policy"] == "hold_target"
    assert (result.yearly["Grid_AC_To_Battery_kWh"] > 0.0).all()
