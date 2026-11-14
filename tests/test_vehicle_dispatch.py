"""ADR 0005 PR 2(a): home dispatch, residual routing and Q1–Q5."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from breos._dispatch import (
    _VEHICLE_LEDGER_COLUMNS,
    _VEHICLE_STATE_COLUMNS,
    LEDGER_SCHEMA_VERSION,
)
from breos.battery import BatteryConfig, _build_simulation_summary, _simulate_core
from breos.dispatch_instructions import DispatchInstructions
from breos.inverter import InverterConversion, InverterUnits
from breos.tariffs import TariffPrices, TariffSchedule, resolve_tariff
from breos.vehicle import VehicleConfig, VehicleSchedule, resolve_vehicle_schedule
from tests.backend_parity import BACKENDS, assert_frames_bit_equal
from tests.energy_conservation import (
    assert_energy_conservation,
    assert_origin_reconciliation,
    assert_vehicle_conservation,
)
from tests.test_ac_several_inverters import _inputs

pytestmark = pytest.mark.usefixtures("vehicle_full_health")

VEHICLE_COLUMNS = (*_VEHICLE_STATE_COLUMNS, *_VEHICLE_LEDGER_COLUMNS)


def vehicle(mode="v2h", **overrides):
    settings = dict(
        mode=mode,
        battery_kwh=40.0,
        departure_soc=0.8,
        min_soc=0.1,
        max_soc=1.0,
        charger_ac_rating_kw=7.4,
        charger_efficiency=0.95,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
        battery_cost_per_kwh=120.0,
        charger_cost=2500.0,
        enable_replacement=False,
        schedule={key: {"arrive": "18:00", "depart": "08:00", "trip_kwh": 9.0} for key in ("weekday", "weekend")},
    )
    if mode == "v2h":
        settings["v2h_min_soc"] = 0.3
    return VehicleConfig(**(settings | overrides))


def schedule(index, hours, countdown=None, plugged=None):
    n = len(index)
    return VehicleSchedule(
        index,
        hours,
        np.ones(n, dtype=bool) if plugged is None else plugged,
        np.arange(n, 0, -1) if countdown is None else countdown,
        np.zeros(n),
        np.zeros(n),
        np.zeros(n),
    )


def run_home(pv, load, energy, countdown, mode="v2h", backend="python", freq="h", **kwargs):
    hours = 1.0 if freq == "h" else 0.25
    index = pd.date_range("2025-06-01", periods=len(pv), freq=freq, tz="UTC")
    cfg = kwargs.pop("vehicle_config", vehicle(mode))
    battery = kwargs.pop("battery_config", BatteryConfig(nominal_energy_wh=0.0))
    core = _simulate_core(
        pv_dc=pd.Series(pv, index=index),
        houseload=pd.DataFrame({"Load": load}, index=index),
        temperature_series=pd.Series(25.0, index=index),
        freq=freq,
        battery_config=battery,
        vehicle_config=cfg,
        vehicle_schedule=schedule(index, hours, countdown),
        initial_vehicle_energy_wh=energy,
        execution_backend=backend,
        **kwargs,
    )
    frame = core.buffers.to_frame(core.rng)
    assert_energy_conservation(frame, battery)
    assert_vehicle_conservation(
        frame,
        cfg,
        hours,
        initial_energy_wh=energy,
        initial_pv_origin_wh=kwargs.get("initial_vehicle_pv_origin_energy_wh", 0.0),
        initial_grid_origin_wh=kwargs.get("initial_vehicle_grid_origin_energy_wh", 0.0),
    )
    return frame


@pytest.mark.parametrize("backend", BACKENDS)
def test_vh1_pv_surplus(backend):
    frame = run_home([5000.0], [1000.0], 20000.0, [10], backend=backend)
    expected = {
        "PV_AC_To_Vehicle": 3800.0,
        "Vehicle_Charge_Conversion_Loss": 181.41568672543053,
        "Vehicle_Charge_Input": 3618.5843132745695,
        "Vehicle_Charge_Loss": 180.92921566372843,
        "Vehicle_Energy": 23437.65509761084,
        "PV_AC_Export": 0.0,
    }
    for column, value in expected.items():
        assert frame[column].iloc[0] == pytest.approx(value, rel=1e-9, abs=1e-9), column


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("mode", ["v1g", "uncontrolled"])
def test_vh2_late_charging_and_exact_target(backend, mode):
    frame = run_home([0.0] * 4, [500.0] * 4, 20000.0, [4, 3, 2, 1], mode=mode, backend=backend)
    imports = [0.0, 0.0, 5885.422166980309, 7400.0] if mode == "v1g" else [7400.0, 5885.422166980309, 0.0, 0.0]
    energies = [20000.0, 20000.0, 25321.5, 32000.0] if mode == "v1g" else [26678.5, 32000.0, 32000.0, 32000.0]
    np.testing.assert_allclose(frame["Grid_AC_To_Vehicle"], imports, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(frame["Vehicle_Energy"], energies, rtol=1e-9, atol=1e-9)
    assert frame["Vehicle_Energy"].iloc[-1] == 32000.0
    assert (
        frame["Vehicle_Energy"].iloc[1 if mode == "uncontrolled" else 2] == energies[1 if mode == "uncontrolled" else 2]
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_vh3_v2h_guard_exact_landing(backend):
    frame = run_home([0.0] * 3, [3000.0] * 3, 30000.0, [3, 2, 1], backend=backend)
    expected = {
        "Vehicle_AC_To_Load": [3000.0, 1204.7512302806317, 0.0],
        "Vehicle_Discharge_DC": [3320.7536568995297, 1357.746343100469, 0.0],
        "Grid_AC_To_Vehicle": [0.0, 0.0, 7400.0],
        "Import_From_Grid": [0.0, 1795.2487697193683, 10400.0],
        "Vehicle_Energy": [26679.24634310047, 25321.5, 32000.0],
    }
    for column, values in expected.items():
        np.testing.assert_allclose(frame[column], values, rtol=1e-9, atol=1e-9, err_msg=column)
    assert frame["Vehicle_Energy"].iloc[1] == 25321.5
    assert frame["Vehicle_Energy"].iloc[2] == 32000.0
    assert (frame["Import_From_Grid"] >= 0.0).all()
    reference = run_home([0.0] * 3, [3000.0] * 3, 30000.0, [3, 2, 1], mode="v1g", backend=backend)
    assert reference["Grid_AC_To_Vehicle"].iloc[-1] == pytest.approx(2221.635598, rel=1e-9)


def stationary(kind):
    ac = kind in {"ac", "several-ac", "retired", "retires", "retired-several-ac"}
    return BatteryConfig(
        nominal_energy_wh=0.0 if kind in {"none", "several"} else 8000.0,
        coupling="ac" if ac else "dc",
        battery_inverter_efficiency=0.95 if ac else None,
        battery_inverter_ac_capacity_w=2500.0 if ac else None,
        inverter_ac_capacity_w=4000.0,
        standby_loss_wh=0.01,
        enable_replacement=False,
        eol_percentage=0.999999 if kind == "retires" else 0.7,
        skipped_replacement_action="retire" if kind == "retires" else "keep",
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("freq", ["h", "15min"])
@pytest.mark.parametrize(
    "kind", ["none", "dc", "ac", "several", "several-ac", "retired", "retired-dc", "retired-several-ac", "retires"]
)
def test_q1_q2_q3_q4_stationary_and_unplugged_bit_identity(backend, freq, kind):
    pv, load, temp, units = _inputs(days=3, freq=freq, rich=True)
    cfg = stationary(kind)
    hours = 1.0 if freq == "h" else 0.25
    target = np.where(pv.index.hour < 6, 0.6, np.nan)
    instructions = DispatchInstructions(
        np.isnan(target), np.zeros(len(pv)), target, None if cfg.coupling == "ac" else 0.95, 6000.0
    )
    instruction_hash = instructions.instruction_hash()
    kwargs = dict(
        pv_dc=pv,
        houseload=load,
        temperature_series=temp,
        freq=freq,
        battery_config=cfg,
        execution_backend=backend,
        dispatch_instructions=instructions,
    )
    if "several" in kind:
        kwargs["inverter_units"] = units
    if kind.startswith("retired"):
        kwargs.update(battery_retired=True, initial_energy_wh=0.0)
    baseline = _simulate_core(**kwargs)
    before = baseline.buffers.to_frame(baseline.rng)
    assert LEDGER_SCHEMA_VERSION == "3.1"
    assert (before[list(VEHICLE_COLUMNS)] == 0.0).all().all()
    cfg_v = vehicle()
    resolved = resolve_vehicle_schedule(cfg_v, pv.index, timezone="UTC", reference_year=2025)
    with_vehicle = _simulate_core(
        **kwargs,
        vehicle_config=cfg_v,
        vehicle_schedule=resolved,
        initial_vehicle_energy_wh=30000.0,
        initial_vehicle_pv_origin_energy_wh=10000.0,
        initial_vehicle_grid_origin_energy_wh=5000.0,
    )
    after = with_vehicle.buffers.to_frame(with_vehicle.rng)
    stable = [
        name for name in before if name not in VEHICLE_COLUMNS and name not in {"PV_AC_Export", "Import_From_Grid"}
    ]
    assert_frames_bit_equal(before[stable], after[stable], context="Q2: ")
    assert before["PV_AC_Export"].sum() > 0.0 or after["Vehicle_AC_To_Load"].sum() > 0.0
    assert instructions.instruction_hash() == instruction_hash
    assert baseline.degradation_tracking == with_vehicle.degradation_tracking
    assert_energy_conservation(after, cfg)
    assert_origin_reconciliation(after, hours)
    assert_vehicle_conservation(
        after, cfg_v, hours, initial_energy_wh=30000.0, initial_pv_origin_wh=10000.0, initial_grid_origin_wh=5000.0
    )
    never = schedule(pv.index, hours, np.zeros(len(pv), dtype=int), np.zeros(len(pv), dtype=bool))
    absent = _simulate_core(**kwargs, vehicle_config=cfg_v, vehicle_schedule=never)
    absent_frame = absent.buffers.to_frame(absent.rng)
    assert_frames_bit_equal(
        before.drop(columns=list(VEHICLE_COLUMNS)), absent_frame.drop(columns=list(VEHICLE_COLUMNS)), context="Q3: "
    )
    assert (absent_frame["Vehicle_Energy"] == 0.0).all()
    assert (absent_frame["Vehicle_Energy_Onboard"] == 32000.0).all()
    if kind == "retires":
        assert with_vehicle.aging.retired
        assert after["Vehicle_AC_To_Load"].iloc[int(24 / hours) :].sum() > 0.0


@pytest.mark.parametrize("freq", ["h", "15min"])
@pytest.mark.parametrize("kind", ["none", "dc", "several-ac", "retired"])
def test_vehicle_python_numba_and_summary_agree(freq, kind):
    pytest.importorskip("numba")
    pv, load, temp, units = _inputs(days=3, freq=freq, rich=True)
    cfg = stationary(kind)
    hours = 1.0 if freq == "h" else 0.25
    kwargs = dict(
        pv_dc=pv,
        houseload=load,
        temperature_series=temp,
        battery_config=cfg,
        freq=freq,
        vehicle_config=vehicle(),
        vehicle_schedule=schedule(pv.index, hours),
        initial_vehicle_energy_wh=20000.0,
    )
    if kind == "several-ac":
        kwargs["inverter_units"] = units
    if kind.startswith("retired"):
        kwargs.update(battery_retired=True, initial_energy_wh=0.0)
    python = _simulate_core(**kwargs)
    compiled = _simulate_core(**kwargs, execution_backend="numba")
    assert_frames_bit_equal(python.buffers.to_frame(python.rng), compiled.buffers.to_frame(compiled.rng))
    summary = _build_simulation_summary(_simulate_core(**kwargs, summary_only=True), return_degradation_state=False)
    for name, values in python.buffers.columns.items():
        assert summary.column_sums[name] == float(np.sum(values)), name


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("freq", ["h", "15min"])
@pytest.mark.parametrize("mode", ["uncontrolled", "v1g", "v2h"])
def test_q5_reachable_home_target_never_short(backend, freq, mode):
    hours = 1.0 if freq == "h" else 0.25
    n = int(12 / hours)
    frame = run_home([0.0] * n, [3000.0] * n, 4000.0, list(range(n, 0, -1)), mode, backend, freq)
    assert frame["Vehicle_Energy"].iloc[-1] == 32000.0


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("freq", ["h", "15min"])
def test_mixed_pv_and_grid_share_one_charger_and_site_room(backend, freq):
    frame = run_home([3500.0], [500.0], 20000.0, [1], backend=backend, freq=freq, grid_import_limit_w=4000.0)
    assert frame["PV_AC_To_Vehicle"].iloc[0] == pytest.approx(2860.0)
    assert frame["Grid_AC_To_Vehicle"].iloc[0] == pytest.approx(1140.0)
    assert frame["Import_From_Grid"].iloc[0] == pytest.approx(1140.0)
    assert frame["Vehicle_Charge_Input"].iloc[0] > 0.0


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("freq", ["h", "15min"])
def test_night_draw_is_kept_on_grid_and_taken_from_pv_before_vehicle(backend, freq):
    hours = 1.0 if freq == "h" else 0.25
    index = pd.date_range("2025-06-01", periods=3, freq=freq, tz="UTC")
    units = InverterUnits(
        ac_capacity_w=(np.inf, np.inf),
        efficiency=(1.0, 1.0),
        conversions=(InverterConversion(), InverterConversion(night_consumption_w=8.0)),
        shares=pd.DataFrame({0: [1.0, 1.0, 1.0], 1: [0.0, 0.0, 0.0]}, index=index),
        ac_derate=pd.DataFrame(1.0, index=index, columns=[0, 1]),
        curve_rows=pd.DataFrame(0.0, index=index, columns=[0, 1]),
    )
    cfg = vehicle()
    core = _simulate_core(
        pv_dc=pd.Series([2000.0, 0.0, 0.0], index=index),
        houseload=pd.DataFrame({"Load": [1000.0, 1000.0, 0.0]}, index=index),
        freq=freq,
        inverter_units=units,
        vehicle_config=cfg,
        vehicle_schedule=schedule(index, hours, [200, 199, 1]),
        initial_vehicle_energy_wh=20000.0,
        execution_backend=backend,
        grid_import_limit_w=2000.0,
    )
    frame = core.buffers.to_frame(core.rng)
    assert frame["PV_AC_To_Vehicle"].iloc[0] == pytest.approx(992.0)
    assert frame["Vehicle_AC_To_Load"].iloc[1] == pytest.approx(1000.0)
    assert frame["Import_From_Grid"].iloc[1] == pytest.approx(8.0)
    assert frame["Grid_AC_To_Vehicle"].iloc[2] == pytest.approx(1992.0)
    assert_energy_conservation(frame, BatteryConfig(nominal_energy_wh=0.0))
    assert_vehicle_conservation(frame, cfg, hours, initial_energy_wh=20000.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_periods_and_v2h_floor(backend):
    index = pd.date_range("2025-06-01", periods=4, freq="h", tz="UTC")
    labels = ["off_peak", "peak", "off_peak", "peak"]
    tariff = resolve_tariff(
        index,
        labels,
        TariffSchedule("test", "1", "UTC", "daily", ("peak", "off_peak")),
        TariffPrices("EUR", {"peak": 0.3, "off_peak": 0.1}, {"all": 0.0}),
        timezone="UTC",
    )
    cfg = vehicle(charge_periods=("off_peak",), discharge_periods=("peak",), v2h_min_soc=0.79)
    frame = run_home(
        [0.0] * 4, [3000.0] * 4, 20000.0, [20, 19, 18, 17], backend=backend, vehicle_config=cfg, vehicle_tariff=tariff
    )
    assert frame["Grid_AC_To_Vehicle"].iloc[0] == 7400.0
    assert frame["Grid_AC_To_Vehicle"].iloc[1] == 0.0
    assert frame["Vehicle_AC_To_Load"].iloc[2] == 0.0
    assert frame["Vehicle_Energy"].iloc[-1] == 31600.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_no_departures_does_not_invent_a_deadline(backend):
    frame = run_home([0.0, 0.0], [1000.0, 1000.0], 20000.0, [0, 0], backend=backend)
    assert frame["Grid_AC_To_Vehicle"].sum() == 0.0
    assert frame["Vehicle_AC_To_Load"].sum() == pytest.approx(2000.0)


def test_rollout_and_schedule_validation():
    index = pd.date_range("2025-06-01", periods=1, freq="h", tz="UTC")
    kwargs = dict(pv_dc=pd.Series(0.0, index=index), houseload=pd.DataFrame({"Load": 0.0}, index=index))
    cfg = vehicle()
    resolved = schedule(index, 1.0)
    allowed = _simulate_core(**kwargs, vehicle_config=replace(cfg, enable_replacement=True), vehicle_schedule=resolved)
    assert allowed.vehicle_carry is not None
    with pytest.raises(ValueError, match="together"):
        _simulate_core(**kwargs, vehicle_config=cfg)
    with pytest.raises(ValueError, match="simulation index"):
        _simulate_core(
            **kwargs, vehicle_config=cfg, vehicle_schedule=replace(resolved, index=index + pd.Timedelta(hours=1))
        )
    with pytest.raises(ValueError, match="vehicle_tariff"):
        _simulate_core(**kwargs, vehicle_config=vehicle(charge_periods=("off_peak",)), vehicle_schedule=resolved)
    with pytest.raises(ValueError, match="SOC window"):
        _simulate_core(**kwargs, vehicle_config=cfg, vehicle_schedule=resolved, initial_vehicle_energy_wh=40001.0)


@pytest.mark.parametrize("column", ["Vehicle_Energy_Onboard", "PV_Origin_Vehicle_Charge_Stored", "Vehicle_AC_To_Load"])
def test_vehicle_closure_helper_catches_corruption(column):
    frame = run_home([5000.0], [1000.0], 20000.0, [10])
    frame.loc[0, column] += 1.0
    with pytest.raises(AssertionError):
        assert_vehicle_conservation(frame, vehicle(), 1.0, initial_energy_wh=20000.0)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("freq", ["h", "15min"])
@pytest.mark.parametrize("mode", ["uncontrolled", "v1g", "v2h"])
def test_deadline_with_varied_reachable_starting_states(backend, freq, mode):
    # Non-round initial states exercise inverse conversion and snapping on
    # both sides of a full-step boundary; all have enough home time.
    hours = 1.0 if freq == "h" else 0.25
    random = np.random.default_rng(420)
    for initial in [32000.0 - 6678.5 * hours, 31999.9999999, *random.uniform(4000.0, 32000.0, 4)]:
        n = int(12 / hours)
        frame = run_home([0.0] * n, [1700.0] * n, initial, list(range(n, 0, -1)), mode, backend, freq)
        assert frame["Vehicle_Energy"].iloc[-1] == 32000.0


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("mode,expected", [("uncontrolled", 32000.0), ("v1g", 40000.0), ("v2h", 40000.0)])
def test_pv_stopping_target_and_capacity_ceiling(backend, mode, expected):
    frame = run_home([20000.0] * 4, [0.0] * 4, 30000.0, [20, 19, 18, 17], mode, backend)
    assert frame["Vehicle_Energy"].iloc[-1] == expected
    assert frame["PV_AC_Export"].iloc[-1] == 19200.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_deadline_overrides_charge_period_and_counts_house_import(backend):
    index = pd.date_range("2025-06-01", periods=1, freq="h", tz="UTC")
    tariff = resolve_tariff(
        index,
        ["peak"],
        TariffSchedule("test", "1", "UTC", "daily", ("peak", "off_peak")),
        TariffPrices("EUR", {"all": 0.3}, {"all": 0.0}),
        timezone="UTC",
    )
    cfg = vehicle(charge_periods=("off_peak",))
    frame = run_home(
        [0.0],
        [1500.0],
        20000.0,
        [1],
        backend=backend,
        vehicle_config=cfg,
        vehicle_tariff=tariff,
        grid_import_limit_w=5000.0,
    )
    assert frame["Grid_AC_To_Vehicle"].iloc[0] == 3500.0
    assert frame["Import_From_Grid"].iloc[0] == 5000.0
    assert frame["Vehicle_Energy"].iloc[0] < 32000.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_vehicle_never_supplies_stationary_grid_charge_or_night_draw(backend):
    battery = BatteryConfig(
        nominal_energy_wh=5000.0,
        inverter_ac_capacity_w=3000.0,
        inverter_conversion=InverterConversion(night_consumption_w=8.0),
    )
    instructions = DispatchInstructions(np.array([False]), np.array([0.0]), np.array([0.8]), 0.95, 10000.0)
    frame = run_home(
        [0.0],
        [0.0],
        32000.0,
        [20],
        backend=backend,
        battery_config=battery,
        dispatch_instructions=instructions,
        initial_energy_wh=1000.0,
    )
    assert frame["Grid_AC_To_Battery"].iloc[0] > 0.0
    assert frame["Vehicle_AC_To_Load"].iloc[0] == 0.0
    assert frame["PV_AC_To_Vehicle"].iloc[0] == 0.0
    assert frame["Import_From_Grid"].iloc[0] == frame["Grid_AC_To_Battery"].iloc[0]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("freq", ["h", "15min"])
@pytest.mark.parametrize("surplus", [79.62884490244426, 31.493228349305014, 68.16479436352995])
def test_mixed_source_target_lands_exactly_despite_rounded_ac_sum(backend, freq, surplus):
    hours = 1.0 if freq == "h" else 0.25
    frame = run_home([surplus / 0.96 / hours], [0.0], 31000.0, [1], backend=backend, freq=freq)
    assert frame["Vehicle_Energy"].iloc[0] == 32000.0


def test_enabling_vehicle_reuses_the_numba_signature():
    pytest.importorskip("numba")
    from breos._numba_dispatch import _kernel

    pv, load, temp, _ = _inputs(days=1, freq="h")
    kwargs = dict(
        pv_dc=pv, houseload=load, temperature_series=temp, battery_config=stationary("dc"), execution_backend="numba"
    )
    _simulate_core(**kwargs)
    signatures = set(_kernel().signatures)
    _simulate_core(**kwargs, vehicle_config=vehicle(), vehicle_schedule=schedule(pv.index, 1.0))
    assert set(_kernel().signatures) == signatures
