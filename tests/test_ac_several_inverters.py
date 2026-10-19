"""An AC-coupled battery beside several PV inverters (ADR 0004 M8, step 5)."""

import numpy as np
import pandas as pd
import pytest

from breos import App
from breos.app_config import resolve_app_config
from breos.battery import BatteryConfig, simulate_energy_balance
from breos.inverter import InverterConversion, InverterUnits
from tests.backend_parity import BACKENDS
from tests.energy_conservation import assert_energy_conservation
from tests.test_inverter_curve import CURVE
from tests.test_multi_inverter import TWO_INVERTERS, timeseries

# The PV inverters' side of the ledger, which the battery must not change (P1).
PV_SIDE = (
    "PV_DC",
    "PV_DC_To_Inverter",
    "PV_DC_Curtailed",
    "PV_Direct_Inverter_Loss",
    "PV_Production",
    "Inverter_Night_Consumption",
)
APP_AC = {
    "battery_kwh": 5.0,
    "battery_coupling": "ac",
    "battery_inverter_ac_rating_kw": 2.5,
    "battery_inverter_efficiency": 0.95,
    "costs": {"battery_inverter_cost_per_kw": 120.0},
}
TOU = {
    "schedule": "pt_mainland_2026_daily_bi",
    "currency": "EUR",
    "import_prices": {"peak": 0.28, "off_peak": 0.11},
    "export_prices": {"all": 0.05},
}


# --- the dispatch -------------------------------------------------------------


def _inputs(days=3, freq="h", rich=False):
    """Synthetic PV and load, and two inverter units.

    ``rich`` gives the first unit an efficiency curve with a curve row and an
    AC derating that vary per step, and leaves the second unit idle through
    each midday, so it draws its night consumption in daylight.
    """
    per_day = 24 if freq == "h" else 96
    index = pd.date_range("2025-06-01", periods=per_day * days, freq=freq, tz="UTC")
    hours = (np.arange(len(index)) % per_day) * 24 / per_day
    pv = pd.Series(np.clip(np.sin((hours - 6) / 12 * np.pi), 0.0, None) * 5000.0, index=index)
    load = pd.DataFrame({"Load": 500.0 + 900.0 * (hours >= 18)}, index=index)
    east = np.where(hours < 12, 0.7, 0.3)
    first = InverterConversion(night_consumption_w=2.0)
    derate = np.ones(len(index))
    rows = np.zeros(len(index))
    if rich:
        east = np.where((hours >= 11) & (hours < 13), 1.0, east)
        first = InverterConversion(
            np.asarray(CURVE.output_fraction), np.asarray(CURVE.efficiency), 1.0, 2e-6, night_consumption_w=2.0
        )
        derate = np.where((hours >= 12) & (hours < 16), 0.8, 1.0)
        rows = 0.5 + 1.5 * (hours / 24)
    units = InverterUnits(
        ac_capacity_w=(1500.0, 3000.0),
        efficiency=(0.97, None),
        conversions=(first, InverterConversion(night_consumption_w=1.0)),
        shares=pd.DataFrame({0: east, 1: 1.0 - east}, index=index),
        ac_derate=pd.DataFrame({0: derate, 1: 1.0}, index=index),
        curve_rows=pd.DataFrame({0: rows, 1: 0.0}, index=index),
        feeder_cable_k_per_w=1e-6,
    )
    return pv, load, pd.Series(25.0, index=index), units


def _run(battery, backend="python", freq="h", rich=False, days=3, **kwargs):
    pv, load, temperature, units = _inputs(days, freq, rich)
    return simulate_energy_balance(
        pv_dc=pv,
        houseload=load,
        battery_config=battery,
        freq=freq,
        temperature_series=temperature,
        inverter_units=units,
        execution_backend=backend,
        **kwargs,
    )[0]


def _ac(**overrides):
    settings = {
        "nominal_energy_wh": 8000.0,
        "coupling": "ac",
        "battery_inverter_ac_capacity_w": 2500.0,
        "battery_inverter_efficiency": 0.95,
        **overrides,
    }
    return BatteryConfig(**settings)


@pytest.mark.parametrize("rich", [False, True], ids=["plain", "curve-derate-idle"])
@pytest.mark.parametrize("freq", ["h", "15min"])
@pytest.mark.parametrize("backend", BACKENDS)
def test_each_pv_inverter_converts_as_without_the_battery(backend, freq, rich):
    pv_only = _run(BatteryConfig(nominal_energy_wh=0.0), backend, freq, rich)
    battery = _ac()
    frame = _run(battery, backend, freq, rich)
    assert pv_only["PV_DC_Curtailed"].sum() > 0.0, "the small inverter must clip"
    assert pv_only["Inverter_Night_Consumption"].sum() > 0.0
    if rich:
        daylight = (pv_only["PV_DC"] > 0.0).to_numpy()
        assert pv_only["Inverter_Night_Consumption"].to_numpy()[daylight].sum() > 0.0, "an idle unit draws by day"
    for column in PV_SIDE:
        np.testing.assert_array_equal(frame[column].to_numpy(), pv_only[column].to_numpy(), err_msg=column)
    assert frame["PV_AC_To_Battery"].sum() > 0.0
    assert frame["Battery_AC_To_Load"].sum() > 0.0
    assert (frame["PV_DC_To_Battery"] == 0.0).all()
    assert_energy_conservation(frame, battery)


def test_both_backends_agree_beside_several_inverters():
    pytest.importorskip("numba")
    python, compiled = _run(_ac(), "python"), _run(_ac(), "numba")
    pd.testing.assert_frame_equal(python, compiled, check_exact=False, rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize("freq", ["h", "15min"])
@pytest.mark.parametrize("backend", BACKENDS)
def test_a_retired_pack_is_the_pv_only_run_with_several_inverters(backend, freq):
    pv_only = _run(BatteryConfig(nominal_energy_wh=0.0), backend, freq, rich=True)
    retired = _run(_ac(), backend, freq, rich=True, battery_retired=True, initial_energy_wh=0.0)
    for column in (*PV_SIDE, "PV_AC_To_Load", "PV_AC_Export", "Import_From_Grid"):
        np.testing.assert_array_equal(retired[column].to_numpy(), pv_only[column].to_numpy(), err_msg=column)


# A pack that crosses its threshold at the first day close and is retired there.
RETIRING = {"eol_percentage": 0.99999, "enable_replacement": False, "skipped_replacement_action": "retire"}


def _retired_steps(frame):
    steps = (frame["Battery_SOH"] < 100.0).to_numpy() & (frame["Battery_Energy"] == 0.0).to_numpy()
    steps &= (frame["Battery_Energy_Beginning"] == 0.0).to_numpy()
    assert 0 < steps.sum() < len(frame), "the pack must retire inside the span"
    return steps


@pytest.mark.parametrize("units", [True, False], ids=["several", "one"])
@pytest.mark.parametrize("freq", ["h", "15min"])
@pytest.mark.parametrize("backend", BACKENDS)
def test_a_pack_retired_during_the_span_reads_the_pv_chain(backend, freq, units):
    # The retired steps read the span's PV chain, with several inverters or one.
    pv, load, temperature, inverter_units = _inputs(3, freq, rich=True)
    kwargs = {
        "pv_dc": pv,
        "houseload": load,
        "freq": freq,
        "temperature_series": temperature,
        "execution_backend": backend,
        "inverter_units": inverter_units if units else None,
    }
    single = {} if units else {"inverter_ac_capacity_w": 2000.0}
    pv_only = simulate_energy_balance(battery_config=BatteryConfig(nominal_energy_wh=0.0, **single), **kwargs)[0]
    frame = simulate_energy_balance(battery_config=_ac(**RETIRING, **single), **kwargs)[0]
    retired = _retired_steps(frame)
    for column in (*PV_SIDE, "PV_AC_To_Load", "PV_AC_Export", "Import_From_Grid"):
        np.testing.assert_array_equal(
            frame[column].to_numpy()[retired], pv_only[column].to_numpy()[retired], err_msg=column
        )


def test_a_dc_coupled_battery_beside_several_inverters_is_refused():
    with pytest.raises(ValueError, match="a battery beside several inverters must be AC-coupled"):
        _run(BatteryConfig(nominal_energy_wh=8000.0))


# --- the App ------------------------------------------------------------------


def test_an_app_run_keeps_each_inverters_pv_and_charges_the_battery(trina, _patch_weather):
    pv_only, _ = timeseries(TWO_INVERTERS)
    frame, result = timeseries({**TWO_INVERTERS, **APP_AC})
    for column in PV_SIDE:
        np.testing.assert_array_equal(frame[column].to_numpy(), pv_only[column].to_numpy(), err_msg=column)
    assert frame["PV_AC_To_Battery"].sum() > 0.0
    waterfall = result["pv_loss_waterfall"]
    assert len(waterfall["inverter"]["inverters"]) == 2
    assert waterfall["battery_inverter"]["ac_capacity_kw"] == 2.5


def test_two_project_years_carry_the_battery_and_discharge_only_runs(
    trina,
    _patch_weather,
    monkeypatch,
):
    import breos.projection as projection_module

    runs = []
    detailed = projection_module._simulate_detailed_run

    def record(*args, **kwargs):
        run = detailed(*args, **kwargs)
        runs.append((kwargs, run))
        return run

    monkeypatch.setattr(projection_module, "_simulate_detailed_run", record)
    discharge_only = {"mode": "discharge_only", "discharge_periods": ["peak"]}
    app = App({**TWO_INVERTERS, **APP_AC, "projection_years": 2, "tariff": TOU, "smart_charging": discharge_only})
    app.simulate()
    result = app.result()
    assert app._artifacts is not None
    years = app._artifacts.yearly_df
    assert len(years) == 2 and len(runs) == 2
    assert (years["PV_AC_To_Battery_kWh"] > 0.0).all()
    assert (years["Grid_AC_To_Battery_kWh"] == 0.0).all()
    # The second year opens on the stored energy and origins the first one closed on.
    first = runs[0][1].results_df.iloc[-1]
    second = runs[1][0]
    assert second["initial_energy_wh"] == first["Battery_Energy_End"]
    assert second["initial_pv_origin_energy_wh"] == first["Battery_PV_Origin_Energy_End"]
    assert second["initial_grid_origin_energy_wh"] == first["Battery_Grid_Origin_Energy_End"]
    # The battery discharges in peak steps and nowhere else.
    for _kwargs, run in runs:
        allowed = run.controller_instructions.discharge_allowed
        assert 0.3 < allowed.mean() < 0.8, "peak is part of each day"
        discharge = run.results_df["Battery_AC_To_Load"].to_numpy()
        assert (discharge[~allowed] == 0.0).all()
        assert discharge[allowed].sum() > 0.0
    rows = result["pv_loss_waterfall"]["inverter"]["inverters"]
    pv_dc = result["pv_loss_waterfall"]["energy_balance"]["pv_dc"]
    assert sum(row["dc_input_kwh"] for row in rows) == pytest.approx(pv_dc["generation_kwh"], abs=0.02)
    assert sum(row["clipping_kwh"] for row in rows) == pytest.approx(pv_dc["curtailed_kwh"], abs=0.02)
    # Each inverter's report is the one it gives without the battery.
    _, pv_only = timeseries({**TWO_INVERTERS, "projection_years": 2, "tariff": TOU})
    assert rows == pv_only["pv_loss_waterfall"]["inverter"]["inverters"]


def test_fixed_target_runs_and_the_planner_is_refused_beside_several_inverters(trina, _patch_weather):
    fixed = {
        "mode": "fixed_target",
        "target_usable_fraction": 0.5,
        "charge_periods": ["off_peak"],
        "discharge_periods": ["peak"],
    }
    frame, _ = timeseries({**TWO_INVERTERS, **APP_AC, "tariff": TOU, "smart_charging": fixed})
    assert frame["Grid_AC_To_Battery"].sum() > 0.0
    planner = {"mode": "daily_persistence", "charge_periods": ["off_peak"], "discharge_periods": ["peak"]}
    with pytest.raises(ValueError, match="plans with one PV inverter, not 2"):
        resolve_app_config({**TWO_INVERTERS, **APP_AC, "tariff": TOU, "smart_charging": planner})


def test_app_refuses_a_dc_battery_beside_several_inverters(trina):
    with pytest.raises(ValueError, match="a battery beside 2 inverters must be AC-coupled"):
        resolve_app_config({**TWO_INVERTERS, "battery_kwh": 5.0})
    App({**TWO_INVERTERS, **APP_AC})


def test_the_daily_target_oracle_refuses_several_inverters_in_both_planning_modes(trina, _patch_weather):
    from tools.oracles.daily_target_dp import YearlyDailyTargetPlanner, daily_target_problem
    from tools.oracles.replay import prepare_replay

    fixed = {
        "mode": "fixed_target",
        "target_usable_fraction": 0.5,
        "charge_periods": ["off_peak"],
        "discharge_periods": ["peak"],
    }
    case = prepare_replay(
        {
            **TWO_INVERTERS,
            **APP_AC,
            "tariff": TOU,
            "smart_charging": fixed,
            "start_date": "2025-01-01",
            "period": {"start": "2025-03-01", "end": "2025-03-05"},
        }
    )
    with pytest.raises(ValueError, match="plans with one PV inverter, not 2"):
        daily_target_problem(case)
    with pytest.raises(ValueError, match="plans with one PV inverter, not 2"):
        YearlyDailyTargetPlanner(
            case, target_levels=np.array([0.0, 1.0]), soc_states=3, free_terminal=False, execution_backend="python"
        )
