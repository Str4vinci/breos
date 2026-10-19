"""A measured inverter efficiency curve, an AC cable and temperature derating in every conversion path."""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from breos._dispatch import _dc_ac, _dc_for_ac
from breos.app_config import resolve_app_config
from breos.battery import BatteryConfig, simulate_energy_balance
from breos.equipment import EfficiencyCurve, InverterSpec, TemperatureDerating
from breos.inverter import NO_CONVERSION, AcCable, InverterConversion, _calculate_dc_ac_power_arrays, ac_derate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "parity"))

from harness import build  # noqa: E402

from tests.energy_conservation import assert_energy_conservation  # noqa: E402

# A synthetic curve with the CEC shape, for the tests only: three voltages,
# efficiency rising with output and peaking at the middle voltage.
CURVE = EfficiencyCurve(
    output_fraction=(0.1, 0.2, 0.3, 0.5, 0.75, 1.0),
    efficiency=(
        (0.950, 0.966, 0.971, 0.974, 0.973, 0.970),
        (0.955, 0.970, 0.975, 0.978, 0.977, 0.975),
        (0.948, 0.965, 0.971, 0.975, 0.975, 0.973),
    ),
    voltages_v=(250.0, 400.0, 550.0),
)
NAMEPLATE = 5000.0


def _conversion(row: float = 1.0, cable_k: float = 0.0) -> InverterConversion:
    return InverterConversion(np.asarray(CURVE.output_fraction), np.asarray(CURVE.efficiency), row, cable_k)


def _forward(dc: float, conversion: InverterConversion, *, nameplate: float = NAMEPLATE) -> tuple[float, float, float]:
    return _dc_ac(dc, nameplate, 0.96, 1.0, 2.0, *conversion.kernel_arguments(1.0))


def _inverse(ac: float, conversion: InverterConversion, *, nameplate: float = NAMEPLATE) -> float:
    return _dc_for_ac(ac, nameplate, 0.96, 1.0, *conversion.kernel_arguments(1.0))


@pytest.mark.parametrize("row", [0.0, 1.0, 0.5, 1.7])
def test_the_curve_passes_through_its_points_and_inverts_exactly(row):
    conversion = _conversion(row)
    lower = min(int(math.floor(row)), 1)
    weight = row - lower
    for j, fraction in enumerate(CURVE.output_fraction):
        eta = CURVE.efficiency[lower][j] + weight * (CURVE.efficiency[lower + 1][j] - CURVE.efficiency[lower][j])
        ac, loss, clipped = _forward(fraction * NAMEPLATE / eta, conversion)
        assert ac == pytest.approx(fraction * NAMEPLATE, rel=1e-12)
        assert clipped == pytest.approx(0.0, abs=1e-9)
    for dc in np.linspace(50.0, 5000.0, 97):
        ac = _forward(dc, conversion)[0]
        assert _inverse(ac, conversion) == pytest.approx(dc, rel=1e-12)


def test_self_consumption_below_the_first_point_and_clipping_above_the_nameplate():
    conversion = _conversion(1.0)
    first_dc = 0.1 * NAMEPLATE / 0.955
    own_use = first_dc - 0.1 * NAMEPLATE
    assert _forward(first_dc / 2, conversion)[0] == pytest.approx(first_dc / 2 - own_use)
    assert _forward(own_use / 2, conversion)[0] == 0.0
    full_dc = NAMEPLATE / 0.975
    ac, loss, clipped = _forward(full_dc + 700.0, conversion)
    assert ac == pytest.approx(NAMEPLATE)
    assert clipped == pytest.approx(700.0)
    assert _inverse(NAMEPLATE * 2, conversion) == pytest.approx(full_dc)


def test_the_ac_cable_takes_its_share_and_the_inverse_accounts_for_it():
    k = 2e-6
    bare, cabled = _conversion(1.0), _conversion(1.0, cable_k=k)
    for dc in (800.0, 2500.0, 4900.0):
        ac = _forward(dc, bare)[0]
        assert _forward(dc, cabled)[0] == pytest.approx(ac - k * ac * ac, rel=1e-12)
        delivered = _forward(dc, cabled)[0]
        assert _inverse(delivered, cabled) == pytest.approx(dc, rel=1e-10)
    # The PVWatts path takes the cable too.
    pvwatts_cabled = InverterConversion(ac_cable_k_per_w=k)
    plain = _forward(3000.0, NO_CONVERSION)[0]
    assert _forward(3000.0, pvwatts_cabled)[0] == pytest.approx(plain - k * plain * plain, rel=1e-12)


def test_the_curve_and_derating_are_validated():
    assert CURVE.row_position(250) == 0 and CURVE.row_position(475) == 1.5 and CURVE.row_position(900) == 2
    with pytest.raises(ValueError, match="increase"):
        EfficiencyCurve((0.5, 0.2), ((0.97, 0.96),))
    with pytest.raises(ValueError, match="one value per output_fraction"):
        EfficiencyCurve((0.2, 0.5), ((0.97,),))
    with pytest.raises(ValueError, match="voltages_v"):
        EfficiencyCurve((0.2, 0.5), ((0.96, 0.97), (0.95, 0.96)))
    with pytest.raises(ValueError, match="rises too steeply"):
        EfficiencyCurve((0.5, 0.51), ((0.90, 0.99),))
    with pytest.raises(ValueError, match="increase"):
        TemperatureDerating((50, 40), (1.0, 0.8))
    spec = InverterSpec(
        nominal_power_w=5000,
        efficiency_curve={"output_fraction": [0.5, 1.0], "efficiency": [[0.97, 0.96]]},
        temperature_derating={"ambient_temp_c": [45, 60], "max_output_fraction": [1.0, 0.5]},
    )
    assert spec.efficiency_curve.efficiency == ((0.97, 0.96),)
    np.testing.assert_allclose(ac_derate(spec.temperature_derating, np.array([20, 45, 52.5, 70])), [1, 1, 0.75, 0.5])


def test_the_ac_cable_loss_coefficient_follows_the_phases():
    single = AcCable(length_m=20, cross_section_mm2=6, phases=1, voltage_v=230)
    three = AcCable(length_m=20, cross_section_mm2=6, phases=3, voltage_v=400, power_factor=0.9)
    r = 1.7241e-8 * (1 + 0.00393 * 30) * 20 / 6e-6
    assert single.conductor_resistance_ohm == pytest.approx(r)
    assert single.loss_coefficient_per_w == pytest.approx(2 * r / 230**2)
    assert three.loss_coefficient_per_w == pytest.approx(r / (400 * 0.9) ** 2)
    assert single.loss_pct_at(4600) == pytest.approx(100 * 2 * r * (4600 / 230) ** 2 / 4600)
    with pytest.raises(ValueError, match="phases"):
        AcCable(length_m=20, cross_section_mm2=6, phases=2, voltage_v=230)


def test_the_pv_only_path_runs_the_kernels_conversion():
    dc = np.linspace(0.0, 6500.0, 211)
    derate = np.linspace(1.0, 0.4, 211)
    conversion = _conversion(0.6, cable_k=1e-6)
    ac, loss, clipped = _calculate_dc_ac_power_arrays(dc, NAMEPLATE, 0.96, conversion=conversion, ac_derate=derate)
    for i in range(len(dc)):
        expected = _dc_ac(float(dc[i]), NAMEPLATE * derate[i], 0.96, 1.0, 2.0, *conversion.kernel_arguments(1.0))
        assert (ac[i], loss[i], clipped[i]) == expected
    # Without either, the vectorised PVWatts path is unchanged.
    plain = _calculate_dc_ac_power_arrays(dc, NAMEPLATE, 0.96)
    np.testing.assert_array_equal(
        _calculate_dc_ac_power_arrays(dc, NAMEPLATE, 0.96, conversion=NO_CONVERSION)[0], plain[0]
    )


def _battery_run(backend: str, *, conversion=None, derate: bool = False):
    pv, load, temp, cfg, _ = build("discharge_limited", "h")
    cfg = {**cfg, "inverter_conversion": conversion}
    derating = None
    if derate:
        # A hot afternoon every day: the rating falls to 60 % at peak.
        hour = pv.index.hour.to_numpy()
        derating = pd.Series(np.where((hour >= 12) & (hour <= 16), 0.6, 1.0), index=pv.index)
    return simulate_energy_balance(
        pv_dc=pv,
        houseload=load,
        battery_config=BatteryConfig(**cfg),
        freq="h",
        temperature_series=temp,
        execution_backend=backend,
        inverter_ac_derate=derating,
    )[0]


def test_a_battery_run_uses_the_curve_cable_and_derating():
    plain = _battery_run("python")
    curved = _battery_run("python", conversion=_conversion(1.0, cable_k=2e-6), derate=True)
    assert curved["PV_DC_Curtailed"].sum() > plain["PV_DC_Curtailed"].sum()
    assert not np.array_equal(curved["Inverter_Loss"].to_numpy(), plain["Inverter_Loss"].to_numpy())
    # The battery takes DC above the derated rating before it is curtailed.
    assert curved["PV_DC_To_Battery"].sum() > plain["PV_DC_To_Battery"].sum()
    # Energy still balances at the inverter's DC side.
    dc = curved["PV_DC"] - curved["PV_DC_To_Battery"] - curved["PV_DC_Curtailed"]
    np.testing.assert_allclose(dc, curved["PV_DC_To_Inverter"], atol=1e-6)


def test_numba_matches_python_with_the_curve_cable_and_derating():
    pytest.importorskip("numba", reason="the compiled backend needs the breos[fast] extra")
    conversion = _conversion(1.3, cable_k=2e-6)
    python = _battery_run("python", conversion=conversion, derate=True)
    compiled = _battery_run("numba", conversion=conversion, derate=True)
    for column in python.columns:
        if column == "Datetime":
            continue
        assert np.array_equal(python[column].to_numpy(), compiled[column].to_numpy()), column


@pytest.fixture
def curved_inverter_config(trina):
    return {
        "location": "porto",
        "annual_consumption_kwh": 4000,
        "pv_arrays": [{"modules": 14, "module": "Trina_TSM_450NEG9R28", "tilt": 30, "azimuth": 180}],
        "inverters": [
            {
                "nominal_power_w": 5000,
                "inverter_efficiency": 0.975,
                "max_dc_voltage_v": 1000,
                "min_mppt_voltage_v": 80,
                "max_mppt_voltage_v": 800,
                "startup_voltage_v": 80,
                "isc_comparison_factor": 1.0,
                "mppts": [{"max_input_current_a": 25, "max_short_circuit_current_a": 40, "max_strings": 2}],
                "efficiency_curve": {
                    "output_fraction": list(CURVE.output_fraction),
                    "efficiency": [list(row) for row in CURVE.efficiency],
                    "voltages_v": list(CURVE.voltages_v),
                },
                "temperature_derating": {"ambient_temp_c": [20, 30], "max_output_fraction": [1.0, 0.7]},
            }
        ],
        "pv_strings": [{"inverter": 0, "mppt": 0, "array": 0, "modules": 7, "parallel": 2}],
        "electrical_environment": {"min_design_temp_c": -5, "max_ambient_temp_c": 40},
        "ac_cable": {"length_m": 25, "cross_section_mm2": 4, "phases": 1, "voltage_v": 230},
    }


def test_app_runs_the_curve_cable_and_derating_and_reports_them(curved_inverter_config, _patch_weather):
    from breos import App

    resolved = resolve_app_config(curved_inverter_config)
    assert resolved.inverter_conversion.curve_row == 1.0
    assert resolved.inverter_conversion.ac_cable_k_per_w == pytest.approx(resolved.ac_cable.loss_coefficient_per_w)
    app = App(curved_inverter_config)
    app.simulate()
    inverter = app.result()["pv_loss_waterfall"]["inverter"]
    assert inverter["efficiency_curve"] == {"row_position": 1.0, "voltage_v": 400.0, "voltage_basis": "fixed"}
    assert inverter["ac_cable"]["conductor_temp_stated"] is False
    assert 0 < inverter["ac_cable"]["loss_kwh"] < inverter["conversion_loss_kwh"]
    assert inverter["temperature_derating"]["ambient"] == "weather"
    assert inverter["temperature_derating"]["min_output_fraction"] < 1

    plain = {k: v for k, v in curved_inverter_config.items() if k != "ac_cable"}
    plain["inverters"] = [
        {k: v for k, v in plain["inverters"][0].items() if k not in ("efficiency_curve", "temperature_derating")}
    ]
    assert resolve_app_config(plain).inverter_conversion is None
    reference = App(plain)
    reference.simulate()
    assert app.result()["usable_ac_system_production_kwh"] != reference.result()["usable_ac_system_production_kwh"]


def test_string_level_energy_reads_the_curve_at_the_strings_voltage(curved_inverter_config):
    resolved = resolve_app_config({**curved_inverter_config, "electrical_fidelity": "string"})
    # 7 modules at 44.6 V: 312.2 V, between the 250 V and 400 V rows.
    assert resolved.inverter_conversion.curve_row == pytest.approx((7 * 44.6 - 250) / 150)


def test_a_curve_with_several_inverters_needs_string_level_energy(curved_inverter_config):
    second = {k: v for k, v in curved_inverter_config["inverters"][0].items() if k != "efficiency_curve"}
    curved_inverter_config["inverters"].append(second)
    curved_inverter_config["pv_arrays"].append({"modules": 7, "module": "Trina_TSM_450NEG9R28"})
    curved_inverter_config["pv_strings"].append({"inverter": 1, "mppt": 0, "array": 1, "modules": 7})
    with pytest.raises(ValueError, match="several inverters needs electrical_fidelity = 'string'"):
        resolve_app_config(curved_inverter_config)


# -- per-step operating voltage and night consumption ---------------------------------------------


def test_the_pv_only_path_reads_each_steps_row():
    dc = np.linspace(0.0, 6500.0, 211)
    rows = np.linspace(0.0, 2.0, 211)
    conversion = _conversion(1.0, cable_k=1e-6)
    ac, loss, clipped = _calculate_dc_ac_power_arrays(dc, NAMEPLATE, 0.96, conversion=conversion, curve_rows=rows)
    curve_ac, curve_eff, _, cable_k = conversion.kernel_arguments(1.0)
    for i in range(len(dc)):
        expected = _dc_ac(float(dc[i]), NAMEPLATE, 0.96, 1.0, 2.0, curve_ac, curve_eff, float(rows[i]), cable_k)
        assert (ac[i], loss[i], clipped[i]) == expected


def _rows(pv: pd.Series) -> pd.Series:
    # A morning at the lowest voltage row, an afternoon at the highest.
    return pd.Series(np.where(pv.index.hour < 12, 0.0, 2.0), index=pv.index)


def _night_run(backend: str, *, conversion, rows: bool = False):
    pv, load, temp, cfg, _ = build("discharge_limited", "h")
    return simulate_energy_balance(
        pv_dc=pv,
        houseload=load,
        battery_config=BatteryConfig(**{**cfg, "inverter_conversion": conversion}),
        freq="h",
        temperature_series=temp,
        execution_backend=backend,
        inverter_curve_row=_rows(pv) if rows else None,
    )[0]


def test_a_row_series_holding_the_conversions_row_changes_no_bit():
    pv, load, temp, cfg, _ = build("discharge_limited", "h")
    conversion = _conversion(1.3, cable_k=2e-6)
    runs = [
        simulate_energy_balance(
            pv_dc=pv,
            houseload=load,
            battery_config=BatteryConfig(**{**cfg, "inverter_conversion": conversion}),
            freq="h",
            temperature_series=temp,
            inverter_curve_row=rows,
        )[0]
        for rows in (None, pd.Series(1.3, index=pv.index))
    ]
    pd.testing.assert_frame_equal(runs[0], runs[1], check_exact=True)


def test_each_steps_row_moves_the_conversion():
    fixed = _night_run("python", conversion=_conversion(1.0))
    moving = _night_run("python", conversion=_conversion(1.0), rows=True)
    # Rows 0 and 2 are both below the middle row's efficiency at every point.
    assert moving["Inverter_Loss"].sum() > fixed["Inverter_Loss"].sum()


def test_the_inverter_draws_its_night_consumption_from_the_grid_while_it_converts_nothing():
    night = InverterConversion(night_consumption_w=6.0)
    plain = _night_run("python", conversion=NO_CONVERSION)
    drawn = _night_run("python", conversion=night)
    idle = (drawn["PV_DC"] <= 0) & (drawn["Battery_Charge_Input"] <= 0) & (drawn["Battery_Discharge_DC"] <= 0)
    assert idle.any() and not idle.all()
    np.testing.assert_array_equal(drawn["Inverter_Night_Consumption"], np.where(idle, 6.0, 0.0))
    np.testing.assert_allclose(
        drawn["Import_From_Grid"] - plain["Import_From_Grid"], drawn["Inverter_Night_Consumption"], atol=1e-9
    )
    pv, load, temp, cfg, _ = build("discharge_limited", "h")
    assert_energy_conservation(drawn, BatteryConfig(**{**cfg, "inverter_conversion": night}))
    # It is a draw on the grid, not a dispatch decision: the battery runs as before.
    for column in ("Battery_Energy", "Battery_Discharge_DC", "PV_AC_Export", "Houseload"):
        np.testing.assert_array_equal(drawn[column], plain[column], err_msg=column)


def test_a_pv_only_run_draws_the_night_consumption_where_there_is_no_pv():
    pv, load, _, _, _ = build("discharge_limited", "h")
    config = BatteryConfig(nominal_energy_wh=0.0, inverter_ac_capacity_w=NAMEPLATE)
    runs = {
        name: simulate_energy_balance(pv_dc=pv, houseload=load, battery_config=replace_conversion(config, conversion))[
            0
        ]
        for name, conversion in (("plain", NO_CONVERSION), ("drawn", InverterConversion(night_consumption_w=4.0)))
    }
    expected = np.where(runs["drawn"]["PV_DC"] <= 0, 4.0, 0.0)
    np.testing.assert_array_equal(runs["drawn"]["Inverter_Night_Consumption"], expected)
    np.testing.assert_allclose(runs["drawn"]["Import_From_Grid"] - runs["plain"]["Import_From_Grid"], expected)


def replace_conversion(config: BatteryConfig, conversion: InverterConversion) -> BatteryConfig:
    from dataclasses import replace

    return replace(config, inverter_conversion=conversion)


def test_numba_matches_python_with_each_steps_row_and_the_night_consumption():
    pytest.importorskip("numba", reason="the compiled backend needs the breos[fast] extra")
    conversion = InverterConversion(
        np.asarray(CURVE.output_fraction), np.asarray(CURVE.efficiency), 1.0, 2e-6, night_consumption_w=7.5
    )
    python = _night_run("python", conversion=conversion, rows=True)
    compiled = _night_run("numba", conversion=conversion, rows=True)
    for column in python.columns:
        if column == "Datetime":
            continue
        assert np.array_equal(python[column].to_numpy(), compiled[column].to_numpy()), column


def test_night_consumption_is_validated():
    with pytest.raises(ValueError, match="night_consumption_w"):
        InverterSpec(nominal_power_w=5000, night_consumption_w=-1)
    with pytest.raises(ValueError, match="night_consumption_w"):
        InverterConversion(night_consumption_w=math.inf)


def test_string_level_energy_reads_the_curve_at_each_steps_operating_voltage(
    curved_inverter_config, _patch_weather, monkeypatch
):
    from breos import App, app_inputs

    config = {**curved_inverter_config, "electrical_fidelity": "string"}
    config["inverters"] = [{**config["inverters"][0], "night_consumption_w": 5.0}]
    resolved = resolve_app_config(config)
    inputs = app_inputs.prepare_simulation_inputs(resolved.cfg, resolved, App._runtime_dependencies())
    voltage = inputs.pv_breakdown.dc_voltage
    rows = inputs.inverter_curve_row
    lit = voltage > 0
    assert lit.any() and (~lit).any()
    np.testing.assert_allclose(rows[lit], np.interp(voltage[lit], CURVE.voltages_v, [0.0, 1.0, 2.0]))
    assert (rows[~lit] == resolved.inverter_conversion.curve_row).all()

    app = App(config)
    app.simulate()
    result = app.result()
    inverter = result["pv_loss_waterfall"]["inverter"]
    curve = inverter["efficiency_curve"]
    assert curve["voltage_basis"] == "operating"
    # 7 modules at their operating point: below their STC 312 V on a warm day.
    assert 200 < curve["operating_voltage_v"] < 7 * 44.6
    night = inverter["night_consumption"]
    assert night["power_w"] == 5.0 and night["hours"] > 0
    assert night["energy_kwh"] == pytest.approx(5.0 * night["hours"] / 1000, abs=0.01)

    # Without the per-step rows the same run converts differently.
    monkeypatch.setattr(app_inputs, "build_inverter_curve_row", lambda *args, **kwargs: None)
    fixed = App(config)
    fixed.simulate()
    assert fixed.result()["usable_ac_system_production_kwh"] != result["usable_ac_system_production_kwh"]


def test_monte_carlo_years_carry_their_own_curve_rows(curved_inverter_config, synthetic_weather):
    from breos.montecarlo import _build_pv_years

    config = {**curved_inverter_config, "electrical_fidelity": "string"}
    resolved = resolve_app_config(config)
    dc, _, inverter = _build_pv_years(resolved.cfg, resolved, {2023: synthetic_weather})
    rows = inverter[2023].curve_row
    assert rows is not None and rows.index.equals(dc[2023].index)
    assert rows[dc[2023] > 0].nunique() > 1
    plain = resolve_app_config(curved_inverter_config)
    assert _build_pv_years(plain.cfg, plain, {2023: synthetic_weather})[2][2023].curve_row is None
