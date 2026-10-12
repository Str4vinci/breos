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
def curved_inverter_config():
    from breos.pv_modules import MODULES, PVModuleParams

    # Trina Solar TSM-450NEG9R.28, datasheet TSM_EN_2024_C, PDF page 2.
    MODULES["Trina_TSM_450NEG9R28"] = PVModuleParams(
        Mpp=450, Vmp=44.6, Imp=10.09, Voc=52.9, Isc=10.74,
        T_Pmax_pct=-0.29, T_Voc_pct=-0.24, T_Isc_pct=0.04, N_Cells=144, Max_System_Voltage=1500,
    )  # fmt: skip
    yield {
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
    del MODULES["Trina_TSM_450NEG9R28"]


def test_app_runs_the_curve_cable_and_derating_and_reports_them(curved_inverter_config, _patch_weather):
    from breos import App

    resolved = resolve_app_config(curved_inverter_config)
    assert resolved.inverter_conversion.curve_row == 1.0
    assert resolved.inverter_conversion.ac_cable_k_per_w == pytest.approx(resolved.ac_cable.loss_coefficient_per_w)
    app = App(curved_inverter_config)
    app.simulate()
    inverter = app.result()["pv_loss_waterfall"]["inverter"]
    assert inverter["efficiency_curve"] == {"row_position": 1.0, "voltage_v": 400.0}
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


def test_a_curve_needs_a_single_inverter(curved_inverter_config):
    second = {k: v for k, v in curved_inverter_config["inverters"][0].items() if k != "efficiency_curve"}
    curved_inverter_config["inverters"].append(second)
    curved_inverter_config["pv_arrays"].append({"modules": 7, "module": "Trina_TSM_450NEG9R28"})
    curved_inverter_config["pv_strings"].append({"inverter": 1, "mppt": 0, "array": 1, "modules": 7})
    with pytest.raises(ValueError, match="needs a single inverter"):
        resolve_app_config(curved_inverter_config)
