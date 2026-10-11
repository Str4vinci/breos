"""String-level energy applies the inverter's input limits and start-up voltage."""

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pvlib
import pytest

from breos import App
from breos.app_config import resolve_app_config
from breos.electrical import (
    ElectricalEnvironment,
    ElectricalSetup,
    PVString,
    check_topology,
    string_limited_dc,
)
from breos.equipment import InverterSpec, MpptSpec
from breos.pv_modules import MODULES, PVModuleParams
from breos.solar import PVProductionBreakdown, _module_dc_before_losses

# Trina Solar TSM-450NEG9R.28, datasheet TSM_EN_2024_C, PDF page 2:
# https://static.trinasolar.com/sites/default/files/Datasheet_Vertex%20S%2B_NEG9R.28_EN_2024_C_web.pdf
TRINA_450 = PVModuleParams(
    Mpp=450,
    Vmp=44.6,
    Imp=10.09,
    Voc=52.9,
    Isc=10.74,
    T_Pmax_pct=-0.29,
    T_Voc_pct=-0.24,
    T_Isc_pct=0.04,
    N_Cells=144,
    Max_System_Voltage=1500,
)

# Jinko Solar JKM440N-54HL4R-B, datasheet EU-JKM430-455N-54HL4R-B-G3-EN, PDF page 2:
# https://jinkosolar.eu/wp-content/uploads/EU-JKM430-455N-54HL4R-B-G3-EN.pdf
JINKO_440 = PVModuleParams(
    Mpp=440,
    Vmp=32.99,
    Imp=13.34,
    Voc=39.57,
    Isc=13.8,
    T_Pmax_pct=-0.29,
    T_Voc_pct=-0.25,
    T_Isc_pct=0.045,
    N_Cells=108,
)

# Fronius Symo GEN24 6.0, technical data, retrieved 2026-10-11:
# https://www.fronius.com/en-gb/uk/solar-energy/installers-partners/technical-data/all-products/inverters/fronius-symo-gen24/fronius-symo-gen24-6-0
FRONIUS_GEN24_6 = {
    "nominal_power_w": 6000,
    "inverter_efficiency": 0.982,
    "max_dc_voltage_v": 1000,
    "max_dc_power_w": 9000,
    "min_mppt_voltage_v": 80,
    "max_mppt_voltage_v": 800,
    "startup_voltage_v": 80,
    "isc_comparison_factor": 1.0,
    "mppts": [
        {"max_input_current_a": 25, "max_short_circuit_current_a": 40, "max_strings": 2},
        {"max_input_current_a": 12.5, "max_short_circuit_current_a": 20, "max_strings": 1},
    ],
}

# The Isc comparison basis was not transcribed; 1 states a direct comparison for these tests.
PORTO = {"min_design_temp_c": -5, "max_ambient_temp_c": 40}


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.setitem(MODULES, "Trina_TSM_450NEG9R28", TRINA_450)
    monkeypatch.setitem(MODULES, "Jinko_JKM440N_54HL4R_B", JINKO_440)


def config(arrays, strings, inverter=FRONIUS_GEN24_6, **extra):
    return {
        "location": "porto",
        "annual_consumption_kwh": 4000,
        "pv_arrays": arrays,
        "inverters": [inverter],
        "pv_strings": strings,
        "electrical_environment": PORTO,
        **extra,
    }


EAST_WEST = config(
    [
        {"modules": 7, "module": "Trina_TSM_450NEG9R28", "tilt": 20, "azimuth": 90},
        {"modules": 7, "module": "Trina_TSM_450NEG9R28", "tilt": 20, "azimuth": 270},
    ],
    [
        {"inverter": 0, "mppt": 0, "array": 0, "modules": 7},
        {"inverter": 0, "mppt": 1, "array": 1, "modules": 7},
    ],
)


def run(cfg):
    app = App(cfg)
    app.simulate()
    return app.result()


def stage(result, key):
    return next((s for s in result["pv_loss_waterfall"]["stages"] if s["key"] == key), None)


def test_a_design_inside_every_limit_matches_the_aggregate_model(modules, _patch_weather):
    aggregate = run(EAST_WEST)
    string = run({**EAST_WEST, "electrical_fidelity": "string"})
    assert string["electrical"]["fidelity"] == "string"
    for key in ("pv_dc_generation_kwh", "usable_ac_system_production_kwh", "npv_savings"):
        assert string[key] == aggregate[key]
    assert stage(string, "mppt_limits")["delta_kwh"] == 0
    assert stage(aggregate, "mppt_limits") is None


def test_the_operating_current_limit_costs_energy_at_peak(modules, _patch_weather):
    # The synthetic test year peaks at 800 W/m2, where Jinko's string carries
    # about 8.8 A after the static losses: under the second input's 12.5 A, so
    # a what-if 8 A limit binds.
    inverter = {**FRONIUS_GEN24_6, "mppts": [FRONIUS_GEN24_6["mppts"][0], {"max_input_current_a": 8}]}
    cfg = config(
        [{"modules": 10, "module": "Jinko_JKM440N_54HL4R_B", "tilt": 35, "azimuth": 180}],
        [{"inverter": 0, "mppt": 1, "array": 0, "modules": 10}],
        inverter=inverter,
    )
    aggregate, string = run(cfg), run({**cfg, "electrical_fidelity": "string"})
    assert "imp_above_mppt_limit" in {f["code"] for f in string["electrical"]["findings"]}
    loss = -stage(string, "mppt_limits")["delta_kwh"]
    assert 0 < loss < 0.2 * aggregate["pv_dc_generation_kwh"]
    assert string["pv_dc_generation_kwh"] == pytest.approx(aggregate["pv_dc_generation_kwh"] - loss, abs=0.02)

    # On the first input, with its 25 A limit, nothing binds.
    first = {**cfg, "pv_strings": [{"inverter": 0, "mppt": 0, "array": 0, "modules": 10}]}
    assert run({**first, "electrical_fidelity": "string"})["pv_dc_generation_kwh"] == aggregate["pv_dc_generation_kwh"]


@pytest.mark.parametrize(
    ("window", "modules_in_series"),
    [({"min_mppt_voltage_v": 150}, 3), ({"max_mppt_voltage_v": 400}, 10)],
    ids=["below_the_range", "above_the_range"],
)
def test_strings_outside_the_mppt_range_lose_energy(modules, _patch_weather, window, modules_in_series):
    # What-if windows: 3 modules run near 120-135 V, 10 modules near 400-450 V.
    inverter = {**FRONIUS_GEN24_6, **window}
    cfg = config(
        [{"modules": modules_in_series, "module": "Trina_TSM_450NEG9R28", "tilt": 35, "azimuth": 180}],
        [{"inverter": 0, "mppt": 0, "array": 0, "modules": modules_in_series}],
        inverter=inverter,
    )
    aggregate, string = run(cfg), run({**cfg, "electrical_fidelity": "string"})
    assert string["pv_dc_generation_kwh"] < aggregate["pv_dc_generation_kwh"]
    assert -stage(string, "mppt_limits")["delta_kwh"] > 0


def _breakdown(module, n_modules, irradiance, temperature, factor):
    times = pd.date_range("2023-06-21 12:00", periods=len(irradiance), freq="h", tz="UTC")
    module_dc, operating = _module_dc_before_losses(
        np.asarray(irradiance, dtype=float),
        np.asarray(temperature, dtype=float),
        module,
        n_modules,
        times,
        name="module_dc_W",
    )
    dc = module_dc * factor
    zero = module_dc * 0
    return PVProductionBreakdown(
        horizontal_reference_dc=zero,
        poa_global_dc=zero,
        front_effective_irradiance_dc=zero,
        rear_gain_dc=zero,
        effective_irradiance_dc=zero,
        module_dc=module_dc,
        dc_after_static_losses=dc,
        dc_after_losses=dc,
        pvwatts_component_losses={},
        pvwatts_components_pct={},
        pvwatts_combined_pct=0.0,
        age_degradation_pct=0.0,
        age_degradation_loss=zero,
        module_operating=operating,
    )


@pytest.mark.parametrize(
    ("v_min", "v_max", "current_limit"),
    [(80, 800, 8.0), (300, 800, 25.0), (80, 250, 25.0), (80, 250, 8.0), (500, 800, 25.0)],
    ids=["current", "low_window", "high_window", "window_and_current", "beyond_voc"],
)
def test_the_operating_point_is_the_best_one_the_limits_allow(v_min, v_max, current_limit):
    """Brute force: the largest V * I(V) on a fine voltage grid that respects both limits."""
    irradiance, temperature, factor, modules = [1000.0, 600.0, 200.0], [45.0, 35.0, 20.0], 0.9, 7
    breakdown = _breakdown(TRINA_450, modules, irradiance, temperature, factor)
    inverter = InverterSpec(
        nominal_power_w=6000,
        min_mppt_voltage_v=v_min,
        max_mppt_voltage_v=v_max,
        startup_voltage_v=0,
        mppts=(MpptSpec(max_input_current_a=current_limit),),
    )
    strings = (PVString(0, 0, 0, modules),)
    environment = ElectricalEnvironment(-5, 40)
    setup = ElectricalSetup(
        (inverter,), strings, environment, check_topology([(TRINA_450, modules)], [inverter], strings, environment)
    )
    dc, loss = string_limited_dc(setup, [breakdown])

    op = breakdown.module_operating
    for step in range(len(irradiance)):
        params = [np.asarray(p)[step] for p in (op.photocurrent, op.saturation_current, op.resistance_series)]
        params += [np.asarray(op.resistance_shunt)[step], np.asarray(op.n_ns_vth)[step]]
        grid = np.linspace(v_min, v_max, 200001)
        current = factor * np.clip(pvlib.pvsystem.i_from_v(grid / modules, *params), 0, None)
        allowed = current <= current_limit + 1e-9
        # No feasible point: the input delivers nothing.
        best = float((grid * current)[allowed].max()) if allowed.any() else 0.0
        unconstrained = factor * modules * op.v_mp[step] * op.i_mp[step]
        assert dc.iloc[step] == pytest.approx(min(best, unconstrained), rel=1e-4, abs=1e-3)
        assert loss.iloc[step] == pytest.approx(unconstrained - dc.iloc[step], rel=1e-6, abs=1e-6)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"inverters": [FRONIUS_GEN24_6, FRONIUS_GEN24_6]}, "one inverter; 2 are configured"),
        ({"pv_strings": [{"inverter": 0, "mppt": 0, "array": 0, "modules": 7}]}, "unconnected_modules"),
        (
            {"inverters": [{**FRONIUS_GEN24_6, "mppts": [{"max_strings": 2}, {"max_strings": 1}]}]},
            "MPPT 0 of inverter 0 has no max_input_current_a",
        ),
        ({"inverters": [{**FRONIUS_GEN24_6, "min_mppt_voltage_v": None}]}, "no min_mppt_voltage_v"),
    ],
)
def test_string_energy_refuses_what_it_cannot_model(modules, change, message):
    with pytest.raises(ValueError, match=message):
        resolve_app_config({**EAST_WEST, **change, "electrical_fidelity": "string"})


def test_string_energy_needs_a_topology_and_a_known_fidelity(modules):
    plain = {"location": "porto", "annual_consumption_kwh": 4000, "n_modules": 10}
    with pytest.raises(ValueError, match="needs 'inverters'"):
        resolve_app_config({**plain, "electrical_fidelity": "string"})
    with pytest.raises(ValueError, match="'aggregate' or 'string'"):
        resolve_app_config({**EAST_WEST, "electrical_fidelity": "iv"})


def test_a_failing_but_wired_design_still_gets_string_energy(modules):
    # Too many strings on the second input is an error, not a wiring gap.
    strings = [{"inverter": 0, "mppt": 1, "array": 0, "modules": 7, "parallel": 2}]
    resolved = resolve_app_config(
        {
            **EAST_WEST,
            "pv_arrays": [{"modules": 14, "module": "Trina_TSM_450NEG9R28"}],
            "pv_strings": strings,
            "electrical_fidelity": "string",
        }
    )
    assert resolved.electrical.check.status == "fail"
    assert resolved.electrical.fidelity == "string"


def test_aggregate_runs_keep_no_operating_point_in_the_result_path(modules):
    assert replace(resolve_app_config(EAST_WEST).electrical).fidelity == "aggregate"


def test_no_feasible_operating_point_delivers_nothing():
    # 7 modules at 1000 W/m2 and 45 °C carry 9.56 A at 250 V; 8 A needs 312 V.
    breakdown = _breakdown(TRINA_450, 7, [1000.0], [45.0], 0.9)
    inverter = InverterSpec(
        nominal_power_w=6000, min_mppt_voltage_v=80, max_mppt_voltage_v=250, startup_voltage_v=0, mppts=(MpptSpec(8.0),)
    )
    strings = (PVString(0, 0, 0, 7),)
    environment = ElectricalEnvironment(-5, 40)
    setup = ElectricalSetup(
        (inverter,), strings, environment, check_topology([(TRINA_450, 7)], [inverter], strings, environment)
    )
    dc, loss = string_limited_dc(setup, [breakdown])
    assert dc.iloc[0] == pytest.approx(0.0, abs=1e-6)
    assert loss.iloc[0] == pytest.approx(breakdown.dc_after_losses.iloc[0])


def test_availability_does_not_lower_the_current_against_the_limit():
    # At STC, 7 Trina modules carry 10.09 A against a 9.9 A limit. A 3 %
    # availability loss is time out of service and must not hide the limit.
    inverter = InverterSpec(
        nominal_power_w=6000, min_mppt_voltage_v=80, max_mppt_voltage_v=800, startup_voltage_v=0, mppts=(MpptSpec(9.9),)
    )
    strings = (PVString(0, 0, 0, 7),)
    environment = ElectricalEnvironment(-5, 40)
    setup = ElectricalSetup(
        (inverter,), strings, environment, check_topology([(TRINA_450, 7)], [inverter], strings, environment)
    )
    full = _breakdown(TRINA_450, 7, [1000.0], [25.0], 1.0)
    in_service = replace(_breakdown(TRINA_450, 7, [1000.0], [25.0], 0.97), pvwatts_components_pct={"availability": 3.0})
    _, loss_full = string_limited_dc(setup, [full])
    _, loss_available = string_limited_dc(setup, [in_service])
    assert loss_full.iloc[0] > 0
    assert loss_available.iloc[0] == pytest.approx(0.97 * loss_full.iloc[0])


def _setup(inverter, strings, arrays):
    environment = ElectricalEnvironment(-5, 40)
    strings = tuple(strings)
    return ElectricalSetup((inverter,), strings, environment, check_topology(arrays, [inverter], strings, environment))


def _inverter(*mppts, **limits):
    values = {"min_mppt_voltage_v": 80, "max_mppt_voltage_v": 800, "startup_voltage_v": 0, **limits}
    return InverterSpec(nominal_power_w=6000, mppts=mppts, **values)


def _module_current(breakdown, step, voltage):
    op = breakdown.module_operating
    params = [np.asarray(p)[step] for p in (op.photocurrent, op.saturation_current, op.resistance_series)]
    params += [np.asarray(op.resistance_shunt)[step], np.asarray(op.n_ns_vth)[step]]
    return pvlib.pvsystem.i_from_v(voltage, *params)


def test_an_input_and_the_inverter_deliver_at_most_their_usable_dc_power():
    # 7 Trina modules at 1000 W/m2 and 45 °C deliver about 2.6 kW after a 10 % loss.
    east = _breakdown(TRINA_450, 7, [1000.0, 300.0], [45.0, 30.0], 0.9)
    west = _breakdown(TRINA_450, 7, [1000.0, 300.0], [45.0, 30.0], 0.9)
    strings = (PVString(0, 0, 0, 7), PVString(0, 1, 1, 7))
    arrays = [(TRINA_450, 7), (TRINA_450, 7)]
    free = string_limited_dc(_setup(_inverter(MpptSpec(25), MpptSpec(25)), strings, arrays), [east, west])[0]
    assert free.iloc[0] > 4000

    per_input = _inverter(MpptSpec(25, max_usable_dc_power_w=2000), MpptSpec(25))
    dc, loss = string_limited_dc(_setup(per_input, strings, arrays), [east, west])
    assert dc.iloc[0] == pytest.approx(2000 + free.iloc[0] / 2)
    assert dc.iloc[1] == free.iloc[1]

    whole = _inverter(MpptSpec(25), MpptSpec(25), max_usable_dc_power_w=3000)
    dc, loss = string_limited_dc(_setup(whole, strings, arrays), [east, west])
    assert dc.iloc[0] == pytest.approx(3000)
    assert loss.iloc[0] == pytest.approx(free.iloc[0] - 3000)
    assert dc.iloc[1] == free.iloc[1]


def test_a_string_connection_limit_bounds_identical_parallel_strings_together():
    breakdown = _breakdown(TRINA_450, 14, [1000.0, 600.0], [45.0, 35.0], 0.9)
    arrays = [(TRINA_450, 14)]
    strings = [PVString(0, 0, 0, 7, parallel=2)]
    by_string = _inverter(MpptSpec(25, max_string_input_current_a=8.0))
    by_input = _inverter(MpptSpec(16.0))
    dc_string = string_limited_dc(_setup(by_string, strings, arrays), [breakdown])[0]
    dc_input = string_limited_dc(_setup(by_input, strings, arrays), [breakdown])[0]
    assert dc_string.iloc[0] < breakdown.dc_after_losses.iloc[0]
    pd.testing.assert_series_equal(dc_string, dc_input)


def test_identical_arrays_in_parallel_match_one_array_with_two_strings():
    one = _breakdown(TRINA_450, 14, [1000.0, 600.0, 200.0], [45.0, 35.0, 20.0], 0.9)
    half = _breakdown(TRINA_450, 7, [1000.0, 600.0, 200.0], [45.0, 35.0, 20.0], 0.9)
    for limit in (25.0, 15.0):
        inverter = _inverter(MpptSpec(limit))
        single = string_limited_dc(_setup(inverter, [PVString(0, 0, 0, 7, parallel=2)], [(TRINA_450, 14)]), [one])
        split = string_limited_dc(
            _setup(inverter, [PVString(0, 0, 0, 7), PVString(0, 0, 1, 7)], [(TRINA_450, 7), (TRINA_450, 7)]),
            [half, half],
        )
        np.testing.assert_allclose(split[0], single[0], rtol=1e-6)


@pytest.mark.parametrize(
    ("v_min", "v_max", "current_limit", "string_limit"),
    [(80, 800, 25.0, math.inf), (80, 800, 15.0, math.inf), (300, 800, 25.0, math.inf), (80, 250, 25.0, 8.0)],
    ids=["free", "input_current", "low_window", "string_current_and_high_window"],
)
def test_parallel_arrays_run_at_the_best_point_of_their_summed_curve(v_min, v_max, current_limit, string_limit):
    """Brute force on east and west strings in parallel: the largest V * (I_east + I_west) the limits allow."""
    irradiance_east, irradiance_west = [900.0, 150.0, 600.0], [200.0, 700.0, 600.0]
    temperature_east, temperature_west = [50.0, 25.0, 40.0], [30.0, 45.0, 40.0]
    east = _breakdown(TRINA_450, 7, irradiance_east, temperature_east, 0.9)
    west = _breakdown(TRINA_450, 7, irradiance_west, temperature_west, 0.85)
    inverter = _inverter(
        MpptSpec(current_limit, max_string_input_current_a=string_limit),
        min_mppt_voltage_v=v_min,
        max_mppt_voltage_v=v_max,
    )
    strings = (PVString(0, 0, 0, 7), PVString(0, 0, 1, 7))
    dc, loss = string_limited_dc(_setup(inverter, strings, [(TRINA_450, 7), (TRINA_450, 7)]), [east, west])

    grid = np.linspace(v_min, v_max, 400001)
    for step in range(3):
        i_east = 0.9 * np.nan_to_num(_module_current(east, step, grid / 7))
        i_west = 0.85 * np.nan_to_num(_module_current(west, step, grid / 7))
        total = i_east + i_west
        allowed = (total <= current_limit + 1e-9) & (np.maximum(i_east, i_west) <= string_limit + 1e-9)
        best = float((grid * np.clip(total, 0, None))[allowed].max()) if allowed.any() else 0.0
        separate = east.dc_after_losses.iloc[step] + west.dc_after_losses.iloc[step]
        assert dc.iloc[step] == pytest.approx(best, rel=1e-5, abs=1e-3)
        assert loss.iloc[step] == pytest.approx(separate - dc.iloc[step], rel=1e-6, abs=1e-6)
    # Unequal irradiance costs some energy even without a limit: the summed
    # curve peaks below the sum of the two peaks.
    if (v_min, v_max, current_limit) == (80, 800, 25.0):
        assert 0 < loss.iloc[0] < 0.02 * dc.iloc[0]
        assert loss.iloc[2] == pytest.approx(0.0, abs=0.05 * 0.01 * dc.iloc[2])


def test_the_inverter_starts_once_an_input_reaches_its_start_up_voltage():
    # A dim morning step, two bright ones, a dim one, night, then dim and bright again.
    irradiance = [15.0, 400.0, 1000.0, 15.0, 0.0, 15.0, 1000.0]
    breakdown = _breakdown(TRINA_450, 3, irradiance, [25.0] * 7, 0.9)
    v_oc = np.asarray(
        [3 * pvlib.pvsystem.v_from_i(0.0, *_params(breakdown, s)) if g else 0.0 for s, g in enumerate(irradiance)]
    )
    startup = 0.5 * (v_oc[0] + v_oc[1])
    assert v_oc[0] < startup < min(v_oc[1], v_oc[2], v_oc[6])
    inverter = _inverter(MpptSpec(25), startup_voltage_v=float(startup))
    dc, loss = string_limited_dc(_setup(inverter, [PVString(0, 0, 0, 3)], [(TRINA_450, 3)]), [breakdown])
    free = breakdown.dc_after_losses
    # Below the start-up voltage before it has started; running once started,
    # down to the dim evening step; stopped overnight and waiting again.
    assert dc.iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert loss.iloc[0] == pytest.approx(free.iloc[0])
    for step in (1, 2, 3, 6):
        assert dc.iloc[step] == free.iloc[step]
    assert dc.iloc[5] == pytest.approx(0.0, abs=1e-9)


def _params(breakdown, step):
    op = breakdown.module_operating
    return [
        np.asarray(p)[step]
        for p in (op.photocurrent, op.saturation_current, op.resistance_series, op.resistance_shunt, op.n_ns_vth)
    ]


def test_string_energy_needs_the_start_up_voltage(modules):
    with pytest.raises(ValueError, match="no startup_voltage_v"):
        resolve_app_config(
            {
                **EAST_WEST,
                "inverters": [{**FRONIUS_GEN24_6, "startup_voltage_v": None}],
                "electrical_fidelity": "string",
            }
        )


def test_app_parallels_two_arrays_on_one_input(modules, _patch_weather):
    cfg = config(
        EAST_WEST["pv_arrays"],
        [
            {"inverter": 0, "mppt": 0, "array": 0, "modules": 7},
            {"inverter": 0, "mppt": 0, "array": 1, "modules": 7},
        ],
    )
    aggregate, string = run(cfg), run({**cfg, "electrical_fidelity": "string"})
    assert "parallel_strings_mismatch" not in {f["code"] for f in string["electrical"]["findings"]}
    mismatch = -stage(string, "mppt_limits")["delta_kwh"]
    # East and west in parallel lose a little to their different curves.
    assert 0 < mismatch < 0.02 * aggregate["pv_dc_generation_kwh"]
