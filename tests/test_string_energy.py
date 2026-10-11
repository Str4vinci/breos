"""String-level energy applies each MPPT input's voltage range and current limit."""

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
    "mppts": [
        {"max_input_current_a": 25, "max_short_circuit_current_a": 40, "max_strings": 2},
        {"max_input_current_a": 12.5, "max_short_circuit_current_a": 20, "max_strings": 1},
    ],
}

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
        best = float((grid * current)[allowed].max()) if allowed.any() else float(v_max * current_limit)
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
