"""Module-level shading: each module's substrings, their bypass diodes and the string's global maximum."""

import json
from dataclasses import replace

import numpy as np
import pvlib
import pytest

from breos.app_config import resolve_app_config
from breos.electrical import PVString, _curves_power, _CurveSource
from breos.pv.mismatch import ShadedStringCurve
from breos.pv.shading import ModuleShading
from breos.pv_modules import MODULES, get_module
from breos.solar import IrradianceParts

STEPS = 3


def _table(value: float) -> list[list[float]]:
    return [[value]]


def _shading(modules: list[dict], diodes: int = 3, parallel: int = 2, **extra) -> ModuleShading:
    # The bundled modules are half-cut: two cell strings per substring.
    return ModuleShading.from_mapping(
        {"azimuth_deg": [0], "elevation_deg": [0], "modules": modules, **extra}
    ).for_module(diodes, parallel)


def _parts(
    beam: float = 800.0, sky: float = 100.0, other: float = 20.0, module: str = "Generic_400W"
) -> IrradianceParts:
    full = np.full(STEPS, 1.0)
    return IrradianceParts(
        beam=beam * full,
        circumsolar=0.0 * full,
        sky=sky * full,
        other=other * full,
        temp_cell=40.0 * full,
        sun_azimuth=180.0 * full,
        sun_elevation=50.0 * full,
        pv_params=get_module(module),
    )


def _curve(modules: list[dict], **kwargs) -> ShadedStringCurve:
    return ShadedStringCurve(_shading(modules), _parts(), np.ones(STEPS), len(modules), **kwargs)


LIT = {"beam_shaded_fraction": _table(0.0)}


def test_the_shape_is_validated():
    with pytest.raises(ValueError, match="either beam_shaded_fraction or substrings"):
        _shading([{"beam_shaded_fraction": _table(0), "substrings": []}])
    with pytest.raises(ValueError, match="holds 2 tables; the module has 3 bypass diodes"):
        _shading([{"substrings": [{"beam_shaded_fraction": _table(0)}] * 2}])
    with pytest.raises(ValueError, match="unknown key"):
        _shading([{**LIT, "limiting_shaded_fraction": _table(1)}])
    with pytest.raises(ValueError, match="from 0 to 1"):
        _shading([{"beam_shaded_fraction": _table(1.5)}])
    with pytest.raises(ValueError, match="unknown key"):
        _shading([LIT], bypass_diodes=2)
    with pytest.raises(ValueError, match="single string"):
        PVString(0, 0, 0, 1, parallel=2, module_shading=_shading([LIT]))
    with pytest.raises(ValueError, match="2 module entries for 3 modules"):
        PVString(0, 0, 0, 3, module_shading=_shading([LIT, LIT]))
    two = _shading([LIT], diodes=2)
    assert two.bypass_diodes == 2 and len(two.modules[0].mean) == 2
    with pytest.raises(ValueError, match="not bound to a module"):
        _ = ModuleShading.from_mapping({"azimuth_deg": [0], "elevation_deg": [0], "modules": [LIT]}).diodes


def test_even_shade_on_every_module_is_the_modules_own_curve():
    even = _curve([{"beam_shaded_fraction": _table(0.3)}] * 8)
    assert even.mismatched.size == 0 and even.uniform.size == STEPS
    np.testing.assert_allclose(even.mpp, even.own_mean_power(), rtol=1e-12)
    np.testing.assert_allclose(even.mpp, even.mpp_bare)


def test_one_shaded_cell_bypasses_its_substring():
    lit = _curve([LIT] * 8)
    # One of the substring's 48 cells fully shaded: the area mean is 1/48.
    cell = {
        "substrings": [
            {"beam_shaded_fraction": _table(1 / 48), "limiting_shaded_fraction": _table(1.0)},
            {"beam_shaded_fraction": _table(0.0)},
            {"beam_shaded_fraction": _table(0.0)},
        ]
    }
    shaded = _curve([cell] + [LIT] * 7)
    assert shaded.mismatched.size == STEPS
    # The bypass diode takes the substring out: about 1 of the string's 24 substrings.
    ratio = shaded.mpp / lit.mpp
    np.testing.assert_allclose(ratio, 23 / 24, atol=0.01)
    # Fed only its 1/48 area mean, the substring would lose about 1/48 of itself.
    mean_only = _curve([{"substrings": [{"beam_shaded_fraction": _table(1 / 48)}] + [LIT] * 2}] + [LIT] * 7)
    assert (mean_only.mpp > shaded.mpp).all()
    assert (mean_only.mpp / lit.mpp > 0.995).all()


def test_a_shaded_module_is_bypassed_at_the_global_maximum():
    lit = _curve([LIT] * 8)
    half = _curve([{"beam_shaded_fraction": _table(0.5)}] + [LIT] * 7)
    # Bypassing the module (7/8) beats running the string at its current (about 0.57).
    np.testing.assert_allclose(half.mpp / lit.mpp, 7 / 8, atol=0.01)
    # The curve's current at its maximum power voltage is its maximum power current.
    every = np.ones(STEPS, dtype=bool)
    np.testing.assert_allclose(half.current(half.v_mp, every), half.i_mp, rtol=5e-3)
    # A window below the bypassed peak's voltage leaves the lower peak.
    _, limited, voltage = _curves_power([_CurveSource(half)], 0.0, half.v_mp[0] * 0.8, 50.0, 50.0)
    assert (limited < half.mpp).all() and (limited > 0.5 * lit.mpp).all()
    # The operating voltage lies in the window and carries the delivered power,
    # to the accuracy of the curve's tabled current.
    assert (voltage <= half.v_mp[0] * 0.8 + 1e-9).all()
    np.testing.assert_allclose(voltage * half.current(voltage, every), limited, rtol=2e-3)
    unconstrained, free, at = _curves_power([_CurveSource(half)], 0.0, 1000.0, 50.0, 50.0)
    np.testing.assert_array_equal(free, unconstrained)
    np.testing.assert_array_equal(at, half.v_mp)


def test_a_higher_bypass_diode_voltage_costs_more():
    module = [{"beam_shaded_fraction": _table(0.5)}] + [LIT] * 7
    low = _curve(module, bypass_diode_voltage_v=0.3)
    high = _curve(module, bypass_diode_voltage_v=0.8)
    assert (high.mpp < low.mpp).all()


def _one_cell(limiting: float) -> dict:
    """A module whose first substring has one of its 48 cells at ``limiting``."""
    return {
        "substrings": [
            {"beam_shaded_fraction": _table(limiting / 48), "limiting_shaded_fraction": _table(limiting)},
            LIT,
            LIT,
        ]
    }


def test_a_half_shaded_cell_does_not_carry_the_string_current():
    # A module with a low forward-fit shunt (195 ohm, 1.4 ohm a cell), which
    # would pass amperes through the shaded cell in reverse bias.
    parts = _parts(module="Erlangen_445W")

    def curve(modules: list[dict]) -> ShadedStringCurve:
        return ShadedStringCurve(_shading(modules), parts, np.ones(STEPS), len(modules))

    lit = curve([LIT] * 10)
    half = curve([_one_cell(0.5)] + [LIT] * 9)
    full = curve([_one_cell(1.0)] + [LIT] * 9)
    # The string runs at the lit modules' current, above what the half-shaded
    # cell can carry: its substring is bypassed as if the cell were dark.
    np.testing.assert_allclose(half.mpp, full.mpp, rtol=1e-6)
    np.testing.assert_allclose(half.mpp / lit.mpp, 29 / 30, atol=0.005)


def test_cells_shaded_alike_cap_the_string_at_their_short_circuit_current():
    curve = _curve([_one_cell(0.5)] * 8)
    # Every substring is current-limited alike: the string runs at most at the
    # shaded cell's short-circuit current, the most it carries without reverse
    # bias. In a half-cut module's 24 cell positions per substring, the shaded
    # cell's own voltage puts the maximum just below it.
    shaded = curve._short_circuits.min(axis=0)
    assert (curve.i_mp <= shaded * (1 + 1e-9)).all()
    np.testing.assert_allclose(curve.i_mp, shaded, rtol=0.01)
    assert (curve.i_mp < 0.75 * _curve([LIT] * 8).i_mp).all()


def _cap_jump(curve: ShadedStringCurve) -> tuple[float, float, float]:
    """The first step's lowest part cap and the string's voltage just below and above it."""
    cap = float(curve._short_circuits[:, 0].min())
    below, above = curve._terminal_voltage(np.array([cap * (1 - 1e-9), cap * (1 + 1e-9)]), np.zeros(2, dtype=int), True)
    return cap, float(below), float(above)


def test_a_window_inside_the_voltage_jump_at_a_cap_has_no_operating_point():
    curve = _curve([_one_cell(0.5)] * 8)
    _, below, above = _cap_jump(curve)
    assert below - above > 50.0
    first = np.array([True, False, False])
    # No point of the curve lies between the jump's two voltages.
    gap = (above + 0.25 * (below - above), above + 0.75 * (below - above))
    power, voltage = curve.constrained_mpp(first, gap[0], gap[1], 50.0)
    assert power[0] == 0.0 and voltage[0] == 0.0
    # A window that ends inside the jump keeps the bypassed peak below it.
    power, voltage = curve.constrained_mpp(first, 0.0, gap[1], 50.0)
    assert voltage[0] <= gap[1]
    assert power[0] == pytest.approx(_brute_force(curve, 0.0, gap[1]), rel=1e-5)


def test_the_tabled_current_holds_the_cap_across_its_jump():
    curve = _curve([_one_cell(0.5)] * 8)
    cap, below, above = _cap_jump(curve)
    first = np.array([True, False, False])
    middle = np.array([0.5 * (below + above)])
    for signed in (False, True):
        assert curve.current(middle, first, signed=signed)[0] == pytest.approx(cap, rel=1e-6)


def test_a_substring_matches_pvlibs_partial_shading_example():
    """pvlib's example cell, 23 lit and one shaded, with reverse breakdown, against no leakage."""
    from scipy.constants import e, k

    thermal = k * (273.15 + 25.0) / e
    breakdown = {"breakdown_factor": 2e-3, "breakdown_exp": 3, "breakdown_voltage": -15.0}

    def cell(irradiance: float) -> tuple[float, ...]:
        params = pvlib.pvsystem.calcparams_desoto(
            irradiance, 25.0, 0.0042, 1.3 * thermal, 8.24, 2.36e-9, 1000.0, 0.00181, EgRef=1.121, dEgdT=-0.0002677
        )
        return tuple(float(p) for p in params)

    current = np.linspace(0.0, 8.2, 4001)

    def reference(irradiance: float, cells: int) -> np.ndarray:
        # Each cell's whole curve, forward and reverse, combined in series at
        # one current, then the bypass diode, as pvlib's example builds it.
        diode = np.linspace(0.99 * breakdown["breakdown_voltage"], 0.75, 20000)
        i, v, _ = pvlib.singlediode.bishop88(diode, *cell(irradiance), **breakdown)
        return cells * np.interp(current, i[::-1], v[::-1], right=-np.inf)

    def part(irradiance: float, cells: int) -> tuple[np.ndarray, ...]:
        photo, saturation, series, shunt, diode = cell(irradiance)
        params = (photo, saturation, series * cells, shunt * cells, diode * cells)
        short_circuit = float(pvlib.pvsystem.i_from_v(0.0, *params))
        return tuple(np.array([p]) for p in params + (short_circuit, -np.inf))

    curve = _curve([LIT])
    rows = np.zeros(1, dtype=int)
    shaded = curve._substring_voltage(current[None, :], [part(200.0, 1), part(1000.0, 23)], rows)[0]
    lit = curve._substring_voltage(current[None, :], [part(1000.0, 24)], rows)[0]
    shaded_reference = np.maximum(reference(200.0, 1) + reference(1000.0, 23), -0.5)
    lit_reference = np.maximum(reference(1000.0, 24), -0.5)
    # Forward bias: the same curve.
    forward = current < part(200.0, 1)[5][0]
    np.testing.assert_allclose(shaded[forward], shaded_reference[forward], atol=5e-3)
    # The 23 lit cells hold the shaded one past -15 V, so it breaks down and
    # its substring keeps a volt or two where this model bypasses it. A string
    # of 10 substrings still finds the same maximum power to 0.2 %.
    for count in (1, 3, 10):
        power = np.max(current * (count * shaded + (10 - count) * lit))
        expected = np.max(current * (count * shaded_reference + (10 - count) * lit_reference))
        assert power == pytest.approx(expected, rel=2e-3)


# -- App -----------------------------------------------------------------------------------------

UNSHADED_GRID = {"azimuth_deg": [0, 90, 180, 270], "elevation_deg": [0, 30, 40]}
CHIMNEY = [[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
CLEAR = [[0.0] * 4] * 3


def _module_shaded_config(trina: str, first_module: list[list[float]]) -> dict:
    def string(mppt: int, tables: list[list[list[float]]]) -> dict:
        return {
            "inverter": 0,
            "mppt": mppt,
            "array": 0,
            "modules": 7,
            "module_shading": {**UNSHADED_GRID, "modules": [{"beam_shaded_fraction": t} for t in tables]},
        }

    return {
        "location": "porto",
        "annual_consumption_kwh": 4000,
        "pv_arrays": [{"modules": 14, "module": trina, "tilt": 30, "azimuth": 180}],
        "inverters": [
            {
                "nominal_power_w": 6000,
                "max_dc_voltage_v": 1000,
                "min_mppt_voltage_v": 80,
                "max_mppt_voltage_v": 800,
                "startup_voltage_v": 80,
                "mppts": [
                    {"max_input_current_a": 16, "max_short_circuit_current_a": 20, "max_strings": 1},
                    {"max_input_current_a": 16, "max_short_circuit_current_a": 20, "max_strings": 1},
                ],
            }
        ],
        "pv_strings": [string(0, [first_module] + [CLEAR] * 6), string(1, [CLEAR] * 7)],
        "electrical_environment": {"min_design_temp_c": -5, "max_ambient_temp_c": 40},
        "electrical_fidelity": "string",
    }


def test_app_reports_the_mismatch_stage_and_each_string(trina, _patch_weather):
    from breos import App

    config = _module_shaded_config(trina, CHIMNEY)
    resolved = resolve_app_config(config)
    assert resolved.cfg["pv_loss_overrides"]["shading"] == 0.0
    app = App(config)
    app.simulate()
    result = app.result()
    waterfall = result["pv_loss_waterfall"]
    keys = [stage["key"] for stage in waterfall["stages"]]
    assert keys.index("near_shading") < keys.index("iam")
    assert keys.index("pvwatts_static") + 1 == keys.index("shading_mismatch")
    mismatch = waterfall["shading_mismatch"]
    assert mismatch["loss_kwh"] > 0 and mismatch["bypass_diode_voltage_stated"] is False
    by_string = {row["string"]: row for row in mismatch["strings"]}
    # The chimney's string loses to mismatch; the clear one does not.
    assert by_string[0]["loss_kwh"] > 0.1 and abs(by_string[1]["loss_kwh"]) < 0.01
    assert by_string[0]["bypass_diodes"] == 3
    assert waterfall["near_shading"]["arrays"][0]["form"] == "modules"
    echoed = result["provenance"]["resolved_config"]["pv_strings"][0]["module_shading"]
    assert echoed["modules"][0]["beam_shaded_fraction"] == CHIMNEY
    json.dumps(result)


def test_all_clear_modules_match_the_unshaded_string_run(trina, _patch_weather):
    from breos import App

    shaded = _module_shaded_config(trina, CLEAR)
    plain = json.loads(json.dumps(shaded))
    for string in plain["pv_strings"]:
        del string["module_shading"]
    plain["pv_loss_overrides"] = {"shading": 0.0}
    results = {}
    for name, config in (("shaded", shaded), ("plain", plain)):
        app = App(config)
        app.simulate()
        results[name] = app.result()["pv_loss_waterfall"]["energy_balance"]["pv_dc"]["generation_kwh"]
    assert results["shaded"] == pytest.approx(results["plain"], rel=1e-6)


def test_differently_shaded_strings_on_one_input_report_the_parallel_mismatch(trina, _patch_weather):
    from breos import App

    split = _module_shaded_config(trina, CHIMNEY)
    together = json.loads(json.dumps(split))
    together["pv_strings"][1]["mppt"] = 0
    # Room for both strings on one input, so no limit binds and the gap is the shared voltage alone.
    together["inverters"][0]["mppts"][0].update(max_input_current_a=32, max_short_circuit_current_a=40, max_strings=2)
    results = {}
    for name, config in (("split", split), ("together", together)):
        app = App(config)
        app.simulate()
        results[name] = app.result()["pv_loss_waterfall"]
    assert results["split"]["parallel_mismatch"] is None
    parallel = results["together"]["parallel_mismatch"]
    assert parallel["inputs"] == [{"inverter": 0, "mppt": 0, "strings": [0, 1], "loss_kwh": parallel["loss_kwh"]}]
    stages = {stage["key"]: stage.get("delta_kwh") for stage in results["together"]["stages"]}
    assert -stages["parallel_mismatch"] == pytest.approx(parallel["loss_kwh"], abs=1e-3)
    assert stages["mppt_limits"] == pytest.approx(0.0, abs=1e-3)
    # Each string's own shading mismatch does not depend on its input.
    assert results["together"]["shading_mismatch"] == results["split"]["shading_mismatch"]
    # What the second input saves is the parallel mismatch, to the results' 0.01 kWh rounding.
    generation = {name: r["energy_balance"]["pv_dc"]["generation_kwh"] for name, r in results.items()}
    assert 0 < parallel["loss_kwh"] == pytest.approx(generation["split"] - generation["together"], abs=0.02)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"electrical_fidelity": "aggregate"}, "needs electrical_fidelity = 'string'"),
        ({"pv_arrays": [{"modules": 14, "tilt": 30, "azimuth": 180, "shading": {**UNSHADED_GRID, "beam_shaded_fraction": CLEAR}}]}, "one or the other"),
    ],
)  # fmt: skip
def test_module_shading_is_refused_where_it_cannot_run(trina, change, message):
    config = {**_module_shaded_config(trina, CHIMNEY), **change}
    if "pv_arrays" in change:
        config["pv_arrays"][0]["module"] = trina
    with pytest.raises(ValueError, match=message):
        resolve_app_config(config)


def test_every_string_of_an_array_or_none(trina):
    config = _module_shaded_config(trina, CHIMNEY)
    del config["pv_strings"][1]["module_shading"]
    with pytest.raises(ValueError, match=r"'pv_strings\[1\]' has no module_shading"):
        resolve_app_config(config)


def test_monte_carlo_years_take_the_module_shading(trina, synthetic_weather):
    from breos.montecarlo import _build_pv_years

    shaded = _module_shaded_config(trina, CHIMNEY)
    clear = _module_shaded_config(trina, CLEAR)
    totals = {}
    for name, config in (("shaded", shaded), ("clear", clear)):
        resolved = resolve_app_config(config)
        dc, _, _ = _build_pv_years(resolved.cfg, resolved, {2023: synthetic_weather})
        totals[name] = dc[2023].sum()
    assert totals["shaded"] < totals["clear"]


# -- review regressions ---------------------------------------------------------------------------


def test_a_negative_perez_sky_term_is_clipped_per_module():
    # Perez's horizon term can make the sky rest negative; with the beam 90 %
    # shaded, the module keeps its clipped sky diffuse, as the array does.
    full = np.ones(STEPS)
    parts = IrradianceParts(
        beam=10.0 * full, circumsolar=50.0 * full, sky=-40.0 * full, other=0.0 * full,
        temp_cell=20.0 * full, sun_azimuth=96.0 * full, sun_elevation=5.0 * full,
        pv_params=get_module("Generic_400W"),
    )  # fmt: skip
    np.testing.assert_allclose(parts.irradiance(np.full(STEPS, 0.9), 1.0), 1.0)
    shading = _shading([{"beam_shaded_fraction": _table(0.9)}] * 4)
    curve = ShadedStringCurve(shading, parts, np.ones(STEPS), 4)
    assert (curve.mpp > 0).all()
    np.testing.assert_allclose(curve.mpp, curve.own_mean_power(), rtol=1e-12)


def test_parallel_strings_open_circuit_where_their_signed_currents_cancel():
    from breos.electrical import _sources_open_circuit

    bright = _curve([LIT] * 8)
    dim = _curve([{"beam_shaded_fraction": _table(0.9)}] * 8)
    voltage = _sources_open_circuit([_CurveSource(bright), _CurveSource(dim)])
    # The root of the two strings' signed currents, found on a fine grid.
    grid = np.linspace(dim.v_oc[0], bright.v_oc[0], 20001)
    every = np.ones(STEPS, dtype=bool)
    total = [
        bright.current(np.full(STEPS, v), every, signed=True)[0] + dim.current(np.full(STEPS, v), every, signed=True)[0]
        for v in grid
    ]
    root = grid[int(np.argmin(np.abs(total)))]
    assert dim.v_oc[0] < root < bright.v_oc[0] - 1.0
    np.testing.assert_allclose(voltage, root, atol=0.05)


def _brute_force(curve: ShadedStringCurve, v_min: float, v_max: float) -> float:
    """The best power on the curve's first step: a fine grid of exact voltages, and the window's edges."""
    select = np.zeros(1, dtype=int)

    def voltage(current: float) -> float:
        return float(curve._terminal_voltage(np.array([current]), select, True)[0])

    currents = np.linspace(0.0, curve._table_i[0, -1], 40001)
    voltages = curve._terminal_voltage(currents, np.zeros(currents.size, dtype=int), True)
    inside = (voltages >= v_min) & (voltages <= v_max)
    best = float((curve.scale[0] * currents * voltages)[inside].max(initial=0.0))
    # Where the curve is steep, the window's edge falls between grid points:
    # solve it, keeping only a side inside the window, since the voltage can
    # also jump across the edge at a part's short-circuit current.
    for edge in (v_min, v_max):
        crossings = np.flatnonzero((voltages[:-1] > edge) & (voltages[1:] <= edge))
        for k in crossings:
            low, high = currents[k], currents[k + 1]
            for _ in range(80):
                middle = 0.5 * (low + high)
                low, high = (middle, high) if voltage(middle) > edge else (low, middle)
            for current in (low, high):
                if v_min <= voltage(current) <= v_max:
                    best = max(best, curve.scale[0] * current * voltage(current))
    return best


@pytest.mark.parametrize(
    ("v_min", "v_max"),
    [pytest.param(80.0, 260.0, id="binding"), pytest.param(80.0, 200.0, id="between-table-points")],
)
def test_a_binding_window_is_solved_on_the_exact_curve(v_min, v_max):
    curve = _curve([{"beam_shaded_fraction": _table(0.99)}] * 7 + [LIT])
    _, limited, _ = _curves_power([_CurveSource(curve)], v_min, v_max, 50.0, 50.0)
    assert limited[0] > 0
    assert limited[0] == pytest.approx(_brute_force(curve, v_min, v_max), rel=1e-4)


class _TwoPeaks:
    """A synthetic string: 10 A to 300 V, then 4.28 A to 697.5 V."""

    count = 1

    def __init__(self) -> None:
        self.mpp = np.full(1, 3000.0)
        self.v_oc = np.full(1, 700.0)

    def current(self, voltage, rows, signed=False):
        v = np.asarray(voltage, dtype=float)
        return np.where(v <= 300.0, 10.0, np.where(v <= 697.5, 4.28, np.clip((700.0 - v) * 1.712, 0.0, None)))


def test_inconsistent_mean_and_limiting_fractions_are_refused(trina):
    config = _module_shaded_config(trina, CLEAR)
    module = config["pv_strings"][0]["module_shading"]["modules"]
    module[0] = {
        "substrings": [
            {"beam_shaded_fraction": CLEAR, "limiting_shaded_fraction": CHIMNEY},
            {"beam_shaded_fraction": CLEAR},
            {"beam_shaded_fraction": CLEAR},
        ]
    }
    with pytest.raises(ValueError, match="below limiting_shaded_fraction / 48"):
        resolve_app_config(config)
    module[0]["substrings"][0]["beam_shaded_fraction"] = [[v / 48 for v in row] for row in CHIMNEY]
    resolve_app_config(config)
    MODULES[trina] = replace(MODULES[trina], bypass_diodes=144)
    module[0]["substrings"] += [{"beam_shaded_fraction": CLEAR}] * 141
    for other in config["pv_strings"][0]["module_shading"]["modules"][1:]:
        other["beam_shaded_fraction"] = CLEAR
    with pytest.raises(ValueError, match="fewer than 2 cells per string"):
        resolve_app_config(config)


def test_reusing_the_brightest_substring_matches_solving_each():
    full = np.ones(STEPS)
    parts = IrradianceParts(
        beam=800.0 * full, circumsolar=0.0 * full, sky=100.0 * full, other=20.0 * full,
        temp_cell=40.0 * full, sun_azimuth=np.array([90.0, 135.0, 180.0]), sun_elevation=20.0 * full,
        pv_params=get_module("Generic_400W"),
    )  # fmt: skip
    grid = {"azimuth_deg": [90, 180], "elevation_deg": [0]}
    shading = ModuleShading.from_mapping(
        {
            **grid,
            "modules": [
                {"beam_shaded_fraction": [[0.6, 0.0]]},
                {"substrings": [
                    {"beam_shaded_fraction": [[0.0, 0.02]], "limiting_shaded_fraction": [[0.0, 0.5]]},
                    {"beam_shaded_fraction": [[0.0, 0.0]]},
                    {"beam_shaded_fraction": [[0.3, 0.3]]},
                ]},
                {"beam_shaded_fraction": [[0.0, 0.0]]},
            ],
        }
    ).for_module(3, 2)  # fmt: skip
    curve = ShadedStringCurve(shading, parts, np.ones(STEPS), 3)
    assert curve.mismatched.size == STEPS
    reused = curve._voltage(curve._table_i)
    curve._substrings = [s._replace(own=~s.split) for s in curve._substrings]
    np.testing.assert_allclose(curve._voltage(curve._table_i), reused, rtol=0, atol=1e-12)
    # A row subset reads the same rows.
    select = np.array([2, 0])
    np.testing.assert_allclose(curve._voltage(curve._table_i[select], select), reused[select], atol=1e-12)


def test_the_dc_balances_with_cables_limits_and_module_shading(trina, _patch_weather):
    from breos import App

    config = _module_shaded_config(trina, CHIMNEY)
    for string in config["pv_strings"]:
        string.update(cable_length_m=40, cable_cross_section_mm2=4)
    for mppt in config["inverters"][0]["mppts"]:
        mppt["max_input_current_a"] = 4
    app = App(config)
    app.simulate()
    waterfall = app.result()["pv_loss_waterfall"]
    stages = {stage["key"]: stage for stage in waterfall["stages"]}
    assert stages["mppt_limits"]["delta_kwh"] < 0 and stages["dc_wiring"]["delta_kwh"] < 0
    assert stages["shading_mismatch"]["delta_kwh"] < 0
    last = waterfall["stages"][-1]["energy_kwh"]
    assert last == pytest.approx(waterfall["energy_balance"]["pv_dc"]["generation_kwh"], abs=0.05)


# -- second review regressions --------------------------------------------------------------------


def _exact_current(curve: ShadedStringCurve, voltage: float) -> float:
    """A mismatched string's signed current at ``voltage`` on its first step, by bisection on its exact curve."""
    select = np.zeros(1, dtype=int)
    low, high = -20.0, float(curve._table_i[0, -1])
    for _ in range(100):
        middle = 0.5 * (low + high)
        if curve._terminal_voltage(np.array([middle]), select, True)[0] > voltage:
            low = middle
        else:
            high = middle
    return curve.scale[0] * 0.5 * (low + high)


def test_a_mismatched_string_draws_current_above_its_open_circuit():
    from breos.electrical import _sources_open_circuit

    clear = _curve([LIT] * 8)
    dim = _curve([{"beam_shaded_fraction": _table(0.9)}] * 7 + [{"beam_shaded_fraction": _table(0.8)}])
    assert dim.mismatched.size == STEPS
    every = np.ones(STEPS, dtype=bool)
    above = dim.v_oc[0] + 1.0
    assert dim.current(np.full(STEPS, above), every, signed=True)[0] == pytest.approx(
        _exact_current(dim, above), abs=0.01
    )
    assert dim.current(np.full(STEPS, above), every, signed=True)[0] < 0

    def total(v: float) -> float:
        return clear.current(np.full(STEPS, v), every, signed=True)[0] + _exact_current(dim, v)

    low, high = dim.v_oc[0], clear.v_oc[0]
    for _ in range(60):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if total(middle) > 0 else (low, middle)
    voltage = _sources_open_circuit([_CurveSource(clear), _CurveSource(dim)])
    assert voltage[0] == pytest.approx(0.5 * (low + high), abs=0.1)
    assert voltage[0] < clear.v_oc[0] - 5.0


def test_more_bypass_diodes_than_cells_is_refused(trina):
    with pytest.raises(ValueError, match="must divide"):
        replace(MODULES[trina], bypass_diodes=145)
    raw = ModuleShading.from_mapping(
        {
            "azimuth_deg": [0],
            "elevation_deg": [0],
            "modules": [{"substrings": [{"beam_shaded_fraction": [[0.0]]}] * 145}],
        }
    )
    problem = raw.for_module(145).inconsistency(144)
    assert problem is not None and "bypass_diodes (145) leave fewer than one of its 144 cells per string" in problem


def test_the_substrings_follow_the_modules_bypass_diodes(trina):
    MODULES[trina] = replace(MODULES[trina], bypass_diodes=2)
    config = _module_shaded_config(trina, CLEAR)
    module = config["pv_strings"][0]["module_shading"]["modules"]
    module[0] = {"substrings": [{"beam_shaded_fraction": CLEAR}] * 3}
    with pytest.raises(ValueError, match="holds 3 tables; the module has 2 bypass diodes"):
        resolve_app_config(config)
    module[0] = {"substrings": [{"beam_shaded_fraction": CLEAR}] * 2}
    resolved = resolve_app_config(config)
    assert resolved.electrical.strings[0].module_shading.bypass_diodes == 2
    assert len(resolved.electrical.strings[1].module_shading.modules[0].mean) == 2


class _FourPlateaus:
    """A synthetic string with four close peaks, the first the highest."""

    count = 1
    step = 700 / 255

    def __init__(self) -> None:
        self.mpp = np.full(1, 3000.0)
        self.v_oc = np.full(1, 700.0)

    def current(self, voltage, rows, signed=False):
        v, h = np.asarray(voltage, dtype=float), self.step
        return np.select(
            [v <= 300.0, v <= 140 * h, v <= 180 * h, v <= 240 * h],
            [10.0, 2996 / (140 * h), 2995 / (180 * h), 2994 / (240 * h)],
            0.0,
        )


# -- third review regressions ---------------------------------------------------------------------


def test_unsigned_current_never_goes_negative_above_open_circuit():
    dim = _curve([{"beam_shaded_fraction": _table(0.9)}] * 7 + [{"beam_shaded_fraction": _table(0.8)}])
    every = np.ones(STEPS, dtype=bool)
    assert (dim.current(np.full(STEPS, dim.v_oc[0] + 1.0), every) == 0).all()
    clear = _curve([LIT] * 8)
    window = (dim.v_oc[0] + 1.0, clear.v_oc[0])
    _, limited, _ = _curves_power([_CurveSource(clear), _CurveSource(dim)], *window, 50.0, 50.0)
    _, alone, _ = _curves_power([_CurveSource(clear)], *window, 50.0, 50.0)
    # Above its open circuit, the dim string carries nothing: the clear one delivers alone.
    assert limited[0] == pytest.approx(alone[0], rel=1e-3)


@pytest.mark.parametrize("fraction", [0.5, 0.57])
def test_near_repeated_grid_currents_keep_their_peak(fraction):
    shaded = {"beam_shaded_fraction": _table(fraction)}
    curve = _curve([LIT, shaded, shaded, LIT])
    assert curve.mpp[0] == pytest.approx(_brute_force(curve, -np.inf, np.inf), rel=1e-5)


class _NarrowFirstPeak:
    count = 1

    def __init__(self) -> None:
        self.mpp = np.full(1, 300.0)
        self.v_oc = np.full(1, 700.0)

    def current(self, voltage, rows, signed=False):
        v = np.asarray(voltage, dtype=float)
        return np.where(v <= 30.0, 10.0, np.where(v <= 697.5, 0.428, np.clip(0.1712 * (700.0 - v), 0.0, None)))


@pytest.mark.parametrize(
    ("source", "power", "rel"),
    [
        pytest.param(_TwoPeaks, 6000.0, 1e-3, id="knee-and-high-voltage-peak"),
        pytest.param(_FourPlateaus, 6000.0, 1e-4, id="several-close-peaks"),
        pytest.param(_NarrowFirstPeak, 600.0, 1e-4, id="narrow-global-peak"),
    ],
)
def test_the_summed_curve_keeps_its_global_peak(source, power, rel):
    _, limited, voltage = _curves_power([source(), source()], 0.0, 800.0, 100.0, 100.0)
    assert limited[0] == pytest.approx(power, rel=rel)
    if source is _TwoPeaks:
        # The global peak lies at the knee, 300 V, not the higher-voltage one near 697.5 V.
        assert voltage[0] == pytest.approx(300.0, rel=1e-3)


def test_string_energy_binds_unbound_shading_to_the_module():
    raw = ModuleShading.from_mapping({"azimuth_deg": [0], "elevation_deg": [0], "modules": [LIT, LIT]})
    bound = ShadedStringCurve(raw.for_module(3, 2), _parts(), np.ones(STEPS), 2)
    unbound = ShadedStringCurve(raw, _parts(), np.ones(STEPS), 2)
    np.testing.assert_array_equal(unbound.mpp, bound.mpp)
    with pytest.raises(ValueError, match="bound to 2 bypass diodes of 2 cell strings; the module has 3 of 2"):
        ShadedStringCurve(raw.for_module(2, 2), _parts(), np.ones(STEPS), 2)
    with pytest.raises(ValueError, match="bound to 3 bypass diodes of 1 cell strings; the module has 3 of 2"):
        ShadedStringCurve(raw.for_module(3), _parts(), np.ones(STEPS), 2)


def test_a_numpy_diode_count_is_stored_as_an_int():
    module = replace(get_module("Generic_400W"), bypass_diodes=np.int64(2))
    assert type(module.bypass_diodes) is int
    shading = ModuleShading.from_mapping({"azimuth_deg": [0], "elevation_deg": [0], "modules": [LIT]})
    assert type(shading.for_module(np.int64(2)).bypass_diodes) is int
    json.dumps(shading.for_module(np.int64(2)).as_dict())
