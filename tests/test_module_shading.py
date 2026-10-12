"""Module-level shading: each module's substrings, their bypass diodes and the string's global maximum."""

import json

import numpy as np
import pytest

from breos.app_config import resolve_app_config
from breos.electrical import PVString, _curves_power, _CurveSource
from breos.pv.mismatch import ShadedStringCurve
from breos.pv.shading import ModuleShading
from breos.pv_modules import MODULES, PVModuleParams, get_module
from breos.solar import IrradianceParts

STEPS = 3


def _table(value: float) -> list[list[float]]:
    return [[value]]


def _shading(modules: list[dict], **extra) -> ModuleShading:
    return ModuleShading.from_mapping({"azimuth_deg": [0], "elevation_deg": [0], "modules": modules, **extra})


def _parts(beam: float = 800.0, sky: float = 100.0, other: float = 20.0) -> IrradianceParts:
    full = np.full(STEPS, 1.0)
    return IrradianceParts(
        beam=beam * full,
        circumsolar=0.0 * full,
        sky=sky * full,
        other=other * full,
        temp_cell=40.0 * full,
        sun_azimuth=180.0 * full,
        sun_elevation=50.0 * full,
        pv_params=get_module("Generic_400W"),
    )


def _curve(modules: list[dict], **kwargs) -> ShadedStringCurve:
    return ShadedStringCurve(_shading(modules), _parts(), np.ones(STEPS), len(modules), **kwargs)


LIT = {"beam_shaded_fraction": _table(0.0)}


def test_the_shape_is_validated():
    with pytest.raises(ValueError, match="either beam_shaded_fraction or substrings"):
        _shading([{"beam_shaded_fraction": _table(0), "substrings": []}])
    with pytest.raises(ValueError, match=r"one table per bypass diode \(3\)"):
        _shading([{"substrings": [{"beam_shaded_fraction": _table(0)}] * 2}])
    with pytest.raises(ValueError, match="unknown key"):
        _shading([{**LIT, "limiting_shaded_fraction": _table(1)}])
    with pytest.raises(ValueError, match="from 0 to 1"):
        _shading([{"beam_shaded_fraction": _table(1.5)}])
    with pytest.raises(ValueError, match="bypass_diodes"):
        _shading([LIT], bypass_diodes=0)
    with pytest.raises(ValueError, match="single string"):
        PVString(0, 0, 0, 1, parallel=2, module_shading=_shading([LIT]))
    with pytest.raises(ValueError, match="2 module entries for 3 modules"):
        PVString(0, 0, 0, 3, module_shading=_shading([LIT, LIT]))
    two = _shading([LIT], bypass_diodes=2)
    assert two.bypass_diodes_stated and len(two.modules[0].mean) == 2


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
    _, limited = _curves_power([_CurveSource(half)], 0.0, half.v_mp[0] * 0.8, 50.0, 50.0)
    assert (limited < half.mpp).all() and (limited > 0.5 * lit.mpp).all()
    unconstrained, free = _curves_power([_CurveSource(half)], 0.0, 1000.0, 50.0, 50.0)
    np.testing.assert_array_equal(free, unconstrained)


def test_a_higher_bypass_diode_voltage_costs_more():
    module = [{"beam_shaded_fraction": _table(0.5)}] + [LIT] * 7
    low = _curve(module, bypass_diode_voltage_v=0.3)
    high = _curve(module, bypass_diode_voltage_v=0.8)
    assert (high.mpp < low.mpp).all()


# -- App -----------------------------------------------------------------------------------------

UNSHADED_GRID = {"azimuth_deg": [0, 90, 180, 270], "elevation_deg": [0, 30, 40]}
CHIMNEY = [[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
CLEAR = [[0.0] * 4] * 3


@pytest.fixture
def trina():
    # Trina Solar TSM-450NEG9R.28, datasheet TSM_EN_2024_C, PDF page 2.
    MODULES["Trina_TSM_450NEG9R28"] = PVModuleParams(
        Mpp=450, Vmp=44.6, Imp=10.09, Voc=52.9, Isc=10.74,
        T_Pmax_pct=-0.29, T_Voc_pct=-0.24, T_Isc_pct=0.04, N_Cells=144, Max_System_Voltage=1500,
    )  # fmt: skip
    yield "Trina_TSM_450NEG9R28"
    del MODULES["Trina_TSM_450NEG9R28"]


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
    assert by_string[0]["bypass_diodes"] == 3 and by_string[0]["bypass_diodes_stated"] is False
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
    best = float((curve.scale[0] * currents * voltages)[inside].max())
    # Where the curve is steep, the window's edge falls between grid points: solve it.
    for edge in (v_min, v_max):
        crossings = np.flatnonzero((voltages[:-1] > edge) & (voltages[1:] <= edge))
        for k in crossings:
            low, high = currents[k], currents[k + 1]
            for _ in range(80):
                middle = 0.5 * (low + high)
                low, high = (middle, high) if voltage(middle) > edge else (low, middle)
            best = max(best, curve.scale[0] * low * voltage(low))
    return best


def test_a_binding_window_is_solved_on_the_exact_curve():
    curve = _curve([{"beam_shaded_fraction": _table(0.99)}] * 7 + [LIT])
    _, limited = _curves_power([_CurveSource(curve)], 80.0, 260.0, 50.0, 50.0)
    assert limited[0] == pytest.approx(_brute_force(curve, 80.0, 260.0), rel=1e-4)


class _TwoPeaks:
    """A synthetic string: 10 A to 300 V, then 4.28 A to 697.5 V."""

    count = 1

    def __init__(self) -> None:
        self.mpp = np.full(1, 3000.0)
        self.v_oc = np.full(1, 700.0)

    def current(self, voltage, rows, signed=False):
        v = np.asarray(voltage, dtype=float)
        return np.where(v <= 300.0, 10.0, np.where(v <= 697.5, 4.28, np.clip((700.0 - v) * 1.712, 0.0, None)))


def test_the_summed_curve_keeps_its_global_peak():
    _, limited = _curves_power([_TwoPeaks(), _TwoPeaks()], 0.0, 800.0, 100.0, 100.0)
    assert limited[0] == pytest.approx(6000.0, rel=1e-3)


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
    config["pv_strings"][0]["module_shading"]["bypass_diodes"] = 100
    module[0]["substrings"] += [{"beam_shaded_fraction": CLEAR}] * 97
    with pytest.raises(ValueError, match="fewer than 2 cells per substring"):
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
    )  # fmt: skip
    curve = ShadedStringCurve(shading, parts, np.ones(STEPS), 3)
    assert curve.mismatched.size == STEPS
    reused = curve._voltage(curve._table_i)
    curve._substrings = [(count, p, np.ones_like(own)) for count, p, own in curve._substrings]
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


def test_a_window_between_two_table_points_is_still_solved():
    curve = _curve([{"beam_shaded_fraction": _table(0.99)}] * 7 + [LIT])
    _, limited = _curves_power([_CurveSource(curve)], 80.0, 200.0, 50.0, 50.0)
    assert limited[0] > 0
    assert limited[0] == pytest.approx(_brute_force(curve, 80.0, 200.0), rel=1e-4)


def test_repeated_grid_currents_do_not_hide_the_peak():
    curve = _curve([LIT, {"beam_shaded_fraction": _table(0.5)}, {"beam_shaded_fraction": _table(0.5)}, LIT])
    assert curve.mpp[0] == pytest.approx(_brute_force(curve, -np.inf, np.inf), rel=1e-5)


def test_more_bypass_diodes_than_cells_is_refused(trina):
    config = _module_shaded_config(trina, CLEAR)
    shading = config["pv_strings"][0]["module_shading"]
    shading["bypass_diodes"] = 145
    shading["modules"][0] = {"substrings": [{"beam_shaded_fraction": CLEAR}] * 145}
    with pytest.raises(ValueError, match="more than the module's 144 cells"):
        resolve_app_config(config)


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


def test_several_close_peaks_keep_the_global_one():
    _, limited = _curves_power([_FourPlateaus(), _FourPlateaus()], 0.0, 800.0, 100.0, 100.0)
    assert limited[0] == pytest.approx(6000.0, rel=1e-4)


# -- third review regressions ---------------------------------------------------------------------


def test_unsigned_current_never_goes_negative_above_open_circuit():
    dim = _curve([{"beam_shaded_fraction": _table(0.9)}] * 7 + [{"beam_shaded_fraction": _table(0.8)}])
    every = np.ones(STEPS, dtype=bool)
    assert (dim.current(np.full(STEPS, dim.v_oc[0] + 1.0), every) == 0).all()
    clear = _curve([LIT] * 8)
    window = (dim.v_oc[0] + 1.0, clear.v_oc[0])
    _, limited = _curves_power([_CurveSource(clear), _CurveSource(dim)], *window, 50.0, 50.0)
    _, alone = _curves_power([_CurveSource(clear)], *window, 50.0, 50.0)
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


def test_a_narrow_global_peak_is_not_filtered_out():
    _, limited = _curves_power([_NarrowFirstPeak(), _NarrowFirstPeak()], 0.0, 800.0, 100.0, 100.0)
    assert limited[0] == pytest.approx(600.0, rel=1e-4)
