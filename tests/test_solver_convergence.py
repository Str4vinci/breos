"""Search error on the represented curves, independent of cell-level accuracy."""

from dataclasses import replace

import numpy as np
import pytest

from breos.electrical import _curves_power, _CurveSource
from breos.pv.mismatch import ShadedStringCurve
from breos.pv.shading import ModuleShading
from breos.pv_modules import get_module
from breos.solar import IrradianceParts

# A 8k-point global grid plus local polishing resolves these represented
# curves much more closely than 0.01%. This tolerance covers the production
# search's finite bracketing and golden-section precision, not compression.
TOL = 1e-4
SEED = 80621
CASES = 16
DENSE_POINTS = 8193


def represented(seed, diodes, parallel, breakdown, cases=CASES, step=None):
    rng = np.random.default_rng(seed)
    module = replace(
        get_module("Generic_400W"),
        bypass_diodes=diodes,
        cell_strings_in_parallel=parallel,
        breakdown_voltage_v=breakdown,
    )
    azimuth = np.arange(cases) * 360 / cases
    one = np.ones(cases)
    parts = IrradianceParts(
        beam=rng.uniform(300, 1100, cases),
        circumsolar=one * 0,
        sky=rng.choice([0.0, 5.0, 80.0], cases),
        other=one * 0,
        temp_cell=rng.uniform(-10, 65, cases),
        sun_azimuth=azimuth,
        sun_elevation=one * 50,
        pv_params=module,
    )
    positions = module.N_Cells / parallel / diodes

    chain_index = 0

    def chain():
        nonlocal chain_index
        limiting = rng.uniform(0.1, 1.0, cases)
        mean = limiting * rng.uniform(1 / positions, 1.0, cases)
        # Structured steps include weakly separated knees and one dark cell.
        mean[0], limiting[0] = 1 / positions, 1.0
        # Three current levels with about equal current times active area:
        # high-current bypass peaks compete with low-current full-string peaks.
        group = (chain_index // parallel) / (4 * diodes)
        mean[1] = limiting[1] = 0.0 if group < 0.25 else (0.5 if group < 0.5 else 0.75)
        chain_index += 1
        return {"beam_shaded_fraction": [mean.tolist()], "limiting_shaded_fraction": [limiting.tolist()]}

    modules = []
    for _ in range(4):
        substrings = []
        for _ in range(diodes):
            substrings.append({"halves": [chain() for _ in range(parallel)]} if parallel > 1 else chain())
        modules.append({"substrings": substrings})
    scale = rng.uniform(0.85, 1.0, cases)
    if step is not None:
        azimuth = azimuth[step : step + 1]
        scale = scale[step : step + 1]
        parts = replace(
            parts,
            **{
                key: getattr(parts, key)[step : step + 1]
                for key in ("beam", "circumsolar", "sky", "other", "temp_cell", "sun_azimuth", "sun_elevation")
            },
        )
        for entry in modules:
            for substring in entry["substrings"]:
                for half in substring.get("halves", [substring]):
                    for key in ("beam_shaded_fraction", "limiting_shaded_fraction"):
                        half[key] = [[half[key][0][step]]]
    shading = ModuleShading.from_mapping(
        {"azimuth_deg": azimuth.tolist(), "elevation_deg": [50], "modules": modules}
    ).for_module(diodes, parallel)
    return ShadedStringCurve(shading, parts, scale, len(modules), cable_ohm=0.2)


def polish(grid, power):
    """Dense global maximum, then an independent local golden search."""
    values = power(grid)
    rows = np.arange(len(grid))
    best = values.argmax(axis=1)
    maximum = values[rows, best]
    low = grid[rows, np.maximum(best - 1, 0)]
    high = grid[rows, np.minimum(best + 1, grid.shape[1] - 1)]
    ratio = (5**0.5 - 1) / 2
    for _ in range(45):
        left, right = high - ratio * (high - low), low + ratio * (high - low)
        pl, pr = power(left[:, None])[:, 0], power(right[:, None])[:, 0]
        maximum = np.maximum(maximum, np.maximum(pl, pr))
        low, high = np.where(pl >= pr, low, left), np.where(pl >= pr, right, high)
    return maximum.clip(min=0.0)


def string_reference(curve, v_min=0.0, v_max=np.inf, current_limit=np.inf):
    select = np.arange(len(curve.mismatched))
    scale = curve.scale[curve.mismatched]
    top = np.minimum(curve._photocurrents.max(axis=0), current_limit / scale)
    low, high = np.zeros(len(top)), top.copy()
    for _ in range(55):
        mid = (low + high) / 2
        above = curve._terminal_voltage(mid, select, True) > v_max
        low, high = np.where(above, mid, low), np.where(above, high, mid)
    start = high if np.isfinite(v_max) else np.zeros(len(top))
    low, high = start.copy(), top.copy()
    for _ in range(55):
        mid = (low + high) / 2
        above = curve._terminal_voltage(mid, select, True) >= v_min
        low, high = np.where(above, mid, low), np.where(above, high, mid)
    end = low
    grid = start[:, None] + (end - start)[:, None] * np.linspace(0, 1, DENSE_POINTS)
    # Include both sides of each represented breakdown discontinuity.
    edges = curve._short_circuits.T
    grid = np.sort(
        np.concatenate(
            (
                grid,
                np.clip(edges * (1 - 1e-10), start[:, None], end[:, None]),
                np.clip(edges * (1 + 1e-10) + 1e-12, start[:, None], end[:, None]),
            ),
            axis=1,
        ),
        axis=1,
    )

    def power(current):
        voltage = curve._voltage(current, select) - scale[:, None] * curve.cable_ohm * current
        ok = (voltage >= v_min) & (voltage <= v_max) & (current * scale[:, None] <= current_limit)
        return np.where(ok, current * voltage * scale[:, None], -np.inf)

    return polish(grid, power)


def parallel_reference(sources, v_min, v_max, current_limit, string_limit):
    rows = np.ones(len(sources[0].mpp), dtype=bool)
    upper = np.minimum(v_max, np.max([s.v_oc for s in sources], axis=0))
    lower = np.full(len(upper), float(v_min))

    def currents(voltage):
        return np.asarray([s.current(voltage, rows, signed=True) for s in sources])

    def allowed(voltage):
        each = currents(voltage)
        return (each.sum(axis=0) <= current_limit + 1e-9) & (each.max(axis=0) <= string_limit + 1e-9)

    low, high = lower.copy(), upper.copy()
    for _ in range(60):
        mid = (low + high) / 2
        ok = allowed(mid)
        low, high = np.where(ok, low, mid), np.where(ok, mid, high)
    lower = np.maximum(lower, high)
    feasible = (lower <= upper) & allowed(lower)
    grid = lower[:, None] + (upper - lower).clip(min=0.0)[:, None] * np.linspace(0, 1, DENSE_POINTS)

    def power(voltage):
        return voltage * currents(voltage).sum(axis=0)

    return np.where(feasible, polish(grid, power), 0.0)


LAYOUTS = [(3, 1, None), (3, 2, -3.0), (2, 2, -5.5), (6, 1, -15.0)]


@pytest.mark.parametrize("diodes,parallel,breakdown", LAYOUTS)
def test_string_peak_convergence(diodes, parallel, breakdown):
    curve = represented(SEED, diodes, parallel, breakdown)
    rows = np.ones(CASES, dtype=bool)
    for limits in [(0.0, np.inf, np.inf), (65.0, 130.0, 3.5)]:
        production, voltage = curve.constrained_mpp(rows, *limits)
        lit = production > 1e-6
        current = np.divide(production, voltage, out=np.zeros(CASES), where=voltage > 0)
        assert np.all(voltage[lit] >= limits[0] - 1e-7)
        assert np.all(voltage[lit] <= limits[1] + 1e-7)
        assert np.all(current[lit] <= limits[2] + 1e-7)
        exact = curve._terminal_voltage(current / curve.scale, np.arange(CASES), True)
        np.testing.assert_allclose(exact[lit], voltage[lit], rtol=TOL, atol=1e-6)
        reference = string_reference(curve, *limits)
        gap = (reference - production) / np.maximum(reference, 1.0)
        print(f"string {diodes, parallel, breakdown} {limits}: worst gap={gap.max():.9g}; excess={-gap.min():.9g}")
        np.testing.assert_allclose(
            production,
            reference,
            rtol=TOL,
            atol=1e-6,
            err_msg=f"seed={SEED}, layout={diodes, parallel, breakdown}, limits={limits}",
        )


@pytest.mark.parametrize("diodes,parallel,breakdown", LAYOUTS)
def test_parallel_peak_convergence(diodes, parallel, breakdown):
    sources = [_CurveSource(represented(SEED + j, diodes, parallel, breakdown)) for j in range(3)]
    for limits in [(0.0, np.inf, np.inf, np.inf), (65.0, 130.0, 9.0, 4.0)]:
        _, production, voltage = _curves_power(sources, *limits)
        lit = production > 1e-6
        each = np.asarray([source.current(voltage, np.ones(CASES, dtype=bool), signed=True) for source in sources])
        assert np.all(voltage[lit] >= limits[0] - 1e-7)
        assert np.all(voltage[lit] <= limits[1] + 1e-7)
        assert np.all(each[:, lit].sum(axis=0) <= limits[2] + 1e-7)
        assert np.all(each[:, lit] <= limits[3] + 1e-7)
        np.testing.assert_allclose(production[lit], (voltage * each.sum(axis=0))[lit], rtol=TOL, atol=1e-6)
        reference = parallel_reference(sources, *limits)
        gap = (reference - production) / np.maximum(reference, 1.0)
        print(f"parallel {diodes, parallel, breakdown} {limits}: worst gap={gap.max():.9g}; excess={-gap.min():.9g}")
        np.testing.assert_allclose(
            production,
            reference,
            rtol=TOL,
            atol=1e-6,
            err_msg=f"seed={SEED}, layout={diodes, parallel, breakdown}, limits={limits}",
        )


def test_global_peaks_at_the_voltage_edge():
    curves = [represented(SEED + j, 3, 1, -3.0, cases=3) for j in range(2)]
    rows = np.ones(3, dtype=bool)
    production, voltage = curves[0].constrained_mpp(rows, 0.0, 1.0, np.inf)
    reference = string_reference(curves[0], 0.0, 1.0)
    np.testing.assert_allclose(production, reference, rtol=TOL, atol=1e-6)
    np.testing.assert_allclose(voltage, 1.0, atol=1e-6)
    sources = [_CurveSource(curve) for curve in curves]
    _, production, voltage = _curves_power(sources, 0.0, 1.0, np.inf, np.inf)
    reference = parallel_reference(sources, 0.0, 1.0, np.inf, np.inf)
    np.testing.assert_allclose(production, reference, rtol=TOL, atol=1e-6)
    np.testing.assert_allclose(voltage, 1.0, atol=1e-6)


def challenging_string():
    # Highest sampled peak: 466.29 W; a lower-ranked bracket refines to
    # 468.04 W. Isolating this step preserves the represented shading inputs.
    return represented(1001, 3, 2, -3.0, cases=64, step=52)


def challenging_parallel():
    # The broad peak wins the 256-point ranking, but another bracket's
    # narrower peak wins after refinement (501.75 W versus 502.26 W).
    return [_CurveSource(represented(1003 + j, 6, 1, -15.0, cases=64, step=41)) for j in range(3)]


def test_a_lower_ranked_string_peak_wins_after_refinement():
    curve = challenging_string()
    reference = string_reference(curve)
    np.testing.assert_allclose(curve.mpp, reference, rtol=TOL, atol=1e-6)


def test_a_lower_ranked_parallel_peak_wins_after_refinement():
    sources = challenging_parallel()
    reference = parallel_reference(sources, 0.0, np.inf, np.inf, np.inf)
    production = _curves_power(sources, 0.0, np.inf, np.inf, np.inf)[1]
    np.testing.assert_allclose(production, reference, rtol=TOL, atol=1e-6)


def test_convergence_cases_detect_reducing_each_peak_count_to_one(monkeypatch):
    import breos.electrical as electrical
    import breos.pv.mismatch as mismatch

    reference = string_reference(challenging_string())
    with monkeypatch.context() as reduced:
        reduced.setattr(mismatch, "_REFINED_PEAKS", 1)
        production = challenging_string().mpp
    assert np.all(production < reference * (1 - TOL))
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(production, reference, rtol=TOL, atol=1e-6)

    sources = challenging_parallel()
    reference = parallel_reference(sources, 0.0, np.inf, np.inf, np.inf)
    with monkeypatch.context() as reduced:
        reduced.setattr(electrical, "_CURVE_PEAKS", 1)
        production = _curves_power(sources, 0.0, np.inf, np.inf, np.inf)[1]
    assert np.all(production < reference * (1 - TOL))
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(production, reference, rtol=TOL, atol=1e-6)
