"""Pure-scipy CEC 6-parameter coefficient fit.

This module computes the reference-condition parameters of the CEC single-diode
("6-parameter") PV module model from the numbers found on a manufacturer
datasheet. It is a drop-in replacement for
:func:`pvlib.ivtools.sdm.fit_cec_sam` that depends only on ``scipy`` and
``pvlib`` — it removes the transitive ``nrel-pysam`` (SAM SDK) dependency that
``fit_cec_sam`` requires, which is the sole blocker for running BREOS on
Python 3.14 (no ``cp314`` wheel and no sdist are published).

The fitted parameters are consumed downstream by
:func:`pvlib.pvsystem.calcparams_cec` and :func:`pvlib.pvsystem.max_power_point`
exactly as before, so model results are unchanged.

Algorithm
---------
The six reference parameters ``(a, I_L, I_o, R_s, R_sh, Adjust)`` are defined by
six constraints evaluated at standard rating conditions (1000 W/m², 25 °C):
the short-circuit, open-circuit and maximum-power points, a zero power-slope at
the maximum-power point, and the datasheet temperature coefficients of open-
circuit voltage (``beta_voc``) and maximum power (``gamma_pmp``). See

    A. Dobos, "An Improved Coefficient Calculator for the California Energy
    Commission 6 Parameter Photovoltaic Module Model", Journal of Solar Energy
    Engineering 134 (2012), DOI:10.1115/1.4005759.

``Adjust`` only rescales how the short-circuit current — and therefore the
maximum power — tracks temperature away from 25 °C; it has no effect on the STC
I–V curve. This lets the problem be solved as a one-dimensional search on
``Adjust`` (matching the datasheet ``gamma_pmp``) wrapping a five-parameter De
Soto solve in which ``alpha_sc`` and ``beta_voc`` carry the ``Adjust`` scaling.

The constants are SAM's (``ssc/shared/6par_solve.h`` and ``6par_gamma.h``), so
the fit reproduces ``fit_cec_sam``: the open-circuit-voltage coefficient is
imposed 5 °C above the reference, temperatures are translated with SAM's
``1/k = 11600 K/eV`` (also in the modeled ``gamma_pmp``), and the accepted
solution must pass SAM's sanity checks on the parameters and on the I–V curve.

The inner solve works in ``(I_L, ln I_o, R_s, 1/R_sh, a)``. The logarithm puts
``I_o`` (~1e-10 A) on the scale of the other unknowns, and the shunt
conductance lets the solution branch followed in ``Adjust`` pass through the
infinite shunt resistance that many modules reach near ``Adjust = 0`` (the De
Soto limit), where ``R_sh`` diverges and changes sign. Only the matched
solution has to be physical.
"""

import warnings
from typing import Tuple

import numpy as np
from pvlib.pvsystem import i_from_v, max_power_point
from scipy.optimize import brentq, root

# SAM's inverse Boltzmann constant in K/eV (6par_solve.h, 6par_gamma.h). It is
# used in the open-circuit-voltage constraint and in the modeled gamma, so the
# fit reproduces SAM rather than pvlib's CODATA value.
_INV_K = 11600.0

# Silicon band-gap reference value and its temperature dependence (Dobos Eqs. 6
# and 7); these are pvlib's calcparams_cec defaults (EgRef, dEgdT).
_EG_REF = 1.121
_DEGDT = -0.0002677

# Temperature offset used for the open-circuit-voltage temperature-coefficient
# constraint (Dobos Eq. 22). SAM's 6parsolve evaluates it 5 °C above the
# reference temperature (``const Real dT = 5`` in 6par_solve.h).
_DELTA_T_VOC = 5.0

# Maximum-power temperature-coefficient grid (Dobos Eq. 10/25): power is
# evaluated from -10 °C to 50 °C and adjacent normalised slopes are averaged.
# For this uniform grid that average telescopes to the secant slope between the
# endpoints, so only the three points needed for it are evaluated. This window
# reproduces the shipped SAM binary (PySAM 7.1.1) exactly.
_GAMMA_T_LOW = -10.0
_GAMMA_T_HIGH = 50.0

# Accepted ranges of the fitted coefficients: SAM's sanity check
# (``module6par::sanity`` in 6par_solve.h). The photocurrent cap is raised from
# SAM's 20 A so that modern high-current modules (210 mm cells, Isc ~18-20 A)
# are not rejected.
_PARAM_BOUNDS = {
    "a_ref": (0.01, 15.0),
    "I_L_ref": (0.01, 100.0),
    "I_o_ref": (1e-16, 1e-7),
    "R_s": (0.001, 75.0),
    "R_sh_ref": (1.0, 100001.0),
    "Adjust": (-100.0, 100.0),
}

_IRRAD_REF = 1000.0
_TEMP_REF = 25.0
_T_REF_K = _TEMP_REF + 273.15

# SAM's technology-specific initial guess (Dobos Tables 3 and 4, and the
# ``guess()`` of 6par_solve.h): the slope/intercept of ``a`` versus the cell
# count, and the C_s / C_sh factors that scale series and shunt resistance.
# ``polySi`` maps to the multi-Si row.
_EMPIRICAL_GUESS = {
    "monosi": ((0.027, -0.0172), 0.32, 4.92),
    "multisi": ((0.0263, 0.0212), 0.34, 5.36),
    "polysi": ((0.0263, 0.0212), 0.34, 5.36),
    "amorphous": ((0.029, 0.5264), 0.59, 0.922),
    "cdte": ((0.012, 1.3565), 0.46, 1.11),
    "cigs": ((0.018, 0.3277), 0.55, 1.22),
    "cis": ((0.021, 0.0897), 0.61, 1.07),
}

# Lowest open-circuit voltage (V) of a series cell assumed by the initial
# guesses. The CEC cell count of shingled and some thin-film modules (e.g.
# Seraphim SEG, Solaria PowerXT, Honda Soltec: 0.11-0.21 V per cell) includes
# parallel strips, which puts an ``a`` proportional to it too far from the
# solution for the solver to converge. The count enters only the guesses.
_CELL_VOC = 0.6

# Tolerance (%/°C) on the modeled-vs-datasheet gamma for a fit to count as a
# clean gamma match rather than a best-effort fallback.
_GAMMA_TOL = 1e-3

# Largest residual of the five reference equations, relative to Isc, for a
# solve to be accepted; the solver's success flag alone is not trusted.
_RESIDUAL_TOL = 1e-8

# Exponents are capped here so that trial points of the solver overflow to a
# large finite residual instead of inf/nan.
_EXP_MAX = 700.0

# Adjust continuation: initial step, the smallest step after halving near a
# failed or invalid solve, and the search limit (SAM's sanity range).
_ADJUST_STEP = 4.0
_ADJUST_MIN_STEP = 0.25
_ADJUST_LIMIT = 100.0

# SAM's I–V shape checks (``module6par::sanity``): power at Vmp within 1.5 % of
# the datasheet, current at Voc below 1.5 % of Imp, and a slope that is never
# positive between 1.5 % and 98 % of Voc, sampled at 100 points.
_SANITY_PMP_TOL = 0.015
_SANITY_IOC_TOL = 0.015
_SANITY_SLOPE_POINTS = 100


def _exp(x):
    """``exp`` with the exponent capped at :data:`_EXP_MAX`."""
    return np.exp(np.minimum(x, _EXP_MAX))


def _expm1(x):
    """``expm1`` with the exponent capped at :data:`_EXP_MAX`."""
    return np.expm1(np.minimum(x, _EXP_MAX))


def _to_unknowns(i_l, i_o, r_s, r_sh, a):
    """Solver unknowns ``[I_L, ln I_o, R_s, 1/R_sh, a]`` of physical parameters."""
    return np.array([i_l, np.log(i_o), r_s, 1.0 / r_sh, a])


def _from_unknowns(x):
    """Physical ``(I_L, I_o, R_s, R_sh, a)`` of solver unknowns (``R_sh`` is inf for a zero conductance)."""
    i_l, ln_i_o, r_s, g_sh, a = (float(v) for v in x)
    r_sh = 1.0 / g_sh if g_sh != 0.0 else np.inf
    return (i_l, float(_exp(ln_i_o)), r_s, r_sh, a)


def _cells_for_guess(v_oc, cells_in_series):
    """Series cells of the initial guesses: at most :data:`_CELL_VOC` per volt of ``Voc``."""
    return min(cells_in_series, v_oc / _CELL_VOC)


def _initial_guess(v_mp, i_mp, v_oc, i_sc, cells_in_series):
    """De Soto five-parameter initial guess (Duffie & Beckman, as in pvlib), as solver unknowns."""
    a0 = 1.5 * _T_REF_K / _INV_K * _cells_for_guess(v_oc, cells_in_series)
    i_o0 = i_sc * np.exp(-v_oc / a0)
    r_s0 = (a0 * np.log1p((i_sc - i_mp) / i_o0) - v_mp) / i_mp
    return _to_unknowns(i_sc, i_o0, r_s0, 100.0, a0)


def _empirical_guess(celltype, v_mp, i_mp, v_oc, i_sc, cells_in_series):
    """SAM's technology-aware guess (Dobos §4.3, Tables 3 and 4; 6par_solve.h ``guess()``), as solver unknowns."""
    (slope, intercept), c_s, c_sh = _EMPIRICAL_GUESS.get(str(celltype).lower(), _EMPIRICAL_GUESS["monosi"])
    a0 = min(max(slope * _cells_for_guess(v_oc, cells_in_series) + intercept, 0.1), 10.0)
    i_o0 = min(max(i_sc * np.exp(-v_oc / a0), 1e-15), 1e-9)
    r_s0 = min(max(c_s * (v_oc - v_mp) / i_mp, 0.02), 60.0)
    r_sh0 = c_sh * v_oc / (i_sc - i_mp)
    return _to_unknowns(i_sc, i_o0, r_s0, r_sh0, a0)


def _translate(x, adjust, alpha_sc, dt):
    """``(I_L, I_o, a)`` ``dt`` kelvin above the reference at full irradiance (Dobos Eqs. 5-9, SAM's 1/k)."""
    i_l, ln_i_o, _r_s, _g_sh, a = x
    t_k = _T_REF_K + dt
    eg = _EG_REF * (1.0 + _DEGDT * dt)
    i_o = _exp(ln_i_o + 3.0 * np.log(t_k / _T_REF_K) + _INV_K * (_EG_REF / _T_REF_K - eg / t_k))
    return i_l + alpha_sc * (1.0 - adjust / 100.0) * dt, i_o, a * t_k / _T_REF_K


def _reference_equations(x, v_mp, i_mp, v_oc, i_sc, alpha_sc, beta_voc, adjust):
    """Residuals of the five reference-condition constraints (Dobos 2012).

    ``x`` holds the solver unknowns ``[I_L, ln I_o, R_s, 1/R_sh, a]`` at the
    reference conditions. ``alpha_sc`` and ``beta_voc`` are the unscaled
    datasheet coefficients; ``Adjust`` scales them as in Eqs. 8 and 9. The
    constraints are the short-circuit point (Eq. 11), the open-circuit point
    (Eq. 12), the maximum-power point (Eq. 13), a zero power-slope there
    (Eq. 19), and the open-circuit-voltage temperature coefficient evaluated
    ``_DELTA_T_VOC`` above the reference (Eq. 22).
    """
    i_l, ln_i_o, r_s, g_sh, a = x
    i_o = _exp(ln_i_o)

    y = np.empty(5)
    # Short-circuit point: V = 0, I = Isc (Eq. 11).
    y[0] = i_sc - i_l + i_o * _expm1(i_sc * r_s / a) + i_sc * r_s * g_sh
    # Open-circuit point: V = Voc, I = 0 (Eq. 12).
    y[1] = -i_l + i_o * _expm1(v_oc / a) + v_oc * g_sh
    # Maximum-power point: V = Vmp, I = Imp (Eq. 13).
    y[2] = i_mp - i_l + i_o * _expm1((v_mp + i_mp * r_s) / a) + (v_mp + i_mp * r_s) * g_sh
    # Zero power-slope at the maximum-power point (Eq. 19).
    exp_mp = _exp((v_mp + i_mp * r_s) / a)
    y[3] = i_mp - v_mp * ((i_o / a) * exp_mp + g_sh) / (1.0 + (i_o * r_s / a) * exp_mp + r_s * g_sh)
    # Open-circuit-voltage temperature coefficient at T' = Tref + dT (Eq. 22).
    dt = _DELTA_T_VOC
    i_l2, i_o2, a2 = _translate(x, adjust, alpha_sc, dt)
    v_oc2 = v_oc + beta_voc * (1.0 + adjust / 100.0) * dt
    y[4] = -i_l2 + i_o2 * _expm1(v_oc2 / a2) + v_oc2 * g_sh
    return y


def _solve_five_params(adjust, v_mp, i_mp, v_oc, i_sc, alpha_sc, beta_voc, x0):
    """Solve the five reference unknowns for a fixed ``Adjust``.

    Returns ``(x, ok)`` with ``x = [I_L, ln I_o, R_s, 1/R_sh, a]``. ``ok``
    requires the solver to report success, finite unknowns and residuals of the
    five equations within :data:`_RESIDUAL_TOL` of ``Isc``; the result may still
    be unphysical (a negative shunt conductance on the way to the solution).
    """
    args = (v_mp, i_mp, v_oc, i_sc, alpha_sc, beta_voc, adjust)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sol = root(_reference_equations, x0, args=args, method="lm")
        ok = bool(sol.success) and bool(np.all(np.isfinite(sol.x)))
        if ok:
            ok = bool(np.max(np.abs(_reference_equations(sol.x, *args))) <= _RESIDUAL_TOL * i_sc)
    return sol.x, ok


def _modeled_gamma(x, adjust, alpha_sc):
    """Modeled maximum-power temperature coefficient in %/°C (Dobos Eq. 10).

    The paper averages adjacent normalised power slopes over -10 → 50 °C; for a
    uniform grid that equals the secant slope between the endpoints, normalised
    by the modeled power at the reference temperature. The temperature
    translation is done here with SAM's constants rather than through
    ``calcparams_cec``, whose Boltzmann constant differs. Returns NaN if the
    maximum-power point cannot be solved.
    """
    dt = np.array([_GAMMA_T_LOW, _TEMP_REF, _GAMMA_T_HIGH]) - _TEMP_REF
    i_l, i_o, a = _translate(x, adjust, alpha_sc, dt)
    r_sh = 1.0 / x[3] if x[3] != 0.0 else np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            p_low, p_ref, p_high = max_power_point(i_l, i_o, x[2], r_sh, a, method="newton")["p_mp"]
        except (RuntimeError, ValueError):
            return float("nan")
    return float(100.0 * (p_high - p_low) / (p_ref * (_GAMMA_T_HIGH - _GAMMA_T_LOW)))


def _is_physical(five, adjust):
    """True if the six parameters fall within SAM's sanity ranges (:data:`_PARAM_BOUNDS`)."""
    i_l_ref, i_o_ref, r_s, r_sh_ref, a_ref = five
    values = {
        "a_ref": a_ref,
        "I_L_ref": i_l_ref,
        "I_o_ref": i_o_ref,
        "R_s": r_s,
        "R_sh_ref": r_sh_ref,
        "Adjust": adjust,
    }
    return all(np.isfinite(v) and lo <= v <= hi for (lo, hi), v in ((_PARAM_BOUNDS[k], values[k]) for k in values))


def _passes_iv_checks(five, v_mp, i_mp, v_oc):
    """SAM's I–V shape checks on the reference curve (``module6par::sanity`` in 6par_solve.h)."""
    i_l, i_o, r_s, r_sh, a = five
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if abs(v_mp * i_from_v(v_mp, i_l, i_o, r_s, r_sh, a) / (v_mp * i_mp) - 1.0) > _SANITY_PMP_TOL:
            return False
        if abs(i_from_v(v_oc, i_l, i_o, r_s, r_sh, a)) > _SANITY_IOC_TOL * i_mp:
            return False
        v = np.linspace(0.015 * v_oc, 0.98 * v_oc, _SANITY_SLOPE_POINTS + 1)
        return bool(np.all(np.diff(i_from_v(v, i_l, i_o, r_s, r_sh, a)) <= 0.0))


def _solve_adjust(v_mp, i_mp, v_oc, i_sc, alpha_sc, beta_voc, gamma_pmp, x0):
    """Find ``Adjust`` so the modeled gamma matches the datasheet ``gamma_pmp``.

    ``gamma`` decreases monotonically with ``Adjust`` along a fixed solution
    branch (a larger ``Adjust`` suppresses the photocurrent's rise with
    temperature, so power falls faster), so the sign of the mismatch at
    ``Adjust = 0`` gives the direction of the search. ``Adjust`` is stepped in
    that direction, each five-parameter solve warm-started from the previous
    one (continuation) so the inner solver stays on one branch. A step whose
    solve fails or whose gamma cannot be evaluated is halved, down to
    :data:`_ADJUST_MIN_STEP`, before the search gives up; it also ends at
    :data:`_ADJUST_LIMIT`. A bracketed match is refined with ``brentq``.

    Returns ``(adjust, x)`` for the matched solution, the closest sample if the
    branch never reaches the target gamma, or ``None`` if ``Adjust = 0`` itself
    cannot be solved.
    """
    args = (v_mp, i_mp, v_oc, i_sc, alpha_sc, beta_voc)
    x, ok = _solve_five_params(0.0, *args, np.asarray(x0, dtype=float))
    residual = _modeled_gamma(x, 0.0, alpha_sc) - gamma_pmp if ok else float("nan")
    if not np.isfinite(residual):
        return None
    direction = 1.0 if residual > 0.0 else -1.0
    adjust, step = 0.0, _ADJUST_STEP
    while abs(residual) > _GAMMA_TOL * 1e-3:
        if abs(adjust) >= _ADJUST_LIMIT:
            break
        trial = direction * min(abs(adjust) + step, _ADJUST_LIMIT)
        x_trial, ok = _solve_five_params(trial, *args, x)
        trial_residual = _modeled_gamma(x_trial, trial, alpha_sc) - gamma_pmp if ok else float("nan")
        if not np.isfinite(trial_residual):
            step /= 2.0
            if step < _ADJUST_MIN_STEP:
                break
            continue
        if residual * trial_residual <= 0.0:
            # Refine within the bracket, warm-starting each solve from the last.
            branch = x.copy()

            def bracket_residual(value):
                solved, solved_ok = _solve_five_params(value, *args, branch)
                if not solved_ok:
                    raise ValueError("inner solve failed inside the Adjust bracket")
                branch[:] = solved
                return _modeled_gamma(solved, value, alpha_sc) - gamma_pmp

            try:
                adjust = brentq(bracket_residual, adjust, trial, xtol=1e-10, rtol=1e-12)
            except (ValueError, RuntimeError):
                return adjust, x
            x, _ok = _solve_five_params(adjust, *args, branch)
            return adjust, x
        adjust, x, residual = trial, x_trial, trial_residual
        step = _ADJUST_STEP
    return adjust, x


def fit_cec_params(
    celltype: str,
    Vmp: float,
    Imp: float,
    Voc: float,
    Isc: float,
    alpha_sc: float,
    beta_voc: float,
    gamma_pmp: float,
    cells_in_series: int,
) -> Tuple[float, float, float, float, float, float]:
    """Fit the CEC single-diode reference parameters from datasheet values.

    Replacement for :func:`pvlib.ivtools.sdm.fit_cec_sam` at its default 25 °C
    reference temperature that needs no ``nrel-pysam``. Implements the
    coefficient calculator of Dobos 2012 (DOI:10.1115/1.4005759) with SAM's
    constants, using ``scipy`` and ``pvlib`` only.

    Args:
        celltype: Cell technology label. It selects the technology-specific
            empirical initial guess (Dobos §4.3), which the solver tries after
            the De Soto guess. An unknown label uses the mono-Si row.
        Vmp: Voltage at the maximum-power point [V].
        Imp: Current at the maximum-power point [A].
        Voc: Open-circuit voltage [V].
        Isc: Short-circuit current [A].
        alpha_sc: Short-circuit-current temperature coefficient [A/°C].
        beta_voc: Open-circuit-voltage temperature coefficient [V/°C].
        gamma_pmp: Maximum-power temperature coefficient [%/°C].
        cells_in_series: Number of cells in series.

    Returns:
        ``(I_L_ref, I_o_ref, R_s, R_sh_ref, a_ref, Adjust)`` — the same tuple,
        in the same order, that ``fit_cec_sam`` returns.

    Raises:
        RuntimeError: if no physical parameter set can be found.
    """
    # Try the reported short-circuit current first and, if no branch matches the
    # datasheet gamma, raise Isc by 1% per iteration (SAM's heuristic in
    # 6par_solve.h, up to five iterations) to move a poorly conditioned module
    # onto a well-shaped I-V curve. The first guess (De Soto) reproduces SAM on
    # well-formed datasheets; the empirical guess (Dobos §4.3) is a robustness
    # fallback. A candidate must pass SAM's parameter ranges and I–V checks; the
    # best-but-imperfect one is retained so a usable result is returned even if
    # no branch lands exactly on the target gamma.
    fallback = None
    for bump in range(6):
        i_sc = Isc * (1.01**bump)
        guesses = (
            _initial_guess(Vmp, Imp, Voc, i_sc, cells_in_series),
            _empirical_guess(celltype, Vmp, Imp, Voc, i_sc, cells_in_series),
        )
        for x0 in guesses:
            result = _solve_adjust(Vmp, Imp, Voc, i_sc, alpha_sc, beta_voc, gamma_pmp, x0)
            if result is None:
                continue
            adjust, x = result
            five = _from_unknowns(x)
            if not _is_physical(five, adjust) or not _passes_iv_checks(five, Vmp, Imp, Voc):
                continue
            residual = abs(_modeled_gamma(x, adjust, alpha_sc) - gamma_pmp)
            if residual <= _GAMMA_TOL:
                i_l_ref, i_o_ref, r_s, r_sh_ref, a_ref = five
                return (i_l_ref, i_o_ref, r_s, r_sh_ref, a_ref, adjust)
            if fallback is None or residual < fallback[1]:
                fallback = ((five, adjust), residual)

    if fallback is not None:
        (five, adjust), residual = fallback
        warnings.warn(
            "CEC parameter fit returned the closest physical solution without "
            f"meeting the gamma tolerance (absolute residual {residual:.6g} %/°C).",
            RuntimeWarning,
            stacklevel=2,
        )
        i_l_ref, i_o_ref, r_s, r_sh_ref, a_ref = five
        return (i_l_ref, i_o_ref, r_s, r_sh_ref, a_ref, adjust)

    raise RuntimeError("CEC parameter fit did not converge to a physical solution.")
