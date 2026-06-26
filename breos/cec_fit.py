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
The modeled ``gamma_pmp`` is evaluated against the same ``calcparams_cec`` /
``max_power_point`` path used at runtime, so the fit is self-consistent with the
simulator.
"""

import warnings
from typing import Tuple

import numpy as np
from pvlib.pvsystem import calcparams_cec, max_power_point
from scipy import constants
from scipy.optimize import brentq, root

# Boltzmann constant in eV/K, taken from the same source pvlib uses, so the
# I_o(T) scaling in the open-circuit-voltage constraint matches calcparams_cec.
_BOLTZMANN_EV_K = constants.value("Boltzmann constant in eV/K")

# Silicon band-gap reference value and its temperature dependence (Dobos Eqs. 6
# and 7); these are pvlib's calcparams_cec defaults (EgRef, dEgdT).
_EG_REF = 1.121
_DEGDT = -0.0002677

# Temperature offset used for the open-circuit-voltage temperature-coefficient
# constraint (Dobos Eq. 22). SAM's 6parsolve and pvlib's own ``fit_desoto``
# both evaluate this constraint 2 °C above the reference temperature.
_DELTA_T_VOC = 2.0

# Maximum-power temperature-coefficient grid (Dobos Eq. 10/25): power is
# evaluated from -10 °C to 50 °C and adjacent normalised slopes are averaged.
# For this uniform grid that average telescopes to the secant slope between the
# endpoints, so only the three points needed for it are evaluated.
_GAMMA_T_LOW = -10.0
_GAMMA_T_HIGH = 50.0

# Sanity ranges for the fitted coefficients, used to reject a converged-but-
# unphysical solution before falling back to heuristics. The lower bounds and
# positivity follow Dobos Table 5 and are what discriminate against spurious
# roots (e.g. negative shunt resistance). The upper bounds are widened well
# beyond the 2012-era Table 5 limits so that modern high-current / high-power
# modules (210 mm cells with Isc ~18-20 A exceed the original 15 A I_L cap) are
# not rejected.
_PARAM_BOUNDS = {
    "a_ref": (0.05, 50.0),
    "I_L_ref": (0.1, 100.0),
    "I_o_ref": (1e-20, 1e-5),
    "R_s": (1e-4, 100.0),
    "R_sh_ref": (1.0, 1e7),
    "Adjust": (-100.0, 100.0),
}

_IRRAD_REF = 1000.0
_TEMP_REF = 25.0


# Per-technology empirical-guess coefficients (Dobos Tables 3 and 4): the
# slope/intercept of ``a`` versus the cell count, and the C_s / C_sh factors
# that scale series and shunt resistance. ``polySi`` maps to the multi-Si row.
_EMPIRICAL_GUESS = {
    "monosi": ((0.027, -0.0172), 0.32, 4.92),
    "multisi": ((0.026, 0.0212), 0.34, 5.36),
    "polysi": ((0.026, 0.0212), 0.34, 5.36),
    "amorphous": ((0.029, 0.5264), 0.59, 0.92),
    "cdte": ((0.012, 1.356), 0.46, 1.11),
    "cigs": ((0.018, 0.327), 0.55, 1.22),
    "cis": ((0.021, 0.0897), 0.61, 1.07),
}

# Tolerance (%/°C) on the modeled-vs-datasheet gamma for a fit to count as a
# clean gamma match rather than a best-effort fallback.
_GAMMA_TOL = 1e-3


def _initial_guess(v_mp, i_mp, v_oc, i_sc, cells_in_series, temp_ref):
    """De Soto five-parameter initial guess (Duffie & Beckman, as in pvlib)."""
    t_ref_k = temp_ref + 273.15
    a0 = 1.5 * _BOLTZMANN_EV_K * t_ref_k * cells_in_series
    i_l0 = i_sc
    i_o0 = i_sc * np.exp(-v_oc / a0)
    r_s0 = (a0 * np.log1p((i_l0 - i_mp) / i_o0) - v_mp) / i_mp
    r_sh0 = 100.0
    return np.array([i_l0, i_o0, r_s0, r_sh0, a0])


def _empirical_guess(celltype, v_mp, i_mp, v_oc, i_sc, cells_in_series):
    """Technology-aware five-parameter guess (Dobos §4.3, Tables 3 and 4)."""
    (slope, intercept), c_s, c_sh = _EMPIRICAL_GUESS.get(str(celltype).lower(), _EMPIRICAL_GUESS["monosi"])
    a0 = slope * cells_in_series + intercept
    i_l0 = i_sc
    i_o0 = i_sc * np.exp(-v_oc / a0)
    r_s0 = c_s * (v_oc - v_mp) / i_mp
    r_sh0 = c_sh * v_oc / (i_sc - i_mp)
    return np.array([i_l0, i_o0, r_s0, r_sh0, a0])


def _reference_equations(params, v_mp, i_mp, v_oc, i_sc, alpha_eff, beta_eff, temp_ref):
    """Residuals of the five reference-condition constraints (Dobos 2012).

    ``params`` is ``[I_L, I_o, R_s, R_sh, a]`` at the reference conditions.
    ``alpha_eff`` and ``beta_eff`` are the short-circuit-current and
    open-circuit-voltage temperature coefficients already scaled by ``Adjust``
    (Eqs. 8 and 9). The constraints are the short-circuit point (Eq. 11), the
    open-circuit point (Eq. 12), the maximum-power point (Eq. 13), a zero
    power-slope there (Eq. 19), and the open-circuit-voltage temperature
    coefficient evaluated ``_DELTA_T_VOC`` above the reference (Eq. 22).
    """
    i_l, i_o, r_s, r_sh, a = params
    t_ref_k = temp_ref + 273.15

    y = np.empty(5)
    # Short-circuit point: V = 0, I = Isc (Eq. 11).
    y[0] = i_sc - i_l + i_o * np.expm1(i_sc * r_s / a) + i_sc * r_s / r_sh
    # Open-circuit point: V = Voc, I = 0 (Eq. 12).
    y[1] = -i_l + i_o * np.expm1(v_oc / a) + v_oc / r_sh
    # Maximum-power point: V = Vmp, I = Imp (Eq. 13).
    y[2] = i_mp - i_l + i_o * np.expm1((v_mp + i_mp * r_s) / a) + (v_mp + i_mp * r_s) / r_sh
    # Zero power-slope at the maximum-power point (Eq. 19).
    exp_mp = np.exp((v_mp + i_mp * r_s) / a)
    y[3] = i_mp - v_mp * ((i_o / a) * exp_mp + 1.0 / r_sh) / (1.0 + (i_o * r_s / a) * exp_mp + r_s / r_sh)
    # Open-circuit-voltage temperature coefficient at T' = Tref + dT (Eq. 22).
    dt = _DELTA_T_VOC
    t2_k = t_ref_k + dt
    a2 = a * t2_k / t_ref_k
    v_oc2 = v_oc + beta_eff * dt
    i_l2 = i_l + alpha_eff * dt
    eg2 = _EG_REF * (1.0 + _DEGDT * dt)
    i_o2 = i_o * (t2_k / t_ref_k) ** 3 * np.exp((1.0 / _BOLTZMANN_EV_K) * (_EG_REF / t_ref_k - eg2 / t2_k))
    y[4] = -i_l2 + i_o2 * np.expm1(v_oc2 / a2) + v_oc2 / r_sh
    return y


def _solve_five_params(adjust, v_mp, i_mp, v_oc, i_sc, alpha_sc, beta_voc, x0, temp_ref):
    """Solve the five reference parameters for a fixed ``Adjust``.

    ``alpha_sc`` and ``beta_voc`` are the *unscaled* datasheet coefficients;
    the ``Adjust`` scaling (Eqs. 8 and 9) is applied here.
    """
    alpha_eff = alpha_sc * (1.0 - adjust / 100.0)
    beta_eff = beta_voc * (1.0 + adjust / 100.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sol = root(
            _reference_equations,
            x0,
            args=(v_mp, i_mp, v_oc, i_sc, alpha_eff, beta_eff, temp_ref),
            method="lm",
        )
    return sol.x, sol.success


def _modeled_pmp(temp, alpha_sc, a_ref, i_l_ref, i_o_ref, r_sh_ref, r_s, adjust):
    """Maximum power at 1000 W/m² and ``temp`` via the runtime model path.

    Uses the same ``calcparams_cec`` / ``max_power_point`` path the simulator
    uses (Newton), falling back to the more robust Brent solver and finally to
    NaN if a (typically unphysical, mid-iteration) parameter set will not
    converge — so an exploratory evaluation never raises.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        params = calcparams_cec(_IRRAD_REF, temp, alpha_sc, a_ref, i_l_ref, i_o_ref, r_sh_ref, r_s, adjust)
        for method in ("newton", "brentq"):
            try:
                return float(max_power_point(*params, method=method)["p_mp"])
            except (RuntimeError, ValueError):
                continue
    return float("nan")


def _modeled_gamma(five, adjust, alpha_sc):
    """Modeled maximum-power temperature coefficient in %/°C (Dobos Eq. 10).

    The paper averages adjacent normalised power slopes over -10 → 50 °C; for a
    uniform grid that equals the secant slope between the endpoints, normalised
    by the modeled power at the reference temperature.
    """
    i_l_ref, i_o_ref, r_s, r_sh_ref, a_ref = five
    p_low = _modeled_pmp(_GAMMA_T_LOW, alpha_sc, a_ref, i_l_ref, i_o_ref, r_sh_ref, r_s, adjust)
    p_high = _modeled_pmp(_GAMMA_T_HIGH, alpha_sc, a_ref, i_l_ref, i_o_ref, r_sh_ref, r_s, adjust)
    p_ref = _modeled_pmp(_TEMP_REF, alpha_sc, a_ref, i_l_ref, i_o_ref, r_sh_ref, r_s, adjust)
    return 100.0 * (p_high - p_low) / (p_ref * (_GAMMA_T_HIGH - _GAMMA_T_LOW))


def _is_physical(five, adjust):
    """True if the six parameters fall within the Dobos Table 5 sanity ranges."""
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


_ADJUST_SCAN = np.arange(0.0, 60.0001, 4.0)


def _solve_adjust(v_mp, i_mp, v_oc, i_sc, alpha_sc, beta_voc, gamma_pmp, x0, temp_ref):
    """Find ``Adjust`` so the modeled gamma matches the datasheet ``gamma_pmp``.

    ``gamma`` decreases monotonically with ``Adjust`` along a fixed solution
    branch (a larger ``Adjust`` suppresses the photocurrent's rise with
    temperature, so power falls faster). The branch is followed by warm-starting
    each five-parameter solve from the previous solution (continuation), which
    keeps the inner solver from hopping to a different root as ``Adjust`` moves
    and makes gamma a smooth, monotone function to bracket and refine.

    Returns ``(adjust, five_params)`` for the matched solution, or ``None`` if no
    physical branch reaches the target gamma.
    """

    def march(direction):
        """Walk Adjust outward from 0, returning physical (adjust, residual, five) samples."""
        samples = []
        guess = np.array(x0, dtype=float)
        for step in _ADJUST_SCAN:
            adjust = direction * step
            five, ok = _solve_five_params(adjust, v_mp, i_mp, v_oc, i_sc, alpha_sc, beta_voc, guess, temp_ref)
            if not ok or not _is_physical(five, adjust):
                break
            gamma = _modeled_gamma(five, adjust, alpha_sc)
            if not np.isfinite(gamma):
                break
            guess = five  # continuation: stay on this branch
            samples.append((adjust, gamma - gamma_pmp, five))
        return samples

    # Sample both directions from 0; the +0 sample is shared, so drop the
    # duplicate from the negative march.
    samples = march(+1.0) + march(-1.0)[1:]
    if not samples:
        return None
    samples.sort(key=lambda s: s[0])

    for (a_lo, r_lo, five_lo), (a_hi, r_hi, _five_hi) in zip(samples, samples[1:]):
        if r_lo == 0.0:
            return a_lo, five_lo
        if r_lo * r_hi < 0.0:
            # Refine within the bracket, warm-starting from the low end each call.
            def residual(adjust, _g=np.array(five_lo, dtype=float)):
                five, _ok = _solve_five_params(adjust, v_mp, i_mp, v_oc, i_sc, alpha_sc, beta_voc, _g, temp_ref)
                _g[:] = five
                return _modeled_gamma(five, adjust, alpha_sc) - gamma_pmp

            adjust = brentq(residual, a_lo, a_hi, xtol=1e-10, rtol=1e-12)
            five, _ = _solve_five_params(
                adjust, v_mp, i_mp, v_oc, i_sc, alpha_sc, beta_voc, np.array(five_lo), temp_ref
            )
            return adjust, five

    # No sign change on this branch: return the closest physical sample so the
    # caller can still surface a usable (if gamma-imperfect) result.
    best = min(samples, key=lambda s: abs(s[1]))
    return best[0], best[2]


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
    temp_ref: float = 25.0,
) -> Tuple[float, float, float, float, float, float]:
    """Fit the CEC single-diode reference parameters from datasheet values.

    Drop-in replacement for :func:`pvlib.ivtools.sdm.fit_cec_sam` that needs no
    ``nrel-pysam``. Implements the coefficient calculator of Dobos 2012
    (DOI:10.1115/1.4005759) with ``scipy`` and ``pvlib`` only.

    Args:
        celltype: Cell technology label (kept for signature parity with
            ``fit_cec_sam``; the solver itself is technology-independent).
        Vmp: Voltage at the maximum-power point [V].
        Imp: Current at the maximum-power point [A].
        Voc: Open-circuit voltage [V].
        Isc: Short-circuit current [A].
        alpha_sc: Short-circuit-current temperature coefficient [A/°C].
        beta_voc: Open-circuit-voltage temperature coefficient [V/°C].
        gamma_pmp: Maximum-power temperature coefficient [%/°C].
        cells_in_series: Number of cells in series.
        temp_ref: Reference temperature [°C] (default 25).

    Returns:
        ``(I_L_ref, I_o_ref, R_s, R_sh_ref, a_ref, Adjust)`` — the same tuple,
        in the same order, that ``fit_cec_sam`` returns.

    Raises:
        RuntimeError: if no physical parameter set can be found.
    """
    guesses = (
        _initial_guess(Vmp, Imp, Voc, Isc, cells_in_series, temp_ref),
        _empirical_guess(celltype, Vmp, Imp, Voc, Isc, cells_in_series),
    )

    # Try the reported short-circuit current first and, if no branch matches the
    # datasheet gamma, raise Isc by 1% per iteration (Dobos §5 heuristic, up to
    # five iterations) to move a poorly conditioned module onto a well-shaped
    # I-V curve. The first guess (De Soto) reproduces SAM on well-formed
    # datasheets; the empirical guess (Dobos §4.3) is a robustness fallback. The
    # best-but-imperfect candidate is retained so a usable result is returned
    # even if no branch lands exactly on the target gamma.
    fallback = None
    for bump in range(6):
        i_sc = Isc * (1.01**bump)
        for x0 in guesses:
            result = _solve_adjust(Vmp, Imp, Voc, i_sc, alpha_sc, beta_voc, gamma_pmp, x0, temp_ref)
            if result is None:
                continue
            adjust, five = result
            if not _is_physical(five, adjust):
                continue
            residual = abs(_modeled_gamma(five, adjust, alpha_sc) - gamma_pmp)
            if residual <= _GAMMA_TOL:
                i_l_ref, i_o_ref, r_s, r_sh_ref, a_ref = five
                return (i_l_ref, i_o_ref, r_s, r_sh_ref, a_ref, adjust)
            if fallback is None or residual < fallback[1]:
                fallback = ((five, adjust), residual)

    if fallback is not None:
        (five, adjust), _ = fallback
        i_l_ref, i_o_ref, r_s, r_sh_ref, a_ref = five
        return (i_l_ref, i_o_ref, r_s, r_sh_ref, a_ref, adjust)

    raise RuntimeError("CEC parameter fit did not converge to a physical solution.")
