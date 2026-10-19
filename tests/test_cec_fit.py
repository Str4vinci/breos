"""Tests for the pure-scipy CEC 6-parameter fit (``breos.cec_fit``)."""

import warnings

import numpy as np
import pytest
from pvlib.pvsystem import calcparams_cec, i_from_v, max_power_point

from breos.cec_fit import _modeled_gamma, _passes_iv_checks, _solve_five_params, _to_unknowns, fit_cec_params
from breos.pv_modules import MODULES, get_module

# A well-formed datasheet (Suntech STP550S) used for the focused single-module
# assertions; the catalog sweep below covers every bundled module.
_KNOWN = dict(
    celltype="monoSi",
    Vmp=42.05,
    Imp=13.08,
    Voc=49.88,
    Isc=14.01,
    alpha_sc=0.05 * 14.01 / 100,
    beta_voc=-0.304 * 49.88 / 100,
    gamma_pmp=-0.34,
    cells_in_series=144,
)


def _mpp_at(params, alpha_sc, irradiance, temp):
    i_l, i_o, r_s, r_sh, a, adjust = params
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cec = calcparams_cec(irradiance, temp, alpha_sc, a, i_l, i_o, r_sh, r_s, adjust)
        return max_power_point(*cec, method="newton")


class TestFitShape:
    def test_returns_six_finite_physical_parameters(self):
        params = fit_cec_params(**_KNOWN)
        assert all(np.isfinite(v) for v in params)
        i_l, i_o, r_s, r_sh, a, adjust = params
        assert i_l > 0
        assert i_o > 0
        assert r_s > 0
        assert r_sh > 0
        assert a > 0
        assert -100.0 <= adjust <= 100.0


class TestParameterRecovery:
    def test_recovers_stc_maximum_power_point(self):
        params = fit_cec_params(**_KNOWN)
        mpp = _mpp_at(params, _KNOWN["alpha_sc"], 1000.0, 25.0)
        # The fit constrains the model to pass through the datasheet MPP at STC.
        assert mpp["p_mp"] == pytest.approx(_KNOWN["Vmp"] * _KNOWN["Imp"], rel=2e-3)
        assert mpp["v_mp"] == pytest.approx(_KNOWN["Vmp"], rel=5e-3)
        assert mpp["i_mp"] == pytest.approx(_KNOWN["Imp"], rel=5e-3)

    def test_open_circuit_current_is_essentially_zero(self):
        i_l, i_o, r_s, r_sh, a, adjust = fit_cec_params(**_KNOWN)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cec = calcparams_cec(1000.0, 25.0, _KNOWN["alpha_sc"], a, i_l, i_o, r_sh, r_s, adjust)
            current_at_voc = float(i_from_v(_KNOWN["Voc"], *cec))
        # Dobos validity check: current at Voc must be well under 1.5% of Imp.
        assert abs(current_at_voc) < 0.015 * _KNOWN["Imp"]


class TestGammaMatch:
    def test_modeled_gamma_matches_datasheet(self):
        i_l, i_o, r_s, r_sh, a, adjust = fit_cec_params(**_KNOWN)
        gamma = _modeled_gamma(_to_unknowns(i_l, i_o, r_s, r_sh, a), adjust, _KNOWN["alpha_sc"])
        assert gamma == pytest.approx(_KNOWN["gamma_pmp"], abs=2e-3)

    def test_gamma_is_monotonic_decreasing_in_adjust(self):
        # On a fixed solution branch, raising Adjust suppresses the photocurrent's
        # rise with temperature, so the modeled power falls faster (more negative
        # gamma). Solve the five params per Adjust and check the trend.
        five0, _ = _solve_five_params(
            0.0,
            _KNOWN["Vmp"],
            _KNOWN["Imp"],
            _KNOWN["Voc"],
            _KNOWN["Isc"],
            _KNOWN["alpha_sc"],
            _KNOWN["beta_voc"],
            _to_unknowns(_KNOWN["Isc"], 1e-10, 0.3, 200.0, 2.0),
        )
        gammas = [_modeled_gamma(five0, adjust, _KNOWN["alpha_sc"]) for adjust in (-10.0, 0.0, 10.0, 20.0)]
        assert all(np.diff(gammas) < 0)

    def test_unreachable_gamma_returns_closest_physical_solution(self):
        # The search runs to Adjust = 100, where I_o leaves SAM's range; the
        # closest sample that passes the checks is returned with a warning.
        inputs = dict(_KNOWN, gamma_pmp=-0.8)
        with pytest.warns(RuntimeWarning, match="closest physical solution"):
            i_l, i_o, r_s, r_sh, a, adjust = fit_cec_params(**inputs)
        assert i_o <= 1e-7
        assert 0.0 < adjust < 100.0
        assert _passes_iv_checks((i_l, i_o, r_s, r_sh, a), _KNOWN["Vmp"], _KNOWN["Imp"], _KNOWN["Voc"])


class TestCatalogModules:
    @pytest.mark.parametrize("name", list(MODULES))
    def test_every_bundled_module_fits_and_recovers_mpp(self, name):
        module = get_module(name)
        params = fit_cec_params(
            celltype=module.celltype,
            Vmp=module.Vmp,
            Imp=module.Imp,
            Voc=module.Voc,
            Isc=module.Isc,
            alpha_sc=module.alpha_sc,
            beta_voc=module.beta_voc,
            gamma_pmp=module.gamma_pmp_effective,
            cells_in_series=module.N_Cells,
        )
        assert all(np.isfinite(v) for v in params)

        mpp = _mpp_at(params, module.alpha_sc, 1000.0, 25.0)
        assert mpp["p_mp"] == pytest.approx(module.Vmp * module.Imp, rel=2e-3)

        gamma = _modeled_gamma(_to_unknowns(*params[:5]), params[5], module.alpha_sc)
        assert gamma == pytest.approx(module.gamma_pmp_effective, abs=5e-3)


# Frozen datasheet inputs and fit_cec_sam results (PySAM 7.1.1) of CEC database
# modules that exercise the solver, so CI does not need PySAM:
# (fit_cec_params arguments, (I_L_ref, I_o_ref, R_s, R_sh_ref, a_ref, Adjust)).
_SAM_REFERENCE = [
    # Nanjing_Daqo_New_Energy_DQ200PSBb: Adjust above 60.
    (
        ("multiSi", 26.9, 7.44, 32.8, 8.24, 0.006543, -0.074718, -0.4613, 54),
        (
            8.269069840391914,
            5.19252505427409e-10,
            0.23116476360784985,
            65.52487796610089,
            1.3999918695791396,
            66.74309014923772,
        ),
    ),
    # Gloria_Solar_GSM6_200D_E1: failed before.
    (
        ("multiSi", 28.8, 6.96, 36.55, 8.05, 0.009741, -0.130118, -0.558, 60),
        (
            8.119128672408591,
            1.5601867134826873e-08,
            0.37819272729460884,
            44.04040006289633,
            1.8309540712831052,
            40.38688694056353,
        ),
    ),
    # Jinko_Solar_Co___Ltd_JKM295M_60HB_V: failed after the 5 degC correction.
    (
        ("monoSi", 32.4, 9.1, 39.7, 9.61, 0.006823, -0.136171, -0.407, 120),
        (
            9.611464874116896,
            2.1898781671452977e-10,
            0.27341469315185174,
            1793.680948568271,
            1.6202325029575404,
            1.3775439987688247,
        ),
    ),
    # Polar_Photovoltaics_TFSM_T_0: thin-film convergence.
    (
        ("amorphous", 58.0, 0.72, 78.0, 0.92, 0.000828, -0.2886, -0.27, 47),
        (
            0.9614565218767317,
            5.2711648119121435e-12,
            15.244005624270613,
            338.29382509533855,
            3.040310499996922,
            -16.00005825282195,
        ),
    ),
    # Centrosolar_America_CP72_325SW: Adjust root skipped by a failed step.
    (
        ("multiSi", 37.52, 8.68, 46.38, 9.19, 0.007352, -0.16233, -0.47, 72),
        (
            9.190201453112051,
            9.64474800810391e-10,
            0.3473254619242431,
            15844.772477244247,
            2.0185177276116404,
            12.315996413393556,
        ),
    ),
    # Tainergy_Tech_TKSD_16501: gamma-tolerance miss.
    (
        ("multiSi", 22.32, 7.41, 25.87, 8.15, 0.004479, -0.08494, -0.459499, 42),
        (
            8.150410115308897,
            1.4720846991878344e-09,
            0.0029924663561067574,
            59.46767200052025,
            1.1559530064085857,
            25.704475355112542,
        ),
    ),
    # Seraphim_Energy_Group_Inc__SEG_E11B_310: shingled, CEC N_s counts parallel strips; SAM raises Isc 3 %.
    (
        ("multiSi", 36.15, 8.58, 44.45, 8.86, 0.00886, -0.13335, -0.439, 340),
        (
            9.133883079326763,
            5.510294716840398e-10,
            0.3214673528979336,
            541.7995451418359,
            1.8897041094338336,
            25.32032562464214,
        ),
    ),
    # Solaria_Corporation_Solaria_PowerXT_340R_BD: shingled, CEC N_s counts parallel strips.
    (
        ("monoSi", 38.6, 8.79, 46.9, 9.36, 0.004549, -0.144874, -0.4202, 360),
        (
            9.367966262671969,
            1.8216883978920352e-10,
            0.2945222455407084,
            346.05038909639075,
            1.9027287580747594,
            11.554975781471429,
        ),
    ),
]


def _grid_power(params, alpha_sc):
    i_l, i_o, r_s, r_sh, a, adjust = params
    g, t = (x.ravel() for x in np.meshgrid([100.0, 400.0, 800.0, 1000.0, 1100.0], [-10.0, 25.0, 45.0, 75.0]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cec = calcparams_cec(g, t, alpha_sc, a, i_l, i_o, r_sh, r_s, adjust)
        return max_power_point(*cec, method="newton")["p_mp"]


class TestSamReference:
    @pytest.mark.parametrize("inputs,reference", _SAM_REFERENCE, ids=[str(c[0][1:4]) for c in _SAM_REFERENCE])
    def test_matches_fit_cec_sam(self, inputs, reference):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)  # no gamma-tolerance fallback
            params = fit_cec_params(*inputs)
        assert params[5] == pytest.approx(reference[5], abs=1e-3)
        np.testing.assert_allclose(_grid_power(params, inputs[5]), _grid_power(reference, inputs[5]), rtol=1e-5)


class TestIVChecks:
    def test_sam_reference_curve_passes(self):
        inputs, reference = _SAM_REFERENCE[0]
        assert _passes_iv_checks(reference[:5], inputs[1], inputs[2], inputs[3])

    def test_curve_far_from_the_datasheet_mpp_is_rejected(self):
        inputs, reference = _SAM_REFERENCE[0]
        i_l, i_o, r_s, _r_sh, a = reference[:5]
        # A 2-ohm shunt drags the power at Vmp far below Vmp * Imp.
        assert not _passes_iv_checks((i_l, i_o, r_s, 2.0, a), inputs[1], inputs[2], inputs[3])
