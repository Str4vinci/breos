"""Tests for the pure-scipy CEC 6-parameter fit (``breos.cec_fit``)."""

import warnings

import numpy as np
import pytest
from pvlib.pvsystem import calcparams_cec, i_from_v, max_power_point

from breos.cec_fit import _modeled_gamma, _solve_five_params, fit_cec_params
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
    gamma_pmp=-0.36,
    cells_in_series=144,
)


def _mpp_at(params, alpha_sc, irradiance, temp):
    i_l, i_o, r_s, r_sh, a, adjust = params
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cec = calcparams_cec(irradiance, temp, alpha_sc, a, i_l, i_o, r_sh, r_s, adjust)
        return max_power_point(*cec, method="newton")


class TestFitShape:
    def test_returns_six_finite_floats(self):
        params = fit_cec_params(**_KNOWN)
        assert len(params) == 6
        assert all(np.isfinite(v) for v in params)

    def test_parameters_are_physical(self):
        i_l, i_o, r_s, r_sh, a, adjust = fit_cec_params(**_KNOWN)
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
        gamma = _modeled_gamma((i_l, i_o, r_s, r_sh, a), adjust, _KNOWN["alpha_sc"])
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
            np.array([_KNOWN["Isc"], 1e-10, 0.3, 200.0, 2.0]),
            25.0,
        )
        gammas = [_modeled_gamma(five0, adjust, _KNOWN["alpha_sc"]) for adjust in (-10.0, 0.0, 10.0, 20.0)]
        assert all(np.diff(gammas) < 0)


class TestEffectiveCoefficientSign:
    def test_effective_alpha_sc_stays_positive(self):
        # calcparams_cec scales alpha_sc by (1 - Adjust/100); for a positive
        # datasheet alpha_sc and Adjust < 100 the effective coefficient must
        # remain positive (Isc still rises with temperature).
        *_, adjust = fit_cec_params(**_KNOWN)
        effective_alpha = _KNOWN["alpha_sc"] * (1.0 - adjust / 100.0)
        assert effective_alpha > 0


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
            gamma_pmp=module.gamma_pmp,
            cells_in_series=module.N_Cells,
        )
        assert all(np.isfinite(v) for v in params)

        mpp = _mpp_at(params, module.alpha_sc, 1000.0, 25.0)
        assert mpp["p_mp"] == pytest.approx(module.Vmp * module.Imp, rel=2e-3)

        gamma = _modeled_gamma(params[:5], params[5], module.alpha_sc)
        assert gamma == pytest.approx(module.gamma_pmp, abs=5e-3)
