"""Validation harness for the pure-scipy CEC fit (``breos.cec_fit``).

Compares :func:`breos.cec_fit.fit_cec_params` against the ``nrel-pysam`` oracle
:func:`pvlib.ivtools.sdm.fit_cec_sam` for every bundled catalog module, on three
levels of increasing relevance:

1. the six reference parameters;
2. the downstream maximum power across a temperature x irradiance grid (the
   number that actually feeds the simulator);
3. a full-year DC energy run per module.

Run this while ``nrel-pysam`` is still installed (it is the oracle):

    .venv/bin/python tools/validate_cec_fit.py

Gates: <=0.1% max Pmp deviation over the grid, and <=0.1% annual-energy
deviation per module. This script is a development artifact; it is not part of
the shipped package and is not exercised by the test suite.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pvlib  # noqa: E402
from pvlib.location import Location  # noqa: E402
from pvlib.pvsystem import calcparams_cec, max_power_point  # noqa: E402

import breos.solar as solar  # noqa: E402
from breos.cec_fit import fit_cec_params  # noqa: E402
from breos.pv_modules import MODULES  # noqa: E402

PARAM_LABELS = ("I_L_ref", "I_o_ref", "R_s", "R_sh_ref", "a_ref", "Adjust")

# Downstream Pmp comparison grid: realistic operating envelope.
GRID_IRRAD = (100.0, 200.0, 400.0, 600.0, 800.0, 1000.0)
GRID_TEMP = (-10.0, 0.0, 10.0, 25.0, 40.0, 55.0, 70.0)

PMP_GATE_PCT = 0.1
ENERGY_GATE_PCT = 0.1


def _oracle(module):
    """Reference six-tuple from SAM via pvlib (requires nrel-pysam)."""
    return pvlib.ivtools.sdm.fit_cec_sam(
        celltype=module.celltype,
        v_mp=module.Vmp,
        i_mp=module.Imp,
        v_oc=module.Voc,
        i_sc=module.Isc,
        alpha_sc=module.alpha_sc,
        beta_voc=module.beta_voc,
        gamma_pmp=module.gamma_pmp,
        cells_in_series=module.N_Cells,
    )


def _candidate(module):
    return fit_cec_params(
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


def _pmp_grid(params, alpha_sc):
    i_l, i_o, r_s, r_sh, a, adjust = params
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for g in GRID_IRRAD:
            for t in GRID_TEMP:
                cec = calcparams_cec(g, t, alpha_sc, a, i_l, i_o, r_sh, r_s, adjust)
                out.append(max_power_point(*cec, method="newton")["p_mp"])
    return np.array(out)


def _synthetic_weather(year=2023):
    """One-year hourly weather with sinusoidal patterns (mirrors test fixture)."""
    n = 8760
    index = pd.date_range(start=f"{year}-01-01", periods=n, freq="h", tz="UTC")
    hour_of_year = np.arange(n, dtype=float)
    day_of_year = hour_of_year / 24.0
    hour_of_day = hour_of_year % 24
    solar_angle = np.clip(np.sin((hour_of_day - 6) / 12 * np.pi), 0, 1)
    seasonal = 0.6 + 0.4 * np.sin((day_of_year - 80) / 365 * 2 * np.pi)
    ghi = solar_angle * seasonal * 800
    temp_air = 15 + 8 * np.sin((day_of_year - 80) / 365 * 2 * np.pi) + 4 * np.sin((hour_of_day - 14) / 24 * 2 * np.pi)
    return pd.DataFrame(
        {"ghi": ghi, "dni": ghi * 0.7, "dhi": ghi * 0.3, "temp_air": temp_air, "wind_speed": np.full(n, 3.0)},
        index=index,
    )


def _annual_kwh(module, weather, location, fit_fn):
    """Annual DC kWh for one module using ``fit_fn`` as the CEC fit."""
    original = solar.fit_cec_params
    solar.fit_cec_params = fit_fn
    solar._cec_param_cache.clear()
    try:
        dc = solar.calculate_pv_production_dc(
            weather_data=weather,
            location=location,
            tilt=35,
            surface_azimuth=180,
            n_modules=1,
            pv_params=module,
            freq="h",
        )
    finally:
        solar.fit_cec_params = original
        solar._cec_param_cache.clear()
    return float(dc.sum()) / 1000.0


def _sam_fit_fn(celltype, Vmp, Imp, Voc, Isc, alpha_sc, beta_voc, gamma_pmp, cells_in_series):
    return pvlib.ivtools.sdm.fit_cec_sam(
        celltype=celltype,
        v_mp=Vmp,
        i_mp=Imp,
        v_oc=Voc,
        i_sc=Isc,
        alpha_sc=alpha_sc,
        beta_voc=beta_voc,
        gamma_pmp=gamma_pmp,
        cells_in_series=cells_in_series,
    )


def main():
    try:
        _oracle(next(iter(MODULES.values())))
    except ImportError:
        print("nrel-pysam is not installed; the oracle is unavailable. Run this before removing the dependency.")
        return 2

    weather = _synthetic_weather()
    location = Location(41.1579, -8.6291, tz="Europe/Lisbon")

    worst_pmp = 0.0
    worst_energy = 0.0
    all_pass = True

    for name, module in MODULES.items():
        sam = _oracle(module)
        new = _candidate(module)

        print(f"\n=== {name}  (gamma_pmp={module.gamma_pmp}, N_Cells={module.N_Cells}) ===")
        for label, s, n in zip(PARAM_LABELS, sam, new):
            rel = abs(s - n) / abs(s) if s else float("nan")
            print(f"  {label:9s}  sam={s: .6e}  new={n: .6e}  rel={rel:.2e}")

        pmp_dev = float(
            np.max(
                np.abs(_pmp_grid(sam, module.alpha_sc) - _pmp_grid(new, module.alpha_sc))
                / _pmp_grid(sam, module.alpha_sc)
            )
            * 100.0
        )
        worst_pmp = max(worst_pmp, pmp_dev)

        kwh_sam = _annual_kwh(module, weather, location, _sam_fit_fn)
        kwh_new = _annual_kwh(module, weather, location, fit_cec_params)
        energy_dev = abs(kwh_sam - kwh_new) / kwh_sam * 100.0
        worst_energy = max(worst_energy, energy_dev)

        pmp_ok = pmp_dev <= PMP_GATE_PCT
        energy_ok = energy_dev <= ENERGY_GATE_PCT
        all_pass = all_pass and pmp_ok and energy_ok
        print(f"  max Pmp deviation over T x G grid : {pmp_dev:.4f}%  [{'PASS' if pmp_ok else 'FAIL'}]")
        print(
            f"  annual DC energy  sam={kwh_sam:.3f} kWh  new={kwh_new:.3f} kWh  "
            f"dev={energy_dev:.4f}%  [{'PASS' if energy_ok else 'FAIL'}]"
        )

    print("\n" + "=" * 70)
    print(f"WORST Pmp deviation   : {worst_pmp:.4f}%   (gate <= {PMP_GATE_PCT}%)")
    print(f"WORST energy deviation: {worst_energy:.4f}%   (gate <= {ENERGY_GATE_PCT}%)")
    print("RESULT:", "ALL GATES PASS" if all_pass else "GATE FAILURE")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
