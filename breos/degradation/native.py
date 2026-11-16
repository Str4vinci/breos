"""The native degradation model's kernels.

Cycle aging and resistance growth follow Naumann et al. (2020), Eq. 5-6,
with the LFP coefficients; calendar aging is the Arrhenius power law with
Naumann's parameters or those of Lam et al. (2025). The energy balance
calls them once per daily degradation window (``breos.battery``); the
documented names stay importable from ``breos.battery``.
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Mapping, Sequence, Tuple, cast

import numpy as np
import pandas as pd

from breos._rainflow import extract_cycles as _extract_rainflow_cycles
from breos.constants import (
    A_Q,
    A_R,
    B_Q,
    B_R,
    C_DOC_Q,
    C_DOC_R,
    D_DOC_Q,
    D_DOC_R,
    LAM_EA_J_MOL,
    LAM_EXPONENT_B,
    LAM_K0_FRAC,
    LAM_SOC_EXPONENT_N,
    NAUMANN_EA_J_MOL,
    NAUMANN_EA_R_J_MOL,
    NAUMANN_EXPONENT_B,
    NAUMANN_EXPONENT_B_R,
    NAUMANN_K0_PERCENT,
    NAUMANN_K0_R_PERCENT,
    NAUMANN_LAM_FIELD_CALIBRATED_EA_J_MOL,
    NAUMANN_LAM_FIELD_CALIBRATED_EXPONENT_B,
    NAUMANN_LAM_FIELD_CALIBRATED_K0_FRAC,
    NAUMANN_LAM_FIELD_CALIBRATED_SOC_EXPONENT_N,
    NAUMANN_LAM_FIELD_CALIBRATED_V2_EA_J_MOL,
    NAUMANN_LAM_FIELD_CALIBRATED_V2_EXPONENT_B,
    NAUMANN_LAM_FIELD_CALIBRATED_V2_K0_FRAC,
    NAUMANN_LAM_FIELD_CALIBRATED_V2_SOC_EXPONENT_N,
    NAUMANN_SOC_EXPONENT_N,
    NAUMANN_SOC_EXPONENT_N_R,
    R_GAS,
    T_REF_K,
    Z_Q,
    Z_R,
)
from breos.utils import _datetime_index_ticks

_FIELD_CALIBRATED_V1 = (
    NAUMANN_LAM_FIELD_CALIBRATED_K0_FRAC,
    NAUMANN_LAM_FIELD_CALIBRATED_EA_J_MOL,
    NAUMANN_LAM_FIELD_CALIBRATED_EXPONENT_B,
    NAUMANN_LAM_FIELD_CALIBRATED_SOC_EXPONENT_N,
)
# Each calendar model's (k0 fraction, activation energy J/mol, time exponent
# b, SOC exponent n). Every model shares Naumann's cycle aging. The keys are
# the configurable names (breos.app_config.CALENDAR_MODELS).
CALENDAR_PARAMETERS: Mapping[str, Tuple[float, float, float, float]] = {
    # Naumann 2020 calendar + cycle (NMC/LFP lab).
    "naumann": (NAUMANN_K0_PERCENT / 100.0, NAUMANN_EA_J_MOL, NAUMANN_EXPONENT_B, NAUMANN_SOC_EXPONENT_N),
    # Naumann cycle + Lam 2025 lab-derived calendar.
    "naumann_lam": (LAM_K0_FRAC, LAM_EA_J_MOL, LAM_EXPONENT_B, LAM_SOC_EXPONENT_N),
    # The v1 field-calibrated fit, under its default alias and its own name.
    "naumann_lam_field_calibrated": _FIELD_CALIBRATED_V1,
    "naumann_lam_field_calibrated_v1": _FIELD_CALIBRATED_V1,
    # The v2 field-calibrated fit: Lam's Ea and n fixed, k0 and b fitted.
    "naumann_lam_field_calibrated_v2": (
        NAUMANN_LAM_FIELD_CALIBRATED_V2_K0_FRAC,
        NAUMANN_LAM_FIELD_CALIBRATED_V2_EA_J_MOL,
        NAUMANN_LAM_FIELD_CALIBRATED_V2_EXPONENT_B,
        NAUMANN_LAM_FIELD_CALIBRATED_V2_SOC_EXPONENT_N,
    ),
}

# Naumann's cycle coefficients (A, B, C_DOC, D_DOC, Z) for capacity and for resistance.
_CAPACITY_CYCLE_COEFFICIENTS = (A_Q, B_Q, C_DOC_Q, D_DOC_Q, Z_Q)
_RESISTANCE_CYCLE_COEFFICIENTS = (A_R, B_R, C_DOC_R, D_DOC_R, Z_R)
# Naumann's resistance calendar parameters (Table 6), in CALENDAR_PARAMETERS' order.
_RESISTANCE_CALENDAR_PARAMETERS = (
    NAUMANN_K0_R_PERCENT / 100.0,
    NAUMANN_EA_R_J_MOL,
    NAUMANN_EXPONENT_B_R,
    NAUMANN_SOC_EXPONENT_N_R,
)


def _get_degradation_params(model: str) -> Tuple[float, float, float, float]:
    """A calendar model's (k0 fraction, Ea, b, n) from :data:`CALENDAR_PARAMETERS`; case and hyphens are ignored."""
    params = CALENDAR_PARAMETERS.get(model.lower().replace("-", "_"))
    if params is None:
        raise ValueError(
            f"Unknown calendar model: {model}. Use 'naumann_lam_field_calibrated', "
            f"'naumann_lam_field_calibrated_v1', "
            f"'naumann_lam_field_calibrated_v2', "
            f"'naumann_lam', or 'naumann'."
        )
    return params


def _cycle_fade(
    cycles: Sequence[Dict],
    fec_cum: float,
    min_doc_fraction: float,
    coefficients: Tuple[float, float, float, float, float],
) -> Tuple[float, float]:
    """Naumann's cycle aging over ``cycles`` from ``fec_cum`` FEC: the fade (fraction) and the FEC after them.

    The differential form of Eq. 5-6: each cycle adds
    ``kC * kDOC * (FEC_new^Z - FEC_old^Z)`` percent.
    """
    a, b, c_doc, d_doc, z = coefficients
    fade = 0.0
    for cyc in cycles:
        doc = max(0.0, min(1.0, cyc["doc"]))
        if doc < min_doc_fraction:
            continue
        # Energy throughput in FEC; a rainflow count is 1.0 for a full cycle and 0.5 for a half.
        fec_new = fec_cum + doc * cyc["count"]
        k_c = max(0.0, a * cyc["mean_c_rate"] + b)
        k_doc = max(0.0, c_doc * ((doc - 0.6) ** 3) + d_doc)
        fade += k_c * k_doc * (fec_new**z - fec_cum**z) / 100.0
        fec_cum = fec_new
    return fade, fec_cum


def _calendar_fade(
    parameters: Tuple[float, float, float, float],
    t_cell_c: float,
    cumulative_seconds: float,
    dt_seconds: float,
    mean_soc_absolute: float,
) -> Tuple[float, float]:
    """The Arrhenius power-law calendar step: the fade (fraction) over ``dt_seconds``, and the seconds after it.

    ``k0 * Arr(T) * ((t + dt)^b - t^b) * SOC^n``, with ``Arr`` relative to 25 °C.
    """
    k0_frac, activation_energy, time_exponent, soc_exponent = parameters
    t_k = t_cell_c + 273.15
    arr_factor = math.exp(-activation_energy / R_GAS * (1.0 / t_k - 1.0 / T_REF_K))
    t_new = cumulative_seconds + dt_seconds
    term_old = math.pow(cumulative_seconds, time_exponent) if cumulative_seconds > 0 else 0.0
    delta_time = math.pow(t_new, time_exponent) - term_old
    soc_stress = max(0.0, mean_soc_absolute) ** soc_exponent
    return k0_frac * arr_factor * delta_time * soc_stress, t_new


def _detect_cycles_rainflow_arrays(
    soc_values: np.ndarray,
    time_ticks: np.ndarray,
    ticks_per_second: float,
    min_doc_fraction: float = 0.01,
) -> List[Dict]:
    """Detect rainflow cycles from arrays used by the simulation hot path."""
    if len(soc_values) < 2:
        return []

    # Count on percent SOC and convert the range back to a fraction; the
    # incremental counter in breos.degradation.protocol rounds the same way.
    soc_pct = soc_values * 100.0

    cycles = []
    for rng, mean, count, i_start, i_end in _extract_rainflow_cycles(soc_pct):
        doc = rng / 100.0  # convert back to fraction
        if doc < min_doc_fraction:
            continue

        mean_soc = mean / 100.0

        # Estimate C-rate from cycle duration
        if i_start < len(time_ticks) and i_end < len(time_ticks):
            duration_seconds = (time_ticks[i_end] - time_ticks[i_start]) / ticks_per_second
            duration_h = duration_seconds / 3600.0
        else:
            duration_h = 0.0
        mean_c_rate = doc / duration_h if duration_h > 0 else 0.0

        cycles.append(
            {
                "doc": doc,
                "mean_soc": mean_soc,
                "count": count,  # 1.0 for full, 0.5 for half
                "mean_c_rate": mean_c_rate,
                "start_idx": i_start,
                "end_idx": i_end,
            }
        )

    return cycles


def update_battery_resistance_cyclewise(
    resistance_growth: float, cycles: List[Dict], fec_cum: float, min_DoD_fraction: float = 0.01
) -> Tuple[float, float]:
    """
    Calculate cycle-induced resistance growth using Naumann's model.

    Uses the same differential form as capacity fade but with resistance
    parameters (A_R, B_R, C_DOC_R, D_DOC_R, Z_R).

    Args:
        resistance_growth: Current cumulative resistance growth (fraction, e.g. 0.05 = 5%)
        cycles: Rainflow cycle dicts, each with ``doc`` (depth, fraction),
            ``count`` (1.0 for a full cycle, 0.5 for a half) and
            ``mean_c_rate``
        fec_cum: Cumulative FEC at start of this period
        min_DoD_fraction: Minimum DOC to count

    Returns:
        Tuple of (new_resistance_growth, delta_resistance_growth)
    """
    delta_R, _fec = _cycle_fade(cycles, fec_cum, min_DoD_fraction, _RESISTANCE_CYCLE_COEFFICIENTS)
    return resistance_growth + delta_R, delta_R


def update_battery_resistance_calendar(
    resistance_growth: float,
    T_cell_C: float,
    cumulative_cal_seconds: float,
    dt_days: float = 1.0,
    mean_soc_absolute: float = 0.5,
) -> Tuple[float, float]:
    """
    Calculate calendar-induced resistance growth using Naumann's model.

    Same Arrhenius + power-law structure as calendar capacity fade,
    but with resistance-specific parameters from Naumann Table 6.

    Args:
        resistance_growth: Current cumulative resistance growth (fraction)
        T_cell_C: Cell temperature (C)
        cumulative_cal_seconds: Total elapsed calendar seconds
        dt_days: Time step in days
        mean_soc_absolute: Mean absolute SOC during period

    Returns:
        Tuple of (new_resistance_growth, delta_resistance_growth)
    """
    dt_seconds = dt_days * 86400.0
    if dt_seconds <= 0:
        return resistance_growth, 0.0
    dR_fraction, _t_new = _calendar_fade(
        _RESISTANCE_CALENDAR_PARAMETERS, T_cell_C, cumulative_cal_seconds, dt_seconds, mean_soc_absolute
    )
    return resistance_growth + dR_fraction, dR_fraction


def resistance_to_efficiency(
    resistance_growth: float,
    base_charge_eff: float,
    base_discharge_eff: float,
) -> Tuple[float, float]:
    """
    Convert resistance growth to effective charge/discharge efficiencies.

    Internal resistance growth increases ohmic losses proportionally. Both
    baseline efficiencies receive the same ``sqrt(1 + growth)`` derating,
    preserving their original ratio.

    RTE_new = RTE_base / (1 + resistance_growth)

    Args:
        resistance_growth: Relative resistance growth (fraction, 0=new cell)
        base_charge_eff: Baseline charge efficiency
        base_discharge_eff: Baseline discharge efficiency

    Returns:
        Tuple of (effective_charge_eff, effective_discharge_eff)
    """
    if resistance_growth <= 0:
        return base_charge_eff, base_discharge_eff

    derate = math.sqrt(1.0 + resistance_growth)
    return base_charge_eff / derate, base_discharge_eff / derate


def _update_battery_soh_from_cycles(
    soh_start_fraction: float,
    cycles: List[Dict],
    *,
    fec_cum: float = 0.0,
    min_DoD_fraction: float = 0.01,
) -> Tuple[float, float, float]:
    qloss_cycle_fraction, fec_cum = _cycle_fade(cycles, fec_cum, min_DoD_fraction, _CAPACITY_CYCLE_COEFFICIENTS)
    return max(0.0, soh_start_fraction - qloss_cycle_fraction), qloss_cycle_fraction, fec_cum


def _update_battery_soh_cyclewise_arrays(
    soh_start_fraction: float,
    soc_values: np.ndarray,
    time_ticks: np.ndarray,
    ticks_per_second: float,
    *,
    fec_cum: float = 0.0,
    min_DoD_fraction: float = 0.01,
) -> Tuple[float, float, float]:
    """Run native rainflow degradation without constructing pandas objects."""
    if len(soc_values) < 2:
        return soh_start_fraction, 0.0, fec_cum
    cycles = _detect_cycles_rainflow_arrays(
        soc_values,
        time_ticks,
        ticks_per_second,
        min_doc_fraction=min_DoD_fraction,
    )
    return _update_battery_soh_from_cycles(
        soh_start_fraction,
        cycles,
        fec_cum=fec_cum,
        min_DoD_fraction=min_DoD_fraction,
    )


def update_battery_soh_cyclewise(
    soh_start_fraction: float,
    soc_series_absolute: pd.Series,
    *,
    fec_cum: float = 0.0,
    min_DoD_fraction: float = 0.01,
) -> Tuple[float, float, float]:
    """
    Calculate cycle-induced degradation using Naumann's semi-empirical model.

    Implements Equation 5-6 from Naumann 2020 paper with the LFP cycle aging
    coefficients. Cycles are counted with rainflow (ASTM E1049) over the
    whole series, and the series' own ends close them.

    Args:
        soh_start_fraction: Starting SOH as fraction (0-1)
        soc_series_absolute: SOC time series
        fec_cum: Cumulative full equivalent cycles
        min_DoD_fraction: Minimum DoD to count as cycle

    Returns:
        Tuple of (soh_after, qloss_cycle_fraction, fec_cum)

    .. deprecated:: 0.8.0
       The simulation ages the battery through its own incremental rainflow
       counter and never calls this function; it will be removed in 0.9.0.
    """
    warnings.warn(
        "breos.battery.update_battery_soh_cyclewise is deprecated and will be removed in BREOS 0.9.0",
        DeprecationWarning,
        stacklevel=2,
    )
    if len(soc_series_absolute) < 2:
        return soh_start_fraction, 0.0, fec_cum

    time_ticks, ticks_per_second = _datetime_index_ticks(cast(pd.DatetimeIndex, soc_series_absolute.index))
    return _update_battery_soh_cyclewise_arrays(
        soh_start_fraction,
        soc_series_absolute.to_numpy(),
        time_ticks,
        ticks_per_second,
        fec_cum=fec_cum,
        min_DoD_fraction=min_DoD_fraction,
    )


def update_battery_soh_calendar(
    soh_start_fraction: float,
    k0_frac: float,
    Ea: float,
    n: float,
    cal_b: float,
    T_cell_C: float = 25.0,
    cumulative_cal_seconds: float = 0.0,
    dt_days: float = 1.0,
    mean_soc_absolute: float = 0.5,
) -> Tuple[float, float, float]:
    """
    Generalized calendar aging using power law physics (Naumann / Lam 2025).

    dSOH = k0_frac * Arr * ((t+dt)^b - t^b) * SOC_stress

    Args:
        soh_start_fraction: Starting SOH as fraction
        k0_frac: Rate constant (fraction per second^b)
        Ea: Activation energy (J/mol)
        n: SOC exponent
        cal_b: Time exponent (0.5 for sqrt-time, 0.75 for Lam)
        T_cell_C: Cell temperature (°C)
        cumulative_cal_seconds: Total elapsed seconds
        dt_days: Time step in days
        mean_soc_absolute: Mean SOC during period

    Returns:
        Tuple of (soh_after, dsoh_fraction, new_cumulative_seconds)
    """
    dt_seconds = dt_days * 86400.0
    if dt_seconds <= 0:
        return soh_start_fraction, 0.0, cumulative_cal_seconds
    d_soh_fraction, t_new = _calendar_fade(
        (k0_frac, Ea, cal_b, n), T_cell_C, cumulative_cal_seconds, dt_seconds, mean_soc_absolute
    )
    return max(0.0, soh_start_fraction - d_soh_fraction), d_soh_fraction, t_new
