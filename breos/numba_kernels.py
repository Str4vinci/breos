"""
Numba-accelerated kernels for performance-critical operations.

This module provides JIT-compiled versions of hot loops for:
- Energy balance simulation
- SOC tracking
- Degradation updates

WARNING: these kernels are APPROXIMATE standalone engines with no
production callers — ``breos.App`` and ``simulate_energy_balance`` always
use the reference Python path in ``battery.py``. The kernels differ from
the reference model: no inverter conversion losses or AC clipping, a
segment-based depth-of-cycle proxy instead of rainflow counting, and no
replacement logic. Do not mix kernel outputs with reference-path results
in published numbers. The LFP temperature derate
(``_lfp_capacity_factor_numba``) is parity-tested against
``battery.lfp_capacity_factor``.
"""

from typing import Tuple

import numpy as np
from numba import jit, prange


@jit(nopython=True, cache=True)
def _lfp_capacity_factor_numba(T_C: float) -> float:
    """LFP temperature capacity derating used by numba degradation kernels.

    Mirrors lfp_capacity_factor in battery.py (LFP_CAP_DERATE_PER_C_*) so
    per-step dispatch limits stay in sync with the Python reference path.
    """
    if T_C >= 25.0:
        return 1.0
    if T_C >= 0.0:
        return 1.0 - 0.002 * (25.0 - T_C)
    base_at_zero = 1.0 - 0.002 * 25.0
    value = base_at_zero - 0.010 * abs(T_C)
    if value < 0.5:
        return 0.5
    return value


@jit(nopython=True, cache=True)
def energy_balance_kernel(
    pv_energy_wh: np.ndarray,
    load_energy_wh: np.ndarray,
    battery_nominal_wh: float,
    max_soc: float,
    min_soc: float,
    charge_efficiency: float,
    discharge_efficiency: float,
    standby_loss_wh: float,
    initial_soh: float,
    hours_per_step: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-optimized energy balance kernel.

    Args:
        pv_energy_wh: PV energy per timestep (Wh)
        load_energy_wh: Load energy per timestep (Wh)
        battery_nominal_wh: Nominal battery capacity (Wh)
        max_soc: Maximum SOC (0-1)
        min_soc: Minimum SOC (0-1)
        charge_efficiency: Charge efficiency (0-1)
        discharge_efficiency: Discharge efficiency (0-1)
        standby_loss_wh: Standby loss per hour (Wh)
        initial_soh: Initial state of health (0-1)
        hours_per_step: Hours per simulation timestep

    Returns:
        Tuple of (import_wh, sell_wh, battery_energy_wh, soc_normalized, soc_absolute)
    """
    n_steps = len(pv_energy_wh)

    # Output arrays
    import_wh = np.zeros(n_steps)
    sell_wh = np.zeros(n_steps)
    battery_energy_wh = np.zeros(n_steps)
    soc_normalized = np.zeros(n_steps)
    soc_absolute = np.zeros(n_steps)

    # Initialize battery state
    soh = initial_soh
    usable_cap = battery_nominal_wh * soh
    Emax = usable_cap * max_soc
    Emin = usable_cap * min_soc
    battery_E = Emax  # Start at max SOC

    for i in range(n_steps):
        pv = pv_energy_wh[i]
        load = load_energy_wh[i]
        surplus = pv - load

        # Apply standby loss
        standby = standby_loss_wh * hours_per_step
        battery_E = max(Emin, battery_E - standby)

        if surplus >= 0:
            # Charging - excess PV
            charge_room = Emax - battery_E
            charge_in = min(surplus, charge_room / charge_efficiency)
            battery_E += charge_in * charge_efficiency
            sell_wh[i] = surplus - charge_in
        else:
            # Discharging - deficit
            deficit = -surplus
            available = battery_E - Emin
            if available > 0:
                draw = min(available, deficit / discharge_efficiency)
                battery_E -= draw
                delivered = draw * discharge_efficiency
                import_wh[i] = max(0.0, deficit - delivered)
            else:
                import_wh[i] = deficit

        # Store battery state
        battery_energy_wh[i] = battery_E

        # Calculate SOC
        if (Emax - Emin) > 0:
            soc_normalized[i] = (battery_E - Emin) / (Emax - Emin)
        else:
            soc_normalized[i] = 0.0

        if usable_cap > 0:
            soc_absolute[i] = battery_E / usable_cap
        else:
            soc_absolute[i] = 0.0

        # Clip SOC values
        soc_normalized[i] = max(0.0, min(1.0, soc_normalized[i]))
        soc_absolute[i] = max(0.0, min(1.0, soc_absolute[i]))

    return import_wh, sell_wh, battery_energy_wh, soc_normalized, soc_absolute


@jit(nopython=True, cache=True, parallel=True)
def batch_energy_balance_kernel(
    pv_scenarios: np.ndarray,  # (n_scenarios, n_steps)
    load_scenarios: np.ndarray,  # (n_scenarios, n_steps)
    battery_nominal_wh: float,
    max_soc: float,
    min_soc: float,
    charge_efficiency: float,
    discharge_efficiency: float,
    standby_loss_wh: float,
    initial_soh: float,
    hours_per_step: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parallel Numba kernel for batch energy balance (Monte Carlo).

    Args:
        pv_scenarios: PV energy per timestep for each scenario (Wh)
        load_scenarios: Load energy per timestep for each scenario (Wh)
        ... (same as energy_balance_kernel)

    Returns:
        Tuple of (total_import, total_sell, grid_independence) arrays
    """
    n_scenarios = pv_scenarios.shape[0]
    n_steps = pv_scenarios.shape[1]

    total_import = np.zeros(n_scenarios)
    total_sell = np.zeros(n_scenarios)
    total_load = np.zeros(n_scenarios)
    grid_independence = np.zeros(n_scenarios)

    for s in prange(n_scenarios):
        soh = initial_soh
        usable_cap = battery_nominal_wh * soh
        Emax = usable_cap * max_soc
        Emin = usable_cap * min_soc
        battery_E = Emax

        imp = 0.0
        sell = 0.0
        load_sum = 0.0

        for i in range(n_steps):
            pv = pv_scenarios[s, i]
            load = load_scenarios[s, i]
            surplus = pv - load
            load_sum += load

            # Standby loss
            standby = standby_loss_wh * hours_per_step
            battery_E = max(Emin, battery_E - standby)

            if surplus >= 0:
                charge_room = Emax - battery_E
                charge_in = min(surplus, charge_room / charge_efficiency)
                battery_E += charge_in * charge_efficiency
                sell += surplus - charge_in
            else:
                deficit = -surplus
                available = battery_E - Emin
                if available > 0:
                    draw = min(available, deficit / discharge_efficiency)
                    battery_E -= draw
                    delivered = draw * discharge_efficiency
                    imp += max(0.0, deficit - delivered)
                else:
                    imp += deficit

        total_import[s] = imp
        total_sell[s] = sell
        total_load[s] = load_sum

        if load_sum > 0:
            grid_independence[s] = 100.0 * (1.0 - imp / load_sum)
        else:
            grid_independence[s] = 100.0

    return total_import, total_sell, grid_independence


# =========================================================================
# Degradation-aware kernels
# =========================================================================


@jit(nopython=True, cache=True)
def _daily_degradation_step(
    soc_day: np.ndarray,
    battery_nominal_wh: float,
    fec_cum: float,
    cumulative_cal_seconds: float,
    soh: float,
    temperature: float,
    hours_per_step: float,
    # Naumann cycle params
    A_Q: float,
    B_Q: float,
    C_DOC_Q: float,
    D_DOC_Q: float,
    Z_Q: float,
    # Calendar params
    k0_frac: float,
    Ea: float,
    cal_b: float,
    n_soc: float,
    R_GAS: float,
    T_REF_K: float,
) -> Tuple[float, float, float, float, float]:
    """
    Numba-compiled daily degradation step using aggregate SOC metrics.

    Instead of full rainflow counting (which can't run in nopython mode),
    this uses daily max-min SOC as a DOC proxy and total energy throughput
    for FEC calculation. This is a deliberate simplification for speed.

    Args:
        soc_day: SOC values for this day (absolute, 0-1)
        battery_nominal_wh: Nominal battery capacity (Wh)
        fec_cum: Cumulative FEC entering this day
        cumulative_cal_seconds: Cumulative calendar seconds entering this day
        soh: Current SOH (fraction)
        temperature: Mean cell temperature for the day (C)
        hours_per_step: Hours per simulation step
        A_Q, B_Q, C_DOC_Q, D_DOC_Q, Z_Q: Naumann cycle aging parameters
        k0_frac, Ea, cal_b, n_soc: Calendar aging parameters
        R_GAS, T_REF_K: Physical constants

    Returns:
        Tuple of (new_soh, new_fec_cum, new_cal_seconds, dSOH_cycle, dSOH_calendar)
    """
    n = len(soc_day)
    if n < 2:
        return soh, fec_cum, cumulative_cal_seconds + 86400.0, 0.0, 0.0

    dSOH_cycle = 0.0

    # Cycle degradation: walk the day's SOC trace and treat each direction
    # reversal as a half-cycle (DOC = |soc_extremum - soc_prev_extremum|).
    # This keeps dFEC semantics aligned with the rainflow reference used in
    # the pandas-based dispatch path while remaining numba-compatible.
    last_ext_idx = 0
    last_direction = 0
    tiny_hysteresis = 1e-12

    for i in range(1, n):
        delta = soc_day[i] - soc_day[i - 1]
        direction = 0
        if delta > tiny_hysteresis:
            direction = 1
        elif delta < -tiny_hysteresis:
            direction = -1

        if direction == 0:
            continue

        if last_direction == 0:
            last_direction = direction
        elif direction != last_direction:
            eidx = i - 1
            DOC = abs(soc_day[eidx] - soc_day[last_ext_idx])
            if DOC >= 0.01:
                duration_h = (eidx - last_ext_idx) * hours_per_step
                mean_c_rate = DOC / duration_h if duration_h > 0.0 else 0.0
                kC = A_Q * mean_c_rate + B_Q
                if kC < 0.0:
                    kC = 0.0
                kDOC = C_DOC_Q * ((DOC - 0.6) ** 3) + D_DOC_Q
                if kDOC < 0.0:
                    kDOC = 0.0
                dFEC = DOC * 0.5
                fec_new = fec_cum + dFEC
                dq_pct = kC * kDOC * (fec_new**Z_Q - fec_cum**Z_Q)
                dSOH_cycle += dq_pct / 100.0
                fec_cum = fec_new

            last_ext_idx = eidx
            last_direction = direction

    # Trailing half-cycle from the final extremum to end of day
    DOC = abs(soc_day[n - 1] - soc_day[last_ext_idx])
    if DOC >= 0.01:
        duration_h = (n - 1 - last_ext_idx) * hours_per_step
        mean_c_rate = DOC / duration_h if duration_h > 0.0 else 0.0
        kC = A_Q * mean_c_rate + B_Q
        if kC < 0.0:
            kC = 0.0
        kDOC = C_DOC_Q * ((DOC - 0.6) ** 3) + D_DOC_Q
        if kDOC < 0.0:
            kDOC = 0.0
        dFEC = DOC * 0.5
        fec_new = fec_cum + dFEC
        dq_pct = kC * kDOC * (fec_new**Z_Q - fec_cum**Z_Q)
        dSOH_cycle += dq_pct / 100.0
        fec_cum = fec_new

    # Calendar degradation
    T_K = temperature + 273.15
    arr_factor = np.exp(-Ea / R_GAS * (1.0 / T_K - 1.0 / T_REF_K))

    t_old = cumulative_cal_seconds
    t_new = t_old + 86400.0  # 1 day

    if t_old > 0:
        term_old = t_old**cal_b
    else:
        term_old = 0.0
    term_new = t_new**cal_b
    delta_time = term_new - term_old

    mean_soc = 0.0
    for i in range(n):
        mean_soc += soc_day[i]
    mean_soc /= n

    soc_stress = mean_soc**n_soc if mean_soc > 0 else 0.0

    dSOH_calendar = k0_frac * arr_factor * delta_time * soc_stress

    new_soh = soh - dSOH_cycle - dSOH_calendar
    if new_soh < 0.0:
        new_soh = 0.0

    return new_soh, fec_cum, t_new, dSOH_cycle, dSOH_calendar


@jit(nopython=True, cache=True)
def energy_balance_kernel_with_degradation(
    pv_energy_wh: np.ndarray,
    load_energy_wh: np.ndarray,
    temperature_c: np.ndarray,
    battery_nominal_wh: float,
    max_soc: float,
    min_soc: float,
    charge_efficiency: float,
    discharge_efficiency: float,
    standby_loss_wh: float,
    initial_soh: float,
    hours_per_step: float,
    steps_per_day: int,
    # Naumann cycle params
    A_Q: float,
    B_Q: float,
    C_DOC_Q: float,
    D_DOC_Q: float,
    Z_Q: float,
    # Calendar params
    k0_frac: float,
    Ea: float,
    cal_b: float,
    n_soc: float,
    R_GAS: float,
    T_REF_K: float,
    initial_fec: float,
    initial_cal_seconds: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """
    Numba-optimized energy balance kernel with daily degradation updates.

    Extends the base kernel by computing daily SOH degradation using
    aggregate metrics (DOC proxy, mean C-rate) instead of full rainflow.

    Args:
        pv_energy_wh: PV energy per timestep (Wh)
        load_energy_wh: Load energy per timestep (Wh)
        temperature_c: Ambient temperature per timestep (C)
        battery_nominal_wh: Nominal battery capacity (Wh)
        max_soc, min_soc: SOC limits (0-1)
        charge_efficiency, discharge_efficiency: Efficiencies (0-1)
        standby_loss_wh: Standby loss per hour (Wh)
        initial_soh: Initial SOH (fraction, 0-1)
        hours_per_step: Hours per timestep
        steps_per_day: Number of steps per day
        A_Q..Z_Q: Naumann cycle aging parameters
        k0_frac..n_soc: Calendar aging parameters
        R_GAS, T_REF_K: Physical constants
        initial_fec: Initial cumulative FEC
        initial_cal_seconds: Initial cumulative calendar seconds

    Returns:
        Tuple of (import_wh, sell_wh, battery_energy_wh, soc_normalized,
                  soc_absolute, final_soh, final_fec, final_cal_seconds)
    """
    n_steps = len(pv_energy_wh)

    import_wh = np.zeros(n_steps)
    sell_wh = np.zeros(n_steps)
    battery_energy_wh = np.zeros(n_steps)
    soc_normalized = np.zeros(n_steps)
    soc_absolute = np.zeros(n_steps)

    soh = initial_soh
    fec_cum = initial_fec
    cal_seconds = initial_cal_seconds

    usable_cap = battery_nominal_wh * soh
    Emax = usable_cap * max_soc
    Emin = usable_cap * min_soc
    battery_E = Emax

    # Buffer for daily SOC values
    day_soc = np.zeros(steps_per_day)
    day_idx = 0
    day_temp_sum = 0.0

    for i in range(n_steps):
        pv = pv_energy_wh[i]
        load = load_energy_wh[i]
        surplus = pv - load

        f_cap = _lfp_capacity_factor_numba(temperature_c[i])
        Emax_step = usable_cap * max_soc * f_cap
        Emin_step = usable_cap * min_soc * f_cap

        standby = standby_loss_wh * hours_per_step
        battery_E = min(Emax_step, max(Emin_step, battery_E - standby))

        if surplus >= 0:
            charge_room = Emax_step - battery_E
            charge_in = min(surplus, charge_room / charge_efficiency)
            battery_E += charge_in * charge_efficiency
            sell_wh[i] = surplus - charge_in
        else:
            deficit = -surplus
            available = battery_E - Emin_step
            if available > 0:
                draw = min(available, deficit / discharge_efficiency)
                battery_E -= draw
                delivered = draw * discharge_efficiency
                import_wh[i] = max(0.0, deficit - delivered)
            else:
                import_wh[i] = deficit

        battery_energy_wh[i] = battery_E

        if (Emax_step - Emin_step) > 0:
            soc_normalized[i] = max(0.0, min(1.0, (battery_E - Emin_step) / (Emax_step - Emin_step)))
        else:
            soc_normalized[i] = 0.0

        soc_abs_val = battery_E / usable_cap if usable_cap > 0 else 0.0
        soc_abs_val = max(0.0, min(1.0, soc_abs_val))
        soc_absolute[i] = soc_abs_val

        # Accumulate daily data
        day_soc[day_idx] = soc_abs_val
        day_temp_sum += temperature_c[i]
        day_idx += 1

        # End of day: apply degradation
        if day_idx >= steps_per_day:
            mean_temp = day_temp_sum / steps_per_day

            soh, fec_cum, cal_seconds, _, _ = _daily_degradation_step(
                day_soc,
                battery_nominal_wh,
                fec_cum,
                cal_seconds,
                soh,
                mean_temp,
                hours_per_step,
                A_Q,
                B_Q,
                C_DOC_Q,
                D_DOC_Q,
                Z_Q,
                k0_frac,
                Ea,
                cal_b,
                n_soc,
                R_GAS,
                T_REF_K,
            )

            # Update nominal SOH capacity limits for next day. Per-step
            # temperature derating is applied before each dispatch step.
            usable_cap = battery_nominal_wh * soh
            Emax = usable_cap * max_soc
            Emin = usable_cap * min_soc
            f_cap = _lfp_capacity_factor_numba(temperature_c[i])
            battery_E = min(battery_E, Emax * f_cap)
            battery_E = max(battery_E, Emin * f_cap)

            day_idx = 0
            day_temp_sum = 0.0

    return import_wh, sell_wh, battery_energy_wh, soc_normalized, soc_absolute, soh, fec_cum, cal_seconds


@jit(nopython=True, cache=True, parallel=True)
def batch_energy_balance_kernel_with_degradation(
    pv_scenarios: np.ndarray,
    load_scenarios: np.ndarray,
    temperature_c: np.ndarray,
    battery_nominal_wh: float,
    max_soc: float,
    min_soc: float,
    charge_efficiency: float,
    discharge_efficiency: float,
    standby_loss_wh: float,
    initial_soh: float,
    hours_per_step: float,
    steps_per_day: int,
    A_Q: float,
    B_Q: float,
    C_DOC_Q: float,
    D_DOC_Q: float,
    Z_Q: float,
    k0_frac: float,
    Ea: float,
    cal_b: float,
    n_soc: float,
    R_GAS: float,
    T_REF_K: float,
    initial_fec: float,
    initial_cal_seconds: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parallel Numba kernel for batch energy balance with degradation (Monte Carlo).

    Each scenario gets independent degradation tracking.

    Args:
        pv_scenarios: (n_scenarios, n_steps) PV energy
        load_scenarios: (n_scenarios, n_steps) Load energy
        temperature_c: (n_steps,) Temperature (shared across scenarios)
        ... (same parameters as energy_balance_kernel_with_degradation)

    Returns:
        Tuple of (total_import, total_sell, grid_independence, final_soh)
    """
    n_scenarios = pv_scenarios.shape[0]
    n_steps = pv_scenarios.shape[1]

    total_import = np.zeros(n_scenarios)
    total_sell = np.zeros(n_scenarios)
    grid_independence = np.zeros(n_scenarios)
    final_soh = np.zeros(n_scenarios)

    for s in prange(n_scenarios):
        soh = initial_soh
        fec_cum = initial_fec
        cal_seconds = initial_cal_seconds

        usable_cap = battery_nominal_wh * soh
        Emax = usable_cap * max_soc
        Emin = usable_cap * min_soc
        battery_E = Emax

        imp = 0.0
        sell = 0.0
        load_sum = 0.0

        day_soc = np.zeros(steps_per_day)
        day_idx = 0
        day_temp_sum = 0.0

        for i in range(n_steps):
            pv = pv_scenarios[s, i]
            load = load_scenarios[s, i]
            surplus = pv - load
            load_sum += load

            f_cap = _lfp_capacity_factor_numba(temperature_c[i])
            Emax_step = usable_cap * max_soc * f_cap
            Emin_step = usable_cap * min_soc * f_cap

            standby = standby_loss_wh * hours_per_step
            battery_E = min(Emax_step, max(Emin_step, battery_E - standby))

            if surplus >= 0:
                charge_room = Emax_step - battery_E
                charge_in = min(surplus, charge_room / charge_efficiency)
                battery_E += charge_in * charge_efficiency
                sell += surplus - charge_in
            else:
                deficit = -surplus
                available = battery_E - Emin_step
                if available > 0:
                    draw = min(available, deficit / discharge_efficiency)
                    battery_E -= draw
                    delivered = draw * discharge_efficiency
                    imp += max(0.0, deficit - delivered)
                else:
                    imp += deficit

            soc_abs_val = battery_E / usable_cap if usable_cap > 0 else 0.0
            soc_abs_val = max(0.0, min(1.0, soc_abs_val))

            day_soc[day_idx] = soc_abs_val
            day_temp_sum += temperature_c[i]
            day_idx += 1

            if day_idx >= steps_per_day:
                mean_temp = day_temp_sum / steps_per_day

                soh, fec_cum, cal_seconds, _, _ = _daily_degradation_step(
                    day_soc,
                    battery_nominal_wh,
                    fec_cum,
                    cal_seconds,
                    soh,
                    mean_temp,
                    hours_per_step,
                    A_Q,
                    B_Q,
                    C_DOC_Q,
                    D_DOC_Q,
                    Z_Q,
                    k0_frac,
                    Ea,
                    cal_b,
                    n_soc,
                    R_GAS,
                    T_REF_K,
                )

                usable_cap = battery_nominal_wh * soh
                Emax = usable_cap * max_soc
                Emin = usable_cap * min_soc
                f_cap = _lfp_capacity_factor_numba(temperature_c[i])
                battery_E = min(battery_E, Emax * f_cap)
                battery_E = max(battery_E, Emin * f_cap)

                day_idx = 0
                day_temp_sum = 0.0

        total_import[s] = imp
        total_sell[s] = sell
        final_soh[s] = soh

        if load_sum > 0:
            grid_independence[s] = 100.0 * (1.0 - imp / load_sum)
        else:
            grid_independence[s] = 100.0

    return total_import, total_sell, grid_independence, final_soh


def simulate_energy_balance_numba(ac_loss, houseload, battery_config, freq: str = "h", start_time=None, end_time=None):
    """
    Numba-accelerated energy balance simulation.

    This is a faster version of simulate_energy_balance() for cases where
    you don't need per-step degradation tracking.

    Args:
        ac_loss: PV AC power series (W)
        houseload: Load DataFrame (W)
        battery_config: BatteryConfig object
        freq: Time frequency ('h' or '15min')
        start_time: Simulation start
        end_time: Simulation end

    Returns:
        Tuple of (results_df, total_pv, summary_df)
    """
    import pandas as pd

    from breos.utils import get_hours_per_step

    hours_per_step = get_hours_per_step(freq)

    # Determine time range
    if start_time is None:
        start_time = ac_loss.index[0]
    if end_time is None:
        end_time = ac_loss.index[-1]

    rng = pd.date_range(start=start_time, end=end_time, freq=freq)

    # Prepare input arrays (convert W to Wh)
    ac_aligned = ac_loss.reindex(rng).fillna(0.0).values * hours_per_step
    load_aligned = houseload.iloc[:, 0].reindex(rng).fillna(0.0).values * hours_per_step

    # Run Numba kernel
    import_wh, sell_wh, battery_E, soc_norm, soc_abs = energy_balance_kernel(
        pv_energy_wh=ac_aligned,
        load_energy_wh=load_aligned,
        battery_nominal_wh=battery_config.nominal_energy_wh,
        max_soc=battery_config.max_soc,
        min_soc=battery_config.min_soc,
        charge_efficiency=battery_config.charge_efficiency,
        discharge_efficiency=battery_config.discharge_efficiency,
        standby_loss_wh=getattr(battery_config, "standby_loss_wh", 0.0),
        initial_soh=battery_config.initial_soh / 100.0,
        hours_per_step=hours_per_step,
    )

    # Build results DataFrame
    results_df = pd.DataFrame(
        {
            "Datetime": rng,
            "PV_Production": ac_aligned / hours_per_step,  # Back to W
            "Houseload": load_aligned / hours_per_step,
            "Import_From_Grid": import_wh / hours_per_step,
            "Sell_To_Grid": sell_wh / hours_per_step,
            "Battery_Energy": battery_E,
            "Battery_SOC_Normalized": soc_norm,
            "Battery_SOC_Absolute": soc_abs,
        }
    )

    # Summary
    total_pv = ac_aligned.sum()
    total_load = load_aligned.sum()
    total_import = import_wh.sum()
    total_sell = sell_wh.sum()

    pct_import = (total_import / total_load * 100) if total_load > 0 else 0

    summary = {
        "Total PV [kWh]": total_pv / 1000,
        "Total Load [kWh]": total_load / 1000,
        "Sell [kWh]": total_sell / 1000,
        "Import [kWh]": total_import / 1000,
        "Import [%]": pct_import,
        "Grid Independence [%]": 100 - pct_import,
    }
    summary_df = pd.DataFrame([summary])

    return results_df, total_pv, summary_df
