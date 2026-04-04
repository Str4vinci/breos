"""
Numba-accelerated kernels for performance-critical operations.

This module provides JIT-compiled versions of hot loops for:
- Energy balance simulation
- SOC tracking
- Degradation updates
"""

from typing import Tuple

import numpy as np
from numba import jit, prange


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

    # Daily aggregate cycle metrics
    soc_max = soc_day[0]
    soc_min = soc_day[0]
    energy_throughput = 0.0

    for i in range(1, n):
        if soc_day[i] > soc_max:
            soc_max = soc_day[i]
        if soc_day[i] < soc_min:
            soc_min = soc_day[i]
        energy_throughput += abs(soc_day[i] - soc_day[i - 1]) * battery_nominal_wh

    DOC = soc_max - soc_min
    dFEC = energy_throughput / battery_nominal_wh if battery_nominal_wh > 0 else 0.0

    # Cycle degradation (Naumann Eq. 5-6)
    dSOH_cycle = 0.0
    if DOC > 0.01 and dFEC > 1e-6:
        # Mean C-rate estimate from total throughput and active hours
        total_hours = n * hours_per_step
        mean_c_rate = (energy_throughput / battery_nominal_wh) / total_hours if total_hours > 0 else 0.0

        kC = A_Q * mean_c_rate + B_Q
        if kC < 0.0:
            kC = 0.0
        kDOC = C_DOC_Q * ((DOC - 0.6) ** 3) + D_DOC_Q
        if kDOC < 0.0:
            kDOC = 0.0

        fec_new = fec_cum + dFEC
        if fec_cum > 0:
            dq_pct = kC * kDOC * (fec_new**Z_Q - fec_cum**Z_Q)
        else:
            dq_pct = kC * kDOC * (fec_new**Z_Q)
        dSOH_cycle = dq_pct / 100.0
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

        standby = standby_loss_wh * hours_per_step
        battery_E = max(Emin, battery_E - standby)

        if surplus >= 0:
            charge_room = Emax - battery_E
            charge_in = min(surplus, charge_room / charge_efficiency)
            battery_E += charge_in * charge_efficiency
            sell_wh[i] = surplus - charge_in
        else:
            deficit = -surplus
            available = battery_E - Emin
            if available > 0:
                draw = min(available, deficit / discharge_efficiency)
                battery_E -= draw
                delivered = draw * discharge_efficiency
                import_wh[i] = max(0.0, deficit - delivered)
            else:
                import_wh[i] = deficit

        battery_energy_wh[i] = battery_E

        if (Emax - Emin) > 0:
            soc_normalized[i] = max(0.0, min(1.0, (battery_E - Emin) / (Emax - Emin)))
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

            # Update capacity limits for next day
            usable_cap = battery_nominal_wh * soh
            Emax = usable_cap * max_soc
            Emin = usable_cap * min_soc
            battery_E = min(battery_E, Emax)
            battery_E = max(battery_E, Emin)

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
                battery_E = min(battery_E, Emax)
                battery_E = max(battery_E, Emin)

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


# =========================================================================
# Combined electrical + thermal kernels (TES + Heat Pump)
# =========================================================================


@jit(nopython=True, cache=True)
def combined_energy_balance_kernel(
    pv_energy_wh: np.ndarray,
    elec_load_wh: np.ndarray,
    thermal_demand_w: np.ndarray,
    temperature_c: np.ndarray,
    # Battery params
    battery_nominal_wh: float,
    max_soc: float,
    min_soc: float,
    charge_efficiency: float,
    discharge_efficiency: float,
    standby_loss_wh: float,
    initial_soh: float,
    inverter_efficiency: float,
    # TES params
    tes_nominal_wh_th: float,
    tes_max_soc: float,
    tes_min_soc: float,
    tes_charge_eff: float,
    tes_discharge_eff: float,
    tes_standby_loss_frac: float,
    # HP params
    hp_rated_kw_th: float,
    hp_sink_temp_c: float,
    hp_carnot_eff: float,
    hp_min_source_temp_c: float,
    # Simulation params
    hours_per_step: float,
    steps_per_day: int,
    # Degradation params
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, float, float, float]:
    """
    Combined electrical + thermal energy balance with battery degradation.

    Thermal priority dispatch: TES discharges first, HP covers remainder,
    PV surplus charges TES via HP before charging battery.

    Returns:
        Tuple of:
        - import_wh: Grid import per step (Wh)
        - sell_wh: Grid export per step (Wh)
        - battery_soc: Battery SOC normalized per step
        - tes_soc: TES SOC per step (fraction)
        - hp_elec_w: HP electrical consumption per step (W)
        - final_soh: Final battery SOH (fraction)
        - final_fec: Final cumulative FEC
        - final_cal_seconds: Final cumulative calendar seconds
        - total_thermal_unmet: Total unmet thermal demand (Wh)
        - total_hp_thermal: Total HP thermal output (Wh)
        - total_tes_discharge: Total TES discharge (Wh)
    """
    n_steps = len(pv_energy_wh)

    import_wh_arr = np.zeros(n_steps)
    sell_wh_arr = np.zeros(n_steps)
    battery_soc_arr = np.zeros(n_steps)
    tes_soc_arr = np.zeros(n_steps)
    hp_elec_arr = np.zeros(n_steps)

    # Battery state
    soh = initial_soh
    fec_cum = initial_fec
    cal_seconds = initial_cal_seconds
    usable_cap = battery_nominal_wh * soh
    batt_Emax = usable_cap * max_soc
    batt_Emin = usable_cap * min_soc
    batt_E = batt_Emax

    # TES state
    tes_Emax = tes_nominal_wh_th * tes_max_soc
    tes_Emin = tes_nominal_wh_th * tes_min_soc
    tes_E = tes_nominal_wh_th * 0.5  # Start at 50%

    # Degradation tracking
    day_soc = np.zeros(steps_per_day)
    day_idx = 0
    day_temp_sum = 0.0

    # Accumulators
    total_thermal_unmet = 0.0
    total_hp_thermal = 0.0
    total_tes_discharge = 0.0

    hp_rated_w_th = hp_rated_kw_th * 1000.0

    for i in range(n_steps):
        pv_wh = pv_energy_wh[i]
        base_load_wh = elec_load_wh[i]
        therm_w = thermal_demand_w[i]
        T_amb = temperature_c[i]

        # COP calculation (Carnot-based)
        cop = 1.0
        if T_amb >= hp_min_source_temp_c:
            T_src_K = T_amb + 273.15
            T_sink_K = hp_sink_temp_c + 273.15
            dT = T_sink_K - T_src_K
            if dT > 0:
                cop = hp_carnot_eff * T_sink_K / dT
                if cop < 1.0:
                    cop = 1.0
                if cop > 10.0:
                    cop = 10.0
            else:
                cop = 10.0

        # TES standby loss
        tes_loss = tes_E * tes_standby_loss_frac * hours_per_step
        tes_E = max(tes_Emin, tes_E - tes_loss)

        # Discharge TES to meet thermal demand
        thermal_from_tes = 0.0
        remaining_therm = therm_w
        if therm_w > 0 and tes_E > tes_Emin:
            demand_wh = therm_w * hours_per_step
            avail_wh = (tes_E - tes_Emin) * tes_discharge_eff
            if avail_wh >= demand_wh:
                withdrawn = demand_wh / tes_discharge_eff
                tes_E -= withdrawn
                thermal_from_tes = therm_w
                remaining_therm = 0.0
            else:
                withdrawn = tes_E - tes_Emin
                delivered = withdrawn * tes_discharge_eff
                tes_E = tes_Emin
                thermal_from_tes = delivered / hours_per_step if hours_per_step > 0 else 0.0
                remaining_therm = therm_w - thermal_from_tes

        total_tes_discharge += thermal_from_tes * hours_per_step

        # HP covers remaining thermal
        hp_thermal = 0.0
        hp_elec = 0.0
        thermal_unmet = 0.0
        if remaining_therm > 0 and hp_rated_w_th > 0:
            hp_thermal = min(remaining_therm, hp_rated_w_th)
            hp_elec = hp_thermal / cop
            thermal_unmet = remaining_therm - hp_thermal
        else:
            thermal_unmet = remaining_therm

        total_thermal_unmet += thermal_unmet * hours_per_step
        total_hp_thermal += hp_thermal * hours_per_step
        hp_elec_arr[i] = hp_elec

        # Total electrical load
        total_load_wh = base_load_wh + hp_elec * hours_per_step

        # Battery dispatch
        batt_standby = standby_loss_wh * hours_per_step
        batt_E = max(batt_Emin, batt_E - batt_standby)

        pv_ac_wh = pv_wh * inverter_efficiency

        if pv_ac_wh >= total_load_wh:
            # Surplus
            dc_to_load = total_load_wh / inverter_efficiency if inverter_efficiency > 0 else total_load_wh
            surplus_dc = pv_wh - dc_to_load

            # Thermal priority: charge TES with surplus via HP
            if tes_nominal_wh_th > 0:
                surplus_ac = surplus_dc * inverter_efficiency
                hp_tes_thermal = surplus_ac * cop
                if hp_tes_thermal > hp_rated_w_th * hours_per_step:
                    hp_tes_thermal = hp_rated_w_th * hours_per_step
                tes_room = (tes_Emax - tes_E) / tes_charge_eff
                tes_charge = min(hp_tes_thermal, tes_room)
                if tes_charge > 0:
                    tes_E += tes_charge * tes_charge_eff
                    elec_used = tes_charge / cop
                    dc_used = elec_used / inverter_efficiency if inverter_efficiency > 0 else elec_used
                    surplus_dc -= dc_used
                    if surplus_dc < 0:
                        surplus_dc = 0.0

            # Charge battery
            charge_room = batt_Emax - batt_E
            charge_in = min(surplus_dc, charge_room / charge_efficiency)
            batt_E += charge_in * charge_efficiency
            remaining_dc = surplus_dc - charge_in
            sell_wh_arr[i] = remaining_dc * inverter_efficiency
        else:
            # Deficit
            deficit = total_load_wh - pv_ac_wh
            available = batt_E - batt_Emin
            if available > 0:
                eff_total = discharge_efficiency * inverter_efficiency
                dc_needed = deficit / eff_total if eff_total > 0 else deficit
                draw = min(available, dc_needed)
                batt_E -= draw
                delivered = draw * eff_total
                import_wh_arr[i] = max(0.0, deficit - delivered)
            else:
                import_wh_arr[i] = deficit

        # SOC
        if (batt_Emax - batt_Emin) > 0:
            battery_soc_arr[i] = max(0.0, min(1.0, (batt_E - batt_Emin) / (batt_Emax - batt_Emin)))

        soc_abs = batt_E / usable_cap if usable_cap > 0 else 0.0
        soc_abs = max(0.0, min(1.0, soc_abs))

        tes_soc_arr[i] = tes_E / tes_nominal_wh_th if tes_nominal_wh_th > 0 else 0.0

        # Daily degradation
        day_soc[day_idx] = soc_abs
        day_temp_sum += T_amb
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
            batt_Emax = usable_cap * max_soc
            batt_Emin = usable_cap * min_soc
            batt_E = min(batt_E, batt_Emax)
            batt_E = max(batt_E, batt_Emin)
            day_idx = 0
            day_temp_sum = 0.0

    return (
        import_wh_arr,
        sell_wh_arr,
        battery_soc_arr,
        tes_soc_arr,
        hp_elec_arr,
        soh,
        fec_cum,
        cal_seconds,
        total_thermal_unmet,
        total_hp_thermal,
        total_tes_discharge,
    )


@jit(nopython=True, cache=True, parallel=True)
def batch_combined_energy_balance_kernel(
    pv_scenarios: np.ndarray,
    load_scenarios: np.ndarray,
    thermal_demand_w: np.ndarray,
    temperature_c: np.ndarray,
    # Battery
    battery_nominal_wh: float,
    max_soc: float,
    min_soc: float,
    charge_efficiency: float,
    discharge_efficiency: float,
    standby_loss_wh: float,
    initial_soh: float,
    inverter_efficiency: float,
    # TES
    tes_nominal_wh_th: float,
    tes_max_soc: float,
    tes_min_soc: float,
    tes_charge_eff: float,
    tes_discharge_eff: float,
    tes_standby_loss_frac: float,
    # HP
    hp_rated_kw_th: float,
    hp_sink_temp_c: float,
    hp_carnot_eff: float,
    hp_min_source_temp_c: float,
    # Sim
    hours_per_step: float,
    steps_per_day: int,
    # Degradation
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parallel batch combined energy balance for optimization.

    Each scenario gets independent electrical+thermal simulation.

    Returns:
        Tuple of per-scenario arrays:
        - total_import: Total grid import (Wh)
        - total_sell: Total grid export (Wh)
        - elec_grid_independence: Electrical grid independence (%)
        - thermal_coverage: Thermal coverage (%)
        - final_soh: Final battery SOH (fraction)
    """
    n_scenarios = pv_scenarios.shape[0]
    n_steps = pv_scenarios.shape[1]

    out_import = np.zeros(n_scenarios)
    out_sell = np.zeros(n_scenarios)
    out_elec_gi = np.zeros(n_scenarios)
    out_thermal_cov = np.zeros(n_scenarios)
    out_soh = np.zeros(n_scenarios)

    hp_rated_w_th = hp_rated_kw_th * 1000.0

    for s in prange(n_scenarios):
        soh = initial_soh
        fec_cum = initial_fec
        cal_seconds = initial_cal_seconds

        usable_cap = battery_nominal_wh * soh
        batt_Emax = usable_cap * max_soc
        batt_Emin = usable_cap * min_soc
        batt_E = batt_Emax

        tes_Emax = tes_nominal_wh_th * tes_max_soc
        tes_Emin = tes_nominal_wh_th * tes_min_soc
        tes_E = tes_nominal_wh_th * 0.5

        imp_sum = 0.0
        sell_sum = 0.0
        load_sum = 0.0
        therm_demand_sum = 0.0
        therm_unmet_sum = 0.0

        day_soc = np.zeros(steps_per_day)
        day_idx = 0
        day_temp_sum = 0.0

        for i in range(n_steps):
            pv_wh = pv_scenarios[s, i]
            base_wh = load_scenarios[s, i]
            therm_w = thermal_demand_w[i]
            T_amb = temperature_c[i]
            load_sum += base_wh

            # COP
            cop = 1.0
            if T_amb >= hp_min_source_temp_c:
                T_src_K = T_amb + 273.15
                T_sink_K = hp_sink_temp_c + 273.15
                dT = T_sink_K - T_src_K
                if dT > 0:
                    cop = hp_carnot_eff * T_sink_K / dT
                    cop = max(1.0, min(10.0, cop))
                else:
                    cop = 10.0

            # TES standby
            tes_E = max(tes_Emin, tes_E - tes_E * tes_standby_loss_frac * hours_per_step)

            # TES discharge
            remaining_therm = therm_w
            therm_demand_sum += therm_w * hours_per_step
            if therm_w > 0 and tes_E > tes_Emin:
                d_wh = therm_w * hours_per_step
                avail = (tes_E - tes_Emin) * tes_discharge_eff
                if avail >= d_wh:
                    tes_E -= d_wh / tes_discharge_eff
                    remaining_therm = 0.0
                else:
                    tes_E = tes_Emin
                    remaining_therm = therm_w - avail / hours_per_step

            # HP
            hp_elec = 0.0
            if remaining_therm > 0 and hp_rated_w_th > 0:
                hp_th = min(remaining_therm, hp_rated_w_th)
                hp_elec = hp_th / cop
                therm_unmet_sum += (remaining_therm - hp_th) * hours_per_step
            else:
                therm_unmet_sum += remaining_therm * hours_per_step

            total_load_wh = base_wh + hp_elec * hours_per_step
            load_sum += hp_elec * hours_per_step

            # Battery dispatch
            batt_E = max(batt_Emin, batt_E - standby_loss_wh * hours_per_step)
            pv_ac = pv_wh * inverter_efficiency

            if pv_ac >= total_load_wh:
                dc_to_load = total_load_wh / inverter_efficiency if inverter_efficiency > 0 else total_load_wh
                surplus_dc = pv_wh - dc_to_load

                # TES charge
                if tes_nominal_wh_th > 0:
                    s_ac = surplus_dc * inverter_efficiency
                    tes_th = min(s_ac * cop, hp_rated_w_th * hours_per_step)
                    tes_room = (tes_Emax - tes_E) / tes_charge_eff
                    tc = min(tes_th, tes_room)
                    if tc > 0:
                        tes_E += tc * tes_charge_eff
                        dc_used = (tc / cop) / inverter_efficiency if inverter_efficiency > 0 else tc / cop
                        surplus_dc = max(0.0, surplus_dc - dc_used)

                cr = batt_Emax - batt_E
                ci = min(surplus_dc, cr / charge_efficiency)
                batt_E += ci * charge_efficiency
                sell_sum += (surplus_dc - ci) * inverter_efficiency
            else:
                deficit = total_load_wh - pv_ac
                avail = batt_E - batt_Emin
                if avail > 0:
                    eff = discharge_efficiency * inverter_efficiency
                    draw = min(avail, deficit / eff if eff > 0 else deficit)
                    batt_E -= draw
                    imp_sum += max(0.0, deficit - draw * eff)
                else:
                    imp_sum += deficit

            soc_abs = batt_E / usable_cap if usable_cap > 0 else 0.0
            soc_abs = max(0.0, min(1.0, soc_abs))

            day_soc[day_idx] = soc_abs
            day_temp_sum += T_amb
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
                batt_Emax = usable_cap * max_soc
                batt_Emin = usable_cap * min_soc
                batt_E = min(batt_E, batt_Emax)
                batt_E = max(batt_E, batt_Emin)
                day_idx = 0
                day_temp_sum = 0.0

        out_import[s] = imp_sum
        out_sell[s] = sell_sum
        out_soh[s] = soh
        out_elec_gi[s] = 100.0 * (1.0 - imp_sum / load_sum) if load_sum > 0 else 100.0
        out_thermal_cov[s] = 100.0 * (1.0 - therm_unmet_sum / therm_demand_sum) if therm_demand_sum > 0 else 100.0

    return out_import, out_sell, out_elec_gi, out_thermal_cov, out_soh


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
