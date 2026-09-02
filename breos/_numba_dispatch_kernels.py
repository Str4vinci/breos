"""Module-scope Numba kernels for the within-day dispatch.

These live at module scope rather than inside a factory for one reason: Numba's
on-disk cache. Its index key for a function defined inside another function
includes the closure cell contents, and a cell holding a ``CPUDispatcher`` is
not stable across processes, so such a function misses the cache in every new
process and appends a fresh ``.nbc`` entry each time. ``_dispatch_day_kernel``
calls the three helpers below; at module scope it resolves them as globals,
which Numba keys by qualified name, and the cache works.

Importing this module requires Numba. :mod:`breos._numba_dispatch` must stay
importable without it -- that is where the availability check and the friendly
error live -- so this module is imported lazily from there, never at its top.

The kernels are a statement-by-statement mirror of ``_dispatch_day_python``.
Operation order and branch structure are part of the contract: the target is a
bit-identical result, so a change that is algebraically equivalent but
reassociates arithmetic is still a defect. ``fastmath=False`` is deliberate.
"""

from __future__ import annotations

import math  # noqa: F401  -- used inside the compiled kernels

import numpy as np
from numba import njit

from breos._numba_dispatch import (
    L_BATTERY_AC_TO_LOAD,
    L_BATTERY_AC_TO_LOAD_PV,
    L_BATTERY_CHARGE_INPUT,
    L_BATTERY_CHARGE_STORED,
    L_BATTERY_DISCHARGE_DC,
    L_BATTERY_ENERGY_DELTA,
    L_BATTERY_INVERTER_LOSS,
    L_CAPACITY_WINDOW_LOSS,
    L_INVERTER_LOSS,
    L_PV_AC_EXPORT,
    L_PV_AC_TO_LOAD,
    L_PV_DC_CURTAILED,
    L_PV_DC_TO_BATTERY,
    L_PV_DC_TO_INVERTER,
    L_PV_DIRECT_INVERTER_LOSS,
    L_PV_ORIGIN_BATTERY_AC_TO_LOAD,
    L_REPLACEMENT_ENERGY_ADDED,
    L_REPLACEMENT_ENERGY_REMOVED,
    L_STANDBY_LOSS,
    LFP_CAP_DERATE_PER_C_COLD,
    LFP_CAP_DERATE_PER_C_MODERATE,
    PVWATTS_CURVE_CONSTANT,
    PVWATTS_CURVE_LINEAR,
    PVWATTS_CURVE_QUADRATIC,
    PVWATTS_REFERENCE_EFFICIENCY,
    R_BATTERY_ENERGY,
    R_BATTERY_ENERGY_BEGIN,
    R_CHARGE_LOSS,
    R_DISCHARGE_LOSS,
    R_GRID_EXPORT,
    R_GRID_IMPORT,
    R_LOAD,
    R_PV_CURTAILMENT,
    R_PV_DC,
    R_PV_DELTA,
    R_PV_ORIGIN_BEGIN,
    R_PV_ORIGIN_END,
    R_PV_PRODUCTION,
    R_SOC_ABSOLUTE,
    R_SOC_NORMALIZED,
    R_SOH,
    R_STANDBY_LOSS,
    R_T_CELL,
)


@njit(cache=True, fastmath=False, nogil=True)
def _lfp_capacity_factor(t_c):
    if t_c >= 25.0:
        return 1.0
    elif t_c >= 0.0:
        return 1.0 - LFP_CAP_DERATE_PER_C_MODERATE * (25.0 - t_c)
    else:
        base_at_zero = 1.0 - LFP_CAP_DERATE_PER_C_MODERATE * 25.0
        return max(0.5, base_at_zero - LFP_CAP_DERATE_PER_C_COLD * abs(t_c))


@njit(cache=True, fastmath=False, nogil=True)
def _dc_ac(pv_dc_power, inverter_ac_power, inverter_efficiency, pow_two, ac_output_scale):
    """Mirror of ``calculate_dc_ac_power``: (ac, conversion_loss, clipping_dc).

    ``pow_two`` carries the literal 2.0 in from the caller. CPython
    evaluates ``zeta ** 2`` as a libm ``pow`` call, and for some inputs
    glibc's ``pow`` differs by one ULP from the correctly rounded square.
    With a constant exponent LLVM rewrites the call to ``zeta * zeta`` and
    picks up that one ULP; keeping the exponent opaque until run time
    forces the same libm call the reference makes. This is load-bearing
    for bit identity, not a stylistic choice.
    """
    pv_dc_power = max(0.0, pv_dc_power)
    inverter_ac_power = max(0.0, inverter_ac_power)
    inverter_efficiency = min(1.0, max(0.0, inverter_efficiency))
    ac_output_scale = max(0.0, ac_output_scale)

    if inverter_efficiency <= 0.0 or inverter_ac_power <= 0.0:
        return 0.0, 0.0, pv_dc_power
    if pv_dc_power <= 0.0:
        return 0.0, 0.0, 0.0
    if not np.isfinite(inverter_ac_power):
        ac_power = pv_dc_power * inverter_efficiency * ac_output_scale
        return ac_power, max(0.0, pv_dc_power - ac_power), 0.0

    pdc0 = inverter_ac_power / inverter_efficiency
    dc_used = min(pv_dc_power, pdc0)
    zeta = dc_used / pdc0
    ac_power = max(
        0.0,
        min(
            min(dc_used, inverter_ac_power),
            (inverter_efficiency / PVWATTS_REFERENCE_EFFICIENCY)
            * pdc0
            * (
                PVWATTS_CURVE_QUADRATIC * math.pow(zeta, pow_two) + PVWATTS_CURVE_LINEAR * zeta + PVWATTS_CURVE_CONSTANT
            ),
        ),
    )
    ac_power *= ac_output_scale
    clipping_loss_dc = max(0.0, pv_dc_power - dc_used)
    conversion_loss = max(0.0, dc_used - ac_power)
    return ac_power, conversion_loss, clipping_loss_dc


@njit(cache=True, fastmath=False, nogil=True)
def _dc_for_ac(ac_power_w, inverter_ac_power, inverter_efficiency, ac_output_scale):
    """Mirror of ``dc_power_for_ac_output``."""
    ac_output_scale = max(0.0, ac_output_scale)
    if ac_output_scale <= 0.0:
        return 0.0
    ac_power_w = ac_power_w / ac_output_scale
    ac_target = max(0.0, min(ac_power_w, max(0.0, inverter_ac_power)))
    inverter_ac_power = max(0.0, inverter_ac_power)
    inverter_efficiency = min(1.0, max(0.0, inverter_efficiency))
    if ac_target <= 0.0 or inverter_ac_power <= 0.0 or inverter_efficiency <= 0.0:
        return 0.0
    if not np.isfinite(inverter_ac_power):
        return ac_target / inverter_efficiency

    upper = inverter_ac_power / inverter_efficiency
    if ac_target >= inverter_ac_power:
        return upper

    normalized_ac = ac_target * PVWATTS_REFERENCE_EFFICIENCY / inverter_ac_power
    a = -PVWATTS_CURVE_QUADRATIC
    b = -PVWATTS_CURVE_LINEAR
    c = normalized_ac - PVWATTS_CURVE_CONSTANT
    discriminant = max(0.0, b * b - 4.0 * a * c)
    zeta = (-b - np.sqrt(discriminant)) / (2.0 * a)
    return min(upper, max(ac_target, zeta * upper))


@njit(cache=True, fastmath=False, nogil=True)
def _dispatch_day_kernel(
    matrix,
    pv_dc_vals,
    load_vals,
    temp_vals,
    lo,
    hi,
    battery_energy,
    pv_origin,
    has_battery,
    nominal_energy_wh,
    soh_fraction,
    soh_percent,
    max_soc,
    min_soc,
    standby_loss_per_step_wh,
    eff_charge,
    eff_discharge,
    inv_eff,
    cap_charge_in_wh,
    cap_discharge_ac_wh,
    inv_cap_ac_wh,
    cap_wh_is_infinite,
    thermal_resistance_kw,
    hours_per_step,
    pow_two,
    ac_output_scale,
):
    t_cell_day_sum = 0.0
    battery_energy_beginning = 0.0

    for i in range(lo, hi):
        pv_dc_power = max(0.0, pv_dc_vals[i] * hours_per_step)
        load = load_vals[i] * hours_per_step
        t_ambient = temp_vals[i]
        t_cell = t_ambient

        battery_energy_beginning = battery_energy if has_battery else 0.0
        pv_origin_beginning = pv_origin if has_battery else 0.0
        capacity_window_loss = 0.0
        battery_standby_loss = 0.0

        if has_battery:
            # Inlined _apply_capacity_window.
            usable_cap = nominal_energy_wh * soh_fraction
            f_cap = _lfp_capacity_factor(t_cell)
            emax = usable_cap * max_soc * f_cap
            emin = usable_cap * min_soc * f_cap

            capacity_window_loss = max(0.0, battery_energy - emax)
            if capacity_window_loss > 0.0 and battery_energy > 0.0:
                pv_origin *= emax / battery_energy
                battery_energy = emax

            removable_for_standby = max(0.0, battery_energy - emin)
            battery_standby_loss = min(standby_loss_per_step_wh, removable_for_standby)
            if battery_standby_loss > 0.0 and battery_energy > 0.0:
                pv_origin *= (battery_energy - battery_standby_loss) / battery_energy
                battery_energy -= battery_standby_loss
        else:
            emax = 0.0
            emin = 0.0

        energy_before_dispatch = battery_energy
        origin_before_dispatch = pv_origin
        if energy_before_dispatch > 0.0:
            origin_fraction = min(1.0, max(0.0, origin_before_dispatch / energy_before_dispatch))
        else:
            origin_fraction = 0.0

        # Inlined _dispatch_dc_step. The ledger is a set of locals here;
        # every entry is initialised to zero exactly as the dict is.
        lg_pv_dc_to_battery = 0.0
        lg_pv_dc_to_inverter = 0.0
        lg_pv_dc_curtailed = 0.0
        lg_pv_ac_to_load = 0.0
        lg_pv_ac_export = 0.0
        lg_battery_charge_input = 0.0
        lg_battery_discharge_dc = 0.0
        lg_battery_ac_to_load = 0.0
        lg_battery_charge_loss = 0.0
        lg_battery_discharge_loss = 0.0
        lg_pv_direct_inverter_loss = 0.0
        lg_battery_inverter_loss = 0.0
        lg_grid_import = 0.0

        pv_ac_max, pv_conv_loss, pv_clip_dc = _dc_ac(pv_dc_power, inv_cap_ac_wh, inv_eff, pow_two, ac_output_scale)

        if has_battery and pv_ac_max >= load:
            lg_pv_ac_to_load = load
            dc_to_load = _dc_for_ac(load, inv_cap_ac_wh, inv_eff, ac_output_scale)
            surplus_dc = max(0.0, pv_dc_power - dc_to_load)

            # Inlined charge().
            drawn = 0.0
            room = max(0.0, emax - battery_energy)
            if not (room <= 0.0 or eff_charge <= 0.0):
                drawn = min(min(surplus_dc, room / eff_charge), cap_charge_in_wh)
                battery_energy += drawn * eff_charge
                lg_pv_dc_to_battery = drawn
                lg_battery_charge_input = drawn
                lg_battery_charge_loss = drawn * (1.0 - eff_charge)

            remaining_dc = surplus_dc - drawn
            direct_ac, direct_conv_loss, direct_clip_dc = _dc_ac(
                dc_to_load + remaining_dc, inv_cap_ac_wh, inv_eff, pow_two, ac_output_scale
            )
            export_ac = max(0.0, direct_ac - load)
            dc_export = max(0.0, dc_to_load + remaining_dc - direct_clip_dc - dc_to_load)
            lg_pv_ac_export = export_ac
            lg_pv_dc_to_inverter = dc_to_load + dc_export
            lg_pv_dc_curtailed = direct_clip_dc
            lg_pv_direct_inverter_loss = direct_conv_loss
        elif has_battery:
            lg_pv_ac_to_load = pv_ac_max
            lg_pv_dc_to_inverter = pv_dc_power - pv_clip_dc
            lg_pv_direct_inverter_loss = pv_conv_loss
            excess_dc = pv_clip_dc
            deficit = load - pv_ac_max
            if excess_dc > 1e-12:
                drawn = 0.0
                room = max(0.0, emax - battery_energy)
                if not (room <= 0.0 or eff_charge <= 0.0):
                    drawn = min(min(excess_dc, room / eff_charge), cap_charge_in_wh)
                    battery_energy += drawn * eff_charge
                    lg_pv_dc_to_battery = drawn
                    lg_battery_charge_input = drawn
                    lg_battery_charge_loss = drawn * (1.0 - eff_charge)
                lg_pv_dc_curtailed = excess_dc - drawn
                lg_grid_import = deficit
            else:
                available = max(0.0, battery_energy - emin)
                # AC correction is applied after the inverter curve and
                # nameplate limit, so the reachable AC ceiling is scaled.
                target_total_ac = min(load, inv_cap_ac_wh * ac_output_scale)
                if available > 0.0 and eff_discharge > 0.0 and target_total_ac > pv_ac_max:
                    total_dc_target = _dc_for_ac(target_total_ac, inv_cap_ac_wh, inv_eff, ac_output_scale)
                    battery_dc = min(available * eff_discharge, max(0.0, total_dc_target - pv_dc_power))

                    if np.isfinite(cap_discharge_ac_wh):
                        total_dc = pv_dc_power + battery_dc
                        conv_ac, _cl, _cd = _dc_ac(total_dc, inv_cap_ac_wh, inv_eff, pow_two, ac_output_scale)
                        if total_dc <= 0.0:
                            unconstrained_battery_ac = 0.0
                        else:
                            unconstrained_battery_ac = conv_ac * battery_dc / total_dc
                        if unconstrained_battery_ac > cap_discharge_ac_wh:
                            lower = 0.0
                            upper = battery_dc
                            for _ in range(40):
                                midpoint = (lower + upper) / 2.0
                                mid_total_dc = pv_dc_power + midpoint
                                mid_ac, _mcl, _mcd = _dc_ac(
                                    mid_total_dc, inv_cap_ac_wh, inv_eff, pow_two, ac_output_scale
                                )
                                if mid_total_dc <= 0.0:
                                    midpoint_battery_ac = 0.0
                                else:
                                    midpoint_battery_ac = mid_ac * midpoint / mid_total_dc
                                if midpoint_battery_ac < cap_discharge_ac_wh:
                                    lower = midpoint
                                else:
                                    upper = midpoint
                            battery_dc = upper

                    total_dc = pv_dc_power + battery_dc
                    conv_ac, conv_loss, _cd2 = _dc_ac(total_dc, inv_cap_ac_wh, inv_eff, pow_two, ac_output_scale)
                    if total_dc <= 0.0:
                        pv_delivered_ac = 0.0
                        delivered_ac = 0.0
                        total_inverter_loss = 0.0
                    else:
                        delivered_ac = conv_ac * battery_dc / total_dc
                        pv_delivered_ac = conv_ac - delivered_ac
                        total_inverter_loss = conv_loss

                    draw = battery_dc / eff_discharge
                    battery_energy -= draw
                    total_inverter_dc = pv_dc_power + battery_dc
                    if total_inverter_dc > 0.0:
                        battery_inverter_loss = total_inverter_loss * battery_dc / total_inverter_dc
                    else:
                        battery_inverter_loss = 0.0
                    lg_battery_discharge_dc = draw
                    lg_battery_ac_to_load = delivered_ac
                    lg_battery_discharge_loss = draw - battery_dc
                    lg_battery_inverter_loss = battery_inverter_loss
                    lg_pv_ac_to_load = pv_delivered_ac
                    lg_pv_direct_inverter_loss = total_inverter_loss - battery_inverter_loss
                    lg_grid_import = max(0.0, load - pv_delivered_ac - delivered_ac)
                else:
                    lg_grid_import = deficit
        else:
            usable_ac = pv_ac_max
            lg_pv_ac_to_load = min(usable_ac, load)
            lg_pv_ac_export = usable_ac - lg_pv_ac_to_load
            lg_pv_dc_to_inverter = pv_dc_power - pv_clip_dc
            lg_pv_dc_curtailed = pv_clip_dc
            lg_pv_direct_inverter_loss = pv_conv_loss
            lg_grid_import = max(0.0, load - lg_pv_ac_to_load)

        charge_stored = lg_battery_charge_input * eff_charge
        pv_origin_discharge_dc = lg_battery_discharge_dc * origin_fraction
        pv_origin_battery_ac = lg_battery_ac_to_load * origin_fraction
        pv_origin = max(0.0, origin_before_dispatch - pv_origin_discharge_dc + charge_stored)
        pv_origin = min(pv_origin, battery_energy)

        if cap_wh_is_infinite:
            # Match the scalar unlimited-inverter compatibility path,
            # including the post-inverter AC correction.
            pv_production = (pv_dc_power - lg_pv_dc_curtailed) * inv_eff * ac_output_scale
        else:
            pv_production = pv_dc_power - lg_pv_dc_curtailed - lg_pv_direct_inverter_loss
        battery_energy_delta = battery_energy - battery_energy_beginning

        if has_battery and thermal_resistance_kw > 0:
            charge_power_w = lg_battery_charge_input / hours_per_step if hours_per_step > 0 else 0.0
            discharge_power_w = lg_battery_discharge_dc / hours_per_step if hours_per_step > 0 else 0.0
            p_loss_charge = charge_power_w * (1.0 - eff_charge)
            p_loss_discharge = discharge_power_w * (1.0 - eff_discharge)
            t_cell = t_ambient + thermal_resistance_kw * (p_loss_charge + p_loss_discharge)
        t_cell_day_sum += t_cell

        if has_battery:
            soc_normalized = (battery_energy - emin) / (emax - emin) if (emax - emin) > 0 else 0.0
            soc_normalized = max(0.0, min(1.0, soc_normalized))
            soc_absolute = battery_energy / (nominal_energy_wh * soh_fraction) if soh_fraction > 0 else 0.0
            soc_absolute = max(0.0, min(1.0, soc_absolute))
        else:
            soc_normalized = 0.0
            soc_absolute = 0.0

        matrix[R_PV_DC, i] = pv_dc_power / hours_per_step
        matrix[R_PV_PRODUCTION, i] = pv_production / hours_per_step
        matrix[R_LOAD, i] = load / hours_per_step
        matrix[R_PV_DELTA, i] = (pv_production - load) / hours_per_step
        matrix[R_GRID_IMPORT, i] = lg_grid_import / hours_per_step
        matrix[R_GRID_EXPORT, i] = lg_pv_ac_export / hours_per_step
        matrix[R_BATTERY_ENERGY, i] = battery_energy if has_battery else 0.0
        matrix[R_SOC_NORMALIZED, i] = soc_normalized
        matrix[R_SOC_ABSOLUTE, i] = soc_absolute
        matrix[R_SOH, i] = soh_percent if has_battery else 100.0
        matrix[R_T_CELL, i] = t_cell
        matrix[R_PV_CURTAILMENT, i] = lg_pv_dc_curtailed / hours_per_step
        matrix[R_CHARGE_LOSS, i] = lg_battery_charge_loss / hours_per_step
        matrix[R_DISCHARGE_LOSS, i] = lg_battery_discharge_loss / hours_per_step
        matrix[R_STANDBY_LOSS, i] = battery_standby_loss / hours_per_step
        matrix[R_BATTERY_ENERGY_BEGIN, i] = battery_energy_beginning
        matrix[R_PV_ORIGIN_BEGIN, i] = pv_origin_beginning
        matrix[R_PV_ORIGIN_END, i] = pv_origin

        matrix[L_PV_DC_TO_BATTERY, i] = lg_pv_dc_to_battery / hours_per_step
        matrix[L_PV_DC_TO_INVERTER, i] = lg_pv_dc_to_inverter / hours_per_step
        matrix[L_PV_DC_CURTAILED, i] = lg_pv_dc_curtailed / hours_per_step
        matrix[L_PV_AC_TO_LOAD, i] = lg_pv_ac_to_load / hours_per_step
        matrix[L_PV_AC_EXPORT, i] = lg_pv_ac_export / hours_per_step
        matrix[L_BATTERY_CHARGE_INPUT, i] = lg_battery_charge_input / hours_per_step
        matrix[L_BATTERY_CHARGE_STORED, i] = charge_stored / hours_per_step
        matrix[L_BATTERY_DISCHARGE_DC, i] = lg_battery_discharge_dc / hours_per_step
        matrix[L_BATTERY_AC_TO_LOAD, i] = lg_battery_ac_to_load / hours_per_step
        matrix[L_BATTERY_AC_TO_LOAD_PV, i] = pv_origin_battery_ac / hours_per_step
        matrix[L_PV_ORIGIN_BATTERY_AC_TO_LOAD, i] = pv_origin_battery_ac / hours_per_step
        matrix[L_PV_DIRECT_INVERTER_LOSS, i] = lg_pv_direct_inverter_loss / hours_per_step
        matrix[L_BATTERY_INVERTER_LOSS, i] = lg_battery_inverter_loss / hours_per_step
        matrix[L_INVERTER_LOSS, i] = (lg_pv_direct_inverter_loss + lg_battery_inverter_loss) / hours_per_step
        matrix[L_STANDBY_LOSS, i] = battery_standby_loss / hours_per_step
        matrix[L_CAPACITY_WINDOW_LOSS, i] = capacity_window_loss / hours_per_step
        matrix[L_REPLACEMENT_ENERGY_REMOVED, i] = 0.0
        matrix[L_REPLACEMENT_ENERGY_ADDED, i] = 0.0
        matrix[L_BATTERY_ENERGY_DELTA, i] = battery_energy_delta / hours_per_step

    return battery_energy, pv_origin, t_cell_day_sum, battery_energy_beginning
