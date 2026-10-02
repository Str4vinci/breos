"""The within-day dispatch step, written once for both execution backends.

The step is greedy self-consumption, gated by per-step instructions (ADR
0002): may the battery discharge, how much it keeps, and whether it charges
from the grid toward a target. No instructions is the greedy step exactly.

Everything here is scalar code over floats and one output matrix: no dicts, no
closures, no dataclasses. That is what lets :mod:`breos._numba_dispatch_kernels`
compile these same functions when the optional Numba backend is selected,
while the Python backend calls them as they stand. There is no second copy to
keep in step, so the backends cannot drift apart.

Operation order and branch structure are part of the contract: the target is a
bit-identical result on both backends, and the compiled side is built with
``fastmath=False`` so LLVM keeps the order written here.

This module must not import Numba. ``import breos`` reaches it through
:mod:`breos.battery`, and the Numba dependency is optional.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np

from breos.constants import DEFAULT_THERMAL_RESISTANCE_K_PER_W, LFP_CAP_DERATE_PER_C_COLD, LFP_CAP_DERATE_PER_C_MODERATE
from breos.dispatch_instructions import DispatchInstructions

if TYPE_CHECKING:
    from breos.battery import BatteryConfig

# PVWatts part-load curve. The inverter cores live here, beside the day loop,
# because Numba's cache is keyed on this file: an edit to them or to these
# constants must invalidate the compiled kernel.
PVWATTS_REFERENCE_EFFICIENCY = 0.9637
PVWATTS_CURVE_QUADRATIC = -0.0162
PVWATTS_CURVE_LINEAR = 0.9858
PVWATTS_CURVE_CONSTANT = -0.0059


def _dc_ac(
    pv_dc_power: float,
    inverter_ac_power: float,
    inverter_efficiency: float,
    ac_output_scale: float,
    pow_two: float,
) -> tuple[float, float, float]:
    """Scalar core of :func:`calculate_dc_ac_power`: ``(ac, conversion_loss, clipping_dc)``.

    Written without dataclasses or ``float()`` coercion so the Numba backend
    compiles this same function.

    ``pow_two`` carries the literal 2.0 in from a Python caller. CPython
    evaluates ``zeta ** 2`` as a libm ``pow`` call, and for some inputs
    glibc's ``pow`` differs by one ULP from the correctly rounded square.
    With a constant exponent LLVM rewrites the call to ``zeta * zeta`` and
    picks up that one ULP; keeping the exponent opaque until run time
    forces the same libm call on both backends. This is load-bearing for bit
    identity, not a stylistic choice.
    """
    pv_dc_power = max(0.0, pv_dc_power)
    inverter_ac_power = max(0.0, inverter_ac_power)
    inverter_efficiency = min(1.0, max(0.0, inverter_efficiency))
    ac_output_scale = min(1.0, max(0.0, ac_output_scale))

    if inverter_efficiency <= 0.0 or inverter_ac_power <= 0.0:
        return 0.0, 0.0, pv_dc_power
    if pv_dc_power <= 0.0:
        return 0.0, 0.0, 0.0

    # A lower-level BatteryConfig may intentionally omit the inverter
    # nameplate. With no rated power there is no part-load ratio to evaluate,
    # so retain the historical unbounded flat-efficiency behavior. App always
    # supplies its sized finite AC rating.
    if not math.isfinite(inverter_ac_power):
        ac_power = pv_dc_power * inverter_efficiency * ac_output_scale
        return ac_power, pv_dc_power - ac_power, 0.0

    # PVWatts defines pdc0 as the DC input at which the inverter reaches its
    # AC nameplate (pac0 = eta_inv_nom * pdc0). BREOS exposes the AC rating,
    # so derive the matching pdc0 here. This is the single conversion path
    # used by both the public solar helper and the App dispatch engine.
    pdc0 = inverter_ac_power / inverter_efficiency
    dc_used = min(pv_dc_power, pdc0)
    zeta = dc_used / pdc0
    ac_power = max(
        0.0,
        min(
            dc_used,
            inverter_ac_power,
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


def _dc_for_ac(
    ac_power_w: float,
    inverter_ac_power: float,
    inverter_efficiency: float,
    ac_output_scale: float,
) -> float:
    """Scalar core of :func:`dc_power_for_ac_output`, compiled as-is by the Numba backend."""
    ac_output_scale = min(1.0, max(0.0, ac_output_scale))
    if ac_output_scale <= 0.0:
        return 0.0
    ac_power_w = ac_power_w / ac_output_scale
    ac_target = max(0.0, min(ac_power_w, max(0.0, inverter_ac_power)))
    inverter_ac_power = max(0.0, inverter_ac_power)
    inverter_efficiency = min(1.0, max(0.0, inverter_efficiency))
    if ac_target <= 0.0 or inverter_ac_power <= 0.0 or inverter_efficiency <= 0.0:
        return 0.0
    if not math.isfinite(inverter_ac_power):
        return ac_target / inverter_efficiency

    upper = inverter_ac_power / inverter_efficiency
    if ac_target >= inverter_ac_power:
        return upper

    # Rearrange the PVWatts polynomial in zeta = pdc / pdc0 and take
    # the root on its monotonic operating interval (0 < zeta < 1).
    normalized_ac = ac_target * PVWATTS_REFERENCE_EFFICIENCY / inverter_ac_power
    a = -PVWATTS_CURVE_QUADRATIC
    b = -PVWATTS_CURVE_LINEAR
    c = normalized_ac - PVWATTS_CURVE_CONSTANT
    discriminant = max(0.0, b * b - 4.0 * a * c)
    zeta = (-b - math.sqrt(discriminant)) / (2.0 * a)
    # At unusually high nominal efficiencies the empirical PVWatts curve can
    # exceed 100% conversion efficiency around its peak. The forward helper
    # caps AC output at DC input to preserve energy conservation, so its
    # inverse must also request at least the target amount of DC.
    return min(upper, max(ac_target, zeta * upper))


# Version of the ledger: the per-step columns below and the App's PV loss
# waterfall built from them. Bump it with any change a consumer could see.
# App, Monte Carlo and SimulationSummary all report it.
# 1.1 adds the bifacial_rear_gain PV loss-waterfall stage, relabels the iam
# stage to name the front side explicitly, and adds the pv_model provenance
# block. All three are additive, so 1.0 consumers keep reading the fields they
# already knew.
# 1.2 removes the year_1_degradation loss-waterfall stage. PV module age is
# counted at the start of each year, so year 1 has no degradation and the stage
# was always 0; pvwatts_static is now the last stage.
# 2.0 splits stored energy into PV, grid and unattributed origins (ADR 0002
# A8, A9): grid-origin balances and per-origin charge, discharge, standby,
# capacity-window and replacement columns. It drops four columns that
# repeated another under a second name: Sell_To_Grid (read PV_AC_Export),
# PV_Curtailment (PV_DC_Curtailed), Battery_Standby_Loss (Standby_Loss) and
# Battery_AC_To_Load_PV (PV_Origin_Battery_AC_To_Load).
# 3.0 takes money out of the ledger (ADR 0003 E4): the per-step
# Replacement_Cost becomes Battery_Replaced_Capacity_Wh, the nominal capacity
# swapped in at that step, and the summary row's Replacement_Cost becomes
# Replaced_Capacity_kWh. The economics prices replacements.
LEDGER_SCHEMA_VERSION = "3.0"

# Per-step state columns, by results-frame name, in their row order inside the
# shared buffer matrix. Stored-energy columns are Wh; every other column is
# average W over the timestep.
_STATE_COLUMNS: Tuple[str, ...] = (
    "PV_DC",
    "PV_Production",
    "Houseload",
    "PV_Delta",
    "Import_From_Grid",
    "Battery_Energy",
    "Battery_SOC_Normalized",
    "Battery_SOC_Absolute",
    "Battery_SOH",
    "T_cell",
    "Battery_Charge_Loss",
    "Battery_Discharge_Loss",
    "Battery_Energy_Beginning",
    "Battery_PV_Origin_Energy_Beginning",
    "Battery_PV_Origin_Energy_End",
    "Battery_Grid_Origin_Energy_Beginning",
    "Battery_Grid_Origin_Energy_End",
)

# Explicit per-step energy flows and losses. Every entry is accumulated in Wh
# during the loop and divided by the step length on write, so the frame
# reports average W. Their rows follow the state rows.
_LEDGER_COLUMNS: Tuple[str, ...] = (
    "PV_DC_To_Battery",
    "PV_DC_To_Inverter",
    "PV_DC_Curtailed",
    "PV_AC_To_Load",
    "PV_AC_Export",
    "Battery_Charge_Input",
    "Battery_Charge_Stored",
    # Grid charging (ADR 0002 A6): AC imported for the battery, the DC it
    # becomes after the hybrid inverter's AC-to-DC conversion, and that
    # conversion's loss. The DC is part of Battery_Charge_Input.
    "Grid_AC_To_Battery",
    "Grid_DC_To_Battery",
    "Grid_Charge_Conversion_Loss",
    "Battery_Discharge_DC",
    "Battery_AC_To_Load",
    "PV_Direct_Inverter_Loss",
    "Battery_Inverter_Loss",
    "Inverter_Loss",
    "Standby_Loss",
    "Capacity_Window_Loss",
    "Battery_Replacement_Energy_Removed",
    "Battery_Replacement_Energy_Added",
    "Battery_Energy_Delta",
    # Stored energy by origin (ADR 0002 A8): PV, grid, and an unattributed
    # remainder that holds a fresh or replacement pack's initial energy. Each
    # flow into or out of the store is split here for PV and grid; the
    # unattributed share of a flow is its total minus those two. Every origin
    # then reconciles step by step from the ledger alone.
    "PV_Origin_Battery_Charge_Stored",
    "PV_Origin_Battery_Discharge_DC",
    "PV_Origin_Battery_AC_To_Load",
    "PV_Origin_Standby_Loss",
    "PV_Origin_Capacity_Window_Loss",
    "PV_Origin_Replacement_Energy_Removed",
    "Grid_Origin_Battery_Charge_Stored",
    "Grid_Origin_Battery_Discharge_DC",
    "Grid_Origin_Battery_AC_To_Load",
    "Grid_Origin_Standby_Loss",
    "Grid_Origin_Capacity_Window_Loss",
    "Grid_Origin_Replacement_Energy_Removed",
)

# Every row of the buffer matrix, by the one name it has everywhere: the
# day loop writes it, the buffers expose it and the results frame reports it
# under this name. The day loop addresses rows by the indices below, so the
# order is its contract.
_ROW_COLUMNS: Tuple[str, ...] = _STATE_COLUMNS + _LEDGER_COLUMNS
_ROW: Dict[str, int] = {name: row for row, name in enumerate(_ROW_COLUMNS)}
_N_ROWS: int = len(_ROW_COLUMNS)

# Row indices as plain module constants, which Numba freezes at compile time.
R_PV_DC = _ROW["PV_DC"]
R_PV_PRODUCTION = _ROW["PV_Production"]
R_LOAD = _ROW["Houseload"]
R_PV_DELTA = _ROW["PV_Delta"]
R_GRID_IMPORT = _ROW["Import_From_Grid"]
R_BATTERY_ENERGY = _ROW["Battery_Energy"]
R_SOC_NORMALIZED = _ROW["Battery_SOC_Normalized"]
R_SOC_ABSOLUTE = _ROW["Battery_SOC_Absolute"]
R_SOH = _ROW["Battery_SOH"]
R_T_CELL = _ROW["T_cell"]
R_CHARGE_LOSS = _ROW["Battery_Charge_Loss"]
R_DISCHARGE_LOSS = _ROW["Battery_Discharge_Loss"]
R_BATTERY_ENERGY_BEGIN = _ROW["Battery_Energy_Beginning"]
R_PV_ORIGIN_BEGIN = _ROW["Battery_PV_Origin_Energy_Beginning"]
R_PV_ORIGIN_END = _ROW["Battery_PV_Origin_Energy_End"]
R_GRID_ORIGIN_BEGIN = _ROW["Battery_Grid_Origin_Energy_Beginning"]
R_GRID_ORIGIN_END = _ROW["Battery_Grid_Origin_Energy_End"]

L_PV_DC_TO_BATTERY = _ROW["PV_DC_To_Battery"]
L_PV_DC_TO_INVERTER = _ROW["PV_DC_To_Inverter"]
L_PV_DC_CURTAILED = _ROW["PV_DC_Curtailed"]
L_PV_AC_TO_LOAD = _ROW["PV_AC_To_Load"]
L_PV_AC_EXPORT = _ROW["PV_AC_Export"]
L_BATTERY_CHARGE_INPUT = _ROW["Battery_Charge_Input"]
L_BATTERY_CHARGE_STORED = _ROW["Battery_Charge_Stored"]
L_GRID_AC_TO_BATTERY = _ROW["Grid_AC_To_Battery"]
L_GRID_DC_TO_BATTERY = _ROW["Grid_DC_To_Battery"]
L_GRID_CHARGE_CONVERSION_LOSS = _ROW["Grid_Charge_Conversion_Loss"]
L_BATTERY_DISCHARGE_DC = _ROW["Battery_Discharge_DC"]
L_BATTERY_AC_TO_LOAD = _ROW["Battery_AC_To_Load"]
L_PV_DIRECT_INVERTER_LOSS = _ROW["PV_Direct_Inverter_Loss"]
L_BATTERY_INVERTER_LOSS = _ROW["Battery_Inverter_Loss"]
L_INVERTER_LOSS = _ROW["Inverter_Loss"]
L_STANDBY_LOSS = _ROW["Standby_Loss"]
L_CAPACITY_WINDOW_LOSS = _ROW["Capacity_Window_Loss"]
L_REPLACEMENT_ENERGY_REMOVED = _ROW["Battery_Replacement_Energy_Removed"]
L_REPLACEMENT_ENERGY_ADDED = _ROW["Battery_Replacement_Energy_Added"]
L_BATTERY_ENERGY_DELTA = _ROW["Battery_Energy_Delta"]
L_PV_ORIGIN_CHARGE_STORED = _ROW["PV_Origin_Battery_Charge_Stored"]
L_PV_ORIGIN_DISCHARGE_DC = _ROW["PV_Origin_Battery_Discharge_DC"]
L_PV_ORIGIN_BATTERY_AC_TO_LOAD = _ROW["PV_Origin_Battery_AC_To_Load"]
L_PV_ORIGIN_STANDBY_LOSS = _ROW["PV_Origin_Standby_Loss"]
L_PV_ORIGIN_CAPACITY_WINDOW_LOSS = _ROW["PV_Origin_Capacity_Window_Loss"]
L_PV_ORIGIN_REPLACEMENT_REMOVED = _ROW["PV_Origin_Replacement_Energy_Removed"]
L_GRID_ORIGIN_CHARGE_STORED = _ROW["Grid_Origin_Battery_Charge_Stored"]
L_GRID_ORIGIN_DISCHARGE_DC = _ROW["Grid_Origin_Battery_Discharge_DC"]
L_GRID_ORIGIN_BATTERY_AC_TO_LOAD = _ROW["Grid_Origin_Battery_AC_To_Load"]
L_GRID_ORIGIN_STANDBY_LOSS = _ROW["Grid_Origin_Standby_Loss"]
L_GRID_ORIGIN_CAPACITY_WINDOW_LOSS = _ROW["Grid_Origin_Capacity_Window_Loss"]
L_GRID_ORIGIN_REPLACEMENT_REMOVED = _ROW["Grid_Origin_Replacement_Energy_Removed"]


def lfp_capacity_factor(T_C: float) -> float:
    """
    Temperature-dependent usable capacity factor for LFP batteries.

    Returns a factor in [0.5, 1.0] relative to nominal capacity at 25°C.
    Uses a piecewise-linear model calibrated to typical LFP characterisation data:
      - ≥25°C: 1.0  (capacity doesn't increase meaningfully above reference)
      - 0–25°C: linear derating at LFP_CAP_DERATE_PER_C_MODERATE per °C
      - <0°C:   steeper derating at LFP_CAP_DERATE_PER_C_COLD per °C below 0

    Args:
        T_C: Battery temperature in °C

    Returns:
        Capacity factor (dimensionless, ≤ 1.0)
    """
    if T_C >= 25.0:
        return 1.0
    elif T_C >= 0.0:
        return 1.0 - LFP_CAP_DERATE_PER_C_MODERATE * (25.0 - T_C)
    else:
        base_at_zero = 1.0 - LFP_CAP_DERATE_PER_C_MODERATE * 25.0  # ~0.95
        return max(0.5, base_at_zero - LFP_CAP_DERATE_PER_C_COLD * abs(T_C))


def compute_cell_temperature(
    T_ambient_C: float,
    charge_power_w: float,
    discharge_power_w: float,
    charge_eff: float,
    discharge_eff: float,
    thermal_resistance_k_per_w: float = DEFAULT_THERMAL_RESISTANCE_K_PER_W,
) -> float:
    """
    Compute battery cell temperature using a quasi-steady-state lumped thermal model.

    Heat is generated by ohmic losses during charge and discharge. The cell
    temperature rises above ambient proportional to heat dissipation and
    thermal resistance of the enclosure.

    Valid for hourly (or longer) timesteps where the battery thermal mass
    reaches approximate equilibrium within each step.

    Args:
        T_ambient_C: Ambient temperature (C)
        charge_power_w: Power flowing into the battery this step (W, DC side)
        discharge_power_w: Power drawn from the battery this step (W, DC side)
        charge_eff: Charge efficiency (0-1)
        discharge_eff: Discharge efficiency (0-1)
        thermal_resistance_k_per_w: Thermal resistance in K/W

    Returns:
        Cell temperature (C)
    """
    # Heat from charging: fraction (1 - eta_charge) is lost as heat
    P_loss_charge = charge_power_w * (1.0 - charge_eff)
    # Heat from discharging: battery delivers more internally than reaches load
    P_loss_discharge = discharge_power_w * (1.0 - discharge_eff)

    P_loss_total = P_loss_charge + P_loss_discharge
    T_cell = T_ambient_C + thermal_resistance_k_per_w * P_loss_total
    return T_cell


def _apply_capacity_window(
    nominal_energy_wh: float,
    soh_fraction: float,
    max_soc: float,
    min_soc: float,
    standby_loss_wh: float,
    energy_wh: float,
    pv_origin_wh: float,
    grid_origin_wh: float,
    t_cell: float,
) -> Tuple[float, float, float, float, float, float, float, float, float, float, float]:
    """Derate the usable SOC window and bleed standby loss, before dispatch.

    Returns ``(energy, pv_origin, grid_origin, emin, emax,
    capacity_window_loss, standby, pv_capacity_window_loss,
    grid_capacity_window_loss, pv_standby, grid_standby)``, all in Wh. Assumes a configured battery; the no-battery case never calls
    this. ``standby_loss_wh`` is already scaled to the timestep.

    ``t_cell`` here is the ambient/indoor temperature at step start, not the
    self-heated cell temperature the thermal model produces later in the same
    step: usable capacity is set by the pack's state *before* this step's
    charge/discharge self-heating, while aging sees the warmed cell. That
    split is intentional.

    A temperature- or SOH-driven fall in ``emax`` is booked as an explicit
    loss — it is neither export nor standby consumption — and the lower
    reserve is a dispatch boundary that must never create energy when it
    rises. Each reduction removes the PV, grid and unattributed origins in
    proportion to their shares, so every origin stays a fraction of what is
    actually stored.
    """
    usable_cap = nominal_energy_wh * soh_fraction
    f_cap = lfp_capacity_factor(t_cell)
    emax = usable_cap * max_soc * f_cap
    emin = usable_cap * min_soc * f_cap

    pv_window_loss = 0.0
    grid_window_loss = 0.0
    capacity_window_loss = max(0.0, energy_wh - emax)
    if capacity_window_loss > 0.0 and energy_wh > 0.0:
        kept = emax / energy_wh
        pv_window_loss = pv_origin_wh
        grid_window_loss = grid_origin_wh
        pv_origin_wh *= kept
        grid_origin_wh *= kept
        pv_window_loss -= pv_origin_wh
        grid_window_loss -= grid_origin_wh
        energy_wh = emax

    pv_standby = 0.0
    grid_standby = 0.0
    removable_for_standby = max(0.0, energy_wh - emin)
    standby = min(standby_loss_wh, removable_for_standby)
    if standby > 0.0 and energy_wh > 0.0:
        kept = (energy_wh - standby) / energy_wh
        pv_standby = pv_origin_wh
        grid_standby = grid_origin_wh
        pv_origin_wh *= kept
        grid_origin_wh *= kept
        pv_standby -= pv_origin_wh
        grid_standby -= grid_origin_wh
        energy_wh -= standby

    return (
        energy_wh,
        pv_origin_wh,
        grid_origin_wh,
        emin,
        emax,
        capacity_window_loss,
        standby,
        pv_window_loss,
        grid_window_loss,
        pv_standby,
        grid_standby,
    )


def _charge(
    surplus_dc: float,
    battery_energy: float,
    emax: float,
    eff_charge: float,
    cap_charge_in_wh: float,
    cap_stored_wh: float,
) -> Tuple[float, float]:
    """Charge from *surplus_dc* up to the window and power limits.

    Returns ``(battery_energy, drawn)``, where ``drawn`` is the DC taken from
    the surplus; ``drawn * eff_charge`` of it is stored.
    """
    room = max(0.0, emax - battery_energy)
    if room <= 0.0 or eff_charge <= 0.0:
        return battery_energy, 0.0
    drawn = min(surplus_dc, room / eff_charge, cap_charge_in_wh, cap_stored_wh / eff_charge)
    return battery_energy + drawn * eff_charge, drawn


def _grid_charge(
    battery_energy: float,
    target_energy: float,
    emax: float,
    eff_charge: float,
    grid_eff: float,
    cap_charge_in_wh: float,
    cap_stored_wh: float,
    inverter_headroom_ac: float,
    site_headroom_ac: float,
    hold_target: bool = False,
) -> Tuple[float, float, float]:
    """Charge from the grid toward *target_energy*, after PV has been allocated.

    Grid AC ``a`` becomes DC charge input ``a * grid_eff``, which then passes
    through the existing charge efficiency like PV charge input (ADR 0002 A6).
    The caps are what PV left of the step's shared limits: charge input,
    stored energy, the inverter's AC rating and the site import limit.

    Returns ``(battery_energy, grid_ac, grid_dc)``.
    """
    room = min(target_energy, emax) - battery_energy
    if room <= 0.0 or eff_charge <= 0.0:
        return battery_energy, 0.0, 0.0
    required_ac = room / eff_charge / grid_eff
    grid_ac = min(
        required_ac,
        cap_charge_in_wh / grid_eff,
        cap_stored_wh / eff_charge / grid_eff,
        inverter_headroom_ac,
        site_headroom_ac,
    )
    if grid_ac <= 0.0:
        return battery_energy, 0.0, 0.0
    grid_dc = grid_ac * grid_eff
    if hold_target and grid_ac == required_ac:
        return min(target_energy, emax), grid_ac, grid_dc
    return battery_energy + grid_dc * eff_charge, grid_ac, grid_dc


def _combined_conversion(
    pv_dc: float,
    battery_dc: float,
    inv_cap_ac_wh: float,
    inv_eff: float,
    ac_output_scale: float,
    pow_two: float,
) -> Tuple[float, float, float]:
    """Convert PV and battery DC at one shared inverter operating point.

    Returns ``(pv_ac, battery_ac, conversion_loss)``, splitting the AC output
    in proportion to each source's share of the DC input.
    """
    total_dc = pv_dc + battery_dc
    ac_power, conversion_loss, _clipping_dc = _dc_ac(total_dc, inv_cap_ac_wh, inv_eff, ac_output_scale, pow_two)
    if total_dc <= 0.0:
        return 0.0, 0.0, 0.0
    battery_ac = ac_power * battery_dc / total_dc
    pv_ac = ac_power - battery_ac
    return pv_ac, battery_ac, conversion_loss


def _dispatch_dc_step(
    pv_dc: float,
    load: float,
    battery_energy: float,
    emin: float,
    emax: float,
    eff_charge: float,
    eff_discharge: float,
    inv_eff: float,
    cap_charge_in_wh: float,
    cap_discharge_ac_wh: float,
    inv_cap_ac_wh: float,
    ac_output_scale: float,
    cap_stored_wh: float,
    pow_two: float,
    discharge_allowed: bool,
    discharge_floor: float,
    grid_target_energy: float,
    grid_eff: float,
    grid_import_cap_wh: float,
) -> Tuple[
    float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float
]:
    """Dispatch one DC-coupled timestep; inputs and outputs are Wh.

    PV serves AC load first. Surplus DC then charges the battery before any
    export. PV and battery discharge share the inverter AC nameplate.
    ``cap_stored_wh`` limits the stored energy gained or released, on top of
    the DC charge-input and AC discharge limits.

    The instruction inputs (ADR 0002) gate that greedy step without
    replacing it. The battery discharges only if ``discharge_allowed``, and
    never below ``discharge_floor``. A finite ``grid_target_energy`` then
    grid-charges toward it with what PV left of each shared limit, but not
    while PV is exported or the battery has discharged. With
    ``discharge_allowed`` true, ``discharge_floor == emin`` and a NaN target
    the step is the greedy step, operation for operation.

    Returns the stored energy after the step followed by the step's ledger:
    ``(battery_energy, pv_dc_to_battery, pv_dc_to_inverter, pv_dc_curtailed,
    pv_ac_to_load, pv_ac_export, battery_charge_input, battery_discharge_dc,
    battery_ac_to_load, battery_charge_loss, battery_discharge_loss,
    pv_direct_inverter_loss, battery_inverter_loss, grid_import, grid_ac,
    grid_dc)``. ``grid_import`` includes ``grid_ac``, and
    ``battery_charge_input`` includes ``grid_dc``.
    """
    pv_dc_curtailed = 0.0
    pv_ac_export = 0.0
    battery_discharge_dc = 0.0
    battery_ac_to_load = 0.0
    battery_discharge_loss = 0.0
    battery_inverter_loss = 0.0
    drawn = 0.0
    grid_ac = 0.0
    grid_dc = 0.0
    hold_target = discharge_allowed and not math.isnan(grid_target_energy) and discharge_floor >= grid_target_energy

    pv_ac_max, pv_conversion_loss, pv_clipping_dc = _dc_ac(pv_dc, inv_cap_ac_wh, inv_eff, ac_output_scale, pow_two)

    if pv_ac_max >= load:
        pv_ac_to_load = load
        dc_to_load = _dc_for_ac(load, inv_cap_ac_wh, inv_eff, ac_output_scale)
        surplus_dc = max(0.0, pv_dc - dc_to_load)
        battery_energy, drawn = _charge(surplus_dc, battery_energy, emax, eff_charge, cap_charge_in_wh, cap_stored_wh)
        remaining_dc = surplus_dc - drawn
        direct_ac, direct_conversion_loss, direct_clipping_dc = _dc_ac(
            dc_to_load + remaining_dc, inv_cap_ac_wh, inv_eff, ac_output_scale, pow_two
        )
        pv_ac_export = max(0.0, direct_ac - load)
        dc_export = max(0.0, dc_to_load + remaining_dc - direct_clipping_dc - dc_to_load)
        pv_dc_to_inverter = dc_to_load + dc_export
        pv_dc_curtailed = direct_clipping_dc
        pv_direct_inverter_loss = direct_conversion_loss
        grid_import = 0.0
    else:
        pv_ac_to_load = pv_ac_max
        pv_dc_to_inverter = pv_dc - pv_clipping_dc
        pv_direct_inverter_loss = pv_conversion_loss
        excess_dc = pv_clipping_dc
        deficit = load - pv_ac_max
        if excess_dc > 1e-12:
            # The inverter is saturated by PV. DC above its immediate AC
            # headroom may charge, but battery discharge has no AC headroom.
            battery_energy, drawn = _charge(
                excess_dc, battery_energy, emax, eff_charge, cap_charge_in_wh, cap_stored_wh
            )
            pv_dc_curtailed = excess_dc - drawn
            grid_import = deficit
        else:
            available = max(0.0, battery_energy - discharge_floor)
            # AC correction is applied after the inverter curve and nameplate
            # limit, so the reachable AC ceiling is the scaled nameplate, which
            # the (0, 1] bound keeps at or below the nameplate itself.
            target_total_ac = min(load, inv_cap_ac_wh * ac_output_scale)
            if discharge_allowed and available > 0.0 and eff_discharge > 0.0 and target_total_ac > pv_ac_max:
                total_dc_target = _dc_for_ac(target_total_ac, inv_cap_ac_wh, inv_eff, ac_output_scale)
                battery_dc = min(
                    available * eff_discharge,
                    max(0.0, total_dc_target - pv_dc),
                    cap_stored_wh * eff_discharge,
                )

                # The public discharge limit is AC delivered. If it binds,
                # solve for the battery DC contribution at the one shared
                # inverter operating point rather than applying a second
                # independent part-load curve.
                if math.isfinite(cap_discharge_ac_wh):
                    unconstrained_battery_ac = _combined_conversion(
                        pv_dc, battery_dc, inv_cap_ac_wh, inv_eff, ac_output_scale, pow_two
                    )[1]
                    if unconstrained_battery_ac > cap_discharge_ac_wh:
                        lower = 0.0
                        upper = battery_dc
                        for _iteration in range(40):
                            midpoint = (lower + upper) / 2.0
                            midpoint_battery_ac = _combined_conversion(
                                pv_dc, midpoint, inv_cap_ac_wh, inv_eff, ac_output_scale, pow_two
                            )[1]
                            if midpoint_battery_ac < cap_discharge_ac_wh:
                                lower = midpoint
                            else:
                                upper = midpoint
                        battery_dc = upper

                pv_delivered_ac, delivered_ac, total_inverter_loss = _combined_conversion(
                    pv_dc, battery_dc, inv_cap_ac_wh, inv_eff, ac_output_scale, pow_two
                )
                draw = battery_dc / eff_discharge
                if hold_target:
                    # Land exactly on a binding floor: multiplying available
                    # energy by efficiency and dividing it back can undershoot
                    # by an ulp and buy that rounding residue on the next step.
                    if battery_dc == available * eff_discharge:
                        draw = available
                    draw = min(draw, available)
                    battery_energy = (
                        discharge_floor if draw == available else max(discharge_floor, battery_energy - draw)
                    )
                else:
                    battery_energy -= draw
                total_inverter_dc = pv_dc + battery_dc
                battery_inverter_loss = (
                    total_inverter_loss * battery_dc / total_inverter_dc if total_inverter_dc > 0.0 else 0.0
                )
                battery_discharge_dc = draw
                battery_ac_to_load = delivered_ac
                battery_discharge_loss = draw - battery_dc
                pv_ac_to_load = pv_delivered_ac
                pv_direct_inverter_loss = total_inverter_loss - battery_inverter_loss
                grid_import = max(0.0, load - pv_delivered_ac - delivered_ac)
            else:
                grid_import = deficit

    battery_charge_input = drawn
    if not math.isnan(grid_target_energy) and pv_ac_export <= 0.0 and battery_discharge_dc <= 0.0:
        # PV keeps priority on every limit the two sources share (A6): grid
        # charge gets the charge input and stored energy PV did not use, the
        # inverter rating less the step's PV AC output, and the site limit
        # less the step's load import.
        battery_energy, grid_ac, grid_dc = _grid_charge(
            battery_energy,
            grid_target_energy,
            emax,
            eff_charge,
            grid_eff,
            cap_charge_in_wh - drawn,
            cap_stored_wh - drawn * eff_charge,
            inv_cap_ac_wh - pv_ac_to_load - pv_ac_export,
            grid_import_cap_wh - grid_import,
            hold_target,
        )
        battery_charge_input = drawn + grid_dc
        grid_import = grid_import + grid_ac

    return (
        battery_energy,
        drawn,
        pv_dc_to_inverter,
        pv_dc_curtailed,
        pv_ac_to_load,
        pv_ac_export,
        battery_charge_input,
        battery_discharge_dc,
        battery_ac_to_load,
        battery_charge_input * (1.0 - eff_charge),
        battery_discharge_loss,
        pv_direct_inverter_loss,
        battery_inverter_loss,
        grid_import,
        grid_ac,
        grid_dc,
    )


def _dispatch_day(
    matrix: np.ndarray,
    pv_dc_vals: np.ndarray,
    load_vals: np.ndarray,
    temp_vals: np.ndarray,
    lo: int,
    hi: int,
    battery_energy: float,
    pv_origin: float,
    grid_origin: float,
    nominal_energy_wh: float,
    soh_fraction: float,
    max_soc: float,
    min_soc: float,
    standby_loss_per_step_wh: float,
    eff_charge: float,
    eff_discharge: float,
    inv_eff: float,
    cap_charge_in_wh: float,
    cap_discharge_ac_wh: float,
    inv_cap_ac_wh: float,
    cap_stored_wh: float,
    thermal_resistance_k_per_w: float,
    hours_per_step: float,
    ac_output_scale: float,
    pow_two: float,
    discharge_allowed: np.ndarray,
    reserve_fraction: np.ndarray,
    grid_target_fraction: np.ndarray,
    grid_eff: float,
    grid_import_cap_wh: float,
) -> None:
    """Dispatch timesteps ``[lo, hi)`` at fixed health, writing rows of *matrix*.

    State of health, resistance-derived efficiencies and the replacement
    decision are unchanged for the duration of the call; the caller advances
    them at the day boundary. Only battery runs come here: PV-only runs take
    the vectorised path in :mod:`breos.battery`.

    ``discharge_allowed``, ``reserve_fraction`` and ``grid_target_fraction``
    are the per-step instruction arrays (ADR 0002), indexed like the inputs.
    A fraction is of each step's usable window, ``emin + f * (emax - emin)``,
    so the energy it names moves with temperature and health (A7).

    Nothing is returned: the state the next window and the day-close
    replacement path need is the final step's row of *matrix*.
    """
    soh_percent = soh_fraction * 100.0
    for i in range(lo, hi):
        # Treat negative model/data artefacts as zero generation, matching the
        # public inverter helper and preventing negative PV from being
        # allocated through the shared PV/battery conversion path.
        pv_dc_power = max(0.0, pv_dc_vals[i] * hours_per_step)  # DC power (Wh) before inverter
        load = load_vals[i] * hours_per_step  # AC Load in Wh
        t_ambient = temp_vals[i]
        t_cell = t_ambient  # default; overridden by thermal model below

        battery_energy_beginning = battery_energy
        pv_origin_beginning = pv_origin
        grid_origin_beginning = grid_origin
        (
            battery_energy,
            pv_origin,
            grid_origin,
            emin,
            emax,
            capacity_window_loss,
            battery_standby_loss,
            pv_window_loss,
            grid_window_loss,
            pv_standby_loss,
            grid_standby_loss,
        ) = _apply_capacity_window(
            nominal_energy_wh,
            soh_fraction,
            max_soc,
            min_soc,
            standby_loss_per_step_wh,
            battery_energy,
            pv_origin,
            grid_origin,
            t_cell,
        )

        # Instructions apply to this step's window. A zero reserve and no
        # target leave the greedy floor, emin, exactly as it is.
        reserve = reserve_fraction[i]
        discharge_floor = emin + reserve * (emax - emin) if reserve > 0.0 else emin
        target = grid_target_fraction[i]
        grid_target_energy = emin + target * (emax - emin) if not math.isnan(target) else math.nan
        if discharge_allowed[i] and reserve == target:
            # A14's shared fraction has one A7 energy bound, exactly.
            discharge_floor = grid_target_energy

        # Discharge takes from every origin in proportion to its share before
        # dispatch. One share per step is exact only because a step either
        # charges or discharges, which is checked below.
        energy_before_dispatch = battery_energy
        origin_before_dispatch = pv_origin
        grid_before_dispatch = grid_origin
        origin_fraction = (
            min(1.0, max(0.0, origin_before_dispatch / energy_before_dispatch)) if energy_before_dispatch > 0.0 else 0.0
        )
        grid_fraction = (
            min(1.0, max(0.0, grid_before_dispatch / energy_before_dispatch)) if energy_before_dispatch > 0.0 else 0.0
        )
        (
            battery_energy,
            pv_dc_to_battery,
            pv_dc_to_inverter,
            pv_dc_curtailed,
            pv_ac_to_load,
            pv_ac_export,
            battery_charge_input,
            battery_discharge_dc,
            battery_ac_to_load,
            battery_charge_loss,
            battery_discharge_loss,
            pv_direct_inverter_loss,
            battery_inverter_loss,
            grid_import,
            grid_ac,
            grid_dc,
        ) = _dispatch_dc_step(
            pv_dc_power,
            load,
            battery_energy,
            emin,
            emax,
            eff_charge,
            eff_discharge,
            inv_eff,
            cap_charge_in_wh,
            cap_discharge_ac_wh,
            inv_cap_ac_wh,
            ac_output_scale,
            cap_stored_wh,
            pow_two,
            discharge_allowed[i],
            discharge_floor,
            grid_target_energy,
            grid_eff,
            grid_import_cap_wh,
        )
        if battery_charge_input > 0.0 and battery_discharge_dc > 0.0:
            raise ValueError("a dispatch step both charged and discharged the battery")
        charge_stored = battery_charge_input * eff_charge
        # Each source's charge adds to its own origin: PV DC to PV, grid to grid.
        pv_charge_stored = pv_dc_to_battery * eff_charge
        grid_charge_stored = grid_dc * eff_charge
        pv_origin_discharge_dc = battery_discharge_dc * origin_fraction
        pv_origin_battery_ac = battery_ac_to_load * origin_fraction
        pv_origin = max(0.0, origin_before_dispatch - pv_origin_discharge_dc + pv_charge_stored)
        pv_origin = min(pv_origin, battery_energy)
        grid_origin_discharge_dc = battery_discharge_dc * grid_fraction
        grid_origin_battery_ac = battery_ac_to_load * grid_fraction
        grid_origin = max(
            0.0,
            min(grid_before_dispatch - grid_origin_discharge_dc + grid_charge_stored, battery_energy - pv_origin),
        )

        # PV output after clipping and the direct PV inverter loss: AC to load
        # and export plus DC to the battery, with or without an inverter rating.
        pv_production = pv_dc_power - pv_dc_curtailed - pv_direct_inverter_loss
        battery_energy_delta = battery_energy - battery_energy_beginning

        # Compute cell temperature via lumped thermal model
        if thermal_resistance_k_per_w > 0:
            # The ledger is in Wh; convert to W for the thermal calculation
            charge_power_w = battery_charge_input / hours_per_step if hours_per_step > 0 else 0.0
            discharge_power_w = battery_discharge_dc / hours_per_step if hours_per_step > 0 else 0.0
            t_cell = compute_cell_temperature(
                t_ambient,
                charge_power_w,
                discharge_power_w,
                eff_charge,
                eff_discharge,
                thermal_resistance_k_per_w,
            )

        soc_normalized = (battery_energy - emin) / (emax - emin) if (emax - emin) > 0 else 0.0
        soc_normalized = max(0.0, min(1.0, soc_normalized))
        soc_absolute = battery_energy / (nominal_energy_wh * soh_fraction) if soh_fraction > 0 else 0.0
        soc_absolute = max(0.0, min(1.0, soc_absolute))

        matrix[R_PV_DC, i] = pv_dc_power / hours_per_step
        matrix[R_PV_PRODUCTION, i] = pv_production / hours_per_step
        matrix[R_LOAD, i] = load / hours_per_step
        matrix[R_PV_DELTA, i] = (pv_production - load) / hours_per_step
        matrix[R_GRID_IMPORT, i] = grid_import / hours_per_step
        matrix[R_BATTERY_ENERGY, i] = battery_energy
        matrix[R_SOC_NORMALIZED, i] = soc_normalized
        matrix[R_SOC_ABSOLUTE, i] = soc_absolute
        matrix[R_SOH, i] = soh_percent
        matrix[R_T_CELL, i] = t_cell
        matrix[R_CHARGE_LOSS, i] = battery_charge_loss / hours_per_step
        matrix[R_DISCHARGE_LOSS, i] = battery_discharge_loss / hours_per_step
        matrix[R_BATTERY_ENERGY_BEGIN, i] = battery_energy_beginning
        matrix[R_PV_ORIGIN_BEGIN, i] = pv_origin_beginning
        matrix[R_PV_ORIGIN_END, i] = pv_origin
        matrix[R_GRID_ORIGIN_BEGIN, i] = grid_origin_beginning
        matrix[R_GRID_ORIGIN_END, i] = grid_origin

        matrix[L_PV_DC_TO_BATTERY, i] = pv_dc_to_battery / hours_per_step
        matrix[L_PV_DC_TO_INVERTER, i] = pv_dc_to_inverter / hours_per_step
        matrix[L_PV_DC_CURTAILED, i] = pv_dc_curtailed / hours_per_step
        matrix[L_PV_AC_TO_LOAD, i] = pv_ac_to_load / hours_per_step
        matrix[L_PV_AC_EXPORT, i] = pv_ac_export / hours_per_step
        matrix[L_BATTERY_CHARGE_INPUT, i] = battery_charge_input / hours_per_step
        matrix[L_BATTERY_CHARGE_STORED, i] = charge_stored / hours_per_step
        matrix[L_GRID_AC_TO_BATTERY, i] = grid_ac / hours_per_step
        matrix[L_GRID_DC_TO_BATTERY, i] = grid_dc / hours_per_step
        matrix[L_GRID_CHARGE_CONVERSION_LOSS, i] = (grid_ac - grid_dc) / hours_per_step
        matrix[L_BATTERY_DISCHARGE_DC, i] = battery_discharge_dc / hours_per_step
        matrix[L_BATTERY_AC_TO_LOAD, i] = battery_ac_to_load / hours_per_step
        matrix[L_PV_DIRECT_INVERTER_LOSS, i] = pv_direct_inverter_loss / hours_per_step
        matrix[L_BATTERY_INVERTER_LOSS, i] = battery_inverter_loss / hours_per_step
        matrix[L_INVERTER_LOSS, i] = (pv_direct_inverter_loss + battery_inverter_loss) / hours_per_step
        matrix[L_STANDBY_LOSS, i] = battery_standby_loss / hours_per_step
        matrix[L_CAPACITY_WINDOW_LOSS, i] = capacity_window_loss / hours_per_step
        matrix[L_REPLACEMENT_ENERGY_REMOVED, i] = 0.0
        matrix[L_REPLACEMENT_ENERGY_ADDED, i] = 0.0
        matrix[L_BATTERY_ENERGY_DELTA, i] = battery_energy_delta / hours_per_step
        matrix[L_PV_ORIGIN_CHARGE_STORED, i] = pv_charge_stored / hours_per_step
        matrix[L_PV_ORIGIN_DISCHARGE_DC, i] = pv_origin_discharge_dc / hours_per_step
        matrix[L_PV_ORIGIN_BATTERY_AC_TO_LOAD, i] = pv_origin_battery_ac / hours_per_step
        matrix[L_PV_ORIGIN_STANDBY_LOSS, i] = pv_standby_loss / hours_per_step
        matrix[L_PV_ORIGIN_CAPACITY_WINDOW_LOSS, i] = pv_window_loss / hours_per_step
        matrix[L_PV_ORIGIN_REPLACEMENT_REMOVED, i] = 0.0
        matrix[L_GRID_ORIGIN_CHARGE_STORED, i] = grid_charge_stored / hours_per_step
        matrix[L_GRID_ORIGIN_DISCHARGE_DC, i] = grid_origin_discharge_dc / hours_per_step
        matrix[L_GRID_ORIGIN_BATTERY_AC_TO_LOAD, i] = grid_origin_battery_ac / hours_per_step
        matrix[L_GRID_ORIGIN_STANDBY_LOSS, i] = grid_standby_loss / hours_per_step
        matrix[L_GRID_ORIGIN_CAPACITY_WINDOW_LOSS, i] = grid_window_loss / hours_per_step
        matrix[L_GRID_ORIGIN_REPLACEMENT_REMOVED, i] = 0.0


def _day_arguments(
    out: Any,
    pv_dc_vals: np.ndarray,
    load_vals: np.ndarray,
    temp_vals: np.ndarray,
    lo: int,
    hi: int,
    *,
    battery_config: "BatteryConfig",
    battery_soh_decimal: float,
    Battery_Energy_Wh: float,
    Battery_PV_Origin_Energy_Wh: float,
    Battery_Grid_Origin_Energy_Wh: float,
    eff_charge: float,
    eff_discharge: float,
    hours_per_step: float,
    standby_loss_per_step_wh: float,
    cap_wh: float,
    cap_charge_wh: float,
    cap_discharge_wh: float,
    cap_stored_wh: float = math.inf,
    instructions: Optional["DispatchInstructions"] = None,
) -> Tuple[Any, ...]:
    """Pack one day's state into the positional arguments of :func:`_dispatch_day`.

    Both backends pack through here, so they cannot be handed different
    inputs. Scalars are coerced to ``float`` so the compiled backend always
    sees one signature. The 2.0 after the scalars is ``pow_two``; see
    :func:`_dc_ac` for why it is passed rather than written.

    ``instructions`` covers the whole run, not just this day, since the day
    loop indexes it like the inputs. None means greedy dispatch; a caller
    that packs many days passes one no-op set rather than building one per
    day.
    """
    if instructions is None:
        instructions = DispatchInstructions.noop(len(pv_dc_vals))
    return (
        out.matrix,
        pv_dc_vals,
        load_vals,
        temp_vals,
        lo,
        hi,
        float(Battery_Energy_Wh),
        float(Battery_PV_Origin_Energy_Wh),
        float(Battery_Grid_Origin_Energy_Wh),
        float(battery_config.nominal_energy_wh),
        float(battery_soh_decimal),
        float(battery_config.max_soc),
        float(battery_config.min_soc),
        float(standby_loss_per_step_wh),
        float(eff_charge),
        float(eff_discharge),
        float(battery_config.inverter_efficiency),
        float(cap_charge_wh),
        float(cap_discharge_wh),
        float(cap_wh),
        float(cap_stored_wh),
        float(battery_config.thermal_resistance_k_per_w),
        float(hours_per_step),
        float(battery_config.ac_output_scale),
        2.0,
        instructions.discharge_allowed,
        instructions.reserve_fraction,
        instructions.grid_target_fraction,
        float(instructions.grid_charge_efficiency),
        float(instructions.grid_import_limit_w * hours_per_step),
    )


def _dispatch_day_python(out: Any, *args: Any, **state: Any) -> None:
    """Run :func:`_dispatch_day` as Python; the reference backend.

    Takes the arguments of :func:`_day_arguments`.
    """
    _dispatch_day(*_day_arguments(out, *args, **state))
