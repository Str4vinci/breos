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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np

from breos.constants import DEFAULT_THERMAL_RESISTANCE_K_PER_W, LFP_CAP_DERATE_PER_C_COLD, LFP_CAP_DERATE_PER_C_MODERATE
from breos.dispatch_instructions import DispatchInstructions

if TYPE_CHECKING:
    from breos.battery import BatteryConfig
    from breos.vehicle import AwayChargingConfig, VehicleConfig, VehicleSchedule

# PVWatts part-load curve. The inverter cores live here, beside the day loop,
# because Numba's cache is keyed on this file: an edit to them or to these
# constants must invalidate the compiled kernel.
PVWATTS_REFERENCE_EFFICIENCY = 0.9637
PVWATTS_CURVE_QUADRATIC = -0.0162
PVWATTS_CURVE_LINEAR = 0.9858
PVWATTS_CURVE_CONSTANT = -0.0059

# Empty schedules use the resolved schedule's readonly array types too, so
# toggling a vehicle does not introduce another Numba signature.
_EMPTY_VEHICLE_MASK: np.ndarray = np.empty(0, dtype=np.bool_)
_EMPTY_VEHICLE_COUNTDOWN: np.ndarray = np.empty(0, dtype=np.int64)
_EMPTY_VEHICLE_MASK.setflags(write=False)
_EMPTY_VEHICLE_COUNTDOWN.setflags(write=False)
_EMPTY_VEHICLE_FLOAT: np.ndarray = np.empty(0)
_EMPTY_VEHICLE_FLOAT.setflags(write=False)


def _curve_efficiency(curve_eff: np.ndarray, curve_row: float, point: int) -> float:
    """Efficiency at one output point, interpolated between the voltage rows at ``curve_row``.

    ``curve_row`` is a position among the rows: 1.5 lies halfway between the
    second and third. Past either end it holds the end row.
    """
    last = curve_eff.shape[0] - 1
    if curve_row <= 0.0:
        return curve_eff[0, point]
    if curve_row >= last:
        return curve_eff[last, point]
    lower = int(math.floor(curve_row))
    fraction = curve_row - lower
    return curve_eff[lower, point] + fraction * (curve_eff[lower + 1, point] - curve_eff[lower, point])


def _curve_ac(
    dc: float, inverter_ac_power: float, curve_ac: np.ndarray, curve_eff: np.ndarray, curve_row: float
) -> float:
    """AC output for DC input ``dc`` at or below the nameplate input, on the efficiency curve.

    The curve gives efficiency against AC output at its points, linear
    between them. Within a segment ``ac = (eta_j - s * a_j) * dc / (1 - s * dc)``
    solves ``ac = eta(ac) * dc`` exactly. Below the first point the inverter
    loses that point's loss in watts, its self-consumption; above the last
    point it keeps that point's efficiency.
    """
    last = curve_ac.shape[0] - 1
    a_lo = curve_ac[0] * inverter_ac_power
    eta_lo = _curve_efficiency(curve_eff, curve_row, 0)
    d_lo = a_lo / eta_lo
    if dc <= d_lo:
        return max(0.0, dc - (d_lo - a_lo))
    for j in range(last):
        a_hi = curve_ac[j + 1] * inverter_ac_power
        eta_hi = _curve_efficiency(curve_eff, curve_row, j + 1)
        d_hi = a_hi / eta_hi
        if dc < d_hi:
            slope = (eta_hi - eta_lo) / (a_hi - a_lo)
            return (eta_lo - slope * a_lo) * dc / (1.0 - slope * dc)
        a_lo, eta_lo, d_lo = a_hi, eta_hi, d_hi
    return eta_lo * dc


def _curve_dc(
    ac: float, inverter_ac_power: float, curve_ac: np.ndarray, curve_eff: np.ndarray, curve_row: float
) -> float:
    """The DC input :func:`_curve_ac` needs for AC output ``ac``: its exact inverse."""
    last = curve_ac.shape[0] - 1
    a_lo = curve_ac[0] * inverter_ac_power
    eta_lo = _curve_efficiency(curve_eff, curve_row, 0)
    if ac <= a_lo:
        return ac + (a_lo / eta_lo - a_lo)
    for j in range(last):
        a_hi = curve_ac[j + 1] * inverter_ac_power
        eta_hi = _curve_efficiency(curve_eff, curve_row, j + 1)
        if ac < a_hi:
            return ac / (eta_lo + (eta_hi - eta_lo) * (ac - a_lo) / (a_hi - a_lo))
        a_lo, eta_lo = a_hi, eta_hi
    return ac / eta_lo


def _cable_gross(net: float, cable_k: float) -> float:
    """AC leaving the inverter that delivers ``net`` past a cable losing ``cable_k * ac**2``; inf if none can."""
    discriminant = 1.0 - 4.0 * cable_k * net
    if discriminant < 0.0:
        return math.inf
    return 2.0 * net / (1.0 + math.sqrt(discriminant))


def _dc_ac(
    pv_dc_power: float,
    inverter_ac_power: float,
    inverter_efficiency: float,
    ac_output_scale: float,
    pow_two: float,
    curve_ac: np.ndarray,
    curve_eff: np.ndarray,
    curve_row: float,
    cable_k: float,
) -> tuple[float, float, float]:
    """Scalar core of :func:`calculate_dc_ac_power`: ``(ac, conversion_loss, clipping_dc)``.

    Written without dataclasses or ``float()`` coercion so the Numba backend
    compiles this same function.

    An empty ``curve_ac`` selects the PVWatts part-load curve. Otherwise
    ``curve_ac`` holds output points as fractions of the AC nameplate and
    ``curve_eff`` one row of efficiencies per DC voltage, read at row position
    ``curve_row`` (:func:`_curve_efficiency`). ``cable_k`` is the AC cable's
    loss over the square of the AC output, per Wh: the cable takes
    ``cable_k * ac**2`` before ``ac_output_scale``. The conversion loss
    includes it. With no curve and no cable the arithmetic is the PVWatts
    path's, operation for operation.

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

    if curve_ac.shape[0] > 0:
        pdc0 = _curve_dc(inverter_ac_power, inverter_ac_power, curve_ac, curve_eff, curve_row)
        dc_used = min(pv_dc_power, pdc0)
        ac_power = max(
            0.0, min(dc_used, inverter_ac_power, _curve_ac(dc_used, inverter_ac_power, curve_ac, curve_eff, curve_row))
        )
        if cable_k > 0.0:
            ac_power = max(0.0, ac_power - cable_k * ac_power * ac_power)
        ac_power *= ac_output_scale
        return ac_power, max(0.0, dc_used - ac_power), max(0.0, pv_dc_power - dc_used)

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
    if cable_k > 0.0:
        ac_power = max(0.0, ac_power - cable_k * ac_power * ac_power)
    ac_power *= ac_output_scale
    clipping_loss_dc = max(0.0, pv_dc_power - dc_used)
    conversion_loss = max(0.0, dc_used - ac_power)
    return ac_power, conversion_loss, clipping_loss_dc


def _dc_for_ac(
    ac_power_w: float,
    inverter_ac_power: float,
    inverter_efficiency: float,
    ac_output_scale: float,
    curve_ac: np.ndarray,
    curve_eff: np.ndarray,
    curve_row: float,
    cable_k: float,
) -> float:
    """Scalar core of :func:`dc_power_for_ac_output`, compiled as-is by the Numba backend.

    Takes the curve and cable of :func:`_dc_ac` and stays its inverse.
    """
    ac_output_scale = min(1.0, max(0.0, ac_output_scale))
    if ac_output_scale <= 0.0:
        return 0.0
    ac_power_w = ac_power_w / ac_output_scale
    if cable_k > 0.0:
        ac_power_w = _cable_gross(max(0.0, ac_power_w), cable_k)
    ac_target = max(0.0, min(ac_power_w, max(0.0, inverter_ac_power)))
    inverter_ac_power = max(0.0, inverter_ac_power)
    inverter_efficiency = min(1.0, max(0.0, inverter_efficiency))
    if ac_target <= 0.0 or inverter_ac_power <= 0.0 or inverter_efficiency <= 0.0:
        return 0.0
    if not math.isfinite(inverter_ac_power):
        return ac_target / inverter_efficiency

    if curve_ac.shape[0] > 0:
        upper = _curve_dc(inverter_ac_power, inverter_ac_power, curve_ac, curve_eff, curve_row)
        if ac_target >= inverter_ac_power:
            return upper
        return min(upper, max(ac_target, _curve_dc(ac_target, inverter_ac_power, curve_ac, curve_eff, curve_row)))

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
# 3.1 adds, without changing a field 3.0 consumers read: the
# Inverter_Night_Consumption column (within Import_From_Grid); the
# near_shading, shading_mismatch, dc_wiring, parallel_mismatch and
# mppt_limits loss-waterfall stages, present only when their inputs are; the
# waterfall's near_shading, shading_mismatch, parallel_mismatch and dc_wiring
# blocks; and the inverter block's AC cable, efficiency curve, temperature
# derating and night consumption entries. It also adds PV_AC_To_Battery and
# PV_Charge_Conversion_Loss, the PV inverter's AC that an AC-coupled battery
# inverter takes in and that inverter's charging loss on it (ADR 0004 C8);
# both are zero in a DC-coupled or PV-only run. Vehicle states, home,
# boundary, away and origin flows also join unreleased 3.1 (ADR 0005 A2).
LEDGER_SCHEMA_VERSION = "3.1"

# Per-step state columns, by results-frame name, in their row order inside the
# shared buffer matrix. Stored-energy columns are Wh, SOC is a fraction,
# stationary SOH is percent, and T_cell is Celsius. Power is average W.
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
    # AC coupling (ADR 0004 C8): PV inverter AC into the battery inverter and
    # that inverter's charging loss on it. Their difference is part of
    # Battery_Charge_Input. Zero when PV reaches the battery as DC.
    "PV_AC_To_Battery",
    "PV_Charge_Conversion_Loss",
    "Battery_Discharge_DC",
    "Battery_AC_To_Load",
    "PV_Direct_Inverter_Loss",
    "Battery_Inverter_Loss",
    "Inverter_Loss",
    # What the inverter draws from the grid while it converts nothing: no PV
    # DC, no battery charge or discharge. Part of Import_From_Grid.
    "Inverter_Night_Consumption",
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
_VEHICLE_STATE_COLUMNS: Tuple[str, ...] = (
    "Vehicle_Plugged",
    "Vehicle_Energy",
    "Vehicle_Energy_Onboard",
    "Vehicle_SOC_Absolute",
    "Vehicle_SOH",
    "Vehicle_PV_Origin_Energy_End",
    "Vehicle_Grid_Origin_Energy_End",
)
_VEHICLE_LEDGER_COLUMNS: Tuple[str, ...] = (
    "PV_AC_To_Vehicle",
    "Grid_AC_To_Vehicle",
    "Vehicle_Charge_Conversion_Loss",
    "Vehicle_Charge_Input",
    "Vehicle_Charge_Loss",
    "Vehicle_Discharge_DC",
    "Vehicle_Discharge_Loss",
    "Vehicle_Charger_Loss",
    "Vehicle_AC_To_Load",
    "PV_Origin_Vehicle_AC_To_Load",
    "Grid_Origin_Vehicle_AC_To_Load",
    "Vehicle_Departure_Energy",
    "Vehicle_Arrival_Energy",
    "Vehicle_Departure_Shortfall",
    "Vehicle_Drive_Energy",
    "Vehicle_Drive_Unserved",
    "Vehicle_Away_Charge_Purchased",
    "Vehicle_Away_Charge_Loss",
    "PV_Origin_Vehicle_Drive_Energy",
    "Grid_Origin_Vehicle_Drive_Energy",
    "Vehicle_Capacity_Window_Loss",
    "Vehicle_Replacement_Energy_Removed",
    "Vehicle_Replacement_Energy_Added",
    "PV_Origin_Vehicle_Capacity_Window_Loss",
    "Grid_Origin_Vehicle_Capacity_Window_Loss",
    "PV_Origin_Vehicle_Replacement_Energy_Removed",
    "Grid_Origin_Vehicle_Replacement_Energy_Removed",
    "PV_Origin_Vehicle_Charge_Stored",
    "Grid_Origin_Vehicle_Charge_Stored",
    "PV_Origin_Vehicle_Discharge_DC",
    "Grid_Origin_Vehicle_Discharge_DC",
)

_STATE_COLUMNS += _VEHICLE_STATE_COLUMNS
_LEDGER_COLUMNS += _VEHICLE_LEDGER_COLUMNS
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
L_PV_AC_TO_BATTERY = _ROW["PV_AC_To_Battery"]
L_PV_CHARGE_CONVERSION_LOSS = _ROW["PV_Charge_Conversion_Loss"]
L_BATTERY_DISCHARGE_DC = _ROW["Battery_Discharge_DC"]
L_BATTERY_AC_TO_LOAD = _ROW["Battery_AC_To_Load"]
L_PV_DIRECT_INVERTER_LOSS = _ROW["PV_Direct_Inverter_Loss"]
L_BATTERY_INVERTER_LOSS = _ROW["Battery_Inverter_Loss"]
L_INVERTER_LOSS = _ROW["Inverter_Loss"]
L_INVERTER_NIGHT_CONSUMPTION = _ROW["Inverter_Night_Consumption"]
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

# Vehicle states are Wh (SOC, SOH and plugged are fractions); flows are average W.
R_VEHICLE_PLUGGED = _ROW["Vehicle_Plugged"]
R_VEHICLE_ENERGY = _ROW["Vehicle_Energy"]
R_VEHICLE_ENERGY_ONBOARD = _ROW["Vehicle_Energy_Onboard"]
R_VEHICLE_SOC_ABSOLUTE = _ROW["Vehicle_SOC_Absolute"]
R_VEHICLE_SOH = _ROW["Vehicle_SOH"]
R_VEHICLE_PV_ORIGIN_ENERGY_END = _ROW["Vehicle_PV_Origin_Energy_End"]
R_VEHICLE_GRID_ORIGIN_ENERGY_END = _ROW["Vehicle_Grid_Origin_Energy_End"]
L_PV_AC_TO_VEHICLE = _ROW["PV_AC_To_Vehicle"]
L_GRID_AC_TO_VEHICLE = _ROW["Grid_AC_To_Vehicle"]
L_VEHICLE_CHARGE_CONVERSION_LOSS = _ROW["Vehicle_Charge_Conversion_Loss"]
L_VEHICLE_CHARGE_INPUT = _ROW["Vehicle_Charge_Input"]
L_VEHICLE_CHARGE_LOSS = _ROW["Vehicle_Charge_Loss"]
L_VEHICLE_DISCHARGE_DC = _ROW["Vehicle_Discharge_DC"]
L_VEHICLE_DISCHARGE_LOSS = _ROW["Vehicle_Discharge_Loss"]
L_VEHICLE_CHARGER_LOSS = _ROW["Vehicle_Charger_Loss"]
L_VEHICLE_AC_TO_LOAD = _ROW["Vehicle_AC_To_Load"]
L_PV_ORIGIN_VEHICLE_AC_TO_LOAD = _ROW["PV_Origin_Vehicle_AC_To_Load"]
L_GRID_ORIGIN_VEHICLE_AC_TO_LOAD = _ROW["Grid_Origin_Vehicle_AC_To_Load"]
L_VEHICLE_DEPARTURE_ENERGY = _ROW["Vehicle_Departure_Energy"]
L_VEHICLE_ARRIVAL_ENERGY = _ROW["Vehicle_Arrival_Energy"]
L_VEHICLE_DEPARTURE_SHORTFALL = _ROW["Vehicle_Departure_Shortfall"]
L_VEHICLE_DRIVE_ENERGY = _ROW["Vehicle_Drive_Energy"]
L_VEHICLE_DRIVE_UNSERVED = _ROW["Vehicle_Drive_Unserved"]
L_VEHICLE_AWAY_CHARGE_PURCHASED = _ROW["Vehicle_Away_Charge_Purchased"]
L_VEHICLE_AWAY_CHARGE_LOSS = _ROW["Vehicle_Away_Charge_Loss"]
L_PV_ORIGIN_VEHICLE_DRIVE_ENERGY = _ROW["PV_Origin_Vehicle_Drive_Energy"]
L_GRID_ORIGIN_VEHICLE_DRIVE_ENERGY = _ROW["Grid_Origin_Vehicle_Drive_Energy"]
L_VEHICLE_CAPACITY_WINDOW_LOSS = _ROW["Vehicle_Capacity_Window_Loss"]
L_VEHICLE_REPLACEMENT_ENERGY_REMOVED = _ROW["Vehicle_Replacement_Energy_Removed"]
L_VEHICLE_REPLACEMENT_ENERGY_ADDED = _ROW["Vehicle_Replacement_Energy_Added"]
L_PV_ORIGIN_VEHICLE_CAPACITY_WINDOW_LOSS = _ROW["PV_Origin_Vehicle_Capacity_Window_Loss"]
L_GRID_ORIGIN_VEHICLE_CAPACITY_WINDOW_LOSS = _ROW["Grid_Origin_Vehicle_Capacity_Window_Loss"]
L_PV_ORIGIN_VEHICLE_REPLACEMENT_ENERGY_REMOVED = _ROW["PV_Origin_Vehicle_Replacement_Energy_Removed"]
L_GRID_ORIGIN_VEHICLE_REPLACEMENT_ENERGY_REMOVED = _ROW["Grid_Origin_Vehicle_Replacement_Energy_Removed"]
L_PV_ORIGIN_VEHICLE_CHARGE_STORED = _ROW["PV_Origin_Vehicle_Charge_Stored"]
L_GRID_ORIGIN_VEHICLE_CHARGE_STORED = _ROW["Grid_Origin_Vehicle_Charge_Stored"]
L_PV_ORIGIN_VEHICLE_DISCHARGE_DC = _ROW["PV_Origin_Vehicle_Discharge_DC"]
L_GRID_ORIGIN_VEHICLE_DISCHARGE_DC = _ROW["Grid_Origin_Vehicle_Discharge_DC"]


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
    curve_ac: np.ndarray,
    curve_eff: np.ndarray,
    curve_row: float,
    cable_k: float,
) -> Tuple[float, float, float]:
    """Convert PV and battery DC at one shared inverter operating point.

    Returns ``(pv_ac, battery_ac, conversion_loss)``, splitting the AC output
    in proportion to each source's share of the DC input.
    """
    total_dc = pv_dc + battery_dc
    ac_power, conversion_loss, _clipping_dc = _dc_ac(
        total_dc, inv_cap_ac_wh, inv_eff, ac_output_scale, pow_two, curve_ac, curve_eff, curve_row, cable_k
    )
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
    curve_ac: np.ndarray,
    curve_eff: np.ndarray,
    curve_row: float,
    cable_k: float,
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

    pv_ac_max, pv_conversion_loss, pv_clipping_dc = _dc_ac(
        pv_dc, inv_cap_ac_wh, inv_eff, ac_output_scale, pow_two, curve_ac, curve_eff, curve_row, cable_k
    )

    if pv_ac_max >= load:
        pv_ac_to_load = load
        dc_to_load = _dc_for_ac(load, inv_cap_ac_wh, inv_eff, ac_output_scale, curve_ac, curve_eff, curve_row, cable_k)
        surplus_dc = max(0.0, pv_dc - dc_to_load)
        battery_energy, drawn = _charge(surplus_dc, battery_energy, emax, eff_charge, cap_charge_in_wh, cap_stored_wh)
        remaining_dc = surplus_dc - drawn
        direct_ac, direct_conversion_loss, direct_clipping_dc = _dc_ac(
            dc_to_load + remaining_dc,
            inv_cap_ac_wh,
            inv_eff,
            ac_output_scale,
            pow_two,
            curve_ac,
            curve_eff,
            curve_row,
            cable_k,
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
            # the (0, 1] bound keeps at or below the nameplate itself, less
            # what an AC cable takes at it.
            ceiling_ac = inv_cap_ac_wh
            if cable_k > 0.0:
                ceiling_ac = max(0.0, ceiling_ac - cable_k * ceiling_ac * ceiling_ac)
            target_total_ac = min(load, ceiling_ac * ac_output_scale)
            if discharge_allowed and available > 0.0 and eff_discharge > 0.0 and target_total_ac > pv_ac_max:
                total_dc_target = _dc_for_ac(
                    target_total_ac, inv_cap_ac_wh, inv_eff, ac_output_scale, curve_ac, curve_eff, curve_row, cable_k
                )
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
                        pv_dc,
                        battery_dc,
                        inv_cap_ac_wh,
                        inv_eff,
                        ac_output_scale,
                        pow_two,
                        curve_ac,
                        curve_eff,
                        curve_row,
                        cable_k,
                    )[1]
                    if unconstrained_battery_ac > cap_discharge_ac_wh:
                        lower = 0.0
                        upper = battery_dc
                        for _iteration in range(40):
                            midpoint = (lower + upper) / 2.0
                            midpoint_battery_ac = _combined_conversion(
                                pv_dc,
                                midpoint,
                                inv_cap_ac_wh,
                                inv_eff,
                                ac_output_scale,
                                pow_two,
                                curve_ac,
                                curve_eff,
                                curve_row,
                                cable_k,
                            )[1]
                            if midpoint_battery_ac < cap_discharge_ac_wh:
                                lower = midpoint
                            else:
                                upper = midpoint
                        battery_dc = upper

                pv_delivered_ac, delivered_ac, total_inverter_loss = _combined_conversion(
                    pv_dc,
                    battery_dc,
                    inv_cap_ac_wh,
                    inv_eff,
                    ac_output_scale,
                    pow_two,
                    curve_ac,
                    curve_eff,
                    curve_row,
                    cable_k,
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


def _battery_inverter_charge(
    ac: float, rating_wh: float, efficiency: float, pow_two: float, flat_ac: np.ndarray, flat_eff: np.ndarray
) -> float:
    """DC an AC-coupled battery inverter delivers for AC input ``ac`` (ADR 0004 C4).

    The PVWatts core run AC to DC: its input limit is the AC rating
    ``rating_wh`` and its DC output at most ``rating_wh * efficiency``. An
    infinite rating is the flat conversion ``ac * efficiency``.
    """
    return _dc_ac(ac, rating_wh * efficiency, efficiency, 1.0, pow_two, flat_ac, flat_eff, 0.0, 0.0)[0]


def _battery_inverter_charge_input(
    dc: float, rating_wh: float, efficiency: float, flat_ac: np.ndarray, flat_eff: np.ndarray
) -> float:
    """AC input :func:`_battery_inverter_charge` needs for DC output ``dc``, saturating at the rating."""
    return _dc_for_ac(dc, rating_wh * efficiency, efficiency, 1.0, flat_ac, flat_eff, 0.0, 0.0)


def _dispatch_ac_step(
    pv_dc: float,
    pv_ac: float,
    pv_conversion_loss: float,
    pv_clipping_dc: float,
    night: float,
    load: float,
    battery_energy: float,
    emax: float,
    eff_charge: float,
    eff_discharge: float,
    battery_inverter_wh: float,
    battery_inverter_eff: float,
    cap_charge_in_wh: float,
    cap_discharge_ac_wh: float,
    cap_stored_wh: float,
    pow_two: float,
    discharge_allowed: bool,
    discharge_floor: float,
    grid_target_energy: float,
    grid_import_cap_wh: float,
    flat_ac: np.ndarray,
    flat_eff: np.ndarray,
) -> Tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]:
    """Dispatch one AC-coupled timestep (ADR 0004 C5, C7); inputs and outputs are Wh.

    The PV inverter has already converted the step: ``pv_ac``,
    ``pv_conversion_loss`` and ``pv_clipping_dc`` are the PV-only run's
    conversion of ``pv_dc``, whatever the battery does. PV AC serves the load
    first, then serves idle PV inverters' draw, then charges the battery
    through its own inverter, then is exported. The battery serves the
    remaining deficit through the same inverter, rated ``battery_inverter_wh``
    (AC, per step; infinite for flat conversion) at nominal efficiency
    ``battery_inverter_eff``. Grid charge
    then runs as in the DC-coupled step, but through the battery inverter:
    PV and grid AC entering it in one step convert at one operating point,
    and the DC is split in proportion to each source's AC.

    ``night`` is the PV inverters' whole draw this step (ADR 0004 M9).
    PV surplus serves it before charging; only the remainder is imported,
    whatever the battery does. It is part of ``grid_import`` before grid
    charge takes what is left of the site limit.

    ``flat_ac`` and ``flat_eff`` are empty curve arrays, which select the
    PVWatts curve for the battery inverter.

    Returns the sixteen values of :func:`_dispatch_dc_step`, with
    ``pv_dc_to_battery`` always 0, followed by ``pv_ac_to_battery`` and
    ``pv_dc_charge``, the DC that PV AC became in the battery inverter.
    ``battery_charge_input`` is ``pv_dc_charge + grid_dc``.
    """
    battery_discharge_dc = 0.0
    battery_ac_to_load = 0.0
    battery_discharge_loss = 0.0
    battery_inverter_loss = 0.0
    pv_ac_to_battery = 0.0
    pv_dc_charge = 0.0
    grid_ac = 0.0
    grid_dc = 0.0
    hold_target = discharge_allowed and not math.isnan(grid_target_energy) and discharge_floor >= grid_target_energy

    pv_ac_to_load = min(pv_ac, load)
    surplus_ac = pv_ac - pv_ac_to_load
    deficit = load - pv_ac_to_load
    pv_to_night = min(surplus_ac, night)
    surplus_ac = surplus_ac - pv_to_night
    energy_before_charge = battery_energy

    # 1. PV surplus charges through the battery inverter.
    room = max(0.0, emax - battery_energy)
    if surplus_ac > 0.0 and room > 0.0 and eff_charge > 0.0:
        dc_wanted = min(room / eff_charge, cap_charge_in_wh, cap_stored_wh / eff_charge)
        if dc_wanted > 0.0:
            pv_ac_to_battery = min(
                surplus_ac,
                _battery_inverter_charge_input(dc_wanted, battery_inverter_wh, battery_inverter_eff, flat_ac, flat_eff),
                battery_inverter_wh,
            )
            pv_dc_charge = min(
                dc_wanted,
                _battery_inverter_charge(
                    pv_ac_to_battery, battery_inverter_wh, battery_inverter_eff, pow_two, flat_ac, flat_eff
                ),
            )
            battery_energy = battery_energy + pv_dc_charge * eff_charge
    pv_ac_export = surplus_ac - pv_ac_to_battery

    # 2. The battery serves the deficit through the battery inverter. A step
    # with a surplus has no deficit, so it never both charges and discharges.
    available = max(0.0, battery_energy - discharge_floor)
    if deficit > 0.0 and discharge_allowed and available > 0.0 and eff_discharge > 0.0:
        ac_target = min(deficit, cap_discharge_ac_wh, battery_inverter_wh)
        battery_dc = min(
            available * eff_discharge,
            _dc_for_ac(ac_target, battery_inverter_wh, battery_inverter_eff, 1.0, flat_ac, flat_eff, 0.0, 0.0),
            cap_stored_wh * eff_discharge,
        )
        battery_ac_to_load = min(
            ac_target,
            _dc_ac(battery_dc, battery_inverter_wh, battery_inverter_eff, 1.0, pow_two, flat_ac, flat_eff, 0.0, 0.0)[0],
        )
        draw = battery_dc / eff_discharge
        if hold_target:
            # Land exactly on a binding floor, as the DC-coupled step does.
            if battery_dc == available * eff_discharge:
                draw = available
            draw = min(draw, available)
            battery_energy = discharge_floor if draw == available else max(discharge_floor, battery_energy - draw)
        else:
            battery_energy -= draw
        battery_discharge_dc = draw
        battery_discharge_loss = draw - battery_dc
        battery_inverter_loss = battery_dc - battery_ac_to_load
    grid_import = max(0.0, deficit - battery_ac_to_load)
    if night > 0.0:
        grid_import = grid_import + (night - pv_to_night)

    # 3. Grid charge toward the target, with what PV left of every shared
    # limit, including the battery inverter's AC input (C7).
    if not math.isnan(grid_target_energy) and pv_ac_export <= 0.0 and battery_discharge_dc <= 0.0 and eff_charge > 0.0:
        goal = min(grid_target_energy, emax)
        if goal - battery_energy > 0.0:
            dc_needed = (goal - energy_before_charge) / eff_charge
            dc_cap = min(cap_charge_in_wh, cap_stored_wh / eff_charge)
            # The inverse saturates at the rating, so a need at or above the
            # rating's DC output is not reachable this step.
            if dc_needed < battery_inverter_wh * battery_inverter_eff:
                required_ac = (
                    _battery_inverter_charge_input(
                        dc_needed, battery_inverter_wh, battery_inverter_eff, flat_ac, flat_eff
                    )
                    - pv_ac_to_battery
                )
            else:
                required_ac = math.inf
            grid_ac = min(
                required_ac,
                _battery_inverter_charge_input(dc_cap, battery_inverter_wh, battery_inverter_eff, flat_ac, flat_eff)
                - pv_ac_to_battery,
                battery_inverter_wh - pv_ac_to_battery,
                grid_import_cap_wh - grid_import,
            )
            if grid_ac > 0.0:
                total_ac = pv_ac_to_battery + grid_ac
                dc_total = _battery_inverter_charge(
                    total_ac, battery_inverter_wh, battery_inverter_eff, pow_two, flat_ac, flat_eff
                )
                grid_dc = dc_total * grid_ac / total_ac
                pv_dc_charge = dc_total - grid_dc
                battery_energy = energy_before_charge + dc_total * eff_charge
                if hold_target and grid_ac == required_ac:
                    battery_energy = goal
                grid_import = grid_import + grid_ac
            else:
                grid_ac = 0.0

    battery_charge_input = pv_dc_charge + grid_dc
    return (
        battery_energy,
        0.0,
        pv_dc - pv_clipping_dc,
        pv_clipping_dc,
        pv_ac_to_load,
        pv_ac_export,
        battery_charge_input,
        battery_discharge_dc,
        battery_ac_to_load,
        battery_charge_input * (1.0 - eff_charge),
        battery_discharge_loss,
        pv_conversion_loss,
        battery_inverter_loss,
        grid_import,
        grid_ac,
        grid_dc,
        pv_ac_to_battery,
        pv_dc_charge,
    )


def _dispatch_vehicle_step(
    energy: float,
    surplus: float,
    deficit: float,
    site_room: float,
    steps_to_departure: int,
    mode: int,
    target: float,
    ceiling: float,
    floor: float,
    rating_wh: float,
    charger_eff: float,
    eff_c: float,
    eff_d: float,
    s_max: float,
    charge_period: bool,
    discharge_period: bool,
    pow_two: float,
    flat_ac: np.ndarray,
    flat_eff: np.ndarray,
) -> tuple[float, float, float, float, float, float, float, float, float, float]:
    """Home dispatch in Wh, after the stationary store (ADR 0005 V4/V5).

    Modes are 0 uncontrolled, 1 V1G, 2 V2H. A zero departure countdown
    means no scheduled departure: PV/period charging may still run, but
    neither mandatory charging nor the discharge deadline binds.

    Returns ending energy, PV AC in, grid AC in, DC charge input, battery
    charge loss, stored-energy discharge draw, battery discharge loss,
    charger discharge loss, AC to load, and PV-origin charge stored.
    """
    need = max(0.0, target - energy)
    future = max(0, steps_to_departure - 1) * s_max
    mandatory = (
        min(s_max, need) if mode == 0 else (min(s_max, max(0.0, need - future)) if steps_to_departure > 0 else 0.0)
    )
    room = max(0.0, (target if mode == 0 else ceiling) - energy)
    room_ac = _battery_inverter_charge_input(room / eff_c, rating_wh, charger_eff, flat_ac, flat_eff)
    pv_ac = min(surplus, rating_wh, room_ac)
    full_charge = rating_wh * charger_eff * eff_c
    desired = min(room, full_charge, max(mandatory, need if charge_period else 0.0))
    total_ac_needed = _battery_inverter_charge_input(desired / eff_c, rating_wh, charger_eff, flat_ac, flat_eff)
    grid_needed = max(0.0, total_ac_needed - pv_ac)
    grid_ac = min(grid_needed, max(0.0, rating_wh - pv_ac), site_room)
    ac_in = pv_ac + grid_ac
    dc_in = _battery_inverter_charge(ac_in, rating_wh, charger_eff, pow_two, flat_ac, flat_eff)
    stored = dc_in * eff_c
    # One charger operating point for both sources, including its start-up loss.
    pv_stored = stored * pv_ac / ac_in if ac_in > 0.0 else 0.0
    draw = 0.0
    discharge_loss = 0.0
    charger_loss = 0.0
    ac_out = 0.0
    if ac_in > 0.0:
        end = energy + stored
        # Reached target and capacity bounds are exact, like hold_target.
        bound = target if mode == 0 else ceiling
        # Check the supplied components, not their rounded sum: PV + (AC
        # request - PV) may be one ulp below the request even when neither
        # source was limited. Such a reached target must still land exactly.
        room_met = pv_ac >= room_ac or grid_ac >= max(0.0, room_ac - pv_ac)
        if stored >= room or (room <= full_charge and room_met):
            end = bound
        elif desired > 0.0 and pv_ac <= total_ac_needed and grid_ac >= grid_needed:
            end = target if desired >= need else energy + desired
        # Derive the stored amount after snapping so origins close in Wh.
        stored = end - energy
        pv_stored = stored * pv_ac / ac_in
        energy = end
    elif mode == 2 and discharge_period:
        guard = target - future if steps_to_departure > 0 else floor
        bound = max(floor, guard)
        available = max(0.0, energy - bound)
        ac_target = min(deficit, rating_wh)
        dc_needed = _dc_for_ac(ac_target, rating_wh, charger_eff, 1.0, flat_ac, flat_eff, 0.0, 0.0)
        dc_out = min(available * eff_d, dc_needed)
        draw = dc_out / eff_d
        ac_out = min(ac_target, _dc_ac(dc_out, rating_wh, charger_eff, 1.0, pow_two, flat_ac, flat_eff, 0.0, 0.0)[0])
        end = energy - draw
        if available > 0.0 and dc_out >= available * eff_d:
            end = bound
            draw = energy - end
        energy = end
        discharge_loss = draw - dc_out
        charger_loss = dc_out - ac_out
    return energy, pv_ac, grid_ac, dc_in, dc_in - stored, draw, discharge_loss, charger_loss, ac_out, pv_stored


def _vehicle_drive(
    energy: float,
    pv_origin: float,
    grid_origin: float,
    requested: float,
    floor: float,
) -> tuple[float, float, float, float, float, float]:
    """Withdraw one driving leg in Wh, proportionally, never below min_soc."""
    served = min(requested, max(0.0, energy - floor))
    pv_used = served * (pv_origin / energy) if energy > 0.0 else 0.0
    grid_used = served * (grid_origin / energy) if energy > 0.0 else 0.0
    # A reserve is a withdrawal limit, never a source of energy.
    end = energy - served
    return end, max(0.0, pv_origin - pv_used), max(0.0, grid_origin - grid_used), served, pv_used, grid_used


def _dispatch_vehicle_away_step(
    energy: float,
    pv_origin: float,
    grid_origin: float,
    floor: float,
    drive_first: float,
    park_hours: float,
    drive_second: float,
    away_target: float,
    away_power_w: float,
    away_efficiency: float,
) -> tuple[float, float, float, float, float, float, float, float, float]:
    """Drive, charge while parked, then drive; away purchases never cross the site meter."""
    energy, pv_origin, grid_origin, first, pv_first, grid_first = _vehicle_drive(
        energy,
        pv_origin,
        grid_origin,
        drive_first,
        floor,
    )
    room = max(0.0, away_target - energy)
    purchased = min(away_power_w * park_hours, room / away_efficiency)
    stored = purchased * away_efficiency
    end = energy + stored
    if room > 0.0 and purchased >= room / away_efficiency:
        end = away_target
    stored = end - energy
    grid_origin += stored
    energy, pv_origin, grid_origin, second, pv_second, grid_second = _vehicle_drive(
        end,
        pv_origin,
        grid_origin,
        drive_second,
        floor,
    )
    served = first + second
    return (
        energy,
        pv_origin,
        grid_origin,
        served,
        max(0.0, drive_first - first) + max(0.0, drive_second - second),
        purchased,
        purchased - stored,
        pv_first + pv_second,
        grid_first + grid_second,
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
    curve_ac: np.ndarray,
    curve_eff: np.ndarray,
    curve_rows: np.ndarray,
    ac_derate: np.ndarray,
    cable_k: float,
    night_wh: float,
    ac_coupled: bool,
    battery_inverter_wh: float,
    battery_inverter_eff: float,
    pv_chain_ac: np.ndarray,
    pv_chain_loss: np.ndarray,
    pv_chain_clip: np.ndarray,
    pv_chain_night: np.ndarray,
    stationary_active: bool,
    vehicle_enabled: bool,
    vehicle_energy: float,
    vehicle_pv_origin: float,
    vehicle_grid_origin: float,
    vehicle_mode: int,
    vehicle_capacity: float,
    vehicle_target: float,
    vehicle_ceiling: float,
    vehicle_floor: float,
    vehicle_rating_wh: float,
    vehicle_charger_eff: float,
    vehicle_eff_c: float,
    vehicle_eff_d: float,
    vehicle_import_cap_wh: float,
    vehicle_plugged: np.ndarray,
    vehicle_countdown: np.ndarray,
    vehicle_charge_period: np.ndarray,
    vehicle_discharge_period: np.ndarray,
    vehicle_was_plugged: bool,
    vehicle_minimum: float,
    vehicle_drive_first: np.ndarray,
    vehicle_park_hours: np.ndarray,
    vehicle_drive_second: np.ndarray,
    vehicle_away_target: float,
    vehicle_away_power_w: float,
    vehicle_away_efficiency: float,
    vehicle_soh_fraction: float,
    vehicle_retired: bool,
) -> None:
    """Dispatch timesteps ``[lo, hi)`` at fixed health, writing rows of *matrix*.

    State of health, resistance-derived efficiencies and the replacement
    decision are unchanged for the duration of the call; the caller advances
    them at the day boundary. Without a stationary pack (or with a retired
    one), stationary_active=False retains the vectorised house rows and
    dispatches only the vehicle. Without a vehicle PV-only stays vectorised.

    ``discharge_allowed``, ``reserve_fraction`` and ``grid_target_fraction``
    are the per-step instruction arrays (ADR 0002), indexed like the inputs.
    A fraction is of each step's usable window, ``emin + f * (emax - emin)``,
    so the energy it names moves with temperature and health (A7).

    ``curve_ac``, ``curve_eff`` and ``cable_k`` are the inverter's conversion
    (:func:`_dc_ac`), with ``curve_rows`` each step's operating voltage as a
    position among the curve's rows. ``ac_derate`` is each step's AC rating
    as a fraction of ``inv_cap_ac_wh``; both are indexed like the inputs.
    ``night_wh`` is what the inverter draws from the grid in a step in which
    it converts nothing: no PV DC and no battery charge or discharge.

    ``ac_coupled`` selects :func:`_dispatch_ac_step` for every step (ADR
    0004). The PV inverter's conversion then comes from the ``pv_chain_*``
    arrays, the PV-only run's AC output, conversion loss, clipped DC and
    night draw in Wh per step, indexed like the inputs; the inverter
    arguments above are not read. ``battery_inverter_wh`` is the battery
    inverter's AC rating per step, infinite for none, at nominal efficiency
    ``battery_inverter_eff``. A DC-coupled run passes empty chain arrays.

    Nothing is returned: the state the next window and the day-close
    replacement path need is the final step's row of *matrix*.
    """
    matrix[R_VEHICLE_PLUGGED : R_VEHICLE_GRID_ORIGIN_ENERGY_END + 1, lo:hi] = 0.0
    matrix[L_PV_AC_TO_VEHICLE:, lo:hi] = 0.0
    soh_percent = soh_fraction * 100.0
    # Empty curve arrays of the right types: the battery inverter's PVWatts curve.
    flat_ac = curve_ac[:0]
    flat_eff = curve_eff[:0]
    vehicle_s_max = (
        _battery_inverter_charge(
            min(vehicle_rating_wh, vehicle_import_cap_wh),
            vehicle_rating_wh,
            vehicle_charger_eff,
            pow_two,
            flat_ac,
            flat_eff,
        )
        * vehicle_eff_c
        if vehicle_enabled
        else 0.0
    )
    for i in range(lo, hi):
        if stationary_active:
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
                min(1.0, max(0.0, origin_before_dispatch / energy_before_dispatch))
                if energy_before_dispatch > 0.0
                else 0.0
            )
            grid_fraction = (
                min(1.0, max(0.0, grid_before_dispatch / energy_before_dispatch))
                if energy_before_dispatch > 0.0
                else 0.0
            )
            if ac_coupled:
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
                    pv_ac_to_battery,
                    pv_dc_charge,
                ) = _dispatch_ac_step(
                    pv_dc_power,
                    pv_chain_ac[i],
                    pv_chain_loss[i],
                    pv_chain_clip[i],
                    pv_chain_night[i],
                    load,
                    battery_energy,
                    emax,
                    eff_charge,
                    eff_discharge,
                    battery_inverter_wh,
                    battery_inverter_eff,
                    cap_charge_in_wh,
                    cap_discharge_ac_wh,
                    cap_stored_wh,
                    pow_two,
                    discharge_allowed[i],
                    discharge_floor,
                    grid_target_energy,
                    grid_import_cap_wh,
                    flat_ac,
                    flat_eff,
                )
                pv_charge_conversion_loss = pv_ac_to_battery - pv_dc_charge
            else:
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
                    inv_cap_ac_wh * ac_derate[i],
                    ac_output_scale,
                    cap_stored_wh,
                    pow_two,
                    discharge_allowed[i],
                    discharge_floor,
                    grid_target_energy,
                    grid_eff,
                    grid_import_cap_wh,
                    curve_ac,
                    curve_eff,
                    curve_rows[i],
                    cable_k,
                )
                pv_ac_to_battery = 0.0
                pv_charge_conversion_loss = 0.0
                # The DC PV put into the battery: PV DC itself when DC-coupled.
                pv_dc_charge = pv_dc_to_battery
            if battery_charge_input > 0.0 and battery_discharge_dc > 0.0:
                raise ValueError("a dispatch step both charged and discharged the battery")
            night_consumption = 0.0
            if ac_coupled:
                # Keep the whole draw; the AC step nets PV surplus against it
                # before charging and imports only the remainder (ADR 0004 M9).
                night_consumption = pv_chain_night[i]
            elif night_wh > 0.0 and pv_dc_power <= 0.0 and battery_charge_input <= 0.0 and battery_discharge_dc <= 0.0:
                night_consumption = night_wh
                grid_import += night_consumption
            charge_stored = battery_charge_input * eff_charge
            # Each source's charge adds to its own origin: PV DC to PV, grid to grid.
            pv_charge_stored = pv_dc_charge * eff_charge
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
            matrix[L_PV_AC_TO_BATTERY, i] = pv_ac_to_battery / hours_per_step
            matrix[L_PV_CHARGE_CONVERSION_LOSS, i] = pv_charge_conversion_loss / hours_per_step
            matrix[L_BATTERY_DISCHARGE_DC, i] = battery_discharge_dc / hours_per_step
            matrix[L_BATTERY_AC_TO_LOAD, i] = battery_ac_to_load / hours_per_step
            matrix[L_PV_DIRECT_INVERTER_LOSS, i] = pv_direct_inverter_loss / hours_per_step
            matrix[L_BATTERY_INVERTER_LOSS, i] = battery_inverter_loss / hours_per_step
            matrix[L_INVERTER_LOSS, i] = (pv_direct_inverter_loss + battery_inverter_loss) / hours_per_step
            matrix[L_INVERTER_NIGHT_CONSUMPTION, i] = night_consumption / hours_per_step
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

        if vehicle_enabled:
            plugged = vehicle_plugged[i]
            # Boundaries are booked before any home or away operation. The
            # previous plug state is carried across windows and project years.
            if vehicle_was_plugged and not plugged:
                matrix[L_VEHICLE_DEPARTURE_ENERGY, i] = vehicle_energy / hours_per_step
                shortfall = max(0.0, vehicle_target - vehicle_energy)
                if shortfall > 1e-9 * vehicle_capacity:
                    matrix[L_VEHICLE_DEPARTURE_SHORTFALL, i] = shortfall / hours_per_step
            elif not vehicle_was_plugged and plugged:
                matrix[L_VEHICLE_ARRIVAL_ENERGY, i] = vehicle_energy / hours_per_step
            if vehicle_retired:
                # A retired pack serves no trips and accepts no energy at
                # home or away. Availability still books site boundaries.
                if not plugged:
                    matrix[L_VEHICLE_DRIVE_UNSERVED, i] = (
                        vehicle_drive_first[i] + vehicle_drive_second[i]
                    ) / hours_per_step
            elif plugged:
                before = vehicle_energy
                pv_fraction = vehicle_pv_origin / before if before > 0.0 else 0.0
                grid_fraction = vehicle_grid_origin / before if before > 0.0 else 0.0
                surplus = matrix[L_PV_AC_EXPORT, i] * hours_per_step
                imported = matrix[R_GRID_IMPORT, i] * hours_per_step
                stationary_grid = matrix[L_GRID_AC_TO_BATTERY, i] * hours_per_step
                night = matrix[L_INVERTER_NIGHT_CONSUMPTION, i] * hours_per_step
                deficit = max(0.0, imported - stationary_grid - night)
                site_room = max(0.0, vehicle_import_cap_wh - imported)
                (
                    vehicle_energy,
                    pv_ac,
                    grid_ac_v,
                    dc_in,
                    charge_loss_v,
                    draw,
                    discharge_loss_v,
                    charger_loss_v,
                    ac_out,
                    pv_stored,
                ) = _dispatch_vehicle_step(
                    vehicle_energy,
                    surplus,
                    deficit,
                    site_room,
                    vehicle_countdown[i],
                    vehicle_mode,
                    vehicle_target,
                    vehicle_ceiling,
                    vehicle_floor,
                    vehicle_rating_wh,
                    vehicle_charger_eff,
                    vehicle_eff_c,
                    vehicle_eff_d,
                    vehicle_s_max,
                    vehicle_charge_period[i],
                    vehicle_discharge_period[i],
                    pow_two,
                    flat_ac,
                    flat_eff,
                )
                grid_stored = dc_in - charge_loss_v - pv_stored
                pv_draw = draw * pv_fraction
                grid_draw = draw * grid_fraction
                vehicle_pv_origin = min(vehicle_energy, max(0.0, vehicle_pv_origin - pv_draw + pv_stored))
                vehicle_grid_origin = min(
                    vehicle_energy - vehicle_pv_origin, max(0.0, vehicle_grid_origin - grid_draw + grid_stored)
                )
                # No arithmetic on an unchanged residual: Q3 also pins signed zeros.
                if pv_ac > 0.0:
                    matrix[L_PV_AC_EXPORT, i] = (surplus - pv_ac) / hours_per_step
                if grid_ac_v > 0.0 or ac_out > 0.0:
                    matrix[R_GRID_IMPORT, i] = (imported + grid_ac_v - ac_out) / hours_per_step
                matrix[L_PV_AC_TO_VEHICLE, i] = pv_ac / hours_per_step
                matrix[L_GRID_AC_TO_VEHICLE, i] = grid_ac_v / hours_per_step
                matrix[L_VEHICLE_CHARGE_CONVERSION_LOSS, i] = (pv_ac + grid_ac_v - dc_in) / hours_per_step
                matrix[L_VEHICLE_CHARGE_INPUT, i] = dc_in / hours_per_step
                matrix[L_VEHICLE_CHARGE_LOSS, i] = charge_loss_v / hours_per_step
                matrix[L_VEHICLE_DISCHARGE_DC, i] = draw / hours_per_step
                matrix[L_VEHICLE_DISCHARGE_LOSS, i] = discharge_loss_v / hours_per_step
                matrix[L_VEHICLE_CHARGER_LOSS, i] = charger_loss_v / hours_per_step
                matrix[L_VEHICLE_AC_TO_LOAD, i] = ac_out / hours_per_step
                matrix[L_PV_ORIGIN_VEHICLE_AC_TO_LOAD, i] = (ac_out * pv_fraction) / hours_per_step
                matrix[L_GRID_ORIGIN_VEHICLE_AC_TO_LOAD, i] = (ac_out * grid_fraction) / hours_per_step
                matrix[L_PV_ORIGIN_VEHICLE_CHARGE_STORED, i] = pv_stored / hours_per_step
                matrix[L_GRID_ORIGIN_VEHICLE_CHARGE_STORED, i] = grid_stored / hours_per_step
                matrix[L_PV_ORIGIN_VEHICLE_DISCHARGE_DC, i] = pv_draw / hours_per_step
                matrix[L_GRID_ORIGIN_VEHICLE_DISCHARGE_DC, i] = grid_draw / hours_per_step
            else:
                (
                    vehicle_energy,
                    vehicle_pv_origin,
                    vehicle_grid_origin,
                    drive_served,
                    drive_unserved,
                    away_purchased,
                    away_loss,
                    pv_drive,
                    grid_drive,
                ) = _dispatch_vehicle_away_step(
                    vehicle_energy,
                    vehicle_pv_origin,
                    vehicle_grid_origin,
                    vehicle_minimum,
                    vehicle_drive_first[i],
                    vehicle_park_hours[i],
                    vehicle_drive_second[i],
                    vehicle_away_target,
                    vehicle_away_power_w,
                    vehicle_away_efficiency,
                )
                matrix[L_VEHICLE_DRIVE_ENERGY, i] = drive_served / hours_per_step
                matrix[L_VEHICLE_DRIVE_UNSERVED, i] = drive_unserved / hours_per_step
                matrix[L_VEHICLE_AWAY_CHARGE_PURCHASED, i] = away_purchased / hours_per_step
                matrix[L_VEHICLE_AWAY_CHARGE_LOSS, i] = away_loss / hours_per_step
                matrix[L_PV_ORIGIN_VEHICLE_DRIVE_ENERGY, i] = pv_drive / hours_per_step
                matrix[L_GRID_ORIGIN_VEHICLE_DRIVE_ENERGY, i] = grid_drive / hours_per_step
            vehicle_was_plugged = plugged
            matrix[R_VEHICLE_PLUGGED, i] = 1.0 if plugged else 0.0
            matrix[R_VEHICLE_ENERGY, i] = vehicle_energy if plugged else 0.0
            matrix[R_VEHICLE_ENERGY_ONBOARD, i] = vehicle_energy
            matrix[R_VEHICLE_SOC_ABSOLUTE, i] = vehicle_energy / vehicle_capacity if vehicle_capacity > 0.0 else 0.0
            matrix[R_VEHICLE_SOH, i] = vehicle_soh_fraction
            matrix[R_VEHICLE_PV_ORIGIN_ENERGY_END, i] = vehicle_pv_origin
            matrix[R_VEHICLE_GRID_ORIGIN_ENERGY_END, i] = vehicle_grid_origin


def _dispatch_day_candidates(
    matrix: np.ndarray,
    pv_dc_vals: np.ndarray,
    load_vals: np.ndarray,
    temp_vals: np.ndarray,
    lo: int,
    hi: int,
    start_energies: np.ndarray,
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
    reserve_levels: np.ndarray,
    target_levels: np.ndarray,
    grid_eff: float,
    grid_import_cap_wh: float,
    curve_ac: np.ndarray,
    curve_eff: np.ndarray,
    curve_rows: np.ndarray,
    ac_derate: np.ndarray,
    cable_k: float,
    night_wh: float,
    ac_coupled: bool,
    battery_inverter_wh: float,
    battery_inverter_eff: float,
    pv_chain_ac: np.ndarray,
    pv_chain_loss: np.ndarray,
    pv_chain_clip: np.ndarray,
    pv_chain_night: np.ndarray,
    stationary_active: bool,
    vehicle_enabled: bool,
    vehicle_energy: float,
    vehicle_pv_origin: float,
    vehicle_grid_origin: float,
    vehicle_mode: int,
    vehicle_capacity: float,
    vehicle_target: float,
    vehicle_ceiling: float,
    vehicle_floor: float,
    vehicle_rating_wh: float,
    vehicle_charger_eff: float,
    vehicle_eff_c: float,
    vehicle_eff_d: float,
    vehicle_import_cap_wh: float,
    vehicle_plugged: np.ndarray,
    vehicle_countdown: np.ndarray,
    vehicle_charge_period: np.ndarray,
    vehicle_discharge_period: np.ndarray,
    vehicle_was_plugged: bool,
    vehicle_minimum: float,
    vehicle_drive_first: np.ndarray,
    vehicle_park_hours: np.ndarray,
    vehicle_drive_second: np.ndarray,
    vehicle_away_target: float,
    vehicle_away_power_w: float,
    vehicle_away_efficiency: float,
    vehicle_soh_fraction: float,
    vehicle_retired: bool,
    candidate_levels: np.ndarray,
    out_import: np.ndarray,
    out_export: np.ndarray,
    out_discharge: np.ndarray,
    out_end: np.ndarray,
) -> None:
    """Run :func:`_dispatch_day` once per candidate of one planner day.

    The arguments are :func:`_dispatch_day`'s, as :func:`_day_arguments`
    packs them, except three: ``start_energies`` is each candidate's start
    energy, and ``reserve_levels`` and ``target_levels`` hold one row of
    instructions per target level, ``candidate_levels`` naming each
    candidate's row. Every candidate dispatches steps ``[lo, hi)`` of
    *matrix* from scratch, as one call of :func:`_dispatch_day` would, and
    leaves its grid import, PV export and battery DC discharge rows in row
    ``c`` of ``out_import``, ``out_export`` and ``out_discharge`` (columns
    ``[0, hi - lo)``) and its end energy in ``out_end[c]``.

    One call per planner day instead of one per candidate is the point: the
    day's inputs are packed and handed over once. The day loop is the same
    function, so a candidate costs what it did, bit for bit.
    """
    n = hi - lo
    for c in range(len(start_energies)):
        level = candidate_levels[c]
        _dispatch_day(
            matrix,
            pv_dc_vals,
            load_vals,
            temp_vals,
            lo,
            hi,
            start_energies[c],
            pv_origin,
            grid_origin,
            nominal_energy_wh,
            soh_fraction,
            max_soc,
            min_soc,
            standby_loss_per_step_wh,
            eff_charge,
            eff_discharge,
            inv_eff,
            cap_charge_in_wh,
            cap_discharge_ac_wh,
            inv_cap_ac_wh,
            cap_stored_wh,
            thermal_resistance_k_per_w,
            hours_per_step,
            ac_output_scale,
            pow_two,
            discharge_allowed,
            reserve_levels[level],
            target_levels[level],
            grid_eff,
            grid_import_cap_wh,
            curve_ac,
            curve_eff,
            curve_rows,
            ac_derate,
            cable_k,
            night_wh,
            ac_coupled,
            battery_inverter_wh,
            battery_inverter_eff,
            pv_chain_ac,
            pv_chain_loss,
            pv_chain_clip,
            pv_chain_night,
            stationary_active,
            vehicle_enabled,
            vehicle_energy,
            vehicle_pv_origin,
            vehicle_grid_origin,
            vehicle_mode,
            vehicle_capacity,
            vehicle_target,
            vehicle_ceiling,
            vehicle_floor,
            vehicle_rating_wh,
            vehicle_charger_eff,
            vehicle_eff_c,
            vehicle_eff_d,
            vehicle_import_cap_wh,
            vehicle_plugged,
            vehicle_countdown,
            vehicle_charge_period,
            vehicle_discharge_period,
            vehicle_was_plugged,
            vehicle_minimum,
            vehicle_drive_first,
            vehicle_park_hours,
            vehicle_drive_second,
            vehicle_away_target,
            vehicle_away_power_w,
            vehicle_away_efficiency,
            vehicle_soh_fraction,
            vehicle_retired,
        )
        out_import[c, :n] = matrix[R_GRID_IMPORT, lo:hi]
        out_export[c, :n] = matrix[L_PV_AC_EXPORT, lo:hi]
        out_discharge[c, :n] = matrix[L_BATTERY_DISCHARGE_DC, lo:hi]
        out_end[c] = matrix[R_BATTERY_ENERGY, hi - 1]


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
    ac_derate: Optional[np.ndarray] = None,
    curve_rows: Optional[np.ndarray] = None,
    pv_chain: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
    stationary_active: bool = True,
    vehicle_config: Optional["VehicleConfig"] = None,
    vehicle_schedule: Optional["VehicleSchedule"] = None,
    vehicle_energy_wh: float = 0.0,
    vehicle_pv_origin_wh: float = 0.0,
    vehicle_grid_origin_wh: float = 0.0,
    vehicle_charge_period: Optional[np.ndarray] = None,
    vehicle_discharge_period: Optional[np.ndarray] = None,
    grid_import_limit_w: float = math.inf,
    vehicle_was_plugged: bool = True,
    vehicle_soh_fraction: float = 1.0,
    vehicle_retired: bool = False,
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

    ``ac_derate`` is the inverter's AC rating per step as a fraction of
    ``cap_wh``, over the whole run like ``instructions``; None is the full
    rating. ``curve_rows`` is each step's position among the efficiency
    curve's voltage rows, likewise; None holds the conversion's own row.
    ``battery_config.inverter_conversion`` supplies the efficiency curve, the
    AC cable and the night consumption.

    An AC-coupled ``battery_config`` needs ``pv_chain``: the PV inverter's
    AC output, conversion loss, clipped DC and night draw in Wh per step over
    the whole run, as the PV-only path computes them (ADR 0004 C3, M2).
    Its grid charge passes through the battery inverter, so its instructions
    carry no ``grid_charge_efficiency``; the kernels receive NaN. A
    DC-coupled one takes no chain and needs the efficiency.
    """
    ac_coupled = stationary_active and battery_config.coupling == "ac"
    if (
        stationary_active
        and instructions is not None
        and instructions.grid_charge_efficiency is None
        and not ac_coupled
    ):
        raise ValueError(
            "a DC-coupled battery needs a grid_charge_efficiency in its dispatch instructions; "
            "only an AC-coupled battery converts grid charge in its own inverter"
        )
    if ac_coupled:
        if pv_chain is None:
            raise ValueError("an AC-coupled battery needs the PV inverter's conversion, pv_chain")
        if any(len(array) != len(pv_dc_vals) for array in pv_chain):
            raise ValueError("pv_chain must cover the same steps as pv_dc_vals")
        chain = pv_chain
        battery_inverter_wh = (
            math.inf
            if battery_config.battery_inverter_ac_capacity_w is None
            else battery_config.battery_inverter_ac_capacity_w * hours_per_step
        )
        assert battery_config.battery_inverter_efficiency is not None
        battery_inverter_eff = battery_config.battery_inverter_efficiency
    else:
        if stationary_active and pv_chain is not None:
            raise ValueError("pv_chain is for an AC-coupled battery; a hybrid inverter converts PV in the step")
        chain = (np.empty(0), np.empty(0), np.empty(0), np.empty(0))
        battery_inverter_wh = math.inf
        battery_inverter_eff = 1.0
    if instructions is None:
        instructions = DispatchInstructions.noop(len(pv_dc_vals))
    if ac_derate is None:
        ac_derate = np.ones(len(pv_dc_vals))
    conversion = battery_config.conversion
    curve_ac, curve_eff, curve_row, cable_k = conversion.kernel_arguments(hours_per_step)
    if curve_rows is None:
        curve_rows = np.full(len(pv_dc_vals), curve_row)
    away = cast("Optional[AwayChargingConfig]", None if vehicle_config is None else vehicle_config.away_charging)
    vehicle_capacity_wh = (
        1.0 if vehicle_config is None else float(vehicle_config.battery_kwh * 1000.0 * vehicle_soh_fraction)
    )
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
        math.nan if instructions.grid_charge_efficiency is None else float(instructions.grid_charge_efficiency),
        float(instructions.grid_import_limit_w * hours_per_step),
        curve_ac,
        curve_eff,
        curve_rows,
        ac_derate,
        float(cable_k),
        float(conversion.night_consumption_w * hours_per_step),
        ac_coupled,
        float(battery_inverter_wh),
        float(battery_inverter_eff),
        *chain,
        stationary_active,
        vehicle_config is not None,
        float(vehicle_energy_wh),
        float(vehicle_pv_origin_wh),
        float(vehicle_grid_origin_wh),
        0 if vehicle_config is None else {"uncontrolled": 0, "v1g": 1, "v2h": 2}[vehicle_config.mode],
        float(vehicle_capacity_wh),
        0.0 if vehicle_config is None else float(vehicle_config.departure_soc * vehicle_capacity_wh),
        0.0 if vehicle_config is None else float(vehicle_config.max_soc * vehicle_capacity_wh),
        0.0
        if vehicle_config is None
        else float(max(vehicle_config.min_soc, vehicle_config.v2h_min_soc or 0.0) * vehicle_capacity_wh),
        1.0 if vehicle_config is None else float(vehicle_config.charger_ac_rating_kw * 1000.0 * hours_per_step),
        1.0 if vehicle_config is None else float(vehicle_config.charger_efficiency),
        1.0 if vehicle_config is None else float(vehicle_config.charge_efficiency),
        1.0 if vehicle_config is None else float(vehicle_config.discharge_efficiency),
        float(min(grid_import_limit_w, instructions.grid_import_limit_w) * hours_per_step),
        _EMPTY_VEHICLE_MASK if vehicle_schedule is None else vehicle_schedule.plugged,
        _EMPTY_VEHICLE_COUNTDOWN if vehicle_schedule is None else vehicle_schedule.steps_to_departure,
        _EMPTY_VEHICLE_MASK if vehicle_charge_period is None else vehicle_charge_period,
        _EMPTY_VEHICLE_MASK if vehicle_discharge_period is None else vehicle_discharge_period,
        bool(vehicle_was_plugged),
        0.0 if vehicle_config is None else float(vehicle_config.min_soc * vehicle_capacity_wh),
        _EMPTY_VEHICLE_FLOAT if vehicle_schedule is None else vehicle_schedule.drive_first_wh,
        _EMPTY_VEHICLE_FLOAT if vehicle_schedule is None else vehicle_schedule.park_hours,
        _EMPTY_VEHICLE_FLOAT if vehicle_schedule is None else vehicle_schedule.drive_second_wh,
        0.0 if away is None else float(away.target_soc * vehicle_capacity_wh),
        0.0 if away is None else float(away.power_kw * 1000.0),
        1.0 if away is None else float(away.efficiency),
        float(vehicle_soh_fraction),
        bool(vehicle_retired),
    )


# Where _day_arguments puts the arguments a planner candidate changes.
_ARG_LO, _ARG_HI, _ARG_ENERGY, _ARG_RESERVE, _ARG_TARGET = 4, 5, 6, 26, 27


def _candidate_arguments(
    day_arguments: Tuple[Any, ...],
    lo: int,
    hi: int,
    start_energies: np.ndarray,
    reserve_levels: np.ndarray,
    target_levels: np.ndarray,
    candidate_levels: np.ndarray,
    outputs: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> Tuple[Any, ...]:
    """The arguments of :func:`_dispatch_day_candidates`, from one packing of :func:`_day_arguments`.

    ``day_arguments`` is packed once for the planner's whole window; only the
    day's bounds, the start energies and the per-level instructions differ
    from one call to the next.
    """
    args = list(day_arguments)
    args[_ARG_LO] = lo
    args[_ARG_HI] = hi
    args[_ARG_ENERGY] = start_energies
    args[_ARG_RESERVE] = reserve_levels
    args[_ARG_TARGET] = target_levels
    return (*args, candidate_levels, *outputs)


# Where _day_arguments puts the state a day of a plain span starts from.
_ARG_PV_ORIGIN, _ARG_GRID_ORIGIN, _ARG_SOH, _ARG_EFF_CHARGE, _ARG_EFF_DISCHARGE = 7, 8, 10, 14, 15


def _restate_day(
    arguments: List[Any],
    lo: int,
    hi: int,
    battery_energy_wh: float,
    pv_origin_wh: float,
    grid_origin_wh: float,
    soh_fraction: float,
    eff_charge: float,
    eff_discharge: float,
) -> None:
    """Put a day's bounds and starting state into a span's packing of :func:`_day_arguments`, in place.

    Over a span with no vehicle and no daily controller, these are the only
    arguments that change from one day to the next: health and efficiency
    after each day's aging, stored energy and its origins after each day's
    dispatch. Everything else is the span's, so the span packs once and
    restates the day here. The coercions are :func:`_day_arguments`' own,
    and ``tests/test_numba_dispatch_parity.py`` holds the restated packing
    to a fresh one.
    """
    arguments[_ARG_LO] = lo
    arguments[_ARG_HI] = hi
    arguments[_ARG_ENERGY] = float(battery_energy_wh)
    arguments[_ARG_PV_ORIGIN] = float(pv_origin_wh)
    arguments[_ARG_GRID_ORIGIN] = float(grid_origin_wh)
    arguments[_ARG_SOH] = float(soh_fraction)
    arguments[_ARG_EFF_CHARGE] = float(eff_charge)
    arguments[_ARG_EFF_DISCHARGE] = float(eff_discharge)


def _run_packed_python(arguments: Sequence[Any]) -> None:
    """Run :func:`_dispatch_day` as Python on a packing of :func:`_day_arguments`."""
    _dispatch_day(*arguments)


def _dispatch_day_python(out: Any, *args: Any, **state: Any) -> None:
    """Run :func:`_dispatch_day` as Python; the reference backend.

    Takes the arguments of :func:`_day_arguments`.
    """
    _run_packed_python(_day_arguments(out, *args, **state))
