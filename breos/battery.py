"""
Battery simulation module.

This module handles battery energy storage simulation including:
- Energy balance calculations
- State of Charge (SOC) tracking
- State of Health (SOH) degradation models (Naumann + Lam)
- Cycle and calendar aging
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import rainflow

from breos.constants import (
    A_Q,
    A_R,
    B_Q,
    B_R,
    C_DOC_Q,
    C_DOC_R,
    D_DOC_Q,
    D_DOC_R,
    DEFAULT_CHARGE_EFFICIENCY,
    DEFAULT_DISCHARGE_EFFICIENCY,
    DEFAULT_MAX_SOC,
    DEFAULT_MIN_SOC,
    DEFAULT_STANDBY_LOSS_WH,
    DEFAULT_THERMAL_RESISTANCE_KW,
    LAM_EA_J_MOL,
    LAM_EXPONENT_B,
    LAM_K0_FRAC,
    LAM_SOC_EXPONENT_N,
    LFP_CAP_DERATE_PER_C_COLD,
    LFP_CAP_DERATE_PER_C_MODERATE,
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
    NAUMANN_LAM_FIELD_CALIBRATED_V1_EA_J_MOL,
    NAUMANN_LAM_FIELD_CALIBRATED_V1_EXPONENT_B,
    NAUMANN_LAM_FIELD_CALIBRATED_V1_K0_FRAC,
    NAUMANN_LAM_FIELD_CALIBRATED_V1_SOC_EXPONENT_N,
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
from breos.economics import BATTERY_REPLACEMENT_COST_PER_KWH
from breos.inverter import calculate_dc_ac_power, dc_power_for_ac_output
from breos.utils import get_hours_per_step, remap_datetime_index_years

SUPPORTED_BATTERY_TYPES: tuple[str, ...] = ("lfp",)


@dataclass
class BatteryConfig:
    """
    Configuration parameters for battery simulation.

    Only DC-coupled systems (hybrid inverters) are modelled:
    - PV → Battery: No inverter loss (stays in DC)
    - Battery → Load: Inverter loss applies (DC to AC)

    AC-coupled dispatch is not implemented; ``dc_coupled=False`` raises.

    Power limits are nameplate powers and therefore scale with the timestep:
    ``max_charge_power_w`` limits DC input to the battery path, while
    ``max_discharge_power_w`` limits battery AC delivered to the load.

    ``eol_percentage`` defaults to 0.70 (replace the battery when its state
    of health falls to 70% of nominal capacity), matching the App config
    default ``battery_eol_percentage``.
    """

    nominal_energy_wh: float  # Required — nominal capacity in Wh
    initial_soh: float = 100.0  # Initial state of health (%)
    eol_percentage: float = 0.70  # End of life threshold (fraction)
    max_soc: float = DEFAULT_MAX_SOC
    min_soc: float = DEFAULT_MIN_SOC
    charge_efficiency: float = DEFAULT_CHARGE_EFFICIENCY
    discharge_efficiency: float = DEFAULT_DISCHARGE_EFFICIENCY
    standby_loss_wh: float = DEFAULT_STANDBY_LOSS_WH
    enable_replacement: bool = True
    replacement_cost: Optional[float] = None  # Auto-computed from cost per kWh if not set
    calendar_model: str = "naumann_lam_field_calibrated"  # v1 field-calibrated default alias
    # Resistance fade (opt-in): grows internal resistance daily and derates
    # the charge/discharge efficiencies in the energy loop so the effective
    # round-trip efficiency declines as the battery ages.
    enable_resistance_fade: bool = False  # Enable Naumann resistance growth model
    initial_resistance_growth: float = 0.0  # Initial relative resistance growth (fraction, 0=new)
    # Thermal model
    thermal_resistance_kw: float = DEFAULT_THERMAL_RESISTANCE_KW  # K/W for lumped thermal model
    # DC-coupled system (hybrid inverter) settings
    dc_coupled: bool = True  # True = hybrid inverter (DC-coupled battery)
    inverter_efficiency: float = 0.96  # Inverter efficiency (for DC→AC conversion)
    # Inverter AC rating (W) shared by PV and battery discharge; AC output is
    # clipped to this each step. None disables clipping (legacy behavior).
    inverter_ac_capacity_w: Optional[float] = None
    # Battery chemistry. The native degradation model is currently LFP-only;
    # unsupported values fail loudly instead of reusing LFP parameters.
    battery_type: str = "lfp"
    max_charge_power_w: Optional[float] = None
    max_discharge_power_w: Optional[float] = None

    def __post_init__(self):
        if not isinstance(self.dc_coupled, bool):
            raise ValueError("dc_coupled must be a bool")
        if not self.dc_coupled:
            raise NotImplementedError(
                "AC-coupled battery dispatch is not implemented. Only DC-coupled "
                "(hybrid inverter) systems are supported; set dc_coupled=True."
            )

        def finite(name: str, value: float) -> float:
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must be a finite number, not a bool")
            try:
                result = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a finite number") from exc
            if not math.isfinite(result):
                raise ValueError(f"{name} must be a finite number")
            return result

        self.nominal_energy_wh = finite("nominal_energy_wh", self.nominal_energy_wh)
        self.initial_soh = finite("initial_soh", self.initial_soh)
        self.eol_percentage = finite("eol_percentage", self.eol_percentage)
        self.min_soc = finite("min_soc", self.min_soc)
        self.max_soc = finite("max_soc", self.max_soc)
        self.charge_efficiency = finite("charge_efficiency", self.charge_efficiency)
        self.discharge_efficiency = finite("discharge_efficiency", self.discharge_efficiency)
        self.inverter_efficiency = finite("inverter_efficiency", self.inverter_efficiency)
        self.standby_loss_wh = finite("standby_loss_wh", self.standby_loss_wh)
        self.initial_resistance_growth = finite("initial_resistance_growth", self.initial_resistance_growth)
        self.thermal_resistance_kw = finite("thermal_resistance_kw", self.thermal_resistance_kw)

        if self.nominal_energy_wh < 0.0:
            raise ValueError("nominal_energy_wh must be non-negative")
        if not 0.0 <= self.initial_soh <= 100.0:
            raise ValueError("initial_soh must be between 0 and 100")
        if not 0.0 <= self.eol_percentage <= 1.0:
            raise ValueError("eol_percentage must be between 0 and 1")
        if not 0.0 <= self.min_soc < self.max_soc <= 1.0:
            raise ValueError("SOC limits must satisfy 0 <= min_soc < max_soc <= 1")
        for name in ("charge_efficiency", "discharge_efficiency", "inverter_efficiency"):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be greater than 0 and at most 1")
        for name in ("standby_loss_wh", "initial_resistance_growth", "thermal_resistance_kw"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.replacement_cost is not None:
            self.replacement_cost = finite("replacement_cost", self.replacement_cost)
            if self.replacement_cost < 0.0:
                raise ValueError("replacement_cost must be non-negative")

        for name in ("inverter_ac_capacity_w", "max_charge_power_w", "max_discharge_power_w"):
            value = getattr(self, name)
            if value is not None:
                if isinstance(value, (bool, np.bool_)):
                    raise ValueError(f"{name} must be a finite non-negative number or None")
                try:
                    value = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{name} must be a finite non-negative number or None") from exc
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(f"{name} must be a finite non-negative number or None")
                setattr(self, name, value)
        self.battery_type = _normalise_battery_type(self.battery_type)
        # Auto-compute replacement cost
        if self.replacement_cost is None:
            if self.nominal_energy_wh > 1:
                self.replacement_cost = BATTERY_REPLACEMENT_COST_PER_KWH * (self.nominal_energy_wh / 1000)
            else:
                self.replacement_cost = 0.0


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
    has_battery: bool,
) -> Tuple[float, Dict[str, float]]:
    """Dispatch one DC-coupled timestep; inputs, outputs and ledger are Wh.

    PV serves AC load first. Surplus DC then charges the battery before any
    export. PV and battery discharge share the inverter AC nameplate.
    """
    ledger = {
        "pv_dc_to_battery": 0.0,
        "pv_dc_to_inverter": 0.0,
        "pv_dc_curtailed": 0.0,
        "pv_ac_to_load": 0.0,
        "pv_ac_export": 0.0,
        "battery_charge_input": 0.0,
        "battery_discharge_dc": 0.0,
        "battery_ac_to_load": 0.0,
        "battery_charge_loss": 0.0,
        "battery_discharge_loss": 0.0,
        "pv_direct_inverter_loss": 0.0,
        "battery_inverter_loss": 0.0,
        "grid_import": 0.0,
    }
    pv_conversion = calculate_dc_ac_power(pv_dc, inv_cap_ac_wh, inv_eff)
    pv_ac_max = pv_conversion.ac_power_w

    def charge(surplus_dc: float) -> float:
        nonlocal battery_energy
        room = max(0.0, emax - battery_energy)
        if room <= 0.0 or eff_charge <= 0.0:
            return 0.0
        drawn = min(surplus_dc, room / eff_charge, cap_charge_in_wh)
        battery_energy += drawn * eff_charge
        ledger["pv_dc_to_battery"] = drawn
        ledger["battery_charge_input"] = drawn
        ledger["battery_charge_loss"] = drawn * (1.0 - eff_charge)
        return drawn

    if has_battery and pv_ac_max >= load:
        ledger["pv_ac_to_load"] = load
        dc_to_load = dc_power_for_ac_output(load, inv_cap_ac_wh, inv_eff)
        surplus_dc = max(0.0, pv_dc - dc_to_load)
        drawn = charge(surplus_dc)
        remaining_dc = surplus_dc - drawn
        direct_conversion = calculate_dc_ac_power(dc_to_load + remaining_dc, inv_cap_ac_wh, inv_eff)
        export_ac = max(0.0, direct_conversion.ac_power_w - load)
        dc_export = max(0.0, dc_to_load + remaining_dc - direct_conversion.clipping_loss_dc_w - dc_to_load)
        ledger["pv_ac_export"] = export_ac
        ledger["pv_dc_to_inverter"] = dc_to_load + dc_export
        ledger["pv_dc_curtailed"] = direct_conversion.clipping_loss_dc_w
        ledger["pv_direct_inverter_loss"] = direct_conversion.conversion_loss_w
    elif has_battery:
        ledger["pv_ac_to_load"] = pv_ac_max
        dc_to_inverter = pv_dc - pv_conversion.clipping_loss_dc_w
        ledger["pv_dc_to_inverter"] = dc_to_inverter
        ledger["pv_direct_inverter_loss"] = pv_conversion.conversion_loss_w
        excess_dc = pv_conversion.clipping_loss_dc_w
        deficit = load - pv_ac_max
        if excess_dc > 1e-12:
            # The inverter is saturated by PV. DC above its immediate AC
            # headroom may charge, but battery discharge has no AC headroom.
            drawn = charge(excess_dc)
            ledger["pv_dc_curtailed"] = excess_dc - drawn
            ledger["grid_import"] = deficit
        else:
            available = max(0.0, battery_energy - emin)
            target_total_ac = min(load, inv_cap_ac_wh)
            if available > 0.0 and eff_discharge > 0.0 and target_total_ac > pv_ac_max:
                total_dc_target = dc_power_for_ac_output(target_total_ac, inv_cap_ac_wh, inv_eff)
                battery_dc = min(available * eff_discharge, max(0.0, total_dc_target - pv_dc))

                def combined_conversion(battery_dc_input: float) -> tuple[float, float, float]:
                    total_dc = pv_dc + battery_dc_input
                    conversion = calculate_dc_ac_power(total_dc, inv_cap_ac_wh, inv_eff)
                    if total_dc <= 0.0:
                        return 0.0, 0.0, 0.0
                    battery_ac = conversion.ac_power_w * battery_dc_input / total_dc
                    pv_ac = conversion.ac_power_w - battery_ac
                    return pv_ac, battery_ac, conversion.conversion_loss_w

                # The public discharge limit is AC delivered. If it binds,
                # solve for the battery DC contribution at the one shared
                # inverter operating point rather than applying a second
                # independent part-load curve.
                if math.isfinite(cap_discharge_ac_wh):
                    _, unconstrained_battery_ac, _ = combined_conversion(battery_dc)
                    if unconstrained_battery_ac > cap_discharge_ac_wh:
                        lower = 0.0
                        upper = battery_dc
                        for _ in range(40):
                            midpoint = (lower + upper) / 2.0
                            _, midpoint_battery_ac, _ = combined_conversion(midpoint)
                            if midpoint_battery_ac < cap_discharge_ac_wh:
                                lower = midpoint
                            else:
                                upper = midpoint
                        battery_dc = upper

                pv_delivered_ac, delivered_ac, total_inverter_loss = combined_conversion(battery_dc)
                draw = battery_dc / eff_discharge
                battery_energy -= draw
                total_inverter_dc = pv_dc + battery_dc
                battery_inverter_loss = (
                    total_inverter_loss * battery_dc / total_inverter_dc if total_inverter_dc > 0.0 else 0.0
                )
                ledger["battery_discharge_dc"] = draw
                ledger["battery_ac_to_load"] = delivered_ac
                ledger["battery_discharge_loss"] = draw - battery_dc
                ledger["battery_inverter_loss"] = battery_inverter_loss
                ledger["pv_ac_to_load"] = pv_delivered_ac
                ledger["pv_direct_inverter_loss"] = total_inverter_loss - battery_inverter_loss
                ledger["grid_import"] = max(0.0, load - pv_delivered_ac - delivered_ac)
            else:
                ledger["grid_import"] = deficit
    else:
        usable_ac = pv_ac_max
        ledger["pv_ac_to_load"] = min(usable_ac, load)
        ledger["pv_ac_export"] = usable_ac - ledger["pv_ac_to_load"]
        dc_to_inverter = pv_dc - pv_conversion.clipping_loss_dc_w
        ledger["pv_dc_to_inverter"] = dc_to_inverter
        ledger["pv_dc_curtailed"] = pv_conversion.clipping_loss_dc_w
        ledger["pv_direct_inverter_loss"] = pv_conversion.conversion_loss_w
        ledger["grid_import"] = max(0.0, load - ledger["pv_ac_to_load"])

    return battery_energy, ledger


def simulate_energy_balance(
    pv_dc: pd.Series,
    houseload: pd.DataFrame,
    battery_config: Optional[BatteryConfig] = None,
    start_time: Optional[pd.Timestamp] = None,
    end_time: Optional[pd.Timestamp] = None,
    freq: str = "h",
    temperature_series: Optional[pd.Series] = None,
    results_directory: Optional[str] = None,
    initial_fec: float = 0.0,
    initial_calendar_seconds: float = 0.0,
    initial_resistance_growth: float = 0.0,
    initial_cumulative_cycle_deg: float = 0.0,
    initial_cumulative_cal_deg: float = 0.0,
    degradation_engine: str = "native",
    blast_model: Optional[str] = None,
    initial_degradation_state: Optional[Dict[str, Any]] = None,
    return_degradation_state: bool = False,
    debug: bool = False,
    initial_energy_wh: Optional[float] = None,
    initial_pv_origin_energy_wh: Optional[float] = None,
) -> (
    Tuple[pd.DataFrame, float, pd.DataFrame, float, int, pd.DataFrame]
    | Tuple[pd.DataFrame, float, pd.DataFrame, float, int, pd.DataFrame, Dict[str, Any]]
):
    """
    Simulate energy balance with battery storage and degradation.

    This function processes PV DC production and load profiles to calculate
    grid interaction, battery state, and degradation for DC-coupled hybrid
    inverter systems. AC-coupled battery dispatch is not implemented.

    Energy flow for DC-coupled (hybrid inverter) systems:
    - PV -> Load: DC -> Inverter -> AC (one inverter loss)
    - PV -> Battery: DC -> Battery (charge efficiency only)
    - Battery -> Load: DC -> Inverter -> AC (discharge efficiency + inverter loss)
    - Grid -> Load: AC (no conversion)

    Args:
        pv_dc: Series with PV DC power production (W) - before inverter
        houseload: DataFrame with electrical load (W) - AC
        battery_config: Battery configuration parameters
        start_time: Simulation start time (defaults to first index of pv_dc)
        end_time: Simulation end time (defaults to last index of pv_dc)
        freq: Time frequency ('h' for hourly, '15min' for 15-minute)
        temperature_series: Battery cell temperature (C), defaults to 25C
        results_directory: Directory for saving results (optional)
        degradation_engine: Degradation backend. ``"native"`` preserves the
            Naumann/Lam model; ``"blast"`` uses the BLAST daily endpoint adapter.
        blast_model: BLAST model key when ``degradation_engine="blast"``.
        initial_degradation_state: Optional state returned by a previous call
            with ``return_degradation_state=True``.
        return_degradation_state: Append final degradation carry state to the
            return tuple when True.
        debug: Enable debug output
        initial_energy_wh: Optional carried stored-energy state (Wh). Defaults
            to the configured max-SOC state for first-run compatibility.
        initial_pv_origin_energy_wh: Optional PV-origin share of the carried
            stored energy (Wh). Defaults to zero.

    Returns:
        Tuple of:
        - results_df: Detailed timestep results
        - total_pv: Total PV AC production after inverter efficiency (Wh)
        - summary_df: Summary statistics
        - replacement_cost: Total battery replacement cost
        - n_replacements: Number of battery replacements
        - degradation_df: Daily degradation tracking
    """
    if battery_config is None:
        battery_config = BatteryConfig(nominal_energy_wh=0)

    # Determine time range
    if start_time is None:
        start_time = pv_dc.index[0]
    if end_time is None:
        end_time = pv_dc.index[-1]

    # Calculate hours per step for energy conversion
    hours_per_step = get_hours_per_step(freq)
    steps_per_day = int(24 / hours_per_step)

    # Create time range
    rng = pd.date_range(start=start_time, end=end_time, freq=freq)

    # Align input data - pv_dc
    pv_dc = pv_dc.reindex(rng).fillna(0.0)

    # Align load data
    if isinstance(houseload.index, pd.DatetimeIndex):
        houseload_series = houseload.iloc[:, 0].copy()
        load_idx = houseload_series.index

        # Work in UTC to avoid DST ambiguity (naive stripping creates
        # duplicates at fall-back transitions, e.g. Oct 26 01:00 in Lisbon).
        if load_idx.tz is not None:
            load_utc = load_idx.tz_convert("UTC")
        else:
            load_utc = load_idx.tz_localize("UTC")

        rng_utc = rng.tz_convert("UTC") if rng.tz is not None else rng.tz_localize("UTC")

        # Only remap year if load covers a single year different from simulation.
        # Use dominant year (most frequent) to handle tz-aware indices that
        # span two calendar years in UTC (e.g., CET midnight = UTC 23:00 prev day).
        load_dominant_year = load_utc.year.value_counts().idxmax()
        sim_dominant_year = rng_utc.year.value_counts().idxmax()
        if load_dominant_year != sim_dominant_year:
            year_offset = sim_dominant_year - load_dominant_year
            houseload_series.index = load_utc
            houseload_series = remap_datetime_index_years(houseload_series, year_offset)
            load_utc = houseload_series.index

        # Convert back to target timezone (UTC→local is always unambiguous)
        if rng.tz is not None:
            new_load_idx = load_utc.tz_convert(rng.tz)
        else:
            new_load_idx = load_utc.tz_localize(None)
        houseload_series.index = new_load_idx
    else:
        houseload_series = houseload.iloc[:, 0].copy()
        houseload_series.index = pv_dc.index
    houseload_series = houseload_series.reindex(rng).fillna(0.0)

    # Temperature series
    if temperature_series is None:
        temperature_series = pd.Series(25.0, index=rng)
    else:
        temperature_series = temperature_series.reindex(rng).fillna(25.0)

    degradation_engine_key = str(degradation_engine).strip().lower()
    if degradation_engine_key not in {"native", "blast"}:
        raise ValueError("degradation_engine must be 'native' or 'blast'")

    if degradation_engine_key == "blast" and not blast_model:
        raise ValueError("blast_model is required when degradation_engine='blast'")

    # Get degradation model parameters
    k0_frac, Ea_val, b_val, n_val = _get_degradation_params(battery_config.calendar_model)

    # Initialize state
    battery_soh_decimal = battery_config.initial_soh / 100.0
    Battery_SOH = battery_config.initial_soh
    default_initial_energy_wh = battery_config.nominal_energy_wh * battery_soh_decimal * battery_config.max_soc
    if initial_energy_wh is None:
        Battery_Energy_Wh = default_initial_energy_wh
    else:
        if isinstance(initial_energy_wh, (bool, np.bool_)):
            raise ValueError("initial_energy_wh must be a finite number, not a bool")
        try:
            Battery_Energy_Wh = float(initial_energy_wh)
        except (TypeError, ValueError) as exc:
            raise ValueError("initial_energy_wh must be a finite number") from exc
        if not math.isfinite(Battery_Energy_Wh):
            raise ValueError("initial_energy_wh must be a finite number")
        if not 0.0 <= Battery_Energy_Wh <= battery_config.nominal_energy_wh:
            raise ValueError(
                f"initial_energy_wh must be between 0 and nominal_energy_wh ({battery_config.nominal_energy_wh:g} Wh)"
            )

    if initial_pv_origin_energy_wh is None:
        Battery_PV_Origin_Energy_Wh = 0.0
    else:
        if isinstance(initial_pv_origin_energy_wh, (bool, np.bool_)):
            raise ValueError("initial_pv_origin_energy_wh must be a finite number, not a bool")
        try:
            Battery_PV_Origin_Energy_Wh = float(initial_pv_origin_energy_wh)
        except (TypeError, ValueError) as exc:
            raise ValueError("initial_pv_origin_energy_wh must be a finite number") from exc
        if not math.isfinite(Battery_PV_Origin_Energy_Wh):
            raise ValueError("initial_pv_origin_energy_wh must be a finite number")
        if not 0.0 <= Battery_PV_Origin_Energy_Wh <= Battery_Energy_Wh:
            raise ValueError("initial_pv_origin_energy_wh must be between 0 and initial_energy_wh")

    # Degradation day-windows are positional (fixed steps_per_day), not
    # calendar-based: DST days and trailing partial days shift/skip windows
    # by design; the Numba kernels share the convention.
    soc_absolute_buffer = np.empty(steps_per_day, dtype=np.float64)
    t_cell_day_buffer = np.empty(steps_per_day, dtype=np.float64)
    soc_buf_idx = 0
    fec_cum = initial_fec
    cumulative_cal_seconds = initial_calendar_seconds
    # The function argument is the multi-year continuation seam (used by the
    # App's year loop); when left at its default the battery's configured
    # starting resistance applies.
    resistance_growth = (
        initial_resistance_growth if initial_resistance_growth > 0.0 else battery_config.initial_resistance_growth
    )
    # Charge/discharge efficiencies, derated by resistance growth when the
    # fade model is enabled; updated after each daily degradation step.
    eff_charge = battery_config.charge_efficiency
    eff_discharge = battery_config.discharge_efficiency
    if battery_config.enable_resistance_fade and resistance_growth > 0.0:
        _rte_derate = math.sqrt(1.0 + resistance_growth)
        eff_charge /= _rte_derate
        eff_discharge /= _rte_derate
    n_replacements = 0
    total_replacement_cost = 0.0
    cumulative_cycle_deg = initial_cumulative_cycle_deg
    cumulative_cal_deg = initial_cumulative_cal_deg
    cumulative_resistance_cycle = 0.0
    cumulative_resistance_calendar = 0.0

    degradation_tracking = []
    T_cell_day_sum = 0.0

    # Pre-extract numpy arrays for fast indexed access (avoids .loc[] overhead)
    _pv_dc_vals = pv_dc.values.astype(np.float64)
    _load_vals = houseload_series.values.astype(np.float64)
    _temp_vals = temperature_series.values.astype(np.float64)
    n_steps = len(rng)

    # Pre-allocate result arrays (avoids per-timestep dict creation)
    _res_pv_dc = np.empty(n_steps)
    _res_pv_prod = np.empty(n_steps)
    _res_load = np.empty(n_steps)
    _res_pv_delta = np.empty(n_steps)
    _res_import = np.empty(n_steps)
    _res_sell = np.empty(n_steps)
    _res_batt_e = np.empty(n_steps)
    _res_soc_norm = np.empty(n_steps)
    _res_soc_abs = np.empty(n_steps)
    _res_soh = np.empty(n_steps)
    _res_tcell = np.empty(n_steps)
    _res_replaced = np.zeros(n_steps, dtype=bool)
    _res_repl_cost = np.zeros(n_steps)
    _res_pv_curtailment = np.empty(n_steps)
    _res_batt_charge_loss = np.empty(n_steps)
    _res_batt_discharge_loss = np.empty(n_steps)
    _res_batt_standby_loss = np.empty(n_steps)
    _ledger_keys = (
        "PV_DC_To_Battery",
        "PV_DC_To_Inverter",
        "PV_DC_Curtailed",
        "PV_AC_To_Load",
        "PV_AC_Export",
        "Battery_Charge_Input",
        "Battery_Charge_Stored",
        "Battery_Discharge_DC",
        "Battery_AC_To_Load",
        "Battery_AC_To_Load_PV",
        "PV_Origin_Battery_AC_To_Load",
        "PV_Direct_Inverter_Loss",
        "Battery_Inverter_Loss",
        "Inverter_Loss",
        "Standby_Loss",
        "Capacity_Window_Loss",
        "Battery_Replacement_Energy_Removed",
        "Battery_Replacement_Energy_Added",
        "Battery_Energy_Delta",
    )
    _ledger_arrays = {key: np.empty(n_steps) for key in _ledger_keys}
    _res_batt_e_begin = np.empty(n_steps)
    _res_batt_pv_begin = np.empty(n_steps)
    _res_batt_pv_end = np.empty(n_steps)
    # Hoist invariant check out of the loop
    has_battery = battery_config.nominal_energy_wh > 1 and (battery_config.max_soc - battery_config.min_soc) > 0
    blast_engine = None
    build_blast_endpoint_day = None
    blast_day_start_soc = battery_config.max_soc if has_battery else 0.0
    blast_day_start_t_cell = float(_temp_vals[0]) if n_steps else 25.0

    if degradation_engine_key == "blast":
        if not has_battery:
            raise ValueError("degradation_engine='blast' requires a configured battery")

        from breos.degradation.engine import BlastEngine, build_endpoint_day

        state_payload = initial_degradation_state or {}
        blast_snapshot = state_payload.get("blast_engine", state_payload)
        if blast_snapshot:
            blast_engine = BlastEngine.from_snapshot(blast_model, blast_snapshot)
        else:
            blast_engine = BlastEngine(blast_model)
            if not math.isclose(battery_config.initial_soh, 100.0):
                raise ValueError(
                    "BLAST starts from a beginning-of-life model unless initial_degradation_state is provided"
                )

        build_blast_endpoint_day = build_endpoint_day
        battery_soh_decimal = blast_engine.soh()
        Battery_SOH = battery_soh_decimal * 100.0
        Battery_Energy_Wh = battery_config.nominal_energy_wh * battery_soh_decimal * battery_config.max_soc
        blast_day_start_soc = float(state_payload.get("day_start_soc_absolute", blast_day_start_soc))
        blast_day_start_t_cell = float(state_payload.get("day_start_temperature_c", blast_day_start_t_cell))

    # Bind capacity factor function once
    _cap_factor_fn = lfp_capacity_factor
    # Inverter AC cap per step (Wh); None keeps the legacy uncapped model
    cap_wh = (
        battery_config.inverter_ac_capacity_w * hours_per_step
        if battery_config.inverter_ac_capacity_w is not None
        else float("inf")
    )
    cap_charge_wh = (
        battery_config.max_charge_power_w * hours_per_step
        if battery_config.max_charge_power_w is not None
        else float("inf")
    )
    cap_discharge_wh = (
        battery_config.max_discharge_power_w * hours_per_step
        if battery_config.max_discharge_power_w is not None
        else float("inf")
    )

    for i in range(n_steps):
        step_time = rng[i]
        # Get values for this timestep via fast array indexing
        # Treat negative model/data artefacts as zero generation, matching the
        # public inverter helper and preventing negative PV from being
        # allocated through the shared PV/battery conversion path.
        pv_dc_power = max(0.0, _pv_dc_vals[i] * hours_per_step)  # DC power (Wh) before inverter
        load = _load_vals[i] * hours_per_step  # AC Load in Wh
        T_ambient = _temp_vals[i]
        T_cell = T_ambient  # default; overridden by thermal model below

        battery_energy_beginning = Battery_Energy_Wh if has_battery else 0.0
        pv_origin_beginning = Battery_PV_Origin_Energy_Wh if has_battery else 0.0
        capacity_window_loss = 0.0
        battery_standby_loss = 0.0

        if has_battery:
            # Calculate usable capacity with temperature derating. f_cap is
            # computed from the ambient/indoor temperature at step start; the
            # lumped thermal model warms T_cell later in the step, so aging
            # sees the warmed cell while derating sees the environment. This
            # is intentional: usable capacity is set by the pack's state
            # before this step's charge/discharge self-heating.
            usable_cap = battery_config.nominal_energy_wh * battery_soh_decimal
            f_cap = _cap_factor_fn(T_cell)
            Emax = usable_cap * battery_config.max_soc * f_cap
            Emin = usable_cap * battery_config.min_soc * f_cap

            # A temperature/SOH-driven reduction in Emax is an explicit loss;
            # it is not export or standby consumption. The lower reserve is a
            # dispatch boundary and must never create energy when it rises.
            capacity_window_loss = max(0.0, Battery_Energy_Wh - Emax)
            if capacity_window_loss > 0.0 and Battery_Energy_Wh > 0.0:
                Battery_PV_Origin_Energy_Wh *= Emax / Battery_Energy_Wh
                Battery_Energy_Wh = Emax

            standby_loss = battery_config.standby_loss_wh * hours_per_step
            removable_for_standby = max(0.0, Battery_Energy_Wh - Emin)
            battery_standby_loss = min(standby_loss, removable_for_standby)
            if battery_standby_loss > 0.0 and Battery_Energy_Wh > 0.0:
                Battery_PV_Origin_Energy_Wh *= (Battery_Energy_Wh - battery_standby_loss) / Battery_Energy_Wh
                Battery_Energy_Wh -= battery_standby_loss
        else:
            Emax = 0.0
            Emin = 0.0

        energy_before_dispatch = Battery_Energy_Wh
        origin_before_dispatch = Battery_PV_Origin_Energy_Wh
        origin_fraction = (
            min(1.0, max(0.0, origin_before_dispatch / energy_before_dispatch)) if energy_before_dispatch > 0.0 else 0.0
        )
        Battery_Energy_Wh, ledger = _dispatch_dc_step(
            pv_dc_power,
            load,
            Battery_Energy_Wh,
            Emin,
            Emax,
            eff_charge,
            eff_discharge,
            battery_config.inverter_efficiency,
            cap_charge_wh,
            cap_discharge_wh,
            cap_wh,
            has_battery,
        )
        charge_stored = ledger["battery_charge_input"] * eff_charge
        pv_origin_discharge_dc = ledger["battery_discharge_dc"] * origin_fraction
        pv_origin_battery_ac = ledger["battery_ac_to_load"] * origin_fraction
        Battery_PV_Origin_Energy_Wh = max(
            0.0,
            origin_before_dispatch - pv_origin_discharge_dc + charge_stored,
        )
        Battery_PV_Origin_Energy_Wh = min(Battery_PV_Origin_Energy_Wh, Battery_Energy_Wh)

        Import = ledger["grid_import"]
        Sell = ledger["pv_ac_export"]
        charge_in = ledger["battery_charge_input"]
        discharge_out = ledger["battery_discharge_dc"]
        pv_curtailment = ledger["pv_dc_curtailed"]
        battery_charge_loss = ledger["battery_charge_loss"]
        battery_discharge_loss = ledger["battery_discharge_loss"]
        # Compatibility field: retain the exact 0.3.4 result when a lower-
        # level caller omits the inverter rating. With a finite inverter, use
        # the explicit part-load conversion loss. Public economics use the AC
        # ledger fields instead.
        if math.isinf(cap_wh):
            pv_production = (pv_dc_power - pv_curtailment) * battery_config.inverter_efficiency
        else:
            pv_production = pv_dc_power - pv_curtailment - ledger["pv_direct_inverter_loss"]
        battery_energy_delta = Battery_Energy_Wh - battery_energy_beginning

        # Compute cell temperature via lumped thermal model
        if has_battery and battery_config.thermal_resistance_kw > 0:
            # charge_in and discharge_out are in Wh; convert to W for thermal calc
            charge_power_w = charge_in / hours_per_step if hours_per_step > 0 else 0.0
            discharge_power_w = discharge_out / hours_per_step if hours_per_step > 0 else 0.0
            T_cell = compute_cell_temperature(
                T_ambient,
                charge_power_w,
                discharge_power_w,
                eff_charge,
                eff_discharge,
                battery_config.thermal_resistance_kw,
            )
        T_cell_day_sum += T_cell

        # SOC calculations (handle no-battery case)
        if has_battery:
            soc_normalized = (Battery_Energy_Wh - Emin) / (Emax - Emin) if (Emax - Emin) > 0 else 0.0
            soc_normalized = max(0.0, min(1.0, soc_normalized))
            soc_absolute = (
                Battery_Energy_Wh / (battery_config.nominal_energy_wh * battery_soh_decimal)
                if battery_soh_decimal > 0
                else 0.0
            )
            soc_absolute = max(0.0, min(1.0, soc_absolute))
        else:
            soc_normalized = 0.0
            soc_absolute = 0.0
        soc_absolute_buffer[soc_buf_idx] = soc_absolute
        t_cell_day_buffer[soc_buf_idx] = T_cell
        soc_buf_idx += 1

        # Store results via array indexing (avoids per-timestep dict overhead)
        _res_pv_dc[i] = pv_dc_power / hours_per_step
        _res_pv_prod[i] = pv_production / hours_per_step
        _res_load[i] = load / hours_per_step
        _res_pv_delta[i] = (pv_production - load) / hours_per_step
        _res_import[i] = Import / hours_per_step
        _res_sell[i] = Sell / hours_per_step
        _res_batt_e[i] = Battery_Energy_Wh if has_battery else 0.0
        _res_soc_norm[i] = soc_normalized
        _res_soc_abs[i] = soc_absolute
        _res_soh[i] = Battery_SOH if has_battery else 100.0
        _res_tcell[i] = T_cell
        _res_pv_curtailment[i] = pv_curtailment / hours_per_step
        _res_batt_charge_loss[i] = battery_charge_loss / hours_per_step
        _res_batt_discharge_loss[i] = battery_discharge_loss / hours_per_step
        _res_batt_standby_loss[i] = battery_standby_loss / hours_per_step
        _res_batt_e_begin[i] = battery_energy_beginning
        _res_batt_pv_begin[i] = pv_origin_beginning
        _res_batt_pv_end[i] = Battery_PV_Origin_Energy_Wh
        ledger_w = {
            "PV_DC_To_Battery": ledger["pv_dc_to_battery"],
            "PV_DC_To_Inverter": ledger["pv_dc_to_inverter"],
            "PV_DC_Curtailed": ledger["pv_dc_curtailed"],
            "PV_AC_To_Load": ledger["pv_ac_to_load"],
            "PV_AC_Export": ledger["pv_ac_export"],
            "Battery_Charge_Input": ledger["battery_charge_input"],
            "Battery_Charge_Stored": charge_stored,
            "Battery_Discharge_DC": ledger["battery_discharge_dc"],
            "Battery_AC_To_Load": ledger["battery_ac_to_load"],
            "Battery_AC_To_Load_PV": pv_origin_battery_ac,
            "PV_Origin_Battery_AC_To_Load": pv_origin_battery_ac,
            "PV_Direct_Inverter_Loss": ledger["pv_direct_inverter_loss"],
            "Battery_Inverter_Loss": ledger["battery_inverter_loss"],
            "Inverter_Loss": ledger["pv_direct_inverter_loss"] + ledger["battery_inverter_loss"],
            "Standby_Loss": battery_standby_loss,
            "Capacity_Window_Loss": capacity_window_loss,
            "Battery_Replacement_Energy_Removed": 0.0,
            "Battery_Replacement_Energy_Added": 0.0,
            "Battery_Energy_Delta": battery_energy_delta,
        }
        for key, value_wh in ledger_w.items():
            _ledger_arrays[key][i] = value_wh / hours_per_step

        # Daily degradation update
        if soc_buf_idx >= steps_per_day:
            # Build Series from buffer using rng slice (avoids pd.date_range overhead)
            day_start_i = i - steps_per_day + 1
            soc_series = pd.Series(
                soc_absolute_buffer[:steps_per_day].copy(),
                index=rng[day_start_i : i + 1],
            )
            t_cell_day = t_cell_day_buffer[:steps_per_day].copy()
            day_end_soc_absolute = float(soc_absolute_buffer[steps_per_day - 1])
            day_end_t_cell = float(t_cell_day[steps_per_day - 1])

            mean_soc_abs = float(soc_series.mean())
            mean_T_cell = T_cell_day_sum / steps_per_day
            effective_rte = battery_config.charge_efficiency * battery_config.discharge_efficiency

            if blast_engine is not None:
                previous_soh_decimal = battery_soh_decimal
                t_secs_day, soc_day, t_cell_day_c = build_blast_endpoint_day(
                    hours_per_step * 3600.0,
                    soc_series.to_numpy(),
                    t_cell_day,
                    start_soc=blast_day_start_soc,
                    start_temperature_c=blast_day_start_t_cell,
                )
                battery_soh_decimal = blast_engine.step(t_secs_day, soc_day, t_cell_day_c)
                fec_cum = float(blast_engine.model.stressors["efc"][-1])
                dSOH_cycle = 0.0
                dSOH_calendar = 0.0
                dSOH_blast = max(0.0, previous_soh_decimal - battery_soh_decimal)
                cumulative_cal_seconds += 86400.0
            else:
                # Cycle degradation
                soh_after_cycle, dSOH_cycle, fec_cum = update_battery_soh_cyclewise(
                    battery_soh_decimal,
                    soc_series,
                    battery_config.nominal_energy_wh,
                    fec_cum=fec_cum,
                    battery_type=battery_config.battery_type,
                    debug=debug,
                )

                # Calendar degradation — use mean cell temperature over the day
                soh_after_calendar, dSOH_calendar, cumulative_cal_seconds = update_battery_soh_calendar(
                    soh_after_cycle,
                    k0_frac=k0_frac,
                    Ea=Ea_val,
                    n=n_val,
                    cal_b=b_val,
                    T_cell_C=mean_T_cell,
                    cumulative_cal_seconds=cumulative_cal_seconds,
                    dt_days=1.0,
                    mean_soc_absolute=mean_soc_abs,
                    debug=debug,
                )

                battery_soh_decimal = soh_after_calendar
                dSOH_blast = 0.0

            Battery_SOH = battery_soh_decimal * 100.0

            cumulative_cycle_deg += dSOH_cycle
            cumulative_cal_deg += dSOH_calendar

            # Resistance fade (opt-in)
            dR_cycle = 0.0
            dR_calendar = 0.0
            if battery_config.enable_resistance_fade:
                # Get cycles for resistance calculation
                time_index = soc_series.index
                cycles = detect_cycles_rainflow(soc_series, time_index, min_doc_fraction=0.01)

                # Compute FEC at start of day (before today's cycles were added)
                day_fec = sum(max(0.0, min(1.0, c["doc"])) * c.get("count", 1.0) for c in cycles)
                fec_before_day = fec_cum - day_fec

                resistance_growth, dR_cycle = update_battery_resistance_cyclewise(
                    resistance_growth, cycles, fec_before_day, debug=debug
                )
                resistance_growth, dR_calendar = update_battery_resistance_calendar(
                    resistance_growth,
                    T_cell_C=mean_T_cell,
                    cumulative_cal_seconds=cumulative_cal_seconds,
                    dt_days=1.0,
                    mean_soc_absolute=mean_soc_abs,
                    debug=debug,
                )
                cumulative_resistance_cycle += dR_cycle
                cumulative_resistance_calendar += dR_calendar

                # Feed the resistance penalty back into the energy loop,
                # split evenly across charge and discharge so their product
                # equals the effective round-trip efficiency.
                _rte_derate = math.sqrt(1.0 + resistance_growth)
                eff_charge = battery_config.charge_efficiency / _rte_derate
                eff_discharge = battery_config.discharge_efficiency / _rte_derate
                effective_rte = eff_charge * eff_discharge

            # Battery replacement check
            if battery_config.enable_replacement and battery_soh_decimal <= battery_config.eol_percentage:
                replacement_energy_removed = Battery_Energy_Wh
                battery_soh_decimal = 1.0
                Battery_SOH = 100.0
                fec_cum = 0.0
                cumulative_cal_seconds = 0.0
                resistance_growth = 0.0
                eff_charge = battery_config.charge_efficiency
                eff_discharge = battery_config.discharge_efficiency
                Battery_Energy_Wh = battery_config.nominal_energy_wh * battery_config.max_soc
                replacement_energy_added = Battery_Energy_Wh
                Battery_PV_Origin_Energy_Wh = 0.0
                n_replacements += 1
                total_replacement_cost += battery_config.replacement_cost
                _res_replaced[i] = True
                _res_repl_cost[i] = battery_config.replacement_cost
                cumulative_cycle_deg = 0.0
                cumulative_cal_deg = 0.0
                cumulative_resistance_cycle = 0.0
                cumulative_resistance_calendar = 0.0
                day_end_soc_absolute = battery_config.max_soc
                if blast_engine is not None:
                    blast_engine.reset()

                # Replacement occurs inside this timestep boundary. Rewrite
                # the already-recorded state so the reported end matches the
                # next timestep's beginning, and expose both external energy
                # transfers for whole-system reconciliation.
                _res_batt_e[i] = Battery_Energy_Wh
                _res_soc_norm[i] = 1.0
                _res_soc_abs[i] = battery_config.max_soc
                _res_soh[i] = 100.0
                _res_batt_pv_end[i] = 0.0
                _ledger_arrays["Battery_Replacement_Energy_Removed"][i] = replacement_energy_removed / hours_per_step
                _ledger_arrays["Battery_Replacement_Energy_Added"][i] = replacement_energy_added / hours_per_step
                _ledger_arrays["Battery_Energy_Delta"][i] = (
                    Battery_Energy_Wh - battery_energy_beginning
                ) / hours_per_step

                if debug:
                    print(f"\n*** BATTERY REPLACED at {step_time} ***")

            degradation_record = {
                "Datetime": step_time,
                "SOH": Battery_SOH,
                "Cycle_Degradation": dSOH_cycle,
                "Calendar_Degradation": dSOH_calendar,
                "Cumulative_Cycle_Degradation": cumulative_cycle_deg,
                "Cumulative_Calendar_Degradation": cumulative_cal_deg,
                "Cumulative_FEC": fec_cum,
                "Cumulative_Calendar_Seconds": cumulative_cal_seconds,
                "Total_Degradation": 1.0 - battery_soh_decimal,
                "Mean_SOC_Absolute": mean_soc_abs,
            }
            if blast_engine is not None:
                degradation_record["BLAST_Model"] = blast_model
                degradation_record["BLAST_Degradation"] = dSOH_blast
            if battery_config.enable_resistance_fade:
                degradation_record["Resistance_Growth"] = resistance_growth
                degradation_record["Effective_RTE"] = effective_rte
            degradation_tracking.append(degradation_record)

            # Reset daily accumulators
            soc_buf_idx = 0
            T_cell_day_sum = 0.0
            blast_day_start_soc = day_end_soc_absolute
            blast_day_start_t_cell = day_end_t_cell

    # Build results DataFrame from pre-allocated arrays
    df = pd.DataFrame(
        {
            "Datetime": rng,
            "PV_DC": _res_pv_dc,
            "PV_Production": _res_pv_prod,
            "Houseload": _res_load,
            "PV_Delta": _res_pv_delta,
            "Import_From_Grid": _res_import,
            "Sell_To_Grid": _res_sell,
            "Battery_Energy": _res_batt_e,
            "Battery_SOC_Normalized": _res_soc_norm,
            "Battery_SOC_Absolute": _res_soc_abs,
            "Battery_SOH": _res_soh,
            "T_cell": _res_tcell,
            "Battery_Replaced": _res_replaced,
            "Replacement_Cost": _res_repl_cost,
            "PV_Curtailment": _res_pv_curtailment,
            "Battery_Charge_Loss": _res_batt_charge_loss,
            "Battery_Discharge_Loss": _res_batt_discharge_loss,
            "Battery_Standby_Loss": _res_batt_standby_loss,
            # Stored-energy state columns are Wh; all explicit flow/loss
            # ledger columns are average W over the timestep.
            "Battery_Energy_Beginning": _res_batt_e_begin,
            "Battery_Energy_End": _res_batt_e,
            "Battery_PV_Origin_Energy_Beginning": _res_batt_pv_begin,
            "Battery_PV_Origin_Energy_End": _res_batt_pv_end,
            **_ledger_arrays,
        }
    )
    deg_df = pd.DataFrame(degradation_tracking) if degradation_tracking else pd.DataFrame()

    # Summary calculations (use numpy sums on arrays directly)
    total_pv = _res_pv_prod.sum() * hours_per_step
    total_load = _res_load.sum() * hours_per_step
    total_sell = _res_sell.sum() * hours_per_step
    total_import = _res_import.sum() * hours_per_step

    percentage_imported = (total_import / total_load * 100) if total_load > 0 else 0

    summary = {
        "Total PV [kWh]": total_pv / 1000.0,
        "Total Load [kWh]": total_load / 1000.0,
        "Sell [kWh]": total_sell / 1000.0,
        "Import [kWh]": total_import / 1000.0,
        "Import [%]": percentage_imported,
        "Grid Independence [%]": 100 - percentage_imported,
        "Final SOH [%]": Battery_SOH,
        "N_Replacements": n_replacements,
        "Replacement_Cost": total_replacement_cost,
    }
    summary_df = pd.DataFrame([summary])

    result = (df, total_pv, summary_df, total_replacement_cost, n_replacements, deg_df)
    if not return_degradation_state:
        return result

    final_degradation_state: Dict[str, Any] = {
        "degradation_engine": degradation_engine_key,
        "fec_cum": float(fec_cum),
        "cumulative_calendar_seconds": float(cumulative_cal_seconds),
        "resistance_growth": float(resistance_growth),
        "cumulative_cycle_degradation": float(cumulative_cycle_deg),
        "cumulative_calendar_degradation": float(cumulative_cal_deg),
    }
    if blast_engine is not None:
        final_degradation_state.update(
            {
                "blast_model": blast_model,
                "blast_engine": blast_engine.state_snapshot(),
                "day_start_soc_absolute": float(blast_day_start_soc),
                "day_start_temperature_c": float(blast_day_start_t_cell),
            }
        )

    return (*result, final_degradation_state)


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
    thermal_resistance_kw: float = DEFAULT_THERMAL_RESISTANCE_KW,
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
        thermal_resistance_kw: Thermal resistance in K/W

    Returns:
        Cell temperature (C)
    """
    # Heat from charging: fraction (1 - eta_charge) is lost as heat
    P_loss_charge = charge_power_w * (1.0 - charge_eff)
    # Heat from discharging: battery delivers more internally than reaches load
    P_loss_discharge = discharge_power_w * (1.0 - discharge_eff)

    P_loss_total = P_loss_charge + P_loss_discharge
    T_cell = T_ambient_C + thermal_resistance_kw * P_loss_total
    return T_cell


def apply_indoor_temperature_model(
    outdoor_temperature: pd.Series,
    setpoint_c: float = 22.0,
    coupling_alpha: float = 0.3,
    floor_c: float = 15.0,
    ceiling_c: float = 35.0,
) -> pd.Series:
    """
    Transform outdoor temperature to indoor temperature for battery simulation.

    Residential batteries are installed indoors where building thermal mass
    buffers outdoor extremes. This stateless preprocessing applies a weighted
    blend with clamp before temperatures enter the simulation loop.

    T_indoor = clamp(alpha * T_outdoor + (1 - alpha) * T_setpoint, floor, ceiling)

    Args:
        outdoor_temperature: Outdoor ambient temperature series (°C)
        setpoint_c: Indoor comfort midpoint (°C)
        coupling_alpha: How much outdoor temp influences indoor (0=insulated, 1=outdoor)
        floor_c: Minimum indoor temperature (°C)
        ceiling_c: Maximum indoor temperature (°C)

    Returns:
        Indoor temperature series (°C), same index as input
    """
    t_indoor = coupling_alpha * outdoor_temperature + (1.0 - coupling_alpha) * setpoint_c
    return t_indoor.clip(lower=floor_c, upper=ceiling_c)


def _get_degradation_params(model: str) -> Tuple[float, float, float, float]:
    """Get degradation model parameters based on model name.

    All LFP models use Naumann (2020) cycle aging + the specified calendar aging parameters.
    The 'naumann' model uses Naumann's own calendar params; 'naumann_lam*' variants use
    Lam et al. (2025) calendar params with different calibrations.

    Models:
        'naumann'                          — Naumann 2020 calendar + cycle (NMC/LFP lab)
        'naumann_lam'                      — Naumann cycle + Lam 2025 lab-derived calendar
        'naumann_lam_field_calibrated'     — v1 field-calibrated fit (default alias)
        'naumann_lam_field_calibrated_v1'  — v1 field-calibrated fit (explicit)
        'naumann_lam_field_calibrated_v2'  — v2 field-calibrated fit with
                                             Lam Ea/n fixed and k0/b fitted
    """
    model_lower = model.lower().replace("-", "_")

    # ── Naumann (pure) ────────────────────────────────────────────────────
    if model_lower == "naumann":
        k0_frac = NAUMANN_K0_PERCENT / 100.0
        return k0_frac, NAUMANN_EA_J_MOL, NAUMANN_EXPONENT_B, NAUMANN_SOC_EXPONENT_N

    # ── Naumann-Lam: lab-derived ──────────────────────────────────────────
    elif model_lower == "naumann_lam":
        return LAM_K0_FRAC, LAM_EA_J_MOL, LAM_EXPONENT_B, LAM_SOC_EXPONENT_N

    # ── Naumann-Lam: field-calibrated v1 (default) ───────────────────────
    elif model_lower == "naumann_lam_field_calibrated":
        return (
            NAUMANN_LAM_FIELD_CALIBRATED_K0_FRAC,
            NAUMANN_LAM_FIELD_CALIBRATED_EA_J_MOL,
            NAUMANN_LAM_FIELD_CALIBRATED_EXPONENT_B,
            NAUMANN_LAM_FIELD_CALIBRATED_SOC_EXPONENT_N,
        )

    elif model_lower == "naumann_lam_field_calibrated_v1":
        return (
            NAUMANN_LAM_FIELD_CALIBRATED_V1_K0_FRAC,
            NAUMANN_LAM_FIELD_CALIBRATED_V1_EA_J_MOL,
            NAUMANN_LAM_FIELD_CALIBRATED_V1_EXPONENT_B,
            NAUMANN_LAM_FIELD_CALIBRATED_V1_SOC_EXPONENT_N,
        )

    elif model_lower == "naumann_lam_field_calibrated_v2":
        return (
            NAUMANN_LAM_FIELD_CALIBRATED_V2_K0_FRAC,
            NAUMANN_LAM_FIELD_CALIBRATED_V2_EA_J_MOL,
            NAUMANN_LAM_FIELD_CALIBRATED_V2_EXPONENT_B,
            NAUMANN_LAM_FIELD_CALIBRATED_V2_SOC_EXPONENT_N,
        )

    else:
        raise ValueError(
            f"Unknown calendar model: {model}. Use 'naumann_lam_field_calibrated', "
            f"'naumann_lam_field_calibrated_v1', "
            f"'naumann_lam_field_calibrated_v2', "
            f"'naumann_lam', or 'naumann'."
        )


def detect_half_cycles_from_soc_series(
    soc_abs_series: pd.Series, time_index: pd.DatetimeIndex, tiny_hysteresis: float = 1e-4
) -> Tuple[List[Dict], pd.Series]:
    """
    Detect charge/discharge half-cycles using local extrema logic.

    Args:
        soc_abs_series: Absolute SOC series
        time_index: Datetime index
        tiny_hysteresis: Minimum change to count as extremum

    Returns:
        Tuple of (half_cycles list, original series)
    """
    soc = soc_abs_series.values
    times = time_index
    n = len(soc)

    if n < 2:
        return [], soc_abs_series

    extrema_idx = [0]
    for i in range(1, n - 1):
        is_peak = soc[i] >= soc[i - 1] + tiny_hysteresis and soc[i] > soc[i + 1] + tiny_hysteresis
        is_trough = soc[i] <= soc[i - 1] - tiny_hysteresis and soc[i] < soc[i + 1] - tiny_hysteresis
        if is_peak or is_trough:
            extrema_idx.append(i)
    extrema_idx.append(n - 1)

    half_cycles = []
    for i in range(1, len(extrema_idx)):
        sidx = extrema_idx[i - 1]
        eidx = extrema_idx[i]
        if eidx == sidx:
            continue

        doc = abs(soc[eidx] - soc[sidx])
        mean_soc = float(np.mean(soc[sidx : eidx + 1]))
        duration_h = (times[eidx] - times[sidx]).total_seconds() / 3600.0
        mean_c_rate = 0.0 if duration_h <= 0 else doc / duration_h

        half_cycles.append(
            {
                "start_idx": sidx,
                "end_idx": eidx,
                "doc": doc,
                "mean_soc": mean_soc,
                "mean_c_rate": mean_c_rate,
                "duration_h": duration_h,
            }
        )

    return half_cycles, soc_abs_series


def detect_cycles_rainflow(
    soc_abs_series: pd.Series, time_index: pd.DatetimeIndex, min_doc_fraction: float = 0.01
) -> List[Dict]:
    """
    Detect charge/discharge cycles using rainflow counting (ASTM E1049).

    Rainflow counting correctly identifies nested cycles common in residential
    PV+storage profiles, which simple extrema-based methods miss.

    Args:
        soc_abs_series: Absolute SOC series (0-1 range)
        time_index: Datetime index for the series
        min_doc_fraction: Minimum depth-of-cycle to include (fraction, 0-1)

    Returns:
        List of cycle dicts with keys: 'doc', 'mean_soc', 'count',
        'mean_c_rate', 'start_idx', 'end_idx'
    """
    if len(soc_abs_series) < 2:
        return []

    # rainflow.extract_cycles expects a sequence; multiply by 100 for percent
    soc_pct = soc_abs_series.values * 100.0

    cycles = []
    for rng, mean, count, i_start, i_end in rainflow.extract_cycles(soc_pct):
        doc = rng / 100.0  # convert back to fraction
        if doc < min_doc_fraction:
            continue

        mean_soc = mean / 100.0

        # Estimate C-rate from cycle duration
        if i_start < len(time_index) and i_end < len(time_index):
            duration_h = (time_index[i_end] - time_index[i_start]).total_seconds() / 3600.0
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


def compute_halfcycle_energy_throughput(hc: Dict, soc_series_absolute: pd.Series, nominal_energy_Wh: float) -> float:
    """Compute energy throughput (Wh) for a half-cycle."""
    s = soc_series_absolute.iloc[hc["start_idx"] : hc["end_idx"] + 1].values
    return abs(s[-1] - s[0]) * nominal_energy_Wh


def k_c_rate_Q(C_rate: float) -> float:
    """Calculate C-rate factor for capacity fade (Naumann Eq. 8)."""
    kC = A_Q * C_rate + B_Q
    return max(0.0, kC)


def k_doc_Q(DOC_frac: float) -> float:
    """Calculate DOC factor for capacity fade (Naumann Eq. 10)."""
    kDOC = C_DOC_Q * ((DOC_frac - 0.6) ** 3) + D_DOC_Q
    return max(0.0, kDOC)


# =========================================================================
# Resistance fade functions (Naumann 2020)
# =========================================================================


def k_c_rate_R(C_rate: float) -> float:
    """Calculate C-rate factor for resistance growth (Naumann Eq. 8 variant)."""
    kC = A_R * C_rate + B_R
    return max(0.0, kC)


def k_doc_R(DOC_frac: float) -> float:
    """Calculate DOC factor for resistance growth (Naumann Eq. 10 variant)."""
    kDOC = C_DOC_R * ((DOC_frac - 0.6) ** 3) + D_DOC_R
    return max(0.0, kDOC)


def update_battery_resistance_cyclewise(
    resistance_growth: float, cycles: List[Dict], fec_cum: float, min_DoD_fraction: float = 0.01, debug: bool = False
) -> Tuple[float, float]:
    """
    Calculate cycle-induced resistance growth using Naumann's model.

    Uses the same differential form as capacity fade but with resistance
    parameters (A_R, B_R, C_DOC_R, D_DOC_R, Z_R).

    Args:
        resistance_growth: Current cumulative resistance growth (fraction, e.g. 0.05 = 5%)
        cycles: List of cycle dicts from detect_cycles_rainflow or detect_half_cycles
        fec_cum: Cumulative FEC at start of this period
        min_DoD_fraction: Minimum DOC to count
        debug: Enable debug output

    Returns:
        Tuple of (new_resistance_growth, delta_resistance_growth)
    """
    delta_R = 0.0
    running_fec = fec_cum

    for cyc in cycles:
        DOC = max(0.0, min(1.0, cyc["doc"]))
        if DOC < min_DoD_fraction:
            continue

        count = cyc.get("count", 1.0)
        dFEC = DOC * count
        mean_c_rate = cyc["mean_c_rate"]

        kC = k_c_rate_R(mean_c_rate)
        kDOC = k_doc_R(DOC)

        fec_new = running_fec + dFEC

        # Differential form: dR% = kC * kDOC * (FEC_new^Z_R - FEC_old^Z_R)
        dR_percent = kC * kDOC * (fec_new**Z_R - running_fec**Z_R)
        dR_fraction = dR_percent / 100.0

        delta_R += dR_fraction
        running_fec = fec_new

        if debug:
            print(f"[R-cycle] DOC={DOC:.4f}, C-rate={mean_c_rate:.4f}, dR={dR_fraction * 100:.6f}%")

    new_growth = resistance_growth + delta_R
    return new_growth, delta_R


def update_battery_resistance_calendar(
    resistance_growth: float,
    T_cell_C: float,
    cumulative_cal_seconds: float,
    dt_days: float = 1.0,
    mean_soc_absolute: float = 0.5,
    debug: bool = False,
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
        debug: Enable debug output

    Returns:
        Tuple of (new_resistance_growth, delta_resistance_growth)
    """
    dt_seconds = dt_days * 86400.0
    if dt_seconds <= 0:
        return resistance_growth, 0.0

    k0_frac = NAUMANN_K0_R_PERCENT / 100.0

    T_K = T_cell_C + 273.15
    arr_factor = math.exp(-NAUMANN_EA_R_J_MOL / R_GAS * (1.0 / T_K - 1.0 / T_REF_K))

    t_old = cumulative_cal_seconds
    t_new = t_old + dt_seconds
    b = NAUMANN_EXPONENT_B_R

    term_old = math.pow(t_old, b) if t_old > 0 else 0.0
    term_new = math.pow(t_new, b)
    delta_time = term_new - term_old

    soc_stress = max(0.0, mean_soc_absolute) ** NAUMANN_SOC_EXPONENT_N_R

    dR_fraction = k0_frac * arr_factor * delta_time * soc_stress

    if debug:
        print(f"[R-calendar] T={T_cell_C:.1f}°C, dR={dR_fraction * 100:.6f}%")

    return resistance_growth + dR_fraction, dR_fraction


def resistance_to_efficiency(
    resistance_growth: float,
    base_charge_eff: float,
    base_discharge_eff: float,
) -> Tuple[float, float]:
    """
    Convert resistance growth to effective charge/discharge efficiencies.

    Internal resistance growth increases ohmic losses proportionally.
    The efficiency penalty is split equally between charge and discharge.

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

    rte_base = base_charge_eff * base_discharge_eff
    rte_new = rte_base / (1.0 + resistance_growth)

    # Split new RTE as sqrt across charge and discharge
    sqrt_rte_new = math.sqrt(max(0.01, rte_new))
    eff_charge = min(base_charge_eff, sqrt_rte_new)
    eff_discharge = min(base_discharge_eff, sqrt_rte_new)

    return eff_charge, eff_discharge


def _normalise_battery_type(battery_type: str) -> str:
    """Normalize and validate the native battery chemistry selector."""
    normalised = str(battery_type).strip().lower()
    if normalised not in SUPPORTED_BATTERY_TYPES:
        available = ", ".join(SUPPORTED_BATTERY_TYPES)
        raise ValueError(
            f"Unsupported battery_type {battery_type!r}. "
            f"The native BREOS degradation model currently supports only: {available}."
        )
    return normalised


def _get_cycle_params(battery_type: str = "lfp") -> Tuple[float, float, float, float, float]:
    """Get cycle aging (Naumann-style) parameters for a battery chemistry.

    Returns:
        Tuple of (a_q, b_q, c_doc_q, d_doc_q, z_q)
    """
    _normalise_battery_type(battery_type)
    return (A_Q, B_Q, C_DOC_Q, D_DOC_Q, Z_Q)


def update_battery_soh_cyclewise(
    soh_start_fraction: float,
    soc_series_absolute: pd.Series,
    nominal_energy_Wh: float,
    fec_cum: float = 0.0,
    min_DoD_fraction: float = 0.01,
    use_rainflow: bool = True,
    battery_type: str = "lfp",
    debug: bool = False,
) -> Tuple[float, float, float]:
    """
    Calculate cycle-induced degradation using Naumann's semi-empirical model.

    Implements Equation 5-6 from Naumann 2020 paper, with technology-specific
    cycle aging coefficients selected by battery_type.

    Args:
        soh_start_fraction: Starting SOH as fraction (0-1)
        soc_series_absolute: SOC time series
        nominal_energy_Wh: Nominal battery capacity
        fec_cum: Cumulative full equivalent cycles
        min_DoD_fraction: Minimum DoD to count as cycle
        use_rainflow: Use rainflow counting (True) or extrema-based detection (False)
        battery_type: Battery chemistry ('lfp')
        debug: Enable debug output

    Returns:
        Tuple of (soh_after, qloss_cycle_fraction, fec_cum)
    """
    if len(soc_series_absolute) < 2:
        return soh_start_fraction, 0.0, fec_cum

    time_index = soc_series_absolute.index

    if use_rainflow:
        cycles = detect_cycles_rainflow(soc_series_absolute, time_index, min_doc_fraction=min_DoD_fraction)
    else:
        cycles, _ = detect_half_cycles_from_soc_series(soc_series_absolute, time_index)

    # Get technology-specific cycle parameters
    a_q, b_q, c_doc_q, d_doc_q, z_q = _get_cycle_params(battery_type)

    qloss_cycle_fraction = 0.0

    for cyc in cycles:
        DOC = max(0.0, min(1.0, cyc["doc"]))
        if DOC < min_DoD_fraction:
            continue

        mean_c_rate = cyc["mean_c_rate"]
        # For rainflow cycles: count is 1.0 (full) or 0.5 (half)
        # For extrema-based: each entry is a half-cycle (count=1 implicitly)
        count = cyc.get("count", 1.0)

        # Energy throughput for this cycle: DOC * count * nominal
        dFEC = DOC * count

        # Naumann-style k-factors with technology-specific coefficients
        kC = max(0.0, a_q * mean_c_rate + b_q)
        kDOC = max(0.0, c_doc_q * ((DOC - 0.6) ** 3) + d_doc_q)

        fec_new = fec_cum + dFEC

        # Differential form using cumulative FEC (Naumann Eq. 5-6)
        dq_percent = kC * kDOC * (fec_new**z_q - fec_cum**z_q)
        dq_fraction = dq_percent / 100.0

        qloss_cycle_fraction += dq_fraction
        fec_cum = fec_new

        if debug:
            print(
                f"[cycle] DOC={DOC:.4f}, C-rate={mean_c_rate:.4f}, count={count}, "
                f"dFEC={dFEC:.6e}, dq={dq_fraction * 100:.6f}%"
            )

    soh_after = max(0.0, soh_start_fraction - qloss_cycle_fraction)
    return soh_after, qloss_cycle_fraction, fec_cum


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
    debug: bool = False,
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
        debug: Enable debug output

    Returns:
        Tuple of (soh_after, dsoh_fraction, new_cumulative_seconds)
    """
    dt_seconds = dt_days * 86400.0
    if dt_seconds <= 0:
        return soh_start_fraction, 0.0, cumulative_cal_seconds

    # Temperature factor (Arrhenius) relative to 25°C
    T_K = T_cell_C + 273.15
    arr_factor = math.exp(-Ea / R_GAS * (1.0 / T_K - 1.0 / T_REF_K))

    # Power law time calculation
    t_old = cumulative_cal_seconds
    t_new = cumulative_cal_seconds + dt_seconds

    term_old = math.pow(t_old, cal_b) if t_old > 0 else 0.0
    term_new = math.pow(t_new, cal_b)
    delta_time_factor = term_new - term_old

    # SOC stress factor
    soc_stress = max(0.0, mean_soc_absolute) ** n

    # Calculate degradation fraction
    d_soh_fraction = k0_frac * arr_factor * delta_time_factor * soc_stress

    soh_after = max(0.0, soh_start_fraction - d_soh_fraction)

    if debug:
        print(
            f"[calendar] T={T_cell_C}°C, b={cal_b:.2f}, Δt^b={delta_time_factor:.2f}, d_soh={d_soh_fraction * 100:.6f}%"
        )

    return soh_after, d_soh_fraction, t_new


def update_battery_soc(
    battery_energy_wh: float, nominal_energy_wh: float, soh_fraction: float, max_soc: float, min_soc: float
) -> Tuple[float, float]:
    """
    Calculate normalized and absolute SOC.

    Returns:
        Tuple of (soc_normalized, soc_absolute)
    """
    usable_cap = nominal_energy_wh * soh_fraction
    Emax = usable_cap * max_soc
    Emin = usable_cap * min_soc

    soc_normalized = (battery_energy_wh - Emin) / (Emax - Emin) if (Emax - Emin) > 0 else 0
    soc_normalized = np.clip(soc_normalized, 0, 1)

    soc_absolute = battery_energy_wh / usable_cap if usable_cap > 0 else 0
    soc_absolute = np.clip(soc_absolute, 0, 1)

    return soc_normalized, soc_absolute
