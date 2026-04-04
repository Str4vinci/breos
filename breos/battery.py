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
from typing import Tuple, Optional, List, Dict

import numpy as np
import pandas as pd
import rainflow

from breos.economics import BATTERY_REPLACEMENT_COST_PER_KWH
from breos.constants import (
    R_GAS, T_REF_K,
    NAUMANN_K0_PERCENT, NAUMANN_EA_J_MOL, NAUMANN_EXPONENT_B, NAUMANN_SOC_EXPONENT_N,
    LAM_K0_FRAC, LAM_EA_J_MOL, LAM_EXPONENT_B, LAM_SOC_EXPONENT_N,
    LAM_CAL_K0_FRAC, LAM_CAL_EA_J_MOL, LAM_CAL_EXPONENT_B, LAM_CAL_SOC_EXPONENT_N,
    LAM_CAL_RELAXED_K0_FRAC, LAM_CAL_RELAXED_EA_J_MOL, LAM_CAL_RELAXED_EXPONENT_B, LAM_CAL_RELAXED_SOC_EXPONENT_N,
    LAM_CAL_HOURLY_K0_FRAC, LAM_CAL_HOURLY_EA_J_MOL, LAM_CAL_HOURLY_EXPONENT_B, LAM_CAL_HOURLY_SOC_EXPONENT_N,
    MODERN_LFP_K0_FRAC, MODERN_LFP_EA_J_MOL, MODERN_LFP_EXPONENT_B, MODERN_LFP_SOC_EXPONENT_N,
    A_Q, B_Q, C_DOC_Q, D_DOC_Q, Z_Q,
    A_R, B_R, C_DOC_R, D_DOC_R, Z_R,
    NAUMANN_K0_R_PERCENT, NAUMANN_EA_R_J_MOL, NAUMANN_EXPONENT_B_R, NAUMANN_SOC_EXPONENT_N_R,
    DEFAULT_CHARGE_EFFICIENCY, DEFAULT_DISCHARGE_EFFICIENCY,
    DEFAULT_STANDBY_LOSS_WH, DEFAULT_MAX_SOC, DEFAULT_MIN_SOC,
    LFP_CAP_DERATE_PER_C_MODERATE, LFP_CAP_DERATE_PER_C_COLD,
    DEFAULT_THERMAL_RESISTANCE_KW,
)
from breos.utils import get_hours_per_step


@dataclass
class BatteryConfig:
    """
    Configuration parameters for battery simulation.
    
    For DC-coupled systems (hybrid inverters):
    - PV → Battery: No inverter loss (stays in DC)
    - Battery → Load: Inverter loss applies (DC to AC)
    
    For AC-coupled systems:
    - All energy goes through inverter first
    """
    nominal_energy_wh: float             # Required — nominal capacity in Wh
    initial_soh: float = 100.0          # Initial state of health (%)
    eol_percentage: float = 0.80        # End of life threshold (fraction)
    max_soc: float = DEFAULT_MAX_SOC
    min_soc: float = DEFAULT_MIN_SOC
    charge_efficiency: float = DEFAULT_CHARGE_EFFICIENCY
    discharge_efficiency: float = DEFAULT_DISCHARGE_EFFICIENCY
    standby_loss_wh: float = DEFAULT_STANDBY_LOSS_WH
    enable_replacement: bool = True
    replacement_cost: Optional[float] = None  # Auto-computed from cost per kWh if not set
    calendar_model: str = 'naumann_lam_calibrated'  # 'naumann_lam_calibrated', 'naumann_lam', 'naumann_lam_modern', 'naumann'
    # Resistance fade tracking
    enable_resistance_fade: bool = False  # Enable Naumann resistance growth model
    initial_resistance_growth: float = 0.0  # Initial relative resistance growth (fraction, 0=new)
    # Thermal model
    thermal_resistance_kw: float = DEFAULT_THERMAL_RESISTANCE_KW  # K/W for lumped thermal model
    # DC-coupled system (hybrid inverter) settings
    dc_coupled: bool = True             # True = hybrid inverter (DC-coupled battery)
    inverter_efficiency: float = 0.96   # Inverter efficiency (for DC→AC conversion)
    # Battery chemistry
    battery_type: str = 'lfp'           # Battery chemistry type

    def __post_init__(self):
        # Auto-compute replacement cost
        if self.replacement_cost is None:
            if self.nominal_energy_wh > 1:
                self.replacement_cost = BATTERY_REPLACEMENT_COST_PER_KWH * (self.nominal_energy_wh / 1000)
            else:
                self.replacement_cost = 0.0


def simulate_energy_balance(
    pv_dc: pd.Series,
    houseload: pd.DataFrame,
    battery_config: Optional[BatteryConfig] = None,
    start_time: Optional[pd.Timestamp] = None,
    end_time: Optional[pd.Timestamp] = None,
    freq: str = 'h',
    temperature_series: Optional[pd.Series] = None,
    results_directory: Optional[str] = None,
    initial_fec: float = 0.0,
    initial_calendar_seconds: float = 0.0,
    initial_resistance_growth: float = 0.0,
    initial_cumulative_cycle_deg: float = 0.0,
    initial_cumulative_cal_deg: float = 0.0,
    debug: bool = False
) -> Tuple[pd.DataFrame, float, pd.DataFrame, float, int, pd.DataFrame]:
    """
    Simulate energy balance with battery storage and degradation.
    
    This function processes PV DC production and load profiles to calculate
    grid interaction, battery state, and degradation. It properly handles
    DC-coupled and AC-coupled battery systems.
    
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
        debug: Enable debug output
        
    Returns:
        Tuple of:
        - results_df: Detailed timestep results
        - total_pv: Total PV DC production (Wh)
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
            load_utc = load_idx.tz_convert('UTC')
        else:
            load_utc = load_idx.tz_localize('UTC')

        rng_utc = rng.tz_convert('UTC') if rng.tz is not None else rng.tz_localize('UTC')

        # Only remap year if load covers a single year different from simulation.
        # Use dominant year (most frequent) to handle tz-aware indices that
        # span two calendar years in UTC (e.g., CET midnight = UTC 23:00 prev day).
        load_dominant_year = load_utc.year.value_counts().idxmax()
        sim_dominant_year = rng_utc.year.value_counts().idxmax()
        if load_dominant_year != sim_dominant_year:
            year_offset = sim_dominant_year - load_dominant_year
            load_utc = load_utc.map(lambda dt: dt.replace(year=dt.year + year_offset))

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
    
    # Get degradation model parameters
    k0_frac, Ea_val, b_val, n_val = _get_degradation_params(battery_config.calendar_model)
    
    # Initialize state
    battery_soh_decimal = battery_config.initial_soh / 100.0
    Battery_SOH = battery_config.initial_soh
    Battery_Energy_Wh = battery_config.nominal_energy_wh * battery_soh_decimal * battery_config.max_soc

    soc_absolute_buffer = np.empty(steps_per_day, dtype=np.float64)
    soc_buf_idx = 0
    fec_cum = initial_fec
    cumulative_cal_seconds = initial_calendar_seconds
    resistance_growth = initial_resistance_growth
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
    # Hoist invariant check out of the loop
    has_battery = battery_config.nominal_energy_wh > 1 and (battery_config.max_soc - battery_config.min_soc) > 0
    # Bind capacity factor function once
    _cap_factor_fn = lfp_capacity_factor

    for i in range(n_steps):
        step_time = rng[i]
        # Get values for this timestep via fast array indexing
        pv_dc_power = _pv_dc_vals[i] * hours_per_step  # DC power (Wh) before inverter
        load = _load_vals[i] * hours_per_step  # AC Load in Wh
        T_ambient = _temp_vals[i]
        T_cell = T_ambient  # default; overridden by thermal model below

        Import = Sell = 0.0
        charge_in = discharge_out = 0.0
        pv_to_load_ac = 0.0  # PV power going directly to load (after inverter)

        if has_battery:
            # Calculate usable capacity with temperature derating
            usable_cap = battery_config.nominal_energy_wh * battery_soh_decimal
            f_cap = _cap_factor_fn(T_cell)
            Emax = usable_cap * battery_config.max_soc * f_cap
            Emin = usable_cap * battery_config.min_soc * f_cap
            
            # Apply standby loss (scaled by timestep)
            standby_loss = battery_config.standby_loss_wh * hours_per_step
            Battery_Energy_Wh = max(Emin, Battery_Energy_Wh - standby_loss)
            
            # Convert PV DC to AC for comparison with load
            pv_ac = pv_dc_power * battery_config.inverter_efficiency
            
            # Energy flows:
            # Load is in AC terms
            # We have pv_dc_power in DC terms
            # Battery operates in DC
            
            if pv_ac >= load:  # Surplus: PV covers load + potential charging
                # PV to load (AC)
                pv_to_load_ac = load
                
                # Remaining DC available for battery or export
                # (pv_dc - pv_to_load_dc) where pv_to_load_dc = load / inverter_efficiency
                dc_to_load = load / battery_config.inverter_efficiency if battery_config.inverter_efficiency > 0 else load
                surplus_dc = pv_dc_power - dc_to_load
                
                # Charge battery with surplus DC (no inverter loss)
                charge_room = Emax - Battery_Energy_Wh
                charge_in = min(surplus_dc, charge_room / battery_config.charge_efficiency)
                Battery_Energy_Wh += charge_in * battery_config.charge_efficiency
                
                # Export remainder (DC -> AC)
                remaining_dc = surplus_dc - charge_in
                Sell = remaining_dc * battery_config.inverter_efficiency
                
            else:  # Deficit: PV insufficient, need battery or grid
                # All PV goes to load first
                pv_to_load_ac = pv_ac
                deficit = load - pv_ac  # AC deficit
                
                # Try to cover deficit from battery
                available = Battery_Energy_Wh - Emin
                if available > 0:
                    # Battery discharge: DC -> AC
                    eff_total = battery_config.discharge_efficiency * battery_config.inverter_efficiency
                    # How much DC do we need to draw to provide 'deficit' AC?
                    dc_needed = deficit / eff_total if eff_total > 0 else deficit
                    draw = min(available, dc_needed)
                    Battery_Energy_Wh -= draw
                    delivered_ac = draw * eff_total
                    discharge_out = draw
                    Import = max(0.0, deficit - delivered_ac)
                else:
                    Import = deficit
        else:
            # No battery: simple DC to AC for load
            pv_ac = pv_dc_power * battery_config.inverter_efficiency
            if pv_ac >= load:
                pv_to_load_ac = load
                Sell = pv_ac - load
            else:
                pv_to_load_ac = pv_ac
                Import = load - pv_ac
        
        # Compute cell temperature via lumped thermal model
        if has_battery and battery_config.thermal_resistance_kw > 0:
            # charge_in and discharge_out are in Wh; convert to W for thermal calc
            charge_power_w = charge_in / hours_per_step if hours_per_step > 0 else 0.0
            discharge_power_w = discharge_out / hours_per_step if hours_per_step > 0 else 0.0
            T_cell = compute_cell_temperature(
                T_ambient,
                charge_power_w,
                discharge_power_w,
                battery_config.charge_efficiency,
                battery_config.discharge_efficiency,
                battery_config.thermal_resistance_kw,
            )
        T_cell_day_sum += T_cell

        # SOC calculations (handle no-battery case)
        if has_battery:
            soc_normalized = (Battery_Energy_Wh - Emin) / (Emax - Emin) if (Emax - Emin) > 0 else 0.0
            soc_normalized = max(0.0, min(1.0, soc_normalized))
            soc_absolute = Battery_Energy_Wh / (battery_config.nominal_energy_wh * battery_soh_decimal) if battery_soh_decimal > 0 else 0.0
            soc_absolute = max(0.0, min(1.0, soc_absolute))
        else:
            soc_normalized = 0.0
            soc_absolute = 0.0
            Emax = 0.0
            Emin = 0.0
        soc_absolute_buffer[soc_buf_idx] = soc_absolute
        soc_buf_idx += 1

        # Store results via array indexing (avoids per-timestep dict overhead)
        _res_pv_dc[i] = pv_dc_power / hours_per_step
        _res_pv_prod[i] = pv_ac / hours_per_step
        _res_load[i] = load / hours_per_step
        _res_pv_delta[i] = (pv_ac - load) / hours_per_step
        _res_import[i] = Import / hours_per_step
        _res_sell[i] = Sell / hours_per_step
        _res_batt_e[i] = Battery_Energy_Wh if has_battery else 0.0
        _res_soc_norm[i] = soc_normalized
        _res_soc_abs[i] = soc_absolute
        _res_soh[i] = Battery_SOH if has_battery else 100.0
        _res_tcell[i] = T_cell
        
        # Daily degradation update
        if soc_buf_idx >= steps_per_day:
            # Build Series from buffer using rng slice (avoids pd.date_range overhead)
            day_start_i = i - steps_per_day + 1
            soc_series = pd.Series(
                soc_absolute_buffer[:steps_per_day].copy(),
                index=rng[day_start_i:i + 1],
            )

            # Cycle degradation
            soh_after_cycle, dSOH_cycle, fec_cum = update_battery_soh_cyclewise(
                battery_soh_decimal,
                soc_series,
                battery_config.nominal_energy_wh,
                fec_cum=fec_cum,
                battery_type=battery_config.battery_type,
                debug=debug
            )

            # Calendar degradation — use mean cell temperature over the day
            mean_soc_abs = float(soc_series.mean())
            mean_T_cell = T_cell_day_sum / steps_per_day
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
                debug=debug
            )

            battery_soh_decimal = soh_after_calendar

            Battery_SOH = battery_soh_decimal * 100.0

            cumulative_cycle_deg += dSOH_cycle
            cumulative_cal_deg += dSOH_calendar

            # Resistance fade (opt-in)
            dR_cycle = 0.0
            dR_calendar = 0.0
            effective_rte = battery_config.charge_efficiency * battery_config.discharge_efficiency
            if battery_config.enable_resistance_fade:
                # Get cycles for resistance calculation
                time_index = soc_series.index
                cycles = detect_cycles_rainflow(soc_series, time_index, min_doc_fraction=0.01)

                # Compute FEC at start of day (before today's cycles were added)
                day_fec = sum(
                    max(0.0, min(1.0, c['doc'])) * c.get('count', 1.0) for c in cycles
                )
                fec_before_day = fec_cum - day_fec

                resistance_growth, dR_cycle = update_battery_resistance_cyclewise(
                    resistance_growth, cycles, fec_before_day,
                    debug=debug
                )
                resistance_growth, dR_calendar = update_battery_resistance_calendar(
                    resistance_growth,
                    T_cell_C=T_cell,
                    cumulative_cal_seconds=cumulative_cal_seconds,
                    dt_days=1.0,
                    mean_soc_absolute=mean_soc_abs,
                    debug=debug
                )
                cumulative_resistance_cycle += dR_cycle
                cumulative_resistance_calendar += dR_calendar

                effective_rte = (battery_config.charge_efficiency * battery_config.discharge_efficiency) / (1.0 + resistance_growth)

            # Battery replacement check
            if battery_config.enable_replacement and battery_soh_decimal <= battery_config.eol_percentage:
                battery_soh_decimal = 1.0
                Battery_SOH = 100.0
                fec_cum = 0.0
                cumulative_cal_seconds = 0.0
                resistance_growth = 0.0
                Battery_Energy_Wh = battery_config.nominal_energy_wh * battery_config.max_soc
                n_replacements += 1
                total_replacement_cost += battery_config.replacement_cost
                _res_replaced[i] = True
                _res_repl_cost[i] = battery_config.replacement_cost
                cumulative_cycle_deg = 0.0
                cumulative_cal_deg = 0.0
                cumulative_resistance_cycle = 0.0
                cumulative_resistance_calendar = 0.0

                if debug:
                    print(f"\n*** BATTERY REPLACED at {step_time} ***")

            degradation_record = {
                'Datetime': step_time,
                'SOH': Battery_SOH,
                'Cycle_Degradation': dSOH_cycle,
                'Calendar_Degradation': dSOH_calendar,
                'Cumulative_Cycle_Degradation': cumulative_cycle_deg,
                'Cumulative_Calendar_Degradation': cumulative_cal_deg,
                'Cumulative_FEC': fec_cum,
                'Cumulative_Calendar_Seconds': cumulative_cal_seconds,
                'Total_Degradation': 1.0 - battery_soh_decimal,
                'Mean_SOC_Absolute': mean_soc_abs,
            }
            if battery_config.enable_resistance_fade:
                degradation_record['Resistance_Growth'] = resistance_growth
                degradation_record['Effective_RTE'] = effective_rte
            degradation_tracking.append(degradation_record)

            # Reset daily accumulators
            soc_buf_idx = 0
            T_cell_day_sum = 0.0
    
    # Build results DataFrame from pre-allocated arrays
    df = pd.DataFrame({
        'Datetime': rng,
        'PV_DC': _res_pv_dc,
        'PV_Production': _res_pv_prod,
        'Houseload': _res_load,
        'PV_Delta': _res_pv_delta,
        'Import_From_Grid': _res_import,
        'Sell_To_Grid': _res_sell,
        'Battery_Energy': _res_batt_e,
        'Battery_SOC_Normalized': _res_soc_norm,
        'Battery_SOC_Absolute': _res_soc_abs,
        'Battery_SOH': _res_soh,
        'T_cell': _res_tcell,
        'Battery_Replaced': _res_replaced,
        'Replacement_Cost': _res_repl_cost,
    })
    deg_df = pd.DataFrame(degradation_tracking) if degradation_tracking else pd.DataFrame()

    # Summary calculations (use numpy sums on arrays directly)
    total_pv = _res_pv_prod.sum() * hours_per_step
    total_load = _res_load.sum() * hours_per_step
    total_sell = _res_sell.sum() * hours_per_step
    total_import = _res_import.sum() * hours_per_step
    
    percentage_imported = (total_import / total_load * 100) if total_load > 0 else 0
    
    summary = {
        'Total PV [kWh]': total_pv / 1000.0,
        'Total Load [kWh]': total_load / 1000.0,
        'Sell [kWh]': total_sell / 1000.0,
        'Import [kWh]': total_import / 1000.0,
        'Import [%]': percentage_imported,
        'Grid Independence [%]': 100 - percentage_imported,
        'Final SOH [%]': Battery_SOH,
        'N_Replacements': n_replacements,
        'Replacement_Cost': total_replacement_cost
    }
    summary_df = pd.DataFrame([summary])
    
    return df, total_pv, summary_df, total_replacement_cost, n_replacements, deg_df


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
    P_loss_discharge = discharge_power_w * (1.0 / discharge_eff - 1.0) if discharge_eff > 0 else 0.0

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
        'naumann'                      — Naumann 2020 calendar + cycle (NMC/LFP lab)
        'naumann_lam'                  — Naumann cycle + Lam 2025 lab-derived calendar
        'naumann_lam_calibrated'       — Naumann cycle + Lam calendar, field-calibrated (default)
        'naumann_lam_calibrated_hourly'— Naumann cycle + Lam calendar, field-calibrated, hourly res.
        'naumann_lam_modern'           — Naumann cycle + Lam calendar, projected 0.5×k₀ for 2020+ cells
    """
    model_lower = model.lower().replace('-', '_')

    # ── Naumann (pure) ────────────────────────────────────────────────────
    if model_lower == 'naumann':
        k0_frac = NAUMANN_K0_PERCENT / 100.0
        return k0_frac, NAUMANN_EA_J_MOL, NAUMANN_EXPONENT_B, NAUMANN_SOC_EXPONENT_N

    # ── Naumann-Lam: lab-derived ──────────────────────────────────────────
    elif model_lower in ('naumann_lam', 'lam'):
        return LAM_K0_FRAC, LAM_EA_J_MOL, LAM_EXPONENT_B, LAM_SOC_EXPONENT_N

    # ── Naumann-Lam: field-calibrated (default) ──────────────────────────
    elif model_lower in ('naumann_lam_calibrated', 'lam_calibrated'):
        return LAM_CAL_K0_FRAC, LAM_CAL_EA_J_MOL, LAM_CAL_EXPONENT_B, LAM_CAL_SOC_EXPONENT_N

    # ── Naumann-Lam: field-calibrated relaxed ─────────────────────────────
    elif model_lower in ('naumann_lam_calibrated_relaxed', 'lam_calibrated_relaxed',
                         'lam_calibrated_1.5ea'):
        return LAM_CAL_RELAXED_K0_FRAC, LAM_CAL_RELAXED_EA_J_MOL, LAM_CAL_RELAXED_EXPONENT_B, LAM_CAL_RELAXED_SOC_EXPONENT_N

    # ── Naumann-Lam: field-calibrated hourly ──────────────────────────────
    elif model_lower in ('naumann_lam_calibrated_hourly', 'lam_calibrated_hourly'):
        return LAM_CAL_HOURLY_K0_FRAC, LAM_CAL_HOURLY_EA_J_MOL, LAM_CAL_HOURLY_EXPONENT_B, LAM_CAL_HOURLY_SOC_EXPONENT_N

    # ── Naumann-Lam: modern LFP projection ───────────────────────────────
    elif model_lower in ('naumann_lam_modern', 'modern_lfp'):
        return MODERN_LFP_K0_FRAC, MODERN_LFP_EA_J_MOL, MODERN_LFP_EXPONENT_B, MODERN_LFP_SOC_EXPONENT_N

    else:
        raise ValueError(
            f"Unknown calendar model: {model}. Use 'naumann_lam_calibrated', 'naumann_lam', "
            f"'naumann_lam_calibrated_hourly', 'naumann_lam_modern', or 'naumann'. "
            f"Legacy aliases ('lam_calibrated', 'lam', 'modern_lfp') are also accepted."
        )


def detect_half_cycles_from_soc_series(
    soc_abs_series: pd.Series,
    time_index: pd.DatetimeIndex,
    tiny_hysteresis: float = 1e-4
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
        mean_soc = float(np.mean(soc[sidx:eidx + 1]))
        duration_h = (times[eidx] - times[sidx]).total_seconds() / 3600.0
        mean_c_rate = 0.0 if duration_h <= 0 else doc / duration_h
        
        half_cycles.append({
            'start_idx': sidx,
            'end_idx': eidx,
            'doc': doc,
            'mean_soc': mean_soc,
            'mean_c_rate': mean_c_rate,
            'duration_h': duration_h
        })
    
    return half_cycles, soc_abs_series


def detect_cycles_rainflow(
    soc_abs_series: pd.Series,
    time_index: pd.DatetimeIndex,
    min_doc_fraction: float = 0.01
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

        cycles.append({
            'doc': doc,
            'mean_soc': mean_soc,
            'count': count,          # 1.0 for full, 0.5 for half
            'mean_c_rate': mean_c_rate,
            'start_idx': i_start,
            'end_idx': i_end,
        })

    return cycles


def compute_halfcycle_energy_throughput(
    hc: Dict,
    soc_series_absolute: pd.Series,
    nominal_energy_Wh: float
) -> float:
    """Compute energy throughput (Wh) for a half-cycle."""
    s = soc_series_absolute.iloc[hc['start_idx']:hc['end_idx'] + 1].values
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
    resistance_growth: float,
    cycles: List[Dict],
    fec_cum: float,
    min_DoD_fraction: float = 0.01,
    debug: bool = False
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
        DOC = max(0.0, min(1.0, cyc['doc']))
        if DOC < min_DoD_fraction:
            continue

        count = cyc.get('count', 1.0)
        dFEC = DOC * count
        mean_c_rate = cyc['mean_c_rate']

        kC = k_c_rate_R(mean_c_rate)
        kDOC = k_doc_R(DOC)

        fec_new = running_fec + dFEC

        # Differential form: dR% = kC * kDOC * (FEC_new^Z_R - FEC_old^Z_R)
        dR_percent = kC * kDOC * (fec_new ** Z_R - running_fec ** Z_R)
        dR_fraction = dR_percent / 100.0

        delta_R += dR_fraction
        running_fec = fec_new

        if debug:
            print(f"[R-cycle] DOC={DOC:.4f}, C-rate={mean_c_rate:.4f}, "
                  f"dR={dR_fraction*100:.6f}%")

    new_growth = resistance_growth + delta_R
    return new_growth, delta_R


def update_battery_resistance_calendar(
    resistance_growth: float,
    T_cell_C: float,
    cumulative_cal_seconds: float,
    dt_days: float = 1.0,
    mean_soc_absolute: float = 0.5,
    debug: bool = False
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
        print(f"[R-calendar] T={T_cell_C:.1f}°C, dR={dR_fraction*100:.6f}%")

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


def _get_cycle_params(battery_type: str = 'lfp') -> Tuple[float, float, float, float, float]:
    """Get cycle aging (Naumann-style) parameters for a battery chemistry.

    Returns:
        Tuple of (a_q, b_q, c_doc_q, d_doc_q, z_q)
    """
    return (A_Q, B_Q, C_DOC_Q, D_DOC_Q, Z_Q)


def update_battery_soh_cyclewise(
    soh_start_fraction: float,
    soc_series_absolute: pd.Series,
    nominal_energy_Wh: float,
    fec_cum: float = 0.0,
    min_DoD_fraction: float = 0.01,
    use_rainflow: bool = True,
    battery_type: str = 'lfp',
    debug: bool = False
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
        DOC = max(0.0, min(1.0, cyc['doc']))
        if DOC < min_DoD_fraction:
            continue

        mean_c_rate = cyc['mean_c_rate']
        # For rainflow cycles: count is 1.0 (full) or 0.5 (half)
        # For extrema-based: each entry is a half-cycle (count=1 implicitly)
        count = cyc.get('count', 1.0)

        # Energy throughput for this cycle: DOC * count * nominal
        dFEC = DOC * count

        # Naumann-style k-factors with technology-specific coefficients
        kC = max(0.0, a_q * mean_c_rate + b_q)
        kDOC = max(0.0, c_doc_q * ((DOC - 0.6) ** 3) + d_doc_q)

        fec_new = fec_cum + dFEC

        # Differential form using cumulative FEC (Naumann Eq. 5-6)
        dq_percent = kC * kDOC * (fec_new ** z_q - fec_cum ** z_q)
        dq_fraction = dq_percent / 100.0

        qloss_cycle_fraction += dq_fraction
        fec_cum = fec_new

        if debug:
            print(f"[cycle] DOC={DOC:.4f}, C-rate={mean_c_rate:.4f}, count={count}, "
                  f"dFEC={dFEC:.6e}, dq={dq_fraction*100:.6f}%")

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
    debug: bool = False
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
        print(f"[calendar] T={T_cell_C}°C, b={cal_b:.2f}, Δt^b={delta_time_factor:.2f}, "
              f"d_soh={d_soh_fraction*100:.6f}%")
    
    return soh_after, d_soh_fraction, t_new


def update_battery_soc(
    battery_energy_wh: float,
    nominal_energy_wh: float,
    soh_fraction: float,
    max_soc: float,
    min_soc: float
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
