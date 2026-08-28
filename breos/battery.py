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
from typing import Any, Dict, List, Optional, Tuple, Union

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
from breos.degradation.protocol import (
    BlastDegradationAdapter,
    DegradationDay,
    DegradationLifecycle,
    NativeDegradationAdapter,
)
from breos.economics import BATTERY_REPLACEMENT_COST_PER_KWH
from breos.execution import (  # noqa: F401  -- EXECUTION_BACKENDS re-exported
    EXECUTION_BACKENDS,
    validate_execution_backend,
)
from breos.inverter import _calculate_dc_ac_power_arrays, calculate_dc_ac_power, dc_power_for_ac_output
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


def _align_simulation_inputs(
    pv_dc: pd.Series,
    houseload: pd.DataFrame,
    temperature_series: Optional[pd.Series],
    rng: pd.DatetimeIndex,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reindex PV, load and temperature onto ``rng`` as float64 arrays.

    The loop indexes these positionally, so everything the simulation reads
    per step is settled here: gaps become zero generation / zero load / 25C,
    and the load profile is year-shifted when it comes from a different year
    than the simulation window.
    """
    pv_values = pv_dc.reindex(rng).fillna(0.0)

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
        houseload_series.index = pv_values.index
    houseload_series = houseload_series.reindex(rng).fillna(0.0)

    if temperature_series is None:
        temperature_series = pd.Series(25.0, index=rng)
    else:
        temperature_series = temperature_series.reindex(rng).fillna(25.0)

    return (
        pv_values.values.astype(np.float64),
        houseload_series.values.astype(np.float64),
        temperature_series.values.astype(np.float64),
    )


def _resolve_degradation_engine(
    degradation_engine: str,
    blast_model: Optional[str],
    initial_degradation_state: Optional[Dict[str, Any]],
    battery_config: BatteryConfig,
) -> str:
    """Normalise the degradation backend name and reject incoherent pairings.

    Every combination that a backend could only honour by silently ignoring
    one of its inputs fails here, before any simulation work.
    """
    engine_key = str(degradation_engine).strip().lower()
    if engine_key not in {"native", "blast"}:
        raise ValueError("degradation_engine must be 'native' or 'blast'")

    if engine_key == "native" and blast_model is not None:
        raise ValueError("blast_model requires degradation_engine='blast'")
    if engine_key == "native" and initial_degradation_state is not None:
        raise ValueError("initial_degradation_state requires degradation_engine='blast'")
    if engine_key == "blast" and not blast_model:
        raise ValueError("blast_model is required when degradation_engine='blast'")
    if engine_key == "blast" and battery_config.enable_resistance_fade:
        raise ValueError("degradation_engine='blast' cannot be combined with enable_resistance_fade")
    return engine_key


def _resolve_carried_energy(
    initial_energy_wh: Optional[float],
    initial_pv_origin_energy_wh: Optional[float],
    battery_config: BatteryConfig,
    battery_soh_decimal: float,
) -> Tuple[float, float]:
    """Validate the carried stored-energy state, returning ``(energy, pv_origin)``.

    Both default for a fresh run: a battery starting full at its configured
    max SOC, with none of that energy attributable to PV.
    """
    if initial_energy_wh is None:
        energy_wh = battery_config.nominal_energy_wh * battery_soh_decimal * battery_config.max_soc
    else:
        if isinstance(initial_energy_wh, (bool, np.bool_)):
            raise ValueError("initial_energy_wh must be a finite number, not a bool")
        try:
            energy_wh = float(initial_energy_wh)
        except (TypeError, ValueError) as exc:
            raise ValueError("initial_energy_wh must be a finite number") from exc
        if not math.isfinite(energy_wh):
            raise ValueError("initial_energy_wh must be a finite number")
        if not 0.0 <= energy_wh <= battery_config.nominal_energy_wh:
            raise ValueError(
                f"initial_energy_wh must be between 0 and nominal_energy_wh ({battery_config.nominal_energy_wh:g} Wh)"
            )

    if initial_pv_origin_energy_wh is None:
        pv_origin_wh = 0.0
    else:
        if isinstance(initial_pv_origin_energy_wh, (bool, np.bool_)):
            raise ValueError("initial_pv_origin_energy_wh must be a finite number, not a bool")
        try:
            pv_origin_wh = float(initial_pv_origin_energy_wh)
        except (TypeError, ValueError) as exc:
            raise ValueError("initial_pv_origin_energy_wh must be a finite number") from exc
        if not math.isfinite(pv_origin_wh):
            raise ValueError("initial_pv_origin_energy_wh must be a finite number")
        if not 0.0 <= pv_origin_wh <= energy_wh:
            raise ValueError("initial_pv_origin_energy_wh must be between 0 and initial_energy_wh")

    return energy_wh, pv_origin_wh


def _build_degradation_lifecycle(
    engine_key: str,
    battery_config: BatteryConfig,
    *,
    battery_soh_decimal: float,
    has_battery: bool,
    blast_model: Optional[str],
    initial_degradation_state: Optional[Dict[str, Any]],
    initial_fec: float,
    initial_calendar_seconds: float,
    initial_cumulative_cycle_deg: float,
    initial_cumulative_cal_deg: float,
    default_day_start_soc: float,
    default_day_start_t_cell: float,
    debug: bool,
) -> Tuple[DegradationLifecycle, float, float]:
    """Construct the degradation backend and its first day-boundary state.

    Returns the lifecycle adapter plus the SOC and cell temperature the first
    daily step should treat as the previous day's endpoint. BLAST consumes
    that boundary pair (its daily endpoint model spans midnight); the native
    adapter ignores it, so the defaults only matter for the BLAST path, where
    a carried snapshot overrides them.
    """
    if engine_key != "blast":
        lifecycle: DegradationLifecycle = NativeDegradationAdapter(
            model_key=battery_config.calendar_model,
            initial_soh_fraction=battery_soh_decimal,
            initial_fec=initial_fec,
            initial_calendar_seconds=initial_calendar_seconds,
            initial_cumulative_cycle_degradation=initial_cumulative_cycle_deg,
            initial_cumulative_calendar_degradation=initial_cumulative_cal_deg,
            nominal_energy_wh=battery_config.nominal_energy_wh,
            battery_type=battery_config.battery_type,
            **_native_degradation_kwargs(battery_config.calendar_model),
            cycle_step=_update_battery_soh_cyclewise_arrays,
            calendar_step=update_battery_soh_calendar,
            debug=debug,
        )
        return lifecycle, default_day_start_soc, default_day_start_t_cell

    if not has_battery:
        raise ValueError("degradation_engine='blast' requires a configured battery")

    state_payload = initial_degradation_state or {}
    blast_snapshot = state_payload.get("blast_engine", state_payload)
    if not blast_snapshot and not math.isclose(battery_config.initial_soh, 100.0):
        raise ValueError("BLAST starts from a beginning-of-life model unless initial_degradation_state is provided")
    lifecycle = BlastDegradationAdapter(
        str(blast_model),
        initial_state=state_payload,
        initial_fec=initial_fec,
        initial_calendar_seconds=initial_calendar_seconds,
        initial_cumulative_cycle_degradation=initial_cumulative_cycle_deg,
        initial_cumulative_calendar_degradation=initial_cumulative_cal_deg,
    )
    return (
        lifecycle,
        float(state_payload.get("day_start_soc_absolute", default_day_start_soc)),
        float(state_payload.get("day_start_temperature_c", default_day_start_t_cell)),
    )


def _native_degradation_kwargs(calendar_model: str) -> Dict[str, float]:
    """Map a calendar-model name onto the native adapter's parameter names."""
    k0_frac, activation_energy, time_exponent, soc_exponent = _get_degradation_params(calendar_model)
    return {
        "k0_fraction": k0_frac,
        "activation_energy": activation_energy,
        "soc_exponent": soc_exponent,
        "time_exponent": time_exponent,
    }


def _apply_capacity_window(
    nominal_energy_wh: float,
    soh_fraction: float,
    max_soc: float,
    min_soc: float,
    standby_loss_wh: float,
    energy_wh: float,
    pv_origin_wh: float,
    t_cell: float,
) -> Tuple[float, float, float, float, float, float]:
    """Derate the usable SOC window and bleed standby loss, before dispatch.

    Returns ``(energy, pv_origin, emin, emax, capacity_window_loss, standby)``,
    all in Wh. Assumes a configured battery; the no-battery case never calls
    this. ``standby_loss_wh`` is already scaled to the timestep.

    ``t_cell`` here is the ambient/indoor temperature at step start, not the
    self-heated cell temperature the thermal model produces later in the same
    step: usable capacity is set by the pack's state *before* this step's
    charge/discharge self-heating, while aging sees the warmed cell. That
    split is intentional.

    A temperature- or SOH-driven fall in ``emax`` is booked as an explicit
    loss — it is neither export nor standby consumption — and the lower
    reserve is a dispatch boundary that must never create energy when it
    rises. The PV-origin share is rescaled with every reduction so it stays a
    fraction of what is actually stored.
    """
    usable_cap = nominal_energy_wh * soh_fraction
    f_cap = lfp_capacity_factor(t_cell)
    emax = usable_cap * max_soc * f_cap
    emin = usable_cap * min_soc * f_cap

    capacity_window_loss = max(0.0, energy_wh - emax)
    if capacity_window_loss > 0.0 and energy_wh > 0.0:
        pv_origin_wh *= emax / energy_wh
        energy_wh = emax

    removable_for_standby = max(0.0, energy_wh - emin)
    standby = min(standby_loss_wh, removable_for_standby)
    if standby > 0.0 and energy_wh > 0.0:
        pv_origin_wh *= (energy_wh - standby) / energy_wh
        energy_wh -= standby

    return energy_wh, pv_origin_wh, emin, emax, capacity_window_loss, standby


def _step_energy_cap(power_w: Optional[float], hours_per_step: float) -> float:
    """Convert a nameplate power limit (W) to a per-step energy cap (Wh).

    ``None`` means unlimited, which the dispatch kernel reads as an infinite
    cap rather than as a separate branch.
    """
    return power_w * hours_per_step if power_w is not None else float("inf")


# Explicit per-step energy flows and losses, in the column order they appear
# in the results frame. Every entry is accumulated in Wh during the loop and
# divided by the step length on write, so the frame reports average W.
_LEDGER_COLUMNS: Tuple[str, ...] = (
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

# Per-step state columns, in their row order inside the shared buffer matrix.
# The ledger columns occupy the rows immediately after them. The compiled
# dispatch kernel addresses rows by these indices, so the order is part of the
# kernel contract and must not be reordered without updating it.
_STATE_ROWS: Tuple[str, ...] = (
    "pv_dc",
    "pv_production",
    "load",
    "pv_delta",
    "grid_import",
    "grid_export",
    "battery_energy",
    "soc_normalized",
    "soc_absolute",
    "soh",
    "t_cell",
    "pv_curtailment",
    "charge_loss",
    "discharge_loss",
    "standby_loss",
    "battery_energy_begin",
    "pv_origin_begin",
    "pv_origin_end",
)
_STATE_ROW_INDEX: Dict[str, int] = {name: row for row, name in enumerate(_STATE_ROWS)}
_LEDGER_ROW0: int = len(_STATE_ROWS)
_LEDGER_ROW_INDEX: Dict[str, int] = {name: _LEDGER_ROW0 + offset for offset, name in enumerate(_LEDGER_COLUMNS)}
_N_ROWS: int = _LEDGER_ROW0 + len(_LEDGER_COLUMNS)


class _ResultBuffers:
    """Pre-allocated per-timestep output arrays and their frame layout.

    The simulation writes into these positionally rather than appending dicts,
    which is what keeps an 8760-step loop out of pandas. Owning both the
    allocation and :meth:`to_frame` here means the column set is described
    once instead of drifting between two ends of a 700-line function.

    ``replaced`` and ``replacement_cost`` are zero-filled because only
    replacement days write them; every other array is fully overwritten each
    step and is left uninitialised.
    """

    __slots__ = (
        "matrix",
        "replaced",
        "replacement_cost",
        "ledger",
    ) + _STATE_ROWS

    def __init__(self, n_steps: int) -> None:
        # One row per per-step column, so a whole day of every output can be
        # handed to a compiled kernel as a single contiguous array. Each named
        # attribute below is a view on its row, not a copy.
        self.matrix = np.empty((_N_ROWS, n_steps))
        for row, name in enumerate(_STATE_ROWS):
            setattr(self, name, self.matrix[row])
        self.replaced = np.zeros(n_steps, dtype=bool)
        self.replacement_cost = np.zeros(n_steps)
        self.ledger = {key: self.matrix[_LEDGER_ROW0 + offset] for offset, key in enumerate(_LEDGER_COLUMNS)}

    def zero_fill(self) -> None:
        """Zero every per-step column the caller is not going to write."""
        self.matrix.fill(0.0)

    def column_arrays(self) -> Dict[str, np.ndarray]:
        """Return every per-timestep output column, keyed by its frame name."""
        return _column_arrays(self)

    def to_frame(self, rng: pd.DatetimeIndex) -> pd.DataFrame:
        """Assemble the public per-timestep results frame."""
        return pd.DataFrame({"Datetime": rng, **self.column_arrays()})


def _column_arrays(buffers: Any) -> Dict[str, np.ndarray]:
    """Return every per-timestep output column of *buffers*, keyed by frame name.

    Both buffer types and both result builders read a run through this one
    mapping, so no path can report a different column set from the one the
    detailed frame exposes.
    """
    return {
        "PV_DC": buffers.pv_dc,
        "PV_Production": buffers.pv_production,
        "Houseload": buffers.load,
        "PV_Delta": buffers.pv_delta,
        "Import_From_Grid": buffers.grid_import,
        "Sell_To_Grid": buffers.grid_export,
        "Battery_Energy": buffers.battery_energy,
        "Battery_SOC_Normalized": buffers.soc_normalized,
        "Battery_SOC_Absolute": buffers.soc_absolute,
        "Battery_SOH": buffers.soh,
        "T_cell": buffers.t_cell,
        "Battery_Replaced": buffers.replaced,
        "Replacement_Cost": buffers.replacement_cost,
        "PV_Curtailment": buffers.pv_curtailment,
        "Battery_Charge_Loss": buffers.charge_loss,
        "Battery_Discharge_Loss": buffers.discharge_loss,
        "Battery_Standby_Loss": buffers.standby_loss,
        # Stored-energy state columns are Wh; all explicit flow/loss
        # ledger columns are average W over the timestep. The
        # end-of-step energy is the same array as "Battery_Energy".
        "Battery_Energy_Beginning": buffers.battery_energy_begin,
        "Battery_Energy_End": buffers.battery_energy,
        "Battery_PV_Origin_Energy_Beginning": buffers.pv_origin_begin,
        "Battery_PV_Origin_Energy_End": buffers.pv_origin_end,
        **buffers.ledger,
    }


def _column_sums(columns: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Total every column, summing each distinct array only once.

    Several names are served by one array: ``Battery_Energy_End`` is the same
    array as ``Battery_Energy`` on every path, and a PV-only run additionally
    shares one zero array between every column it never writes. Keying the
    work by array identity rather than by column name means those columns
    cost one reduction between them instead of one apiece, and reports the
    same float for each, because it is literally the same reduction.
    """
    totals: Dict[int, float] = {}
    sums: Dict[str, float] = {}
    for name, values in columns.items():
        key = id(values)
        total = totals.get(key)
        if total is None:
            # Reduced over the same contiguous values, in the same order
            # pandas would use, so a summary total is bit-identical to the
            # detailed frame's total rather than merely close to it.
            total = totals[key] = float(np.sum(values))
        sums[name] = total
    return sums


# The per-step columns a PV-only run actually writes. Everything else in
# :func:`_column_arrays` stays at zero when there is no battery.
_PV_ONLY_STATE_ROWS: Tuple[str, ...] = (
    "pv_dc",
    "pv_production",
    "load",
    "pv_delta",
    "grid_import",
    "grid_export",
    "soh",
    "t_cell",
    "pv_curtailment",
)
_PV_ONLY_LEDGER_COLUMNS: Tuple[str, ...] = (
    "PV_DC_To_Inverter",
    "PV_AC_To_Load",
    "PV_Direct_Inverter_Loss",
)


class _PvOnlySummaryBuffers:
    """Reduced per-step buffers for a PV-only run that only owes a summary.

    A system with no battery leaves 24 of the columns :func:`_column_arrays`
    reports at zero for every step, and writes three more with values it has
    already written under another name. Allocating the full
    ``(_N_ROWS, n_steps)`` matrix to hold that costs about three times the
    memory such a run needs, and a Monte Carlo study pays it once per
    simulated year in every worker at once -- which is memory traffic, not
    arithmetic, and so is exactly what stops the study scaling across cores.

    This type presents the same reading interface over twelve written arrays,
    one shared zero array and one shared zero mask. It is deliberately a
    summary-path type with no ``to_frame``: the detailed frame must not hand
    a caller aliased columns it could write through.
    """

    __slots__ = ("zeros", "replaced", "replacement_cost", "ledger") + _STATE_ROWS

    def __init__(self, n_steps: int) -> None:
        self.zeros = np.zeros(n_steps)
        written = frozenset(_PV_ONLY_STATE_ROWS)
        for name in _STATE_ROWS:
            # Written rows are left uninitialised; the dispatch overwrites
            # every element of each one before anything reads it.
            setattr(self, name, np.empty(n_steps) if name in written else self.zeros)
        self.replaced = np.zeros(n_steps, dtype=bool)
        self.replacement_cost = self.zeros

        ledger: Dict[str, np.ndarray] = {name: np.empty(n_steps) for name in _PV_ONLY_LEDGER_COLUMNS}
        # Three ledger columns hold, step for step, values the run has
        # already produced under another name. Pointing them at that array
        # keeps every reported sum identical and drops three more
        # full-length allocations per simulated year.
        ledger["PV_DC_Curtailed"] = self.pv_curtailment
        ledger["PV_AC_Export"] = self.grid_export
        ledger["Inverter_Loss"] = ledger["PV_Direct_Inverter_Loss"]
        for name in _LEDGER_COLUMNS:
            ledger.setdefault(name, self.zeros)
        self.ledger = ledger

    def zero_fill(self) -> None:
        """No-op: unwritten columns are already served by the zero array."""

    def column_arrays(self) -> Dict[str, np.ndarray]:
        """Return every per-timestep output column, keyed by its frame name."""
        return _column_arrays(self)


@dataclass(slots=True)
class _AgingState:
    """Battery health state that only changes at a daily boundary.

    Everything here is read by the per-step loop but written once per day by
    :func:`_apply_daily_degradation`, so the loop can keep hot copies in
    locals and refresh them when a day closes.

    ``cumulative_resistance_cycle`` and ``cumulative_resistance_calendar`` are
    accumulated and reset with the rest of the fade state but are not reported
    anywhere; they are kept because they mirror the SOH accumulators, and
    dropping them would silently remove the only running total of where
    resistance growth came from.
    """

    soh_fraction: float
    soh_percent: float
    fec_cum: float
    cumulative_cal_seconds: float
    cumulative_cycle_deg: float
    cumulative_cal_deg: float
    resistance_growth: float
    cumulative_resistance_cycle: float
    cumulative_resistance_calendar: float
    eff_charge: float
    eff_discharge: float
    n_replacements: int
    total_replacement_cost: float
    day_start_soc: float
    day_start_t_cell: float


def _apply_resistance_fade(
    aging: _AgingState,
    battery_config: BatteryConfig,
    soc_values: np.ndarray,
    time_ticks: np.ndarray,
    ticks_per_second: float,
    *,
    mean_t_cell: float,
    mean_soc_absolute: float,
    debug: bool,
) -> float:
    """Grow internal resistance for one day and return the effective RTE.

    The cycle term is charged against the FEC standing at the *start* of the
    day, so today's own cycles do not count towards their own aging; the
    lifecycle step has already folded them into the cumulative FEC, so they
    are subtracted back out here.
    """
    cycles = _detect_cycles_rainflow_arrays(
        soc_values,
        time_ticks,
        ticks_per_second,
        min_doc_fraction=0.01,
    )
    day_fec = sum(max(0.0, min(1.0, c["doc"])) * c.get("count", 1.0) for c in cycles)
    fec_before_day = aging.fec_cum - day_fec

    aging.resistance_growth, dR_cycle = update_battery_resistance_cyclewise(
        aging.resistance_growth, cycles, fec_before_day, debug=debug
    )
    aging.resistance_growth, dR_calendar = update_battery_resistance_calendar(
        aging.resistance_growth,
        T_cell_C=mean_t_cell,
        cumulative_cal_seconds=aging.cumulative_cal_seconds,
        dt_days=1.0,
        mean_soc_absolute=mean_soc_absolute,
        debug=debug,
    )
    aging.cumulative_resistance_cycle += dR_cycle
    aging.cumulative_resistance_calendar += dR_calendar

    # Feed the resistance penalty back into the energy loop using the same
    # mapping as the initial dispatch state.
    aging.eff_charge, aging.eff_discharge = resistance_to_efficiency(
        aging.resistance_growth,
        battery_config.charge_efficiency,
        battery_config.discharge_efficiency,
    )
    return aging.eff_charge * aging.eff_discharge


def _apply_battery_replacement(
    aging: _AgingState,
    battery_config: BatteryConfig,
    lifecycle: DegradationLifecycle,
    out: _ResultBuffers,
    *,
    step_index: int,
    hours_per_step: float,
    battery_energy_wh: float,
    battery_energy_beginning: float,
) -> Tuple[float, float, float]:
    """Swap in a new pack, returning ``(energy, pv_origin, day_end_soc)``.

    Replacement happens *inside* the closing timestep, after that step's
    results were already recorded. The recorded end-of-step state is
    therefore rewritten so it matches the next step's beginning, and both
    external energy transfers are exposed for whole-system reconciliation.
    """
    replacement_energy_removed = battery_energy_wh
    aging.soh_fraction = 1.0
    aging.soh_percent = 100.0
    aging.fec_cum = 0.0
    aging.cumulative_cal_seconds = 0.0
    aging.resistance_growth = 0.0
    aging.eff_charge = battery_config.charge_efficiency
    aging.eff_discharge = battery_config.discharge_efficiency
    aging.cumulative_cycle_deg = 0.0
    aging.cumulative_cal_deg = 0.0
    aging.cumulative_resistance_cycle = 0.0
    aging.cumulative_resistance_calendar = 0.0
    aging.n_replacements += 1
    aging.total_replacement_cost += battery_config.replacement_cost

    battery_energy_wh = battery_config.nominal_energy_wh * battery_config.max_soc
    replacement_energy_added = battery_energy_wh
    lifecycle.reset()

    out.replaced[step_index] = True
    out.replacement_cost[step_index] = battery_config.replacement_cost
    out.battery_energy[step_index] = battery_energy_wh
    out.soc_normalized[step_index] = 1.0
    out.soc_absolute[step_index] = battery_config.max_soc
    out.soh[step_index] = 100.0
    out.pv_origin_end[step_index] = 0.0
    out.ledger["Battery_Replacement_Energy_Removed"][step_index] = replacement_energy_removed / hours_per_step
    out.ledger["Battery_Replacement_Energy_Added"][step_index] = replacement_energy_added / hours_per_step
    out.ledger["Battery_Energy_Delta"][step_index] = (battery_energy_wh - battery_energy_beginning) / hours_per_step

    return battery_energy_wh, 0.0, battery_config.max_soc


def _dispatch_day_python(
    out: "_ResultBuffers",
    _pv_dc_vals: np.ndarray,
    _load_vals: np.ndarray,
    _temp_vals: np.ndarray,
    lo: int,
    hi: int,
    *,
    battery_config: BatteryConfig,
    has_battery: bool,
    battery_soh_decimal: float,
    Battery_SOH: float,
    Battery_Energy_Wh: float,
    Battery_PV_Origin_Energy_Wh: float,
    eff_charge: float,
    eff_discharge: float,
    hours_per_step: float,
    standby_loss_per_step_wh: float,
    cap_wh: float,
    cap_charge_wh: float,
    cap_discharge_wh: float,
) -> Tuple[float, float, float, float]:
    """Dispatch timesteps ``[lo, hi)`` at fixed health, filling *out* in place.

    This is the reference per-step loop, lifted out of the year loop so that a
    whole day can be handed to one call. State of health, resistance-derived
    efficiencies and the replacement decision are unchanged for the duration
    of the call; the caller advances them at the day boundary.

    Returns ``(battery_energy, pv_origin, t_cell_day_sum, battery_energy_beginning)``,
    where the last value is the beginning-of-step stored energy of the final
    step in the window, which the day-close replacement path needs.
    """
    _ledger_arrays = out.ledger
    T_cell_day_sum = 0.0
    battery_energy_beginning = 0.0
    for i in range(lo, hi):
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
            (
                Battery_Energy_Wh,
                Battery_PV_Origin_Energy_Wh,
                Emin,
                Emax,
                capacity_window_loss,
                battery_standby_loss,
            ) = _apply_capacity_window(
                battery_config.nominal_energy_wh,
                battery_soh_decimal,
                battery_config.max_soc,
                battery_config.min_soc,
                standby_loss_per_step_wh,
                Battery_Energy_Wh,
                Battery_PV_Origin_Energy_Wh,
                T_cell,
            )
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
        # Store results via array indexing (avoids per-timestep dict overhead)
        out.pv_dc[i] = pv_dc_power / hours_per_step
        out.pv_production[i] = pv_production / hours_per_step
        out.load[i] = load / hours_per_step
        out.pv_delta[i] = (pv_production - load) / hours_per_step
        out.grid_import[i] = Import / hours_per_step
        out.grid_export[i] = Sell / hours_per_step
        out.battery_energy[i] = Battery_Energy_Wh if has_battery else 0.0
        out.soc_normalized[i] = soc_normalized
        out.soc_absolute[i] = soc_absolute
        out.soh[i] = Battery_SOH if has_battery else 100.0
        out.t_cell[i] = T_cell
        out.pv_curtailment[i] = pv_curtailment / hours_per_step
        out.charge_loss[i] = battery_charge_loss / hours_per_step
        out.discharge_loss[i] = battery_discharge_loss / hours_per_step
        out.standby_loss[i] = battery_standby_loss / hours_per_step
        out.battery_energy_begin[i] = battery_energy_beginning
        out.pv_origin_begin[i] = pv_origin_beginning
        out.pv_origin_end[i] = Battery_PV_Origin_Energy_Wh
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
    return Battery_Energy_Wh, Battery_PV_Origin_Energy_Wh, T_cell_day_sum, battery_energy_beginning


def _dispatch_no_battery_vectorized(
    out: Union["_ResultBuffers", "_PvOnlySummaryBuffers"],
    pv_dc_values: np.ndarray,
    load_values: np.ndarray,
    temperature_values: np.ndarray,
    *,
    battery_config: BatteryConfig,
    hours_per_step: float,
    cap_wh: float,
) -> None:
    """Fill a PV-only result buffer without entering the timestep loop."""
    out.zero_fill()

    pv_dc_wh = np.maximum(0.0, pv_dc_values * hours_per_step)
    load_wh = load_values * hours_per_step
    ac_wh, conversion_loss_wh, clipping_loss_dc_wh = _calculate_dc_ac_power_arrays(
        pv_dc_wh,
        cap_wh,
        battery_config.inverter_efficiency,
    )
    pv_ac_to_load_wh = np.minimum(ac_wh, load_wh)
    grid_export_wh = ac_wh - pv_ac_to_load_wh
    grid_import_wh = np.maximum(0.0, load_wh - pv_ac_to_load_wh)
    if math.isinf(cap_wh):
        pv_production_wh = (pv_dc_wh - clipping_loss_dc_wh) * battery_config.inverter_efficiency
    else:
        # Keep the scalar reference's subtraction order. Returning ``ac_wh``
        # here is algebraically equivalent but differs by one ULP at low load.
        pv_production_wh = pv_dc_wh - clipping_loss_dc_wh - conversion_loss_wh

    # The three columns reported twice under different names are divided
    # once and assigned twice. On the reduced summary buffer the second
    # assignment writes the array the first one already filled, which is
    # what makes sharing it safe: both names carry the same values either way.
    curtailment_w = clipping_loss_dc_wh / hours_per_step
    grid_export_w = grid_export_wh / hours_per_step
    conversion_w = conversion_loss_wh / hours_per_step

    out.pv_dc[:] = pv_dc_wh / hours_per_step
    out.pv_production[:] = pv_production_wh / hours_per_step
    out.load[:] = load_wh / hours_per_step
    out.pv_delta[:] = (pv_production_wh - load_wh) / hours_per_step
    out.grid_import[:] = grid_import_wh / hours_per_step
    out.grid_export[:] = grid_export_w
    out.soh.fill(100.0)
    out.t_cell[:] = temperature_values
    out.pv_curtailment[:] = curtailment_w

    out.ledger["PV_DC_To_Inverter"][:] = (pv_dc_wh - clipping_loss_dc_wh) / hours_per_step
    out.ledger["PV_DC_Curtailed"][:] = curtailment_w
    out.ledger["PV_AC_To_Load"][:] = pv_ac_to_load_wh / hours_per_step
    out.ledger["PV_AC_Export"][:] = grid_export_w
    out.ledger["PV_Direct_Inverter_Loss"][:] = conversion_w
    out.ledger["Inverter_Loss"][:] = conversion_w


def _apply_daily_degradation(
    aging: _AgingState,
    lifecycle: DegradationLifecycle,
    battery_config: BatteryConfig,
    out: _ResultBuffers,
    degradation_tracking: List[Dict[str, Any]],
    *,
    step_index: int,
    step_time: pd.Timestamp,
    day_index: pd.DatetimeIndex,
    soc_absolute_day: np.ndarray,
    t_cell_day: np.ndarray,
    t_cell_day_sum: float,
    steps_per_day: int,
    hours_per_step: float,
    battery_energy_wh: float,
    pv_origin_energy_wh: float,
    battery_energy_beginning: float,
    debug: bool,
) -> Tuple[float, float]:
    """Close out one degradation day, returning ``(energy, pv_origin)``.

    Runs the lifecycle step, optional resistance fade and the end-of-life
    replacement check in that order, mutating *aging* in place and appending
    one row to *degradation_tracking*. The two stored-energy values are
    returned rather than carried on *aging* because the per-step loop owns
    them and only a replacement changes them here.

    The day's SOC and cell-temperature endpoints become the next day's
    starting boundary; a replacement moves that endpoint to the fresh pack's
    max SOC, since the recorded state was rewritten to match.
    """
    time_ticks, ticks_per_second = _datetime_index_ticks(day_index)
    day_end_soc_absolute = float(soc_absolute_day[steps_per_day - 1])
    day_end_t_cell = float(t_cell_day[steps_per_day - 1])

    mean_soc_abs = float(np.mean(soc_absolute_day))
    mean_t_cell = t_cell_day_sum / steps_per_day
    effective_rte = battery_config.charge_efficiency * battery_config.discharge_efficiency

    degradation_step = lifecycle.step(
        DegradationDay(
            soc=soc_absolute_day,
            time_ticks=time_ticks,
            ticks_per_second=ticks_per_second,
            temperature_c=t_cell_day,
            step_seconds=hours_per_step * 3600.0,
            start_soc=aging.day_start_soc,
            start_temperature_c=aging.day_start_t_cell,
        )
    )
    aging.soh_fraction = degradation_step.soh_fraction
    aging.soh_percent = aging.soh_fraction * 100.0
    aging.fec_cum = degradation_step.fec
    aging.cumulative_cal_seconds = degradation_step.calendar_seconds
    aging.cumulative_cycle_deg += degradation_step.cycle_degradation
    aging.cumulative_cal_deg += degradation_step.calendar_degradation

    if battery_config.enable_resistance_fade:
        effective_rte = _apply_resistance_fade(
            aging,
            battery_config,
            soc_absolute_day,
            time_ticks,
            ticks_per_second,
            mean_t_cell=mean_t_cell,
            mean_soc_absolute=mean_soc_abs,
            debug=debug,
        )

    if battery_config.enable_replacement and aging.soh_fraction <= battery_config.eol_percentage:
        battery_energy_wh, pv_origin_energy_wh, day_end_soc_absolute = _apply_battery_replacement(
            aging,
            battery_config,
            lifecycle,
            out,
            step_index=step_index,
            hours_per_step=hours_per_step,
            battery_energy_wh=battery_energy_wh,
            battery_energy_beginning=battery_energy_beginning,
        )
        if debug:
            print(f"\n*** BATTERY REPLACED at {step_time} ***")

    degradation_record = {
        "Datetime": step_time,
        "SOH": aging.soh_percent,
        "Cycle_Degradation": degradation_step.cycle_degradation,
        "Calendar_Degradation": degradation_step.calendar_degradation,
        "Cumulative_Cycle_Degradation": aging.cumulative_cycle_deg,
        "Cumulative_Calendar_Degradation": aging.cumulative_cal_deg,
        "Cumulative_FEC": aging.fec_cum,
        "Cumulative_Calendar_Seconds": aging.cumulative_cal_seconds,
        "Total_Degradation": 1.0 - aging.soh_fraction,
        "Mean_SOC_Absolute": mean_soc_abs,
    }
    degradation_record.update(lifecycle.tracking_fields(degradation_step))
    if battery_config.enable_resistance_fade:
        degradation_record["Resistance_Growth"] = aging.resistance_growth
        degradation_record["Effective_RTE"] = effective_rte
    degradation_tracking.append(degradation_record)

    aging.day_start_soc = day_end_soc_absolute
    aging.day_start_t_cell = day_end_t_cell
    return battery_energy_wh, pv_origin_energy_wh


def _build_summary_row(
    buffers: _ResultBuffers,
    hours_per_step: float,
    *,
    final_soh_percent: float,
    n_replacements: int,
    total_replacement_cost: float,
) -> Tuple[Dict[str, float], float]:
    """Summarise a completed run, returning ``(summary_row, total_pv_wh)``.

    ``total_pv`` is both a summary row and a separate public return value, so
    it is computed once here and handed back rather than recomputed.
    """
    total_pv = np.sum(buffers.pv_production) * hours_per_step
    total_load = np.sum(buffers.load) * hours_per_step
    total_sell = np.sum(buffers.grid_export) * hours_per_step
    total_import = np.sum(buffers.grid_import) * hours_per_step

    percentage_imported = (total_import / total_load * 100) if total_load > 0 else 0

    summary = {
        "Total PV [kWh]": total_pv / 1000.0,
        "Total Load [kWh]": total_load / 1000.0,
        "Sell [kWh]": total_sell / 1000.0,
        "Import [kWh]": total_import / 1000.0,
        "Import [%]": percentage_imported,
        "Grid Independence [%]": 100 - percentage_imported,
        "Final SOH [%]": final_soh_percent,
        "N_Replacements": n_replacements,
        "Replacement_Cost": total_replacement_cost,
    }
    return summary, total_pv


def _build_final_degradation_state(
    degradation_lifecycle: DegradationLifecycle,
    *,
    day_start_soc: float,
    day_start_temperature_c: float,
    resistance_growth: float,
) -> Dict[str, Any]:
    """Assemble the carry state a follow-on run can be resumed from.

    The engine-independent keys are listed first and explicitly, so the
    schema a caller round-trips does not depend on adapter dict ordering;
    whatever else the adapter reports (BLAST engine internals) follows.
    Resistance growth is owned by the energy loop, not by either adapter.
    """
    adapter_snapshot = degradation_lifecycle.snapshot(
        day_start_soc=day_start_soc,
        day_start_temperature_c=day_start_temperature_c,
    )
    return {
        "degradation_engine": adapter_snapshot.pop("degradation_engine"),
        "fec_cum": float(adapter_snapshot.pop("fec_cum")),
        "cumulative_calendar_seconds": float(adapter_snapshot.pop("cumulative_calendar_seconds")),
        "resistance_growth": float(resistance_growth),
        "cumulative_cycle_degradation": float(adapter_snapshot.pop("cumulative_cycle_degradation")),
        "cumulative_calendar_degradation": float(adapter_snapshot.pop("cumulative_calendar_degradation")),
        **adapter_snapshot,
    }


def _resolve_dispatch_day(execution_backend: str) -> Any:
    """Return the within-day dispatch implementation for a backend name.

    Resolution happens once per simulated year, before any timestep runs, so
    a missing optional dependency is reported at the start of a study rather
    than part-way through one. The name check lives in :mod:`breos.execution`
    so App and Monte Carlo cannot disagree about what is valid.
    """
    validate_execution_backend(execution_backend)
    if execution_backend == "python":
        return _dispatch_day_python
    from breos._numba_dispatch import require_numba_dispatch_day

    return require_numba_dispatch_day()


@dataclass(slots=True)
class _CoreRun:
    """Everything one simulated span produced, before any result is shaped.

    Both public entry points run the same core and then differ only in what
    they build from this: the detailed path materialises frames, the summary
    path reduces the buffers in place. Keeping the split here is what makes
    the two paths comparable field by field.
    """

    buffers: _ResultBuffers
    rng: pd.DatetimeIndex
    aging: _AgingState
    lifecycle: DegradationLifecycle
    degradation_tracking: List[Dict[str, Any]]
    hours_per_step: float
    has_battery: bool
    final_soh_percent: float


@dataclass(frozen=True, slots=True)
class SimulationSummary:
    """Annual totals and carry state, without a per-timestep frame.

    ``column_sums`` holds the plain sum of every column the detailed results
    frame exposes, under the same names and in the same units, so a caller
    that used to write ``results_df[col].sum()`` reads ``column_sums[col]``
    and gets the identical value. Unit scaling stays with the caller, because
    the order of a scaling expression is itself observable in floating point.

    The remaining fields are the state a multi-year caller has to carry, plus
    the diagnostics needed to establish parity between execution paths:
    stored and PV-origin energy at the seam, the four cumulative degradation
    accumulators, resistance growth, and the exact steps at which the pack
    was replaced.
    """

    n_steps: int
    hours_per_step: float
    has_battery: bool
    column_sums: Dict[str, float]
    total_pv_wh: float
    summary_row: Dict[str, float]
    final_soh_percent: float
    n_replacements: int
    total_replacement_cost: float
    opening_energy_wh: float
    opening_pv_origin_energy_wh: float
    carried_energy_wh: float
    carried_pv_origin_energy_wh: float
    has_degradation_rows: bool
    fec_cum: float
    cumulative_calendar_seconds: float
    cumulative_cycle_degradation: float
    cumulative_calendar_degradation: float
    resistance_growth: float
    replacement_steps: Tuple[int, ...]
    final_degradation_state: Optional[Dict[str, Any]] = None


def _build_simulation_summary(core: _CoreRun, *, return_degradation_state: bool) -> SimulationSummary:
    """Reduce a completed core run to its annual totals and carry state."""
    buffers = core.buffers
    aging = core.aging
    column_sums = _column_sums(buffers.column_arrays())

    summary_row, total_pv = _build_summary_row(
        buffers,
        core.hours_per_step,
        final_soh_percent=core.final_soh_percent,
        n_replacements=aging.n_replacements,
        total_replacement_cost=aging.total_replacement_cost,
    )

    final_state = None
    if return_degradation_state:
        final_state = _build_final_degradation_state(
            core.lifecycle,
            day_start_soc=aging.day_start_soc,
            day_start_temperature_c=aging.day_start_t_cell,
            resistance_growth=aging.resistance_growth,
        )

    return SimulationSummary(
        n_steps=len(buffers.battery_energy),
        hours_per_step=core.hours_per_step,
        has_battery=core.has_battery,
        column_sums=column_sums,
        total_pv_wh=total_pv,
        summary_row=summary_row,
        final_soh_percent=core.final_soh_percent,
        n_replacements=aging.n_replacements,
        total_replacement_cost=aging.total_replacement_cost,
        # Read from the recorded end-of-step state rather than from the loop
        # locals: a replacement inside the closing step rewrites what the next
        # year must resume from.
        # The opening state makes the year-to-year seam checkable from a
        # summary alone: the first step's beginning energy must equal what the
        # previous year carried out.
        opening_energy_wh=float(buffers.battery_energy_begin[0]),
        opening_pv_origin_energy_wh=float(buffers.pv_origin_begin[0]),
        carried_energy_wh=float(buffers.battery_energy[-1]),
        carried_pv_origin_energy_wh=float(buffers.pv_origin_end[-1]),
        has_degradation_rows=bool(core.degradation_tracking),
        fec_cum=aging.fec_cum,
        cumulative_calendar_seconds=aging.cumulative_cal_seconds,
        cumulative_cycle_degradation=aging.cumulative_cycle_deg,
        cumulative_calendar_degradation=aging.cumulative_cal_deg,
        resistance_growth=aging.resistance_growth,
        replacement_steps=tuple(int(i) for i in np.flatnonzero(buffers.replaced)),
        final_degradation_state=final_state,
    )


def _simulate_core(
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
    debug: bool = False,
    initial_energy_wh: Optional[float] = None,
    initial_pv_origin_energy_wh: Optional[float] = None,
    execution_backend: str = "python",
    summary_only: bool = False,
) -> "_CoreRun":
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
        initial_degradation_state: Optional BLAST state returned by a previous
            call with ``return_degradation_state=True``.
        return_degradation_state: Append final degradation carry state to the
            return tuple when True.
        debug: Enable debug output
        initial_energy_wh: Optional carried stored-energy state (Wh). Defaults
            to the configured max-SOC state for first-run compatibility.
        initial_pv_origin_energy_wh: Optional PV-origin share of the carried
            stored energy (Wh). Defaults to zero.
        execution_backend: Which implementation runs the within-day dispatch
            arithmetic. ``"python"`` is the default and the numerical
            reference; ``"numba"`` selects the optional compiled kernel and
            requires ``breos[fast]``. Everything outside the day window runs
            in Python either way.

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

    _pv_dc_vals, _load_vals, _temp_vals = _align_simulation_inputs(pv_dc, houseload, temperature_series, rng)

    degradation_engine_key = _resolve_degradation_engine(
        degradation_engine,
        blast_model,
        initial_degradation_state,
        battery_config,
    )

    # Initialize state
    battery_soh_decimal = battery_config.initial_soh / 100.0
    Battery_SOH = battery_config.initial_soh
    Battery_Energy_Wh, Battery_PV_Origin_Energy_Wh = _resolve_carried_energy(
        initial_energy_wh,
        initial_pv_origin_energy_wh,
        battery_config,
        battery_soh_decimal,
    )

    # Degradation day-windows are positional (fixed steps_per_day), not
    # calendar-based: DST days and trailing partial days shift/skip windows
    # by design; the compiled dispatch backend shares the convention.
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
    if battery_config.enable_resistance_fade:
        eff_charge, eff_discharge = resistance_to_efficiency(
            resistance_growth,
            battery_config.charge_efficiency,
            battery_config.discharge_efficiency,
        )

    degradation_tracking: List[Dict[str, Any]] = []

    n_steps = len(rng)

    # Hoist invariant check out of the loop
    has_battery = battery_config.nominal_energy_wh > 1 and (battery_config.max_soc - battery_config.min_soc) > 0
    # The vectorized PV-only dispatch below is the only producer that leaves
    # most columns at zero, so it is the only one whose output can be served
    # from the reduced buffer -- and only when the caller wants a summary,
    # since the detailed frame must own a writable array per column.
    pv_only_summary = summary_only and not has_battery and execution_backend == "python"

    # Pre-allocate result arrays (avoids per-timestep dict creation)
    out = _PvOnlySummaryBuffers(n_steps) if pv_only_summary else _ResultBuffers(n_steps)
    degradation_day_start_soc = battery_config.max_soc if has_battery else 0.0
    degradation_day_start_t_cell = float(_temp_vals[0]) if n_steps else 25.0

    degradation_lifecycle, degradation_day_start_soc, degradation_day_start_t_cell = _build_degradation_lifecycle(
        degradation_engine_key,
        battery_config,
        battery_soh_decimal=battery_soh_decimal,
        has_battery=has_battery,
        blast_model=blast_model,
        initial_degradation_state=initial_degradation_state,
        initial_fec=initial_fec,
        initial_calendar_seconds=initial_calendar_seconds,
        initial_cumulative_cycle_deg=initial_cumulative_cycle_deg,
        initial_cumulative_cal_deg=initial_cumulative_cal_deg,
        default_day_start_soc=degradation_day_start_soc,
        default_day_start_t_cell=degradation_day_start_t_cell,
        debug=debug,
    )

    battery_soh_decimal = degradation_lifecycle.soh()
    Battery_SOH = battery_soh_decimal * 100.0
    if degradation_engine_key == "blast" and initial_energy_wh is None:
        Battery_Energy_Wh = battery_config.nominal_energy_wh * battery_soh_decimal * battery_config.max_soc

    # Health state advanced only at daily boundaries. The loop keeps hot
    # copies of the four fields it reads every step (SOH and the two
    # efficiencies) and refreshes them whenever a day closes.
    aging = _AgingState(
        soh_fraction=battery_soh_decimal,
        soh_percent=Battery_SOH,
        fec_cum=initial_fec,
        cumulative_cal_seconds=initial_calendar_seconds,
        cumulative_cycle_deg=initial_cumulative_cycle_deg,
        cumulative_cal_deg=initial_cumulative_cal_deg,
        resistance_growth=resistance_growth,
        cumulative_resistance_cycle=0.0,
        cumulative_resistance_calendar=0.0,
        eff_charge=eff_charge,
        eff_discharge=eff_discharge,
        n_replacements=0,
        total_replacement_cost=0.0,
        day_start_soc=degradation_day_start_soc,
        day_start_t_cell=degradation_day_start_t_cell,
    )

    # Per-step energy caps (Wh) for the shared inverter AC nameplate and the
    # battery's own charge/discharge power limits.
    standby_loss_per_step_wh = battery_config.standby_loss_wh * hours_per_step
    cap_wh = _step_energy_cap(battery_config.inverter_ac_capacity_w, hours_per_step)
    cap_charge_wh = _step_energy_cap(battery_config.max_charge_power_w, hours_per_step)
    cap_discharge_wh = _step_energy_cap(battery_config.max_discharge_power_w, hours_per_step)

    dispatch_day = _resolve_dispatch_day(execution_backend)

    if not has_battery and execution_backend == "python":
        _dispatch_no_battery_vectorized(
            out,
            _pv_dc_vals,
            _load_vals,
            _temp_vals,
            battery_config=battery_config,
            hours_per_step=hours_per_step,
            cap_wh=cap_wh,
        )
        return _CoreRun(
            buffers=out,
            rng=rng,
            aging=aging,
            lifecycle=degradation_lifecycle,
            degradation_tracking=degradation_tracking,
            hours_per_step=hours_per_step,
            has_battery=False,
            final_soh_percent=Battery_SOH,
        )

    # Dispatch advances one degradation day at a time. Health state is fixed
    # for the whole window and advanced here, between windows, so every
    # scientifically sensitive transition stays on this path regardless of
    # which backend ran the arithmetic inside the window.
    window_start = 0
    while window_start < n_steps:
        window_end = min(window_start + steps_per_day, n_steps)
        (
            Battery_Energy_Wh,
            Battery_PV_Origin_Energy_Wh,
            T_cell_day_sum,
            battery_energy_beginning,
        ) = dispatch_day(
            out,
            _pv_dc_vals,
            _load_vals,
            _temp_vals,
            window_start,
            window_end,
            battery_config=battery_config,
            has_battery=has_battery,
            battery_soh_decimal=battery_soh_decimal,
            Battery_SOH=Battery_SOH,
            Battery_Energy_Wh=Battery_Energy_Wh,
            Battery_PV_Origin_Energy_Wh=Battery_PV_Origin_Energy_Wh,
            eff_charge=eff_charge,
            eff_discharge=eff_discharge,
            hours_per_step=hours_per_step,
            standby_loss_per_step_wh=standby_loss_per_step_wh,
            cap_wh=cap_wh,
            cap_charge_wh=cap_charge_wh,
            cap_discharge_wh=cap_discharge_wh,
        )
        if window_end - window_start < steps_per_day:
            # Degradation windows are positional and whole-day. A trailing
            # partial day is simulated and reported but never closes a window,
            # which is what the per-step loop did before this was hoisted.
            break

        if has_battery:
            last_step = window_end - 1
            Battery_Energy_Wh, Battery_PV_Origin_Energy_Wh = _apply_daily_degradation(
                aging,
                degradation_lifecycle,
                battery_config,
                out,
                degradation_tracking,
                step_index=last_step,
                # Timestamps are read once per closed day rather than once per
                # step; nothing inside the window depends on the calendar.
                step_time=rng[last_step],
                day_index=rng[window_start:window_end],
                # Copied before the call so a replacement rewriting the closing
                # step's recorded state cannot reach the day the aging model saw.
                soc_absolute_day=out.soc_absolute[window_start:window_end].copy(),
                t_cell_day=out.t_cell[window_start:window_end].copy(),
                t_cell_day_sum=T_cell_day_sum,
                steps_per_day=steps_per_day,
                hours_per_step=hours_per_step,
                battery_energy_wh=Battery_Energy_Wh,
                pv_origin_energy_wh=Battery_PV_Origin_Energy_Wh,
                battery_energy_beginning=battery_energy_beginning,
                debug=debug,
            )
            # Refresh the loop's hot copies of the daily-boundary state.
            battery_soh_decimal = aging.soh_fraction
            Battery_SOH = aging.soh_percent
            eff_charge = aging.eff_charge
            eff_discharge = aging.eff_discharge
        window_start = window_end

    return _CoreRun(
        buffers=out,
        rng=rng,
        aging=aging,
        lifecycle=degradation_lifecycle,
        degradation_tracking=degradation_tracking,
        hours_per_step=hours_per_step,
        has_battery=has_battery,
        final_soh_percent=Battery_SOH,
    )


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
    execution_backend: str = "python",
) -> (
    Tuple[pd.DataFrame, float, pd.DataFrame, float, int, pd.DataFrame]
    | Tuple[pd.DataFrame, float, pd.DataFrame, float, int, pd.DataFrame, Dict[str, Any]]
):
    """Simulate an energy balance and return the detailed per-timestep results.

    This is the reference path and the full public contract; see
    :func:`_simulate_core` for the argument semantics. Callers that only need
    annual totals and carry state should use
    :func:`simulate_energy_balance_summary`, which runs the same physics
    without materialising the results frame.

    Returns:
        Tuple of:
        - results_df: Detailed timestep results
        - total_pv: Total PV AC production after inverter efficiency (Wh)
        - summary_df: Summary statistics
        - replacement_cost: Total battery replacement cost
        - n_replacements: Number of battery replacements
        - degradation_df: Daily degradation tracking
    """
    core = _simulate_core(
        pv_dc=pv_dc,
        houseload=houseload,
        battery_config=battery_config,
        start_time=start_time,
        end_time=end_time,
        freq=freq,
        temperature_series=temperature_series,
        results_directory=results_directory,
        initial_fec=initial_fec,
        initial_calendar_seconds=initial_calendar_seconds,
        initial_resistance_growth=initial_resistance_growth,
        initial_cumulative_cycle_deg=initial_cumulative_cycle_deg,
        initial_cumulative_cal_deg=initial_cumulative_cal_deg,
        degradation_engine=degradation_engine,
        blast_model=blast_model,
        initial_degradation_state=initial_degradation_state,
        debug=debug,
        initial_energy_wh=initial_energy_wh,
        initial_pv_origin_energy_wh=initial_pv_origin_energy_wh,
        execution_backend=execution_backend,
    )
    df = core.buffers.to_frame(core.rng)
    deg_df = pd.DataFrame(core.degradation_tracking) if core.degradation_tracking else pd.DataFrame()
    summary_row, total_pv = _build_summary_row(
        core.buffers,
        core.hours_per_step,
        final_soh_percent=core.final_soh_percent,
        n_replacements=core.aging.n_replacements,
        total_replacement_cost=core.aging.total_replacement_cost,
    )
    summary_df = pd.DataFrame([summary_row])

    result = (df, total_pv, summary_df, core.aging.total_replacement_cost, core.aging.n_replacements, deg_df)
    if not return_degradation_state:
        return result

    final_degradation_state = _build_final_degradation_state(
        core.lifecycle,
        day_start_soc=core.aging.day_start_soc,
        day_start_temperature_c=core.aging.day_start_t_cell,
        resistance_growth=core.aging.resistance_growth,
    )
    return (*result, final_degradation_state)


def simulate_energy_balance_summary(
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
    execution_backend: str = "python",
) -> SimulationSummary:
    """Simulate an energy balance and return annual totals and carry state.

    Runs exactly the physics :func:`simulate_energy_balance` runs, with the
    same arguments, and skips only the construction of the per-timestep
    results frame and the daily degradation frame. Multi-year callers such as
    Monte Carlo need the aggregates and the year-to-year seam, not the
    35,040 rows they were being reduced from.
    """
    core = _simulate_core(
        pv_dc=pv_dc,
        houseload=houseload,
        battery_config=battery_config,
        start_time=start_time,
        end_time=end_time,
        freq=freq,
        temperature_series=temperature_series,
        results_directory=results_directory,
        initial_fec=initial_fec,
        initial_calendar_seconds=initial_calendar_seconds,
        initial_resistance_growth=initial_resistance_growth,
        initial_cumulative_cycle_deg=initial_cumulative_cycle_deg,
        initial_cumulative_cal_deg=initial_cumulative_cal_deg,
        degradation_engine=degradation_engine,
        blast_model=blast_model,
        initial_degradation_state=initial_degradation_state,
        debug=debug,
        initial_energy_wh=initial_energy_wh,
        initial_pv_origin_energy_wh=initial_pv_origin_energy_wh,
        execution_backend=execution_backend,
        # Reduced per-step buffers are safe here: nothing downstream of this
        # call materialises a per-timestep frame.
        summary_only=True,
    )
    return _build_simulation_summary(core, return_degradation_state=return_degradation_state)


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


_TICKS_PER_SECOND = {
    "s": 1.0,
    "ms": 1_000.0,
    "us": 1_000_000.0,
    "ns": 1_000_000_000.0,
}


def _datetime_index_ticks(time_index: pd.DatetimeIndex) -> Tuple[np.ndarray, float]:
    """Return integer timestamps and their scale without changing resolution."""
    try:
        ticks_per_second = _TICKS_PER_SECOND[time_index.unit]
    except KeyError as exc:
        raise ValueError(f"Unsupported DatetimeIndex resolution: {time_index.unit}") from exc
    return time_index.asi8, ticks_per_second


def _detect_cycles_rainflow_arrays(
    soc_values: np.ndarray,
    time_ticks: np.ndarray,
    ticks_per_second: float,
    min_doc_fraction: float = 0.01,
) -> List[Dict]:
    """Detect rainflow cycles from arrays used by the simulation hot path."""
    if len(soc_values) < 2:
        return []

    # rainflow.extract_cycles expects a sequence; multiply by 100 for percent
    soc_pct = soc_values * 100.0

    cycles = []
    for rng, mean, count, i_start, i_end in rainflow.extract_cycles(soc_pct):
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
    time_ticks, ticks_per_second = _datetime_index_ticks(time_index)
    return _detect_cycles_rainflow_arrays(
        soc_abs_series.to_numpy(),
        time_ticks,
        ticks_per_second,
        min_doc_fraction=min_doc_fraction,
    )


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


def _update_battery_soh_from_cycles(
    soh_start_fraction: float,
    cycles: List[Dict],
    nominal_energy_Wh: float,
    fec_cum: float = 0.0,
    min_DoD_fraction: float = 0.01,
    battery_type: str = "lfp",
    debug: bool = False,
) -> Tuple[float, float, float]:
    # Get technology-specific cycle parameters
    del nominal_energy_Wh
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


def _update_battery_soh_cyclewise_arrays(
    soh_start_fraction: float,
    soc_values: np.ndarray,
    time_ticks: np.ndarray,
    ticks_per_second: float,
    nominal_energy_Wh: float,
    fec_cum: float = 0.0,
    min_DoD_fraction: float = 0.01,
    battery_type: str = "lfp",
    debug: bool = False,
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
        nominal_energy_Wh,
        fec_cum=fec_cum,
        min_DoD_fraction=min_DoD_fraction,
        battery_type=battery_type,
        debug=debug,
    )


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
        time_ticks, ticks_per_second = _datetime_index_ticks(time_index)
        return _update_battery_soh_cyclewise_arrays(
            soh_start_fraction,
            soc_series_absolute.to_numpy(),
            time_ticks,
            ticks_per_second,
            nominal_energy_Wh,
            fec_cum=fec_cum,
            min_DoD_fraction=min_DoD_fraction,
            battery_type=battery_type,
            debug=debug,
        )

    cycles, _ = detect_half_cycles_from_soc_series(soc_series_absolute, time_index)
    return _update_battery_soh_from_cycles(
        soh_start_fraction,
        cycles,
        nominal_energy_Wh,
        fec_cum=fec_cum,
        min_DoD_fraction=min_DoD_fraction,
        battery_type=battery_type,
        debug=debug,
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
