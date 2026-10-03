"""
Battery simulation module.

This module handles battery energy storage simulation including:
- Energy balance calculations
- State of Charge (SOC) tracking
- State of Health (SOH) degradation models (Naumann + Lam)
- Cycle and calendar aging
"""

import dataclasses
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
import rainflow

from breos._controller import ControllerBatteryState, ControllerCarry, DailyDispatchController, _ControllerSession
from breos._dispatch import (  # noqa: F401  -- the dispatch step moved; its names stay importable here
    _LEDGER_COLUMNS,
    _N_ROWS,
    _ROW,
    _ROW_COLUMNS,
    LEDGER_SCHEMA_VERSION,
    _dispatch_day_python,
    compute_cell_temperature,
    lfp_capacity_factor,
)
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
    DEFAULT_EOL_PERCENTAGE,
    DEFAULT_INDOOR_CEILING_C,
    DEFAULT_INDOOR_COUPLING_ALPHA,
    DEFAULT_INDOOR_FLOOR_C,
    DEFAULT_INDOOR_SETPOINT_C,
    DEFAULT_MAX_SOC,
    DEFAULT_MIN_SOC,
    DEFAULT_STANDBY_LOSS_WH,
    DEFAULT_THERMAL_RESISTANCE_K_PER_W,
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
from breos.degradation.protocol import (
    BlastDegradationAdapter,
    DegradationDay,
    DegradationLifecycle,
    NativeDegradationAdapter,
)
from breos.dispatch_instructions import DispatchInstructions
from breos.execution import is_pv_only_dispatch, single_thread_blas, validate_execution_backend
from breos.inverter import _calculate_dc_ac_power_arrays
from breos.utils import _datetime_index_ticks, get_hours_per_step, remap_datetime_index_years

if TYPE_CHECKING:
    from breos.tariffs import ResolvedTariff


@dataclass
class BatteryConfig:
    """
    Configuration parameters for battery simulation.

    Only DC-coupled systems (hybrid inverters) are modelled:
    - PV → Battery: No inverter loss (stays in DC)
    - Battery → Load: Inverter loss applies (DC to AC)

    AC-coupled dispatch is not implemented.

    Power limits are nameplate powers and therefore scale with the timestep:
    ``max_charge_power_w`` limits DC input to the battery path, while
    ``max_discharge_power_w`` limits battery AC delivered to the load. Both are
    absolute wattages that do not track ``nominal_energy_wh``. Set
    ``power_limit_c_rate`` instead to derive a symmetric limit from capacity,
    which is what a capacity sweep normally wants. It limits the stored
    energy, like a cell current rating: at 1 C a 5 kWh pack stores or releases
    at most 5 kW in either direction, so its DC input while charging and its AC
    output while discharging differ from that by the conversion losses.

    ``ac_output_scale`` derates AC delivery for shortfall the chain does not
    model and is bounded to ``(0, 1]``. It applies inside dispatch, not to a
    finished result, so battery discharge decisions and the reachable AC
    ceiling respond to it. It is one constant approximating the combined
    annual effect of things like availability, curtailment and downstream
    wiring, not a model of any of them individually. While it is active,
    the reported inverter loss is the whole DC-to-AC shortfall rather than
    the converter's own loss alone.

    It is the AC-side half of a pair: the optimizer's ``dc_output_scale``
    corrects the array itself before dispatch, so clipping, charging and the
    part-load ratio all respond to it, and it is the correct knob when the
    model under-predicts measured yield.

    ``eol_percentage`` defaults to 0.70 (replace the battery when its state
    of health falls to 70% of nominal capacity), matching the App config
    default ``battery_eol_percentage``.

    ``allow_terminal_replacement`` decides whether the end-of-life check at
    the close of this span's final degradation period may replace the pack.
    That period is the one ending on the span's last step: its last whole
    day, or a trailing partial day, or the whole span when it is shorter
    than a day. The default ``True`` replaces there as anywhere else.
    ``False`` skips only that swap, so no pack is bought that would deliver
    no service inside the span. The period is still aged, finalized and
    recorded, and the old pack's state is reported. A complete day followed
    by a partial day is not the final period and keeps its replacement.
    A call treats its own span as the horizon: a caller that splits one
    horizon across several calls must leave it ``True`` on every span but
    the last, because the next span inherits the pack.

    ``replacement_min_remaining_years`` skips any end-of-life swap that
    would leave the new pack less than this many years of the horizon to
    serve. The default 0 skips none. Time is counted as a projection books a
    swap's ``Replacement_Time_Years``: this span is one project year, and
    ``replacement_years_after_span`` whole project years follow it. A swap at
    the close of a period ending on span step ``k`` of ``n`` therefore has
    ``replacement_years_after_span + (n - k) / n`` years left, and it is
    skipped when that is below the minimum. The test is made in steps,
    ``replacement_years_after_span * n + n - k`` against
    ``replacement_min_remaining_years * n``, so a swap with exactly the
    minimum left still happens. A skipped pack keeps ageing below end of
    life and is reported as it is, as with the terminal guard. Any positive
    minimum also skips the final period's swap, whatever
    ``allow_terminal_replacement`` says. A call with
    ``replacement_years_after_span = 0`` treats its own span as the last
    project year; a projection sets the field for each year it runs.
    """

    nominal_energy_wh: float  # Required — nominal capacity in Wh
    initial_soh: float = 100.0  # Initial state of health (%)
    eol_percentage: float = DEFAULT_EOL_PERCENTAGE  # End of life threshold (fraction)
    max_soc: float = DEFAULT_MAX_SOC
    min_soc: float = DEFAULT_MIN_SOC
    charge_efficiency: float = DEFAULT_CHARGE_EFFICIENCY
    discharge_efficiency: float = DEFAULT_DISCHARGE_EFFICIENCY
    standby_loss_wh: float = DEFAULT_STANDBY_LOSS_WH
    enable_replacement: bool = True
    # False skips the end-of-life swap at the close of the span's final
    # degradation period only; that period still ages the pack.
    allow_terminal_replacement: bool = True
    # Skips a swap that leaves the new pack less than this many project
    # years to serve; 0 skips none. The years that follow this span count
    # toward what is left.
    replacement_min_remaining_years: float = 0.0
    replacement_years_after_span: int = 0
    calendar_model: str = "naumann_lam_field_calibrated"  # v1 field-calibrated default alias
    # Resistance fade (opt-in): grows internal resistance daily and derates
    # the charge/discharge efficiencies in the energy loop so the effective
    # round-trip efficiency declines as the battery ages.
    enable_resistance_fade: bool = False  # Enable Naumann resistance growth model
    initial_resistance_growth: float = 0.0  # Initial relative resistance growth (fraction, 0=new)
    # Thermal model
    thermal_resistance_k_per_w: float = DEFAULT_THERMAL_RESISTANCE_K_PER_W  # K/W for lumped thermal model
    inverter_efficiency: float = 0.96  # Inverter efficiency (for DC→AC conversion)
    # Inverter AC rating (W) shared by PV and battery discharge; AC output is
    # clipped to this each step. None disables clipping (legacy behavior).
    inverter_ac_capacity_w: Optional[float] = None
    max_charge_power_w: Optional[float] = None
    max_discharge_power_w: Optional[float] = None
    # Capacity-proportional alternative to the two absolute limits above. When
    # set, the stored energy changes by at most ``power_limit_c_rate *
    # nominal_energy_wh`` per hour in either direction, so a sizing sweep keeps
    # one C-rate instead of one wattage across capacities. The absolute limits
    # apply at other boundaries, so setting either with it raises.
    power_limit_c_rate: Optional[float] = None
    # In-dispatch derate applied to inverter AC output after the part-load
    # curve and every inverter limit. It stands in for AC-side shortfall the
    # chain does not model, without moving the clipping threshold or the
    # part-load ratio, which is why it is not folded into
    # ``inverter_efficiency``. Bounded to (0, 1]: above 1 the inverter would
    # exceed its nameplate and emit more AC than the DC entering it. Correct
    # an under-predicting model on the DC side instead. The default 1.0 is a
    # no-op and reproduces prior behaviour bit-for-bit.
    ac_output_scale: float = 1.0

    def __post_init__(self):
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

        # Checked rather than coerced: a string such as "false" is truthy.
        if not isinstance(self.allow_terminal_replacement, (bool, np.bool_)):
            raise ValueError("allow_terminal_replacement must be a bool")
        self.allow_terminal_replacement = bool(self.allow_terminal_replacement)
        if isinstance(self.replacement_years_after_span, (bool, np.bool_)) or not isinstance(
            self.replacement_years_after_span, (int, np.integer)
        ):
            raise ValueError("replacement_years_after_span must be a non-negative integer")
        if self.replacement_years_after_span < 0:
            raise ValueError("replacement_years_after_span must be a non-negative integer")
        self.replacement_years_after_span = int(self.replacement_years_after_span)
        self.replacement_min_remaining_years = finite(
            "replacement_min_remaining_years", self.replacement_min_remaining_years
        )
        if self.replacement_min_remaining_years < 0.0:
            raise ValueError("replacement_min_remaining_years must be non-negative")

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
        self.thermal_resistance_k_per_w = finite("thermal_resistance_k_per_w", self.thermal_resistance_k_per_w)

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
        for name in ("standby_loss_wh", "initial_resistance_growth", "thermal_resistance_k_per_w"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")

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
        if self.power_limit_c_rate is not None:
            if self.max_charge_power_w is not None or self.max_discharge_power_w is not None:
                raise ValueError(
                    "power_limit_c_rate limits both directions from capacity; do not also set "
                    "max_charge_power_w or max_discharge_power_w"
                )
            rate = finite("power_limit_c_rate", self.power_limit_c_rate)
            if rate <= 0.0:
                raise ValueError("power_limit_c_rate must be greater than 0")
            self.power_limit_c_rate = rate
        self.ac_output_scale = finite("ac_output_scale", self.ac_output_scale)
        if not 0.0 < self.ac_output_scale <= 1.0:
            raise ValueError(
                "ac_output_scale must be greater than 0 and at most 1; it is applied after the "
                "inverter nameplate limit, so a value above 1 would deliver more AC than the "
                "nameplate and more AC than the DC entering the inverter. Scale the DC series "
                "instead to correct an under-predicting model"
            )

    @property
    def stored_power_limit_w(self) -> Optional[float]:
        """The C-rate limit on stored-energy change (W), or None without one."""
        if self.power_limit_c_rate is None:
            return None
        return self.power_limit_c_rate * self.nominal_energy_wh


def _require_complete_series(values: pd.Series, name: str) -> None:
    """Reject a series that leaves any simulation step without a finite value.

    A missing reading is not a zero. Filling it would turn incomplete input
    into a plausible-looking result, so gaps must be filled explicitly by the
    caller, where the choice stays visible.
    """
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    invalid = ~np.isfinite(numeric)
    if invalid.any():
        first = values.index[int(np.argmax(invalid))]
        raise ValueError(
            f"{name} has no finite value for {int(invalid.sum())} of {len(values)} simulation steps "
            f"(first at {first}). Supply input that covers the whole simulation range at its "
            "interval, or fill the gaps explicitly before simulating."
        )


def _repeat_annual_load_across_year_edges(load_utc: pd.Series, rng_utc: pd.DatetimeIndex) -> pd.Series:
    """Extend an annual load profile by one year on each side the window needs.

    A load profile covers one civil year in its own timezone, while weather
    read from UTC covers one UTC year. East of UTC the last simulated hours
    fall in the next civil year; west of UTC the first ones fall in the
    previous year. Those steps take the profile's value one year earlier or
    later, which is the same instant of a repeating annual profile. The
    original values always win, so a gap inside the data is not covered and
    still fails validation.
    """
    if load_utc.empty:
        return load_utc
    parts = [load_utc]
    if rng_utc[0] < load_utc.index[0]:
        parts.append(remap_datetime_index_years(load_utc, -1))
    if rng_utc[-1] > load_utc.index[-1]:
        parts.append(remap_datetime_index_years(load_utc, 1))
    if len(parts) == 1:
        return load_utc
    combined = pd.concat(parts)
    return combined[~combined.index.duplicated(keep="first")].sort_index()


def _align_input_arrays(
    pv_dc: pd.Series,
    houseload: pd.DataFrame,
    temperature_series: Optional[pd.Series],
    rng: pd.DatetimeIndex,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reindex PV, load and temperature onto ``rng`` as float64 arrays.

    The loop indexes these positionally, so everything the simulation reads
    per step is settled here. PV and load must cover every step with a finite
    value, and load must not be negative. A temperature series must cover
    every step too; only an omitted one defaults to 25C. The load profile is year-shifted when it comes from a different year
    than the simulation window, and it repeats across the year edge (see
    :func:`_repeat_annual_load_across_year_edges`).
    """
    pv_values = pv_dc.reindex(rng)
    _require_complete_series(pv_values, "pv_dc")

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
        houseload_series.index = load_utc
        if load_dominant_year != sim_dominant_year:
            year_offset = sim_dominant_year - load_dominant_year
            houseload_series = remap_datetime_index_years(houseload_series, year_offset)
        houseload_series = _repeat_annual_load_across_year_edges(houseload_series, rng_utc)
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
    houseload_series = houseload_series.reindex(rng)
    _require_complete_series(houseload_series, "houseload")
    negative = houseload_series.to_numpy(dtype=float) < 0.0
    if negative.any():
        raise ValueError(
            f"houseload is negative at {int(negative.sum())} simulation steps "
            f"(minimum {float(houseload_series.min()):.6g} W). Load must be gross building "
            "demand; a net-meter reading that includes on-site generation cannot stand in for it."
        )

    if temperature_series is None:
        temperature_series = pd.Series(25.0, index=rng)
    else:
        temperature_series = temperature_series.reindex(rng)
        _require_complete_series(temperature_series, "temperature_series")

    return (
        pv_values.values.astype(np.float64),
        houseload_series.values.astype(np.float64),
        temperature_series.values.astype(np.float64),
    )


@dataclass(frozen=True, slots=True)
class AlignedSimulationInputs:
    """PV, load and temperature already aligned onto one simulation calendar.

    Both simulation entry points do this work internally on every call: they
    build the calendar, reindex three series onto it, and year-shift the load
    profile when it comes from a different year than the simulated window.
    :func:`simulate_energy_balance_summary` will take the result instead, via
    its ``aligned`` parameter.

    A Monte Carlo study repeats that for every simulated year of every
    trajectory against the same weather calendar, with a load profile that
    differs only by a scalar -- tens of thousands of times per study for one
    of nineteen distinct answers. Aligning the calendar once and scaling
    afterwards is exact rather than merely equivalent: reindexing only
    selects elements, and multiplying by a scalar commutes with that, so every
    element is the same product either way.

    Attributes:
        index: The simulation calendar. Everything else is positional on it.
        pv_dc_w: PV DC power (W) per step. Every step has a finite value.
        load_w: AC load (W) per step. Every step is finite and non-negative.
        temperature_c: Battery cell temperature (C) per step, 25 C if none was given.
        freq: The calendar's step, which converts power to energy. A run on
            these inputs takes it from here.
    """

    index: pd.DatetimeIndex
    pv_dc_w: np.ndarray
    load_w: np.ndarray
    temperature_c: np.ndarray
    freq: str
    pv_chain: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    pv_chain_key: Optional[Tuple[int, int, int, float, float, float, float]] = None

    def resolve_freq(self, freq: Optional[str]) -> str:
        """Return the inputs' own step, refusing a ``freq`` that disagrees with it."""
        if freq is not None and get_hours_per_step(freq) != get_hours_per_step(self.freq):
            raise ValueError(
                f"freq={freq!r} disagrees with the aligned inputs, which step at {self.freq!r}; omit freq to use theirs"
            )
        return self.freq

    def scaled(self, *, pv_factor: float = 1.0, load_factor: float = 1.0) -> "AlignedSimulationInputs":
        """Return the same calendar with PV and load scaled by a constant.

        A change in PV invalidates any memoized chain; a change in load does
        not, because the chain stops at the inverter terminals.
        """
        return AlignedSimulationInputs(
            index=self.index,
            pv_dc_w=self.pv_dc_w * pv_factor if pv_factor != 1.0 else self.pv_dc_w,
            load_w=self.load_w * load_factor if load_factor != 1.0 else self.load_w,
            temperature_c=self.temperature_c,
            freq=self.freq,
            pv_chain=self.pv_chain if pv_factor == 1.0 else None,
            pv_chain_key=self.pv_chain_key if pv_factor == 1.0 else None,
        )

    def _pv_chain_cache_key(
        self,
        battery_config: "BatteryConfig",
        hours_per_step: float,
        *,
        pv_dc_w: Optional[np.ndarray] = None,
    ) -> Tuple[int, int, int, float, float, float, float]:
        """Identify the aligned PV values and inverter settings used by the memo."""
        cap_wh = _step_energy_cap(battery_config.inverter_ac_capacity_w, hours_per_step)
        pv_values = self.pv_dc_w if pv_dc_w is None else pv_dc_w
        return (
            id(self.index),
            id(pv_values),
            len(pv_values),
            float(hours_per_step),
            float(cap_wh),
            float(battery_config.inverter_efficiency),
            float(battery_config.ac_output_scale),
        )

    def with_pv_only_chain(
        self, battery_config: "BatteryConfig", *, freq: Optional[str] = None
    ) -> "AlignedSimulationInputs":
        """Return the same inputs with the DC-to-AC conversion memoized.

        A PV-only step converts DC to AC through the PVWatts efficiency curve
        before it ever looks at the load, so that conversion depends only on
        the weather year and the project year's degradation factor -- never on
        the trajectory. A Monte Carlo study evaluates a few hundred distinct
        (weather year, project year) pairs tens of thousands of times each,
        recomputing the curve every time.

        What is stored is exactly the tuple
        :func:`breos.inverter._calculate_dc_ac_power_arrays` returns for these
        inputs, so a run that uses it takes the same values through the same
        expressions as a run that does not. There is no second arithmetic
        path to keep in step.

        The returned inputs own a read-only copy of the PV array and
        read-only conversion arrays. That keeps the memo stable without
        changing write access to the caller's aligned arrays.
        """
        hours_per_step = get_hours_per_step(self.resolve_freq(freq))
        cap_wh = _step_energy_cap(battery_config.inverter_ac_capacity_w, hours_per_step)
        pv_dc_w = self.pv_dc_w.copy()
        pv_dc_w.setflags(write=False)
        pv_dc_wh = np.maximum(0.0, pv_dc_w * hours_per_step)
        chain = _calculate_dc_ac_power_arrays(
            pv_dc_wh,
            cap_wh,
            battery_config.inverter_efficiency,
            battery_config.ac_output_scale,
        )
        for values in chain:
            values.setflags(write=False)
        pv_chain_key = self._pv_chain_cache_key(battery_config, hours_per_step, pv_dc_w=pv_dc_w)
        return AlignedSimulationInputs(
            index=self.index,
            pv_dc_w=pv_dc_w,
            load_w=self.load_w,
            temperature_c=self.temperature_c,
            freq=self.freq,
            pv_chain=chain,
            pv_chain_key=pv_chain_key,
        )


def align_simulation_inputs(
    pv_dc: pd.Series,
    houseload: pd.DataFrame,
    temperature_series: Optional[pd.Series] = None,
    *,
    freq: str = "h",
    start_time: Optional[pd.Timestamp] = None,
    end_time: Optional[pd.Timestamp] = None,
) -> AlignedSimulationInputs:
    """Align simulation inputs once so many runs can share the result.

    The arguments and their defaults match
    :func:`simulate_energy_balance_summary`, which accepts the result through
    its ``aligned`` parameter. A caller that runs the same calendar
    repeatedly -- a Monte Carlo study, or any sweep over a scalar -- aligns
    once and passes the result to every run.
    """
    if start_time is None:
        start_time = pv_dc.index[0]
    if end_time is None:
        end_time = pv_dc.index[-1]
    rng = pd.date_range(start=start_time, end=end_time, freq=freq)
    pv_values, load_values, temperature_values = _align_input_arrays(pv_dc, houseload, temperature_series, rng)
    return AlignedSimulationInputs(
        index=rng,
        pv_dc_w=pv_values,
        load_w=load_values,
        temperature_c=temperature_values,
        freq=freq,
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
    if initial_degradation_state is not None:
        state_engine = initial_degradation_state.get("degradation_engine")
        if (engine_key == "native" and state_engine != "native") or (
            engine_key == "blast" and state_engine not in (None, "blast")
        ):
            raise ValueError(f"initial_degradation_state requires degradation_engine={state_engine!r}")
    if engine_key == "blast" and not blast_model:
        raise ValueError("blast_model is required when degradation_engine='blast'")
    if engine_key == "blast" and battery_config.enable_resistance_fade:
        raise ValueError("degradation_engine='blast' cannot be combined with enable_resistance_fade")
    return engine_key


def _carried_origin(value: Optional[float], name: str, limit_wh: float, limit_name: str) -> float:
    """Validate one carried origin against the energy it may take, returning it in Wh.

    ``limit_wh`` is often a difference such as ``E - pv``, which can round a
    few ULPs below what a caller split exactly (``E * f`` and ``E * (1 - f)``).
    A value within that rounding of the limit is accepted and clamped to it.
    """
    if value is None:
        return 0.0
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number, not a bool")
    try:
        origin_wh = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(origin_wh):
        raise ValueError(f"{name} must be a finite number")
    slack = 4.0 * math.ulp(max(abs(limit_wh), abs(origin_wh)))
    if not 0.0 <= origin_wh <= limit_wh + slack:
        raise ValueError(f"{name} must be between 0 and {limit_name}")
    # Adding 0.0 turns a -0.0 input into 0.0.
    return min(origin_wh, max(0.0, limit_wh)) + 0.0


def _resolve_carried_energy(
    initial_energy_wh: Optional[float],
    initial_pv_origin_energy_wh: Optional[float],
    battery_config: BatteryConfig,
    battery_soh_decimal: float,
    initial_grid_origin_energy_wh: Optional[float] = None,
) -> Tuple[float, float, float]:
    """Validate the carried stored-energy state, returning ``(energy, pv_origin, grid_origin)``.

    All three default for a fresh run: a battery starting full at its
    configured max SOC, with none of that energy attributable to PV or to the
    grid. Whatever the two origins do not cover is unattributed.
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

    pv_origin_wh = _carried_origin(
        initial_pv_origin_energy_wh, "initial_pv_origin_energy_wh", energy_wh, "initial_energy_wh"
    )
    grid_origin_wh = _carried_origin(
        initial_grid_origin_energy_wh,
        "initial_grid_origin_energy_wh",
        energy_wh - pv_origin_wh,
        "initial_energy_wh minus initial_pv_origin_energy_wh",
    )
    return energy_wh, pv_origin_wh, grid_origin_wh


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
    default_day_start_soc: float,
    default_day_start_t_cell: float,
) -> Tuple[DegradationLifecycle, float, float]:
    """Construct the degradation backend and its first day-boundary state.

    Returns the lifecycle adapter plus the SOC and cell temperature the first
    step should treat as the preceding endpoint. Both adapters consume that
    boundary pair, and a carried snapshot overrides the defaults.
    """
    if engine_key != "blast":
        state_payload = initial_degradation_state or {}
        lifecycle: DegradationLifecycle = NativeDegradationAdapter(
            model_key=battery_config.calendar_model,
            initial_soh_fraction=battery_soh_decimal,
            initial_fec=float(state_payload.get("fec_cum", initial_fec)),
            initial_calendar_seconds=float(state_payload.get("cumulative_calendar_seconds", initial_calendar_seconds)),
            **_native_degradation_kwargs(battery_config.calendar_model),
            cycle_step=_update_battery_soh_from_cycles,
            calendar_step=update_battery_soh_calendar,
            initial_rainflow_state=state_payload.get("native_rainflow_state"),
        )
        return (
            lifecycle,
            float(state_payload.get("day_start_soc_absolute", default_day_start_soc)),
            float(state_payload.get("day_start_temperature_c", default_day_start_t_cell)),
        )

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


def _step_energy_cap(power_w: Optional[float], hours_per_step: float) -> float:
    """Convert a nameplate power limit (W) to a per-step energy cap (Wh).

    ``None`` means unlimited, which the dispatch kernel reads as an infinite
    cap rather than as a separate branch.
    """
    return power_w * hours_per_step if power_w is not None else float("inf")


# Results-frame column order. Every matrix row appears once, under its row
# name. The replacement flag and the capacity it swapped (Wh) are the only
# columns that are not matrix rows, and ``Battery_Energy_End`` is the ``Battery_Energy`` array
# under a second name.
_FRAME_COLUMNS: Tuple[str, ...] = (
    *_ROW_COLUMNS[: _ROW["T_cell"] + 1],
    "Battery_Replaced",
    "Battery_Replaced_Capacity_Wh",
    *_ROW_COLUMNS[_ROW["T_cell"] + 1 : _ROW["Battery_Energy_Beginning"] + 1],
    "Battery_Energy_End",
    *_ROW_COLUMNS[_ROW["Battery_Energy_Beginning"] + 1 :],
)


def _frame_mapping(rows: Mapping[str, np.ndarray], replaced: np.ndarray, replaced_capacity: np.ndarray):
    """Return every frame column by name, in frame order, as a read-only mapping.

    Read-only so a misspelt name fails as a ``KeyError`` instead of adding a
    column nothing reads.
    """
    specials = {
        "Battery_Replaced": replaced,
        "Battery_Replaced_Capacity_Wh": replaced_capacity,
        "Battery_Energy_End": rows["Battery_Energy"],
    }
    return MappingProxyType({name: specials[name] if name in specials else rows[name] for name in _FRAME_COLUMNS})


class _ResultBuffers:
    """Pre-allocated per-timestep output arrays and their frame layout.

    The simulation writes into these positionally rather than appending dicts,
    which is what keeps an 8760-step loop out of pandas. Owning both the
    allocation and :meth:`to_frame` here means the column set is described
    once instead of drifting between two ends of a 700-line function.

    ``columns`` holds every frame column by its frame name, which is also the
    name the day loop's row constants are built from. ``replaced`` and
    ``replaced_capacity`` are zero-filled because only replacement days write
    them; every matrix row is fully overwritten each step and is left
    uninitialised.
    """

    __slots__ = ("matrix", "replaced", "replaced_capacity", "columns")

    def __init__(self, n_steps: int) -> None:
        # One row per per-step column, so a whole day of every output can be
        # handed to a compiled kernel as a single contiguous array. Each
        # column below is a view on its row, not a copy.
        self.matrix: np.ndarray = np.empty((_N_ROWS, n_steps))
        self.replaced: np.ndarray = np.zeros(n_steps, dtype=bool)
        self.replaced_capacity: np.ndarray = np.zeros(n_steps)
        rows = {name: self.matrix[row] for row, name in enumerate(_ROW_COLUMNS)}
        self.columns: Mapping[str, np.ndarray] = _frame_mapping(rows, self.replaced, self.replaced_capacity)

    def zero_fill(self) -> None:
        """Zero every per-step column the caller is not going to write."""
        self.matrix.fill(0.0)

    def column_arrays(self) -> Dict[str, np.ndarray]:
        """Return every per-timestep output column, keyed by its frame name."""
        return dict(self.columns)

    def to_frame(self, rng: pd.DatetimeIndex) -> pd.DataFrame:
        """Assemble the public per-timestep results frame."""
        return pd.DataFrame({"Datetime": rng, **self.columns})


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


# The per-step rows a PV-only run actually writes. Everything else stays at
# zero when there is no battery.
_PV_ONLY_ROWS: Tuple[str, ...] = (
    "PV_DC",
    "PV_Production",
    "Houseload",
    "PV_Delta",
    "Import_From_Grid",
    "Battery_SOH",
    "T_cell",
    "PV_DC_To_Inverter",
    "PV_DC_Curtailed",
    "PV_AC_To_Load",
    "PV_AC_Export",
    "PV_Direct_Inverter_Loss",
)
# Rows a PV-only run fills with values it already wrote under another name.
_PV_ONLY_ALIASES: Dict[str, str] = {
    "Inverter_Loss": "PV_Direct_Inverter_Loss",
}


class _PvOnlySummaryBuffers:
    """Reduced per-step buffers for a PV-only run that only owes a summary.

    A system with no battery leaves most of the frame's columns at zero for
    every step, and writes ``Inverter_Loss`` with the values it already wrote
    as ``PV_Direct_Inverter_Loss``. Allocating the full ``(_N_ROWS, n_steps)``
    matrix to hold that costs several times the memory such a run needs, and
    a Monte Carlo study pays it once per simulated year in every worker at
    once -- which is memory traffic, not arithmetic, and so is exactly what
    stops the study scaling across cores.

    This type presents the same ``columns`` mapping over the written arrays
    (``_PV_ONLY_ROWS``), one shared zero array and one shared zero mask. It
    is deliberately a summary-path type with no ``to_frame``: the detailed
    frame must not hand a caller aliased columns it could write through.
    """

    __slots__ = ("zeros", "replaced", "replaced_capacity", "columns")

    def __init__(self, n_steps: int) -> None:
        self.zeros: np.ndarray = np.zeros(n_steps)
        self.replaced: np.ndarray = np.zeros(n_steps, dtype=bool)
        self.replaced_capacity: np.ndarray = self.zeros
        # Written rows are left uninitialised; the dispatch overwrites every
        # element of each one before anything reads it. An aliased row keeps
        # every reported sum identical and drops one more full-length
        # allocation per simulated year.
        rows: Dict[str, np.ndarray] = {name: np.empty(n_steps) for name in _PV_ONLY_ROWS}
        for alias, name in _PV_ONLY_ALIASES.items():
            rows[alias] = rows[name]
        for name in _ROW_COLUMNS:
            rows.setdefault(name, self.zeros)
        # Many columns share these two arrays, so a stray write would show up
        # in all of them; nothing may write them.
        self.zeros.flags.writeable = False
        self.replaced.flags.writeable = False
        self.columns: Mapping[str, np.ndarray] = _frame_mapping(rows, self.replaced, self.replaced_capacity)

    def zero_fill(self) -> None:
        """No-op: unwritten columns are already served by the zero array."""

    def column_arrays(self) -> Dict[str, np.ndarray]:
        """Return every per-timestep output column, keyed by its frame name."""
        return dict(self.columns)


# What an end-of-life crossing did to the installed pack (ADR 0003 E11): it
# was replaced, or it stayed in service below its threshold.
END_OF_LIFE_ACTIONS: Tuple[str, ...] = ("replaced", "kept")
# Why: the pack reached its threshold and a replacement was allowed; or the
# swap was skipped because less than the minimum service time remained, or
# because it fell in the span's final period with terminal replacement off,
# or because replacement is disabled.
END_OF_LIFE_REASONS: Tuple[str, ...] = (
    "end_of_life",
    "min_remaining_years",
    "terminal_period",
    "replacement_disabled",
)


# The detailed degradation frame's attrs key for the span's crossings.
END_OF_LIFE_EVENTS_ATTR = "end_of_life_events"


@dataclass(frozen=True, slots=True)
class EndOfLifeEvent:
    """One crossing of the end-of-life threshold inside a simulated span.

    A crossing is a degradation period that closes with the installed pack's
    state of health at or below ``eol_percentage``. ``step`` is the
    zero-based closing step of that period and ``timestamp`` its index
    label; the event happens at that step's end, the instant a replacement
    is booked at. ``soh_pct`` is the health the end-of-life check compared
    with the threshold. ``action`` is one of :data:`END_OF_LIFE_ACTIONS` and
    ``reason`` one of :data:`END_OF_LIFE_REASONS`.

    A replaced pack's successor can cross again, so a span may hold several
    replacements. A pack that stays in service below its threshold crosses
    once: later periods, and later spans that inherit it, record nothing
    more. A pack that starts a span at or below its threshold has already
    crossed, unless that span replaces it.
    """

    step: int
    timestamp: pd.Timestamp
    action: str
    reason: str
    soh_pct: float

    def to_record(self) -> Dict[str, Any]:
        """The event as JSON-safe values, its timestamp in ISO 8601."""
        return {
            "step": int(self.step),
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "reason": self.reason,
            "soh_pct": float(self.soh_pct),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "EndOfLifeEvent":
        """The event a :meth:`to_record` dict describes."""
        return cls(
            int(record["step"]),
            pd.Timestamp(record["timestamp"]),
            str(record["action"]),
            str(record["reason"]),
            float(record["soh_pct"]),
        )


@dataclass(slots=True)
class _AgingState:
    """Battery health state that only changes at a daily boundary.

    Everything here is read by the per-step loop but written once per period by
    :func:`_apply_daily_degradation`, so the loop can keep hot copies in
    locals and refresh them when a period closes.

    ``cumulative_cycle_deg`` and ``cumulative_cal_deg`` split the installed
    pack's SOH loss into its cycle and calendar parts. The degradation
    adapters report only each period's increment, so this is the one running
    total; the carry state reads it, like ``resistance_growth``.

    ``fec_cum`` belongs to the pack currently installed and is reset to zero
    when that pack is replaced, so it cannot be differenced across a
    replacement. ``fec_lifetime`` adds up rainflow counts across packs without
    resetting, so a retired pack's final period remains in the run total.
    """

    soh_fraction: float
    soh_percent: float
    fec_cum: float
    fec_lifetime: float
    cumulative_cal_seconds: float
    cumulative_cycle_deg: float
    cumulative_cal_deg: float
    resistance_growth: float
    eff_charge: float
    eff_discharge: float
    n_replacements: int
    # Nominal capacity swapped in, summed over the replacements (Wh). The
    # economics prices it; the physics carries no money (ADR 0003 E4).
    replaced_capacity_wh: float
    day_start_soc: float
    day_start_t_cell: float
    # True once the installed pack is in service at or below its end-of-life
    # threshold without a replacement, so the crossing is recorded once.
    past_end_of_life: bool = False
    end_of_life_events: List[EndOfLifeEvent] = field(default_factory=list)


def _apply_resistance_fade(
    aging: _AgingState,
    battery_config: BatteryConfig,
    cycles: tuple[Dict[str, Any], ...],
    *,
    mean_t_cell: float,
    mean_soc_absolute: float,
    dt_days: float,
) -> float:
    """Grow internal resistance for one period and return the effective RTE.

    The cycle term is charged against the FEC standing at the *start* of the
    period, so this period's cycles do not count towards their own aging; the
    lifecycle step has already folded them into the cumulative FEC, so they
    are subtracted back out here.
    """
    day_fec = sum(max(0.0, min(1.0, c["doc"])) * c["count"] for c in cycles)
    fec_before_day = aging.fec_cum - day_fec

    aging.resistance_growth, _ = update_battery_resistance_cyclewise(aging.resistance_growth, cycles, fec_before_day)
    aging.resistance_growth, _ = update_battery_resistance_calendar(
        aging.resistance_growth,
        T_cell_C=mean_t_cell,
        cumulative_cal_seconds=aging.cumulative_cal_seconds,
        dt_days=dt_days,
        mean_soc_absolute=mean_soc_absolute,
    )
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
    pv_origin_energy_wh: float,
    grid_origin_energy_wh: float,
    battery_energy_beginning: float,
) -> Tuple[float, float, float, float]:
    """Swap in a new pack, returning ``(energy, pv_origin, grid_origin, day_end_soc)``.

    Replacement happens *inside* the closing timestep, after that step's
    results were already recorded. The recorded end-of-step state is
    therefore rewritten so it matches the next step's beginning, and both
    external energy transfers are exposed for whole-system reconciliation.
    The retired pack takes every origin with it; the new pack's energy is
    unattributed (ADR 0002 A8).
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
    aging.n_replacements += 1
    aging.replaced_capacity_wh += battery_config.nominal_energy_wh

    battery_energy_wh = battery_config.nominal_energy_wh * battery_config.max_soc
    replacement_energy_added = battery_energy_wh
    lifecycle.reset()

    out.replaced[step_index] = True
    out.replaced_capacity[step_index] = battery_config.nominal_energy_wh
    out.columns["Battery_Energy"][step_index] = battery_energy_wh
    out.columns["Battery_SOC_Normalized"][step_index] = 1.0
    out.columns["Battery_SOC_Absolute"][step_index] = battery_config.max_soc
    out.columns["Battery_SOH"][step_index] = 100.0
    out.columns["Battery_PV_Origin_Energy_End"][step_index] = 0.0
    out.columns["Battery_Grid_Origin_Energy_End"][step_index] = 0.0
    out.columns["Battery_Replacement_Energy_Removed"][step_index] = replacement_energy_removed / hours_per_step
    out.columns["PV_Origin_Replacement_Energy_Removed"][step_index] = pv_origin_energy_wh / hours_per_step
    out.columns["Grid_Origin_Replacement_Energy_Removed"][step_index] = grid_origin_energy_wh / hours_per_step
    out.columns["Battery_Replacement_Energy_Added"][step_index] = replacement_energy_added / hours_per_step
    out.columns["Battery_Energy_Delta"][step_index] = (battery_energy_wh - battery_energy_beginning) / hours_per_step

    return battery_energy_wh, 0.0, 0.0, battery_config.max_soc


def _dispatch_no_battery_vectorized(
    out: Union["_ResultBuffers", "_PvOnlySummaryBuffers"],
    pv_dc_values: np.ndarray,
    load_values: np.ndarray,
    temperature_values: np.ndarray,
    *,
    battery_config: BatteryConfig,
    hours_per_step: float,
    cap_wh: float,
    pv_chain: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> None:
    """Fill a PV-only result buffer without entering the timestep loop."""
    out.zero_fill()

    pv_dc_wh = np.maximum(0.0, pv_dc_values * hours_per_step)
    load_wh = load_values * hours_per_step
    if pv_chain is None:
        ac_wh, conversion_loss_wh, clipping_loss_dc_wh = _calculate_dc_ac_power_arrays(
            pv_dc_wh,
            cap_wh,
            battery_config.inverter_efficiency,
            battery_config.ac_output_scale,
        )
    else:
        # Memoized by the caller for these exact inputs. Same values, same
        # expressions downstream; see AlignedSimulationInputs.with_pv_only_chain.
        ac_wh, conversion_loss_wh, clipping_loss_dc_wh = pv_chain
    pv_ac_to_load_wh = np.minimum(ac_wh, load_wh)
    grid_export_wh = ac_wh - pv_ac_to_load_wh
    grid_import_wh = np.maximum(0.0, load_wh - pv_ac_to_load_wh)
    # Keep the scalar reference's subtraction order. Returning ``ac_wh``
    # here is algebraically equivalent but differs by one ULP at low load.
    pv_production_wh = pv_dc_wh - clipping_loss_dc_wh - conversion_loss_wh

    # Inverter_Loss is PV_Direct_Inverter_Loss when there is no battery: it is
    # divided once and assigned twice. On the reduced summary buffer the
    # second assignment writes the array the first one already filled, which
    # is what makes sharing it safe: both names carry the same values.
    curtailment_w = clipping_loss_dc_wh / hours_per_step
    grid_export_w = grid_export_wh / hours_per_step
    conversion_w = conversion_loss_wh / hours_per_step

    out.columns["PV_DC"][:] = pv_dc_wh / hours_per_step
    out.columns["PV_Production"][:] = pv_production_wh / hours_per_step
    out.columns["Houseload"][:] = load_wh / hours_per_step
    out.columns["PV_Delta"][:] = (pv_production_wh - load_wh) / hours_per_step
    out.columns["Import_From_Grid"][:] = grid_import_wh / hours_per_step
    out.columns["Battery_SOH"].fill(100.0)
    out.columns["T_cell"][:] = temperature_values

    out.columns["PV_DC_To_Inverter"][:] = (pv_dc_wh - clipping_loss_dc_wh) / hours_per_step
    out.columns["PV_DC_Curtailed"][:] = curtailment_w
    out.columns["PV_AC_To_Load"][:] = pv_ac_to_load_wh / hours_per_step
    out.columns["PV_AC_Export"][:] = grid_export_w
    out.columns["PV_Direct_Inverter_Loss"][:] = conversion_w
    out.columns["Inverter_Loss"][:] = conversion_w


def _apply_daily_degradation(
    aging: _AgingState,
    lifecycle: DegradationLifecycle,
    battery_config: BatteryConfig,
    out: _ResultBuffers,
    degradation_tracking: List[Dict[str, Any]],
    *,
    step_index: int,
    step_time: pd.Timestamp,
    time_ticks: np.ndarray,
    ticks_per_second: float,
    soc_absolute_day: np.ndarray,
    t_cell_day: np.ndarray,
    finalize_cycles: bool,
    hours_per_step: float,
    battery_energy_wh: float,
    pv_origin_energy_wh: float,
    grid_origin_energy_wh: float,
    battery_energy_beginning: float,
    replacement_skipped_by: Optional[str] = None,
) -> Tuple[float, float, float]:
    """Close out one degradation period, returning ``(energy, pv_origin, grid_origin)``.

    Runs the lifecycle step, optional resistance fade and the end-of-life
    replacement check in that order, mutating *aging* in place and appending
    one row to *degradation_tracking*. The stored-energy values are
    returned rather than carried on *aging* because the per-step loop owns
    them and only a replacement changes them here.

    The period's SOC and cell-temperature endpoints become the next starting
    boundary; a replacement moves that endpoint to the fresh pack's
    max SOC, since the recorded state was rewritten to match.

    ``replacement_skipped_by`` names the rule that skips this period's
    end-of-life swap, one of :data:`END_OF_LIFE_REASONS`: the span's final
    period when the battery does not allow a terminal replacement, or any
    period that closes with less than the battery's minimum service time
    left. None allows the swap. The period still ages the pack and is still
    recorded; only the swap is skipped. A crossing is recorded on
    ``aging.end_of_life_events``, with what was done and why.
    """
    period_steps = len(soc_absolute_day)
    period_seconds = period_steps * hours_per_step * 3600.0
    dt_days = period_seconds / 86400.0
    day_end_soc_absolute = float(soc_absolute_day[-1])
    day_end_t_cell = float(t_cell_day[-1])

    mean_soc_abs = float(np.mean(soc_absolute_day))
    mean_t_cell = float(np.mean(t_cell_day))
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
            finalize_cycles=finalize_cycles,
        )
    )
    aging.soh_fraction = degradation_step.soh_fraction
    aging.soh_percent = aging.soh_fraction * 100.0
    # The lifecycle reports the installed pack's running total, so the day's
    # own rainflow count is the increment over yesterday's total. Taking it
    # here, before the replacement check below can zero the pack counter, is
    # what keeps a retired pack's final part-period in the lifetime figure.
    aging.fec_lifetime += degradation_step.fec - aging.fec_cum
    aging.fec_cum = degradation_step.fec
    aging.cumulative_cal_seconds = degradation_step.calendar_seconds
    aging.cumulative_cycle_deg += degradation_step.cycle_degradation
    aging.cumulative_cal_deg += degradation_step.calendar_degradation

    if battery_config.enable_resistance_fade:
        effective_rte = _apply_resistance_fade(
            aging,
            battery_config,
            degradation_step.cycle_records,
            mean_t_cell=mean_t_cell,
            mean_soc_absolute=mean_soc_abs,
            dt_days=dt_days,
        )

    reached_end_of_life = aging.soh_fraction <= battery_config.eol_percentage
    if reached_end_of_life and replacement_skipped_by is None and battery_config.enable_replacement:
        aging.end_of_life_events.append(
            EndOfLifeEvent(step_index, step_time, "replaced", "end_of_life", aging.soh_percent)
        )
        # A retired native pack owns its unresolved terminal half cycles.
        # Settle them before reset so its lifetime FEC remains complete while
        # the replacement starts with a clean rainflow residue.
        terminal_cycles = lifecycle.finalize_cycles()
        aging.fec_lifetime += terminal_cycles.fec - aging.fec_cum
        aging.fec_cum = terminal_cycles.fec
        aging.soh_fraction = terminal_cycles.soh_fraction
        aging.soh_percent = aging.soh_fraction * 100.0
        aging.cumulative_cycle_deg += terminal_cycles.cycle_degradation
        cycle_degradation_for_row = degradation_step.cycle_degradation + terminal_cycles.cycle_degradation
        battery_energy_wh, pv_origin_energy_wh, grid_origin_energy_wh, day_end_soc_absolute = (
            _apply_battery_replacement(
                aging,
                battery_config,
                lifecycle,
                out,
                step_index=step_index,
                hours_per_step=hours_per_step,
                battery_energy_wh=battery_energy_wh,
                pv_origin_energy_wh=pv_origin_energy_wh,
                grid_origin_energy_wh=grid_origin_energy_wh,
                battery_energy_beginning=battery_energy_beginning,
            )
        )
        aging.past_end_of_life = False
    else:
        cycle_degradation_for_row = degradation_step.cycle_degradation
        if reached_end_of_life and not aging.past_end_of_life:
            reason = "replacement_disabled" if not battery_config.enable_replacement else replacement_skipped_by
            assert reason is not None
            aging.end_of_life_events.append(EndOfLifeEvent(step_index, step_time, "kept", reason, aging.soh_percent))
            aging.past_end_of_life = True

    degradation_record = {
        "Datetime": step_time,
        "SOH": aging.soh_percent,
        "Cycle_Degradation": cycle_degradation_for_row,
        "Calendar_Degradation": degradation_step.calendar_degradation,
        "Cumulative_Cycle_Degradation": aging.cumulative_cycle_deg,
        "Cumulative_Calendar_Degradation": aging.cumulative_cal_deg,
        "Cumulative_FEC": aging.fec_cum,
        "Cumulative_FEC_All_Packs": aging.fec_lifetime,
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
    return battery_energy_wh, pv_origin_energy_wh, grid_origin_energy_wh


def _replacement_skipped_by(
    battery_config: BatteryConfig, *, final_period: bool, remaining_steps: int, min_remaining_steps: float
) -> Optional[str]:
    """The rule that skips an end-of-life swap at a period close, or None.

    A pack bought with less than the minimum left to serve is not bought.
    The period closing on the span's last step is its final one, whole or
    partial; a pack bought there would serve no step. Any positive minimum
    skips that period too, so it is named first.
    """
    if remaining_steps < min_remaining_steps:
        return "min_remaining_years"
    if final_period and not battery_config.allow_terminal_replacement:
        return "terminal_period"
    return None


def _build_summary_row(
    buffers: _ResultBuffers,
    hours_per_step: float,
    *,
    final_soh_percent: float,
    n_replacements: int,
    replaced_capacity_wh: float,
) -> Tuple[Dict[str, float], float]:
    """Summarise a completed run, returning ``(summary_row, total_pv_wh)``.

    ``total_pv`` is both a summary row and a separate public return value, so
    it is computed once here and handed back rather than recomputed.
    """
    total_pv = np.sum(buffers.columns["PV_Production"]) * hours_per_step
    total_load = np.sum(buffers.columns["Houseload"]) * hours_per_step
    total_sell = np.sum(buffers.columns["PV_AC_Export"]) * hours_per_step
    total_import = np.sum(buffers.columns["Import_From_Grid"]) * hours_per_step

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
        "Replaced_Capacity_kWh": replaced_capacity_wh / 1000.0,
    }
    return summary, total_pv


def _build_final_degradation_state(
    degradation_lifecycle: DegradationLifecycle,
    aging: _AgingState,
) -> Dict[str, Any]:
    """Assemble the carry state a follow-on run can be resumed from.

    The engine-independent keys are listed first and explicitly, so the
    schema a caller round-trips does not depend on adapter dict ordering;
    whatever else the adapter reports (BLAST engine internals) follows.
    Resistance growth and the cycle/calendar degradation split are owned by
    the energy loop, not by either adapter.
    """
    adapter_snapshot = degradation_lifecycle.snapshot(
        day_start_soc=aging.day_start_soc,
        day_start_temperature_c=aging.day_start_t_cell,
    )
    return {
        "degradation_engine": adapter_snapshot.pop("degradation_engine"),
        "fec_cum": float(adapter_snapshot.pop("fec_cum")),
        "cumulative_calendar_seconds": float(adapter_snapshot.pop("cumulative_calendar_seconds")),
        "resistance_growth": float(aging.resistance_growth),
        "cumulative_cycle_degradation": float(aging.cumulative_cycle_deg),
        "cumulative_calendar_degradation": float(aging.cumulative_cal_deg),
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
    # The daily controller's carry; None without a controller.
    controller_carry: Optional[ControllerCarry] = None
    # A copy of the instructions a daily controller's decisions dispatched,
    # one per simulated step; None without a controller.
    controller_instructions: Optional[DispatchInstructions] = None


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
    column_sums: Dict[str, float]
    total_pv_wh: float
    summary_row: Dict[str, float]
    final_soh_percent: float
    n_replacements: int
    replaced_capacity_wh: float
    opening_energy_wh: float
    opening_pv_origin_energy_wh: float
    opening_grid_origin_energy_wh: float
    carried_energy_wh: float
    carried_pv_origin_energy_wh: float
    carried_grid_origin_energy_wh: float
    has_degradation_rows: bool
    fec_cum: float
    cumulative_calendar_seconds: float
    cumulative_cycle_degradation: float
    cumulative_calendar_degradation: float
    resistance_growth: float
    replacement_steps: Tuple[int, ...]
    # Rainflow cycles every pack accumulated in the span, a retired pack's
    # part-period included; ``fec_cum`` restarts at zero on replacement.
    fec_all_packs: float = 0.0
    final_degradation_state: Optional[Dict[str, Any]] = None
    # Each end-of-life crossing in the span, in step order (ADR 0003 E11).
    end_of_life_events: Tuple[EndOfLifeEvent, ...] = ()
    # ``sum(column * weights)`` for each requested (column, weights) pair,
    # such as import power times the step's import price.
    weighted_sums: Dict[str, float] = field(default_factory=dict)
    # The ledger schema the column sums follow; App and Monte Carlo report
    # the same version.
    ledger_schema_version: str = LEDGER_SCHEMA_VERSION


def frame_replaced_capacity_wh(results_df: pd.DataFrame) -> float:
    """The nominal capacity a results frame's replacements swapped in (Wh).

    Added one event at a time from 0.0, in step order, as the day loop adds
    it to ``SimulationSummary.replaced_capacity_wh``, so the detailed and
    summary paths report the same float. Only replacement steps are non-zero.
    """
    total = 0.0
    swapped = results_df["Battery_Replaced_Capacity_Wh"].to_numpy(dtype=float)
    for value in swapped[swapped != 0.0]:
        total += float(value)
    return total


def weighted_column_sums(
    columns: Mapping[str, np.ndarray], weights: Optional[Mapping[str, Tuple[str, np.ndarray]]]
) -> Dict[str, float]:
    """``sum(columns[column] * w)`` for each ``name: (column, w)`` in ``weights``.

    Frames and summaries both reduce through this, so a priced total is the
    same float whichever path produced the column. The dot products run on one
    BLAS thread, so it is also the same float in a serial run and in a worker
    process, whatever the core count.
    """
    if not weights:
        return {}
    sums: Dict[str, float] = {}
    with single_thread_blas():
        for name, (column, values) in weights.items():
            array = np.asarray(columns[column], dtype=float)
            if len(values) != len(array):
                raise ValueError(f"weights {name!r} have {len(values)} steps; the simulation has {len(array)}")
            sums[name] = float(np.dot(array, np.asarray(values, dtype=float)))
    return sums


def _build_simulation_summary(
    core: _CoreRun,
    *,
    return_degradation_state: bool,
    weights: Optional[Mapping[str, Tuple[str, np.ndarray]]] = None,
) -> SimulationSummary:
    """Reduce a completed core run to its annual totals and carry state."""
    buffers = core.buffers
    aging = core.aging
    columns = buffers.column_arrays()
    column_sums = _column_sums(columns)

    summary_row, total_pv = _build_summary_row(
        buffers,
        core.hours_per_step,
        final_soh_percent=aging.soh_percent,
        n_replacements=aging.n_replacements,
        replaced_capacity_wh=aging.replaced_capacity_wh,
    )

    final_state = None
    if return_degradation_state:
        final_state = _build_final_degradation_state(core.lifecycle, aging)

    return SimulationSummary(
        n_steps=len(buffers.columns["Battery_Energy"]),
        hours_per_step=core.hours_per_step,
        column_sums=column_sums,
        total_pv_wh=total_pv,
        summary_row=summary_row,
        final_soh_percent=aging.soh_percent,
        n_replacements=aging.n_replacements,
        replaced_capacity_wh=aging.replaced_capacity_wh,
        # Read from the recorded end-of-step state rather than from the loop
        # locals: a replacement inside the closing step rewrites what the next
        # year must resume from.
        # The opening state makes the year-to-year seam checkable from a
        # summary alone: the first step's beginning energy must equal what the
        # previous year carried out.
        opening_energy_wh=float(buffers.columns["Battery_Energy_Beginning"][0]),
        opening_pv_origin_energy_wh=float(buffers.columns["Battery_PV_Origin_Energy_Beginning"][0]),
        opening_grid_origin_energy_wh=float(buffers.columns["Battery_Grid_Origin_Energy_Beginning"][0]),
        carried_energy_wh=float(buffers.columns["Battery_Energy"][-1]),
        carried_pv_origin_energy_wh=float(buffers.columns["Battery_PV_Origin_Energy_End"][-1]),
        carried_grid_origin_energy_wh=float(buffers.columns["Battery_Grid_Origin_Energy_End"][-1]),
        has_degradation_rows=bool(core.degradation_tracking),
        fec_cum=aging.fec_cum,
        cumulative_calendar_seconds=aging.cumulative_cal_seconds,
        cumulative_cycle_degradation=aging.cumulative_cycle_deg,
        cumulative_calendar_degradation=aging.cumulative_cal_deg,
        resistance_growth=aging.resistance_growth,
        replacement_steps=tuple(int(i) for i in np.flatnonzero(buffers.replaced)),
        fec_all_packs=aging.fec_lifetime,
        final_degradation_state=final_state,
        end_of_life_events=tuple(aging.end_of_life_events),
        weighted_sums=weighted_column_sums(columns, weights),
    )


def _resolve_finalize_degradation(finalize_degradation: Optional[bool], return_degradation_state: bool) -> bool:
    """Resolve whether terminal rainflow residue belongs to this span.

    A span finalizes unless the caller opts out explicitly. Returning state is
    not enough to leave cycles open: a caller that asks for state but does not
    feed it back would otherwise lose the unresolved half cycles.
    """
    if finalize_degradation is None:
        return True
    if not finalize_degradation and not return_degradation_state:
        raise ValueError("finalize_degradation=False requires return_degradation_state=True")
    return bool(finalize_degradation)


def _simulate_core(
    pv_dc: Optional[pd.Series] = None,
    houseload: Optional[pd.DataFrame] = None,
    battery_config: Optional[BatteryConfig] = None,
    start_time: Optional[pd.Timestamp] = None,
    end_time: Optional[pd.Timestamp] = None,
    freq: str = "h",
    temperature_series: Optional[pd.Series] = None,
    initial_fec: float = 0.0,
    initial_calendar_seconds: float = 0.0,
    initial_resistance_growth: Optional[float] = None,
    initial_cumulative_cycle_deg: float = 0.0,
    initial_cumulative_cal_deg: float = 0.0,
    degradation_engine: str = "native",
    blast_model: Optional[str] = None,
    initial_degradation_state: Optional[Dict[str, Any]] = None,
    finalize_degradation: bool = True,
    initial_energy_wh: Optional[float] = None,
    initial_pv_origin_energy_wh: Optional[float] = None,
    initial_grid_origin_energy_wh: Optional[float] = None,
    dispatch_instructions: Optional[DispatchInstructions] = None,
    execution_backend: str = "python",
    summary_only: bool = False,
    aligned: Optional[AlignedSimulationInputs] = None,
    day_controller: Optional[DailyDispatchController] = None,
    controller_tariff: Optional["ResolvedTariff"] = None,
    controller_carry: Optional[ControllerCarry] = None,
    projection_year: int = 0,
    replay_seam: bool = False,
    instruction_planner: Optional[Callable[[ControllerBatteryState], DispatchInstructions]] = None,
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
        pv_dc: Series with PV DC power production (W) - before inverter.
            It must have a finite value at every simulation step.
        houseload: DataFrame with electrical load (W) - AC. It must have a
            finite, non-negative value at every simulation step; gaps raise
            ``ValueError`` rather than being read as zero load.
        battery_config: Battery configuration parameters
        start_time: Simulation start time (defaults to first index of pv_dc)
        end_time: Simulation end time (defaults to last index of pv_dc)
        freq: Time frequency ('h' for hourly, '15min' for 15-minute)
        temperature_series: Ambient (battery-location) temperature (C),
            defaults to 25C. The dispatch derives ``T_cell`` from it.
        degradation_engine: Degradation backend. ``"native"`` preserves the
            Naumann/Lam model; ``"blast"`` uses the BLAST daily endpoint adapter.
        blast_model: BLAST model key when ``degradation_engine="blast"``.
        initial_degradation_state: Optional native or BLAST state returned by
            a previous call with ``return_degradation_state=True``.
        finalize_degradation: Count any remaining rainflow half cycles at the
            end of this span (the default). Set False only when returning state
            that a later span resumes from; otherwise those cycles are lost.
        initial_energy_wh: Optional carried stored-energy state (Wh). Defaults
            to the configured max-SOC state for first-run compatibility.
        initial_pv_origin_energy_wh: Optional PV-origin share of the carried
            stored energy (Wh). Defaults to zero.
        initial_grid_origin_energy_wh: Optional grid-origin share of the
            carried stored energy (Wh). Defaults to zero; whatever neither
            origin covers is unattributed.
        dispatch_instructions: Optional per-step instructions (ADR 0002) for
            the battery dispatch, one entry per simulation step. None is
            greedy self-consumption. PV-only runs ignore them.
        execution_backend: Which implementation runs the within-day dispatch
            arithmetic. ``"python"`` is the default and the numerical
            reference; ``"numba"`` selects the optional compiled kernel and
            requires ``breos[fast]``. Everything outside the day window runs
            in Python either way.
        day_controller: Optional private daily controller (ADR 0002 A11),
            asked for each configured-zone civil day's instructions instead of
            ``dispatch_instructions``. It needs a battery and
            ``controller_tariff``, the resolved tariff whose ``day_starts``
            are the civil-day boundaries.
        controller_tariff: The resolved tariff a ``day_controller`` runs on,
            resolved on this span's simulation index. Its ``day_starts`` are
            the civil-day boundaries, and its labels and prices feed each
            day's known tariff horizon. Required with ``day_controller`` and
            unused without one.
        controller_carry: The controller carry a previous projection year
            returned; None starts a new projection.
        projection_year: The project year this span simulates, for the
            controller's input.
        replay_seam: True when the next span replays this calendar (ADR 0002
            A2), so a civil day cut by the span's end continues at its head.
            False for a standalone span such as a ``[period]``.
        instruction_planner: Optional private hook (ADR 0002 A16), called
            once before the first step with the state the span opens in, after
            the degradation engine has restored its health. It returns the
            span's ``dispatch_instructions``; pass it instead of them.

    Returns:
        A :class:`_CoreRun` holding the filled result buffers, the calendar,
        the aging state and the lifecycle; the public entry points shape
        their results from it. With a ``day_controller`` it also holds the
        controller carry and a copy of the instructions the span dispatched.
    """
    if battery_config is None:
        battery_config = BatteryConfig(nominal_energy_wh=0)

    # Calculate hours per step for energy conversion
    hours_per_step = get_hours_per_step(freq)
    steps_per_day = int(24 / hours_per_step)

    if aligned is None:
        if pv_dc is None or houseload is None:
            raise ValueError("pass either pv_dc and houseload, or aligned inputs")
        # Determine time range
        if start_time is None:
            start_time = pv_dc.index[0]
        if end_time is None:
            end_time = pv_dc.index[-1]
        # Create time range
        rng = pd.date_range(start=start_time, end=end_time, freq=freq)
        _pv_dc_vals, _load_vals, _temp_vals = _align_input_arrays(pv_dc, houseload, temperature_series, rng)
    else:
        # The caller has already built the calendar and reindexed onto it.
        rng = aligned.index
        _pv_dc_vals = aligned.pv_dc_w
        _load_vals = aligned.load_w
        _temp_vals = aligned.temperature_c

    degradation_engine_key = _resolve_degradation_engine(
        degradation_engine,
        blast_model,
        initial_degradation_state,
        battery_config,
    )

    state_payload = initial_degradation_state or {}
    if degradation_engine_key == "native" and initial_degradation_state is not None:
        initial_fec = float(state_payload.get("fec_cum", initial_fec))
        initial_calendar_seconds = float(state_payload.get("cumulative_calendar_seconds", initial_calendar_seconds))
        initial_cumulative_cycle_deg = float(
            state_payload.get("cumulative_cycle_degradation", initial_cumulative_cycle_deg)
        )
        initial_cumulative_cal_deg = float(
            state_payload.get("cumulative_calendar_degradation", initial_cumulative_cal_deg)
        )

    # Initialize state
    battery_soh_decimal = battery_config.initial_soh / 100.0
    if degradation_engine_key == "native" and initial_degradation_state is not None:
        battery_soh_decimal = float(state_payload.get("soh_fraction", battery_soh_decimal))
    Battery_Energy_Wh, Battery_PV_Origin_Energy_Wh, Battery_Grid_Origin_Energy_Wh = _resolve_carried_energy(
        initial_energy_wh,
        initial_pv_origin_energy_wh,
        battery_config,
        battery_soh_decimal,
        initial_grid_origin_energy_wh,
    )

    # Degradation windows are positional (fixed steps_per_day), not
    # calendar-based: DST days shift the windows by design, and a trailing
    # partial window is processed at the end. Dispatch shares this convention.
    # The function argument is the multi-year continuation seam (used by the
    # App's year loop). A native state snapshot takes precedence; otherwise
    # None means to use the battery's configured starting resistance.
    if degradation_engine_key == "native" and "resistance_growth" in state_payload:
        resistance_growth = float(state_payload["resistance_growth"])
    else:
        resistance_growth = (
            battery_config.initial_resistance_growth if initial_resistance_growth is None else initial_resistance_growth
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
    has_battery = not is_pv_only_dispatch(
        battery_config.nominal_energy_wh,
        battery_config.max_soc,
        battery_config.min_soc,
    )
    if not has_battery and aligned is not None and aligned.pv_chain is not None:
        expected_chain_key = aligned._pv_chain_cache_key(battery_config, hours_per_step)
        if aligned.pv_chain_key != expected_chain_key:
            raise ValueError("memoized PV chain does not match the aligned PV inputs or inverter settings")
    # The vectorized PV-only dispatch below is the only producer that leaves
    # most columns at zero, so it is the only one whose output can be served
    # from the reduced buffer -- and only when the caller wants a summary,
    # since the detailed frame must own a writable array per column.
    pv_only_summary = summary_only and not has_battery

    # Pre-allocate result arrays (avoids per-timestep dict creation)
    out = _PvOnlySummaryBuffers(n_steps) if pv_only_summary else _ResultBuffers(n_steps)
    if has_battery and battery_soh_decimal > 0.0:
        degradation_day_start_soc = min(
            1.0,
            max(0.0, Battery_Energy_Wh / (battery_config.nominal_energy_wh * battery_soh_decimal)),
        )
    else:
        degradation_day_start_soc = 0.0
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
        default_day_start_soc=degradation_day_start_soc,
        default_day_start_t_cell=degradation_day_start_t_cell,
    )

    battery_soh_decimal = degradation_lifecycle.soh()
    Battery_SOH = battery_soh_decimal * 100.0
    if degradation_engine_key == "blast" and initial_energy_wh is None:
        Battery_Energy_Wh = battery_config.nominal_energy_wh * battery_soh_decimal * battery_config.max_soc
        # The carried origins were checked against the energy before BLAST
        # restored its SOH; they must also fit the energy the run starts with.
        Battery_PV_Origin_Energy_Wh = _carried_origin(
            initial_pv_origin_energy_wh,
            "initial_pv_origin_energy_wh",
            Battery_Energy_Wh,
            "the starting energy at the restored BLAST SOH",
        )
        Battery_Grid_Origin_Energy_Wh = _carried_origin(
            initial_grid_origin_energy_wh,
            "initial_grid_origin_energy_wh",
            Battery_Energy_Wh - Battery_PV_Origin_Energy_Wh,
            "the starting energy at the restored BLAST SOH minus initial_pv_origin_energy_wh",
        )

    # Health state advanced only at daily boundaries. The loop keeps hot
    # copies of the four fields it reads every step (SOH and the two
    # efficiencies) and refreshes them whenever a day closes.
    aging = _AgingState(
        soh_fraction=battery_soh_decimal,
        soh_percent=Battery_SOH,
        fec_cum=initial_fec,
        # Lifetime FEC starts this span at zero whatever the installed pack
        # already carries, so the end-of-span value is the FEC the span itself
        # accumulated across every pack it used.
        fec_lifetime=0.0,
        cumulative_cal_seconds=initial_calendar_seconds,
        # Coerced so a NumPy scalar argument, such as float32, is summed in
        # float64 like the carried state always was.
        cumulative_cycle_deg=float(initial_cumulative_cycle_deg),
        cumulative_cal_deg=float(initial_cumulative_cal_deg),
        resistance_growth=resistance_growth,
        eff_charge=eff_charge,
        eff_discharge=eff_discharge,
        n_replacements=0,
        replaced_capacity_wh=0.0,
        day_start_soc=degradation_day_start_soc,
        day_start_t_cell=degradation_day_start_t_cell,
        past_end_of_life=battery_soh_decimal <= battery_config.eol_percentage,
    )

    # Per-step energy caps (Wh) for the shared inverter AC nameplate and the
    # battery's own charge/discharge power limits.
    standby_loss_per_step_wh = battery_config.standby_loss_wh * hours_per_step
    cap_wh = _step_energy_cap(battery_config.inverter_ac_capacity_w, hours_per_step)
    cap_charge_wh = _step_energy_cap(battery_config.max_charge_power_w, hours_per_step)
    cap_discharge_wh = _step_energy_cap(battery_config.max_discharge_power_w, hours_per_step)
    cap_stored_wh = _step_energy_cap(battery_config.stored_power_limit_w, hours_per_step)

    dispatch_day = _resolve_dispatch_day(execution_backend)
    if instruction_planner is not None:
        if dispatch_instructions is not None or day_controller is not None:
            raise ValueError("pass one of dispatch_instructions, an instruction planner or a daily controller")
        if not has_battery:
            raise ValueError("an instruction planner needs a battery; a PV-only run has nothing to dispatch")
        # Planned from the state the first step dispatches from, the one a
        # daily controller's first decision would see.
        dispatch_instructions = instruction_planner(
            ControllerBatteryState(
                energy_wh=Battery_Energy_Wh,
                pv_origin_energy_wh=Battery_PV_Origin_Energy_Wh,
                grid_origin_energy_wh=Battery_Grid_Origin_Energy_Wh,
                soh_fraction=battery_soh_decimal,
                resistance_growth=aging.resistance_growth,
                charge_efficiency=eff_charge,
                discharge_efficiency=eff_discharge,
            )
        )
        if not isinstance(dispatch_instructions, DispatchInstructions):
            raise TypeError("an instruction planner must return DispatchInstructions")
    if dispatch_instructions is not None and len(dispatch_instructions) != n_steps:
        raise ValueError(
            f"dispatch_instructions cover {len(dispatch_instructions)} steps; the simulation has {n_steps}"
        )
    # No instructions is greedy dispatch; one no-op set serves every day.
    instructions = dispatch_instructions if dispatch_instructions is not None else DispatchInstructions.noop(n_steps)
    session: Optional[_ControllerSession] = None
    if day_controller is not None:
        if dispatch_instructions is not None:
            raise ValueError("pass either dispatch_instructions or a daily controller, not both")
        if controller_tariff is None:
            raise ValueError("a daily controller needs the resolved tariff's civil calendar")
        if not has_battery:
            raise ValueError("a daily controller needs a battery; a PV-only run has nothing to dispatch")
        session = _ControllerSession(
            day_controller,
            controller_tariff,
            rng,
            hours_per_step=hours_per_step,
            carry=controller_carry,
            projection_year=projection_year,
            replay_seam=replay_seam,
            battery_config=MappingProxyType(dataclasses.asdict(battery_config)),
        )
    # The dispatch reads instructions at global positions; a controller's
    # buffer is filled day by day ahead of each segment.
    dispatch_arrays: Any = instructions if session is None else session.instructions

    # PV-only balance is already vectorized with NumPy and does not need the
    # general per-step dispatcher. This common path is faster than either the
    # Python or Numba day loop and keeps one numerical implementation.
    if not has_battery:
        _dispatch_no_battery_vectorized(
            out,
            _pv_dc_vals,
            _load_vals,
            _temp_vals,
            battery_config=battery_config,
            hours_per_step=hours_per_step,
            cap_wh=cap_wh,
            pv_chain=None if aligned is None else aligned.pv_chain,
        )
        return _CoreRun(
            buffers=out,
            rng=rng,
            aging=aging,
            lifecycle=degradation_lifecycle,
            degradation_tracking=degradation_tracking,
            hours_per_step=hours_per_step,
        )

    # Dispatch advances one degradation day at a time. Health state is fixed
    # for the whole window and advanced here, between windows, so every
    # scientifically sensitive transition stays on this path regardless of
    # which backend ran the arithmetic inside the window.
    # Slicing the DatetimeIndex once per day was a measurable share of a
    # compiled year, so the aging model gets views of one tick array instead.
    time_ticks, ticks_per_second = _datetime_index_ticks(rng)
    # The horizon's steps after this span, and the fewest steps a new pack
    # must have left to serve; both count this span as one project year
    # (see BatteryConfig.replacement_min_remaining_years).
    steps_after_span = battery_config.replacement_years_after_span * n_steps
    min_remaining_steps = battery_config.replacement_min_remaining_years * n_steps
    window_start = 0
    while window_start < n_steps:
        window_end = min(window_start + steps_per_day, n_steps)
        # Without a controller a window is one dispatch call. With one, a
        # civil-day start inside the window splits it (ADR 0002 A11): each
        # segment continues from the previous segment's stored energy and
        # origins, and health stays fixed until the window closes below.
        segment_start = window_start
        while segment_start < window_end:
            if session is None:
                segment_end = window_end
            else:
                if session.decides_at(segment_start):
                    # A day starting where the previous window closed sees the
                    # state after that close's aging and any replacement.
                    session.decide(
                        segment_start,
                        ControllerBatteryState(
                            energy_wh=Battery_Energy_Wh,
                            pv_origin_energy_wh=Battery_PV_Origin_Energy_Wh,
                            grid_origin_energy_wh=Battery_Grid_Origin_Energy_Wh,
                            soh_fraction=battery_soh_decimal,
                            resistance_growth=aging.resistance_growth,
                            charge_efficiency=eff_charge,
                            discharge_efficiency=eff_discharge,
                        ),
                    )
                segment_end = session.prepare_segment(segment_start, window_end)
            dispatch_day(
                out,
                _pv_dc_vals,
                _load_vals,
                _temp_vals,
                segment_start,
                segment_end,
                battery_config=battery_config,
                battery_soh_decimal=battery_soh_decimal,
                Battery_Energy_Wh=Battery_Energy_Wh,
                Battery_PV_Origin_Energy_Wh=Battery_PV_Origin_Energy_Wh,
                Battery_Grid_Origin_Energy_Wh=Battery_Grid_Origin_Energy_Wh,
                eff_charge=eff_charge,
                eff_discharge=eff_discharge,
                hours_per_step=hours_per_step,
                standby_loss_per_step_wh=standby_loss_per_step_wh,
                cap_wh=cap_wh,
                cap_charge_wh=cap_charge_wh,
                cap_discharge_wh=cap_discharge_wh,
                cap_stored_wh=cap_stored_wh,
                instructions=dispatch_arrays,
            )
            if session is not None:
                session.complete_segment(_pv_dc_vals, _load_vals, _temp_vals)
            segment_last = segment_end - 1
            Battery_Energy_Wh = float(out.columns["Battery_Energy"][segment_last])
            Battery_PV_Origin_Energy_Wh = float(out.columns["Battery_PV_Origin_Energy_End"][segment_last])
            Battery_Grid_Origin_Energy_Wh = float(out.columns["Battery_Grid_Origin_Energy_End"][segment_last])
            segment_start = segment_end
        last_step = window_end - 1
        # The window's closing row, written by its final segment, holds the
        # state the day close continues from.
        battery_energy_beginning = float(out.columns["Battery_Energy_Beginning"][last_step])
        Battery_Energy_Wh, Battery_PV_Origin_Energy_Wh, Battery_Grid_Origin_Energy_Wh = _apply_daily_degradation(
            aging,
            degradation_lifecycle,
            battery_config,
            out,
            degradation_tracking,
            step_index=last_step,
            # Timestamps are read once per closed day rather than once per
            # step; nothing inside the window depends on the calendar.
            step_time=rng[last_step],
            time_ticks=time_ticks[window_start:window_end],
            ticks_per_second=ticks_per_second,
            # Copied before the call so a replacement rewriting the closing
            # step's recorded state cannot reach the day the aging model saw.
            soc_absolute_day=out.columns["Battery_SOC_Absolute"][window_start:window_end].copy(),
            t_cell_day=out.columns["T_cell"][window_start:window_end].copy(),
            finalize_cycles=finalize_degradation and window_end == n_steps,
            hours_per_step=hours_per_step,
            battery_energy_wh=Battery_Energy_Wh,
            pv_origin_energy_wh=Battery_PV_Origin_Energy_Wh,
            grid_origin_energy_wh=Battery_Grid_Origin_Energy_Wh,
            battery_energy_beginning=battery_energy_beginning,
            replacement_skipped_by=_replacement_skipped_by(
                battery_config,
                final_period=window_end == n_steps,
                remaining_steps=steps_after_span + n_steps - window_end,
                min_remaining_steps=min_remaining_steps,
            ),
        )
        # Refresh the loop's hot copies of the daily-boundary state.
        battery_soh_decimal = aging.soh_fraction
        eff_charge = aging.eff_charge
        eff_discharge = aging.eff_discharge
        if window_end - window_start < steps_per_day:
            break
        window_start = window_end

    return _CoreRun(
        buffers=out,
        rng=rng,
        aging=aging,
        lifecycle=degradation_lifecycle,
        degradation_tracking=degradation_tracking,
        hours_per_step=hours_per_step,
        controller_carry=None if session is None else session.finish(),
        controller_instructions=None if session is None else session.executed_instructions(),
    )


def simulate_energy_balance(
    pv_dc: pd.Series,
    houseload: pd.DataFrame,
    battery_config: Optional[BatteryConfig] = None,
    start_time: Optional[pd.Timestamp] = None,
    end_time: Optional[pd.Timestamp] = None,
    freq: str = "h",
    temperature_series: Optional[pd.Series] = None,
    *,
    initial_fec: float = 0.0,
    initial_calendar_seconds: float = 0.0,
    initial_resistance_growth: Optional[float] = None,
    initial_cumulative_cycle_deg: float = 0.0,
    initial_cumulative_cal_deg: float = 0.0,
    degradation_engine: str = "native",
    blast_model: Optional[str] = None,
    initial_degradation_state: Optional[Dict[str, Any]] = None,
    return_degradation_state: bool = False,
    initial_energy_wh: Optional[float] = None,
    initial_pv_origin_energy_wh: Optional[float] = None,
    initial_grid_origin_energy_wh: Optional[float] = None,
    dispatch_instructions: Optional[DispatchInstructions] = None,
    execution_backend: str = "python",
    finalize_degradation: Optional[bool] = None,
) -> (
    Tuple[pd.DataFrame, float, pd.DataFrame, int, pd.DataFrame]
    | Tuple[pd.DataFrame, float, pd.DataFrame, int, pd.DataFrame, Dict[str, Any]]
):
    """Simulate an energy balance and return the detailed per-timestep results.

    This is the reference path and the full public contract; see
    :func:`_simulate_core` for the argument semantics. Callers that only need
    annual totals and carry state should use
    :func:`simulate_energy_balance_summary`, which runs the same physics
    without materialising the results frame.

    Native rainflow state carries unresolved cycles between daily windows. By
    default the remaining half cycles are counted at the end of the span,
    whether or not a carry state is returned. A multi-year caller that feeds
    the returned state into the next span passes
    ``return_degradation_state=True, finalize_degradation=False`` for every
    span but the last, so that residue continues into the next span instead.

    Returns:
        Tuple of:
        - results_df: Detailed timestep results
        - total_pv: Total PV AC production after inverter efficiency (Wh)
        - summary_df: Summary statistics
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
        initial_fec=initial_fec,
        initial_calendar_seconds=initial_calendar_seconds,
        initial_resistance_growth=initial_resistance_growth,
        initial_cumulative_cycle_deg=initial_cumulative_cycle_deg,
        initial_cumulative_cal_deg=initial_cumulative_cal_deg,
        degradation_engine=degradation_engine,
        blast_model=blast_model,
        initial_degradation_state=initial_degradation_state,
        finalize_degradation=_resolve_finalize_degradation(finalize_degradation, return_degradation_state),
        initial_energy_wh=initial_energy_wh,
        initial_pv_origin_energy_wh=initial_pv_origin_energy_wh,
        initial_grid_origin_energy_wh=initial_grid_origin_energy_wh,
        dispatch_instructions=dispatch_instructions,
        execution_backend=execution_backend,
    )
    result = _detailed_frames(core)
    if not return_degradation_state:
        return result

    final_degradation_state = _build_final_degradation_state(core.lifecycle, core.aging)
    return (*result, final_degradation_state)


def _detailed_frames(core: _CoreRun) -> Tuple[pd.DataFrame, float, pd.DataFrame, int, pd.DataFrame]:
    """The detailed path's frames: results, total PV, summary, replacements and degradation.

    The degradation frame's ``attrs["end_of_life_events"]`` holds the span's
    end-of-life crossings, as :class:`SimulationSummary` does, so the
    public return tuple keeps its shape. They are
    :meth:`EndOfLifeEvent.to_record` dicts, because pandas writes a frame's
    attrs as JSON (``to_parquet``) and copies them into derived frames.
    """
    df = core.buffers.to_frame(core.rng)
    deg_df = pd.DataFrame(core.degradation_tracking) if core.degradation_tracking else pd.DataFrame()
    deg_df.attrs[END_OF_LIFE_EVENTS_ATTR] = [event.to_record() for event in core.aging.end_of_life_events]
    summary_row, total_pv = _build_summary_row(
        core.buffers,
        core.hours_per_step,
        final_soh_percent=core.aging.soh_percent,
        n_replacements=core.aging.n_replacements,
        replaced_capacity_wh=core.aging.replaced_capacity_wh,
    )
    return df, total_pv, pd.DataFrame([summary_row]), core.aging.n_replacements, deg_df


@dataclass(frozen=True, slots=True)
class _DetailedRun:
    """A detailed span with its carry states, for the projection loop's controller path.

    The first six fields are what ``simulate_energy_balance(...,
    return_degradation_state=True)`` returns; the controller carry and the
    executed controller instructions are kept beside them so neither that
    tuple nor the degradation payload changes.
    """

    results_df: pd.DataFrame
    total_pv_wh: float
    summary_df: pd.DataFrame
    n_replacements: int
    degradation_df: pd.DataFrame
    degradation_state: Dict[str, Any]
    controller_carry: Optional[ControllerCarry]
    controller_instructions: Optional[DispatchInstructions]


def _simulate_detailed_run(
    pv_dc: pd.Series,
    houseload: pd.DataFrame,
    *,
    finalize_degradation: bool,
    **core_kwargs: Any,
) -> _DetailedRun:
    """Run :func:`_simulate_core` with per-step frames and return every carry state.

    Takes the core's keyword arguments, including the private daily
    controller's; see :func:`_simulate_core`.
    """
    core = _simulate_core(pv_dc=pv_dc, houseload=houseload, finalize_degradation=finalize_degradation, **core_kwargs)
    df, total_pv, summary_df, n_replacements, deg_df = _detailed_frames(core)
    return _DetailedRun(
        results_df=df,
        total_pv_wh=total_pv,
        summary_df=summary_df,
        n_replacements=n_replacements,
        degradation_df=deg_df,
        degradation_state=_build_final_degradation_state(core.lifecycle, core.aging),
        controller_carry=core.controller_carry,
        controller_instructions=core.controller_instructions,
    )


def simulate_energy_balance_summary(
    pv_dc: Optional[pd.Series] = None,
    houseload: Optional[pd.DataFrame] = None,
    battery_config: Optional[BatteryConfig] = None,
    start_time: Optional[pd.Timestamp] = None,
    end_time: Optional[pd.Timestamp] = None,
    freq: Optional[str] = None,
    temperature_series: Optional[pd.Series] = None,
    *,
    initial_fec: float = 0.0,
    initial_calendar_seconds: float = 0.0,
    initial_resistance_growth: Optional[float] = None,
    initial_cumulative_cycle_deg: float = 0.0,
    initial_cumulative_cal_deg: float = 0.0,
    degradation_engine: str = "native",
    blast_model: Optional[str] = None,
    initial_degradation_state: Optional[Dict[str, Any]] = None,
    return_degradation_state: bool = False,
    initial_energy_wh: Optional[float] = None,
    initial_pv_origin_energy_wh: Optional[float] = None,
    initial_grid_origin_energy_wh: Optional[float] = None,
    dispatch_instructions: Optional[DispatchInstructions] = None,
    execution_backend: str = "python",
    aligned: Optional[AlignedSimulationInputs] = None,
    finalize_degradation: Optional[bool] = None,
    weights: Optional[Mapping[str, Tuple[str, np.ndarray]]] = None,
) -> SimulationSummary:
    """Simulate an energy balance and return annual totals and carry state.

    ``weights`` maps a name to a results column and one weight per step; the
    summary's ``weighted_sums`` holds ``sum(column * weight)`` for each, which
    is how a tariff prices a year without per-step frames.

    Runs exactly the physics :func:`simulate_energy_balance` runs, with the
    same arguments, and skips only the construction of the per-timestep
    results frame and the daily degradation frame. Multi-year callers such as
    Monte Carlo need the aggregates and the year-to-year seam, not the
    35,040 rows they were being reduced from.

    The ``finalize_degradation`` option controls whether the native rainflow
    residue is charged as terminal half cycles. It defaults to True; pass
    False together with ``return_degradation_state=True`` only when the
    returned state is fed into a later span that continues the same trace.

    Pass either ``pv_dc`` and ``houseload``, or a
    :class:`AlignedSimulationInputs` built by
    :func:`align_simulation_inputs` as ``aligned``. The second form skips
    building the calendar and reindexing onto it, which a caller running the
    same calendar many times has already done; ``freq``, ``start_time`` and
    ``end_time`` are then read from the aligned inputs, and
    ``temperature_series`` is ignored in favour of the aligned one. A
    ``freq`` that disagrees with the aligned inputs raises ``ValueError``.
    Without aligned inputs, ``freq`` defaults to ``"h"``.
    """
    freq = aligned.resolve_freq(freq) if aligned is not None else (freq or "h")
    core = _simulate_core(
        pv_dc=pv_dc,
        houseload=houseload,
        battery_config=battery_config,
        start_time=start_time,
        end_time=end_time,
        freq=freq,
        temperature_series=temperature_series,
        initial_fec=initial_fec,
        initial_calendar_seconds=initial_calendar_seconds,
        initial_resistance_growth=initial_resistance_growth,
        initial_cumulative_cycle_deg=initial_cumulative_cycle_deg,
        initial_cumulative_cal_deg=initial_cumulative_cal_deg,
        degradation_engine=degradation_engine,
        blast_model=blast_model,
        initial_degradation_state=initial_degradation_state,
        finalize_degradation=_resolve_finalize_degradation(finalize_degradation, return_degradation_state),
        initial_energy_wh=initial_energy_wh,
        initial_pv_origin_energy_wh=initial_pv_origin_energy_wh,
        initial_grid_origin_energy_wh=initial_grid_origin_energy_wh,
        dispatch_instructions=dispatch_instructions,
        execution_backend=execution_backend,
        # Reduced per-step buffers are safe here: nothing downstream of this
        # call materialises a per-timestep frame.
        summary_only=True,
        aligned=aligned,
    )
    return _build_simulation_summary(core, return_degradation_state=return_degradation_state, weights=weights)


def apply_indoor_temperature_model(
    outdoor_temperature: pd.Series,
    setpoint_c: float = DEFAULT_INDOOR_SETPOINT_C,
    coupling_alpha: float = DEFAULT_INDOOR_COUPLING_ALPHA,
    floor_c: float = DEFAULT_INDOOR_FLOOR_C,
    ceiling_c: float = DEFAULT_INDOOR_CEILING_C,
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
    elif model_lower in ("naumann_lam_field_calibrated", "naumann_lam_field_calibrated_v1"):
        return (
            NAUMANN_LAM_FIELD_CALIBRATED_K0_FRAC,
            NAUMANN_LAM_FIELD_CALIBRATED_EA_J_MOL,
            NAUMANN_LAM_FIELD_CALIBRATED_EXPONENT_B,
            NAUMANN_LAM_FIELD_CALIBRATED_SOC_EXPONENT_N,
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


# =========================================================================
# Resistance fade functions (Naumann 2020)
# =========================================================================


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
    delta_R = 0.0
    running_fec = fec_cum

    for cyc in cycles:
        DOC = max(0.0, min(1.0, cyc["doc"]))
        if DOC < min_DoD_fraction:
            continue

        dFEC = DOC * cyc["count"]
        mean_c_rate = cyc["mean_c_rate"]

        kC = max(0.0, A_R * mean_c_rate + B_R)
        kDOC = max(0.0, C_DOC_R * ((DOC - 0.6) ** 3) + D_DOC_R)

        fec_new = running_fec + dFEC

        # Differential form: dR% = kC * kDOC * (FEC_new^Z_R - FEC_old^Z_R)
        dR_percent = kC * kDOC * (fec_new**Z_R - running_fec**Z_R)
        dR_fraction = dR_percent / 100.0

        delta_R += dR_fraction
        running_fec = fec_new

    new_growth = resistance_growth + delta_R
    return new_growth, delta_R


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
    qloss_cycle_fraction = 0.0

    for cyc in cycles:
        DOC = max(0.0, min(1.0, cyc["doc"]))
        if DOC < min_DoD_fraction:
            continue

        mean_c_rate = cyc["mean_c_rate"]
        # Energy throughput for this cycle in FEC; a rainflow count is 1.0
        # for a full cycle and 0.5 for a half
        dFEC = DOC * cyc["count"]

        # Naumann-style k-factors with the LFP cycle aging coefficients
        kC = max(0.0, A_Q * mean_c_rate + B_Q)
        kDOC = max(0.0, C_DOC_Q * ((DOC - 0.6) ** 3) + D_DOC_Q)

        fec_new = fec_cum + dFEC

        # Differential form using cumulative FEC (Naumann Eq. 5-6)
        dq_percent = kC * kDOC * (fec_new**Z_Q - fec_cum**Z_Q)
        dq_fraction = dq_percent / 100.0

        qloss_cycle_fraction += dq_fraction
        fec_cum = fec_new

    soh_after = max(0.0, soh_start_fraction - qloss_cycle_fraction)
    return soh_after, qloss_cycle_fraction, fec_cum


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
    """
    if len(soc_series_absolute) < 2:
        return soh_start_fraction, 0.0, fec_cum

    time_ticks, ticks_per_second = _datetime_index_ticks(soc_series_absolute.index)
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

    return soh_after, d_soh_fraction, t_new
