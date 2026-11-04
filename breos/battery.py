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
import warnings
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from breos._controller import ControllerBatteryState, ControllerCarry, DailyDispatchController, _ControllerSession
from breos._dispatch import (  # noqa: F401  -- the dispatch step moved; its names stay importable here
    _N_ROWS,
    _ROW,
    _ROW_COLUMNS,
    LEDGER_SCHEMA_VERSION,
    _dispatch_day_python,
    compute_cell_temperature,
)
from breos.constants import (
    BATTERY_COUPLINGS,
    DEFAULT_CHARGE_EFFICIENCY,
    DEFAULT_DISCHARGE_EFFICIENCY,
    DEFAULT_EOL_PERCENTAGE,
    DEFAULT_MAX_SOC,
    DEFAULT_MIN_SOC,
    DEFAULT_STANDBY_LOSS_WH,
    DEFAULT_THERMAL_RESISTANCE_K_PER_W,
    SKIPPED_REPLACEMENT_ACTIONS,
)
from breos.degradation.native import (  # noqa: F401  -- the documented primitives stay importable here
    _get_degradation_params,
    _update_battery_soh_from_cycles,
    resistance_to_efficiency,
    update_battery_resistance_calendar,
    update_battery_resistance_cyclewise,
    update_battery_soh_calendar,
    update_battery_soh_cyclewise,
)
from breos.degradation.protocol import (
    BlastDegradationAdapter,
    DegradationDay,
    DegradationLifecycle,
    NativeDegradationAdapter,
)
from breos.degradation.replacement import replacement_skip_reason
from breos.dispatch_instructions import DispatchInstructions
from breos.execution import is_pv_only_dispatch, single_thread_blas, validate_execution_backend
from breos.inverter import (
    InverterConversion,
    InverterUnits,
    _calculate_dc_ac_power_arrays,
    as_inverter_conversion,
    convert_units,
)
from breos.utils import _datetime_index_ticks, get_hours_per_step, remap_datetime_index_years
from breos.vehicle import VehicleCarry, VehicleConfig, VehicleSchedule

if TYPE_CHECKING:
    from breos.vehicle.lifecycle import VehicleLifecycle

if TYPE_CHECKING:
    from breos.tariffs import ResolvedTariff


def _finite(name: str, value: Any) -> float:
    """``value`` as a float, refusing a bool, a non-number and a non-finite number by ``name``."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number, not a bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


@dataclass
class BatteryConfig:
    """
    Configuration parameters for battery simulation.

    ``coupling`` selects the topology (ADR 0004). ``"dc"``, the default, is a
    hybrid inverter: PV and the battery share one DC bus and one inverter, so
    PV charges the battery as DC and battery discharge crosses the shared
    inverter. ``"ac"`` gives the battery its own bidirectional inverter on the
    AC bus beside the PV inverter, described by
    ``battery_inverter_ac_capacity_w`` and ``battery_inverter_efficiency``;
    the inverter fields above then describe the PV inverter only. ``"dc"``
    refuses the battery inverter fields, and ``"ac"`` requires the
    efficiency. With ``"ac"`` PV converts as in a PV-only run, its surplus
    AC charges the battery through the battery inverter, and grid charge
    takes instructions without a ``grid_charge_efficiency``. Instruction
    planners and daily controllers run with it; with several PV inverters the
    App refuses the daily-target planner, which plans through one.

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
    ceiling respond to it. With ``"ac"`` it scales the PV inverter's
    conversion alone; the battery inverter's output and ceiling are
    unchanged. It is one constant approximating the combined
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

    ``skipped_replacement_action`` decides what happens to a pack whose
    end-of-life swap is skipped, by the minimum service time, the terminal
    guard or ``enable_replacement = False``. ``"keep"`` (the default) leaves
    it in service below its threshold until its health reaches zero, when it
    is retired. ``"retire"`` switches it off at the
    crossing, the instant a swap would have been booked at: the stored
    energy leaves the system as a replacement's does, and from the next step
    the battery neither charges nor discharges, so PV serves the load and
    the grid covers the rest as in a PV-only system. Dispatch instructions
    have no effect after it. The retired pack stays installed at zero
    charge, and the aging model still runs on it. A span that inherits a retired pack continues
    it retired from the returned degradation state, or takes
    ``battery_retired=True``.
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
    # What a skipped swap does with the old pack: "keep" it in service or
    # "retire" it and finish the span PV-only.
    skipped_replacement_action: str = "keep"
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
    # The inverter's measured efficiency curve and AC cable; None keeps the
    # PVWatts part-load curve and no cable. A mapping of its fields is accepted.
    inverter_conversion: Optional[InverterConversion] = None
    # Battery topology (ADR 0004): "dc" for a hybrid inverter, "ac" for a
    # battery inverter of its own on the AC bus.
    coupling: str = "dc"
    # The AC-coupled battery inverter: its AC rating (W), the AC output when
    # discharging and the AC input when charging, None for flat unclipped
    # conversion; and its nominal efficiency, which has no default. Both are
    # refused with "dc".
    battery_inverter_ac_capacity_w: Optional[float] = None
    battery_inverter_efficiency: Optional[float] = None

    def __post_init__(self):
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
        self.replacement_min_remaining_years = _finite(
            "replacement_min_remaining_years", self.replacement_min_remaining_years
        )
        if self.replacement_min_remaining_years < 0.0:
            raise ValueError("replacement_min_remaining_years must be non-negative")
        if self.skipped_replacement_action not in SKIPPED_REPLACEMENT_ACTIONS:
            raise ValueError(f"skipped_replacement_action must be one of: {', '.join(SKIPPED_REPLACEMENT_ACTIONS)}")

        self.nominal_energy_wh = _finite("nominal_energy_wh", self.nominal_energy_wh)
        self.initial_soh = _finite("initial_soh", self.initial_soh)
        self.eol_percentage = _finite("eol_percentage", self.eol_percentage)
        self.min_soc = _finite("min_soc", self.min_soc)
        self.max_soc = _finite("max_soc", self.max_soc)
        self.charge_efficiency = _finite("charge_efficiency", self.charge_efficiency)
        self.discharge_efficiency = _finite("discharge_efficiency", self.discharge_efficiency)
        self.inverter_efficiency = _finite("inverter_efficiency", self.inverter_efficiency)
        self.standby_loss_wh = _finite("standby_loss_wh", self.standby_loss_wh)
        self.initial_resistance_growth = _finite("initial_resistance_growth", self.initial_resistance_growth)
        self.thermal_resistance_k_per_w = _finite("thermal_resistance_k_per_w", self.thermal_resistance_k_per_w)

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

        if self.coupling not in BATTERY_COUPLINGS:
            raise ValueError(f"coupling must be one of: {', '.join(BATTERY_COUPLINGS)}; got {self.coupling!r}")
        if self.coupling == "dc":
            for name in ("battery_inverter_ac_capacity_w", "battery_inverter_efficiency"):
                if getattr(self, name) is not None:
                    raise ValueError(f"{name} describes an AC-coupled battery's own inverter; it needs coupling = 'ac'")
        else:
            if self.battery_inverter_efficiency is None:
                raise ValueError("an AC-coupled battery needs battery_inverter_efficiency; it has no default")
            self.battery_inverter_efficiency = _finite("battery_inverter_efficiency", self.battery_inverter_efficiency)
            if not 0.0 < self.battery_inverter_efficiency <= 1.0:
                raise ValueError("battery_inverter_efficiency must be greater than 0 and at most 1")

        for name in (
            "inverter_ac_capacity_w",
            "max_charge_power_w",
            "max_discharge_power_w",
            "battery_inverter_ac_capacity_w",
        ):
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
            rate = _finite("power_limit_c_rate", self.power_limit_c_rate)
            if rate <= 0.0:
                raise ValueError("power_limit_c_rate must be greater than 0")
            self.power_limit_c_rate = rate
        if self.inverter_conversion is not None:
            self.inverter_conversion = as_inverter_conversion(self.inverter_conversion)
        self.ac_output_scale = _finite("ac_output_scale", self.ac_output_scale)
        if not 0.0 < self.ac_output_scale <= 1.0:
            raise ValueError(
                "ac_output_scale must be greater than 0 and at most 1; it is applied after the "
                "inverter nameplate limit, so a value above 1 would deliver more AC than the "
                "nameplate and more AC than the DC entering the inverter. Scale the DC series "
                "instead to correct an under-predicting model"
            )

    @property
    def conversion(self) -> InverterConversion:
        """The inverter conversion the kernels run, the default when none is configured."""
        return as_inverter_conversion(self.inverter_conversion)

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


def _align_ac_derate(inverter_ac_derate: Optional[pd.Series], rng: pd.DatetimeIndex) -> Optional[np.ndarray]:
    """Reindex the inverter's per-step AC rating fraction onto ``rng``; None when there is none."""
    if inverter_ac_derate is None:
        return None
    values = inverter_ac_derate.reindex(rng)
    _require_complete_series(values, "inverter_ac_derate")
    array = values.values.astype(np.float64)
    if (array < 0.0).any() or (array > 1.0).any():
        raise ValueError("inverter_ac_derate must be a fraction in [0, 1] at every step")
    return array


def _align_curve_rows(inverter_curve_row: Optional[pd.Series], rng: pd.DatetimeIndex) -> Optional[np.ndarray]:
    """Reindex the efficiency curve's per-step row position onto ``rng``; None when there is none."""
    if inverter_curve_row is None:
        return None
    values = inverter_curve_row.reindex(rng)
    _require_complete_series(values, "inverter_curve_row")
    array = np.ascontiguousarray(values.values, dtype=np.float64)
    if not np.isfinite(array).all() or (array < 0.0).any():
        raise ValueError("inverter_curve_row must be a finite row position of at least 0 at every step")
    return array


def _align_pv_dc(pv_dc: pd.Series, rng: pd.DatetimeIndex) -> np.ndarray:
    """Reindex PV DC power onto ``rng`` as float64, refusing a step without a finite value."""
    pv_values = pv_dc.reindex(rng)
    _require_complete_series(pv_values, "pv_dc")
    return pv_values.values.astype(np.float64)


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
    pv_values = _align_pv_dc(pv_dc, rng)

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
        houseload_series.index = rng
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
        pv_values,
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
        inverter_ac_derate: The inverter's AC rating per step as a fraction
            of its nameplate, from temperature derating; None for none.
        inverter_curve_row: The efficiency curve's row position per step,
            from the operating DC voltage; None reads the conversion's own row.
    """

    index: pd.DatetimeIndex
    pv_dc_w: np.ndarray
    load_w: np.ndarray
    temperature_c: np.ndarray
    freq: str
    pv_chain: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    pv_chain_key: Optional[Tuple[Any, ...]] = None
    inverter_ac_derate: Optional[np.ndarray] = None
    inverter_curve_row: Optional[np.ndarray] = None

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
            inverter_ac_derate=self.inverter_ac_derate,
            inverter_curve_row=self.inverter_curve_row,
        )

    def with_pv_dc(self, pv_dc: pd.Series) -> "AlignedSimulationInputs":
        """Return the same calendar, load and temperature with ``pv_dc`` aligned onto it as the PV.

        ``pv_dc`` must start and end where this calendar does. The PV is then
        reindexed and checked exactly as :func:`align_simulation_inputs`
        does, so the result equals what that function builds from ``pv_dc``
        and this calendar's load and temperature, without aligning the load
        again. A design sweep on one weather year aligns its load once.
        """
        if len(pv_dc.index) == 0 or pv_dc.index[0] != self.index[0] or pv_dc.index[-1] != self.index[-1]:
            raise ValueError("pv_dc must start and end where the aligned calendar does")
        return AlignedSimulationInputs(
            index=self.index,
            pv_dc_w=_align_pv_dc(pv_dc, self.index),
            load_w=self.load_w,
            temperature_c=self.temperature_c,
            freq=self.freq,
            inverter_ac_derate=self.inverter_ac_derate,
            inverter_curve_row=self.inverter_curve_row,
        )

    def _pv_chain_cache_key(
        self,
        battery_config: "BatteryConfig",
        hours_per_step: float,
        *,
        pv_dc_w: Optional[np.ndarray] = None,
    ) -> Tuple[Any, ...]:
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
            battery_config.conversion.cache_key(),
            None if self.inverter_ac_derate is None else id(self.inverter_ac_derate),
            None if self.inverter_curve_row is None else id(self.inverter_curve_row),
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
            conversion=battery_config.conversion,
            ac_derate=self.inverter_ac_derate,
            curve_rows=self.inverter_curve_row,
            hours_per_step=hours_per_step,
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
            inverter_ac_derate=self.inverter_ac_derate,
            inverter_curve_row=self.inverter_curve_row,
        )


def align_simulation_inputs(
    pv_dc: pd.Series,
    houseload: pd.DataFrame,
    temperature_series: Optional[pd.Series] = None,
    *,
    freq: str = "h",
    start_time: Optional[pd.Timestamp] = None,
    end_time: Optional[pd.Timestamp] = None,
    inverter_ac_derate: Optional[pd.Series] = None,
    inverter_curve_row: Optional[pd.Series] = None,
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
        inverter_ac_derate=_align_ac_derate(inverter_ac_derate, rng),
        inverter_curve_row=_align_curve_rows(inverter_curve_row, rng),
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
    origin_wh = _finite(name, value)
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
        energy_wh = _finite("initial_energy_wh", initial_energy_wh)
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


@dataclass(frozen=True)
class _StartState:
    """The aging state a span starts from."""

    fec: float
    calendar_seconds: float
    cumulative_cycle_deg: float
    cumulative_cal_deg: float
    soh_fraction: float
    resistance_growth: float


def _resolve_start_state(
    engine_key: str,
    battery_config: BatteryConfig,
    initial_degradation_state: Optional[Dict[str, Any]],
    *,
    initial_fec: float,
    initial_calendar_seconds: float,
    initial_cumulative_cycle_deg: float,
    initial_cumulative_cal_deg: float,
    initial_resistance_growth: Optional[float],
) -> _StartState:
    """The span's starting aging state: a native snapshot's values, else the arguments and the battery's own.

    The arguments are the multi-year continuation seam (the App's year loop);
    a native snapshot takes precedence over them. ``initial_resistance_growth``
    None is the battery's configured starting resistance. BLAST restores its
    own state from the snapshot (:func:`_build_degradation_lifecycle`).
    """
    native = engine_key == "native" and initial_degradation_state is not None
    payload: Mapping[str, Any] = initial_degradation_state if native and initial_degradation_state else {}

    def carried(key: str, default: Any) -> Any:
        return float(payload.get(key, default)) if native else default

    if "resistance_growth" in payload:
        resistance_growth = float(payload["resistance_growth"])
    elif initial_resistance_growth is None:
        resistance_growth = battery_config.initial_resistance_growth
    else:
        resistance_growth = initial_resistance_growth
    return _StartState(
        fec=carried("fec_cum", initial_fec),
        calendar_seconds=carried("cumulative_calendar_seconds", initial_calendar_seconds),
        cumulative_cycle_deg=carried("cumulative_cycle_degradation", initial_cumulative_cycle_deg),
        cumulative_cal_deg=carried("cumulative_calendar_degradation", initial_cumulative_cal_deg),
        soh_fraction=carried("soh_fraction", battery_config.initial_soh / 100.0),
        resistance_growth=resistance_growth,
    )


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
    rainflow_kernel: Optional[Callable[..., int]] = None,
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
            initial_fec=float(initial_fec),
            initial_calendar_seconds=float(initial_calendar_seconds),
            **_native_degradation_kwargs(battery_config.calendar_model),
            cycle_step=_update_battery_soh_from_cycles,
            calendar_step=update_battery_soh_calendar,
            initial_rainflow_state=state_payload.get("native_rainflow_state"),
            rainflow_kernel=rainflow_kernel,
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

    __slots__ = ("matrix", "replaced", "replaced_capacity", "columns", "vehicle_events")

    def __init__(self, n_steps: int) -> None:
        # One row per per-step column, so a whole day of every output can be
        # handed to a compiled kernel as a single contiguous array. Each
        # column below is a view on its row, not a copy.
        self.vehicle_events: List[Dict[str, Any]] = []
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
        frame = pd.DataFrame({"Datetime": rng, **self.columns})
        if self.vehicle_events:
            frame.attrs["vehicle_end_of_life_events"] = list(self.vehicle_events)
        return frame


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
    "Inverter_Night_Consumption",
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


# What an end-of-life crossing did to the installed pack (ADR 0003 E11, E12):
# it was replaced, it stayed in service below its threshold, or it was
# switched off.
END_OF_LIFE_ACTIONS: Tuple[str, ...] = ("replaced", "kept", "retired")
# Why: the pack reached its threshold and a replacement was allowed; or the
# swap was skipped because less than the minimum service time remained, or
# because it fell in the span's final period with terminal replacement off,
# or because replacement is disabled; or a pack kept below its threshold
# reached zero health and was retired.
END_OF_LIFE_REASONS: Tuple[str, ...] = (
    "end_of_life",
    "min_remaining_years",
    "terminal_period",
    "replacement_disabled",
    "zero_health",
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
    replacements. A pack that stays in service below its threshold, or is
    retired, crosses once: later periods, and later spans that inherit it,
    record nothing more, except that a kept pack whose health reaches zero
    is retired then, an event with the reason ``"zero_health"``. A pack continued from an earlier span (through
    ``initial_degradation_state``) at or below its threshold has already
    crossed there, unless this span replaces it. A fresh pack that starts at
    or below its threshold crosses at its first period close.
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


def retiring_step(events: Sequence[EndOfLifeEvent]) -> Optional[int]:
    """The step at whose end a span's events retire the pack, or None.

    A retired pack crosses no more, so a span holds at most one retirement.
    """
    return next((event.step for event in events if event.action == "retired"), None)


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
    # True once the pack is switched off (BatteryConfig.skipped_replacement_action).
    retired: bool = False


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


def _settle_terminal_cycles(aging: _AgingState, lifecycle: DegradationLifecycle) -> float:
    """Charge a pack leaving service with its unresolved terminal half cycles.

    A replaced or retired native pack owns them. Settling them keeps its
    lifetime FEC complete, and leaves a clean rainflow residue for a new
    pack or for a retired pack that no longer cycles. Returns the cycle
    degradation they added.
    """
    terminal_cycles = lifecycle.finalize_cycles()
    aging.fec_lifetime += terminal_cycles.fec - aging.fec_cum
    aging.fec_cum = terminal_cycles.fec
    aging.soh_fraction = terminal_cycles.soh_fraction
    aging.soh_percent = aging.soh_fraction * 100.0
    aging.cumulative_cycle_deg += terminal_cycles.cycle_degradation
    return terminal_cycles.cycle_degradation


def _apply_battery_retirement(
    aging: _AgingState,
    out: _ResultBuffers,
    *,
    step_index: int,
    hours_per_step: float,
    battery_energy_wh: float,
    pv_origin_energy_wh: float,
    grid_origin_energy_wh: float,
    battery_energy_beginning: float,
) -> Tuple[float, float, float, float]:
    """Switch the pack off, returning ``(energy, pv_origin, grid_origin, day_end_soc)``.

    Like a replacement it happens *inside* the closing timestep, after that
    step was dispatched. The stored energy leaves the system as a replaced
    pack's does, booked as ``Battery_Replacement_Energy_Removed`` with its
    origins, and nothing is added, so the ledger closes and every later step
    starts empty. The pack stays installed and keeps its health.
    """
    aging.retired = True
    out.columns["Battery_Energy"][step_index] = 0.0
    out.columns["Battery_SOC_Normalized"][step_index] = 0.0
    out.columns["Battery_SOC_Absolute"][step_index] = 0.0
    out.columns["Battery_PV_Origin_Energy_End"][step_index] = 0.0
    out.columns["Battery_Grid_Origin_Energy_End"][step_index] = 0.0
    out.columns["Battery_Replacement_Energy_Removed"][step_index] = battery_energy_wh / hours_per_step
    out.columns["PV_Origin_Replacement_Energy_Removed"][step_index] = pv_origin_energy_wh / hours_per_step
    out.columns["Grid_Origin_Replacement_Energy_Removed"][step_index] = grid_origin_energy_wh / hours_per_step
    out.columns["Battery_Energy_Delta"][step_index] = (0.0 - battery_energy_beginning) / hours_per_step
    return 0.0, 0.0, 0.0, 0.0


def _pv_inverter_conversion(
    pv_dc_wh: np.ndarray,
    *,
    battery_config: BatteryConfig,
    hours_per_step: float,
    cap_wh: float,
    pv_chain: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    ac_derate: Optional[np.ndarray] = None,
    curve_rows: Optional[np.ndarray] = None,
    units: Optional[InverterUnits] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """The PV inverters' conversion of ``pv_dc_wh`` on its own: ``(ac, conversion_loss, clipping, night)``.

    Wh per step, as a PV-only run converts it. ``night`` is the inverters'
    draw on the steps they convert nothing, None when there is none. A
    PV-only run and an AC-coupled battery's PV inverter (ADR 0004 C3) both
    take their conversion from here, so they agree bit for bit. ``ac_derate``
    and ``curve_rows`` are aligned to ``pv_dc_wh``; ``units`` and a memoized
    ``pv_chain`` cover a whole span.
    """
    if units is not None:
        assert pv_chain is None
        converted = convert_units(
            pv_dc_wh,
            units,
            inverter_efficiency=battery_config.inverter_efficiency,
            ac_output_scale=battery_config.ac_output_scale,
            hours_per_step=hours_per_step,
        )
        # With several inverters, each one draws on the steps it receives no DC.
        night = converted.night if converted.night.any() else None
        return converted.ac, converted.conversion_loss, converted.clipping, night
    if pv_chain is None:
        ac_wh, conversion_loss_wh, clipping_loss_dc_wh = _calculate_dc_ac_power_arrays(
            pv_dc_wh,
            cap_wh,
            battery_config.inverter_efficiency,
            battery_config.ac_output_scale,
            conversion=battery_config.conversion,
            ac_derate=ac_derate,
            curve_rows=curve_rows,
            hours_per_step=hours_per_step,
        )
    else:
        # Memoized by the caller for these exact inputs. Same values, same
        # expressions downstream; see AlignedSimulationInputs.with_pv_only_chain.
        ac_wh, conversion_loss_wh, clipping_loss_dc_wh = pv_chain
    # The inverter's draw while it converts nothing, from the grid.
    night_wh = battery_config.conversion.night_consumption_w * hours_per_step
    night = np.where(pv_dc_wh <= 0.0, night_wh, 0.0) if night_wh > 0.0 else None
    return ac_wh, conversion_loss_wh, clipping_loss_dc_wh, night


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
    steps: slice = slice(None),
    ac_derate: Optional[np.ndarray] = None,
    curve_rows: Optional[np.ndarray] = None,
    units: Optional[InverterUnits] = None,
    span_chain: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
) -> None:
    """Fill a PV-only result buffer without entering the timestep loop.

    ``steps`` fills only those steps of a full buffer, which is how a retired
    battery's steps are dispatched: element by element the same expressions,
    so each step equals the PV-only system's. A memoized ``pv_chain`` covers
    the whole span only. ``units``, aligned to the span, converts each
    inverter's share of the DC on its own instead of ``battery_config``'s
    single conversion (:func:`~breos.inverter.convert_units`).
    ``span_chain`` is an AC-coupled battery's PV inverter conversion over the
    whole span, ``(ac, conversion_loss, clipping, night)`` in Wh, which its
    retired steps read instead of converting again (ADR 0004 C12).
    """
    if steps == slice(None):
        out.zero_fill()
    else:
        assert isinstance(out, _ResultBuffers) and pv_chain is None
        out.matrix[:, steps] = 0.0

    assert units is None or steps == slice(None)
    pv_dc_wh = np.maximum(0.0, pv_dc_values[steps] * hours_per_step)
    load_wh = load_values[steps] * hours_per_step
    night_consumption_wh: Optional[np.ndarray]
    if span_chain is not None:
        assert pv_chain is None and units is None
        ac_wh, conversion_loss_wh, clipping_loss_dc_wh, night_consumption_wh = (part[steps] for part in span_chain)
        if not night_consumption_wh.any():
            night_consumption_wh = None
    else:
        ac_wh, conversion_loss_wh, clipping_loss_dc_wh, night_consumption_wh = _pv_inverter_conversion(
            pv_dc_wh,
            battery_config=battery_config,
            hours_per_step=hours_per_step,
            cap_wh=cap_wh,
            pv_chain=pv_chain,
            ac_derate=None if ac_derate is None else ac_derate[steps],
            curve_rows=None if curve_rows is None else curve_rows[steps],
            units=units,
        )
    pv_ac_to_load_wh = np.minimum(ac_wh, load_wh)
    grid_export_wh = ac_wh - pv_ac_to_load_wh
    grid_import_wh = np.maximum(0.0, load_wh - pv_ac_to_load_wh)
    if night_consumption_wh is not None:
        # Idle units share the bus: their draw precedes export. The battery
        # never supplies it (ADR 0004 M9).
        pv_to_night_wh = np.minimum(grid_export_wh, night_consumption_wh)
        grid_export_wh = grid_export_wh - pv_to_night_wh
        grid_import_wh = grid_import_wh + (night_consumption_wh - pv_to_night_wh)
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

    out.columns["PV_DC"][steps] = pv_dc_wh / hours_per_step
    out.columns["PV_Production"][steps] = pv_production_wh / hours_per_step
    out.columns["Houseload"][steps] = load_wh / hours_per_step
    out.columns["PV_Delta"][steps] = (pv_production_wh - load_wh) / hours_per_step
    out.columns["Import_From_Grid"][steps] = grid_import_wh / hours_per_step
    out.columns["Battery_SOH"][steps] = 100.0
    out.columns["T_cell"][steps] = temperature_values[steps]

    out.columns["PV_DC_To_Inverter"][steps] = (pv_dc_wh - clipping_loss_dc_wh) / hours_per_step
    out.columns["PV_DC_Curtailed"][steps] = curtailment_w
    out.columns["PV_AC_To_Load"][steps] = pv_ac_to_load_wh / hours_per_step
    out.columns["PV_AC_Export"][steps] = grid_export_w
    out.columns["PV_Direct_Inverter_Loss"][steps] = conversion_w
    out.columns["Inverter_Night_Consumption"][steps] = (
        0.0 if night_consumption_wh is None else night_consumption_wh / hours_per_step
    )
    out.columns["Inverter_Loss"][steps] = conversion_w


def _mean(values: np.ndarray) -> float:
    """``float(np.mean(values))``, without np.mean's wrapper for float64.

    For float64, np.mean is add.reduce divided by the count, so this returns
    the same bits; any other dtype goes to np.mean, which may accumulate in
    another precision.
    """
    if values.dtype == np.float64 and values.size:
        return float(np.add.reduce(values)) / values.size
    return float(np.mean(values))


def _apply_daily_degradation(
    aging: _AgingState,
    lifecycle: DegradationLifecycle,
    battery_config: BatteryConfig,
    out: _ResultBuffers,
    degradation_tracking: List[Dict[str, Any]],
    *,
    step_index: int,
    step_times: pd.DatetimeIndex,
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
    keep_record: bool = True,
) -> Tuple[float, float, float]:
    """Close out one degradation period, returning ``(energy, pv_origin, grid_origin)``.

    Runs the lifecycle step, optional resistance fade and the end-of-life
    replacement check in that order, mutating *aging* in place and appending
    one row to *degradation_tracking*, unless ``keep_record`` is false (a
    summary run, which builds no degradation frame). The stored-energy values are
    returned rather than carried on *aging* because the per-step loop owns
    them and only a replacement changes them here.

    The period's SOC and cell-temperature endpoints become the next starting
    boundary; a replacement moves that endpoint to the fresh pack's
    max SOC, since the recorded state was rewritten to match.

    ``replacement_skipped_by`` names the rule that skips this period's
    end-of-life swap, one of the skip reasons of :data:`END_OF_LIFE_REASONS`
    (never ``"zero_health"``): the span's final
    period when the battery does not allow a terminal replacement, or any
    period that closes with less than the battery's minimum service time
    left. None allows the swap. The period still ages the pack and is still
    recorded; only the swap is skipped. A crossing is recorded on
    ``aging.end_of_life_events``, with what was done and why. A skipped swap
    with ``skipped_replacement_action = "retire"`` switches the pack off at
    the period's close, and so does a kept pack's period that closes at zero
    health; a retired pack is never replaced.
    """
    period_steps = len(soc_absolute_day)
    period_seconds = period_steps * hours_per_step * 3600.0
    dt_days = period_seconds / 86400.0
    day_end_soc_absolute = float(soc_absolute_day[-1])
    day_end_t_cell = float(t_cell_day[-1])

    # np.mean is add.reduce over the size; doing it directly skips ~4 us of
    # wrapper per call and returns the same bits.
    mean_soc_abs = _mean(soc_absolute_day)
    mean_t_cell = _mean(t_cell_day)
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
            soc_mean=mean_soc_abs,
            temperature_mean_c=mean_t_cell,
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
    if (
        reached_end_of_life
        and replacement_skipped_by is None
        and battery_config.enable_replacement
        and not aging.retired
    ):
        aging.end_of_life_events.append(
            EndOfLifeEvent(step_index, step_times[step_index], "replaced", "end_of_life", aging.soh_percent)
        )
        cycle_degradation_for_row = degradation_step.cycle_degradation + _settle_terminal_cycles(aging, lifecycle)
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
        retire = False
        if reached_end_of_life and not aging.past_end_of_life:
            reason = "replacement_disabled" if not battery_config.enable_replacement else replacement_skipped_by
            assert reason is not None
            retire = battery_config.skipped_replacement_action == "retire"
            aging.end_of_life_events.append(
                EndOfLifeEvent(
                    step_index, step_times[step_index], "retired" if retire else "kept", reason, aging.soh_percent
                )
            )
            aging.past_end_of_life = True
        if not retire and not aging.retired and aging.soh_fraction <= 0.0:
            # A kept pack at zero health has no capacity left to serve, so it
            # is retired as a skipped swap's "retire" retires one.
            retire = True
            aging.end_of_life_events.append(
                EndOfLifeEvent(step_index, step_times[step_index], "retired", "zero_health", aging.soh_percent)
            )
        if retire:
            # The pack's last cycles end here: the energy it holds is
            # written off, not discharged, and it never cycles again.
            cycle_degradation_for_row += _settle_terminal_cycles(aging, lifecycle)
            battery_energy_wh, pv_origin_energy_wh, grid_origin_energy_wh, day_end_soc_absolute = (
                _apply_battery_retirement(
                    aging,
                    out,
                    step_index=step_index,
                    hours_per_step=hours_per_step,
                    battery_energy_wh=battery_energy_wh,
                    pv_origin_energy_wh=pv_origin_energy_wh,
                    grid_origin_energy_wh=grid_origin_energy_wh,
                    battery_energy_beginning=battery_energy_beginning,
                )
            )

    if not keep_record:
        aging.day_start_soc = day_end_soc_absolute
        aging.day_start_t_cell = day_end_t_cell
        return battery_energy_wh, pv_origin_energy_wh, grid_origin_energy_wh
    degradation_record = {
        "Datetime": step_times[step_index],
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
    return replacement_skip_reason(
        allow_terminal_replacement=battery_config.allow_terminal_replacement,
        final_period=final_period,
        remaining_steps=remaining_steps,
        min_remaining_steps=min_remaining_steps,
    )


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
        # Only a retired pack carries the flag, so other states are unchanged.
        **({"battery_retired": True} if aging.retired else {}),
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


def _resolve_rainflow_kernel(execution_backend: str) -> Optional[Callable[..., int]]:
    """The compiled native rainflow day scan for the numba backend; None (numpy and Python) otherwise."""
    validate_execution_backend(execution_backend)
    if execution_backend == "python":
        return None
    from breos._numba_dispatch import require_numba_rainflow_day

    return require_numba_rainflow_day()


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
    # The degradation periods closed; a summary run keeps no tracking rows.
    degradation_periods: int = 0
    # The daily controller's carry; None without a controller.
    controller_carry: Optional[ControllerCarry] = None
    # A copy of the instructions a daily controller's decisions dispatched,
    # one per simulated step; None without a controller.
    controller_instructions: Optional[DispatchInstructions] = None
    vehicle_carry: Optional[VehicleCarry] = None
    vehicle_end_of_life_events: Tuple[EndOfLifeEvent, ...] = ()
    vehicle_degradation_tracking: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SimulationSummary:
    """Annual totals and carry state, without a per-timestep frame.

    ``column_sums`` holds the plain sum of every column the detailed results
    frame exposes, under the same names and in the same units:
    ``column_sums[col]`` is the value ``results_df[col].sum()`` gives. Unit scaling stays with the caller, because
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
    # ``T_cell`` summed over the steps the pack served, through the one that
    # retired it, when the span retired it; None otherwise.
    in_service_t_cell_sum: Optional[float] = None
    # The ledger schema the column sums follow; App and Monte Carlo report
    # the same version.
    ledger_schema_version: str = LEDGER_SCHEMA_VERSION
    vehicle_carry: Optional[VehicleCarry] = None
    vehicle_end_of_life_events: Tuple[EndOfLifeEvent, ...] = ()
    vehicle_degradation_tracking: List[Dict[str, Any]] = field(default_factory=list)


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
    retiring = retiring_step(aging.end_of_life_events)

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
        has_degradation_rows=core.degradation_periods > 0,
        fec_cum=aging.fec_cum,
        cumulative_calendar_seconds=aging.cumulative_cal_seconds,
        cumulative_cycle_degradation=aging.cumulative_cycle_deg,
        cumulative_calendar_degradation=aging.cumulative_cal_deg,
        resistance_growth=aging.resistance_growth,
        replacement_steps=tuple(int(i) for i in np.flatnonzero(buffers.replaced)),
        fec_all_packs=aging.fec_lifetime,
        final_degradation_state=final_state,
        vehicle_carry=core.vehicle_carry,
        vehicle_end_of_life_events=core.vehicle_end_of_life_events,
        vehicle_degradation_tracking=core.vehicle_degradation_tracking,
        end_of_life_events=tuple(aging.end_of_life_events),
        weighted_sums=weighted_column_sums(columns, weights),
        in_service_t_cell_sum=float(np.sum(columns["T_cell"][: retiring + 1])) if retiring is not None else None,
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
    battery_retired: bool = False,
    inverter_ac_derate: Optional[pd.Series] = None,
    inverter_curve_row: Optional[pd.Series] = None,
    inverter_units: Optional[InverterUnits] = None,
    vehicle_config: Optional[VehicleConfig] = None,
    vehicle_schedule: Optional[VehicleSchedule] = None,
    vehicle_tariff: Optional["ResolvedTariff"] = None,
    grid_import_limit_w: float = math.inf,
    initial_vehicle_energy_wh: Optional[float] = None,
    initial_vehicle_pv_origin_energy_wh: Optional[float] = None,
    initial_vehicle_grid_origin_energy_wh: Optional[float] = None,
    vehicle_carry: Optional[VehicleCarry] = None,
    vehicle_temperature_series: Optional[pd.Series] = None,
    vehicle_allow_terminal_replacement: Optional[bool] = None,
    vehicle_replacement_years_after_span: int = 0,
) -> "_CoreRun":
    """
    Simulate energy balance with battery storage and degradation.

    This function processes PV DC production and load profiles to calculate
    grid interaction, battery state, and degradation, for a DC-coupled
    battery behind a hybrid inverter or an AC-coupled one behind its own
    inverter (``battery_config.coupling``, ADR 0004).

    Energy flow for DC-coupled (hybrid inverter) systems:
    - PV -> Load: DC -> Inverter -> AC (one inverter loss)
    - PV -> Battery: DC -> Battery (charge efficiency only)
    - Battery -> Load: DC -> Inverter -> AC (discharge efficiency + inverter loss)
    - Grid -> Load: AC (no conversion)

    For AC-coupled systems PV converts in the PV inverter as in a PV-only
    run, and the battery inverter converts both ways:
    - PV -> Battery: DC -> PV inverter -> AC -> battery inverter -> DC
    - Battery -> Load: DC -> battery inverter -> AC
    - Grid -> Battery: AC -> battery inverter -> DC

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
        inverter_ac_derate: Optional inverter AC rating per step as a fraction
            of ``battery_config.inverter_ac_capacity_w``, from temperature
            derating. It must cover every step. None is the full rating.
        inverter_curve_row: Optional position among the inverter efficiency
            curve's voltage rows per step, from the operating DC voltage. It
            must cover every step. None reads the conversion's own row.
        inverter_units: Several inverters, each converting its own share of
            the DC (:class:`~breos.inverter.InverterUnits`), for a run without
            a battery or with an AC-coupled one; a DC-coupled battery is
            refused. They replace ``battery_config``'s single conversion and
            night draw, and must cover every step. None converts through
            ``battery_config`` alone.
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
        battery_retired: The span inherits a battery a previous span retired
            (``BatteryConfig.skipped_replacement_action = "retire"``). It
            holds no energy, every step dispatches PV-only, and the pack
            only ages. A carried stored energy must be zero. A retired span's
            returned degradation state holds ``"battery_retired": True``, so
            passing it as ``initial_degradation_state`` continues it retired
            without this argument; True with a state from a pack that was not
            retired is refused.

        vehicle_config, vehicle_schedule: Lower-level vehicle dispatch and
            its resolved calendar. App remains gated until PR 4. Replacement
            follows its own lifecycle, with same-SOC pack replacement.
        vehicle_tariff: Resolved tariff for vehicle charge/discharge periods.
        grid_import_limit_w: Site limit from [smart_charging], also usable
            without stationary instructions. A vehicle uses the smaller of
            this and an existing instruction limit, without modifying them.
        initial_vehicle_energy_wh, initial_vehicle_pv_origin_energy_wh,
            initial_vehicle_grid_origin_energy_wh: On-board starting energy
            and attributed shares in Wh. Default: departure target, unattributed.
        vehicle_carry: Previous span's energy, origins and last plug state.
            Pass instead of initial_vehicle values. Without carry the first
            step starts where its schedule says, with no synthetic boundary.
        vehicle_temperature_series: Independently prepared vehicle cell/ambient
            temperatures, without an indoor model. Otherwise weather.build_battery_temperature_series
            resolves vehicle.battery_temperature (weather uses the input ambient
            series, default 25 C if none; fixed and CSV use their own values).
        vehicle_allow_terminal_replacement, vehicle_replacement_years_after_span:
            The vehicle's horizon policy, as for BatteryConfig. project_years
            allows earlier year-end swaps and supplies remaining project years.

    Returns:
        A :class:`_CoreRun` holding the filled result buffers, the calendar,
        the aging state and the lifecycle; the public entry points shape
        their results from it. With a ``day_controller`` it also holds the
        controller carry and a copy of the instructions the span dispatched.
    """
    if (vehicle_config is None) != (vehicle_schedule is None):
        raise ValueError("pass vehicle_config and its resolved vehicle_schedule together")
    if vehicle_carry is not None:
        if vehicle_config is None:
            raise ValueError("vehicle_carry needs a vehicle_config and resolved vehicle_schedule")
        if any(
            value is not None
            for value in (
                initial_vehicle_energy_wh,
                initial_vehicle_pv_origin_energy_wh,
                initial_vehicle_grid_origin_energy_wh,
            )
        ):
            raise ValueError("pass vehicle_carry or initial_vehicle energy/origin values, not both")
        if not isinstance(vehicle_carry.plugged, (bool, np.bool_)):
            raise ValueError("vehicle_carry.plugged must be a bool")
        initial_vehicle_energy_wh = vehicle_carry.energy_wh
        initial_vehicle_pv_origin_energy_wh = vehicle_carry.pv_origin_energy_wh
        initial_vehicle_grid_origin_energy_wh = vehicle_carry.grid_origin_energy_wh
    if vehicle_allow_terminal_replacement is None:
        vehicle_allow_terminal_replacement = (
            True if vehicle_config is None else vehicle_config.allow_terminal_replacement
        )
    if not isinstance(vehicle_allow_terminal_replacement, (bool, np.bool_)):
        raise ValueError("vehicle_allow_terminal_replacement must be a bool")
    if (
        isinstance(vehicle_replacement_years_after_span, (bool, np.bool_))
        or not isinstance(vehicle_replacement_years_after_span, (int, np.integer))
        or vehicle_replacement_years_after_span < 0
    ):
        raise ValueError("vehicle_replacement_years_after_span must be a non-negative integer")
    if math.isnan(grid_import_limit_w) or grid_import_limit_w <= 0.0:
        raise ValueError("grid_import_limit_w must be positive, or math.inf")
    if vehicle_config is not None and (day_controller is not None or instruction_planner is not None):
        raise ValueError("daily_persistence controllers and planners are not supported with a vehicle")
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
        _derate_vals = _align_ac_derate(inverter_ac_derate, rng)
        _curve_rows = _align_curve_rows(inverter_curve_row, rng)
        _units = None if inverter_units is None else inverter_units.aligned(rng)
    else:
        # The caller has already built the calendar and reindexed onto it.
        if inverter_ac_derate is not None or inverter_curve_row is not None:
            raise ValueError(
                "pass inverter_ac_derate and inverter_curve_row to align_simulation_inputs with aligned inputs"
            )
        if inverter_units is not None:
            raise ValueError("inverter_units need the per-step series, not aligned inputs")
        _units = None
        rng = aligned.index
        _pv_dc_vals = aligned.pv_dc_w
        _load_vals = aligned.load_w
        _temp_vals = aligned.temperature_c
        _derate_vals = aligned.inverter_ac_derate
        _curve_rows = aligned.inverter_curve_row

    degradation_engine_key = _resolve_degradation_engine(
        degradation_engine,
        blast_model,
        initial_degradation_state,
        battery_config,
    )

    state_payload = initial_degradation_state or {}
    start = _resolve_start_state(
        degradation_engine_key,
        battery_config,
        initial_degradation_state,
        initial_fec=initial_fec,
        initial_calendar_seconds=initial_calendar_seconds,
        initial_cumulative_cycle_deg=initial_cumulative_cycle_deg,
        initial_cumulative_cal_deg=initial_cumulative_cal_deg,
        initial_resistance_growth=initial_resistance_growth,
    )
    battery_soh_decimal = start.soh_fraction
    resistance_growth = start.resistance_growth
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
    degradation_periods = 0

    n_steps = len(rng)

    # Hoist invariant check out of the loop
    has_battery = not is_pv_only_dispatch(
        battery_config.nominal_energy_wh,
        battery_config.max_soc,
        battery_config.min_soc,
    )
    if has_battery and _units is not None and battery_config.coupling != "ac":
        raise ValueError(
            "a battery beside several inverters must be AC-coupled (coupling = 'ac'); a DC-coupled battery "
            "sits behind one hybrid inverter"
        )
    if not has_battery and aligned is not None and aligned.pv_chain is not None:
        expected_chain_key = aligned._pv_chain_cache_key(battery_config, hours_per_step)
        if aligned.pv_chain_key != expected_chain_key:
            raise ValueError("memoized PV chain does not match the aligned PV inputs or inverter settings")
    # The vectorized PV-only dispatch below is the only producer that leaves
    # most columns at zero, so it is the only one whose output can be served
    # from the reduced buffer -- and only when the caller wants a summary,
    # since the detailed frame must own a writable array per column.
    pv_only_summary = summary_only and not has_battery and vehicle_config is None

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
        initial_fec=start.fec,
        initial_calendar_seconds=start.calendar_seconds,
        default_day_start_soc=degradation_day_start_soc,
        default_day_start_t_cell=degradation_day_start_t_cell,
        rainflow_kernel=_resolve_rainflow_kernel(execution_backend),
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

    # A state from a span that retired the pack continues it retired.
    state_retired = bool(state_payload.get("battery_retired", False))
    if battery_retired and initial_degradation_state is not None and not state_retired:
        raise ValueError(
            "battery_retired=True, but initial_degradation_state comes from a battery that was not retired"
        )
    battery_retired = battery_retired or state_retired
    if battery_retired:
        if not has_battery:
            raise ValueError("battery_retired needs a battery; a PV-only run has none to retire")
        if initial_energy_wh not in (None, 0.0):
            raise ValueError("a retired battery holds no energy; initial_energy_wh must be 0 or None")
        Battery_Energy_Wh = Battery_PV_Origin_Energy_Wh = Battery_Grid_Origin_Energy_Wh = 0.0

    # Health state advanced only at daily boundaries. The loop keeps hot
    # copies of the four fields it reads every step (SOH and the two
    # efficiencies) and refreshes them whenever a day closes.
    aging = _AgingState(
        soh_fraction=battery_soh_decimal,
        soh_percent=Battery_SOH,
        fec_cum=start.fec,
        # Lifetime FEC starts this span at zero whatever the installed pack
        # already carries, so the end-of-span value is the FEC the span itself
        # accumulated across every pack it used.
        fec_lifetime=0.0,
        cumulative_cal_seconds=start.calendar_seconds,
        # Coerced so a NumPy scalar argument, such as float32, is summed in
        # float64 like the carried state always was.
        cumulative_cycle_deg=float(start.cumulative_cycle_deg),
        cumulative_cal_deg=float(start.cumulative_cal_deg),
        resistance_growth=resistance_growth,
        eff_charge=eff_charge,
        eff_discharge=eff_discharge,
        n_replacements=0,
        replaced_capacity_wh=0.0,
        day_start_soc=0.0 if battery_retired else degradation_day_start_soc,
        day_start_t_cell=degradation_day_start_t_cell,
        # A pack continued at or below its threshold crossed it in an earlier
        # span, which recorded the crossing. A fresh pack that starts there
        # crosses at its first period close.
        past_end_of_life=battery_retired
        or (initial_degradation_state is not None and battery_soh_decimal <= battery_config.eol_percentage),
        retired=battery_retired,
    )

    # Per-step energy caps (Wh) for the shared inverter AC nameplate and the
    # battery's own charge/discharge power limits.
    standby_loss_per_step_wh = battery_config.standby_loss_wh * hours_per_step
    cap_wh = _step_energy_cap(battery_config.inverter_ac_capacity_w, hours_per_step)
    cap_charge_wh = _step_energy_cap(battery_config.max_charge_power_w, hours_per_step)
    cap_discharge_wh = _step_energy_cap(battery_config.max_discharge_power_w, hours_per_step)
    cap_stored_wh = _step_energy_cap(battery_config.stored_power_limit_w, hours_per_step)

    dispatch_day = _resolve_dispatch_day(execution_backend)
    ac_coupled = has_battery and battery_config.coupling == "ac"

    def battery_state() -> ControllerBatteryState:
        # The state the next step dispatches from, as a planner or a daily
        # controller decides on it.
        return ControllerBatteryState(
            energy_wh=Battery_Energy_Wh,
            pv_origin_energy_wh=Battery_PV_Origin_Energy_Wh,
            grid_origin_energy_wh=Battery_Grid_Origin_Energy_Wh,
            soh_fraction=battery_soh_decimal,
            resistance_growth=aging.resistance_growth,
            charge_efficiency=eff_charge,
            discharge_efficiency=eff_discharge,
        )

    if instruction_planner is not None:
        if dispatch_instructions is not None or day_controller is not None:
            raise ValueError("pass one of dispatch_instructions, an instruction planner or a daily controller")
        if not has_battery:
            raise ValueError("an instruction planner needs a battery; a PV-only run has nothing to dispatch")
        # Planned from the state the first step dispatches from, the one a
        # daily controller's first decision would see.
        dispatch_instructions = instruction_planner(battery_state())
        if not isinstance(dispatch_instructions, DispatchInstructions):
            raise TypeError("an instruction planner must return DispatchInstructions")
    if dispatch_instructions is not None and len(dispatch_instructions) != n_steps:
        raise ValueError(
            f"dispatch_instructions cover {len(dispatch_instructions)} steps; the simulation has {n_steps}"
        )
    if (
        battery_config.coupling == "dc"
        and dispatch_instructions is not None
        and dispatch_instructions.grid_charge_efficiency is None
    ):
        # Checked here, before a PV-only or retired run skips the day loop.
        raise ValueError(
            "a DC-coupled battery needs a grid_charge_efficiency in its dispatch instructions; "
            "only an AC-coupled battery converts grid charge in its own inverter"
        )
    if (
        ac_coupled
        and dispatch_instructions is not None
        and dispatch_instructions.grid_charge_efficiency is not None
        and not np.isnan(dispatch_instructions.grid_target_fraction).all()
    ):
        raise ValueError(
            "an AC-coupled battery grid-charges through its own inverter; its dispatch instructions "
            "take grid_charge_efficiency = None"
        )
    # No instructions is greedy dispatch; one no-op set serves every day.
    instructions = dispatch_instructions if dispatch_instructions is not None else DispatchInstructions.noop(n_steps)
    # Vehicle ageing has its own lifecycle and raw temperature input; no
    # stationary capacity-temperature factor, standby or resistance fade.
    vehicle_state: Dict[str, Any] = {}
    vehicle_lifecycle: Optional["VehicleLifecycle"] = None
    vehicle_temp = np.empty(0)
    vehicle_ticks = np.empty(0, dtype=np.int64)
    vehicle_ticks_per_second = 1.0
    if vehicle_config is not None:
        assert vehicle_schedule is not None
        if not vehicle_schedule.index.equals(rng) or vehicle_schedule.hours_per_step != hours_per_step:
            raise ValueError("vehicle_schedule must match the simulation index and hours_per_step")
        from breos.vehicle.lifecycle import VehicleLifecycle
        from breos.weather import build_battery_temperature_series

        vehicle_temp = build_battery_temperature_series(
            vehicle_config.battery_temperature if vehicle_temperature_series is None else "weather",
            rng,
            weather_df=pd.DataFrame({"temp_air": _temp_vals}, index=rng)
            if vehicle_temperature_series is None
            else pd.DataFrame({"temp_air": vehicle_temperature_series}),
            indoor_model={"enabled": False},
        ).to_numpy(dtype=float)
        vehicle_ticks, vehicle_ticks_per_second = _datetime_index_ticks(rng)
        initial_vehicle_soh = 1.0 if vehicle_carry is None else vehicle_carry.soh_fraction
        if not math.isfinite(initial_vehicle_soh) or not 0.0 <= initial_vehicle_soh <= 1.0:
            raise ValueError("vehicle carried health must be finite and between 0 and 1")
        retired_v = False if vehicle_carry is None else vehicle_carry.retired
        capacity_v = vehicle_config.battery_kwh * 1000.0 * initial_vehicle_soh
        energy_v = (
            vehicle_config.departure_soc * capacity_v
            if initial_vehicle_energy_wh is None
            else float(initial_vehicle_energy_wh)
        )
        if (
            not math.isfinite(energy_v)
            or not (0.0 if retired_v else vehicle_config.min_soc * capacity_v)
            <= energy_v
            <= vehicle_config.max_soc * capacity_v
            or (retired_v and energy_v != 0.0)
        ):
            raise ValueError("initial_vehicle_energy_wh must be within the vehicle SOC window")
        pv_v = _carried_origin(
            initial_vehicle_pv_origin_energy_wh, "initial_vehicle_pv_origin_energy_wh", energy_v, "vehicle energy"
        )
        grid_v = _carried_origin(
            initial_vehicle_grid_origin_energy_wh,
            "initial_vehicle_grid_origin_energy_wh",
            energy_v - pv_v,
            "vehicle energy less its PV origin",
        )
        vehicle_lifecycle = VehicleLifecycle(
            vehicle_config,
            vehicle_carry,
            initial_soc=energy_v / capacity_v if capacity_v > 0.0 else 0.0,
            initial_temperature=float(vehicle_temp[0]),
            rainflow_kernel=_resolve_rainflow_kernel(execution_backend),
        )
        charge_v = np.zeros(n_steps, dtype=np.bool_)
        discharge_v = np.ones(n_steps, dtype=np.bool_)
        if vehicle_config.charge_periods or vehicle_config.discharge_periods:
            if vehicle_tariff is None or not vehicle_tariff.index.equals(rng):
                raise ValueError("vehicle charge/discharge periods need a vehicle_tariff on the simulation index")
            unknown = (set(vehicle_config.charge_periods) | set(vehicle_config.discharge_periods)) - set(
                vehicle_tariff.schedule.periods
            )
            if unknown:
                raise ValueError(f"unknown vehicle tariff periods: {sorted(unknown)}")
            charge_v = np.isin(vehicle_tariff.period_labels, vehicle_config.charge_periods)
            if vehicle_config.discharge_periods:
                discharge_v = np.isin(vehicle_tariff.period_labels, vehicle_config.discharge_periods)
        charge_v.setflags(write=False)
        discharge_v.setflags(write=False)
        vehicle_state = dict(
            vehicle_config=vehicle_config,
            vehicle_schedule=vehicle_schedule,
            vehicle_energy_wh=energy_v,
            vehicle_pv_origin_wh=pv_v,
            vehicle_grid_origin_wh=grid_v,
            vehicle_soh_fraction=vehicle_lifecycle.soh,
            vehicle_retired=vehicle_lifecycle.retired,
            vehicle_was_plugged=bool(vehicle_schedule.plugged[0])
            if vehicle_carry is None
            else bool(vehicle_carry.plugged),
            vehicle_charge_period=charge_v,
            vehicle_discharge_period=discharge_v,
            grid_import_limit_w=grid_import_limit_w,
        )

    def final_vehicle_carry() -> Optional[VehicleCarry]:
        return None if vehicle_lifecycle is None else vehicle_lifecycle.carry(out.columns, n_steps)

    def close_vehicle_day(lo: int, hi: int) -> None:
        if vehicle_lifecycle is None:
            return
        vehicle_lifecycle.close(
            out.columns,
            lo=lo,
            hi=hi,
            time_ticks=vehicle_ticks,
            ticks_per_second=vehicle_ticks_per_second,
            temperature=vehicle_temp,
            hours=hours_per_step,
            step_time=rng[hi - 1],
            finalize=finalize_degradation and hi == n_steps,
            allow_terminal_replacement=vehicle_allow_terminal_replacement,
            years_after_span=vehicle_replacement_years_after_span,
            n_steps=n_steps,
        )
        assert isinstance(out, _ResultBuffers)
        out.vehicle_events = [event.to_record() for event in vehicle_lifecycle.events]
        last = hi - 1
        vehicle_state.update(
            vehicle_energy_wh=float(out.columns["Vehicle_Energy_Onboard"][last]),
            vehicle_pv_origin_wh=float(out.columns["Vehicle_PV_Origin_Energy_End"][last]),
            vehicle_grid_origin_wh=float(out.columns["Vehicle_Grid_Origin_Energy_End"][last]),
            vehicle_was_plugged=bool(out.columns["Vehicle_Plugged"][last]),
            vehicle_soh_fraction=vehicle_lifecycle.soh,
            vehicle_retired=vehicle_lifecycle.retired,
        )

    def dispatch_vehicle_only(lo: int, hi: int) -> None:
        # House rows have already been filled by the unchanged vectorised
        # PV-only path. The shared day loop then dispatches only the vehicle.
        dispatch_day(
            out,
            _pv_dc_vals,
            _load_vals,
            _temp_vals,
            lo,
            hi,
            battery_config=battery_config,
            battery_soh_decimal=battery_soh_decimal,
            Battery_Energy_Wh=0.0,
            Battery_PV_Origin_Energy_Wh=0.0,
            Battery_Grid_Origin_Energy_Wh=0.0,
            eff_charge=eff_charge,
            eff_discharge=eff_discharge,
            hours_per_step=hours_per_step,
            standby_loss_per_step_wh=0.0,
            cap_wh=cap_wh,
            cap_charge_wh=cap_charge_wh,
            cap_discharge_wh=cap_discharge_wh,
            instructions=instructions,
            ac_derate=_ones,
            curve_rows=_own_curve_rows,
            stationary_active=False,
            **vehicle_state,
        )

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
    # The kernels read a per-step rating fraction and curve row. Without
    # per-step series, one array of each serves every day: _day_arguments
    # would otherwise build a whole-run array for every day it packs.
    _ones = np.ones(n_steps)
    _full_rating = _ones if _derate_vals is None else _derate_vals
    _own_curve_rows = np.full(n_steps, battery_config.conversion.curve_row)
    _dispatch_curve_rows = _own_curve_rows if _curve_rows is None else _curve_rows

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
            ac_derate=_derate_vals,
            curve_rows=_curve_rows,
            units=_units,
        )
        if vehicle_config is not None:
            for lo in range(0, n_steps, steps_per_day):
                hi = min(lo + steps_per_day, n_steps)
                dispatch_vehicle_only(lo, hi)
                close_vehicle_day(lo, hi)
        return _CoreRun(
            buffers=out,
            rng=rng,
            aging=aging,
            lifecycle=degradation_lifecycle,
            degradation_tracking=degradation_tracking,
            degradation_periods=degradation_periods,
            hours_per_step=hours_per_step,
            vehicle_carry=final_vehicle_carry(),
            vehicle_end_of_life_events=() if vehicle_lifecycle is None else tuple(vehicle_lifecycle.events),
            vehicle_degradation_tracking=[] if vehicle_lifecycle is None else vehicle_lifecycle.tracking,
        )

    # An AC-coupled battery's PV inverter converts as in a PV-only run,
    # whatever the battery does, so its conversion is computed once for the
    # span (ADR 0004 C3, M2).
    ac_pv_chain: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None
    if ac_coupled:
        chain_ac, chain_loss, chain_clip, chain_night = _pv_inverter_conversion(
            np.maximum(0.0, _pv_dc_vals * hours_per_step),
            battery_config=battery_config,
            hours_per_step=hours_per_step,
            cap_wh=cap_wh,
            ac_derate=_derate_vals,
            curve_rows=_curve_rows,
            units=_units,
        )
        ac_pv_chain = (chain_ac, chain_loss, chain_clip, np.zeros(n_steps) if chain_night is None else chain_night)

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
                    session.decide(segment_start, battery_state())
                segment_end = session.prepare_segment(segment_start, window_end)
            if aging.retired:
                # A retired pack neither charges nor discharges, whatever the
                # instructions say: the steps are the PV-only system's.
                _dispatch_no_battery_vectorized(
                    out,
                    _pv_dc_vals,
                    _load_vals,
                    _temp_vals,
                    battery_config=battery_config,
                    hours_per_step=hours_per_step,
                    cap_wh=cap_wh,
                    steps=slice(segment_start, segment_end),
                    ac_derate=_derate_vals,
                    curve_rows=_curve_rows,
                    span_chain=ac_pv_chain,
                )
                out.columns["Battery_SOH"][segment_start:segment_end] = aging.soh_percent
                if vehicle_config is not None:
                    dispatch_vehicle_only(segment_start, segment_end)
            else:
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
                    ac_derate=_full_rating,
                    curve_rows=_dispatch_curve_rows,
                    pv_chain=ac_pv_chain,
                    **vehicle_state,
                )
            if session is not None:
                session.complete_segment(_pv_dc_vals, _load_vals, _temp_vals)
            segment_last = segment_end - 1
            if vehicle_config is not None:
                vehicle_state["vehicle_was_plugged"] = bool(out.columns["Vehicle_Plugged"][segment_last])
                vehicle_state["vehicle_energy_wh"] = float(out.columns["Vehicle_Energy_Onboard"][segment_last])
                vehicle_state["vehicle_pv_origin_wh"] = float(out.columns["Vehicle_PV_Origin_Energy_End"][segment_last])
                vehicle_state["vehicle_grid_origin_wh"] = float(
                    out.columns["Vehicle_Grid_Origin_Energy_End"][segment_last]
                )
            Battery_Energy_Wh = float(out.columns["Battery_Energy"][segment_last])
            Battery_PV_Origin_Energy_Wh = float(out.columns["Battery_PV_Origin_Energy_End"][segment_last])
            Battery_Grid_Origin_Energy_Wh = float(out.columns["Battery_Grid_Origin_Energy_End"][segment_last])
            segment_start = segment_end
        last_step = window_end - 1
        # The window's closing row, written by its final segment, holds the
        # state the day close continues from.
        battery_energy_beginning = float(out.columns["Battery_Energy_Beginning"][last_step])
        degradation_periods += 1
        Battery_Energy_Wh, Battery_PV_Origin_Energy_Wh, Battery_Grid_Origin_Energy_Wh = _apply_daily_degradation(
            aging,
            degradation_lifecycle,
            battery_config,
            out,
            degradation_tracking,
            step_index=last_step,
            # The timestamp is read only for a record or an end-of-life
            # event; nothing inside the window depends on the calendar.
            step_times=rng,
            # A summary reads only whether any period closed.
            keep_record=not summary_only,
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
        if vehicle_lifecycle is not None:
            close_vehicle_day(window_start, window_end)
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
        degradation_periods=degradation_periods,
        hours_per_step=hours_per_step,
        vehicle_carry=final_vehicle_carry(),
        vehicle_end_of_life_events=() if vehicle_lifecycle is None else tuple(vehicle_lifecycle.events),
        vehicle_degradation_tracking=[] if vehicle_lifecycle is None else vehicle_lifecycle.tracking,
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
    battery_retired: bool = False,
    inverter_ac_derate: Optional[pd.Series] = None,
    inverter_curve_row: Optional[pd.Series] = None,
    inverter_units: Optional[InverterUnits] = None,
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
        battery_retired=battery_retired,
        inverter_ac_derate=inverter_ac_derate,
        inverter_curve_row=inverter_curve_row,
        inverter_units=inverter_units,
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
    vehicle_carry: Optional[VehicleCarry] = None
    vehicle_end_of_life_events: Tuple[EndOfLifeEvent, ...] = ()
    vehicle_degradation_tracking: List[Dict[str, Any]] = field(default_factory=list)


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
        vehicle_carry=core.vehicle_carry,
        vehicle_end_of_life_events=core.vehicle_end_of_life_events,
        vehicle_degradation_tracking=core.vehicle_degradation_tracking,
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
    battery_retired: bool = False,
    inverter_ac_derate: Optional[pd.Series] = None,
    inverter_curve_row: Optional[pd.Series] = None,
    inverter_units: Optional[InverterUnits] = None,
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
        battery_retired=battery_retired,
        inverter_ac_derate=inverter_ac_derate,
        inverter_curve_row=inverter_curve_row,
        inverter_units=inverter_units,
    )
    return _build_simulation_summary(core, return_degradation_state=return_degradation_state, weights=weights)


def __getattr__(name: str) -> Any:
    # breos.battery.lfp_capacity_factor is deprecated in 0.8.0 and goes in 0.9.0;
    # the function itself stays in breos._dispatch, where the dispatch step uses it.
    if name == "lfp_capacity_factor":
        from breos._dispatch import lfp_capacity_factor

        warnings.warn(
            "breos.battery.lfp_capacity_factor is deprecated and will be removed in BREOS 0.9.0",
            DeprecationWarning,
            stacklevel=2,
        )
        return lfp_capacity_factor
    # Moved to breos.weather in 0.8.0, its only caller; the old name goes in 0.9.0.
    if name == "apply_indoor_temperature_model":
        from breos.weather import apply_indoor_temperature_model

        warnings.warn(
            "breos.battery.apply_indoor_temperature_model is deprecated and will be removed in BREOS 0.9.0; "
            "import it from breos.weather",
            DeprecationWarning,
            stacklevel=2,
        )
        return apply_indoor_temperature_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
