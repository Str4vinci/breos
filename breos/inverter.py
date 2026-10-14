"""
Inverter conversion and sizing helpers.

This module handles:
- DC-to-AC conversion with the PVWatts part-load curve and clipping
- The inverse: the DC input a requested AC output needs
- The inverter AC nameplate for a DC peak and loading ratio
- A measured efficiency curve, an AC cable and temperature derating
  (:class:`InverterConversion`, :class:`AcCable`, :func:`ac_derate`)
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Optional

import numpy as np
import pandas as pd

from breos._dispatch import (
    PVWATTS_CURVE_CONSTANT,
    PVWATTS_CURVE_LINEAR,
    PVWATTS_CURVE_QUADRATIC,
    PVWATTS_REFERENCE_EFFICIENCY,
    _dc_ac,
    _dc_for_ac,
)
from breos.electrical import (
    CONDUCTOR_RESISTIVITY_20C_OHM_M,
    CONDUCTOR_TEMPERATURE_COEFFICIENT_PER_K,
    DEFAULT_CONDUCTOR_TEMP_C,
)
from breos.equipment import EfficiencyCurve, TemperatureDerating

_NO_CURVE_AC = np.zeros(0)
_NO_CURVE_EFF = np.zeros((0, 0))


@dataclass(frozen=True, eq=False)
class InverterConversion:
    """What converts DC to AC beyond the nameplate and the single efficiency.

    ``curve_ac`` holds the efficiency curve's output points as fractions of the
    AC nameplate and ``curve_eff`` one row of efficiencies per DC voltage;
    ``curve_row`` is the operating voltage's position among the rows. Empty
    arrays keep the PVWatts part-load curve. ``ac_cable_k_per_w`` is the AC
    cable's loss over the square of the AC output, in 1/W (:class:`AcCable`).
    ``night_consumption_w`` is what the inverter draws from the grid while it
    converts nothing. The default converts exactly as no conversion does.
    """

    curve_ac: np.ndarray = field(default_factory=lambda: _NO_CURVE_AC)
    curve_eff: np.ndarray = field(default_factory=lambda: _NO_CURVE_EFF)
    curve_row: float = 0.0
    ac_cable_k_per_w: float = 0.0
    night_consumption_w: float = 0.0

    def __post_init__(self) -> None:
        curve_ac = np.ascontiguousarray(self.curve_ac, dtype=np.float64).reshape(-1)
        curve_eff = np.ascontiguousarray(self.curve_eff, dtype=np.float64)
        if curve_ac.size == 0:
            curve_eff = _NO_CURVE_EFF
        elif curve_eff.ndim != 2 or curve_eff.shape[1] != curve_ac.size or curve_eff.shape[0] < 1:
            raise ValueError("curve_eff must hold one row of len(curve_ac) efficiencies per voltage")
        k = float(self.ac_cable_k_per_w)
        if not math.isfinite(k) or k < 0.0:
            raise ValueError("ac_cable_k_per_w must be a finite number of at least 0")
        row = float(self.curve_row)
        if not math.isfinite(row) or row < 0.0:
            raise ValueError("curve_row must be a finite number of at least 0")
        night = float(self.night_consumption_w)
        if not math.isfinite(night) or night < 0.0:
            raise ValueError("night_consumption_w must be a finite number of at least 0")
        object.__setattr__(self, "curve_ac", curve_ac)
        object.__setattr__(self, "curve_eff", curve_eff)
        object.__setattr__(self, "curve_row", row)
        object.__setattr__(self, "ac_cable_k_per_w", k)
        object.__setattr__(self, "night_consumption_w", night)

    @classmethod
    def build(
        cls,
        curve: Optional[EfficiencyCurve] = None,
        *,
        voltage_v: Optional[float] = None,
        ac_cable: Optional["AcCable"] = None,
        night_consumption_w: Optional[float] = None,
    ) -> "InverterConversion":
        """The conversion for an efficiency curve read at DC voltage ``voltage_v``, an AC cable and a night draw.

        Without ``voltage_v`` a curve is read at its middle row: the nominal
        voltage of a three-row curve.
        """
        k = 0.0 if ac_cable is None else ac_cable.loss_coefficient_per_w
        night = 0.0 if night_consumption_w is None else night_consumption_w
        if curve is None:
            return cls(ac_cable_k_per_w=k, night_consumption_w=night)
        rows = len(curve.efficiency)
        position = (rows - 1) / 2.0 if voltage_v is None else curve.row_position(voltage_v)
        return cls(np.asarray(curve.output_fraction), np.asarray(curve.efficiency), position, k, night)

    @property
    def is_default(self) -> bool:
        """Whether DC converts to AC as the PVWatts curve with no cable does; the night draw is apart."""
        return self.curve_ac.size == 0 and self.ac_cable_k_per_w == 0.0

    def cache_key(self) -> tuple[Any, ...]:
        """A hashable identity of the DC-to-AC conversion, for memo keys; the night draw is not part of it."""
        return (tuple(self.curve_ac), tuple(map(tuple, self.curve_eff)), self.curve_row, self.ac_cable_k_per_w)

    def kernel_arguments(self, hours_per_step: float) -> tuple[np.ndarray, np.ndarray, float, float]:
        """``(curve_ac, curve_eff, curve_row, cable_k)`` as the kernels take them, with the cable per Wh."""
        cable_k = self.ac_cable_k_per_w / hours_per_step if self.ac_cable_k_per_w > 0.0 else 0.0
        return self.curve_ac, self.curve_eff, self.curve_row, cable_k


NO_CONVERSION = InverterConversion()


def as_inverter_conversion(value: Any) -> InverterConversion:
    """An :class:`InverterConversion` from itself, a mapping of its fields, or None for the default."""
    if value is None:
        return NO_CONVERSION
    if isinstance(value, InverterConversion):
        return value
    if isinstance(value, Mapping):
        return InverterConversion(**value)
    raise ValueError("inverter_conversion must be an InverterConversion, a mapping of its fields, or None")


@dataclass(frozen=True)
class AcCable:
    """The AC cable from the inverter to the meter.

    ``length_m`` is one-way and ``cross_section_mm2`` each conductor's.
    ``phases`` is 1, with ``voltage_v`` line to neutral and two current-carrying
    conductors, or 3, with ``voltage_v`` line to line and three. ``power_factor``
    is the inverter's, 1 by default. ``conductor_temp_c`` is the conductors'
    operating temperature; unstated, :data:`~breos.electrical.DEFAULT_CONDUCTOR_TEMP_C`
    is assumed.
    """

    length_m: float
    cross_section_mm2: float
    phases: int
    voltage_v: float
    conductor: str = "copper"
    power_factor: float = 1.0
    conductor_temp_c: Optional[float] = None

    def __post_init__(self) -> None:
        for name in ("length_m", "cross_section_mm2", "voltage_v", "power_factor"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if self.length_m < 0:
            raise ValueError("length_m must be at least 0")
        if self.cross_section_mm2 <= 0 or self.voltage_v <= 0:
            raise ValueError("cross_section_mm2 and voltage_v must be above 0")
        if not 0 < self.power_factor <= 1:
            raise ValueError("power_factor must be in (0, 1]")
        if self.phases not in (1, 3) or isinstance(self.phases, bool):
            raise ValueError("phases must be 1 or 3")
        if self.conductor not in CONDUCTOR_RESISTIVITY_20C_OHM_M:
            raise ValueError(
                f"conductor must be one of {sorted(CONDUCTOR_RESISTIVITY_20C_OHM_M)}, got {self.conductor!r}"
            )
        if self.conductor_temp_c is not None and (
            isinstance(self.conductor_temp_c, bool)
            or not isinstance(self.conductor_temp_c, (int, float))
            or not math.isfinite(self.conductor_temp_c)
        ):
            raise ValueError("conductor_temp_c must be a finite number")

    @property
    def conductor_c(self) -> float:
        return float(DEFAULT_CONDUCTOR_TEMP_C if self.conductor_temp_c is None else self.conductor_temp_c)

    @property
    def conductor_resistance_ohm(self) -> float:
        """One conductor's resistance at the conductor temperature."""
        rho = CONDUCTOR_RESISTIVITY_20C_OHM_M[self.conductor] * (
            1.0 + CONDUCTOR_TEMPERATURE_COEFFICIENT_PER_K[self.conductor] * (self.conductor_c - 20.0)
        )
        return float(rho * self.length_m / (self.cross_section_mm2 * 1e-6))

    @property
    def loss_coefficient_per_w(self) -> float:
        """The cable's loss over the square of the inverter's AC output, in 1/W.

        Single-phase: two conductors carry ``P / (V * pf)``. Three-phase: three
        carry ``P / (sqrt(3) * V * pf)``, which loses ``R * P**2 / (V * pf)**2``.
        """
        conductors = 2.0 if self.phases == 1 else 1.0
        return conductors * self.conductor_resistance_ohm / (self.voltage_v * self.power_factor) ** 2

    def loss_pct_at(self, ac_power_w: float) -> float:
        """The share of ``ac_power_w`` the cable dissipates, in percent."""
        return 100.0 * self.loss_coefficient_per_w * ac_power_w


def ac_derate(derating: TemperatureDerating, ambient_c: np.ndarray) -> np.ndarray:
    """Each step's allowed AC output as a fraction of the nameplate, at ambient ``ambient_c`` (°C)."""
    return np.interp(np.asarray(ambient_c, dtype=float), derating.ambient_temp_c, derating.max_output_fraction)


@dataclass(frozen=True)
class InverterConversionResult:
    """AC conversion result with explicit DC-side clipping bookkeeping."""

    ac_power_w: float
    conversion_loss_w: float
    clipping_loss_dc_w: float
    clipping_loss_ac_equivalent_w: float

    @property
    def total_dc_input_w(self) -> float:
        """DC input reconstructed from AC output, conversion loss, and clipping."""
        return self.ac_power_w + self.conversion_loss_w + self.clipping_loss_dc_w


def _clamped_ac_output_scale(value: float) -> float:
    """Clamp an AC-side derate into ``[0, 1]``.

    The upper bound is the physical one: the factor is applied after the
    inverter nameplate limit, so a value above 1 would deliver more AC than
    the nameplate and more AC than the DC entering the converter.

    This clamp is a backstop, not the validation. Every configured route
    rejects an out-of-range value before reaching here, and loudly:
    :class:`~breos.battery.BatteryConfig` for anything that dispatches, and
    the study-config validators in :mod:`breos.optimization` for the
    optimizer. What remains is a direct call into these helpers, where
    clamping matches how ``inverter_efficiency`` already behaves beside it.
    """
    return min(1.0, max(0.0, float(value)))


def calculate_dc_ac_power(
    pv_dc_power: float,
    inverter_ac_power: float,
    inverter_efficiency: float = 0.96,
    ac_output_scale: float = 1.0,
) -> InverterConversionResult:
    """
    Calculate AC output and loss buckets with the PVWatts part-load curve.

    Clipping is reported on the DC side: power above the DC input required
    to saturate the AC rating is ``clipping_loss_dc_w``. The AC-equivalent
    clipping value is also exposed for reports that compare against
    ``pv_dc_power * inverter_efficiency``.

    ``ac_output_scale`` multiplies the converted AC power *after* the
    part-load curve and every inverter limit, so it derates AC delivery
    without moving the clipping threshold ``pdc0`` or the part-load ratio.

    It is an **in-dispatch derate, not a post-processing multiplier**. It is
    applied inside the conversion the dispatcher calls, so battery discharge
    decisions and the reachable AC ceiling respond to it, which is the correct
    behaviour for a derate that is really there. Multiplying a finished result
    series instead would leave dispatch believing in AC that was never
    delivered.

    It is a single constant standing in for AC-side shortfall the chain does
    not model, such as availability, curtailment or downstream wiring. One
    constant approximates their combined annual effect; it is not a model of
    any of them individually, and it cannot represent their time structure.

    It is bounded to ``(0, 1]``. A factor above 1 would let the inverter
    deliver more than its nameplate and more AC than the DC entering it,
    leaving ``conversion_loss_w`` pinned at zero. An under-predicting model is
    corrected on the DC side with ``dc_output_scale``, which keeps clipping
    and the part-load ratio responsive, or through ``inverter_efficiency``
    when the converter itself is modelled too pessimistically.

    While the derate is active, ``conversion_loss_w`` is the whole DC-to-AC
    shortfall and no longer only the inverter's own conversion loss: it
    carries the derated energy too. Reports that attribute it specifically to
    the converter must account for that.

    The default ``1.0`` is a no-op and reproduces prior behaviour
    bit-for-bit. ``dc_power_for_ac_output`` takes the same argument and stays
    its exact inverse.

    Args:
        pv_dc_power: DC power from PV array (W)
        inverter_ac_power: Inverter AC rating (W)
        inverter_efficiency: Nominal inverter efficiency
        ac_output_scale: In-dispatch AC-side derate applied after conversion, in (0, 1]

    Returns:
        InverterConversionResult with AC output and loss buckets.
    """
    pv_dc_power = max(0.0, float(pv_dc_power))
    inverter_ac_power = max(0.0, float(inverter_ac_power))
    inverter_efficiency = min(1.0, max(0.0, float(inverter_efficiency)))
    ac_output_scale = _clamped_ac_output_scale(ac_output_scale)
    ac_power, conversion_loss, clipping_loss_dc = _dc_ac(
        pv_dc_power, inverter_ac_power, inverter_efficiency, ac_output_scale, 2.0, _NO_CURVE_AC, _NO_CURVE_EFF, 0.0, 0.0
    )
    # Only a converter that is really there has an AC side to express the
    # clipped DC on; the core's early returns carry no clipping otherwise.
    if inverter_efficiency > 0.0 and inverter_ac_power > 0.0:
        clipping_loss_ac_equiv = clipping_loss_dc * inverter_efficiency * ac_output_scale
    else:
        clipping_loss_ac_equiv = 0.0

    return InverterConversionResult(
        ac_power_w=ac_power,
        conversion_loss_w=conversion_loss,
        clipping_loss_dc_w=clipping_loss_dc,
        clipping_loss_ac_equivalent_w=clipping_loss_ac_equiv,
    )


def _calculate_dc_ac_power_arrays(
    pv_dc_power: np.ndarray,
    inverter_ac_power: float,
    inverter_efficiency: float = 0.96,
    ac_output_scale: float = 1.0,
    *,
    conversion: InverterConversion = NO_CONVERSION,
    ac_derate: Optional[np.ndarray] = None,
    curve_rows: Optional[np.ndarray] = None,
    hours_per_step: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized counterpart of :func:`calculate_dc_ac_power`.

    Returns AC power, conversion loss, and DC-side clipping arrays. Expression
    order mirrors the scalar helper so the simulation can retain bit parity.

    With a non-default ``conversion`` or an ``ac_derate`` (each step's AC
    rating as a fraction of ``inverter_ac_power``), each step runs the
    dispatch kernels' scalar conversion, so the PV-only and battery paths
    agree exactly. ``curve_rows`` reads the efficiency curve at each step's
    row position instead of the conversion's own. ``hours_per_step`` converts
    the cable coefficient to the Wh the inputs are in.
    """
    pv_dc_power = np.maximum(0.0, np.asarray(pv_dc_power, dtype=np.float64))
    inverter_ac_power = max(0.0, float(inverter_ac_power))
    inverter_efficiency = min(1.0, max(0.0, float(inverter_efficiency)))
    ac_output_scale = _clamped_ac_output_scale(ac_output_scale)

    if not conversion.is_default or ac_derate is not None or curve_rows is not None:
        curve_ac, curve_eff, curve_row, cable_k = conversion.kernel_arguments(hours_per_step)
        derate = np.ones(pv_dc_power.shape) if ac_derate is None else np.asarray(ac_derate, dtype=np.float64)
        rows = np.full(pv_dc_power.shape, curve_row) if curve_rows is None else np.asarray(curve_rows, np.float64)
        ac_power = np.empty_like(pv_dc_power)
        conversion_loss = np.empty_like(pv_dc_power)
        clipping_loss_dc = np.empty_like(pv_dc_power)
        for i in range(pv_dc_power.shape[0]):
            ac_power[i], conversion_loss[i], clipping_loss_dc[i] = _dc_ac(
                float(pv_dc_power[i]),
                inverter_ac_power * float(derate[i]),
                inverter_efficiency,
                ac_output_scale,
                2.0,
                curve_ac,
                curve_eff,
                float(rows[i]),
                cable_k,
            )
        return ac_power, conversion_loss, clipping_loss_dc

    if inverter_efficiency <= 0.0 or inverter_ac_power <= 0.0:
        zeros = np.zeros_like(pv_dc_power)
        return zeros, zeros, pv_dc_power

    if not math.isfinite(inverter_ac_power):
        ac_power = pv_dc_power * inverter_efficiency * ac_output_scale
        return ac_power, pv_dc_power - ac_power, np.zeros_like(pv_dc_power)

    pdc0 = inverter_ac_power / inverter_efficiency
    dc_used = np.minimum(pv_dc_power, pdc0)
    zeta = dc_used / pdc0
    curve_power = (
        (inverter_efficiency / PVWATTS_REFERENCE_EFFICIENCY)
        * pdc0
        * (PVWATTS_CURVE_QUADRATIC * zeta**2 + PVWATTS_CURVE_LINEAR * zeta + PVWATTS_CURVE_CONSTANT)
    )
    ac_power = np.maximum(0.0, np.minimum(np.minimum(dc_used, inverter_ac_power), curve_power))
    ac_power = ac_power * ac_output_scale
    clipping_loss_dc = np.maximum(0.0, pv_dc_power - dc_used)
    conversion_loss = np.maximum(0.0, dc_used - ac_power)
    return ac_power, conversion_loss, clipping_loss_dc


@dataclass(frozen=True, eq=False)
class InverterUnits:
    """Several inverters, each converting its own share of the system's DC.

    String-level energy without a battery (ADR 0007 D10) or beside an
    AC-coupled one (ADR 0004). Entry ``i`` of
    ``ac_capacity_w``, ``efficiency`` and ``conversions`` is inverter ``i``'s
    AC nameplate, single efficiency (None for the run's own, which the
    conversion is given) and conversion: its efficiency curve, its own AC
    cable and its night draw. ``shares`` holds each step's share of
    the system DC per inverter, one column per inverter; ``ac_derate`` each
    inverter's AC rating as a fraction of its nameplate, 1 without
    temperature derating; and ``curve_rows`` each inverter's efficiency-curve
    row position. ``feeder_cable_k_per_w`` is the shared AC cable's loss over
    the square of the inverters' summed AC output, in 1/W.
    """

    ac_capacity_w: tuple[float, ...]
    efficiency: tuple[float | None, ...]
    conversions: tuple[InverterConversion, ...]
    shares: pd.DataFrame
    ac_derate: pd.DataFrame
    curve_rows: pd.DataFrame
    feeder_cable_k_per_w: float = 0.0

    def __post_init__(self) -> None:
        n = len(self.ac_capacity_w)
        if n < 2 or len(self.efficiency) != n or len(self.conversions) != n:
            raise ValueError("InverterUnits needs at least two inverters and one entry of each field per inverter")
        for frame in (self.shares, self.ac_derate, self.curve_rows):
            if frame.shape[1] != n or not frame.index.equals(self.shares.index):
                raise ValueError("InverterUnits per-step frames need one column per inverter on one index")
        k = float(self.feeder_cable_k_per_w)
        if not math.isfinite(k) or k < 0.0:
            raise ValueError("feeder_cable_k_per_w must be a finite number of at least 0")

    def aligned(self, index: pd.DatetimeIndex) -> "InverterUnits":
        """These units on ``index``, which their per-step frames must cover with finite values."""
        frames = []
        for name in ("shares", "ac_derate", "curve_rows"):
            frame = getattr(self, name).reindex(index)
            if not np.isfinite(frame.to_numpy(dtype=float)).all():
                raise ValueError(f"InverterUnits.{name} must hold a finite value at every simulated step")
            frames.append(frame)
        return replace(self, shares=frames[0], ac_derate=frames[1], curve_rows=frames[2])

    def efficiencies(self, default: float) -> tuple[float, ...]:
        """Each inverter's single efficiency, ``default`` where it states none."""
        return tuple(default if value is None else value for value in self.efficiency)


@dataclass(frozen=True)
class UnitsConversion:
    """What :func:`convert_units` returns, in the energy per step its DC came in.

    ``ac``, ``conversion_loss``, ``clipping`` and ``night`` are the system's:
    the AC delivered past the shared cable, the DC that entered the
    inverters less that AC, the DC above the inverters' limits, and their
    night draw. ``inverters`` holds one mapping per inverter of its ``dc``,
    ``ac`` (past its own cable), ``conversion_loss``, ``clipping``, ``night``
    and ``cable_loss``; ``feeder_cable_loss`` is the shared cable's.
    """

    ac: np.ndarray
    conversion_loss: np.ndarray
    clipping: np.ndarray
    night: np.ndarray
    inverters: tuple[dict[str, np.ndarray], ...]
    feeder_cable_loss: np.ndarray


def _cable_loss_from_net(net: np.ndarray, cable_k: float) -> np.ndarray:
    """What a cable losing ``cable_k * ac**2`` took from the AC it delivered as ``net``."""
    if cable_k <= 0.0:
        return np.zeros_like(net)
    root = np.sqrt(np.clip(1.0 - 4.0 * cable_k * net, 0.0, None))
    return 2.0 * net / (1.0 + root) - net


def convert_units(
    pv_dc: np.ndarray,
    units: InverterUnits,
    *,
    inverter_efficiency: float = 0.96,
    ac_output_scale: float = 1.0,
    hours_per_step: float = 1.0,
) -> UnitsConversion:
    """Convert the system DC ``pv_dc`` (energy per step) through each inverter of ``units`` on its own.

    ``units`` must be aligned to ``pv_dc``'s steps. Each inverter takes its
    share of the DC, clips at its own derated nameplate and converts along
    its own curve, past its own AC cable; it draws its night consumption on
    the steps it receives no DC. The shared cable then takes its loss from
    the summed AC, and ``ac_output_scale`` applies last, as in the single
    inverter's conversion. ``inverter_efficiency`` is the single efficiency
    of the inverters that state none.
    """
    pv_dc = np.maximum(0.0, np.asarray(pv_dc, dtype=np.float64))
    shares = units.shares.to_numpy(dtype=np.float64)
    derates = units.ac_derate.to_numpy(dtype=np.float64)
    rows = units.curve_rows.to_numpy(dtype=np.float64)
    gross = np.zeros_like(pv_dc)
    clipping = np.zeros_like(pv_dc)
    night = np.zeros_like(pv_dc)
    inverters = []
    efficiencies = units.efficiencies(inverter_efficiency)
    for i, conversion in enumerate(units.conversions):
        dc = pv_dc * shares[:, i]
        derate = derates[:, i]
        reads_rows = conversion.curve_ac.size > 0 and conversion.curve_eff.shape[0] > 1
        ac, _, clip = _calculate_dc_ac_power_arrays(
            dc,
            _step_cap(units.ac_capacity_w[i], hours_per_step),
            efficiencies[i],
            1.0,
            conversion=conversion,
            ac_derate=None if (derate == 1.0).all() else derate,
            curve_rows=rows[:, i] if reads_rows else None,
            hours_per_step=hours_per_step,
        )
        draw = np.where(dc <= 0.0, conversion.night_consumption_w * hours_per_step, 0.0)
        cable_k = conversion.ac_cable_k_per_w / hours_per_step
        inverters.append(
            {
                "dc": dc,
                "ac": ac,
                "conversion_loss": np.maximum(0.0, dc - clip - ac),
                "clipping": clip,
                "night": draw,
                "cable_loss": _cable_loss_from_net(ac, cable_k),
            }
        )
        gross += ac
        clipping += clip
        night += draw
    feeder_k = units.feeder_cable_k_per_w / hours_per_step
    delivered = np.maximum(0.0, gross - feeder_k * gross * gross) if feeder_k > 0.0 else gross
    ac = delivered * _clamped_ac_output_scale(ac_output_scale)
    return UnitsConversion(
        ac=ac,
        conversion_loss=np.maximum(0.0, pv_dc - clipping - ac),
        clipping=clipping,
        night=night,
        inverters=tuple(inverters),
        feeder_cable_loss=gross - delivered,
    )


def _step_cap(power_w: float, hours_per_step: float) -> float:
    return power_w * hours_per_step


def dc_power_for_ac_output(
    ac_power_w: float,
    inverter_ac_power: float,
    inverter_efficiency: float = 0.96,
    ac_output_scale: float = 1.0,
) -> float:
    """Return the minimum DC input required for a requested PVWatts AC output.

    The inverse is solved on the monotonic operating range up to the inverter
    nameplate. Requests above the nameplate are clamped to it. Keeping this
    inverse beside :func:`calculate_dc_ac_power` prevents dispatch from
    silently reverting to a flat-efficiency approximation.

    ``ac_output_scale`` matches the forward helper, including its ``(0, 1]``
    bound: the request is divided by it before the inverse is solved, so
    ``calculate_dc_ac_power`` applied to the returned DC reproduces the
    requested AC at the same scale. Dispatch must pass the same value to both,
    or it would size DC against one boundary and deliver against another.
    """
    ac_output_scale = _clamped_ac_output_scale(ac_output_scale)
    if ac_output_scale <= 0.0:
        return 0.0
    return _dc_for_ac(
        float(ac_power_w),
        float(inverter_ac_power),
        float(inverter_efficiency),
        ac_output_scale,
        _NO_CURVE_AC,
        _NO_CURVE_EFF,
        0.0,
        0.0,
    )


def inverter_ac_capacity_w(pv_peak_w: float, loading_ratio: Optional[float]) -> Optional[float]:
    """Return the inverter AC nameplate in W for a DC peak and DC/AC loading ratio.

    This is the one sizing rule the App, Monte Carlo and the optimizer share.
    A loading ratio that is unset or not positive gives None, which the
    dispatch reads as no AC clipping.
    """
    if loading_ratio is None or not loading_ratio > 0:
        return None
    return pv_peak_w / loading_ratio
