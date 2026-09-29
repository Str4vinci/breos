"""
Inverter module for PV system sizing and efficiency.

This module handles:
- Inverter sizing based on PV array power
- DC/AC coupling configurations
- Efficiency calculations
"""

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Optional

import numpy as np

from breos._dispatch import (
    PVWATTS_CURVE_CONSTANT,
    PVWATTS_CURVE_LINEAR,
    PVWATTS_CURVE_QUADRATIC,
    PVWATTS_REFERENCE_EFFICIENCY,
    _dc_ac,
    _dc_for_ac,
)


def _require_optional_non_negative_finite(name: str, value: Optional[float]) -> None:
    """Reject an invalid supplied quantity while allowing an unknown one."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number when provided")


def _require_positive_finite(name: str, value: float) -> None:
    """Reject a ratio or other quantity that must be strictly positive."""
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")


def _require_efficiency(name: str, value: float) -> None:
    """Reject efficiencies outside the physically meaningful interval (0, 1]."""
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError(f"{name} must be a finite number in (0, 1]")


def _require_positive_integer(name: str, value: int) -> None:
    """Reject an MPPT count that cannot describe hardware."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass
class InverterConfig:
    """
    Inverter configuration parameters.

    Attributes:
        nominal_power_w: Inverter nominal AC power (W), or None if unknown.
        dc_ac_ratio: DC/AC sizing ratio (typical: 1.1-1.25)
        inverter_efficiency: Peak inverter efficiency (typical: 0.96-0.98)
        is_hybrid: Whether this is a hybrid inverter with battery support
        mppt_channels: Number of MPPT channels
    """

    nominal_power_w: Optional[float] = None
    dc_ac_ratio: float = 1.25  # Default 1.25
    inverter_efficiency: float = 0.96
    is_hybrid: bool = True
    mppt_channels: int = 2

    def __post_init__(self) -> None:
        """Validate the configuration without dependencies."""
        _require_optional_non_negative_finite("nominal_power_w", self.nominal_power_w)
        _require_positive_finite("dc_ac_ratio", self.dc_ac_ratio)
        _require_efficiency("inverter_efficiency", self.inverter_efficiency)
        if not isinstance(self.is_hybrid, bool):
            raise ValueError("is_hybrid must be a bool")
        _require_positive_integer("mppt_channels", self.mppt_channels)


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
        pv_dc_power, inverter_ac_power, inverter_efficiency, ac_output_scale, 2.0
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized counterpart of :func:`calculate_dc_ac_power`.

    Returns AC power, conversion loss, and DC-side clipping arrays. Expression
    order mirrors the scalar helper so the simulation can retain bit parity.
    """
    pv_dc_power = np.maximum(0.0, np.asarray(pv_dc_power, dtype=np.float64))
    inverter_ac_power = max(0.0, float(inverter_ac_power))
    inverter_efficiency = min(1.0, max(0.0, float(inverter_efficiency)))
    ac_output_scale = _clamped_ac_output_scale(ac_output_scale)

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
    return _dc_for_ac(float(ac_power_w), float(inverter_ac_power), float(inverter_efficiency), ac_output_scale)


def inverter_ac_capacity_w(pv_peak_w: float, loading_ratio: Optional[float]) -> Optional[float]:
    """Return the inverter AC nameplate in W for a DC peak and DC/AC loading ratio.

    This is the one sizing rule the App, Monte Carlo and the optimizer share.
    A loading ratio that is unset or not positive gives None, which the
    dispatch reads as no AC clipping.
    """
    if loading_ratio is None or not loading_ratio > 0:
        return None
    return pv_peak_w / loading_ratio
