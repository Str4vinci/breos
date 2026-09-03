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

PVWATTS_REFERENCE_EFFICIENCY = 0.9637
PVWATTS_CURVE_QUADRATIC = -0.0162
PVWATTS_CURVE_LINEAR = 0.9858
PVWATTS_CURVE_CONSTANT = -0.0059


def _require_optional_non_negative_finite(name: str, value: Optional[float]) -> None:
    """Reject invalid supplied datasheet quantities while allowing unknown limits."""
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
    """Reject MPPT counts and parallel-string limits that cannot describe hardware."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass
class InverterConfig:
    """
    Inverter configuration parameters.

    Attributes:
        nominal_power_w: Inverter nominal AC power (W). If None, sized from PV.
        dc_ac_ratio: DC/AC sizing ratio (typical: 1.1-1.25)
        inverter_efficiency: Peak inverter efficiency (typical: 0.96-0.98)
        is_hybrid: Whether this is a hybrid inverter with battery support
        mppt_channels: Number of MPPT channels
        cost_per_kw_simple: Cost per kW for simple (grid-tie) inverter
        cost_per_kw_hybrid: Cost per kW for hybrid inverter (with battery)
        max_dc_voltage_v: Absolute maximum DC input voltage from the datasheet (V).
        max_dc_power_w: Maximum recommended or permitted DC input power (W).
        min_mppt_voltage_v: Lower bound of the MPPT operating window (V).
        max_mppt_voltage_v: Upper bound of the MPPT operating window (V).
        startup_voltage_v: DC voltage required for inverter startup (V).
        max_strings_per_mppt: Maximum parallel strings permitted on each MPPT.
        max_input_current_per_mppt_a: Maximum operating input current per MPPT (A).
        max_short_circuit_current_per_mppt_a: Maximum short-circuit current per MPPT (A).

    The datasheet fields are optional so existing aggregate simulations and
    callers which do not yet know a particular inverter's nameplate limits
    remain valid. When a field is supplied, it is validated here rather than
    relying on an API boundary to do so.
    """

    nominal_power_w: Optional[float] = None
    dc_ac_ratio: float = 1.25  # Default 1.25
    inverter_efficiency: float = 0.96
    is_hybrid: bool = True
    mppt_channels: int = 2
    cost_per_kw_simple: float = 48.37  # €/kW for simple grid-tie inverter
    cost_per_kw_hybrid: float = 102.58  # €/kW for hybrid inverter
    max_dc_voltage_v: Optional[float] = None
    max_dc_power_w: Optional[float] = None
    min_mppt_voltage_v: Optional[float] = None
    max_mppt_voltage_v: Optional[float] = None
    startup_voltage_v: Optional[float] = None
    max_strings_per_mppt: Optional[int] = None
    max_input_current_per_mppt_a: Optional[float] = None
    max_short_circuit_current_per_mppt_a: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate configuration and supplied datasheet limits without dependencies."""
        _require_optional_non_negative_finite("nominal_power_w", self.nominal_power_w)
        _require_positive_finite("dc_ac_ratio", self.dc_ac_ratio)
        _require_efficiency("inverter_efficiency", self.inverter_efficiency)
        if not isinstance(self.is_hybrid, bool):
            raise ValueError("is_hybrid must be a bool")
        _require_positive_integer("mppt_channels", self.mppt_channels)
        _require_optional_non_negative_finite("cost_per_kw_simple", self.cost_per_kw_simple)
        _require_optional_non_negative_finite("cost_per_kw_hybrid", self.cost_per_kw_hybrid)
        _require_optional_non_negative_finite("max_dc_voltage_v", self.max_dc_voltage_v)
        _require_optional_non_negative_finite("max_dc_power_w", self.max_dc_power_w)
        _require_optional_non_negative_finite("min_mppt_voltage_v", self.min_mppt_voltage_v)
        _require_optional_non_negative_finite("max_mppt_voltage_v", self.max_mppt_voltage_v)
        _require_optional_non_negative_finite("startup_voltage_v", self.startup_voltage_v)
        _require_optional_non_negative_finite("max_input_current_per_mppt_a", self.max_input_current_per_mppt_a)
        _require_optional_non_negative_finite(
            "max_short_circuit_current_per_mppt_a", self.max_short_circuit_current_per_mppt_a
        )

        if self.max_strings_per_mppt is not None:
            _require_positive_integer("max_strings_per_mppt", self.max_strings_per_mppt)

        if (
            self.min_mppt_voltage_v is not None
            and self.max_mppt_voltage_v is not None
            and self.min_mppt_voltage_v > self.max_mppt_voltage_v
        ):
            raise ValueError("min_mppt_voltage_v must not exceed max_mppt_voltage_v")

        if (
            self.max_dc_voltage_v is not None
            and self.max_mppt_voltage_v is not None
            and self.max_mppt_voltage_v > self.max_dc_voltage_v
        ):
            raise ValueError("max_mppt_voltage_v must not exceed max_dc_voltage_v")

        if (
            self.max_dc_voltage_v is not None
            and self.min_mppt_voltage_v is not None
            and self.min_mppt_voltage_v > self.max_dc_voltage_v
        ):
            raise ValueError("min_mppt_voltage_v must not exceed max_dc_voltage_v")

        # Only the physical ceiling is enforced. Startup voltage is deliberately
        # not required to sit inside the MPPT window: plenty of real datasheets
        # quote a startup well below the MPP range minimum (Fronius Primo starts
        # at ~80 V against an MPP range from ~240 V), because startup marks where
        # the inverter wakes up, not where it can track. Requiring containment
        # would reject a faithful transcription of those sheets.
        if (
            self.startup_voltage_v is not None
            and self.max_dc_voltage_v is not None
            and self.startup_voltage_v > self.max_dc_voltage_v
        ):
            raise ValueError("startup_voltage_v must not exceed max_dc_voltage_v")

    def size_from_pv(self, pv_peak_power_w: float) -> float:
        """
        Size inverter based on PV peak power.

        Args:
            pv_peak_power_w: Total PV array peak power (Wp)

        Returns:
            Inverter nominal AC power (W)
        """
        return pv_peak_power_w / self.dc_ac_ratio

    def get_cost(self, pv_peak_power_w: Optional[float] = None) -> float:
        """
        Calculate inverter cost.

        Args:
            pv_peak_power_w: PV peak power for sizing (uses nominal_power if provided)

        Returns:
            Inverter cost in €
        """
        if self.nominal_power_w is not None:
            power = self.nominal_power_w
        elif pv_peak_power_w is not None:
            power = self.size_from_pv(pv_peak_power_w)
        else:
            raise ValueError("Either nominal_power_w or pv_peak_power_w must be provided")

        cost_per_kw = self.cost_per_kw_hybrid if self.is_hybrid else self.cost_per_kw_simple
        power_kw = power / 1000  # Convert W to kW
        return power_kw * cost_per_kw


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


# Common inverter presets
INVERTER_PRESETS = {
    "residential_hybrid": InverterConfig(
        dc_ac_ratio=1.25,
        inverter_efficiency=0.96,
        is_hybrid=True,
    ),
    "residential_simple": InverterConfig(
        dc_ac_ratio=1.25,
        inverter_efficiency=0.96,
        is_hybrid=False,
    ),
    "commercial_hybrid": InverterConfig(
        dc_ac_ratio=1.25,
        inverter_efficiency=0.98,
        is_hybrid=True,
    ),
    "oversized_1.5": InverterConfig(
        dc_ac_ratio=1.5,
        inverter_efficiency=0.96,
        is_hybrid=True,
    ),
}


def get_inverter_preset(name: str) -> InverterConfig:
    """
    Get a pre-defined inverter configuration.

    Available presets:
    - residential_hybrid: 1.25 ratio, 0.96 efficiency, hybrid
    - residential_simple: 1.25 ratio, 0.96 efficiency, grid-tie only
    - commercial_hybrid: 1.25 ratio, 0.98 efficiency, hybrid
    - oversized_1.5: 1.5 ratio for high DC/AC

    Args:
        name: Preset name

    Returns:
        InverterConfig object
    """
    if name not in INVERTER_PRESETS:
        available = ", ".join(INVERTER_PRESETS.keys())
        raise KeyError(f"Preset '{name}' not found. Available: {available}")
    return INVERTER_PRESETS[name]


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

    if inverter_efficiency <= 0.0 or inverter_ac_power <= 0.0:
        return InverterConversionResult(
            ac_power_w=0.0,
            conversion_loss_w=0.0,
            clipping_loss_dc_w=pv_dc_power,
            clipping_loss_ac_equivalent_w=0.0,
        )

    if pv_dc_power <= 0.0:
        return InverterConversionResult(
            ac_power_w=0.0,
            conversion_loss_w=0.0,
            clipping_loss_dc_w=0.0,
            clipping_loss_ac_equivalent_w=0.0,
        )

    # A lower-level BatteryConfig may intentionally omit the inverter
    # nameplate. With no rated power there is no part-load ratio to evaluate,
    # so retain the historical unbounded flat-efficiency behavior. App always
    # supplies its sized finite AC rating.
    if not math.isfinite(inverter_ac_power):
        ac_power = pv_dc_power * inverter_efficiency * ac_output_scale
        return InverterConversionResult(
            ac_power_w=ac_power,
            conversion_loss_w=pv_dc_power - ac_power,
            clipping_loss_dc_w=0.0,
            clipping_loss_ac_equivalent_w=0.0,
        )

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
            * (PVWATTS_CURVE_QUADRATIC * zeta**2 + PVWATTS_CURVE_LINEAR * zeta + PVWATTS_CURVE_CONSTANT),
        ),
    )
    ac_power *= ac_output_scale
    clipping_loss_dc = max(0.0, pv_dc_power - dc_used)
    conversion_loss = max(0.0, dc_used - ac_power)
    clipping_loss_ac_equiv = clipping_loss_dc * inverter_efficiency * ac_output_scale

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
    ac_power_w = float(ac_power_w) / ac_output_scale
    ac_target = max(0.0, min(float(ac_power_w), max(0.0, float(inverter_ac_power))))
    inverter_ac_power = max(0.0, float(inverter_ac_power))
    inverter_efficiency = min(1.0, max(0.0, float(inverter_efficiency)))
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
