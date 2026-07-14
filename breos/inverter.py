"""
Inverter module for PV system sizing and efficiency.

This module handles:
- Inverter sizing based on PV array power
- DC/AC coupling configurations
- Efficiency calculations
"""

import math
from dataclasses import dataclass
from typing import Optional

PVWATTS_REFERENCE_EFFICIENCY = 0.9637
PVWATTS_CURVE_QUADRATIC = -0.0162
PVWATTS_CURVE_LINEAR = 0.9858
PVWATTS_CURVE_CONSTANT = -0.0059


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
    """

    nominal_power_w: Optional[float] = None
    dc_ac_ratio: float = 1.25  # Default 1.25
    inverter_efficiency: float = 0.96
    is_hybrid: bool = True
    mppt_channels: int = 2
    cost_per_kw_simple: float = 48.37  # €/kW for simple grid-tie inverter
    cost_per_kw_hybrid: float = 102.58  # €/kW for hybrid inverter

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


def calculate_dc_ac_power(
    pv_dc_power: float, inverter_ac_power: float, inverter_efficiency: float = 0.96
) -> InverterConversionResult:
    """
    Calculate AC output and loss buckets with the PVWatts part-load curve.

    Clipping is reported on the DC side: power above the DC input required
    to saturate the AC rating is ``clipping_loss_dc_w``. The AC-equivalent
    clipping value is also exposed for reports that compare against
    ``pv_dc_power * inverter_efficiency``.

    Args:
        pv_dc_power: DC power from PV array (W)
        inverter_ac_power: Inverter AC rating (W)
        inverter_efficiency: Nominal inverter efficiency

    Returns:
        InverterConversionResult with AC output and loss buckets.
    """
    pv_dc_power = max(0.0, float(pv_dc_power))
    inverter_ac_power = max(0.0, float(inverter_ac_power))
    inverter_efficiency = min(1.0, max(0.0, float(inverter_efficiency)))

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
        ac_power = pv_dc_power * inverter_efficiency
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
            inverter_ac_power,
            (inverter_efficiency / PVWATTS_REFERENCE_EFFICIENCY)
            * pdc0
            * (PVWATTS_CURVE_QUADRATIC * zeta**2 + PVWATTS_CURVE_LINEAR * zeta + PVWATTS_CURVE_CONSTANT),
        ),
    )
    clipping_loss_dc = max(0.0, pv_dc_power - dc_used)
    conversion_loss = max(0.0, dc_used - ac_power)
    clipping_loss_ac_equiv = clipping_loss_dc * inverter_efficiency

    return InverterConversionResult(
        ac_power_w=ac_power,
        conversion_loss_w=conversion_loss,
        clipping_loss_dc_w=clipping_loss_dc,
        clipping_loss_ac_equivalent_w=clipping_loss_ac_equiv,
    )


def dc_power_for_ac_output(ac_power_w: float, inverter_ac_power: float, inverter_efficiency: float = 0.96) -> float:
    """Return the minimum DC input required for a requested PVWatts AC output.

    The inverse is solved on the monotonic operating range up to the inverter
    nameplate. Requests above the nameplate are clamped to it. Keeping this
    inverse beside :func:`calculate_dc_ac_power` prevents dispatch from
    silently reverting to a flat-efficiency approximation.
    """
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
    return min(upper, max(0.0, zeta * upper))
