"""
Inverter module for PV system sizing and efficiency.

This module handles:
- Inverter sizing based on PV array power
- DC/AC coupling configurations
- Efficiency calculations
"""

from dataclasses import dataclass, field
from typing import Optional

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
    dc_ac_ratio: float = 1.25         # Default 1.25
    inverter_efficiency: float = 0.96
    is_hybrid: bool = True
    mppt_channels: int = 2
    cost_per_kw_simple: float = 48.37    # €/kW for simple grid-tie inverter
    cost_per_kw_hybrid: float = 102.58   # €/kW for hybrid inverter
    
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
    - residential_hybrid: 1.1 ratio, hybrid
    - residential_simple: 1.1 ratio, grid-tie only
    - commercial_hybrid: 1.25 ratio, hybrid
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


def calculate_dc_ac_efficiency(
    pv_dc_power: float,
    inverter_ac_power: float,
    inverter_efficiency: float = 0.96
) -> float:
    """
    Calculate AC output considering inverter clipping.
    
    If DC power exceeds inverter rating, clips to inverter limit.
    
    Args:
        pv_dc_power: DC power from PV array (W)
        inverter_ac_power: Inverter AC rating (W)
        inverter_efficiency: Inverter efficiency at MPP
        
    Returns:
        AC power output (W)
    """
    max_ac = inverter_ac_power
    theoretical_ac = pv_dc_power * inverter_efficiency
    return min(theoretical_ac, max_ac)
