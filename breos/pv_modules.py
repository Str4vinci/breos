"""
PV Module Database

Pre-defined PV module specifications for common modules used in simulations.
Add new modules to the MODULES dictionary as needed.

Usage:
    from breos.pv_modules import get_module, list_modules

    # Get a pre-defined module
    pv_params = get_module("Suntech_STP550S_STC")

    # List available modules
    list_modules()

    # Override a parameter
    custom = get_module("Suntech_STP550S_STC")
    custom.Mpp = 545  # Slightly different power
"""

from dataclasses import replace
from typing import Dict, List, Optional

from breos.solar import PVModuleParams

# =============================================================================
# MODULE CATALOG
# =============================================================================

MODULES: Dict[str, PVModuleParams] = {
    "Suntech_STP550S_STC": PVModuleParams(
        Mpp=550,  # W - Maximum Power Point
        Vmp=42.05,  # V - Voltage at MPP
        Imp=13.08,  # A - Current at MPP
        Voc=49.88,  # V - Open circuit voltage
        Isc=14.01,  # A - Short circuit current
        celltype="monoSi",
        Module_Efficiency=0.213,  # fraction - Module Efficiency (21.3 %)
        T_Pmax_pct=-0.36,  # %/°C - Power temperature coefficient
        T_Voc_pct=-0.304,  # %/°C - Voltage temperature coefficient
        T_Isc_pct=0.05,  # %/°C - Current temperature coefficient
        N_Cells=6 * 24,  # 144 cells
        Name="Suntech_STP550S-C72/Vmh",
    ),
    # -------------------------------------------------------------------------
    # 445W Mono-Si Module (Used in max_case.py - Erlangen, Germany)
    # -------------------------------------------------------------------------
    "Erlangen_445W": PVModuleParams(
        Mpp=445,  # W - Maximum Power Point
        Vmp=44.3,  # V - Voltage at MPP
        Imp=10.05,  # A - Current at MPP
        Voc=52.6,  # V - Open circuit voltage
        Isc=10.71,  # A - Short circuit current
        celltype="monoSi",
        T_Pmax_pct=-0.30,  # %/°C - Power temperature coefficient
        T_Voc_pct=-0.24,  # %/°C - Voltage temperature coefficient
        T_Isc_pct=0.04,  # %/°C - Current temperature coefficient
        N_Cells=144,
    ),
    # -------------------------------------------------------------------------
    # Generic 400W Module (common 72-cell / 144-half-cell residential panel)
    # Representative mono-PERC specs (LONGi LR4-72HPH-400M family).
    # -------------------------------------------------------------------------
    "Generic_400W": PVModuleParams(
        Mpp=400,
        Vmp=41.0,  # V - Voltage at MPP
        Imp=9.76,  # A - Current at MPP
        Voc=49.3,  # V - Open circuit voltage
        Isc=10.30,  # A - Short circuit current
        celltype="monoSi",
        T_Pmax_pct=-0.35,  # %/°C - Power temperature coefficient
        T_Voc_pct=-0.265,  # %/°C - Voltage temperature coefficient
        T_Isc_pct=0.05,  # %/°C - Current temperature coefficient
        N_Cells=144,  # 72-cell module (144 half-cells)
        Name="Generic 400W (LONGi LR4-72HPH-400M ref)",
    ),
    # -------------------------------------------------------------------------
    # Generic 600W Bifacial Module (utility-scale, high-voltage 144-half-cell)
    # Representative mono-PERC specs; the model treats it as a front-side module.
    # -------------------------------------------------------------------------
    "Generic_600W_Bifacial": PVModuleParams(
        Mpp=600,
        Vmp=44.6,
        Imp=13.46,
        Voc=53.7,
        Isc=14.25,
        celltype="monoSi",
        T_Pmax_pct=-0.34,
        T_Voc_pct=-0.26,
        T_Isc_pct=0.046,
        N_Cells=144,
        Name="Generic 600W bifacial (utility-scale ref)",
    ),
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_module(name: str) -> PVModuleParams:
    """
    Get a PV module by name from the catalog.

    Args:
        name: Module name (case-insensitive)

    Returns:
        PVModuleParams object (copy, safe to modify)

    Raises:
        KeyError: If module name not found

    Example:
        >>> pv_params = get_module("Suntech_STP550S_STC")
        >>> pv_params.Mpp
        550
    """
    # Case-insensitive lookup
    name_lower = name.lower()
    for key, value in MODULES.items():
        if key.lower() == name_lower:
            # Return a copy so user can modify without affecting catalog
            return replace(value)

    available = ", ".join(MODULES.keys())
    raise KeyError(f"Module '{name}' not found. Available: {available}")


def list_modules() -> List[str]:
    """
    List all available module names.

    Returns:
        List of module names

    Example:
        >>> list_modules()
        ['Suntech_STP550S_STC', 'Erlangen_445W', 'Generic_400W', 'Generic_600W_Bifacial']
    """
    return list(MODULES.keys())


def get_module_info(name: str) -> str:
    """
    Get a formatted string with module specifications.

    Args:
        name: Module name

    Returns:
        Formatted string with module info
    """
    m = get_module(name)
    return f"""
{name}
{"=" * len(name)}
Power:      {m.Mpp} W
Vmp:        {m.Vmp} V
Imp:        {m.Imp} A
Voc:        {m.Voc} V
Isc:        {m.Isc} A
Cell Type:  {m.celltype}
Cells:      {m.N_Cells}
T_Pmax:     {m.T_Pmax_pct} %/°C
Name:       {m.Name}
Efficiency: {f"{m.Module_Efficiency * 100:.1f} %" if m.Module_Efficiency is not None else "n/a"}
"""


def add_module(name: str, params: PVModuleParams) -> None:
    """
    Add a new module to the catalog (runtime only, not persisted).

    Args:
        name: Module name
        params: PVModuleParams object

    Example:
        >>> add_module("Custom_500W", PVModuleParams(Mpp=500, ...))
    """
    MODULES[name] = params
