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
    custom.Mpp = 545  # Slightly different power (must stay within 2% of Vmp * Imp)

    # Change the STC point together
    from dataclasses import replace
    resized = replace(custom, Mpp=560, Vmp=42.4, Imp=13.21)
"""

import math
from copy import copy
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, ClassVar, Dict, List, Optional

import numpy as np


@dataclass
class PVModuleParams:
    """Datasheet parameters for a PV module.

    Every field holds what the user supplied. The temperature coefficients the
    models use are read-only properties resolved from the current field
    values: ``alpha_sc`` (A/°C), ``beta_voc`` (V/°C) and ``gamma_pmp_effective``
    (%/°C). ``gamma_pmp`` stays ``None`` unless it was given, and
    ``gamma_pmp_effective`` then follows ``T_Pmax_pct``. Because nothing
    derived is stored, in-place edits, ``dataclasses.replace``, a
    ``dataclasses.asdict`` round trip, copies and pickles all see current
    coefficients.

    Field values are validated on construction and on every assignment. A
    rejected assignment leaves the module unchanged. ``Mpp`` must match
    ``Vmp * Imp`` within 2%, so change the STC point together with
    ``dataclasses.replace(module, Mpp=..., Vmp=..., Imp=...)`` when a single
    edit would leave it inconsistent.
    """

    Mpp: float  # W (STC power)
    Vmp: float  # V
    Imp: float  # A
    Voc: float  # V
    Isc: float  # A

    T_Pmax_pct: float  # %/°C
    T_Voc_pct: float  # %/°C
    T_Isc_pct: float  # %/°C

    N_Cells: int  # Number of cells (eg 6*24 or 144)

    Name: Optional[str] = None  # Metadata: specific module model name
    # Module efficiency fraction, e.g. 0.213. Feeds the PVsyst and SAM NOCT
    # cell-temperature models; when unset the PVsyst path uses
    # breos.pv.temperature.DEFAULT_MODULE_EFFICIENCY and noct-sam refuses to run.
    Module_Efficiency: Optional[float] = None
    celltype: str = "monoSi"

    alpha_sc_abs: Optional[float] = None  # A/°C - if provided, overrides T_Isc_pct conversion
    beta_voc_abs: Optional[float] = None  # V/°C - if provided, overrides T_Voc_pct conversion
    gamma_pmp: Optional[float] = None  # %/°C - if provided, overrides T_Pmax_pct
    # Appended after all pre-0.5 fields to preserve positional construction.
    bifaciality: Optional[float] = None  # Metadata: rear/front maximum-power ratio (inert by itself)
    NOCT: Optional[float] = None  # Metadata: nominal operating cell temperature (°C), required by noct-sam

    _POSITIVE_FIELDS = frozenset({"Mpp", "Vmp", "Imp", "Voc", "Isc"})
    _STC_POINT_FIELDS = ("Mpp", "Vmp", "Imp")
    _FINITE_FIELDS = frozenset({"T_Pmax_pct", "T_Voc_pct", "T_Isc_pct", "alpha_sc_abs", "beta_voc_abs", "gamma_pmp"})
    _MPP_RELATIVE_TOLERANCE = 0.02

    # Set per instance once __post_init__ has checked the STC point.
    _module_params_ready: ClassVar[bool] = False

    def __setattr__(self, name: str, value: Any) -> None:
        """Validate a field before it is stored; a rejected edit changes nothing."""
        if name in self._POSITIVE_FIELDS:
            self._validate_number(name, value, minimum=0.0, minimum_strict=True)
        elif name in self._FINITE_FIELDS:
            if value is not None or name in {"T_Pmax_pct", "T_Voc_pct", "T_Isc_pct"}:
                self._validate_number(name, value)
            if name == "T_Pmax_pct" and float(value) >= 0:
                raise ValueError("T_Pmax_pct must be negative")
        elif name == "N_Cells":
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0:
                raise ValueError("N_Cells must be a positive integer")
        elif name == "Module_Efficiency" and value is not None:
            self._validate_number(name, value, minimum=0.0, maximum=1.0, minimum_strict=True)
        elif name == "NOCT" and value is not None:
            self._validate_number(name, value, minimum=0.0, maximum=100.0, minimum_strict=True)
        elif name == "bifaciality" and value is not None:
            try:
                self._validate_number(name, value, minimum=0.0, maximum=1.0, minimum_strict=True)
            except ValueError as exc:
                raise ValueError("bifaciality must be between 0 (exclusive) and 1 (inclusive)") from exc
        elif name == "celltype" and (not isinstance(value, str) or not value.strip()):
            raise ValueError("celltype must be a non-empty string")

        # During __init__ the STC point is checked once, in __post_init__,
        # after all three fields exist.
        if name in self._STC_POINT_FIELDS and self._module_params_ready:
            point = {field: getattr(self, field) for field in self._STC_POINT_FIELDS}
            point[name] = value
            self._validate_stc_point(point["Mpp"], point["Vmp"], point["Imp"])

        object.__setattr__(self, name, value)

    @staticmethod
    def _validate_number(
        name: str,
        value: Any,
        *,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
        minimum_strict: bool = False,
        maximum_strict: bool = False,
    ) -> None:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a finite number")
        number = float(value)
        if minimum is not None and (number <= minimum if minimum_strict else number < minimum):
            bracket = "greater than" if minimum_strict else "at least"
            raise ValueError(f"{name} must be {bracket} {minimum}")
        if maximum is not None and (number >= maximum if maximum_strict else number > maximum):
            bracket = "less than" if maximum_strict else "at most"
            raise ValueError(f"{name} must be {bracket} {maximum}")

    @classmethod
    def _validate_stc_point(cls, mpp: float, vmp: float, imp: float) -> None:
        if not math.isclose(mpp, vmp * imp, rel_tol=cls._MPP_RELATIVE_TOLERANCE):
            raise ValueError(
                f"Mpp must match Vmp * Imp within {cls._MPP_RELATIVE_TOLERANCE:.0%} for a datasheet STC point "
                f"(Mpp={mpp}, Vmp * Imp={vmp * imp:.6g}); use dataclasses.replace() to change "
                "Mpp, Vmp and Imp together"
            )

    @property
    def alpha_sc(self) -> float:
        """Short-circuit-current temperature coefficient in A/°C."""
        if self.alpha_sc_abs is not None:
            return float(self.alpha_sc_abs)
        return float((self.T_Isc_pct * self.Isc) / 100)

    @property
    def beta_voc(self) -> float:
        """Open-circuit-voltage temperature coefficient in V/°C."""
        if self.beta_voc_abs is not None:
            return float(self.beta_voc_abs)
        return float((self.T_Voc_pct * self.Voc) / 100)

    @property
    def gamma_pmp_effective(self) -> float:
        """Maximum-power temperature coefficient in %/°C: ``gamma_pmp`` if set, else ``T_Pmax_pct``."""
        if self.gamma_pmp is not None:
            return float(self.gamma_pmp)
        return float(self.T_Pmax_pct)

    def __post_init__(self) -> None:
        self._validate_stc_point(self.Mpp, self.Vmp, self.Imp)
        object.__setattr__(self, "_module_params_ready", True)


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
        # Suntech EN-STP-Ultra-V-NO3.02-Rev 2022 coefficients for the
        # monofacial C72/Vmh revision represented by these STC values.
        T_Pmax_pct=-0.34,  # %/°C - Power temperature coefficient
        T_Voc_pct=-0.26,  # %/°C - Voltage temperature coefficient
        T_Isc_pct=0.05,  # %/°C - Current temperature coefficient
        N_Cells=6 * 24,  # 144 cells
        Name="Suntech_STP550S-C72/Vmh",
    ),
    # -------------------------------------------------------------------------
    # 445W Mono-Si Module (Used in max_case.py - Erlangen, Germany)
    # No manufacturer or datasheet is recorded for this entry, so its module
    # efficiency is deliberately left unset rather than back-derived from an
    # assumed frame area. The PVsyst temperature models fall back to
    # breos.pv.temperature.DEFAULT_MODULE_EFFICIENCY for it.
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
    # The published LR4-72HPH bins start well above 400 W, so no datasheet
    # efficiency can be quoted for a 400 W unit of that family without inventing
    # a frame area. Left unset: the PVsyst temperature models fall back to
    # breos.pv.temperature.DEFAULT_MODULE_EFFICIENCY.
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
    # Representative mono-PERC specs. The 0.70 maximum-power bifaciality, the
    # -0.34 %/°C power coefficient, and the 21.2 % module efficiency are sourced
    # from Trina Solar's 600 W Vertex TSM-DEG20C.20 datasheet (TSM_EN_2020_PA2),
    # which lists 21.2 % at the 600 W bin of its STC electrical-data table:
    # https://d2fp8gxcp7iq0s.cloudfront.net/documents/jz96t7q3r8tgfwMcpj79rB4LgxqrX9QsmzpaVycj.pdf
    # The remaining electricals stay generic: Trina's module is a 120-cell
    # 210 mm design (Voc 41.7 V), whereas this entry represents the 144-half-cell
    # format. Bifaciality remains inert until bifacial modeling is explicitly
    # activated by the caller.
    # -------------------------------------------------------------------------
    "Generic_600W_Bifacial": PVModuleParams(
        Mpp=600,
        Vmp=44.6,
        Imp=13.46,
        Voc=53.7,
        Isc=14.25,
        celltype="monoSi",
        Module_Efficiency=0.212,  # fraction - Module Efficiency (21.2 %)
        T_Pmax_pct=-0.34,
        T_Voc_pct=-0.26,
        T_Isc_pct=0.046,
        N_Cells=144,
        bifaciality=0.70,
        Name="Generic 600W bifacial (Trina Vertex bifaciality ref)",
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
            return copy(value)

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
Bifaciality: {f"{m.bifaciality * 100:.1f} %" if m.bifaciality is not None else "n/a"}
NOCT:       {f"{m.NOCT:.1f} °C" if m.NOCT is not None else "n/a (not sourced in bundled catalog)"}
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
