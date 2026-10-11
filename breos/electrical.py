"""Check a declared PV string topology against module and inverter datasheet limits.

The check is pure: it reads datasheet values, design temperatures and a topology
and returns structured findings. It does not change the topology, propose one,
or affect simulated energy. Severities separate three kinds of finding:

- ``error``: a hard limit is broken (overvoltage, short-circuit current, a
  string too short to track or start, a wiring inconsistency). The design is
  not buildable as declared.
- ``warning``: an operating limit is exceeded. The inverter holds its limit and
  some energy is lost, but the design is safe.
- ``missing``: a datasheet value the check needs is unknown, so that check
  could not run. A missing value is never replaced by a default.

Temperatures follow the usual string-sizing rules. Cold Voc uses the coldest
design ambient, when no irradiance warms the cell. Hot Vmp, hot Voc and the
design Isc use the hottest ambient plus a mounting-dependent rise. The upper MPPT
check reads Vmp at the cold operating cell, the coldest ambient plus the same
rise, because the module is producing only when irradiance warms it. Vmp moves
with ``PVModuleParams.T_Vmp_pct_effective``, which derives the coefficient from
the power and current ones when the datasheet does not publish it.
"""

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal

from breos.equipment import InverterSpec
from breos.pv_modules import PVModuleParams

#: Cell temperature above ambient at full irradiance, by mounting, in °C.
#: Industry rules of thumb: flush-mounted modules run hottest.
MOUNTING_TEMPERATURE_RISE_C = {"flush": 30.0, "tilted_rack": 25.0, "ground_mount": 20.0}

#: Rear over front irradiance for a bifacial module's design Isc: 135 W/m²
#: behind 1000 W/m², the bifacial nameplate irradiance of IEC TS 60904-1-2.
BIFACIAL_REAR_IRRADIANCE_RATIO = 0.135

Severity = Literal["error", "warning", "missing"]
Status = Literal["pass", "warning", "incomplete", "fail"]

_STC_C = 25.0
_TOLERANCE = 1e-9


@dataclass(frozen=True)
class ElectricalEnvironment:
    """Site design temperatures in °C and the module mounting.

    ``min_design_temp_c`` is the coldest design ambient (for example the
    ASHRAE 99.6 % extreme minimum) and ``max_ambient_temp_c`` the hottest.
    ``mounting`` is a key of :data:`MOUNTING_TEMPERATURE_RISE_C`.
    """

    min_design_temp_c: float
    max_ambient_temp_c: float
    mounting: str = "tilted_rack"

    def __post_init__(self) -> None:
        for name in ("min_design_temp_c", "max_ambient_temp_c"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if self.min_design_temp_c >= self.max_ambient_temp_c:
            raise ValueError("min_design_temp_c must be below max_ambient_temp_c")
        if self.mounting not in MOUNTING_TEMPERATURE_RISE_C:
            raise ValueError(f"mounting must be one of {sorted(MOUNTING_TEMPERATURE_RISE_C)}, got {self.mounting!r}")

    @property
    def cold_cell_c(self) -> float:
        """Cell temperature for cold Voc: the cold ambient, without irradiance."""
        return float(self.min_design_temp_c)

    @property
    def cold_operating_cell_c(self) -> float:
        """Coldest cell temperature while producing, for the upper MPPT check."""
        return float(self.min_design_temp_c + MOUNTING_TEMPERATURE_RISE_C[self.mounting])

    @property
    def hot_cell_c(self) -> float:
        """Hottest cell temperature, for hot Vmp, hot Voc and the design Isc."""
        return float(self.max_ambient_temp_c + MOUNTING_TEMPERATURE_RISE_C[self.mounting])


@dataclass(frozen=True)
class PVString:
    """``parallel`` identical series strings on one MPPT input.

    Each string holds ``modules`` modules in series, all from PV array
    ``array``, and connects to input ``mppt`` of inverter ``inverter``. The
    three references are 0-based positions in the arrays, the inverters and
    that inverter's ``mppts``.
    """

    inverter: int
    mppt: int
    array: int
    modules: int
    parallel: int = 1

    def __post_init__(self) -> None:
        for name in ("inverter", "mppt", "array", "modules", "parallel"):
            value = getattr(self, name)
            minimum = 1 if name in ("modules", "parallel") else 0
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}")


@dataclass(frozen=True)
class ElectricalFinding:
    """One result of a topology check.

    ``value`` and ``limit`` are in the unit the message names (V, A, W or a
    count). ``inverter``, ``mppt``, ``string`` and ``array`` locate the finding
    as 0-based positions; ``string`` indexes the declared strings. ``field``
    names the datasheet value a ``missing`` finding needs.
    """

    code: str
    severity: Severity
    message: str
    value: float | None = None
    limit: float | None = None
    inverter: int | None = None
    mppt: int | None = None
    string: int | None = None
    array: int | None = None
    field: str | None = None


@dataclass(frozen=True)
class TopologyCheck:
    """All findings of one check, in a stable order."""

    findings: tuple[ElectricalFinding, ...]

    @property
    def status(self) -> Status:
        """``fail`` on any error, else ``incomplete`` on missing data, else ``warning`` or ``pass``."""
        severities = {finding.severity for finding in self.findings}
        if "error" in severities:
            return "fail"
        if "missing" in severities:
            return "incomplete"
        return "warning" if "warning" in severities else "pass"


def _at(stc_value: float, coefficient_pct_per_c: float, cell_c: float) -> float:
    return stc_value * (1.0 + (cell_c - _STC_C) * coefficient_pct_per_c / 100.0)


def _design_isc(module: PVModuleParams, environment: ElectricalEnvironment) -> float:
    """Hot-cell Isc, never below STC, plus the rear gain of a bifacial module."""
    isc_hot = max(module.Isc, _at(module.Isc, module.T_Isc_pct, environment.hot_cell_c))
    return isc_hot * (1.0 + (module.bifaciality or 0.0) * BIFACIAL_REAR_IRRADIANCE_RATIO)


class _Findings:
    def __init__(self) -> None:
        self.items: list[ElectricalFinding] = []
        self._missing: set[tuple[str, int | None, int | None, int | None]] = set()

    def add(self, code: str, severity: Severity, message: str, **where: float | int | str | None) -> None:
        self.items.append(ElectricalFinding(code, severity, message, **where))  # type: ignore[arg-type]

    def missing(
        self, field: str, owner: str, *, inverter: int | None = None, mppt: int | None = None, array: int | None = None
    ) -> None:
        key = (field, inverter, mppt, array)
        if key not in self._missing:
            self._missing.add(key)
            self.add(
                "missing_value",
                "missing",
                f"{owner} has no {field}; the checks that need it did not run.",
                inverter=inverter,
                mppt=mppt,
                array=array,
                field=field,
            )


def check_topology(
    arrays: Sequence[tuple[PVModuleParams, int]],
    inverters: Sequence[InverterSpec],
    strings: Sequence[PVString],
    environment: ElectricalEnvironment,
    *,
    dc_ac_ratio_range: tuple[float, float] | None = None,
) -> TopologyCheck:
    """Check declared strings against the arrays, the inverters and the site.

    ``arrays`` lists each PV array's module and module count. Every module
    must be in exactly one string: an array with modules left over, or with
    more modules strung than it has, is an error. Exclude a module by lowering
    its array's count. ``dc_ac_ratio_range`` optionally warns when an
    inverter's connected STC power over its AC nameplate falls outside
    ``(low, high)``.
    """
    found = _Findings()
    if not inverters:
        found.add("no_inverters", "error", "No inverter is declared.")
    if not strings:
        found.add("no_strings", "error", "No string is declared.")

    strung = [0] * len(arrays)
    on_input: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, string in enumerate(strings):
        where = {"inverter": string.inverter, "mppt": string.mppt, "string": index, "array": string.array}
        if string.array >= len(arrays):
            found.add(
                "unknown_array", "error", f"String {index} names array {string.array}, which does not exist.", **where
            )
            continue
        strung[string.array] += string.modules * string.parallel
        if string.inverter >= len(inverters):
            found.add(
                "unknown_inverter",
                "error",
                f"String {index} names inverter {string.inverter}, which does not exist.",
                **where,
            )
            continue
        inverter = inverters[string.inverter]
        if inverter.mppts is None:
            found.missing("mppts", f"Inverter {string.inverter}", inverter=string.inverter)
        elif string.mppt >= len(inverter.mppts):
            found.add(
                "unknown_mppt",
                "error",
                f"String {index} uses MPPT {string.mppt}, but inverter {string.inverter} has {len(inverter.mppts)}.",
                value=string.mppt,
                limit=len(inverter.mppts) - 1,
                **where,
            )
            continue
        _check_string(found, index, string, arrays[string.array][0], inverter, environment)
        if inverter.mppts is not None:
            on_input[(string.inverter, string.mppt)].append(index)

    for index, (_, count) in enumerate(arrays):
        if strung[index] < count:
            found.add(
                "unconnected_modules",
                "error",
                f"Array {index} has {count - strung[index]} of its {count} modules in no string; "
                "lower its module count to exclude them.",
                value=count - strung[index],
                limit=0,
                array=index,
            )
        elif strung[index] > count:
            found.add(
                "array_overcommitted",
                "error",
                f"Strings take {strung[index]} modules from array {index}, which has {count}.",
                value=strung[index],
                limit=count,
                array=index,
            )

    for (inverter_index, mppt_index), members in on_input.items():
        _check_mppt(found, inverter_index, mppt_index, [strings[i] for i in members], arrays, inverters, environment)

    for inverter_index, inverter in enumerate(inverters):
        _check_inverter(found, inverter_index, inverter, strings, arrays, dc_ac_ratio_range)
    return TopologyCheck(tuple(found.items))


def _check_string(
    found: _Findings,
    index: int,
    string: PVString,
    module: PVModuleParams,
    inverter: InverterSpec,
    environment: ElectricalEnvironment,
) -> None:
    where = {"inverter": string.inverter, "mppt": string.mppt, "string": index, "array": string.array}
    n = string.modules

    ceilings = []
    if inverter.max_dc_voltage_v is None:
        found.missing("max_dc_voltage_v", f"Inverter {string.inverter}", inverter=string.inverter)
    else:
        ceilings.append(inverter.max_dc_voltage_v)
    if module.Max_System_Voltage is None:
        found.missing("Max_System_Voltage", f"The module of array {string.array}", array=string.array)
    else:
        ceilings.append(module.Max_System_Voltage)
    voc_cold = n * _at(module.Voc, module.T_Voc_pct, environment.cold_cell_c)
    if ceilings and voc_cold > min(ceilings) + _TOLERANCE:
        found.add(
            "cold_voc_above_limit",
            "error",
            f"String {index}: cold Voc is {voc_cold:.1f} V, above the {min(ceilings):.1f} V DC limit.",
            value=voc_cold,
            limit=min(ceilings),
            **where,
        )

    if inverter.startup_voltage_v is None:
        found.missing("startup_voltage_v", f"Inverter {string.inverter}", inverter=string.inverter)
    else:
        voc_hot = n * _at(module.Voc, module.T_Voc_pct, environment.hot_cell_c)
        if voc_hot < inverter.startup_voltage_v - _TOLERANCE:
            found.add(
                "hot_voc_below_startup",
                "error",
                f"String {index}: hot-cell Voc is {voc_hot:.1f} V, below the {inverter.startup_voltage_v:.1f} V "
                "startup voltage, so the inverter cannot restart on the hottest days.",
                value=voc_hot,
                limit=inverter.startup_voltage_v,
                **where,
            )

    vmp_coefficient = module.T_Vmp_pct_effective
    if inverter.min_mppt_voltage_v is None:
        found.missing("min_mppt_voltage_v", f"Inverter {string.inverter}", inverter=string.inverter)
    else:
        vmp_hot = n * _at(module.Vmp, vmp_coefficient, environment.hot_cell_c)
        if vmp_hot < inverter.min_mppt_voltage_v - _TOLERANCE:
            found.add(
                "hot_vmp_below_mppt_min",
                "error",
                f"String {index}: hot-cell Vmp is {vmp_hot:.1f} V, below the {inverter.min_mppt_voltage_v:.1f} V "
                "MPPT minimum, so tracking drops out on the hottest days.",
                value=vmp_hot,
                limit=inverter.min_mppt_voltage_v,
                **where,
            )
    if inverter.max_mppt_voltage_v is None:
        found.missing("max_mppt_voltage_v", f"Inverter {string.inverter}", inverter=string.inverter)
    else:
        vmp_cold = n * _at(module.Vmp, vmp_coefficient, environment.cold_operating_cell_c)
        if vmp_cold > inverter.max_mppt_voltage_v + _TOLERANCE:
            found.add(
                "vmp_above_mppt_max",
                "warning",
                f"String {index}: Vmp at the cold operating cell is {vmp_cold:.1f} V, above the "
                f"{inverter.max_mppt_voltage_v:.1f} V MPPT maximum; the inverter limits its voltage and loses yield.",
                value=vmp_cold,
                limit=inverter.max_mppt_voltage_v,
                **where,
            )


def _check_mppt(
    found: _Findings,
    inverter_index: int,
    mppt_index: int,
    members: list[PVString],
    arrays: Sequence[tuple[PVModuleParams, int]],
    inverters: Sequence[InverterSpec],
    environment: ElectricalEnvironment,
) -> None:
    inverter = inverters[inverter_index]
    assert inverter.mppts is not None
    spec = inverter.mppts[mppt_index]
    where = {"inverter": inverter_index, "mppt": mppt_index}
    owner = f"MPPT {mppt_index} of inverter {inverter_index}"
    parallel = sum(string.parallel for string in members)

    if len({(string.array, string.modules) for string in members}) > 1:
        found.add(
            "parallel_strings_mismatch",
            "error",
            f"{owner}: parallel strings must have the same length and come from the same array.",
            **where,
        )

    if spec.max_strings is None:
        found.missing("max_strings", owner, **where)
    elif parallel > spec.max_strings:
        found.add(
            "too_many_strings",
            "error",
            f"{owner} has {parallel} strings, above its limit of {spec.max_strings}.",
            value=parallel,
            limit=spec.max_strings,
            **where,
        )

    factor = inverter.isc_comparison_factor or 1.0
    isc = factor * sum(string.parallel * _design_isc(arrays[string.array][0], environment) for string in members)
    if spec.max_short_circuit_current_a is None:
        found.missing("max_short_circuit_current_a", owner, **where)
    elif isc > spec.max_short_circuit_current_a + _TOLERANCE:
        found.add(
            "isc_above_mppt_limit",
            "error",
            f"{owner}: design Isc is {isc:.2f} A, above its {spec.max_short_circuit_current_a:.2f} A rating.",
            value=isc,
            limit=spec.max_short_circuit_current_a,
            **where,
        )

    imp = sum(string.parallel * arrays[string.array][0].Imp for string in members)
    if spec.max_input_current_a is None:
        found.missing("max_input_current_a", owner, **where)
    elif imp > spec.max_input_current_a + _TOLERANCE:
        found.add(
            "imp_above_mppt_limit",
            "warning",
            f"{owner}: operating current is {imp:.2f} A, above its {spec.max_input_current_a:.2f} A input limit; "
            "the inverter limits its current and loses yield near peak production.",
            value=imp,
            limit=spec.max_input_current_a,
            **where,
        )


def _check_inverter(
    found: _Findings,
    index: int,
    inverter: InverterSpec,
    strings: Sequence[PVString],
    arrays: Sequence[tuple[PVModuleParams, int]],
    dc_ac_ratio_range: tuple[float, float] | None,
) -> None:
    connected = [s for s in strings if s.inverter == index and s.array < len(arrays)]
    if not connected:
        return
    stc_w = sum(s.modules * s.parallel * arrays[s.array][0].Mpp for s in connected)
    if inverter.max_dc_power_w is not None and stc_w > inverter.max_dc_power_w + _TOLERANCE:
        found.add(
            "dc_power_above_max",
            "warning",
            f"Inverter {index}: {stc_w:.0f} W of modules at STC exceeds its {inverter.max_dc_power_w:.0f} W DC input.",
            value=stc_w,
            limit=inverter.max_dc_power_w,
            inverter=index,
        )
    if dc_ac_ratio_range is not None:
        low, high = dc_ac_ratio_range
        ratio = stc_w / inverter.nominal_power_w
        if not low - _TOLERANCE <= ratio <= high + _TOLERANCE:
            found.add(
                "dc_ac_ratio_outside_range",
                "warning",
                f"Inverter {index}: DC/AC ratio is {ratio:.2f}, outside {low:.2f} to {high:.2f}.",
                value=ratio,
                limit=low if ratio < low else high,
                inverter=index,
            )
