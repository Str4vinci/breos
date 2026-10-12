"""Check a declared PV string topology against module and inverter datasheet limits.

The check is pure: it reads datasheet values, design temperatures and a topology
and returns structured findings. It does not change the topology or propose
one. Severities separate three kinds of finding (:data:`FINDING_CODES` lists
every code):

- ``error``: a hard limit is broken (voltage rating, short-circuit current,
  string count, maximum generator power) or the wiring is inconsistent. The
  design is not buildable as declared.
- ``warning``: an operating or yield limit is exceeded. The equipment is safe,
  but it loses energy or may not restart on the hottest days.
- ``missing``: a datasheet value the check needs is unknown, so that check
  could not run. A missing value is never replaced by a default.

Temperatures follow the usual string-sizing rules. Cold Voc uses the extreme
annual minimum ambient, before appreciable solar heating. Hot Vmp, hot Voc and
the hot-cell Isc use the hottest ambient plus a mounting-dependent screening
rise, and the upper MPPT check reads Vmp at the cold operating cell, the
coldest ambient plus the same rise; a caller may state both cell temperatures
instead. Each array may set its own mounting and cell temperatures
(:class:`ArrayMounting`), since a rack and a flush roof on one site differ by
several degrees. Voc and Isc move with the coefficients the energy model uses
(``beta_voc`` and ``alpha_sc``, which honour absolute overrides), and Vmp with
``PVModuleParams.T_Vmp_pct_effective``, which derives the coefficient when the
datasheet does not publish it.

:func:`string_limited_dc` is the energy side: it applies each MPPT input's
voltage, current and power limits, the inverter's power limit and its start-up
voltage to the production of the wired strings.
"""

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from numbers import Integral, Real
from typing import Any, Literal

import numpy as np
import pandas as pd
import pvlib

from breos.equipment import InverterSpec
from breos.pv_modules import PVModuleParams

#: Cell temperature above ambient at full irradiance, by mounting, in °C.
#: Screening values from common string-sizing practice; they do not bound
#: every mounting and wind condition. State ``hot_cell_temp_c`` and
#: ``cold_operating_cell_temp_c`` to override them.
MOUNTING_TEMPERATURE_RISE_C = {"flush": 30.0, "tilted_rack": 25.0, "ground_mount": 20.0}

#: Default rear irradiance for a bifacial module's design Isc, as a fraction of
#: 1000 W/m² front: 135 W/m², the bifacial nameplate irradiance of IEC TS
#: 60904-1-2. ``ElectricalEnvironment.bifacial_rear_irradiance_w_m2`` overrides it.
BIFACIAL_REAR_IRRADIANCE_RATIO = 0.135

Severity = Literal["error", "warning", "missing"]
Status = Literal["pass", "warning", "incomplete", "fail"]

_STC_C = 25.0
_TOLERANCE = 1e-9


@dataclass(frozen=True)
class FindingKind:
    """The fixed contract of one finding code.

    ``unit`` is the unit of ``value`` and ``limit`` (``"V"``, ``"A"``, ``"W"``,
    ``"count"`` or ``"ratio"``), or None when the finding has neither.
    ``located_by`` lists the location fields the finding sets; the others are
    None.
    """

    severity: Severity
    unit: str | None
    located_by: tuple[str, ...]


_STRING = ("inverter", "mppt", "string", "array")
_INPUT = ("inverter", "mppt")

#: Every finding code with its severity, unit and location fields. Codes and
#: severities are stable; a new code may be added in a later release.
FINDING_CODES: dict[str, FindingKind] = {
    "no_inverters": FindingKind("error", None, ()),
    "no_strings": FindingKind("error", None, ()),
    "unknown_array": FindingKind("error", None, _STRING),
    "unknown_inverter": FindingKind("error", None, _STRING),
    "unknown_mppt": FindingKind("error", None, _STRING),
    "unconnected_modules": FindingKind("error", "count", ("array",)),
    "array_overcommitted": FindingKind("error", "count", ("array",)),
    "cold_voc_above_limit": FindingKind("error", "V", _STRING),
    "hot_voc_below_startup": FindingKind("warning", "V", _STRING),
    "hot_vmp_below_mppt_min": FindingKind("warning", "V", _STRING),
    "vmp_above_mppt_max": FindingKind("warning", "V", _STRING),
    "parallel_strings_mismatch": FindingKind("error", None, _INPUT),
    "too_many_strings": FindingKind("error", "count", _INPUT),
    "isc_above_mppt_limit": FindingKind("error", "A", _INPUT),
    "imp_above_mppt_limit": FindingKind("warning", "A", _INPUT),
    "isc_above_string_limit": FindingKind("error", "A", _STRING),
    "imp_above_string_limit": FindingKind("warning", "A", _STRING),
    "mppt_dc_power_above_max": FindingKind("error", "W", _INPUT),
    "isc_above_inverter_limit": FindingKind("error", "A", ("inverter",)),
    "dc_power_above_max": FindingKind("error", "W", ("inverter",)),
    "dc_power_above_recommended": FindingKind("warning", "W", ("inverter",)),
    "dc_ac_ratio_outside_range": FindingKind("warning", "ratio", ("inverter",)),
    "missing_value": FindingKind("missing", None, ()),
}


@dataclass(frozen=True)
class ElectricalEnvironment:
    """Site design temperatures in °C, the module mounting and bifacial design irradiance.

    ``min_design_temp_c`` is the extreme annual minimum ambient the design must
    withstand, from the basis the jurisdiction or manufacturer requires;
    ``max_ambient_temp_c`` is the hottest design ambient. ``mounting`` is a key
    of :data:`MOUNTING_TEMPERATURE_RISE_C`, whose rise gives the hot and cold
    operating cell temperatures unless ``hot_cell_temp_c`` or
    ``cold_operating_cell_temp_c`` states them. ``bifacial_rear_irradiance_w_m2``
    is the rear irradiance behind 1000 W/m² front for a bifacial design Isc.
    """

    min_design_temp_c: float
    max_ambient_temp_c: float
    mounting: str = "tilted_rack"
    hot_cell_temp_c: float | None = None
    cold_operating_cell_temp_c: float | None = None
    bifacial_rear_irradiance_w_m2: float = 1000.0 * BIFACIAL_REAR_IRRADIANCE_RATIO

    def __post_init__(self) -> None:
        for name in ("min_design_temp_c", "max_ambient_temp_c", "bifacial_rear_irradiance_w_m2"):
            _finite(name, getattr(self, name))
        for name in ("hot_cell_temp_c", "cold_operating_cell_temp_c"):
            if getattr(self, name) is not None:
                _finite(name, getattr(self, name))
        if self.min_design_temp_c >= self.max_ambient_temp_c:
            raise ValueError("min_design_temp_c must be below max_ambient_temp_c")
        if self.mounting not in MOUNTING_TEMPERATURE_RISE_C:
            raise ValueError(f"mounting must be one of {sorted(MOUNTING_TEMPERATURE_RISE_C)}, got {self.mounting!r}")
        if self.bifacial_rear_irradiance_w_m2 < 0:
            raise ValueError("bifacial_rear_irradiance_w_m2 must be at least 0")

    @property
    def cold_cell_c(self) -> float:
        """Cell temperature for cold Voc: the cold ambient, before appreciable solar heating."""
        return float(self.min_design_temp_c)

    @property
    def cold_operating_cell_c(self) -> float:
        """Coldest cell temperature while producing, for the upper MPPT check."""
        if self.cold_operating_cell_temp_c is not None:
            return float(self.cold_operating_cell_temp_c)
        return float(self.min_design_temp_c + MOUNTING_TEMPERATURE_RISE_C[self.mounting])

    @property
    def hot_cell_c(self) -> float:
        """Hottest cell temperature, for hot Vmp, hot Voc and the hot-cell Isc."""
        if self.hot_cell_temp_c is not None:
            return float(self.hot_cell_temp_c)
        return float(self.max_ambient_temp_c + MOUNTING_TEMPERATURE_RISE_C[self.mounting])


@dataclass(frozen=True)
class ArrayMounting:
    """One array's mounting and cell temperatures, overriding the site's for its strings.

    Each field left None inherits from the :class:`ElectricalEnvironment`. The
    most specific value wins: a temperature the array states, then the rise of
    the array's ``mounting``, then a temperature the site states, then the rise
    of the site's ``mounting``. So an array that states its ``mounting`` takes
    its cell temperatures from that rise, not from the site's stated ones.
    """

    mounting: str | None = None
    hot_cell_temp_c: float | None = None
    cold_operating_cell_temp_c: float | None = None

    def __post_init__(self) -> None:
        for name in ("hot_cell_temp_c", "cold_operating_cell_temp_c"):
            if getattr(self, name) is not None:
                _finite(name, getattr(self, name))
        if self.mounting is not None and self.mounting not in MOUNTING_TEMPERATURE_RISE_C:
            raise ValueError(f"mounting must be one of {sorted(MOUNTING_TEMPERATURE_RISE_C)}, got {self.mounting!r}")

    def applied_to(self, environment: ElectricalEnvironment) -> ElectricalEnvironment:
        """The site ``environment`` as this array sees it."""
        if self.mounting is not None:
            environment = replace(
                environment, mounting=self.mounting, hot_cell_temp_c=None, cold_operating_cell_temp_c=None
            )
        return replace(
            environment,
            hot_cell_temp_c=_first_set(self.hot_cell_temp_c, environment.hot_cell_temp_c),
            cold_operating_cell_temp_c=_first_set(
                self.cold_operating_cell_temp_c, environment.cold_operating_cell_temp_c
            ),
        )


def _first_set(value: float | None, fallback: float | None) -> float | None:
    return fallback if value is None else value


def _finite(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")


@dataclass(frozen=True)
class PVString:
    """``parallel`` identical series strings on one MPPT input.

    Each string holds ``modules`` modules in series, all from PV array
    ``array``, and connects to input ``mppt`` of inverter ``inverter``. An input
    is an independent tracker. The three references are 0-based positions in
    the arrays, the inverters and that inverter's ``mppts``.
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
    """One result of a topology check; :data:`FINDING_CODES` describes each ``code``.

    ``value`` and ``limit`` are in ``unit``. ``inverter``, ``mppt``,
    ``string`` and ``array`` locate the finding as 0-based positions; ``string``
    indexes the declared strings. ``limit_field`` names the datasheet value the
    limit came from, and ``field`` the datasheet value a ``missing`` finding
    needs.
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
    unit: str | None = None
    limit_field: str | None = None


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


def _voc(module: PVModuleParams, cell_c: float) -> float:
    """Module Voc at ``cell_c``, with the coefficient the energy model uses."""
    return float(module.Voc + module.beta_voc * (cell_c - _STC_C))


def _vmp(module: PVModuleParams, cell_c: float) -> float:
    return float(module.Vmp * (1.0 + (cell_c - _STC_C) * module.T_Vmp_pct_effective / 100.0))


def _rear_gain(module: PVModuleParams, environment: ElectricalEnvironment) -> float:
    return 1.0 + (module.bifaciality or 0.0) * environment.bifacial_rear_irradiance_w_m2 / 1000.0


def _isc_stc(module: PVModuleParams, environment: ElectricalEnvironment) -> float:
    """STC Isc plus the rear gain of a bifacial module."""
    return float(module.Isc * _rear_gain(module, environment))


def _isc_hot(module: PVModuleParams, environment: ElectricalEnvironment) -> float:
    """Hot-cell Isc, never below STC, plus the rear gain of a bifacial module."""
    isc_hot = max(module.Isc, module.Isc + module.alpha_sc * (environment.hot_cell_c - _STC_C))
    return float(isc_hot * _rear_gain(module, environment))


def _design_isc(module: PVModuleParams, factor: float | None, environment: ElectricalEnvironment) -> float:
    """One string's design Isc: the manufacturer's rule on STC Isc or the hot-cell Isc, whichever is larger.

    An unknown factor leaves the hot-cell Isc, the physical maximum; the check
    reports the factor missing.
    """
    return max((factor or 0.0) * _isc_stc(module, environment), _isc_hot(module, environment))


class _Findings:
    def __init__(self) -> None:
        self.items: list[ElectricalFinding] = []
        self._missing: set[tuple[str, int | None, int | None, int | None]] = set()

    def add(self, code: str, message: str, **where: Any) -> None:
        kind = FINDING_CODES[code]
        self.items.append(ElectricalFinding(code, kind.severity, message, unit=kind.unit, **where))

    def missing(
        self, field: str, owner: str, *, inverter: int | None = None, mppt: int | None = None, array: int | None = None
    ) -> None:
        key = (field, inverter, mppt, array)
        if key not in self._missing:
            self._missing.add(key)
            self.add(
                "missing_value",
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
    mountings: Sequence[ArrayMounting | None] | None = None,
) -> TopologyCheck:
    """Check declared strings against the arrays, the inverters and the site.

    ``arrays`` lists each PV array's module and module count. Every module
    must be in exactly one string: an array with modules left over, or with
    more modules strung than it has, is an error. Exclude a module by lowering
    its array's count. ``dc_ac_ratio_range`` optionally warns when an
    inverter's connected STC power over its AC nameplate falls outside
    ``(low, high)``. ``mountings`` optionally gives each array, by position,
    its own :class:`ArrayMounting`; None inherits the site's.
    :data:`FINDING_CODES` lists every finding.
    """
    if mountings is not None and len(mountings) != len(arrays):
        raise ValueError(f"mountings has {len(mountings)} entries for {len(arrays)} arrays")
    sites = [
        environment if mounting is None else mounting.applied_to(environment)
        for mounting in (mountings if mountings is not None else [None] * len(arrays))
    ]
    found = _Findings()
    if not inverters:
        found.add("no_inverters", "No inverter is declared.")
    if not strings:
        found.add("no_strings", "No string is declared.")

    strung = [0] * len(arrays)
    on_input: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, string in enumerate(strings):
        where = {"inverter": string.inverter, "mppt": string.mppt, "string": index, "array": string.array}
        if string.array >= len(arrays):
            found.add("unknown_array", f"String {index} names array {string.array}, which does not exist.", **where)
            continue
        strung[string.array] += string.modules * string.parallel
        if string.inverter >= len(inverters):
            found.add(
                "unknown_inverter", f"String {index} names inverter {string.inverter}, which does not exist.", **where
            )
            continue
        inverter = inverters[string.inverter]
        if inverter.mppts is None:
            found.missing("mppts", f"Inverter {string.inverter}", inverter=string.inverter)
        elif string.mppt >= len(inverter.mppts):
            found.add(
                "unknown_mppt",
                f"String {index} uses MPPT {string.mppt}, but inverter {string.inverter} has {len(inverter.mppts)}.",
                **where,
            )
            continue
        _check_string(found, index, string, arrays[string.array][0], inverter, sites[string.array])
        if inverter.mppts is not None:
            on_input[(string.inverter, string.mppt)].append(index)

    for index, (_, count) in enumerate(arrays):
        if strung[index] < count:
            found.add(
                "unconnected_modules",
                f"Array {index} has {count - strung[index]} of its {count} modules in no string; "
                "lower its module count to exclude them.",
                value=count - strung[index],
                limit=0,
                array=index,
            )
        elif strung[index] > count:
            found.add(
                "array_overcommitted",
                f"Strings take {strung[index]} modules from array {index}, which has {count}.",
                value=strung[index],
                limit=count,
                array=index,
                limit_field="modules",
            )

    for (inverter_index, mppt_index), members in on_input.items():
        _check_mppt(found, inverter_index, mppt_index, [(i, strings[i]) for i in members], arrays, inverters, sites)

    for inverter_index, inverter in enumerate(inverters):
        _check_inverter(found, inverter_index, inverter, strings, arrays, sites, dc_ac_ratio_range)
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

    voc_cold = n * _voc(module, environment.cold_cell_c)
    ceilings = (
        ("max_dc_voltage_v", inverter.max_dc_voltage_v, f"Inverter {string.inverter}", {"inverter": string.inverter}),
        (
            "Max_System_Voltage",
            module.Max_System_Voltage,
            f"The module of array {string.array}",
            {"array": string.array},
        ),
    )
    for field, ceiling, owner, missing_where in ceilings:
        if ceiling is None:
            found.missing(field, owner, **missing_where)
        elif voc_cold > ceiling + _TOLERANCE:
            found.add(
                "cold_voc_above_limit",
                f"String {index}: cold Voc is {voc_cold:.1f} V, above the {ceiling:.1f} V {field}.",
                value=voc_cold,
                limit=ceiling,
                limit_field=field,
                **where,
            )

    if inverter.startup_voltage_v is None:
        found.missing("startup_voltage_v", f"Inverter {string.inverter}", inverter=string.inverter)
    else:
        voc_hot = n * _voc(module, environment.hot_cell_c)
        if voc_hot < inverter.startup_voltage_v - _TOLERANCE:
            found.add(
                "hot_voc_below_startup",
                f"String {index}: hot-cell Voc is {voc_hot:.1f} V, below the {inverter.startup_voltage_v:.1f} V "
                "start-up voltage, so the inverter may not restart on the hottest days.",
                value=voc_hot,
                limit=inverter.startup_voltage_v,
                limit_field="startup_voltage_v",
                **where,
            )

    if inverter.min_mppt_voltage_v is None:
        found.missing("min_mppt_voltage_v", f"Inverter {string.inverter}", inverter=string.inverter)
    else:
        vmp_hot = n * _vmp(module, environment.hot_cell_c)
        if vmp_hot < inverter.min_mppt_voltage_v - _TOLERANCE:
            found.add(
                "hot_vmp_below_mppt_min",
                f"String {index}: hot-cell Vmp is {vmp_hot:.1f} V, below the {inverter.min_mppt_voltage_v:.1f} V "
                "MPPT minimum; on the hottest days the inverter works off the maximum power point and loses yield.",
                value=vmp_hot,
                limit=inverter.min_mppt_voltage_v,
                limit_field="min_mppt_voltage_v",
                **where,
            )
    if inverter.max_mppt_voltage_v is None:
        found.missing("max_mppt_voltage_v", f"Inverter {string.inverter}", inverter=string.inverter)
    else:
        vmp_cold = n * _vmp(module, environment.cold_operating_cell_c)
        if vmp_cold > inverter.max_mppt_voltage_v + _TOLERANCE:
            found.add(
                "vmp_above_mppt_max",
                f"String {index}: Vmp at the cold operating cell is {vmp_cold:.1f} V, above the "
                f"{inverter.max_mppt_voltage_v:.1f} V MPPT maximum; the inverter limits its voltage and loses yield.",
                value=vmp_cold,
                limit=inverter.max_mppt_voltage_v,
                limit_field="max_mppt_voltage_v",
                **where,
            )


def _check_mppt(
    found: _Findings,
    inverter_index: int,
    mppt_index: int,
    indexed: list[tuple[int, PVString]],
    arrays: Sequence[tuple[PVModuleParams, int]],
    inverters: Sequence[InverterSpec],
    sites: Sequence[ElectricalEnvironment],
) -> None:
    inverter = inverters[inverter_index]
    assert inverter.mppts is not None
    spec = inverter.mppts[mppt_index]
    where = {"inverter": inverter_index, "mppt": mppt_index}
    owner = f"MPPT {mppt_index} of inverter {inverter_index}"
    members = [string for _, string in indexed]
    parallel = sum(string.parallel for string in members)

    # Strings from different arrays may share an input, as a roof face split
    # into several arrays or an east-west pair often does: they share its
    # voltage, so they need the same module and the same length.
    first_module = arrays[members[0].array][0]
    same_module = all(arrays[string.array][0] == first_module for string in members)
    if len({string.modules for string in members}) > 1 or not same_module:
        found.add(
            "parallel_strings_mismatch",
            f"{owner}: parallel strings must have the same module and the same length.",
            **where,
        )

    if spec.max_strings is None:
        found.missing("max_strings", owner, **where)
    elif parallel > spec.max_strings:
        found.add(
            "too_many_strings",
            f"{owner} has {parallel} strings, above its limit of {spec.max_strings}.",
            value=parallel,
            limit=spec.max_strings,
            limit_field="max_strings",
            **where,
        )

    # The manufacturer's comparison factor applies to STC Isc; the hot-cell
    # Isc is the physical maximum. The input must take the larger of the two.
    factor = inverter.isc_comparison_factor
    if factor is None:
        found.missing("isc_comparison_factor", f"Inverter {inverter_index}", inverter=inverter_index)
    isc = sum(string.parallel * _design_isc(arrays[string.array][0], factor, sites[string.array]) for string in members)
    if spec.max_short_circuit_current_a is None:
        found.missing("max_short_circuit_current_a", owner, **where)
    elif isc > spec.max_short_circuit_current_a + _TOLERANCE:
        found.add(
            "isc_above_mppt_limit",
            f"{owner}: design Isc is {isc:.2f} A, above its {spec.max_short_circuit_current_a:.2f} A rating.",
            value=isc,
            limit=spec.max_short_circuit_current_a,
            limit_field="max_short_circuit_current_a",
            **where,
        )

    imp = sum(string.parallel * arrays[string.array][0].Imp for string in members)
    if spec.max_input_current_a is None:
        found.missing("max_input_current_a", owner, **where)
    elif imp > spec.max_input_current_a + _TOLERANCE:
        found.add(
            "imp_above_mppt_limit",
            f"{owner}: operating current is {imp:.2f} A, above its {spec.max_input_current_a:.2f} A input limit; "
            "the inverter limits its current and loses yield near peak production.",
            value=imp,
            limit=spec.max_input_current_a,
            limit_field="max_input_current_a",
            **where,
        )

    # Each string connection has its own rating on some inverters; unset, the
    # input's rating is the only one.
    for index, string in indexed:
        module = arrays[string.array][0]
        at = {**where, "string": index, "array": string.array}
        string_isc = _design_isc(module, factor, sites[string.array])
        string_isc_limit = spec.max_string_short_circuit_current_a
        if string_isc_limit is not None and string_isc > string_isc_limit + _TOLERANCE:
            found.add(
                "isc_above_string_limit",
                f"String {index}: design Isc is {string_isc:.2f} A, above the {string_isc_limit:.2f} A rating "
                f"of a string connection on {owner}.",
                value=string_isc,
                limit=string_isc_limit,
                limit_field="max_string_short_circuit_current_a",
                **at,
            )
        string_imp_limit = spec.max_string_input_current_a
        if string_imp_limit is not None and module.Imp > string_imp_limit + _TOLERANCE:
            found.add(
                "imp_above_string_limit",
                f"String {index}: operating current is {module.Imp:.2f} A, above the {string_imp_limit:.2f} A "
                f"limit of a string connection on {owner}; the inverter limits its current and loses yield.",
                value=module.Imp,
                limit=string_imp_limit,
                limit_field="max_string_input_current_a",
                **at,
            )

    stc_w = sum(s.modules * s.parallel * arrays[s.array][0].Mpp for s in members)
    if spec.max_dc_power_w is not None and stc_w > spec.max_dc_power_w + _TOLERANCE:
        found.add(
            "mppt_dc_power_above_max",
            f"{owner}: {stc_w:.0f} W of modules at STC exceeds its {spec.max_dc_power_w:.0f} W maximum "
            "PV generator power.",
            value=stc_w,
            limit=spec.max_dc_power_w,
            limit_field="max_dc_power_w",
            **where,
        )


def _check_inverter(
    found: _Findings,
    index: int,
    inverter: InverterSpec,
    strings: Sequence[PVString],
    arrays: Sequence[tuple[PVModuleParams, int]],
    sites: Sequence[ElectricalEnvironment],
    dc_ac_ratio_range: tuple[float, float] | None,
) -> None:
    connected = [s for s in strings if s.inverter == index and s.array < len(arrays)]
    if not connected:
        return
    if inverter.max_short_circuit_current_a is not None:
        isc = sum(
            s.parallel * _design_isc(arrays[s.array][0], inverter.isc_comparison_factor, sites[s.array])
            for s in connected
        )
        if isc > inverter.max_short_circuit_current_a + _TOLERANCE:
            found.add(
                "isc_above_inverter_limit",
                f"Inverter {index}: design Isc over all its inputs is {isc:.2f} A, above its "
                f"{inverter.max_short_circuit_current_a:.2f} A rating.",
                value=isc,
                limit=inverter.max_short_circuit_current_a,
                limit_field="max_short_circuit_current_a",
                inverter=index,
            )
    stc_w = sum(s.modules * s.parallel * arrays[s.array][0].Mpp for s in connected)
    if inverter.max_dc_power_w is not None and stc_w > inverter.max_dc_power_w + _TOLERANCE:
        found.add(
            "dc_power_above_max",
            f"Inverter {index}: {stc_w:.0f} W of modules at STC exceeds its {inverter.max_dc_power_w:.0f} W "
            "maximum PV generator power.",
            value=stc_w,
            limit=inverter.max_dc_power_w,
            limit_field="max_dc_power_w",
            inverter=index,
        )
    if inverter.recommended_dc_power_w is not None and stc_w > inverter.recommended_dc_power_w + _TOLERANCE:
        found.add(
            "dc_power_above_recommended",
            f"Inverter {index}: {stc_w:.0f} W of modules at STC exceeds its recommended "
            f"{inverter.recommended_dc_power_w:.0f} W.",
            value=stc_w,
            limit=inverter.recommended_dc_power_w,
            limit_field="recommended_dc_power_w",
            inverter=index,
        )
    if dc_ac_ratio_range is not None:
        low, high = dc_ac_ratio_range
        ratio = stc_w / inverter.nominal_power_w
        if not low - _TOLERANCE <= ratio <= high + _TOLERANCE:
            found.add(
                "dc_ac_ratio_outside_range",
                f"Inverter {index}: DC/AC ratio is {ratio:.2f}, outside {low:.2f} to {high:.2f}.",
                value=ratio,
                limit=low if ratio < low else high,
                inverter=index,
            )


@dataclass(frozen=True)
class ElectricalSetup:
    """The specified equipment, topology and site of an App run, with its check."""

    inverters: tuple[InverterSpec, ...]
    strings: tuple[PVString, ...]
    environment: ElectricalEnvironment
    check: TopologyCheck
    # "aggregate", or "string" for energy per MPPT input.
    fidelity: str = "aggregate"


def electrical_result(setup: ElectricalSetup | None) -> dict[str, Any]:
    """The ``electrical`` block of an App result.

    ``fidelity`` says how energy was modelled: ``"aggregate"`` for the system
    as a whole, or ``"string"`` when the inputs' and the inverter's limits
    applied (:func:`string_limited_dc`). Without a topology, ``status`` is ``"not_evaluated"``.
    """
    if setup is None:
        return {"fidelity": "aggregate", "status": "not_evaluated", "findings": []}
    return {
        "fidelity": setup.fidelity,
        "status": setup.check.status,
        "findings": [asdict(finding) for finding in setup.check.findings],
    }


# Findings that leave the string model without a definite wiring.
_STRUCTURAL_CODES = frozenset(
    {
        "no_inverters",
        "no_strings",
        "unknown_array",
        "unknown_inverter",
        "unknown_mppt",
        "unconnected_modules",
        "array_overcommitted",
        "parallel_strings_mismatch",
    }
)


def string_model_problems(setup: ElectricalSetup) -> list[str]:
    """Why the string energy model cannot run for this setup; empty when it can.

    The model needs one inverter, a wiring with no structural error, the MPPT
    voltage range, the start-up voltage and each used input's operating-current
    limit. Other errors, such as an overvoltage, do not stop it: their energy is
    a diagnostic, as in aggregate mode.
    """
    problems = []
    if len(setup.inverters) != 1:
        problems.append(f"it models one inverter; {len(setup.inverters)} are configured")
    structural = sorted({f.code for f in setup.check.findings if f.code in _STRUCTURAL_CODES})
    if structural:
        problems.append("the wiring has " + ", ".join(structural))
    if setup.inverters:
        inverter = setup.inverters[0]
        for name in ("min_mppt_voltage_v", "max_mppt_voltage_v", "startup_voltage_v"):
            if getattr(inverter, name) is None:
                problems.append(f"inverter 0 has no {name}")
        if inverter.mppts is None:
            problems.append("inverter 0 has no mppts")
        else:
            used = sorted({s.mppt for s in setup.strings if s.inverter == 0 and s.mppt < len(inverter.mppts)})
            for index in used:
                if inverter.mppts[index].max_input_current_a is None:
                    problems.append(f"MPPT {index} of inverter 0 has no max_input_current_a")
    return problems


def string_limited_dc(setup: ElectricalSetup, arrays: Sequence[Any]) -> tuple[pd.Series, pd.Series]:
    """DC power after each MPPT input's limits and the inverter's, in W.

    ``arrays`` holds one :class:`~breos.solar.PVProductionBreakdown` per PV
    array, in order, each with its module operating point. Each input runs its
    strings at their maximum power point when that point is inside the MPPT
    voltage range and under the input's and each string connection's
    operating-current limit. Otherwise it moves along the strings' IV curve to
    the best point that respects them: the nearest end of the range, or the
    voltage at which the current falls to the limit. When no point of the curve
    respects them, the input delivers nothing. Strings from different arrays on
    one input share its voltage, so the input runs at the maximum of their
    summed IV curve, which can lie below the sum of their maximum powers.

    An input then delivers at most its usable DC power, and the inverter at most
    its own. The inverter starts at the first producing step of a day on which
    the open-circuit voltage of an input reaches the start-up voltage, and stops
    when production does. Static losses scale the current, except availability,
    which applies after the limits, so the unconstrained power equals the
    aggregate model's.

    Returns the limited DC and the DC the limits cost. Where no limit binds,
    the DC is the aggregate model's, bit for bit.
    """
    problems = string_model_problems(setup)
    if problems:
        raise ValueError("String-level energy cannot run: " + "; ".join(problems))
    inverter = setup.inverters[0]
    assert inverter.mppts is not None and inverter.min_mppt_voltage_v is not None
    assert inverter.max_mppt_voltage_v is not None and inverter.startup_voltage_v is not None
    for breakdown in arrays:
        if breakdown.module_operating is None:
            raise ValueError("String-level energy needs each array's module operating point")
    in_service = {_in_service(breakdown) for breakdown in arrays}
    if len(in_service) != 1:
        raise ValueError("String-level energy needs one availability loss for every array")
    availability = in_service.pop()

    index = arrays[0].dc_after_losses.index
    # Powers before availability: delivered after the limits, and their cost.
    delivered = np.zeros(len(index))
    cost = np.zeros(len(index))
    open_circuit = np.zeros(len(index))
    by_input: dict[int, list[PVString]] = defaultdict(list)
    for string in setup.strings:
        by_input[string.mppt].append(string)
    for mppt_index, members in sorted(by_input.items()):
        spec = inverter.mppts[mppt_index]
        assert spec.max_input_current_a is not None
        string_limit = spec.max_string_input_current_a
        groups = [
            _StringGroup(arrays[array], sum(s.parallel for s in members if s.array == array))
            for array in sorted({s.array for s in members})
        ]
        unconstrained, limited = _input_power(
            groups,
            members[0].modules,
            inverter.min_mppt_voltage_v,
            inverter.max_mppt_voltage_v,
            spec.max_input_current_a,
            math.inf if string_limit is None else string_limit,
        )
        cost += np.clip(unconstrained - limited, 0.0, None)
        if spec.max_usable_dc_power_w is not None:
            cost += _excess(limited, spec.max_usable_dc_power_w)
            limited = np.minimum(limited, spec.max_usable_dc_power_w)
        delivered += limited
        if inverter.startup_voltage_v > 0:
            open_circuit = np.maximum(open_circuit, members[0].modules * _open_circuit_voltage(groups))
    if inverter.max_usable_dc_power_w is not None:
        cost += _excess(delivered, inverter.max_usable_dc_power_w)
        delivered = np.minimum(delivered, inverter.max_usable_dc_power_w)
    if inverter.startup_voltage_v > 0:
        idle = ~_running(delivered > 0, open_circuit >= inverter.startup_voltage_v - _TOLERANCE)
        cost += np.where(idle, delivered, 0.0)

    total = sum((b.dc_after_losses for b in arrays[1:]), arrays[0].dc_after_losses)
    loss_series = pd.Series(cost * availability, index=index, name="mppt_limit_loss_W")
    return (total - loss_series).rename("dc_power_W"), loss_series


@dataclass(frozen=True)
class _StringGroup:
    """``parallel`` strings of one array on one input."""

    breakdown: Any
    parallel: int

    @property
    def operating(self) -> Any:
        return self.breakdown.module_operating

    @property
    def string_scale(self) -> np.ndarray:
        """Each string's current over one module's, from the static losses other than availability."""
        module_dc = np.asarray(self.breakdown.module_dc, dtype=float)
        dc = np.asarray(self.breakdown.dc_after_losses, dtype=float)
        in_service = _in_service(self.breakdown)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where((module_dc > 0) & (in_service > 0), dc / module_dc / in_service, 0.0)


def _in_service(breakdown: Any) -> float:
    # Availability is time out of service, not less current while running, so
    # it applies after the limits; every other static loss scales the current.
    return 1.0 - float(breakdown.pvwatts_components_pct.get("availability", 0.0)) / 100.0


def _excess(power: np.ndarray, limit: float) -> np.ndarray:
    return np.clip(power - limit, 0.0, None)


def _running(producing: np.ndarray, can_start: np.ndarray) -> np.ndarray:
    """Whether the inverter runs: from the first step of a producing spell that can start it, to the spell's end."""
    spell = np.cumsum(~producing)
    started = pd.Series(producing & can_start).groupby(spell).cummax().to_numpy(dtype=bool)
    return producing & started


def _input_power(
    groups: list[_StringGroup],
    modules: int,
    v_min: float,
    v_max: float,
    current_limit: float,
    string_limit: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One input's unconstrained and limited DC before availability, in W at each step.

    The limited power equals the unconstrained array, element for element,
    wherever no limit binds.
    """
    if len(groups) == 1:
        group = groups[0]
        # Identical strings share the current, so each string's limit bounds the input at parallel times it.
        return _single_array_power(
            group.operating,
            group.string_scale * group.parallel,
            modules,
            v_min,
            v_max,
            min(current_limit, group.parallel * string_limit),
        )
    return _parallel_arrays_power(groups, modules, v_min, v_max, current_limit, string_limit)


def _single_array_power(
    operating: Any, current_scale: np.ndarray, modules: int, v_min: float, v_max: float, current_limit: float
) -> tuple[np.ndarray, np.ndarray]:
    v_mp = np.nan_to_num(operating.v_mp)
    i_mp = np.nan_to_num(operating.i_mp)
    unconstrained = modules * v_mp * current_scale * i_mp

    with np.errstate(divide="ignore", invalid="ignore"):
        module_current_limit = np.where(current_scale > 0, current_limit / current_scale, np.inf)
    lower_v, upper_v = v_min / modules, v_max / modules
    producing = unconstrained > 0
    limited = producing & ((v_mp < lower_v) | (v_mp > upper_v) | (i_mp > module_current_limit))
    result = unconstrained.copy()
    if not limited.any():
        return unconstrained, result

    params = [np.asarray(p)[limited] for p in _diode_params(operating)]
    v_at_limit = np.zeros(int(limited.sum()))
    binding = np.isfinite(module_current_limit[limited]) & (module_current_limit[limited] < params[0])
    if binding.any():
        v_at_limit[binding] = np.nan_to_num(
            pvlib.pvsystem.v_from_i(module_current_limit[limited][binding], *[p[binding] for p in params])
        ).clip(min=0.0)
    lower = np.maximum(lower_v, v_at_limit)
    voltage = np.clip(v_mp[limited], lower, upper_v)
    current = np.clip(np.nan_to_num(pvlib.pvsystem.i_from_v(voltage, *params)), 0.0, None)
    # No point of the curve is inside the range and under the current limit:
    # the input has no usable operating point and delivers nothing.
    current = np.where(lower > upper_v, 0.0, current)
    result[limited] = np.minimum(modules * voltage * current_scale[limited] * current, unconstrained[limited])
    return unconstrained, result


# Bisection and golden-section steps for the summed IV curve of parallel
# strings from different arrays, and the coarse grid that brackets its maximum.
_BISECTION_STEPS = 60
_GOLDEN_STEPS = 60
_GRID_POINTS = 64


def _parallel_arrays_power(
    groups: list[_StringGroup], modules: int, v_min: float, v_max: float, current_limit: float, string_limit: float
) -> tuple[np.ndarray, np.ndarray]:
    """Strings from several arrays in parallel: the best point of their summed IV curve within the limits."""
    scales = [group.string_scale for group in groups]
    unconstrained = sum(
        modules * group.parallel * scale * np.nan_to_num(group.operating.v_mp) * np.nan_to_num(group.operating.i_mp)
        for group, scale in zip(groups, scales, strict=False)
    )
    assert isinstance(unconstrained, np.ndarray)
    result = unconstrained.copy()
    steps = unconstrained > 0
    if not steps.any():
        return unconstrained, result
    params = [[np.asarray(p, dtype=float)[steps] for p in _diode_params(g.operating)] for g in groups]
    string_scales = [scale[steps] for scale in scales]
    input_scales = [g.parallel * scale for g, scale in zip(groups, string_scales, strict=False)]

    everywhere = np.ones(len(string_scales[0]), dtype=bool)

    def module_currents(v: np.ndarray, rows: np.ndarray = everywhere) -> list[np.ndarray]:
        shape = (-1,) + (1,) * (v.ndim - 1)
        return [np.nan_to_num(pvlib.pvsystem.i_from_v(v, *[p[rows].reshape(shape) for p in group])) for group in params]

    def input_current(v: np.ndarray) -> np.ndarray:
        shape = (-1,) + (1,) * (v.ndim - 1)
        return sum(s.reshape(shape) * i for s, i in zip(input_scales, module_currents(v), strict=False))

    def allowed(v: np.ndarray, rows: np.ndarray = everywhere) -> np.ndarray:
        currents = module_currents(v, rows)
        total = sum(s[rows] * i for s, i in zip(input_scales, currents, strict=False))
        largest = np.max([s[rows] * i for s, i in zip(string_scales, currents, strict=False)], axis=0)
        return (total <= current_limit + _TOLERANCE) & (largest <= string_limit + _TOLERANCE)

    # Above the highest open-circuit voltage the summed curve carries no current.
    v_oc = np.max([np.nan_to_num(pvlib.pvsystem.v_from_i(0.0, *group)) for group in params], axis=0)
    lower = np.full(len(v_oc), v_min / modules)
    upper = np.minimum(v_max / modules, v_oc)
    # The currents fall with voltage, so the allowed voltages start where they meet their limits.
    blocked = ~allowed(lower)
    if blocked.any():
        low, high = lower[blocked].copy(), upper[blocked].copy()
        for _ in range(_BISECTION_STEPS):
            middle = 0.5 * (low + high)
            ok = allowed(middle, blocked)
            high = np.where(ok, middle, high)
            low = np.where(ok, low, middle)
        lower[blocked] = high
    feasible = (lower <= upper) & allowed(np.minimum(lower, upper))

    def power(v: np.ndarray) -> np.ndarray:
        return modules * v * np.clip(input_current(v), 0.0, None)

    grid = lower[:, None] + (upper - lower).clip(min=0.0)[:, None] * np.linspace(0.0, 1.0, _GRID_POINTS)
    values = power(grid)
    best = np.argmax(values, axis=1)
    rows = np.arange(len(best))
    a = grid[rows, np.maximum(best - 1, 0)]
    b = grid[rows, np.minimum(best + 1, _GRID_POINTS - 1)]
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(_GOLDEN_STEPS):
        c, d = b - ratio * (b - a), a + ratio * (b - a)
        left = power(c) >= power(d)
        a, b = np.where(left, a, c), np.where(left, d, b)
    best_power = np.maximum(values[rows, best], power(0.5 * (a + b)))
    result[steps] = np.minimum(np.where(feasible, best_power, 0.0), unconstrained[steps])
    return unconstrained, result


def _open_circuit_voltage(groups: list[_StringGroup]) -> np.ndarray:
    """One module's open-circuit voltage on an input at each step, in V; 0 in the dark.

    Strings in parallel settle where their currents cancel, between their own
    open-circuit voltages.
    """
    lit = np.all([np.nan_to_num(np.asarray(g.operating.photocurrent, dtype=float)) > 0 for g in groups], axis=0)
    voltage = np.zeros(len(lit))
    if not lit.any():
        return voltage
    params = [[np.asarray(p, dtype=float)[lit] for p in _diode_params(g.operating)] for g in groups]
    own = [np.nan_to_num(pvlib.pvsystem.v_from_i(0.0, *group)) for group in params]
    if len(groups) == 1:
        voltage[lit] = own[0]
        return voltage
    scales = [g.parallel * g.string_scale[lit] for g in groups]
    low, high = np.min(own, axis=0), np.max(own, axis=0)
    for _ in range(_BISECTION_STEPS):
        middle = 0.5 * (low + high)
        current = sum(
            s * np.nan_to_num(pvlib.pvsystem.i_from_v(middle, *group)) for s, group in zip(scales, params, strict=False)
        )
        positive = np.asarray(current) > 0
        low, high = np.where(positive, middle, low), np.where(positive, high, middle)
    voltage[lit] = 0.5 * (low + high)
    return voltage


def _diode_params(operating: Any) -> tuple[np.ndarray, ...]:
    return (
        operating.photocurrent,
        operating.saturation_current,
        operating.resistance_series,
        operating.resistance_shunt,
        operating.n_ns_vth,
    )
