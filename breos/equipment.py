"""Generic inverter datasheet values, independent of product catalogs and laws.

The specification stores hardware limits; it does not validate a string topology
or make the aggregate simulator model individual MPPTs. ``as_app_config`` supplies
only the AC nameplate and optional efficiency the simulator actually consumes.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any


@dataclass(frozen=True)
class MpptSpec:
    """One MPPT input's datasheet limits, with unknown values left unset.

    Currents are A and powers W. Inverters with asymmetric inputs list one
    ``MpptSpec`` per input, so each input keeps its own current and string
    limits. A limit of ``math.inf`` means the input has none.

    The last four limits are published by some datasheets only, so leaving
    them unset is not reported missing. ``max_string_input_current_a`` and
    ``max_string_short_circuit_current_a`` rate each string connection of the
    input; ``max_dc_power_w`` is the PV generator power the input may carry, and
    ``max_usable_dc_power_w`` the DC power the input converts, above which the
    inverter moves the input off its maximum power point.
    """

    max_input_current_a: float | None = None
    max_short_circuit_current_a: float | None = None
    max_strings: int | None = None
    max_string_input_current_a: float | None = None
    max_string_short_circuit_current_a: float | None = None
    max_dc_power_w: float | None = None
    max_usable_dc_power_w: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "max_input_current_a",
            "max_short_circuit_current_a",
            "max_string_input_current_a",
            "max_string_short_circuit_current_a",
            "max_dc_power_w",
            "max_usable_dc_power_w",
        ):
            value = getattr(self, name)
            if value is not None:
                _upper_limit(name, value)
        if self.max_strings is not None:
            _positive_integer("max_strings", self.max_strings)
        _ordered(self, "max_input_current_a", "max_short_circuit_current_a")
        _ordered(self, "max_string_input_current_a", "max_string_short_circuit_current_a")


@dataclass(frozen=True)
class EfficiencyCurve:
    """An inverter's conversion efficiency against its AC output, at one to three DC voltages.

    ``output_fraction`` holds the AC output points as fractions of the AC
    nameplate, increasing, such as the CEC points 0.1, 0.2, 0.3, 0.5, 0.75
    and 1. ``efficiency`` holds one row of efficiencies, fractions in
    (0, 1], per DC voltage in ``voltages_v``, lowest voltage first; a single
    row needs no voltage. Efficiency is linear in output between the points.
    This is the shape of the CEC and OND test data, so an importer can fill it.
    """

    output_fraction: tuple[float, ...]
    efficiency: tuple[tuple[float, ...], ...]
    voltages_v: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        fractions = _float_tuple("output_fraction", self.output_fraction)
        if not fractions or any(f <= 0 for f in fractions):
            raise ValueError("output_fraction must hold at least one fraction above 0")
        if any(b <= a for a, b in zip(fractions, fractions[1:], strict=False)):
            raise ValueError("output_fraction must increase")
        if not isinstance(self.efficiency, (list, tuple)) or not self.efficiency:
            raise ValueError("efficiency must hold at least one row")
        rows = tuple(_float_tuple("efficiency", row) for row in self.efficiency)
        for row in rows:
            if len(row) != len(fractions):
                raise ValueError("each efficiency row needs one value per output_fraction")
            if any(not 0 < e <= 1 for e in row):
                raise ValueError("efficiency values must be in (0, 1]")
            # The DC input of each point must rise with its output, or the
            # curve would map one DC input to two outputs.
            dc = [f / e for f, e in zip(fractions, row, strict=False)]
            if any(b <= a for a, b in zip(dc, dc[1:], strict=False)):
                raise ValueError("efficiency rises too steeply: each point's DC input must exceed the previous one's")
        voltages = None if self.voltages_v is None else _float_tuple("voltages_v", self.voltages_v)
        if voltages is None:
            if len(rows) != 1:
                raise ValueError("give voltages_v, one per efficiency row, when there are several rows")
        else:
            if len(voltages) != len(rows):
                raise ValueError("give one voltage in voltages_v per efficiency row")
            if any(v <= 0 for v in voltages) or any(b <= a for a, b in zip(voltages, voltages[1:], strict=False)):
                raise ValueError("voltages_v must be positive and increase")
        if len(rows) > 3:
            raise ValueError("efficiency takes one to three voltage rows")
        object.__setattr__(self, "output_fraction", fractions)
        object.__setattr__(self, "efficiency", rows)
        object.__setattr__(self, "voltages_v", voltages)

    def row_position(self, voltage_v: float) -> float:
        """The position of DC voltage ``voltage_v`` among the rows, for interpolating between them."""
        if self.voltages_v is None or len(self.voltages_v) == 1:
            return 0.0
        voltages = self.voltages_v
        if voltage_v <= voltages[0]:
            return 0.0
        for index in range(len(voltages) - 1):
            if voltage_v < voltages[index + 1]:
                return index + (voltage_v - voltages[index]) / (voltages[index + 1] - voltages[index])
        return float(len(voltages) - 1)


@dataclass(frozen=True)
class TemperatureDerating:
    """The AC output an inverter allows at each ambient temperature.

    ``max_output_fraction`` is the allowed AC output as a fraction of the
    nameplate at each ambient in ``ambient_temp_c`` (°C, increasing), linear
    between the points and held beyond the ends.
    """

    ambient_temp_c: tuple[float, ...]
    max_output_fraction: tuple[float, ...]

    def __post_init__(self) -> None:
        temps = _float_tuple("ambient_temp_c", self.ambient_temp_c)
        fractions = _float_tuple("max_output_fraction", self.max_output_fraction)
        if not temps or len(temps) != len(fractions):
            raise ValueError("give one max_output_fraction per ambient_temp_c, at least one")
        if any(b <= a for a, b in zip(temps, temps[1:], strict=False)):
            raise ValueError("ambient_temp_c must increase")
        if any(not 0 <= f <= 1 for f in fractions):
            raise ValueError("max_output_fraction values must be in [0, 1]")
        object.__setattr__(self, "ambient_temp_c", temps)
        object.__setattr__(self, "max_output_fraction", fractions)


def _float_tuple(name: str, values: object) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a sequence of numbers")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{name} must hold finite numbers")
    return tuple(float(value) for value in values)


@dataclass(frozen=True)
class InverterSpec:
    """A specified inverter, with unknown datasheet values left unset.

    Powers are W, voltages V, and efficiency is a fraction. The DC voltage
    limits apply to every input; ``mppts`` lists the inputs in datasheet order,
    each with its own current and string limits, as ``MpptSpec`` instances or
    mappings of their fields. ``None`` means a value is unknown; an upper limit
    of ``math.inf`` or a lower limit of 0 means the inverter has none.
    ``isc_comparison_factor`` is
    the multiplier the manufacturer applies to array Isc before comparing it
    with an input's STC short-circuit current, such as 1.25; give 1 where the
    manufacturer compares Isc as it is. ``max_dc_power_w`` is the maximum PV
    generator power the manufacturer permits, and ``recommended_dc_power_w``
    its recommended PV power. ``max_short_circuit_current_a`` is the array
    short-circuit current the inverter takes over all its inputs, and
    ``max_usable_dc_power_w`` the DC power it converts over all its inputs;
    like those two powers, they are published by some datasheets only and
    are not reported missing when unset. ``efficiency_curve`` replaces the
    single ``inverter_efficiency`` with the measured curve, and
    ``temperature_derating`` lowers the AC rating in hot ambients; both accept
    mappings of their fields. ``night_consumption_w`` is what the inverter
    draws from the grid while it converts nothing, as datasheets state it;
    unset, it is not modelled and not reported missing. ``ac_cable`` is
    the inverter's own AC cable, given as a mapping of the fields of
    :class:`~breos.inverter.AcCable` and held as their sorted
    ``(field, value)`` pairs. No prices, product identities, certifications or sizing
    presets live here.
    """

    nominal_power_w: float
    inverter_efficiency: float | None = None
    is_hybrid: bool | None = None
    max_dc_voltage_v: float | None = None
    max_dc_power_w: float | None = None
    min_mppt_voltage_v: float | None = None
    max_mppt_voltage_v: float | None = None
    startup_voltage_v: float | None = None
    mppts: tuple[MpptSpec, ...] | None = None
    isc_comparison_factor: float | None = None
    recommended_dc_power_w: float | None = None
    max_short_circuit_current_a: float | None = None
    max_usable_dc_power_w: float | None = None
    efficiency_curve: EfficiencyCurve | None = None
    temperature_derating: TemperatureDerating | None = None
    night_consumption_w: float | None = None
    ac_cable: tuple[tuple[str, Any], ...] | None = None

    def __post_init__(self) -> None:
        _positive("nominal_power_w", self.nominal_power_w)
        if self.ac_cable is not None:
            # Held as sorted (field, value) pairs, so the spec stays hashable and picklable.
            cable = self.ac_cable
            if isinstance(cable, Mapping):
                cable = tuple(sorted(cable.items()))
            elif isinstance(cable, (list, tuple)):
                cable = tuple(sorted(tuple(pair) for pair in cable))
            else:
                raise ValueError("ac_cable must be a mapping of AcCable's fields")
            object.__setattr__(self, "ac_cable", cable)
        for name, kind in (("efficiency_curve", EfficiencyCurve), ("temperature_derating", TemperatureDerating)):
            value = getattr(self, name)
            if isinstance(value, Mapping):
                object.__setattr__(self, name, kind(**value))
            elif value is not None and not isinstance(value, kind):
                raise ValueError(f"{name} must be a {kind.__name__} or a mapping of its fields")
        for name in ("inverter_efficiency", "isc_comparison_factor"):
            value = getattr(self, name)
            if value is not None:
                _positive(name, value)
        for name in (
            "max_dc_voltage_v",
            "max_dc_power_w",
            "max_mppt_voltage_v",
            "recommended_dc_power_w",
            "max_short_circuit_current_a",
            "max_usable_dc_power_w",
        ):
            value = getattr(self, name)
            if value is not None:
                _upper_limit(name, value)
        for name in ("min_mppt_voltage_v", "startup_voltage_v", "night_consumption_w"):
            value = getattr(self, name)
            if value is not None:
                _lower_limit(name, value)
        if self.inverter_efficiency is not None and self.inverter_efficiency > 1:
            raise ValueError("inverter_efficiency must be in (0, 1]")
        if self.isc_comparison_factor is not None and self.isc_comparison_factor < 1:
            raise ValueError("isc_comparison_factor must be at least 1")
        if self.is_hybrid is not None and not isinstance(self.is_hybrid, bool):
            raise ValueError("is_hybrid must be a boolean when supplied")
        if self.mppts is not None:
            if not isinstance(self.mppts, (list, tuple)) or not self.mppts:
                raise ValueError("mppts must be a non-empty sequence of MpptSpec")
            mppts = tuple(MpptSpec(**m) if isinstance(m, Mapping) else m for m in self.mppts)
            if not all(isinstance(m, MpptSpec) for m in mppts):
                raise ValueError("mppts must be a non-empty sequence of MpptSpec")
            object.__setattr__(self, "mppts", mppts)

        for lower, upper in (
            ("min_mppt_voltage_v", "max_mppt_voltage_v"),
            ("min_mppt_voltage_v", "max_dc_voltage_v"),
            ("max_mppt_voltage_v", "max_dc_voltage_v"),
            ("startup_voltage_v", "max_dc_voltage_v"),
            ("recommended_dc_power_w", "max_dc_power_w"),
        ):
            _ordered(self, lower, upper)

    def as_app_config(self) -> dict[str, float | None]:
        """Select this nameplate in App, clearing any previous sizing ratio.

        Merge into the caller's configuration before construction. When efficiency
        is unknown, App keeps its existing/default efficiency assumption. DC/MPPT
        limits remain metadata: passing them does not add string-aware simulation.
        """
        config: dict[str, float | None] = {
            "inverter_ac_rating_kw": self.nominal_power_w / 1000,
            "inverter_loading_ratio": None,
        }
        if self.inverter_efficiency is not None:
            config["inverter_efficiency"] = self.inverter_efficiency
        return config


def _positive(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")


def _upper_limit(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or math.isnan(value) or value <= 0:
        raise ValueError(f"{name} must be a positive number or math.inf")


def _lower_limit(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite number of at least 0")


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _ordered(spec: object, lower: str, upper: str) -> None:
    low, high = getattr(spec, lower), getattr(spec, upper)
    if low is not None and high is not None and low > high:
        raise ValueError(f"{lower} must not exceed {upper}")
