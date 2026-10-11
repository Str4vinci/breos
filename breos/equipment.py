"""Generic inverter datasheet values, independent of product catalogs and laws.

The specification stores hardware limits; it does not validate a string topology
or make the aggregate simulator model individual MPPTs. ``as_app_config`` supplies
only the AC nameplate and optional efficiency the simulator actually consumes.
"""

import math
from dataclasses import dataclass
from numbers import Integral, Real


@dataclass(frozen=True)
class MpptSpec:
    """One MPPT input's datasheet limits, with unknown values left unset.

    Currents are A. Inverters with asymmetric inputs list one ``MpptSpec`` per
    input, so each input keeps its own current and string limits.
    """

    max_input_current_a: float | None = None
    max_short_circuit_current_a: float | None = None
    max_strings: int | None = None

    def __post_init__(self) -> None:
        for name in ("max_input_current_a", "max_short_circuit_current_a"):
            value = getattr(self, name)
            if value is not None:
                _positive(name, value)
        if self.max_strings is not None:
            _positive_integer("max_strings", self.max_strings)
        _ordered(self, "max_input_current_a", "max_short_circuit_current_a")


@dataclass(frozen=True)
class InverterSpec:
    """A specified inverter, with unknown datasheet values left unset.

    Powers are W, voltages V, and efficiency is a fraction. The DC voltage
    limits apply to every input; ``mppts`` lists the inputs in datasheet order,
    each with its own current and string limits. ``isc_comparison_factor`` is
    the multiplier the manufacturer applies to array Isc before comparing it
    with an input's short-circuit rating, such as 1.25. No prices, product
    identities, certifications or sizing presets live here.
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

    def __post_init__(self) -> None:
        _positive("nominal_power_w", self.nominal_power_w)
        for name in (
            "inverter_efficiency",
            "max_dc_voltage_v",
            "max_dc_power_w",
            "min_mppt_voltage_v",
            "max_mppt_voltage_v",
            "startup_voltage_v",
            "isc_comparison_factor",
        ):
            value = getattr(self, name)
            if value is not None:
                _positive(name, value)
        if self.inverter_efficiency is not None and self.inverter_efficiency > 1:
            raise ValueError("inverter_efficiency must be in (0, 1]")
        if self.isc_comparison_factor is not None and self.isc_comparison_factor < 1:
            raise ValueError("isc_comparison_factor must be at least 1")
        if self.is_hybrid is not None and not isinstance(self.is_hybrid, bool):
            raise ValueError("is_hybrid must be a boolean when supplied")
        if self.mppts is not None:
            mppts = tuple(self.mppts) if isinstance(self.mppts, list) else self.mppts
            if not isinstance(mppts, tuple) or not mppts or not all(isinstance(m, MpptSpec) for m in mppts):
                raise ValueError("mppts must be a non-empty sequence of MpptSpec")
            object.__setattr__(self, "mppts", mppts)

        for lower, upper in (
            ("min_mppt_voltage_v", "max_mppt_voltage_v"),
            ("min_mppt_voltage_v", "max_dc_voltage_v"),
            ("max_mppt_voltage_v", "max_dc_voltage_v"),
            ("startup_voltage_v", "max_dc_voltage_v"),
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


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _ordered(spec: object, lower: str, upper: str) -> None:
    low, high = getattr(spec, lower), getattr(spec, upper)
    if low is not None and high is not None and low > high:
        raise ValueError(f"{lower} must not exceed {upper}")
