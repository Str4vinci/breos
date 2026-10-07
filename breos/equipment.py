"""Generic inverter datasheet values, independent of product catalogs and laws.

The specification stores hardware limits; it does not validate a string topology
or make the aggregate simulator model individual MPPTs. ``as_app_config`` supplies
only the AC nameplate and optional efficiency the simulator actually consumes.
"""

import math
from dataclasses import dataclass
from numbers import Integral, Real


@dataclass(frozen=True)
class InverterSpec:
    """A specified inverter, with unknown datasheet values left unset.

    Powers are W, voltages V, currents A, and efficiency is a fraction. MPPT
    limits describe identical channels; equipment with asymmetric channels
    needs a richer model before those limits can support topology checks.
    No prices, product identities, certifications or sizing presets live here.
    """

    nominal_power_w: float
    inverter_efficiency: float | None = None
    is_hybrid: bool | None = None
    mppt_channels: int | None = None
    max_dc_voltage_v: float | None = None
    max_dc_power_w: float | None = None
    min_mppt_voltage_v: float | None = None
    max_mppt_voltage_v: float | None = None
    startup_voltage_v: float | None = None
    max_strings_per_mppt: int | None = None
    max_input_current_per_mppt_a: float | None = None
    max_short_circuit_current_per_mppt_a: float | None = None

    def __post_init__(self) -> None:
        _positive("nominal_power_w", self.nominal_power_w)
        for name in (
            "inverter_efficiency",
            "max_dc_voltage_v",
            "max_dc_power_w",
            "min_mppt_voltage_v",
            "max_mppt_voltage_v",
            "startup_voltage_v",
            "max_input_current_per_mppt_a",
            "max_short_circuit_current_per_mppt_a",
        ):
            value = getattr(self, name)
            if value is not None:
                _positive(name, value)
        if self.inverter_efficiency is not None and self.inverter_efficiency > 1:
            raise ValueError("inverter_efficiency must be in (0, 1]")
        for name in ("mppt_channels", "max_strings_per_mppt"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, Integral) or value <= 0):
                raise ValueError(f"{name} must be a positive integer")
        if self.is_hybrid is not None and not isinstance(self.is_hybrid, bool):
            raise ValueError("is_hybrid must be a boolean when supplied")

        for lower, upper in (
            ("min_mppt_voltage_v", "max_mppt_voltage_v"),
            ("min_mppt_voltage_v", "max_dc_voltage_v"),
            ("max_mppt_voltage_v", "max_dc_voltage_v"),
            ("startup_voltage_v", "max_dc_voltage_v"),
            ("max_input_current_per_mppt_a", "max_short_circuit_current_per_mppt_a"),
        ):
            low, high = getattr(self, lower), getattr(self, upper)
            if low is not None and high is not None and low > high:
                raise ValueError(f"{lower} must not exceed {upper}")

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
