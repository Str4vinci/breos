"""Validated vehicle inputs, independent of App and of dispatch."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from breos.config_schema import TableSpec, boolean, choice, list_of, number, table, text
from breos.constants import (
    DEFAULT_CHARGE_EFFICIENCY,
    DEFAULT_DISCHARGE_EFFICIENCY,
    DEFAULT_EOL_PERCENTAGE,
    DEFAULT_MIN_SOC,
    DEGRADATION_ENGINES,
    SKIPPED_REPLACEMENT_ACTIONS,
)
from breos.degradation.native import CALENDAR_PARAMETERS
from breos.degradation.profiles import ENABLED_BLAST_MODEL_KEYS

_SOC = number(minimum=0, maximum=1)
_EFFICIENCY = number(minimum=0, min_exclusive=True, maximum=1)
_POSITIVE = number(minimum=0, min_exclusive=True)
_NON_NEGATIVE = number(minimum=0)


def _clock(value: Any, where: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", value):
        raise ValueError(f"'{where}' must be a civil time in HH:MM format (00:00 to 23:59)")
    return value


DAY_SCHEDULE_TABLE = TableSpec(
    "vehicle.schedule.weekday",
    keys={"arrive": _clock, "depart": _clock, "trip_kwh": _NON_NEGATIVE, "away_charging": boolean},
    required=frozenset({"arrive", "depart", "trip_kwh"}),
    docs={
        "arrive": "Local arrival time, HH:MM; at or before departure means arrival on the following day",
        "depart": "Local departure time, HH:MM; the departure date selects the day type",
        "trip_kwh": "Gross stored energy withdrawn while driving, in kWh; half in the first and last away steps",
        "away_charging": "Allow away charging in the parked steps between the two legs; default true",
    },
)


def _schedule_form(values: dict[str, Any], where: str) -> None:
    if "file" in values:
        if len(values) != 1:
            raise ValueError(f"'{where}.file' replaces weekday and weekend rules; do not set both forms")
    elif not {"weekday", "weekend"} <= values.keys():
        raise ValueError(f"'{where}' needs both weekday and weekend rules, or file")


VEHICLE_SCHEDULE_TABLE = TableSpec(
    "vehicle.schedule",
    keys={"weekday": table(DAY_SCHEDULE_TABLE), "weekend": table(DAY_SCHEDULE_TABLE), "file": text},
    check=_schedule_form,
    docs={
        "weekday": "Monday–Friday rule; see vehicle.schedule.weekday",
        "weekend": "Saturday–Sunday rule, with the same keys as vehicle.schedule.weekday",
        "file": "Trip-segment CSV path, relative to the working directory; replaces both day rules",
    },
)

AWAY_CHARGING_TABLE = TableSpec(
    "vehicle.away_charging",
    keys={"target_soc": _SOC, "power_kw": _POSITIVE, "efficiency": _EFFICIENCY, "price": _NON_NEGATIVE},
    required=frozenset({"target_soc", "power_kw", "efficiency", "price"}),
    docs={
        "target_soc": "Stopping target within the vehicle SOC window, not a ceiling on energy already on board",
        "power_kw": "Charging power at the purchased-energy boundary, in kW; greater than 0, no default",
        "efficiency": "Stored energy per purchased energy, in (0, 1]; no default",
        "price": "Price per purchased kWh in the run currency; required, 0 means free charging",
    },
)


def _temperature(value: Any, where: str) -> Any:
    if isinstance(value, str):
        if value.lower() != "weather" and not Path(value).is_file():
            raise FileNotFoundError(f"{where} file not found: {value}")
        return value
    return number()(value, where)


VEHICLE_TABLE = TableSpec(
    "vehicle",
    keys={
        "mode": choice(("uncontrolled", "v1g", "v2h")),
        "battery_kwh": _POSITIVE,
        "min_soc": _SOC,
        "max_soc": _SOC,
        "departure_soc": _SOC,
        "v2h_min_soc": _SOC,
        "charge_efficiency": _EFFICIENCY,
        "discharge_efficiency": _EFFICIENCY,
        "charger_ac_rating_kw": _POSITIVE,
        "charger_efficiency": _EFFICIENCY,
        "charge_periods": list_of(text, min_length=1),
        "discharge_periods": list_of(text, min_length=1),
        "reference_mode": choice(("uncontrolled", "v1g", "v2h", "none")),
        "charger_cost": _NON_NEGATIVE,
        "battery_cost_per_kwh": _NON_NEGATIVE,
        "outside_charging_price": _NON_NEGATIVE,
        "eol_percentage": _SOC,
        "enable_replacement": boolean,
        "allow_terminal_replacement": boolean,
        "replacement_min_remaining_years": _NON_NEGATIVE,
        "skipped_replacement_action": choice(SKIPPED_REPLACEMENT_ACTIONS),
        "battery_replacement_cost": _NON_NEGATIVE,
        "degradation_engine": choice(DEGRADATION_ENGINES),
        "blast_model": text,
        "calendar_model": choice(CALENDAR_PARAMETERS),
        "battery_temperature": _temperature,
        "away_charging": table(AWAY_CHARGING_TABLE),
        "schedule": table(VEHICLE_SCHEDULE_TABLE),
    },
    required=frozenset(
        {
            "mode",
            "battery_kwh",
            "departure_soc",
            "charger_ac_rating_kw",
            "charger_efficiency",
            "battery_cost_per_kwh",
            "schedule",
        }
    ),
    docs={
        "mode": 'Required study mode: "uncontrolled", "v1g" or "v2h"; no default',
        "battery_kwh": "Nominal pack capacity in kWh; greater than 0, no default",
        "min_soc": f"Minimum SOC; default {DEFAULT_MIN_SOC:g}",
        "max_soc": "Maximum SOC; default 1",
        "departure_soc": "Required SOC at each departure, within the SOC window; no default",
        "v2h_min_soc": "V2H discharge floor within the SOC window; v2h only; default min_soc",
        "charge_efficiency": f"Battery DC-in to stored efficiency; default {DEFAULT_CHARGE_EFFICIENCY:g}",
        "discharge_efficiency": f"Battery stored to DC-out efficiency; default {DEFAULT_DISCHARGE_EFFICIENCY:g}",
        "charger_ac_rating_kw": "Charger AC input/output rating in kW; greater than 0, no default",
        "charger_efficiency": "Charger nominal efficiency in (0, 1]; no default",
        "charge_periods": "Disjoint tariff periods for grid charging; needs tariff; refused with uncontrolled",
        "discharge_periods": "Tariff periods for discharge; needs tariff and v2h; disjoint from charge_periods",
        "reference_mode": "Reference household mode; default v1g for v2h, uncontrolled for v1g, none for uncontrolled; must differ from mode",
        "charger_cost": "Installed charger premium in run currency; required with v2h, no default; otherwise default 0",
        "battery_cost_per_kwh": "Pack-only replacement/wear price per nominal kWh in run currency; required, no default",
        "outside_charging_price": "Optional penalty per kWh of unserved driving; restores no energy; no default",
        "eol_percentage": f"Pack replacement health threshold; default {DEFAULT_EOL_PERCENTAGE:g}",
        "enable_replacement": "Replace the pack at its health threshold; default true",
        "allow_terminal_replacement": "Allow a swap at the project's last day close; default true",
        "replacement_min_remaining_years": "Minimum project years remaining for a replacement; default 0",
        "skipped_replacement_action": '"keep" (default) or "retire" the old pack when replacement is skipped',
        "battery_replacement_cost": "Optional price of one swap, including installation; terminal credit remains pack-only",
        "degradation_engine": '"native" (default) or "blast"; BLAST is refused with Monte Carlo',
        "blast_model": "Enabled BLAST cell-model key; required with blast, refused with native",
        "calendar_model": 'Native calendar model; default "naumann_lam_field_calibrated"',
        "battery_temperature": '"weather" (default), fixed °C or timestamped CSV; no indoor model',
        "away_charging": "Optional away charging equipment and price; all four nested keys required",
        "schedule": "Required weekday/weekend rules or trip-segment CSV; see vehicle.schedule",
    },
)


@dataclass(frozen=True)
class VehicleDaySchedule:
    """An absence starting on this day type, in configured-zone civil time."""

    arrive: str
    depart: str
    trip_kwh: float
    away_charging: bool = True

    def __post_init__(self) -> None:
        DAY_SCHEDULE_TABLE.validate(asdict(self))


@dataclass(frozen=True)
class VehicleScheduleSpec:
    """Exactly one input form: two day rules, or a trip-segment CSV."""

    weekday: VehicleDaySchedule | Mapping[str, Any] | None = None
    weekend: VehicleDaySchedule | Mapping[str, Any] | None = None
    file: str | None = None

    def __post_init__(self) -> None:
        values = {key: value for key, value in asdict(self).items() if value is not None}
        checked = VEHICLE_SCHEDULE_TABLE.validate(values)
        for name in ("weekday", "weekend"):
            if name in checked:
                object.__setattr__(self, name, VehicleDaySchedule(**checked[name]))


@dataclass(frozen=True)
class AwayChargingConfig:
    """Away charger facts, all at the purchased-energy boundary."""

    target_soc: float
    power_kw: float
    efficiency: float
    price: float

    def __post_init__(self) -> None:
        AWAY_CHARGING_TABLE.validate(asdict(self))


@dataclass(frozen=True)
class VehicleConfig:
    """One vehicle's equipment, controls and prices; never a stationary pack.

    Study mode, equipment facts and pack price have no defaults. ``charger_cost`` has
    no default in v2h mode; other modes use zero when it is omitted.
    """

    mode: str
    battery_kwh: float
    departure_soc: float
    charger_ac_rating_kw: float
    charger_efficiency: float
    battery_cost_per_kwh: float
    schedule: VehicleScheduleSpec | Mapping[str, Any]
    min_soc: float = DEFAULT_MIN_SOC
    max_soc: float = 1.0
    v2h_min_soc: float | None = None
    charge_efficiency: float = DEFAULT_CHARGE_EFFICIENCY
    discharge_efficiency: float = DEFAULT_DISCHARGE_EFFICIENCY
    charge_periods: tuple[str, ...] = ()
    discharge_periods: tuple[str, ...] = ()
    reference_mode: str | None = None
    charger_cost: float | None = None
    outside_charging_price: float | None = None
    eol_percentage: float = DEFAULT_EOL_PERCENTAGE
    enable_replacement: bool = True
    allow_terminal_replacement: bool = True
    replacement_min_remaining_years: float = 0.0
    skipped_replacement_action: str = "keep"
    battery_replacement_cost: float | None = None
    degradation_engine: str = "native"
    blast_model: str | None = None
    calendar_model: str = "naumann_lam_field_calibrated"
    battery_temperature: float | str = "weather"
    away_charging: AwayChargingConfig | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        optional_keys = {
            "v2h_min_soc",
            "reference_mode",
            "charger_cost",
            "outside_charging_price",
            "battery_replacement_cost",
            "blast_model",
            "away_charging",
        }
        values = {
            key: value
            for key, value in asdict(self).items()
            if not (key in optional_keys and value is None)
            and not (key in ("charge_periods", "discharge_periods") and not value)
        }
        # asdict includes the unused schedule form's None fields.
        if isinstance(values["schedule"], Mapping):
            values["schedule"] = {key: value for key, value in values["schedule"].items() if value is not None}
        checked = VEHICLE_TABLE.validate(values)
        for name, value in checked.items():
            object.__setattr__(self, name, value)
        if not self.min_soc <= self.departure_soc <= self.max_soc:
            raise ValueError("'vehicle' SOC limits must satisfy 0 <= min_soc <= departure_soc <= max_soc <= 1")
        if self.v2h_min_soc is not None and self.mode != "v2h":
            raise ValueError("'vehicle.v2h_min_soc' needs mode = 'v2h'")
        floor = self.min_soc if self.v2h_min_soc is None else self.v2h_min_soc
        if not self.min_soc <= floor <= self.max_soc:
            raise ValueError("'vehicle' SOC limits must satisfy min_soc <= v2h_min_soc <= max_soc")
        if self.mode == "v2h":
            object.__setattr__(self, "v2h_min_soc", floor)
        if self.mode == "uncontrolled" and (self.charge_periods or self.discharge_periods):
            raise ValueError("'vehicle' charge_periods and discharge_periods are refused with mode = 'uncontrolled'")
        if self.mode != "v2h" and self.discharge_periods:
            raise ValueError("'vehicle.discharge_periods' needs mode = 'v2h'")
        if set(self.charge_periods) & set(self.discharge_periods):
            raise ValueError("'vehicle' charge_periods and discharge_periods must be disjoint")
        for name in ("charge_periods", "discharge_periods"):
            object.__setattr__(self, name, tuple(dict.fromkeys(getattr(self, name))))
        reference = self.reference_mode
        if reference is None:
            reference = {"v2h": "v1g", "v1g": "uncontrolled", "uncontrolled": "none"}[self.mode]
        if reference == self.mode:
            raise ValueError("'vehicle.reference_mode' must differ from vehicle.mode")
        if self.mode == "uncontrolled" and reference != "none":
            raise ValueError("'vehicle.reference_mode' must be 'none' with mode = 'uncontrolled'")
        object.__setattr__(self, "reference_mode", reference)
        if self.charger_cost is None:
            if self.mode == "v2h":
                raise ValueError(
                    "A V2H vehicle needs 'vehicle.charger_cost'; it has no default. Set 0 for hardware already owned"
                )
            object.__setattr__(self, "charger_cost", 0.0)
        if self.degradation_engine == "blast":
            if self.blast_model not in ENABLED_BLAST_MODEL_KEYS:
                raise ValueError("'vehicle.degradation_engine=blast' needs an enabled vehicle.blast_model")
        elif self.blast_model is not None:
            raise ValueError("'vehicle.blast_model' requires 'vehicle.degradation_engine=blast'")
        object.__setattr__(self, "schedule", VehicleScheduleSpec(**checked["schedule"]))
        if self.away_charging is not None:
            away = AwayChargingConfig(**checked["away_charging"])
            if not self.min_soc <= away.target_soc <= self.max_soc:
                raise ValueError("'vehicle.away_charging.target_soc' must be within the vehicle SOC window")
            object.__setattr__(self, "away_charging", away)


def validate_vehicle(value: Any, *, periods: tuple[str, ...] | None, montecarlo: bool = False) -> VehicleConfig:
    """Validate an App table and its tariff/runner dependencies."""
    checked = VEHICLE_TABLE.validate(value)
    config = VehicleConfig(**checked)
    for name in ("charge_periods", "discharge_periods"):
        selected = getattr(config, name)
        if selected and periods is None:
            raise ValueError(f"'vehicle.{name}' needs a [tariff]: its periods are tariff periods")
        unknown = sorted(set(selected) - set(periods or ()))
        if unknown:
            raise ValueError(f"'vehicle.{name}' has unknown tariff period(s): {', '.join(unknown)}")
    if montecarlo and config.degradation_engine == "blast":
        raise ValueError("'vehicle.degradation_engine=blast' is not supported with Monte Carlo yet")
    return config
