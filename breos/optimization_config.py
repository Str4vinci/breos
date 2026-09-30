"""The optimization config: its nested tables, their defaults, and one resolver (#181, #162).

:func:`breos.optimization.optimize_system_multi_objective` and
:func:`breos.optimization.evaluate_projected_design` take a nested config
(``[location]``, ``[pv]``, ``[battery]``, ``[costs]``, ``[financials]``,
``[constraints]``, ...), not the flat App config. :func:`resolve_optimization_config`
checks it once, before any candidate is scored: every table allows a fixed set
of keys, so a misspelt or unsupported key raises instead of being ignored,
and every default the optimizer applies is filled in here, by name, rather
than inside the code that reads it. The values a key shares with the App
(the tariff and smart-charging tables, the cost keys, the indoor model, the
projection horizon, PV degradation, inverter efficiency and the default
module) are checked and defaulted the way the App does.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import fields
from typing import Any, Mapping

from breos.app_config import (
    COSTS_TABLE,
    DEFAULTS,
    INDOOR_MODEL_TABLE,
    SMART_CHARGING_TABLE,
    TARIFF_TABLE,
    check_calendar_model,
    default_module_key,
)
from breos.config_schema import (
    TableSpec,
    anything,
    boolean,
    choice,
    number,
    table,
    text,
)
from breos.economics import DEFAULT_DISCOUNT_RATE, DEFAULT_INFLATION_RATE
from breos.emissions import EmissionsParams
from breos.pv.model_options import PV_MODEL_CONFIG_KEYS

# Search defaults. They apply when [constraints] leaves a key out, and the
# resolved values are recorded in the optimizer's provenance.
DEFAULT_BUDGET = 10000.0
DEFAULT_MAX_AREA_M2 = 20.0
DEFAULT_MAX_MODULES = 60
DEFAULT_MAX_BATTERY_KWH = 30.0
DEFAULT_MIN_TILT_DEG = 10.0
DEFAULT_MAX_TILT_DEG = 90.0
DEFAULT_TILT_MARGIN_DEG = 15.0
DEFAULT_TIMEZONE = "UTC"
DEFAULT_RESOLUTION = "h"
DEFAULT_OBJECTIVE_BASIS = "projected"

# NSGA-II run settings: an ``optimize_system_multi_objective`` argument, else
# the [optimization] key, else this.
DEFAULT_RUN_SETTINGS: Mapping[str, Any] = {"pop_size": 40, "n_gen": 100, "n_offsprings": None, "seed": 1}

# A removed key raises with its replacement rather than as unknown.
_REMOVED_KEYS = {
    ("constraints", "budget_eur"): (
        "constraints.budget_eur was renamed to constraints.budget in 0.7.0; the budget is in the run's currency."
    ),
    ("costs", "panel_wp"): (
        "costs.panel_wp was removed in 0.7.0: CAPEX is priced at the selected module's own "
        "rating. Remove the key, or select a module with the wattage you meant."
    ),
}


def _integer(minimum: int) -> Any:
    """A whole number, returned as ``int``; ``20.0`` is accepted, ``20.5`` is not."""

    def check(value: Any, where: str) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"'{where}' must be a whole number")
        if isinstance(value, float) and not (math.isfinite(value) and value.is_integer()):
            raise ValueError(f"'{where}' must be a whole number")
        if value < minimum:
            raise ValueError(f"'{where}' must be >= {minimum}")
        return int(value)

    return check


def _optional(checker: Any) -> Any:
    return lambda value, where: None if value is None else checker(value, where)


def _real(minimum: float, maximum: float) -> Any:
    """A number in range, returned as given, so an integer stays one where it is reported."""
    check = number(minimum=minimum, maximum=maximum)

    def keep(value: Any, where: str) -> Any:
        check(value, where)
        return value

    return keep


def _scale(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("must be a number")
    return float(value)


def _dc_output_scale(value: Any, where: str) -> float:
    """A DC-side yield correction, applied before dispatch; not bounded above."""
    scale = _scale(value)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("dc_output_scale must be finite and greater than 0")
    return scale


def _ac_output_scale(value: Any, where: str) -> float:
    """An AC-side derate, applied after the inverter nameplate limit, so at most 1."""
    scale = _scale(value)
    if not math.isfinite(scale) or not 0.0 < scale <= 1.0:
        raise ValueError(
            "ac_output_scale must be finite, greater than 0 and at most 1; it is applied after "
            "the inverter nameplate limit. Use dc_output_scale to correct an under-predicting "
            "model on the DC side"
        )
    return scale


def _max_tilt(value: Any, where: str) -> Any:
    if isinstance(value, str):
        if value.strip().lower() == "adjust":
            return "adjust"
        raise ValueError(f"Unsupported {where} value: {value!r}. Use a number or 'adjust'.")
    return number(minimum=0, maximum=90)(value, where)


def _objective_basis(value: Any, where: str) -> str:
    if isinstance(value, str) and value.strip().lower() == "steady_state":
        raise ValueError(
            "optimization.objective_basis = 'steady_state' was removed in 0.7.0: candidates are "
            "scored over the projected lifetime only. Use 'projected' or omit the key."
        )
    if not (isinstance(value, str) and value.strip().lower() == DEFAULT_OBJECTIVE_BASIS):
        raise ValueError("optimization.objective_basis must be 'projected'")
    return DEFAULT_OBJECTIVE_BASIS


def _early_stop(value: Any, where: str) -> Any:
    if value is None or isinstance(value, bool):
        return value
    return EARLY_STOP_TABLE.validate(value, where)


_MODULE_PARAMS_TABLE = TableSpec(
    "pv.params",
    keys={
        **dict.fromkeys(("Mpp", "Vmp", "Imp", "Voc", "Isc"), number(minimum=0, min_exclusive=True)),
        **dict.fromkeys(("T_Pmax_pct", "T_Pmax", "T_Voc_pct", "T_Voc", "T_Isc_pct", "T_Isc"), number()),
        "N_Cells": _integer(1),
        "celltype": text,
    },
    # An inline module states its electrical rating; only the temperature
    # coefficients, cell count and cell type have defaults.
    required=frozenset({"Mpp", "Vmp", "Imp", "Voc", "Isc"}),
)
_DIMENSIONS_TABLE = TableSpec(
    "pv.dimensions",
    keys=dict.fromkeys(("width", "length"), number(minimum=0, min_exclusive=True)),
    required=frozenset({"width", "length"}),
)

LOCATION_TABLE = TableSpec(
    "location",
    keys={
        "latitude": number(minimum=-90, maximum=90),
        "longitude": number(minimum=-180, maximum=180),
        "timezone": text,
        "altitude": _optional(number()),
        "name": anything,
    },
    required=frozenset({"latitude", "longitude"}),
)
PV_TABLE = TableSpec(
    "pv",
    keys={
        "module": _optional(text),
        "params": table(_MODULE_PARAMS_TABLE),
        "dimensions": table(_DIMENSIONS_TABLE),
        "module_width_m": number(minimum=0, min_exclusive=True),
        "module_length_m": number(minimum=0, min_exclusive=True),
        "degradation_rate": number(minimum=0, maximum=1, max_exclusive=True),
    },
)
BATTERY_TABLE = TableSpec(
    "battery",
    keys={
        # Forwarded to BatteryConfig, which checks the values.
        **dict.fromkeys(
            (
                "min_soc",
                "max_soc",
                "charge_efficiency",
                "discharge_efficiency",
                "standby_loss_wh",
                "eol_percentage",
                "max_charge_power_w",
                "max_discharge_power_w",
                "power_limit_c_rate",
                "enable_resistance_fade",
            ),
            anything,
        ),
        # Stored normalised, as App stores its top-level calendar_model.
        "calendar_model": check_calendar_model,
        # Read by the battery temperature builder, which checks it.
        "temperature": anything,
        "indoor_model": _optional(table(INDOOR_MODEL_TABLE)),
        "degradation_engine": choice(("native", "blast")),
        "blast_model": anything,
        "replacement_cost": anything,
        "enable_replacement": boolean,
        "initial_soh": number(minimum=0, maximum=100, min_exclusive=True),
    },
)
FINANCIALS_TABLE = TableSpec(
    "financials",
    keys={
        "inflation_rate": number(),
        "sell_price_inflation": number(),
        "import_price_escalation": _optional(number()),
        "om_escalation": _optional(number()),
        "replacement_cost_learning": _optional(number()),
        "discount_rate": number(),
        "project_lifespan": _integer(1),
        "pv_degradation_rate": number(minimum=0, maximum=1, max_exclusive=True),
        # Flat prices, used when [costs] does not set them.
        "electricity_cost": number(minimum=0),
        "electricity_sold_cost": number(minimum=0),
    },
)
OPTIMIZATION_COSTS_TABLE = TableSpec(
    "costs",
    keys={
        **COSTS_TABLE.keys,
        # The optimizer's inverter sizing: AC rating = DC peak / dc_ac_ratio.
        "dc_ac_ratio": number(minimum=0, min_exclusive=True),
    },
)
CONSTRAINTS_TABLE = TableSpec(
    "constraints",
    keys={
        "budget": number(minimum=0),
        "max_area_m2": number(minimum=0, min_exclusive=True),
        # A bound, not a count: the repair snaps candidates inside it.
        "max_modules": _real(1, float("inf")),
        "max_battery_kwh": number(minimum=0),
        "min_tilt_deg": number(minimum=0, maximum=90),
        "max_tilt_deg": _max_tilt,
        "tilt_margin_deg": number(minimum=0),
        "enforce_zeb": boolean,
    },
)
MODE_TABLE = TableSpec("mode", keys={"fixed_azimuth": _optional(_real(-360, 360))})
EARLY_STOP_TABLE = TableSpec(
    "optimization.early_stop",
    keys={
        "enabled": boolean,
        "ftol": number(minimum=0, min_exclusive=True),
        "period": _integer(1),
        "n_skip": _integer(0),
        "min_gen": _integer(1),
        "only_feasible": boolean,
    },
)
OPTIMIZATION_TABLE = TableSpec(
    "optimization",
    keys={
        "algorithm": choice(("nsga2",)),
        "objective_basis": _objective_basis,
        "early_stop": _early_stop,
        "pop_size": _integer(1),
        "n_gen": _integer(1),
        "n_offsprings": _optional(_integer(1)),
        "seed": _integer(0),
    },
)
SIMULATION_TABLE = TableSpec(
    "simulation",
    keys={"resolution": choice(("h", "15min")), "years_projection": _integer(1)},
)
INVERTER_TABLE = TableSpec("inverter", keys={"efficiency": number(minimum=0, maximum=1, min_exclusive=True)})
EMISSIONS_TABLE = TableSpec("emissions", keys={field.name: anything for field in fields(EmissionsParams)})

# Every table the optimizer reads, by its top-level key. [tariff] and
# [smart_charging] are the App's own tables.
OPTIMIZATION_TABLES: Mapping[str, TableSpec] = {
    "location": LOCATION_TABLE,
    "pv": PV_TABLE,
    "battery": BATTERY_TABLE,
    "costs": OPTIMIZATION_COSTS_TABLE,
    "financials": FINANCIALS_TABLE,
    "constraints": CONSTRAINTS_TABLE,
    "mode": MODE_TABLE,
    "optimization": OPTIMIZATION_TABLE,
    "simulation": SIMULATION_TABLE,
    "inverter": INVERTER_TABLE,
    "emissions": EMISSIONS_TABLE,
    "tariff": TARIFF_TABLE,
    "smart_charging": SMART_CHARGING_TABLE,
}
# Top-level scalars: the App keys the optimizer reads under the same name.
OPTIMIZATION_SCALARS: Mapping[str, Any] = {
    "pv_module": text,
    "inverter_efficiency": number(minimum=0, maximum=1, min_exclusive=True),
    "dc_output_scale": _dc_output_scale,
    "ac_output_scale": _ac_output_scale,
    **dict.fromkeys(PV_MODEL_CONFIG_KEYS, anything),
}


def _first_set(where: tuple[tuple[str, Any], ...], default: Any) -> Any:
    """The first spelling of a setting that is set, in the documented order, else ``default``."""
    for _name, value in where:
        if value is not None:
            return value
    return default


def adjusted_max_tilt_deg(latitude: float, margin_deg: float) -> float:
    """The ``max_tilt_deg = "adjust"`` bound: 5° × round((|latitude| + margin) / 5), within 60–90°."""
    return float(min(max(5.0 * round((abs(float(latitude)) + float(margin_deg)) / 5.0), 60.0), 90.0))


def resolve_optimization_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Check an optimization config and return it with every default filled in.

    The input is left unchanged. Resolving a resolved config gives it back,
    so an entry point can resolve what it is handed without knowing whether a
    caller already did.

    Raises:
        TypeError: If a table is not a mapping or a value has the wrong type.
        ValueError: If a key is unknown or removed, a required key is missing,
            or a value is out of range.

    Where two keys set one thing, the first one set wins:
    ``simulation.years_projection`` over ``financials.project_lifespan``,
    ``pv.degradation_rate`` over ``financials.pv_degradation_rate``,
    ``inverter_efficiency`` over ``inverter.efficiency``, and ``pv.module``
    over ``pv_module``.
    """
    if not isinstance(config, Mapping):
        raise TypeError("The optimization config must be a table/dict")
    simulation_table = config.get("simulation")
    if "period" in config or (isinstance(simulation_table, Mapping) and "period" in simulation_table):
        raise ValueError(
            "'period' is not supported by the optimizer: it ranks designs on their lifetime economics, which a "
            "window shorter than a year does not have. Remove 'period', or run the window with breos.App."
        )
    unknown = sorted(key for key in config if key not in OPTIMIZATION_TABLES and key not in OPTIMIZATION_SCALARS)
    if unknown:
        hint = (
            " Pass execution_backend to the function instead: the optimizer never reads it from the config."
            if "execution_backend" in unknown
            else ""
        )
        available = ", ".join(sorted({*OPTIMIZATION_TABLES, *OPTIMIZATION_SCALARS}))
        raise ValueError(f"Unknown optimization config key(s): {', '.join(unknown)}.{hint} Available: {available}")
    if "location" not in config:
        raise ValueError("The optimization config needs a [location] table with latitude and longitude")

    resolved: dict[str, Any] = {}
    for key, value in config.items():
        if key in OPTIMIZATION_SCALARS:
            resolved[key] = OPTIMIZATION_SCALARS[key](value, key)
            continue
        # An empty [emissions] table turns emissions off, as it always has;
        # an empty [tariff] or [smart_charging] is checked like any other.
        if value is None or (key == "emissions" and not value):
            resolved[key] = None
            continue
        if isinstance(value, Mapping):
            for name in value:
                if (key, name) in _REMOVED_KEYS:
                    raise ValueError(_REMOVED_KEYS[(key, name)])
        # Validation normalises values; the tariff and smart-charging tables
        # are kept as given and resolved by the shared App resolvers.
        checked = OPTIMIZATION_TABLES[key].validate(value, key)
        resolved[key] = deepcopy(dict(value)) if key in ("tariff", "smart_charging") else checked

    location = resolved["location"]
    location.setdefault("timezone", DEFAULT_TIMEZONE)
    location.setdefault("altitude", None)
    location.setdefault("name", "")

    pv = resolved.setdefault("pv", {}) or {}
    resolved["pv"] = pv
    financials = resolved.setdefault("financials", {}) or {}
    resolved["financials"] = financials
    financials.setdefault("inflation_rate", DEFAULT_INFLATION_RATE)
    financials.setdefault("sell_price_inflation", 0.0)
    financials.setdefault("discount_rate", DEFAULT_DISCOUNT_RATE)
    simulation = resolved.setdefault("simulation", {}) or {}
    resolved["simulation"] = simulation
    inverter = resolved.pop("inverter", None) or {}

    simulation.setdefault("resolution", DEFAULT_RESOLUTION)
    simulation["years_projection"] = _first_set(
        (
            ("simulation.years_projection", simulation.get("years_projection")),
            ("financials.project_lifespan", financials.get("project_lifespan")),
        ),
        DEFAULTS["projection_years"],
    )
    pv["degradation_rate"] = _first_set(
        (
            ("pv.degradation_rate", pv.get("degradation_rate")),
            ("financials.pv_degradation_rate", financials.get("pv_degradation_rate")),
        ),
        DEFAULTS["pv_degradation_rate"],
    )
    resolved["inverter_efficiency"] = _first_set(
        (
            ("inverter_efficiency", resolved.get("inverter_efficiency")),
            ("inverter.efficiency", inverter.get("efficiency")),
        ),
        DEFAULTS["inverter_efficiency"],
    )
    module = _first_set((("pv.module", pv.get("module")), ("pv_module", resolved.pop("pv_module", None))), None)
    if "params" not in pv:
        pv["module"] = module or default_module_key()
    elif module is not None:
        raise ValueError("pv.params gives the module inline, so it is not also named; remove pv.module or pv_module")
    resolved.setdefault("dc_output_scale", 1.0)
    resolved.setdefault("ac_output_scale", 1.0)

    battery = resolved.setdefault("battery", {}) or {}
    resolved["battery"] = battery
    battery.setdefault("temperature", DEFAULTS["battery_temperature"])
    battery.setdefault("indoor_model", None)
    battery.setdefault("degradation_engine", "native")
    battery.setdefault("blast_model", None)
    battery.setdefault("replacement_cost", None)
    battery.setdefault("enable_replacement", True)
    battery.setdefault("initial_soh", 100.0)
    resolved["costs"] = resolved.get("costs") or {}

    constraints = resolved.setdefault("constraints", {}) or {}
    resolved["constraints"] = constraints
    constraints.setdefault("budget", DEFAULT_BUDGET)
    constraints.setdefault("max_area_m2", DEFAULT_MAX_AREA_M2)
    constraints.setdefault("max_modules", DEFAULT_MAX_MODULES)
    constraints.setdefault("max_battery_kwh", DEFAULT_MAX_BATTERY_KWH)
    constraints.setdefault("min_tilt_deg", DEFAULT_MIN_TILT_DEG)
    constraints.setdefault("max_tilt_deg", DEFAULT_MAX_TILT_DEG)
    constraints.setdefault("tilt_margin_deg", DEFAULT_TILT_MARGIN_DEG)
    constraints.setdefault("enforce_zeb", False)
    max_tilt = constraints["max_tilt_deg"]
    if max_tilt == "adjust":
        max_tilt = adjusted_max_tilt_deg(location["latitude"], constraints["tilt_margin_deg"])
    if constraints["min_tilt_deg"] > max_tilt:
        raise ValueError(
            f"constraints.min_tilt_deg ({constraints['min_tilt_deg']:g}) is above the maximum tilt "
            f"({max_tilt:g}); lower it or raise constraints.max_tilt_deg"
        )

    mode = resolved.setdefault("mode", {}) or {}
    resolved["mode"] = mode
    mode.setdefault("fixed_azimuth", None)

    optimization = resolved.setdefault("optimization", {}) or {}
    resolved["optimization"] = optimization
    optimization.setdefault("algorithm", "nsga2")
    optimization.setdefault("objective_basis", DEFAULT_OBJECTIVE_BASIS)
    optimization.setdefault("early_stop", None)
    for key in ("emissions", "tariff", "smart_charging"):
        resolved.setdefault(key, None)
    return resolved


def resolve_run_settings(config: Mapping[str, Any], **arguments: Any) -> dict[str, Any]:
    """The NSGA-II run settings: each argument given, else the [optimization] key, else its default.

    Raises:
        ValueError: If an argument and the [optimization] key are both set and differ.
    """
    table = config.get("optimization") or {}
    settings: dict[str, Any] = {}
    for key, default in DEFAULT_RUN_SETTINGS.items():
        argument, configured = arguments.get(key), table.get(key)
        if argument is not None and configured is not None and argument != configured:
            raise ValueError(
                f"{key} = {argument!r} was passed, but optimization.{key} = {configured!r}; set one of them"
            )
        settings[key] = argument if argument is not None else configured if configured is not None else default
    return settings
