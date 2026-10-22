"""Configuration and resource resolution for the public App facade."""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, fields, replace
from datetime import date, datetime
from functools import partial
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

from breos.config_schema import (
    Checker,
    Choice,
    TableSpec,
    anything,
    boolean,
    choice,
    integer,
    list_of,
    mapping_of,
    number,
    optional,
    table,
    text,
)
from breos.constants import (
    BATTERY_COUPLINGS,
    DEFAULT_CHARGE_EFFICIENCY,
    DEFAULT_DISCHARGE_EFFICIENCY,
    DEFAULT_EOL_PERCENTAGE,
    DEFAULT_INDOOR_CEILING_C,
    DEFAULT_INDOOR_COUPLING_ALPHA,
    DEFAULT_INDOOR_FLOOR_C,
    DEFAULT_INDOOR_MODEL_ENABLED,
    DEFAULT_INDOOR_SETPOINT_C,
    DEFAULT_MAX_SOC,
    DEFAULT_MIN_SOC,
    DEGRADATION_ENGINES,
    RESOLUTIONS,
    SKIPPED_REPLACEMENT_ACTIONS,
)
from breos.degradation.native import CALENDAR_PARAMETERS
from breos.degradation.profiles import ENABLED_BLAST_MODEL_KEYS
from breos.economics import (
    COST_CONFIG_KEY_TO_PARAM,
    DEFAULT_DISCOUNT_RATE,
    DEFAULT_INFLATION_RATE,
    CostParams,
    calculate_costs,
    default_amount_cost_keys,
    replacement_event_cost,
)
from breos.electrical import (
    DEFAULT_CONDUCTOR_TEMP_C,
    MOUNTING_TEMPERATURE_RISE_C,
    ArrayMounting,
    ElectricalEnvironment,
    ElectricalSetup,
    PVString,
    cable_summary,
    check_topology,
    string_model_problems,
)
from breos.emissions import EmissionsParams
from breos.equipment import InverterSpec, MpptSpec
from breos.execution import (
    DEFAULT_EXECUTION_BACKEND,
    EXECUTION_BACKENDS,
    config_has_battery,
    validate_execution_backend,
)
from breos.inverter import AcCable, InverterConversion, inverter_ac_capacity_w
from breos.load_profiles import PROFILE_UNITS, resolve_profile_key, validate_profile_options
from breos.pv.horizon import normalise_horizon_profile
from breos.pv.mismatch import DEFAULT_BYPASS_DIODE_VOLTAGE_V
from breos.pv.model_options import PER_ARRAY_MODEL_OPTION_KEYS, is_valid_albedo, normalise_model_name
from breos.pv.shading import ShadingTable
from breos.pv.temperature import validate_temperature_inputs
from breos.pv_modules import MODULES, PVModuleParams, catalogue_snapshot, get_module
from breos.resources import load_config_json
from breos.smart_charging import (
    DECISION_BOUNDARIES,
    DECISION_BOUNDARY_KEY,
    DEFAULT_PUBLICATION_DELAY_HOURS,
    FORECAST_FILES_KEY,
    FORECAST_KEY,
    FORECASTS,
    OVERLAP_POLICIES,
    PLANNER_KEYS,
    PLANNER_MODES,
    PLANNER_SETTINGS,
    PUBLICATION_DELAY_KEY,
    SMART_CHARGING_MODES,
    WEAR_COST_KEY,
    SmartChargingSpec,
    check_charge_windows,
    check_mode_settings,
    check_overlap_policy,
    check_period_overlap,
)
from breos.solar import (
    BIFACIAL_MODELS,
    DEFAULT_BIFACIAL_MODEL,
    DEFAULT_DIFFUSE_IAM,
    DEFAULT_IAM_MODEL,
    DEFAULT_PEREZ_MODEL,
    DEFAULT_SOLAR_POSITION,
    DEFAULT_TEMPERATURE_MODEL,
    DEFAULT_TRANSPOSITION_MODEL,
    DIFFUSE_IAM_METHODS,
    IAM_MODELS,
    PEREZ_MODELS,
    SOLAR_POSITION_METHODS,
    SURFACE_TYPES,
    TEMPERATURE_MODELS,
    TRANSPOSITION_MODELS,
    estimate_optimal_tilt,
    resolve_pvwatts_losses,
)
from breos.solar import default_azimuth as default_azimuth_fn
from breos.tariffs import (
    BOUNDARY_POLICIES,
    DEFAULT_CURRENCY,
    SCHEDULE_CYCLES,
    AnnualNetworkCredit,
    ReferenceTariffSpec,
    ScheduleDefinition,
    TariffPrices,
    TariffSchedule,
    TariffSpec,
    available_tariff_schedules,
    check_currency,
    get_schedule_definition,
    parse_schedule_definition,
    run_currency,
    schedule_resolution_minutes,
    validate_season_prices,
)
from breos.utils import get_hours_per_step
from breos.vehicle.config import VEHICLE_TABLE, VehicleConfig, validate_vehicle
from breos.weather import IRRADIANCE_RESAMPLING_POLICIES, validate_irradiance_resampling

_NO_DEFAULT = object()
# Checks several keys share: a growth or discount rate, and an efficiency.
_RATE = number(minimum=-1, min_exclusive=True)
_EFFICIENCY = number(minimum=0, maximum=1, min_exclusive=True)
_EFFICIENCY_OR_NONE = number(minimum=0, maximum=1, min_exclusive=True, allow_none=True)
_TRACKING_MODES = ("fixed", "single_axis", "dual_axis")


def _count(value: Any, where: str) -> int:
    """A whole number of at least 1, such as a module count."""
    if not _is_int(value) or value < 1:
        raise ValueError(f"'{where}' must be >= 1")
    return value


def _module_key(value: Any, where: str) -> str:
    """A key of the PV module catalogue, read when checked, so a module added at run time counts."""
    if value not in MODULES:
        raise ValueError(f"Unknown PV module {value!r} for '{where}'. Available: {', '.join(sorted(MODULES))}")
    return value


def _albedo(value: Any, where: str) -> float:
    """A ground reflectance in [0, 1]."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not is_valid_albedo(value):
        raise ValueError(f"'{where}' must be a number between 0 and 1")
    return float(value)


def _angle(low: float, high: float) -> Checker:
    return number(minimum=low, maximum=high, allow_none=True)


# The PV keys an array may set as well as the top level, checked alike in
# both places. None leaves a key unset, except where noted.
_PV_CHECKS: dict[str, Checker] = {
    "tilt": _angle(0, 90),
    "azimuth": _angle(0, 360),
    # Not optional: the top level always resolves a mode. An array takes optional(...).
    "tracking": choice(_TRACKING_MODES, case_insensitive=False),
    "axis_tilt": _angle(0, 90),
    "axis_azimuth": _angle(0, 360),
    "max_angle": _angle(0, 90),
    # A string such as "no" is truthy and would leave backtracking on.
    "backtrack": optional(boolean),
    "cross_axis_tilt": _angle(-90, 90),
    "dual_axis_max_tilt": _angle(0, 90),
    "temperature_model": choice(TEMPERATURE_MODELS),
    # gcr feeds the infinite_sheds view factors and the backtracking schedule;
    # pvlib takes any ratio and quietly computes a different rotation.
    "gcr": number(minimum=0, maximum=1, min_exclusive=True),
    "transposition_model": optional(choice(TRANSPOSITION_MODELS)),
    "albedo": optional(_albedo),
    "surface_type": optional(choice(SURFACE_TYPES, case_insensitive=False)),
    "model_perez": optional(choice(PEREZ_MODELS, case_insensitive=False)),
    # None reads as "none", as the model name does.
    "bifacial_model": optional(choice(BIFACIAL_MODELS)),
    "pvrow_height": number(minimum=0, min_exclusive=True, allow_none=True),
    "pvrow_pitch": number(minimum=0, min_exclusive=True, allow_none=True),
}


@dataclass(frozen=True)
class AppConfigField:
    """Declarative metadata for one public App configuration key.

    Scientific constraints intentionally remain in the focused validators
    below. This registry owns the mechanical contract that had previously
    drifted across defaults, the allowed-key list, CLI arguments, and CLI
    override propagation.

    ``doc`` describes the key in the generated configuration key reference
    (``tools/generate_config_docs.py``). ``default_doc`` replaces the default
    shown there when the value alone would mislead: a required key, or a
    table that is unset by default.

    ``summary`` is the key's ``"section.name"`` place in the resolved-config
    summary that ``breos validate-config`` and ``breos run --dry-run`` print.
    Only the runner sections have none.

    ``check`` checks the value on its own, after the ``normalizer``; it is
    not written back. Rules that span keys stay in the validators. A
    :func:`~breos.config_schema.choice` check also gives ``cli_choices``.

    ``affects_inputs=False`` marks a key that does not reach
    :func:`breos.app_inputs.prepare_simulation_inputs`, directly or through
    the resolved location, orientation, tracking, arrays or module: it feeds
    dispatch, degradation, valuation, emissions or the backend, which run
    later, so the input cache leaves it out. Each such key is pinned by a
    test that changes it and compares the prepared inputs. The default is
    True, so a new key is part of the cache key until shown not to matter.

    ``revalue`` marks a price key that :meth:`breos.App.revalue` may change.
    ``exclusive_with`` names a key that sets the same thing, so one config
    layer may set only one of the two; a later layer that sets this key drops
    the other (:func:`override_config`).
    """

    default: Any = _NO_DEFAULT
    cli_flags: tuple[str, ...] = ()
    cli_type: Callable[[str], Any] | None = None
    cli_choices: tuple[str, ...] | None = None
    cli_help: str | None = None
    normalizer: Callable[[Any], Any] | None = None
    doc: str = ""
    default_doc: str | None = None
    summary: str | None = None
    check: Checker | None = None
    affects_inputs: bool = True
    revalue: bool = False
    exclusive_with: str | None = None

    def __post_init__(self) -> None:
        if self.cli_choices is None and isinstance(self.check, Choice):
            object.__setattr__(self, "cli_choices", self.check.options)

    @property
    def has_default(self) -> bool:
        return self.default is not _NO_DEFAULT


def _lower(value: Any) -> Any:
    return value.lower() if isinstance(value, str) and value else value


def _upper(value: Any) -> Any:
    return value.upper() if isinstance(value, str) and value else value


def _underscored(value: Any) -> Any:
    return value.replace("-", "_") if isinstance(value, str) and value else value


def _path_string(value: Any) -> str | None:
    return str(value) if value is not None else None


def _model_key(value: Any) -> Any:
    return value.strip().lower().replace("-", "_") if isinstance(value, str) else value


CALENDAR_MODELS: tuple[str, ...] = tuple(CALENDAR_PARAMETERS)


def check_calendar_model(value: Any, where: str) -> str:
    """Return a calendar aging model name as the aging model looks it up.

    The App and the optimizer both store the returned spelling, so a name
    that passes here cannot fail in ``_get_degradation_params``.
    """
    key = _model_key(value)
    if not isinstance(key, str) or key not in CALENDAR_MODELS:
        raise ValueError(f"'{where}' must be one of: {', '.join(CALENDAR_MODELS)}")
    return key


APP_CONFIG_FIELDS: dict[str, AppConfigField] = {
    # CLI-exposed fields are kept in parser display order. Required inputs have
    # no default; argparse still leaves them optional so --config can supply
    # them, and the existing App validators remain the source of required-key
    # errors.
    "location": AppConfigField(
        cli_flags=("--location",),
        cli_help="Location preset key, for example 'porto'.",
        normalizer=_lower,
        doc=(
            'Preset key (e.g. `"porto"`, `"berlin"`) or `{"latitude": ..., "longitude": ..., "timezone": ...}`. See '
            "[Custom location](configuration.md#custom-location)"
        ),
        default_doc="*required*",
        summary="location.key",
    ),
    "n_modules": AppConfigField(
        cli_flags=("--n-modules",),
        cli_type=int,
        cli_help="Number of PV modules.",
        doc="Number of PV modules",
        default_doc="*required unless `pv_arrays` is set*",
        summary="pv.n_modules",
    ),
    "annual_consumption_kwh": AppConfigField(
        check=number(minimum=0, min_exclusive=True),
        cli_flags=("--annual-consumption-kwh",),
        cli_type=float,
        cli_help="Annual electricity demand in kWh.",
        doc="Annual electricity demand (kWh)",
        default_doc="*required*",
        summary="load.annual_consumption_kwh",
    ),
    "battery_kwh": AppConfigField(
        affects_inputs=False,
        check=number(minimum=0),
        default=0.0,
        cli_flags=("--battery-kwh",),
        cli_type=float,
        cli_help="Battery capacity in kWh.",
        doc=(
            "Nominal battery capacity in kWh (`0` = no battery). The SOC window sets the usable share; see [Battery "
            "capacity and the SOC window](configuration.md#battery-capacity-and-the-soc-window)"
        ),
        summary="battery.capacity_kwh",
    ),
    "battery_max_charge_power_w": AppConfigField(
        affects_inputs=False,
        check=number(minimum=0, allow_none=True),
        default=None,
        cli_flags=("--battery-max-charge-power-w",),
        cli_type=float,
        cli_help="Maximum DC power entering the battery charge path in W (default: unlimited).",
        doc="Maximum DC power entering the battery charge path; `None` is unlimited",
        summary="battery.max_charge_power_w",
    ),
    "battery_max_discharge_power_w": AppConfigField(
        affects_inputs=False,
        check=number(minimum=0, allow_none=True),
        default=None,
        cli_flags=("--battery-max-discharge-power-w",),
        cli_type=float,
        cli_help="Maximum battery AC power delivered to load in W (default: unlimited).",
        doc="Maximum battery AC power delivered to load; `None` is unlimited",
        summary="battery.max_discharge_power_w",
    ),
    "battery_power_limit_c_rate": AppConfigField(
        affects_inputs=False,
        check=number(minimum=0, min_exclusive=True, allow_none=True),
        default=None,
        cli_flags=("--battery-power-limit-c-rate",),
        cli_type=float,
        cli_help=(
            "Symmetric charge and discharge limit on the stored energy as a multiple of capacity, "
            "for example 1.0 for 1 C. Scales with battery_kwh; cannot be combined with the absolute "
            "battery_max_charge_power_w or battery_max_discharge_power_w."
        ),
        doc=(
            "Charge and discharge limit on the stored energy, as a multiple of capacity (1.0 = 1 C). It scales with "
            "`battery_kwh` and cannot be combined with `battery_max_charge_power_w` or "
            "`battery_max_discharge_power_w`"
        ),
        summary="battery.power_limit_c_rate",
    ),
    "battery_coupling": AppConfigField(
        check=choice(BATTERY_COUPLINGS, case_insensitive=False),
        default="dc",
        cli_flags=("--battery-coupling",),
        cli_help="Battery topology: 'dc' for a hybrid inverter, 'ac' for a battery inverter of its own.",
        doc=(
            '`"dc"` for a hybrid inverter that PV and the battery share, `"ac"` for a battery with its own '
            'bidirectional inverter on the AC bus beside the PV inverter. With `"ac"` the `inverter_*` keys '
            "describe the PV inverter only"
        ),
        summary="battery.coupling",
    ),
    "battery_inverter_ac_rating_kw": AppConfigField(
        exclusive_with="battery_inverter_kw_per_kwh",
        check=number(minimum=0, min_exclusive=True, allow_none=True),
        default=None,
        cli_flags=("--battery-inverter-ac-rating-kw",),
        cli_type=float,
        cli_help="AC-coupled battery inverter's AC rating in kW.",
        doc=(
            "The AC-coupled battery inverter's AC rating in kW: its AC output discharging and AC input charging. "
            'With `battery_coupling = "ac"` and a battery, set this or `battery_inverter_kw_per_kwh`'
        ),
        summary="battery.inverter_ac_rating_kw",
    ),
    "battery_inverter_kw_per_kwh": AppConfigField(
        exclusive_with="battery_inverter_ac_rating_kw",
        check=number(minimum=0, min_exclusive=True, allow_none=True),
        default=None,
        cli_flags=("--battery-inverter-kw-per-kwh",),
        cli_type=float,
        cli_help="AC-coupled battery inverter's AC rating per kWh of nominal battery capacity.",
        doc=(
            "The AC-coupled battery inverter's AC rating per kWh of nominal capacity, so a capacity sweep keeps one "
            "ratio. Instead of `battery_inverter_ac_rating_kw`"
        ),
        summary="battery.inverter_kw_per_kwh",
    ),
    "battery_inverter_efficiency": AppConfigField(
        check=_EFFICIENCY_OR_NONE,
        default=None,
        cli_flags=("--battery-inverter-efficiency",),
        cli_type=float,
        cli_help="AC-coupled battery inverter's nominal efficiency (no default).",
        doc=(
            "The AC-coupled battery inverter's nominal efficiency, from a source for the product. It has no default "
            'and is required with `battery_coupling = "ac"` and a battery. Both directions use the PVWatts '
            "part-load curve; the charging direction is an assumption"
        ),
        summary="battery.inverter_efficiency",
    ),
    "battery_replacement_cost": AppConfigField(
        revalue=True,
        check=number(minimum=0, allow_none=True),
        default=None,
        cli_flags=("--battery-replacement-cost",),
        cli_type=float,
        cli_help="Price of one battery replacement at t = 0 (default: the pack at storage_cost_per_kwh).",
        doc=(
            "The price of one battery replacement at t = 0, such as a unit sold as pack and inverter. `None` prices "
            "the pack at `storage_cost_per_kwh`. The terminal credit stays the pack's"
        ),
        # A price, so App.revalue may change it.
        summary="economics.battery_replacement_cost",
    ),
    "cost_preset": AppConfigField(
        revalue=True,
        affects_inputs=False,
        default=None,
        cli_flags=("--cost-preset",),
        cli_help="Cost preset key, for example 'residential-pt'.",
        normalizer=_underscored,
        doc=(
            "Cost preset key from the packaged defaults; see [Packaged options](options.md#cost-presets). `None` uses "
            "the {py:class}`~breos.economics.CostParams` defaults"
        ),
        summary="economics.cost_preset",
    ),
    "currency": AppConfigField(
        revalue=True,
        affects_inputs=False,
        default=None,
        cli_flags=("--currency",),
        cli_help="ISO 4217 code of the run's money, for example 'USD' (default: the tariff's, else EUR).",
        normalizer=_upper,
        doc=(
            'ISO 4217 code of the run\'s money, such as `"USD"`. Every money input must be in it, and every money '
            "output is; BREOS does not convert. Unset, the run takes the `[tariff]` currency, else EUR. A run in "
            "another currency than EUR gives its costs in that currency; see [Currency](configuration.md#currency)"
        ),
        default_doc='*the `[tariff]` currency, else `"EUR"`*',
        summary="economics.currency",
    ),
    "emissions_country": AppConfigField(
        affects_inputs=False,
        default=None,
        cli_flags=("--emissions-country",),
        cli_help="Country code for emissions, for example 'pt'.",
        normalizer=_upper,
        doc=(
            'Country code for CO2 calculations (`"PT"`, `"DE"`, `"ES"`, ...); see [Packaged '
            "options](options.md#emissions-factors)"
        ),
        summary="emissions.country",
    ),
    "pv_module": AppConfigField(
        check=optional(_module_key),
        default=None,
        cli_flags=("--pv-module",),
        cli_help="PV module catalogue key.",
        doc="Module key from the built-in catalogue. `None` uses the catalogue's first entry, `Suntech_STP550S_STC`",
        summary="pv.module",
    ),
    "load_profile": AppConfigField(
        default="demandlib_h0",
        cli_flags=("--load-profile",),
        cli_help="Load profile key; see 'breos list load-profiles'.",
        doc=(
            "Load profile key; see [Load profiles](configuration.md#load-profiles) and [Packaged "
            "options](options.md#load-profiles)"
        ),
        summary="load.load_profile",
    ),
    "rlp_directory": AppConfigField(
        default=None,
        cli_flags=("--rlp-directory",),
        cli_type=Path,
        cli_help="Directory containing licensed external RLP CSV files.",
        normalizer=_path_string,
        doc="Directory containing licensed external RLP CSVs for non-bundled load profiles",
        summary="load.rlp_directory",
    ),
    "load_profile_file": AppConfigField(
        default=None,
        cli_flags=("--load-profile-file",),
        cli_type=Path,
        cli_help=(
            "Load-profile CSV to read instead of the key's filename pattern; required for 'custom'. "
            "A relative path is taken inside rlp_directory when that is set."
        ),
        normalizer=_path_string,
        doc=(
            'Load-profile CSV to read instead of the key\'s filename pattern; required for `load_profile = "custom"`. '
            "A relative path is taken inside `rlp_directory` when that is set"
        ),
        summary="load.load_profile_file",
    ),
    "load_profile_column": AppConfigField(
        default=None,
        cli_flags=("--load-profile-column",),
        cli_help="For load_profile 'custom': the CSV column holding the load, if the file has several.",
        doc=(
            'For `load_profile = "custom"` only: the CSV column holding the load, if the file has several. Refused '
            "for any other profile"
        ),
        summary="load.load_profile_column",
    ),
    "load_profile_unit": AppConfigField(
        default=None,
        cli_flags=("--load-profile-unit",),
        cli_choices=PROFILE_UNITS,
        cli_help="For load_profile 'custom': W or kW (mean power per row), or Wh or kWh (energy per row).",
        doc=(
            'Required for `load_profile = "custom"`, and refused for any other profile: `W` or `kW` (mean power per '
            "row), or `Wh` or `kWh` (energy per row)"
        ),
        summary="load.load_profile_unit",
    ),
    "tilt": AppConfigField(
        check=_PV_CHECKS["tilt"],
        default=None,
        cli_flags=("--tilt",),
        cli_type=float,
        cli_help="PV tilt angle in degrees.",
        doc="Tilt angle (degrees). `None` estimates it from the latitude",
        summary="pv.tilt",
    ),
    "azimuth": AppConfigField(
        check=_PV_CHECKS["azimuth"],
        default=None,
        cli_flags=("--azimuth",),
        cli_type=float,
        cli_help="PV surface azimuth in degrees.",
        doc="Surface azimuth (degrees). `None` faces the equator: 180 in the northern hemisphere",
        summary="pv.azimuth",
    ),
    "transposition_model": AppConfigField(
        check=_PV_CHECKS["transposition_model"],
        default=DEFAULT_TRANSPOSITION_MODEL,
        cli_flags=("--transposition-model", "--sky-model"),
        cli_choices=tuple(TRANSPOSITION_MODELS),
        cli_help="Sky-diffusion model for POA transposition (default: isotropic).",
        doc=(
            "Sky-diffusion model used to project GHI/DHI/DNI onto the plane of array; see [Sky-diffusion "
            "model](configuration.md#sky-diffusion-transposition-model)"
        ),
        summary="pv.transposition_model",
    ),
    "albedo": AppConfigField(
        check=_PV_CHECKS["albedo"],
        default=None,
        cli_flags=("--albedo",),
        cli_type=float,
        cli_help="Ground reflectance 0-1 (default: pvlib 0.25). Excludes --surface-type.",
        doc=(
            "Ground reflectance (0-1) for the ground-diffuse component; `None` uses pvlib's 0.25 default. Mutually "
            "exclusive with `surface_type`"
        ),
        summary="pv.albedo",
    ),
    "surface_type": AppConfigField(
        check=_PV_CHECKS["surface_type"],
        default=None,
        cli_flags=("--surface-type",),
        cli_choices=tuple(SURFACE_TYPES),
        cli_help="Named ground cover mapped to an albedo (alternative to --albedo).",
        doc="Named ground cover mapped to an albedo; an alternative to `albedo`",
        summary="pv.surface_type",
    ),
    "model_perez": AppConfigField(
        check=_PV_CHECKS["model_perez"],
        default=DEFAULT_PEREZ_MODEL,
        cli_flags=("--perez-model",),
        cli_choices=tuple(PEREZ_MODELS),
        cli_help="Perez coefficient set (only used with --transposition-model perez).",
        doc='Perez coefficient set; only used when `transposition_model = "perez"`',
        summary="pv.model_perez",
    ),
    "irradiance_resampling": AppConfigField(
        default="auto",
        cli_flags=("--irradiance-resampling",),
        cli_choices=IRRADIANCE_RESAMPLING_POLICIES,
        cli_help=(
            "Hourly irradiance to 15 minutes: auto conserves interval means; clear_sky reconstructs instant samples; "
            "quadrature derives GHI from one-minute DNI and DHI."
        ),
        doc=(
            'Hourly-to-15-minute irradiance policy: `"auto"` conserves declared interval means and uses '
            '`"clear_sky"` otherwise. `"clear_sky_energy_conserving"` requires interval-mean weather metadata. '
            "Each component is conserved independently; no GHI/DNI/DHI closure is enforced. "
            '`"quadrature"` derives GHI from DNI and DHI in every one-minute sample and averages the samples '
            "into quarter-hours, conserving interval-mean DNI and DHI"
        ),
        summary="simulation.irradiance_resampling",
    ),
    "solar_position": AppConfigField(
        check=choice(SOLAR_POSITION_METHODS),
        default=DEFAULT_SOLAR_POSITION,
        cli_flags=("--solar-position",),
        cli_help=(
            "Where within each timestep the sun position is evaluated (default: auto, which reads the "
            "weather metadata and falls back to interval-start with a warning when it has no time basis)."
        ),
        doc=(
            'Where within each timestep the sun position is evaluated. `"weather"` reads the representative-time '
            "offset from content-bound weather metadata, including provider offsets for instantaneous irradiance "
            "and left- or right-labelled interval means, and raises when the metadata has no `radiation_time_basis`. "
            '`"auto"` is `"weather"` when the metadata states a `radiation_time_basis` and `"interval-start"`, '
            'with a warning, when it does not. `"interval-start"` takes the sun at the step label (the default '
            'before 0.8.0) and `"mid-interval"` half a timestep later'
        ),
        summary="pv.solar_position",
    ),
    "iam_model": AppConfigField(
        check=choice(IAM_MODELS),
        default=DEFAULT_IAM_MODEL,
        cli_flags=("--iam-model",),
        cli_help="Beam incidence-angle modifier (default: ashrae, historical compatibility).",
        doc=(
            'Beam incidence-angle modifier. `"physical"` uses pvlib\'s physical optics model and `"martin_ruiz"` its '
            "empirical model; the Ashrae default preserves historical results"
        ),
        summary="pv.iam_model",
    ),
    "diffuse_iam": AppConfigField(
        # None reads as "none": the PV model resolves the name with str().
        check=optional(choice(DIFFUSE_IAM_METHODS)),
        default=DEFAULT_DIFFUSE_IAM,
        cli_choices=tuple(DIFFUSE_IAM_METHODS),
        cli_flags=("--diffuse-iam",),
        cli_help=(
            "Whether IAM is also applied to the diffuse POA components. 'marion' weighs sky- and "
            "ground-diffuse with the view-factor-integrated selected IAM model (default: none, beam-only)."
        ),
        doc=(
            'Whether the incidence-angle modifier also applies to the diffuse POA components. `"marion"` weighs sky- '
            "and ground-diffuse with the view-factor-integrated selected IAM model (Marion 2017); the default applies "
            "IAM to beam only, a known ~0.5-1% overestimate"
        ),
        summary="pv.diffuse_iam",
    ),
    "temperature_model": AppConfigField(
        check=_PV_CHECKS["temperature_model"],
        default=DEFAULT_TEMPERATURE_MODEL,
        cli_flags=("--temperature-model",),
        cli_help=(
            "Cell-temperature model / mounting preset. The pvsyst-* and sapm-* presets use documented "
            "mounting coefficients; noct-sam additionally requires sourced module NOCT and efficiency "
            "metadata (not yet bundled). Default: faiman, open rack."
        ),
        doc=(
            'Cell-temperature model and mounting preset. `"pvsyst-*"` and `"sapm-*"` use documented mounting '
            'coefficients; `"noct-sam"` needs sourced module NOCT and efficiency metadata, which no bundled module '
            "has yet. The default is Faiman, open rack. A `pv_arrays` entry may set its own"
        ),
        summary="pv.temperature_model",
    ),
    "bifacial_model": AppConfigField(
        check=_PV_CHECKS["bifacial_model"],
        default=DEFAULT_BIFACIAL_MODEL,
        cli_flags=("--bifacial-model",),
        cli_choices=tuple(BIFACIAL_MODELS),
        cli_help=(
            "Rear-irradiance model (default: none; infinite_sheds requires bifacial module metadata and row geometry)."
        ),
        doc=(
            'Rear-irradiance model. `"none"` is front-only production; `"infinite_sheds"` needs sourced module '
            "bifaciality plus `gcr`, `pvrow_height` and `pvrow_pitch`"
        ),
        summary="pv.bifacial_model",
    ),
    "pvrow_height": AppConfigField(
        check=_PV_CHECKS["pvrow_height"],
        default=None,
        cli_flags=("--pvrow-height",),
        cli_type=float,
        cli_help="PV row center height above ground; use the same unit as --pvrow-pitch.",
        doc=(
            'Height of the PV row center above ground; required by `"infinite_sheds"`, in the same unit as '
            "`pvrow_pitch`"
        ),
        summary="pv.pvrow_height",
    ),
    "pvrow_pitch": AppConfigField(
        check=_PV_CHECKS["pvrow_pitch"],
        default=None,
        cli_flags=("--pvrow-pitch",),
        cli_type=float,
        cli_help="Distance between PV rows; use the same unit as --pvrow-height.",
        doc='Distance between adjacent PV rows; required by `"infinite_sheds"`, in the same unit as `pvrow_height`',
        summary="pv.pvrow_pitch",
    ),
    "gcr": AppConfigField(
        check=_PV_CHECKS["gcr"],
        default=0.35,
        cli_flags=("--gcr",),
        cli_type=float,
        cli_help="PV row ground coverage ratio (default: 0.35).",
        doc="Ground coverage ratio for single-axis tracking and infinite-sheds bifacial geometry",
        summary="pv.gcr",
    ),
    "resolution": AppConfigField(
        check=choice(RESOLUTIONS, case_insensitive=False),
        default="h",
        cli_flags=("--resolution",),
        cli_help="Simulation time resolution.",
        doc="Simulation time resolution",
        summary="load.resolution",
    ),
    "projection_years": AppConfigField(
        affects_inputs=False,
        check=_count,
        default=20,
        cli_flags=("--projection-years",),
        cli_type=int,
        cli_help="Economic projection horizon.",
        doc="Economic projection horizon in years",
        summary="economics.projection_years",
    ),
    "inflation_rate": AppConfigField(
        revalue=True,
        affects_inputs=False,
        check=_RATE,
        default=DEFAULT_INFLATION_RATE,
        cli_flags=("--inflation-rate",),
        cli_type=float,
        cli_help=(
            "General annual inflation. Import energy, the fixed charge and O&M escalate at it unless "
            "their own rate is set, and replacement prices inflate at it."
        ),
        doc=(
            "General annual inflation (nominal). Import energy, the fixed charge and O&M escalate at it unless their "
            "own rate is set; replacement prices inflate at it"
        ),
        summary="economics.inflation_rate",
    ),
    # ADR 0003 E2: separate escalators. None inherits inflation_rate, so a run
    # that sets none of them prices exactly as before.
    "import_price_escalation": AppConfigField(
        revalue=True,
        affects_inputs=False,
        check=optional(_RATE),
        default=None,
        cli_flags=("--import-price-escalation",),
        cli_type=float,
        cli_help="Annual escalation of the import price and the fixed charge. Default: inflation_rate.",
        doc="Annual escalation of the import price and the fixed charge; `None` uses `inflation_rate`",
        summary="economics.import_price_escalation",
    ),
    "om_escalation": AppConfigField(
        revalue=True,
        affects_inputs=False,
        check=optional(_RATE),
        default=None,
        cli_flags=("--om-escalation",),
        cli_type=float,
        cli_help="Annual escalation of O&M costs. Default: inflation_rate.",
        doc="Annual escalation of O&M costs; `None` uses `inflation_rate`",
        summary="economics.om_escalation",
    ),
    "replacement_cost_learning": AppConfigField(
        revalue=True,
        affects_inputs=False,
        check=number(minimum=0, maximum=1, max_exclusive=True),
        default=0.0,
        cli_flags=("--replacement-cost-learning",),
        cli_type=float,
        cli_help="Annual fall in the battery replacement price, on top of inflation. Default 0.",
        doc=(
            "Annual fall in the battery replacement price on top of inflation: a swap at `t` years costs `C0 × (1 + "
            "inflation_rate)^t × (1 − learning)^t`"
        ),
        summary="economics.replacement_cost_learning",
    ),
    "sell_price_inflation": AppConfigField(
        revalue=True,
        affects_inputs=False,
        check=number(minimum=-1, maximum=1, min_exclusive=True, max_exclusive=True),
        default=0.0,
        cli_flags=("--sell-price-inflation",),
        cli_type=float,
        cli_help="Annual inflation of the grid export (sell) price. Default 0.",
        doc="Annual escalation of the grid export (sell) price",
        summary="economics.sell_price_inflation",
    ),
    "export_emissions_factor_gco2_kwh": AppConfigField(
        affects_inputs=False,
        check=number(minimum=0, allow_none=True),
        default=None,
        cli_flags=("--export-emissions-factor-gco2-kwh",),
        cli_type=float,
        cli_help="Exported-generation displacement factor in gCO2/kWh (default: grid avoided factor).",
        doc=(
            "Displacement factor for exported PV, in gCO2/kWh. `None` uses the preset's avoided-grid factor and "
            "reports that fallback"
        ),
        summary="emissions.export_factor_gco2_kwh",
    ),
    "discount_rate": AppConfigField(
        revalue=True,
        affects_inputs=False,
        check=_RATE,
        default=DEFAULT_DISCOUNT_RATE,
        cli_flags=("--discount-rate",),
        cli_type=float,
        cli_help="Discount rate for NPV calculations.",
        doc="Nominal discount rate for NPV",
        summary="economics.discount_rate",
    ),
    "pv_degradation_rate": AppConfigField(
        affects_inputs=False,
        check=number(minimum=0, maximum=1, max_exclusive=True),
        default=0.005,
        cli_flags=("--pv-degradation-rate",),
        cli_type=float,
        cli_help="Annual compound PV degradation rate; year 1 has none.",
        doc=(
            "Annual PV degradation rate, compounded and counted from the start of each year, so year 1 has none; see "
            "[Module aging](../api/pv.md#module-aging)"
        ),
        summary="pv.degradation_rate",
    ),
    "calendar_model": AppConfigField(
        affects_inputs=False,
        default="naumann_lam_field_calibrated",
        cli_flags=("--calendar-model",),
        cli_help="Battery calendar aging model.",
        normalizer=_model_key,
        doc=(
            "Battery calendar aging model. The default is the v1 field calibration; "
            '`"naumann_lam_field_calibrated_v2"` is the v2 fit with Lam `Ea`/`n` fixed and `k0`/`b` fitted'
        ),
        summary="battery.calendar_model",
    ),
    "degradation_engine": AppConfigField(
        affects_inputs=False,
        default="native",
        cli_flags=("--degradation-engine",),
        cli_choices=DEGRADATION_ENGINES,
        cli_help="Battery degradation engine (default: native Naumann/Lam).",
        doc='`"native"` keeps Naumann/Lam; `"blast"` opts into a vendored BLAST cell model',
        summary="battery.degradation_engine",
    ),
    "blast_model": AppConfigField(
        affects_inputs=False,
        default=None,
        cli_flags=("--blast-model",),
        cli_help="Stable BLAST battery-model key; requires --degradation-engine blast.",
        doc='Stable BLAST model key; required with `degradation_engine = "blast"` and invalid with the native engine',
        summary="battery.blast_model",
    ),
    "inverter_efficiency": AppConfigField(
        affects_inputs=False,
        check=_EFFICIENCY,
        default=0.96,
        cli_flags=("--inverter-efficiency",),
        cli_type=float,
        cli_help="Inverter efficiency.",
        doc="Nominal inverter efficiency used by the PVWatts part-load curve",
        summary="inverter.efficiency",
    ),
    "inverter_loading_ratio": AppConfigField(
        exclusive_with="inverter_ac_rating_kw",
        affects_inputs=False,
        default=1.25,
        cli_flags=("--inverter-loading-ratio",),
        cli_type=float,
        cli_help="DC/AC oversizing ratio.",
        doc=(
            "DC/AC oversizing ratio; also sets the inverter AC rating that clips production and that CAPEX "
            "prices. Not with `inverter_ac_rating_kw`"
        ),
        summary="inverter.loading_ratio",
    ),
    "inverter_ac_rating_kw": AppConfigField(
        exclusive_with="inverter_loading_ratio",
        affects_inputs=False,
        check=number(minimum=0, min_exclusive=True, allow_none=True),
        default=None,
        cli_flags=("--inverter-ac-rating-kw",),
        cli_type=float,
        cli_help="Absolute inverter AC rating in kW, instead of --inverter-loading-ratio.",
        doc=(
            "Absolute inverter AC rating in kW, instead of `inverter_loading_ratio`: it clips production and "
            "CAPEX prices it. Not with `inverter_loading_ratio`"
        ),
        summary="inverter.ac_rating_kw",
    ),
    "inverters": AppConfigField(
        default=None,
        doc=(
            "Specified inverters, checked with `pv_strings` against `electrical_environment`; see "
            "[`[[inverters]]`](#inverters). Their summed `nominal_power_w` replaces `inverter_loading_ratio` and "
            "`inverter_ac_rating_kw`. Set with `pv_strings` and `electrical_environment`"
        ),
        summary="inverter.specs",
    ),
    "pv_strings": AppConfigField(
        default=None,
        doc=(
            "String topology: which modules of which array connect to which inverter input; see "
            "[`[[pv_strings]]`](#pv_strings). Set with `inverters` and `electrical_environment`"
        ),
        summary="inverter.strings",
    ),
    "electrical_environment": AppConfigField(
        default=None,
        doc=(
            "Design temperatures and mounting for the string check; see "
            "[`[electrical_environment]`](#electrical_environment). Set with `inverters` and `pv_strings`"
        ),
        summary="inverter.electrical_environment",
    ),
    "ac_cable": AppConfigField(
        default=None,
        doc=(
            "The AC cable from the inverter to the meter, whose I²R loss the dispatch takes from the inverter's "
            "output; see [`[ac_cable]`](#ac_cable) and [AC cable](configuration.md#ac-cable)"
        ),
        summary="inverter.ac_cable",
    ),
    "electrical_fidelity": AppConfigField(
        check=choice(("aggregate", "string"), case_insensitive=False),
        default="aggregate",
        normalizer=_lower,
        doc=(
            'How energy depends on the string topology: `"aggregate"` ignores it; `"string"` applies each MPPT '
            "input's voltage, current and power limits, the inverter's power limit and its start-up voltage. "
            '`"string"` needs `inverters`, `pv_strings` and '
            "`electrical_environment`, a wiring without structural errors, and one inverter, or several with no "
            'battery or an AC-coupled one (`battery_coupling = "ac"`)'
        ),
        summary="inverter.electrical_fidelity",
    ),
    "start_date": AppConfigField(
        default="2023-01-01",
        cli_flags=("--start-date",),
        cli_help="First simulated day: 1 January of the study year, YYYY-01-01.",
        doc=(
            "1 January of the study year, `YYYY-01-01`. The App simulates that year, or the `period` window in it. "
            "Monte Carlo does not use it for the load or weather; its `target_year` sets the study year"
        ),
        summary="load.start_date",
    ),
    # The [period] table (#242). Omitted: the whole calendar year of start_date.
    "period": AppConfigField(
        doc=(
            "Simulate only the window from `start` to `end`, local dates in the year of `start_date`, `end` "
            "exclusive. The window runs once: energy and year-1 money cover the window, and lifetime economics are "
            "`None`. See "
            "[`[period]`](#period) and [Simulate part of a year](configuration.md#simulate-part-of-a-year)"
        ),
        default_doc="*unset*",
        summary="simulation.period",
    ),
    # The [tariff] table (ADR 0002). Omitted: flat prices from the cost preset.
    "tariff": AppConfigField(
        revalue=True,
        affects_inputs=False,
        default=None,
        doc=(
            "Time-of-use import and export prices on a bundled or custom schedule, replacing the flat "
            "`costs.electricity_cost`, "
            "`costs.electricity_sold_cost` and `costs.daily_power_cost`; see [`[tariff]`](#tariff) and [Time-of-use "
            "tariffs](configuration.md#time-of-use-tariffs)"
        ),
        default_doc="*unset*",
        summary="economics.tariff",
    ),
    # The [reference_tariff] table. Omitted: the no-system household pays the
    # system's own prices, the [tariff] or the flat costs.
    "reference_tariff": AppConfigField(
        revalue=True,
        affects_inputs=False,
        default=None,
        doc=(
            "What the household would pay without the system: an import price, by period or flat, and a fixed "
            "charge, independent of the system's `[tariff]`. Unset, the no-system baseline is priced at the "
            "system's own prices; see [`[reference_tariff]`](#reference_tariff) and [No-system reference "
            "tariff](configuration.md#no-system-reference-tariff)"
        ),
        default_doc="*unset*",
        summary="economics.reference_tariff",
    ),
    "terminal_value": AppConfigField(
        revalue=True,
        default=None,
        doc=(
            "Optional estimate of the final battery pack's residual value from its capacity health; see "
            "[`[terminal_value]`](#terminal_value) and "
            "[Estimated battery residual value](configuration.md#estimated-battery-residual-value)"
        ),
        default_doc="*unset (basis = none)*",
        summary="economics.terminal_value",
    ),
    # The [smart_charging] table (ADR 0002). Omitted: greedy self-consumption.
    "smart_charging": AppConfigField(
        affects_inputs=False,
        default=None,
        doc=(
            "Battery dispatch by tariff period: grid charging toward a fixed target or, experimentally, a daily "
            "planned target in the listed charge periods, or discharge only in the listed periods; see "
            "[`[smart_charging]`](#smart_charging) and [Smart charging](configuration.md#smart-charging)"
        ),
        default_doc="*unset*",
        summary="battery.smart_charging",
    ),
    "vehicle": AppConfigField(
        doc="Vehicle equipment, prices and schedule; validated, but vehicle simulation is not available yet. See [vehicle](#vehicle)",
        default_doc="*unset*",
    ),
    "weather_file": AppConfigField(
        exclusive_with="weather_source",
        default=None,
        cli_flags=("--weather-file",),
        cli_type=Path,
        cli_help=(
            "One-year weather CSV to simulate, instead of the weather/ cache or a PVGIS fetch. "
            "A relative path is taken from the working directory."
        ),
        normalizer=_path_string,
        doc=(
            "One-year weather CSV (or gzip-compressed `.csv.gz`) to simulate, in the layout of the cached "
            "`weather/` files, instead of the `weather/` cache or a PVGIS fetch. Any filename and any location "
            "work. A relative path is taken from the working directory. The file's path and SHA-256 digest are "
            "recorded under `provenance.weather`; see [Run offline with cached "
            "weather](../how-to/offline-weather.md#name-the-weather-file)"
        ),
        summary="simulation.weather_file",
    ),
    "weather_source": AppConfigField(
        exclusive_with="weather_file",
        default=None,
        cli_flags=("--weather-source",),
        cli_help=(
            "Source part of the cached weather/<location>_tmy_*_<source>.csv (or .csv.gz) file to use when several "
            "TMY files exist for the location preset, for example 'pvgis-sarah3'."
        ),
        doc=(
            "Source part of the cached `weather/<location>_tmy_<years>_<source>.csv` (or gzip-compressed `.csv.gz`) "
            'file to load, e.g. `"pvgis-sarah3"`. Needed only when several TMY files exist for a location preset; see '
            "[Run offline with cached weather](../how-to/offline-weather.md)"
        ),
        summary="simulation.weather_source",
    ),
    # Config-file/API-only fields.
    "costs": AppConfigField(
        revalue=True,
        affects_inputs=False,
        doc="Cost overrides layered over the selected preset and the built-in defaults; see [`[costs]`](#costs)",
        default_doc="*unset*",
        summary="economics.costs",
    ),
    "pv_arrays": AppConfigField(
        default=None,
        doc=(
            "List of arrays, each with at least `modules`; other keys inherit from the top level. The array module "
            "total replaces "
            "`n_modules`; see [`[[pv_arrays]]`](#pv_arrays)"
        ),
        summary="pv.arrays",
    ),
    "tracking": AppConfigField(
        check=_PV_CHECKS["tracking"],
        default="fixed",
        doc="Tracking mode",
        summary="pv.tracking",
    ),
    "axis_tilt": AppConfigField(
        check=_PV_CHECKS["axis_tilt"],
        default=0.0,
        doc="Single-axis tracker axis tilt (degrees)",
        summary="pv.axis_tilt",
    ),
    "axis_azimuth": AppConfigField(
        check=_PV_CHECKS["axis_azimuth"],
        default=None,
        doc="Tracker axis azimuth (degrees). `None` sets it from the latitude",
        summary="pv.axis_azimuth",
    ),
    "max_angle": AppConfigField(
        check=_PV_CHECKS["max_angle"],
        default=60.0,
        doc="Single-axis tracker maximum rotation angle (degrees)",
        summary="pv.max_angle",
    ),
    "backtrack": AppConfigField(
        check=_PV_CHECKS["backtrack"],
        default=True,
        doc="Whether single-axis trackers backtrack to avoid row shading",
        summary="pv.backtrack",
    ),
    "cross_axis_tilt": AppConfigField(
        check=_PV_CHECKS["cross_axis_tilt"],
        default=0.0,
        doc="Cross-axis terrain slope for single-axis tracking (degrees)",
        summary="pv.cross_axis_tilt",
    ),
    "dual_axis_max_tilt": AppConfigField(
        check=_PV_CHECKS["dual_axis_max_tilt"],
        default=90.0,
        doc="Maximum panel tilt for dual-axis tracking (degrees)",
        summary="pv.dual_axis_max_tilt",
    ),
    # The BatteryConfig defaults, so the App and the optimizer, which leaves
    # unset battery settings to BatteryConfig, resolve the same window.
    "battery_min_soc": AppConfigField(
        affects_inputs=False,
        default=DEFAULT_MIN_SOC,
        doc="Battery SOC floor, as a fraction of nominal SOH-derated capacity",
        summary="battery.min_soc",
    ),
    "battery_max_soc": AppConfigField(
        affects_inputs=False,
        default=DEFAULT_MAX_SOC,
        doc="Battery SOC ceiling, on the same basis as `battery_min_soc`",
        summary="battery.max_soc",
    ),
    "battery_eol_percentage": AppConfigField(
        affects_inputs=False,
        check=number(minimum=0, maximum=1, min_exclusive=True, max_exclusive=True),
        default=DEFAULT_EOL_PERCENTAGE,
        doc=(
            "SOH fraction at which the battery reaches end of life: it is replaced, or, when the replacement is "
            "skipped, kept or retired"
        ),
        summary="battery.eol_percentage",
    ),
    "battery_enable_replacement": AppConfigField(
        affects_inputs=False,
        check=boolean,
        default=True,
        doc=(
            "Whether a battery that reaches end of life is replaced. `false` skips every replacement: the battery "
            "keeps ageing below its threshold, or is retired, as `battery_skipped_replacement_action` says. See "
            "[Running without replacement](configuration.md#running-without-replacement)"
        ),
        summary="battery.enable_replacement",
    ),
    "battery_allow_terminal_replacement": AppConfigField(
        affects_inputs=False,
        check=boolean,
        default=True,
        doc=(
            "Whether a battery that reaches end of life in the horizon's final degradation period is replaced. That "
            "period ends on the last simulated step: the last whole day, a trailing partial day, or a span shorter "
            "than a day. `false` skips only that replacement and its cost; the period is still aged. See "
            "[Battery replacement at the end of the horizon]"
            "(configuration.md#battery-replacement-at-the-end-of-the-horizon)"
        ),
        summary="battery.allow_terminal_replacement",
    ),
    "battery_replacement_min_remaining_years": AppConfigField(
        affects_inputs=False,
        check=number(minimum=0),
        default=0.0,
        doc=(
            "Fewest project years a replacement battery must have left to serve. A battery that reaches end of "
            "life with less than this left before the end of the horizon is not replaced; it keeps ageing below "
            "its end-of-life threshold. `0` replaces at every end of life. See "
            "[Battery replacement at the end of the horizon]"
            "(configuration.md#battery-replacement-at-the-end-of-the-horizon)"
        ),
        summary="battery.replacement_min_remaining_years",
    ),
    "battery_skipped_replacement_action": AppConfigField(
        affects_inputs=False,
        check=choice(SKIPPED_REPLACEMENT_ACTIONS, case_insensitive=False),
        default="keep",
        doc=(
            'What happens to a battery whose end-of-life replacement is skipped. `"keep"` leaves it in service '
            'below its threshold until its state of health reaches 0, when it is retired; `"retire"` switches it '
            "off, and the project finishes PV-only. Applies only when "
            "`battery_enable_replacement`, `battery_replacement_min_remaining_years` or "
            "`battery_allow_terminal_replacement` skips a swap. See "
            "[Retiring the battery instead of keeping it](configuration.md#retiring-the-battery-instead-of-keeping-it)"
        ),
        summary="battery.skipped_replacement_action",
    ),
    "battery_rte": AppConfigField(
        affects_inputs=False,
        check=_EFFICIENCY_OR_NONE,
        default=None,
        doc=(
            "Battery round-trip efficiency, split evenly across charge and discharge; `None` is "
            f"{DEFAULT_CHARGE_EFFICIENCY * DEFAULT_DISCHARGE_EFFICIENCY:.2f}"
        ),
        summary="battery.round_trip_efficiency",
    ),
    "enable_resistance_fade": AppConfigField(
        affects_inputs=False,
        check=boolean,
        default=False,
        doc=(
            "Grow the battery's internal resistance as it ages (Naumann), which lowers its charge and discharge "
            'efficiencies. Native engine only: it cannot be combined with `degradation_engine = "blast"`'
        ),
        summary="battery.enable_resistance_fade",
    ),
    "pv_loss_overrides": AppConfigField(
        default=None,
        doc='Per-component overrides (percent) for the fixed PVWatts system losses, e.g. `{"shading": 0.0}`',
        summary="pv.pv_loss_overrides",
    ),
    "horizon_profile": AppConfigField(
        default=None,
        doc=(
            "Far-horizon profile as `[[azimuth_deg, elevation_deg], ...]`. Points are circularly interpolated; direct "
            "beam is removed while the sun is on or below the terrain line. Needs weather explicitly marked as "
            "unshaded"
        ),
        summary="pv.horizon_profile",
    ),
    "battery_temperature": AppConfigField(
        default="weather",
        doc=(
            'Battery temperature used for degradation: `"weather"`, a fixed temperature in °C, or a timestamped CSV '
            "path. The indoor model then remaps it unless `battery_indoor_model` disables it"
        ),
        summary="battery.temperature",
    ),
    "battery_indoor_model": AppConfigField(
        default=None,
        doc=(
            "Indoor-temperature model settings. `None` applies the default indoor buffering; `{enabled = false}` uses "
            "`battery_temperature` without remapping. See [`[battery_indoor_model]`](#battery_indoor_model)"
        ),
        summary="battery.indoor_model",
    ),
    "execution_backend": AppConfigField(
        affects_inputs=False,
        default=DEFAULT_EXECUTION_BACKEND,
        cli_flags=("--execution-backend",),
        cli_choices=EXECUTION_BACKENDS,
        cli_help=(
            "Within-day dispatch implementation. 'python' is the numerical reference; "
            "'numba' is an optional compiled path that reproduces it bit for bit and "
            'needs the extra: pip install "breos[fast]".'
        ),
        doc=(
            'Within-day dispatch implementation. `"python"` is the numerical reference; `"numba"` is an optional '
            'compiled path that reproduces it bit for bit and needs `pip install "breos[fast]"`'
        ),
        summary="simulation.execution_backend",
    ),
    # Runner sections are accepted by App resolution so each workflow can use
    # the same base config validation. The CLI validates their own structure.
    "montecarlo": AppConfigField(
        doc=(
            "Monte Carlo study controls, read by `breos montecarlo`; see [Monte "
            "Carlo](monte-carlo.md#configure-a-study)"
        ),
        default_doc="*unset*",
    ),
    "sweep": AppConfigField(
        doc="Parameter grid, read by `breos sweep`; see [Run a parameter sweep](../how-to/parameter-sweep.md)",
        default_doc="*unset*",
    ),
}

DEFAULTS: dict[str, Any] = {name: field.default for name, field in APP_CONFIG_FIELDS.items() if field.has_default}

# Everything at the top level must be registered so typos (e.g.
# ``batery_kwh``) fail loudly instead of being silently dropped by defaults.
ALLOWED_CONFIG_KEYS: frozenset[str] = frozenset(APP_CONFIG_FIELDS)

# Cost override keys deliberately use the existing preset-catalog vocabulary;
# the canonical translation to CostParams lives in ``breos.economics`` so the
# App and lower-level construction helper cannot drift.
COST_OVERRIDE_KEYS: frozenset[str] = frozenset(COST_CONFIG_KEY_TO_PARAM)
COSTS_TABLE = TableSpec(
    "costs",
    keys=dict.fromkeys(COST_OVERRIDE_KEYS, number(minimum=0)),
    docs={
        "electricity_cost": "Flat import price, per kWh at year-1 prices",
        "electricity_sold_cost": "Flat export price, per kWh at year-1 prices",
        "daily_power_cost": "Fixed grid-connection charge per day, at year-1 prices",
        "module_cost_per_w": "PV module price per W of module rating",
        "storage_cost_per_kwh": (
            "Battery price per kWh of nominal capacity; also the t = 0 price of each replacement, unless "
            "`battery_replacement_cost` sets one"
        ),
        "inverter_cost_per_kw_hybrid": (
            'Hybrid inverter price per kW of AC rating, for a DC-coupled system with a battery (`battery_coupling = "dc"`)'
        ),
        "inverter_cost_per_kw_simple": (
            'PV inverter price per kW of AC rating, for a PV-only system and, with `battery_coupling = "ac"`, for '
            "the PV inverter whether or not there is a battery"
        ),
        "battery_inverter_cost_per_kw": (
            'An AC-coupled battery inverter\'s price per kW of its AC rating, required with `battery_coupling = "ac"` '
            "and a battery. It has no default; 0 states that another price, such as `storage_cost_per_kwh`, "
            "already includes it"
        ),
        "installation_cost_per_module": "Installation cost per PV module",
        "installation_cost_battery": "Fixed battery installation cost, for a system with a battery",
        "other_cost_per_module": "Other cost per PV module, such as cabling",
        "other_costs": "Fixed other initial cost",
        "land_cost": "Land cost, part of the initial investment",
        "maintenance_cost_per_panel": "O&M cost per PV module per year, at year-1 prices",
        "maintenance_cost": "Fixed O&M cost per year, at year-1 prices",
        "operation_cost": "Further operation cost per year, at year-1 prices",
    },
)


def _check_indoor_temperature_band(model: dict[str, Any], where: str) -> None:
    floor, ceiling = model.get("floor_c"), model.get("ceiling_c")
    if floor is not None and ceiling is not None and floor > ceiling:
        raise ValueError(f"'{where}.floor_c' must not exceed '{where}.ceiling_c'")


INDOOR_MODEL_TABLE = TableSpec(
    "battery_indoor_model",
    keys={
        "enabled": boolean,
        "setpoint_c": number(),
        "coupling_alpha": number(minimum=0, maximum=1),
        "floor_c": number(),
        "ceiling_c": number(),
    },
    check=_check_indoor_temperature_band,
    docs={
        "enabled": (
            "Remap the resolved `battery_temperature` (weather, fixed or CSV) to an indoor battery temperature "
            f"(default `{str(DEFAULT_INDOOR_MODEL_ENABLED).lower()}`). `false` uses `battery_temperature` as given"
        ),
        "setpoint_c": f"Indoor comfort midpoint in °C (default {DEFAULT_INDOOR_SETPOINT_C:g})",
        "coupling_alpha": (
            "Share of the input temperature in the indoor one, from 0 (fully insulated) to 1 (no buffering): "
            f"`alpha × input + (1 − alpha) × setpoint_c` (default {DEFAULT_INDOOR_COUPLING_ALPHA:g})"
        ),
        "floor_c": f"Lowest indoor temperature in °C (default {DEFAULT_INDOOR_FLOOR_C:g})",
        "ceiling_c": f"Highest indoor temperature in °C (default {DEFAULT_INDOOR_CEILING_C:g}); not below `floor_c`",
    },
)


def _period_date(value: Any, where: str) -> date:
    # TOML and Python give dates; JSON and the CLI give ISO strings. A window
    # starts and ends at local midnight, so a time of day is refused.
    if isinstance(value, datetime):
        raise TypeError(f"'{where}' must be a date such as 2025-06-01, not a date and time")
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"'{where}' must be an ISO date such as 2025-06-01, got {value!r}") from exc
    raise TypeError(f"'{where}' must be a date such as 2025-06-01")


def _check_period_order(table: dict[str, Any], where: str) -> None:
    if not table["start"] < table["end"]:
        raise ValueError(
            f"'{where}.start' ({table['start'].isoformat()}) must be before '{where}.end' "
            f"({table['end'].isoformat()}); the end date is exclusive, so a one-day window ends the next day"
        )


PERIOD_TABLE = TableSpec(
    "period",
    keys={"start": _period_date, "end": _period_date},
    required=frozenset({"start", "end"}),
    check=_check_period_order,
    docs={
        "start": "First simulated day, a date in the year of `start_date`. The window starts at its local midnight",
        "end": (
            "Day after the last simulated day: the window ends at its local midnight, so `end` is exclusive. At "
            "most 1 January of the next year"
        ),
    },
)


@dataclass(frozen=True)
class SimulationPeriod:
    """The window a ``period`` config simulates, from ``start`` to ``end`` exclusive.

    Both are civil dates in ``timezone``, the location's IANA zone (the #180
    calendar contract): the window starts at local midnight of ``start`` and
    ends at local midnight of ``end``, whatever clock the weather is on.
    """

    start: date
    end: date
    timezone: str

    @property
    def start_time(self) -> pd.Timestamp:
        """The instant the window starts, local midnight of ``start``."""
        return _local_midnight(self.start, self.timezone)

    @property
    def end_time(self) -> pd.Timestamp:
        """The instant the window ends, local midnight of ``end``; not simulated."""
        return _local_midnight(self.end, self.timezone)

    @property
    def days(self) -> int:
        """Civil days in the window, which the fixed charge is billed on."""
        return (self.end - self.start).days

    def record(self) -> dict[str, Any]:
        """A JSON-safe description of the window, for results and provenance."""
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "end_exclusive": True,
            "timezone": self.timezone,
            "days": self.days,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            # The window runs once, whatever projection_years says.
            "projection_years_used": 1,
        }


def _local_midnight(day: date, timezone: str) -> pd.Timestamp:
    # A zone that springs forward at midnight has no 00:00 that day, and one
    # that falls back at midnight has two; the day starts at the earliest
    # instant that exists.
    return pd.Timestamp(day).tz_localize(timezone, nonexistent="shift_forward", ambiguous=True)


def _validate_period(cfg: dict[str, Any]) -> None:
    """Check a [period] against the year of ``start_date``, and store its dates as ISO strings."""
    if cfg.get("period") is None:
        return
    table = PERIOD_TABLE.validate(cfg["period"])
    start, end = table["start"], table["end"]
    year = date.fromisoformat(cfg["start_date"]).year
    if start.year != year or end > date(year + 1, 1, 1):
        raise ValueError(
            f"'period' ({start.isoformat()} to {end.isoformat()}) must lie in {year}, the year of start_date: "
            f"'period.start' on or after {year}-01-01 and 'period.end' on or before {year + 1}-01-01. "
            f"Set start_date = '{start.year}-01-01' to simulate a window in {start.year}."
        )
    if (start, end) == (date(year, 1, 1), date(year + 1, 1, 1)):
        raise ValueError(
            f"'period' covers the whole of {year}; remove it. Without 'period' the App simulates the whole "
            "calendar year of start_date and its lifetime economics."
        )
    cfg["period"] = {"start": start.isoformat(), "end": end.isoformat()}


def _path(value: Any, where: str) -> str:
    """A file path, as a string or a path object; returned as a string."""
    return os.fspath(value) if isinstance(value, os.PathLike) else text(value, where)


# The [montecarlo] table: the MonteCarloSettings fields, checked for type
# and range before the settings are made. MonteCarloSettings makes the
# range checks again for Python callers, with those that span settings
# (max_load_scale >= min_load_scale, the weather years); run_montecarlo
# checks what needs the App config.
MONTECARLO_TABLE = TableSpec(
    "montecarlo",
    keys={
        "weather_file": _path,
        "n_runs": integer(minimum=1),
        "years_per_run": optional(integer(minimum=1)),
        "load_uncertainty": number(minimum=0),
        "load_distribution": choice(("normal", "uniform")),
        "target_year": integer(),
        "weather_start_year": optional(integer()),
        "weather_end_year": optional(integer()),
        "seed": optional(integer(minimum=0)),
        "min_load_scale": number(minimum=0),
        "max_load_scale": number(allow_none=True),
        "collect_yearly": boolean,
        "n_procs": integer(minimum=1),
        "execution_backend": optional(choice(EXECUTION_BACKENDS)),
        "model_uncertainty_pct": number(minimum=0, allow_none=True),
    },
)


def validate_montecarlo_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """The checked ``[montecarlo]`` table, empty when the config has none.

    Run before runner setup or weather access, so a typo is reported first.
    """
    if "montecarlo" not in cfg:
        return {}
    return MONTECARLO_TABLE.validate(cfg["montecarlo"])


@dataclass(frozen=True)
class ResolvedAppConfig:
    """Config values resolved to runtime objects used by the App pipeline."""

    cfg: dict[str, Any]
    lat: float
    lon: float
    timezone: str
    loc_key: str | None
    pv_arrays: list[dict[str, Any]]
    pv_params: PVModuleParams
    # Catalogue key of ``pv_params`` (the first array's module with pv_arrays),
    # which App accepts back as ``pv_module``; ``pv_params.Name`` is a display
    # name and is not.
    pv_module_key: str
    avg_module_power_w: float
    system_kwp: float
    tilt: float
    azimuth: float
    tracking: str
    axis_azimuth: float
    # AC nameplate that clips dispatch, sized like the inverter CAPEX: the DC
    # peak over inverter_loading_ratio, or inverter_ac_rating_kw. Every
    # consumer reads this one value: the dispatch, CAPEX and the reports.
    inverter_ac_capacity_w: float | None
    # The configured [tariff], or None for flat prices.
    tariff: TariffSpec | None
    # The configured [smart_charging], or None for greedy self-consumption.
    smart_charging: SmartChargingSpec | None
    cost_params: CostParams
    emissions_params: EmissionsParams | None
    # The currency of every money input and output (run_currency).
    currency: str
    # The configured [period], or None for the whole calendar year of start_date.
    period: SimulationPeriod | None = None
    # The configured [reference_tariff], or None: the no-system baseline is
    # then priced at the system's own prices.
    reference_tariff: ReferenceTariffSpec | None = None
    # The specified inverters, strings and site with their check, or None
    # without a string topology.
    electrical: ElectricalSetup | None = None
    # The configured [ac_cable], or None.
    ac_cable: AcCable | None = None
    # The inverter's efficiency curve and AC cable as the dispatch runs them,
    # or None for the PVWatts curve and no cable.
    inverter_conversion: InverterConversion | None = None
    # The module catalogue when the configuration was resolved
    # (:func:`breos.pv_modules.catalogue_snapshot`): every field above that
    # names or describes a module was taken from it.
    module_catalogue: str = ""
    # An AC-coupled battery's own inverter rating (W), resolved against the
    # nominal capacity, or None for a DC-coupled or PV-only system.
    battery_inverter_ac_capacity_w: float | None = None
    vehicle: VehicleConfig | None = None


def normalize_config_keys(config: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with hyphens changed to underscores in every table key.

    TOML and JSON input, Python mappings, and CLI overrides all pass through
    this normalization before schema validation. Reject collisions rather
    than silently picking whichever spelling happened to be visited last.
    """
    if not isinstance(config, dict):
        raise TypeError("'config' must be a dict")

    def normalize_value(value: Any, path: str) -> Any:
        if isinstance(value, dict):
            normalized: dict[str, Any] = {}
            for key, child in value.items():
                if not isinstance(key, str):
                    raise TypeError(f"Configuration table '{path}' must use string keys")
                normalized_key = key.replace("-", "_")
                if normalized_key in normalized:
                    raise ValueError(
                        f"Duplicate config key '{normalized_key}' in '{path}' after replacing hyphens with underscores"
                    )
                child_path = f"{path}.{normalized_key}" if path else normalized_key
                normalized[normalized_key] = normalize_value(child, child_path)
            return normalized
        if isinstance(value, list):
            return [normalize_value(child, f"{path}[{index}]") for index, child in enumerate(value)]
        if isinstance(value, tuple):
            return tuple(normalize_value(child, f"{path}[{index}]") for index, child in enumerate(value))
        return value

    return normalize_value(config, "")


def merge_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Apply user values over the global defaults."""
    return {**DEFAULTS, **config}


def default_module_key() -> str:
    """Return the catalog key used when a config names no PV module.

    ``DEFAULTS["pv_module"]`` is ``None`` rather than a key, so the real
    default is the catalog's insertion order. Resolving it here keeps the
    validation, resolution, and results layers from each re-deriving it and
    silently disagreeing if the catalog is reordered.
    """
    return next(iter(MODULES))


# Tracker geometry, inherited by every tracking array that does not set it.
_TRACKER_GEOMETRY_KEYS = (
    "axis_tilt",
    "axis_azimuth",
    "max_angle",
    "backtrack",
    "cross_axis_tilt",
    "dual_axis_max_tilt",
)
# Per-array overrides of electrical_environment for the string check.
_PV_ARRAY_MOUNTING_KEYS = tuple(f.name for f in fields(ArrayMounting))
_PV_ARRAY_KEYS = frozenset(
    (
        "modules",
        "module",
        "tilt",
        "azimuth",
        "tracking",
        *_TRACKER_GEOMETRY_KEYS,
        *PER_ARRAY_MODEL_OPTION_KEYS,
        *_PV_ARRAY_MOUNTING_KEYS,
        "shading",
    )
)


def _check_albedo_or_surface(settings: Mapping[str, Any], where: str) -> None:
    prefix = f"{where}." if where else ""
    if settings.get("albedo") is not None and settings.get("surface_type") is not None:
        raise ValueError(f"Set either '{prefix}albedo' or '{prefix}surface_type', not both.")


def _shading_table(value: Any, where: str) -> Any:
    _validate_shading_table(value, where)
    return value


PV_ARRAY_TABLE = TableSpec(
    "pv_arrays[i]",
    keys={
        **dict.fromkeys(_PV_ARRAY_KEYS, anything),
        **{key: optional(check) for key, check in _PV_CHECKS.items()},
        # An array that sets these must give a value.
        "temperature_model": _PV_CHECKS["temperature_model"],
        "gcr": _PV_CHECKS["gcr"],
        "modules": _count,
        "module": optional(_module_key),
        "shading": _shading_table,
    },
    check=_check_albedo_or_surface,
    docs={
        "modules": "Number of PV modules in this array, at least 1",
        "module": "Module catalogue key for this array; inherits `pv_module`",
        # Every other key is its top-level namesake, for this array only.
        **{
            key: f"`{key}` for this array; inherits the top-level value"
            for key in ("tilt", "azimuth", "tracking", *_TRACKER_GEOMETRY_KEYS, *PER_ARRAY_MODEL_OPTION_KEYS)
        },
        "mounting": "String-check mounting for this array; inherits `electrical_environment.mounting`",
        "hot_cell_temp_c": "Hottest cell temperature (°C) for this array's strings, instead of any mounting rise",
        "cold_operating_cell_temp_c": (
            "Coldest cell temperature while producing (°C) for this array's strings, instead of any mounting rise"
        ),
        "shading": (
            "Near-shading table: `azimuth_deg`, `elevation_deg`, `beam_shaded_fraction` (one row per elevation) "
            "and optional `sky_view_factor`; every array or none"
        ),
    },
)


INVERTER_TABLE = TableSpec(
    "inverters[i]",
    keys=dict.fromkeys((f.name for f in fields(InverterSpec)), anything),
    required=frozenset({"nominal_power_w"}),
    docs={
        "nominal_power_w": "AC nameplate (W)",
        "inverter_efficiency": (
            "Efficiency fraction. When every inverter sets it, their nameplate-weighted mean replaces "
            "`inverter_efficiency`"
        ),
        "is_hybrid": "Whether the inverter also connects a battery; metadata",
        "max_dc_voltage_v": "Maximum DC input voltage (V), or `inf` for none",
        "max_dc_power_w": "Maximum PV generator power the manufacturer permits (W), or `inf` for none",
        "recommended_dc_power_w": "Recommended PV generator power (W), or `inf` for none",
        "min_mppt_voltage_v": "Lower end of the MPPT voltage range (V), or 0 for none",
        "max_mppt_voltage_v": "Upper end of the MPPT voltage range (V), or `inf` for none",
        "startup_voltage_v": "Start-up voltage (V), or 0 for none",
        "mppts": "One table per MPPT input, in datasheet order; see [`mppts`](#mppts)",
        "isc_comparison_factor": (
            "Multiplier the manufacturer applies to the array's STC Isc before comparing it with an input's "
            "short-circuit rating, at least 1; 1 where Isc is compared as it is. Unset is reported as missing"
        ),
        "max_short_circuit_current_a": (
            "Maximum array short-circuit current over all inputs (A), or `inf` for none; checked when set"
        ),
        "max_usable_dc_power_w": (
            "DC power the inverter converts over all inputs (W), or `inf` for none; string-level energy applies it"
        ),
        "efficiency_curve": (
            "Measured efficiency against AC output, replacing the single efficiency: `output_fraction` (AC "
            "output over the nameplate, increasing), `efficiency` (one row per DC voltage) and `voltages_v` "
            "(one per row, increasing; optional with one row). See "
            "[Inverter efficiency curve](configuration.md#inverter-efficiency-curve)"
        ),
        "temperature_derating": (
            "Allowed AC output against ambient: `ambient_temp_c` (°C, increasing) and `max_output_fraction` "
            "(of the nameplate). See [Inverter efficiency curve](configuration.md#inverter-efficiency-curve)"
        ),
        "night_consumption_w": (
            "What the inverter draws from the AC bus while it converts nothing (W), as its datasheet states; "
            "PV surplus after the house load serves it before charging or export, and only the remainder is imported. "
            "Unset, it is not modelled. See [Inverter efficiency curve](configuration.md#inverter-efficiency-curve)"
        ),
        "ac_cable": (
            "The inverter's own AC cable, with the keys of [`[ac_cable]`](#ac_cable); with several inverters it "
            "needs string-level energy, and `[ac_cable]` is then the run they share. See "
            "[Several inverters](configuration.md#several-inverters)"
        ),
    },
)

MPPT_TABLE = TableSpec(
    "inverters[i].mppts[j]",
    keys=dict.fromkeys((f.name for f in fields(MpptSpec)), anything),
    docs={
        "max_input_current_a": "Maximum operating input current (A), or `inf` for none",
        "max_short_circuit_current_a": "Maximum array short-circuit current (A), or `inf` for none",
        "max_strings": "Strings the input takes in parallel",
        "max_string_input_current_a": (
            "Maximum operating current of each string connection (A), or `inf` for none; checked when set"
        ),
        "max_string_short_circuit_current_a": (
            "Maximum short-circuit current of each string connection (A), or `inf` for none; checked when set"
        ),
        "max_dc_power_w": "Maximum PV generator power on the input (W), or `inf` for none; checked when set",
        "max_usable_dc_power_w": ("DC power the input converts (W), or `inf` for none; string-level energy applies it"),
    },
)

PV_STRING_TABLE = TableSpec(
    "pv_strings[i]",
    keys=dict.fromkeys((f.name for f in fields(PVString)), anything),
    required=frozenset({"inverter", "mppt", "array", "modules"}),
    docs={
        "inverter": "0-based position of the inverter in `inverters`",
        "mppt": "0-based position of the input in that inverter's `mppts`",
        "array": "0-based position of the array in `pv_arrays`; 0 without `pv_arrays`",
        "modules": "Modules in series in each string",
        "parallel": "Identical strings in parallel on the input; 1 by default",
        "cable_length_m": (
            "One-way length (m) of each string's cable to the inverter input; with `cable_cross_section_mm2`, "
            "the cable's resistance replaces the flat `wiring` loss. Give it on every string or none"
        ),
        "cable_cross_section_mm2": "Conductor cross-section (mm²) of each string's cable",
        "conductor": 'Cable conductor, `"copper"` or `"aluminium"`; `"copper"` by default',
        "module_shading": (
            "Each module's own shading, for string-level energy: `azimuth_deg`, `elevation_deg` and `modules`, "
            "one entry per module in series order with a whole-module `beam_shaded_fraction` or one `substrings` "
            "table per bypass diode of the module (its `bypass_diodes`, 3 by default); every string of an array "
            "or none"
        ),
    },
)

ELECTRICAL_ENVIRONMENT_TABLE = TableSpec(
    "electrical_environment",
    keys=dict.fromkeys((f.name for f in fields(ElectricalEnvironment)), anything),
    required=frozenset({"min_design_temp_c", "max_ambient_temp_c"}),
    docs={
        "min_design_temp_c": (
            "Extreme annual minimum ambient (°C) on the basis the jurisdiction or manufacturer requires; cold Voc "
            "is taken at it"
        ),
        "max_ambient_temp_c": "Hottest design ambient (°C)",
        "hot_cell_temp_c": "Hottest cell temperature (°C), instead of `max_ambient_temp_c` plus the mounting rise",
        "cold_operating_cell_temp_c": (
            "Coldest cell temperature while producing (°C), instead of `min_design_temp_c` plus the mounting rise"
        ),
        "bifacial_rear_irradiance_w_m2": "Rear irradiance behind 1000 W/m² front for a bifacial design Isc; 135",
        "inverter_ambient_temp_c": (
            "The inverter's ambient (°C) for its `temperature_derating`, for an inverter indoors; unstated, the "
            "weather's air temperature applies"
        ),
        "bypass_diode_voltage_v": (
            "Forward voltage (V) of the modules' bypass diodes, for `module_shading`; "
            f"{DEFAULT_BYPASS_DIODE_VOLTAGE_V:g} V is assumed, and reported as assumed, when unstated"
        ),
        "conductor_temp_c": (
            "Operating temperature (°C) of the string cables' conductors; "
            f"{DEFAULT_CONDUCTOR_TEMP_C:g} °C is assumed, and reported as assumed, when unstated"
        ),
        "mounting": (
            "Module mounting, which sets the screening cell temperature rise: "
            + ", ".join(f'`"{key}"` ({rise:g} °C)' for key, rise in MOUNTING_TEMPERATURE_RISE_C.items())
            + '; `"tilted_rack"` by default'
        ),
    },
)

AC_CABLE_TABLE = TableSpec(
    "ac_cable",
    keys=dict.fromkeys((f.name for f in fields(AcCable)), anything),
    required=frozenset({"length_m", "cross_section_mm2", "phases", "voltage_v"}),
    docs={
        "length_m": "One-way length (m) from the inverter to the meter",
        "cross_section_mm2": "Cross-section (mm²) of each conductor",
        "phases": "1 or 3",
        "voltage_v": "Grid voltage (V): line to neutral for 1 phase, line to line for 3",
        "conductor": '`"copper"` or `"aluminium"`; `"copper"` by default',
        "power_factor": "The inverter's power factor; 1 by default",
        "conductor_temp_c": (
            "Operating temperature (°C) of the conductors; "
            f"{DEFAULT_CONDUCTOR_TEMP_C:g} °C is assumed, and reported as assumed, when unstated"
        ),
    },
)

_ELECTRICAL_KEYS = ("inverters", "pv_strings", "electrical_environment")


# Upper limits that may be unbounded. JSON has no infinity, so "inf" states it.
_UNBOUNDED_FIELDS = frozenset(
    {
        "max_dc_voltage_v",
        "max_dc_power_w",
        "max_mppt_voltage_v",
        "recommended_dc_power_w",
        "max_input_current_a",
        "max_short_circuit_current_a",
        "max_string_input_current_a",
        "max_string_short_circuit_current_a",
        "max_usable_dc_power_w",
    }
)


def _built(cls: Callable[..., Any], table: dict[str, Any], where: str) -> Any:
    """Construct a dataclass from a checked table, naming the table in a refusal."""
    table = {
        key: math.inf if key in _UNBOUNDED_FIELDS and isinstance(value, str) and value.lower() == "inf" else value
        for key, value in table.items()
    }
    try:
        return cls(**table)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{where}': {exc}") from exc


def _electrical_inputs(
    cfg: Mapping[str, Any],
) -> tuple[tuple[InverterSpec, ...], tuple[PVString, ...], ElectricalEnvironment] | None:
    """Validate the three string-topology keys and build their objects, or None when unset."""
    present = [key for key in _ELECTRICAL_KEYS if cfg.get(key) is not None]
    if not present:
        return None
    if len(present) < len(_ELECTRICAL_KEYS):
        missing = ", ".join(f"'{key}'" for key in _ELECTRICAL_KEYS if key not in present)
        raise ValueError(f"'inverters', 'pv_strings' and 'electrical_environment' go together; set {missing}")
    for key in ("inverters", "pv_strings"):
        if not isinstance(cfg[key], (list, tuple)):
            raise TypeError(f"'{key}' must be a list of tables")
    if not cfg["inverters"]:
        raise ValueError("'inverters' needs at least one inverter")
    inverters = []
    for i, entry in enumerate(cfg["inverters"]):
        table = INVERTER_TABLE.validate(entry, f"inverters[{i}]")
        if table.get("mppts") is not None:
            if not isinstance(table["mppts"], (list, tuple)):
                raise TypeError(f"'inverters[{i}].mppts' must be a list of tables")
            table["mppts"] = tuple(
                _built(MpptSpec, MPPT_TABLE.validate(mppt, f"inverters[{i}].mppts[{j}]"), f"inverters[{i}].mppts[{j}]")
                for j, mppt in enumerate(table["mppts"])
            )
        if table.get("ac_cable") is not None:
            cable = AC_CABLE_TABLE.validate(table["ac_cable"], f"inverters[{i}].ac_cable")
            _built(AcCable, cable, f"inverters[{i}].ac_cable")
            table["ac_cable"] = cable
        inverters.append(_built(InverterSpec, table, f"inverters[{i}]"))
    strings = tuple(
        _built(PVString, PV_STRING_TABLE.validate(entry, f"pv_strings[{i}]"), f"pv_strings[{i}]")
        for i, entry in enumerate(cfg["pv_strings"])
    )
    environment = _built(
        ElectricalEnvironment,
        ELECTRICAL_ENVIRONMENT_TABLE.validate(cfg["electrical_environment"]),
        "electrical_environment",
    )
    return tuple(inverters), strings, environment


def _check_electrical_sizing(config: Mapping[str, Any]) -> None:
    """Specified inverters size the AC side; refuse the keys that would size it twice."""
    if config.get("inverters") is None:
        return
    for key in ("inverter_ac_rating_kw", "inverter_loading_ratio"):
        if config.get(key) is not None:
            raise ValueError(f"'inverters' sets the AC rating from their nameplates; remove '{key}'")
    entries = config["inverters"] if isinstance(config["inverters"], (list, tuple)) else ()
    if config.get("inverter_efficiency") is not None and any(
        isinstance(entry, Mapping) and entry.get("inverter_efficiency") is not None for entry in entries
    ):
        raise ValueError("'inverters' set their own efficiency; remove 'inverter_efficiency'")


def _validate_bifacial_settings(
    model: Any,
    module: Any,
    pvrow_height: Any,
    pvrow_pitch: Any,
    where: str = "",
) -> None:
    """Validate opt-in bifacial module metadata and row geometry.

    Shares its predicates with :mod:`breos.pv.model_options` but reports them
    against config keys. ``pvrow_*`` geometry is required only for an active
    rear-side model. ``gcr`` is range-checked with the other PV settings,
    because tracking reads it without bifacial modeling.
    """
    prefix = f"{where}." if where else ""
    if normalise_model_name(model) == "none":
        return

    module_key = module or default_module_key()
    if module_key in MODULES and MODULES[module_key].bifaciality is None:
        raise ValueError(
            f"'{prefix}bifacial_model=infinite_sheds' requires bifaciality metadata for PV module {module_key!r}"
        )
    if pvrow_height is None:
        raise ValueError(f"'{prefix}pvrow_height' is required when bifacial_model='infinite_sheds'")
    if pvrow_pitch is None:
        raise ValueError(f"'{prefix}pvrow_pitch' is required when bifacial_model='infinite_sheds'")


def _check_fields(cfg: Mapping[str, Any]) -> None:
    """Check each top-level value its registry entry gives a ``check`` for; values are left as given."""
    for key, field in APP_CONFIG_FIELDS.items():
        if field.check is not None and key in cfg:
            field.check(cfg[key], key)


@dataclass(frozen=True)
class ValidatedSpecs:
    """What :func:`validate_config` builds while it checks, for the resolver to keep."""

    tariff: TariffSpec | None
    reference_tariff: ReferenceTariffSpec | None
    smart_charging: SmartChargingSpec | None
    currency: str
    vehicle: VehicleConfig | None = None


def validate_config(cfg: dict[str, Any]) -> ValidatedSpecs:
    """Validate user-facing App config before resolving derived values."""
    has_arrays = _validate_structure_and_location(cfg)
    _refuse_dc_battery_inverter_keys(cfg)
    _check_fields(cfg)
    _validate_pv_and_inverter(cfg, has_arrays)
    _validate_time_and_weather(cfg)
    _validate_economics(cfg)
    tariff_spec = reference_spec = None
    if cfg["tariff"] is not None or cfg["reference_tariff"] is not None:
        # Resolve location before PV module metadata or weather work. The
        # schedule's civil-time zone is part of the tariff definition, not a
        # conversion that may be deferred until the simulation is running.
        timezone = resolve_location(cfg)[2]
    if cfg["tariff"] is not None:
        tariff_spec = resolve_tariff_spec(cfg, timezone)
    currency = run_currency(cfg["currency"], tariff_spec.prices.currency if tariff_spec is not None else None)
    if cfg["reference_tariff"] is not None:
        reference_spec = resolve_reference_tariff_spec(cfg, timezone, currency)
    check_cost_currency(
        currency,
        given=cfg.get("costs") or {},
        preset=cfg.get("cost_preset"),
        tariff=tariff_spec is not None,
        # Not config_has_battery: the SOC limits it reads are checked only
        # below, and a bad one must fail there, by its key.
        battery=cfg["battery_kwh"] * 1000 > 1,
        coupling=cfg["battery_coupling"],
    )
    _validate_battery_and_degradation(cfg)
    _validate_battery_coupling(cfg)
    _validate_period(cfg)
    vehicle = None
    if "vehicle" in cfg:
        periods = tuple(tariff_spec.definition.schedule.periods) if tariff_spec is not None else None
        vehicle = validate_vehicle(cfg["vehicle"], periods=periods, montecarlo="montecarlo" in cfg)
        smart = cfg.get("smart_charging")
        if isinstance(smart, Mapping) and str(smart.get("mode")).strip().lower() == "daily_persistence":
            raise ValueError("'smart_charging.mode' = 'daily_persistence' is not supported with [vehicle]")
    smart_charging = resolve_smart_charging_spec(cfg, tariff_spec)
    _electrical_inputs(cfg)
    if cfg["electrical_fidelity"] == "string" and cfg.get("inverters") is None:
        raise ValueError("electrical_fidelity = 'string' needs 'inverters', 'pv_strings' and 'electrical_environment'")
    return ValidatedSpecs(tariff_spec, reference_spec, smart_charging, currency, vehicle)


def _tariff_study_date(value: Any, where: str) -> date:
    # TOML and Python give dates; JSON and the CLI give ISO strings.
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"'{where}' must be an ISO date such as 2027-07-01") from exc
    raise TypeError(f"'{where}' must be a date")


def _month(value: Any, where: str) -> int:
    month = integer(minimum=1)(value, where)
    if month > 12:
        raise ValueError(f"'{where}' must be a month from 1 through 12")
    return month


def _tariff_timezone(value: Any, where: str) -> str:
    timezone = text(value, where)
    try:
        ZoneInfo(timezone)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unknown IANA tariff timezone: {timezone!r} at '{where}'") from exc
    return timezone


def _holiday_dates(value: Any, where: str) -> Any:
    """Holiday dates by year; the parser checks the dates, this the four-digit year keys TOML gives."""
    if not isinstance(value, Mapping) or not value:
        raise TypeError(f"'{where}' must map years to lists of ISO dates")
    for year in value:
        if not (isinstance(year, str) and re.fullmatch(r"\d{4}", year)) or year == "0000":
            raise ValueError(f"'{where}' keys must be four-digit years from 0001 through 9999, not {year!r}")
    return value


# The keys a [tariff].custom_schedule allows. Their values are checked by
# breos.tariffs.parse_schedule_definition; these tables refuse unknown keys,
# which the parser ignores, and lowercase the selectors. A rule's season is
# checked against the schedule's own seasons in _custom_schedule.
_CUSTOM_RULE_TABLE = TableSpec(
    "custom_schedule.rules[i]",
    keys={"days": choice(("all", "saturday", "sunday", "weekday")), "season": anything, "intervals": anything},
    required=frozenset({"days", "season", "intervals"}),
)


def _custom_rules(value: Any, where: str) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)) or not value:
        raise TypeError(f"'{where}' must be a non-empty list")
    return [_CUSTOM_RULE_TABLE.validate(rule, f"{where}[{position}]") for position, rule in enumerate(value)]


_CUSTOM_HOLIDAYS_TABLE = TableSpec(
    "custom_schedule.holidays",
    keys={"day_type": choice(("saturday", "sunday", "weekday")), "dates": _holiday_dates, "source": text},
    required=frozenset({"day_type", "dates"}),
)
_CUSTOM_SCHEDULE_TABLE = TableSpec(
    "custom_schedule",
    keys={
        "identifier": text,
        "version": text,
        "timezone": _tariff_timezone,
        "cycle": choice(tuple(sorted(SCHEDULE_CYCLES))),
        "periods": anything,
        "rules": _custom_rules,
        "source": text,
        "source_url": text,
        "note": text,
        "effective_from": anything,
        "effective_to": anything,
        "holidays": optional(table(_CUSTOM_HOLIDAYS_TABLE)),
        "seasons": optional(mapping_of(text, list_of(_month, min_length=1))),
    },
    required=frozenset({"identifier", "version", "timezone", "cycle", "periods", "rules"}),
)


def _custom_schedule(value: Any, where: str) -> ScheduleDefinition:
    """Check the App's strict nested shape, then build the schedule with the shared parser."""
    raw = _CUSTOM_SCHEDULE_TABLE.validate(value, where)
    seasons = raw.get("seasons")
    selectors = ("all", *sorted(seasons)) if seasons else ("all", "dst", "standard")
    for position, rule in enumerate(raw["rules"]):
        rule["season"] = choice(selectors)(rule["season"], f"{where}.rules[{position}].season")
    identifier = raw.pop("identifier")
    return parse_schedule_definition(identifier, raw, where=where)


def _selected_schedule(table: Mapping[str, Any]) -> ScheduleDefinition:
    if "custom_schedule" in table:
        return table["custom_schedule"]
    return get_schedule_definition(table["schedule"])


def _check_tariff_prices(table: dict[str, Any], where: str) -> None:
    if ("schedule" in table) == ("custom_schedule" in table):
        raise ValueError(f"'{where}' must set exactly one of 'schedule' or 'custom_schedule'")
    _check_period_prices(table, where, ("import_prices", "export_prices"))
    _check_network_credit_periods(table, where)


def _check_period_prices(table: dict[str, Any], where: str, names: tuple[str, ...]) -> None:
    definition = _selected_schedule(table)
    schedule = definition.schedule
    periods = set(schedule.periods)
    for name in names:
        nested = [isinstance(value, Mapping) for value in table[name].values()]
        if any(nested) and not all(nested):
            raise ValueError(
                f"'{where}.{name}' mixes prices and season tables. Give every entry as a period price, or every "
                "entry as a table of period prices for one month season."
            )
        if nested and all(nested):
            validate_season_prices(definition, table[name], f"{where}.{name}")
            continue
        given = set(table[name])
        unknown = sorted(given - periods - {"all"})
        if unknown:
            raise ValueError(
                f"'{where}.{name}' has period(s) {', '.join(unknown)} that schedule {schedule.identifier!r} "
                f"does not have. Its periods: {', '.join(sorted(periods))}; 'all' prices every period."
            )
        missing = sorted(periods - given) if "all" not in given else []
        if missing:
            raise ValueError(
                f"'{where}.{name}' has no price for {', '.join(missing)}. Price every period of "
                f"{schedule.identifier!r}, or give 'all'."
            )


_PERIOD_PRICES = mapping_of(text, number(minimum=0))


def _price_entry(value: Any, where: str) -> Any:
    """A period's price, or, for a schedule with month seasons, one season's period prices."""
    if isinstance(value, Mapping):
        return _PERIOD_PRICES(value, where)
    return number(minimum=0)(value, where)


# Prices by period, or by month season and then period; a sweep can address
# either level (tariff.import_prices.peak, tariff.import_prices.winter.peak).
_PRICE_LIST = mapping_of(text, _price_entry, depth=2)

ANNUAL_NETWORK_CREDIT_TABLE = TableSpec(
    "annual_network_credit",
    keys={
        "amount_per_year": number(minimum=0),
        "network_fixed_per_year": number(minimum=0),
        "network_import_prices": _PRICE_LIST,
    },
    required=frozenset({"amount_per_year", "network_fixed_per_year", "network_import_prices"}),
    docs={
        "amount_per_year": "Annual reduction of the household's network charges, at year-1 prices; an explicit 0 is valid",
        "network_fixed_per_year": (
            "Network part of the fixed charge per year, gross, at year-1 prices: at most 365 days of "
            "`fixed_charge_per_day`, and never added to it. An explicit 0 is valid"
        ),
        "network_import_prices": (
            "Network part of the import price per kWh, gross, by period name, in the shape of `import_prices`; "
            "`all` prices every period. At most the period's import price, and never added to it"
        ),
    },
)


def _annual_network_credit(value: Any, where: str) -> dict[str, Any]:
    return ANNUAL_NETWORK_CREDIT_TABLE.validate(value, where)


def _network_credit(table: Mapping[str, Any]) -> AnnualNetworkCredit | None:
    """The annual network credit a validated [tariff] or [reference_tariff] table sets, or None."""
    credit = table.get("annual_network_credit")
    if credit is None:
        return None
    return AnnualNetworkCredit(
        amount_per_year=credit["amount_per_year"],
        network_fixed_per_year=credit["network_fixed_per_year"],
        network_import_prices=credit["network_import_prices"],
    )


def _check_network_credit_periods(table: dict[str, Any], where: str) -> None:
    # The network prices name the schedule's periods, as the import prices do.
    if "annual_network_credit" in table:
        credit = {**table, **table["annual_network_credit"]}
        _check_period_prices(credit, f"{where}.annual_network_credit", ("network_import_prices",))


TARIFF_TABLE = TableSpec(
    "tariff",
    keys={
        "schedule": choice(available_tariff_schedules()),
        "custom_schedule": _custom_schedule,
        "currency": check_currency,
        "import_prices": _PRICE_LIST,
        "export_prices": _PRICE_LIST,
        "fixed_charge_per_day": number(minimum=0),
        "boundary_policy": choice(tuple(sorted(BOUNDARY_POLICIES))),
        "study_date": _tariff_study_date,
        "annual_network_credit": _annual_network_credit,
    },
    required=frozenset({"currency", "import_prices", "export_prices"}),
    check=_check_tariff_prices,
    docs={
        "schedule": (
            "Bundled schedule key, which fixes the periods in local civil time; see "
            "[Bundled schedules](../api/tariffs.md#bundled-schedules). Set this or `custom_schedule`, not both"
        ),
        "custom_schedule": (
            "Inline schedule definition with `identifier`, `version`, `timezone`, `cycle`, `periods`, and `rules`, "
            "and optional calendar-month `seasons`; set this or `schedule`, not both. See "
            "[Custom App schedules](../api/tariffs.md#custom-app-schedules)"
        ),
        "currency": (
            "ISO 4217 code of the prices, such as `EUR`. It is the run's currency unless the top-level `currency` "
            "is set, which it must then equal; BREOS does not convert"
        ),
        "import_prices": (
            "Import price per kWh by period name, at year-1 prices; `all` prices every period. With month seasons, "
            "a table of period prices for every season instead"
        ),
        "export_prices": (
            "Export price per kWh by period name, at year-1 prices; `all` prices every period. With month seasons, "
            "a table of period prices for every season instead"
        ),
        "fixed_charge_per_day": "Fixed charge per day, at year-1 prices (default 0)",
        "boundary_policy": (
            "How a period boundary inside a step is handled. `strict`, the default, refuses it. One of "
            + ", ".join(f"`{policy}`" for policy in sorted(BOUNDARY_POLICIES))
        ),
        "study_date": "A date in the schedule's effective window, needed when the simulated year is outside it",
        "annual_network_credit": (
            "Annual network credit of the system household, capped at its own network charges; see "
            "[`annual_network_credit`](#annual_network_credit)"
        ),
    },
)
# Flat-price cost keys a tariff replaces; setting both would price energy twice.
_TARIFF_REPLACES_COSTS = ("electricity_cost", "electricity_sold_cost", "daily_power_cost")


def _validate_tariff(cfg: dict[str, Any]) -> tuple[dict[str, Any], ScheduleDefinition]:
    table = TARIFF_TABLE.validate(cfg["tariff"])
    clashing = sorted(key for key in _TARIFF_REPLACES_COSTS if key in (cfg.get("costs") or {}))
    if clashing:
        raise ValueError(
            f"A [tariff] sets the energy prices and the fixed charge, so {', '.join(f'costs.{k}' for k in clashing)} "
            "would price them twice. Remove them, or remove [tariff]."
        )
    schedule = _selected_schedule(table)
    _check_schedule_resolution(cfg, schedule)
    return table, schedule


def _check_schedule_resolution(cfg: dict[str, Any], schedule: ScheduleDefinition) -> None:
    step_minutes = int(get_hours_per_step(cfg["resolution"]) * 60)
    # The clock changes of the study year count; the optimizer's adapted config
    # has no start_date, and classifying its index checks them instead.
    start = cfg.get("start_date")
    years = None if start is None else ((start if isinstance(start, date) else date.fromisoformat(start)).year,)
    required = schedule_resolution_minutes(schedule, years)
    if required % step_minutes:
        fitting = [freq for freq in ("h", "15min") if required % int(get_hours_per_step(freq) * 60) == 0]
        remedy = (
            f'use resolution = "{fitting[0]}"' if fitting else f"App offers no step that divides {required} minutes"
        )
        identifier = schedule.schedule.identifier
        raise ValueError(
            f"Schedule {identifier!r} needs steps that divide {required} minutes, which "
            f"{cfg['resolution']!r} steps do not; {remedy}."
        )


def _check_schedule_timezone(schedule: ScheduleDefinition, timezone: str) -> None:
    metadata = schedule.schedule
    if metadata.timezone != timezone:
        raise ValueError(
            f"Schedule {metadata.identifier!r} is defined in {metadata.timezone} civil time, but the "
            f"location's timezone is {timezone}. BREOS does not move a schedule to another zone."
        )


def resolve_tariff_spec(cfg: dict[str, Any], timezone: str) -> TariffSpec | None:
    """Validate and build a tariff for App or an adapted optimizer config.

    ``cfg`` supplies ``tariff``, ``resolution``, optional ``costs`` and, from
    App, ``start_date``, whose year's clock changes count toward the
    resolution. Keeping the price-conflict and resolution checks here gives
    both entry points the same validation before they run the PV model.
    """
    if cfg["tariff"] is None:
        return None
    table, schedule = _validate_tariff(cfg)
    _check_schedule_timezone(schedule, timezone)
    prices = TariffPrices(
        currency=table["currency"],
        import_prices=table["import_prices"],
        export_prices=table["export_prices"],
        fixed_charge_per_day=table.get("fixed_charge_per_day", 0.0),
        identifier="config",
        version="1",
        annual_network_credit=_network_credit(table),
    )
    return TariffSpec(
        schedule=schedule,
        prices=prices,
        boundary_policy=table.get("boundary_policy", "strict"),
        study_date=table.get("study_date"),
    )


def _check_reference_prices(table: dict[str, Any], where: str) -> None:
    if "schedule" in table and "custom_schedule" in table:
        raise ValueError(f"'{where}' sets both 'schedule' and 'custom_schedule'; set one, or neither for a flat price")
    if "schedule" in table or "custom_schedule" in table:
        _check_period_prices(table, where, ("import_prices",))
        _check_network_credit_periods(table, where)
        return
    # Without a schedule the reference is one flat price.
    if set(table["import_prices"]) != {"all"} or isinstance(table["import_prices"].get("all"), Mapping):
        raise ValueError(
            f"'{where}' has no schedule, so it is one flat price: set '{where}.import_prices' = {{ all = <price> }}, "
            "or set a schedule or custom_schedule"
        )
    network = (table.get("annual_network_credit") or {}).get("network_import_prices")
    if network is not None and (set(network) != {"all"} or isinstance(network.get("all"), Mapping)):
        raise ValueError(
            f"'{where}' has no schedule, so its network price is flat too: set "
            f"'{where}.annual_network_credit.network_import_prices' = {{ all = <price> }}"
        )
    scheduled = sorted(key for key in ("boundary_policy", "study_date") if key in table)
    if scheduled:
        raise ValueError(
            f"{', '.join(f'{where}.{key}' for key in scheduled)} applies to a schedule, and '{where}' has none"
        )


REFERENCE_TARIFF_TABLE = TableSpec(
    "reference_tariff",
    keys={
        "schedule": choice(available_tariff_schedules()),
        "custom_schedule": _custom_schedule,
        "currency": check_currency,
        "import_prices": _PRICE_LIST,
        "fixed_charge_per_day": number(minimum=0),
        "boundary_policy": choice(tuple(sorted(BOUNDARY_POLICIES))),
        "study_date": _tariff_study_date,
        "import_price_escalation": number(minimum=-1, min_exclusive=True),
        "annual_network_credit": _annual_network_credit,
    },
    required=frozenset({"currency", "import_prices", "fixed_charge_per_day"}),
    check=_check_reference_prices,
    docs={
        "schedule": (
            "Bundled schedule key of the reference; see [Bundled schedules](../api/tariffs.md#bundled-schedules). "
            "Set this, `custom_schedule`, or neither for one flat price"
        ),
        "custom_schedule": (
            "Inline schedule definition of the reference, in the shape of `tariff.custom_schedule`; set this, "
            "`schedule`, or neither for one flat price"
        ),
        "currency": (
            "ISO 4217 code of the prices. Must be the run's currency: `currency`, else the `[tariff]` currency, "
            "else EUR"
        ),
        "import_prices": (
            "Import price per kWh by period name, at year-1 prices; `all` prices every period. With month seasons, "
            "a table of period prices for every season instead. Without a schedule, only `all`"
        ),
        "fixed_charge_per_day": "Fixed charge per day without the system, at year-1 prices; an explicit 0 is valid",
        "boundary_policy": (
            "How a period boundary inside a step is handled, as in `tariff.boundary_policy`; needs a schedule"
        ),
        "study_date": "A date in the schedule's effective window, as in `tariff.study_date`; needs a schedule",
        "import_price_escalation": (
            "Annual escalation of the reference energy and fixed charge. Default: the system's import escalation"
        ),
        "annual_network_credit": (
            "Annual network credit of the no-system household, capped at its own network charges, in the shape of "
            "[`annual_network_credit`](#annual_network_credit). Without a schedule, "
            "`network_import_prices = { all = <price> }`"
        ),
    },
)


def resolve_reference_tariff_spec(cfg: dict[str, Any], timezone: str, currency: str) -> ReferenceTariffSpec | None:
    """Validate and build the no-system reference tariff for App or an adapted optimizer config.

    ``cfg`` supplies ``reference_tariff`` and ``resolution``, and from App
    ``start_date``, as :func:`resolve_tariff_spec` reads them. ``currency``
    is the run's (:func:`~breos.tariffs.run_currency`): the reference must be
    in it, since BREOS does not convert.
    """
    if cfg.get("reference_tariff") is None:
        return None
    table = REFERENCE_TARIFF_TABLE.validate(cfg["reference_tariff"])
    if table["currency"] != currency:
        raise ValueError(
            f"'reference_tariff.currency' is {table['currency']}, but the run is in {currency}. BREOS does not "
            f"convert currencies: give the reference prices in {currency}."
        )
    schedule: ScheduleDefinition | None = None
    if "schedule" in table or "custom_schedule" in table:
        schedule = _selected_schedule(table)
        _check_schedule_resolution(cfg, schedule)
        _check_schedule_timezone(schedule, timezone)
    prices = TariffPrices(
        currency=table["currency"],
        import_prices=table["import_prices"],
        # The no-system household exports nothing.
        export_prices={"all": 0.0},
        fixed_charge_per_day=table["fixed_charge_per_day"],
        identifier="reference",
        version="1",
        annual_network_credit=_network_credit(table),
    )
    return ReferenceTariffSpec(
        prices=prices,
        schedule=schedule,
        boundary_policy=table.get("boundary_policy", "strict"),
        study_date=table.get("study_date"),
        import_price_escalation=table.get("import_price_escalation"),
    )


def _check_smart_charging_keys(table: dict[str, Any], where: str, coupling: str = "dc") -> None:
    check_overlap_policy(table["mode"], table.get("overlap_policy", "reject"))
    check_mode_settings(table["mode"], [key for key in table if key != "mode"], coupling, table.get(FORECAST_KEY))
    if table["mode"] != "disabled":
        check_period_overlap(
            table.get("charge_periods", ()), table["discharge_periods"], table.get("overlap_policy", "reject")
        )


SMART_CHARGING_TABLE = TableSpec(
    "smart_charging",
    keys={
        "mode": choice(SMART_CHARGING_MODES),
        "overlap_policy": choice(OVERLAP_POLICIES),
        "target_usable_fraction": number(minimum=0, maximum=1),
        "charge_periods": list_of(text, min_length=1),
        "discharge_periods": list_of(text, min_length=1),
        "grid_charge_efficiency": number(minimum=0, maximum=1, min_exclusive=True),
        # None, as well as omitting the key, leaves site import unlimited.
        "grid_import_limit_w": number(minimum=0, min_exclusive=True, allow_none=True),
        **{name: integer(minimum=minimum) for name, (_default, minimum) in PLANNER_SETTINGS.items()},
        WEAR_COST_KEY: number(minimum=0),
        DECISION_BOUNDARY_KEY: choice(DECISION_BOUNDARIES),
        FORECAST_KEY: choice(FORECASTS),
        FORECAST_FILES_KEY: list_of(text, min_length=1),
        PUBLICATION_DELAY_KEY: number(minimum=0),
    },
    required=frozenset({"mode"}),
    check=_check_smart_charging_keys,
    docs={
        "overlap_policy": (
            "Default `reject`: charge and discharge periods must be disjoint. `hold_target` (`fixed_target` and "
            "`daily_persistence`) uses the grid target as the discharge floor on overlapping steps. Above it the "
            "battery may discharge; below it the grid may charge; PV may charge above it. Under "
            "`daily_persistence` the floor is each day's planned target, and 0 on a day with no grid target"
        ),
        "mode": (
            "`fixed_target` charges from the grid toward a fixed target; `daily_persistence` (experimental, App "
            "only) plans each day's target; `discharge_only` discharges only in `discharge_periods` and never "
            "charges from the grid; `disabled` is greedy self-consumption"
        ),
        "target_usable_fraction": (
            "Grid-charging target as a fraction of the usable window: 0 is `battery_min_soc`, 1 is "
            "`battery_max_soc`. `fixed_target` only"
        ),
        "forecast_horizon_days": (
            f"`daily_persistence` only: civil days the planner looks ahead, today included; under "
            f"`charge_window_start`, the windows that start within that many days. An integer of at "
            f"least {PLANNER_SETTINGS['forecast_horizon_days'][1]}; default "
            f"{PLANNER_SETTINGS['forecast_horizon_days'][0]}"
        ),
        "target_levels": (
            f"`daily_persistence` only: candidate targets, evenly spaced from 0 to 1 of the usable window. An "
            f"integer of at least {PLANNER_SETTINGS['target_levels'][1]} (one level selects target 0); default "
            f"{PLANNER_SETTINGS['target_levels'][0]}"
        ),
        "soc_states": (
            f"`daily_persistence` only: stored-energy grid points of the planner's value function. An integer of "
            f"at least {PLANNER_SETTINGS['soc_states'][1]}; default {PLANNER_SETTINGS['soc_states'][0]}"
        ),
        WEAR_COST_KEY: (
            "`daily_persistence` only: a planning weight, in the tariff's currency per kWh of DC energy the "
            "battery discharges, that the planner adds to each day's cost. Not a degradation model: realised "
            "ageing still comes from the battery model. At least 0; default 0"
        ),
        DECISION_BOUNDARY_KEY: (
            "`daily_persistence` only: when the planner decides a target. `civil_day` (default) decides one per "
            "civil day at local midnight; `charge_window_start` decides one per charge window at its start, held "
            "until the next window starts. A window is a run of consecutive steps in `charge_periods`; it may "
            "cross midnight or span a weekend. Needs a tariff period outside `charge_periods`"
        ),
        FORECAST_KEY: (
            "`daily_persistence` only (experimental): what the planner forecasts PV with. `persistence` (default) "
            "repeats the day before the decision; `weather` reads day-ahead forecast weather from "
            f"`{FORECAST_FILES_KEY}` and takes, for each planned step, the youngest forecast published before "
            "the decision, keeping persistence beyond the last lead; `perfect` plans on the run's own PV, load "
            "and temperature, a perfect-information reference no controller can have. Load and temperature are "
            "persistence under `persistence` and `weather`"
        ),
        FORECAST_FILES_KEY: (
            f'`{FORECAST_KEY} = "weather"` only: forecast weather files, the first for lead day 1 (the values '
            "forecast at least one day before their valid hour), the next for lead day 2, and so on, each in "
            "the layout of `weather_file` and on the same year. A relative path is taken from the working "
            "directory"
        ),
        PUBLICATION_DELAY_KEY: (
            f'`{FORECAST_KEY} = "weather"` only: hours between a forecast\'s nominal issue (its valid hour less '
            "one day per lead day) and the first decision that may use it. At least 0; default "
            f"{DEFAULT_PUBLICATION_DELAY_HOURS:g}"
        ),
        "charge_periods": "Tariff periods in which the grid may charge the battery. Not with `discharge_only`",
        "discharge_periods": (
            "Tariff periods in which the battery may discharge to the load; overlap needs `hold_target`. Under "
            "`discharge_only` the battery holds its charge in every other period"
        ),
        "grid_charge_efficiency": (
            "AC-to-DC conversion efficiency of the grid-charging path, before the battery's own charge efficiency. "
            'No default. Not with `discharge_only`, nor with `battery_coupling = "ac"`, whose battery inverter '
            "converts grid charge at `battery_inverter_efficiency`"
        ),
        "grid_import_limit_w": (
            "Site import limit in W for grid charging, which may import up to the limit minus the load's import. "
            "Load import is never cut. Unset is unlimited. Not with `discharge_only`. With `[vehicle]`, "
            'also accepted without a stationary battery alongside `mode = "disabled"`; stationary instructions stay disabled'
        ),
    },
)
# The same table for an AC-coupled battery, which refuses grid_charge_efficiency.
AC_SMART_CHARGING_TABLE = replace(SMART_CHARGING_TABLE, check=partial(_check_smart_charging_keys, coupling="ac"))


def smart_charging_table(coupling: str) -> TableSpec:
    """The [smart_charging] table for a battery with ``coupling``."""
    return AC_SMART_CHARGING_TABLE if coupling == "ac" else SMART_CHARGING_TABLE


TERMINAL_VALUE_TABLE = TableSpec(
    "terminal_value",
    {"basis": choice(("none", "battery_health_fraction"))},
    docs={
        "basis": 'Accounting sensitivity: `"none"` (default) or `"battery_health_fraction"`. Inherits the resolved '
        "replacement-pack price and physical `battery_eol_percentage`; introduces no separate threshold. "
        "No resale value, PV, inverter or stored-energy credit; projected optimization ignores this table.",
    },
)


# The nested tables of an App config, by top-level key. ``pv_arrays`` is a
# list of tables and is not here: a sweep or merge cannot address one entry.
NESTED_TABLE_SPECS: Mapping[str, TableSpec] = {
    "costs": COSTS_TABLE,
    "battery_indoor_model": INDOOR_MODEL_TABLE,
    "tariff": TARIFF_TABLE,
    "reference_tariff": REFERENCE_TARIFF_TABLE,
    "terminal_value": TERMINAL_VALUE_TABLE,
    "smart_charging": SMART_CHARGING_TABLE,
    "vehicle": VEHICLE_TABLE,
    "period": PERIOD_TABLE,
    "electrical_environment": ELECTRICAL_ENVIRONMENT_TABLE,
    "ac_cable": AC_CABLE_TABLE,
}


def _checked_smart_charging(
    value: Any,
    schedule: TariffSchedule | None,
    battery_kwh: float,
    battery_key: str = "battery_kwh",
    coupling: str = "dc",
) -> dict[str, Any]:
    """Check a [smart_charging] table against the configured tariff schedule and battery.

    ``battery_key`` names the setting ``battery_kwh`` came from, for the error.
    ``coupling`` selects the table: an AC-coupled battery refuses
    ``grid_charge_efficiency`` where a DC-coupled one needs it (ADR 0004 C7).
    """
    table = smart_charging_table(coupling).validate(value)
    mode = table["mode"]
    if mode == "disabled":
        return table
    if schedule is None:
        raise ValueError(
            f"'smart_charging.mode' = '{mode}' needs a [tariff]: its charge and discharge periods are tariff periods"
        )
    if not battery_kwh > 0:
        raise ValueError(f"'smart_charging.mode' = '{mode}' needs a battery; set {battery_key} > 0")
    periods = schedule.periods
    for name in ("charge_periods", "discharge_periods"):
        unknown = sorted(set(table.get(name, ())) - set(periods))
        if unknown:
            raise ValueError(
                f"'smart_charging.{name}' has period(s) {', '.join(unknown)} that schedule {schedule.identifier!r} "
                f"does not have. Its periods: {', '.join(sorted(periods))}."
            )
    check_charge_windows(table.get(DECISION_BOUNDARY_KEY), table.get("charge_periods", ()), periods)
    for index, path in enumerate(table.get(FORECAST_FILES_KEY, ())):
        if not Path(path).is_file():
            raise FileNotFoundError(f"smart_charging.{FORECAST_FILES_KEY}[{index}] not found: {path}")
    return table


def resolve_smart_charging_spec(
    cfg: dict[str, Any], tariff_spec: TariffSpec | None, battery_key: str = "battery_kwh"
) -> SmartChargingSpec | None:
    """Validate and build a smart-charging spec for App or an adapted optimizer config.

    ``cfg`` supplies ``smart_charging`` and ``battery_kwh``; ``tariff_spec``
    is what :func:`resolve_tariff_spec` returned for the same config, whose
    periods the charge and discharge periods must name. ``battery_key`` is
    the setting the error names when ``battery_kwh`` is zero.
    """
    if cfg.get("smart_charging") is None:
        return None
    value = cfg["smart_charging"]
    # Decision 18 adds only a site limit. Keep the stationary disabled spec
    # and its instructions identical, and retain the limit in cfg for the
    # future vehicle runner. Every other key still goes through the same
    # table and mode checks.
    if "vehicle" in cfg and isinstance(value, Mapping) and str(value.get("mode")).strip().lower() == "disabled":
        if "grid_import_limit_w" in value:
            SMART_CHARGING_TABLE.keys["grid_import_limit_w"](
                value["grid_import_limit_w"], "smart_charging.grid_import_limit_w"
            )
            value = {key: item for key, item in value.items() if key != "grid_import_limit_w"}
    schedule = tariff_spec.definition.schedule if tariff_spec is not None else None
    coupling = cfg.get("battery_coupling", "dc")
    table = _checked_smart_charging(value, schedule, cfg["battery_kwh"], battery_key, coupling)
    if table["mode"] == "disabled":
        return SmartChargingSpec(mode="disabled")
    return SmartChargingSpec(
        mode=table["mode"],
        overlap_policy=table.get("overlap_policy", "reject"),
        target_usable_fraction=table.get("target_usable_fraction"),
        # A period named twice is still one period.
        charge_periods=tuple(dict.fromkeys(table.get("charge_periods", ()))),
        discharge_periods=tuple(dict.fromkeys(table["discharge_periods"])),
        grid_charge_efficiency=table.get("grid_charge_efficiency"),
        grid_import_limit_w=table.get("grid_import_limit_w"),
        # Omitted planner settings take the planner's defaults, which the
        # spec then records.
        **{name: table[name] for name in PLANNER_KEYS if name in table},
        battery_coupling=coupling,
    )


def _validate_structure_and_location(cfg: dict[str, Any]) -> bool:
    """Validate top-level keys, required inputs, and location structure."""
    unknown = set(cfg) - ALLOWED_CONFIG_KEYS
    if unknown:
        available = ", ".join(sorted(ALLOWED_CONFIG_KEYS))
        raise ValueError(f"Unknown config key(s): {', '.join(sorted(unknown))}. Available: {available}")

    validate_montecarlo_config(cfg)

    for key in ("location", "annual_consumption_kwh"):
        if key not in cfg:
            raise ValueError(f"Missing required config key: '{key}'")

    has_arrays = bool(cfg.get("pv_arrays"))
    if not has_arrays and "n_modules" not in cfg:
        raise ValueError("Missing required config key: 'n_modules'")

    loc = cfg["location"]
    if isinstance(loc, dict):
        for field in ("latitude", "longitude", "timezone"):
            if field not in loc:
                raise ValueError(f"Custom location must include '{field}'")
        number(minimum=-90, maximum=90)(loc["latitude"], "location.latitude")
        number(minimum=-180, maximum=180)(loc["longitude"], "location.longitude")
        if not isinstance(loc["timezone"], str):
            raise TypeError("'location.timezone' must be an IANA timezone string")
        try:
            ZoneInfo(loc["timezone"])
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Unknown IANA timezone: {loc['timezone']!r}") from exc
    elif not isinstance(loc, str):
        raise TypeError("'location' must be a string key or a dict with latitude/longitude/timezone")

    _validate_load_profile(cfg)

    return has_arrays


def _validate_load_profile(cfg: dict[str, Any]) -> None:
    """Make the profile key canonical and check the options that go with it.

    The file itself is resolved when the profile is loaded.
    """
    cfg["load_profile"] = resolve_profile_key(cfg["load_profile"])
    validate_profile_options(cfg["load_profile"], cfg["load_profile_column"], cfg["load_profile_unit"])
    if cfg["load_profile"] == "custom" and cfg["load_profile_file"] is None:
        raise ValueError("load_profile 'custom' needs load_profile_file, the CSV to read.")


def _validate_pv_and_inverter(cfg: dict[str, Any], has_arrays: bool) -> None:
    """Validate PV sizing, array geometry, sky models, and inverter inputs."""
    if not has_arrays:
        _count(cfg["n_modules"], "n_modules")
    _check_albedo_or_surface(cfg, "")
    if has_arrays:
        if not isinstance(cfg["pv_arrays"], list):
            raise TypeError("'pv_arrays' must be a list")
        for i, arr in enumerate(cfg["pv_arrays"]):
            PV_ARRAY_TABLE.validate(arr, f"pv_arrays[{i}]")
            if "modules" not in arr:
                raise ValueError(f"'pv_arrays[{i}].modules' must be >= 1")
            module = arr.get("module", cfg.get("pv_module"))
            mounting_keys = [key for key in _PV_ARRAY_MOUNTING_KEYS if key in arr]
            if mounting_keys and cfg.get("electrical_environment") is None:
                raise ValueError(
                    f"'pv_arrays[{i}].{mounting_keys[0]}' applies to the string check; "
                    "set 'inverters', 'pv_strings' and 'electrical_environment'"
                )
            _validate_bifacial_settings(
                arr.get("bifacial_model", cfg["bifacial_model"]),
                module or default_module_key(),
                arr.get("pvrow_height", cfg["pvrow_height"]),
                arr.get("pvrow_pitch", cfg["pvrow_pitch"]),
                where=f"pv_arrays[{i}]",
            )
        by_modules = _module_shaded_arrays(cfg)
        for i, arr in enumerate(cfg["pv_arrays"]):
            if "shading" in arr and i in by_modules:
                raise ValueError(
                    f"'pv_arrays[{i}]' has a 'shading' table and strings with 'module_shading'; its modules' "
                    "shading sets its irradiance, so give one or the other"
                )
        shaded = ["shading" in arr or i in by_modules for i, arr in enumerate(cfg["pv_arrays"])]
        if any(shaded) and not all(shaded):
            raise ValueError(
                f"'pv_arrays[{shaded.index(False)}]' has no 'shading' table. Give every array a shading table "
                "or strings with module_shading, or none; an unshaded array takes an all-zero table"
            )
    if not has_arrays:
        _validate_bifacial_settings(
            cfg["bifacial_model"],
            cfg.get("pv_module") or default_module_key(),
            cfg["pvrow_height"],
            cfg["pvrow_pitch"],
        )
    if (
        cfg["inverter_ac_rating_kw"] is None
        and cfg.get("inverters") is None
        and _finite_real(cfg["inverter_loading_ratio"], "inverter_loading_ratio") <= 0
    ):
        raise ValueError("'inverter_loading_ratio' must be > 0")


def _module_shaded_arrays(cfg: Mapping[str, Any]) -> set[int]:
    """The arrays some string gives ``module_shading`` for, from the raw ``pv_strings``."""
    strings = cfg.get("pv_strings") or []
    return {
        entry["array"]
        for entry in strings
        if isinstance(entry, Mapping) and entry.get("module_shading") is not None and _is_int(entry.get("array"))
    }


def _validate_shading_table(value: Any, where: str) -> None:
    """Raise unless ``value`` is a valid near-shading table, naming ``where``."""
    if not isinstance(value, Mapping):
        raise ValueError(f"'{where}' must be a table of azimuth_deg, elevation_deg and beam_shaded_fraction")
    try:
        ShadingTable.from_mapping(value)
    except ValueError as error:
        raise ValueError(f"'{where}': {error}") from None


def _validate_time_and_weather(cfg: dict[str, Any]) -> None:
    """Validate simulation horizon, resolution, and solar-model selections."""
    cfg["irradiance_resampling"] = validate_irradiance_resampling(cfg["irradiance_resampling"])
    _validate_weather_source(cfg)
    _validate_weather_file(cfg)
    cfg["horizon_profile"] = normalise_horizon_profile(cfg["horizon_profile"])
    cfg["bifacial_model"] = normalise_model_name(cfg["bifacial_model"])
    overrides = cfg.get("pv_loss_overrides")
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise TypeError("'pv_loss_overrides' must be a dict of loss component percentages")
        for name, value in overrides.items():
            number(minimum=0, maximum=100)(value, f"pv_loss_overrides[{name!r}]")
        # Resolve the component names here so a typo cannot survive App
        # construction and fail after a TMY weather request.
        resolve_pvwatts_losses(overrides)


def _validate_weather_source(cfg: dict[str, Any]) -> None:
    """Check ``weather_source`` can name a cached TMY file for a location preset.

    The value is matched against the ``<source>`` part of
    ``<location>_tmy_<years>_<source>.csv`` in the ``weather/`` cache, which
    only location presets consult; coordinate-dict locations always fetch.
    Whether a matching file exists is checked when the weather is loaded.
    """
    source = cfg["weather_source"]
    if source is None:
        return
    if not isinstance(source, str) or re.fullmatch(r"[\w-]+", source) is None:
        raise ValueError(
            "'weather_source' must be the source part of a cached weather filename, such as "
            f"'pvgis-sarah3' (letters, digits, '_' or '-'), got {source!r}"
        )
    if isinstance(cfg["location"], dict):
        raise ValueError(
            "'weather_source' selects a cached weather file by location preset key; "
            "coordinate-dict locations have no cache key and always fetch PVGIS weather"
        )


def _validate_weather_file(cfg: dict[str, Any]) -> None:
    """Check ``weather_file`` names an existing file, and is not combined with ``weather_source``.

    A relative path is taken from the working directory, as for the other
    path keys. The file is read when the weather is loaded.
    """
    weather_file = cfg["weather_file"]
    if weather_file is None:
        return
    if cfg["weather_source"] is not None:
        raise ValueError(
            "'weather_file' names the weather file directly, and 'weather_source' selects one from the "
            "weather/ cache; set one of them"
        )
    if not Path(weather_file).is_file():
        raise FileNotFoundError(f"weather_file not found: {weather_file}")


def _validate_economics(cfg: dict[str, Any]) -> None:
    """Validate financial and optional export-emissions inputs."""
    if cfg["terminal_value"] is not None:
        cfg["terminal_value"] = TERMINAL_VALUE_TABLE.validate(cfg["terminal_value"])
        cfg["terminal_value"].setdefault("basis", "none")
    if "costs" in cfg:
        COSTS_TABLE.validate(cfg["costs"])
    if cfg["currency"] is not None:
        cfg["currency"] = check_currency(cfg["currency"], "currency")


def _validate_battery_and_degradation(cfg: dict[str, Any]) -> None:
    """Validate battery dispatch and explicit degradation-engine selection."""
    min_soc = _finite_real(cfg["battery_min_soc"], "battery_min_soc")
    max_soc = _finite_real(cfg["battery_max_soc"], "battery_max_soc")
    if not 0 <= min_soc < max_soc <= 1:
        raise ValueError("'battery_min_soc' and 'battery_max_soc' must satisfy 0 <= min < max <= 1")
    if cfg["battery_power_limit_c_rate"] is not None:
        if cfg["battery_max_charge_power_w"] is not None or cfg["battery_max_discharge_power_w"] is not None:
            raise ValueError(
                "'battery_power_limit_c_rate' limits both directions from capacity; do not also "
                "set 'battery_max_charge_power_w' or 'battery_max_discharge_power_w'"
            )
    battery_temperature = cfg["battery_temperature"]
    if isinstance(battery_temperature, Real) and not isinstance(battery_temperature, bool):
        _finite_real(battery_temperature, "battery_temperature")
    elif not isinstance(battery_temperature, str):
        raise TypeError("'battery_temperature' must be 'weather', a CSV path, or a finite temperature")
    elif battery_temperature.lower() != "weather" and not Path(battery_temperature).is_file():
        raise FileNotFoundError(f"battery_temperature file not found: {battery_temperature}")
    if cfg["battery_indoor_model"] is not None:
        INDOOR_MODEL_TABLE.validate(cfg["battery_indoor_model"])
    check_calendar_model(cfg["calendar_model"], "calendar_model")
    start_date = cfg["start_date"]
    # datetime subclasses date, so it is excluded before the date case.
    if isinstance(start_date, date) and not isinstance(start_date, datetime):
        start_date = start_date.isoformat()
        cfg["start_date"] = start_date
    elif not isinstance(start_date, str):
        raise TypeError("'start_date' must be an ISO date string or datetime.date (YYYY-MM-DD)")
    try:
        start = date.fromisoformat(start_date)
    except ValueError as exc:
        raise ValueError("'start_date' must be a valid ISO date (YYYY-MM-DD)") from exc
    if (start.month, start.day) != (1, 1):
        raise ValueError(
            f"'start_date' must be 1 January of the study year, got {cfg['start_date']!r}. BREOS simulates "
            f"whole calendar years: weather, load and every projected year start on 1 January, so use "
            f"'{start.year}-01-01'. To simulate part of the year, set [period]."
        )

    cfg["battery_replacement_min_remaining_years"] = float(cfg["battery_replacement_min_remaining_years"])

    # Validated through breos.execution so App and Monte Carlo cannot disagree
    # about which names exist. The default stays "python": the compiled path is
    # an optimisation that must be asked for, never selected by ambient state.
    cfg["execution_backend"] = validate_execution_backend(cfg["execution_backend"])

    degradation_engine = str(cfg["degradation_engine"]).strip().lower()
    if degradation_engine not in DEGRADATION_ENGINES:
        raise ValueError(f"'degradation_engine' must be one of: {', '.join(DEGRADATION_ENGINES)}")
    cfg["degradation_engine"] = degradation_engine

    if degradation_engine == "blast":
        if cfg["battery_kwh"] <= 0:
            raise ValueError("'degradation_engine=blast' requires 'battery_kwh' > 0")
        if "montecarlo" in cfg:
            raise ValueError("'degradation_engine=blast' is not supported with Monte Carlo yet")
        if cfg["enable_resistance_fade"]:
            raise ValueError("'degradation_engine=blast' cannot be combined with 'enable_resistance_fade'")
        if cfg["blast_model"] not in ENABLED_BLAST_MODEL_KEYS:
            available = ", ".join(ENABLED_BLAST_MODEL_KEYS)
            raise ValueError(f"Unknown blast_model {cfg['blast_model']!r}. Available: {available}")
    elif cfg["blast_model"] is not None:
        raise ValueError("'blast_model' requires 'degradation_engine=blast'; native degradation remains the default")


# An AC-coupled battery's own inverter (ADR 0004 C2): "dc" refuses them all.
BATTERY_INVERTER_KEYS: tuple[str, ...] = (
    "battery_inverter_ac_rating_kw",
    "battery_inverter_kw_per_kwh",
    "battery_inverter_efficiency",
)


def _refuse_dc_battery_inverter_keys(cfg: Mapping[str, Any]) -> None:
    """A DC-coupled battery has no inverter of its own, so its keys are refused before their values are checked."""
    if cfg["battery_coupling"] != "dc":
        return
    given = [key for key in BATTERY_INVERTER_KEYS if cfg[key] is not None]
    costs = cfg.get("costs")
    if isinstance(costs, Mapping) and "battery_inverter_cost_per_kw" in costs:
        given.append("costs.battery_inverter_cost_per_kw")
    if given:
        raise ValueError(
            f"{', '.join(repr(key) for key in given)} describe an AC-coupled battery's own inverter; "
            "set battery_coupling = 'ac' or remove them"
        )


def _validate_battery_coupling(cfg: dict[str, Any]) -> None:
    """Validate the battery topology and an AC-coupled battery's inverter (ADR 0004 C2, C9).

    ``"dc"`` refuses the battery inverter keys and its price, so no setting
    is silently ignored; :func:`_refuse_dc_battery_inverter_keys` does that
    first, in :func:`validate_config`. ``"ac"`` with a battery needs one rating key and
    the efficiency; without a battery it takes them and needs neither, so a
    capacity sweep from 0 keeps one config. Its price is checked when the
    costs resolve, since a preset could carry it.
    """
    coupling = cfg["battery_coupling"]
    if cfg["battery_inverter_ac_rating_kw"] is not None and cfg["battery_inverter_kw_per_kwh"] is not None:
        raise ValueError(
            "'battery_inverter_ac_rating_kw' and 'battery_inverter_kw_per_kwh' both size the battery inverter; "
            "set one of them"
        )
    efficiency = cfg["battery_inverter_efficiency"]
    if coupling == "ac" and config_has_battery(cfg):
        if cfg["battery_inverter_ac_rating_kw"] is None and cfg["battery_inverter_kw_per_kwh"] is None:
            raise ValueError(
                "An AC-coupled battery needs its inverter's rating: set 'battery_inverter_ac_rating_kw' or "
                "'battery_inverter_kw_per_kwh'"
            )
        if efficiency is None:
            raise ValueError(
                "An AC-coupled battery needs 'battery_inverter_efficiency', from a source for the product; "
                "it has no default"
            )


def battery_inverter_ac_capacity_w(cfg: Mapping[str, Any]) -> float | None:
    """An AC-coupled battery's inverter rating in W, or None without one.

    ``battery_inverter_kw_per_kwh`` resolves against the nominal capacity, so
    the rating is fixed for the project (ADR 0004 C2).
    """
    if cfg["battery_coupling"] != "ac" or not config_has_battery(cfg):
        return None
    if cfg["battery_inverter_ac_rating_kw"] is not None:
        return float(cfg["battery_inverter_ac_rating_kw"]) * 1000
    return float(cfg["battery_inverter_kw_per_kwh"]) * float(cfg["battery_kwh"]) * 1000


def _is_int(value: Any) -> bool:
    """Return whether *value* is an integer, excluding booleans."""
    return isinstance(value, int) and not isinstance(value, bool)


# A finite number, returned as float.
_finite_real = number()


def resolve_location(cfg: dict[str, Any]) -> tuple[float, float, str, str | None]:
    """Resolve a location preset or custom coordinate dict."""
    loc = cfg["location"]
    if isinstance(loc, str):
        locations = load_config_json("locations.json")
        if loc not in locations:
            available = ", ".join(sorted(locations))
            raise ValueError(f"Unknown location '{loc}'. Available: {available}")
        loc_data = locations[loc]
        return loc_data["latitude"], loc_data["longitude"], loc_data["timezone"], loc
    return loc["latitude"], loc["longitude"], loc["timezone"], None


def resolve_orientation(cfg: dict[str, Any], lat: float) -> tuple[float, float, float]:
    """Return the top-level tilt, azimuth and tracker axis azimuth.

    An unset angle is derived from the latitude: the tilt by
    :func:`estimate_optimal_tilt`, both azimuths toward the equator. The PV
    arrays inherit the same values.
    """
    tilt = cfg["tilt"] if cfg["tilt"] is not None else estimate_optimal_tilt(lat)
    azimuth = cfg["azimuth"] if cfg["azimuth"] is not None else default_azimuth_fn(lat)
    axis_azimuth = cfg["axis_azimuth"] if cfg["axis_azimuth"] is not None else default_azimuth_fn(lat)
    return tilt, azimuth, axis_azimuth


def normalise_pv_arrays(
    arrays: list[dict[str, Any]] | None,
    cfg: dict[str, Any],
    *,
    tilt: float,
    azimuth: float,
    axis_azimuth: float,
) -> list[dict[str, Any]]:
    """Apply App-level PV defaults to each configured PV array.

    ``tilt``, ``azimuth`` and ``axis_azimuth`` are the top-level values from
    :func:`resolve_orientation`.
    """
    if not arrays:
        return []

    default_module = cfg["pv_module"] or default_module_key()

    # Tracker settings are inherited from the top level like module, tilt and
    # azimuth, before the PV model's own fallbacks for an array without them.
    tracker_defaults = {key: cfg[key] for key in ("tracking", *_TRACKER_GEOMETRY_KEYS)}
    tracker_defaults["axis_azimuth"] = axis_azimuth

    normalized: list[dict[str, Any]] = []
    for arr in arrays:
        entry = {
            "modules": int(arr["modules"]),
            "module": arr.get("module") or default_module,
            "tilt": float(arr.get("tilt", tilt)),
            "azimuth": float(arr.get("azimuth", azimuth)),
            "tracking": arr.get("tracking", tracker_defaults["tracking"]),
        }
        if entry["tracking"] != "fixed":
            for key in _TRACKER_GEOMETRY_KEYS:
                value = arr.get(key)
                entry[key] = tracker_defaults[key] if value is None else value
        for key in (*PER_ARRAY_MODEL_OPTION_KEYS, *_PV_ARRAY_MOUNTING_KEYS):
            if key in arr:
                entry[key] = arr[key]
        if "shading" in arr:
            entry["shading"] = _plain_shading(arr["shading"])
        normalized.append(entry)
    return normalized


def _plain_shading(value: Mapping[str, Any]) -> dict[str, Any]:
    """A validated shading table as plain lists of floats, as the result echoes it."""
    table = ShadingTable.from_mapping(value)
    plain: dict[str, Any] = {
        "azimuth_deg": list(table.azimuth_deg),
        "elevation_deg": list(table.elevation_deg),
        "beam_shaded_fraction": [list(row) for row in table.beam_shaded_fraction],
    }
    if table.sky_view_factor_stated:
        plain["sky_view_factor"] = table.sky_view_factor
    return plain


def resolve_pv_system(
    cfg: dict[str, Any], *, tilt: float, azimuth: float, axis_azimuth: float
) -> tuple[list[dict[str, Any]], str, PVModuleParams, int, float, float]:
    """Resolve PV module, array, and system sizing details.

    ``tilt``, ``azimuth`` and ``axis_azimuth`` are the top-level values from
    :func:`resolve_orientation`, which the arrays inherit.

    The module is returned twice: as its catalogue key and as its parameters.

    Returns the resolved module count rather than writing it back into ``cfg``;
    the caller materialises it so the dict wrapped by the frozen
    :class:`ResolvedAppConfig` is built once and not mutated in place.
    """
    pv_arrays = normalise_pv_arrays(cfg["pv_arrays"], cfg, tilt=tilt, azimuth=azimuth, axis_azimuth=axis_azimuth)
    if pv_arrays:
        n_modules = sum(arr["modules"] for arr in pv_arrays)
        total_power_w = sum(arr["modules"] * get_module(arr["module"]).Mpp for arr in pv_arrays)
        avg_module_power_w = total_power_w / n_modules
        system_kwp = total_power_w / 1000
        module_name = pv_arrays[0]["module"]
    else:
        n_modules = cfg["n_modules"]
        module_name = cfg["pv_module"]

    if module_name is None:
        module_name = default_module_key()
    pv_params = get_module(module_name)

    if not pv_arrays:
        avg_module_power_w = pv_params.Mpp
        system_kwp = n_modules * pv_params.Mpp / 1000
    return pv_arrays, module_name, pv_params, n_modules, avg_module_power_w, system_kwp


def validate_temperature_module_metadata(
    temperature_model: str,
    pv_arrays: list[dict[str, Any]],
    pv_params: PVModuleParams,
) -> None:
    """Validate any module metadata required by the selected thermal model.

    Array configurations may name different modules and thermal models, so SAM
    NOCT needs each array checked during App config resolution rather than
    failing after weather loading. The thermal kernel repeats this validation
    for direct solar calls.
    """
    if not pv_arrays:
        validate_temperature_inputs(
            normalise_model_name(temperature_model), pv_params.Module_Efficiency, pv_params.NOCT
        )
    for arr in pv_arrays:
        module = get_module(arr["module"])
        model = normalise_model_name(arr.get("temperature_model", temperature_model))
        validate_temperature_inputs(model, module.Module_Efficiency, module.NOCT)


def check_cost_currency(
    currency: str,
    *,
    given: Iterable[str],
    preset: str | None = None,
    tariff: bool,
    battery: bool | None,
    coupling: str = "dc",
    where: str = "[costs]",
) -> None:
    """Refuse cost inputs that are not in the run's ``currency``.

    BREOS does not convert, and never relabels an amount: a cost preset must
    be in the run's currency, and in a run whose currency is not EUR every
    cost the run prices with whose :class:`~breos.economics.CostParams`
    default is a EUR amount must be given, in ``given`` (the cost keys set
    explicitly) or by a preset in that currency. Defaults of zero are zero in
    any currency and stay. ``tariff``, ``battery`` and ``coupling`` select the
    keys the run prices with (:func:`~breos.economics.default_amount_cost_keys`).

    Raises:
        ValueError: If the preset's currency differs, or a EUR default would be read.
    """
    supplied = set(given)
    if preset:
        costs_db = load_config_json("costs.json")
        if preset not in costs_db:
            raise ValueError(f"Unknown cost preset '{preset}'. Available: {', '.join(sorted(costs_db))}")
        preset_currency = str(costs_db[preset]["currency"])
        if preset_currency != currency:
            raise ValueError(
                f"cost_preset '{preset}' is priced in {preset_currency}, but the run is in {currency}. BREOS does "
                f"not convert currencies: remove cost_preset and give the costs in {currency} under {where}, or "
                f"choose a cost preset in {currency}."
            )
        supplied.update(costs_db[preset])
    if currency == DEFAULT_CURRENCY:
        return
    missing = [
        key
        for key in default_amount_cost_keys(tariff=tariff, battery=battery, coupling=coupling)
        if key not in supplied
    ]
    if missing:
        raise ValueError(
            f"The run is in {currency}, but {', '.join(missing)} would come from the built-in "
            f"{DEFAULT_CURRENCY} defaults. BREOS does not convert currencies: set them in {currency} under {where}."
        )


def resolve_costs(cfg: dict[str, Any]) -> CostParams:
    """Build CostParams from packaged presets, overrides, and financial defaults.

    Preset keys override the :class:`CostParams` dataclass defaults; a key
    missing from a preset falls back to the same default used when no
    preset is configured, so the two paths cannot diverge.
    """
    params: dict[str, Any] = {}

    if cfg.get("cost_preset"):
        costs_db = load_config_json("costs.json")
        preset_key = cfg["cost_preset"]
        if preset_key not in costs_db:
            available = ", ".join(sorted(costs_db))
            raise ValueError(f"Unknown cost preset '{preset_key}'. Available: {available}")
        preset = costs_db[preset_key]
        for config_key, param_key in COST_CONFIG_KEY_TO_PARAM.items():
            if config_key in preset:
                params[param_key] = preset[config_key]

    # Explicit values are the final layer: user overrides > named preset >
    # CostParams defaults. Validation has already guaranteed this is a known,
    # finite, non-negative table.
    for config_key, value in cfg.get("costs", {}).items():
        params[COST_CONFIG_KEY_TO_PARAM[config_key]] = value

    if cfg["inverter_loading_ratio"] is not None:
        params["dc_ac_ratio"] = cfg["inverter_loading_ratio"]
    params["inflation_rate"] = cfg["inflation_rate"]
    params["sell_price_inflation"] = cfg["sell_price_inflation"]
    params["discount_rate"] = cfg["discount_rate"]
    params["pv_degradation_rate"] = cfg["pv_degradation_rate"]

    return CostParams(**params)


def resolve_emissions(cfg: dict[str, Any]) -> EmissionsParams | None:
    """Resolve optional emissions preset."""
    if not cfg["emissions_country"]:
        return None

    emissions_db = load_config_json("emissions.json")
    key = cfg["emissions_country"]
    if key not in emissions_db:
        available = ", ".join(sorted(emissions_db))
        raise ValueError(f"Unknown emissions country '{key}'. Available: {available}")
    params = dict(emissions_db[key])
    if cfg["export_emissions_factor_gco2_kwh"] is not None:
        params["export_displacement_carbon_intensity_gco2_kwh"] = cfg["export_emissions_factor_gco2_kwh"]
    return EmissionsParams(**params)


def build_costs_dict(cfg: dict[str, Any], resolved: ResolvedAppConfig) -> dict[str, float]:
    """Build the cost-analysis input dictionary for the resolved system.

    A replacement is priced at the storage cost per kWh of the configured
    capacity (ADR 0003 E4), the price the optimizer uses too, unless
    ``battery_replacement_cost`` states the price of one (ADR 0004 C9).
    """
    return calculate_costs(
        n_modules=cfg["n_modules"],
        module_power_w=resolved.avg_module_power_w,
        battery_capacity_wh=cfg["battery_kwh"] * 1000,
        cost_params=resolved.cost_params,
        # The rating that clips production is the one CAPEX prices (#181).
        inverter_ac_capacity_w=resolved.inverter_ac_capacity_w,
        replacement_cost_each=replacement_event_cost(
            cfg["battery_kwh"], resolved.cost_params.battery_cost_per_kwh, cfg.get("battery_replacement_cost")
        ),
        battery_coupling=cfg.get("battery_coupling", "dc"),
        battery_inverter_ac_capacity_w=resolved.battery_inverter_ac_capacity_w,
    )


def resolve_period(cfg: dict[str, Any], timezone: str) -> SimulationPeriod | None:
    """The simulated window of a validated config, or None for the whole calendar year."""
    period = cfg.get("period")
    if period is None:
        return None
    return SimulationPeriod(
        start=date.fromisoformat(period["start"]), end=date.fromisoformat(period["end"]), timezone=timezone
    )


def _normalise_config_values(cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply registry-owned value normalizers to config files and API input."""
    for key, field in APP_CONFIG_FIELDS.items():
        if field.normalizer is not None and key in cfg:
            cfg[key] = field.normalizer(cfg[key])
    return cfg


# Keys that set the same thing, so one config layer may set only one of them.
EXCLUSIVE_ALTERNATIVES: Mapping[str, str] = {
    key: field.exclusive_with for key, field in APP_CONFIG_FIELDS.items() if field.exclusive_with is not None
}


def override_config(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """``base`` without the alternatives of the keys ``overrides`` sets, so the override replaces them.

    Setting ``inverter_ac_rating_kw`` over a file that sets
    ``inverter_loading_ratio`` replaces the ratio, as any flag replaces the
    file's value. Both set in one layer still raise when the config resolves.
    """
    dropped = {EXCLUSIVE_ALTERNATIVES[key] for key in overrides if key in EXCLUSIVE_ALTERNATIVES}
    return {key: value for key, value in base.items() if key not in dropped or key in overrides}


def _check_one_inverter_sizing(config: Mapping[str, Any]) -> None:
    """``inverter_ac_rating_kw`` and ``inverter_loading_ratio`` size the same inverter; allow one."""
    if config.get("inverter_ac_rating_kw") is not None and config.get("inverter_loading_ratio") is not None:
        raise ValueError("'inverter_ac_rating_kw' and 'inverter_loading_ratio' both size the inverter; set one of them")


def resolve_app_config(config: dict[str, Any]) -> ResolvedAppConfig:
    """Merge, validate, and resolve App configuration."""
    module_catalogue = catalogue_snapshot()
    normalized = normalize_config_keys(config)
    _check_one_inverter_sizing(normalized)
    _check_electrical_sizing(normalized)
    cfg = merge_defaults(normalized)
    if cfg["inverter_ac_rating_kw"] is not None or cfg["inverters"] is not None:
        # An absolute rating replaces the ratio, so the ratio is reported unset.
        cfg["inverter_loading_ratio"] = None
    _normalise_config_values(cfg)
    specs = validate_config(cfg)
    if specs.vehicle is not None:
        raise ValueError("[vehicle] is validated, but vehicle simulation is not available yet")
    tariff = specs.tariff
    if cfg.get("pv_arrays"):
        # The result echoes the tables as plain floats, whatever numbers they came as.
        cfg["pv_arrays"] = [
            {**arr, "shading": _plain_shading(arr["shading"])} if "shading" in arr else arr for arr in cfg["pv_arrays"]
        ]

    lat, lon, timezone, loc_key = resolve_location(cfg)
    tilt, azimuth, axis_azimuth = resolve_orientation(cfg, lat)
    pv_arrays, pv_module_key, pv_params, n_modules, avg_module_power_w, system_kwp = resolve_pv_system(
        cfg, tilt=tilt, azimuth=azimuth, axis_azimuth=axis_azimuth
    )
    validate_temperature_module_metadata(cfg["temperature_model"], pv_arrays, pv_params)
    tracking = cfg["tracking"]

    # Materialise the resolved module count (derived from pv_arrays when set)
    # into a fresh dict rather than mutating the merged config in place.
    cfg = {**cfg, "n_modules": n_modules}
    cost_params = resolve_costs(cfg)
    battery_inverter_w = battery_inverter_ac_capacity_w(cfg)
    if battery_inverter_w is not None and cost_params.battery_inverter_cost_per_kw is None:
        raise ValueError(
            "An AC-coupled battery needs '[costs] battery_inverter_cost_per_kw', the price of its own inverter per "
            "kW; BREOS has no default for it. Set 0 if another price already includes the inverter"
        )
    electrical = _resolve_electrical(cfg, pv_arrays, pv_params, n_modules)
    if electrical is not None and any(string.module_shading is not None for string in electrical.strings):
        # The result echoes each module's tables as plain floats.
        cfg["pv_strings"] = [
            {**entry, "module_shading": string.module_shading.as_config()}
            if string.module_shading is not None
            else entry
            for entry, string in zip(cfg["pv_strings"], electrical.strings, strict=True)
        ]
    if electrical is not None and electrical.dc_wiring is not None:
        overrides = cfg["pv_loss_overrides"] or {}
        if "wiring" not in overrides:
            # The cables model the DC wiring. A stated wiring loss stays, as the
            # AC side's until AC cables are modelled.
            cfg["pv_loss_overrides"] = {**overrides, "wiring": 0.0}
    if any("shading" in arr for arr in pv_arrays) or _module_shaded_arrays(cfg):
        overrides = cfg["pv_loss_overrides"] or {}
        if "shading" not in overrides:
            # The tables model the near shading. A stated flat loss stays, on
            # top of them, for whatever they leave out.
            cfg["pv_loss_overrides"] = {**overrides, "shading": 0.0}
    if electrical is not None:
        cfg["inverter_ac_rating_kw"] = sum(inverter.nominal_power_w for inverter in electrical.inverters) / 1000
        rated = [(i.nominal_power_w, i.inverter_efficiency) for i in electrical.inverters]
        known = [(power, efficiency) for power, efficiency in rated if efficiency is not None]
        if len(known) == len(rated):
            # The nameplate-weighted mean, as the aggregate model has one inverter.
            cfg["inverter_efficiency"] = sum(p * e for p, e in known) / sum(p for p, _ in known)
        elif known:
            raise ValueError("Give every entry of 'inverters' an inverter_efficiency, or none")
    ac_capacity_w: float | None
    if cfg["inverter_ac_rating_kw"] is not None:
        ac_capacity_w = float(cfg["inverter_ac_rating_kw"]) * 1000
        # The ratio the rating implies, so CostParams does not keep a stale one.
        cost_params = replace(cost_params, dc_ac_ratio=n_modules * avg_module_power_w / ac_capacity_w)
    else:
        ac_capacity_w = inverter_ac_capacity_w(n_modules * avg_module_power_w, cfg["inverter_loading_ratio"])

    ac_cable = (
        None if cfg["ac_cable"] is None else _built(AcCable, AC_CABLE_TABLE.validate(cfg["ac_cable"]), "ac_cable")
    )
    if electrical is not None and any(inverter.ac_cable is not None for inverter in electrical.inverters):
        if len(electrical.inverters) == 1:
            # One inverter has one cable to the meter, given in either place.
            if ac_cable is not None:
                raise ValueError("Give the inverter's AC cable as '[ac_cable]' or 'inverters[0].ac_cable', not both")
            own = electrical.inverters[0].ac_cable
            assert own is not None
            ac_cable = _built(AcCable, dict(own), "inverters[0].ac_cable")
        elif electrical.fidelity != "string":
            raise ValueError(
                "An inverter's own 'ac_cable' with several inverters needs electrical_fidelity = 'string', which "
                "converts each inverter on its own"
            )
    return ResolvedAppConfig(
        cfg=cfg,
        lat=lat,
        lon=lon,
        timezone=timezone,
        loc_key=loc_key,
        pv_arrays=pv_arrays,
        pv_params=pv_params,
        pv_module_key=pv_module_key,
        avg_module_power_w=avg_module_power_w,
        system_kwp=system_kwp,
        tilt=tilt,
        azimuth=azimuth,
        tracking=tracking,
        axis_azimuth=axis_azimuth,
        inverter_ac_capacity_w=ac_capacity_w,
        tariff=tariff,
        smart_charging=specs.smart_charging,
        vehicle=specs.vehicle,
        cost_params=cost_params,
        emissions_params=resolve_emissions(cfg),
        period=resolve_period(cfg, timezone),
        reference_tariff=specs.reference_tariff,
        currency=specs.currency,
        electrical=electrical,
        ac_cable=ac_cable,
        inverter_conversion=_inverter_conversion(electrical, ac_cable, pv_arrays, pv_params),
        module_catalogue=module_catalogue,
        battery_inverter_ac_capacity_w=battery_inverter_w,
    )


def _inverter_conversion(
    electrical: ElectricalSetup | None,
    ac_cable: AcCable | None,
    pv_arrays: list[dict[str, Any]],
    pv_params: PVModuleParams,
) -> InverterConversion | None:
    """The efficiency curve, AC cable and night consumption the dispatch runs, or None for none of them.

    Aggregate energy reads the curve at its middle row, the nominal voltage of
    a three-row curve. String-level energy reads it at the strings' STC
    maximum power voltage, weighted by their STC power; that row stands for
    the steps without PV, and each step with PV reads its own operating
    voltage (:func:`breos.app_inputs.build_inverter_curve_row`). The night
    consumption is every inverter's that states one.

    String-level energy with several inverters converts each on its own
    (:func:`breos.app_inputs.build_inverter_units`), so there is no single
    conversion: None.
    """
    inverters = electrical.inverters if electrical is not None else ()
    if len(inverters) > 1 and electrical is not None and electrical.fidelity == "string":
        return None
    if len(inverters) > 1 and any(i.efficiency_curve or i.temperature_derating for i in inverters):
        raise ValueError(
            "An inverter efficiency_curve or temperature_derating with several inverters needs "
            f"electrical_fidelity = 'string'; {len(inverters)} are configured"
        )
    curve = inverters[0].efficiency_curve if inverters else None
    stated = [i.night_consumption_w for i in inverters if i.night_consumption_w is not None]
    night = sum(stated) if stated else None
    if curve is None and ac_cable is None and night is None:
        return None
    voltage = None
    if curve is not None and electrical is not None and electrical.fidelity == "string":
        modules = [get_module(arr["module"]) for arr in pv_arrays] or [pv_params]
        weighted = [
            (s.modules * s.parallel * modules[s.array].Mpp, s.modules * modules[s.array].Vmp)
            for s in electrical.strings
            if s.array < len(modules)
        ]
        power = sum(p for p, _ in weighted)
        voltage = sum(p * v for p, v in weighted) / power if power > 0 else None
    return InverterConversion.build(curve, voltage_v=voltage, ac_cable=ac_cable, night_consumption_w=night)


def _check_module_shading(
    cfg: Mapping[str, Any], strings: Sequence[PVString], arrays: Sequence[tuple[PVModuleParams, int]]
) -> tuple[PVString, ...]:
    """The strings with their module shading bound to their array's module; refuse shading string energy cannot run.

    Each string's tables take its module's bypass diodes
    (:meth:`~breos.pv.shading.ModuleShading.for_module`).
    """
    shaded = [string.module_shading is not None for string in strings]
    if not any(shaded):
        return tuple(strings)
    n_arrays = len(cfg["pv_arrays"] or [])
    bound = []
    for i, string in enumerate(strings):
        if string.module_shading is not None and string.array < len(arrays):
            module = arrays[string.array][0]
            try:
                shading = string.module_shading.for_module(module.bypass_diodes, module.cell_strings_in_parallel)
            except ValueError as error:
                raise ValueError(f"'pv_strings[{i}].module_shading': {error}") from None
            problem = shading.inconsistency(module.N_Cells)
            if problem:
                raise ValueError(f"'pv_strings[{i}].module_shading': {problem}")
            string = replace(string, module_shading=shading)
        bound.append(string)
    first = shaded.index(True)
    if cfg["electrical_fidelity"] != "string":
        raise ValueError(f"'pv_strings[{first}].module_shading' needs electrical_fidelity = 'string'")
    if not n_arrays:
        raise ValueError(f"'pv_strings[{first}].module_shading' needs 'pv_arrays'")
    for array in sorted({string.array for string in strings}):
        members = [i for i, string in enumerate(strings) if string.array == array]
        flags = [shaded[i] for i in members]
        if any(flags) and not all(flags):
            raise ValueError(
                f"'pv_strings[{members[flags.index(False)]}]' has no module_shading, but another string of "
                f"array {array} does. Give every string of an array module_shading, or none"
            )
    return tuple(bound)


# The DC/AC ratio range outside which the electrical check warns, per inverter.
DC_AC_RATIO_RANGE = (0.8, 1.6)


def _resolve_electrical(
    cfg: Mapping[str, Any], pv_arrays: list[dict[str, Any]], pv_params: PVModuleParams, n_modules: int
) -> ElectricalSetup | None:
    """Check the configured string topology against the resolved arrays, or None without one."""
    inputs = _electrical_inputs(cfg)
    if inputs is None:
        return None
    inverters, strings, environment = inputs
    arrays = [(get_module(arr["module"]), arr["modules"]) for arr in pv_arrays] or [(pv_params, n_modules)]
    mountings = [
        _built(ArrayMounting, {key: arr[key] for key in _PV_ARRAY_MOUNTING_KEYS if key in arr}, f"pv_arrays[{i}]")
        for i, arr in enumerate(pv_arrays)
    ] or None
    cabled = [string.has_cable for string in strings]
    if any(cabled) and not all(cabled):
        raise ValueError(
            "Give every 'pv_strings' entry cable_length_m and cable_cross_section_mm2, or none: the cables "
            "replace the flat 'wiring' loss for the whole system"
        )
    strings = _check_module_shading(cfg, strings, arrays)
    setup = ElectricalSetup(
        inverters,
        strings,
        environment,
        check_topology(
            arrays, inverters, strings, environment, dc_ac_ratio_range=DC_AC_RATIO_RANGE, mountings=mountings
        ),
        fidelity=cfg["electrical_fidelity"],
        dc_wiring=cable_summary(arrays, strings, environment),
    )
    if setup.fidelity == "string":
        problems = string_model_problems(setup)
        if len(inverters) > 1 and config_has_battery(cfg) and cfg["battery_coupling"] != "ac":
            problems.append(
                f"a battery beside {len(inverters)} inverters must be AC-coupled (battery_coupling = 'ac'); a "
                "DC-coupled battery sits behind one hybrid inverter"
            )
        smart_charging = cfg.get("smart_charging") or {}
        if len(inverters) > 1 and config_has_battery(cfg) and smart_charging.get("mode") in PLANNER_MODES:
            # The planner converts one forecast through one PV inverter (ADR 0004 M7).
            problems.append(
                f"smart_charging.mode = '{smart_charging['mode']}' plans with one PV inverter, not "
                f"{len(inverters)}; use mode = 'fixed_target' or 'discharge_only'"
            )
        if problems:
            raise ValueError("electrical_fidelity = 'string' cannot run: " + "; ".join(problems))
    return setup
