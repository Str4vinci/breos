"""Configuration and resource resolution for the public App facade."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from datetime import date, datetime
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

from breos.config_schema import TableSpec, anything, boolean, choice, list_of, mapping_of, number, text
from breos.constants import (
    DEFAULT_CHARGE_EFFICIENCY,
    DEFAULT_DISCHARGE_EFFICIENCY,
    DEFAULT_INDOOR_CEILING_C,
    DEFAULT_INDOOR_COUPLING_ALPHA,
    DEFAULT_INDOOR_FLOOR_C,
    DEFAULT_INDOOR_MODEL_ENABLED,
    DEFAULT_INDOOR_SETPOINT_C,
    DEFAULT_MAX_SOC,
    DEFAULT_MIN_SOC,
)
from breos.degradation.profiles import ENABLED_BLAST_MODEL_KEYS, apply_battery_profile_defaults
from breos.economics import (
    COST_CONFIG_KEY_TO_PARAM,
    DEFAULT_DISCOUNT_RATE,
    DEFAULT_INFLATION_RATE,
    CostParams,
    calculate_costs,
    replacement_event_cost,
)
from breos.emissions import EmissionsParams
from breos.execution import DEFAULT_EXECUTION_BACKEND, EXECUTION_BACKENDS, validate_execution_backend
from breos.inverter import inverter_ac_capacity_w
from breos.load_profiles import PROFILE_UNITS, resolve_profile_key, validate_profile_options
from breos.pv.horizon import normalise_horizon_profile
from breos.pv.model_options import is_known_model, is_valid_albedo, is_valid_gcr, normalise_model_name
from breos.pv.temperature import validate_temperature_inputs
from breos.pv_modules import MODULES, PVModuleParams, get_module
from breos.resources import load_config_json
from breos.smart_charging import SMART_CHARGING_MODES, SmartChargingSpec
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
    SUPPORTED_CURRENCIES,
    TariffPrices,
    TariffSpec,
    available_tariff_schedules,
    get_tariff_schedule,
    schedule_resolution_minutes,
)
from breos.utils import get_hours_per_step

_NO_DEFAULT = object()
_TRACKING_MODES = ("fixed", "single_axis", "dual_axis")


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
    Only the runner sections and the removed ``battery_type`` have none.
    """

    default: Any = _NO_DEFAULT
    cli_flags: tuple[str, ...] = ()
    cli_type: Callable[[str], Any] | None = None
    cli_choices: tuple[str, ...] | None = None
    cli_action: str | None = None
    cli_help: str | None = None
    normalizer: Callable[[Any], Any] | None = None
    doc: str = ""
    default_doc: str | None = None
    summary: str | None = None

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


CALENDAR_MODELS: tuple[str, ...] = (
    "naumann",
    "naumann_lam",
    "naumann_lam_field_calibrated",
    "naumann_lam_field_calibrated_v1",
    "naumann_lam_field_calibrated_v2",
)


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
        cli_flags=("--annual-consumption-kwh",),
        cli_type=float,
        cli_help="Annual electricity demand in kWh.",
        doc="Annual electricity demand (kWh)",
        default_doc="*required*",
        summary="load.annual_consumption_kwh",
    ),
    "battery_kwh": AppConfigField(
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
        default=None,
        cli_flags=("--battery-max-charge-power-w",),
        cli_type=float,
        cli_help="Maximum DC power entering the battery charge path in W (default: unlimited).",
        doc="Maximum DC power entering the battery charge path; `None` is unlimited",
        summary="battery.max_charge_power_w",
    ),
    "battery_max_discharge_power_w": AppConfigField(
        default=None,
        cli_flags=("--battery-max-discharge-power-w",),
        cli_type=float,
        cli_help="Maximum battery AC power delivered to load in W (default: unlimited).",
        doc="Maximum battery AC power delivered to load; `None` is unlimited",
        summary="battery.max_discharge_power_w",
    ),
    "battery_power_limit_c_rate": AppConfigField(
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
    "cost_preset": AppConfigField(
        default=None,
        cli_flags=("--cost-preset",),
        cli_help="Cost preset key, for example 'residential-pt'.",
        normalizer=_underscored,
        doc=(
            "Cost preset key from the packaged defaults; see [Packaged options](options.md#cost-presets). `None` uses "
            "the {py:class}`~breos.CostParams` defaults"
        ),
        summary="economics.cost_preset",
    ),
    "emissions_country": AppConfigField(
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
        default=None,
        cli_flags=("--pv-module",),
        cli_help="PV module catalogue key.",
        doc="Module key from the built-in catalogue. `None` uses the first available",
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
        default=None,
        cli_flags=("--tilt",),
        cli_type=float,
        cli_help="PV tilt angle in degrees.",
        doc="Tilt angle (degrees). `None` estimates it from the latitude",
        summary="pv.tilt",
    ),
    "azimuth": AppConfigField(
        default=None,
        cli_flags=("--azimuth",),
        cli_type=float,
        cli_help="PV surface azimuth in degrees.",
        doc="Surface azimuth (degrees). `None` faces the equator: 180 in the northern hemisphere",
        summary="pv.azimuth",
    ),
    "transposition_model": AppConfigField(
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
        default=None,
        cli_flags=("--surface-type",),
        cli_choices=tuple(SURFACE_TYPES),
        cli_help="Named ground cover mapped to an albedo (alternative to --albedo).",
        doc="Named ground cover mapped to an albedo; an alternative to `albedo`",
        summary="pv.surface_type",
    ),
    "model_perez": AppConfigField(
        default=DEFAULT_PEREZ_MODEL,
        cli_flags=("--perez-model",),
        cli_choices=tuple(PEREZ_MODELS),
        cli_help="Perez coefficient set (only used with --transposition-model perez).",
        doc='Perez coefficient set; only used when `transposition_model = "perez"`',
        summary="pv.model_perez",
    ),
    "solar_position": AppConfigField(
        default=DEFAULT_SOLAR_POSITION,
        cli_flags=("--solar-position",),
        cli_choices=tuple(SOLAR_POSITION_METHODS),
        cli_help=(
            "Where within each timestep the sun position is evaluated. 'mid-interval' matches "
            "PVWatts/SAM for interval-averaged weather (default: interval-start)."
        ),
        doc=(
            'Where within each timestep the sun position is evaluated. `"mid-interval"` adds half a timestep. '
            '`"weather"` reads the representative-time offset from content-bound weather metadata, including provider '
            "offsets for instantaneous irradiance and left- or right-labelled interval means"
        ),
        summary="pv.solar_position",
    ),
    "iam_model": AppConfigField(
        default=DEFAULT_IAM_MODEL,
        cli_flags=("--iam-model",),
        cli_choices=tuple(IAM_MODELS),
        cli_help="Beam incidence-angle modifier (default: ashrae, historical compatibility).",
        doc=(
            'Beam incidence-angle modifier. `"physical"` uses pvlib\'s physical optics model and `"martin_ruiz"` its '
            "empirical model; the Ashrae default preserves historical results"
        ),
        summary="pv.iam_model",
    ),
    "diffuse_iam": AppConfigField(
        default=DEFAULT_DIFFUSE_IAM,
        cli_flags=("--diffuse-iam",),
        cli_choices=tuple(DIFFUSE_IAM_METHODS),
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
        default=DEFAULT_TEMPERATURE_MODEL,
        cli_flags=("--temperature-model",),
        cli_choices=tuple(TEMPERATURE_MODELS),
        cli_help=(
            "Cell-temperature model / mounting preset. The pvsyst-* and sapm-* presets use documented "
            "mounting coefficients; noct-sam additionally requires sourced module NOCT and efficiency "
            "metadata (not yet bundled). Default: faiman, open rack."
        ),
        doc=(
            'Cell-temperature model and mounting preset. `"pvsyst-*"` and `"sapm-*"` use documented mounting '
            'coefficients; `"noct-sam"` needs sourced module NOCT and efficiency metadata, which no bundled module '
            "has yet. The default is Faiman, open rack"
        ),
        summary="pv.temperature_model",
    ),
    "bifacial_model": AppConfigField(
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
        default=None,
        cli_flags=("--pvrow-pitch",),
        cli_type=float,
        cli_help="Distance between PV rows; use the same unit as --pvrow-height.",
        doc='Distance between adjacent PV rows; required by `"infinite_sheds"`, in the same unit as `pvrow_height`',
        summary="pv.pvrow_pitch",
    ),
    "gcr": AppConfigField(
        default=0.35,
        cli_flags=("--gcr",),
        cli_type=float,
        cli_help="PV row ground coverage ratio (default: 0.35).",
        doc="Ground coverage ratio for single-axis tracking and infinite-sheds bifacial geometry",
        summary="pv.gcr",
    ),
    "resolution": AppConfigField(
        default="h",
        cli_flags=("--resolution",),
        cli_choices=("h", "15min"),
        cli_help="Simulation time resolution.",
        doc="Simulation time resolution",
        summary="load.resolution",
    ),
    "projection_years": AppConfigField(
        default=20,
        cli_flags=("--projection-years",),
        cli_type=int,
        cli_help="Economic projection horizon.",
        doc="Economic projection horizon in years",
        summary="economics.projection_years",
    ),
    "inflation_rate": AppConfigField(
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
        default=None,
        cli_flags=("--import-price-escalation",),
        cli_type=float,
        cli_help="Annual escalation of the import price and the fixed charge. Default: inflation_rate.",
        doc="Annual escalation of the import price and the fixed charge; `None` uses `inflation_rate`",
        summary="economics.import_price_escalation",
    ),
    "om_escalation": AppConfigField(
        default=None,
        cli_flags=("--om-escalation",),
        cli_type=float,
        cli_help="Annual escalation of O&M costs. Default: inflation_rate.",
        doc="Annual escalation of O&M costs; `None` uses `inflation_rate`",
        summary="economics.om_escalation",
    ),
    "replacement_cost_learning": AppConfigField(
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
        default=0.0,
        cli_flags=("--sell-price-inflation",),
        cli_type=float,
        cli_help="Annual inflation of the grid export (sell) price. Default 0.",
        doc="Annual escalation of the grid export (sell) price",
        summary="economics.sell_price_inflation",
    ),
    "export_emissions_factor_gco2_kwh": AppConfigField(
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
        default=DEFAULT_DISCOUNT_RATE,
        cli_flags=("--discount-rate",),
        cli_type=float,
        cli_help="Discount rate for NPV calculations.",
        doc="Nominal discount rate for NPV",
        summary="economics.discount_rate",
    ),
    "pv_degradation_rate": AppConfigField(
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
        default="native",
        cli_flags=("--degradation-engine",),
        cli_choices=("native", "blast"),
        cli_help="Battery degradation engine (default: native Naumann/Lam).",
        doc='`"native"` keeps Naumann/Lam; `"blast"` opts into a vendored BLAST cell model',
        summary="battery.degradation_engine",
    ),
    "blast_model": AppConfigField(
        default=None,
        cli_flags=("--blast-model",),
        cli_help="Stable BLAST battery-model key; requires --degradation-engine blast.",
        doc='Stable BLAST model key; required with `degradation_engine = "blast"` and invalid with the native engine',
        summary="battery.blast_model",
    ),
    "dc_coupled": AppConfigField(
        default=True,
        cli_flags=("--dc-coupled",),
        cli_action="store_true",
        cli_help="Use the supported DC-coupled/hybrid battery model.",
        doc="DC-coupled or hybrid inverter. `False` is not supported and raises",
        summary="inverter.dc_coupled",
    ),
    "inverter_efficiency": AppConfigField(
        default=0.96,
        cli_flags=("--inverter-efficiency",),
        cli_type=float,
        cli_help="Inverter efficiency.",
        doc="Nominal inverter efficiency used by the PVWatts part-load curve",
        summary="inverter.efficiency",
    ),
    "inverter_loading_ratio": AppConfigField(
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
    "start_date": AppConfigField(
        default="2023-01-01",
        cli_flags=("--start-date",),
        cli_help="First simulated day: 1 January of the study year, YYYY-01-01.",
        doc="1 January of the study year, `YYYY-01-01`. The App simulates that year, or the `period` window in it",
        summary="load.start_date",
    ),
    # The [period] table (#242). Omitted: the whole calendar year of start_date.
    "period": AppConfigField(
        doc=(
            "Simulate only the window from `start` to `end`, local dates in the year of `start_date`, `end` "
            "exclusive. The window runs once and reports energy only: lifetime economics are `None`. See "
            "[`[period]`](#period) and [Simulate part of a year](recipes.md#simulate-part-of-a-year)"
        ),
        default_doc="*unset*",
        summary="simulation.period",
    ),
    # The [tariff] table (ADR 0002). Omitted: flat prices from the cost preset.
    "tariff": AppConfigField(
        default=None,
        doc=(
            "Time-of-use import and export prices on a bundled schedule, replacing the flat `costs.electricity_cost`, "
            "`costs.electricity_sold_cost` and `costs.daily_power_cost`; see [`[tariff]`](#tariff) and [Time-of-use "
            "tariffs](configuration.md#time-of-use-tariffs)"
        ),
        default_doc="*unset*",
        summary="economics.tariff",
    ),
    # The [smart_charging] table (ADR 0002). Omitted: greedy self-consumption.
    "smart_charging": AppConfigField(
        default=None,
        doc=(
            "Grid charging toward a target in the tariff's cheap periods; see [`[smart_charging]`](#smart_charging) "
            "and [Smart charging](configuration.md#smart-charging)"
        ),
        default_doc="*unset*",
        summary="battery.smart_charging",
    ),
    "weather_source": AppConfigField(
        default=None,
        cli_flags=("--weather-source",),
        cli_help=(
            "Source part of the cached weather/<location>_tmy_*_<source>.csv file to use when several "
            "TMY files exist for the location preset, for example 'pvgis-sarah3'."
        ),
        doc=(
            "Source part of the cached `weather/<location>_tmy_<years>_<source>.csv` file to load, e.g. "
            '`"pvgis-sarah3"`. Needed only when several TMY files exist for a location preset; see [Offline runs with '
            "cached weather](recipes.md#offline-runs-with-cached-weather)"
        ),
        summary="simulation.weather_source",
    ),
    # Config-file/API-only fields.
    "costs": AppConfigField(
        doc="Cost overrides layered over the selected preset and the built-in defaults; see [`[costs]`](#costs)",
        default_doc="*unset*",
        summary="economics.costs",
    ),
    "pv_arrays": AppConfigField(
        default=None,
        doc=(
            "List of arrays, each with `modules`, `module`, `tilt` and `azimuth`. The array module total replaces "
            "`n_modules`; see [`[[pv_arrays]]`](#pv_arrays)"
        ),
        summary="pv.arrays",
    ),
    "tracking": AppConfigField(
        default="fixed",
        doc="Tracking mode: " + ", ".join(f'`"{mode}"`' for mode in _TRACKING_MODES),
        summary="pv.tracking",
    ),
    "axis_tilt": AppConfigField(default=0.0, doc="Single-axis tracker axis tilt (degrees)", summary="pv.axis_tilt"),
    "axis_azimuth": AppConfigField(
        default=None,
        doc="Tracker axis azimuth (degrees). `None` sets it from the latitude",
        summary="pv.axis_azimuth",
    ),
    "max_angle": AppConfigField(
        default=60.0,
        doc="Single-axis tracker maximum rotation angle (degrees)",
        summary="pv.max_angle",
    ),
    "backtrack": AppConfigField(
        default=True,
        doc="Whether single-axis trackers backtrack to avoid row shading",
        summary="pv.backtrack",
    ),
    "cross_axis_tilt": AppConfigField(
        default=0.0,
        doc="Cross-axis terrain slope for single-axis tracking (degrees)",
        summary="pv.cross_axis_tilt",
    ),
    "dual_axis_max_tilt": AppConfigField(
        default=90.0,
        doc="Maximum panel tilt for dual-axis tracking (degrees)",
        summary="pv.dual_axis_max_tilt",
    ),
    # The BatteryConfig defaults, so the App and the optimizer, which leaves
    # unset battery settings to BatteryConfig, resolve the same window.
    "battery_min_soc": AppConfigField(
        default=DEFAULT_MIN_SOC,
        doc="Battery SOC floor, as a fraction of nominal SOH-derated capacity",
        summary="battery.min_soc",
    ),
    "battery_max_soc": AppConfigField(
        default=DEFAULT_MAX_SOC,
        doc="Battery SOC ceiling, on the same basis as `battery_min_soc`",
        summary="battery.max_soc",
    ),
    "battery_eol_percentage": AppConfigField(
        default=0.70,
        doc="SOH fraction that triggers a battery replacement",
        summary="battery.eol_percentage",
    ),
    "battery_rte": AppConfigField(
        default=None,
        doc=(
            "Battery round-trip efficiency, split evenly across charge and discharge; `None` is "
            f"{DEFAULT_CHARGE_EFFICIENCY * DEFAULT_DISCHARGE_EFFICIENCY:.2f}"
        ),
        summary="battery.round_trip_efficiency",
    ),
    "enable_resistance_fade": AppConfigField(
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
        doc="Parameter grid, read by `breos sweep`; see [Parameter sweep](recipes.md#parameter-sweep)",
        default_doc="*unset*",
    ),
    # Kept solely to preserve the existing actionable legacy-selector error.
    "battery_type": AppConfigField(
        doc=(
            'Removed legacy selector. Setting it raises an error: use `degradation_engine = "native"`, or '
            '`degradation_engine = "blast"` with `blast_model`'
        ),
        default_doc="*removed*",
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
        "storage_cost_per_kwh": "Battery price per kWh of nominal capacity; also the t = 0 price of each replacement",
        "inverter_cost_per_kw_hybrid": "Hybrid inverter price per kW of AC rating, for a system with a battery",
        "inverter_cost_per_kw_simple": "Inverter price per kW of AC rating, for a PV-only system",
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


# Keep runner-table keys explicit until the shared configuration schema from
# #181 can describe these sections alongside App fields.
MONTECARLO_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "weather_file",
        "n_runs",
        "years_per_run",
        "load_uncertainty",
        "load_distribution",
        "target_year",
        "weather_start_year",
        "weather_end_year",
        "seed",
        "min_load_scale",
        "max_load_scale",
        "preserve_irradiance_energy",
        "collect_yearly",
        "n_procs",
        "execution_backend",
    }
)


def validate_montecarlo_config(cfg: dict[str, Any]) -> None:
    """Reject unknown Monte Carlo settings before runner setup or weather access."""
    if "montecarlo" not in cfg:
        return
    montecarlo = cfg["montecarlo"]
    if not isinstance(montecarlo, dict):
        raise TypeError("'montecarlo' must be a table/dict of Monte Carlo settings")
    unknown_mc = set(montecarlo) - MONTECARLO_CONFIG_KEYS
    if unknown_mc:
        available = ", ".join(sorted(f"montecarlo.{key}" for key in MONTECARLO_CONFIG_KEYS))
        unknown_text = ", ".join(sorted(f"montecarlo.{key}" for key in unknown_mc))
        raise ValueError(f"Unknown Monte Carlo config key(s): {unknown_text}. Available: {available}")


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
    # The configured [period], or None for the whole calendar year of start_date.
    period: SimulationPeriod | None = None


def load_json(name: str) -> dict[str, Any]:
    """Load a packaged App configuration resource."""
    return load_config_json(name)


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
    """Apply user values over BLAST profile defaults over global defaults."""
    return apply_battery_profile_defaults(DEFAULTS, config)


def default_module_key() -> str:
    """Return the catalog key used when a config names no PV module.

    ``DEFAULTS["pv_module"]`` is ``None`` rather than a key, so the real
    default is the catalog's insertion order. Resolving it here keeps the
    validation, resolution, and results layers from each re-deriving it and
    silently disagreeing if the catalog is reordered.
    """
    return next(iter(MODULES))


def _validate_sky_settings(
    transposition_model: Any,
    albedo: Any,
    surface_type: Any,
    model_perez: Any,
    where: str = "",
) -> None:
    """Validate the sky-diffusion settings shared by the top level and arrays.

    ``where`` prefixes the key name in error messages (e.g. ``pv_arrays[0]``);
    ``None`` values are treated as "not set" and skipped, so per-array overrides
    only validate the keys they actually provide.

    The rules come from :mod:`breos.pv.model_options`; only the config-key
    phrasing and the None-means-unset handling are this layer's own.
    """
    prefix = f"{where}." if where else ""
    if transposition_model is not None and not is_known_model(transposition_model, TRANSPOSITION_MODELS):
        valid = ", ".join(TRANSPOSITION_MODELS)
        raise ValueError(f"'{prefix}transposition_model' must be one of: {valid}")
    if albedo is not None and surface_type is not None:
        raise ValueError(f"Set either '{prefix}albedo' or '{prefix}surface_type', not both.")
    if albedo is not None and (not isinstance(albedo, (int, float)) or not is_valid_albedo(albedo)):
        raise ValueError(f"'{prefix}albedo' must be a number between 0 and 1")
    if surface_type is not None and surface_type not in SURFACE_TYPES:
        valid = ", ".join(SURFACE_TYPES)
        raise ValueError(f"'{prefix}surface_type' must be one of: {valid}")
    if model_perez is not None and model_perez not in PEREZ_MODELS:
        valid = ", ".join(PEREZ_MODELS)
        raise ValueError(f"'{prefix}model_perez' must be one of: {valid}")


# Tracker geometry, inherited by every tracking array that does not set it.
_TRACKER_GEOMETRY_KEYS = (
    "axis_tilt",
    "axis_azimuth",
    "max_angle",
    "backtrack",
    "cross_axis_tilt",
    "dual_axis_max_tilt",
)
_TRACKER_ANGLE_RANGES = {
    "axis_tilt": (0.0, 90.0),
    "axis_azimuth": (0.0, 360.0),
    "max_angle": (0.0, 90.0),
    "cross_axis_tilt": (-90.0, 90.0),
    "dual_axis_max_tilt": (0.0, 90.0),
}
# Per-array settings passed through to the PV model when an array sets them.
_PV_ARRAY_OPTION_KEYS = (
    "gcr",
    "transposition_model",
    "albedo",
    "surface_type",
    "model_perez",
    "bifacial_model",
    "pvrow_height",
    "pvrow_pitch",
)
_PV_ARRAY_KEYS = frozenset(
    ("modules", "module", "tilt", "azimuth", "tracking", *_TRACKER_GEOMETRY_KEYS, *_PV_ARRAY_OPTION_KEYS)
)
PV_ARRAY_TABLE = TableSpec(
    "pv_arrays[i]",
    keys=dict.fromkeys(_PV_ARRAY_KEYS, anything),
    docs={
        "modules": "Number of PV modules in this array, at least 1",
        "module": "Module catalogue key for this array; inherits `pv_module`",
        # Every other key is its top-level namesake, for this array only.
        **{
            key: f"`{key}` for this array; inherits the top-level value"
            for key in ("tilt", "azimuth", "tracking", *_TRACKER_GEOMETRY_KEYS, *_PV_ARRAY_OPTION_KEYS)
        },
    },
)


def _validate_tracker_settings(settings: dict[str, Any], where: str = "") -> None:
    """Validate the tracker keys shared by the top level and arrays.

    ``None`` means "not set" and is skipped, so an array only validates the
    keys it provides; the rest are inherited from the already-checked top
    level. ``backtrack`` must be a real bool, because a string such as
    ``"no"`` is truthy and would leave backtracking on.
    """
    prefix = f"{where}." if where else ""
    tracking = settings.get("tracking")
    if tracking is not None and tracking not in _TRACKING_MODES:
        raise ValueError(f"'{prefix}tracking' must be 'fixed', 'single_axis', or 'dual_axis', got {tracking!r}")
    for key, (low, high) in _TRACKER_ANGLE_RANGES.items():
        value = settings.get(key)
        if value is not None and not low <= _finite_real(value, f"{prefix}{key}") <= high:
            raise ValueError(f"'{prefix}{key}' must be between {low:g} and {high:g}")
    backtrack = settings.get("backtrack")
    if backtrack is not None and not isinstance(backtrack, bool):
        raise TypeError(f"'{prefix}backtrack' must be true or false, got {backtrack!r}")


def _validate_gcr(gcr: Any, prefix: str = "") -> None:
    """Validate a ground coverage ratio, whichever model consumes it.

    ``gcr`` has two consumers and neither is a shading calculation: the
    ``infinite_sheds`` rear-side view factors, and — on the tracking path —
    the backtracking rotation schedule that pvlib's ``singleaxis`` derives
    from it. The second is why this is checked even with no rear-side model
    active. pvlib does not reject a nonsensical ratio; it quietly computes a
    different rotation, so a mistyped ``3.5`` returns roughly half the annual
    energy with no error anywhere. ``prefix`` already carries its trailing
    dot, matching the other config-key messages.
    """
    if not is_valid_gcr(_finite_real(gcr, f"{prefix}gcr")):
        raise ValueError(f"'{prefix}gcr' must be between 0 (exclusive) and 1 (inclusive)")


def _validate_bifacial_settings(
    model: Any,
    module: Any,
    gcr: Any,
    pvrow_height: Any,
    pvrow_pitch: Any,
    where: str = "",
) -> None:
    """Validate opt-in bifacial module metadata and row geometry.

    Shares its predicates with :mod:`breos.pv.model_options` but reports them
    against config keys. ``pvrow_*`` geometry is required only for an active
    rear-side model; ``gcr`` is also checked independently after the existing
    validators because tracking can consume it without bifacial modeling.
    """
    prefix = f"{where}." if where else ""
    normalised = normalise_model_name(model)
    if not is_known_model(model, BIFACIAL_MODELS):
        valid = ", ".join(BIFACIAL_MODELS)
        raise ValueError(f"'{prefix}bifacial_model' must be one of: {valid}")

    for key, value in (("pvrow_height", pvrow_height), ("pvrow_pitch", pvrow_pitch)):
        if value is not None and _finite_real(value, f"{prefix}{key}") <= 0:
            raise ValueError(f"'{prefix}{key}' must be > 0 when configured")

    if normalised == "none":
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
    # Deliberately last, so an active rear-side model still reports its
    # missing metadata and geometry before quibbling about the ratio.
    _validate_gcr(gcr, prefix)


def validate_config(cfg: dict[str, Any]) -> None:
    """Validate user-facing App config before resolving derived values."""
    has_arrays = _validate_structure_and_location(cfg)
    _validate_pv_and_inverter(cfg, has_arrays)
    _validate_time_and_weather(cfg)
    _validate_economics(cfg)
    _validate_tariff(cfg)
    _validate_battery_and_degradation(cfg)
    _validate_period(cfg)
    _validate_smart_charging(cfg)
    _validate_reachable_gcr(cfg, has_arrays)


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


def _check_tariff_prices(table: dict[str, Any], where: str) -> None:
    periods = set(get_tariff_schedule(table["schedule"]).periods)
    for name in ("import_prices", "export_prices"):
        given = set(table[name])
        unknown = sorted(given - periods - {"all"})
        if unknown:
            raise ValueError(
                f"'{where}.{name}' has period(s) {', '.join(unknown)} that schedule {table['schedule']!r} "
                f"does not have. Its periods: {', '.join(sorted(periods))}; 'all' prices every period."
            )
        missing = sorted(periods - given) if "all" not in given else []
        if missing:
            raise ValueError(
                f"'{where}.{name}' has no price for {', '.join(missing)}. Price every period of "
                f"{table['schedule']!r}, or give 'all'."
            )


TARIFF_TABLE = TableSpec(
    "tariff",
    keys={
        "schedule": choice(available_tariff_schedules()),
        "currency": choice(tuple(sorted(SUPPORTED_CURRENCIES))),
        "import_prices": mapping_of(text, number(minimum=0)),
        "export_prices": mapping_of(text, number(minimum=0)),
        "fixed_charge_per_day": number(minimum=0),
        "boundary_policy": choice(tuple(sorted(BOUNDARY_POLICIES))),
        "study_date": _tariff_study_date,
    },
    required=frozenset({"schedule", "currency", "import_prices", "export_prices"}),
    check=_check_tariff_prices,
    docs={
        "schedule": (
            "Bundled schedule key, which fixes the periods in local civil time; see "
            "[Bundled schedules](../api/tariffs.md#bundled-schedules)"
        ),
        "currency": (
            f"Currency of the prices: {', '.join(sorted(SUPPORTED_CURRENCIES))}. The cost preset should be in the "
            "same currency; BREOS does not convert"
        ),
        "import_prices": "Import price per kWh by period name, at year-1 prices; `all` prices every period",
        "export_prices": "Export price per kWh by period name, at year-1 prices; `all` prices every period",
        "fixed_charge_per_day": "Fixed charge per day, at year-1 prices (default 0)",
        "boundary_policy": (
            "How a period boundary inside a step is handled. `strict`, the default, refuses it. One of "
            + ", ".join(f"`{policy}`" for policy in sorted(BOUNDARY_POLICIES))
        ),
        "study_date": "A date in the schedule's effective window, needed when the simulated year is outside it",
    },
)
# Flat-price cost keys a tariff replaces; setting both would price energy twice.
_TARIFF_REPLACES_COSTS = ("electricity_cost", "electricity_sold_cost", "daily_power_cost")


def _validate_tariff(cfg: dict[str, Any]) -> None:
    if cfg["tariff"] is None:
        return
    table = TARIFF_TABLE.validate(cfg["tariff"])
    clashing = sorted(key for key in _TARIFF_REPLACES_COSTS if key in (cfg.get("costs") or {}))
    if clashing:
        raise ValueError(
            f"A [tariff] sets the energy prices and the fixed charge, so {', '.join(f'costs.{k}' for k in clashing)} "
            "would price them twice. Remove them, or remove [tariff]."
        )
    step_minutes = int(get_hours_per_step(cfg["resolution"]) * 60)
    required = schedule_resolution_minutes(table["schedule"])
    if required % step_minutes:
        raise ValueError(
            f"Schedule {table['schedule']!r} has boundaries every {required} minutes, which "
            f'{cfg["resolution"]!r} steps cannot represent; use resolution = "15min" (ADR 0002 A3).'
        )


def resolve_tariff_spec(cfg: dict[str, Any], timezone: str) -> TariffSpec | None:
    """Validate and build a tariff for App or an adapted optimizer config.

    ``cfg`` supplies ``tariff``, ``resolution`` and optional ``costs``. Keeping
    the price-conflict and resolution checks here gives both entry points
    the same validation before they run the PV model.
    """
    if cfg["tariff"] is None:
        return None
    _validate_tariff(cfg)
    table = TARIFF_TABLE.validate(cfg["tariff"])
    schedule = get_tariff_schedule(table["schedule"])
    if schedule.timezone != timezone:
        raise ValueError(
            f"Schedule {schedule.identifier!r} is defined in {schedule.timezone} civil time, but the "
            f"location's timezone is {timezone}. BREOS does not move a schedule to another zone."
        )
    prices = TariffPrices(
        currency=table["currency"],
        import_prices=table["import_prices"],
        export_prices=table["export_prices"],
        fixed_charge_per_day=table.get("fixed_charge_per_day", 0.0),
        identifier="config",
        version="1",
    )
    return TariffSpec(
        schedule=schedule.identifier,
        prices=prices,
        boundary_policy=table.get("boundary_policy", "strict"),
        study_date=table.get("study_date"),
    )


# Keys fixed-target mode must set. grid_charge_efficiency has no default: the
# inverter model has no AC-to-DC path to derive one from (ADR 0002 A6).
_FIXED_TARGET_REQUIRED = ("target_usable_fraction", "charge_periods", "discharge_periods", "grid_charge_efficiency")


def _check_smart_charging_keys(table: dict[str, Any], where: str) -> None:
    if table["mode"] == "disabled":
        extra = sorted(key for key in table if key != "mode")
        if extra:
            raise ValueError(
                f"'{where}.mode' = 'disabled' takes no other keys; remove {', '.join(f'{where}.{k}' for k in extra)}"
            )
        return
    missing = [key for key in _FIXED_TARGET_REQUIRED if key not in table]
    if missing:
        raise ValueError(f"'{where}' needs {', '.join(f'{where}.{k}' for k in missing)} for mode = 'fixed_target'")
    overlap = sorted(set(table["charge_periods"]) & set(table["discharge_periods"]))
    if overlap:
        raise ValueError(
            f"'{where}.charge_periods' and '{where}.discharge_periods' share {', '.join(overlap)}; "
            "every step either charges or discharges (ADR 0002 A8)"
        )


SMART_CHARGING_TABLE = TableSpec(
    "smart_charging",
    keys={
        "mode": choice(SMART_CHARGING_MODES),
        "target_usable_fraction": number(minimum=0, maximum=1),
        "charge_periods": list_of(text, min_length=1),
        "discharge_periods": list_of(text, min_length=1),
        "grid_charge_efficiency": number(minimum=0, maximum=1, min_exclusive=True),
        # None, as well as omitting the key, leaves site import unlimited.
        "grid_import_limit_w": number(minimum=0, min_exclusive=True, allow_none=True),
    },
    required=frozenset({"mode"}),
    check=_check_smart_charging_keys,
    docs={
        "mode": "`fixed_target` charges from the grid toward a target; `disabled` is greedy self-consumption",
        "target_usable_fraction": (
            "Grid-charging target as a fraction of the usable window: 0 is `battery_min_soc`, 1 is `battery_max_soc`"
        ),
        "charge_periods": "Tariff periods in which the grid may charge the battery",
        "discharge_periods": "Tariff periods in which the battery may discharge to the load; not a charge period",
        "grid_charge_efficiency": (
            "AC-to-DC conversion efficiency of the grid-charging path, before the battery's own charge efficiency. "
            "No default"
        ),
        "grid_import_limit_w": (
            "Site import limit in W for grid charging, which may import up to the limit minus the load's import. "
            "Load import is never cut. Unset is unlimited"
        ),
    },
)

# The nested tables of an App config, by top-level key. ``pv_arrays`` is a
# list of tables and is not here: a sweep or merge cannot address one entry.
NESTED_TABLE_SPECS: Mapping[str, TableSpec] = {
    "costs": COSTS_TABLE,
    "battery_indoor_model": INDOOR_MODEL_TABLE,
    "tariff": TARIFF_TABLE,
    "smart_charging": SMART_CHARGING_TABLE,
    "period": PERIOD_TABLE,
}


def _checked_smart_charging(
    value: Any, schedule: str | None, battery_kwh: float, battery_key: str = "battery_kwh"
) -> dict[str, Any]:
    """Check a [smart_charging] table against the configured tariff schedule and battery.

    ``battery_key`` names the setting ``battery_kwh`` came from, for the error.
    """
    table = SMART_CHARGING_TABLE.validate(value)
    if table["mode"] == "disabled":
        return table
    if schedule is None:
        raise ValueError(
            "'smart_charging.mode' = 'fixed_target' needs a [tariff]: its charge and discharge periods are "
            "tariff periods"
        )
    if not battery_kwh > 0:
        raise ValueError(f"'smart_charging.mode' = 'fixed_target' needs a battery; set {battery_key} > 0")
    periods = get_tariff_schedule(schedule).periods
    for name in ("charge_periods", "discharge_periods"):
        unknown = sorted(set(table[name]) - set(periods))
        if unknown:
            raise ValueError(
                f"'smart_charging.{name}' has period(s) {', '.join(unknown)} that schedule {schedule!r} "
                f"does not have. Its periods: {', '.join(sorted(periods))}."
            )
    return table


def _validate_smart_charging(cfg: dict[str, Any]) -> None:
    if cfg["smart_charging"] is None:
        return
    schedule = TARIFF_TABLE.validate(cfg["tariff"])["schedule"] if cfg["tariff"] is not None else None
    _checked_smart_charging(cfg["smart_charging"], schedule, cfg["battery_kwh"])


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
    schedule = tariff_spec.schedule if tariff_spec is not None else None
    table = _checked_smart_charging(cfg["smart_charging"], schedule, cfg["battery_kwh"], battery_key)
    if table["mode"] == "disabled":
        return SmartChargingSpec(mode="disabled")
    return SmartChargingSpec(
        mode=table["mode"],
        target_usable_fraction=table["target_usable_fraction"],
        # A period named twice is still one period.
        charge_periods=tuple(dict.fromkeys(table["charge_periods"])),
        discharge_periods=tuple(dict.fromkeys(table["discharge_periods"])),
        grid_charge_efficiency=table["grid_charge_efficiency"],
        grid_import_limit_w=table.get("grid_import_limit_w"),
    )


def _validate_reachable_gcr(cfg: dict[str, Any], has_arrays: bool) -> None:
    """Check every ``gcr`` that can reach the model, once everything else passes.

    Runs last, and deliberately so: an out-of-range ``gcr`` used to be caught
    only under an active bifacial model, so checking it earlier would change
    which error an already-broken config reports. Running it here makes the
    check purely additive — a config failing on some other key keeps failing
    on that key, and ``gcr`` is only ever the *new* reason a config is
    rejected.

    Arrays that set no ``gcr`` inherit the top-level value, which is also the
    function-level default handed to the multi-array entry point, so the
    top-level value is checked even when every array overrides it. Bifacial
    arrays have already had their effective ``gcr`` checked in the loop above;
    re-checking an explicit override here is harmless and covers the
    non-bifacial tracking path that nothing else reaches.
    """
    _validate_gcr(cfg["gcr"])
    if has_arrays:
        for index, array in enumerate(cfg["pv_arrays"]):
            if "gcr" in array:
                _validate_gcr(array["gcr"], f"pv_arrays[{index}].")


def _validate_structure_and_location(cfg: dict[str, Any]) -> bool:
    """Validate top-level keys, required inputs, and location structure."""
    if "battery_type" in cfg:
        raise ValueError(
            "'battery_type' is an ambiguous legacy selector and is not supported by App. "
            "Use degradation_engine='native' (default), or set degradation_engine='blast' with blast_model='<key>'."
        )
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
        lat = _finite_real(loc["latitude"], "location.latitude")
        lon = _finite_real(loc["longitude"], "location.longitude")
        if not -90 <= lat <= 90:
            raise ValueError("'location.latitude' must be between -90 and 90")
        if not -180 <= lon <= 180:
            raise ValueError("'location.longitude' must be between -180 and 180")
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
    if not has_arrays and (not _is_int(cfg["n_modules"]) or cfg["n_modules"] < 1):
        raise ValueError("'n_modules' must be >= 1")
    _validate_tracker_settings(cfg)
    if has_arrays:
        if not isinstance(cfg["pv_arrays"], list):
            raise TypeError("'pv_arrays' must be a list")
        for i, arr in enumerate(cfg["pv_arrays"]):
            # Only the key set is checked here; the values are checked below
            # with the top-level rules they share.
            PV_ARRAY_TABLE.validate(arr, f"pv_arrays[{i}]")
            _validate_tracker_settings(arr, where=f"pv_arrays[{i}]")
            modules = arr.get("modules", 0)
            if not _is_int(modules) or modules < 1:
                raise ValueError(f"'pv_arrays[{i}].modules' must be >= 1")
            module = arr.get("module", cfg.get("pv_module"))
            if module is not None and module not in MODULES:
                available = ", ".join(sorted(MODULES))
                raise ValueError(f"Unknown PV module {module!r} in pv_arrays[{i}]. Available: {available}")
            tilt = arr.get("tilt", cfg.get("tilt"))
            azimuth = arr.get("azimuth", cfg.get("azimuth"))
            if tilt is not None and not 0 <= _finite_real(tilt, f"pv_arrays[{i}].tilt") <= 90:
                raise ValueError(f"'pv_arrays[{i}].tilt' must be between 0 and 90")
            if azimuth is not None and not 0 <= _finite_real(azimuth, f"pv_arrays[{i}].azimuth") <= 360:
                raise ValueError(f"'pv_arrays[{i}].azimuth' must be between 0 and 360")
            _validate_sky_settings(
                arr.get("transposition_model"),
                arr.get("albedo"),
                arr.get("surface_type"),
                arr.get("model_perez"),
                where=f"pv_arrays[{i}]",
            )
            _validate_bifacial_settings(
                arr.get("bifacial_model", cfg["bifacial_model"]),
                module or default_module_key(),
                arr.get("gcr", cfg["gcr"]),
                arr.get("pvrow_height", cfg["pvrow_height"]),
                arr.get("pvrow_pitch", cfg["pvrow_pitch"]),
                where=f"pv_arrays[{i}]",
            )
    if _finite_real(cfg["annual_consumption_kwh"], "annual_consumption_kwh") <= 0:
        raise ValueError("'annual_consumption_kwh' must be > 0")
    if _finite_real(cfg["battery_kwh"], "battery_kwh") < 0:
        raise ValueError("'battery_kwh' must be >= 0")
    if cfg.get("pv_module") is not None and cfg["pv_module"] not in MODULES:
        available = ", ".join(sorted(MODULES))
        raise ValueError(f"Unknown PV module {cfg['pv_module']!r}. Available: {available}")
    tilt = cfg.get("tilt")
    if tilt is not None and not 0 <= _finite_real(tilt, "tilt") <= 90:
        raise ValueError("'tilt' must be between 0 and 90")
    azimuth = cfg.get("azimuth")
    if azimuth is not None and not 0 <= _finite_real(azimuth, "azimuth") <= 360:
        raise ValueError("'azimuth' must be between 0 and 360")
    if not has_arrays:
        _validate_bifacial_settings(
            cfg["bifacial_model"],
            cfg.get("pv_module") or default_module_key(),
            cfg["gcr"],
            cfg["pvrow_height"],
            cfg["pvrow_pitch"],
        )
    if not 0 < _finite_real(cfg["inverter_efficiency"], "inverter_efficiency") <= 1:
        raise ValueError("'inverter_efficiency' must be between 0 (exclusive) and 1 (inclusive)")
    if cfg["inverter_ac_rating_kw"] is not None:
        if _finite_real(cfg["inverter_ac_rating_kw"], "inverter_ac_rating_kw") <= 0:
            raise ValueError("'inverter_ac_rating_kw' must be > 0")
    elif _finite_real(cfg["inverter_loading_ratio"], "inverter_loading_ratio") <= 0:
        raise ValueError("'inverter_loading_ratio' must be > 0")


def _validate_time_and_weather(cfg: dict[str, Any]) -> None:
    """Validate simulation horizon, resolution, and solar-model selections."""
    if not _is_int(cfg["projection_years"]) or cfg["projection_years"] < 1:
        raise ValueError("'projection_years' must be >= 1")
    if not 0 <= _finite_real(cfg["pv_degradation_rate"], "pv_degradation_rate") < 1:
        raise ValueError("'pv_degradation_rate' must be between 0 (inclusive) and 1 (exclusive)")
    if cfg["resolution"] not in ("h", "15min"):
        raise ValueError("'resolution' must be 'h' or '15min'")
    _validate_weather_source(cfg)
    cfg["horizon_profile"] = normalise_horizon_profile(cfg["horizon_profile"])
    _validate_sky_settings(cfg["transposition_model"], cfg["albedo"], cfg["surface_type"], cfg["model_perez"])
    if not is_known_model(cfg["solar_position"], SOLAR_POSITION_METHODS):
        valid = ", ".join(SOLAR_POSITION_METHODS)
        raise ValueError(f"'solar_position' must be one of: {valid}")
    if not is_known_model(cfg["iam_model"], IAM_MODELS):
        valid = ", ".join(IAM_MODELS)
        raise ValueError(f"'iam_model' must be one of: {valid}")
    if not is_known_model(cfg["diffuse_iam"], DIFFUSE_IAM_METHODS):
        valid = ", ".join(DIFFUSE_IAM_METHODS)
        raise ValueError(f"'diffuse_iam' must be one of: {valid}")
    if not is_known_model(cfg["temperature_model"], TEMPERATURE_MODELS):
        valid = ", ".join(TEMPERATURE_MODELS)
        raise ValueError(f"'temperature_model' must be one of: {valid}")
    cfg["bifacial_model"] = normalise_model_name(cfg["bifacial_model"])
    overrides = cfg.get("pv_loss_overrides")
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise TypeError("'pv_loss_overrides' must be a dict of loss component percentages")
        for name, value in overrides.items():
            if not 0 <= _finite_real(value, f"pv_loss_overrides[{name!r}]") <= 100:
                raise ValueError(f"'pv_loss_overrides[{name!r}]' must be a percentage between 0 and 100")
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


def _validate_economics(cfg: dict[str, Any]) -> None:
    """Validate financial and optional export-emissions inputs."""
    for key in ("inflation_rate", "discount_rate"):
        if _finite_real(cfg[key], key) <= -1:
            raise ValueError(f"'{key}' must be greater than -1")
    if not -1 < _finite_real(cfg["sell_price_inflation"], "sell_price_inflation") < 1:
        raise ValueError("'sell_price_inflation' must be between -1 and 1 (exclusive)")
    for key in ("import_price_escalation", "om_escalation"):
        if cfg[key] is not None and _finite_real(cfg[key], key) <= -1:
            raise ValueError(f"'{key}' must be greater than -1 when configured")
    if not 0 <= _finite_real(cfg["replacement_cost_learning"], "replacement_cost_learning") < 1:
        raise ValueError("'replacement_cost_learning' must be at least 0 and below 1")
    if "costs" in cfg:
        COSTS_TABLE.validate(cfg["costs"])
    if cfg["export_emissions_factor_gco2_kwh"] is not None:
        if _finite_real(cfg["export_emissions_factor_gco2_kwh"], "export_emissions_factor_gco2_kwh") < 0:
            raise ValueError("'export_emissions_factor_gco2_kwh' must be >= 0 when configured")


def _validate_battery_and_degradation(cfg: dict[str, Any]) -> None:
    """Validate battery dispatch and explicit degradation-engine selection."""
    min_soc = _finite_real(cfg["battery_min_soc"], "battery_min_soc")
    max_soc = _finite_real(cfg["battery_max_soc"], "battery_max_soc")
    if not 0 <= min_soc < max_soc <= 1:
        raise ValueError("'battery_min_soc' and 'battery_max_soc' must satisfy 0 <= min < max <= 1")
    if not 0 < _finite_real(cfg["battery_eol_percentage"], "battery_eol_percentage") < 1:
        raise ValueError("'battery_eol_percentage' must be between 0 and 1 (exclusive)")
    if cfg["battery_rte"] is not None and not 0 < _finite_real(cfg["battery_rte"], "battery_rte") <= 1:
        raise ValueError("'battery_rte' must be between 0 (exclusive) and 1 (inclusive)")
    for key in ("battery_max_charge_power_w", "battery_max_discharge_power_w"):
        if cfg[key] is not None and _finite_real(cfg[key], key) < 0:
            raise ValueError(f"'{key}' must be >= 0 when configured")
    if cfg["battery_power_limit_c_rate"] is not None:
        if _finite_real(cfg["battery_power_limit_c_rate"], "battery_power_limit_c_rate") <= 0:
            raise ValueError("'battery_power_limit_c_rate' must be greater than 0 when configured")
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
    if not isinstance(cfg["dc_coupled"], bool):
        raise TypeError("'dc_coupled' must be a boolean")
    if not cfg["dc_coupled"]:
        raise NotImplementedError("BREOS 0.3.x supports DC-coupled/hybrid battery dispatch only")
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

    if not isinstance(cfg["enable_resistance_fade"], bool):
        raise TypeError("'enable_resistance_fade' must be a boolean")

    # Validated through breos.execution so App and Monte Carlo cannot disagree
    # about which names exist. The default stays "python": the compiled path is
    # an optimisation that must be asked for, never selected by ambient state.
    cfg["execution_backend"] = validate_execution_backend(cfg["execution_backend"])

    degradation_engine = str(cfg["degradation_engine"]).strip().lower()
    if degradation_engine not in ("native", "blast"):
        raise ValueError("'degradation_engine' must be one of: native, blast")
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


def _is_int(value: Any) -> bool:
    """Return whether *value* is an integer, excluding booleans."""
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_real(value: Any, key: str) -> float:
    """Return a finite float or raise an actionable public config error."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"'{key}' must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"'{key}' must be a finite number")
    return result


def resolve_location(cfg: dict[str, Any]) -> tuple[float, float, str, str | None]:
    """Resolve a location preset or custom coordinate dict."""
    loc = cfg["location"]
    if isinstance(loc, str):
        locations = load_json("locations.json")
        if loc not in locations:
            available = ", ".join(sorted(locations))
            raise ValueError(f"Unknown location '{loc}'. Available: {available}")
        loc_data = locations[loc]
        return loc_data["latitude"], loc_data["longitude"], loc_data["timezone"], loc
    return loc["latitude"], loc["longitude"], loc["timezone"], None


def normalise_pv_arrays(arrays: list[dict[str, Any]] | None, cfg: dict[str, Any], lat: float) -> list[dict[str, Any]]:
    """Apply App-level PV defaults to each configured PV array."""
    if not arrays:
        return []

    default_module = cfg.get("pv_module") or default_module_key()
    configured_tilt = cfg.get("tilt")
    configured_azimuth = cfg.get("azimuth")
    default_tilt = configured_tilt if configured_tilt is not None else estimate_optimal_tilt(lat)
    default_azimuth = configured_azimuth if configured_azimuth is not None else default_azimuth_fn(lat)

    # Tracker settings are inherited from the top level like module, tilt and
    # azimuth. The PV model has its own fallbacks for an array without them,
    # and those used to win: a top-level single-axis tracker ran fixed-tilt
    # once pv_arrays was set.
    tracker_defaults = {key: cfg.get(key, DEFAULTS[key]) for key in ("tracking", *_TRACKER_GEOMETRY_KEYS)}
    if tracker_defaults["axis_azimuth"] is None:
        tracker_defaults["axis_azimuth"] = default_azimuth_fn(lat)

    normalized: list[dict[str, Any]] = []
    for arr in arrays:
        entry = {
            "modules": int(arr["modules"]),
            "module": arr.get("module") or default_module,
            "tilt": float(arr.get("tilt", default_tilt)),
            "azimuth": float(arr.get("azimuth", default_azimuth)),
            "tracking": arr.get("tracking", tracker_defaults["tracking"]),
        }
        if entry["tracking"] != "fixed":
            for key in _TRACKER_GEOMETRY_KEYS:
                value = arr.get(key)
                entry[key] = tracker_defaults[key] if value is None else value
        for key in _PV_ARRAY_OPTION_KEYS:
            if key in arr:
                entry[key] = arr[key]
        normalized.append(entry)
    return normalized


def resolve_pv_system(
    cfg: dict[str, Any], lat: float
) -> tuple[list[dict[str, Any]], str, PVModuleParams, int, float, float, float, float]:
    """Resolve PV module, array, tilt, azimuth, and system sizing details.

    The module is returned twice: as its catalogue key and as its parameters.

    Returns the resolved module count rather than writing it back into ``cfg``;
    the caller materialises it so the dict wrapped by the frozen
    :class:`ResolvedAppConfig` is built once and not mutated in place.
    """
    pv_arrays = normalise_pv_arrays(cfg.get("pv_arrays"), cfg, lat)
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

    tilt = cfg["tilt"] if cfg["tilt"] is not None else estimate_optimal_tilt(lat)
    azimuth = cfg["azimuth"] if cfg["azimuth"] is not None else default_azimuth_fn(lat)
    return pv_arrays, module_name, pv_params, n_modules, avg_module_power_w, system_kwp, tilt, azimuth


def validate_temperature_module_metadata(
    temperature_model: str,
    pv_arrays: list[dict[str, Any]],
    pv_params: PVModuleParams,
) -> None:
    """Validate any module metadata required by the selected thermal model.

    Array configurations may name different modules, so SAM NOCT needs each
    one checked during App config resolution rather than failing after weather
    loading. The thermal kernel repeats this validation for direct solar calls.
    """
    model = normalise_model_name(temperature_model)
    modules = [get_module(arr["module"]) for arr in pv_arrays] if pv_arrays else [pv_params]
    for module in modules:
        validate_temperature_inputs(model, module.Module_Efficiency, module.NOCT)


def resolve_tracking(cfg: dict[str, Any], lat: float) -> tuple[str, float]:
    """Resolve tracker mode and orientation defaults."""
    tracking = cfg["tracking"]
    if tracking not in _TRACKING_MODES:
        raise ValueError(f"tracking must be 'fixed', 'single_axis', or 'dual_axis', got {tracking!r}")
    axis_azimuth = cfg["axis_azimuth"] if cfg["axis_azimuth"] is not None else default_azimuth_fn(lat)
    return tracking, axis_azimuth


def resolve_costs(cfg: dict[str, Any]) -> CostParams:
    """Build CostParams from packaged presets, overrides, and financial defaults.

    Preset keys override the :class:`CostParams` dataclass defaults; a key
    missing from a preset falls back to the same default used when no
    preset is configured, so the two paths cannot diverge.
    """
    params: dict[str, Any] = {}

    if cfg.get("cost_preset"):
        costs_db = load_json("costs.json")
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

    emissions_db = load_json("emissions.json")
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
    capacity (ADR 0003 E4), the price the optimizer uses too.
    """
    return calculate_costs(
        n_modules=cfg["n_modules"],
        module_power_w=resolved.avg_module_power_w,
        battery_capacity_wh=cfg["battery_kwh"] * 1000,
        cost_params=resolved.cost_params,
        # The rating that clips production is the one CAPEX prices (#181).
        inverter_ac_capacity_w=resolved.inverter_ac_capacity_w,
        replacement_cost_each=replacement_event_cost(cfg["battery_kwh"], resolved.cost_params.battery_cost_per_kwh),
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
# A later layer (a CLI flag over a file, a sweep value over the base) that
# sets one drops the other from the layers below: see override_config.
EXCLUSIVE_ALTERNATIVES: Mapping[str, str] = {
    "inverter_ac_rating_kw": "inverter_loading_ratio",
    "inverter_loading_ratio": "inverter_ac_rating_kw",
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
    normalized = normalize_config_keys(config)
    _check_one_inverter_sizing(normalized)
    cfg = merge_defaults(normalized)
    if cfg["inverter_ac_rating_kw"] is not None:
        # The absolute rating replaces the ratio, so the ratio is reported unset.
        cfg["inverter_loading_ratio"] = None
    _normalise_config_values(cfg)
    validate_config(cfg)

    lat, lon, timezone, loc_key = resolve_location(cfg)
    pv_arrays, pv_module_key, pv_params, n_modules, avg_module_power_w, system_kwp, tilt, azimuth = resolve_pv_system(
        cfg, lat
    )
    validate_temperature_module_metadata(cfg["temperature_model"], pv_arrays, pv_params)
    tracking, axis_azimuth = resolve_tracking(cfg, lat)

    # Materialise the resolved module count (derived from pv_arrays when set)
    # into a fresh dict rather than mutating the merged config in place.
    cfg = {**cfg, "n_modules": n_modules}
    tariff = resolve_tariff_spec(cfg, timezone)
    cost_params = resolve_costs(cfg)
    ac_capacity_w: float | None
    if cfg["inverter_ac_rating_kw"] is not None:
        ac_capacity_w = float(cfg["inverter_ac_rating_kw"]) * 1000
        # The ratio the rating implies, so CostParams does not keep a stale one.
        cost_params = replace(cost_params, dc_ac_ratio=n_modules * avg_module_power_w / ac_capacity_w)
    else:
        ac_capacity_w = inverter_ac_capacity_w(n_modules * avg_module_power_w, cfg["inverter_loading_ratio"])

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
        smart_charging=resolve_smart_charging_spec(cfg, tariff),
        cost_params=cost_params,
        emissions_params=resolve_emissions(cfg),
        period=resolve_period(cfg, timezone),
    )
