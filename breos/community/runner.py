"""Internal full-horizon community physics and subsequent meter settlement.

Project years replay the commissioning calendar, exactly as App does. Their
settlement frames are keyed by 1-based project year, not restamped to new
calendar years (which would change leap-day and tariff-clock semantics).
No community pricing, ownership accounting or public facade lives here.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from breos._daily_persistence import DailyPersistenceController
from breos.app_config import ResolvedAppConfig, resolve_location
from breos.app_inputs import (
    AppRuntimeDependencies,
    _input_cache_key,
    load_consumption_profile,
    prepare_planner_forecast,
    prepare_simulation_inputs_cached,
    reuse_prepared_inputs,
    unknown_source_metadata,
)
from breos.battery import align_simulation_inputs
from breos.community.config import CommunityConfig, HouseholdConfig, _thaw
from breos.community.settlement import SETTLEMENT_SCHEMA_VERSION, MeterBalances, settle_meter_balances
from breos.execution import aggregate_jit_cache_states, backend_provenance, config_has_battery
from breos.load_profiles import LOAD_PROFILE_METADATA_KEY, load_profile
from breos.projection import ProjectionRun, ProjectionYear, run_projection
from breos.pv.model_options import solar_position_record, solar_position_warning_scope, warn_solar_position_fallback
from breos.smart_charging import PLANNER_MODES, FixedTargetDayController, resolve_instructions
from breos.tariffs import ResolvedTariff
from breos.utils import local_datetime_index
from breos.weather import (
    WEATHER_METADATA_KEY,
    build_battery_temperature_series,
    fetch_tmy_weather_data,
    load_weather,
    resample_to_15min,
)


@dataclass(frozen=True)
class SiteRun:
    """A site's standalone physics and the unrounded inputs PR 4 will price.

    ``projection`` is absent for a load-only household; its ``load_w`` and
    ``meters`` then hold all physical flows. PV sites retain full-horizon
    priced flows in W, year rows, carry and event records. Their large first
    dispatch frame is discarded unless ``retain_site_frames=True`` was asked.
    ``meters`` are Wh and own their arrays, separate from projection powers.
    """

    config: ResolvedAppConfig | HouseholdConfig
    index: pd.DatetimeIndex
    projection: ProjectionRun | None
    meters: tuple[MeterBalances, ...]
    load_w: pd.Series
    tariff: ResolvedTariff | None
    reference_tariff: ResolvedTariff | None
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class CommunityRun:
    config: CommunityConfig
    sites: Mapping[str, SiteRun]
    settlements: Mapping[int, pd.DataFrame]
    settlement_totals: Mapping[int, pd.DataFrame]
    provenance: Mapping[str, Any]


def iter_settlements(run: CommunityRun) -> Iterator[tuple[int, pd.DataFrame]]:
    """Yield each project year's settlement, rebuilding unretained frames.

    Later frames are temporary unless the caller retains them. Meter arrays
    and full-horizon projection carry remain unchanged.
    """
    config = run.config
    index = pd.DatetimeIndex(run.settlements[1].index.get_level_values("interval").unique())
    for year in range(1, config.community.shared.projection_years + 1):
        if year in run.settlements:
            yield year, run.settlements[year]
            continue
        frame = settle_meter_balances(
            {site.id: run.sites[site.id].meters[year - 1] for site in config.participants},
            {asset.id: run.sites[asset.id].meters[year - 1] for asset in config.assets},
            index=index,
            allocation=config.community.allocation,
            coefficients=config.community.coefficients,
            pool_participant_export=config.community.pool_participant_export,
        )
        frame.attrs["project_year"] = year
        yield year, frame
        del frame


def _dependencies() -> AppRuntimeDependencies:
    return AppRuntimeDependencies(
        load_profile=load_profile,
        load_weather=load_weather,
        fetch_tmy_weather_data=fetch_tmy_weather_data,
        resample_to_15min=resample_to_15min,
        build_battery_temperature_series=build_battery_temperature_series,
    )


def _calendar(cfg: Mapping[str, Any], timezone: str) -> pd.DatetimeIndex:
    year = int(cfg["start_date"][:4])
    return pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="15min", inclusive="left", tz=timezone)


def _zero_load_profile(*, start_date: str, timezone: str, **_kwargs: Any) -> pd.DataFrame:
    """An asset's real zero load, never a scaled household placeholder."""
    frame = pd.DataFrame({"Load": 0.0}, index=_calendar({"start_date": start_date}, timezone))
    frame.attrs[LOAD_PROFILE_METADATA_KEY] = {"source": "community_zero_load"}
    return frame


def _config_hash(site: ResolvedAppConfig | HouseholdConfig) -> str:
    # Hash the actual resolved config, not a rounded public result or an
    # input-cache key (which deliberately omits dispatch and economic keys).
    record = {"config": _thaw(site.cfg)}
    if isinstance(site, ResolvedAppConfig):
        record["module_catalogue"] = site.module_catalogue
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _meter_balances(import_w: np.ndarray, export_w: np.ndarray) -> MeterBalances:
    # A4: dispatch powers become interval energies exactly once. Do not net
    # imports/exports or remove inverter draw or grid charging here.
    return MeterBalances(import_w * 0.25, export_w * 0.25)


def _check_calendar(index: pd.DatetimeIndex, expected: pd.DatetimeIndex) -> None:
    if index.tz is None or not index.tz_convert(expected.tz).equals(expected):
        raise ValueError("site does not cover the community's full 15-minute calendar")


def _run_household(site: HouseholdConfig, deps: AppRuntimeDependencies, index: pd.DatetimeIndex) -> SiteRun:
    cfg = _thaw(site.cfg)
    load = load_consumption_profile(cfg, deps, site.timezone)
    _check_calendar(pd.DatetimeIndex(load.index), _calendar(cfg, site.timezone))
    values = load.iloc[:, 0].to_numpy(dtype=float, copy=True)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("household load must be finite and nonnegative")
    # Match App's legal-clock load alignment, including repeated annual
    # edges when a UTC weather year differs from the household's civil year.
    aligned = align_simulation_inputs(pd.Series(0.0, index=index), load, freq="15min")
    values = aligned.load_w.copy()
    load_w = pd.Series(values, index=index, name="Houseload")
    tariff = site.tariff.resolve(index, site.timezone) if site.tariff else None
    reference = site.reference_tariff.resolve(index, site.timezone) if site.reference_tariff else None
    # The same calendar and load replay every year, so equal arrays are kept
    # once, like ProjectionRun.priced_flows. No equipment carry is invented.
    meters = _meter_balances(values, np.zeros_like(values))
    return SiteRun(
        site,
        index,
        None,
        (meters,) * cfg["projection_years"],
        load_w,
        tariff,
        reference,
        {
            "config_hash": _config_hash(site),
            "load_profile": deepcopy(
                load.attrs.get(LOAD_PROFILE_METADATA_KEY) or unknown_source_metadata("load-profile")
            ),
            "execution": {"effective": "load_only"},
        },
    )


@solar_position_warning_scope()
def _run_pv_site(site: ResolvedAppConfig, deps: AppRuntimeDependencies, *, retain_site_frames: bool) -> SiteRun:
    cfg = site.cfg
    has_battery = config_has_battery(cfg)
    execution = backend_provenance(cfg["execution_backend"], pv_only=not has_battery)
    inputs = prepare_simulation_inputs_cached(cfg, site, deps)
    index = pd.DatetimeIndex(inputs.dc_system_base.index)
    solar_position = solar_position_record(cfg["solar_position"], inputs.weather, cfg["resolution"])
    warn_solar_position_fallback(solar_position["requested"], solar_position["resolved"])
    tariff = site.tariff.resolve(index, site.timezone) if site.tariff else None
    reference = site.reference_tariff.resolve(index, site.timezone) if site.reference_tariff else None
    spec = site.smart_charging
    controller: DailyPersistenceController | FixedTargetDayController | None = None
    if spec is not None and spec.mode in PLANNER_MODES and has_battery:
        assert tariff is not None
        forecast = prepare_planner_forecast(cfg, site, deps, inputs, index)
        controller = DailyPersistenceController.for_run(
            spec, tariff, freq=cfg["resolution"], execution_backend=cfg["execution_backend"], forecast=forecast
        )
    elif spec is not None and has_battery:
        instructions = resolve_instructions(spec, tariff)
        if instructions is not None:
            controller = FixedTargetDayController(instructions)

    def year_inputs(year_idx: int) -> ProjectionYear:
        factor = (1 - cfg["pv_degradation_rate"]) ** year_idx
        return ProjectionYear(
            pv_degradation_factor=factor,
            pv_dc=inputs.dc_system_base * factor,
            houseload=inputs.load_data,
            temperature_series=inputs.temperature_series,
            inverter_ac_derate=inputs.inverter_ac_derate,
            inverter_curve_row=inputs.inverter_curve_row,
            inverter_units=inputs.inverter_units,
        )

    projection = run_projection(
        cfg,
        site,
        cfg["projection_years"],
        year_inputs,
        has_battery=has_battery,
        execution_backend=cfg["execution_backend"],
        observe_jit_per_year=True,
        tariff=tariff,
        reference_tariff=reference,
        day_controller=controller,
        record_period_energy=True,
        record_priced_flows=True,
    )
    assert projection.priced_flows is not None and projection.first_year_results_df is not None
    first_frame = projection.first_year_results_df
    _check_calendar(local_datetime_index(first_frame["Datetime"]), index)
    load_w = first_frame["Houseload"].copy()
    load_w.index = index
    meters = tuple(_meter_balances(year["Import_From_Grid"], year["PV_AC_Export"]) for year in projection.priced_flows)
    if not retain_site_frames:
        projection = replace(projection, first_year_results_df=None)
    if cfg["execution_backend"] == "numba":
        execution["jit_cache"] = aggregate_jit_cache_states(projection.jit_cache_states)
    return SiteRun(
        site,
        index,
        projection,
        meters,
        load_w,
        tariff,
        reference,
        {
            "config_hash": _config_hash(site),
            "weather": deepcopy(inputs.weather.attrs.get(WEATHER_METADATA_KEY) or unknown_source_metadata("weather")),
            "load_profile": deepcopy(
                inputs.load_data.attrs.get(LOAD_PROFILE_METADATA_KEY) or unknown_source_metadata("load-profile")
            ),
            "solar_position": solar_position,
            "execution": execution,
        },
    )


def run_community(
    config: CommunityConfig,
    *,
    deps: AppRuntimeDependencies | None = None,
    retain_site_frames: bool = False,
    retain_settlement_frames: bool = False,
) -> CommunityRun:
    """Run standalone sites for the whole horizon, then settle each year.

    ``config`` is PR 2's parsed configuration. Dependency injection has the
    same shape as App's and supports offline weather/load providers. All
    projections use per-step years and record meter powers even when a
    price-aware controller runs. No valuation function is called.

    PV preparations are grouped by the resolved form of
    ``input_configuration_key`` and reused within ``reuse_prepared_inputs``.
    We use its internal key directly: re-resolving materialised inverter
    ratings or an asset's zero load as an App config is invalid. No key is
    dropped to manufacture a hit. Different asset dependencies also remain
    separate in the preparation cache. Only one cached preparation is kept.

    Returned sites and settlements are in sorted id/project-year order.
    The PV sites must share the same interval instants; their weather clock
    defines the simulation calendar, as in App. Load-only households align
    to it with App's alignment helper. Without PV the civil load calendar is
    used, with no weather fetch. No UTC weather year is cropped to civil dates.
    Each settlement replays the commissioning calendar; ``project_year`` in
    frame attrs identifies its position in the horizon. The full first-year
    settlement and every year's energy totals are retained by default; set
    ``retain_settlement_frames`` to retain every frame. Later frames can be
    rebuilt lazily with ``iter_settlements`` from the meter arrays.
    """
    deps = deps or _dependencies()
    asset_deps = replace(deps, load_profile=_zero_load_profile)
    definitions: dict[str, ResolvedAppConfig | HouseholdConfig] = {
        **{participant.id: participant.site for participant in config.participants},
        **{asset.id: asset.site for asset in config.assets},
    }
    asset_ids = {asset.id for asset in config.assets}
    sites = {}
    groups = []
    for site_id in sorted(definitions):
        site = definitions[site_id]
        if isinstance(site, ResolvedAppConfig):
            groups.append((_input_cache_key(site.cfg, site), site_id))
    # An uncacheable config runs alone, with no accidental equality to a
    # real key. The id is the deterministic tie-breaker within every group.
    groups.sort(key=lambda item: (item[0] is None, item[0] or "", item[1]))
    with reuse_prepared_inputs():
        for _key, site_id in groups:
            pv_site = definitions[site_id]
            assert isinstance(pv_site, ResolvedAppConfig)
            try:
                sites[site_id] = _run_pv_site(
                    pv_site, asset_deps if site_id in asset_ids else deps, retain_site_frames=retain_site_frames
                )
            except (ValueError, TypeError, OSError) as exc:
                raise type(exc)(f"site '{site_id}': {exc}") from exc
    sites = {key: sites[key] for key in sorted(sites)}
    shared = config.community.shared
    timezone = resolve_location(shared.app_payload())[2]
    index = sites[min(sites)].index.tz_convert(timezone) if sites else _calendar(shared.app_payload(), timezone)
    for site_id in sorted(definitions):
        site = definitions[site_id]
        if isinstance(site, HouseholdConfig):
            try:
                sites[site_id] = _run_household(site, deps, index)
            except (ValueError, TypeError, OSError) as exc:
                raise type(exc)(f"site '{site_id}': {exc}") from exc
    sites = {key: sites[key] for key in sorted(sites)}
    for site_id, site_run in sites.items():
        try:
            _check_calendar(site_run.index, index)
        except ValueError as exc:
            raise ValueError(f"site '{site_id}': {exc}") from exc
    settlements = {}
    settlement_totals = {}
    for year_idx in range(shared.projection_years):
        frame = settle_meter_balances(
            {site.id: sites[site.id].meters[year_idx] for site in config.participants},
            {asset.id: sites[asset.id].meters[year_idx] for asset in config.assets},
            index=index,
            allocation=config.community.allocation,
            coefficients=config.community.coefficients,
            pool_participant_export=config.community.pool_participant_export,
        )
        frame.attrs["project_year"] = year_idx + 1
        settlement_totals[year_idx + 1] = (
            frame.drop(columns="Coefficient").groupby(level=["site_type", "site_id"], sort=True).sum()
        )
        if year_idx == 0 or retain_settlement_frames:
            settlements[year_idx + 1] = frame
        del frame
    return CommunityRun(
        config,
        MappingProxyType(sites),
        MappingProxyType(settlements),
        MappingProxyType(settlement_totals),
        {
            "settlement_schema_version": SETTLEMENT_SCHEMA_VERSION,
            "site_ids": list(sites),
            "site_config_hashes": {key: site.provenance["config_hash"] for key, site in sites.items()},
            "calendar": "replayed_commissioning_year",
            "first_interval": index[0].isoformat(),
            "last_interval": index[-1].isoformat(),
            "meter_energy_unit": "Wh",
            "dispatch_power_unit": "W",
        },
    )
