"""Shared pieces of the multi-year projection that App and Monte Carlo run.

App (``breos.runners.app``) and Monte Carlo (``breos.montecarlo``) simulate a
configured system year by year. What they share is how a configuration turns
into the battery the dispatch runs, so it is built here once and neither loop
can drift from the other (#179).
"""

from __future__ import annotations

import math
from typing import Any

from breos.app_config import ResolvedAppConfig
from breos.battery import BatteryConfig


def build_battery_config(cfg: dict[str, Any], resolved: ResolvedAppConfig, *, initial_soh: float) -> BatteryConfig:
    """Build the battery one projection year runs, starting at ``initial_soh``.

    A configured round-trip efficiency is split evenly across charge and
    discharge, the BatteryConfig default convention. Replacement is on, and
    a replacement is priced at the configured storage cost per kWh.
    """
    battery_kwh = cfg["battery_kwh"]
    efficiency: dict[str, Any] = {}
    if cfg["battery_rte"] is not None:
        one_way = math.sqrt(cfg["battery_rte"])
        efficiency = {"charge_efficiency": one_way, "discharge_efficiency": one_way}
    return BatteryConfig(
        nominal_energy_wh=battery_kwh * 1000,
        initial_soh=initial_soh,
        eol_percentage=cfg["battery_eol_percentage"],
        max_soc=cfg["battery_max_soc"],
        min_soc=cfg["battery_min_soc"],
        dc_coupled=cfg["dc_coupled"],
        inverter_efficiency=cfg["inverter_efficiency"],
        inverter_ac_capacity_w=resolved.inverter_ac_capacity_w,
        enable_replacement=True,
        replacement_cost=resolved.cost_params.battery_cost_per_kwh * battery_kwh,
        calendar_model=cfg["calendar_model"],
        max_charge_power_w=cfg["battery_max_charge_power_w"],
        max_discharge_power_w=cfg["battery_max_discharge_power_w"],
        power_limit_c_rate=cfg["battery_power_limit_c_rate"],
        enable_resistance_fade=cfg.get("enable_resistance_fade", False),
        **efficiency,
    )


def build_pv_only_battery_config(cfg: dict[str, Any], resolved: ResolvedAppConfig) -> BatteryConfig:
    """Build the config a PV-only year runs under.

    PV-only runs still go through the inverter model, so the configured
    efficiency and AC clipping apply as they do with a battery. Monte Carlo's
    PV chain memo uses this too: if it resolved the inverter separately, a
    drift would memoize a conversion for one inverter and spend it on another.
    """
    return BatteryConfig(
        nominal_energy_wh=0,
        inverter_efficiency=cfg["inverter_efficiency"],
        inverter_ac_capacity_w=resolved.inverter_ac_capacity_w,
    )
