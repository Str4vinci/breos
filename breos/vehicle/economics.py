"""Replacement-aware vehicle wear, separate from system cash flows."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from breos.economics import _replacement_outlay, replacement_event_cost, terminal_health_credit
from breos.vehicle.config import VehicleConfig

VEHICLE_MONEY_KEYS = ("charger_cost", "battery_cost_per_kwh", "battery_replacement_cost", "outside_charging_price")


def require_vehicle_prices(config: VehicleConfig) -> None:
    """Refuse missing equipment prices even at economics entry points."""
    for name in ("battery_cost_per_kwh", "charger_cost"):
        value = getattr(config, name)
        required = name == "battery_cost_per_kwh" or config.mode == "v2h"
        if value is None and not required:
            continue
        if value is None:
            raise ValueError(
                f"Vehicle economics needs 'vehicle.{name}'; it has no default. Set 0 explicitly if appropriate"
            )
        if isinstance(value, bool) or not math.isfinite(value) or value < 0.0:
            raise ValueError(f"vehicle.{name} must be a finite non-negative price")


def vehicle_pack_value(
    config: VehicleConfig,
    events: Sequence[Mapping[str, Any]],
    final_soh: float | None,
    *,
    horizon: int,
    inflation: float,
    learning: float,
    discount: float,
) -> dict[str, Any]:
    """P and H in V10; every invoice is booked at its own swap instant."""
    require_vehicle_prices(config)
    invoice = replacement_event_cost(config.battery_kwh, config.battery_cost_per_kwh, config.battery_replacement_cost)
    history = []
    for event in events:
        time = float(event["time_years"])
        base = invoice if event["action"] == "replaced" else 0.0
        nominal = float(_replacement_outlay(np.array([base]), np.array([time]), inflation, learning)[0])
        history.append(
            {
                **event,
                "outlay_t0_prices": base,
                "outlay_nominal": nominal,
                "outlay_npv": nominal / (1 + discount) ** time,
            }
        )
    # A threshold of 1 is a valid immediate-swap study policy: no health
    # above that threshold exists and the pack earns no credit.
    credit = 0.0
    if config.eol_percentage < 1.0:
        credit = terminal_health_credit(
            final_soh_fraction=final_soh,
            threshold=config.eol_percentage,
            replacement_cost_each=config.battery_kwh * config.battery_cost_per_kwh,
            inflation_rate=inflation,
            replacement_cost_learning=learning,
            discount_rate=discount,
            horizon_years=horizon,
            npv_savings=0.0,
            enable_replacement=config.enable_replacement,
            allow_terminal_replacement=config.allow_terminal_replacement,
            replacement_min_remaining_years=config.replacement_min_remaining_years,
            skipped_replacement_action=config.skipped_replacement_action,
        ).npv
    return {
        "replacement_history": history,
        "replacement_outlays_npv": sum(event["outlay_npv"] for event in history),
        "terminal_pack_credit_npv": credit,
    }


def vehicle_wear(study: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, float]:
    outlays = float(study["replacement_outlays_npv"] - reference["replacement_outlays_npv"])
    return {
        "replacement_outlays_difference_npv": outlays,
        "wear_npv": outlays + reference["terminal_pack_credit_npv"] - study["terminal_pack_credit_npv"],
    }
