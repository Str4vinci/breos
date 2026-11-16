"""Vehicle year-row reductions and their JSON names (ADR 0005 V11)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from breos.battery import EndOfLifeEvent
from breos.vehicle.config import VehicleConfig
from breos.vehicle.state import VehicleCarry

# Flows are mean W in the ledger and kWh in year rows.
VEHICLE_ENERGY_FIELDS = {
    "pv_ac_charge_kwh": "PV_AC_To_Vehicle",
    "grid_ac_charge_kwh": "Grid_AC_To_Vehicle",
    "ac_to_load_kwh": "Vehicle_AC_To_Load",
    "pv_origin_ac_to_load_kwh": "PV_Origin_Vehicle_AC_To_Load",
    "grid_origin_ac_to_load_kwh": "Grid_Origin_Vehicle_AC_To_Load",
    "drive_served_kwh": "Vehicle_Drive_Energy",
    "drive_unserved_kwh": "Vehicle_Drive_Unserved",
    "pv_origin_driving_kwh": "PV_Origin_Vehicle_Drive_Energy",
    "away_purchased_kwh": "Vehicle_Away_Charge_Purchased",
    "departure_shortfall_kwh": "Vehicle_Departure_Shortfall",
    "capacity_window_loss_kwh": "Vehicle_Capacity_Window_Loss",
    "replacement_energy_removed_kwh": "Vehicle_Replacement_Energy_Removed",
    "replacement_energy_added_kwh": "Vehicle_Replacement_Energy_Added",
}


def vehicle_year_fields(
    sums: Mapping[str, float],
    hours: float,
    carry: VehicleCarry,
    events: Sequence[EndOfLifeEvent],
    short_departures: int,
    n_steps: int,
    year_idx: int,
    config: VehicleConfig,
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        f"{column}_kWh": sums[column] * hours / 1000.0 for column in VEHICLE_ENERGY_FIELDS.values()
    }
    swaps = [event for event in events if event.action == "replaced"]
    fields.update(
        Vehicle_Departures_Short=short_departures,
        Vehicle_SOH=carry.soh_fraction,
        Vehicle_Replacements=len(swaps),
        Vehicle_Replaced_Capacity_kWh=len(swaps) * config.battery_kwh,
        Vehicle_Replacement_Time_Years=[year_idx + (event.step + 1) / n_steps for event in swaps],
        Vehicle_Energy_Onboard_End_Wh=carry.energy_wh,
        Vehicle_Retired=carry.retired,
    )
    return fields


def vehicle_year_record(row: Mapping[Any, Any]) -> dict[str, Any]:
    """Keep energy and valuation inputs unrounded until result serialization."""
    return {
        "year": int(row["Year"]),
        **{name: float(row[f"{column}_kWh"]) for name, column in VEHICLE_ENERGY_FIELDS.items()},
        "departures_short": int(row["Vehicle_Departures_Short"]),
        "soh_fraction": float(row["Vehicle_SOH"]),
        "replacements": int(row["Vehicle_Replacements"]),
        "replaced_capacity_kwh": float(row["Vehicle_Replaced_Capacity_kWh"]),
        "replacement_time_years": list(row["Vehicle_Replacement_Time_Years"]),
        "onboard_energy_end_wh": float(row["Vehicle_Energy_Onboard_End_Wh"]),
        "retired": bool(row["Vehicle_Retired"]),
        "away_cost_year1_prices": float(row.get("Vehicle_Away_Charge_Cost", 0.0)),
        "unserved_cost_year1_prices": float(row.get("Vehicle_Drive_Unserved_Cost", 0.0)),
    }
