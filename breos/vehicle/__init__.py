"""Vehicle configuration and schedules; dispatch is available in _simulate_core."""

from breos.vehicle.config import AwayChargingConfig, VehicleConfig, VehicleDaySchedule, VehicleScheduleSpec
from breos.vehicle.schedule import VehicleSchedule, resolve_vehicle_schedule
from breos.vehicle.state import VehicleCarry

__all__ = [
    "AwayChargingConfig",
    "VehicleConfig",
    "VehicleCarry",
    "VehicleDaySchedule",
    "VehicleSchedule",
    "VehicleScheduleSpec",
    "resolve_vehicle_schedule",
]
