"""Physical vehicle carry between dispatch windows and simulation spans."""

from dataclasses import dataclass


@dataclass(frozen=True)
class VehicleCarry:
    """On-board energy and origins in Wh, and the last step's plug state.

    The next resolved schedule supplies the remaining driving legs and parked
    hours. ``plugged`` preserves the position inside an absence: continuing
    away does not depart again, and the next home step books arrival once.
    Health and replacement state are added in PR 3.
    """

    energy_wh: float
    pv_origin_energy_wh: float = 0.0
    grid_origin_energy_wh: float = 0.0
    plugged: bool = True
