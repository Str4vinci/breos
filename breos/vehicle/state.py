"""Physical and lifecycle vehicle carry between dispatch windows and spans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from breos.battery import EndOfLifeEvent


@dataclass(frozen=True)
class VehicleCarry:
    """Pack energy/origins (Wh), lifecycle snapshot and project swap history.

    Schedule arrays supply the remaining legs and parked hours. ``plugged``
    detects boundaries across spans. Lifecycle endpoints belong to the SOC
    and temperature samples the engine consumed, before day-close rewrites.
    History steps are on the carried project clock; a run's events are local.
    """

    energy_wh: float
    pv_origin_energy_wh: float = 0.0
    grid_origin_energy_wh: float = 0.0
    plugged: bool = True
    soh_fraction: float = 1.0
    degradation_state: dict[str, Any] | None = None
    retired: bool = False
    past_end_of_life: bool = False
    n_replacements: int = 0
    replaced_capacity_wh: float = 0.0
    fec_all_packs: float = 0.0
    elapsed_steps: int = 0
    end_of_life_events: tuple[EndOfLifeEvent, ...] = ()
