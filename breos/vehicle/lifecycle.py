"""Vehicle day close: ageing, clipping and same-SOC pack replacement.

Imported lazily by battery._simulate_core. The stationary and vehicle models
share the lifecycle protocol, native equations, event type and policy guards;
the vehicle's inventory and day-close ledger operations are its own.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Mapping

import numpy as np
import pandas as pd

from breos.battery import EndOfLifeEvent
from breos.degradation.native import (
    _get_degradation_params,
    _update_battery_soh_from_cycles,
    update_battery_soh_calendar,
)
from breos.degradation.protocol import (
    BlastDegradationAdapter,
    DegradationDay,
    DegradationLifecycle,
    NativeDegradationAdapter,
)
from breos.degradation.replacement import replacement_skip_reason
from breos.vehicle.config import VehicleConfig
from breos.vehicle.state import VehicleCarry

VEHICLE_END_OF_LIFE_EVENTS_ATTR = "vehicle_end_of_life_events"


def _build_vehicle_lifecycle(config: VehicleConfig, carry: VehicleCarry | None) -> DegradationLifecycle:
    state = {} if carry is None else (carry.degradation_state or {})
    if state and state.get("degradation_engine") != config.degradation_engine:
        raise ValueError("vehicle degradation engine differs from the carried lifecycle")
    if config.degradation_engine == "blast":
        if carry is not None and not state and carry.soh_fraction != 1.0:
            raise ValueError("a faded BLAST vehicle needs its lifecycle snapshot")
        if state and state.get("blast_model") != config.blast_model:
            raise ValueError("vehicle BLAST model differs from the carried lifecycle")
        return BlastDegradationAdapter(
            str(config.blast_model),
            initial_state=state,
            initial_fec=float(state.get("fec_cum", 0.0)),
            initial_calendar_seconds=float(state.get("cumulative_calendar_seconds", 0.0)),
        )
    if state and state.get("calendar_model", config.calendar_model) != config.calendar_model:
        raise ValueError("vehicle calendar model differs from the carried lifecycle")
    if (
        carry is not None
        and "soh_fraction" in state
        and not math.isclose(float(state["soh_fraction"]), carry.soh_fraction, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError("vehicle carried health disagrees with its lifecycle snapshot")
    k0, activation, exponent, soc_exponent = _get_degradation_params(config.calendar_model)
    return NativeDegradationAdapter(
        model_key=config.calendar_model,
        initial_soh_fraction=1.0 if carry is None else carry.soh_fraction,
        initial_fec=float(state.get("fec_cum", 0.0)),
        initial_calendar_seconds=float(state.get("cumulative_calendar_seconds", 0.0)),
        k0_fraction=k0,
        activation_energy=activation,
        soc_exponent=soc_exponent,
        time_exponent=exponent,
        cycle_step=_update_battery_soh_from_cycles,
        calendar_step=update_battery_soh_calendar,
        initial_rainflow_state=state.get("native_rainflow_state"),
    )


class VehicleLifecycle:
    """The independent vehicle pack, advanced only between dispatch windows."""

    def __init__(
        self, config: VehicleConfig, carry: VehicleCarry | None, *, initial_soc: float, initial_temperature: float
    ) -> None:
        self.config = config
        self.initial_carry = carry
        self.lifecycle = _build_vehicle_lifecycle(config, carry)
        self.soh = self.lifecycle.soh()
        if not math.isfinite(self.soh) or not 0.0 <= self.soh <= 1.0:
            raise ValueError("vehicle health must be finite and between 0 and 1")
        if carry is not None and not math.isclose(carry.soh_fraction, self.soh, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("vehicle carried health disagrees with its lifecycle snapshot")
        state = {} if carry is None else (carry.degradation_state or {})
        self.start_soc = float(state.get("day_start_soc_absolute", initial_soc))
        self.start_temperature = float(state.get("day_start_temperature_c", initial_temperature))
        self.fec = float(state.get("fec_cum", 0.0))
        self.calendar_seconds = float(state.get("cumulative_calendar_seconds", 0.0))
        self.fec_all = 0.0 if carry is None else carry.fec_all_packs
        self.cycle_deg = float(state.get("cumulative_cycle_degradation", 0.0))
        self.calendar_deg = float(state.get("cumulative_calendar_degradation", 0.0))
        self.retired = False if carry is None else carry.retired
        self.past_eol = False if carry is None else carry.past_end_of_life
        self.n_replacements = 0 if carry is None else carry.n_replacements
        self.replaced_wh = 0.0 if carry is None else carry.replaced_capacity_wh
        self.history = [] if carry is None else list(carry.end_of_life_events)
        self.elapsed_steps = 0 if carry is None else carry.elapsed_steps
        self.events: list[EndOfLifeEvent] = []
        self.tracking: list[dict[str, Any]] = []

    def _settle(self) -> float:
        terminal = self.lifecycle.finalize_cycles()
        self.fec_all += terminal.fec - self.fec
        self.fec = terminal.fec
        self.soh = terminal.soh_fraction
        self.cycle_deg += terminal.cycle_degradation
        return terminal.cycle_degradation

    def close(
        self,
        columns: Mapping[str, np.ndarray],
        *,
        lo: int,
        hi: int,
        time_ticks: np.ndarray,
        ticks_per_second: float,
        temperature: np.ndarray,
        hours: float,
        step_time: pd.Timestamp,
        finalize: bool,
        allow_terminal_replacement: bool,
        years_after_span: int,
        n_steps: int,
    ) -> None:
        """Age real on-board SOC, then settle departures from service and clip."""
        last = hi - 1
        soc = columns["Vehicle_SOC_Absolute"][lo:hi].copy()
        day = DegradationDay(
            soc,
            time_ticks[lo:hi],
            ticks_per_second,
            temperature[lo:hi],
            hours * 3600.0,
            self.start_soc,
            self.start_temperature,
            finalize,
        )
        step = self.lifecycle.step(day)
        self.soh = step.soh_fraction
        self.calendar_seconds = step.calendar_seconds
        self.fec_all += step.fec - self.fec
        self.fec = step.fec
        self.cycle_deg += step.cycle_degradation
        self.calendar_deg += step.calendar_degradation
        cycle_deg = step.cycle_degradation
        self.start_soc = float(soc[-1])
        self.start_temperature = float(temperature[last])
        energy = float(columns["Vehicle_Energy_Onboard"][last])
        pv = float(columns["Vehicle_PV_Origin_Energy_End"][last])
        grid = float(columns["Vehicle_Grid_Origin_Energy_End"][last])
        reason = replacement_skip_reason(
            allow_terminal_replacement=allow_terminal_replacement,
            final_period=hi == n_steps,
            remaining_steps=years_after_span * n_steps + n_steps - hi,
            min_remaining_steps=self.config.replacement_min_remaining_years * n_steps,
        )
        reached = self.soh <= self.config.eol_percentage
        swap = reached and self.config.enable_replacement and reason is None and not self.retired
        retire = False
        events = []
        if swap:
            events.append(EndOfLifeEvent(last, step_time, "replaced", "end_of_life", self.soh * 100.0))
        elif reached and not self.past_eol and not self.retired:
            reason = "replacement_disabled" if not self.config.enable_replacement else reason
            assert reason is not None
            retire = self.config.skipped_replacement_action == "retire"
            events.append(EndOfLifeEvent(last, step_time, "retired" if retire else "kept", reason, self.soh * 100.0))
            self.past_eol = True
        if not swap and not retire and not self.retired and self.soh <= 0.0:
            retire = True
            events.append(EndOfLifeEvent(last, step_time, "retired", "zero_health", self.soh * 100.0))
        if swap or retire:
            cycle_deg += self._settle()
        # An outgoing pack owns its unresolved wear. Its current capacity
        # sets its SOC and any clipping before inventory is removed.
        cap = self.config.battery_kwh * 1000.0 * self.soh
        clipped = max(0.0, energy - self.config.max_soc * cap)
        if clipped > 0.0:
            pv_loss = clipped * pv / energy if energy > 0.0 else 0.0
            grid_loss = clipped * grid / energy if energy > 0.0 else 0.0
            columns["Vehicle_Capacity_Window_Loss"][last] = clipped / hours
            columns["PV_Origin_Vehicle_Capacity_Window_Loss"][last] = pv_loss / hours
            columns["Grid_Origin_Vehicle_Capacity_Window_Loss"][last] = grid_loss / hours
            energy -= clipped
            pv -= pv_loss
            grid -= grid_loss
        if swap or retire:
            columns["Vehicle_Replacement_Energy_Removed"][last] = energy / hours
            columns["PV_Origin_Vehicle_Replacement_Energy_Removed"][last] = pv / hours
            columns["Grid_Origin_Vehicle_Replacement_Energy_Removed"][last] = grid / hours
            if swap:
                # At zero capacity SOC has no denominator. Retain the last
                # measured SOC within the configured window; all energy of
                # its successor is still an explicit inventory addition.
                old_soc = (
                    energy / cap if cap > 0.0 else min(self.config.max_soc, max(self.config.min_soc, float(soc[-1])))
                )
                energy = old_soc * self.config.battery_kwh * 1000.0
                columns["Vehicle_Replacement_Energy_Added"][last] = energy / hours
                self.lifecycle.reset()
                self.soh = 1.0
                self.fec = self.cycle_deg = self.calendar_deg = self.calendar_seconds = 0.0
                self.n_replacements += 1
                self.replaced_wh += self.config.battery_kwh * 1000.0
                self.past_eol = False
                self.start_soc = old_soc
            else:
                self.retired = True
                energy = 0.0
                self.start_soc = 0.0
            pv = grid = 0.0
        for event in events:
            self.events.append(event)
            self.history.append(replace(event, step=self.elapsed_steps + last))
        columns["Vehicle_Energy_Onboard"][last] = energy
        columns["Vehicle_Energy"][last] = energy if columns["Vehicle_Plugged"][last] else 0.0
        columns["Vehicle_PV_Origin_Energy_End"][last] = pv
        columns["Vehicle_Grid_Origin_Energy_End"][last] = grid
        columns["Vehicle_SOH"][last] = self.soh
        capacity = self.config.battery_kwh * 1000.0 * self.soh
        columns["Vehicle_SOC_Absolute"][last] = energy / capacity if capacity > 0.0 else 0.0
        record = dict(
            Datetime=step_time,
            SOH=self.soh * 100.0,
            Cycle_Degradation=cycle_deg,
            Calendar_Degradation=step.calendar_degradation,
            Cumulative_Cycle_Degradation=self.cycle_deg,
            Cumulative_Calendar_Degradation=self.calendar_deg,
            Cumulative_FEC=self.fec,
            Cumulative_FEC_All_Packs=self.fec_all,
            Cumulative_Calendar_Seconds=self.calendar_seconds,
            Mean_SOC_Absolute=day.mean_soc,
            Mean_Temperature_C=day.mean_temperature_c,
        )
        record.update(self.lifecycle.tracking_fields(step))
        self.tracking.append(record)

    def carry(self, columns: Mapping[str, np.ndarray], n_steps: int) -> VehicleCarry:
        snapshot = self.lifecycle.snapshot(day_start_soc=self.start_soc, day_start_temperature_c=self.start_temperature)
        snapshot.update(
            cumulative_cycle_degradation=self.cycle_deg,
            cumulative_calendar_degradation=self.calendar_deg,
            calendar_model=self.config.calendar_model,
        )
        return VehicleCarry(
            energy_wh=float(columns["Vehicle_Energy_Onboard"][-1]),
            pv_origin_energy_wh=float(columns["Vehicle_PV_Origin_Energy_End"][-1]),
            grid_origin_energy_wh=float(columns["Vehicle_Grid_Origin_Energy_End"][-1]),
            plugged=bool(columns["Vehicle_Plugged"][-1]),
            soh_fraction=self.soh,
            degradation_state=snapshot,
            retired=self.retired,
            past_end_of_life=self.past_eol,
            n_replacements=self.n_replacements,
            replaced_capacity_wh=self.replaced_wh,
            fec_all_packs=self.fec_all,
            elapsed_steps=self.elapsed_steps + n_steps,
            end_of_life_events=tuple(self.history),
        )
