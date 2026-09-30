"""Private causal civil-day dispatch controllers (ADR 0002 A11).

A daily controller decides one configured-zone civil day at a time and returns
canonical :class:`~breos.dispatch_instructions.DispatchInstructions` for it.
It receives the day's calendar, the known tariff for the day, the current
battery state, and observations of days that have already completed.
Simulation truth is captured by the core after each dispatch segment and is
never part of a decision input: no current or future PV, load or temperature
array reaches the controller.

The seam is private. :class:`_ControllerSession` is what
:func:`breos.battery._simulate_core` drives: it splits dispatch at civil-day
boundaries, keeps a monotonic project clock, stitches a logical day across an
ADR 0002 A2 year seam, and owns the observation history. Degradation windows
stay positional; the session never moves them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING, Any, Mapping, Protocol, Sequence, TypeAlias, cast

import numpy as np
import pandas as pd

from breos.dispatch_instructions import DispatchInstructions

if TYPE_CHECKING:
    from breos.tariffs import ResolvedTariff

# A configured-zone wall-clock slot: seconds after local midnight, and the DST
# fold (0 for the first occurrence of a wall time, 1 for its repeat). It holds
# no date, so a logical day stitched across an A2 seam compares by wall slot
# only; the date is a label.
SlotKey: TypeAlias = tuple[int, int]


@dataclass(frozen=True, slots=True)
class ObservedCivilDay:
    """One fully observed configured-zone day, in its local-slot order.

    ``pv_dc_w``, ``load_w`` and ``temperature_c`` are the simulation inputs
    the dispatch read on those slots. ``temperature_c`` is the input
    (ambient or battery-location) temperature, not ``T_cell``: the dispatch
    and the daily-target planner each derive the cell temperature from it.
    """

    logical_day_ordinal: int
    local_date: str
    timezone: str
    slot_keys: tuple[SlotKey, ...]
    pv_dc_w: tuple[float, ...]
    load_w: tuple[float, ...]
    temperature_c: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class PendingObservedCivilDay:
    """A possibly incomplete day; missing observations are represented by None."""

    logical_day_ordinal: int
    local_date: str
    timezone: str
    slot_keys: tuple[SlotKey, ...]
    pv_dc_w: tuple[float | None, ...]
    load_w: tuple[float | None, ...]
    temperature_c: tuple[float | None, ...]

    @classmethod
    def empty(
        cls, logical_day_ordinal: int, local_date: str, timezone: str, slot_keys: tuple[SlotKey, ...]
    ) -> PendingObservedCivilDay:
        empty = (None,) * len(slot_keys)
        return cls(logical_day_ordinal, local_date, timezone, slot_keys, empty, empty, empty)

    def with_segment(
        self,
        start: int,
        slot_keys: Sequence[SlotKey],
        pv_dc_w: Sequence[float],
        load_w: Sequence[float],
        temperature_c: Sequence[float],
    ) -> PendingObservedCivilDay:
        """Record the observations of consecutive expected slots from ``start``."""
        stop = start + len(slot_keys)
        if start < 0 or stop > len(self.slot_keys) or tuple(self.slot_keys[start:stop]) != tuple(slot_keys):
            raise ValueError("observed slots are outside their pending civil day")
        if any(value is not None for value in self.pv_dc_w[start:stop]):
            raise ValueError("an observed slot was captured more than once")

        def splice(values: tuple[float | None, ...], new: Sequence[float]) -> tuple[float | None, ...]:
            return values[:start] + tuple(map(float, new)) + values[stop:]

        return PendingObservedCivilDay(
            self.logical_day_ordinal,
            self.local_date,
            self.timezone,
            self.slot_keys,
            splice(self.pv_dc_w, pv_dc_w),
            splice(self.load_w, load_w),
            splice(self.temperature_c, temperature_c),
        )

    @property
    def captured_slots(self) -> int:
        """How many expected slots have been observed so far."""
        return len(self.pv_dc_w) - self.pv_dc_w.count(None)

    @property
    def complete(self) -> bool:
        return None not in self.pv_dc_w

    def promote(self) -> ObservedCivilDay:
        if not self.complete:
            raise ValueError("an incomplete civil day cannot become forecast history")
        # Segments are recorded whole, so a complete day holds floats only.
        return ObservedCivilDay(
            self.logical_day_ordinal,
            self.local_date,
            self.timezone,
            self.slot_keys,
            cast("tuple[float, ...]", self.pv_dc_w),
            cast("tuple[float, ...]", self.load_w),
            cast("tuple[float, ...]", self.temperature_c),
        )


@dataclass(frozen=True, slots=True)
class KnownTariffHorizon:
    """Tariff/calendar facts aligned to one controller decision span.

    The horizon starts with the decision span and continues for up to the
    controller's ``tariff_horizon_days`` logical days, in replay order.
    ``civil_day_offsets`` holds the offset of each logical day's first slot,
    followed by the horizon length. A standalone span such as a ``[period]``
    has no day after its end, so near that end the horizon holds fewer than
    ``tariff_horizon_days`` days (as few as the decision span alone) and
    ``civil_day_offsets`` is correspondingly shorter. ``slot_keys`` gives
    each slot's configured-zone wall slot and DST fold, and
    ``calendar_positions`` its position in the replayed resolved tariff.
    These are schedule data, not weather or load observations.
    """

    timezone: str
    slot_keys: tuple[SlotKey, ...]
    period_labels: tuple[str, ...]
    period_codes: tuple[int, ...]
    import_price_per_kwh: tuple[float, ...]
    export_price_per_kwh: tuple[float, ...]
    civil_day_offsets: tuple[int, ...]
    calendar_positions: tuple[int, ...]
    schedule_hash: str


@dataclass(frozen=True, slots=True)
class ControllerBatteryState:
    """Detached physical and health state visible when a day is decided."""

    energy_wh: float
    pv_origin_energy_wh: float
    grid_origin_energy_wh: float
    soh_fraction: float
    resistance_growth: float
    charge_efficiency: float
    discharge_efficiency: float


@dataclass(frozen=True, slots=True)
class ControllerDayInput:
    """Causal information supplied to a private daily controller.

    ``expected_slot_keys`` is the whole civil day; the decision covers
    ``decision_step_count`` of them from ``civil_slot_offset``. The current
    simulation call dispatches ``segment_step_count`` of those from
    ``segment_offset``; a day stitched across an A2 seam dispatches the rest
    in the next call. ``initial_partial_day`` marks a projection that starts
    mid-day, ``clipped_start`` and ``clipped_end`` a span that lacks the day's
    first or last slots for any other reason, such as a ``[period]`` edge.
    """

    projection_year: int
    project_step_ordinal: int
    logical_day_ordinal: int
    local_date: date
    timezone: str
    expected_slot_keys: tuple[SlotKey, ...]
    decision_step_count: int
    segment_offset: int
    segment_step_count: int
    civil_slot_offset: int
    initial_partial_day: bool
    clipped_start: bool
    clipped_end: bool
    tariff: KnownTariffHorizon
    battery_config: Mapping[str, object]
    battery_state: ControllerBatteryState
    last_complete_observed_day: ObservedCivilDay | None
    complete_days_observed: int


@dataclass(frozen=True, slots=True)
class ControllerDayDecision:
    """Instructions for the logical decision span and the next policy state."""

    instructions: DispatchInstructions
    next_policy_state: object | None = None


@dataclass(frozen=True, slots=True)
class CarriedDayDecision:
    """Immutable instructions and cursor for a decision crossing a call edge.

    The decision covers ``expected_slot_keys`` from ``civil_slot_offset``;
    ``next_slot_offset`` is the first of its instructions not yet dispatched.
    """

    logical_day_ordinal: int
    local_date: str
    timezone: str
    expected_slot_keys: tuple[SlotKey, ...]
    civil_slot_offset: int
    instructions: DispatchInstructions
    next_slot_offset: int


@dataclass(frozen=True, slots=True)
class ControllerCarry:
    """Policy, causal observations, and monotonic clock across projection years.

    It is kept apart from the degradation state: the degradation engine owns
    that payload, and this one belongs to the controller seam.
    """

    policy_state: object | None = None
    last_complete_observed_day: ObservedCivilDay | None = None
    complete_days_observed: int = 0
    pending_day: PendingObservedCivilDay | None = None
    active_day_decision: CarriedDayDecision | None = None
    next_project_step_ordinal: int = 0
    next_project_day_ordinal: int = 0


class DailyDispatchController(Protocol):
    """A private causal policy that returns canonical dispatch instructions.

    ``tariff_horizon_days`` is how many logical days of known tariff the
    controller sees from the start of each decision.
    """

    @property
    def tariff_horizon_days(self) -> int: ...

    def decide_day(self, day: ControllerDayInput, policy_state: object | None) -> ControllerDayDecision: ...


def _nanoseconds(index: pd.DatetimeIndex) -> np.ndarray:
    """The index's integer nanoseconds (UTC for an aware index, wall time for a naive one)."""
    return np.asarray(cast(Any, index.as_unit("ns")).asi8, dtype=np.int64)


def _slot_arrays(utc_ns: np.ndarray, timezone: str) -> tuple[np.ndarray, np.ndarray]:
    """Wall-clock seconds after local midnight and DST fold for UTC instants."""
    utc_ns = np.asarray(utc_ns, dtype=np.int64)
    utc = pd.DatetimeIndex(utc_ns.astype("datetime64[ns]")).tz_localize("UTC")
    naive = utc.tz_convert(timezone).tz_localize(None)
    wall = (_nanoseconds(naive) - _nanoseconds(naive.normalize())) // 1_000_000_000
    # Relocalising a wall time at its first occurrence gives another instant
    # only for the repeated hour of a fall-back day.
    first = _nanoseconds(naive.tz_localize(timezone, ambiguous=np.ones(len(naive), dtype=bool)))
    return wall, (first != utc_ns).astype(np.int64)


def _local_midnight_ns(day: date, timezone: str) -> int:
    midnight = pd.Timestamp(day).tz_localize(timezone, ambiguous=True, nonexistent="shift_forward")
    return int(midnight.as_unit("ns").value)


def _locate(present: tuple[SlotKey, ...], expected: tuple[SlotKey, ...]) -> int:
    """Where the present slots sit, contiguously, in the expected civil day."""
    try:
        offset = expected.index(present[0])
    except (IndexError, ValueError) as exc:
        raise ValueError("the civil day's slots are not on the simulation's step grid") from exc
    if expected[offset : offset + len(present)] != present:
        raise ValueError("the civil day's slots are not contiguous on the simulation's step grid")
    return offset


class _InstructionBuffer:
    """Call-sized instruction arrays filled from each day's decision.

    The dispatch day indexes instructions at global input positions, so each
    decision's current segment is copied into the matching positions here.
    The arrays the kernel reads are read-only views, the same array contract
    (dtype, contiguity, writeability) a :class:`DispatchInstructions` gives
    both backends. The two scalars are kernel arguments, so every decision in
    one simulation must agree on them.
    """

    __slots__ = (
        "_discharge",
        "_reserve",
        "_target",
        "discharge_allowed",
        "reserve_fraction",
        "grid_target_fraction",
        "grid_charge_efficiency",
        "grid_import_limit_w",
    )

    def __init__(self, n_steps: int) -> None:
        self._discharge = np.zeros(n_steps, dtype=np.bool_)
        self._reserve = np.zeros(n_steps, dtype=np.float64)
        self._target = np.full(n_steps, np.nan, dtype=np.float64)
        self.discharge_allowed = self._read_only(self._discharge)
        self.reserve_fraction = self._read_only(self._reserve)
        self.grid_target_fraction = self._read_only(self._target)
        self.grid_charge_efficiency: float | None = None
        self.grid_import_limit_w: float | None = None

    @staticmethod
    def _read_only(array: np.ndarray) -> np.ndarray:
        view = array.view()
        view.setflags(write=False)
        return view

    def bind_scalars(self, instructions: DispatchInstructions) -> None:
        scalars = (instructions.grid_charge_efficiency, instructions.grid_import_limit_w)
        if self.grid_charge_efficiency is None:
            self.grid_charge_efficiency, self.grid_import_limit_w = scalars
        elif (self.grid_charge_efficiency, self.grid_import_limit_w) != scalars:
            raise ValueError(
                "every daily decision in one simulation must keep grid_charge_efficiency and grid_import_limit_w"
            )

    def write(self, lo: int, instructions: DispatchInstructions, offset: int, count: int) -> None:
        if offset < 0 or offset + count > len(instructions):
            raise ValueError("the daily decision does not cover the dispatched segment")
        hi = lo + count
        self._discharge[lo:hi] = instructions.discharge_allowed[offset : offset + count]
        self._reserve[lo:hi] = instructions.reserve_fraction[offset : offset + count]
        self._target[lo:hi] = instructions.grid_target_fraction[offset : offset + count]

    def executed(self) -> DispatchInstructions | None:
        """A copy of the instructions the call dispatched, one per simulated step.

        Only the slots this call ran are here. A decision's slots that the
        next call runs, or that no call runs because this call was the last,
        are not. None when no decision was made.
        """
        if self.grid_charge_efficiency is None or self.grid_import_limit_w is None:
            return None
        return DispatchInstructions(
            discharge_allowed=self._discharge,
            reserve_fraction=self._reserve,
            grid_target_fraction=self._target,
            grid_charge_efficiency=self.grid_charge_efficiency,
            grid_import_limit_w=self.grid_import_limit_w,
        )


def concatenate_instructions(parts: Sequence[DispatchInstructions]) -> DispatchInstructions | None:
    """Join executed instruction traces in dispatch order; None for no parts.

    Every part must keep the same two kernel scalars, as the decisions of one
    simulation do.
    """
    if not parts:
        return None
    first = parts[0]
    scalars = (first.grid_charge_efficiency, first.grid_import_limit_w)
    if any((part.grid_charge_efficiency, part.grid_import_limit_w) != scalars for part in parts[1:]):
        raise ValueError(
            "executed instructions with different grid_charge_efficiency or grid_import_limit_w cannot be joined"
        )
    return DispatchInstructions(
        discharge_allowed=np.concatenate([part.discharge_allowed for part in parts]),
        reserve_fraction=np.concatenate([part.reserve_fraction for part in parts]),
        grid_target_fraction=np.concatenate([part.grid_target_fraction for part in parts]),
        grid_charge_efficiency=scalars[0],
        grid_import_limit_w=scalars[1],
    )


@dataclass(slots=True)
class _ActiveDecision:
    logical_day_ordinal: int
    local_date: str
    expected_slot_keys: tuple[SlotKey, ...]
    civil_slot_offset: int
    instructions: DispatchInstructions
    cursor: int


class _ControllerSession:
    """One simulation call's controller state, driven by the dispatch core.

    At each dispatch segment start the core asks :meth:`decides_at`, calls
    :meth:`decide` with the current battery state when a civil day begins,
    takes the segment end from :meth:`prepare_segment`, dispatches, and
    reports the finished segment through :meth:`complete_segment`. Civil
    boundaries come from ``tariff.day_starts`` only.
    """

    def __init__(
        self,
        controller: DailyDispatchController,
        tariff: ResolvedTariff,
        index: pd.DatetimeIndex,
        *,
        hours_per_step: float,
        carry: ControllerCarry | None,
        projection_year: int,
        replay_seam: bool,
        battery_config: Mapping[str, object],
    ) -> None:
        n_steps = len(index)
        utc_ns = _nanoseconds(index.tz_convert("UTC"))
        tariff_ns = _nanoseconds(tariff.index.tz_convert("UTC"))
        if len(tariff_ns) != n_steps or not np.array_equal(tariff_ns, utc_ns):
            raise ValueError("the controller's tariff calendar must be resolved on the simulated index")
        day_starts = tuple(int(position) for position in tariff.day_starts)
        if (
            day_starts[0] != 0
            or day_starts[-1] != n_steps
            or any(left >= right for left, right in zip(day_starts, day_starts[1:], strict=False))
        ):
            raise ValueError("the tariff's civil-day starts must rise from 0 to the simulated length")
        horizon_days = controller.tariff_horizon_days
        if isinstance(horizon_days, bool) or not isinstance(horizon_days, int) or horizon_days < 1:
            raise ValueError("a daily controller's tariff_horizon_days must be a positive integer")

        self._controller = controller
        self._tariff = tariff
        self._timezone = tariff.timezone
        self._n_steps = n_steps
        self._day_starts = day_starts
        self._day_of_start = {start: day for day, start in enumerate(day_starts[:-1])}
        self._n_days = len(day_starts) - 1
        self._utc_ns = utc_ns
        self._step_ns = int(round(hours_per_step * 3600 * 1_000_000_000))
        empty = np.empty(0, dtype=np.int64)
        slots: tuple[np.ndarray, np.ndarray] = _slot_arrays(utc_ns, self._timezone) if n_steps else (empty, empty)
        self._wall, self._fold = slots
        self._horizon_days = horizon_days
        self._projection_year = int(projection_year)
        self._replay_seam = bool(replay_seam)
        self._battery_config = battery_config

        self._labels = np.asarray(tariff.period_labels, dtype=object)
        self._codes = np.asarray(tariff.period_codes, dtype=np.int64)
        self._import = np.asarray(tariff.import_price_per_kwh, dtype=np.float64)
        self._export = np.asarray(tariff.export_price_per_kwh, dtype=np.float64)

        carry = carry or ControllerCarry()
        self._carry_in = carry
        self._base_step = carry.next_project_step_ordinal
        self._policy_state = carry.policy_state
        self._last_complete = carry.last_complete_observed_day
        self._complete_days = carry.complete_days_observed
        self._next_day_ordinal = carry.next_project_day_ordinal
        self._pending: PendingObservedCivilDay | None = None
        self._active: _ActiveDecision | None = None
        self.instructions = _InstructionBuffer(n_steps)
        self._day = -1
        self._segment: tuple[int, int, int] | None = None
        self._seam_head: tuple[SlotKey, ...] | None = None

    # -- calendar ---------------------------------------------------------

    def _slot_keys(self, lo: int, hi: int) -> tuple[SlotKey, ...]:
        return tuple(zip(self._wall[lo:hi].tolist(), self._fold[lo:hi].tolist(), strict=True))

    def _local_date(self, position: int) -> date:
        return pd.Timestamp(int(self._utc_ns[position]), tz="UTC").tz_convert(self._timezone).date()

    def _expected_slot_keys(self, day: int) -> tuple[SlotKey, ...]:
        """Every slot of the civil day on the simulation's step grid."""
        lo, hi = self._day_starts[day], self._day_starts[day + 1]
        if 0 < day < self._n_days - 1:
            # A day with neighbours on both sides is whole in a regular index.
            return self._slot_keys(lo, hi)
        local = self._local_date(lo)
        start_ns = _local_midnight_ns(local, self._timezone)
        stop_ns = _local_midnight_ns(local + timedelta(days=1), self._timezone)
        anchor = int(self._utc_ns[0])
        first = anchor - ((anchor - start_ns) // self._step_ns) * self._step_ns
        wall, fold = _slot_arrays(np.arange(first, stop_ns, self._step_ns, dtype=np.int64), self._timezone)
        return tuple(zip(wall.tolist(), fold.tolist(), strict=True))

    def _head_completes_tail(self) -> bool:
        """Whether the replayed head day completes the final civil day across an A2 seam."""
        if self._seam_head is None:
            head: tuple[SlotKey, ...] = ()
            if self._replay_seam and self._n_days >= 2:
                tail = self._n_days - 1
                present = self._slot_keys(self._day_starts[tail], self._n_steps)
                expected = self._expected_slot_keys(tail)
                remainder = expected[_locate(present, expected) + len(present) :]
                candidate = self._slot_keys(0, self._day_starts[1])
                if remainder and remainder == candidate:
                    head = candidate
            self._seam_head = head
        return bool(self._seam_head)

    def _logical_day(self, start: int) -> tuple[list[range], int | None]:
        """Calendar ranges of the logical day starting at ``start``, and the next day's start.

        A2 replays one calendar, so in a replay the day after the final civil
        day is the head of the same calendar. When the head completes the
        final day, the two are one logical day. A standalone span has no day
        after its end.
        """
        end = self._day_starts[self._day_of_start[start] + 1]
        ranges = [range(start, end)]
        if end < self._n_steps:
            return ranges, end
        if not self._replay_seam:
            return ranges, None
        if self._head_completes_tail():
            ranges.append(range(0, self._day_starts[1]))
            return ranges, self._day_starts[1]
        return ranges, 0

    def _horizon(self, span: list[range], next_start: int | None) -> KnownTariffHorizon:
        days = [span]
        while len(days) < self._horizon_days and next_start is not None:
            ranges, next_start = self._logical_day(next_start)
            days.append(ranges)
        offsets = [0]
        for ranges in days:
            offsets.append(offsets[-1] + sum(len(part) for part in ranges))
        parts = [part for ranges in days for part in ranges]
        positions = np.concatenate([np.arange(part.start, part.stop) for part in parts])
        return KnownTariffHorizon(
            timezone=self._timezone,
            slot_keys=tuple(key for part in parts for key in self._slot_keys(part.start, part.stop)),
            period_labels=tuple(self._labels[positions].tolist()),
            period_codes=tuple(self._codes[positions].tolist()),
            import_price_per_kwh=tuple(self._import[positions].tolist()),
            export_price_per_kwh=tuple(self._export[positions].tolist()),
            civil_day_offsets=tuple(offsets),
            calendar_positions=tuple(positions.tolist()),
            schedule_hash=self._tariff.schedule_hash,
        )

    # -- the core's calls -------------------------------------------------

    def decides_at(self, position: int) -> bool:
        """Whether a civil day begins at ``position``, which is about to be dispatched."""
        return self._day + 1 < self._n_days and position == self._day_starts[self._day + 1]

    def decide(self, position: int, battery_state: ControllerBatteryState) -> None:
        """Begin the civil day at ``position``: resume a carried decision or ask the controller."""
        day = self._day + 1
        lo, hi = self._day_starts[day], self._day_starts[day + 1]
        if position != lo:
            raise RuntimeError(f"civil day {day} begins at step {lo}, not {position}")
        self._day = day
        present = self._slot_keys(lo, hi)
        if day == 0 and self._continue_carried(present):
            return

        # The previous day has ended. A complete day was promoted as it
        # completed; an incomplete one is not forecast history and is dropped.
        self._pending = None
        self._active = None
        expected = self._expected_slot_keys(day)
        offset = _locate(present, expected)
        ordinal = self._next_day_ordinal
        self._next_day_ordinal += 1

        ranges, next_start = self._logical_day(lo)
        span_count = sum(len(part) for part in ranges)
        missing_start = offset > 0
        initial_partial = missing_start and ordinal == 0 and self._replay_seam
        local = self._local_date(lo)
        label = local.isoformat()
        day_input = ControllerDayInput(
            projection_year=self._projection_year,
            project_step_ordinal=self._base_step + lo,
            logical_day_ordinal=ordinal,
            local_date=local,
            timezone=self._timezone,
            expected_slot_keys=expected,
            decision_step_count=span_count,
            segment_offset=0,
            segment_step_count=hi - lo,
            civil_slot_offset=offset,
            initial_partial_day=initial_partial,
            clipped_start=missing_start and not initial_partial,
            clipped_end=offset + span_count < len(expected),
            tariff=self._horizon(ranges, next_start),
            battery_config=self._battery_config,
            battery_state=battery_state,
            last_complete_observed_day=self._last_complete,
            complete_days_observed=self._complete_days,
        )
        decision = self._controller.decide_day(day_input, self._policy_state)
        if not isinstance(decision, ControllerDayDecision):
            raise TypeError("a daily controller must return a ControllerDayDecision")
        instructions = decision.instructions
        if not isinstance(instructions, DispatchInstructions):
            raise TypeError("a daily decision's instructions must be DispatchInstructions")
        if len(instructions) != span_count:
            raise ValueError(f"the daily decision covers {len(instructions)} steps; its decision span has {span_count}")
        self.instructions.bind_scalars(instructions)
        self._policy_state = decision.next_policy_state
        self._active = _ActiveDecision(ordinal, label, expected, offset, instructions, 0)
        self._pending = PendingObservedCivilDay.empty(ordinal, label, self._timezone, expected)

    def _continue_carried(self, present: tuple[SlotKey, ...]) -> bool:
        """Resume a day that began before this call, when the replay continues it."""
        carried, pending = self._carry_in.active_day_decision, self._carry_in.pending_day
        if not self._replay_seam or carried is None or pending is None:
            return False
        if carried.logical_day_ordinal != pending.logical_day_ordinal:
            return False
        start = carried.civil_slot_offset + carried.next_slot_offset
        fits = carried.next_slot_offset + len(present) <= len(carried.instructions)
        if not fits or carried.expected_slot_keys[start : start + len(present)] != present:
            return False
        self.instructions.bind_scalars(carried.instructions)
        self._active = _ActiveDecision(
            carried.logical_day_ordinal,
            carried.local_date,
            carried.expected_slot_keys,
            carried.civil_slot_offset,
            carried.instructions,
            carried.next_slot_offset,
        )
        self._pending = pending
        return True

    def prepare_segment(self, position: int, window_end: int) -> int:
        """Fill the instructions for the segment from ``position``; return its end."""
        active = self._active
        if active is None:
            raise RuntimeError("a dispatch segment began before its civil day was decided")
        end = min(window_end, self._day_starts[self._day + 1])
        count = end - position
        self.instructions.write(position, active.instructions, active.cursor, count)
        self._segment = (position, end, active.civil_slot_offset + active.cursor)
        active.cursor += count
        return end

    def complete_segment(self, pv_dc_w: np.ndarray, load_w: np.ndarray, temperature_c: np.ndarray) -> None:
        """Capture the dispatched segment's actual inputs and promote a completed day."""
        if self._segment is None or self._pending is None:
            raise RuntimeError("no dispatch segment is open")
        lo, hi, slot = self._segment
        self._segment = None
        pending = self._pending.with_segment(
            slot,
            self._slot_keys(lo, hi),
            pv_dc_w[lo:hi].tolist(),
            load_w[lo:hi].tolist(),
            temperature_c[lo:hi].tolist(),
        )
        if pending.complete:
            self._last_complete = pending.promote()
            self._complete_days += 1
            self._pending = None
        else:
            self._pending = pending

    def executed_instructions(self) -> DispatchInstructions | None:
        """A copy of the instructions this call dispatched; see :meth:`_InstructionBuffer.executed`."""
        return self.instructions.executed()

    def finish(self) -> ControllerCarry:
        """The carry the next projection-year call resumes from."""
        carried = None
        active = self._active
        if active is not None and active.cursor < len(active.instructions):
            carried = CarriedDayDecision(
                logical_day_ordinal=active.logical_day_ordinal,
                local_date=active.local_date,
                timezone=self._timezone,
                expected_slot_keys=active.expected_slot_keys,
                civil_slot_offset=active.civil_slot_offset,
                instructions=active.instructions,
                next_slot_offset=active.cursor,
            )
        return ControllerCarry(
            policy_state=self._policy_state,
            last_complete_observed_day=self._last_complete,
            complete_days_observed=self._complete_days,
            pending_day=self._pending,
            active_day_decision=carried,
            next_project_step_ordinal=self._base_step + self._n_steps,
            next_project_day_ordinal=self._next_day_ordinal,
        )
