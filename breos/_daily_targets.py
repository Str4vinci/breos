"""One grid-charge target per day, chosen by a gridded dynamic program.

This is private groundwork for daily-persistence smart charging (plan step
7), not public API. It ports the daily-target dynamic program of the legacy
research tool (``tools/compute_a2_daily_sc_oracle.py`` at ``07dc0e40``,
``solve_daily_sc_schedule``). The policy class is the fixed-target
instruction layout with one change: every charge step of a day takes that
day's target, chosen from a grid of usable fractions. The discharge gate, the
reserve, the grid-charge efficiency and the import limit stay as given.

The state is stored energy at a day boundary and the stage cost is the day's
import cost less its export revenue. Standing charges are left out: they are
the same for every schedule and cannot move the argmin. Each transition is one
day of the production dispatch step, :func:`breos._dispatch._dispatch_day`,
run on a scratch buffer by the selected backend. There is no second battery
model to drift from the first.

The value function lives on a uniform grid of stored energy and is read back
by linear interpolation. The forward pass carries the exact, unsnapped
energy, so the grid only limits how close to optimal the schedule is. A
schedule's cost is what production physics gives when the schedule is
replayed, not the planner's own number: the planner holds state of health and
the efficiencies fixed over its whole window, as the legacy tool held them
over a project year.

Days are civil days (ADR 0002 A1), given as the resolved tariff's
``day_starts``. The legacy tool used positional ``steps_per_day`` windows.
Production still advances health on positional degradation windows, so a
closed-loop controller that plans civil days will run a degradation window
in two ``_dispatch_day`` calls when a civil day starts inside it. Carrying
the stored energy and both origins from the first call into the second gives
the same energy as one call. The day-close replacement path must then take
``battery_energy_beginning`` from the call that holds the window's last step.

Two choices depart from the legacy tool on purpose. The terminal refill is
priced at the cheapest step that may grid-charge, not the cheapest step of
the window: energy cannot be bought back in a period the instructions never
charge in, and a cheaper period elsewhere would undervalue what the battery
holds at the end and push a rolling controller to drain it. The refill
target is the max-SOC energy at the last step's temperature, the most the
battery can then hold, rather than at the reference capacity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Sequence

import numpy as np

from breos._dispatch import L_PV_AC_EXPORT, R_GRID_IMPORT, lfp_capacity_factor
from breos.battery import (
    BatteryConfig,
    _dispatch_no_battery_vectorized,
    _resolve_dispatch_day,
    _ResultBuffers,
    _step_energy_cap,
)
from breos.dispatch_instructions import DispatchInstructions
from breos.execution import is_pv_only_dispatch
from breos.tariffs import ResolvedTariff
from breos.utils import get_hours_per_step

# The closed-loop controller replans each day over this many days.
DEFAULT_HORIZON_DAYS = 2
# 0.0, 0.1, ..., 1.0 of the usable window.
DEFAULT_TARGET_LEVELS = 11
DEFAULT_SOC_STATES = 21
# A later level must beat the best so far by more than this to replace it.
_TIE_TOLERANCE = 1e-12


def target_grid(target_levels: int | Sequence[float]) -> np.ndarray:
    """The candidate targets, as usable fractions in increasing order.

    An integer ``n`` spaces ``n`` levels evenly from 0 to 1; one level is 0.
    A sequence is taken as the levels themselves. The order matters: ties go
    to the lowest target, so grid energy that does not pay for itself is not
    bought.
    """
    if isinstance(target_levels, Integral) and not isinstance(target_levels, bool):
        if target_levels < 1:
            raise ValueError("'target_levels' must be at least 1")
        return np.linspace(0.0, 1.0, int(target_levels))
    levels = np.array(target_levels, dtype=np.float64)
    if levels.ndim != 1 or len(levels) == 0:
        raise ValueError("'target_levels' must be a positive integer or a non-empty sequence of fractions")
    if not np.isfinite(levels).all() or ((levels < 0.0) | (levels > 1.0)).any():
        raise ValueError("'target_levels' must be fractions between 0 and 1")
    if (np.diff(levels) <= 0.0).any():
        raise ValueError("'target_levels' must be strictly increasing")
    return levels


def daily_target_instructions(
    instructions: DispatchInstructions, day_starts: Sequence[int], targets: Sequence[float] | np.ndarray
) -> DispatchInstructions:
    """``instructions`` with every charge step of day ``d`` targeting ``targets[d]``.

    A charge step is one with a finite grid target; the other steps and every
    other field are kept. ``day_starts`` is the first step of every day
    followed by the step count, as :attr:`ResolvedTariff.day_starts`. A NaN
    target leaves its day with no grid charge.
    """
    starts = _checked_day_starts(day_starts, len(instructions))
    values = np.asarray(targets, dtype=np.float64)
    n_days = len(starts) - 1
    if values.shape != (n_days,):
        raise ValueError(f"'targets' must hold one target per day ({n_days}), got shape {values.shape}")
    per_step = np.repeat(values, np.diff(starts))
    base = instructions.grid_target_fraction
    return DispatchInstructions(
        discharge_allowed=instructions.discharge_allowed,
        reserve_fraction=instructions.reserve_fraction,
        grid_target_fraction=np.where(np.isnan(base), np.nan, per_step),
        grid_charge_efficiency=instructions.grid_charge_efficiency,
        grid_import_limit_w=instructions.grid_import_limit_w,
    )


def _checked_day_starts(day_starts: Sequence[int], n_steps: int) -> np.ndarray:
    starts = np.asarray(day_starts)
    if starts.ndim != 1 or len(starts) < 1 or not np.issubdtype(starts.dtype, np.integer):
        raise ValueError("'day_starts' must be a 1-D sequence of step positions")
    if starts[0] != 0 or starts[-1] != n_steps or (np.diff(starts) <= 0).any():
        raise ValueError(
            f"'day_starts' must rise strictly from 0 to the step count ({n_steps}), as ResolvedTariff.day_starts does"
        )
    return starts.astype(np.int64)


def _slice(instructions: DispatchInstructions, lo: int, hi: int) -> DispatchInstructions:
    return DispatchInstructions(
        discharge_allowed=instructions.discharge_allowed[lo:hi],
        reserve_fraction=instructions.reserve_fraction[lo:hi],
        grid_target_fraction=instructions.grid_target_fraction[lo:hi],
        grid_charge_efficiency=instructions.grid_charge_efficiency,
        grid_import_limit_w=instructions.grid_import_limit_w,
    )


@dataclass(frozen=True)
class DailyTargetProblem:
    """What the planner sees over its window, one entry per step.

    ``pv_dc_w``, ``load_w`` and ``temperature_c`` are what the planner
    believes: truth for an offline oracle, a forecast for a controller. The
    tariff prices and ``day_starts`` are known. ``instructions`` is the
    fixed-target layout the targets are placed on. ``soh_fraction`` and the
    two efficiencies are the battery's state at the start of the window and
    hold for all of it; None takes the configured values.
    """

    pv_dc_w: np.ndarray
    load_w: np.ndarray
    temperature_c: np.ndarray
    import_price_per_kwh: np.ndarray
    export_price_per_kwh: np.ndarray
    day_starts: tuple[int, ...]
    instructions: DispatchInstructions
    battery_config: BatteryConfig
    hours_per_step: float
    soh_fraction: float | None = None
    eff_charge: float | None = None
    eff_discharge: float | None = None

    def __post_init__(self) -> None:
        n = len(self.instructions)
        for name in ("pv_dc_w", "load_w", "temperature_c", "import_price_per_kwh", "export_price_per_kwh"):
            array = np.array(getattr(self, name), dtype=np.float64)
            if array.shape != (n,):
                raise ValueError(f"'{name}' must hold one value per instruction step ({n}), got shape {array.shape}")
            if not np.isfinite(array).all():
                raise ValueError(f"'{name}' must be finite")
            # Read-only here; the planner dispatches on writable copies, so
            # the compiled kernel keeps the one signature production uses.
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        object.__setattr__(self, "day_starts", tuple(int(s) for s in _checked_day_starts(self.day_starts, n)))
        if not (math.isfinite(self.hours_per_step) and self.hours_per_step > 0.0):
            raise ValueError("'hours_per_step' must be finite and positive")

    @classmethod
    def from_tariff(
        cls,
        tariff: ResolvedTariff,
        instructions: DispatchInstructions,
        battery_config: BatteryConfig,
        *,
        pv_dc_w: Any,
        load_w: Any,
        temperature_c: Any,
        freq: str,
        **state: float,
    ) -> DailyTargetProblem:
        """A problem on ``tariff``'s calendar: its prices and civil days."""
        return cls(
            pv_dc_w=pv_dc_w,
            load_w=load_w,
            temperature_c=temperature_c,
            import_price_per_kwh=np.asarray(tariff.import_price_per_kwh),
            export_price_per_kwh=np.asarray(tariff.export_price_per_kwh),
            day_starts=tariff.day_starts,
            instructions=instructions,
            battery_config=battery_config,
            hours_per_step=get_hours_per_step(freq),
            **state,
        )

    @property
    def n_days(self) -> int:
        return len(self.day_starts) - 1

    def health(self) -> tuple[float, float, float]:
        """``(soh_fraction, eff_charge, eff_discharge)``, the configured value for each one left None."""
        config = self.battery_config
        return (
            config.initial_soh / 100.0 if self.soh_fraction is None else float(self.soh_fraction),
            config.charge_efficiency if self.eff_charge is None else float(self.eff_charge),
            config.discharge_efficiency if self.eff_discharge is None else float(self.eff_discharge),
        )

    def horizon(self, first_day: int, horizon_days: int = DEFAULT_HORIZON_DAYS) -> DailyTargetProblem:
        """The same problem over days ``[first_day, first_day + horizon_days)``, cut at the last day.

        A controller then swaps in its forecast for the believed PV, load and
        temperature, with :func:`dataclasses.replace`.
        """
        if not 0 <= first_day < self.n_days:
            raise ValueError(f"'first_day' must be a day of the problem (0 to {self.n_days - 1})")
        if horizon_days < 1:
            raise ValueError("'horizon_days' must be at least 1")
        last = min(self.n_days, first_day + horizon_days)
        lo, hi = self.day_starts[first_day], self.day_starts[last]
        return DailyTargetProblem(
            pv_dc_w=self.pv_dc_w[lo:hi],
            load_w=self.load_w[lo:hi],
            temperature_c=self.temperature_c[lo:hi],
            import_price_per_kwh=self.import_price_per_kwh[lo:hi],
            export_price_per_kwh=self.export_price_per_kwh[lo:hi],
            day_starts=tuple(start - lo for start in self.day_starts[first_day : last + 1]),
            instructions=_slice(self.instructions, lo, hi),
            battery_config=self.battery_config,
            hours_per_step=self.hours_per_step,
            soh_fraction=self.soh_fraction,
            eff_charge=self.eff_charge,
            eff_discharge=self.eff_discharge,
        )


@dataclass(frozen=True)
class DailyTargetPlan:
    """A planned schedule and what the planner expects it to cost.

    ``objective`` is ``stage_cost`` plus ``terminal_cost``, in the tariff's
    currency. ``stage_cost`` is import cost less export revenue over the
    window; ``terminal_cost`` buys back, at the cheapest import price of a
    step that may grid-charge, the stored energy that ends below the
    terminal target.
    """

    targets: np.ndarray
    objective: float
    stage_cost: float
    terminal_cost: float
    start_energy_wh: float
    end_energy_wh: float
    soc_grid_wh: np.ndarray = field(repr=False)


class _DayEvaluator:
    """Run one day of production dispatch on a scratch buffer and price it."""

    def __init__(self, problem: DailyTargetProblem, levels: np.ndarray, execution_backend: str) -> None:
        config = problem.battery_config
        hours = problem.hours_per_step
        self.problem = problem
        self.buffers = _ResultBuffers(len(problem.instructions))
        self.series = (np.array(problem.pv_dc_w), np.array(problem.load_w), np.array(problem.temperature_c))
        self.dispatch_day = _resolve_dispatch_day(execution_backend)
        self.level_instructions = [
            daily_target_instructions(problem.instructions, problem.day_starts, np.full(problem.n_days, level))
            for level in levels
        ]
        soh, eff_charge, eff_discharge = problem.health()
        self.state = {
            "battery_config": config,
            "battery_soh_decimal": soh,
            # Origins never feed a dispatch decision, so they cannot move a cost.
            "Battery_PV_Origin_Energy_Wh": 0.0,
            "Battery_Grid_Origin_Energy_Wh": 0.0,
            "eff_charge": eff_charge,
            "eff_discharge": eff_discharge,
            "hours_per_step": hours,
            "standby_loss_per_step_wh": config.standby_loss_wh * hours,
            "cap_wh": _step_energy_cap(config.inverter_ac_capacity_w, hours),
            "cap_charge_wh": _step_energy_cap(config.max_charge_power_w, hours),
            "cap_discharge_wh": _step_energy_cap(config.max_discharge_power_w, hours),
            "cap_stored_wh": _step_energy_cap(config.stored_power_limit_w, hours),
        }

    def run(self, day: int, energy_wh: float, level: int) -> tuple[float, float]:
        """``(cost, end_energy_wh)`` of ``day`` from ``energy_wh`` at target level ``level``."""
        problem = self.problem
        lo, hi = problem.day_starts[day], problem.day_starts[day + 1]
        self.dispatch_day(
            self.buffers,
            *self.series,
            lo,
            hi,
            Battery_Energy_Wh=energy_wh,
            instructions=self.level_instructions[level],
            **self.state,
        )
        end_energy = self.buffers.columns["Battery_Energy"][hi - 1]
        return _window_cost(self.buffers.matrix, problem, lo, hi), float(end_energy)


def _window_cost(matrix: np.ndarray, problem: DailyTargetProblem, lo: int, hi: int) -> float:
    """Import cost less export revenue of steps ``[lo, hi)`` as ``matrix`` holds them."""
    money_w = float(
        matrix[R_GRID_IMPORT, lo:hi] @ problem.import_price_per_kwh[lo:hi]
        - matrix[L_PV_AC_EXPORT, lo:hi] @ problem.export_price_per_kwh[lo:hi]
    )
    return money_w * problem.hours_per_step / 1000.0


def solve_daily_targets(
    problem: DailyTargetProblem,
    *,
    initial_energy_wh: float | None = None,
    target_levels: int | Sequence[float] = DEFAULT_TARGET_LEVELS,
    soc_states: int = DEFAULT_SOC_STATES,
    terminal_energy_wh: float | None = None,
    free_terminal: bool = False,
    execution_backend: str = "python",
) -> DailyTargetPlan:
    """Pick one charge target per day of ``problem`` that minimises its cost.

    ``initial_energy_wh`` defaults to a full battery at the configured max
    SOC, as a fresh simulation starts. Unless ``free_terminal``, energy that
    ends the window below ``terminal_energy_wh`` is bought back at the
    cheapest import price of a step with a grid target (of any step if none
    has one), through both charge efficiencies. The default target is the
    max-SOC energy at the last step's temperature. Without the refill the
    plan would drain the battery on the last day, a gain the next window
    pays for.

    With no battery, or a single level, there is nothing to choose: the days
    are chained at the lowest level.
    """
    levels = target_grid(target_levels)
    if isinstance(soc_states, bool) or not isinstance(soc_states, Integral) or soc_states < 2:
        raise ValueError("'soc_states' must be an integer of at least 2")
    config = problem.battery_config
    soh, eff_charge, _eff_discharge = problem.health()
    usable_wh = config.nominal_energy_wh * soh
    temperatures = problem.temperature_c
    start = usable_wh * config.max_soc if initial_energy_wh is None else float(initial_energy_wh)
    end_capacity = lfp_capacity_factor(float(temperatures[-1])) if len(temperatures) else 1.0
    target = usable_wh * config.max_soc * end_capacity if terminal_energy_wh is None else float(terminal_energy_wh)
    if not (math.isfinite(start) and math.isfinite(target)):
        raise ValueError("'initial_energy_wh' and 'terminal_energy_wh' must be finite")

    # Unrounded, unlike the legacy tool: the grid must span the window the
    # dispatch computes from these same temperatures.
    factors = [lfp_capacity_factor(float(t)) for t in np.unique(temperatures)]
    soc_grid = np.linspace(
        config.min_soc * usable_wh * min(factors, default=1.0),
        config.max_soc * usable_wh * max(factors, default=1.0),
        int(soc_states),
    )
    chargeable = ~np.isnan(problem.instructions.grid_target_fraction)
    refill_prices = problem.import_price_per_kwh[chargeable] if chargeable.any() else problem.import_price_per_kwh
    refill_per_wh = (
        float(refill_prices.min(initial=math.inf)) / 1000.0 / (problem.instructions.grid_charge_efficiency * eff_charge)
    )

    def terminal_cost(end_wh: Any) -> Any:
        shortfall = np.maximum(0.0, target - np.asarray(end_wh, dtype=np.float64))
        return np.zeros_like(shortfall) if free_terminal else shortfall * refill_per_wh

    n_days = problem.n_days
    if is_pv_only_dispatch(config.nominal_energy_wh, config.max_soc, config.min_soc):
        # The PV-only path never reads the instructions, so no level differs.
        buffers = _ResultBuffers(len(problem.instructions))
        _dispatch_no_battery_vectorized(
            buffers,
            problem.pv_dc_w,
            problem.load_w,
            problem.temperature_c,
            battery_config=config,
            hours_per_step=problem.hours_per_step,
            cap_wh=_step_energy_cap(config.inverter_ac_capacity_w, problem.hours_per_step),
        )
        stage = sum(
            _window_cost(buffers.matrix, problem, problem.day_starts[day], problem.day_starts[day + 1])
            for day in range(n_days)
        )
        return _plan(np.full(n_days, levels[0]), float(stage), 0.0, 0.0, 0.0, soc_grid)

    evaluator = _DayEvaluator(problem, levels, execution_backend)
    if len(levels) == 1:
        energy, stage = start, 0.0
        for day in range(n_days):
            cost, energy = evaluator.run(day, energy, 0)
            stage += cost
        return _plan(np.full(n_days, levels[0]), stage, float(terminal_cost(energy)), start, energy, soc_grid)

    # Backward pass: the cheapest cost to go from each grid energy at each day boundary.
    values = np.empty((n_days + 1, len(soc_grid)))
    values[n_days] = terminal_cost(soc_grid)
    for day in range(n_days - 1, -1, -1):
        following = values[day + 1]
        for state, energy_wh in enumerate(soc_grid):
            best = math.inf
            for level in range(len(levels)):
                cost, end_wh = evaluator.run(day, float(energy_wh), level)
                best = min(best, cost + float(np.interp(end_wh, soc_grid, following)))
            values[day, state] = best

    # Forward pass from the exact energy; the grid only prices what follows.
    schedule = np.empty(n_days)
    energy, stage = start, 0.0
    for day in range(n_days):
        following = values[day + 1]
        best_total, best_level, best_cost, best_end = math.inf, 0, 0.0, energy
        for level in range(len(levels)):
            cost, end_wh = evaluator.run(day, energy, level)
            total = cost + float(np.interp(end_wh, soc_grid, following))
            # Levels rise, and a strict margin keeps a tie on the lower one.
            if total < best_total - _TIE_TOLERANCE:
                best_total, best_level, best_cost, best_end = total, level, cost, end_wh
        schedule[day] = levels[best_level]
        stage += best_cost
        energy = best_end
    return _plan(schedule, stage, float(terminal_cost(energy)), start, energy, soc_grid)


def _plan(
    targets: np.ndarray, stage: float, terminal: float, start_wh: float, end_wh: float, soc_grid: np.ndarray
) -> DailyTargetPlan:
    return DailyTargetPlan(
        targets=targets,
        objective=float(stage + terminal),
        stage_cost=float(stage),
        terminal_cost=terminal,
        start_energy_wh=float(start_wh),
        end_energy_wh=float(end_wh),
        soc_grid_wh=soc_grid,
    )
