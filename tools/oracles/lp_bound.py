"""A perfect-foresight lower bound on one year's electricity bill, by linear programming.

The linear program sees the whole first project year at once: PV, load,
battery temperature and the tariff's per-step prices. It chooses every flow
of every step (PV to the inverter or the battery, AC to the load or the
grid, battery discharge, grid charge and stored energy) to minimise import
cost less export revenue. It is solved with HiGHS through
:func:`scipy.optimize.linprog`. Its feasible set contains every flow the
production dispatch step can deliver, under any
:class:`~breos.dispatch_instructions.DispatchInstructions`. Its optimum is
therefore a cost that no controller can beat, causal or not, on the same
year at the same health. The standing charge is left out: every schedule
pays it.

This is the BREOS counterpart of the legacy perfect-foresight bound
(``tools/compute_a2_perfect_foresight_bound.py`` and
``tools/a2_lp_replay.py`` at ``07dc0e40``). It is a validation tool, not
public API.

What the program keeps exactly, per step:

- PV DC, AC load and the battery temperature the dispatch reads, and the
  tariff's import and export prices.
- The charge and discharge efficiencies, the grid-charge (AC-to-DC)
  efficiency, the charge-input, discharge-AC and C-rate stored-energy
  limits, and the inverter AC rating, with ``ac_output_scale``. Grid charge
  shares the AC rating with the PV output, as it does in the dispatch.
- The usable window's ceiling, ``max_soc`` of the capacity at the step's
  temperature (A7), and the start at max SOC.

What it relaxes, so that the bound claim holds:

- **Fixed health.** State of health and both efficiencies are held at the
  year's opening values (``soh_fraction``, the configured efficiencies).
  Health only falls within a year, which lowers the dispatch's ceiling and
  so keeps the bound valid. It also lowers the floor, by ``min_soc`` times
  the fade, which is the one way fixed health favours the dispatch. The
  floor is therefore taken at ``floor_soh_fraction``, the same value by
  default; the bound holds for any dispatch whose health stays at or above
  it. The report gives the lowest health each replay reached and whether
  the floor covers it (``floor_soh_covers_run``); ``--floor-soh`` sets it.
- **The inverter curve is convexified.** The PVWatts part-load curve the
  dispatch uses is not concave near zero load. The program bounds AC output
  from above by a concave function over it: a line from the origin at the
  curve's peak AC/DC ratio (about 1.003 times the nominal efficiency),
  tangents to the curve between that peak and the rated point, AC no
  greater than DC, and the AC rating (:func:`inverter_hull_cuts`). Each
  step then tightens this with its load (:func:`load_secants`): the
  inverter delivers at most the load unless PV serves all of it and
  exports, and below the peak point no output up to the load converts
  better than the load itself. What remains optimistic is a step that
  delivers less than its load below the peak point, and exported AC, which
  are credited up to the conversion at the load or the peak ratio. The same
  bounds apply to the PV and the battery share of the shared operating
  point on their own, so AC is not credited to the source that did not
  make it. DC above the rated point is clipped, as in the dispatch, and the
  clipped DC may still charge the battery.
- **Standby loss is charged in part.** The dispatch bleeds
  ``min(standby, energy - emin)`` each step after the window clip. The
  energy that leaves is not a convex function of the energy stored, so the
  program charges the chord of it over the window: the full standby loss
  at the ceiling, none at the floor, and a share in between. A spill
  variable absorbs the capacity-window loss when a colder step shrinks the
  window, and any standby the chord does not charge.
- **A lower floor after a cold spell.** The dispatch never discharges below
  the step's floor, but stored energy can sit below a floor that rises with
  temperature. The program's floor is the running minimum of the floor
  over the steps so far.
- **No dispatch order.** The step serves the load from PV first, charges
  from surplus PV before it exports, never grid-charges while it exports or
  discharges, and either charges or discharges (ADR 0002 A8). The program
  keeps none of these rules. It may charge and discharge in one step,
  discharge while PV exports (so stored energy in effect reaches the grid),
  and curtail PV. The site import limit bounds grid charge alone: its exact
  form, grid charge only in the headroom the load's import leaves, is not
  convex.
- **Free terminal energy.** The year may end with any stored energy. A
  cyclic end state, the legacy default, is not a bound on one year's bill:
  a controller may end the year lower than it started.

Grid charging is available with the ``[smart_charging]`` table's
grid-charge efficiency and site limit. Without a fixed-target table the
program has no grid-charge converter to model and bounds self-consumption
dispatch only, unless a grid-charge efficiency is given.

The bound is the optimum within the solver's tolerances. The schedule can be
turned into instructions (:func:`lp_instructions`) and replayed through the
production run with :mod:`tools.oracles.replay`, which shows the gap between
what the relaxed physics promised and what production delivers. Legacy
options not ported: the per-project-year and year-one-reused LP modes, the
wear price, the terminal energy value and the lifetime NPV sums; this tool
bounds the first year's bill.

Usage:
    python tools/oracles/lp_bound.py --config my.toml --output bound.json --csv schedule.csv
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from breos._dispatch import (
    PVWATTS_CURVE_CONSTANT,
    PVWATTS_CURVE_LINEAR,
    PVWATTS_CURVE_QUADRATIC,
    PVWATTS_REFERENCE_EFFICIENCY,
    _dc_ac,
    _dc_for_ac,
    lfp_capacity_factor,
)
from breos.app_inputs import reuse_prepared_inputs
from breos.battery import BatteryConfig, _step_energy_cap
from breos.dispatch_instructions import DispatchInstructions
from breos.tariffs import result_currency
from breos.utils import get_hours_per_step
from tools.oracles._output import load_config, write_csv, write_json
from tools.oracles.replay import (
    ReplayCase,
    ReplayResult,
    Tolerance,
    plan_comparison,
    prepare_replay,
    reference_dispatch,
    replay_instructions,
    replay_summary,
)

LP_BOUND_SCHEMA = "breos_lp_bound_v1"
LP_SCHEDULE_SCHEMA = "breos_lp_bound_schedule_v1"
DEFAULT_TANGENTS = 5
# A planned flow below this (Wh over the step) is solver noise, not an action.
ACTION_THRESHOLD_WH = 1e-6
# The replay compares the schedule's flows with this tolerance.
DEFAULT_REPLAY_TOLERANCE = Tolerance(atol_wh=1.0)

RELAXATIONS = (
    "fixed_health",
    "inverter_concave_hull",
    "partial_standby_loss",
    "running_minimum_floor",
    "no_dispatch_order",
    "free_terminal_energy",
)

# The schedule's flows, Wh over each step, in variable-block order.
SCHEDULE_FLOWS = (
    "pv_dc_to_inverter_wh",
    "pv_dc_to_battery_wh",
    "pv_ac_to_load_wh",
    "pv_ac_export_wh",
    "battery_ac_to_load_wh",
    "battery_discharge_stored_wh",
    "grid_ac_to_battery_wh",
    "grid_import_to_load_wh",
    "stored_energy_spill_wh",
    "standby_loss_wh",
    "battery_energy_end_wh",
)
_PI, _PB, _LP, _EX, _BA, _DS, _GC, _GL, _SP, _SB, _EN = range(len(SCHEDULE_FLOWS))


def inverter_hull_cuts(
    inverter_efficiency: float,
    inverter_ac_wh: float,
    ac_output_scale: float = 1.0,
    tangents: int = DEFAULT_TANGENTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Lines ``ac <= intercept + slope * dc`` whose minimum bounds the dispatch's conversion from above.

    The dispatch converts DC (Wh over a step) to AC along the PVWatts
    part-load curve, capped at DC and at the AC rating ``inverter_ac_wh``,
    and scales the result by ``ac_output_scale``. Near zero load the curve
    falls below zero and is clipped there, so it is not concave. The lines
    are the line from the origin at the curve's peak AC/DC ratio, which
    touches the curve at ``sqrt(c / q)`` of the rated DC, ``tangents - 1``
    more tangents spaced evenly from there to the rated point, and
    ``ac <= dc``, all times the scale. Each lies on or above the curve
    everywhere, since the curve is concave where it is positive. The AC
    rating is a separate bound. Without a rating the conversion is linear
    and so is the one line.

    Returns ``(intercepts_wh, slopes)``.
    """
    eta = min(1.0, max(0.0, float(inverter_efficiency)))
    scale = min(1.0, max(0.0, float(ac_output_scale)))
    if not math.isfinite(inverter_ac_wh):
        return np.zeros(1), np.array([eta * scale])
    if tangents < 1:
        raise ValueError("'tangents' must be at least 1")
    pdc0 = inverter_ac_wh / eta
    gain = eta / PVWATTS_REFERENCE_EFFICIENCY * scale
    q, l, c = PVWATTS_CURVE_QUADRATIC, PVWATTS_CURVE_LINEAR, PVWATTS_CURVE_CONSTANT
    peak = math.sqrt(c / q)
    points = np.linspace(peak, 1.0, tangents)
    # The tangent at z: g(z) + g'(z) (x - z pdc0), with g = gain pdc0 (q z^2 + l z + c).
    intercepts = gain * pdc0 * (c - q * points**2)
    slopes = gain * (2.0 * q * points + l)
    # The first is the peak-ratio line; set its intercept to the zero it is analytically.
    intercepts[0] = 0.0
    return np.append(intercepts, 0.0), np.append(slopes, scale)


def load_secants(
    load_wh: np.ndarray, inverter_efficiency: float, inverter_ac_wh: float, ac_output_scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Per-step ``(dc_per_ac, dc_per_export)``: ``dc >= dc_per_ac * ac_to_load + dc_per_export * export``.

    The dispatch's inverter delivers at most the step's load unless PV
    serves the whole load and exports the rest. Below the operating point
    of the peak AC/DC ratio, DC per unit of AC only falls as output rises,
    so no output up to a load ``L`` below that point converts better than
    ``L`` itself: ``dc_per_ac`` is DC per AC at ``L``. Exported AC past
    ``L`` costs at least the marginal DC per AC there, since the DC-for-AC
    curve is convex. A step with no load, or a load at or above the peak
    point, takes the peak ratio and no export term, which the hull cuts
    already hold. Both factors are shaded by 1e-9 so round-off in the
    dispatch's own conversion cannot cut off a point it delivers.
    """
    eta = min(1.0, max(0.0, float(inverter_efficiency)))
    scale = min(1.0, max(0.0, float(ac_output_scale)))
    n = len(load_wh)
    if not math.isfinite(inverter_ac_wh) or eta <= 0.0 or scale <= 0.0:
        return np.full(n, 1.0 / max(eta * scale, 1e-300)), np.zeros(n)
    pdc0 = inverter_ac_wh / eta
    q, l, c = PVWATTS_CURVE_QUADRATIC, PVWATTS_CURVE_LINEAR, PVWATTS_CURVE_CONSTANT
    peak_dc = math.sqrt(c / q) * pdc0
    peak_ac, _loss, _clip = _dc_ac(peak_dc, inverter_ac_wh, eta, scale, 2.0)
    shade = 1.0 - 1e-9
    dc_per_ac = np.full(n, peak_dc / peak_ac * shade)
    dc_per_export = np.zeros(n)
    for step in np.flatnonzero((load_wh > 0.0) & (load_wh < peak_ac)):
        load = float(load_wh[step])
        dc = _dc_for_ac(load, inverter_ac_wh, eta, scale)
        slope = scale * eta / PVWATTS_REFERENCE_EFFICIENCY * (2.0 * q * dc / pdc0 + l)
        dc_per_ac[step] = dc / load * shade
        dc_per_export[step] = shade / max(slope, scale)
    return dc_per_ac, dc_per_export


@dataclass(frozen=True)
class LpBoundProblem:
    """One year's per-step inputs and the battery the program plans for.

    ``pv_dc_w``, ``load_w`` and ``temperature_c`` are the dispatch's own
    per-step inputs, and the prices are per kWh. ``soh_fraction`` sets the
    ceiling and the start energy and ``floor_soh_fraction`` the floor; both
    default to the battery's ``initial_soh``. ``grid_charge_efficiency`` is
    the AC-to-DC conversion a grid charge passes through; None forbids grid
    charge. ``grid_import_limit_w`` bounds grid charge; math.inf is no
    limit. ``initial_energy_wh`` defaults to a full battery at max SOC, as a
    fresh simulation starts.
    """

    pv_dc_w: np.ndarray
    load_w: np.ndarray
    temperature_c: np.ndarray
    import_price_per_kwh: np.ndarray
    export_price_per_kwh: np.ndarray
    battery_config: BatteryConfig
    hours_per_step: float
    grid_charge_efficiency: float | None = None
    grid_import_limit_w: float = math.inf
    soh_fraction: float | None = None
    floor_soh_fraction: float | None = None
    initial_energy_wh: float | None = None
    tangents: int = DEFAULT_TANGENTS

    def __post_init__(self) -> None:
        n = len(np.asarray(self.pv_dc_w))
        for name in ("pv_dc_w", "load_w", "temperature_c", "import_price_per_kwh", "export_price_per_kwh"):
            array = np.array(getattr(self, name), dtype=np.float64)
            if array.shape != (n,) or not np.isfinite(array).all():
                raise ValueError(f"'{name}' must hold {n} finite values")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if (self.load_w < 0.0).any():
            raise ValueError("'load_w' must not be negative")
        if not (math.isfinite(self.hours_per_step) and self.hours_per_step > 0.0):
            raise ValueError("'hours_per_step' must be finite and positive")
        efficiency = self.grid_charge_efficiency
        if efficiency is not None and not (math.isfinite(efficiency) and 0.0 < efficiency <= 1.0):
            raise ValueError("'grid_charge_efficiency' must be between 0 (exclusive) and 1 (inclusive), or None")
        if math.isnan(self.grid_import_limit_w) or self.grid_import_limit_w <= 0.0:
            raise ValueError("'grid_import_limit_w' must be greater than 0, or math.inf for no limit")
        soh, floor = self.health()
        if not (0.0 < floor <= soh <= 1.0):
            raise ValueError("health must satisfy 0 < floor_soh_fraction <= soh_fraction <= 1")

    @classmethod
    def from_case(cls, case: ReplayCase, **overrides: Any) -> LpBoundProblem:
        """The first project year of a prepared replay case.

        Grid charge takes the ``[smart_charging]`` table's grid-charge
        efficiency and site limit, when the table sets them. ``overrides``
        replaces any field.
        """
        aligned = case.aligned_inputs()
        spec = case.resolved.smart_charging
        settings: dict[str, Any] = {}
        if spec is not None and spec.grid_charge_efficiency is not None:
            settings["grid_charge_efficiency"] = spec.grid_charge_efficiency
            if spec.grid_import_limit_w is not None:
                settings["grid_import_limit_w"] = spec.grid_import_limit_w
        return cls(
            pv_dc_w=aligned.pv_dc_w,
            load_w=aligned.load_w,
            temperature_c=aligned.temperature_c,
            import_price_per_kwh=np.asarray(case.tariff.import_price_per_kwh),
            export_price_per_kwh=np.asarray(case.tariff.export_price_per_kwh),
            battery_config=case.battery_config(),
            hours_per_step=get_hours_per_step(case.cfg["resolution"]),
            **{**settings, **overrides},
        )

    def __len__(self) -> int:
        return len(self.pv_dc_w)

    def health(self) -> tuple[float, float]:
        """``(soh_fraction, floor_soh_fraction)``, each defaulting to the battery's ``initial_soh``."""
        soh = self.battery_config.initial_soh / 100.0 if self.soh_fraction is None else float(self.soh_fraction)
        floor = soh if self.floor_soh_fraction is None else float(self.floor_soh_fraction)
        return soh, floor

    def capacity_factors(self) -> np.ndarray:
        """The usable-capacity factor at each step's temperature, as the dispatch computes it."""
        unique, inverse = np.unique(self.temperature_c, return_inverse=True)
        return np.array([lfp_capacity_factor(float(t)) for t in unique])[inverse]

    def window(self) -> tuple[np.ndarray, np.ndarray]:
        """``(emin_wh, emax_wh)`` per step: the dispatch's usable window at ``soh_fraction``."""
        config = self.battery_config
        usable = config.nominal_energy_wh * self.health()[0] * self.capacity_factors()
        return usable * config.min_soc, usable * config.max_soc

    def start_energy_wh(self) -> float:
        config = self.battery_config
        if self.initial_energy_wh is not None:
            return float(self.initial_energy_wh)
        return config.nominal_energy_wh * self.health()[0] * config.max_soc

    def floor_wh(self) -> np.ndarray:
        """The program's floor: the running minimum of the floor at ``floor_soh_fraction`` and the start."""
        config = self.battery_config
        floor = config.nominal_energy_wh * self.health()[1] * config.min_soc * self.capacity_factors()
        return np.minimum(np.minimum.accumulate(floor), self.start_energy_wh())


@dataclass(frozen=True)
class LpBound:
    """The program's optimum: the bound, and the schedule that attains it.

    ``objective`` is import cost less export revenue over the problem, in
    the tariff's currency; no dispatch on the same inputs and health pays
    less. ``schedule`` maps each name in :data:`SCHEDULE_FLOWS` to its Wh
    per step, with solver round-off below zero set to zero.
    """

    objective: float
    import_cost: float
    export_revenue: float
    schedule: dict[str, np.ndarray]
    start_energy_wh: float
    solve_seconds: float
    n_variables: int
    n_constraints: int
    solver_message: str


@dataclass(frozen=True)
class LinearProgram:
    """``min cost @ x`` subject to ``a_ub @ x <= b_ub``, ``a_eq @ x == b_eq`` and ``lower <= x <= upper``.

    ``x`` holds one block of ``n_steps`` values per name in
    :data:`SCHEDULE_FLOWS`, in that order, each in Wh over the step.
    """

    cost: np.ndarray
    a_ub: sparse.csr_matrix
    b_ub: np.ndarray
    a_eq: sparse.csr_matrix
    b_eq: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    n_steps: int
    start_energy_wh: float

    def block(self, name: str) -> slice:
        """The positions of flow ``name`` in ``x``."""
        first = SCHEDULE_FLOWS.index(name) * self.n_steps
        return slice(first, first + self.n_steps)

    def violation(self, x: np.ndarray) -> dict[str, float]:
        """The largest violation of each kind of constraint at ``x``, in Wh (0 where all hold)."""
        return {
            "equality": float(np.abs(self.a_eq @ x - self.b_eq).max(initial=0.0)),
            "inequality": float(np.maximum(0.0, self.a_ub @ x - self.b_ub).max(initial=0.0)),
            "lower": float(np.maximum(0.0, self.lower - x).max(initial=0.0)),
            "upper": float(np.maximum(0.0, x - self.upper).max(initial=0.0)),
        }


def ledger_point(program: LinearProgram, results: pd.DataFrame, hours_per_step: float) -> np.ndarray:
    """A dispatch ledger, the first project year's per-step frame, as a point ``x`` of ``program``.

    Every flow the dispatch delivered has a variable. If the program is a
    relaxation of the dispatch, every such point is feasible:
    :meth:`LinearProgram.violation` is zero up to round-off. The tests check
    exactly that.
    """
    columns = {
        "pv_dc_to_inverter_wh": results["PV_DC_To_Inverter"],
        "pv_dc_to_battery_wh": results["PV_DC_To_Battery"],
        "pv_ac_to_load_wh": results["PV_AC_To_Load"],
        "pv_ac_export_wh": results["PV_AC_Export"],
        "battery_ac_to_load_wh": results["Battery_AC_To_Load"],
        "battery_discharge_stored_wh": results["Battery_Discharge_DC"],
        "grid_ac_to_battery_wh": results["Grid_AC_To_Battery"],
        "grid_import_to_load_wh": results["Import_From_Grid"] - results["Grid_AC_To_Battery"],
        "stored_energy_spill_wh": results["Capacity_Window_Loss"],
        "standby_loss_wh": results["Standby_Loss"],
    }
    x = np.zeros(len(SCHEDULE_FLOWS) * program.n_steps)
    for name, values in columns.items():
        x[program.block(name)] = np.asarray(values, dtype=np.float64) * hours_per_step
    # Stored energy is a state, reported in Wh rather than as average power.
    x[program.block("battery_energy_end_wh")] = results["Battery_Energy_End"].to_numpy(dtype=np.float64)
    return x


def build_linear_program(problem: LpBoundProblem) -> LinearProgram:
    """The linear program for ``problem``; the module docstring says what it keeps and relaxes."""
    config = problem.battery_config
    n = len(problem)
    hours = problem.hours_per_step
    eff_c, eff_d = config.charge_efficiency, config.discharge_efficiency
    grid_eff = problem.grid_charge_efficiency or 0.0
    pv = np.maximum(0.0, problem.pv_dc_w) * hours
    load = problem.load_w * hours
    emin, emax = problem.window()
    floor = problem.floor_wh()
    start = problem.start_energy_wh()
    if not (math.isfinite(start) and 0.0 <= start <= config.nominal_energy_wh):
        raise ValueError(f"'initial_energy_wh' must be between 0 and {config.nominal_energy_wh:g} Wh")

    cap_ac = _step_energy_cap(config.inverter_ac_capacity_w, hours)
    cap_in = _step_energy_cap(config.max_charge_power_w, hours)
    cap_dis = _step_energy_cap(config.max_discharge_power_w, hours)
    cap_stored = _step_energy_cap(config.stored_power_limit_w, hours)
    cap_grid = _step_energy_cap(problem.grid_import_limit_w, hours) if problem.grid_charge_efficiency else 0.0
    scale = min(1.0, max(0.0, config.ac_output_scale))
    intercepts, slopes = inverter_hull_cuts(config.inverter_efficiency, cap_ac, scale, problem.tangents)

    steps = np.arange(n)
    n_vars = len(SCHEDULE_FLOWS) * n

    def var(block: int) -> np.ndarray:
        return block * n + steps

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    rhs: list[np.ndarray] = []

    def add_rows(terms: Sequence[tuple[np.ndarray, Any]], bound: np.ndarray, row_list: list[int]) -> None:
        # One row per step; terms are (variable indices, coefficient) pairs.
        first = row_list[0]
        for index, coefficient in terms:
            rows.append(first + np.arange(len(index)))
            cols.append(index)
            vals.append(np.broadcast_to(np.asarray(coefficient, dtype=np.float64), (len(index),)).copy())
        rhs.append(np.broadcast_to(np.asarray(bound, dtype=np.float64), (n,)).copy())
        row_list[0] += n

    # Equalities: the load balance, and stored energy step to step.
    counter = [0]
    add_rows([(var(_LP), 1.0), (var(_BA), 1.0), (var(_GL), 1.0)], load, counter)
    energy_rhs = np.zeros(n)
    energy_rhs[0] = start
    first_energy_row = counter[0]
    add_rows(
        [
            (var(_EN), 1.0),
            (var(_SP), 1.0),
            (var(_SB), 1.0),
            (var(_PB), -eff_c),
            (var(_GC), -eff_c * grid_eff),
            (var(_DS), 1.0),
        ],
        energy_rhs,
        counter,
    )
    rows.append(first_energy_row + steps[1:])
    cols.append(var(_EN)[:-1])
    vals.append(np.full(n - 1, -1.0))
    a_eq = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(counter[0], n_vars)
    )
    b_eq = np.concatenate(rhs)

    rows, cols, vals, rhs = [], [], [], []
    counter = [0]
    add_rows([(var(_PI), 1.0), (var(_PB), 1.0)], pv, counter)
    # Before the step, stored energy fits the step's window: a spill takes the rest.
    pre_rhs = emax.copy()
    pre_rhs[0] -= start
    pre_first = counter[0]
    add_rows([(var(_SP), -1.0)], pre_rhs, counter)
    rows.append(pre_first + steps[1:])
    cols.append(var(_EN)[:-1])
    vals.append(np.ones(n - 1))
    standby = config.standby_loss_wh * hours
    if standby > 0.0:
        # The dispatch bleeds min(standby, energy - emin) after the window
        # clip: the energy left is convex in the clipped energy, so the chord
        # over [floor, emax] bounds it from above. Standby is charged in
        # full at emax and not at all at the floor.
        top = np.maximum(emin, emax - standby)
        slope = (top - emin) / (emax - floor)
        chord_rhs = emin - slope * floor
        chord_rhs[0] -= (1.0 - slope[0]) * start
        chord_first = counter[0]
        add_rows([(var(_SP), slope - 1.0), (var(_SB), -1.0)], chord_rhs, counter)
        rows.append(chord_first + steps[1:])
        cols.append(var(_EN)[:-1])
        vals.append(1.0 - slope[1:])
    if math.isfinite(cap_in):
        add_rows([(var(_PB), 1.0), (var(_GC), grid_eff)], np.full(n, cap_in), counter)
    if math.isfinite(cap_stored):
        add_rows([(var(_PB), eff_c), (var(_GC), eff_c * grid_eff)], np.full(n, cap_stored), counter)
    if math.isfinite(cap_ac):
        # Grid charge takes the AC rating PV leaves; with no grid charge the
        # output also stays within the scaled rating.
        add_rows([(var(_LP), 1.0), (var(_EX), 1.0), (var(_BA), 1.0), (var(_GC), 1.0)], np.full(n, cap_ac), counter)
        if scale < 1.0:
            add_rows([(var(_LP), 1.0), (var(_EX), 1.0), (var(_BA), 1.0)], np.full(n, scale * cap_ac), counter)
    for intercept, slope in zip(intercepts, slopes, strict=True):
        bound = np.full(n, intercept)
        # The shared operating point.
        add_rows(
            [(var(_LP), 1.0), (var(_EX), 1.0), (var(_BA), 1.0), (var(_PI), -slope), (var(_DS), -slope * eff_d)],
            bound,
            counter,
        )
        # Each share on its own too, so AC is not credited to the source that did not make it.
        add_rows([(var(_LP), 1.0), (var(_EX), 1.0), (var(_PI), -slope)], bound, counter)
        add_rows([(var(_BA), 1.0), (var(_DS), -slope * eff_d)], bound, counter)
    if math.isfinite(cap_ac):
        # Below the peak ratio, the conversion is no better than at the step's load.
        dc_per_ac, dc_per_export = load_secants(load, config.inverter_efficiency, cap_ac, scale)
        zero = np.zeros(n)
        add_rows(
            [
                (var(_LP), dc_per_ac),
                (var(_BA), dc_per_ac),
                (var(_EX), dc_per_export),
                (var(_PI), -1.0),
                (var(_DS), -eff_d),
            ],
            zero,
            counter,
        )
        add_rows([(var(_LP), dc_per_ac), (var(_EX), dc_per_export), (var(_PI), -1.0)], zero, counter)
        add_rows([(var(_BA), dc_per_ac), (var(_DS), -eff_d)], zero, counter)
    a_ub = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(counter[0], n_vars)
    )
    b_ub = np.concatenate(rhs)

    lower = np.zeros(n_vars)
    upper = np.full(n_vars, math.inf)
    lower[var(_EN)] = floor
    upper[var(_EN)] = emax
    upper[var(_DS)] = cap_stored
    upper[var(_BA)] = cap_dis
    upper[var(_GC)] = cap_grid
    if eff_c <= 0.0:
        upper[var(_PB)] = 0.0
        upper[var(_GC)] = 0.0

    import_price = problem.import_price_per_kwh / 1000.0
    export_price = problem.export_price_per_kwh / 1000.0
    cost = np.zeros(n_vars)
    cost[var(_GL)] = import_price
    cost[var(_GC)] = import_price
    cost[var(_EX)] = -export_price
    return LinearProgram(
        cost=cost,
        a_ub=a_ub,
        b_ub=b_ub,
        a_eq=a_eq,
        b_eq=b_eq,
        lower=lower,
        upper=upper,
        n_steps=n,
        start_energy_wh=start,
    )


def solve_lp_bound(problem: LpBoundProblem) -> LpBound:
    """Solve ``problem``'s linear program with HiGHS.

    Raises:
        RuntimeError: If HiGHS does not report an optimum.
    """
    program = build_linear_program(problem)
    started = time.perf_counter()
    result = linprog(
        program.cost,
        A_ub=program.a_ub,
        b_ub=program.b_ub,
        A_eq=program.a_eq,
        b_eq=program.b_eq,
        bounds=np.column_stack([program.lower, program.upper]),
        method="highs",
    )
    elapsed = time.perf_counter() - started
    if result.status != 0:
        raise RuntimeError(f"The LP bound did not solve (status {result.status}): {result.message}")

    x = np.asarray(result.x)
    schedule = {name: np.maximum(0.0, x[program.block(name)]) for name in SCHEDULE_FLOWS}
    import_price = problem.import_price_per_kwh / 1000.0
    export_price = problem.export_price_per_kwh / 1000.0
    imports = x[program.block("grid_import_to_load_wh")] + x[program.block("grid_ac_to_battery_wh")]
    return LpBound(
        objective=float(result.fun),
        import_cost=float(imports @ import_price),
        export_revenue=float(x[program.block("pv_ac_export_wh")] @ export_price),
        schedule=schedule,
        start_energy_wh=program.start_energy_wh,
        solve_seconds=elapsed,
        n_variables=len(program.cost),
        n_constraints=program.a_eq.shape[0] + program.a_ub.shape[0],
        solver_message=str(result.message),
    )


def lp_instructions(
    bound: LpBound,
    problem: LpBoundProblem,
    *,
    grid_import_limit_w: float | None = None,
    threshold_wh: float = ACTION_THRESHOLD_WH,
) -> DispatchInstructions:
    """The schedule as dispatch instructions, for a replay.

    A step that grid-charges targets the schedule's end-of-step energy. A
    step that discharges (and does not grid-charge) may discharge down to
    that energy, as a reserve. Every other step holds the battery: it may
    not discharge, while surplus PV still charges it, which no instruction
    prevents. Fractions are of the dispatch's window at ``soh_fraction``.
    The site limit defaults to the problem's.
    """
    emin, emax = problem.window()
    span = emax - emin
    energy = bound.schedule["battery_energy_end_wh"]
    fraction = np.clip(np.divide(energy - emin, span, out=np.zeros_like(span), where=span > 0.0), 0.0, 1.0)
    charging = bound.schedule["grid_ac_to_battery_wh"] > threshold_wh
    discharging = (bound.schedule["battery_discharge_stored_wh"] > threshold_wh) & ~charging
    return DispatchInstructions(
        discharge_allowed=discharging,
        reserve_fraction=np.where(discharging, fraction, 0.0),
        grid_target_fraction=np.where(charging, fraction, np.nan),
        grid_charge_efficiency=problem.grid_charge_efficiency or 1.0,
        grid_import_limit_w=problem.grid_import_limit_w if grid_import_limit_w is None else grid_import_limit_w,
    )


def lp_planned_flows(bound: LpBound, problem: LpBoundProblem) -> dict[str, np.ndarray]:
    """The schedule as the replay's planned flows (see :data:`tools.oracles.replay.PLANNED_FLOWS`)."""
    hours = problem.hours_per_step
    schedule = bound.schedule
    return {
        "pv_charge_dc_w": schedule["pv_dc_to_battery_wh"] / hours,
        "grid_charge_ac_w": schedule["grid_ac_to_battery_wh"] / hours,
        "discharge_ac_w": schedule["battery_ac_to_load_wh"] / hours,
        "battery_energy_wh": schedule["battery_energy_end_wh"],
    }


@dataclass(frozen=True)
class LpBoundResult:
    """The bound on one configuration, against what production dispatch delivers.

    ``reference`` replays the instructions App itself dispatches with: the
    fixed-target ones, or greedy dispatch without a ``[smart_charging]``
    table. ``lp_replay`` replays the schedule's instructions, when asked.
    Every cost compared is the first project year's import cost less export
    revenue.
    """

    problem: LpBoundProblem
    bound: LpBound
    reference: ReplayResult
    instructions: DispatchInstructions | None
    lp_replay: ReplayResult | None
    schema: str = LP_BOUND_SCHEMA


def run_lp_bound(
    case: ReplayCase,
    *,
    replay: bool = True,
    tolerance: Tolerance = DEFAULT_REPLAY_TOLERANCE,
    execution_backend: str | None = None,
    **problem_overrides: Any,
) -> LpBoundResult:
    """Bound ``case``'s first-year bill, replay App's own dispatch and, with ``replay``, the schedule."""
    problem = LpBoundProblem.from_case(case, **problem_overrides)
    bound = solve_lp_bound(problem)
    reference = replay_instructions(case, case.configured_instructions(), execution_backend=execution_backend)
    instructions = lp_instructions(bound, problem) if replay else None
    lp_replay = (
        replay_instructions(
            case,
            instructions,
            planned=lp_planned_flows(bound, problem),
            tolerance=tolerance,
            execution_backend=execution_backend,
        )
        if instructions is not None
        else None
    )
    return LpBoundResult(
        problem=problem, bound=bound, reference=reference, instructions=instructions, lp_replay=lp_replay
    )


def report(result: LpBoundResult, case: ReplayCase) -> dict[str, Any]:
    """A JSON-safe summary of ``result``."""
    problem, bound = result.problem, result.bound
    soh, floor_soh = problem.health()
    reference = replay_summary(result.reference, problem.hours_per_step)
    schedule = bound.schedule
    payload: dict[str, Any] = {
        "schema": result.schema,
        "currency": result_currency(case.tariff),
        "n_steps": len(problem),
        "resolution": case.cfg["resolution"],
        "start": str(case.index[0]),
        "end": str(case.index[-1]),
        "battery_kwh": problem.battery_config.nominal_energy_wh / 1000.0,
        "cost_basis": "first project year import cost less export revenue; standing charge excluded",
        "lp": {
            "relaxations": list(RELAXATIONS),
            "soh_fraction": soh,
            "floor_soh_fraction": floor_soh,
            "grid_charge_efficiency": problem.grid_charge_efficiency,
            "grid_import_limit_w": problem.grid_import_limit_w,
            "inverter_tangents": problem.tangents,
            "bound": bound.objective,
            "import_cost": bound.import_cost,
            "export_revenue": bound.export_revenue,
            "import_kwh": float((schedule["grid_import_to_load_wh"] + schedule["grid_ac_to_battery_wh"]).sum() / 1000),
            "export_kwh": float(schedule["pv_ac_export_wh"].sum() / 1000),
            "grid_charge_kwh": float(schedule["grid_ac_to_battery_wh"].sum() / 1000),
            "battery_ac_to_load_kwh": float(schedule["battery_ac_to_load_wh"].sum() / 1000),
            "start_energy_wh": bound.start_energy_wh,
            "end_energy_wh": float(schedule["battery_energy_end_wh"][-1]),
            "n_variables": bound.n_variables,
            "n_constraints": bound.n_constraints,
            "solve_seconds": bound.solve_seconds,
            "solver_message": bound.solver_message,
        },
        "reference": {
            "dispatch": reference_dispatch(case),
            **reference,
            "minus_bound": reference["first_year_cost"] - bound.objective,
            # The bound provably covers this run only if its floor health is at or below the run's.
            "floor_soh_covers_run": floor_soh * 100.0 <= reference["min_soh_pct"],
        },
    }
    if result.lp_replay is not None:
        replayed = replay_summary(result.lp_replay, problem.hours_per_step)
        payload["lp_replay"] = {
            **replayed,
            "minus_bound": replayed["first_year_cost"] - bound.objective,
            "floor_soh_covers_run": floor_soh * 100.0 <= replayed["min_soh_pct"],
            **plan_comparison(result.lp_replay),
        }
    return payload


def schedule_frame(result: LpBoundResult, case: ReplayCase) -> pd.DataFrame:
    """The schedule, one row per step, with the replay's delivered flows when there is one."""
    frame = pd.DataFrame({"timestamp": case.index.astype(str), **result.bound.schedule})
    if result.instructions is not None:
        frame["discharge_allowed"] = result.instructions.discharge_allowed
        frame["reserve_fraction"] = result.instructions.reserve_fraction
        frame["grid_target_fraction"] = result.instructions.grid_target_fraction
    if result.lp_replay is not None:
        delivered = result.lp_replay.artifacts.first_year_results_df
        hours = result.problem.hours_per_step
        frame["delivered_grid_ac_to_battery_wh"] = delivered["Grid_AC_To_Battery"].to_numpy() * hours
        frame["delivered_battery_ac_to_load_wh"] = delivered["Battery_AC_To_Load"].to_numpy() * hours
        frame["delivered_battery_energy_end_wh"] = delivered["Battery_Energy_End"].to_numpy()
        frame["delivered_step_cost"] = result.lp_replay.first_year_step_cost
    return frame


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config", required=True, help="App configuration (TOML or JSON) with a [tariff] and a battery"
    )
    parser.add_argument("--output", default="-", help="JSON summary path; '-' (default) writes to standard output")
    parser.add_argument("--csv", help="Write the per-step schedule to this CSV")
    parser.add_argument("--no-replay", action="store_true", help="Skip replaying the schedule's instructions")
    parser.add_argument("--floor-soh", type=float, help="State of health (fraction) the floor is taken at")
    parser.add_argument("--tangents", type=int, default=DEFAULT_TANGENTS, help="Tangents to the inverter curve")
    parser.add_argument(
        "--grid-charge-efficiency",
        type=float,
        help="AC-to-DC grid-charge efficiency; default the [smart_charging] table's, or no grid charge",
    )
    parser.add_argument("--atol-wh", type=float, default=DEFAULT_REPLAY_TOLERANCE.atol_wh, help="Replay tolerance")
    parser.add_argument("--execution-backend", choices=("python", "numba"), help="Replay backend")
    args = parser.parse_args(argv)

    overrides: dict[str, Any] = {"tangents": args.tangents}
    if args.floor_soh is not None:
        overrides["floor_soh_fraction"] = args.floor_soh
    if args.grid_charge_efficiency is not None:
        overrides["grid_charge_efficiency"] = args.grid_charge_efficiency
    with reuse_prepared_inputs():
        case = prepare_replay(load_config(args.config))
        result = run_lp_bound(
            case,
            replay=not args.no_replay,
            tolerance=Tolerance(atol_wh=args.atol_wh),
            execution_backend=args.execution_backend,
            **overrides,
        )
    write_json(report(result, case), args.output)
    if args.csv:
        write_csv(schedule_frame(result, case), LP_SCHEDULE_SCHEMA, args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
