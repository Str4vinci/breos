"""A perfect-foresight lower bound on one year's electricity bill, by linear programming.

The linear program sees the whole first project year at once: PV, load,
battery temperature and the tariff's per-step prices. It chooses every flow
of every step (PV to the inverter or the battery, AC to the load or the
grid, battery discharge, grid charge and stored energy) to minimise import
cost less export revenue. It is solved with HiGHS through
:func:`scipy.optimize.linprog`. Its feasible set contains every flow the
production dispatch step can deliver, under any
:class:`~breos.dispatch_instructions.DispatchInstructions`, as long as the
battery's health stays at or above the program's floor health and no
battery is replaced during the year. Under that condition its optimum is a
cost no controller can beat, causal or not. The standing charge is left
out: every schedule pays it.

The floor health is the one setting the claim depends on. By default
(``floor_soh="auto"``, ``--floor-soh auto``) the program is solved at the
opening health, the reference dispatch and the program's own schedule are
replayed, and it is solved again with the floor at the lowest health,
unrounded, that those replays reached. The reported bound then provably
covers every run it reports, and any dispatch whose health stays at or
above that floor; it does not cover a controller that fades harder. The
optimum at the opening health is reported next to it as a
``fixed_health_estimate``: a close estimate, not a bound, since every run
fades. ``bound_is_strict`` says whether the bound covers every reported
run, and a warning names each one it does not.

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

- **Opening health for the ceiling and the efficiencies.** State of health
  and both efficiencies are held at the year's opening values
  (``soh_fraction``, the configured efficiencies). Between replacements,
  health only falls, which lowers the dispatch's ceiling, and resistance
  fade only lowers its efficiencies; both keep the bound valid. With fade a
  dispatch ledger is not exactly a point of the program, since the
  dispatch drew more energy than the program's efficiencies need for the
  same stored energy, but the point :func:`ledger_point` builds from the
  stored energies is, and it costs no more. Falling health also lowers the
  floor, by ``min_soc`` times the fade, which is the one way the opening
  health would favour the program: hence the floor health above.
- **No replacement.** A replacement adds stored energy and restores health
  within the year, which the program does not model. A replayed run that
  replaced its battery in the first year is never counted as covered.
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
  ``min(standby, energy - emin)`` each step after the window clip, so the
  energy left is ``max(emin, energy - standby)`` above ``emin`` and the
  energy itself below it. The program charges the line through ``(emin,
  emin)`` and the ceiling's value, which lies on or above that map: the
  full standby loss at the ceiling, none at ``emin``, a share in between.
  A spill variable absorbs the capacity-window loss when a colder step
  shrinks the window, and any standby the line does not charge.
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

The bound is the optimum within the solver's tolerances. The schedule of
the fixed-health estimate can be turned into instructions (:func:`lp_instructions`) and replayed through the
production run with :mod:`tools.oracles.replay`, which shows the gap between
what the relaxed physics promised and what production delivers. Legacy
options not ported: the per-project-year and year-one-reused LP modes, the
wear price, the terminal energy value and the lifetime NPV sums; this tool
bounds the first year's bill.

Usage:
    python tools/oracles/lp_bound.py --config my.toml --output bound.json --csv schedule.csv
    python tools/oracles/lp_bound.py --config my.toml --floor-soh 0.9    # a floor of your own
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    first_year_replacements,
    plan_comparison,
    prepare_replay,
    reference_dispatch,
    replay_instructions,
    replay_summary,
)

LP_BOUND_SCHEMA = "breos_lp_bound_v1"
LP_SCHEDULE_SCHEMA = "breos_lp_bound_schedule_v1"
DEFAULT_TANGENTS = 5
# Below this load (Wh over the step) a step takes the peak-ratio line, not its own secant.
MIN_SECANT_LOAD_WH = 1.0
# A planned flow below this (Wh over the step) is solver noise, not an action.
ACTION_THRESHOLD_WH = 1e-6
# The replay compares the schedule's flows with this tolerance.
DEFAULT_REPLAY_TOLERANCE = Tolerance(atol_wh=1.0)

# What the program relaxes. The floor health is not among them: it is the
# condition the bound holds under (see run_lp_bound and the report's bound_scope).
RELAXATIONS = (
    "opening_health_ceiling_and_efficiencies",
    "inverter_concave_hull",
    "standby_loss_by_chord",
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
    curve is convex. A step with a load below 1 Wh, or at or above the
    peak point, takes the peak ratio and no export term, which the hull
    cuts already hold. Near zero load the DC-for-AC inverse sits on the
    curve's zero crossing, where its ratio is large and round-off in it is
    too, so a secant there would not be worth its margin. Both factors are
    shaded by 1e-9 so round-off in the dispatch's own conversion cannot cut
    off a point it delivers.
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
    for step in np.flatnonzero((load_wh >= MIN_SECANT_LOAD_WH) & (load_wh < peak_ac)):
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
            hours_per_step=get_hours_per_step(case.resolved.cfg["resolution"]),
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
    the tariff's currency. No dispatch on the same inputs pays less if its
    health stays at or above the problem's floor health and it replaces no
    battery. ``schedule`` maps each name in :data:`SCHEDULE_FLOWS` to its Wh
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
    eff_charge: float
    eff_discharge: float
    grid_charge_efficiency: float

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

    Every flow the dispatch delivered has a variable. Charge and discharge
    are mapped through the energy they store and deliver at the program's
    efficiencies, and any stored energy that leaves otherwise goes to the
    spill. With the configured efficiencies this is the ledger itself.
    With resistance fade the dispatch's efficiencies are lower, so the
    point draws less PV or grid energy and releases less stored energy than
    production did, for the same stored energy and the same AC: it costs no
    more than production and the bound still holds, but it is not the
    ledger one to one.

    If the program is a relaxation of the dispatch, every such point is
    feasible: :meth:`LinearProgram.violation` is zero up to round-off. That
    round-off includes the dispatch's own, for example its discharge-limit
    bisection, which can overshoot the limit by about 1e-9 Wh. The tests
    check exactly that.
    """
    stored_pv = results["PV_Origin_Battery_Charge_Stored"].to_numpy(dtype=np.float64)
    stored_grid = results["Grid_Origin_Battery_Charge_Stored"].to_numpy(dtype=np.float64)
    battery_dc = (results["Battery_Discharge_DC"] - results["Battery_Discharge_Loss"]).to_numpy(dtype=np.float64)
    grid_ac = results["Grid_AC_To_Battery"].to_numpy(dtype=np.float64)
    grid_gain = program.eff_charge * program.grid_charge_efficiency
    discharge = battery_dc / program.eff_discharge if program.eff_discharge > 0.0 else battery_dc
    columns = {
        "pv_dc_to_inverter_wh": results["PV_DC_To_Inverter"].to_numpy(dtype=np.float64),
        "pv_dc_to_battery_wh": stored_pv / program.eff_charge if program.eff_charge > 0.0 else stored_pv,
        "pv_ac_to_load_wh": results["PV_AC_To_Load"].to_numpy(dtype=np.float64),
        "pv_ac_export_wh": results["PV_AC_Export"].to_numpy(dtype=np.float64),
        "battery_ac_to_load_wh": results["Battery_AC_To_Load"].to_numpy(dtype=np.float64),
        "battery_discharge_stored_wh": discharge,
        "grid_ac_to_battery_wh": stored_grid / grid_gain if grid_gain > 0.0 else grid_ac,
        "grid_import_to_load_wh": (results["Import_From_Grid"] - results["Grid_AC_To_Battery"]).to_numpy(
            dtype=np.float64
        ),
        "stored_energy_spill_wh": (results["Capacity_Window_Loss"] + results["Battery_Discharge_DC"]).to_numpy(
            dtype=np.float64
        )
        - discharge,
        "standby_loss_wh": results["Standby_Loss"].to_numpy(dtype=np.float64),
    }
    x = np.zeros(len(SCHEDULE_FLOWS) * program.n_steps)
    for name, values in columns.items():
        x[program.block(name)] = values * hours_per_step
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
        # After the window clip the dispatch bleeds min(standby, E - emin)
        # down to emin, leaving E below emin and max(emin, E - standby) above
        # it. That map is not convex across emin, but the line through
        # (emin, emin) and (emax, max(emin, emax - standby)) lies on or above
        # it everywhere: above emin the map is convex and meets the line at
        # both ends, and below emin it is the identity, under a line of slope
        # at most 1 through (emin, emin). With fade the dispatch's emin is
        # lower, which only lowers the map. Standby is charged in full at
        # emax and not at all at emin.
        top = np.maximum(emin, emax - standby)
        span = emax - emin
        slope = np.divide(top - emin, span, out=np.zeros_like(span), where=span > 0.0)
        chord_rhs = emin * (1.0 - slope)
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
        eff_charge=eff_c,
        eff_discharge=eff_d,
        grid_charge_efficiency=grid_eff,
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


FLOOR_AUTO = "auto"
FLOOR_OPENING = "opening"


@dataclass(frozen=True)
class CoveredRun:
    """A replayed run the bound was checked against, and whether it provably covers it.

    The bound covers a run when the run's lowest first-year health is at or
    above the program's floor health and the run replaced no battery in the
    first year. ``reason`` says why not, when it does not.
    """

    name: str
    replay: ReplayResult
    min_soh_fraction: float
    replacements: int
    covered: bool
    reason: str | None


@dataclass(frozen=True)
class LpBoundResult:
    """The bound on one configuration, against what production dispatch delivers.

    ``bound`` is solved on ``problem``, whose floor health ``floor_mode``
    chose. ``fixed_health_estimate`` is the optimum with the floor at the
    opening health: a close estimate, but not a bound on a run whose health
    fades, which every real run does. The schedule replayed is the
    estimate's. ``reference`` replays the instructions App itself
    dispatches with: the fixed-target ones, or greedy dispatch without a
    ``[smart_charging]`` table. ``lp_replay`` replays the schedule, when
    asked. ``runs`` are every replay the bound was checked against, and
    ``bound_is_strict`` says it covers all of them. Every cost compared is
    the first project year's import cost less export revenue.
    """

    problem: LpBoundProblem
    bound: LpBound
    fixed_health_estimate: LpBound
    reference: ReplayResult
    instructions: DispatchInstructions | None
    lp_replay: ReplayResult | None
    runs: tuple[CoveredRun, ...]
    floor_mode: str
    schema: str = LP_BOUND_SCHEMA

    @property
    def bound_is_strict(self) -> bool:
        return all(run.covered for run in self.runs)


def _min_soh(replay: ReplayResult) -> float:
    return float(replay.artifacts.first_year_results_df["Battery_SOH"].min()) / 100.0


def _covered_run(name: str, replay: ReplayResult, floor_soh: float) -> CoveredRun:
    health = _min_soh(replay)
    replacements = first_year_replacements(replay)
    reasons = []
    if replacements:
        reasons.append(
            f"it replaced the battery {replacements} time(s) in the first year, which adds stored energy "
            "and restores health the program does not model"
        )
    if health < floor_soh:
        reasons.append(f"its health fell to {health:.6f}, below the floor health {floor_soh:.6f}")
    return CoveredRun(
        name=name,
        replay=replay,
        min_soh_fraction=health,
        replacements=replacements,
        covered=not reasons,
        reason="; ".join(reasons) or None,
    )


def run_lp_bound(
    case: ReplayCase,
    *,
    replay: bool = True,
    floor_soh: str | float = FLOOR_AUTO,
    covering: Mapping[str, ReplayResult] | None = None,
    tolerance: Tolerance = DEFAULT_REPLAY_TOLERANCE,
    execution_backend: str | None = None,
    **problem_overrides: Any,
) -> LpBoundResult:
    """Bound ``case``'s first-year bill, and replay App's own dispatch and, with ``replay``, the schedule.

    ``floor_soh`` sets the floor health. ``"auto"`` (the default) first
    solves at the opening health, replays the reference and the schedule,
    then solves again with the floor at the lowest health, unrounded, that
    any of those replays or the ``covering`` ones reached. The bound then
    holds for every run it reports, and for any dispatch whose health stays
    at or above that floor; it does not cover a controller that fades
    harder. ``"opening"`` keeps the opening health, which covers no run
    that fades. A number is a floor health fraction.

    A run that replaces the battery in its first year is never covered.
    When a reported run is not covered, a warning says so and
    ``bound_is_strict`` is False.
    """
    if "floor_soh_fraction" in problem_overrides:
        raise TypeError("Set the floor health with 'floor_soh', not 'floor_soh_fraction'")
    opening = LpBoundProblem.from_case(case, **problem_overrides)
    estimate = solve_lp_bound(opening)
    reference = replay_instructions(case, case.configured_instructions(), execution_backend=execution_backend)
    instructions = lp_instructions(estimate, opening) if replay else None
    lp_replay = (
        replay_instructions(
            case,
            instructions,
            planned=lp_planned_flows(estimate, opening),
            tolerance=tolerance,
            execution_backend=execution_backend,
        )
        if instructions is not None
        else None
    )
    named = {"reference": reference, **({"lp_replay": lp_replay} if lp_replay is not None else {}), **(covering or {})}

    opening_soh = opening.health()[0]
    if floor_soh == FLOOR_AUTO:
        floor = min([opening_soh, *(_min_soh(run) for run in named.values())])
    elif floor_soh == FLOOR_OPENING:
        floor = opening_soh
    elif isinstance(floor_soh, str):
        raise ValueError(f"'floor_soh' must be {FLOOR_AUTO!r}, {FLOOR_OPENING!r} or a fraction")
    else:
        floor = float(floor_soh)
    problem = dataclasses.replace(opening, floor_soh_fraction=floor)
    bound = estimate if floor == opening_soh else solve_lp_bound(problem)

    runs = tuple(_covered_run(name, run, floor) for name, run in named.items())
    for run in runs:
        if not run.covered:
            warnings.warn(f"The LP bound does not provably cover the {run.name} run: {run.reason}", stacklevel=2)
    return LpBoundResult(
        problem=problem,
        bound=bound,
        fixed_health_estimate=estimate,
        reference=reference,
        instructions=instructions,
        lp_replay=lp_replay,
        runs=runs,
        floor_mode=floor_soh if isinstance(floor_soh, str) else "given",
    )


def report(result: LpBoundResult, case: ReplayCase) -> dict[str, Any]:
    """A JSON-safe summary of ``result``."""
    problem, bound = result.problem, result.bound
    soh, floor_soh = problem.health()
    schedule = bound.schedule
    coverage = {run.name: run for run in result.runs}

    def run_fields(name: str, replay: ReplayResult) -> dict[str, Any]:
        summary = replay_summary(replay, problem.hours_per_step)
        run = coverage[name]
        return {
            **summary,
            "minus_bound": summary["first_year_cost"] - bound.objective,
            "covered_by_bound": run.covered,
            "not_covered_because": run.reason,
        }

    payload: dict[str, Any] = {
        "schema": result.schema,
        "currency": result_currency(case.tariff),
        "n_steps": len(problem),
        "resolution": case.resolved.cfg["resolution"],
        "start": str(case.index[0]),
        "end": str(case.index[-1]),
        "battery_kwh": problem.battery_config.nominal_energy_wh / 1000.0,
        "cost_basis": "first project year import cost less export revenue; standing charge excluded",
        "bound_is_strict": result.bound_is_strict,
        "lp": {
            "bound": bound.objective,
            "bound_scope": (
                "a lower bound for any dispatch of this year whose state of health stays at or above "
                "floor_soh_fraction and that replaces no battery"
            ),
            "floor_mode": result.floor_mode,
            "floor_soh_fraction": floor_soh,
            "soh_fraction": soh,
            "fixed_health_estimate": result.fixed_health_estimate.objective,
            "fixed_health_estimate_note": (
                "the optimum with the floor at the opening health; not a bound on a run whose health fades"
            ),
            "relaxations": list(RELAXATIONS),
            "grid_charge_efficiency": problem.grid_charge_efficiency,
            "grid_import_limit_w": problem.grid_import_limit_w,
            "inverter_tangents": problem.tangents,
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
        "reference": {"dispatch": reference_dispatch(case), **run_fields("reference", result.reference)},
    }
    if result.lp_replay is not None:
        payload["lp_replay"] = {
            "schedule": "fixed_health_estimate",
            **run_fields("lp_replay", result.lp_replay),
            **plan_comparison(result.lp_replay),
        }
    others = [run for run in result.runs if run.name not in ("reference", "lp_replay")]
    if others:
        payload["covering"] = {run.name: run_fields(run.name, run.replay) for run in others}
    return payload


def schedule_frame(result: LpBoundResult, case: ReplayCase) -> pd.DataFrame:
    """The replayed schedule, one row per step, with the replay's delivered flows when there is one.

    The schedule is the fixed-health estimate's, the one the instructions
    come from; ``bound_battery_energy_end_wh`` is the stored energy of the
    schedule that attains the reported bound.
    """
    frame = pd.DataFrame({"timestamp": case.index.astype(str), **result.fixed_health_estimate.schedule})
    frame["bound_battery_energy_end_wh"] = result.bound.schedule["battery_energy_end_wh"]
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
    parser.add_argument(
        "--floor-soh",
        default=FLOOR_AUTO,
        help=(
            "Floor health: 'auto' (default) takes the lowest health the reported replays reached, "
            "'opening' the opening health (not a bound on a run that fades), or a fraction"
        ),
    )
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
    floor_soh: str | float = args.floor_soh if args.floor_soh in (FLOOR_AUTO, FLOOR_OPENING) else float(args.floor_soh)
    if args.grid_charge_efficiency is not None:
        overrides["grid_charge_efficiency"] = args.grid_charge_efficiency
    with reuse_prepared_inputs():
        case = prepare_replay(load_config(args.config))
        result = run_lp_bound(
            case,
            replay=not args.no_replay,
            floor_soh=floor_soh,
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
