"""The IV curve of a series string whose modules are shaded unevenly (design record 0008, level 2).

Each module splits into bypass-diode substrings. A substring is its cells in
series: the most-shaded cell at the irradiance its limiting shaded fraction
leaves, and the other cells at the irradiance that keeps the substring's
area mean. Each part is a single-diode curve with its share of the module's
series and shunt resistance and diode voltage. Below the bypass diode's
forward voltage the diode conducts and holds the substring there, so a
shaded cell in reverse bias sees no more than the rest of its substring; the
reverse characteristic before that is the shunt alone, with no avalanche
breakdown. The string carries one current through every substring, and its
voltage is their sum, less the cable's drop.

Steps where every substring of the string sees the same light use the
module's own curve exactly; only the others build the string's curve, on a
grid of currents refined to its global maximum power point.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pvlib

from breos.pv.shading import ModuleShading

if TYPE_CHECKING:
    from breos.solar import IrradianceParts

#: Forward voltage of a bypass diode, in V, when the topology does not state
#: it; the value pvlib's partial-shading example assumes.
DEFAULT_BYPASS_DIODE_VOLTAGE_V = 0.5

# Points of the current grid each mismatched step's curve is built on, and
# golden-section steps refining its maximum power point.
_CURVE_POINTS = 120
_GOLDEN_STEPS = 30
_BISECTION_STEPS = 40
# Local maxima of each step's tabled curve refined on the exact curve.
_REFINED_PEAKS = 4
_GOLDEN_RATIO = (5.0**0.5 - 1.0) / 2.0


def _newton_mpp(*params):
    return pvlib.pvsystem.max_power_point(*params, method="newton")


def _quiet(function, *args):
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        return function(*args)


class ShadedStringCurve:
    """One string's power and current from its modules' own shading, at each step.

    ``scale`` is each step's string current over one module's current, from
    the array's static losses, and ``cable_ohm`` the string cable's loop
    resistance. Currents are the string's and voltages its terminal voltage.
    """

    def __init__(
        self,
        shading: ModuleShading,
        parts: IrradianceParts,
        scale: np.ndarray,
        modules: int,
        cable_ohm: float = 0.0,
        bypass_diode_voltage_v: float = DEFAULT_BYPASS_DIODE_VOLTAGE_V,
    ) -> None:
        if len(shading.modules) != modules:
            raise ValueError(f"module_shading has {len(shading.modules)} modules for a string of {modules}")
        self.parts = parts
        self.scale = np.asarray(scale, dtype=float)
        self.modules = modules
        self.cable_ohm = float(cable_ohm)
        self.bypass_v = float(bypass_diode_voltage_v)
        steps = len(self.scale)
        cells = float(parts.pv_params.N_Cells)
        diodes = shading.bypass_diodes
        self.cells = cells
        per_substring = cells / diodes

        weights = shading.axes.grid_weights(parts.sun_azimuth, parts.sun_elevation)
        mean, limiting = shading.substring_fractions(weights)
        view = shading.sky_view_factors[:, None, None]

        def irradiance(fraction: np.ndarray) -> np.ndarray:
            return parts.irradiance(fraction, view)

        e_mean = irradiance(mean)
        limited = (limiting > mean) & (per_substring > 1.0)
        rest = np.clip((per_substring * mean - limiting) / max(per_substring - 1.0, 1.0), 0.0, 1.0)
        e_limit = irradiance(limiting)
        e_rest = irradiance(np.where(limited, rest, mean))

        lit = (e_mean.max(axis=(0, 1)) > 0) & (self.scale > 0)
        same = np.all(e_mean == e_mean[:1, :1], axis=(0, 1)) & ~limited.any(axis=(0, 1))
        self.uniform = np.flatnonzero(lit & same)
        self.mismatched = np.flatnonzero(lit & ~same)

        # Each string's own mean irradiance, the reference its mismatch is measured from.
        self.mean_irradiance = e_mean.mean(axis=(0, 1))

        self.mpp_bare = np.zeros(steps)
        self.mpp = np.zeros(steps)
        self.v_mp = np.zeros(steps)
        self.i_mp = np.zeros(steps)
        self.v_oc = np.zeros(steps)
        self._uniform_params: tuple[np.ndarray, ...] = ()
        self._table_v = np.zeros((0, _CURVE_POINTS))
        self._table_i = np.zeros((0, _CURVE_POINTS))
        self._uniform_curve()
        if self.mismatched.size:
            self._substrings = self._substring_elements(shading, e_mean, e_limit, e_rest, limited, per_substring, cells)
            self._mismatched_curve()

    # -- uniform steps: the module's own curve -------------------------------------------------

    def _uniform_curve(self) -> None:
        rows = self.uniform
        if not rows.size:
            return
        params = list(self.parts.diode_parameters(self.mean_irradiance[rows], rows))
        bare = _quiet(_newton_mpp, *params)
        scale = self.scale[rows]
        self.mpp_bare[rows] = self.modules * scale * np.nan_to_num(np.asarray(bare["p_mp"], dtype=float))
        # The cable as each module's share of series resistance, as for an unshaded string.
        params[2] = params[2] + scale * self.cable_ohm / self.modules
        cabled = _quiet(_newton_mpp, *params)
        self.mpp[rows] = self.modules * scale * np.nan_to_num(np.asarray(cabled["p_mp"], dtype=float))
        self.v_mp[rows] = self.modules * np.nan_to_num(np.asarray(cabled["v_mp"], dtype=float))
        self.i_mp[rows] = scale * np.nan_to_num(np.asarray(cabled["i_mp"], dtype=float))
        self.v_oc[rows] = self.modules * np.nan_to_num(np.asarray(_quiet(pvlib.pvsystem.v_from_i, 0.0, *params)))
        self._uniform_params = tuple(params)

    # -- mismatched steps: the string's curve from its substrings ------------------------------

    def _substring_elements(
        self,
        shading: ModuleShading,
        e_mean: np.ndarray,
        e_limit: np.ndarray,
        e_rest: np.ndarray,
        limited: np.ndarray,
        per_substring: float,
        cells: float,
    ) -> list[tuple[int, list[tuple[np.ndarray, ...]], np.ndarray]]:
        """Each distinct substring's count and its parts' single-diode parameters on the mismatched steps."""
        rows = self.mismatched
        groups: dict[tuple[Any, ...], list[tuple[int, int]]] = {}
        for m, module in enumerate(shading.modules):
            for k in range(shading.bypass_diodes):
                key = (id(module.mean[k]), id(module.limiting[k]), module.sky_view_factor)
                groups.setdefault(key, []).append((m, k))

        def element(irradiance: np.ndarray, share: float) -> tuple[np.ndarray, ...]:
            photo, saturation, series, shunt, diode = self.parts.diode_parameters(irradiance, rows)
            return (photo, saturation, series * share, shunt * share, diode * share)

        # Most substrings are lit like the brightest on most steps: they share
        # its curve there, and only the others are solved.
        brightest = np.max(np.where(limited[:, :, rows], -np.inf, e_mean[:, :, rows]), axis=(0, 1))
        brightest = np.where(np.isfinite(brightest), brightest, e_mean[:, :, rows].max(axis=(0, 1)))
        self._reference = [element(brightest, per_substring / cells)]
        substrings = []
        for members in groups.values():
            m, k = members[0]
            own = limited[m, k, rows] | (e_mean[m, k, rows] != brightest)
            if limited[m, k, rows].any():
                lim = limited[m, k, rows]
                low = np.where(lim, e_limit[m, k, rows], e_mean[m, k, rows])
                high = np.where(lim, e_rest[m, k, rows], e_mean[m, k, rows])
                # Unlimited steps split the substring the same way at one irradiance.
                parts = [element(low, 1.0 / cells), element(high, (per_substring - 1.0) / cells)]
            else:
                parts = [element(e_mean[m, k, rows], per_substring / cells)]
            substrings.append((len(members), parts, own))
        # Each part's photocurrent: the curve's knees lie just below them.
        self._photocurrents = np.asarray([p[0] for _, parts, _ in substrings for p in parts] + [self._reference[0][0]])
        return substrings

    def _substring_voltage(self, current: np.ndarray, parts: list[tuple[np.ndarray, ...]], rows: Any) -> np.ndarray:
        """One substring's voltage at ``current`` on the mismatched steps ``rows``, held by its bypass diode."""
        shape = (-1,) + (1,) * (current.ndim - 1)
        voltage = np.zeros(current.shape)
        for params in parts:
            part = _quiet(pvlib.pvsystem.v_from_i, current, *(p[rows].reshape(shape) for p in params))
            voltage += np.nan_to_num(np.asarray(part, dtype=float), nan=-np.inf, posinf=np.inf, neginf=-np.inf)
        return np.maximum(voltage, -self.bypass_v)

    def _voltage(self, current: np.ndarray, select: np.ndarray | None = None) -> np.ndarray:
        """The string's voltage before its cable at module-level ``current``.

        ``select`` holds positions among the mismatched steps, one per row of
        ``current``; all of them, in order, when None.
        """
        if select is None:
            select = np.arange(self.mismatched.size)
        reference = self._substring_voltage(current, self._reference, select)
        total = np.zeros(current.shape)
        for count, parts, own in self._substrings:
            voltage = reference.copy()
            mine = np.flatnonzero(own[select])
            if mine.size:
                voltage[mine] = self._substring_voltage(current[mine], parts, select[mine])
            total += count * voltage
        return total

    def _current_grid(self) -> np.ndarray:
        """Each mismatched step's module-level currents for its curve, dense below every photocurrent."""
        currents = self._photocurrents
        levels = np.quantile(currents, [0.0, 0.25, 0.5, 0.75, 1.0], axis=0).T
        top = currents.max(axis=0)
        unit = np.linspace(0.0, 1.0, _CURVE_POINTS)
        knee = np.array([0.98, 0.995])
        # Negative current, above open circuit, for the start-up voltage of
        # strings in parallel: a brighter string can drive up to its own
        # short-circuit current back through this one.
        reverse = -float(self.parts.pv_params.Isc) * 1.5 * np.geomspace(1e-4, 1.0, 24)
        pieces = [top[:, None] * unit[None, :], np.broadcast_to(reverse, (len(top), reverse.size))]
        for j in range(levels.shape[1]):
            pieces.append(levels[:, j : j + 1] * np.concatenate((unit[:: max(_CURVE_POINTS // 24, 1)], knee))[None, :])
        # Every part's own knee, where a substring starts to need its bypass diode.
        pieces.append((currents.T[:, :, None] * knee[None, None, :]).reshape(len(top), -1))
        return np.sort(np.concatenate(pieces, axis=1), axis=1)

    def _mismatched_curve(self) -> None:
        rows = self.mismatched
        scale = self.scale[rows]
        grid = self._current_grid()
        bare = self._voltage(grid)
        drop = (scale * self.cable_ohm)[:, None] * grid
        self._table_i = grid
        self._table_v = bare - drop
        self.v_oc[rows] = self._voltage(np.zeros(len(rows)))
        everything = np.arange(len(rows))
        unlimited = np.full(len(rows), np.inf)
        bare_power, bare_current = self._refine(everything, grid, bare, False, unlimited, -np.inf, np.inf)
        self.mpp_bare[rows] = bare_power
        if self.cable_ohm == 0:
            # Without a cable both curves are one.
            power, current = bare_power, bare_current
        else:
            power, current = self._refine(everything, grid, self._table_v, True, unlimited, -np.inf, np.inf)
        self.mpp[rows] = power
        self.i_mp[rows] = scale * current
        with np.errstate(divide="ignore", invalid="ignore"):
            self.v_mp[rows] = np.where(scale * current > 0, power / (scale * current), 0.0)

    def _terminal_voltage(self, current: np.ndarray, select: np.ndarray, cabled: bool) -> np.ndarray:
        voltage = self._voltage(current, select)
        if cabled:
            voltage = voltage - self.scale[self.mismatched[select]] * self.cable_ohm * current
        return voltage

    def _refine(
        self,
        select: np.ndarray,
        grid: np.ndarray,
        table: np.ndarray,
        cabled: bool,
        current_limit: np.ndarray,
        v_min: float,
        v_max: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """The best power and its module-level current on the mismatched steps ``select``, within the limits.

        The table is exact at its points. Brackets around its best local
        maxima, and around where the voltage range and the current limit cut
        the curve, are refined on the exact curve: cut to what the limits
        allow by bisection, then searched by golden section.
        """
        scale = self.scale[self.mismatched[select]]
        rows = np.arange(len(select))
        points = grid.shape[1]
        feasible = (table >= v_min) & (table <= v_max) & (grid <= current_limit[:, None])
        power = np.where(feasible, scale[:, None] * grid * table, -np.inf)
        best_power = np.max(power, axis=1).clip(min=0.0)
        best_current = grid[rows, np.argmax(power, axis=1)]
        # A repeated current, to rounding, is one point of the curve: only its
        # first copy can be a peak.
        tolerance = 1e-9 * np.abs(grid).max(axis=1, keepdims=True)
        repeat = np.zeros_like(grid, dtype=bool)
        repeat[:, 1:] = grid[:, 1:] - grid[:, :-1] <= tolerance
        # Peaks among distinct currents: a repeat's power, which differs only
        # by rounding, does not compete with its first copy.
        distinct = np.where(repeat, -np.inf, power)
        peak = np.zeros_like(power, dtype=bool)
        peak[:, 1:-1] = (distinct[:, 1:-1] >= distinct[:, :-2]) & (distinct[:, 1:-1] >= distinct[:, 2:])
        peak[:, 0] = distinct[:, 0] >= distinct[:, 1]
        peak[:, -1] = distinct[:, -1] >= distinct[:, -2]
        candidate = peak & feasible & ~repeat
        ranked = np.argsort(np.where(candidate, power, -np.inf), axis=1)[:, ::-1][:, :_REFINED_PEAKS]

        def neighbours(k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            # The nearest distinct currents below and above point ``k``.
            here = grid[rows, k][:, None]
            below = np.clip((grid < here - tolerance).sum(axis=1) - 1, 0, points - 1)
            above = np.clip((grid <= here + tolerance).sum(axis=1), 0, points - 1)
            return below, above

        brackets: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for j in range(ranked.shape[1]):
            k = ranked[:, j]
            below, above = neighbours(k)
            brackets.append((below, above, candidate[rows, k]))
        # Where a limit cuts the curve between two points, the best point can
        # lie between them with neither point allowed.
        if np.isfinite(v_max):
            k = np.clip((table > v_max).sum(axis=1), 1, points - 1)
            brackets.append((k - 1, k, (table[:, 0] > v_max) & (table[:, -1] <= v_max)))
        if np.isfinite(v_min):
            k = np.clip((table >= v_min).sum(axis=1), 1, points - 1)
            brackets.append((k - 1, k, (table[:, 0] >= v_min) & (table[:, -1] < v_min)))
        if np.isfinite(current_limit).any():
            k = np.clip((grid <= current_limit[:, None]).sum(axis=1), 1, points - 1)
            brackets.append((k - 1, k, np.isfinite(current_limit) & (grid[:, -1] > current_limit)))

        def at(current: np.ndarray) -> np.ndarray:
            return scale * current * self._terminal_voltage(current, select, cabled)

        def bisect(low: np.ndarray, high: np.ndarray, target: float) -> np.ndarray:
            # The voltage falls with current: the current where it crosses ``target``.
            for _ in range(_BISECTION_STEPS):
                middle = 0.5 * (low + high)
                above = self._terminal_voltage(middle, select, cabled) > target
                low, high = np.where(above, middle, low), np.where(above, high, middle)
            return 0.5 * (low + high)

        for lo_k, hi_k, usable in brackets:
            if not usable.any():
                continue
            low, high = grid[rows, lo_k], grid[rows, hi_k]
            v_low, v_high = table[rows, lo_k], table[rows, hi_k]
            ok = usable & (v_high <= v_max) & (v_low >= v_min) & (low <= current_limit)
            if np.isfinite(v_max):
                cut = ok & (v_low > v_max)
                if cut.any():
                    low = np.where(cut, bisect(low.copy(), high.copy(), v_max), low)
            if np.isfinite(v_min):
                cut = ok & (v_high < v_min)
                if cut.any():
                    high = np.where(cut, bisect(low.copy(), high.copy(), v_min), high)
            high = np.minimum(high, current_limit)
            ok &= low <= high
            if not ok.any():
                continue
            a, b = low.copy(), np.where(ok, high, low)
            c, d = b - _GOLDEN_RATIO * (b - a), a + _GOLDEN_RATIO * (b - a)
            fc, fd = at(c), at(d)
            for _ in range(_GOLDEN_STEPS):
                # Each step keeps one interior point and evaluates one new one.
                left = fc >= fd
                a, b = np.where(left, a, c), np.where(left, d, b)
                new = np.where(left, b - _GOLDEN_RATIO * (b - a), a + _GOLDEN_RATIO * (b - a))
                fnew = at(new)
                c, d, fc, fd = (
                    np.where(left, new, d),
                    np.where(left, c, new),
                    np.where(left, fnew, fd),
                    np.where(left, fc, fnew),
                )
            for point in (low, np.where(ok, high, low), 0.5 * (a + b)):
                value = np.where(ok, at(point), -np.inf)
                better = value > best_power
                best_power = np.where(better, value, best_power)
                best_current = np.where(better, point, best_current)
        return best_power.clip(min=0.0), best_current

    # -- the solver's view ---------------------------------------------------------------------

    def constrained_mpp(
        self, rows: np.ndarray, v_min: float, v_max: float, current_limit: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """The string's best power, and its terminal voltage, on the steps ``rows`` (a boolean mask) with
        the voltage in ``[v_min, v_max]`` and the current at most ``current_limit``; 0 where no point
        qualifies."""
        selected = np.flatnonzero(rows)
        out = np.zeros(selected.size)
        voltage = np.zeros(selected.size)
        mismatched = np.isin(selected, self.mismatched)
        if mismatched.any():
            select = np.searchsorted(self.mismatched, selected[mismatched])
            scale = self.scale[selected[mismatched]]
            with np.errstate(divide="ignore", invalid="ignore"):
                limit = np.where(scale > 0, current_limit / scale, np.inf)
            power, current = self._refine(
                select, self._table_i[select], self._table_v[select], True, limit, v_min, v_max
            )
            out[mismatched] = power
            with np.errstate(divide="ignore", invalid="ignore"):
                voltage[mismatched] = np.where(power > 0, power / (scale * current), 0.0)
        uniform = ~mismatched
        if uniform.any():
            out[uniform], voltage[uniform] = self._uniform_constrained(selected[uniform], v_min, v_max, current_limit)
        return out, voltage

    def _uniform_constrained(
        self, steps: np.ndarray, v_min: float, v_max: float, current_limit: float
    ) -> tuple[np.ndarray, np.ndarray]:
        positions = np.searchsorted(self.uniform, steps)
        params = [p[positions] for p in self._uniform_params]
        scale = self.scale[steps]
        v_mp, i_mp = self.v_mp[steps], self.i_mp[steps]
        result = self.mpp[steps].copy()
        operating_v = v_mp.copy()
        bound = (v_mp < v_min) | (v_mp > v_max) | (i_mp > current_limit)
        if not bound.any():
            return result, operating_v
        # One module's curve: the best point is the nearest edge of what the limits allow.
        p = [x[bound] for x in params]
        s = scale[bound]
        with np.errstate(divide="ignore", invalid="ignore"):
            module_limit = np.where(s > 0, current_limit / s, np.inf)
        at_limit = np.zeros(int(bound.sum()))
        binding = np.isfinite(module_limit) & (module_limit < p[0])
        if binding.any():
            at_limit[binding] = self.modules * np.nan_to_num(
                _quiet(pvlib.pvsystem.v_from_i, module_limit[binding], *[x[binding] for x in p])
            ).clip(min=0.0)
        lower = np.maximum(v_min, at_limit)
        voltage = np.clip(v_mp[bound], lower, v_max)
        current = s * np.clip(np.nan_to_num(_quiet(pvlib.pvsystem.i_from_v, voltage / self.modules, *p)), 0.0, None)
        result[bound] = np.where(lower > v_max, 0.0, np.minimum(voltage * current, result[bound]))
        operating_v[bound] = np.where(result[bound] > 0, voltage, 0.0)
        return result, operating_v

    def current(self, voltage: np.ndarray, rows: np.ndarray, signed: bool = False) -> np.ndarray:
        """The string's current at terminal ``voltage`` on the steps ``rows``, a boolean mask.

        ``voltage`` has one row per selected step and any further columns. The
        current is never negative unless ``signed``, as when solving where
        strings in parallel cancel at open circuit.
        """
        selected = np.flatnonzero(rows)
        out = np.zeros(voltage.shape)
        if not selected.size:
            return out
        uniform = np.isin(selected, self.uniform)
        if uniform.any():
            positions = np.searchsorted(self.uniform, selected[uniform])
            shape = (-1,) + (1,) * (voltage.ndim - 1)
            params = [p[positions].reshape(shape) for p in self._uniform_params]
            scale = self.scale[selected[uniform]].reshape(shape)
            module = np.nan_to_num(
                np.asarray(_quiet(pvlib.pvsystem.i_from_v, voltage[uniform] / self.modules, *params), dtype=float)
            )
            out[uniform] = scale * (module if signed else np.clip(module, 0.0, None))
        mismatched = np.isin(selected, self.mismatched)
        if mismatched.any():
            positions = np.searchsorted(self.mismatched, selected[mismatched])
            table_v, table_i = self._table_v[positions], self._table_i[positions]
            scale = self.scale[selected[mismatched]]
            target = voltage[mismatched]
            columns = target.reshape(len(positions), -1)
            values = np.stack(
                [_table_current(table_v, table_i, columns[:, j], signed) for j in range(columns.shape[1])], 1
            )
            out[mismatched] = (scale[:, None] * values).reshape(target.shape)
        return out

    def own_mean_power(self) -> np.ndarray:
        """The string's power at its maximum power point with every module at the string's mean irradiance, bare."""
        steps = len(self.scale)
        power = np.zeros(steps)
        rows = np.flatnonzero((self.mean_irradiance > 0) & (self.scale > 0))
        if rows.size:
            params = self.parts.diode_parameters(self.mean_irradiance[rows], rows)
            mpp = _quiet(_newton_mpp, *params)
            power[rows] = self.modules * self.scale[rows] * np.nan_to_num(np.asarray(mpp["p_mp"], dtype=float))
        return power


def _table_current(table_v: np.ndarray, table_i: np.ndarray, voltage: np.ndarray, signed: bool = False) -> np.ndarray:
    """Each row's current at ``voltage`` on its curve, the voltages falling along the row.

    Above the open-circuit voltage the current is 0, or, when ``signed``,
    the curve's first segment extended into negative current.
    """
    points = table_v.shape[1]
    above = (table_v > voltage[:, None]).sum(axis=1)
    upper = np.clip(above, 1, points - 1)
    lower = upper - 1
    index = np.arange(len(voltage))
    v0, v1 = table_v[index, lower], table_v[index, upper]
    i0, i1 = table_i[index, lower], table_i[index, upper]
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = np.clip(np.where(v0 != v1, (v0 - voltage) / (v0 - v1), 0.0), 0.0, 1.0)
        current = i0 + weight * (i1 - i0)
        if signed:
            # Above open circuit, extend the curve's first segment of falling
            # voltage into negative current.
            first = np.argmax(table_v < table_v[:, :1] - 1e-9, axis=1)
            vf, i_first = table_v[index, first], table_i[index, first]
            slope = np.where(first > 0, (i_first - table_i[:, 0]) / (vf - table_v[:, 0]), 0.0)
            extended = table_i[:, 0] + slope * (voltage - table_v[:, 0])
            current = np.where(above == 0, extended, current)
        else:
            # The reverse-current part of the table lies above open circuit.
            current = np.clip(np.where(above == 0, 0.0, current), 0.0, None)
    return np.where(above >= points, table_i[:, -1], current)
