"""The IV curve of a series string whose modules are shaded unevenly (design record 0008, level 2).

Each module splits into bypass-diode substrings. A substring is its cells in
series, its shade concentrated: as many cells as its area mean allows at the
irradiance its limiting shaded fraction leaves, one cell at the remainder of
the mean, and the others clear. A shadow that leaves at most one cell partly
shaded, every other cell lit or dark, is described exactly. In a module of several cell strings per substring, such as a
half-cut one (design record 0009), each cell position is that many cells in
parallel; strings shaded alike behave as one string of positions, and
strings shaded differently share the substring's voltage and add their
currents. Each part is a single-diode curve with its share of the module's
series and shunt resistance and diode voltage, used in forward bias. A part
carries no more than its own short-circuit current: the module's shunt is a
fit to the forward curve, and split per cell it would pass amperes in
reverse bias, so above that current the part's voltage falls without limit
and the bypass diode conducts, holding the substring at the diode's forward
voltage, unless the module states its cells' breakdown voltage: past its
short-circuit current a part then holds at its cells' summed breakdown
voltage. Reverse leakage below breakdown is not modelled. The
string carries one current through every substring, and its voltage is
their sum, less the cable's drop.

Steps where every substring of the string sees the same light use the
module's own curve exactly; only the others build the string's curve, on a
grid of currents refined to its global maximum power point.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

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
# Relative distance of the grid points either side of a part's short-circuit
# current, on its own curve below it and on the bypass diode above it.
_EDGE = 1e-7
# Just above a short-circuit current of 0, where a dark cell breaks down at
# any current: a relative step alone would sample 0 again.
_ABSOLUTE_EDGE = 1e-9
# A substring whose parallel strings are shaded differently is tabled: each
# string's voltage on a grid of its own currents, from reverse current up to
# its cap, and the substring's current on a grid of voltages.
_STRING_POINTS = 240
_SECTION_POINTS = 320


def _concentrated(
    cells: float, mean: np.ndarray, limiting: np.ndarray, limited: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """A chain of ``cells`` cells in three parts: each part's cell count and shaded fraction.

    Where ``limited``, ``floor(cells * mean / limiting)`` cells are at the
    limiting fraction, one cell at what is left of the mean and the rest
    clear; elsewhere every cell is at the mean. Where no cell is clear, the
    remainder's cell is split between the last two parts, so no part is
    empty and every count stays positive.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        dark = np.floor(np.where(limited, cells * mean / limiting, cells - 1.0) + 1e-9)
    dark = np.clip(dark, 1.0, cells - 1.0)
    remainder = np.where(limited, np.clip(cells * mean - dark * limiting, 0.0, 1.0), mean)
    clear = cells - dark - 1.0
    empty = clear < 0.5
    counts = [dark, np.where(empty, 0.5, 1.0), np.where(empty, 0.5, clear)]
    fractions = [np.where(limited, limiting, mean), remainder, np.where(limited & ~empty, 0.0, remainder)]
    return counts, fractions


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
        diodes = parts.pv_params.bypass_diodes
        parallel = parts.pv_params.cell_strings_in_parallel
        if shading.bypass_diodes is None:
            shading = shading.for_module(diodes, parallel)
            problem = shading.inconsistency(parts.pv_params.N_Cells)
            if problem:
                raise ValueError(f"module_shading: {problem}")
        elif (shading.bypass_diodes, shading.cell_strings_in_parallel) != (diodes, parallel):
            raise ValueError(
                f"module_shading is bound to {shading.bypass_diodes} bypass diodes of "
                f"{shading.cell_strings_in_parallel} cell strings; the module has {diodes} of {parallel}"
            )
        self.parts = parts
        # A shaded cell's breakdown voltage, or -inf: it then carries at most
        # its short-circuit current and its bypass diode takes the substring.
        breakdown = parts.pv_params.breakdown_voltage_v
        self.breakdown_v = -np.inf if breakdown is None else float(breakdown)
        self.scale = np.asarray(scale, dtype=float)
        self.modules = modules
        self.cable_ohm = float(cable_ohm)
        self.bypass_v = float(bypass_diode_voltage_v)
        steps = len(self.scale)
        # Cell positions in series across the module: each is the module's
        # parallel cell strings' cells side by side.
        self.strings = shading.cell_strings_in_parallel
        cells = float(parts.pv_params.N_Cells) / self.strings
        diodes = shading.diodes
        self.cells = cells
        per_substring = cells / diodes

        weights = shading.axes.grid_weights(parts.sun_azimuth, parts.sun_elevation)
        mean, limiting = shading.substring_fractions(weights)
        view = shading.sky_view_factors[:, None, None]
        self._sky_view = shading.sky_view_factors

        def irradiance(fraction: np.ndarray) -> np.ndarray:
            return parts.irradiance(fraction, view)

        # Substrings whose parallel strings are shaded differently.
        fractions = shading.half_fractions(weights)
        halved = np.zeros(mean.shape, dtype=bool)
        self._halves: tuple[np.ndarray, ...] | None = None
        if fractions is not None:
            half_mean, half_limiting = fractions
            halved = (half_mean != half_mean[:, :, :1]).any(axis=2) | (half_limiting != half_limiting[:, :, :1]).any(
                axis=2
            )
            self._halves = (parts.irradiance(half_mean, view[..., None]), half_mean, half_limiting)
            # Halves shaded alike are one substring of their shared fractions.
            limiting = np.where(halved, limiting, half_limiting[:, :, 0])

        e_mean = irradiance(mean)
        limited = (limiting > mean) & (per_substring > 1.0)
        counts, part_fractions = _concentrated(per_substring, mean, limiting, limited)
        e_parts = [irradiance(fraction) for fraction in part_fractions]

        lit = (e_mean.max(axis=(0, 1)) > 0) & (self.scale > 0)
        same = np.all(e_mean == e_mean[:1, :1], axis=(0, 1)) & ~limited.any(axis=(0, 1)) & ~halved.any(axis=(0, 1))
        self._halved = halved
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
            self._substrings = self._substring_elements(shading, e_mean, e_parts, counts, limited, per_substring, cells)
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
        e_parts: list[np.ndarray],
        counts: list[np.ndarray],
        limited: np.ndarray,
        per_substring: float,
        cells: float,
    ) -> list[_Substring]:
        """Each distinct substring's count and its parts' single-diode parameters on the mismatched steps."""
        rows = self.mismatched
        groups: dict[tuple[Any, ...], list[tuple[int, int]]] = {}
        for m, module in enumerate(shading.modules):
            for k in range(shading.diodes):
                halves = module.substring_halves(k)
                key = (
                    id(module.mean[k]),
                    id(module.limiting[k]),
                    module.sky_view_factor,
                    None if halves is None else tuple(id(t) for pair in halves for t in pair),
                )
                groups.setdefault(key, []).append((m, k))

        def element(irradiance: np.ndarray, share: Any) -> tuple[np.ndarray, ...]:
            photo, saturation, series, shunt, diode = self.parts.diode_parameters(irradiance, rows)
            params = (photo, saturation, series * share, shunt * share, diode * share)
            # The most current the part carries before it breaks down or needs its bypass diode.
            # A dark part's can come out a roundoff below 0, which would break it
            # down at open circuit.
            short_circuit = np.maximum(
                np.nan_to_num(np.asarray(_quiet(pvlib.pvsystem.i_from_v, 0.0, *params), dtype=float)), 0.0
            )
            return params + (short_circuit, np.broadcast_to(share * cells * self.breakdown_v, rows.shape))

        # Most substrings are lit like the brightest on most steps: they share
        # its curve there, and only the others are solved.
        brightest = np.max(np.where(limited[:, :, rows], -np.inf, e_mean[:, :, rows]), axis=(0, 1))
        brightest = np.where(np.isfinite(brightest), brightest, e_mean[:, :, rows].max(axis=(0, 1)))
        self._reference = [element(brightest, per_substring / cells)]
        substrings: list[_Substring] = []
        # Each part's photocurrent: the curve's knees lie just below them.
        photocurrents = [self._reference[0][0]]
        # Each part's short-circuit current, where its substring drops to the bypass diode.
        short_circuits = [self._reference[0][5]]
        weights, merged = self._merge_alike(
            list(groups.values()), e_mean, e_parts, counts, limited, brightest, per_substring
        )
        for members, weight, absorbed in zip(groups.values(), weights, merged, strict=True):
            m, k = members[0]
            # Only the steps whose parallel strings are shaded differently need
            # a half section; on the others the halves are one substring.
            halved = self._halved[m, k, rows]
            split = halved & ~absorbed
            own = (limited[m, k, rows] | (e_mean[m, k, rows] != brightest)) & ~halved & ~absorbed
            if limited[m, k, rows].any():
                # Unlimited steps split the substring the same way at one irradiance.
                parts = [element(e[m, k, rows], c[m, k, rows] / cells) for e, c in zip(e_parts, counts, strict=True)]
            else:
                parts = [element(e_mean[m, k, rows], per_substring / cells)]
            photocurrents.extend(p[0] for p in parts)
            short_circuits.extend(p[5] for p in parts)
            section = None
            if halved.any():
                # Built on every step its halves differ, merged or not: its
                # table and the grid's points then do not depend on merging.
                section = self._half_section(m, k, rows[halved], per_substring, cells)
                # The section's knees and caps on its own steps; elsewhere the
                # first part's repeat a point the grid already holds.
                for values, fill, into in (
                    (section.knees, parts[0][0], photocurrents),
                    (section.caps, parts[0][5], short_circuits),
                ):
                    for value in values:
                        spread = np.array(fill, dtype=float)
                        spread[halved] = value
                        into.append(spread)
            substrings.append(_Substring(weight, parts, own, section, split, np.cumsum(halved) - 1))
        self._photocurrents = np.asarray(photocurrents)
        self._short_circuits = np.asarray(short_circuits)
        return substrings

    def _merge_alike(
        self,
        groups: list[list[tuple[int, int]]],
        e_mean: np.ndarray,
        e_parts: list[np.ndarray],
        counts: list[np.ndarray],
        limited: np.ndarray,
        brightest: np.ndarray,
        per_substring: float,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Each group's weight on the mismatched steps, and the steps another group alike solves for it.

        On a step where two groups are built from the same inputs their
        curves are one: the first solves it at both groups' weight and the
        other's weight there is 0. Steps on the brightest curve are left as
        they are; they cost nothing to share. Whether a group's parts and
        each of its half section's chains are concentrated is decided over
        all its steps, so groups merge only where that choice agrees too.
        """
        rows = self.mismatched

        def inputs(m: int, k: int) -> np.ndarray:
            arrays = [e_mean[m, k, rows], limited[m, k, rows], self._halved[m, k, rows]]
            arrays += [e[m, k, rows] for e in e_parts] + [c[m, k, rows] for c in counts]
            if self._halves is not None:
                arrays += [table[m, k, h, rows] for table in self._halves for h in range(self.strings)]
            arrays.append(np.broadcast_to(self._sky_view[m], rows.shape))
            arrays.append(np.broadcast_to(limited[m, k, rows].any(), rows.shape))
            if self._halves is not None:
                halved = self._halved[m, k, rows]
                _, half_mean, half_limiting = self._halves
                for h in range(self.strings):
                    lim = (half_limiting[m, k, h, rows] > half_mean[m, k, h, rows]) & (per_substring > 1.0)
                    arrays.append(np.broadcast_to((lim & halved).any(), rows.shape))
            return np.stack([np.asarray(a, dtype=float) for a in arrays])

        firsts = [members[0] for members in groups]
        features = [inputs(m, k) for m, k in firsts]
        solved = [limited[m, k, rows] | (e_mean[m, k, rows] != brightest) | self._halved[m, k, rows] for m, k in firsts]
        weights = [np.full(rows.size, float(len(members))) for members in groups]
        merged = [np.zeros(rows.size, dtype=bool) for _ in groups]
        for h in range(len(groups)):
            for g in range(h):
                alike = solved[h] & ~merged[h] & ~merged[g] & np.all(features[g] == features[h], axis=0)
                weights[g][alike] += weights[h][alike]
                merged[h] |= alike
            weights[h][merged[h]] = 0.0
        return weights, merged

    def _half_section(self, m: int, k: int, rows: np.ndarray, per_substring: float, cells: float) -> _HalfSection:
        """Substring ``k`` of module ``m``, whose parallel strings are shaded differently, on the steps ``rows``.

        Each string's shade is concentrated as a substring's is, with one
        cell per position: a position's share of the
        module's current and diode saturation, and the string count times
        its share of the resistances.
        """
        assert self._halves is not None
        e_half_mean, half_mean, half_limiting = self._halves
        strings = self.strings

        def element(irradiance: np.ndarray, count: Any) -> tuple[np.ndarray, ...]:
            photo, saturation, series, shunt, diode = self.parts.diode_parameters(irradiance, rows)
            share = count / cells
            params = (
                photo / strings,
                saturation / strings,
                series * share * strings,
                shunt * share * strings,
                diode * share,
            )
            short_circuit = np.maximum(
                np.nan_to_num(np.asarray(_quiet(pvlib.pvsystem.i_from_v, 0.0, *params), dtype=float)), 0.0
            )
            return params + (short_circuit, np.broadcast_to(count * self.breakdown_v, rows.shape))

        chains = []
        for h in range(strings):
            mean = half_mean[m, k, h, rows]
            limit = half_limiting[m, k, h, rows]
            lim = (limit > mean) & (per_substring > 1.0)
            if lim.any():
                counts, fractions = _concentrated(per_substring, mean, limit, lim)
                chains.append(
                    [
                        element(self._part_irradiance(m, rows, fraction), count)
                        for fraction, count in zip(fractions, counts, strict=True)
                    ]
                )
            else:
                chains.append([element(e_half_mean[m, k, h, rows], per_substring)])
        return _HalfSection(chains, self.bypass_v, float(self.parts.pv_params.Isc) / strings)

    def _part_irradiance(self, m: int, rows: np.ndarray, shaded: np.ndarray) -> np.ndarray:
        """A cell of module ``m`` on the steps ``rows``: its irradiance at the beam ``shaded`` fraction."""
        fraction = np.zeros(len(self.scale))
        fraction[rows] = shaded
        return self.parts.irradiance(fraction, self._sky_view[m])[rows]

    def _substring_voltage(self, current: np.ndarray, parts: Any, rows: Any) -> np.ndarray:
        """One substring's voltage at ``current`` on the mismatched steps ``rows``, held by its bypass diode."""
        shape = (-1,) + (1,) * (current.ndim - 1)
        voltage = np.zeros(current.shape)
        for params in parts:
            voltage += _part_voltage(current, [p[rows].reshape(shape) for p in params])
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
        for substring in self._substrings:
            voltage = reference.copy()
            mine = np.flatnonzero(substring.own[select])
            if mine.size:
                voltage[mine] = self._substring_voltage(current[mine], substring.parts, select[mine])
            if substring.section is not None:
                cut = np.flatnonzero(substring.split[select])
                if cut.size:
                    voltage[cut] = substring.section.voltage(current[cut], substring.position[select[cut]])
            total += substring.weight[select].reshape((-1,) + (1,) * (current.ndim - 1)) * voltage
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
        # Every part's own knee, and both sides of its short-circuit current,
        # where its substring drops to the bypass diode: the table then reads
        # the voltage's jump there at that current.
        pieces.append((currents.T[:, :, None] * knee[None, None, :]).reshape(len(top), -1))
        pieces.append(self._short_circuits.T * (1.0 - _EDGE))
        above = self._short_circuits.T * (1.0 + _EDGE)
        pieces.append(above + _ABSOLUTE_EDGE if np.isfinite(self.breakdown_v) else above)
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
            # The power at ``current``; -inf where the limits do not allow it,
            # as where the voltage jumps across the range at a part's cap.
            voltage = self._terminal_voltage(current, select, cabled)
            allowed = (voltage >= v_min) & (voltage <= v_max) & (current <= current_limit)
            return np.where(allowed, scale * current * voltage, -np.inf)

        def bisect(low: np.ndarray, high: np.ndarray, target: float) -> tuple[np.ndarray, np.ndarray]:
            # The voltage falls with current: the currents just above and just
            # below ``target`` where it crosses or jumps across it.
            for _ in range(_BISECTION_STEPS):
                middle = 0.5 * (low + high)
                above = self._terminal_voltage(middle, select, cabled) > target
                low, high = np.where(above, middle, low), np.where(above, high, middle)
            return low, high

        for lo_k, hi_k, usable in brackets:
            if not usable.any():
                continue
            low, high = grid[rows, lo_k], grid[rows, hi_k]
            v_low, v_high = table[rows, lo_k], table[rows, hi_k]
            ok = usable & (v_high <= v_max) & (v_low >= v_min) & (low <= current_limit)
            if np.isfinite(v_max):
                cut = ok & (v_low > v_max)
                if cut.any():
                    low = np.where(cut, bisect(low.copy(), high.copy(), v_max)[1], low)
            if np.isfinite(v_min):
                cut = ok & (v_high < v_min)
                if cut.any():
                    high = np.where(cut, bisect(low.copy(), high.copy(), v_min)[0], high)
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


def _part_voltage(current: np.ndarray, params: Sequence[np.ndarray]) -> np.ndarray:
    """A part's voltage at ``current``: its forward curve up to its short-circuit current, its breakdown beyond.

    ``params`` are its single-diode parameters, its short-circuit current and
    its cells' summed breakdown voltage, broadcastable against ``current``.
    Without a breakdown voltage (-inf) the part carries no more than its
    short-circuit current.
    """
    *diode, short_circuit, breakdown = params
    part = _quiet(pvlib.pvsystem.v_from_i, current, *diode)
    part = np.nan_to_num(np.asarray(part, dtype=float), nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    return np.where(current > short_circuit, breakdown, part)


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


def _rowwise_interp(x: np.ndarray, y: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Row by row, ``y`` at ``query`` on the piecewise-linear curve through ``(x, y)``.

    ``x`` rises along each row; ``query`` has one row per row of ``x`` and
    any number of columns, and holds at the ends of its row's ``x``.
    """
    rows, points = x.shape
    span = float(np.nanmax(x) - np.nanmin(x)) + 1.0
    offset = (np.arange(rows, dtype=float) * 2.0 * span)[:, None]
    clipped = np.clip(query, x[:, :1], x[:, -1:])
    flat = (x + offset).ravel()
    # On a run of equal ``x`` this takes the first point's ``y``.
    position = np.searchsorted(flat, (clipped + offset).ravel(), side="left").reshape(query.shape)
    index = np.clip(position - 1 - (np.arange(rows) * points)[:, None], 0, points - 2)
    x0, x1 = np.take_along_axis(x, index, 1), np.take_along_axis(x, index + 1, 1)
    y0, y1 = np.take_along_axis(y, index, 1), np.take_along_axis(y, index + 1, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = np.where(x1 > x0, (clipped - x0) / (x1 - x0), 0.0)
    return y0 + weight * (y1 - y0)


class _Substring(NamedTuple):
    """Identical substrings of a string, on its mismatched steps.

    ``weight`` is how many substrings each step's curve stands for: their
    count, plus any substrings lit alike on that step, or 0 where another
    group solves it. ``parts`` are the substring's parts in series; ``own`` marks the steps
    they are solved on, the others sharing the brightest substring's curve.
    On the ``split`` steps its parallel strings are shaded differently and
    ``section`` holds it instead, at row ``position`` of its table.
    """

    weight: np.ndarray
    parts: list[tuple[np.ndarray, ...]]
    own: np.ndarray
    section: _HalfSection | None
    split: np.ndarray
    position: np.ndarray


class _HalfSection:
    """A substring of parallel cell strings shaded differently, on the mismatched steps it was built on.

    Each string is its parts in series and carries at most its smallest
    part's short-circuit current. The strings share the substring's voltage
    and add their currents; the bypass diode holds the substring at minus its
    forward voltage once their sum falls short of the string's current.
    """

    def __init__(self, chains: list[list[tuple[np.ndarray, ...]]], bypass_v: float, chain_isc: float) -> None:
        self.bypass_v = bypass_v
        rows = chains[0][0][0].shape[0]
        unit = np.unique(
            np.concatenate((np.linspace(0.0, 1.0, _STRING_POINTS - 64), 1.0 - np.geomspace(1e-7, 0.25, 40)))
        )
        # Above its open circuit a chain takes current back from the others,
        # up to what the brightest of them drives: at least a chain's
        # short-circuit current at STC (``chain_isc``), whatever its own light.
        caps = [np.min([p[5] for p in parts], axis=0) for parts in chains]
        reach = np.maximum(np.max(caps, axis=0), chain_isc)
        reverse = -1.5 * np.geomspace(1e-4, 1.0, 24)[::-1]
        tables = []
        knees = []
        brightest: list[np.ndarray] = []
        corners = []
        top = np.full(rows, -np.inf)
        for parts, cap in zip(chains, caps, strict=True):
            part_caps = np.asarray([p[5] for p in parts])
            breaks = np.isfinite(parts[0][6]).any()
            pieces = [reach[:, None] * reverse[None, :]]
            if breaks:
                # Past its short-circuit current a part holds at its breakdown
                # voltage, so the chain carries current up to its largest
                # part's, each part's jump tabled on both sides.
                pieces.append(part_caps.max(axis=0)[:, None] * unit[None, :])
                below = part_caps[:, :, None] * (1.0 - np.geomspace(1e-7, 0.25, 40))[None, None, :]
                above = part_caps[:, :, None] * (1.0 + np.array([_EDGE, 1e-4]))[None, None, :] + _ABSOLUTE_EDGE
                for edges in (below, above):
                    pieces.append(edges.transpose(1, 0, 2).reshape(rows, -1))
                current = np.sort(np.concatenate(pieces, axis=1), axis=1)
            else:
                pieces.append(cap[:, None] * unit[None, :])
                current = np.concatenate(pieces, axis=1)
            voltage = np.zeros(current.shape)
            if not breaks:
                for params in parts:
                    diode = [p[:, None] for p in params[:5]]
                    voltage += np.nan_to_num(np.asarray(_quiet(pvlib.pvsystem.v_from_i, current, *diode), dtype=float))
            else:
                for params in parts:
                    voltage += _part_voltage(current, [p[:, None] for p in params])
                # The voltages on both sides of each jump, for the substring's voltage grid.
                for part_cap in part_caps:
                    for edge in (part_cap * (1.0 - _EDGE), part_cap * (1.0 + _EDGE) + _ABSOLUTE_EDGE):
                        at = sum(_part_voltage(edge[:, None], [p[:, None] for p in q])[:, 0] for q in parts)
                        corners.append(at)
            # The voltage falls as the current rises: tabled against falling voltage, it rises.
            tables.append((-voltage, current))
            knees.append(np.min([p[0] for p in parts], axis=0))
            if breaks:
                brightest.append(np.max([p[0] for p in parts], axis=0))
            top = np.maximum(top, voltage[:, 0])
        # The substring's current over a grid of voltages from its bypass clamp up past open circuit.
        levels = np.concatenate(
            [np.linspace(0.0, 1.0, _SECTION_POINTS - 8 * len(chains))[None, :].repeat(rows, 0)]
            + [
                ((-t[0][:, [-1, -8, -16, -24, -40, -60, -90, -120]] + bypass_v) / (top + bypass_v)[:, None])
                for t in tables
            ],
            axis=1,
        )
        if corners:
            levels = np.concatenate(
                [levels] + [((np.asarray(c) + bypass_v) / (top + bypass_v))[:, None] for c in corners], axis=1
            )
        volts = np.sort(-bypass_v + np.clip(levels, 0.0, 1.0) * (top + bypass_v)[:, None], axis=1)
        total = np.zeros(volts.shape)
        for falling, current in tables:
            total += _rowwise_interp(falling, current, -volts)
        # Current rises as the voltage falls: the substring's curve against rising current.
        self._current = total[:, ::-1]
        self._voltage = volts[:, ::-1]
        # Where the diode takes over: the most the strings carry together;
        # with breakdown, also the substring's currents at each jump of a
        # string, so the string's own table samples both sides of them.
        self.caps = [self._current[:, -1]]
        if corners:
            at = np.clip(np.stack(corners, axis=1), -bypass_v, top[:, None])
            jumps = np.zeros(at.shape)
            for falling, current in tables:
                jumps += _rowwise_interp(falling, current, -at)
            self.caps.extend(jumps.T)
        self.knees = [sum(knees)] + ([sum(brightest)] if brightest else [])

    def voltage(self, current: np.ndarray, rows: Any) -> np.ndarray:
        """The substring's voltage at ``current`` on the steps ``rows`` of its table."""
        shape = current.shape
        table_i, table_v = self._current[rows], self._voltage[rows]
        query = current.reshape(table_i.shape[0], -1)
        voltage = _rowwise_interp(table_i, table_v, query)
        # Beyond what the strings carry together, the bypass diode takes the current.
        voltage = np.where(query > table_i[:, -1:], -self.bypass_v, voltage)
        return np.maximum(voltage, -self.bypass_v).reshape(shape)
