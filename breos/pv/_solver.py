"""Checked single-diode solves and per-run recovery counts."""

from __future__ import annotations

import warnings
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pvlib
from scipy.optimize import brentq

DARK_FRACTION = 1e-9


def floor_photocurrent(photocurrent, reference):
    """Treat photocurrent below a fraction of the module's reference as dark."""
    photo = np.asarray(photocurrent, dtype=float)
    return np.where(photo < DARK_FRACTION * reference, 0.0, photo)


@dataclass
class SolverDiagnostics:
    """Distinct recovered steps, overall and within each solve stage."""

    stages: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    def result(self) -> dict[str, Any]:
        return {
            "recovered_steps": len(set().union(*self.stages.values())),
            "by_stage": {stage: len(steps) for stage, steps in sorted(self.stages.items())},
        }


_diagnostics: ContextVar[SolverDiagnostics | None] = ContextVar("solver_diagnostics", default=None)
_location: ContextVar[dict[str, Any]] = ContextVar("solver_location", default={})


@contextmanager
def solver_scope(*, diagnostics: SolverDiagnostics | None = None, **location: Any):
    """Supply the run's collector and any known equipment and step indices."""
    token = _location.set({**_location.get(), **location})
    collector = _diagnostics.set(diagnostics) if diagnostics is not None else None
    try:
        yield
    finally:
        _location.reset(token)
        if collector is not None:
            _diagnostics.reset(collector)


def _quiet(function, *args, **kwargs):
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        return function(*args, **kwargs)


def solver_location() -> dict[str, Any]:
    """Retain equipment context for curves evaluated after construction."""
    return _location.get()


def _reverse_voltage(current, photo, saturation, series, shunt, diode):
    """Brent's voltage root at or above open circuit, outside pvlib's default bracket."""
    result = np.empty(np.shape(current))
    for k in range(result.size):
        il, io, rs, rsh, n = (p[k] for p in (photo, saturation, series, shunt, diode))
        amps = current[k]
        upper = n * np.log1p((il - amps) / io)

        def residual(junction):
            return il - io * np.expm1(junction / n) - junction / rsh - amps

        # The analytic endpoint can round just below the root.
        for _ in range(8):
            if residual(upper) <= 0.0:
                break
            upper = np.nextafter(upper * (1.0 + 1e-8), np.inf)
        else:
            raise ValueError("Could not bracket reverse-current voltage")
        result[k] = brentq(residual, 0.0, upper) - amps * rs
    return result


def _step_label(flat, shape, steps, location):
    row = np.unravel_index(flat, shape)[0] if shape else 0
    labels = steps if steps is not None else location.get("steps")
    label = labels[row] if labels is not None else row
    timeline = location.get("steps")
    if steps is not None and timeline is not None and isinstance(label, (int, np.integer)):
        label = timeline[label]
    return str(label)


def _problem(message, stage, bad, diode, steps, location):
    location = {**_location.get(), **(location or {})}
    flat = np.flatnonzero(bad)[0]
    label = _step_label(flat, np.shape(bad), steps, location)
    details = ", ".join(
        f"{name}={p.flat[flat]!r}"
        for name, p in zip(("photocurrent", "saturation_current", "Rs", "Rsh", "nNsVth"), diode, strict=True)
    )
    equipment = ", ".join(f"{k}={v}" for k, v in location.items() if k != "steps")
    equipment = f", {equipment}" if equipment else ""
    return ValueError(f"{message} at {stage}, step {label}{equipment}: {details}")


def _solve(kind, params, value=0.0, *, stage, steps=None, feasible=True, location=None):
    """Keep valid primary values; retry only failed lit points with Brent's method.

    IV curves may carry negative current or voltage. Their caller supplies
    ``feasible`` where a bypass clamp replaces the diode's forward curve.
    Dark parts retain their IV curve, including reverse current. Their roots
    may also need bracketing, but never count as numerical recoveries.
    """
    photo = np.asarray(params[0])
    nonfinite = ~np.isfinite(photo)
    for parameter in params[1:]:
        nonfinite = nonfinite | ~np.isfinite(parameter)
    bad_inputs = nonfinite & ((photo > 0) | ~np.isfinite(photo))
    if bad_inputs.any():
        diode = np.broadcast_arrays(*params)
        raise _problem("Electrical input has non-finite diode parameters", stage, bad_inputs, diode, steps, location)

    function = {
        "mpp": pvlib.pvsystem.max_power_point,
        "current": pvlib.pvsystem.i_from_v,
        "voltage": pvlib.pvsystem.v_from_i,
    }[kind]

    def call(method, selected=None):
        args = params if selected is None else [p[selected] for p in diode]
        if kind != "mpp":
            args = [value if selected is None else arrays[5][selected], *args]
        try:
            if kind == "voltage" and method == "brentq" and np.any(np.asarray(args[0]) <= 0):
                reverse = np.asarray(args[0]) <= 0
                result = np.empty(reverse.shape)
                result[reverse] = _quiet(_reverse_voltage, *[p[reverse] for p in args])
                if (~reverse).any():
                    result[~reverse] = _quiet(function, *[p[~reverse] for p in args], method=method)
            else:
                result = _quiet(function, *args, method=method)
            if kind == "mpp":
                return {k: np.asarray(result[k], dtype=float) for k in ("p_mp", "v_mp", "i_mp")}
            return np.asarray(result, dtype=float)
        except (RuntimeError, ValueError, OverflowError):
            size = (
                np.broadcast_shapes(*(np.shape(p) for p in params), *([] if kind == "mpp" else [np.shape(value)]))
                if selected is None
                else (int(selected.sum()),)
            )
            empty = np.full(size, np.nan)
            return {k: empty.copy() for k in ("p_mp", "v_mp", "i_mp")} if kind == "mpp" else empty

    def invalid(result, selected=None):
        p = [np.asarray(p) for p in params] if selected is None else [p[selected] for p in diode]
        if kind != "mpp":
            query = np.asarray(value) if selected is None else arrays[5][selected]
            volts, amps = (query, result) if kind == "current" else (result, query)
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                junction = np.asarray(volts + amps * p[2])
                leakage = junction / p[3]
                np.divide(junction, p[4], out=junction)
                np.exp(junction, out=junction)
                junction *= p[1]
                # Scale by the equation's terms, not by the small difference
                # left after cancellation. The floor allows 1e-8 A roundoff.
                scale = np.asarray(np.maximum(np.maximum(np.abs(p[0]), np.abs(amps)), np.abs(junction)))
                np.maximum(scale, np.abs(leakage), out=scale)
                np.maximum(scale, 0.1, out=scale)
                # Reuse the diode-current buffer for the residual.
                junction += leakage
                junction += amps
                junction -= p[0]
                junction -= p[1]
                off_curve = ~np.isfinite(junction)
                np.abs(junction, out=junction)
                scale *= 1e-7
                off_curve |= junction > scale
            tolerance = 1e-7 * np.maximum(1.0, np.abs(p[0]))
            if kind == "current":
                return (
                    ~np.isfinite(result)
                    | off_curve
                    | ((query >= 0) & (result > p[0] + tolerance))
                    | ((query <= 0) & (result < -tolerance))
                )
            with np.errstate(divide="ignore", invalid="ignore"):
                upper = p[4] * np.log1p(p[0] / p[1])
            return (
                ~np.isfinite(result)
                | off_curve
                | ((query <= 0) & (result < -tolerance))
                | ((query >= 0) & (result > upper + 1e-7 * np.maximum(1.0, np.abs(upper))))
            )
        power, voltage, current = (result[k] for k in ("p_mp", "v_mp", "i_mp"))
        # Positive I and V on the diode equation imply V <= Voc and I <=
        # Isc. Analytic upper bounds and the residual check avoid two extra
        # endpoint solves on every valid MPP, including vanishing light.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            voc = p[4] * np.log1p(p[0] / p[1])
            isc = p[0] / (1.0 + p[2] / p[3])
            junction = voltage + current * p[2]
            on_curve = p[0] - p[1] * np.expm1(junction / p[4]) - junction / p[3]
        tolerance = 1e-7
        return (
            ~np.isfinite(power + voltage + current + voc + isc)
            | (power < -1e-8)
            | (voltage < -1e-8)
            | (current < -1e-8)
            | ((p[0] > 1e-8) & ((power <= 0) | (voltage <= 0) | (current <= 0)))
            | (voltage > voc + tolerance * np.maximum(1.0, np.abs(voc)))
            | (current > isc + tolerance * np.maximum(1.0, np.abs(isc)))
            | ~np.isclose(power, voltage * current, rtol=tolerance, atol=1e-8)
            | ~np.isclose(current, on_curve, rtol=tolerance, atol=1e-8)
        )

    result = call("newton" if kind == "mpp" else "lambertw")
    if kind == "mpp":
        result = {key: np.where(photo == 0.0, 0.0, column) for key, column in result.items()}
    else:
        # Both dark open circuit and dark short circuit are exactly zero.
        result = np.where((photo == 0.0) & (np.asarray(value) == 0.0), 0.0, result)
    bad_primary = np.asarray(invalid(result))
    if not bad_primary.any():
        return result
    shape = np.shape(result["p_mp"] if kind == "mpp" else result)
    if kind == "voltage":
        dark_reverse = bad_primary & (photo == 0.0) & (np.asarray(value) < 0.0)
        if dark_reverse.any():
            arrays = np.broadcast_arrays(*params, value)
            diode = arrays[:5]
            p = [p[dark_reverse] for p in diode]
            amps = arrays[5][dark_reverse]
            # The ideal-diode endpoint is also the root for infinite Rsh;
            # at the dark floor its leakage is usually negligible. Check it
            # before paying for scalar bracketing on those table points.
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                candidate = p[4] * np.log1p(-amps / p[1]) - amps * p[2]
            result[dark_reverse] = candidate
            bad_primary[dark_reverse] = invalid(candidate, dark_reverse)
    active = (photo > 0) if kind == "mpp" else (photo >= 0)
    failed = bad_primary & active & np.broadcast_to(feasible, shape)
    if failed.any():
        arrays = np.broadcast_arrays(*params, *([] if kind == "mpp" else [value]))
        diode = arrays[:5]
        # A dark part may have no finite forward root (for example with
        # infinite Rsh). It must not spoil a lit point's independent retry.
        for lit in (True, False):
            selected = failed & ((photo > 0.0) if lit else (photo == 0.0))
            if not selected.any():
                continue
            fallback = call("brentq", selected)
            bad = invalid(fallback, selected)
            positions = np.flatnonzero(selected)
            if lit and bad.any():
                bad_points = np.zeros(shape, dtype=bool)
                bad_points.flat[positions[np.flatnonzero(bad)]] = True
                raise _problem("Electrical solver failed", stage, bad_points, diode, steps, location)
            if kind == "mpp":
                for key in result:
                    result[key][selected] = fallback[key]
            else:
                result[selected] = fallback
            collector = _diagnostics.get()
            if lit and collector is not None:
                context = {**_location.get(), **(location or {})}
                collector.stages[stage].update(_step_label(flat, shape, steps, context) for flat in positions)
    if kind == "mpp":
        return {key: np.nan_to_num(column, copy=False, posinf=0.0, neginf=0.0) for key, column in result.items()}
    return np.nan_to_num(result, copy=False, posinf=0.0, neginf=0.0)


def mpp(*params, stage="mpp", **kwargs):
    return _solve("mpp", params, stage=stage, **kwargs)


def current(voltage, *params, stage="current", **kwargs):
    return _solve("current", params, voltage, stage=stage, **kwargs)


def voltage(current, *params, stage="voltage", **kwargs):
    return _solve("voltage", params, current, stage=stage, **kwargs)
