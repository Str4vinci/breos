"""The sun positions and clear-sky frames computed last, by everything they depend on.

The sun is most of a PV run's time, and the optimizer, the terrain horizon,
sweeps and Monte Carlo ask for the same grid again and again: Monte Carlo
restamps every weather year onto one calendar, so each year's resample and
PV model ask for the same instants. At most about 20 MB in all: 45 hourly
years or 11 quarter-hourly ones of sun positions, the oldest dropped first.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import Any, Callable

import pandas as pd
from pvlib.location import Location

_FRAMES: OrderedDict[tuple, pd.DataFrame] = OrderedDict()
_LOCK = threading.Lock()


def _new_lock_after_fork() -> None:
    # A worker forked while another thread held the lock would inherit it held.
    global _LOCK
    _LOCK = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_new_lock_after_fork)
_MAX_ROWS = 400_000
# The Location methods each kind runs: get_clearsky asks the site for its
# sun position and air mass when it is not given them.
_DEPENDS_ON = {
    "solar_position": ("get_solarposition",),
    "clear_sky": ("get_clearsky", "get_solarposition", "get_airmass"),
}
_PVLIB_METHODS = {name: getattr(Location, name) for names in _DEPENDS_ON.values() for name in names}


def clear() -> None:
    """Forget every kept frame."""
    with _LOCK:
        _FRAMES.clear()


def _key(kind: str, location: Any, times: pd.DatetimeIndex) -> tuple | None:
    """What pvlib's ``kind`` frame at ``times`` depends on, or None if it is not kept.

    Only a plain pvlib ``Location`` whose methods are pvlib's own is kept: a
    subclass or a replaced method may compute something else for the site.
    pvlib reads the instants in UTC, the calendar dates of the clear sky
    included. The clear-sky key still holds the time zone, so a pvlib that
    reads local dates cannot be served another zone's frame.
    """
    if type(location) is not Location or any(
        getattr(Location, name) is not _PVLIB_METHODS[name] or name in vars(location) for name in _DEPENDS_ON[kind]
    ):
        return None
    return (
        kind,
        location.latitude,
        location.longitude,
        location.altitude,
        str(times.tz) if kind == "clear_sky" else None,
        times.as_unit("ns").asi8.tobytes(),  # type: ignore[attr-defined]  # pandas-stubs omits asi8
    )


def sun_frame(kind: str, location: Any, times: pd.DatetimeIndex, compute: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    """Return a copy of ``compute()``, the ``kind`` frame of ``location`` at ``times``, kept for reuse.

    ``kind`` is ``"solar_position"`` (``location.get_solarposition(times)``)
    or ``"clear_sky"`` (``location.get_clearsky(times)``, with pvlib's
    default model, optionally given the site's own solar position at
    ``times``). ``compute`` must return exactly that; it runs only when no
    equal request is kept. The copy is indexed by ``times`` itself, so the
    same instants asked for on another clock come back on that clock.
    """
    key = _key(kind, location, times)
    frame = None
    if key is not None:
        with _LOCK:
            frame = _FRAMES.get(key)
            if frame is not None:
                _FRAMES.move_to_end(key)
    if frame is None:
        frame = compute()
        if key is not None and len(times) <= _MAX_ROWS:
            with _LOCK:
                _FRAMES[key] = frame
                while sum(len(kept) for kept in _FRAMES.values()) > _MAX_ROWS:
                    _FRAMES.popitem(last=False)
    copy = frame.copy()
    # A new Index object, so renaming the returned index touches neither the
    # caller's times nor the kept frame.
    copy.index = times.copy()
    return copy
