"""Compile the shared dispatch functions with Numba.

Nothing here is dispatch logic. :func:`breos._dispatch._dispatch_day` and the
scalar helpers it calls are the functions the Python backend runs; this module
registers the helpers as jitable and compiles the day loop from the same
function object, so the two backends execute one source.

The day loop lives at module scope in :mod:`breos._dispatch` rather than in a
factory for one reason: Numba's on-disk cache. Its index key for a function
defined inside another function includes the closure cell contents, which are
not stable across processes, so such a function misses the cache in every new
process and appends a fresh ``.nbc`` entry each time.

The cache is keyed on the day loop's own source file, so every compiled
helper, the inverter cores included, lives in :mod:`breos._dispatch`. Numba
freezes module globals at compile time, and constants imported from
:mod:`breos.constants` are not part of that key: after editing one, clear
``breos/__pycache__/_dispatch.*.nb[ic]`` (or the ``NUMBA_CACHE_DIR``), or the
compiled backend keeps the old value and the parity tests will say so.

Importing this module requires Numba. :mod:`breos._numba_dispatch` must stay
importable without it -- that is where the availability check and the friendly
error live -- so this module is imported lazily from there, never at its top.

``fastmath=False`` is deliberate: enabling it would let LLVM reassociate and
contract the arithmetic and would break bit identity with the Python backend.
"""

from __future__ import annotations

from numba import njit
from numba.extending import register_jitable

from breos import _dispatch

for _helper in (
    _dispatch._dc_ac,
    _dispatch._dc_for_ac,
    _dispatch.lfp_capacity_factor,
    _dispatch.compute_cell_temperature,
    _dispatch._apply_capacity_window,
    _dispatch._charge,
    _dispatch._grid_charge,
    _dispatch._combined_conversion,
    _dispatch._dispatch_dc_step,
):
    register_jitable(fastmath=False)(_helper)

_dispatch_day_kernel = njit(cache=True, fastmath=False, nogil=True)(_dispatch._dispatch_day)
