"""Optional compiled within-day dispatch kernel.

This is a private accelerator, not public API. It runs the dispatch day loop
of :mod:`breos._dispatch` compiled by Numba, for one degradation day at a
time, with state of health, resistance-derived efficiencies and the
replacement decision held fixed for the duration of the call. Everything
scientifically sensitive -- rainflow counting, calendar and cycle degradation,
resistance growth, replacement, and the state carried between days and years --
stays in the Python path and is never compiled.

The compiled function is the Python backend's own ``_dispatch_day``, not a
copy of it, and both backends pack their arguments through
``_day_arguments``. See :mod:`breos._numba_dispatch_kernels` for how it is
compiled and cached.
"""

from __future__ import annotations

from typing import Any, Tuple

from breos._dispatch import _day_arguments


class NumbaUnavailableError(ImportError):
    """Raised when the Numba backend is selected but the extra is not installed."""


def numba_available() -> bool:
    """Return whether the optional compiled backend can be imported."""
    try:
        import numba  # noqa: F401
    except ImportError:
        return False
    return True


def numba_versions() -> dict[str, str]:
    """Return the compiler versions a bit-identity claim is scoped to."""
    try:
        import llvmlite
        import numba
    except ImportError:
        return {"numba": "not installed", "llvmlite": "not installed"}
    return {"numba": numba.__version__, "llvmlite": llvmlite.__version__}


_JIT_CACHE_STATE: str | None = None


def reset_jit_cache_observation() -> None:
    """Start a new observation at the next compiled dispatch call."""
    global _JIT_CACHE_STATE
    _JIT_CACHE_STATE = None


def jit_cache_state() -> str | None:
    """Return the cache outcome observed by Numba for the current trajectory.

    Cache-file presence is not evidence of a hit. Numba can reject an index
    after a source or toolchain change, and a shared cache directory can hold
    entries from another checkout. The dispatch wrapper therefore reads the
    CPU dispatcher's hit and miss counters after its first call.
    """
    return _JIT_CACHE_STATE


def observed_jit_cache_state() -> str | None:
    """Return the cache outcome observed by the current worker process."""
    return jit_cache_state()


def _cache_event_count(events: Any) -> int:
    """Return the number of cache events in a Numba dispatcher counter."""
    return int(sum(events.values()))


def _build_kernel() -> Any:
    """Return the compiled dispatch kernel, importing the compiled module lazily.

    :mod:`breos._numba_dispatch_kernels` imports Numba at its top, so it is
    imported here rather than above: this module must stay importable without
    the optional dependency, because :func:`require_numba_dispatch_day` is
    what turns a missing Numba into a readable error.
    """
    from breos._numba_dispatch_kernels import _dispatch_day_kernel

    return _dispatch_day_kernel


_KERNEL: Any = None


def _kernel() -> Any:
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = _build_kernel()
    return _KERNEL


def _dispatch_day_numba(out: Any, *args: Any, **state: Any) -> Tuple[float, float, float]:
    """Run the compiled ``_dispatch_day``; takes the arguments of ``_day_arguments``."""
    global _JIT_CACHE_STATE

    kernel = _kernel()
    observe_cache = _JIT_CACHE_STATE is None
    if observe_cache:
        had_compiled_signature = bool(kernel.signatures)
        cache_hits_before = _cache_event_count(kernel.stats.cache_hits)
        cache_misses_before = _cache_event_count(kernel.stats.cache_misses)

    result = kernel(*_day_arguments(out, *args, **state))
    if observe_cache:
        cache_hits_after = _cache_event_count(kernel.stats.cache_hits)
        cache_misses_after = _cache_event_count(kernel.stats.cache_misses)
        if cache_misses_after > cache_misses_before:
            _JIT_CACHE_STATE = "cold"
        elif cache_hits_after > cache_hits_before or had_compiled_signature:
            _JIT_CACHE_STATE = "warm"
        else:
            # Telemetry about a run must never be able to destroy the run. The
            # counters read here are Numba internals with no stability
            # guarantee, so a future release can stop populating them; a study
            # that is hours old should not die because its provenance field
            # could not be filled in. "unknown" records honestly that the
            # observation failed, and the simulation continues -- its numbers
            # do not depend on this.
            _JIT_CACHE_STATE = "unknown"
    return result


def require_numba_dispatch_day() -> Any:
    """Return the compiled dispatch callable, or explain what is missing.

    Called once per simulated span, before any timestep runs, so a study that
    asks for this backend without the extra installed stops immediately rather
    than part-way through a long run.
    """
    if not numba_available():
        raise NumbaUnavailableError(
            "execution_backend='numba' requires the optional Numba dependency. "
            'Install it with: pip install "breos[fast]"'
        )
    return _dispatch_day_numba
