"""One place to validate the execution backend and record what ran.

App and Monte Carlo both choose a dispatch backend, and both have to say which
one they used. Two copies of that logic would drift -- a validation message
here, a provenance key there, a missing toolchain record in whichever path was
added second -- and a bit-identity claim is only checkable against a stated
toolchain. So the name check, the fail-before-start dependency check, the
toolchain record and the JIT cache observation live here, and both callers go
through them.

The default is ``"python"`` everywhere, deliberately. The Python path is the
numerical reference; the compiled one is an optimisation that has to earn its
selection explicitly, never by ambient configuration.
"""

from __future__ import annotations

import platform
from typing import Any

import numpy as np
import pandas as pd

EXECUTION_BACKENDS: tuple[str, ...] = ("python", "numba")

#: The numerical reference. Anything else must be asked for by name.
DEFAULT_EXECUTION_BACKEND = "python"

#: Recorded when the cache outcome could not be classified. Never an error:
#: provenance bookkeeping must not be able to fail a completed run.
UNKNOWN_JIT_CACHE_STATE = "unknown"

#: Recorded as ``dispatch_path`` when a run took the vectorized PV-only
#: balance instead of a per-timestep dispatch loop. The backend name alone
#: would not say so: that path is shared, so ``execution_backend = "numba"``
#: and "the compiled kernel ran" are no longer the same claim.
PV_ONLY_DISPATCH_PATH = "pv_only_vectorized"

#: A battery below this cannot move a meaningful amount of energy in a
#: timestep, so dispatch treats it as absent rather than accumulating
#: rounding noise into a ledger.
MIN_DISPATCHABLE_ENERGY_WH = 1.0


def is_pv_only_dispatch(nominal_energy_wh: float, max_soc: float, min_soc: float) -> bool:
    """Return True when dispatch degenerates to the vectorized PV-only path.

    The dispatch loop decides this from the battery config it was handed,
    while App and Monte Carlo decide it from their own config dicts before
    either config exists. Spelling it separately in each place is how a run
    ends up on the vectorized path while its provenance still claims a
    compiled kernel, so all three ask here.

    A battery is dispatchable only if it holds a usable amount of energy
    *and* has room to move between its SOC limits; either alone leaves
    nothing to dispatch.
    """
    return not (nominal_energy_wh > MIN_DISPATCHABLE_ENERGY_WH and (max_soc - min_soc) > 0.0)


def validate_execution_backend(execution_backend: Any) -> str:
    """Return a known backend name, or say what was asked for and what exists."""
    if execution_backend not in EXECUTION_BACKENDS:
        raise ValueError(f"execution_backend must be one of {EXECUTION_BACKENDS}, got {execution_backend!r}")
    return str(execution_backend)


def require_backend(execution_backend: str) -> None:
    """Fail before any timestep runs if the chosen backend cannot be used.

    Called once per study or per App run, so a missing optional dependency
    stops the job at the start rather than part-way through a long one.
    """
    validate_execution_backend(execution_backend)
    if execution_backend == "numba":
        from breos._numba_dispatch import require_numba_dispatch_day

        require_numba_dispatch_day()


def backend_provenance(
    execution_backend: str,
    *,
    jit_cache_states: list[str] | None = None,
    pv_only: bool = False,
) -> dict[str, Any]:
    """Check the backend is usable and record the toolchain it ran on.

    A bit-identity claim cannot be checked after the fact without a stated
    toolchain, so the versions are recorded for every run rather than only for
    benchmarks.

    A ``numba`` record always carries a ``jit_cache`` field. Callers that
    aggregate worker observations pass them in; callers that cannot observe
    their workers -- a driver that fans work out to subprocesses, for instance
    -- pass nothing and get ``"unknown"``. A field that admits it could not
    tell is provenance; a missing field is a gap that reads as an oversight
    later, which is worse on a run that took hours.

    ``pv_only`` records that the run took the shared vectorized PV-only
    balance, which no backend name implies on its own. Callers decide it with
    :func:`is_pv_only_dispatch` so the record and the dispatch cannot disagree.
    """
    validate_execution_backend(execution_backend)
    provenance: dict[str, Any] = {
        "execution_backend": execution_backend,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    if execution_backend == "numba":
        from breos._numba_dispatch import numba_versions, require_numba_dispatch_day

        require_numba_dispatch_day()
        provenance.update(numba_versions())
        provenance["jit_cache"] = aggregate_jit_cache_states(list(jit_cache_states or []))
    if pv_only:
        provenance["dispatch_path"] = PV_ONLY_DISPATCH_PATH
    return provenance


def reset_jit_cache_observation(execution_backend: str) -> None:
    """Open a fresh JIT cache observation window, if the backend has one.

    A no-op for the Python backend, so callers do not have to branch. Each
    observation boundary belongs to one unit of work whose compile cost is
    being attributed -- one Monte Carlo trajectory in a worker, or one App run.
    """
    if execution_backend == "numba":
        from breos._numba_dispatch import reset_jit_cache_observation as _reset

        _reset()


def observed_jit_cache_state(execution_backend: str) -> str | None:
    """Return the cache outcome observed since the last reset, or None."""
    if execution_backend != "numba":
        return None
    from breos._numba_dispatch import observed_jit_cache_state as _observed

    return _observed()


def aggregate_jit_cache_states(states: list[str]) -> str:
    """Summarise several observations into one claim about a run.

    ``cold`` beats ``warm`` because one compile means the run paid for a
    compile. ``unknown`` beats both: if any observation could not be
    classified, the run-level claim cannot be trusted either. An empty list is
    ``unknown`` too -- nothing observed is not evidence of a warm cache.

    This never raises. Provenance bookkeeping must not be able to fail a run
    that produced valid results.
    """
    if not states or any(state not in {"warm", "cold"} for state in states):
        return UNKNOWN_JIT_CACHE_STATE
    return "cold" if "cold" in states else "warm"
