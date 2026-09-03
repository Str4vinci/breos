"""Optional compiled within-day dispatch kernel.

This is a private accelerator, not public API. It reproduces the reference
per-step loop in :mod:`breos.battery` for one degradation day at a time, with
state of health, resistance-derived efficiencies and the replacement decision
held fixed for the duration of the call. Everything scientifically sensitive --
rainflow counting, calendar and cycle degradation, resistance growth,
replacement, and the state carried between days and years -- stays in the
Python reference path and is never compiled.

The kernel is a statement-by-statement mirror of ``_dispatch_day_python``,
including the inlined bodies of ``calculate_dc_ac_power``,
``dc_power_for_ac_output``, ``_apply_capacity_window`` and
``lfp_capacity_factor``. Operation order and branch structure are part of the
contract: the target is a bit-identical result, so a change that is
algebraically equivalent but reassociates arithmetic is still a defect.

Compiled with ``fastmath=False``. Enabling it would let LLVM reassociate and
contract these expressions and would break bit identity with the reference.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from breos.battery import _LEDGER_ROW_INDEX, _STATE_ROW_INDEX, BatteryConfig, _ResultBuffers
from breos.constants import LFP_CAP_DERATE_PER_C_COLD, LFP_CAP_DERATE_PER_C_MODERATE
from breos.inverter import (
    PVWATTS_CURVE_CONSTANT,
    PVWATTS_CURVE_LINEAR,
    PVWATTS_CURVE_QUADRATIC,
    PVWATTS_REFERENCE_EFFICIENCY,
)

# Row indices into the shared buffer matrix. Read from the reference module so
# a reordering there is picked up rather than silently mismatched; Numba
# freezes them as compile-time constants.
R_PV_DC = _STATE_ROW_INDEX["pv_dc"]
R_PV_PRODUCTION = _STATE_ROW_INDEX["pv_production"]
R_LOAD = _STATE_ROW_INDEX["load"]
R_PV_DELTA = _STATE_ROW_INDEX["pv_delta"]
R_GRID_IMPORT = _STATE_ROW_INDEX["grid_import"]
R_GRID_EXPORT = _STATE_ROW_INDEX["grid_export"]
R_BATTERY_ENERGY = _STATE_ROW_INDEX["battery_energy"]
R_SOC_NORMALIZED = _STATE_ROW_INDEX["soc_normalized"]
R_SOC_ABSOLUTE = _STATE_ROW_INDEX["soc_absolute"]
R_SOH = _STATE_ROW_INDEX["soh"]
R_T_CELL = _STATE_ROW_INDEX["t_cell"]
R_PV_CURTAILMENT = _STATE_ROW_INDEX["pv_curtailment"]
R_CHARGE_LOSS = _STATE_ROW_INDEX["charge_loss"]
R_DISCHARGE_LOSS = _STATE_ROW_INDEX["discharge_loss"]
R_STANDBY_LOSS = _STATE_ROW_INDEX["standby_loss"]
R_BATTERY_ENERGY_BEGIN = _STATE_ROW_INDEX["battery_energy_begin"]
R_PV_ORIGIN_BEGIN = _STATE_ROW_INDEX["pv_origin_begin"]
R_PV_ORIGIN_END = _STATE_ROW_INDEX["pv_origin_end"]

L_PV_DC_TO_BATTERY = _LEDGER_ROW_INDEX["PV_DC_To_Battery"]
L_PV_DC_TO_INVERTER = _LEDGER_ROW_INDEX["PV_DC_To_Inverter"]
L_PV_DC_CURTAILED = _LEDGER_ROW_INDEX["PV_DC_Curtailed"]
L_PV_AC_TO_LOAD = _LEDGER_ROW_INDEX["PV_AC_To_Load"]
L_PV_AC_EXPORT = _LEDGER_ROW_INDEX["PV_AC_Export"]
L_BATTERY_CHARGE_INPUT = _LEDGER_ROW_INDEX["Battery_Charge_Input"]
L_BATTERY_CHARGE_STORED = _LEDGER_ROW_INDEX["Battery_Charge_Stored"]
L_BATTERY_DISCHARGE_DC = _LEDGER_ROW_INDEX["Battery_Discharge_DC"]
L_BATTERY_AC_TO_LOAD = _LEDGER_ROW_INDEX["Battery_AC_To_Load"]
L_BATTERY_AC_TO_LOAD_PV = _LEDGER_ROW_INDEX["Battery_AC_To_Load_PV"]
L_PV_ORIGIN_BATTERY_AC_TO_LOAD = _LEDGER_ROW_INDEX["PV_Origin_Battery_AC_To_Load"]
L_PV_DIRECT_INVERTER_LOSS = _LEDGER_ROW_INDEX["PV_Direct_Inverter_Loss"]
L_BATTERY_INVERTER_LOSS = _LEDGER_ROW_INDEX["Battery_Inverter_Loss"]
L_INVERTER_LOSS = _LEDGER_ROW_INDEX["Inverter_Loss"]
L_STANDBY_LOSS = _LEDGER_ROW_INDEX["Standby_Loss"]
L_CAPACITY_WINDOW_LOSS = _LEDGER_ROW_INDEX["Capacity_Window_Loss"]
L_REPLACEMENT_ENERGY_REMOVED = _LEDGER_ROW_INDEX["Battery_Replacement_Energy_Removed"]
L_REPLACEMENT_ENERGY_ADDED = _LEDGER_ROW_INDEX["Battery_Replacement_Energy_Added"]
L_BATTERY_ENERGY_DELTA = _LEDGER_ROW_INDEX["Battery_Energy_Delta"]


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

    The kernels live at module scope in :mod:`breos._numba_dispatch_kernels` so
    that Numba's on-disk cache can key them by qualified name. That module
    imports Numba at its top, so it is imported here rather than above: this
    module must stay importable without the optional dependency, because
    :func:`require_numba_dispatch_day` is what turns a missing Numba into a
    readable error.
    """
    from breos._numba_dispatch_kernels import _dispatch_day_kernel

    return _dispatch_day_kernel


_KERNEL: Any = None


def _kernel() -> Any:
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = _build_kernel()
    return _KERNEL


def _dispatch_day_numba(
    out: _ResultBuffers,
    _pv_dc_vals: np.ndarray,
    _load_vals: np.ndarray,
    _temp_vals: np.ndarray,
    lo: int,
    hi: int,
    *,
    battery_config: BatteryConfig,
    has_battery: bool,
    battery_soh_decimal: float,
    Battery_SOH: float,
    Battery_Energy_Wh: float,
    Battery_PV_Origin_Energy_Wh: float,
    eff_charge: float,
    eff_discharge: float,
    hours_per_step: float,
    standby_loss_per_step_wh: float,
    cap_wh: float,
    cap_charge_wh: float,
    cap_discharge_wh: float,
) -> Tuple[float, float, float, float]:
    """Compiled counterpart of ``_dispatch_day_python`` with the same contract."""
    global _JIT_CACHE_STATE

    kernel = _kernel()
    observe_cache = _JIT_CACHE_STATE is None
    if observe_cache:
        had_compiled_signature = bool(kernel.signatures)
        cache_hits_before = _cache_event_count(kernel.stats.cache_hits)
        cache_misses_before = _cache_event_count(kernel.stats.cache_misses)

    result = kernel(
        out.matrix,
        _pv_dc_vals,
        _load_vals,
        _temp_vals,
        lo,
        hi,
        float(Battery_Energy_Wh),
        float(Battery_PV_Origin_Energy_Wh),
        bool(has_battery),
        float(battery_config.nominal_energy_wh),
        float(battery_soh_decimal),
        float(Battery_SOH),
        float(battery_config.max_soc),
        float(battery_config.min_soc),
        float(standby_loss_per_step_wh),
        float(eff_charge),
        float(eff_discharge),
        float(battery_config.inverter_efficiency),
        float(cap_charge_wh),
        float(cap_discharge_wh),
        float(cap_wh),
        bool(np.isinf(cap_wh)),
        float(battery_config.thermal_resistance_kw),
        float(hours_per_step),
        2.0,
        float(battery_config.ac_output_scale),
    )
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
