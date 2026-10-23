"""NumPy settlement of quarter-hour meter energies (ADR 0006 K4–K6).

Settlement changes billed balances only. Inputs are already Wh, not the mean
W stored in dispatch frames: the caller converts those exactly once, by 0.25.
No dispatch, configuration, tariffs or ownership are resolved here.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

SETTLEMENT_SCHEMA_VERSION = "1.0"
SETTLEMENT_COLUMNS = (
    "Metered_Import",
    "Metered_Export",
    "Pooled_Injection",
    "Coefficient",
    "Shared_Allocated",
    "Shared_Used",
    "Participant_Surplus",
    "Supplier_Import",
    "Own_Export",
    "Attributed_Shared_Used",
    "Attributed_Surplus",
    "Energy_For_Sharing",
    "Shared_Used_Total",
    "Community_Surplus",
)


@dataclass(frozen=True)
class MeterBalances:
    """One site's aligned 1-D import and export arrays, in Wh per 15 minutes.

    Values must be finite, nonnegative and mutually exclusive in each interval.
    Assets may import (for example inverter night draw), but never receive a
    share. Neither array is modified by settlement.
    """

    import_wh: ArrayLike
    export_wh: ArrayLike


def _meter_arrays(balances: MeterBalances, n: int, site_id: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    arrays = []
    for name, values in (("import", balances.import_wh), ("export", balances.export_wh)):
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (n,):
            raise ValueError(f"{site_id}: {name} must have shape ({n},)")
        if not np.all(np.isfinite(array)) or np.any(array < 0):
            raise ValueError(f"{site_id}: {name} must be finite and nonnegative")
        arrays.append(array)
    imports, exports = arrays
    if np.any((imports > 0) & (exports > 0)):
        raise ValueError(f"{site_id}: simultaneous metered import and export")
    return imports, exports


def _sum_in_id_order(values: NDArray[np.float64]) -> NDArray[np.float64]:
    # Rows have already been put in canonical id order. Do not reduce a mapping
    # in its insertion order, or add separately reduced asset/participant sums.
    with np.errstate(over="raise", invalid="raise"):
        try:
            return np.sum(values, axis=0)
        except FloatingPointError as exc:
            raise ValueError("community energy total exceeds float64 range") from exc


def settle_meter_balances(
    participants: Mapping[str, MeterBalances],
    assets: Mapping[str, MeterBalances],
    *,
    index: pd.DatetimeIndex,
    allocation: Literal["fixed", "proportional"],
    coefficients: Mapping[str, float] | None = None,
    pool_participant_export: bool = True,
) -> pd.DataFrame:
    """Allocate metered injections and return settlement schema ``"1.0"``.

    ``index`` labels consecutive 15-minute intervals (interval-start or
    interval-end labels are preserved). DST transitions are checked by elapsed
    time. Inputs must have one value per label, and at least one participant.
    Ids are unique across sites and follow ADR K3's lowercase id syntax.

    Fixed coefficients must name exactly the participants, be finite and
    nonnegative, and sum to 1 within absolute float64 tolerance 1e-12. They are
    renormalised over non-exporting participants, including zero-import sites.
    Proportional allocation takes no configured coefficients. A zero receiver
    denominator leaves all energy as community surplus, without allocation.
    Participant surplus is never redistributed. Exporters receive nothing even
    when their export is not pooled.

    The frame has index levels ``interval``, ``site_type``, ``site_id``. Each
    interval contains all sites in sorted id order, followed by a community row
    (type ``community``, id ``""``). Site types are ``participant`` and ``asset``.
    K6's eleven site columns are zero on community rows; its three community
    columns are zero on site rows. ``Coefficient`` is dimensionless; every
    other column is Wh. Version and units live in ``DataFrame.attrs``. All
    numerical work uses NumPy float64; pandas only packages the output.

    Roundoff can put total use a few ulps above injection. Community surplus
    is clamped to zero in that case; no energy is redistributed or rounded.
    Closure therefore holds to floating-point precision, as specified by K6.
    """
    if not isinstance(index, pd.DatetimeIndex) or index.hasnans or len(index) == 0:
        raise ValueError("index must be a nonempty DatetimeIndex without NaT")
    if not np.all(index[1:] - index[:-1] == pd.Timedelta(minutes=15)):
        raise ValueError("settlement requires consecutive 15-minute intervals")
    if not participants:
        raise ValueError("settlement requires at least one participant")
    if any(
        not isinstance(key, str) or re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", key) is None
        for key in (*participants, *assets)
    ):
        raise ValueError("site ids must match ^[a-z0-9][a-z0-9_-]{0,63}$")
    if set(participants) & set(assets):
        raise ValueError("participant and asset ids must be unique")
    if allocation not in ("fixed", "proportional"):
        raise ValueError("allocation must be 'fixed' or 'proportional'")
    if not isinstance(pool_participant_export, bool):
        raise ValueError("pool_participant_export must be a bool")

    site_ids = sorted((*participants, *assets))
    participant_ids = sorted(participants)
    participant_positions = np.array([site_ids.index(key) for key in participant_ids])
    n = len(index)
    meters = [_meter_arrays(participants[key] if key in participants else assets[key], n, key) for key in site_ids]
    imports = np.stack([meter[0] for meter in meters])
    exports = np.stack([meter[1] for meter in meters])
    is_participant = np.array([key in participants for key in site_ids])
    pooled = np.where(is_participant[:, None] & (not pool_participant_export), 0.0, exports)
    energy = _sum_in_id_order(pooled)
    consumption = imports[participant_positions]
    receivers = exports[participant_positions] == 0

    if allocation == "fixed":
        if coefficients is None or set(coefficients) != set(participants):
            raise ValueError("fixed coefficients must name exactly the participants")
        base = np.array([coefficients[key] for key in participant_ids], dtype=np.float64)
        if (
            not np.all(np.isfinite(base))
            or np.any(base < 0)
            or not np.isclose(_sum_in_id_order(base), 1.0, rtol=0, atol=1e-12)
        ):
            raise ValueError("fixed coefficients must be finite, nonnegative and sum to 1 (atol=1e-12)")
        weights = np.where(receivers, base[:, None], 0.0)
    else:
        if coefficients is not None:
            raise ValueError("proportional allocation does not take fixed coefficients")
        weights = np.where(receivers, consumption, 0.0)
    denominator = _sum_in_id_order(weights)
    effective = np.divide(weights, denominator, out=np.zeros_like(weights), where=denominator > 0)
    allocated = effective * energy
    used = np.minimum(allocated, consumption)
    used_total = _sum_in_id_order(used)
    surplus = np.maximum(energy - used_total, 0.0)
    injection_fraction = np.divide(pooled, energy, out=np.zeros_like(pooled), where=energy > 0)

    # Build one interval-major block. The numeric columns are all float64,
    # including zeros in fields that do not apply to a particular row type.
    values = np.zeros((n, len(site_ids) + 1, len(SETTLEMENT_COLUMNS)), dtype=np.float64)
    sites = values[:, :-1, :]
    sites[:, :, 0] = imports.T
    sites[:, :, 1] = exports.T
    sites[:, :, 2] = pooled.T
    sites[:, participant_positions, 3] = effective.T
    sites[:, participant_positions, 4] = allocated.T
    sites[:, participant_positions, 5] = used.T
    sites[:, participant_positions, 6] = (allocated - used).T
    sites[:, :, 7] = imports.T
    sites[:, participant_positions, 7] = (consumption - used).T
    sites[:, :, 8] = (exports - pooled).T
    sites[:, :, 9] = (injection_fraction * used_total).T
    sites[:, :, 10] = (injection_fraction * surplus).T
    values[:, -1, 11:] = np.stack((energy, used_total, surplus), axis=1)

    types = ["participant" if key in participants else "asset" for key in site_ids] + ["community"]
    frame_index = pd.MultiIndex.from_arrays(
        [
            index.repeat(len(site_ids) + 1),
            np.tile(types, n),
            np.tile([*site_ids, ""], n),
        ],
        names=["interval", "site_type", "site_id"],
    )
    frame = pd.DataFrame(values.reshape(-1, len(SETTLEMENT_COLUMNS)), index=frame_index, columns=SETTLEMENT_COLUMNS)
    frame.attrs.update(
        settlement_schema_version=SETTLEMENT_SCHEMA_VERSION,
        energy_unit="Wh",
        interval_minutes=15,
        allocation=allocation,
        pool_participant_export=pool_participant_export,
    )
    return frame
