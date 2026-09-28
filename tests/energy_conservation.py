"""Shared per-step energy-conservation checks for simulation result frames.

Every test that runs the energy balance can call
:func:`assert_energy_conservation` on its per-step results, whatever the
backend, degradation engine or resolution. The frame's power columns share
one unit, so the identities hold on the values as they are.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# AC energy the grid puts into the battery. No dispatch strategy does this
# yet; the grid-charging work planned for 0.7 will, and its column then counts
# as an external input like PV. Absent columns count as zero.
GRID_TO_BATTERY = "Grid_AC_To_Battery"


# Every column the identities read, apart from the optional grid charging.
LEDGER_COLUMNS = (
    "PV_DC",
    "PV_DC_To_Battery",
    "PV_DC_To_Inverter",
    "PV_DC_Curtailed",
    "Houseload",
    "PV_AC_To_Load",
    "Battery_AC_To_Load",
    "Import_From_Grid",
    "PV_AC_Export",
    "Battery_Charge_Input",
    "Battery_Charge_Stored",
    "Battery_Charge_Loss",
    "Battery_Discharge_DC",
    "Battery_Discharge_Loss",
    "Battery_Energy_Delta",
    "Standby_Loss",
    "Capacity_Window_Loss",
    "Battery_Replacement_Energy_Removed",
    "Battery_Replacement_Energy_Added",
    "Inverter_Loss",
    "PV_Direct_Inverter_Loss",
    "Battery_Inverter_Loss",
)


def _column(results: pd.DataFrame, name: str) -> pd.Series:
    if name in results.columns:
        return results[name]
    return pd.Series(0.0, index=results.index)


def _assert_balanced(actual, desired, atol: float, label: str) -> None:
    # rtol=0 makes atol the whole bound; NumPy's default rtol would add a
    # term that grows with the values.
    np.testing.assert_allclose(actual, desired, rtol=0, atol=atol, err_msg=label)


def assert_energy_conservation(results: pd.DataFrame, config, *, atol: float = 1e-7) -> None:
    """Assert every per-step ledger identity of ``simulate_energy_balance`` results.

    ``config`` is the run's :class:`breos.battery.BatteryConfig`, for the
    charge efficiency, which is fixed unless resistance fade is enabled.
    """
    columns = [*LEDGER_COLUMNS, *([GRID_TO_BATTERY] if GRID_TO_BATTERY in results.columns else [])]
    ledger = results[columns].to_numpy(dtype=float)
    # assert_allclose treats NaN as equal to NaN, so a NaN row would pass
    # every identity.
    bad_rows, bad_columns = np.nonzero(~np.isfinite(ledger))
    if bad_rows.size:
        names = ", ".join(sorted({columns[i] for i in bad_columns}))
        raise AssertionError(
            f"non-finite ledger values in {bad_rows.size} cells, first at {results.index[bad_rows[0]]}; "
            f"columns: {names}"
        )

    grid_to_battery = _column(results, GRID_TO_BATTERY)
    _assert_balanced(
        results["PV_DC"],
        results["PV_DC_To_Battery"] + results["PV_DC_To_Inverter"] + results["PV_DC_Curtailed"],
        atol,
        "PV DC split",
    )
    _assert_balanced(
        results["Houseload"],
        results["PV_AC_To_Load"] + results["Battery_AC_To_Load"] + results["Import_From_Grid"] - grid_to_battery,
        atol,
        "load supply",
    )
    _assert_balanced(
        results["Battery_Charge_Stored"],
        results["Battery_Charge_Input"] - results["Battery_Charge_Loss"],
        atol,
        "charge loss",
    )
    if not config.enable_resistance_fade:
        # Resistance growth lowers the charge efficiency over the pack's life.
        _assert_balanced(
            results["Battery_Charge_Stored"],
            results["Battery_Charge_Input"] * config.charge_efficiency,
            atol,
            "charge efficiency",
        )
    _assert_balanced(
        results["Battery_Energy_Delta"],
        results["Battery_Charge_Stored"]
        - results["Battery_Discharge_DC"]
        - results["Standby_Loss"]
        - results["Capacity_Window_Loss"]
        - results["Battery_Replacement_Energy_Removed"]
        + results["Battery_Replacement_Energy_Added"],
        atol,
        "battery energy change",
    )
    _assert_balanced(
        results["Inverter_Loss"],
        results["PV_Direct_Inverter_Loss"] + results["Battery_Inverter_Loss"],
        atol,
        "inverter loss split",
    )
    # PV, grid charging and replacement-added energy are the external inputs.
    # Delivered energy, losses, net battery movement, and energy removed with
    # a replaced pack are outputs.
    outputs = (
        results["PV_AC_To_Load"]
        + results["PV_AC_Export"]
        + results["Battery_AC_To_Load"]
        + results["PV_DC_Curtailed"]
        + results["Battery_Charge_Loss"]
        + results["Battery_Discharge_Loss"]
        + results["PV_Direct_Inverter_Loss"]
        + results["Battery_Inverter_Loss"]
        + results["Standby_Loss"]
        + results["Capacity_Window_Loss"]
        + results["Battery_Replacement_Energy_Removed"]
        + results["Battery_Energy_Delta"]
    )
    inputs = results["PV_DC"] + grid_to_battery + results["Battery_Replacement_Energy_Added"]
    _assert_balanced(inputs, outputs, atol, "whole-system balance")


# Stored energy by origin (ADR 0002 A8). The unattributed origin has no
# columns of its own: its share of a flow is the total minus the PV and grid
# shares.
_ORIGIN_FLOWS = (
    # (total column, per-origin column suffix, sign in the balance)
    ("Battery_Charge_Stored", "Battery_Charge_Stored", 1.0),
    ("Battery_Discharge_DC", "Battery_Discharge_DC", -1.0),
    ("Standby_Loss", "Standby_Loss", -1.0),
    ("Capacity_Window_Loss", "Capacity_Window_Loss", -1.0),
    ("Battery_Replacement_Energy_Removed", "Replacement_Energy_Removed", -1.0),
)


def assert_origin_reconciliation(results: pd.DataFrame, hours_per_step: float, *, atol: float = 1e-6) -> None:
    """Assert that the PV, grid and unattributed origins each reconcile step by step.

    Balances are Wh and flows average W, so flows are scaled by the step
    length. Each origin's ending balance must equal its beginning balance
    plus charge stored, minus discharge and every removal; the unattributed
    origin also gains a replacement pack's energy. Every origin's share of a
    flow must be non-negative and the shares must not exceed the total.
    """
    balances = {
        origin: (
            results[f"Battery_{origin}_Origin_Energy_Beginning"],
            results[f"Battery_{origin}_Origin_Energy_End"],
        )
        for origin in ("PV", "Grid")
    }
    balances["Unattributed"] = (
        results["Battery_Energy_Beginning"] - balances["PV"][0] - balances["Grid"][0],
        results["Battery_Energy_End"] - balances["PV"][1] - balances["Grid"][1],
    )
    expected = {origin: begin.copy() for origin, (begin, _end) in balances.items()}
    for total_column, suffix, sign in _ORIGIN_FLOWS:
        shares = {origin: _column(results, f"{origin}_Origin_{suffix}") for origin in ("PV", "Grid")}
        shares["Unattributed"] = results[total_column] - shares["PV"] - shares["Grid"]
        for origin, share in shares.items():
            if (share < -atol).any():
                raise AssertionError(f"{origin}-origin share of {total_column} is negative at {share.idxmin()}")
            expected[origin] = expected[origin] + sign * share * hours_per_step
    expected["Unattributed"] = expected["Unattributed"] + results["Battery_Replacement_Energy_Added"] * hours_per_step
    for origin, (_begin, end) in balances.items():
        if (end < -atol).any():
            raise AssertionError(f"{origin}-origin balance is negative at {end.idxmin()}")
        _assert_balanced(end, expected[origin], atol, f"{origin}-origin balance")
    for origin in ("PV", "Grid"):
        share = results[f"{origin}_Origin_Battery_AC_To_Load"]
        if (share < -atol).any() or (share > results["Battery_AC_To_Load"] + atol).any():
            raise AssertionError(f"{origin}-origin battery AC to load falls outside [0, Battery_AC_To_Load]")
