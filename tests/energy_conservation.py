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


def _column(results: pd.DataFrame, name: str) -> pd.Series:
    if name in results.columns:
        return results[name]
    return pd.Series(0.0, index=results.index)


def assert_energy_conservation(results: pd.DataFrame, config, *, atol: float = 1e-7) -> None:
    """Assert every per-step ledger identity of ``simulate_energy_balance`` results.

    ``config`` is the run's :class:`breos.battery.BatteryConfig`, for the
    charge efficiency, which is fixed unless resistance fade is enabled.
    """
    grid_to_battery = _column(results, GRID_TO_BATTERY)
    np.testing.assert_allclose(
        results["PV_DC"],
        results["PV_DC_To_Battery"] + results["PV_DC_To_Inverter"] + results["PV_DC_Curtailed"],
        atol=atol,
        err_msg="PV DC split",
    )
    np.testing.assert_allclose(
        results["Houseload"],
        results["PV_AC_To_Load"] + results["Battery_AC_To_Load"] + results["Import_From_Grid"] - grid_to_battery,
        atol=atol,
        err_msg="load supply",
    )
    np.testing.assert_allclose(results["PV_AC_Export"], results["Sell_To_Grid"], atol=atol, err_msg="export alias")
    np.testing.assert_allclose(
        results["Battery_Charge_Stored"],
        results["Battery_Charge_Input"] - results["Battery_Charge_Loss"],
        atol=atol,
        err_msg="charge loss",
    )
    if not config.enable_resistance_fade:
        # Resistance growth lowers the charge efficiency over the pack's life.
        np.testing.assert_allclose(
            results["Battery_Charge_Stored"],
            results["Battery_Charge_Input"] * config.charge_efficiency,
            atol=atol,
            err_msg="charge efficiency",
        )
    np.testing.assert_allclose(
        results["Battery_Energy_Delta"],
        results["Battery_Charge_Stored"]
        - results["Battery_Discharge_DC"]
        - results["Standby_Loss"]
        - results["Capacity_Window_Loss"]
        - results["Battery_Replacement_Energy_Removed"]
        + results["Battery_Replacement_Energy_Added"],
        atol=atol,
        err_msg="battery energy change",
    )
    np.testing.assert_allclose(
        results["Inverter_Loss"],
        results["PV_Direct_Inverter_Loss"] + results["Battery_Inverter_Loss"],
        atol=atol,
        err_msg="inverter loss split",
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
    np.testing.assert_allclose(inputs, outputs, atol=atol, err_msg="whole-system balance")
