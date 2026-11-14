"""Shared per-step energy-conservation checks for simulation result frames.

Every test that runs the energy balance can call
:func:`assert_energy_conservation` on its per-step results, whatever the
backend, degradation engine or resolution. The frame's power columns share
one unit, so the identities hold on the values as they are.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# AC energy the grid puts into the battery (ADR 0002 A6), an external input
# like PV, with the DC it becomes and the conversion loss between them.
# Absent columns count as zero.
GRID_TO_BATTERY = "Grid_AC_To_Battery"
GRID_CHARGE_COLUMNS = (GRID_TO_BATTERY, "Grid_DC_To_Battery", "Grid_Charge_Conversion_Loss")
# PV inverter AC an AC-coupled battery inverter takes in, and that inverter's
# charging loss on it (ADR 0004 C8). Internal flows: zero unless AC-coupled.
PV_AC_CHARGE_COLUMNS = ("PV_AC_To_Battery", "PV_Charge_Conversion_Loss")


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
    optional = (*GRID_CHARGE_COLUMNS, *PV_AC_CHARGE_COLUMNS, "Inverter_Night_Consumption")
    columns = [*LEDGER_COLUMNS, *(column for column in optional if column in results.columns)]
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
    grid_dc_to_battery = _column(results, "Grid_DC_To_Battery")
    grid_conversion_loss = _column(results, "Grid_Charge_Conversion_Loss")
    _assert_balanced(grid_to_battery, grid_dc_to_battery + grid_conversion_loss, atol, "grid-charge conversion")
    pv_ac_to_battery = _column(results, "PV_AC_To_Battery")
    pv_charge_conversion_loss = _column(results, "PV_Charge_Conversion_Loss")
    _assert_balanced(
        results["Battery_Charge_Input"],
        results["PV_DC_To_Battery"] + pv_ac_to_battery - pv_charge_conversion_loss + grid_dc_to_battery,
        atol,
        "charge input split",
    )
    # Idle-inverter draw is served after the house load, before charging or
    # export (ADR 0004 M9). Derive its PV share from the existing ledger.
    night = _column(results, "Inverter_Night_Consumption")
    pv_surplus = (results["PV_DC_To_Inverter"] - results["PV_Direct_Inverter_Loss"] - results["PV_AC_To_Load"]).clip(
        lower=0.0
    )
    pv_to_night = np.minimum(pv_surplus, night)
    # What the PV inverter takes in and loses leaves it as AC: to the house,
    # idle inverters, export, or an AC-coupled battery inverter.
    _assert_balanced(
        results["PV_DC_To_Inverter"] - results["PV_Direct_Inverter_Loss"],
        results["PV_AC_To_Load"]
        + pv_to_night
        + results["PV_AC_Export"]
        + pv_ac_to_battery
        + _column(results, "PV_AC_To_Vehicle"),
        atol,
        "PV inverter output",
    )
    _assert_balanced(
        results["PV_DC"],
        results["PV_DC_To_Battery"] + results["PV_DC_To_Inverter"] + results["PV_DC_Curtailed"],
        atol,
        "PV DC split",
    )
    # Import also feeds grid charging and the part of the draw PV cannot cover.
    _assert_balanced(
        results["Houseload"],
        results["PV_AC_To_Load"]
        + results["Battery_AC_To_Load"]
        + _column(results, "Vehicle_AC_To_Load")
        + results["Import_From_Grid"]
        - _column(results, "Grid_AC_To_Vehicle")
        - grid_to_battery
        - (night - pv_to_night),
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
        # Each source's charge stores into its own origin (ADR 0002 A8,
        # ADR 0004 C8): PV's DC, as DC or after the battery inverter.
        pv_origin_stored = _column(results, "PV_Origin_Battery_Charge_Stored")
        grid_origin_stored = _column(results, "Grid_Origin_Battery_Charge_Stored")
        _assert_balanced(
            pv_origin_stored,
            (results["PV_DC_To_Battery"] + pv_ac_to_battery - pv_charge_conversion_loss) * config.charge_efficiency,
            atol,
            "PV-origin charge stored",
        )
        _assert_balanced(
            grid_origin_stored, grid_dc_to_battery * config.charge_efficiency, atol, "grid-origin charge stored"
        )
    if getattr(config, "coupling", "dc") == "ac":
        # No PV DC reaches the battery, and the battery inverter's AC rating
        # bounds what enters it, PV and grid together, and what leaves it.
        _assert_balanced(results["PV_DC_To_Battery"], 0.0, 0.0, "AC coupling stores no PV DC")
        rating = config.battery_inverter_ac_capacity_w
        if rating is not None:
            entering = pv_ac_to_battery + grid_to_battery
            for label, flow in (
                ("battery inverter AC in", entering),
                ("battery inverter AC out", results["Battery_AC_To_Load"]),
            ):
                if (flow > rating + atol).any():
                    raise AssertionError(f"{label} exceeds its rating at {flow.idxmax()}: {flow.max()} > {rating}")
    else:
        _assert_balanced(pv_ac_to_battery, 0.0, 0.0, "a hybrid inverter takes no PV AC into the battery")
        _assert_balanced(pv_charge_conversion_loss, 0.0, 0.0, "a hybrid inverter has no PV charging conversion")
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
    # PV_AC_To_Battery is internal: it leaves the PV inverter and enters the
    # battery inverter, so it is in neither side.
    # Delivered energy, losses, net battery movement, and energy removed with
    # a replaced pack are outputs.
    outputs = (
        results["PV_AC_To_Load"]
        + pv_to_night
        + results["PV_AC_Export"]
        + results["Battery_AC_To_Load"]
        + results["PV_DC_Curtailed"]
        + results["Battery_Charge_Loss"]
        + results["Battery_Discharge_Loss"]
        + results["PV_Direct_Inverter_Loss"]
        + results["Battery_Inverter_Loss"]
        + grid_conversion_loss
        + pv_charge_conversion_loss
        + results["Standby_Loss"]
        + results["Capacity_Window_Loss"]
        + results["Battery_Replacement_Energy_Removed"]
        + results["Battery_Energy_Delta"]
    )
    # Vehicle home conversion, battery losses and home movement. The vehicle
    # helper additionally checks actual site movement and boundary bookings;
    # away operations never enter this site balance.
    outputs = outputs + _column(results, "Vehicle_AC_To_Load")
    for name in (
        "Vehicle_Charge_Conversion_Loss",
        "Vehicle_Charge_Loss",
        "Vehicle_Discharge_Loss",
        "Vehicle_Charger_Loss",
    ):
        outputs = outputs + _column(results, name)
    vehicle_movement = (
        _column(results, "Vehicle_Charge_Input")
        - _column(results, "Vehicle_Charge_Loss")
        - _column(results, "Vehicle_Discharge_DC")
    )
    outputs = outputs + vehicle_movement
    inputs = (
        results["PV_DC"]
        + grid_to_battery
        + results["Battery_Replacement_Energy_Added"]
        + _column(results, "Grid_AC_To_Vehicle")
    )
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


def assert_vehicle_conservation(
    results: pd.DataFrame,
    vehicle_config,
    hours_per_step: float,
    *,
    schedule=None,
    initial_carry=None,
    initial_energy_wh: float | None = None,
    initial_pv_origin_wh: float = 0.0,
    initial_grid_origin_wh: float = 0.0,
    initial_plugged: bool | None = None,
    atol: float = 1e-7,
) -> None:
    """ADR 0005 identities 3–11, including home/away transitions.

    States are Wh and transfers average W. ``schedule`` additionally checks
    scheduled driving, leg order and the parked-time purchase limits.
    A fresh run starts where its first schedule step says, without a boundary
    booking. A resumed span uses the preceding ``initial_carry``.
    """
    from breos._dispatch import _VEHICLE_LEDGER_COLUMNS, _VEHICLE_STATE_COLUMNS

    columns = (*_VEHICLE_STATE_COLUMNS, *_VEHICLE_LEDGER_COLUMNS)
    assert np.isfinite(results[list(columns)].to_numpy()).all(), "non-finite vehicle ledger"
    col = results.__getitem__
    capacity = vehicle_config.battery_kwh * 1000.0
    if initial_carry is not None:
        initial_energy_wh = initial_carry.energy_wh
        initial_pv_origin_wh = initial_carry.pv_origin_energy_wh
        initial_grid_origin_wh = initial_carry.grid_origin_energy_wh
        initial_plugged = initial_carry.plugged
    initial_soh = 1.0 if initial_carry is None else initial_carry.soh_fraction
    capacity_end = capacity * col("Vehicle_SOH")
    capacity_before = capacity_end.shift(1, fill_value=capacity * initial_soh)
    retired_before = pd.Series(False if initial_carry is None else initial_carry.retired, index=results.index)
    retired_after = retired_before.copy()
    for event in results.attrs.get("vehicle_end_of_life_events", ()):
        if event["action"] == "retired":
            retired_before.iloc[event["step"] + 1 :] = True
            retired_after.iloc[event["step"] :] = True
    initial = capacity * initial_soh * vehicle_config.departure_soc if initial_energy_wh is None else initial_energy_wh
    onboard = col("Vehicle_Energy_Onboard")
    begin = onboard.shift(1, fill_value=initial)
    plugged = col("Vehicle_Plugged").astype(bool)
    previous = plugged.shift(1, fill_value=bool(plugged.iloc[0]) if initial_plugged is None else initial_plugged)
    departure = previous & ~plugged
    arrival = ~previous & plugged
    _assert_balanced(col("Vehicle_Energy"), np.where(plugged, onboard, 0.0), 0.0, "vehicle site boundary")
    _assert_balanced(
        col("Vehicle_Departure_Energy") * hours_per_step, np.where(departure, begin, 0.0), atol, "vehicle departure"
    )
    _assert_balanced(
        col("Vehicle_Arrival_Energy") * hours_per_step, np.where(arrival, begin, 0.0), atol, "vehicle arrival"
    )
    shortfall = (capacity_before * vehicle_config.departure_soc - begin).clip(lower=0.0)
    _assert_balanced(
        col("Vehicle_Departure_Shortfall") * hours_per_step,
        np.where(departure & (shortfall > 1e-9 * capacity_before), shortfall, 0.0),
        atol,
        "departure target shortfall",
    )
    _assert_balanced(
        col("Vehicle_SOC_Absolute"), (onboard / capacity_end.replace(0.0, np.nan)).fillna(0.0), atol, "vehicle SOC"
    )
    assert col("Vehicle_SOH").between(0.0, 1.0).all(), "invalid vehicle health"
    ac_in = col("PV_AC_To_Vehicle") + col("Grid_AC_To_Vehicle")
    stored = col("Vehicle_Charge_Input") - col("Vehicle_Charge_Loss")
    draw = col("Vehicle_Discharge_DC")
    drive = col("Vehicle_Drive_Energy")
    away_stored = col("Vehicle_Away_Charge_Purchased") - col("Vehicle_Away_Charge_Loss")
    _assert_balanced(
        ac_in - col("Vehicle_Charge_Conversion_Loss"), col("Vehicle_Charge_Input"), atol, "vehicle charger in"
    )
    _assert_balanced(
        draw - col("Vehicle_Discharge_Loss") - col("Vehicle_Charger_Loss"),
        col("Vehicle_AC_To_Load"),
        atol,
        "vehicle charger out",
    )
    movement = (
        stored
        - draw
        + away_stored
        - drive
        - col("Vehicle_Capacity_Window_Loss")
        - col("Vehicle_Replacement_Energy_Removed")
        + col("Vehicle_Replacement_Energy_Added")
    )
    _assert_balanced(onboard - begin, movement * hours_per_step, atol, "vehicle on-board movement")

    # The full site meter supplies the house, both stores and idle inverters.
    # Driving and away purchases are deliberately absent from this identity.
    initial_site = initial if bool(previous.iloc[0]) else 0.0
    delta_site = col("Vehicle_Energy") - col("Vehicle_Energy").shift(1, fill_value=initial_site)
    losses = sum(
        col(name)
        for name in (
            "Vehicle_Charge_Conversion_Loss",
            "Vehicle_Charge_Loss",
            "Vehicle_Discharge_Loss",
            "Vehicle_Charger_Loss",
        )
    )
    inputs = (
        col("PV_DC")
        + col("Import_From_Grid")
        + col("Battery_Replacement_Energy_Added")
        + col("Vehicle_Arrival_Energy")
        + col("Vehicle_Replacement_Energy_Added") * plugged
    )
    outputs = sum(
        col(name)
        for name in (
            "Houseload",
            "PV_AC_Export",
            "PV_DC_Curtailed",
            "Inverter_Loss",
            "Inverter_Night_Consumption",
            "Battery_Charge_Loss",
            "Battery_Discharge_Loss",
            "Grid_Charge_Conversion_Loss",
            "PV_Charge_Conversion_Loss",
            "Standby_Loss",
            "Capacity_Window_Loss",
            "Battery_Replacement_Energy_Removed",
            "Battery_Energy_Delta",
            "Vehicle_Departure_Energy",
        )
    )
    outputs = (
        outputs
        + losses
        + delta_site / hours_per_step
        + (col("Vehicle_Capacity_Window_Loss") + col("Vehicle_Replacement_Energy_Removed")) * plugged
    )
    _assert_balanced(inputs, outputs, atol, "whole-site balance including vehicle boundaries")
    if ((ac_in > atol) & (draw > atol)).any():
        raise AssertionError("vehicle charges and discharges in the same step")
    rating = vehicle_config.charger_ac_rating_kw * 1000.0
    assert (ac_in <= rating + atol).all(), "vehicle input exceeds charger rating"
    assert (col("Vehicle_AC_To_Load") <= rating + atol).all(), "vehicle output exceeds charger rating"
    residual = (col("Houseload") - col("PV_AC_To_Load") - col("Battery_AC_To_Load")).clip(lower=0.0)
    assert (col("Vehicle_AC_To_Load") <= residual + atol).all(), "vehicle exports or supplies another store"
    assert (col("Import_From_Grid") >= -atol).all(), "vehicle exports to the grid"
    assert (onboard >= np.where(retired_after, 0.0, vehicle_config.min_soc * capacity_end) - atol).all(), (
        "vehicle below minimum SOC"
    )
    assert (onboard <= vehicle_config.max_soc * capacity_end + atol).all(), "vehicle above maximum SOC"
    away_columns = (
        "Vehicle_Drive_Energy",
        "Vehicle_Drive_Unserved",
        "Vehicle_Away_Charge_Purchased",
        "Vehicle_Away_Charge_Loss",
        "PV_Origin_Vehicle_Drive_Energy",
        "Grid_Origin_Vehicle_Drive_Energy",
    )
    boundaries = ("Vehicle_Departure_Energy", "Vehicle_Arrival_Energy", "Vehicle_Departure_Shortfall")
    for name in _VEHICLE_LEDGER_COLUMNS:
        assert (col(name) >= -atol).all(), f"negative {name}"
        if name in away_columns:
            _assert_balanced(col(name)[plugged], 0.0, 0.0, f"home away flow {name}")
        elif name not in boundaries and not any(
            part in name for part in ("Capacity_Window_Loss", "Replacement_Energy")
        ):
            _assert_balanced(col(name)[~plugged], 0.0, 0.0, f"away home flow {name}")

    expected_first = None
    if schedule is not None:
        np.testing.assert_array_equal(plugged, schedule.plugged)
        first = np.where(plugged, 0.0, schedule.drive_first_wh)
        second = np.where(plugged, 0.0, schedule.drive_second_wh)
        minimum = vehicle_config.min_soc * capacity_before
        expected_first = np.minimum(first, (begin - minimum).clip(lower=0.0))
        expected_first = pd.Series(np.where(retired_before, 0.0, expected_first), index=results.index)
        after_first = begin - expected_first
        away = vehicle_config.away_charging
        if away is None:
            expected_away = np.zeros(len(results))
        else:
            expected_away = np.minimum(
                away.power_kw * 1000.0 * schedule.park_hours * away.efficiency,
                (away.target_soc * capacity_before - after_first).clip(lower=0.0),
            )
        expected_away = np.where(plugged | retired_before, 0.0, expected_away)
        expected_second = np.minimum(second, (after_first + expected_away - minimum).clip(lower=0.0))
        expected_second = np.where(retired_before, 0.0, expected_second)
        _assert_balanced(drive * hours_per_step, expected_first + expected_second, atol, "ordered driving withdrawals")
        _assert_balanced(
            (drive + col("Vehicle_Drive_Unserved")) * hours_per_step, first + second, atol, "scheduled driving"
        )
        _assert_balanced(
            away_stored * hours_per_step, expected_away, atol, "away charging target and parked-power limit"
        )
        efficiency = 1.0 if away is None else away.efficiency
        _assert_balanced(
            away_stored, col("Vehicle_Away_Charge_Purchased") * efficiency, atol, "away charging efficiency"
        )

    initial_origins = {
        "PV": initial_pv_origin_wh,
        "Grid": initial_grid_origin_wh,
        "Unattributed": initial - initial_pv_origin_wh - initial_grid_origin_wh,
    }
    ends = {origin: col(f"Vehicle_{origin}_Origin_Energy_End") for origin in ("PV", "Grid")}
    ends["Unattributed"] = onboard - ends["PV"] - ends["Grid"]
    for origin, end in ends.items():
        start = end.shift(1, fill_value=initial_origins[origin])
        if origin == "Unattributed":
            charged = stored - col("PV_Origin_Vehicle_Charge_Stored") - col("Grid_Origin_Vehicle_Charge_Stored")
            discharged = draw - col("PV_Origin_Vehicle_Discharge_DC") - col("Grid_Origin_Vehicle_Discharge_DC")
            driven = drive - col("PV_Origin_Vehicle_Drive_Energy") - col("Grid_Origin_Vehicle_Drive_Energy")
        else:
            charged = col(f"{origin}_Origin_Vehicle_Charge_Stored")
            discharged = col(f"{origin}_Origin_Vehicle_Discharge_DC")
            driven = col(f"{origin}_Origin_Vehicle_Drive_Energy")
            fraction = (start / begin.replace(0.0, np.nan)).fillna(0.0)
            _assert_balanced(discharged, draw * fraction, atol, f"vehicle {origin} proportional withdrawal")
            _assert_balanced(
                col(f"{origin}_Origin_Vehicle_AC_To_Load"),
                col("Vehicle_AC_To_Load") * fraction,
                atol,
                f"vehicle {origin} AC share",
            )
            ac_share = col("PV_AC_To_Vehicle" if origin == "PV" else "Grid_AC_To_Vehicle")
            _assert_balanced(
                charged,
                stored * (ac_share / ac_in.replace(0.0, np.nan)).fillna(0.0),
                atol,
                f"vehicle {origin} charge share",
            )
            if expected_first is not None:
                first_share = expected_first * fraction
                origin_before_second = start - first_share + (expected_away if origin == "Grid" else 0.0)
                second_fraction = (origin_before_second / (after_first + expected_away).replace(0.0, np.nan)).fillna(
                    0.0
                )
                _assert_balanced(
                    driven * hours_per_step,
                    first_share + expected_second * second_fraction,
                    atol,
                    f"vehicle {origin} driving share",
                )
        if origin == "Unattributed":
            removal = (
                col("Vehicle_Capacity_Window_Loss")
                + col("Vehicle_Replacement_Energy_Removed")
                - col("PV_Origin_Vehicle_Capacity_Window_Loss")
                - col("Grid_Origin_Vehicle_Capacity_Window_Loss")
                - col("PV_Origin_Vehicle_Replacement_Energy_Removed")
                - col("Grid_Origin_Vehicle_Replacement_Energy_Removed")
            )
        else:
            window_loss = col(f"{origin}_Origin_Vehicle_Capacity_Window_Loss")
            removed = col(f"{origin}_Origin_Vehicle_Replacement_Energy_Removed")
            removal = window_loss + removed
            origin_preclose = (
                start + (charged - discharged - driven + (away_stored if origin == "Grid" else 0.0)) * hours_per_step
            )
            energy_preclose = begin + (stored - draw - drive + away_stored) * hours_per_step
            share = (origin_preclose / energy_preclose.replace(0.0, np.nan)).fillna(0.0)
            _assert_balanced(
                window_loss, col("Vehicle_Capacity_Window_Loss") * share, atol, f"vehicle {origin} capacity loss"
            )
            remaining_energy = energy_preclose - col("Vehicle_Capacity_Window_Loss") * hours_per_step
            remaining_origin = origin_preclose - window_loss * hours_per_step
            removal_share = (remaining_origin / remaining_energy.replace(0.0, np.nan)).fillna(0.0)
            _assert_balanced(
                removed,
                col("Vehicle_Replacement_Energy_Removed") * removal_share,
                atol,
                f"vehicle {origin} pack removal",
            )
        added = (
            away_stored
            if origin == "Grid"
            else (col("Vehicle_Replacement_Energy_Added") if origin == "Unattributed" else 0.0)
        )
        assert (end >= -atol).all(), f"negative vehicle {origin} origin"
        _assert_balanced(
            end,
            start + (charged + added - discharged - driven - removal) * hours_per_step,
            atol,
            f"vehicle {origin} origin balance",
        )
