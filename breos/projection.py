"""The multi-year projection App and Monte Carlo share.

App (``breos.runners.app``) and Monte Carlo (``breos.montecarlo``) simulate a
configured system year by year. The battery each year runs, the state carried
between years, the year row, and the loop itself are defined here once, so
neither can drift from the other (#179). App runs every year with per-step
frames; Monte Carlo runs summaries.
"""

from __future__ import annotations

import calendar
import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Sequence, TypeAlias, cast

import numpy as np
import pandas as pd

from breos._controller import (
    ControllerBatteryState,
    ControllerCarry,
    DailyDispatchController,
    concatenate_instructions,
)
from breos.app_config import ResolvedAppConfig, build_costs_dict
from breos.battery import (
    END_OF_LIFE_EVENTS_ATTR,
    AlignedSimulationInputs,
    BatteryConfig,
    EndOfLifeEvent,
    SimulationSummary,
    _simulate_detailed_run,
    frame_replaced_capacity_wh,
    retiring_step,
    simulate_energy_balance,
    simulate_energy_balance_summary,
    weighted_column_sums,
)
from breos.dispatch_instructions import DispatchInstructions
from breos.economics import (
    TerminalHealthCredit,
    cost_analysis_projection,
    price_year_rows,
    projection_rates_record,
    replacement_fraction_from_steps,
    terminal_health_credit,
)
from breos.execution import config_has_battery, observed_jit_cache_state, reset_jit_cache_observation
from breos.tariffs import ResolvedTariff
from breos.utils import get_hours_per_step


def build_battery_config(cfg: dict[str, Any], resolved: ResolvedAppConfig, *, initial_soh: float) -> BatteryConfig:
    """Build the battery one projection year runs, starting at ``initial_soh``.

    A configured round-trip efficiency is split evenly across charge and
    discharge, the BatteryConfig default convention. Replacement is on unless
    ``battery_enable_replacement`` turns it off; the economics prices each one
    (ADR 0003 E4). ``battery_allow_terminal_replacement``
    is the project's policy for its final period; :func:`project_years`
    applies it to the final year only. ``battery_replacement_min_remaining_years``
    is the project's minimum service time for a new pack; :func:`project_years`
    tells each year how many project years follow it.
    ``battery_skipped_replacement_action`` keeps or retires a pack whose swap
    any of the three skips.
    """
    battery_kwh = cfg["battery_kwh"]
    efficiency: dict[str, Any] = {}
    if cfg["battery_rte"] is not None:
        one_way = math.sqrt(cfg["battery_rte"])
        efficiency = {"charge_efficiency": one_way, "discharge_efficiency": one_way}
    return BatteryConfig(
        nominal_energy_wh=battery_kwh * 1000,
        initial_soh=initial_soh,
        eol_percentage=cfg["battery_eol_percentage"],
        max_soc=cfg["battery_max_soc"],
        min_soc=cfg["battery_min_soc"],
        inverter_efficiency=cfg["inverter_efficiency"],
        inverter_ac_capacity_w=resolved.inverter_ac_capacity_w,
        enable_replacement=cfg.get("battery_enable_replacement", True),
        allow_terminal_replacement=cfg.get("battery_allow_terminal_replacement", True),
        replacement_min_remaining_years=cfg.get("battery_replacement_min_remaining_years", 0.0),
        skipped_replacement_action=cfg.get("battery_skipped_replacement_action", "keep"),
        calendar_model=cfg["calendar_model"],
        max_charge_power_w=cfg["battery_max_charge_power_w"],
        max_discharge_power_w=cfg["battery_max_discharge_power_w"],
        power_limit_c_rate=cfg["battery_power_limit_c_rate"],
        enable_resistance_fade=cfg.get("enable_resistance_fade", False),
        **efficiency,
    )


def build_pv_only_battery_config(cfg: dict[str, Any], resolved: ResolvedAppConfig) -> BatteryConfig:
    """Build the config a PV-only year runs under.

    PV-only runs still go through the inverter model, so the configured
    efficiency and AC clipping apply as they do with a battery. Monte Carlo's
    PV chain memo uses this too: if it resolved the inverter separately, a
    drift would memoize a conversion for one inverter and spend it on another.
    """
    return BatteryConfig(
        nominal_energy_wh=0,
        inverter_efficiency=cfg["inverter_efficiency"],
        inverter_ac_capacity_w=resolved.inverter_ac_capacity_w,
    )


@dataclass(frozen=True)
class CarryState:
    """Battery state one projection year hands to the next.

    ``degradation_state`` is the engine's native payload (#246). For the native
    engine it already holds the throughput, calendar time, degradation, SOH
    and resistance, and overrides the scalars when the kernel restores it; the
    scalars carry them for year one and for engines whose payload does not.
    ``energy_wh`` is None until a year has run, so year one starts at the
    battery's own initial state of charge. ``controller_carry`` is a daily
    controller's policy state, observations and project clock (ADR 0002
    A11); it is not degradation state and never enters
    :meth:`simulation_kwargs`.
    """

    energy_wh: float | None = None
    pv_origin_energy_wh: float | None = None
    grid_origin_energy_wh: float | None = None
    fec: float = 0.0
    calendar_seconds: float = 0.0
    cycle_degradation: float = 0.0
    calendar_degradation: float = 0.0
    resistance_growth: float = 0.0
    soh_pct: float = 100.0
    degradation_state: dict[str, Any] | None = None
    controller_carry: ControllerCarry | None = None
    # True once a year has retired the battery (ADR 0003 E12); every later
    # year starts with it switched off.
    battery_retired: bool = False

    def simulation_kwargs(self) -> dict[str, Any]:
        """Keyword arguments that start a simulation from this state."""
        kwargs: dict[str, Any] = {
            "initial_fec": self.fec,
            "initial_calendar_seconds": self.calendar_seconds,
            "initial_resistance_growth": self.resistance_growth,
            "initial_cumulative_cycle_deg": self.cycle_degradation,
            "initial_cumulative_cal_deg": self.calendar_degradation,
            "initial_degradation_state": self.degradation_state,
        }
        if self.energy_wh is not None:
            kwargs["initial_energy_wh"] = self.energy_wh
            kwargs["initial_pv_origin_energy_wh"] = self.pv_origin_energy_wh or 0.0
            kwargs["initial_grid_origin_energy_wh"] = self.grid_origin_energy_wh or 0.0
        if self.battery_retired:
            kwargs["battery_retired"] = True
        return kwargs

    def after_frames(
        self,
        results_df: pd.DataFrame,
        degradation_df: pd.DataFrame,
        degradation_state: dict[str, Any] | None,
        *,
        has_battery: bool,
    ) -> CarryState:
        """The state after a year run with per-step frames."""
        if not has_battery:
            return replace(self, degradation_state=degradation_state)
        changes: dict[str, Any] = {
            "energy_wh": float(results_df["Battery_Energy_End"].iloc[-1]),
            "pv_origin_energy_wh": float(results_df["Battery_PV_Origin_Energy_End"].iloc[-1]),
            "grid_origin_energy_wh": float(results_df["Battery_Grid_Origin_Energy_End"].iloc[-1]),
            "degradation_state": degradation_state,
        }
        if not degradation_df.empty:
            last = degradation_df.iloc[-1]
            changes.update(
                fec=last["Cumulative_FEC"],
                calendar_seconds=last["Cumulative_Calendar_Seconds"],
                cycle_degradation=last["Cumulative_Cycle_Degradation"],
                calendar_degradation=last["Cumulative_Calendar_Degradation"],
                soh_pct=last["SOH"],
            )
            # The frame reports resistance only when the fade model is on;
            # otherwise the carried value stays put.
            if "Resistance_Growth" in degradation_df.columns:
                changes["resistance_growth"] = last["Resistance_Growth"]
        return replace(self, **changes)

    def after_summary(self, summary: SimulationSummary, *, has_battery: bool, resistance_fade: bool) -> CarryState:
        """The state after a year run as a summary; the same rule as :meth:`after_frames`."""
        if not has_battery:
            return replace(self, degradation_state=summary.final_degradation_state)
        changes: dict[str, Any] = {
            "energy_wh": summary.carried_energy_wh,
            "pv_origin_energy_wh": summary.carried_pv_origin_energy_wh,
            "grid_origin_energy_wh": summary.carried_grid_origin_energy_wh,
            "degradation_state": summary.final_degradation_state,
        }
        if summary.has_degradation_rows:
            changes.update(
                fec=summary.fec_cum,
                calendar_seconds=summary.cumulative_calendar_seconds,
                cycle_degradation=summary.cumulative_cycle_degradation,
                calendar_degradation=summary.cumulative_calendar_degradation,
                soh_pct=summary.final_soh_percent,
            )
            if resistance_fade:
                changes["resistance_growth"] = summary.resistance_growth
        return replace(self, **changes)


# Ledger columns every year row reports as annual kWh, keyed by row column.
_DIAGNOSTIC_COLUMNS = {
    "PV_Direct_Inverter_Loss_kWh": "PV_Direct_Inverter_Loss",
    "Battery_Inverter_Loss_kWh": "Battery_Inverter_Loss",
    "Battery_Charge_Input_kWh": "Battery_Charge_Input",
    "Grid_AC_To_Battery_kWh": "Grid_AC_To_Battery",
    "Grid_Charge_Conversion_Loss_kWh": "Grid_Charge_Conversion_Loss",
    "Battery_Discharge_DC_kWh": "Battery_Discharge_DC",
    "Battery_AC_To_Load_kWh": "Battery_AC_To_Load",
    "Battery_Charge_Loss_kWh": "Battery_Charge_Loss",
    "Battery_Discharge_Loss_kWh": "Battery_Discharge_Loss",
    "Battery_Standby_Loss_kWh": "Standby_Loss",
    "Capacity_Window_Loss_kWh": "Capacity_Window_Loss",
    "Replacement_Energy_Removed_kWh": "Battery_Replacement_Energy_Removed",
    "Replacement_Energy_Added_kWh": "Battery_Replacement_Energy_Added",
}
_ROW_SUM_COLUMNS = (
    "PV_DC",
    "PV_Production",
    "PV_AC_To_Load",
    "PV_Origin_Battery_AC_To_Load",
    "Grid_Origin_Battery_AC_To_Load",
    "Houseload",
    "Import_From_Grid",
    "PV_AC_Export",
    "PV_DC_Curtailed",
    "Inverter_Loss",
    "Battery_Charge_Stored",
    "Battery_SOC_Normalized",
    "Battery_SOC_Absolute",
    "T_cell",
    *_DIAGNOSTIC_COLUMNS.values(),
)


def build_year_row(
    year_idx: int,
    sums_w: Mapping[str, float],
    hours_per_step: float,
    carry: CarryState,
    *,
    has_battery: bool,
    n_replacements: int,
    replaced_capacity_wh: float,
    replacement_steps: Sequence[int],
    n_steps: int,
    pv_degradation_factor: float,
    in_service_steps: int,
    in_service_t_cell_sum: float | None,
    annual_fec: float = 0.0,
    extra: Mapping[str, Any] | None = None,
    money: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Build the one year-row schema App, Monte Carlo and the optimizer report.

    ``sums_w`` holds each results column's sum over the year's steps; a
    summary's ``column_sums`` and a frame's column sums are the same floats.
    ``carry`` is the state at the end of the year and ``annual_fec`` the
    rainflow cycles every pack accumulated in it. ``in_service_steps`` are
    the steps the battery served, every step unless it was retired, and
    ``in_service_t_cell_sum`` its ``T_cell`` summed over them. ``extra``
    columns (Monte Carlo's sampled weather year and load scale) follow
    ``PV_Degradation_Factor``.
    """

    def kwh(column: str) -> float:
        return float(sums_w[column] * hours_per_step / 1000)

    direct_pv_ac_kwh = kwh("PV_AC_To_Load")
    pv_origin_battery_ac_kwh = kwh("PV_Origin_Battery_AC_To_Load")
    total_load = (sums_w["Houseload"] / 1000) * hours_per_step
    total_import = (sums_w["Import_From_Grid"] / 1000) * hours_per_step
    total_export = (sums_w["PV_AC_Export"] / 1000) * hours_per_step
    row: dict[str, Any] = {
        "Year": year_idx + 1,
        # Delivered AC: PV straight to load, PV-origin battery discharge to
        # load, and export.
        "PV_Production_kWh": direct_pv_ac_kwh + pv_origin_battery_ac_kwh + total_export,
        "PV_DC_Generation_kWh": kwh("PV_DC"),
        "Direct_PV_AC_Load_kWh": direct_pv_ac_kwh,
        "PV_Origin_Battery_AC_Load_kWh": pv_origin_battery_ac_kwh,
        # Grid energy shifted through the battery (ADR 0002 A10): what the
        # battery delivered from grid charge, and what that charge imported.
        "Grid_Origin_Battery_AC_Load_kWh": kwh("Grid_Origin_Battery_AC_To_Load"),
        "Self_Consumption_kWh": direct_pv_ac_kwh + pv_origin_battery_ac_kwh,
        "Curtailment_DC_kWh": kwh("PV_DC_Curtailed"),
        "Load_kWh": total_load,
        "Import_kWh": total_import,
        "Export_kWh": total_export,
        "Grid_Independence_%": (1 - total_import / total_load) * 100 if total_load > 0 else 0,
        "Battery_SOH_%": carry.soh_pct if has_battery else None,
        "Battery_Cumulative_FEC": carry.fec,
        "Battery_Cumulative_Calendar_Seconds": carry.calendar_seconds,
        "Battery_Cumulative_Cycle_Degradation": carry.cycle_degradation,
        "Battery_Cumulative_Calendar_Degradation": carry.calendar_degradation,
        "Battery_Resistance_Growth": carry.resistance_growth,
        "Replacements": n_replacements,
        # The nominal capacity swapped in; the economics prices it (ADR 0003 E4).
        "Replaced_Capacity_kWh": replaced_capacity_wh / 1000,
        # Where in the year the pack was swapped, so the economics can book
        # the outlay at that instant rather than at a year boundary. NaN in a
        # year without a replacement.
        "Replacement_Year_Fraction": replacement_fraction_from_steps(replacement_steps, n_steps),
        "PV_Degradation_Factor": pv_degradation_factor,
        # The simulated duration, which the fixed charge is billed on (ADR 0003
        # E5): 8760 hours for a common year, 8784 for a leap year.
        "Simulated_Hours": n_steps * hours_per_step,
        **(extra or {}),
    }
    # Diagnostics: reductions of ledger columns, so one execution path can be
    # compared with another field by field without rerunning it.
    row.update({name: kwh(column) for name, column in _DIAGNOSTIC_COLUMNS.items()})
    row["Battery_Carried_Energy_Wh"] = float(carry.energy_wh) if has_battery and carry.energy_wh is not None else None
    row["Battery_Carried_PV_Origin_Energy_Wh"] = (
        float(carry.pv_origin_energy_wh) if has_battery and carry.pv_origin_energy_wh is not None else None
    )
    row["Battery_Carried_Grid_Origin_Energy_Wh"] = (
        float(carry.grid_origin_energy_wh) if has_battery and carry.grid_origin_energy_wh is not None else None
    )
    row["Replacement_Steps"] = ";".join(str(step) for step in replacement_steps)
    row["Inverter_Loss_kWh"] = kwh("Inverter_Loss")
    # Cell-side energy in and out, so the pair reflects round-trip loss and
    # feeds cycle ageing directly. Charge is measured after charging losses
    # and discharge before inverter losses.
    row["Battery_Charge_Throughput_kWh"] = kwh("Battery_Charge_Stored")
    row["Battery_Discharge_Throughput_kWh"] = kwh("Battery_Discharge_DC")
    # State levels are averaged, not integrated. Normalized SOC is the
    # position in the usable window; absolute SOC is the fraction of the
    # SOH-derated pack, so it rises as the pack fades.
    row["Battery_SOC_Normalized_Mean_%"] = float(sums_w["Battery_SOC_Normalized"] / n_steps * 100.0)
    row["Battery_SOC_Absolute_Mean_%"] = float(sums_w["Battery_SOC_Absolute"] / n_steps * 100.0)
    # The time the battery served, and the cell temperature the aging model
    # saw, averaged over it: a retired battery has no cell temperature, and
    # a PV-only system no battery.
    row["Battery_In_Service_Hours"] = in_service_steps * hours_per_step if has_battery else None
    row["Battery_Cell_Temperature_Mean_C"] = (
        float(in_service_t_cell_sum / in_service_steps)
        if has_battery and in_service_steps and in_service_t_cell_sum is not None
        else None
    )
    # Cumulative FEC belongs to the installed pack and restarts at zero on
    # replacement, so the year's own count comes from the all-pack total.
    row["Battery_Annual_FEC"] = float(annual_fec)
    # Year-1-price money from a tariff (ADR 0003 E7); without one, economics
    # prices the energy at the flat rates.
    row.update(money or {})
    return row


def _tariff_weights(tariff: ResolvedTariff) -> dict[str, tuple[str, np.ndarray]]:
    import_prices = np.asarray(tariff.import_price_per_kwh, dtype=float)
    weights = {
        "Import_Cost": ("Import_From_Grid", import_prices),
        "Export_Revenue": ("PV_AC_Export", np.asarray(tariff.export_price_per_kwh, dtype=float)),
        # The no-system household buys its whole load at the same prices.
        "Baseline_Import_Cost": ("Houseload", import_prices),
        # The part of Import_Cost bought to charge the battery.
        "Grid_Charge_Cost": ("Grid_AC_To_Battery", import_prices),
    }
    if tariff.network_price_per_kwh is not None:
        network_prices = np.asarray(tariff.network_price_per_kwh, dtype=float)
        # The network part of each household's import, which caps its annual
        # network credit (ADR 0002 A15).
        weights[_NETWORK_IMPORT] = ("Import_From_Grid", network_prices)
        weights[_BASELINE_NETWORK_IMPORT] = ("Houseload", network_prices)
    return weights


# The weighted sums of each household's import at the network prices.
_NETWORK_IMPORT = "Network_Import_Charge"
_BASELINE_NETWORK_IMPORT = "Baseline_Network_Import_Charge"
# The year-row columns of an annual network credit, at year-1 prices: each
# household's eligible network charges (the cap basis) and its credit.
SYSTEM_NETWORK_CREDIT_COLUMNS = ("Network_Charge", "Network_Credit")
BASELINE_NETWORK_CREDIT_COLUMNS = ("Baseline_Network_Charge", "Baseline_Network_Credit")


def _credit_year_fraction(tariff: ResolvedTariff, billed_days: float | None) -> float:
    """The share of a year an annual network credit is billed on (ADR 0002 A15).

    A simulated year is a whole year, a leap year too. A [period] window of
    ``billed_days`` civil days lies in one calendar year, its tariff's first
    civil year, and is that share of its days.
    """
    if billed_days is None or pd.isna(billed_days):
        return 1.0
    year = int(tariff.index[0].tz_convert(tariff.timezone).year)
    return float(billed_days) / (366 if calendar.isleap(year) else 365)


def _network_credit_money(
    tariff: ResolvedTariff, network_import_charge: float, billed_days: float | None, columns: tuple[str, str]
) -> dict[str, float]:
    """One household's eligible network charges and its credit, from its import at the network prices.

    The credit is the annual amount, capped at the eligible network charges:
    the network part of the import plus the network fixed amount, both for
    the billed share of the year.
    """
    credit = tariff.prices.annual_network_credit
    assert credit is not None
    fraction = _credit_year_fraction(tariff, billed_days)
    charge = network_import_charge + credit.network_fixed_per_year * fraction
    return {columns[0]: charge, columns[1]: min(credit.amount_per_year * fraction, charge)}


# The flows a tariff prices, by the money column each fills and the frame
# column it is summed from, with the step price that applies to it.
_PRICED_FLOWS: dict[str, tuple[str, str]] = {
    "Import_Cost": ("Import_From_Grid", "import"),
    "Export_Revenue": ("PV_AC_Export", "export"),
    "Baseline_Import_Cost": ("Houseload", "import"),
    "Grid_Charge_Cost": ("Grid_AC_To_Battery", "import"),
}
# The frame columns those flows are summed from, once each.
_PRICED_FLOW_COLUMNS = tuple(dict.fromkeys(column for column, _kind in _PRICED_FLOWS.values()))


def _period_energy_name(money_column: str, bucket: str) -> str:
    return f"{_PRICED_FLOWS[money_column][0]}_kWh@{bucket}"


def _price_buckets(tariff: ResolvedTariff) -> tuple[str, ...]:
    """Each step's price bucket: its period, or its month season and period.

    Every step in a bucket has one price under any prices for the schedule,
    so energy summed by bucket can be re-priced exactly.
    """
    if tariff.season_labels is None:
        return tariff.period_labels
    return tuple(
        f"{season}/{period}" for season, period in zip(tariff.season_labels, tariff.period_labels, strict=True)
    )


def _period_weights(tariff: ResolvedTariff) -> dict[str, tuple[str, np.ndarray]]:
    """One 0/1 mask per price bucket and priced flow, so each year also records its energy by bucket.

    A bucket is a tariff period, or a month season and period for a schedule
    with month seasons. That energy is what :func:`reprice_tariff_year_rows`
    re-prices a year from when only the prices change.
    """
    buckets = _price_buckets(tariff)
    labels = np.asarray(buckets, dtype=object)
    weights: dict[str, tuple[str, np.ndarray]] = {}
    for bucket in dict.fromkeys(buckets):
        mask = (labels == bucket).astype(float)
        for money_column, (column, _price) in _PRICED_FLOWS.items():
            weights[_period_energy_name(money_column, bucket)] = (column, mask)
    return weights


def _fixed_charge(tariff: ResolvedTariff, simulated_hours: float) -> float:
    # Billed on the simulated duration, as the flat path is (ADR 0003 E5).
    return tariff.prices.fixed_charge_per_day * (simulated_hours / 24)


def _billed_fixed_charge(tariff: ResolvedTariff, row: pd.Series) -> float:
    # A [period] row bills its civil days (Billed_Days), whatever DST does to
    # its hours; any other row bills its simulated duration.
    billed_days = row.get("Billed_Days")
    if billed_days is not None and not pd.isna(billed_days):
        return tariff.prices.fixed_charge_per_day * float(billed_days)
    return _fixed_charge(tariff, row["Simulated_Hours"])


def _tariff_money(
    tariff: ResolvedTariff,
    weighted_w: Mapping[str, float],
    hours_per_step: float,
    n_steps: int,
    billed_days: float | None = None,
) -> dict[str, float]:
    """A year's money at year-1 prices from its price-weighted power sums.

    The fixed charge is billed on ``billed_days`` when given (a [period]
    window's civil days), otherwise on the simulated duration.
    """
    money = {name: float(weighted_w[name] * hours_per_step / 1000) for name in _PRICED_FLOWS}
    money["Fixed_Charge"] = (
        tariff.prices.fixed_charge_per_day * billed_days
        if billed_days is not None
        else _fixed_charge(tariff, n_steps * hours_per_step)
    )
    if tariff.prices.annual_network_credit is not None:
        for name, columns in (
            (_NETWORK_IMPORT, SYSTEM_NETWORK_CREDIT_COLUMNS),
            (_BASELINE_NETWORK_IMPORT, BASELINE_NETWORK_CREDIT_COLUMNS),
        ):
            money.update(
                _network_credit_money(tariff, float(weighted_w[name] * hours_per_step / 1000), billed_days, columns)
            )
    return money


def _reference_weights(reference: ResolvedTariff) -> dict[str, tuple[str, np.ndarray]]:
    # The no-system household buys its whole load at the reference prices.
    weights = {"Baseline_Import_Cost": ("Houseload", np.asarray(reference.import_price_per_kwh, dtype=float))}
    if reference.network_price_per_kwh is not None:
        weights[_BASELINE_NETWORK_IMPORT] = ("Houseload", np.asarray(reference.network_price_per_kwh, dtype=float))
    return weights


def _reference_money(
    reference: ResolvedTariff,
    weighted_w: Mapping[str, float],
    hours_per_step: float,
    n_steps: int,
    billed_days: float | None = None,
) -> dict[str, float]:
    """A year's no-system money at the reference tariff's year-1 prices, billed as :func:`_tariff_money` bills.

    The reference's own annual network credit, if it has one, replaces any
    the system tariff gave the no-system household.
    """
    money = {
        "Baseline_Import_Cost": float(weighted_w["Baseline_Import_Cost"] * hours_per_step / 1000),
        "Baseline_Fixed_Charge": (
            reference.prices.fixed_charge_per_day * billed_days
            if billed_days is not None
            else _fixed_charge(reference, n_steps * hours_per_step)
        ),
    }
    if reference.prices.annual_network_credit is not None:
        money.update(
            _network_credit_money(
                reference,
                float(weighted_w[_BASELINE_NETWORK_IMPORT] * hours_per_step / 1000),
                billed_days,
                BASELINE_NETWORK_CREDIT_COLUMNS,
            )
        )
    return money


def price_reference_year_rows(
    yearly_df: pd.DataFrame, houseload_w: np.ndarray, reference: ResolvedTariff, freq: str
) -> pd.DataFrame:
    """Year rows with the no-system money re-priced at ``reference``, from one year's per-step load.

    For revaluation without re-simulation. The reference prices only the
    household load, which no price changes, and every App year replays the
    one load on the one calendar, so ``houseload_w``, the first year's
    ``Houseload``, is every year's. The sums are the year loop's own, so a
    reference priced here gives the floats a fresh run gives.
    """
    if len(houseload_w) != len(reference.index):
        raise ValueError("the household load and the reference tariff are on different calendars")
    hours_per_step = get_hours_per_step(freq)
    weights = _reference_weights(reference)
    weighted = weighted_column_sums({"Houseload": np.asarray(houseload_w)}, weights)
    repriced = yearly_df.drop(columns=[column for column in BASELINE_NETWORK_CREDIT_COLUMNS if column in yearly_df])
    repriced["Baseline_Import_Cost"] = float(weighted["Baseline_Import_Cost"] * hours_per_step / 1000)
    repriced["Baseline_Fixed_Charge"] = [_billed_fixed_charge(reference, row) for _, row in repriced.iterrows()]
    if reference.prices.annual_network_credit is not None:
        network = float(weighted[_BASELINE_NETWORK_IMPORT] * hours_per_step / 1000)
        credits = [
            _network_credit_money(reference, network, _billed_days(row), BASELINE_NETWORK_CREDIT_COLUMNS)
            for _, row in repriced.iterrows()
        ]
        for column in BASELINE_NETWORK_CREDIT_COLUMNS:
            repriced[column] = [money[column] for money in credits]
    return repriced


def _billed_days(row: pd.Series) -> float | None:
    billed_days = row.get("Billed_Days")
    return None if billed_days is None or pd.isna(billed_days) else float(billed_days)


def _period_prices(tariff: ResolvedTariff, kind: str) -> dict[str, float]:
    prices = {
        "import": tariff.import_price_per_kwh,
        "export": tariff.export_price_per_kwh,
        "network": tariff.network_price_per_kwh,
    }[kind]
    assert prices is not None
    by_bucket: dict[str, float] = {}
    for bucket, price in zip(_price_buckets(tariff), prices, strict=True):
        by_bucket.setdefault(bucket, float(price))
    return by_bucket


def reprice_tariff_year_rows(
    yearly_df: pd.DataFrame, period_energy: pd.DataFrame, tariff: ResolvedTariff
) -> pd.DataFrame:
    """Year rows re-priced at ``tariff``'s prices, from each year's energy by tariff period.

    For revaluation without re-simulation: ``period_energy`` is the
    :attr:`ProjectionRun.period_energy` of a run on a tariff with the same
    schedule, so only the prices differ. Each money column becomes the sum over
    periods (month season and period, with month seasons) of energy times
    price, and the fixed charge the new daily charge
    over the simulated hours. A fresh simulation sums energy times price per
    step instead, so the two agree to rounding, not bit for bit. The annual
    network credit of each household is re-priced too
    (:func:`reprice_network_credit_year_rows`).
    """
    repriced = yearly_df.copy()
    prices = {kind: _period_prices(tariff, kind) for kind in ("import", "export")}
    for money_column, (_column, kind) in _PRICED_FLOWS.items():
        total = np.zeros(len(repriced))
        for period, price in prices[kind].items():
            total = total + period_energy[_period_energy_name(money_column, period)].to_numpy(dtype=float) * price
        repriced[money_column] = total
    repriced["Fixed_Charge"] = [_billed_fixed_charge(tariff, row) for _, row in repriced.iterrows()]
    return reprice_network_credit_year_rows(repriced, period_energy, tariff)


def reprice_network_credit_year_rows(
    yearly_df: pd.DataFrame, period_energy: pd.DataFrame, tariff: ResolvedTariff
) -> pd.DataFrame:
    """Year rows with only the annual network credits re-priced at ``tariff``'s, from each year's energy by period.

    The credit of each household (ADR 0002 A15), with its cap basis, from
    the same retained energy :func:`reprice_tariff_year_rows` re-prices from;
    without a credit on ``tariff`` the columns are dropped. Every other
    column is kept as it is.
    """
    repriced = yearly_df.drop(
        columns=[
            column
            for column in (*SYSTEM_NETWORK_CREDIT_COLUMNS, *BASELINE_NETWORK_CREDIT_COLUMNS)
            if column in yearly_df
        ]
    )
    if tariff.prices.annual_network_credit is None:
        return repriced
    network_prices = _period_prices(tariff, "network")
    for money_column, columns in (
        ("Import_Cost", SYSTEM_NETWORK_CREDIT_COLUMNS),
        ("Baseline_Import_Cost", BASELINE_NETWORK_CREDIT_COLUMNS),
    ):
        network = np.zeros(len(repriced))
        for period, price in network_prices.items():
            network = network + period_energy[_period_energy_name(money_column, period)].to_numpy(dtype=float) * price
        credits = [
            _network_credit_money(tariff, float(charge), _billed_days(row), columns)
            for charge, (_, row) in zip(network, repriced.iterrows(), strict=True)
        ]
        for column in columns:
            repriced[column] = [money[column] for money in credits]
    return repriced


def _year_money(
    tariff: ResolvedTariff | None,
    reference: ResolvedTariff | None,
    weighted_w: Mapping[str, float],
    hours_per_step: float,
    n_steps: int,
    billed_days: float | None,
) -> dict[str, float]:
    """A year's money from the system tariff, then the no-system reference, which prices that household instead."""
    money = _tariff_money(tariff, weighted_w, hours_per_step, n_steps, billed_days) if tariff is not None else {}
    if reference is not None:
        if reference.prices.annual_network_credit is None:
            for column in BASELINE_NETWORK_CREDIT_COLUMNS:
                money.pop(column, None)
        money.update(_reference_money(reference, weighted_w, hours_per_step, n_steps, billed_days))
    return money


def price_tariff_year_rows_by_step(
    yearly_df: pd.DataFrame, priced_flows: Sequence[Mapping[str, np.ndarray]], tariff: ResolvedTariff, freq: str
) -> pd.DataFrame:
    """Year rows priced at ``tariff`` from each year's retained step flows, on any schedule.

    For revaluation without re-simulation of a run whose dispatch never saw
    a tariff: ``priced_flows`` is its :attr:`ProjectionRun.priced_flows`,
    and ``tariff`` is resolved on its calendar. The money columns of the old
    prices, a reference tariff's included, give way to the ``tariff``'s, and
    the no-system household pays the ``tariff`` too;
    :func:`price_reference_year_rows` then prices a reference. The sums are
    the year loop's own, so the rows carry the floats a fresh run gives.
    """
    if len(priced_flows) != len(yearly_df):
        raise ValueError(f"{len(priced_flows)} years of step flows for {len(yearly_df)} year rows")
    hours_per_step = get_hours_per_step(freq)
    weights = _tariff_weights(tariff)
    money_rows = []
    for (_, row), flows in zip(yearly_df.iterrows(), priced_flows, strict=True):
        n_steps = len(flows["Import_From_Grid"])
        if n_steps != len(tariff.index):
            raise ValueError("the step flows and the tariff are on different calendars")
        billed_days = row.get("Billed_Days")
        money_rows.append(
            _tariff_money(
                tariff,
                weighted_column_sums(flows, weights),
                hours_per_step,
                n_steps,
                None if billed_days is None or pd.isna(billed_days) else float(billed_days),
            )
        )
    money_columns = (
        *_PRICED_FLOWS,
        "Fixed_Charge",
        "Baseline_Fixed_Charge",
        *SYSTEM_NETWORK_CREDIT_COLUMNS,
        *BASELINE_NETWORK_CREDIT_COLUMNS,
    )
    unpriced = yearly_df.drop(columns=[column for column in money_columns if column in yearly_df.columns])
    # The money columns go last, where the year loop puts them.
    return pd.concat([unpriced, pd.DataFrame(money_rows, index=yearly_df.index)], axis=1)


def _check_tariff_calendar(tariff: ResolvedTariff, index: pd.DatetimeIndex) -> None:
    same = len(index) == len(tariff.index) and bool((index.tz_convert("UTC") == tariff.index.tz_convert("UTC")).all())
    if not same:
        raise ValueError(
            "The tariff was resolved on a different calendar from the simulated year; resolve it on the "
            "simulation index."
        )


@dataclass(frozen=True)
class ProjectionYear:
    """The inputs of one projection year.

    Give either the per-step series (``pv_dc`` and ``houseload``, with
    ``temperature_series`` when there is a battery), which the projection
    runs with full frames, or ``aligned`` inputs, which it runs as a summary.
    """

    pv_degradation_factor: float
    pv_dc: pd.Series | None = None
    houseload: pd.DataFrame | None = None
    temperature_series: pd.Series | None = None
    aligned: AlignedSimulationInputs | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class YearStart:
    """What a year planner sees as a project year begins (ADR 0002 A16).

    ``year`` is the year's inputs, with its PV degradation already applied,
    and ``battery_config`` the battery it runs. ``battery_state`` is the
    state the year's first step dispatches from: the carried stored energy
    and origins, and the health and efficiencies after the degradation
    engine has restored them.
    """

    year_idx: int
    year: ProjectionYear
    battery_config: BatteryConfig
    battery_state: ControllerBatteryState


# Returns one project year's instructions, on the replayed calendar.
YearPlanner: TypeAlias = Callable[[YearStart], DispatchInstructions]
# One set for every year, one set per year, or a planner asked at each year start.
YearInstructions: TypeAlias = DispatchInstructions | Sequence[DispatchInstructions] | YearPlanner


def _instructions_by_year(
    instructions: YearInstructions | None, years: int
) -> Callable[[int], DispatchInstructions | YearPlanner | None]:
    """Each year's static set or planner, from what :func:`project_years` was given."""
    if instructions is None or isinstance(instructions, DispatchInstructions) or callable(instructions):
        single = instructions
        return lambda _year_idx: single
    sets = tuple(instructions)
    if len(sets) != years:
        raise ValueError(f"per-year instructions hold {len(sets)} sets; the projection has {years} years")
    if not all(isinstance(item, DispatchInstructions) for item in sets):
        raise TypeError("per-year instructions must be DispatchInstructions, one per project year")
    return sets.__getitem__


@dataclass(frozen=True)
class ProjectionRun:
    """What a multi-year projection produced."""

    yearly_df: pd.DataFrame
    carry: CarryState
    total_replacements: int
    # The first year's per-step frame; None for a summary projection.
    first_year_results_df: pd.DataFrame | None
    jit_cache_states: list[str]
    # With a tariff, each year's priced energy by tariff period, or by month
    # season and period (kWh), one row per year, so a price change can be
    # re-priced without re-simulating.
    period_energy: pd.DataFrame | None = None
    # With a daily controller, the instructions it executed, one per
    # simulated step in project order across every year (ADR 0002 A11).
    # Slots of a decision that no year dispatched are not in it.
    controller_instructions: DispatchInstructions | None = None
    # With static instructions or a year planner, the set each project year
    # dispatched on, in year order (ADR 0002 A16).
    year_instructions: tuple[DispatchInstructions, ...] | None = None
    # With record_priced_flows, each year's per-step flows a tariff prices
    # (W), by frame column, so the run can be priced on another schedule
    # without re-simulating. A flow equal to the year before's is that
    # array, kept once: the load, and the zero grid charge of a greedy run.
    priced_flows: tuple[dict[str, np.ndarray], ...] | None = None
    # Each end-of-life crossing, in project order, as end_of_life_record
    # reports it (ADR 0003 E11).
    end_of_life_events: tuple[dict[str, Any], ...] = ()


def end_of_life_record(event: EndOfLifeEvent, year_idx: int, n_steps: int) -> dict[str, Any]:
    """One span's end-of-life crossing on the project clock.

    ``time_years`` is measured from commissioning as ``Replacement_Time_Years``
    books a swap, ``year_idx + (step + 1) / n_steps``, so a replaced
    crossing's time is the booked one. ``date`` is the closing step's
    calendar date moved forward by ``year_idx`` years: every project year
    replays the first year's calendar (ADR 0002 A2). App and the optimizer
    reuse one year's series; Monte Carlo restamps every sampled weather year
    to ``target_year``, so its dates run from ``target_year``.
    ``soh_pct`` is the health the check compared with the threshold, before
    any swap.
    """
    return {
        "year": year_idx + 1,
        "time_years": year_idx + (event.step + 1.0) / n_steps,
        "date": (event.timestamp + pd.DateOffset(years=year_idx)).date().isoformat(),
        "action": event.action,
        "reason": event.reason,
        "soh_pct": float(event.soh_pct),
    }


def first_end_of_life_metrics(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """A run's first end-of-life crossing as flat metrics, NaN and None without one.

    Monte Carlo reports these per trajectory and the projected optimizer per
    design, beside their replacement counts.
    """
    first: Mapping[str, Any] = events[0] if events else {}
    return {
        "first_end_of_life_years": float(first.get("time_years", math.nan)),
        "first_end_of_life_action": first.get("action"),
        "first_end_of_life_reason": first.get("reason"),
        "first_end_of_life_soh_pct": float(first.get("soh_pct", math.nan)),
    }


def project_years(
    years: int,
    year_inputs: Callable[[int], ProjectionYear],
    *,
    battery_config: Callable[[float], BatteryConfig],
    freq: str,
    has_battery: bool,
    execution_backend: str,
    degradation_engine: str = "native",
    blast_model: str | None = None,
    initial_carry: CarryState | None = None,
    observe_jit_per_year: bool = False,
    tariff: ResolvedTariff | None = None,
    instructions: YearInstructions | None = None,
    record_period_energy: bool = False,
    day_controller: DailyDispatchController | None = None,
    replay_seam: bool = True,
    reference_tariff: ResolvedTariff | None = None,
    record_priced_flows: bool = False,
) -> ProjectionRun:
    """Simulate ``years`` project years, carrying the battery from one to the next.

    App, Monte Carlo and the projected optimizer all run this loop.
    ``year_inputs(year_idx)`` supplies each year, and ``battery_config(soh)``
    builds the battery at the carried SOH. The rainflow residue is counted
    once, at the end of the last year. ``observe_jit_per_year`` records the
    Numba cache state of every year, as App reports it.

    The battery's ``allow_terminal_replacement`` is the project's policy for
    the final degradation period of its last year. Every earlier year runs a
    copy that allows it, because the next year inherits the pack a year-end
    replacement installs. Its ``replacement_min_remaining_years`` is measured
    to the end of the project: with a positive minimum each year runs a copy
    whose ``replacement_years_after_span`` is the number of years after it,
    so a swap is skipped when the project time left after its
    ``Replacement_Time_Years`` is below the minimum. A skipped swap keeps or
    retires the pack (``skipped_replacement_action``); a retired pack is
    carried switched off into every later year. The run's
    ``end_of_life_events`` lists every crossing of the end-of-life threshold,
    replaced or not, on the project clock (:func:`end_of_life_record`).

    With a ``tariff``, resolved on the simulation calendar, each year row
    carries its import cost, export revenue, no-system import cost and fixed
    charge at year-1 prices, from the step energy times the step price. Every
    year replays the one calendar (ADR 0002 A2), and so do smart-charging
    ``instructions``, resolved on that calendar. ``record_period_energy``
    also keeps each year's priced energy by tariff period, which App.revalue
    re-prices from. ``record_priced_flows`` keeps each per-step year's
    priced flows step by step, with or without a tariff, which App.revalue
    prices on another schedule from when no tariff steered the dispatch.

    ``instructions`` is one set every year replays, a sequence of one set
    per project year, or a :data:`YearPlanner` (ADR 0002 A16). A planner is
    asked once as each per-step year begins, with a :class:`YearStart` that
    holds the state the year opens in, and returns that year's set. The run's
    ``year_instructions`` records the set each year dispatched on.

    A private ``day_controller`` (ADR 0002 A11) decides each civil day of
    the ``tariff``'s calendar in place of static ``instructions``, on
    per-step years with a battery. Its carry crosses the years beside the
    battery state, and the run's ``controller_instructions`` join the
    instructions each year executed. ``replay_seam`` says the next year
    replays the calendar, so a civil day cut by a year's end continues at
    the next year's head; a standalone ``[period]`` passes False.

    A ``reference_tariff``, resolved on the same calendar, prices the
    no-system household instead: each row's ``Baseline_Import_Cost`` is the
    year's load at its prices and ``Baseline_Fixed_Charge`` its fixed
    charge. It never touches the dispatch. Without one, the no-system
    household pays the system's own prices.

    A tariff or reference with an annual network credit (ADR 0002 A15) adds
    its household's eligible network charges and credit to each row. The
    system tariff's credit covers both households unless a reference prices
    the no-system one, which then has the reference's credit or none.
    """
    instructions_for_year = _instructions_by_year(instructions, years)
    if day_controller is not None:
        if instructions is not None:
            raise ValueError("pass either instructions or a daily controller, not both")
        if tariff is None:
            raise ValueError("a daily controller needs a resolved tariff")
        if not has_battery:
            raise ValueError("a daily controller needs a battery")
    hours_per_step = get_hours_per_step(freq)
    record_period_energy = record_period_energy and tariff is not None
    period_weights = _period_weights(tariff) if record_period_energy and tariff is not None else {}
    weights = {**_tariff_weights(tariff), **period_weights} if tariff is not None else None
    if reference_tariff is not None:
        weights = {**(weights or {}), **_reference_weights(reference_tariff)}
    carry = initial_carry or CarryState()
    rows: list[dict[str, Any]] = []
    period_rows: list[dict[str, float]] = []
    total_replacements = 0
    first_year_results_df: pd.DataFrame | None = None
    jit_cache_states: list[str] = []
    executed: list[DispatchInstructions] = []
    dispatched: list[DispatchInstructions] = []
    priced_flows: list[dict[str, np.ndarray]] = []
    end_of_life_events: list[dict[str, Any]] = []

    for year_idx in range(years):
        year = year_inputs(year_idx)
        started_retired = carry.battery_retired
        batt_cfg = battery_config(carry.soh_pct)
        if year_idx < years - 1 and not batt_cfg.allow_terminal_replacement:
            batt_cfg = replace(batt_cfg, allow_terminal_replacement=True)
        if batt_cfg.replacement_min_remaining_years > 0.0:
            batt_cfg = replace(batt_cfg, replacement_years_after_span=years - 1 - year_idx)
        year_set = instructions_for_year(year_idx)
        planner = None if year_set is None or isinstance(year_set, DispatchInstructions) else year_set
        static = year_set if isinstance(year_set, DispatchInstructions) else None
        common = {
            **carry.simulation_kwargs(),
            "battery_config": batt_cfg,
            "freq": freq,
            "degradation_engine": degradation_engine,
            "blast_model": blast_model,
            "return_degradation_state": True,
            "finalize_degradation": year_idx == years - 1,
            "execution_backend": execution_backend,
            "dispatch_instructions": static,
        }

        if observe_jit_per_year:
            # A multi-year run enters the kernel once per year, so the compile
            # is paid in year one and every later year should observe a warm
            # cache; aggregating over the years keeps the run-level claim
            # honest rather than reporting only the first.
            reset_jit_cache_observation(execution_backend)

        if year.aligned is not None:
            if day_controller is not None:
                raise ValueError("a daily controller runs on per-step projection years, not aligned summaries")
            if planner is not None:
                raise ValueError("a year planner runs on per-step projection years, not aligned summaries")
            if record_priced_flows:
                raise ValueError("step flows are recorded on per-step projection years, not aligned summaries")
            for priced in (tariff, reference_tariff):
                if priced is not None:
                    _check_tariff_calendar(priced, year.aligned.index)
            summary = simulate_energy_balance_summary(aligned=year.aligned, weights=weights, **common)
            weighted_w: Mapping[str, float] = summary.weighted_sums
            carry = carry.after_summary(
                summary, has_battery=has_battery, resistance_fade=batt_cfg.enable_resistance_fade
            )
            sums_w: Mapping[str, float] = summary.column_sums
            n_rep, replaced_wh = summary.n_replacements, summary.replaced_capacity_wh
            replacement_steps: Sequence[int] = summary.replacement_steps
            n_steps = summary.n_steps
            annual_fec = summary.fec_all_packs if has_battery and summary.has_degradation_rows else 0.0
            span_events: Sequence[EndOfLifeEvent] = summary.end_of_life_events
            retiring_t_cell_sum = summary.in_service_t_cell_sum
        else:
            if year.pv_dc is None or year.houseload is None:
                raise ValueError("a projection year needs aligned inputs, or pv_dc and houseload")
            # The simulation runs on this range (align_simulation_inputs).
            # Checked first, so a year off the tariff's calendar fails here
            # rather than on the instructions' step count.
            for priced in (tariff, reference_tariff):
                if priced is not None:
                    _check_tariff_calendar(priced, pd.date_range(year.pv_dc.index[0], year.pv_dc.index[-1], freq=freq))
            if planner is not None:
                core_kwargs = {
                    key: value
                    for key, value in common.items()
                    if key not in ("return_degradation_state", "dispatch_instructions")
                }
                planned: list[DispatchInstructions] = []

                def plan_year(
                    state: ControllerBatteryState,
                    planner: YearPlanner = planner,
                    year_idx: int = year_idx,
                    year: ProjectionYear = year,
                    batt_cfg: BatteryConfig = batt_cfg,
                    planned: list[DispatchInstructions] = planned,
                ) -> DispatchInstructions:
                    chosen = planner(YearStart(year_idx, year, batt_cfg, state))
                    planned.append(chosen)
                    return chosen

                detailed = _simulate_detailed_run(
                    pv_dc=year.pv_dc,
                    houseload=year.houseload,
                    temperature_series=year.temperature_series if has_battery else None,
                    instruction_planner=plan_year,
                    **core_kwargs,
                )
                results_df, n_rep, degradation_df = (
                    detailed.results_df,
                    detailed.n_replacements,
                    detailed.degradation_df,
                )
                carry = carry.after_frames(
                    results_df, degradation_df, detailed.degradation_state, has_battery=has_battery
                )
                dispatched.extend(planned)
            elif day_controller is None:
                results_df, _total_pv, _summary_df, n_rep, degradation_df, state = cast(
                    "tuple[pd.DataFrame, float, pd.DataFrame, int, pd.DataFrame, dict[str, Any]]",
                    simulate_energy_balance(
                        pv_dc=year.pv_dc,
                        houseload=year.houseload,
                        temperature_series=year.temperature_series if has_battery else None,
                        **common,
                    ),
                )
                carry = carry.after_frames(results_df, degradation_df, state, has_battery=has_battery)
            else:
                assert tariff is not None
                core_kwargs = {key: value for key, value in common.items() if key != "return_degradation_state"}
                detailed = _simulate_detailed_run(
                    pv_dc=year.pv_dc,
                    houseload=year.houseload,
                    temperature_series=year.temperature_series,
                    day_controller=day_controller,
                    controller_tariff=tariff,
                    controller_carry=carry.controller_carry,
                    projection_year=year_idx,
                    replay_seam=replay_seam,
                    **core_kwargs,
                )
                results_df, n_rep, degradation_df = (
                    detailed.results_df,
                    detailed.n_replacements,
                    detailed.degradation_df,
                )
                carry = replace(
                    carry.after_frames(results_df, degradation_df, detailed.degradation_state, has_battery=has_battery),
                    controller_carry=detailed.controller_carry,
                )
                if detailed.controller_instructions is not None:
                    executed.append(detailed.controller_instructions)
            sums_w = {column: float(results_df[column].sum()) for column in _ROW_SUM_COLUMNS}
            replaced_wh = frame_replaced_capacity_wh(results_df)
            replacement_steps = np.flatnonzero(results_df["Battery_Replaced"].to_numpy()).tolist()
            n_steps = len(results_df)
            span_events = [
                EndOfLifeEvent.from_record(record) for record in degradation_df.attrs.get(END_OF_LIFE_EVENTS_ATTR, ())
            ]
            # Reduced as the summary reduces it, so both paths report one float.
            last_served = retiring_step(span_events)
            retiring_t_cell_sum = (
                float(np.sum(results_df["T_cell"].to_numpy()[: last_served + 1])) if last_served is not None else None
            )
            # Each project year is its own simulation span, so the span's
            # all-pack total is exactly this year's FEC.
            annual_fec = (
                float(degradation_df["Cumulative_FEC_All_Packs"].iloc[-1])
                if has_battery and not degradation_df.empty
                else 0.0
            )
            if first_year_results_df is None:
                first_year_results_df = results_df
            weighted_w = weighted_column_sums(
                {column: results_df[column].to_numpy() for column, _ in (weights or {}).values()}, weights
            )
            if record_priced_flows:
                previous = priced_flows[-1] if priced_flows else {}
                flows: dict[str, np.ndarray] = {}
                for column in _PRICED_FLOW_COLUMNS:
                    # A copy, so the year's frame is not kept alive by a view.
                    values = results_df[column].to_numpy(dtype=float, copy=True)
                    same = previous.get(column)
                    flows[column] = same if same is not None and np.array_equal(same, values) else values
                priced_flows.append(flows)

        if static is not None:
            dispatched.append(static)

        if observe_jit_per_year:
            state_name = observed_jit_cache_state(execution_backend)
            if state_name is not None:
                jit_cache_states.append(state_name)

        total_replacements += n_rep
        end_of_life_events.extend(end_of_life_record(event, year_idx, n_steps) for event in span_events)
        # The battery serves through the step that retires it, and not at all
        # in a year that starts retired.
        retiring = retiring_step(span_events)
        if started_retired:
            in_service_steps, in_service_t_cell_sum = 0, None
        elif retiring is None:
            in_service_steps, in_service_t_cell_sum = n_steps, sums_w["T_cell"]
        else:
            in_service_steps, in_service_t_cell_sum = retiring + 1, retiring_t_cell_sum
        if retiring is not None:
            carry = replace(carry, battery_retired=True)
        rows.append(
            build_year_row(
                year_idx,
                sums_w,
                hours_per_step,
                carry,
                has_battery=has_battery,
                n_replacements=n_rep,
                replaced_capacity_wh=replaced_wh,
                replacement_steps=replacement_steps,
                n_steps=n_steps,
                pv_degradation_factor=year.pv_degradation_factor,
                annual_fec=annual_fec,
                in_service_steps=in_service_steps,
                in_service_t_cell_sum=in_service_t_cell_sum,
                extra=year.extra,
                money=_year_money(
                    tariff, reference_tariff, weighted_w, hours_per_step, n_steps, year.extra.get("Billed_Days")
                ),
            )
        )
        if record_period_energy:
            period_rows.append({name: float(weighted_w[name] * hours_per_step / 1000) for name in period_weights})

    if not rows:
        raise RuntimeError("projection_years must be at least 1")
    return ProjectionRun(
        yearly_df=pd.DataFrame(rows),
        carry=carry,
        total_replacements=total_replacements,
        first_year_results_df=first_year_results_df,
        jit_cache_states=jit_cache_states,
        period_energy=pd.DataFrame(period_rows) if record_period_energy else None,
        controller_instructions=concatenate_instructions(executed),
        year_instructions=tuple(dispatched) if instructions is not None else None,
        priced_flows=tuple(priced_flows) if record_priced_flows else None,
        end_of_life_events=tuple(end_of_life_events),
    )


def run_projection(
    cfg: dict[str, Any],
    resolved: ResolvedAppConfig,
    years: int,
    year_inputs: Callable[[int], ProjectionYear],
    *,
    has_battery: bool,
    execution_backend: str,
    observe_jit_per_year: bool = False,
    tariff: ResolvedTariff | None = None,
    instructions: YearInstructions | None = None,
    record_period_energy: bool = False,
    day_controller: DailyDispatchController | None = None,
    replay_seam: bool = True,
    reference_tariff: ResolvedTariff | None = None,
    record_priced_flows: bool = False,
) -> ProjectionRun:
    """Run :func:`project_years` for an App configuration.

    The battery, the degradation engine and the resolution come from the
    configuration; App and Monte Carlo call this.
    """

    def battery_config(soh_pct: float) -> BatteryConfig:
        if has_battery:
            return build_battery_config(cfg, resolved, initial_soh=soh_pct)
        return build_pv_only_battery_config(cfg, resolved)

    return project_years(
        years,
        year_inputs,
        battery_config=battery_config,
        freq=cfg["resolution"],
        has_battery=has_battery,
        execution_backend=execution_backend,
        degradation_engine=str(cfg.get("degradation_engine", "native")).strip().lower(),
        blast_model=cfg.get("blast_model"),
        observe_jit_per_year=observe_jit_per_year,
        tariff=tariff,
        instructions=instructions,
        record_period_energy=record_period_energy,
        day_controller=day_controller,
        replay_seam=replay_seam,
        reference_tariff=reference_tariff,
        record_priced_flows=record_priced_flows,
    )


@dataclass(frozen=True)
class ProjectionValue:
    """A priced projection: the cost dict, the priced year rows, the cost projection, and LCOE.

    ``total_replacement_cost`` is the replacements at t = 0 prices.
    """

    costs: dict[str, float]
    yearly_df: pd.DataFrame
    cost_projection: pd.DataFrame
    lcoe: float
    total_replacement_cost: float
    terminal_health: TerminalHealthCredit | None = None


def effective_reference_escalation(resolved: ResolvedAppConfig) -> float:
    """The escalation of a run's no-system reference tariff: its own, else the system's import escalation."""
    if resolved.reference_tariff is not None and resolved.reference_tariff.import_price_escalation is not None:
        return float(resolved.reference_tariff.import_price_escalation)
    return projection_rates_record(resolved.cfg)["import_price_escalation"]


def value_projection(cfg: dict[str, Any], resolved: ResolvedAppConfig, run: ProjectionRun) -> ProjectionValue:
    """Price a projection from its year rows; App and Monte Carlo value their runs the same way.

    The rows gain the year-1-price money columns (ADR 0003 E7) before the
    projection escalates and discounts them.
    """
    costs = build_costs_dict(cfg, resolved)
    yearly_df = price_year_rows(run.yearly_df, costs)
    cost_projection = cost_analysis_projection(
        yearly_summary_df=yearly_df,
        costs=costs,
        num_years=len(yearly_df),
        inflation_rate=cfg["inflation_rate"],
        sell_price_inflation=cfg["sell_price_inflation"],
        # Absent from a hand-built cfg: then the escalators inherit inflation.
        import_price_escalation=cfg.get("import_price_escalation"),
        om_escalation=cfg.get("om_escalation"),
        replacement_cost_learning=cfg.get("replacement_cost_learning", 0.0),
        discount_rate=cfg["discount_rate"],
        emissions_params=resolved.emissions_params,
        currency=resolved.currency,
        baseline_import_price_escalation=(
            resolved.reference_tariff.import_price_escalation if resolved.reference_tariff is not None else None
        ),
    )
    terminal = None
    if (cfg.get("terminal_value") or {}).get("basis", "none") == "battery_health_fraction" and resolved.period is None:
        terminal = terminal_health_credit(
            final_soh_fraction=run.carry.soh_pct / 100 if config_has_battery(cfg) else None,
            threshold=cfg["battery_eol_percentage"],
            replacement_cost_each=costs["replacement_cost_each"],
            inflation_rate=cfg["inflation_rate"],
            replacement_cost_learning=cfg.get("replacement_cost_learning", 0.0),
            discount_rate=cfg["discount_rate"],
            horizon_years=len(yearly_df),
            npv_savings=float(cost_projection.attrs["final_npv_savings"]),
            enable_replacement=cfg.get("battery_enable_replacement", True),
            allow_terminal_replacement=cfg.get("battery_allow_terminal_replacement", True),
            replacement_min_remaining_years=cfg.get("battery_replacement_min_remaining_years", 0.0),
            skipped_replacement_action=cfg.get("battery_skipped_replacement_action", "keep"),
        )
    return ProjectionValue(
        costs=costs,
        yearly_df=yearly_df,
        cost_projection=cost_projection,
        lcoe=cost_projection.attrs["lcoe_per_kwh"],
        total_replacement_cost=cost_projection.attrs["total_replacement_cost"],
        terminal_health=terminal,
    )
