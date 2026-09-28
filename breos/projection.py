"""The multi-year projection App and Monte Carlo share.

App (``breos.runners.app``) and Monte Carlo (``breos.montecarlo``) simulate a
configured system year by year. The battery each year runs, the state carried
between years, the year row, and the loop itself are defined here once, so
neither can drift from the other (#179). App runs every year with per-step
frames; Monte Carlo runs summaries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Sequence, cast

import numpy as np
import pandas as pd

from breos.app_config import ResolvedAppConfig, build_costs_dict
from breos.battery import (
    AlignedSimulationInputs,
    BatteryConfig,
    SimulationSummary,
    simulate_energy_balance,
    simulate_energy_balance_summary,
    weighted_column_sums,
)
from breos.dispatch_instructions import DispatchInstructions
from breos.economics import (
    calculate_lcoe_from_projection,
    cost_analysis_projection,
    price_year_rows,
    replacement_fraction_from_steps,
)
from breos.execution import observed_jit_cache_state, reset_jit_cache_observation
from breos.tariffs import ResolvedTariff
from breos.utils import get_hours_per_step


def build_battery_config(cfg: dict[str, Any], resolved: ResolvedAppConfig, *, initial_soh: float) -> BatteryConfig:
    """Build the battery one projection year runs, starting at ``initial_soh``.

    A configured round-trip efficiency is split evenly across charge and
    discharge, the BatteryConfig default convention. Replacement is on, and
    a replacement is priced at the configured storage cost per kWh.
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
        dc_coupled=cfg["dc_coupled"],
        inverter_efficiency=cfg["inverter_efficiency"],
        inverter_ac_capacity_w=resolved.inverter_ac_capacity_w,
        enable_replacement=True,
        replacement_cost=resolved.cost_params.battery_cost_per_kwh * battery_kwh,
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
    battery's own initial state of charge.
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
    replacement_cost: float,
    replacement_steps: Sequence[int],
    n_steps: int,
    pv_degradation_factor: float,
    annual_fec: float = 0.0,
    extra: Mapping[str, Any] | None = None,
    money: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Build the one year-row schema App, Monte Carlo and the optimizer report.

    ``sums_w`` holds each results column's sum over the year's steps; a
    summary's ``column_sums`` and a frame's column sums are the same floats.
    ``carry`` is the state at the end of the year and ``annual_fec`` the
    rainflow cycles every pack accumulated in it. ``extra`` columns (Monte
    Carlo's sampled weather year and load scale) follow
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
        "Legacy_PV_Production_kWh": kwh("PV_Production"),
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
        "Replacement_Cost": replacement_cost,
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
    # Cumulative FEC belongs to the installed pack and restarts at zero on
    # replacement, so the year's own count comes from the all-pack total.
    row["Battery_Annual_FEC"] = float(annual_fec)
    # Year-1-price money from a tariff (ADR 0003 E7); without one, economics
    # prices the energy at the flat rates.
    row.update(money or {})
    return row


def _tariff_weights(tariff: ResolvedTariff) -> dict[str, tuple[str, np.ndarray]]:
    import_prices = np.asarray(tariff.import_price_per_kwh, dtype=float)
    return {
        "Import_Cost": ("Import_From_Grid", import_prices),
        "Export_Revenue": ("PV_AC_Export", np.asarray(tariff.export_price_per_kwh, dtype=float)),
        # The no-system household buys its whole load at the same prices.
        "Baseline_Import_Cost": ("Houseload", import_prices),
        # The part of Import_Cost bought to charge the battery.
        "Grid_Charge_Cost": ("Grid_AC_To_Battery", import_prices),
    }


def _tariff_money(
    tariff: ResolvedTariff, weighted_w: Mapping[str, float], hours_per_step: float, n_steps: int
) -> dict[str, float]:
    """A year's money at year-1 prices from its price-weighted power sums."""
    money = {name: float(total * hours_per_step / 1000) for name, total in weighted_w.items()}
    # Billed on the simulated duration, as the flat path is (ADR 0003 E5).
    money["Fixed_Charge"] = tariff.prices.fixed_charge_per_day * (n_steps * hours_per_step / 24)
    return money


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
class ProjectionRun:
    """What a multi-year projection produced."""

    yearly_df: pd.DataFrame
    carry: CarryState
    total_replacements: int
    total_replacement_cost: float
    # The first year's per-step frame; None for a summary projection.
    first_year_results_df: pd.DataFrame | None
    jit_cache_states: list[str]


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
    instructions: DispatchInstructions | None = None,
) -> ProjectionRun:
    """Simulate ``years`` project years, carrying the battery from one to the next.

    App, Monte Carlo and the projected optimizer all run this loop.
    ``year_inputs(year_idx)`` supplies each year, and ``battery_config(soh)``
    builds the battery at the carried SOH. The rainflow residue is counted
    once, at the end of the last year. ``observe_jit_per_year`` records the
    Numba cache state of every year, as App reports it.

    With a ``tariff``, resolved on the simulation calendar, each year row
    carries its import cost, export revenue, no-system import cost and fixed
    charge at year-1 prices, from the step energy times the step price. Every
    year replays the one calendar (ADR 0002 A2), and so do smart-charging
    ``instructions``, resolved on that calendar.
    """
    hours_per_step = get_hours_per_step(freq)
    weights = _tariff_weights(tariff) if tariff is not None else None
    carry = initial_carry or CarryState()
    rows: list[dict[str, Any]] = []
    total_replacements = 0
    total_replacement_cost = 0.0
    first_year_results_df: pd.DataFrame | None = None
    jit_cache_states: list[str] = []

    for year_idx in range(years):
        year = year_inputs(year_idx)
        batt_cfg = battery_config(carry.soh_pct)
        common = {
            **carry.simulation_kwargs(),
            "battery_config": batt_cfg,
            "freq": freq,
            "degradation_engine": degradation_engine,
            "blast_model": blast_model,
            "return_degradation_state": True,
            "finalize_degradation": year_idx == years - 1,
            "execution_backend": execution_backend,
            "dispatch_instructions": instructions,
        }

        if observe_jit_per_year:
            # A multi-year run enters the kernel once per year, so the compile
            # is paid in year one and every later year should observe a warm
            # cache; aggregating over the years keeps the run-level claim
            # honest rather than reporting only the first.
            reset_jit_cache_observation(execution_backend)

        if year.aligned is not None:
            if tariff is not None:
                _check_tariff_calendar(tariff, year.aligned.index)
            summary = simulate_energy_balance_summary(aligned=year.aligned, weights=weights, **common)
            weighted_w: Mapping[str, float] = summary.weighted_sums
            carry = carry.after_summary(
                summary, has_battery=has_battery, resistance_fade=batt_cfg.enable_resistance_fade
            )
            sums_w: Mapping[str, float] = summary.column_sums
            n_rep, rep_cost = summary.n_replacements, summary.total_replacement_cost
            replacement_steps: Sequence[int] = summary.replacement_steps
            n_steps = summary.n_steps
            annual_fec = summary.fec_all_packs if has_battery and summary.has_degradation_rows else 0.0
        else:
            if year.pv_dc is None or year.houseload is None:
                raise ValueError("a projection year needs aligned inputs, or pv_dc and houseload")
            if tariff is not None:
                # The simulation runs on this range (align_simulation_inputs).
                # Checked first, so a year off the tariff's calendar fails
                # here rather than on the instructions' step count.
                _check_tariff_calendar(tariff, pd.date_range(year.pv_dc.index[0], year.pv_dc.index[-1], freq=freq))
            results_df, _total_pv, _summary_df, rep_cost, n_rep, degradation_df, state = cast(
                "tuple[pd.DataFrame, float, pd.DataFrame, float, int, pd.DataFrame, dict[str, Any]]",
                simulate_energy_balance(
                    pv_dc=year.pv_dc,
                    houseload=year.houseload,
                    temperature_series=year.temperature_series if has_battery else None,
                    **common,
                ),
            )
            carry = carry.after_frames(results_df, degradation_df, state, has_battery=has_battery)
            sums_w = {column: float(results_df[column].sum()) for column in _ROW_SUM_COLUMNS}
            replacement_steps = (
                np.flatnonzero(results_df["Battery_Replaced"].to_numpy()).tolist()
                if "Battery_Replaced" in results_df.columns
                else []
            )
            n_steps = len(results_df)
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

        if observe_jit_per_year:
            state_name = observed_jit_cache_state(execution_backend)
            if state_name is not None:
                jit_cache_states.append(state_name)

        total_replacements += n_rep
        total_replacement_cost += rep_cost
        rows.append(
            build_year_row(
                year_idx,
                sums_w,
                hours_per_step,
                carry,
                has_battery=has_battery,
                n_replacements=n_rep,
                replacement_cost=rep_cost,
                replacement_steps=replacement_steps,
                n_steps=n_steps,
                pv_degradation_factor=year.pv_degradation_factor,
                annual_fec=annual_fec,
                extra=year.extra,
                money=_tariff_money(tariff, weighted_w, hours_per_step, n_steps) if tariff is not None else None,
            )
        )

    if not rows:
        raise RuntimeError("projection_years must be at least 1")
    return ProjectionRun(
        yearly_df=pd.DataFrame(rows),
        carry=carry,
        total_replacements=total_replacements,
        total_replacement_cost=total_replacement_cost,
        first_year_results_df=first_year_results_df,
        jit_cache_states=jit_cache_states,
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
    instructions: DispatchInstructions | None = None,
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
    )


@dataclass(frozen=True)
class ProjectionValue:
    """A priced projection: the cost dict, the priced year rows, the cost projection, and LCOE."""

    costs: dict[str, float]
    yearly_df: pd.DataFrame
    cost_projection: pd.DataFrame
    lcoe: float


def value_projection(cfg: dict[str, Any], resolved: ResolvedAppConfig, run: ProjectionRun) -> ProjectionValue:
    """Price a projection from its year rows; App and Monte Carlo value their runs the same way.

    The rows gain the year-1-price money columns (ADR 0003 E7) before the
    projection escalates and discounts them.
    """
    costs = build_costs_dict(cfg, resolved)
    yearly_df = price_year_rows(run.yearly_df, costs)
    cost_projection = cost_analysis_projection(
        results_df=None,
        costs=costs,
        num_years=len(yearly_df),
        inflation_rate=cfg["inflation_rate"],
        sell_price_inflation=cfg["sell_price_inflation"],
        discount_rate=cfg["discount_rate"],
        freq=cfg["resolution"],
        yearly_summary_df=yearly_df,
        total_replacement_cost=run.total_replacement_cost,
        emissions_params=resolved.emissions_params,
    )
    lcoe = calculate_lcoe_from_projection(
        cost_projection,
        total_investment=costs["total_initial_cost"],
        discount_rate=cfg["discount_rate"],
    )
    return ProjectionValue(costs=costs, yearly_df=yearly_df, cost_projection=cost_projection, lcoe=lcoe)
