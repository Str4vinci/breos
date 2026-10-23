"""App's three physical households, retained independently of their valuation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, cast

import numpy as np
import pandas as pd

from breos.app_config import ResolvedAppConfig, build_costs_dict
from breos.battery import BatteryConfig
from breos.economics import add_co2_projection, price_year_rows, projection_rates_record
from breos.projection import (
    BASELINE_NETWORK_CREDIT_COLUMNS,
    SYSTEM_NETWORK_CREDIT_COLUMNS,
    ProjectionRun,
    ProjectionValue,
    ProjectionYear,
    effective_reference_escalation,
    price_tariff_year_rows_by_step,
    project_years,
    value_projection,
)
from breos.tariffs import ResolvedTariff
from breos.vehicle import AwayChargingConfig, VehicleConfig, VehicleSchedule
from breos.vehicle.economics import vehicle_pack_value, vehicle_wear
from breos.vehicle.results import VEHICLE_ENERGY_FIELDS, vehicle_year_record


def household_configs(study: VehicleConfig) -> dict[str, VehicleConfig]:
    """Only controls and charger premium differ; trips, pack and policy agree."""

    def with_mode(mode: str) -> VehicleConfig:
        return replace(
            study,
            mode=mode,
            reference_mode="none",
            charger_cost=0.0,
            v2h_min_soc=study.v2h_min_soc if mode == "v2h" else None,
            charge_periods=study.charge_periods if mode != "uncontrolled" else (),
            discharge_periods=study.discharge_periods if mode == "v2h" else (),
        )

    configs = {"R0": with_mode("uncontrolled"), "R2": study}
    if study.reference_mode != "none":
        configs["R1"] = with_mode(str(study.reference_mode))
    return configs


@dataclass(frozen=True)
class VehicleHouseholds:
    runs: dict[str, ProjectionRun]
    schedule: VehicleSchedule

    def provenance(self, config: VehicleConfig) -> dict[str, Any]:
        return vehicle_provenance(config, self.schedule)


def vehicle_provenance(config: VehicleConfig, schedule: VehicleSchedule) -> dict[str, Any]:
    """The physical household definitions shared by App and Monte Carlo."""
    configs = household_configs(config)
    return {
        "config": asdict(config),
        "schedule_hash": schedule.schedule_hash(),
        "rounding": {"departure": "floor", "arrival": "ceil", "timezone": "configured civil time"},
        "households": {
            key: {"system": "none" if key == "R0" else "configured", "mode": configs[key].mode} for key in configs
        },
    }


def run_vehicle_baseline(
    years: int,
    year_inputs: Callable[[int], ProjectionYear],
    config: VehicleConfig,
    *,
    freq: str,
    execution_backend: str,
    grid_import_limit_w: float,
) -> ProjectionRun:
    """R0 contains no PV equipment, inverter draw or stationary storage."""

    def baseline_inputs(year_idx: int) -> ProjectionYear:
        year = year_inputs(year_idx)
        assert year.pv_dc is not None
        return replace(
            year,
            pv_dc=year.pv_dc * 0.0,
            inverter_units=None,
            inverter_curve_row=None,
            inverter_ac_derate=None,
            pv_degradation_factor=1.0,
        )

    return project_years(
        years,
        baseline_inputs,
        battery_config=lambda _: BatteryConfig(nominal_energy_wh=0.0),
        freq=freq,
        has_battery=False,
        execution_backend=execution_backend,
        vehicle_config=config,
        grid_import_limit_w=grid_import_limit_w,
        record_priced_flows=True,
    )


_MONEY_COLUMNS = (
    "Import_Cost",
    "Export_Revenue",
    "Baseline_Import_Cost",
    "Grid_Charge_Cost",
    "Fixed_Charge",
    "Baseline_Fixed_Charge",
    *SYSTEM_NETWORK_CREDIT_COLUMNS,
    *BASELINE_NETWORK_CREDIT_COLUMNS,
)


def _price_run(run: ProjectionRun, tariff: ResolvedTariff | None, costs: dict[str, float], freq: str) -> pd.DataFrame:
    rows = run.yearly_df.drop(columns=[column for column in _MONEY_COLUMNS if column in run.yearly_df])
    if tariff is not None:
        assert run.priced_flows is not None
        rows = price_tariff_year_rows_by_step(rows, run.priced_flows, tariff, freq)
    return price_year_rows(rows, costs)


def value_vehicle_households(
    cfg: dict[str, Any],
    resolved: ResolvedAppConfig,
    households: VehicleHouseholds,
    tariff: ResolvedTariff | None,
    reference: ResolvedTariff | None,
) -> tuple[ProjectionValue, dict[str, Any]]:
    """Price each year's actual purchases, then value V10's two comparisons."""
    config = resolved.vehicle
    assert config is not None
    configs = household_configs(config)
    costs = build_costs_dict(cfg, resolved)
    rows = {
        key: _price_run(run, (reference or tariff) if key == "R0" else tariff, costs, cfg["resolution"])
        for key, run in households.runs.items()
    }
    baseline = rows["R0"]
    for key, row in rows.items():
        row["Baseline_Import_Cost"] = baseline["Import_Cost"]
        row["Baseline_Fixed_Charge"] = baseline["Fixed_Charge"]
        row["Baseline_Import_kWh"] = baseline["Import_kWh"]
        row["Baseline_Away_kWh"] = baseline["Vehicle_Away_Charge_Purchased_kWh"]
        for system_column, baseline_column in zip(
            SYSTEM_NETWORK_CREDIT_COLUMNS, BASELINE_NETWORK_CREDIT_COLUMNS, strict=True
        ):
            if system_column in baseline:
                row[baseline_column] = baseline[system_column]
            elif baseline_column in row:
                row.drop(columns=baseline_column, inplace=True)
        away_price = 0.0 if config.away_charging is None else cast(AwayChargingConfig, config.away_charging).price
        row["Vehicle_Away_Charge_Cost"] = row["Vehicle_Away_Charge_Purchased_kWh"] * away_price
        row["Vehicle_Drive_Unserved_Cost"] = row["Vehicle_Drive_Unserved_kWh"] * (config.outside_charging_price or 0.0)
    rates = projection_rates_record(cfg)
    discount = float(cfg["discount_rate"])
    horizon = len(baseline)
    valued = {}
    for key, run in households.runs.items():
        row = rows[key]
        carry = run.carry.vehicle
        assert carry is not None
        pack = vehicle_pack_value(
            configs[key],
            run.vehicle_end_of_life_events,
            None if carry.retired else carry.soh_fraction,
            horizon=horizon,
            inflation=cfg["inflation_rate"],
            learning=cfg["replacement_cost_learning"],
            discount=discount,
        )
        years = row["Year"].to_numpy(dtype=float)
        home_rate = effective_reference_escalation(resolved) if key == "R0" else rates["import_price_escalation"]

        def annual_npv(amounts: Any, escalation: float) -> float:
            return float(np.sum(np.asarray(amounts) * (1 + escalation) ** (years - 1) / (1 + discount) ** years))

        home = annual_npv(row["Import_Cost"] + row["Fixed_Charge"] - row.get("Network_Credit", 0.0), home_rate)
        export = annual_npv(row["Export_Revenue"], cfg["sell_price_inflation"])
        away = annual_npv(row["Vehicle_Away_Charge_Cost"], rates["import_price_escalation"])
        unserved = annual_npv(row["Vehicle_Drive_Unserved_Cost"], rates["import_price_escalation"])
        co2 = None
        if resolved.emissions_params is not None:
            co2 = row[["PV_Production_kWh", "Export_kWh"]].copy()
            add_co2_projection(co2, row, resolved.emissions_params)
        year_records = [vehicle_year_record(year) for year in row.to_dict("records")]
        for record, (_, year) in zip(year_records, row.iterrows(), strict=True):
            record.update(
                grid_import_kwh=float(year["Import_kWh"]),
                grid_export_kwh=float(year["Export_kWh"]),
                home_import_cost_year1_prices=float(year["Import_Cost"]),
                export_revenue_year1_prices=float(year["Export_Revenue"]),
                fixed_charge_year1_prices=float(year["Fixed_Charge"]),
                network_credit_year1_prices=float(year.get("Network_Credit", 0.0)),
            )
            if co2 is not None:
                record["avoided_co2_kg"] = float(co2.iloc[record["year"] - 1]["CO2_Avoided_Total_kg"])
            events = [event for event in pack["replacement_history"] if event["year"] == record["year"]]
            record["replacement_outlays_npv"] = sum(event["outlay_npv"] for event in events)
            record["replacement_outlays_t0_prices"] = sum(event["outlay_t0_prices"] for event in events)
        totals = {name: sum(record[name] for record in year_records) for name in VEHICLE_ENERGY_FIELDS}
        totals.update(
            departures_short=sum(record["departures_short"] for record in year_records),
            replacements=carry.n_replacements,
            replaced_capacity_kwh=carry.replaced_capacity_wh / 1000.0,
        )
        if co2 is not None:
            totals["avoided_co2_kg"] = float(co2["CO2_Avoided_Total_kg"].sum())
        valued[key] = {
            "mode": configs[key].mode,
            "yearly": year_records,
            "totals": totals,
            "final_soh_fraction": carry.soh_fraction,
            "retired": carry.retired,
            **pack,
            "home_bill_npv": home,
            "export_revenue_npv": export,
            "away_purchases_npv": away,
            "unserved_driving_cost_npv": unserved,
            "operating_cost_npv": home - export + away + unserved,
        }
    value = value_projection(cfg, resolved, replace(households.runs["R2"], yearly_df=rows["R2"]))
    system_npv = float(value.cost_projection.attrs["final_npv_savings"])

    def comparison(reference_key: str) -> dict[str, Any]:
        study, ref = valued["R2"], valued[reference_key]
        comparable = (
            config.outside_charging_price is not None
            or study["totals"]["drive_unserved_kwh"] <= ref["totals"]["drive_unserved_kwh"] + 1e-9
        )
        return {
            **vehicle_wear(study, ref),
            "comparable": comparable,
            "reason": None if comparable else "R2 leaves more driving unserved; outside_charging_price is unset",
        }

    adjusted = comparison("R0")
    away_difference = valued["R2"]["away_purchases_npv"] - valued["R0"]["away_purchases_npv"]
    unserved_difference = valued["R2"]["unserved_driving_cost_npv"] - valued["R0"]["unserved_driving_cost_npv"]
    adjusted.update(
        system_npv=system_npv,
        away_cost_difference_npv=away_difference,
        unserved_cost_difference_npv=unserved_difference,
        npv=system_npv - away_difference - unserved_difference - adjusted["wear_npv"],
    )
    benefit = None
    if "R1" in valued:
        benefit = comparison("R1")
        study, ref = valued["R2"], valued["R1"]
        saving = ref["operating_cost_npv"] - study["operating_cost_npv"]
        premium = float(config.charger_cost or 0.0)
        benefit.update(
            operating_savings_npv=saving,
            charger_premium=premium,
            home_bill_savings_npv=ref["home_bill_npv"] - study["home_bill_npv"],
            export_revenue_gain_npv=study["export_revenue_npv"] - ref["export_revenue_npv"],
            away_savings_npv=ref["away_purchases_npv"] - study["away_purchases_npv"],
            unserved_savings_npv=ref["unserved_driving_cost_npv"] - study["unserved_driving_cost_npv"],
            npv=saving - premium - benefit["wear_npv"],
        )
    if resolved.period is not None:
        # A window is not a project lifetime, just as for system economics.
        benefit = None
        for household in valued.values():
            for event in household["replacement_history"]:
                event["outlay_npv"] = event["outlay_nominal"] = None
            for record in household["yearly"]:
                record["replacement_outlays_npv"] = None
            for name in tuple(household):
                if name.endswith("_npv"):
                    household[name] = None
    return value, {
        "mode": config.mode,
        "reference_mode": config.reference_mode,
        "currency": resolved.currency,
        "households": valued,
        "household_adjusted_npv": adjusted if resolved.period is None else None,
        "v2h_benefit": benefit,
    }
