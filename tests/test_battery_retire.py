"""Retiring a battery whose end-of-life replacement is skipped (#404, ADR 0003 E12).

``battery_skipped_replacement_action = "retire"`` (``BatteryConfig.
skipped_replacement_action``, the optimizer's ``[battery]
skipped_replacement_action``) switches the battery off at a crossing whose
swap ``battery_replacement_min_remaining_years`` or
``battery_allow_terminal_replacement`` skips. From that instant it neither
charges nor discharges, whatever the dispatch instructions say, and the
project finishes as a PV-only system. Every earlier step and cash flow is
unchanged, and no replacement is bought. The default ``"keep"`` changes
nothing.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import breos.optimization as optimization_module
from breos.app_config import resolve_app_config
from breos.battery import BatteryConfig, align_simulation_inputs, simulate_energy_balance
from breos.dispatch_instructions import DispatchInstructions
from breos.montecarlo import MonteCarloSettings, run_montecarlo
from breos.optimization_config import resolve_optimization_config
from breos.projection import ProjectionYear, project_years
from tests.test_replacement_min_remaining_years import YEAR_STEPS
from tests.test_terminal_replacement import (
    EVERY_PERIOD_EOL,
    _app,
    _assert_same_run,
    _inputs,
    _mc_config,
    _projected_metrics,
    _replaced_steps,
    _require_numba,
    _run,
)

# Flows a retired step shares with the PV-only system, by results column.
PV_ONLY_COLUMNS = (
    "PV_DC",
    "PV_Production",
    "Houseload",
    "PV_Delta",
    "Import_From_Grid",
    "PV_DC_To_Inverter",
    "PV_DC_Curtailed",
    "PV_AC_To_Load",
    "PV_AC_Export",
    "PV_Direct_Inverter_Loss",
    "Inverter_Loss",
    "T_cell",
)
# Battery flows, all zero once the battery is retired.
BATTERY_FLOW_COLUMNS = (
    "Battery_Charge_Input",
    "Battery_Charge_Stored",
    "Battery_Discharge_DC",
    "Battery_AC_To_Load",
    "Grid_AC_To_Battery",
    "Battery_Inverter_Loss",
    "Standby_Loss",
    "Battery_Energy",
    "Battery_Energy_Beginning",
)
# Year-row flows a retired year shares with the PV-only system.
PV_ONLY_YEAR_COLUMNS = (
    "PV_Production_kWh",
    "PV_DC_Generation_kWh",
    "Direct_PV_AC_Load_kWh",
    "Self_Consumption_kWh",
    "Curtailment_DC_kWh",
    "Load_kWh",
    "Import_kWh",
    "Export_kWh",
    "Inverter_Loss_kWh",
)
# A flat price on a two-period schedule: grid charging saves nothing and
# loses its conversion and round-trip losses.
FLAT_TOU = {
    "schedule": "pt_mainland_2026_daily_bi",
    "currency": "EUR",
    "import_prices": {"peak": 0.20, "off_peak": 0.20},
    "export_prices": {"all": 0.05},
}
FILL_OFF_PEAK = {
    "mode": "fixed_target",
    "target_usable_fraction": 1.0,
    "charge_periods": ["off_peak"],
    "discharge_periods": ["peak"],
    "grid_charge_efficiency": 0.9,
}


def _pv_only(n_steps: int, freq: str = "h"):
    pv, load, temperature = _inputs(n_steps, freq)
    return simulate_energy_balance(
        pv_dc=pv,
        houseload=load,
        battery_config=BatteryConfig(nominal_energy_wh=0.0),
        freq=freq,
        temperature_series=temperature,
    )[0]


def _grid_charging(n_steps: int) -> DispatchInstructions:
    """Grid-charge to full before 06:00; discharge after it."""
    hour = _inputs(n_steps)[0].index.hour.to_numpy()
    return DispatchInstructions(
        discharge_allowed=hour >= 6,
        reserve_fraction=np.zeros(n_steps),
        grid_target_fraction=np.where(hour < 6, 1.0, np.nan),
        grid_charge_efficiency=0.9,
        grid_import_limit_w=math.inf,
    )


# -- the settings ------------------------------------------------------------------


class TestSettings:
    BASE = {"location": "porto", "n_modules": 8, "annual_consumption_kwh": 4000, "battery_kwh": 5}
    LOCATION = {"location": {"latitude": 41.15, "longitude": -8.61}}

    def test_battery_config_keeps_by_default_and_refuses_other_actions(self):
        assert BatteryConfig(nominal_energy_wh=5000.0).skipped_replacement_action == "keep"
        assert BatteryConfig(nominal_energy_wh=5000.0, skipped_replacement_action="retire")
        for value in ("replace", "Retire", None, True):
            with pytest.raises(ValueError, match="skipped_replacement_action must be one of: keep, retire"):
                BatteryConfig(nominal_energy_wh=5000.0, skipped_replacement_action=value)

    def test_the_app_key_resolves_keep_and_reaches_the_battery(self):
        assert resolve_app_config(self.BASE).cfg["battery_skipped_replacement_action"] == "keep"
        resolved = resolve_app_config({**self.BASE, "battery_skipped_replacement_action": "retire"})
        from breos.projection import build_battery_config

        assert build_battery_config(resolved.cfg, resolved, initial_soh=100.0).skipped_replacement_action == "retire"
        with pytest.raises(ValueError, match="'battery_skipped_replacement_action' must be one of: keep, retire"):
            resolve_app_config({**self.BASE, "battery_skipped_replacement_action": "replace"})

    def test_the_optimizer_key_resolves_keep_and_is_forwarded(self):
        assert resolve_optimization_config(self.LOCATION)["battery"]["skipped_replacement_action"] == "keep"
        config = {**self.LOCATION, "battery": {"skipped_replacement_action": "retire"}}
        assert resolve_optimization_config(config)["battery"]["skipped_replacement_action"] == "retire"
        battery = optimization_module._build_battery_config_from_spec(
            {"skipped_replacement_action": "retire"}, nominal_energy_wh=5000.0
        )
        assert battery.skipped_replacement_action == "retire"
        with pytest.raises(ValueError, match="battery.skipped_replacement_action"):
            resolve_optimization_config({**self.LOCATION, "battery": {"skipped_replacement_action": "sell"}})


# -- one span ---------------------------------------------------------------------


# One swap at step 23, then the close of step 47 is skipped.
SKIP_SECOND = {"eol_percentage": EVERY_PERIOD_EOL, "replacement_min_remaining_years": 0.6}


def test_retire_changes_nothing_before_the_crossing_and_dispatches_pv_only_after():
    keep = _run(96, **SKIP_SECOND)
    retire = _run(96, **SKIP_SECOND, skipped_replacement_action="retire")
    pv_only = _pv_only(96)

    assert _replaced_steps(keep) == _replaced_steps(retire) == [23]
    assert [(e.step, e.action, e.reason) for e in retire[4].attrs["end_of_life_events"]] == [
        (23, "replaced", "end_of_life"),
        (47, "retired", "min_remaining_years"),
    ]
    # Every step before the crossing's is the same, and the crossing step was
    # dispatched the same; only its closing state moves.
    pd.testing.assert_frame_equal(keep[0].iloc[:47], retire[0].iloc[:47], check_exact=True)
    for column in PV_ONLY_COLUMNS + ("Battery_Charge_Input", "Battery_Discharge_DC", "Battery_Energy_Beginning"):
        assert keep[0][column].iloc[47] == retire[0][column].iloc[47]
    # From the next step on, the PV-only system's flows, bit for bit.
    for column in PV_ONLY_COLUMNS:
        np.testing.assert_array_equal(retire[0][column].iloc[48:], pv_only[column].iloc[48:], err_msg=column)
    for column in BATTERY_FLOW_COLUMNS:
        assert (retire[0][column].iloc[48:] == 0.0).all(), column
    # Keep went on dispatching.
    assert keep[0]["Battery_Discharge_DC"].iloc[48:].sum() > 0.0


def test_the_stored_energy_leaves_with_the_retired_pack():
    keep = _run(96, **SKIP_SECOND)
    retire = _run(96, **SKIP_SECOND, skipped_replacement_action="retire")
    step = retire[0].iloc[47]

    held = keep[0]["Battery_Energy"].iloc[47]
    assert held > 0.0
    assert step["Battery_Replacement_Energy_Removed"] == held
    assert step["Battery_Replacement_Energy_Added"] == 0.0
    assert not step["Battery_Replaced"] and step["Battery_Replaced_Capacity_Wh"] == 0.0
    assert step["Battery_Energy"] == step["Battery_Energy_End"] == 0.0
    assert step["Battery_Energy_Delta"] == -step["Battery_Energy_Beginning"]
    assert retire[3] == keep[3] == 1
    assert retire[5]["native_rainflow_state"]["residue"] == []


def test_a_retired_battery_ignores_grid_charging_instructions():
    instructions = _grid_charging(96)
    pv, load, temperature = _inputs(96)

    def run(action: str):
        return simulate_energy_balance(
            pv_dc=pv,
            houseload=load,
            battery_config=BatteryConfig(nominal_energy_wh=5000.0, skipped_replacement_action=action, **SKIP_SECOND),
            freq="h",
            temperature_series=temperature,
            dispatch_instructions=instructions,
        )[0]

    keep, retire = run("keep"), run("retire")
    assert keep["Grid_AC_To_Battery"].iloc[48:].sum() > 0.0
    assert (retire["Grid_AC_To_Battery"].iloc[48:] == 0.0).all()
    pd.testing.assert_frame_equal(keep.iloc[:47], retire.iloc[:47], check_exact=True)
    for column in PV_ONLY_COLUMNS:
        np.testing.assert_array_equal(retire[column].iloc[48:], _pv_only(96)[column].iloc[48:], err_msg=column)


def test_the_terminal_guard_retires_on_the_last_day():
    retire = _run(
        96, eol_percentage=EVERY_PERIOD_EOL, allow_terminal_replacement=False, skipped_replacement_action="retire"
    )
    events = retire[4].attrs["end_of_life_events"]
    assert (events[-1].step, events[-1].action, events[-1].reason) == (95, "retired", "terminal_period")
    assert retire[0]["Battery_Energy"].iloc[-1] == 0.0


def test_keep_is_the_default():
    _assert_same_run(_run(96, **SKIP_SECOND), _run(96, **SKIP_SECOND, skipped_replacement_action="keep"))
    # Without a skipped swap the action changes nothing.
    _assert_same_run(
        _run(96, eol_percentage=EVERY_PERIOD_EOL),
        _run(96, eol_percentage=EVERY_PERIOD_EOL, skipped_replacement_action="retire"),
    )


def test_a_span_that_inherits_a_retired_battery_is_pv_only():
    pv, load, temperature = _inputs(48)
    run = simulate_energy_balance(
        pv_dc=pv,
        houseload=load,
        battery_config=BatteryConfig(nominal_energy_wh=5000.0, initial_soh=60.0),
        freq="h",
        temperature_series=temperature,
        initial_energy_wh=0.0,
        battery_retired=True,
        dispatch_instructions=_grid_charging(48),
    )
    for column in PV_ONLY_COLUMNS:
        np.testing.assert_array_equal(run[0][column], _pv_only(48)[column], err_msg=column)
    assert run[3] == 0 and run[4].attrs["end_of_life_events"] == ()
    with pytest.raises(ValueError, match="a retired battery holds no energy"):
        simulate_energy_balance(
            pv_dc=pv,
            houseload=load,
            battery_config=BatteryConfig(nominal_energy_wh=5000.0),
            freq="h",
            temperature_series=temperature,
            initial_energy_wh=100.0,
            battery_retired=True,
        )


@pytest.mark.parametrize("freq", ["h", "15min"])
def test_python_and_numba_retire_the_same_way(freq):
    _require_numba()
    n_steps = 4 * (24 if freq == "h" else 96)
    config = {**SKIP_SECOND, "skipped_replacement_action": "retire"}
    _assert_same_run(_run(n_steps, freq, **config), _run(n_steps, freq, backend="numba", **config))


# -- the projection ------------------------------------------------------------------


def _project(action: str, *, aligned: bool = False, backend: str = "python", instructions=None, battery_kwh=5.0):
    pv, load, temperature = _inputs(YEAR_STEPS)
    year_aligned = align_simulation_inputs(pv, load, temperature, freq="h") if aligned else None

    def year_inputs(_year_idx):
        if year_aligned is not None:
            return ProjectionYear(1.0, aligned=year_aligned)
        return ProjectionYear(1.0, pv_dc=pv, houseload=load, temperature_series=temperature)

    def battery(soh_pct):
        return BatteryConfig(
            nominal_energy_wh=battery_kwh * 1000.0,
            initial_soh=soh_pct,
            eol_percentage=EVERY_PERIOD_EOL,
            replacement_min_remaining_years=1.5,
            skipped_replacement_action=action,
        )

    return project_years(
        3,
        year_inputs,
        battery_config=battery,
        freq="h",
        has_battery=battery_kwh > 0,
        execution_backend=backend,
        instructions=instructions,
    )


def test_a_retired_battery_stays_off_in_the_later_years():
    keep, retire, pv_only = _project("keep"), _project("retire"), _project("keep", battery_kwh=0.0)

    # Year 1 swaps three times; year 2 swaps at its first close and retires at its second.
    assert retire.yearly_df["Replacements"].tolist() == keep.yearly_df["Replacements"].tolist() == [3, 1, 0]
    assert [(e["year"], e["action"]) for e in retire.end_of_life_events][-2:] == [(2, "replaced"), (2, "retired")]
    assert retire.carry.battery_retired and not keep.carry.battery_retired
    pd.testing.assert_frame_equal(keep.yearly_df.iloc[:1], retire.yearly_df.iloc[:1], check_exact=True)
    last = retire.yearly_df.iloc[2]
    for column in PV_ONLY_YEAR_COLUMNS:
        assert last[column] == pv_only.yearly_df[column].iloc[2], column
    assert last["Battery_Charge_Throughput_kWh"] == last["Battery_Discharge_Throughput_kWh"] == 0.0
    assert last["Battery_Carried_Energy_Wh"] == 0.0
    assert keep.yearly_df["Battery_Discharge_Throughput_kWh"].iloc[2] > 0.0


def test_retired_projection_years_ignore_grid_charging():
    instructions = _grid_charging(YEAR_STEPS)
    keep, retire = _project("keep", instructions=instructions), _project("retire", instructions=instructions)
    assert keep.yearly_df["Grid_AC_To_Battery_kWh"].iloc[2] > 0.0
    assert retire.yearly_df["Grid_AC_To_Battery_kWh"].iloc[2] == 0.0


def test_frames_summaries_and_numba_agree_on_a_retired_projection():
    frames = _project("retire")
    summary = _project("retire", aligned=True)
    pd.testing.assert_frame_equal(frames.yearly_df, summary.yearly_df, check_exact=True)
    assert frames.end_of_life_events == summary.end_of_life_events
    assert summary.carry.battery_retired
    _require_numba()
    numba = _project("retire", backend="numba")
    pd.testing.assert_frame_equal(frames.yearly_df, numba.yearly_df, check_exact=True)


# -- App, Monte Carlo and the optimizer ------------------------------------------------


def _three(**overrides):
    """Replace, keep and retire on one two-year App case whose year-2 swaps the minimum skips."""
    replace = _app(**overrides)
    keep = _app(battery_replacement_min_remaining_years=1, **overrides)
    retire = _app(battery_replacement_min_remaining_years=1, battery_skipped_replacement_action="retire", **overrides)
    return replace, keep, retire


def test_replace_keep_and_retire_share_every_flow_before_the_crossing():
    replace, keep, retire = _three(terminal_value={"basis": "battery_health_fraction"})
    rows = [app._artifacts.yearly_df for app in (replace, keep, retire)]
    costs = [app._artifacts.cost_projection for app in (replace, keep, retire)]
    results = [app.result() for app in (replace, keep, retire)]

    # The investment and year 1 are the same in all three.
    assert len({result["financial"][0]["balance"] for result in results}) == 1
    for other in (1, 2):
        pd.testing.assert_frame_equal(rows[0].iloc[:1], rows[other].iloc[:1], check_exact=True)
        pd.testing.assert_frame_equal(costs[0].iloc[:1], costs[other].iloc[:1], check_exact=True)
    # Year 2: replace buys 365 packs, keep and retire none.
    assert [row["Replacements"].iloc[1] for row in rows] == [365, 0, 0]
    assert costs[0]["Cost_Replacement"].iloc[1] > 0.0
    assert costs[1]["Cost_Replacement"].iloc[1] == costs[2]["Cost_Replacement"].iloc[1] == 0.0
    # Retire stops the battery after the first day of year 2.
    assert results[2]["battery_end_of_life_events"][-1]["action"] == "retired"
    assert results[1]["battery_end_of_life_events"][-1]["action"] == "kept"
    assert rows[2]["Battery_Discharge_Throughput_kWh"].iloc[1] < rows[1]["Battery_Discharge_Throughput_kWh"].iloc[1]
    # A kept or retired pack is at or below its threshold: no residual value.
    assert results[1]["terminal_health_credit"] == results[2]["terminal_health_credit"] == 0.0
    provenance = results[2]["provenance"]
    assert provenance["resolved_config"]["battery_skipped_replacement_action"] == "retire"
    assert provenance["terminal_value"]["replacement_policy"]["skipped_replacement_action"] == "retire"


def test_retire_beats_keep_when_grid_charging_loses_money():
    _replace, keep, retire = _three(tariff=FLAT_TOU, smart_charging=FILL_OFF_PEAK)
    keep_result, retire_result = keep.result(), retire.result()
    keep_rows, retire_rows = keep._artifacts.yearly_df, retire._artifacts.yearly_df

    pd.testing.assert_frame_equal(keep_rows.iloc[:1], retire_rows.iloc[:1], check_exact=True)
    # Kept, the worn battery grid-charges all year at a loss; retired, only
    # on the first day of year 2.
    assert retire_rows["Grid_AC_To_Battery_kWh"].iloc[1] < 0.01 * keep_rows["Grid_AC_To_Battery_kWh"].iloc[1]
    assert retire_rows["Import_Cost"].iloc[1] < keep_rows["Import_Cost"].iloc[1]
    assert retire_result["npv_savings"] > keep_result["npv_savings"]


def test_a_daily_controller_cannot_dispatch_a_retired_battery():
    from tests.test_daily_persistence import COARSE, DAILY, TOU

    _replace, keep, retire = _three(tariff=TOU, smart_charging={**DAILY, **COARSE})
    keep_rows, retire_rows = keep._artifacts.yearly_df, retire._artifacts.yearly_df

    pd.testing.assert_frame_equal(keep_rows.iloc[:1], retire_rows.iloc[:1], check_exact=True)
    assert keep_rows["Grid_AC_To_Battery_kWh"].iloc[1] > 0.0
    # The controller still decides every day, but nothing reaches the battery.
    assert retire_rows["Grid_AC_To_Battery_kWh"].iloc[1] == 0.0
    executed = retire._artifacts.projection.controller_instructions
    assert executed is not None and len(executed) == 2 * 8760


def test_an_explicit_keep_is_the_default_app_run():
    default = _app(battery_replacement_min_remaining_years=1).result()
    explicit = _app(battery_replacement_min_remaining_years=1, battery_skipped_replacement_action="keep").result()
    for result in (default, explicit):
        result["provenance"].pop("execution", None)
    explicit["provenance"]["resolved_config"].pop("battery_skipped_replacement_action")
    default["provenance"]["resolved_config"].pop("battery_skipped_replacement_action")
    assert default == explicit


def test_monte_carlo_retires_each_trajectory(tmp_path, write_multiyear_weather):
    weather = write_multiyear_weather(tmp_path / "multi.csv")
    settings = MonteCarloSettings(weather_file=str(weather), n_runs=2, years_per_run=2, seed=7)
    config = {"battery_eol_percentage": EVERY_PERIOD_EOL, "battery_replacement_min_remaining_years": 1.0}

    keep = run_montecarlo(_mc_config(**config), settings)
    retire = run_montecarlo(_mc_config(**config, battery_skipped_replacement_action="retire"), settings)

    assert retire.provenance["resolved_config"]["battery_skipped_replacement_action"] == "retire"
    assert (retire.runs["total_replacements"] == keep.runs["total_replacements"]).all()
    assert (retire.runs["mean_pv_origin_battery_ac_load_kwh"] < keep.runs["mean_pv_origin_battery_ac_load_kwh"]).all()
    assert (retire.runs["first_end_of_life_action"] == "replaced").all()


def test_the_projected_optimizer_retires_and_records_the_setting(monkeypatch):
    keep, _captured = _projected_metrics(monkeypatch, replacement_min_remaining_years=2.0)
    retire, _captured = _projected_metrics(
        monkeypatch, replacement_min_remaining_years=2.0, skipped_replacement_action="retire"
    )
    assert retire["Projected_First_End_Of_Life_Action"] == "retired"
    assert keep["Projected_First_End_Of_Life_Action"] == "kept"
    assert retire["Projected_Total_Replacements"] == keep["Projected_Total_Replacements"] == 0
    years = retire["_yearly_summary_df"]
    assert years["Battery_Discharge_Throughput_kWh"].iloc[1] == 0.0
    assert (
        optimization_module._battery_replacement_treatment({"skipped_replacement_action": "retire"})[
            "skipped_replacement_action"
        ]
        == "retire"
    )


def test_the_key_reference_lists_the_setting():
    from pathlib import Path

    reference = (Path(__file__).resolve().parents[1] / "docs" / "getting-started" / "config-reference.md").read_text()
    assert '| `battery_skipped_replacement_action` | `"keep"` | — |' in reference


@pytest.mark.filterwarnings("ignore::breos.degradation.validation.BlastExperimentalRangeWarning")
def test_a_blast_pack_retires_the_same_way():
    from tests.test_terminal_replacement import BLAST

    retire = _run(96, engine=BLAST, **SKIP_SECOND, skipped_replacement_action="retire")
    assert [e.action for e in retire[4].attrs["end_of_life_events"]] == ["replaced", "retired"]
    for column in PV_ONLY_COLUMNS:
        np.testing.assert_array_equal(retire[0][column].iloc[48:], _pv_only(96)[column].iloc[48:], err_msg=column)
    # Calendar time still ages the pack a little at zero charge.
    soh = retire[4]["SOH"].to_numpy()
    assert soh[3] <= soh[2] <= soh[1]
