"""Replacement control in App and the per-year degradation history (#377).

``battery_enable_replacement = false`` skips every end-of-life replacement,
as ``BatteryConfig.enable_replacement`` and the optimizer's ``[battery]
enable_replacement`` do. The pack keeps ageing below its threshold, or is
retired, as ``battery_skipped_replacement_action`` says. Every App result
with a battery lists the pack's state at the end of each project year in
``battery_degradation_history``; a quantity the degradation model does not
supply is None, never 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import breos.optimization as optimization_module
from breos.app_config import resolve_app_config
from breos.montecarlo import MonteCarloSettings, run_montecarlo
from breos.projection import build_battery_config
from tests.test_terminal_replacement import BLAST, EVERY_PERIOD_EOL, _app, _mc_config

pytestmark = pytest.mark.filterwarnings("ignore::breos.degradation.validation.BlastExperimentalRangeWarning")

BASE = {"location": "porto", "n_modules": 8, "annual_consumption_kwh": 4000, "battery_kwh": 5}
HISTORY_FIELDS = [
    "year",
    "soh_pct",
    "capacity_kwh",
    "usable_capacity_kwh",
    "replacements",
    "charge_throughput_kwh",
    "discharge_throughput_kwh",
    "fec",
    "cumulative_fec",
    "mean_soc_pct",
    "mean_cell_temperature_c",
    "cycle_loss_pct",
    "calendar_loss_pct",
    "resistance_growth_pct",
    "round_trip_efficiency",
]


@pytest.fixture(scope="module")
def disabled():
    """Two years with replacement off and a threshold every period crosses."""
    return _app(battery_enable_replacement=False, terminal_value={"basis": "battery_health_fraction"})


# -- the setting ---------------------------------------------------------------------


def test_the_key_defaults_to_replacing_and_reaches_the_battery():
    assert resolve_app_config(BASE).cfg["battery_enable_replacement"] is True
    resolved = resolve_app_config({**BASE, "battery_enable_replacement": False})
    assert build_battery_config(resolved.cfg, resolved, initial_soh=100.0).enable_replacement is False
    with pytest.raises(TypeError, match="'battery_enable_replacement' must be a boolean"):
        resolve_app_config({**BASE, "battery_enable_replacement": "false"})


def test_an_explicit_true_is_the_default_run():
    default = _app().result()
    explicit = _app(battery_enable_replacement=True).result()
    for result in (default, explicit):
        result["provenance"].pop("execution", None)
        result["provenance"]["resolved_config"].pop("battery_enable_replacement")
    assert default == explicit


# -- running without replacement -------------------------------------------------


def test_a_pack_without_replacement_keeps_ageing_below_its_threshold(disabled):
    result = disabled.result()
    rows = disabled._artifacts.yearly_df
    costs = disabled._artifacts.cost_projection

    assert result["battery_replacements"] == 0
    assert (rows["Replacements"] == 0).all()
    assert (costs["Cost_Replacement"] == 0.0).all()
    assert result["battery_replacement_cost_npv"] == 0.0
    # Crossed in the first period, recorded once, and never replaced.
    assert [(e["year"], e["action"], e["reason"]) for e in result["battery_end_of_life_events"]] == [
        (1, "kept", "replacement_disabled")
    ]
    # The health is not frozen: the pack goes on cycling and fading.
    history = result["battery_degradation_history"]
    assert 100.0 * EVERY_PERIOD_EOL > history[0]["soh_pct"] > history[1]["soh_pct"]
    assert history[1]["discharge_throughput_kwh"] > 0.0
    assert history[1]["cumulative_fec"] > history[0]["cumulative_fec"] > 0.0
    # Below the threshold, the residual value is zero, and the policy is recorded.
    assert result["terminal_health_credit"] == 0.0
    assert result["provenance"]["resolved_config"]["battery_enable_replacement"] is False
    assert result["provenance"]["terminal_value"]["replacement_policy"]["enable_replacement"] is False


def test_a_pack_without_replacement_retires_at_its_first_crossing():
    result = _app(battery_enable_replacement=False, battery_skipped_replacement_action="retire").result()
    assert [(e["action"], e["reason"]) for e in result["battery_end_of_life_events"]] == [
        ("retired", "replacement_disabled")
    ]
    history = result["battery_degradation_history"]
    assert history[1]["charge_throughput_kwh"] == history[1]["discharge_throughput_kwh"] == 0.0
    assert history[1]["soh_pct"] == history[0]["soh_pct"]


def test_disabled_replacement_names_its_own_reason_over_the_other_rules():
    result = _app(
        battery_enable_replacement=False,
        battery_replacement_min_remaining_years=5.0,
        battery_allow_terminal_replacement=False,
    ).result()
    assert [e["reason"] for e in result["battery_end_of_life_events"]] == ["replacement_disabled"]


def test_monte_carlo_runs_each_trajectory_without_replacement(tmp_path, write_multiyear_weather):
    weather = write_multiyear_weather(tmp_path / "multi.csv")
    settings = MonteCarloSettings(weather_file=str(weather), n_runs=2, years_per_run=2, seed=7)
    config = _mc_config(battery_eol_percentage=EVERY_PERIOD_EOL, battery_enable_replacement=False)
    runs = run_montecarlo(config, settings)
    assert runs.provenance["resolved_config"]["battery_enable_replacement"] is False
    assert (runs.runs["total_replacements"] == 0).all()
    assert (runs.runs["first_end_of_life_reason"] == "replacement_disabled").all()


def test_the_optimizer_records_its_replacement_setting():
    assert optimization_module._battery_replacement_treatment({})["enable_replacement"] is True
    treatment = optimization_module._battery_replacement_treatment({"enable_replacement": False})
    assert treatment["enable_replacement"] is False


# -- the degradation history -------------------------------------------------------


def test_the_history_reports_each_year_from_the_year_rows(disabled):
    history = disabled.result()["battery_degradation_history"]
    rows = disabled._artifacts.yearly_df
    assert [list(entry) for entry in history] == [HISTORY_FIELDS] * 2
    for entry, (_, row) in zip(history, rows.iterrows(), strict=True):
        assert entry["year"] == row["Year"]
        assert entry["soh_pct"] == round(row["Battery_SOH_%"], 2)
        assert entry["capacity_kwh"] == round(5 * row["Battery_SOH_%"] / 100, 3)
        assert entry["usable_capacity_kwh"] == round(5 * row["Battery_SOH_%"] / 100 * 0.8, 3)
        assert entry["fec"] == round(row["Battery_Annual_FEC"], 2)
        assert entry["mean_soc_pct"] == round(row["Battery_SOC_Absolute_Mean_%"], 2)
        # The native split adds up to the health lost since commissioning.
        assert entry["cycle_loss_pct"] + entry["calendar_loss_pct"] == pytest.approx(100 - entry["soh_pct"], abs=0.02)
        assert entry["cycle_loss_pct"] > 0.0 and entry["calendar_loss_pct"] > 0.0
        # Without resistance fade the model supplies neither.
        assert entry["resistance_growth_pct"] is None and entry["round_trip_efficiency"] is None


def test_the_mean_cell_temperature_is_the_years_mean(disabled):
    first = disabled._artifacts.first_year_results_df
    row = disabled._artifacts.yearly_df.iloc[0]
    assert row["Battery_Cell_Temperature_Mean_C"] == pytest.approx(float(first["T_cell"].mean()), rel=1e-12)


def test_a_replaced_pack_restarts_its_history():
    history = _app().result()["battery_degradation_history"]
    # Every period crosses and is replaced, so the year ends on a new pack.
    assert [entry["replacements"] for entry in history] == [365, 365]
    for entry in history:
        assert entry["soh_pct"] == 100.0 and entry["cumulative_fec"] == 0.0
        assert entry["cycle_loss_pct"] == entry["calendar_loss_pct"] == 0.0
        assert entry["fec"] > 0.0


def test_blast_reports_no_cycle_and_calendar_split():
    history = _app(battery_enable_replacement=False, **BLAST).result()["battery_degradation_history"]
    for entry in history:
        assert entry["cycle_loss_pct"] is None and entry["calendar_loss_pct"] is None
        assert entry["resistance_growth_pct"] is None and entry["round_trip_efficiency"] is None
        assert entry["fec"] > 0.0 and entry["soh_pct"] < 100.0


def test_resistance_fade_reports_growth_and_efficiency():
    app = _app(battery_eol_percentage=0.7, enable_resistance_fade=True, battery_rte=0.9)
    history = app.result()["battery_degradation_history"]
    growth = app._artifacts.yearly_df["Battery_Resistance_Growth"].to_numpy()
    assert [entry["resistance_growth_pct"] for entry in history] == list(np.round(growth * 100, 2))
    assert 0.0 < history[0]["resistance_growth_pct"] < history[1]["resistance_growth_pct"]
    assert history[0]["round_trip_efficiency"] == round(0.9 / (1 + growth[0]), 4)
    assert 0.9 > history[0]["round_trip_efficiency"] > history[1]["round_trip_efficiency"]


def test_a_pv_only_run_has_no_history():
    assert "battery_degradation_history" not in _app(battery_kwh=0).result()


def test_the_history_follows_a_revaluation(disabled):
    revalued = disabled.revalue({"discount_rate": 0.07})
    assert revalued["battery_degradation_history"] == disabled.result()["battery_degradation_history"]


def test_summaries_and_frames_agree_on_the_new_row_column():
    from tests.test_battery_retire import _project

    frames, summary = _project("keep"), _project("keep", aligned=True)
    pd.testing.assert_series_equal(
        frames.yearly_df["Battery_Cell_Temperature_Mean_C"],
        summary.yearly_df["Battery_Cell_Temperature_Mean_C"],
        check_exact=True,
    )


def test_the_key_reference_lists_the_setting():
    from pathlib import Path

    reference = (Path(__file__).resolve().parents[1] / "docs" / "getting-started" / "config-reference.md").read_text()
    assert "| `battery_enable_replacement` | `True` | — |" in reference
