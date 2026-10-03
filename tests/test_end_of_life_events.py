"""End-of-life events: when the battery crossed its threshold, what was done and why (ADR 0003 E11).

The final state of health does not say whether a pack was replaced late or
kept below its threshold. Every crossing is therefore recorded, with the
action (``"replaced"`` or ``"kept"``), the reason and the health at the
crossing, and reported by App, Monte Carlo and the projected optimizer. The
events add fields only: every existing value is unchanged.
"""

from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from breos.battery import (
    END_OF_LIFE_ACTIONS,
    END_OF_LIFE_REASONS,
    BatteryConfig,
    EndOfLifeEvent,
    simulate_energy_balance_summary,
)
from breos.economics import replacement_booking_time
from breos.montecarlo import MonteCarloSettings, run_montecarlo
from tests.test_replacement_min_remaining_years import YEAR_STEPS, _late_last_year_eol, _project
from tests.test_terminal_replacement import (
    EVERY_PERIOD_EOL,
    _app,
    _inputs,
    _mc_config,
    _projected_metrics,
    _replaced_steps,
    _require_numba,
    _run,
)


def _events(run) -> tuple[EndOfLifeEvent, ...]:
    return tuple(EndOfLifeEvent.from_record(record) for record in run[4].attrs["end_of_life_events"])


def _summary(n_steps: int, **config):
    pv, load, temperature = _inputs(n_steps)
    return simulate_energy_balance_summary(
        pv_dc=pv,
        houseload=load,
        temperature_series=temperature,
        battery_config=BatteryConfig(nominal_energy_wh=5000.0, **config),
        freq="h",
    )


def test_the_vocabularies_are_closed_lists():
    assert END_OF_LIFE_ACTIONS == ("replaced", "kept")
    assert END_OF_LIFE_REASONS == ("end_of_life", "min_remaining_years", "terminal_period", "replacement_disabled")


# -- one span -------------------------------------------------------------------------


def test_every_replacement_is_an_event_at_its_swap_step():
    run = _run(96, eol_percentage=EVERY_PERIOD_EOL)
    events = _events(run)

    assert [event.step for event in events] == _replaced_steps(run) == [23, 47, 71, 95]
    assert {event.action for event in events} == {"replaced"}
    assert {event.reason for event in events} == {"end_of_life"}
    index = _inputs(96)[0].index
    assert [event.timestamp for event in events] == [index[step] for step in (23, 47, 71, 95)]
    # The health the check compared, before the swap reset it to 100.
    assert all(99.0 < event.soh_pct <= 100.0 * EVERY_PERIOD_EOL for event in events)
    assert (run[4]["SOH"] == 100.0).all()


def test_a_run_that_never_reaches_end_of_life_has_no_event():
    assert _events(_run(96)) == ()
    assert _summary(96).end_of_life_events == ()


@pytest.mark.parametrize(
    "config,reason",
    [
        ({"replacement_min_remaining_years": 0.25}, "min_remaining_years"),
        ({"allow_terminal_replacement": False}, "terminal_period"),
        # A positive minimum also skips the final period; it is named first.
        ({"replacement_min_remaining_years": 0.25, "allow_terminal_replacement": False}, "min_remaining_years"),
    ],
)
def test_a_skipped_swap_is_a_kept_event_with_its_reason(config, reason):
    run = _run(96, eol_percentage=EVERY_PERIOD_EOL, **config)
    events = _events(run)

    assert _replaced_steps(run) == [23, 47, 71]
    assert [(event.step, event.action) for event in events] == [
        (23, "replaced"),
        (47, "replaced"),
        (71, "replaced"),
        (95, "kept"),
    ]
    assert events[-1].reason == reason
    assert events[-1].soh_pct == run[4]["SOH"].iloc[-1] <= 100.0 * EVERY_PERIOD_EOL


def test_a_kept_pack_crosses_once():
    # Replacement off: the pack crosses on the first close and stays below.
    run = _run(96, eol_percentage=EVERY_PERIOD_EOL, enable_replacement=False)
    assert [(event.step, event.action, event.reason) for event in _events(run)] == [
        (23, "kept", "replacement_disabled")
    ]
    # A one-year minimum on a one-year span skips every close; only the first records.
    guarded = _run(96, eol_percentage=EVERY_PERIOD_EOL, replacement_min_remaining_years=1.0)
    assert [(event.step, event.action, event.reason) for event in _events(guarded)] == [
        (23, "kept", "min_remaining_years")
    ]


def test_a_pack_that_starts_below_its_threshold_has_already_crossed():
    run = _run(48, eol_percentage=0.8, initial_soh=75.0, enable_replacement=False)
    assert _events(run) == ()
    # Unless the span replaces it.
    replaced = _run(48, eol_percentage=0.8, initial_soh=75.0)
    assert [(event.step, event.action) for event in _events(replaced)] == [(23, "replaced")]


def test_the_degradation_frame_attrs_stay_json_safe():
    run = _run(96, eol_percentage=EVERY_PERIOD_EOL, replacement_min_remaining_years=0.6)
    records = run[4].attrs["end_of_life_events"]
    assert records[0] == {
        "step": 23,
        "timestamp": "2025-01-01T23:00:00+00:00",
        "action": "replaced",
        "reason": "end_of_life",
        "soh_pct": records[0]["soh_pct"],
    }
    # pandas writes attrs as JSON in to_parquet and copies them into derived frames.
    assert json.loads(json.dumps(run[4].attrs)) == run[4].attrs
    assert run[4].iloc[:2].attrs == run[4].attrs


def test_a_degradation_frame_with_a_crossing_writes_to_parquet(tmp_path):
    pytest.importorskip("pyarrow")
    run = _run(96, eol_percentage=EVERY_PERIOD_EOL, replacement_min_remaining_years=0.6)
    run[4].to_parquet(tmp_path / "degradation.parquet")
    assert pd.read_parquet(tmp_path / "degradation.parquet").attrs == run[4].attrs


def test_summaries_hold_the_frames_events():
    config = {"eol_percentage": EVERY_PERIOD_EOL, "replacement_min_remaining_years": 0.6}
    assert _summary(96, **config).end_of_life_events == _events(_run(96, **config))


def test_python_and_numba_record_the_same_events():
    _require_numba()
    config = {"eol_percentage": EVERY_PERIOD_EOL, "replacement_min_remaining_years": 0.6}
    assert _events(_run(96, backend="numba", **config)) == _events(_run(96, **config))


# -- the projection ---------------------------------------------------------------------


def test_a_late_last_year_crossing_is_reported_as_kept():
    eol, soh = _late_last_year_eol()
    default = _project(0.0, eol=eol)
    guarded = _project(1.0, eol=eol)

    # The crossing closes the last year's second day: step 47 of 72.
    time_years = float(replacement_booking_time([3], [48 / YEAR_STEPS], [True])[0])
    assert time_years == 2.0 + 48 / YEAR_STEPS
    [replaced] = default.end_of_life_events
    [kept] = guarded.end_of_life_events
    assert replaced == {
        "year": 3,
        "time_years": time_years,
        "date": "2027-01-02",
        "action": "replaced",
        "reason": "end_of_life",
        "soh_pct": soh[-2],
    }
    assert kept == {**replaced, "action": "kept", "reason": "min_remaining_years"}


def test_a_kept_pack_is_not_reported_again_by_the_next_years():
    run = _project(1.5, years=3)
    events = run.end_of_life_events

    assert run.yearly_df["Replacements"].tolist() == [3, 1, 0]
    assert [(event["year"], event["action"]) for event in events] == [
        (1, "replaced"),
        (1, "replaced"),
        (1, "replaced"),
        (2, "replaced"),
        (2, "kept"),
    ]
    assert events[-1]["reason"] == "min_remaining_years"
    assert events[-1]["time_years"] == 1.0 + 48 / YEAR_STEPS
    # A replaced crossing's time is the booked replacement time.
    assert events[3]["time_years"] == float(replacement_booking_time([2], [24 / YEAR_STEPS], [True])[0])


def test_frames_summaries_and_numba_report_the_same_events():
    frames = _project(1.5)
    assert _project(1.5, aligned=True).end_of_life_events == frames.end_of_life_events
    _require_numba()
    assert _project(1.5, backend="numba").end_of_life_events == frames.end_of_life_events


# -- App, Monte Carlo and the optimizer ------------------------------------------------


def test_app_lists_every_crossing_and_repeats_the_first():
    result = _app(battery_replacement_min_remaining_years=1).result()
    events = result["battery_end_of_life_events"]

    # Year 1 swaps at all 365 closes; year 2 keeps the pack from its first close.
    assert len(events) == 366
    assert {event["action"] for event in events[:365]} == {"replaced"}
    assert events[-1] | {"soh_pct": None} == {
        "year": 2,
        "time_years": round(1.0 + 24 / 8760, 4),
        "date": "2026-01-01",
        "action": "kept",
        "reason": "min_remaining_years",
        "soh_pct": None,
    }
    assert events[-1]["soh_pct"] < 100.0 * EVERY_PERIOD_EOL
    assert result["battery_first_end_of_life_years"] == events[0]["time_years"] == round(24 / 8760, 4)
    assert result["battery_first_end_of_life_action"] == "replaced"
    assert result["battery_first_end_of_life_reason"] == "end_of_life"
    assert result["battery_first_end_of_life_soh_pct"] == events[0]["soh_pct"]


def test_app_without_a_crossing_reports_none():
    result = _app(battery_eol_percentage=0.7).result()
    assert result["battery_end_of_life_events"] == []
    for field in ("years", "action", "reason", "soh_pct"):
        assert result[f"battery_first_end_of_life_{field}"] is None


def test_monte_carlo_reports_each_trajectorys_first_crossing(tmp_path, write_multiyear_weather):
    weather = write_multiyear_weather(tmp_path / "multi.csv")
    settings = MonteCarloSettings(weather_file=str(weather), n_runs=2, years_per_run=2, seed=7)

    crossing = run_montecarlo(_mc_config(battery_eol_percentage=EVERY_PERIOD_EOL), settings).runs
    assert (crossing["first_end_of_life_years"] == 24 / 8760).all()
    assert (crossing["first_end_of_life_action"] == "replaced").all()
    assert (crossing["first_end_of_life_reason"] == "end_of_life").all()
    assert (crossing["first_end_of_life_soh_pct"] <= 100.0 * EVERY_PERIOD_EOL).all()

    none = run_montecarlo(_mc_config(), settings).runs
    assert none["first_end_of_life_years"].isna().all() and none["first_end_of_life_soh_pct"].isna().all()
    assert none["first_end_of_life_action"].isna().all()


def test_the_projected_optimizer_reports_the_first_crossing(monkeypatch):
    # Every close of the first year swaps, so the first crossing ends step 24.
    metrics, _captured = _projected_metrics(monkeypatch, replacement_min_remaining_years=2.0)
    assert metrics["Projected_First_End_Of_Life_Years"] == 24 / YEAR_STEPS
    assert metrics["Projected_First_End_Of_Life_Action"] == "kept"
    assert metrics["Projected_First_End_Of_Life_Reason"] == "min_remaining_years"
    assert metrics["Projected_First_End_Of_Life_SOH_%"] <= 100.0 * EVERY_PERIOD_EOL

    never, _captured = _projected_metrics(monkeypatch, eol_percentage=0.0)
    assert math.isnan(never["Projected_First_End_Of_Life_Years"])
    assert never["Projected_First_End_Of_Life_Action"] is None
    assert pd.isna(never["Projected_First_End_Of_Life_SOH_%"])


def test_monte_carlo_dates_run_from_the_target_year(tmp_path, write_multiyear_weather, monkeypatch):
    import breos.montecarlo as montecarlo_module

    seen: list = []
    real = montecarlo_module.first_end_of_life_metrics

    def recording(events):
        seen.append(events)
        return real(events)

    monkeypatch.setattr(montecarlo_module, "first_end_of_life_metrics", recording)
    weather = write_multiyear_weather(tmp_path / "multi.csv")
    settings = MonteCarloSettings(weather_file=str(weather), n_runs=2, years_per_run=3, seed=7, target_year=2030)
    run_montecarlo(
        _mc_config(battery_eol_percentage=EVERY_PERIOD_EOL, battery_replacement_min_remaining_years=1.0), settings
    )

    # Every sampled weather year is restamped to target_year, so a year's
    # dates are target_year's moved forward by the year index.
    for events in seen:
        assert events[0]["date"] == "2030-01-01"
        assert {event["date"][:4] for event in events if event["year"] == 2} == {"2031"}
        assert events[-1]["action"] == "kept" and events[-1]["date"] == "2032-01-01"
