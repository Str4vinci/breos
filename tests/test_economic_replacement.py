"""The economic end-of-life decision (#403, ADR 0003 E14).

With ``battery_replacement_decision = "economic"`` (App and Monte Carlo) or
``[battery] replacement_decision = "economic"`` (the projected optimizer),
each end-of-life crossing compares replacing the pack with keeping or
retiring it over the rest of the horizon, and takes the highest NPV. The
default ``"threshold"`` changes nothing.
"""

from __future__ import annotations

import copy
import tomllib

import pandas as pd
import pytest

from breos import App
from breos.battery import BatteryConfig
from breos.montecarlo import MonteCarloSettings, run_montecarlo
from breos.optimization import evaluate_projected_design
from breos.projection import CarryState, ProjectionRun, decide_replacements
from tests.offline_app import serve_app_inputs
from tests.test_app_replacement_control import (
    DESIGN_EOL,
    LOCATION,
    OPTIMIZER_COSTS,
    OPTIMIZER_RATES,
    _design_inputs,
)
from tests.test_terminal_replacement import EVERY_PERIOD_EOL, _app, _mc_config, _run
from tests.test_vehicle_config import VEHICLE

pytestmark = pytest.mark.filterwarnings("ignore::breos.degradation.validation.BlastExperimentalRangeWarning")

# Native v1 on the test home crosses 93 % in year 2, and its successor
# again in year 3.
CROSSES_TWICE = {"projection_years": 3, "battery_eol_percentage": 0.93}
# A replacement nobody would buy, and one that costs nothing.
DEAR = {"battery_replacement_cost": 50_000.0}
FREE = {"battery_replacement_cost": 0.0}


def _events(result):
    return result["battery_end_of_life_events"]


# -- the decision over the branches ---------------------------------------------------


def _fake_run(swaps: int, events_by_limit):
    """A ``run(limit)`` whose projections carry only their swaps, crossings and limit."""

    def run(limit):
        made = swaps if limit is None else limit[0]
        return ProjectionRun(
            # The limit the branch ran under, so a fake NPV can be looked up.
            yearly_df=pd.DataFrame({"limit": [limit]}),
            carry=CarryState(),
            total_replacements=made,
            first_year_results_df=None,
            jit_cache_states=[f"run{limit}"],
            end_of_life_events=events_by_limit(limit),
        )

    return run


def _crossings(limit, swaps=2):
    """Every swap the limit allows, then the crossing it decides."""
    made = swaps if limit is None else limit[0]
    events = [{"action": "replaced", "reason": "end_of_life", "year": k + 1} for k in range(made)]
    if limit is not None:
        events.append({"action": "kept" if limit[1] == "keep" else "retired", "reason": "economic", "year": made + 1})
    return tuple(events)


@pytest.mark.parametrize(
    ("npv", "chosen"),
    [
        # Keep at the first crossing.
        ({None: 1.0, (1, "keep"): 2.0, (1, "retire"): 0.0, (0, "keep"): 5.0, (0, "retire"): 4.0}, (0, "keep")),
        # Replace once, then retire.
        ({None: 1.0, (1, "keep"): 2.0, (1, "retire"): 3.0, (0, "keep"): 0.0, (0, "retire"): 0.0}, (1, "retire")),
        # The threshold rule is best.
        ({None: 9.0, (1, "keep"): 2.0, (1, "retire"): 3.0, (0, "keep"): 0.0, (0, "retire"): 0.0}, None),
    ],
)
def test_the_best_branch_is_chosen_and_every_crossing_reports_its_alternatives(npv, chosen):
    run = _fake_run(2, _crossings)
    projection = decide_replacements(run, lambda p: npv[_limit_of(p)])
    record = projection.replacement_decision
    assert _limit_of(projection) == chosen
    assert record["chosen"]["npv"] == max(npv.values())
    assert record["threshold_npv"] == npv[None]
    assert record["extra_projections"] == 4
    assert [(b["replacements"], b["then"]) for b in record["branches"]] == [
        (2, None),
        (1, "keep"),
        (1, "retire"),
        (0, "keep"),
        (0, "retire"),
    ]
    # Every projection's cache observations are kept.
    assert len(projection.jit_cache_states) == 5
    decided = [event["decision"] for event in projection.end_of_life_events if "decision" in event]
    for k, decision in enumerate(decided, start=1):
        # Replacing is worth the best branch that makes this swap.
        assert decision["npv"]["replace"] == max(
            value for limit, value in npv.items() if limit is None or limit[0] >= k
        )
        assert decision["npv"]["keep"] == npv[(k - 1, "keep")]
        assert decision["npv"]["retire"] == npv[(k - 1, "retire")]
        best = max(decision["npv"], key=decision["npv"].get)
        assert decision["chosen"] == best


def _limit_of(projection):
    return projection.yearly_df["limit"].iloc[0]


def test_a_tie_keeps_the_threshold_rule_then_more_swaps_then_keep():
    run = _fake_run(2, _crossings)
    assert decide_replacements(run, lambda p: 0.0).total_replacements == 2
    npv = {None: 0.0, (1, "keep"): 1.0, (1, "retire"): 1.0, (0, "keep"): 1.0, (0, "retire"): 1.0}
    projection = decide_replacements(run, lambda p: npv[_limit_of(p)])
    assert _limit_of(projection) == (1, "keep")


def test_without_a_swap_there_is_nothing_to_decide():
    run = _fake_run(0, lambda limit: ())
    projection = decide_replacements(run, lambda p: 3.0)
    assert projection.replacement_decision["branches"] == [{"replacements": 0, "then": None, "npv": 3.0}]
    assert projection.replacement_decision["extra_projections"] == 0


# -- the battery's replacement limit --------------------------------------------------


@pytest.mark.parametrize("action", ["keep", "retire"])
def test_a_limit_makes_its_swaps_then_skips_the_next_crossing(action):
    *_, deg_df, _state = _run(
        24 * 5, replacement_limit=2, replacement_limit_action=action, eol_percentage=EVERY_PERIOD_EOL
    )
    events = deg_df.attrs["end_of_life_events"]
    assert [(e["action"], e["reason"]) for e in events] == [
        ("replaced", "end_of_life"),
        ("replaced", "end_of_life"),
        ("kept" if action == "keep" else "retired", "economic"),
    ]


def test_without_a_limit_the_battery_runs_as_before():
    default = _run(24 * 3, eol_percentage=EVERY_PERIOD_EOL)
    unlimited = _run(24 * 3, eol_percentage=EVERY_PERIOD_EOL, replacement_limit=None)
    pd.testing.assert_frame_equal(default[0], unlimited[0], check_exact=True)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"replacement_limit": -1}, "replacement_limit"),
        ({"replacement_limit": True}, "replacement_limit"),
        ({"replacement_limit": 1.0}, "replacement_limit"),
        ({"replacement_limit_action": "replace"}, "replacement_limit_action"),
    ],
)
def test_the_limit_is_checked(kwargs, message):
    with pytest.raises(ValueError, match=message):
        BatteryConfig(nominal_energy_wh=5000.0, **kwargs)


# -- App ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dear():
    return _app(**CROSSES_TWICE, **DEAR, battery_replacement_decision="economic")


@pytest.fixture(scope="module")
def dear_threshold():
    return _app(**CROSSES_TWICE, **DEAR)


def test_a_dear_replacement_is_not_bought(dear, dear_threshold):
    result, threshold = dear.result(), dear_threshold.result()
    assert [e["action"] for e in _events(threshold)] == ["replaced", "replaced"]
    (event,) = _events(result)
    assert (event["action"], event["reason"]) == ("kept", "economic")
    assert result["battery_replacements"] == 0
    assert result["npv_savings"] > threshold["npv_savings"]
    decision = event["decision"]
    record = result["battery_replacement_decision"]
    assert decision["chosen"] == "keep"
    # Replacing is valued at its best continuation: keep, retire or replace again at the second crossing.
    assert decision["npv"]["replace"] == max(b["npv"] for b in record["branches"] if b["replacements"] >= 1)
    assert decision["npv"]["keep"] == result["npv_savings"]
    assert record["mode"] == "economic"
    assert record["objective"] == "npv"
    assert record["look_ahead"] == "perfect_information"
    assert record["threshold_replacements"] == 2
    assert record["extra_projections"] == 4
    assert record["threshold_npv"] == threshold["npv_savings"]
    assert record["chosen"] == {"replacements": 0, "then": "keep", "npv": result["npv_savings"]}


def test_keeping_is_the_run_without_replacement(dear):
    disabled = _app(**CROSSES_TWICE, **DEAR, battery_enable_replacement=False).result()
    result = dear.result()
    assert result["npv_savings"] == disabled["npv_savings"]
    assert result["battery_soh_end_pct"] == disabled["battery_soh_end_pct"]
    assert [(e["year"], e["date"], e["action"]) for e in _events(result)] == [
        (e["year"], e["date"], e["action"]) for e in _events(disabled)
    ]


def test_a_free_replacement_is_bought():
    result = _app(**CROSSES_TWICE, **FREE, battery_replacement_decision="economic").result()
    threshold = _app(**CROSSES_TWICE, **FREE).result()
    assert [(e["action"], e["reason"]) for e in _events(result)] == [("replaced", "end_of_life")] * 2
    assert [e["decision"]["chosen"] for e in _events(result)] == ["replace", "replace"]
    assert [e["decision"]["crossing"] for e in _events(result)] == [1, 2]
    assert result["npv_savings"] == threshold["npv_savings"]
    assert result["battery_replacement_decision"]["chosen"]["then"] is None


def test_the_terminal_credit_is_in_the_objective():
    result = _app(
        **CROSSES_TWICE,
        **DEAR,
        battery_replacement_decision="economic",
        terminal_value={"basis": "battery_health_fraction"},
    ).result()
    record = result["battery_replacement_decision"]
    assert record["objective"] == "npv_with_terminal_health_credit"
    assert record["chosen"]["npv"] == result["npv_savings_terminal_adjusted"]


def test_a_revaluation_decides_again(dear):
    revalued = dear.revalue(FREE)
    assert revalued["provenance"]["revaluation"]["method"] == "resimulated"
    assert [e["action"] for e in _events(revalued)] == ["replaced", "replaced"]
    assert revalued["battery_replacement_decision"]["chosen"]["then"] is None


def test_the_threshold_rule_is_the_default(dear_threshold):
    explicit = _app(**CROSSES_TWICE, **DEAR, battery_replacement_decision="threshold").result()
    default = copy.deepcopy(dear_threshold.result())
    for result in (explicit, default):
        result["provenance"].pop("execution", None)
        result["provenance"].pop("config", None)
    assert "battery_replacement_decision" not in default
    assert all("decision" not in event for event in _events(default))
    assert explicit == default


def test_a_pv_only_run_decides_nothing():
    result = _app(battery_kwh=0, battery_replacement_decision="economic").result()
    assert "battery_replacement_decision" not in result


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"period": {"start": "2025-03-01", "end": "2025-04-01"}}, r"\[period\]"),
        ({"vehicle": VEHICLE}, r"\[vehicle\]"),
    ],
)
def test_the_decision_needs_one_buildings_lifetime(extra, message):
    with pytest.raises(ValueError, match=message):
        App(
            {
                "location": "porto",
                "n_modules": 8,
                "annual_consumption_kwh": 4000,
                "battery_kwh": 5,
                "start_date": "2025-01-01",
                "battery_replacement_decision": "economic",
                **extra,
            }
        )


# -- the projected optimizer ----------------------------------------------------------


def _design(decision, replacement_cost=None):
    weather, load = _design_inputs()
    battery = {
        "temperature": 20.0,
        "indoor_model": {"enabled": False},
        "eol_percentage": DESIGN_EOL,
        "replacement_decision": decision,
    }
    if replacement_cost is not None:
        battery["replacement_cost"] = replacement_cost
    config = {
        "location": LOCATION,
        "simulation": {"resolution": "h", "years_projection": 3},
        "pv": {"module": "Suntech_STP550S_STC", "degradation_rate": 0.007},
        "battery": battery,
        "costs": {**OPTIMIZER_COSTS, "dc_ac_ratio": 1.25},
        "financials": {**OPTIMIZER_RATES, "pv_degradation_rate": 0.007, "project_lifespan": 3},
    }
    return evaluate_projected_design(
        weather, load, config, n_modules=4, battery_kwh=5.0, tilt=30.0, azimuth=180.0, execution_backend="python"
    )


def test_the_optimizer_scores_the_decided_design():
    threshold, economic = _design("threshold"), _design("economic")
    assert threshold.metrics["Projected_Total_Replacements"] > 0
    assert "Projected_Threshold_Rule_NPV" not in threshold.metrics
    assert economic.metrics["Projected_Threshold_Rule_NPV"] == threshold.metrics["Projected_NPV"]
    assert economic.metrics["Projected_NPV"] >= threshold.metrics["Projected_NPV"]
    assert economic.metrics["Projected_Replacement_Decision_Projections"] == (
        1 + 2 * threshold.metrics["Projected_Total_Replacements"]
    )


def test_app_and_the_optimizer_decide_alike(monkeypatch):
    weather, load = _design_inputs()
    serve_app_inputs(monkeypatch, weather, load)
    app = App(
        {
            "location": LOCATION,
            "n_modules": 4,
            "annual_consumption_kwh": float(load["Load"].sum() / 1000),
            "battery_kwh": 5.0,
            "pv_module": "Suntech_STP550S_STC",
            "tilt": 30.0,
            "azimuth": 180.0,
            "projection_years": 3,
            "resolution": "h",
            "start_date": "2023-01-01",
            "costs": OPTIMIZER_COSTS,
            "inverter_loading_ratio": 1.25,
            **OPTIMIZER_RATES,
            "pv_degradation_rate": 0.007,
            "battery_temperature": 20.0,
            "battery_indoor_model": {"enabled": False},
            "execution_backend": "python",
            "battery_eol_percentage": DESIGN_EOL,
            "battery_replacement_decision": "economic",
        }
    )
    app.simulate()
    design = _design("economic")
    result = app.result()
    assert result["npv_savings"] == round(design.metrics["Projected_NPV"], 2)
    assert result["battery_replacements"] == design.metrics["Projected_Total_Replacements"]


# -- Monte Carlo ----------------------------------------------------------------------


def test_monte_carlo_decides_on_the_years_it_drew(tmp_path, write_multiyear_weather):
    weather = write_multiyear_weather(tmp_path / "multi.csv")
    settings = MonteCarloSettings(weather_file=str(weather), n_runs=3, years_per_run=3, seed=11, collect_yearly=True)
    config = _mc_config(projection_years=3, battery_eol_percentage=0.93, **DEAR)
    threshold = run_montecarlo(config, settings)
    economic = run_montecarlo({**config, "battery_replacement_decision": "economic"}, settings)
    draws = ["run", "Year", "Weather_Year", "Load_Scale"]
    pd.testing.assert_frame_equal(threshold.yearly[draws], economic.yearly[draws], check_exact=True)
    assert threshold.yearly["Replacements"].sum() > 0
    assert economic.yearly["Replacements"].sum() == 0


def test_a_community_site_cannot_decide_on_its_own_npv():
    from breos.community.config import parse_community_config
    from tests.test_community_config import EXAMPLE

    config = tomllib.loads(EXAMPLE)
    config["participants"][1].update(battery_kwh=5.0, battery_replacement_decision="economic")
    with pytest.raises(ValueError, match="community"):
        parse_community_config(config)
