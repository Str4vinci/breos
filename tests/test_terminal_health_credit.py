"""Terminal-health credit is a reporting sensitivity, with no changes to physical or financial flows."""

from dataclasses import replace
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from breos import App
from breos.app_config import resolve_app_config
from breos.battery import align_simulation_inputs
from breos.economics import terminal_health_credit
from breos.montecarlo import MonteCarloSettings, _simulate_trajectory, _summarize, run_montecarlo
from breos.optimization import evaluate_projected_design
from breos.projection import ProjectionYear, run_projection, value_projection
from tests.test_terminal_replacement import BLAST, EVERY_PERIOD_EOL, PRICING, _design_inputs, _inputs
from tools.generate_app_golden import SCENARIOS, _fake_fetch

BASE = {**SCENARIOS["native_h_battery"], "execution_backend": "python", **PRICING}
ENABLED = {"basis": "battery_health_fraction"}
FIELDS = ("terminal_health_credit", "terminal_health_credit_npv", "npv_savings_terminal_adjusted")
pytestmark = pytest.mark.filterwarnings("ignore::breos.degradation.validation.BlastExperimentalRangeWarning")


def _credit(health=1.0, **overrides):
    return terminal_health_credit(
        **{
            "final_soh_fraction": health,
            "threshold": 0.7,
            "replacement_cost_each": 2000.0,
            "inflation_rate": 0.02,
            "replacement_cost_learning": 0.03,
            "discount_rate": 0.05,
            "horizon_years": 3,
            "npv_savings": -500.0,
            "allow_terminal_replacement": True,
            **overrides,
        }
    )


@pytest.mark.parametrize("health,fraction", [(0.0, 0), (0.69, 0), (0.7, 0), (0.85, 0.5), (1.0, 1), (1.1, 1)])
def test_health_is_clipped_and_fresh_pack_earns_the_full_year_t_price(health, fraction):
    credit = _credit(health)
    price = 2000 * 1.02**3 * 0.97**3
    assert credit.nominal == pytest.approx(price * fraction)
    assert credit.npv == pytest.approx(price * fraction / 1.05**3)
    assert credit.adjusted_npv == -500 + credit.npv
    assert credit.provenance["health_fraction"] == pytest.approx(fraction)
    assert credit.provenance["booking_time_years"] == 3


@pytest.mark.parametrize(
    "key",
    [
        "final_soh_fraction",
        "threshold",
        "replacement_cost_each",
        "inflation_rate",
        "replacement_cost_learning",
        "discount_rate",
        "horizon_years",
        "npv_savings",
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_credit_inputs_are_rejected(key, value):
    with pytest.raises(ValueError, match="finite"):
        _credit(**{key: value})


@pytest.mark.parametrize("threshold", [1.0, 1.1])
def test_threshold_at_or_above_one_is_rejected(threshold):
    with pytest.raises(ValueError, match="threshold"):
        _credit(threshold=threshold)
    with pytest.raises(ValueError, match="battery_eol_percentage"):
        App({**BASE, "terminal_value": ENABLED, "battery_eol_percentage": threshold})


@pytest.mark.parametrize("table", [{"basis": "resale"}, {"threshold": 0.8}, {"price": 1}])
def test_terminal_table_rejects_other_bases_and_keys(table):
    with pytest.raises(ValueError, match="terminal_value"):
        App({**BASE, "terminal_value": table})


def _app(**overrides):
    config = {**BASE, **overrides}
    if "period" in config:
        config.pop("projection_years")
    with (
        mock.patch("breos.app.fetch_tmy_weather_data", _fake_fetch),
        mock.patch("breos.app.load_weather", return_value=None),
    ):
        app = App(config)
        app.simulate()
    return app


@pytest.fixture(scope="module")
def app():
    return _app()


def _old_values(result):
    result = {key: value for key, value in result.items() if key not in FIELDS}
    result["provenance"] = {
        key: value for key, value in result["provenance"].items() if key not in ("terminal_value", "revaluation")
    }
    result["provenance"]["resolved_config"] = {
        key: value for key, value in result["provenance"]["resolved_config"].items() if key != "terminal_value"
    }
    return result


@pytest.mark.parametrize("table", [None, {}, {"basis": "none"}, ENABLED])
def test_off_and_enabled_keep_all_existing_results_bit_identical(app, table):
    result = app.revalue({"terminal_value": table})
    assert _old_values(result) == _old_values(app.result())
    if table != ENABLED:
        assert all(result[field] is None for field in FIELDS)
        assert "terminal_value" not in result["provenance"]
    else:
        assert result["terminal_health_credit"] > 0
    original = app._artifacts
    assert original is not None and original.projection is not None
    resolved = resolve_app_config({**BASE, "terminal_value": table})
    valued = value_projection(resolved.cfg, resolved, original.projection)
    pd.testing.assert_frame_equal(valued.cost_projection, original.cost_projection, check_exact=True)


@pytest.mark.parametrize("backend", ["python", "numba"])
@pytest.mark.parametrize("engine", [{}, BLAST], ids=["native", "blast"])
@pytest.mark.parametrize("freq", ["h", "15min"])
@pytest.mark.parametrize("allow", [True, False], ids=["final-replacement", "guarded"])
def test_final_installed_pack_is_credited_without_changing_any_outlay_or_state(backend, engine, freq, allow):
    if backend == "numba":
        pytest.importorskip("numba")
    pv, load, temp = _inputs(72 if freq == "h" else 288, freq)
    resolved = resolve_app_config(
        {
            **BASE,
            **engine,
            "resolution": freq,
            "execution_backend": backend,
            "terminal_value": ENABLED,
            "battery_eol_percentage": EVERY_PERIOD_EOL,
            "battery_allow_terminal_replacement": allow,
        }
    )
    cfg = resolved.cfg
    run = run_projection(
        cfg,
        resolved,
        3,
        lambda year: ProjectionYear(1.0, pv_dc=pv, houseload=load, temperature_series=temp),
        has_battery=True,
        execution_backend=backend,
    )
    value = value_projection(cfg, resolved, run)
    disabled = value_projection({**cfg, "terminal_value": None}, resolved, run)
    credit = value.terminal_health
    assert credit is not None
    pd.testing.assert_frame_equal(value.cost_projection, disabled.cost_projection, check_exact=True)
    pd.testing.assert_frame_equal(value.yearly_df, disabled.yearly_df, check_exact=True)
    assert value.total_replacement_cost == disabled.total_replacement_cost == (9 if allow else 8) * 2000
    assert run.yearly_df["Replacements"].tolist() == ([3, 3, 3] if allow else [3, 3, 2])
    assert credit.provenance["final_soh_fraction"] == run.carry.soh_pct / 100
    assert credit.provenance["replacement_policy"]["allow_terminal_replacement"] is allow
    if allow:
        assert run.carry.soh_pct == 100
        assert credit.nominal == pytest.approx(2000 * 1.02**3 * 0.97**3)
        # A single replacement booked at T must use exactly this full-pack price.
        final_event = run.yearly_df.copy()
        final_event["Replacements"] = 0
        final_event["Replacement_Year_Fraction"] = np.nan
        final_event.loc[final_event.index[-1], "Replacements"] = 1
        final_event.loc[final_event.index[-1], "Replacement_Year_Fraction"] = 1.0
        booked = value_projection(cfg, resolved, replace(run, yearly_df=final_event))
        assert credit.nominal == booked.cost_projection["Cost_Replacement"].iloc[-1]
    else:
        assert run.carry.soh_pct / 100 <= EVERY_PERIOD_EOL
        assert credit.nominal == credit.npv == 0
    # The same routine must inherit a resolved full-pack price override, without any independent price setting.
    with mock.patch("breos.projection.build_costs_dict", return_value={**value.costs, "replacement_cost_each": 1234.0}):
        override = value_projection(cfg, resolved, run).terminal_health
    assert override is not None
    assert override.provenance["replacement_cost_each_t0_prices"] == 1234.0
    assert override.nominal == pytest.approx(credit.nominal * 1234 / 2000)


def test_enabled_pv_only_reports_zero_credit():
    result = _app(battery_kwh=0, terminal_value=ENABLED).result()
    assert result["terminal_health_credit"] == result["terminal_health_credit_npv"] == 0
    assert result["npv_savings_terminal_adjusted"] == result["npv_savings"]
    assert result["provenance"]["terminal_value"]["final_soh_fraction"] is None


def test_period_keeps_credit_null_also_on_revalue():
    app = _app(period={"start": "2025-07-01", "end": "2025-07-04"}, terminal_value=ENABLED)
    for result in (app.result(), app.revalue({"costs": {"storage_cost_per_kwh": 100}})):
        assert all(result[field] is None for field in (*FIELDS, "npv_savings"))
        assert set(FIELDS) <= set(result["period"]["skipped_fields"])
        assert "terminal_value" not in result["provenance"]


def test_revalue_uses_retained_health_for_basis_price_and_rate_changes(app, monkeypatch):
    monkeypatch.setattr("breos.app.run_app_simulation", mock.Mock(side_effect=AssertionError("unexpected simulation")))
    monkeypatch.setattr(
        "breos.runners.app.run_app_simulation", mock.Mock(side_effect=AssertionError("unexpected simulation"))
    )
    changes = {
        "terminal_value": ENABLED,
        "inflation_rate": 0.04,
        "discount_rate": 0.08,
        "replacement_cost_learning": 0.01,
        "costs": {"storage_cost_per_kwh": 250},
    }
    result = app.revalue(changes)
    assert result["provenance"]["revaluation"]["method"] == "repriced"
    assert app._artifacts is not None
    credit = _credit(
        app._artifacts.current_soh / 100,
        replacement_cost_each=1250,
        inflation_rate=0.04,
        discount_rate=0.08,
        replacement_cost_learning=0.01,
    )
    assert result["terminal_health_credit"] == round(credit.nominal, 2)
    assert result["terminal_health_credit_npv"] == round(credit.npv, 2)
    assert result["npv_savings_terminal_adjusted"] == pytest.approx(result["npv_savings"] + credit.npv, abs=0.01)
    assert all(app.result()[field] is None for field in FIELDS)


def test_revalue_can_disable_an_enabled_credit(monkeypatch):
    app = _app(terminal_value=ENABLED)
    monkeypatch.setattr(
        "breos.runners.app.run_app_simulation", mock.Mock(side_effect=AssertionError("unexpected simulation"))
    )
    for table in (None, {"basis": "none"}):
        result = app.revalue({"terminal_value": table})
        assert all(result[field] is None for field in FIELDS)
        assert _old_values(result) == _old_values(app.result())


@pytest.mark.parametrize("capacity", [0, 5])
def test_montecarlo_enabled_and_disabled_keep_existing_trajectory_values(capacity):
    pv, load, temp = _inputs(72)
    aligned = align_simulation_inputs(pv, load, temp, freq="h")
    metrics, trajectories = [], []
    for table in (None, ENABLED):
        resolved = resolve_app_config({**BASE, "battery_kwh": capacity, "terminal_value": table})
        row, trajectory = _simulate_trajectory(
            resolved.cfg,
            resolved,
            np.array([2025]),
            3,
            MonteCarloSettings(weather_file="unused.csv", execution_backend="python", load_uncertainty=0),
            np.random.default_rng(0),
            {2025: aligned},
            None,
        )
        metrics.append(row)
        trajectories.append(trajectory)
    pd.testing.assert_frame_equal(trajectories[0], trajectories[1], check_exact=True)
    existing = [key for key in metrics[0] if key not in FIELDS]
    pd.testing.assert_series_equal(
        pd.Series({key: metrics[0][key] for key in existing}),
        pd.Series({key: metrics[1][key] for key in existing}),
        check_exact=True,
    )
    assert all(np.isnan(metrics[0][field]) for field in FIELDS)
    assert not set(FIELDS) & _summarize(pd.DataFrame([metrics[0]])).keys()
    if capacity == 0:
        assert metrics[1]["terminal_health_credit"] == metrics[1]["terminal_health_credit_npv"] == 0
        assert metrics[1]["npv_savings_terminal_adjusted"] == metrics[1]["npv_savings"]


def test_montecarlo_values_each_trajectory_before_aggregation(monkeypatch):
    pv, load, temp = _inputs(72)
    aligned = align_simulation_inputs(pv, load, temp, freq="h")
    resolved = resolve_app_config({**BASE, "terminal_value": ENABLED})
    settings = MonteCarloSettings(weather_file="unused.csv", n_runs=3, years_per_run=3, execution_backend="python")
    outputs = []
    for i, scale in enumerate((0.2, 1.0, 4.0)):
        scaled = replace(aligned, load_w=aligned.load_w * scale)
        metrics, trajectory = _simulate_trajectory(
            resolved.cfg,
            resolved,
            np.array([2025]),
            3,
            replace(settings, load_uncertainty=0),
            np.random.default_rng(i),
            {2025: scaled},
            None,
        )
        credit = _credit(metrics["final_soh_pct"] / 100, npv_savings=metrics["npv_savings"])
        for field, expected in zip(FIELDS, (credit.nominal, credit.npv, credit.adjusted_npv), strict=True):
            assert metrics[field] == expected
        outputs.append((i, metrics, trajectory, None))
    assert len({output[1]["terminal_health_credit"] for output in outputs}) == 3
    # Feed the actual per-trajectory valuations through study assembly and its statistics.
    monkeypatch.setattr("breos.montecarlo._precompute_year_caches", lambda *args, **kwargs: ({2025: pv}, {2025: temp}))
    monkeypatch.setattr("breos.montecarlo._run_trajectory_index", lambda i: outputs[i])
    result = run_montecarlo(resolved.cfg, settings)
    for field in FIELDS:
        assert result.summary[field] == _summarize(result.runs)[field]
        assert result.summary[field]["count"] == 3
        assert result.summary[field]["mean"] == result.runs[field].mean()
    records = result.provenance["terminal_value"]["trajectories"]
    assert [record["final_soh_fraction"] for record in records] == (result.runs["final_soh_pct"] / 100).tolist()
    assert [record["run"] for record in records] == [1, 2, 3]
    assert "_terminal_value_provenance" not in result.runs


def test_projected_optimizer_accepts_and_ignores_the_table(monkeypatch):
    weather, load, pv = _design_inputs()
    monkeypatch.setattr("breos.optimization.calculate_pv_production_dc", lambda **kwargs: pv)
    config = {
        "location": {"latitude": 41.15, "longitude": -8.63},
        "simulation": {"resolution": "h", "years_projection": 2},
        "financials": {"project_lifespan": 2},
        "battery": {"eol_percentage": EVERY_PERIOD_EOL},
    }
    arguments = {"n_modules": 4, "battery_kwh": 5.0, "tilt": 35.0, "azimuth": 180.0}
    default = evaluate_projected_design(weather, load, config, **arguments)
    enabled = evaluate_projected_design(weather, load, {**config, "terminal_value": ENABLED}, **arguments)
    assert default.metrics == enabled.metrics
    pd.testing.assert_frame_equal(default.yearly, enabled.yearly, check_exact=True)
    assert "terminal_value" not in enabled.provenance
    assert not any("terminal" in key for key in enabled.metrics)
