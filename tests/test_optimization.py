"""Tests for optimization guardrails."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pymoo")

from breos.optimization import (
    SolarDesignProblem,
    optimize_system_multi_objective,
)
from tests.conftest import _stub_projection_balance, declare_instant_at_labels

LOCATION = {"latitude": 41.15, "longitude": -8.61, "timezone": "UTC"}
LOOSE = {"budget": 100000.0, "max_area_m2": 100.0}


@pytest.fixture
def tiny_inputs():
    """Two dark hourly steps of weather and a flat 500 W load."""
    idx = pd.date_range("2025-01-01 00:00", periods=2, freq="h", tz="UTC")
    tmy_data = declare_instant_at_labels(pd.DataFrame({"temp_air": [15.0, 16.0], "ghi": [0.0, 0.0]}, index=idx))
    houseload = pd.DataFrame({"Load": [500.0, 500.0]}, index=idx)
    return idx, tmy_data, houseload


def _config(**tables):
    """The site with a fixed azimuth, plus the tables given."""
    return {"location": LOCATION, "mode": {"fixed_azimuth": 180}, **tables}


def test_projected_scoring_is_the_default_with_two_objectives_and_an_optional_zeb_constraint(tiny_inputs):
    """A design is chosen for its lifetime, so lifetime scoring is the default.

    The annual basis scored a candidate on its first year alone, before any PV
    degradation, battery fade, or replacement. Leaving that as the default made
    the cheap approximation the one a caller got without asking.
    """
    _idx, tmy_data, houseload = tiny_inputs

    problem = SolarDesignProblem(tmy_data, houseload, _config(constraints={"enforce_zeb": True}))

    assert problem.objective_basis == "projected"
    assert problem.n_obj == 2
    assert problem.n_ieq_constr == 3


def test_steady_state_objective_basis_was_removed(tiny_inputs):
    """The annual basis is gone, and naming it fails instead of falling back."""
    _idx, tmy_data, houseload = tiny_inputs
    config = _config(optimization={"objective_basis": "steady_state"})

    with pytest.raises(ValueError, match=r"'steady_state' was removed in 0\.7\.0.*Use 'projected'"):
        SolarDesignProblem(tmy_data, houseload, config)


def test_projected_objective_basis_rejects_unknown_value(tiny_inputs):
    _idx, tmy_data, houseload = tiny_inputs

    with pytest.raises(ValueError, match="objective_basis"):
        SolarDesignProblem(
            tmy_data, houseload, {"location": LOCATION, "optimization": {"objective_basis": "lifetime-ish"}}
        )


# The four metrics candidate scoring reads, for tests that stub the projection.
_PROJECTED_STUB = {
    "Projected_Grid_Independence_%": 50.0,
    "Projected_ZEB_Ratio": 0.5,
    "Projected_NPV": 1000.0,
    "Projected_Initial_Cost": 500.0,
}


def test_projected_zeb_constraint_uses_projected_diagnostic(monkeypatch, tiny_inputs):
    idx, tmy_data, houseload = tiny_inputs
    projected = {
        "Projected_Grid_Independence_%": 50.0,
        "Projected_ZEB_Ratio": 0.8,
        "Projected_NPV": 1000.0,
        "Projected_Initial_Cost": 0.0,
    }
    monkeypatch.setattr("breos.optimization.calculate_pv_production_dc", lambda **kwargs: pd.Series(0.0, index=idx))
    monkeypatch.setattr("breos.optimization._evaluate_projected_design_metrics", lambda **kwargs: projected)
    config = _config(
        constraints={**LOOSE, "enforce_zeb": True},
        simulation={"resolution": "h", "years_projection": 2},
        financials={"project_lifespan": 2},
    )
    out = {}
    problem = SolarDesignProblem(tmy_data, houseload, config)
    problem._evaluate(np.array([2.0, 0.0, 10.0]), out)

    assert out["F"] == pytest.approx([0.5, -1000.0])
    assert out["ZEB_Ratio"] == pytest.approx(0.8)
    assert out["G"][2] == pytest.approx(0.2)


def test_solar_design_problem_area_constraint_uses_pv_dimensions(monkeypatch, tiny_inputs):
    idx, tmy_data, houseload = tiny_inputs
    config = _config(
        simulation={"resolution": "h"},
        constraints={"budget": 100000, "max_area_m2": 10.0, "max_modules": 5},
        pv={
            "module": "Suntech_STP550S_STC",
            "dimensions": {"width": 2.0, "length": 3.0},
        },
        battery={"max_soc": 0.9, "min_soc": 0.1},
    )

    monkeypatch.setattr("breos.optimization.calculate_pv_production_dc", lambda **kwargs: pd.Series(0.0, index=idx))
    monkeypatch.setattr("breos.optimization._evaluate_projected_design_metrics", lambda **kwargs: _PROJECTED_STUB)

    problem = SolarDesignProblem(tmy_data, houseload, config)
    out: dict = {}
    problem._evaluate(np.array([2.0, 0.0, 10.0], dtype=float), out)

    assert out["G"][1] == pytest.approx(2.0)


def test_the_search_bounds_and_tilt_floor_are_the_resolved_ones(tiny_inputs):
    _idx, tmy_data, houseload = tiny_inputs

    default = SolarDesignProblem(tmy_data, houseload, {"location": LOCATION})
    bounds = default.config["constraints"]
    assert (bounds["budget"], bounds["max_area_m2"], default.xl[2]) == (10000.0, 20.0, 10.0)
    assert (default.xu[0], default.xu[1], default.xu[2]) == (60, 30.0, 90.0)

    raised = SolarDesignProblem(tmy_data, houseload, {"location": LOCATION, "constraints": {"min_tilt_deg": 25.0}})
    assert raised.xl[2] == 25.0

    explicit = SolarDesignProblem(
        tmy_data, houseload, _config(constraints={"max_modules": 5, "max_battery_kwh": 7.0, "max_tilt_deg": 45.0})
    )
    assert explicit.xu[0] == pytest.approx(5.0)
    assert explicit.xu[1] == pytest.approx(7.0)
    assert explicit.xu[2] == pytest.approx(45.0)


def test_discrete_repair_keeps_candidates_inside_the_bounds(tiny_inputs):
    """Snapping to the grid must not round a candidate past a bound.

    Plain rounding took 62.9 degrees to 65 under a 63 degree maximum, and
    29.5 kWh to 30 under a 29.5 kWh cap.
    """
    from breos.optimization import DiscreteGridRepair

    _idx, tmy_data, houseload = tiny_inputs
    config = {
        "location": LOCATION,
        "constraints": {"max_modules": 7.5, "max_battery_kwh": 29.5, "max_tilt_deg": 63.0},
    }
    problem = SolarDesignProblem(tmy_data, houseload, config)
    X = np.array(
        [
            [7.5, 29.5, 62.9, 268.0],
            [1.0, 0.0, 10.0, 90.0],
            [3.4, 12.6, 37.4, 181.0],
        ]
    )
    repaired = DiscreteGridRepair()._do(problem, X.copy())

    assert repaired.tolist() == [
        [7.0, 29.0, 60.0, 270.0],
        [1.0, 0.0, 10.0, 90.0],
        [3.0, 13.0, 35.0, 180.0],
    ]
    assert np.all(repaired >= problem.xl) and np.all(repaired <= problem.xu)

    # pymoo calls the repair through Repair.do with a population; _do receives
    # its design matrix and the result is written back onto the population.
    from pymoo.core.population import Population

    population = DiscreteGridRepair().do(problem, Population.new("X", X.copy()))
    assert population.get("X").tolist() == repaired.tolist()


def test_optimize_system_multi_objective_reports_a_missing_pymoo(monkeypatch):
    missing = ImportError("No module named 'pymoo'")
    monkeypatch.setattr("breos.optimization._PYMOO_IMPORT_ERROR", missing)

    with pytest.raises(ImportError, match=r"pip install 'breos\[optimization\]'") as excinfo:
        optimize_system_multi_objective(pd.DataFrame(), pd.DataFrame(), {})
    assert excinfo.value.__cause__ is missing


def test_solar_design_problem_uses_configured_resolution(monkeypatch):
    idx = pd.date_range("2025-01-01 00:00", periods=2, freq="15min", tz="UTC")
    tmy_data = declare_instant_at_labels(pd.DataFrame({"temp_air": [15.0, 16.0], "ghi": [0.0, 0.0]}, index=idx))
    houseload = pd.DataFrame({"Load": [1000.0, 1000.0]}, index=idx)
    dc = pd.Series([0.0, 0.0], index=idx)
    captured: dict = {}
    config = _config(
        simulation={"resolution": "15min", "years_projection": 1},
        constraints=LOOSE,
        battery={"temperature": 20.0, "indoor_model": {"enabled": False}},
    )

    def fake_pv(**kwargs):
        captured["pv_freq"] = kwargs["freq"]
        return dc

    monkeypatch.setattr("breos.optimization.calculate_pv_production_dc", fake_pv)
    _stub_projection_balance(monkeypatch, idx, captured, Houseload=1000.0, Import_From_Grid=600.0, PV_AC_To_Load=400.0)

    problem = SolarDesignProblem(tmy_data, houseload, config)
    out: dict = {}
    problem._evaluate(np.array([2.0, 1.0, 10.0], dtype=float), out)

    assert captured["pv_freq"] == "15min"
    assert captured["freq"] == "15min"
    # Two 15-minute steps at 400 W are 0.2 kWh, not the 0.8 kWh of two hours.
    assert out["Projected_PV_Production_Year1_kWh"] == pytest.approx(0.2)
    assert list(captured["temperature_series"]) == [20.0, 20.0]


def test_solar_design_problem_scores_zeb_from_explicit_ac_ledger(monkeypatch, tiny_inputs):
    idx, _tmy_data, houseload = tiny_inputs  # 1 kWh of load
    tmy_data = declare_instant_at_labels(pd.DataFrame({"temp_air": [15.0, 16.0], "ghi": [500.0, 500.0]}, index=idx))
    dc = pd.Series([1000.0, 1000.0], index=idx)  # 2 kWh raw DC
    config = _config(
        simulation={"resolution": "h", "years_projection": 1}, constraints=LOOSE, battery={"temperature": 20.0}
    )

    monkeypatch.setattr("breos.optimization.calculate_pv_production_dc", lambda **kwargs: dc)
    # 0.85 kWh usable AC; the legacy PV_Production field must not drive ZEB.
    _stub_projection_balance(
        monkeypatch,
        idx,
        Houseload=500.0,
        PV_AC_To_Load=300.0,
        PV_Origin_Battery_AC_To_Load=50.0,
        PV_AC_Export=75.0,
        PV_Production=1000.0,
    )

    problem = SolarDesignProblem(tmy_data, houseload, config)
    out = {}
    problem._evaluate(np.array([2.0, 1.0, 10.0], dtype=float), out)

    assert out["ZEB_Ratio"] == pytest.approx(0.85)
    assert out["Projected_ZEB_Ratio"] == pytest.approx(0.85)
    assert problem.battery_replacement_treatment["method"] == "simulated_yearly_state_propagation"


def test_solar_design_problem_uses_simulated_load_for_objective_denominator(monkeypatch, tiny_inputs):
    idx, tmy_data, _houseload = tiny_inputs
    load_idx = pd.date_range("2025-01-01 00:00", periods=3, freq="h", tz="UTC")
    houseload = pd.DataFrame({"Load": [500.0, 500.0, 500.0]}, index=load_idx)
    config = _config(
        simulation={"resolution": "h", "years_projection": 1}, constraints=LOOSE, battery={"temperature": 20.0}
    )

    monkeypatch.setattr("breos.optimization.calculate_pv_production_dc", lambda **kwargs: pd.Series(0.0, index=idx))
    # The simulation aligned the three-hour input onto two steps: 1 kWh of
    # load, of which 0.5 kWh was imported.
    _stub_projection_balance(monkeypatch, idx, Houseload=500.0, Import_From_Grid=250.0)

    problem = SolarDesignProblem(tmy_data, houseload, config)
    out = {}
    problem._evaluate(np.array([2.0, 0.0, 10.0], dtype=float), out)

    # 0.5 / 1.0 kWh simulated, not 0.5 / 1.5 kWh of raw input.
    assert out["F"][0] == pytest.approx(0.5)
    assert out["Projected_Grid_Independence_%"] == pytest.approx(50.0)


# A small search space for the stubbed four-step NSGA-II runs.
_SMALL_SEARCH = {**LOOSE, "max_modules": 4, "max_battery_kwh": 3.0, "max_tilt_deg": 30.0}


def test_projected_optimization_smoke_reports_two_objective_semantics(monkeypatch):
    idx = pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC")
    tmy_data = declare_instant_at_labels(
        pd.DataFrame(
            {
                "temp_air": [15.0, 16.0, 17.0, 18.0],
                "ghi": [0.0, 500.0, 500.0, 0.0],
            },
            index=idx,
        )
    )
    houseload = pd.DataFrame({"Load": [500.0] * 4}, index=idx)
    dc = pd.Series([0.0, 1000.0, 1000.0, 0.0], index=idx)

    def fake_projected(**kwargs):
        modules = kwargs["n_modules"]
        return {
            "Projected_Grid_Independence_%": 40.0 + modules,
            "Projected_Grid_Independence_Year1_%": 41.0 + modules,
            "Projected_Grid_Independence_FinalYear_%": 39.0 + modules,
            "Projected_Grid_Independence_Mean_%": 40.0 + modules,
            "Projected_Grid_Independence_Min_%": 39.0 + modules,
            "Projected_ZEB_Ratio": 0.8 + modules / 100.0,
            "Projected_ZEB_Ratio_Year1": 0.81 + modules / 100.0,
            "Projected_ZEB_Ratio_FinalYear": 0.79 + modules / 100.0,
            "Projected_ZEB_Ratio_Mean": 0.8 + modules / 100.0,
            "Projected_ZEB_Ratio_Min": 0.79 + modules / 100.0,
            "Projected_NPV": 1000.0 - modules,
            "Projected_Payback_Year": 8.0,
            "Projected_Payback_Year_Interpolated": 7.5,
            "Projected_Initial_Cost": 500.0,
            "Projected_Replacement_Cost_T0_Prices": 0.0,
            "Projected_Total_Replacements": 0,
            "Projected_Final_SOH_%": 90.0,
            "Projected_PV_Production_Year1_kWh": 1.0,
            "Projected_PV_Production_FinalYear_kWh": 0.9,
        }

    monkeypatch.setattr("breos.optimization.calculate_pv_production_dc", lambda **kwargs: dc)
    monkeypatch.setattr("breos.optimization._evaluate_projected_design_metrics", fake_projected)

    config = _config(
        simulation={"resolution": "h", "years_projection": 2},
        optimization={"early_stop": False},
        constraints=_SMALL_SEARCH,
        battery={"temperature": 20.0},
        financials={"project_lifespan": 2},
    )
    result = optimize_system_multi_objective(
        tmy_data,
        houseload,
        config,
        pop_size=4,
        n_gen=1,
        seed=1,
        verbose=False,
    )

    pareto = result.details["pareto"]
    assert result.details["problem"].n_obj == 2
    assert result.details["objective_basis"] == "projected"
    assert result.iterations == 1
    assert not pareto.empty
    assert set(pareto.columns) >= {
        "Modules",
        "Battery_kWh",
        "Tilt",
        "Azimuth",
        "Grid_Independence_%",
        "NPV",
        "ZEB_Ratio",
    }
    assert np.array_equal(pareto["Grid_Independence_%"], pareto["Projected_Grid_Independence_%"])
    assert np.array_equal(pareto["NPV"], pareto["Projected_NPV"])
    assert np.array_equal(pareto["ZEB_Ratio"], pareto["Projected_ZEB_Ratio"])
    assert np.allclose(pareto["Objective_Grid_Independence_%"], pareto["Projected_Grid_Independence_%"])
    assert np.allclose(pareto["Objective_NPV"], pareto["Projected_NPV"])
    assert "Objective_ZEB_Ratio" not in pareto
    assert not [column for column in pareto.columns if column.startswith("SteadyState_")]
    assert result.details["objective_names"] == ["Projected_Grid_Independence_%", "Projected_NPV"]
    assert result.details["battery_replacement_treatment"]["method"] == "simulated_yearly_state_propagation"


def test_projected_optimization_uses_worker_diagnostics_without_parent_rescoring(monkeypatch):
    import os

    import breos.optimization as optimization_module

    idx = pd.date_range("2025-06-21", periods=24, freq="h", tz="UTC")
    ghi = [0, 0, 0, 0, 0, 0, 100, 250, 400, 550, 700, 800, 850, 800, 700, 550, 400, 250, 100, 0, 0, 0, 0, 0]
    tmy_data = declare_instant_at_labels(
        pd.DataFrame(
            {
                "temp_air": [20.0] * len(idx),
                "ghi": ghi,
                "dni": [value * 0.75 for value in ghi],
                "dhi": [value * 0.25 for value in ghi],
                "wind_speed": [2.0] * len(idx),
            },
            index=idx,
        )
    )
    houseload = pd.DataFrame({"Load": [500.0] * len(idx)}, index=idx)
    config = _config(
        simulation={"resolution": "h", "years_projection": 1},
        optimization={"early_stop": False},
        constraints=_SMALL_SEARCH,
        battery={"temperature": 20.0},
        financials={"project_lifespan": 1},
    )
    parent_pid = os.getpid()
    parent_metric_calls = 0
    original_evaluator = optimization_module._evaluate_projected_design_metrics

    def count_parent_evaluations(**kwargs):
        nonlocal parent_metric_calls
        if os.getpid() == parent_pid:
            parent_metric_calls += 1
        return original_evaluator(**kwargs)

    monkeypatch.setattr(optimization_module, "_evaluate_projected_design_metrics", count_parent_evaluations)
    result = optimize_system_multi_objective(
        tmy_data,
        houseload,
        config,
        pop_size=4,
        n_gen=1,
        seed=1,
        verbose=False,
        n_procs=2,
    )

    pareto = result.details["pareto"]
    assert not pareto.empty
    assert {"Projected_NPV", "Projected_Grid_Independence_%"} <= set(pareto.columns)
    np.testing.assert_array_equal(
        pareto["Projected_NPV"].to_numpy(),
        result.details["pymoo_result"].opt.get("Projected_NPV"),
    )
    assert parent_metric_calls == 0


def _projected_capture(monkeypatch, config, x):
    """Score one design on a stubbed projection; return the calls the scorer made."""
    idx = pd.date_range("2025-01-01 00:00", periods=2, freq="h", tz="UTC")
    tmy_data = declare_instant_at_labels(pd.DataFrame({"temp_air": [15.0, 16.0], "ghi": [0.0, 0.0]}, index=idx))
    houseload = pd.DataFrame({"Load": [500.0, 500.0]}, index=idx)
    captured: dict = {"balance": {}}

    monkeypatch.setattr("breos.optimization.calculate_pv_production_dc", lambda **kwargs: pd.Series(0.0, index=idx))
    _stub_projection_balance(monkeypatch, idx, captured["balance"], Houseload=500.0, Import_From_Grid=500.0)
    problem = SolarDesignProblem(tmy_data, houseload, _config(constraints=LOOSE, **config))
    out: dict = {}
    problem._evaluate(np.array(x, dtype=float), out)
    captured["out"] = out
    return captured


def test_projected_scoring_uses_the_projected_horizon_and_pv_degradation(monkeypatch):
    # simulation.years_projection and pv.degradation_rate win over the
    # financials pair.
    seen: dict = {}

    def fake_projected(**kwargs):
        seen.update(kwargs)
        return _PROJECTED_STUB

    monkeypatch.setattr("breos.optimization._evaluate_projected_design_metrics", fake_projected)
    _projected_capture(
        monkeypatch,
        {
            "simulation": {"resolution": "h", "years_projection": 3},
            "pv": {"degradation_rate": 0.10},
            "financials": {"project_lifespan": 2, "pv_degradation_rate": 0.02, "discount_rate": 0.04},
        },
        [2.0, 0.0, 10.0],
    )

    assert seen["years_projection"] == 3
    assert seen["degradation_rate"] == pytest.approx(0.10)
    assert seen["fin_cfg"]["discount_rate"] == pytest.approx(0.04)


@pytest.mark.parametrize(("battery_kwh", "engine", "model"), [(5.0, "blast", "nmc_gr_50ah_b1"), (0.0, "native", None)])
def test_projected_simulation_uses_the_configured_degradation_engine(monkeypatch, battery_kwh, engine, model):
    captured = _projected_capture(
        monkeypatch,
        {
            "simulation": {"resolution": "h", "years_projection": 1},
            "battery": {"degradation_engine": "blast", "blast_model": "nmc_gr_50ah_b1", "temperature": 20.0},
        },
        [2.0, battery_kwh, 10.0],
    )

    # A PV-only candidate has no pack for BLAST to age.
    assert captured["balance"]["degradation_engine"] == engine
    assert captured["balance"]["blast_model"] == model


@pytest.mark.parametrize(
    ("battery", "match"),
    [
        ({"degradation_engine": "blast", "blast_model": "nope"}, "Unknown battery.blast_model 'nope'"),
        ({"degradation_engine": "blast"}, "Unknown battery.blast_model None"),
        ({"blast_model": "nmc_gr_50ah_b1"}, "requires battery.degradation_engine = 'blast'"),
        ({"degradation_engine": "physics"}, "must be one of: native, blast"),
    ],
)
def test_invalid_degradation_settings_raise_when_the_problem_is_built(battery, match, tiny_inputs):
    _idx, tmy_data, houseload = tiny_inputs

    with pytest.raises(ValueError, match=match):
        SolarDesignProblem(tmy_data, houseload, {"location": LOCATION, "battery": battery})


def test_projected_blast_scores_a_pv_only_candidate(synthetic_weather):
    from breos.load_profiles import load_profile

    load = load_profile("demandlib_h0", 3500, start_date="2023-01-01", timezone="UTC")
    config = _config(
        constraints=LOOSE,
        simulation={"resolution": "h", "years_projection": 1},
        battery={"degradation_engine": "blast", "blast_model": "nmc_gr_50ah_b1"},
    )
    problem = SolarDesignProblem(synthetic_weather, load, config)
    out: dict = {}
    # This used to raise "degradation_engine='blast' requires a configured battery".
    problem._evaluate(np.array([6.0, 0.0, 35.0], dtype=float), out)

    assert np.isfinite(out["Projected_NPV"])
    assert 0.0 < out["Projected_Grid_Independence_%"] < 100.0


def test_early_stopping_optimizer_result_pickles(monkeypatch):
    # The minimum-generation wrapper was a local class, so a result run with
    # early stopping could not be pickled (#217).
    import pickle

    idx = pd.date_range("2025-01-01 00:00", periods=4, freq="h", tz="UTC")
    monkeypatch.setattr(
        "breos.optimization.calculate_pv_production_dc", lambda **kwargs: pd.Series([0.0, 1e3, 1e3, 0.0], index=idx)
    )
    monkeypatch.setattr("breos.optimization._evaluate_projected_design_metrics", lambda **kwargs: _PROJECTED_STUB)
    config = _config(
        simulation={"resolution": "h"},
        constraints={"budget": 1e5, "max_area_m2": 100.0, "max_modules": 4, "max_battery_kwh": 3.0},
        optimization={"early_stop": {"min_gen": 1}},
        battery={"temperature": 20.0},
    )
    result = optimize_system_multi_objective(
        declare_instant_at_labels(pd.DataFrame({"temp_air": 15.0, "ghi": 0.0}, index=idx)),
        pd.DataFrame({"Load": 500.0}, index=idx),
        config,
        pop_size=4,
        n_gen=2,
        seed=1,
        verbose=False,
    )

    restored = pickle.loads(pickle.dumps(result.details["pymoo_result"]))
    assert restored.X.shape == result.details["pymoo_result"].X.shape


@pytest.mark.parametrize("n_procs", [0, -1, True])
def test_optimizer_needs_a_positive_worker_count(n_procs):
    from tests.conftest import _build_synthetic_weather

    weather = _build_synthetic_weather(2023)
    load = pd.DataFrame({"Load": 400.0}, index=weather.index)
    config = {
        "location": LOCATION,
        "constraints": {**LOOSE, "max_modules": 8, "max_battery_kwh": 5.0},
        "simulation": {"resolution": "h", "years_projection": 1},
    }

    with pytest.raises(ValueError, match="n_procs must be a positive integer"):
        optimize_system_multi_objective(weather, load, config, pop_size=4, n_gen=1, n_procs=n_procs)


def _stubbed_search(monkeypatch, *, cached: bool):
    """Run a seeded search on a stubbed projection, with or without the score cache.

    Returns the result, the design batches the search asked for, and the
    number of projections it ran.
    """
    from pymoo.core.problem import ElementwiseProblem

    idx = pd.date_range("2025-01-01 00:00", periods=4, freq="h", tz="UTC")
    dc = pd.Series([0.0, 1000.0, 1000.0, 0.0], index=idx)
    projections = []

    def fake_projected(**kwargs):
        projections.append(kwargs["n_modules"])
        modules, battery = kwargs["n_modules"], kwargs["battery_kwh"]
        yield_kwh = float(kwargs["base_dc_power"].sum()) / 1000.0
        return {
            **_PROJECTED_STUB,
            "Projected_Grid_Independence_%": min(95.0, 6.0 * modules * yield_kwh + 4.0 * battery),
            "Projected_ZEB_Ratio": 0.3 * modules * yield_kwh,
            "Projected_NPV": 900.0 * modules * yield_kwh - 700.0 * battery - 40.0 * modules**2,
            "Projected_Initial_Cost": 400.0 * modules + 600.0 * battery,
        }

    # Tilt changes the yield, so every gene moves the objectives.
    monkeypatch.setattr(
        "breos.optimization.calculate_pv_production_dc", lambda **kwargs: dc * (1.0 + kwargs["tilt"] / 100.0)
    )
    monkeypatch.setattr("breos.optimization._evaluate_projected_design_metrics", fake_projected)
    implementation = SolarDesignProblem._evaluate_elementwise if cached else ElementwiseProblem._evaluate_elementwise
    batches = []

    def recording(problem, X, out, *args, **kwargs):
        batches.append(np.array(X, copy=True))
        implementation(problem, X, out, *args, **kwargs)

    monkeypatch.setattr(SolarDesignProblem, "_evaluate_elementwise", recording)
    config = {
        "location": {"latitude": 41.15, "longitude": -8.61, "timezone": "UTC"},
        "simulation": {"resolution": "h"},
        # The budget and the ZEB constraint leave part of the space infeasible.
        "constraints": {
            "budget": 3000.0,
            "max_area_m2": 100.0,
            "max_modules": 5,
            "max_battery_kwh": 3.0,
            "min_tilt_deg": 10.0,
            "max_tilt_deg": 30.0,
            "enforce_zeb": True,
        },
        "mode": {"fixed_azimuth": 180},
        "battery": {"temperature": 20.0},
    }
    result = optimize_system_multi_objective(
        declare_instant_at_labels(pd.DataFrame({"temp_air": 15.0, "ghi": 0.0}, index=idx)),
        pd.DataFrame({"Load": 500.0}, index=idx),
        config,
        pop_size=8,
        n_gen=12,
        seed=3,
    )
    return result, batches, len(projections)


def test_score_cache_keeps_the_search_and_front_of_an_uncached_search(monkeypatch):
    # The cache only skips repeat simulations: the same seed asks for the same
    # designs, generation by generation, and ends on the same front.
    cached, cached_batches, cached_projections = _stubbed_search(monkeypatch, cached=True)
    monkeypatch.undo()
    uncached, uncached_batches, uncached_projections = _stubbed_search(monkeypatch, cached=False)

    assert len(cached_batches) == len(uncached_batches)
    for cached_batch, uncached_batch in zip(cached_batches, uncached_batches):
        np.testing.assert_array_equal(cached_batch, uncached_batch)
    cached_result, uncached_result = cached.details["pymoo_result"], uncached.details["pymoo_result"]
    for name in ("X", "F", "G"):
        np.testing.assert_array_equal(cached_result.pop.get(name), uncached_result.pop.get(name))
        np.testing.assert_array_equal(cached_result.opt.get(name), uncached_result.opt.get(name))
    pd.testing.assert_frame_equal(cached.details["pareto"], uncached.details["pareto"])

    # Without the cache every evaluation is a simulation; with it, every
    # distinct design is simulated once.
    n_eval = cached_result.algorithm.evaluator.n_eval
    distinct = {tuple(row) for row in np.vstack(cached_batches)}
    assert uncached_projections == n_eval == uncached_result.algorithm.evaluator.n_eval
    assert cached_projections == len(distinct) < n_eval
    assert cached.details["evaluation_counts"] == {
        "evaluations": n_eval,
        "simulations": len(distinct),
        "distinct_designs": len(distinct),
    }


def test_evaluated_designs_hold_every_scored_design_and_its_constraints(monkeypatch):
    import pickle

    result, batches, _ = _stubbed_search(monkeypatch, cached=True)
    evaluated = result.details["evaluated"]
    pareto = result.details["pareto"]
    problem = result.details["problem"]

    # One row per distinct design, in the order first asked for.
    designs = list(dict.fromkeys(tuple(row) for row in np.vstack(batches)))
    assert len(evaluated) == len(designs)
    np.testing.assert_array_equal(evaluated[["Modules", "Battery_kWh", "Tilt"]].to_numpy(dtype=float), designs)
    assert evaluated["Evaluations"].sum() == result.details["evaluation_counts"]["evaluations"]
    assert evaluated["Evaluations"].max() > 1

    # The Pareto columns, then the constraints the search checked.
    assert list(evaluated.columns) == [
        *pareto.columns,
        "Constraint_Budget",
        "Constraint_Area",
        "Constraint_ZEB",
        "Feasible",
        "Evaluations",
    ]
    np.testing.assert_allclose(evaluated["Constraint_Budget"], evaluated["Projected_Initial_Cost"] - 3000.0)
    np.testing.assert_allclose(evaluated["Constraint_ZEB"], 1.0 - evaluated["Projected_ZEB_Ratio"])
    assert evaluated["Feasible"].any() and not evaluated["Feasible"].all()

    # Every front design is an evaluated, feasible row with the same values.
    front = pareto.merge(evaluated, on=["Modules", "Battery_kWh", "Tilt"], suffixes=("", "_evaluated"))
    assert len(front) == len(pareto)
    assert front["Feasible"].all()
    np.testing.assert_array_equal(front["NPV"], front["NPV_evaluated"])

    # Worker tasks pickle the problem without the growing record.
    assert pickle.loads(pickle.dumps(problem)).scored == {}


def test_parallel_search_scores_and_reports_the_designs_of_a_serial_search(synthetic_weather):
    from breos.load_profiles import load_profile

    load = load_profile("demandlib_h0", 3500, start_date="2023-01-01", timezone="UTC")
    config = {
        "location": {"latitude": 41.15, "longitude": -8.61, "timezone": "UTC"},
        "constraints": {"budget": 8000.0, "max_area_m2": 100.0, "max_modules": 6, "max_battery_kwh": 2.0},
        "simulation": {"resolution": "h", "years_projection": 1},
        "mode": {"fixed_azimuth": 180},
    }
    run = {"pop_size": 6, "n_gen": 4, "seed": 2}

    serial = optimize_system_multi_objective(synthetic_weather, load, config, **run)
    parallel = optimize_system_multi_objective(synthetic_weather, load, config, n_procs=2, **run)

    pd.testing.assert_frame_equal(serial.details["evaluated"], parallel.details["evaluated"])
    pd.testing.assert_frame_equal(serial.details["pareto"], parallel.details["pareto"])
    assert serial.details["evaluation_counts"] == parallel.details["evaluation_counts"]

    # The public Pareto columns, in their order and with their types, written
    # out rather than built by the code under test (as on 567df2a).
    pareto = serial.details["pareto"]
    assert list(pareto.columns) == [
        "Modules",
        "Battery_kWh",
        "Tilt",
        "Azimuth",
        "Layout",
        "Grid_Independence_%",
        "NPV",
        "Objective_Grid_Independence_%",
        "Objective_NPV",
        "Projected_Final_SOH_%",
        "Projected_First_End_Of_Life_Action",
        "Projected_First_End_Of_Life_Reason",
        "Projected_First_End_Of_Life_SOH_%",
        "Projected_First_End_Of_Life_Years",
        "Projected_Grid_Independence_%",
        "Projected_Grid_Independence_FinalYear_%",
        "Projected_Grid_Independence_Mean_%",
        "Projected_Grid_Independence_Min_%",
        "Projected_Grid_Independence_Year1_%",
        "Projected_Initial_Cost",
        "Projected_Inverter_Loss_Year1_kWh",
        "Projected_LCOE_per_kWh",
        "Projected_NPV",
        "Projected_PV_DC_Curtailed_Year1_kWh",
        "Projected_PV_DC_FinalYear_kWh",
        "Projected_PV_DC_Year1_kWh",
        "Projected_PV_Production_FinalYear_kWh",
        "Projected_PV_Production_Year1_kWh",
        "Projected_Payback_Year",
        "Projected_Payback_Year_Interpolated",
        "Projected_Replacement_Cost_T0_Prices",
        "Projected_Total_Replacements",
        "Projected_ZEB_Ratio",
        "Projected_ZEB_Ratio_FinalYear",
        "Projected_ZEB_Ratio_Mean",
        "Projected_ZEB_Ratio_Min",
        "Projected_ZEB_Ratio_Year1",
        "ZEB_Ratio",
    ]
    assert pareto["Modules"].dtype == np.int64 and pareto["Azimuth"].dtype == np.int64
    assert (pareto["Grid_Independence_%"] == pareto["Projected_Grid_Independence_%"]).all()
    assert (pareto["NPV"] == pareto["Projected_NPV"]).all()
    assert (pareto["ZEB_Ratio"] == pareto["Projected_ZEB_Ratio"]).all()


def test_evaluated_designs_record_the_currency_of_the_search(monkeypatch):
    # Like the front, the evaluated table names its currency, so a plot of it
    # labels money in it and refuses another.
    from breos.plotting import plot_pareto_front

    result, _, _ = _stubbed_search(monkeypatch, cached=True)
    evaluated = result.details["evaluated"]

    assert evaluated.attrs["currency"] == result.details["provenance"]["currency"] == "EUR"
    with pytest.raises(ValueError, match="records its currency as 'EUR'"):
        plot_pareto_front(evaluated, currency="USD")
