"""Tests for optimization guardrails."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pymoo")

from breos.optimization import SolarDesignProblem, optimize_system_multi_objective


def test_projected_objective_basis_uses_two_objectives_and_optional_zeb_constraint():
    idx = pd.date_range("2025-01-01 00:00", periods=2, freq="h", tz="UTC")
    tmy_data = pd.DataFrame({"temp_air": [15.0, 16.0], "ghi": [0.0, 0.0]}, index=idx)
    houseload = pd.DataFrame({"Load": [500.0, 500.0]}, index=idx)
    config = {
        "location": {"latitude": 41.15, "longitude": -8.61, "timezone": "UTC"},
        "optimization": {"objective_basis": "projected"},
        "constraints": {"enforce_zeb": True},
        "mode": {"fixed_azimuth": 180},
    }

    problem = SolarDesignProblem(tmy_data, houseload, config, "results/_test_run/problem_projected")

    assert problem.n_obj == 2
    assert problem.n_ieq_constr == 3


def test_projected_objective_basis_rejects_unknown_value():
    idx = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
    tmy_data = pd.DataFrame({"temp_air": [15.0, 16.0], "ghi": [0.0, 0.0]}, index=idx)
    houseload = pd.DataFrame({"Load": [500.0, 500.0]}, index=idx)

    with pytest.raises(ValueError, match="objective_basis"):
        SolarDesignProblem(
            tmy_data,
            houseload,
            {
                "location": {"latitude": 41.15, "longitude": -8.61, "timezone": "UTC"},
                "optimization": {"objective_basis": "lifetime-ish"},
            },
            "results/_test_run/problem_invalid_basis",
        )


def test_projected_zeb_constraint_uses_projected_diagnostic(monkeypatch):
    idx = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
    tmy_data = pd.DataFrame({"temp_air": [15.0, 16.0], "ghi": [0.0, 0.0]}, index=idx)
    houseload = pd.DataFrame({"Load": [500.0, 500.0]}, index=idx)
    results = pd.DataFrame(
        {
            "Houseload": [500.0, 500.0],
            "PV_AC_To_Load": [0.0, 0.0],
            "Battery_AC_To_Load_PV": [0.0, 0.0],
            "PV_AC_Export": [0.0, 0.0],
        },
        index=idx,
    )
    summary = pd.DataFrame({"Import [kWh]": [1.0], "Sell [kWh]": [0.0]})
    projected = {
        "Projected_Grid_Independence_%": 50.0,
        "Projected_ZEB_Ratio": 0.8,
        "Projected_NPV_Eur": 1000.0,
    }
    monkeypatch.setattr("breos.optimization.calculate_pv_production_dc", lambda **kwargs: pd.Series(0.0, index=idx))
    monkeypatch.setattr(
        "breos.optimization.simulate_energy_balance",
        lambda **kwargs: (results, 0.0, summary, 0.0, 0, pd.DataFrame()),
    )
    monkeypatch.setattr("breos.optimization.calculate_financials", lambda *args, **kwargs: (0.0, 0.0))
    monkeypatch.setattr("breos.optimization._evaluate_projected_design_metrics", lambda **kwargs: projected)
    config = {
        "location": {"latitude": 41.15, "longitude": -8.61, "timezone": "UTC"},
        "optimization": {"objective_basis": "projected"},
        "constraints": {"budget_eur": 100000.0, "max_area_m2": 100.0, "enforce_zeb": True},
        "simulation": {"resolution": "h", "years_projection": 2},
        "financials": {"project_lifespan": 2},
        "mode": {"fixed_azimuth": 180},
    }
    out = {}
    problem = SolarDesignProblem(tmy_data, houseload, config, "results/_test_run/problem_projected_zeb")
    problem._evaluate(np.array([2.0, 0.0, 10.0]), out)

    assert out["F"] == pytest.approx([0.5, -1000.0])
    assert out["ZEB_Ratio"] == pytest.approx(0.8)
    assert out["G"][2] == pytest.approx(0.2)


def test_solar_design_problem_area_constraint_uses_pv_dimensions(monkeypatch):
    idx = pd.date_range("2025-01-01 00:00", periods=2, freq="h", tz="UTC")
    tmy_data = pd.DataFrame({"temp_air": [15.0, 16.0], "ghi": [0.0, 0.0]}, index=idx)
    houseload = pd.DataFrame({"Load": [500.0, 500.0]}, index=idx)
    dc = pd.Series([0.0, 0.0], index=idx)
    summary = pd.DataFrame({"Import [kWh]": [1.0], "Sell [kWh]": [0.0]})
    config = {
        "location": {"latitude": 41.15, "longitude": -8.61, "timezone": "UTC"},
        "simulation": {"resolution": "h"},
        "constraints": {"budget_eur": 100000, "max_area_m2": 10.0, "max_modules": 5},
        "mode": {"fixed_azimuth": 180},
        "pv": {
            "module": "Suntech_STP550S_STC",
            "dimensions": {"width": 2.0, "length": 3.0},
        },
        "battery": {"max_soc": 0.9, "min_soc": 0.1},
    }

    monkeypatch.setattr("breos.optimization.calculate_pv_production_dc", lambda **kwargs: dc)
    monkeypatch.setattr(
        "breos.optimization.simulate_energy_balance",
        lambda **kwargs: (pd.DataFrame(), 0.0, summary, 0.0, 0, pd.DataFrame()),
    )
    monkeypatch.setattr("breos.optimization.calculate_financials", lambda *args, **kwargs: (0.0, 0.0))

    problem = SolarDesignProblem(tmy_data, houseload, config, "results/_test_run/problem_area")
    out: dict = {}
    problem._evaluate(np.array([2.0, 0.0, 10.0], dtype=float), out)

    assert out["G"][1] == pytest.approx(2.0)


def test_solar_design_problem_honors_module_and_tilt_bounds():
    idx = pd.date_range("2025-01-01 00:00", periods=2, freq="h", tz="UTC")
    tmy_data = pd.DataFrame({"temp_air": [15.0, 16.0], "ghi": [0.0, 0.0]}, index=idx)
    houseload = pd.DataFrame({"Load": [500.0, 500.0]}, index=idx)
    config = {
        "location": {"latitude": 41.15, "longitude": -8.61, "timezone": "UTC"},
        "constraints": {"max_modules": 5, "max_battery_kwh": 7.0, "max_tilt_deg": 45.0},
        "mode": {"fixed_azimuth": 180},
    }

    problem = SolarDesignProblem(tmy_data, houseload, config, "results/_test_run/problem_bounds")

    assert problem.xu[0] == pytest.approx(5.0)
    assert problem.xu[1] == pytest.approx(7.0)
    assert problem.xu[2] == pytest.approx(45.0)


def test_solar_design_problem_uses_configured_resolution(monkeypatch):
    idx = pd.date_range("2025-01-01 00:00", periods=2, freq="15min", tz="UTC")
    tmy_data = pd.DataFrame({"temp_air": [15.0, 16.0], "ghi": [0.0, 0.0]}, index=idx)
    houseload = pd.DataFrame({"Load": [1000.0, 1000.0]}, index=idx)
    dc = pd.Series([0.0, 0.0], index=idx)
    summary = pd.DataFrame({"Import [kWh]": [0.5], "Sell [kWh]": [0.0]})
    captured: dict = {}
    config = {
        "location": {"latitude": 41.15, "longitude": -8.61, "timezone": "UTC"},
        "simulation": {"resolution": "15min"},
        "constraints": {"budget_eur": 100000, "max_area_m2": 100.0},
        "mode": {"fixed_azimuth": 180},
        "battery": {"temperature": 20.0, "indoor_model": {"enabled": False}},
    }

    def fake_pv(**kwargs):
        captured["pv_freq"] = kwargs["freq"]
        return dc

    def fake_balance(**kwargs):
        captured["balance_freq"] = kwargs["freq"]
        captured["temperature_series"] = kwargs["temperature_series"]
        return pd.DataFrame(), 0.0, summary, 0.0, 0, pd.DataFrame()

    def fake_financials(*args, **kwargs):
        captured["annual_load_kwh"] = args[4]
        return 0.0, 0.0

    monkeypatch.setattr("breos.optimization.calculate_pv_production_dc", fake_pv)
    monkeypatch.setattr("breos.optimization.simulate_energy_balance", fake_balance)
    monkeypatch.setattr("breos.optimization.calculate_financials", fake_financials)

    problem = SolarDesignProblem(tmy_data, houseload, config, "results/_test_run/problem_resolution")
    out: dict = {}
    problem._evaluate(np.array([2.0, 0.0, 10.0], dtype=float), out)

    assert captured["pv_freq"] == "15min"
    assert captured["balance_freq"] == "15min"
    assert captured["annual_load_kwh"] == pytest.approx(0.5)
    assert list(captured["temperature_series"]) == [20.0, 20.0]


def test_solar_design_problem_scores_zeb_from_explicit_ac_ledger(monkeypatch):
    idx = pd.date_range("2025-01-01 00:00", periods=2, freq="h", tz="UTC")
    tmy_data = pd.DataFrame({"temp_air": [15.0, 16.0], "ghi": [500.0, 500.0]}, index=idx)
    houseload = pd.DataFrame({"Load": [500.0, 500.0]}, index=idx)  # 1 kWh
    dc = pd.Series([1000.0, 1000.0], index=idx)  # 2 kWh raw DC
    results = pd.DataFrame(
        {
            "PV_AC_To_Load": [300.0, 300.0],
            "Battery_AC_To_Load_PV": [50.0, 50.0],
            "PV_AC_Export": [75.0, 75.0],
            "PV_Production": [1000.0, 1000.0],
        },
        index=idx,
    )  # 0.85 kWh usable AC
    summary = pd.DataFrame({"Import [kWh]": [0.3], "Sell [kWh]": [0.15], "Final SOH [%]": [85.0]})
    captured = {}
    config = {
        "location": {"latitude": 41.15, "longitude": -8.61, "timezone": "UTC"},
        "simulation": {"resolution": "h"},
        "constraints": {"budget_eur": 100000, "max_area_m2": 100.0},
        "mode": {"fixed_azimuth": 180},
        "battery": {"temperature": 20.0},
    }

    monkeypatch.setattr("breos.optimization.calculate_pv_production_dc", lambda **kwargs: dc)
    monkeypatch.setattr(
        "breos.optimization.simulate_energy_balance",
        lambda **kwargs: (results, 2000.0, summary, 0.0, 0, pd.DataFrame()),
    )

    def fake_financials(*args, **kwargs):
        captured.update(kwargs)
        return 0.0, 0.0

    monkeypatch.setattr("breos.optimization.calculate_financials", fake_financials)

    problem = SolarDesignProblem(tmy_data, houseload, config, "results/_test_run/problem_zeb_ac")
    out = {}
    problem._evaluate(np.array([2.0, 1.0, 10.0], dtype=float), out)

    assert -out["F"][2] == pytest.approx(0.85)
    assert captured["annual_pv_kwh"] == pytest.approx(0.85)
    assert captured["annual_battery_soh_loss_pct"] == pytest.approx(15.0)
    assert problem.battery_replacement_treatment["method"] == "repeat_simulated_year_1_soh_loss_to_eol"


def test_solar_design_problem_uses_simulated_load_for_objective_denominator(monkeypatch):
    idx = pd.date_range("2025-01-01 00:00", periods=2, freq="h", tz="UTC")
    load_idx = pd.date_range("2025-01-01 00:00", periods=3, freq="h", tz="UTC")
    tmy_data = pd.DataFrame({"temp_air": [15.0, 16.0], "ghi": [0.0, 0.0]}, index=idx)
    houseload = pd.DataFrame({"Load": [500.0, 500.0, 500.0]}, index=load_idx)
    dc = pd.Series([0.0, 0.0], index=idx)
    results = pd.DataFrame(
        {
            "Houseload": [500.0, 500.0],
            "PV_AC_To_Load": [0.0, 0.0],
            "Battery_AC_To_Load_PV": [0.0, 0.0],
            "PV_AC_Export": [0.0, 0.0],
        },
        index=idx,
    )
    summary = pd.DataFrame({"Import [kWh]": [1.0], "Sell [kWh]": [0.0]})
    captured = {}
    config = {
        "location": {"latitude": 41.15, "longitude": -8.61, "timezone": "UTC"},
        "simulation": {"resolution": "h"},
        "constraints": {"budget_eur": 100000, "max_area_m2": 100.0},
        "mode": {"fixed_azimuth": 180},
        "battery": {"temperature": 20.0},
    }

    monkeypatch.setattr("breos.optimization.calculate_pv_production_dc", lambda **kwargs: dc)
    monkeypatch.setattr(
        "breos.optimization.simulate_energy_balance",
        lambda **kwargs: (results, 0.0, summary, 0.0, 0, pd.DataFrame()),
    )

    def fake_financials(*args, **kwargs):
        captured["annual_load_kwh"] = args[4]
        return 0.0, 0.0

    monkeypatch.setattr("breos.optimization.calculate_financials", fake_financials)

    problem = SolarDesignProblem(tmy_data, houseload, config, "results/_test_run/problem_aligned_load")
    out = {}
    problem._evaluate(np.array([2.0, 0.0, 10.0], dtype=float), out)

    assert captured["annual_load_kwh"] == pytest.approx(1.0)
    assert out["F"][0] == pytest.approx(1.0)


def test_optimize_system_multi_objective_returns_pareto_dataframe(monkeypatch):
    idx = pd.date_range("2025-01-01 00:00", periods=4, freq="h", tz="UTC")
    tmy_data = pd.DataFrame({"temp_air": [15.0, 16.0, 17.0, 18.0], "ghi": [0.0, 500.0, 500.0, 0.0]}, index=idx)
    houseload = pd.DataFrame({"Load": [500.0, 500.0, 500.0, 500.0]}, index=idx)
    dc = pd.Series([0.0, 1000.0, 1000.0, 0.0], index=idx)
    summary = pd.DataFrame({"Import [kWh]": [1.0], "Sell [kWh]": [0.25]})
    config = {
        "location": {"latitude": 41.15, "longitude": -8.61, "timezone": "UTC"},
        "simulation": {"resolution": "h"},
        "constraints": {
            "budget_eur": 100000.0,
            "max_area_m2": 100.0,
            "max_modules": 4,
            "max_battery_kwh": 3.0,
            "max_tilt_deg": 30.0,
        },
        "mode": {"fixed_azimuth": 180},
        "battery": {"temperature": 20.0},
    }

    monkeypatch.setattr("breos.optimization.calculate_pv_production_dc", lambda **kwargs: dc)
    monkeypatch.setattr(
        "breos.optimization.simulate_energy_balance",
        lambda **kwargs: (pd.DataFrame(), 0.0, summary, 0.0, 0, pd.DataFrame()),
    )
    monkeypatch.setattr("breos.optimization.calculate_financials", lambda *args, **kwargs: (1000.0, 2500.0))

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
    assert result.iterations >= 1
    assert not pareto.empty
    assert set(pareto.columns) >= {
        "Modules",
        "Battery_kWh",
        "Tilt",
        "Azimuth",
        "Grid_Independence_%",
        "NPV_Eur",
        "ZEB_Ratio",
    }
    assert result.details["battery_replacement_treatment"]["method"] == ("repeat_simulated_year_1_soh_loss_to_eol")


def test_projected_optimization_smoke_reports_two_objective_semantics(monkeypatch):
    idx = pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC")
    tmy_data = pd.DataFrame(
        {
            "temp_air": [15.0, 16.0, 17.0, 18.0],
            "ghi": [0.0, 500.0, 500.0, 0.0],
        },
        index=idx,
    )
    houseload = pd.DataFrame({"Load": [500.0] * 4}, index=idx)
    dc = pd.Series([0.0, 1000.0, 1000.0, 0.0], index=idx)
    results = pd.DataFrame(
        {
            "Houseload": [500.0] * 4,
            "PV_AC_To_Load": [0.0, 400.0, 400.0, 0.0],
            "Battery_AC_To_Load_PV": [0.0] * 4,
            "PV_AC_Export": [0.0, 100.0, 100.0, 0.0],
        },
        index=idx,
    )
    summary = pd.DataFrame({"Import [kWh]": [1.2], "Sell [kWh]": [0.2]})

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
            "Projected_NPV_Eur": 1000.0 - modules,
            "Projected_Breakeven_Year": 8.0,
            "Projected_Breakeven_Year_Exact": 7.5,
            "Projected_Initial_Cost_Eur": 500.0,
            "Projected_Replacement_Cost_Eur": 0.0,
            "Projected_Total_Replacements": 0,
            "Projected_Final_SOH_%": 90.0,
            "Projected_PV_Production_Year1_kWh": 1.0,
            "Projected_PV_Production_FinalYear_kWh": 0.9,
        }

    monkeypatch.setattr("breos.optimization.calculate_pv_production_dc", lambda **kwargs: dc)
    monkeypatch.setattr(
        "breos.optimization.simulate_energy_balance",
        lambda **kwargs: (results, 1000.0, summary, 0.0, 0, pd.DataFrame()),
    )
    monkeypatch.setattr("breos.optimization.calculate_financials", lambda *args, **kwargs: (500.0, 750.0))
    monkeypatch.setattr("breos.optimization._evaluate_projected_design_metrics", fake_projected)

    config = {
        "location": {"latitude": 41.15, "longitude": -8.61, "timezone": "UTC"},
        "simulation": {"resolution": "h", "years_projection": 2},
        "optimization": {"objective_basis": "projected", "early_stop": False},
        "constraints": {
            "budget_eur": 100000.0,
            "max_area_m2": 100.0,
            "max_modules": 4,
            "max_battery_kwh": 3.0,
            "max_tilt_deg": 30.0,
        },
        "mode": {"fixed_azimuth": 180},
        "battery": {"temperature": 20.0},
        "financials": {"project_lifespan": 2},
    }
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
    assert np.array_equal(pareto["Grid_Independence_%"], pareto["Projected_Grid_Independence_%"])
    assert np.array_equal(pareto["NPV_Eur"], pareto["Projected_NPV_Eur"])
    assert np.array_equal(pareto["ZEB_Ratio"], pareto["Projected_ZEB_Ratio"])
    assert np.allclose(pareto["Objective_Grid_Independence_%"], pareto["Projected_Grid_Independence_%"])
    assert np.allclose(pareto["Objective_NPV_Eur"], pareto["Projected_NPV_Eur"])
    assert "Objective_ZEB_Ratio" not in pareto
    assert result.details["objective_names"] == ["Projected_Grid_Independence_%", "Projected_NPV_Eur"]
