"""Tariff prices reach fixed-design evaluation and optimizer scoring."""

import pickle
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from breos import optimization

TARIFF = {
    "schedule": "pt_mainland_2026_daily_bi",
    "currency": "EUR",
    "import_prices": {"peak": 0.50, "off_peak": 0.10},
    "export_prices": {"all": 0.03},
    "fixed_charge_per_day": 0.40,
}
FIXED_TARGET = {
    "mode": "fixed_target",
    "target_usable_fraction": 0.5,
    "charge_periods": ["off_peak"],
    "discharge_periods": ["peak"],
    "grid_charge_efficiency": 0.95,
}


@pytest.fixture
def tariff_case(monkeypatch):
    index = pd.date_range("2026-01-05", periods=48, freq="h", tz="Europe/Lisbon")
    weather = pd.DataFrame({"temp_air": 20.0}, index=index)
    load = pd.DataFrame({"Load": 1000.0}, index=index)
    pv = pd.Series(np.where((index.hour >= 10) & (index.hour < 16), 1800.0, 0.0), index=index)
    monkeypatch.setattr(optimization, "calculate_pv_production_dc", lambda **kwargs: pv.copy())
    config = {
        "location": {"latitude": 41.15, "longitude": -8.61, "timezone": "Europe/Lisbon", "altitude": 100},
        "simulation": {"resolution": "h", "years_projection": 2},
        "battery": {"temperature": 20.0},
        "mode": {"fixed_azimuth": 180.0},
        "constraints": {"budget": 100000, "max_area_m2": 100, "max_tilt_deg": 60},
        "tariff": deepcopy(TARIFF),
    }
    return weather, load, config


def evaluate(case, battery_kwh=0.0):
    weather, load, config = case
    return optimization.evaluate_projected_design(
        weather, load, config, n_modules=4, battery_kwh=battery_kwh, tilt=30.0, azimuth=180.0
    )


def test_optimizer_provenance_records_the_schema_version_and_currency(tariff_case):
    weather, load, config = tariff_case
    flat = deepcopy(config)
    del flat["tariff"]

    for case in (tariff_case, (weather, load, flat)):
        provenance = evaluate(case).provenance
        assert provenance["result_schema_version"] == "1.1"
        assert provenance["currency"] == "EUR"
    assert "tariff" not in evaluate((weather, load, flat)).provenance


def test_tariff_changes_optimizer_money_without_changing_energy(tariff_case):
    weather, load, config = tariff_case
    first = evaluate(tariff_case)
    expensive = deepcopy(config)
    expensive["tariff"]["import_prices"] = {"all": 2.0}
    second = evaluate((weather, load, expensive))

    assert second.metrics["Projected_NPV"] > first.metrics["Projected_NPV"]
    pd.testing.assert_frame_equal(
        first.yearly[["Import_kWh", "Export_kWh", "PV_Production_kWh"]],
        second.yearly[["Import_kWh", "Export_kWh", "PV_Production_kWh"]],
    )


@pytest.mark.parametrize("entrypoint", ["fixed", "problem"])
def test_optimizer_rejects_invalid_tariff_before_pv(tariff_case, monkeypatch, entrypoint):
    weather, load, config = tariff_case
    config["tariff"]["import_prices"]["peak"] = -1
    monkeypatch.setattr(
        optimization, "calculate_pv_production_dc", lambda **kwargs: pytest.fail("PV ran before tariff validation")
    )
    with pytest.raises(ValueError, match="tariff.*peak"):
        if entrypoint == "fixed":
            evaluate(tariff_case)
        else:
            pytest.importorskip("pymoo")
            optimization.SolarDesignProblem(weather, load, config, "unused")


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"costs": {"electricity_cost": 0.2}}, "price them twice"),
        ({"tariff": {**TARIFF, "schedule": "pt_mainland_2027_daily_bi"}}, "15min"),
        ({"smart_charging": {**FIXED_TARGET, "charge_periods": ["night"]}}, r"smart_charging\.charge_periods.*night"),
    ],
)
def test_optimizer_checks_conflicts_and_unsupported_dispatch(tariff_case, monkeypatch, extra, message):
    weather, load, config = tariff_case
    config.update(extra)
    monkeypatch.setattr(
        optimization, "calculate_pv_production_dc", lambda **kwargs: pytest.fail("PV ran before config validation")
    )
    with pytest.raises(ValueError, match=message):
        evaluate((weather, load, config), battery_kwh=5.0)


def test_optimizer_prices_fixed_target_charging(tariff_case):
    weather, load, config = tariff_case
    greedy = evaluate(tariff_case, battery_kwh=5.0)
    config["smart_charging"] = FIXED_TARGET
    charged = evaluate((weather, load, config), battery_kwh=5.0)

    assert charged.yearly["Grid_AC_To_Battery_kWh"].sum() > 0.0
    assert (greedy.yearly["Grid_AC_To_Battery_kWh"] == 0.0).all()
    assert charged.yearly["Grid_Charge_Cost"].sum() > 0.0
    assert charged.metrics["Projected_NPV"] != greedy.metrics["Projected_NPV"]
    record = charged.provenance["smart_charging"]
    assert record["mode"] == "fixed_target"
    assert record["schedule_hash"] == charged.provenance["tariff"]["schedule_hash"]
    assert "smart_charging" not in greedy.provenance


def test_optimizer_search_shares_fixed_target_scoring_and_provenance(tariff_case):
    pytest.importorskip("pymoo")
    weather, load, config = tariff_case
    config["smart_charging"] = FIXED_TARGET
    fixed = evaluate((weather, load, config), battery_kwh=5.0)
    problem = optimization.SolarDesignProblem(weather, load, config, "unused")
    restored = pickle.loads(pickle.dumps(problem))
    out = {}
    restored._evaluate(np.array([4, 5.0, 30.0]), out)

    assert out["Projected_NPV"] == fixed.metrics["Projected_NPV"]
    assert restored.pricing.instructions == problem.pricing.instructions
    result = optimization.optimize_system_multi_objective(weather, load, config, pop_size=4, n_gen=1, seed=42)
    assert result.details["provenance"]["smart_charging"] == fixed.provenance["smart_charging"]


def test_optimizer_fixed_target_needs_a_battery(tariff_case):
    weather, load, config = tariff_case
    config["smart_charging"] = FIXED_TARGET
    with pytest.raises(ValueError, match=r"'smart_charging\.mode' = 'fixed_target' needs a battery"):
        evaluate((weather, load, config), battery_kwh=0.0)


def test_optimizer_search_without_a_battery_names_its_constraint(tariff_case):
    weather, load, config = tariff_case
    config["smart_charging"] = FIXED_TARGET
    config["constraints"] = {**config["constraints"], "max_battery_kwh": 0}
    with pytest.raises(ValueError, match=r"needs a battery; set constraints\.max_battery_kwh > 0"):
        optimization.SolarDesignProblem(weather, load, config, "unused")


def test_optimizer_accepts_disabled_smart_charging_unchanged(tariff_case):
    weather, load, config = tariff_case
    plain = evaluate(tariff_case, battery_kwh=5.0)
    config = {**config, "smart_charging": {"mode": "disabled"}}
    disabled = evaluate((weather, load, config), battery_kwh=5.0)

    pd.testing.assert_series_equal(pd.Series(disabled.metrics), pd.Series(plain.metrics))
    pd.testing.assert_frame_equal(disabled.yearly, plain.yearly)


def test_optimizer_money_matches_the_step_ledger(tariff_case, monkeypatch):
    from breos import projection

    frames = []
    simulate = projection.simulate_energy_balance

    def record(**kwargs):
        result = simulate(**kwargs)
        frames.append(result[0])
        return result

    monkeypatch.setattr(projection, "simulate_energy_balance", record)
    result = evaluate(tariff_case)
    for i, frame in enumerate(frames):
        local = pd.DatetimeIndex(frame["Datetime"]).tz_convert("Europe/Lisbon")
        prices = np.where((local.hour >= 8) & (local.hour < 22), 0.50, 0.10)
        row = result.yearly.iloc[i]
        assert row["Import_Cost"] == pytest.approx((frame["Import_From_Grid"] * prices).sum() / 1000)
        assert row["Export_Revenue"] == pytest.approx(frame["PV_AC_Export"].sum() / 1000 * 0.03)
        assert row["Baseline_Import_Cost"] == pytest.approx(16.0)
        assert row["Fixed_Charge"] == pytest.approx(0.80)
    assert result.provenance["tariff"]["calendar_policy"] == "replay_start_year"


def test_search_and_fixed_design_share_tariff_scoring_and_pickle(tariff_case):
    pytest.importorskip("pymoo")
    weather, load, config = tariff_case
    fixed = evaluate(tariff_case)
    problem = optimization.SolarDesignProblem(weather, load, config, "unused")
    restored = pickle.loads(pickle.dumps(problem))
    out = {}
    restored._evaluate(np.array([4, 0.0, 30.0]), out)

    assert out["Projected_NPV"] == fixed.metrics["Projected_NPV"]
    assert out["F"][1] == -fixed.metrics["Projected_NPV"]
    assert restored.tariff.schedule_hash == fixed.provenance["tariff"]["schedule_hash"]
    with pytest.raises(TypeError):
        restored.tariff.prices.import_prices["peak"] = 100


def test_search_result_records_tariff_provenance(tariff_case):
    pytest.importorskip("pymoo")
    weather, load, config = tariff_case
    result = optimization.optimize_system_multi_objective(weather, load, config, pop_size=4, n_gen=1, seed=42)
    restored = pickle.loads(pickle.dumps(result))
    assert restored.details["provenance"]["tariff"]["import_prices"] == TARIFF["import_prices"]
    assert restored.details["provenance"]["tariff"]["calendar_year"] == 2026


def test_invalid_tariff_fails_before_starting_workers(tariff_case, monkeypatch):
    pytest.importorskip("pymoo")
    weather, load, config = tariff_case
    config["tariff"]["currency"] = "USD"
    monkeypatch.setattr("multiprocessing.Pool", lambda *args, **kwargs: pytest.fail("Started worker pool"))
    with pytest.raises(ValueError, match="tariff.currency"):
        optimization.optimize_system_multi_objective(weather, load, config, n_procs=2)


@pytest.mark.parametrize(
    "smart_charging", [None, {**FIXED_TARGET, "target_usable_fraction": 1.0}], ids=["greedy", "fixed-target"]
)
@pytest.mark.parametrize(("freq", "backend"), [("h", "python"), ("15min", "numba")])
def test_three_year_tariff_design_reproduces_through_app(
    monkeypatch, open_meteo_weather, freq, backend, smart_charging
):
    import breos.app as app_module
    from breos import App
    from breos.app_inputs import AppRuntimeDependencies
    from breos.economics import COST_CONFIG_KEY_TO_PARAM
    from breos.weather import build_battery_temperature_series

    if backend == "numba":
        pytest.importorskip("numba")
    index = pd.date_range("2026-01-01", "2027-01-01", inclusive="left", freq=freq, tz="Europe/Lisbon")
    weather = open_meteo_weather(index)
    load = pd.DataFrame({"Load": 500.0}, index=index)
    tariff = deepcopy(TARIFF)
    if freq == "15min":
        tariff.update(
            schedule="pt_mainland_2026_weekly_tri", import_prices={"off_peak": 0.10, "mid_peak": 0.25, "peak": 0.50}
        )
    app_config = {
        "location": {"latitude": 41.15, "longitude": -8.61, "timezone": "Europe/Lisbon"},
        "n_modules": 8,
        "annual_consumption_kwh": 4380,
        "battery_kwh": 5.0,
        "tilt": 30.0,
        "azimuth": 180.0,
        "start_date": "2026-01-01",
        "resolution": freq,
        "projection_years": 3,
        "battery_eol_percentage": 0.99,
        "battery_temperature": 20.0,
        "enable_resistance_fade": True,
        "execution_backend": backend,
        "tariff": tariff,
        **({"smart_charging": smart_charging} if smart_charging else {}),
    }
    dependencies = AppRuntimeDependencies(
        load_profile=lambda **kwargs: load.copy(),
        load_weather=lambda **kwargs: None,
        fetch_tmy_weather_data=lambda **kwargs: (weather.copy(), {}),
        resample_to_15min=lambda frame, **kwargs: frame,
        build_battery_temperature_series=build_battery_temperature_series,
    )
    monkeypatch.setattr(App, "_runtime_dependencies", staticmethod(lambda: dependencies))
    artifacts = []
    run_app = app_module.run_app_simulation

    def record_app(*args):
        result = run_app(*args)
        artifacts.append(result)
        return result

    monkeypatch.setattr(app_module, "run_app_simulation", record_app)
    app = App(app_config)
    cfg, params = app._cfg, app._resolved.cost_params
    costs = {
        key: getattr(params, field)
        for key, field in COST_CONFIG_KEY_TO_PARAM.items()
        if key not in ("electricity_cost", "electricity_sold_cost", "daily_power_cost")
    }
    costs["dc_ac_ratio"] = cfg["inverter_loading_ratio"]
    config = {
        "location": cfg["location"],
        "simulation": {"resolution": freq, "years_projection": 3},
        "pv": {"module": cfg["pv_module"], "degradation_rate": cfg["pv_degradation_rate"]},
        "battery": {"temperature": 20.0, "eol_percentage": 0.99, "enable_resistance_fade": True},
        "costs": costs,
        "financials": {
            "discount_rate": cfg["discount_rate"],
            "inflation_rate": cfg["inflation_rate"],
            "sell_price_inflation": cfg["sell_price_inflation"],
        },
        "inverter_efficiency": cfg["inverter_efficiency"],
        "tariff": tariff,
        **({"smart_charging": smart_charging} if smart_charging else {}),
    }
    fixed = optimization.evaluate_projected_design(
        weather, load, config, n_modules=8, battery_kwh=5.0, tilt=30.0, azimuth=180.0, execution_backend=backend
    )
    app.simulate()
    result = app.result()
    pd.testing.assert_frame_equal(
        artifacts[0].yearly_df[fixed.yearly.columns], fixed.yearly, check_exact=False, rtol=1e-12, atol=1e-10
    )
    assert fixed.metrics["Projected_Total_Replacements"] > 0
    assert result["npv_savings"] == pytest.approx(fixed.metrics["Projected_NPV"], abs=0.0051, rel=0)
    assert result["provenance"]["tariff"] == fixed.provenance["tariff"]
    if smart_charging:
        assert result["provenance"]["smart_charging"] == fixed.provenance["smart_charging"]
        assert fixed.yearly["Grid_AC_To_Battery_kWh"].min() > 0.0, "the design must grid-charge"
    for app_row, (_, opt_row) in zip(result["financial"][1:], fixed.financial.iterrows(), strict=True):
        for app_key, column in (
            ("cost_import", "Cost_Import"),
            ("revenue_export", "Revenue_Export"),
            ("cost_fixed_charge", "Cost_Daily"),
            ("cost_replacement", "Cost_Replacement"),
        ):
            assert app_row[app_key] == pytest.approx(opt_row[column], abs=0.0051, rel=0)
