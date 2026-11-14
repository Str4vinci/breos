"""The optimization config is checked and defaulted once, before any candidate is scored (#181, #162)."""

import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from breos import optimization
from breos.app_config import DEFAULTS, default_module_key
from breos.optimization_config import (
    DEFAULT_BUDGET,
    DEFAULT_MAX_AREA_M2,
    DEFAULT_MIN_TILT_DEG,
    OPTIMIZATION_TABLES,
    resolve_optimization_config,
    resolve_run_settings,
)

EXAMPLE = Path(__file__).resolve().parents[1] / "configs" / "optimization" / "projected-optimization.toml"
MINIMAL = {"location": {"latitude": 41.15, "longitude": -8.61}}


def test_the_example_config_resolves():
    with EXAMPLE.open("rb") as handle:
        config = tomllib.load(handle)
    resolved = resolve_optimization_config(config)
    assert resolved["constraints"]["budget"] == 15000.0
    assert resolved["optimization"]["pop_size"] == 40


def test_every_default_is_filled_in_by_name():
    resolved = resolve_optimization_config(MINIMAL)
    assert resolved["constraints"] == {
        "budget": DEFAULT_BUDGET,
        "max_area_m2": DEFAULT_MAX_AREA_M2,
        "max_modules": 60,
        "max_battery_kwh": 30.0,
        "min_tilt_deg": DEFAULT_MIN_TILT_DEG,
        "max_tilt_deg": 90.0,
        "tilt_margin_deg": 15.0,
        "enforce_zeb": False,
    }
    assert (DEFAULT_BUDGET, DEFAULT_MAX_AREA_M2, DEFAULT_MIN_TILT_DEG) == (10000.0, 20.0, 10.0)
    # The settings the App shares default as the App's do.
    assert resolved["simulation"] == {
        "resolution": "h",
        "years_projection": DEFAULTS["projection_years"],
        "irradiance_resampling": "auto",
    }
    assert resolved["pv"] == {"module": default_module_key(), "degradation_rate": DEFAULTS["pv_degradation_rate"]}
    assert resolved["inverter_efficiency"] == DEFAULTS["inverter_efficiency"]
    assert resolved["battery"]["temperature"] == DEFAULTS["battery_temperature"]
    assert resolved["location"] == {**MINIMAL["location"], "timezone": "UTC", "altitude": None, "name": ""}
    assert resolved["financials"] == {"inflation_rate": 0.02, "sell_price_inflation": 0.0, "discount_rate": 0.03}


@pytest.mark.parametrize("spelling", ["Naumann-Lam", " naumann_lam "])
def test_the_calendar_model_is_stored_as_the_aging_model_reads_it(spelling, monkeypatch):
    config = {**MINIMAL, "battery": {"calendar_model": spelling, "temperature": 20.0}}
    assert resolve_optimization_config(config)["battery"]["calendar_model"] == "naumann_lam"

    # The evaluated design ages with it: surrounding spaces used to pass here
    # and then fail in the aging model.
    index = pd.date_range("2026-01-05", periods=48, freq="h", tz="UTC")
    pv = pd.Series(np.where((index.hour >= 10) & (index.hour < 16), 1800.0, 0.0), index=index)
    monkeypatch.setattr(optimization, "calculate_pv_production_dc", lambda **kwargs: pv.copy())

    def final_soh(calendar_model):
        result = optimization.evaluate_projected_design(
            pd.DataFrame({"temp_air": 20.0}, index=index),
            pd.DataFrame({"Load": 1000.0}, index=index),
            {
                **MINIMAL,
                "battery": {"calendar_model": calendar_model, "temperature": 20.0},
                "simulation": {"resolution": "h", "years_projection": 1},
            },
            n_modules=4,
            battery_kwh=5.0,
            tilt=30.0,
            azimuth=180.0,
        )
        return result.yearly["Battery_SOH_%"].iloc[-1]

    assert final_soh(spelling) == final_soh("naumann_lam") < 100.0


def test_an_unknown_calendar_model_raises_before_the_search():
    with pytest.raises(ValueError, match=r"'battery\.calendar_model' must be one of"):
        resolve_optimization_config({**MINIMAL, "battery": {"calendar_model": "invented"}})


def test_resolving_twice_changes_nothing():
    resolved = resolve_optimization_config(MINIMAL)
    assert resolve_optimization_config(resolved) == resolved


@pytest.mark.parametrize("table", sorted(OPTIMIZATION_TABLES))
def test_an_unknown_key_in_any_table_raises(table):
    config = {**MINIMAL, table: {"surely_not_a_key": 1}}
    if table == "location":
        config[table] = {**MINIMAL["location"], "surely_not_a_key": 1}
    with pytest.raises(ValueError, match=f"'{table}.surely_not_a_key'"):
        resolve_optimization_config(config)


@pytest.mark.parametrize(
    ("key", "message"),
    [
        ("load", "Unknown optimization config key"),
        ("pv_specs", "Unknown optimization config key"),
        ("execution_backend", "Pass execution_backend to the function"),
    ],
)
def test_an_unknown_top_level_key_raises(key, message):
    with pytest.raises(ValueError, match=message):
        resolve_optimization_config({**MINIMAL, key: {}})


_PARAMS = {"Mpp": 400, "Vmp": 40, "Imp": 10, "Voc": 48, "Isc": 11}


@pytest.mark.parametrize(
    ("change", "match"),
    [
        # Each could hold only one value, and BatteryConfig no longer has either.
        ({"battery": {"battery_type": "lfp"}}, "Unknown key 'battery.battery_type'"),
        ({"battery": {"dc_coupled": True}}, "Unknown key 'battery.dc_coupled'"),
        # The single-value algorithm key and the short temperature-coefficient
        # aliases were removed in 0.7.0.
        ({"optimization": {"algorithm": "nsga2"}}, r"Unknown key '[a-z.]*optimization.algorithm'"),
        ({"pv": {"params": {**_PARAMS, "T_Pmax": -0.3}}}, r"Unknown key '[a-z.]*T_Pmax'"),
        ({"pv": {"params": {**_PARAMS, "T_Voc": -0.3}}}, r"Unknown key '[a-z.]*T_Voc'"),
        ({"pv": {"params": {**_PARAMS, "T_Isc": 0.05}}}, r"Unknown key '[a-z.]*T_Isc'"),
        # Keys nothing reads.
        ({"simulation": {"weather_file": "weather/porto.csv"}}, "Unknown"),
        ({"name": "study"}, "Unknown"),
    ],
)
def test_removed_and_unread_keys_raise_like_any_unknown_key(change, match):
    with pytest.raises(ValueError, match=match):
        resolve_optimization_config({**MINIMAL, **change})


def test_removed_keys_name_their_replacement():
    with pytest.raises(ValueError, match=r"constraints\.budget_eur was renamed to constraints\.budget"):
        resolve_optimization_config({**MINIMAL, "constraints": {"budget_eur": 5000.0}})
    with pytest.raises(ValueError, match="costs.panel_wp was removed"):
        resolve_optimization_config({**MINIMAL, "costs": {"panel_wp": 500.0}})


def test_the_first_spelling_set_wins():
    resolved = resolve_optimization_config(
        {
            **MINIMAL,
            "simulation": {"years_projection": 3},
            "financials": {"project_lifespan": 20, "pv_degradation_rate": 0.01},
            "inverter": {"efficiency": 0.9},
            "inverter_efficiency": 0.97,
        }
    )
    assert resolved["simulation"]["years_projection"] == 3
    assert resolved["pv"]["degradation_rate"] == 0.01
    assert resolved["inverter_efficiency"] == 0.97


def test_an_inline_module_states_its_rating():
    with pytest.raises(ValueError, match="pv.params.Isc"):
        resolve_optimization_config({**MINIMAL, "pv": {"params": {"Mpp": 400, "Vmp": 40, "Imp": 10, "Voc": 48}}})


@pytest.mark.parametrize(
    ("section", "value", "error"),
    [
        ("constraints", {"max_tilt_deg": "steep"}, "Use a number or 'adjust'"),
        ("constraints", {"enforce_zeb": "yes"}, "must be true or false"),
        ("optimization", {"early_stop": {"ftol": 0}}, "optimization.early_stop.ftol"),
        ("simulation", {"resolution": "30min"}, "must be one of: h, 15min"),
        ("simulation", {"irradiance_resampling": "invalid"}, "must be one of: auto, clear_sky"),
        ("battery", {"indoor_model": {"setpoint": 22}}, "battery.indoor_model.setpoint"),
    ],
)
def test_bad_values_raise(section, value, error):
    with pytest.raises((TypeError, ValueError), match=error):
        resolve_optimization_config({**MINIMAL, section: value})


def test_run_settings_come_from_the_argument_or_the_config():
    config = {"optimization": {"pop_size": 12, "seed": 7}}
    assert resolve_run_settings(config) == {"pop_size": 12, "n_gen": 100, "n_offsprings": None, "seed": 7}
    assert resolve_run_settings(config, pop_size=12, n_gen=5)["n_gen"] == 5
    with pytest.raises(ValueError, match="pop_size = 20 was passed, but optimization.pop_size = 12"):
        resolve_run_settings(config, pop_size=20)


def test_whole_numbers_may_be_written_as_floats():
    resolved = resolve_optimization_config({**MINIMAL, "simulation": {"years_projection": 20.0, "resolution": "H"}})
    assert resolved["simulation"] == {"years_projection": 20, "resolution": "h", "irradiance_resampling": "auto"}
    with pytest.raises(ValueError, match="simulation.years_projection' must be a whole number"):
        resolve_optimization_config({**MINIMAL, "simulation": {"years_projection": 20.5}})


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"tariff": {}}, "tariff"),
        (
            {"pv": {"module": "Generic_400W", "params": {"Mpp": 400, "Vmp": 40, "Imp": 10, "Voc": 48, "Isc": 11}}},
            "pv.params gives the module inline",
        ),
        ({"constraints": {"min_tilt_deg": 50.0, "max_tilt_deg": 40.0}}, "min_tilt_deg"),
        ({"constraints": {"min_tilt_deg": 80.0, "max_tilt_deg": "adjust"}}, "above the maximum tilt \\(60\\)"),
        ({"optimization": {"objective_basis": "steady_state"}}, "'steady_state' was removed"),
        ({"optimization": {"objective_basis": "annual"}}, "must be 'projected'"),
    ],
)
def test_the_resolver_refuses_what_the_search_would(changes, error):
    with pytest.raises(ValueError, match=error):
        resolve_optimization_config({**MINIMAL, **changes})


@pytest.mark.parametrize(("low", "high"), [(11.0, 14.0), (12.0, 12.0)])
def test_tilt_bounds_without_a_grid_point_are_refused(low, high):
    # Repair snaps tilt to 5° steps; 11–14° holds none, and clipping the
    # empty range used to return 10°, outside the recorded bounds.
    with pytest.raises(ValueError, match=r"holds no multiple of 5°"):
        resolve_optimization_config({**MINIMAL, "constraints": {"min_tilt_deg": low, "max_tilt_deg": high}})
    assert resolve_optimization_config({**MINIMAL, "constraints": {"min_tilt_deg": 11.0, "max_tilt_deg": 15.0}})


def test_snapping_refuses_bounds_without_a_grid_point():
    with pytest.raises(ValueError, match="hold no multiple of 5"):
        optimization._snap_to_grid_within_bounds(np.array([12.0]), 5.0, 11.0, 14.0)


def test_a_fixed_design_with_a_battery_needs_its_battery_prices_in_the_run_currency():
    # The search bound says no battery, but the evaluated design installs one:
    # its battery prices would otherwise be the EUR defaults, labelled USD.
    from breos.economics import default_amount_cost_keys
    from tests.conftest import _build_synthetic_weather

    weather = _build_synthetic_weather(2023)
    load = pd.DataFrame({"Load": np.full(len(weather), 500.0)}, index=weather.index)
    config = {
        "location": {"latitude": 41.15, "longitude": -8.61, "timezone": "UTC", "altitude": 0},
        "simulation": {"years_projection": 1},
        "currency": "USD",
        "constraints": {"budget": 10000, "max_battery_kwh": 0},
        "costs": dict.fromkeys(default_amount_cost_keys(tariff=False, battery=False), 0.0),
    }
    with pytest.raises(ValueError, match=r"The run is in USD, but storage_cost_per_kwh"):
        optimization.evaluate_projected_design(
            weather, load, config, n_modules=2, battery_kwh=5, tilt=30, azimuth=180, execution_backend="python"
        )
    result = optimization.evaluate_projected_design(
        weather, load, config, n_modules=2, battery_kwh=0, tilt=30, azimuth=180, execution_backend="python"
    )
    assert result.provenance["currency"] == "USD"


def _fixed_design_inputs():
    from tests.conftest import _build_synthetic_weather

    weather = _build_synthetic_weather(2023)
    load = pd.DataFrame({"Load": np.full(len(weather), 500.0)}, index=weather.index)
    config = {
        "location": {"latitude": 41.15, "longitude": -8.61, "timezone": "UTC", "altitude": 0},
        "simulation": {"years_projection": 1},
    }
    return weather, load, config


@pytest.mark.parametrize("constraints", [{}, {"max_battery_kwh": 0}, {"budget": 10000}])
def test_a_fixed_design_without_a_battery_needs_no_battery_prices_and_no_budget(constraints):
    # A fixed design reads no [constraints]: neither the search's battery bound
    # nor its budget decides which prices it must have in the run currency.
    from breos.economics import default_amount_cost_keys

    weather, load, config = _fixed_design_inputs()
    config = {
        **config,
        "currency": "USD",
        "constraints": constraints,
        "costs": dict.fromkeys(default_amount_cost_keys(tariff=False, battery=False), 0.0),
    }
    result = optimization.evaluate_projected_design(
        weather, load, config, n_modules=2, battery_kwh=0, tilt=30, azimuth=180, execution_backend="python"
    )
    assert result.provenance["currency"] == "USD"


def test_an_ac_fixed_design_with_a_battery_needs_its_inverter_rating():
    # The search bound excludes a battery, but the design installs one.
    weather, load, config = _fixed_design_inputs()
    config = {**config, "battery": {"coupling": "ac"}, "constraints": {"max_battery_kwh": 0}}
    with pytest.raises(ValueError, match=r"needs its inverter's rating: set battery.inverter_ac_rating_kw"):
        optimization.evaluate_projected_design(
            weather, load, config, n_modules=2, battery_kwh=5, tilt=30, azimuth=180, execution_backend="python"
        )


def test_tilt_bounds_without_a_grid_point_are_kept_where_no_tilt_gene_reads_them():
    tilt_bounds = {"min_tilt_deg": 11.0, "max_tilt_deg": 14.0}
    # An East-West search fixes its tilt at mode.east_west_tilt_deg.
    resolved = resolve_optimization_config({**MINIMAL, "mode": {"layouts": ["east_west"]}, "constraints": tilt_bounds})
    assert resolved["constraints"]["min_tilt_deg"] == 11.0
    with pytest.raises(ValueError, match=r"holds no multiple of 5°"):
        resolve_optimization_config({**MINIMAL, "mode": {"layouts": ["single"]}, "constraints": tilt_bounds})
    # A fixed design brings its own tilt.
    weather, load, config = _fixed_design_inputs()
    result = optimization.evaluate_projected_design(
        weather,
        load,
        {**config, "constraints": tilt_bounds},
        n_modules=2,
        battery_kwh=0,
        tilt=12,
        azimuth=180,
        execution_backend="python",
    )
    assert result.metrics["Tilt"] == 12.0


def test_a_search_without_a_feasible_design_names_the_broken_constraint_and_keeps_the_scored_designs():
    pytest.importorskip("pymoo")
    weather, load, config = _fixed_design_inputs()
    config = {**config, "constraints": {"budget": 10, "max_modules": 4, "max_battery_kwh": 2}}
    with pytest.raises(optimization.NoFeasibleDesignError) as raised:
        optimization.optimize_system_multi_objective(
            weather, load, config, pop_size=4, n_gen=1, seed=1, execution_backend="python"
        )
    evaluated = raised.value.evaluated
    assert len(evaluated) > 0
    assert not evaluated["Feasible"].any()
    assert evaluated.attrs["currency"] == "EUR"
    total = len(evaluated)
    assert f"constraints.budget by every design ({total} of {total})" in str(raised.value)
    assert "constraints.max_area_m2" not in str(raised.value)
