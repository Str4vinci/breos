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
    assert resolved["simulation"] == {"resolution": "h", "years_projection": DEFAULTS["projection_years"]}
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


def test_keys_nothing_read_are_refused():
    for config in (
        {**MINIMAL, "simulation": {"weather_file": "weather/porto.csv"}},
        {**MINIMAL, "simulation": {"irradiance_resampling": "clear_sky_energy_conserving"}},
        {**MINIMAL, "name": "study"},
    ):
        with pytest.raises(ValueError, match="Unknown"):
            resolve_optimization_config(config)


@pytest.mark.parametrize(
    ("section", "value", "key"),
    [
        ("optimization", {"algorithm": "nsga2"}, "optimization.algorithm"),
        ("pv", {"params": {"Mpp": 400, "Vmp": 40, "Imp": 10, "Voc": 48, "Isc": 11, "T_Pmax": -0.3}}, "T_Pmax"),
        ("pv", {"params": {"Mpp": 400, "Vmp": 40, "Imp": 10, "Voc": 48, "Isc": 11, "T_Voc": -0.3}}, "T_Voc"),
        ("pv", {"params": {"Mpp": 400, "Vmp": 40, "Imp": 10, "Voc": 48, "Isc": 11, "T_Isc": 0.05}}, "T_Isc"),
    ],
)
def test_keys_removed_in_0_7_0_are_unknown(section, value, key):
    # The single-value algorithm key and the short temperature-coefficient
    # aliases were removed; they raise like any other unknown key.
    with pytest.raises(ValueError, match=rf"Unknown key '[a-z.]*{key}'"):
        resolve_optimization_config({**MINIMAL, section: value})


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
    assert resolved["simulation"] == {"years_projection": 20, "resolution": "h"}
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
