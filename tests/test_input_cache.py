"""Reusing prepared inputs across sweep runs (#181).

``INPUT_INDEPENDENT_KEYS`` lists the config keys the input stage never reads.
Each is pinned here by changing it and comparing the prepared weather, PV,
load and battery temperature, so a key cannot join the list, or stay on it,
unless it truly leaves the inputs unchanged.
"""

import csv
import pickle
from unittest import mock

import pandas as pd
import pytest

from breos import App, cli
from breos.app_config import APP_CONFIG_FIELDS
from breos.app_inputs import (
    INPUT_INDEPENDENT_KEYS,
    _input_cache_key,
    prepare_simulation_inputs,
    reuse_prepared_inputs,
)
from tools.generate_app_golden import _fake_fetch

BASE = {
    "location": "porto",
    "n_modules": 8,
    "annual_consumption_kwh": 4000,
    "start_date": "2025-01-01",
    "projection_years": 3,
    "cost_preset": "residential_pt",
    "emissions_country": "PT",
    "battery_kwh": 5,
    "resolution": "h",
}
TOU = {
    "schedule": "pt_mainland_2026_daily_bi",
    "currency": "EUR",
    "import_prices": {"peak": 0.28, "off_peak": 0.11},
    "export_prices": {"all": 0.05},
}
FIXED_TARGET = {
    "mode": "fixed_target",
    "target_usable_fraction": 0.5,
    "charge_periods": ["off_peak"],
    "discharge_periods": ["peak"],
    "grid_charge_efficiency": 0.95,
}

# One change per input-independent key, with the companions some need. The
# companions are input-independent keys too.
CHANGES = {
    "battery_kwh": {"battery_kwh": 10},
    "battery_min_soc": {"battery_min_soc": 0.2},
    "battery_max_soc": {"battery_max_soc": 0.8},
    "battery_max_charge_power_w": {"battery_max_charge_power_w": 2000.0},
    "battery_max_discharge_power_w": {"battery_max_discharge_power_w": 2000.0},
    "battery_power_limit_c_rate": {"battery_power_limit_c_rate": 0.5},
    "battery_eol_percentage": {"battery_eol_percentage": 0.8},
    "battery_rte": {"battery_rte": 0.9},
    "enable_resistance_fade": {"enable_resistance_fade": True},
    "inverter_efficiency": {"inverter_efficiency": 0.97},
    "inverter_loading_ratio": {"inverter_loading_ratio": 1.1},
    "calendar_model": {"calendar_model": "naumann"},
    "degradation_engine": {"degradation_engine": "blast", "blast_model": "nmc_gr_50ah_b1"},
    "blast_model": {"degradation_engine": "blast", "blast_model": "lfp_gr_250ah_prismatic"},
    "pv_degradation_rate": {"pv_degradation_rate": 0.01},
    "projection_years": {"projection_years": 5},
    "cost_preset": {"cost_preset": "residential_es"},
    "costs": {"costs": {"storage_cost_per_kwh": 300.0}},
    "inflation_rate": {"inflation_rate": 0.03},
    "sell_price_inflation": {"sell_price_inflation": 0.01},
    "discount_rate": {"discount_rate": 0.05},
    "tariff": {"tariff": TOU},
    "smart_charging": {"tariff": TOU, "smart_charging": FIXED_TARGET},
    "emissions_country": {"emissions_country": "ES"},
    "export_emissions_factor_gco2_kwh": {"export_emissions_factor_gco2_kwh": 100.0},
    "execution_backend": {"execution_backend": "numba"},
}


def _prepared(config):
    with (
        mock.patch("breos.app.fetch_tmy_weather_data", _fake_fetch),
        mock.patch("breos.app.load_weather", lambda **_kwargs: None),
    ):
        app = App(config)
        return prepare_simulation_inputs(app._cfg, app._resolved, app._runtime_dependencies())


@pytest.fixture(scope="module")
def base_inputs():
    return pickle.dumps(_prepared(BASE))


def test_every_input_independent_key_is_a_registry_key_with_a_pinning_change():
    assert INPUT_INDEPENDENT_KEYS <= set(APP_CONFIG_FIELDS)
    assert set(CHANGES) == INPUT_INDEPENDENT_KEYS


@pytest.mark.parametrize("key", sorted(CHANGES))
def test_an_input_independent_key_leaves_the_prepared_inputs_unchanged(key, base_inputs):
    changed = {**BASE, **CHANGES[key]}
    assert _input_cache_key(App(changed)._cfg) == _input_cache_key(App(BASE)._cfg)
    # Pickled, the frames compare values, index, dtypes and attrs at once.
    assert pickle.dumps(_prepared(changed)) == base_inputs


@pytest.mark.parametrize(
    "change",
    [
        {"n_modules": 10},
        {"tilt": 20.0},
        {"annual_consumption_kwh": 3000},
        {"battery_temperature": 20.0},
        {"resolution": "15min"},
    ],
)
def test_an_input_key_changes_the_cache_key_and_the_inputs(change, base_inputs):
    changed = {**BASE, **change}
    assert _input_cache_key(App(changed)._cfg) != _input_cache_key(App(BASE)._cfg)
    assert pickle.dumps(_prepared(changed)) != base_inputs


def test_a_value_json_cannot_hold_is_not_cached():
    assert _input_cache_key({"battery_temperature": pd.Series([20.0])}) is None
    assert _input_cache_key({"battery_temperature": 20.0}) is not None


def test_inputs_are_prepared_once_per_input_config_inside_the_block_only(monkeypatch):
    import breos.runners.app as runner

    calls = []
    prepare = runner.prepare_simulation_inputs

    def counting(*args):
        calls.append(args[0]["n_modules"])
        return prepare(*args)

    monkeypatch.setattr(runner, "prepare_simulation_inputs", counting)
    monkeypatch.setattr("breos.app.fetch_tmy_weather_data", _fake_fetch)
    monkeypatch.setattr("breos.app.load_weather", lambda **_kwargs: None)

    def run(config):
        app = App(config)
        app.simulate()
        return app.result()

    run(BASE)
    run(BASE)
    assert calls == [8, 8]

    calls.clear()
    with reuse_prepared_inputs():
        first = run(BASE)
        again = run({**BASE, "tariff": TOU, "smart_charging": FIXED_TARGET})
        same = run(BASE)
        run({**BASE, "n_modules": 10})
    assert calls == [8, 10]
    assert same == first
    assert again["grid_charge_cost_year1_prices"] > 0.0

    calls.clear()
    run(BASE)
    assert calls == [8]


SWEEP_CONFIG = """
location = "porto"
annual_consumption_kwh = 4000
start_date = "2025-01-01"
projection_years = 2
cost_preset = "residential_pt"
emissions_country = "PT"
battery_kwh = 5.0

[smart_charging]
mode = "fixed_target"
target_usable_fraction = 0.5
charge_periods = ["off_peak"]
discharge_periods = ["peak"]
grid_charge_efficiency = 0.95

[tariff]
schedule = "pt_mainland_2026_daily_bi"
currency = "EUR"
import_prices = { peak = 0.2310, off_peak = 0.1210 }
export_prices = { all = 0.0500 }
fixed_charge_per_day = 0.30

[sweep]
n_modules = [8, 10]
battery_kwh = [5.0, 10.0]
"tariff.import_prices.off_peak" = [0.10, 0.14]
"smart_charging.target_usable_fraction" = [0.5, 0.9]
"""


def test_a_sweep_writes_the_same_csv_bytes_with_and_without_the_cache(monkeypatch, tmp_path):
    import contextlib

    import breos.runners.app as runner

    monkeypatch.setattr("breos.app.fetch_tmy_weather_data", _fake_fetch)
    monkeypatch.setattr("breos.app.load_weather", lambda **_kwargs: None)
    calls = []
    prepare = runner.prepare_simulation_inputs

    def counting(*args):
        calls.append(args[0]["n_modules"])
        return prepare(*args)

    monkeypatch.setattr(runner, "prepare_simulation_inputs", counting)
    config_path = tmp_path / "sweep.toml"
    config_path.write_text(SWEEP_CONFIG, encoding="utf-8")

    cached = tmp_path / "cached.csv"
    assert cli.main(["sweep", "--config", str(config_path), "--output", str(cached)]) == 0
    assert calls == [8, 10]

    calls.clear()
    monkeypatch.setattr(cli, "reuse_prepared_inputs", contextlib.nullcontext)
    fresh = tmp_path / "fresh.csv"
    assert cli.main(["sweep", "--config", str(config_path), "--output", str(fresh)]) == 0
    assert len(calls) == 16

    assert cached.read_bytes() == fresh.read_bytes()
    rows = list(csv.DictReader(fresh.open(encoding="utf-8")))
    assert len(rows) == 16
    # The runs differ, so the equality is not between identical rows.
    assert len({row["grid_import_cost_year1_prices"] for row in rows}) > 8
