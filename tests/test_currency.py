"""Currency selection (#376, ADR 0003 E13): one currency per run, checked, carried, never converted."""

import csv
import json
import re
from copy import deepcopy
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from breos import cli, optimization
from breos.app import REVALUATION_KEYS, App
from breos.app_config import resolve_app_config
from breos.economics import default_amount_cost_keys, find_payback_year_interpolated, write_cost_projection
from breos.montecarlo import MonteCarloSettings, run_montecarlo
from breos.optimization_config import resolve_optimization_config
from tools.generate_app_golden import SCENARIOS, _fake_fetch
from tools.parity.app_parity import flatten

# A power of two, so every scaled sum and product is the exact scaled float.
K = 128.0

# The residential_pt preset, written out, with the zero-default costs set too.
EUR_COSTS = {
    "electricity_cost": 0.2582,
    "electricity_sold_cost": 0.04,
    "daily_power_cost": 0.3,
    "module_cost_per_w": 0.125,
    "inverter_cost_per_kw_hybrid": 102.58,
    "inverter_cost_per_kw_simple": 48.37,
    "storage_cost_per_kwh": 500.0,
    "installation_cost_per_module": 350.0,
    "installation_cost_battery": 350.0,
    "other_cost_per_module": 30.0,
    "other_costs": 120.0,
    "land_cost": 200.0,
    "maintenance_cost_per_panel": 10.0,
    "maintenance_cost": 15.0,
    "operation_cost": 5.0,
}
FLAT_PRICES = ("electricity_cost", "electricity_sold_cost", "daily_power_cost")
CAPEX_COSTS = {key: value for key, value in EUR_COSTS.items() if key not in FLAT_PRICES}
BASE = {key: value for key, value in SCENARIOS["native_h_replacement"].items() if key not in ("cost_preset",)} | {
    "execution_backend": "python",
    "costs": EUR_COSTS,
}
TOU = {
    "schedule": "pt_mainland_2026_daily_bi",
    "currency": "EUR",
    "import_prices": {"peak": 0.28, "off_peak": 0.11},
    "export_prices": {"all": 0.05},
    "fixed_charge_per_day": 0.25,
    "annual_network_credit": {
        "amount_per_year": 146.58,
        "network_fixed_per_year": 39.70,
        "network_import_prices": {"peak": 0.0888, "off_peak": 0.0311},
    },
}
REFERENCE = {
    "currency": "EUR",
    "import_prices": {"all": 0.30},
    "fixed_charge_per_day": 0.40,
    "annual_network_credit": {
        "amount_per_year": 146.58,
        "network_fixed_per_year": 39.70,
        "network_import_prices": {"all": 0.0888},
    },
}
FIXED_TARGET = {
    "mode": "fixed_target",
    "target_usable_fraction": 0.5,
    "charge_periods": ["off_peak"],
    "discharge_periods": ["peak"],
    "grid_charge_efficiency": 0.95,
}
TARIFF_RUN = {
    **BASE,
    "costs": CAPEX_COSTS,
    "tariff": TOU,
    "reference_tariff": REFERENCE,
    "smart_charging": FIXED_TARGET,
    "terminal_value": {"basis": "battery_health_fraction"},
}
# The experimental planner chooses each day's target from the prices, and a
# wear weight is money too.
PLANNER_RUN = {
    **TARIFF_RUN,
    "projection_years": 2,
    "smart_charging": {
        "mode": "daily_persistence",
        "charge_periods": ["off_peak"],
        "discharge_periods": ["peak"],
        "grid_charge_efficiency": 0.95,
        "target_levels": 5,
        "soc_states": 7,
        "wear_cost_per_kwh": 0.02,
    },
}
USD_COSTS = {key: value * 1.1 for key, value in EUR_COSTS.items()}


def _offline():
    return (
        mock.patch("breos.app.fetch_tmy_weather_data", _fake_fetch),
        mock.patch("breos.app.load_weather", lambda **_kwargs: None),
    )


def _scaled_prices(table, factor):
    """A tariff or reference table with every amount scaled; the schedule and names unchanged."""
    table = deepcopy(table)
    for key in ("import_prices", "export_prices", "network_import_prices"):
        if key in table:
            table[key] = {
                period: (
                    {inner: price * factor for inner, price in value.items()}
                    if isinstance(value, dict)
                    else value * factor
                )
                for period, value in table[key].items()
            }
    for key in ("fixed_charge_per_day", "amount_per_year", "network_fixed_per_year"):
        if key in table:
            table[key] *= factor
    if "annual_network_credit" in table:
        table["annual_network_credit"] = _scaled_prices(table["annual_network_credit"], factor)
    return table


def _scaled_config(config, factor, currency):
    """``config`` with every money input multiplied by ``factor`` and labelled ``currency``."""
    scaled = deepcopy(config)
    scaled["currency"] = currency
    scaled["costs"] = {key: value * factor for key, value in config["costs"].items()}
    for key in ("tariff", "reference_tariff"):
        if scaled.get(key) is not None:
            scaled[key] = {**_scaled_prices(scaled[key], factor), "currency": currency}
    if "wear_cost_per_kwh" in (scaled.get("smart_charging") or {}):
        scaled["smart_charging"]["wear_cost_per_kwh"] *= factor
    return scaled


def _run(config):
    weather_patch, load_patch = _offline()
    with weather_patch, load_patch:
        app = App(config)
        app.simulate()
    return app


# A money output is named for money and not for a physical unit; every one
# must scale by K (a zero stays zero), and every other output must not move.
_MONEY = re.compile(
    r"cost|price|revenue|saving|npv|lcoe|investment|credit|balance|(fixed|network)_charge|reference", re.I
)
_PHYSICAL = re.compile(
    r"(?<!per)_kwh$|_pct$|_%$|_kg$|_wh$|(^|_)soh(_|$)|_years?$|_w$|_hours?$|(^|_)(fraction|ratio|count|replacements)$",
    re.I,
)


def _is_money(name):
    leaf = name.rsplit(".", 1)[-1]
    return bool(_MONEY.search(leaf)) and not _PHYSICAL.search(leaf)


def _assert_frames_scale(base, scaled, *, money):
    """Every money column of ``scaled`` is ``K`` times ``base``, every other column equal; the named ``money`` move."""
    assert list(base.columns) == list(scaled.columns)
    moved = set()
    for column in base.columns:
        left, right = base[column], scaled[column]
        if pd.api.types.is_numeric_dtype(left) and _is_money(column):
            np.testing.assert_array_equal(right.to_numpy(), K * left.to_numpy(), err_msg=column)
            moved.update([column] if (left != 0).any() else [])
        else:
            assert left.equals(right), column
    assert set(money) <= moved


def _assert_results_scale(base, scaled):
    """Rounded result leaves: money ones K times, to the 0.01 rounding, and every other one equal."""
    left, right = flatten(base), flatten(scaled)
    assert left.keys() == right.keys()
    money = set()
    for key, value in left.items():
        if key.startswith("provenance."):
            continue
        other = right[key]
        if isinstance(value, float) and _is_money(key):
            assert other == pytest.approx(K * value, rel=1e-12, abs=0.005 * (K + 1)), key
            money.update([key] if value != 0 else [])
        else:
            assert value == other or (isinstance(value, float) and np.isnan(value) and np.isnan(other)), key
    return money


@pytest.mark.parametrize("config", [BASE, TARIFF_RUN, PLANNER_RUN], ids=["flat", "tariff", "planner"])
def test_scaling_every_money_input_scales_every_money_output(config):
    base, scaled = _run(config), _run(_scaled_config(config, K, "JPY"))
    base_result, scaled_result = base.result(), scaled.result()

    assert base_result["provenance"]["currency"] == "EUR"
    assert scaled_result["provenance"]["currency"] == "JPY"
    money = _assert_results_scale(base_result, scaled_result)
    assert {"total_investment", "npv_savings", "lcoe_per_kwh", "battery_replacement_cost_t0_prices"} <= money
    # Energy, ageing and the payback year do not move.
    assert base_result["yearly"] == scaled_result["yearly"]
    assert base_result["battery_replacements"] == scaled_result["battery_replacements"] > 0
    assert base_result["payback_year"] == scaled_result["payback_year"]

    left, right = base._artifacts.cost_projection, scaled._artifacts.cost_projection
    _assert_frames_scale(
        left,
        right,
        money=["Cost_Import", "Cost_Operation", "Cost_Daily", "Cost_Replacement", "Savings_Cumulative_NPV"],
    )
    assert right.attrs["currency"] == "JPY"
    assert find_payback_year_interpolated(left) == find_payback_year_interpolated(right)
    for key in ("total_investment", "lcoe_per_kwh", "final_npv_savings"):
        assert right.attrs[key] == K * left.attrs[key]
    _assert_frames_scale(base.timeseries(), scaled.timeseries(), money=[])
    if config is not BASE:
        assert {"terminal_health_credit", "network_credit_year1_prices"} <= money
        assert scaled_result["network_credit_year1_prices"] > 0


def test_revalue_moves_a_run_to_another_currency():
    base = _run(BASE)
    scaled = _scaled_config(BASE, K, "JPY")
    weather_patch, load_patch = _offline()
    with weather_patch, load_patch:
        revalued = base.revalue({"currency": "JPY", "costs": scaled["costs"]})
        with pytest.raises(ValueError, match=r"The run is in JPY, but electricity_cost"):
            base.revalue({"currency": "JPY", "costs": {"electricity_cost": None}})
    assert "currency" in REVALUATION_KEYS
    assert revalued["provenance"]["revaluation"] == {"method": "repriced", "changed_keys": ["costs", "currency"]}
    del revalued["provenance"]["revaluation"]
    assert flatten(revalued) == flatten(_run(scaled).result())


def _named_amounts(error, currency):
    match = re.search(rf"to \w+, but (.+) (?:is an amount|are amounts) in {currency}\.", str(error.value))
    assert match, error.value
    return sorted(match.group(1).split(", "))


def test_revalue_into_another_currency_restates_every_cost():
    # An explicit zero is zero in any currency and need not be restated.
    costs = {**EUR_COSTS, "land_cost": 0.0}
    base = _run({**BASE, "projection_years": 2, "cost_preset": "residential_pt", "costs": costs})
    usd = {key: value * K for key, value in costs.items() if key != "land_cost"}
    weather_patch, load_patch = _offline()
    with weather_patch, load_patch:
        # A new label alone would report the EUR amounts as USD.
        with pytest.raises(ValueError, match=r"cost_preset 'residential_pt' is priced in EUR, but the run is in USD"):
            base.revalue({"currency": "USD"})
        with pytest.raises(ValueError, match=r"from EUR to USD, but costs\.daily_power_cost, .* are amounts in EUR"):
            base.revalue({"currency": "USD", "cost_preset": None})
        # One price restated, thirteen still in EUR.
        with pytest.raises(ValueError) as error:
            base.revalue({"currency": "USD", "cost_preset": None, "costs": {"module_cost_per_w": 0.30}})
        assert _named_amounts(error, "EUR") == sorted(f"costs.{key}" for key in usd if key != "module_cost_per_w")
        with pytest.raises(ValueError, match=r"cost_preset 'residential_pt' is priced in EUR, but the run is in USD"):
            base.revalue({"currency": "USD", "costs": usd})
        revalued = base.revalue({"currency": "USD", "costs": usd, "cost_preset": None})
    assert revalued["provenance"]["currency"] == "USD"
    assert revalued["total_investment"] == pytest.approx(K * base.result()["total_investment"], abs=0.005 * (K + 1))


def test_revalue_into_another_currency_restates_every_tariff_amount():
    # A zero price list is zero in any currency and need not be restated.
    config = {
        **TARIFF_RUN,
        "projection_years": 2,
        "smart_charging": None,
        "terminal_value": None,
        "tariff": {**TOU, "export_prices": {"all": 0.0}},
    }
    base = _run(config)
    usd = _scaled_config(config, K, "USD")
    credit = ("amount_per_year", "network_fixed_per_year", "network_import_prices")
    weather_patch, load_patch = _offline()
    with weather_patch, load_patch:
        # Only the labels change: every tariff and reference amount would be EUR read as USD.
        with pytest.raises(ValueError) as error:
            base.revalue(
                {
                    "currency": "USD",
                    "costs": usd["costs"],
                    "tariff": {"currency": "USD"},
                    "reference_tariff": {"currency": "USD"},
                }
            )
        assert _named_amounts(error, "EUR") == sorted(
            ["tariff.import_prices", "tariff.fixed_charge_per_day", "reference_tariff.import_prices"]
            + ["reference_tariff.fixed_charge_per_day"]
            + [f"{table}.annual_network_credit.{key}" for table in ("tariff", "reference_tariff") for key in credit]
        )
        # The price lists restated, the fixed charges and the network credits kept.
        prices = {key: usd["tariff"][key] for key in ("currency", "import_prices")}
        with pytest.raises(ValueError) as error:
            base.revalue(
                {
                    "currency": "USD",
                    "costs": usd["costs"],
                    "tariff": prices,
                    "reference_tariff": {key: usd["reference_tariff"][key] for key in ("currency", "import_prices")},
                }
            )
        assert _named_amounts(error, "EUR") == sorted(
            ["tariff.fixed_charge_per_day", "reference_tariff.fixed_charge_per_day"]
            + [f"{table}.annual_network_credit.{key}" for table in ("tariff", "reference_tariff") for key in credit]
        )
        # A network credit removed counts as restated.
        removed = base.revalue(
            {
                "currency": "USD",
                "costs": usd["costs"],
                "tariff": {
                    **prices,
                    "fixed_charge_per_day": usd["tariff"]["fixed_charge_per_day"],
                    "annual_network_credit": None,
                },
                "reference_tariff": usd["reference_tariff"],
            }
        )
        revalued = base.revalue({key: usd[key] for key in ("currency", "costs", "tariff", "reference_tariff")})
    assert removed["provenance"]["currency"] == "USD"
    assert removed["provenance"]["resolved_config"]["tariff"].get("annual_network_credit") is None
    del revalued["provenance"]["revaluation"]
    assert flatten(revalued) == flatten(_run(usd).result())


def test_revalue_by_the_tariff_currency_alone_restates_every_tariff_amount():
    # Without the currency key, the run's currency follows the tariff's.
    config = {**TARIFF_RUN, "projection_years": 2, "smart_charging": None, "reference_tariff": None}
    base = _run(config)
    usd = _scaled_config(config, K, "USD")
    weather_patch, load_patch = _offline()
    with weather_patch, load_patch:
        with pytest.raises(ValueError, match=r"from EUR to USD, but tariff\.import_prices, .* are amounts in EUR"):
            base.revalue({"costs": usd["costs"], "tariff": {"currency": "USD"}})
        revalued = base.revalue({"costs": usd["costs"], "tariff": usd["tariff"]})
    del revalued["provenance"]["revaluation"]
    assert flatten(revalued) == flatten(_run({**usd, "currency": None}).result())


def test_revalue_back_to_the_default_currency_restates_every_cost():
    config = {**BASE, "projection_years": 2}
    base = _run(_scaled_config(config, K, "JPY"))
    weather_patch, load_patch = _offline()
    with weather_patch, load_patch:
        with pytest.raises(ValueError, match=r"from JPY to EUR, but costs\.\w+, .* are amounts in JPY"):
            base.revalue({"currency": None})
        revalued = base.revalue({"currency": None, "costs": EUR_COSTS})
        defaults = base.revalue({"currency": None, "costs": {key: None for key in EUR_COSTS}})
    del revalued["provenance"]["revaluation"], defaults["provenance"]["revaluation"]
    assert revalued["provenance"]["currency"] == defaults["provenance"]["currency"] == "EUR"
    assert flatten(revalued) == flatten(_run({**config, "currency": None}).result())
    assert flatten(defaults) == flatten(_run({**config, "currency": None, "costs": {}}).result())


def test_revalue_refuses_a_currency_change_under_a_wear_cost():
    base = _run(PLANNER_RUN)
    scaled = _scaled_config(PLANNER_RUN, K, "JPY")
    weather_patch, load_patch = _offline()
    with weather_patch, load_patch:
        with pytest.raises(ValueError, match=r"smart_charging\.wear_cost_per_kwh \(0\.02\) is in EUR, and revalue"):
            base.revalue({key: scaled[key] for key in ("currency", "costs", "tariff", "reference_tariff")})


def test_monte_carlo_money_scales_and_records_the_currency(tmp_path, write_multiyear_weather):
    weather = write_multiyear_weather(tmp_path / "multi.csv")
    settings = MonteCarloSettings(weather_file=str(weather), n_runs=2, years_per_run=2, seed=3)
    config = {**BASE, "projection_years": 2}
    base = run_montecarlo(config, settings)
    scaled = run_montecarlo(_scaled_config(config, K, "JPY"), settings)

    assert scaled.provenance["currency"] == scaled.runs.attrs["currency"] == "JPY"
    _assert_frames_scale(base.runs, scaled.runs, money=["npv_savings", "lcoe_per_kwh"])


def _optimizer_case(monkeypatch):
    index = pd.date_range("2026-01-05", periods=48, freq="h", tz="Europe/Lisbon")
    weather = pd.DataFrame({"temp_air": 20.0}, index=index)
    load = pd.DataFrame({"Load": 1000.0}, index=index)
    pv = pd.Series(np.where((index.hour >= 10) & (index.hour < 16), 1800.0, 0.0), index=index)
    monkeypatch.setattr(optimization, "calculate_pv_production_dc", lambda **kwargs: pv.copy())
    config = {
        "location": {"latitude": 41.15, "longitude": -8.61, "timezone": "Europe/Lisbon", "altitude": 100},
        "simulation": {"resolution": "h", "years_projection": 2},
        "battery": {"temperature": 20.0, "replacement_cost": 2400.0},
        "mode": {"fixed_azimuth": 180.0},
        "costs": dict(CAPEX_COSTS),
        "constraints": {
            "budget": 8000.0,
            "max_modules": 16,
            "max_area_m2": 100,
            "max_tilt_deg": 60,
            "max_battery_kwh": 10,
        },
        "tariff": {key: value for key, value in TOU.items() if key != "annual_network_credit"},
        "reference_tariff": REFERENCE,
    }
    return weather, load, config


def _scaled_optimizer_config(config, factor, currency):
    scaled = _scaled_config(config, factor, currency)
    scaled["battery"]["replacement_cost"] *= factor
    scaled["constraints"]["budget"] *= factor
    return scaled


def test_optimizer_ranks_and_constrains_scaled_money_alike(monkeypatch):
    pytest.importorskip("pymoo")
    weather, load, config = _optimizer_case(monkeypatch)
    scaled_config = _scaled_optimizer_config(config, K, "JPY")

    designs = [(4, 0.0), (8, 5.0), (12, 10.0), (16, 0.0)]
    base = [
        optimization.evaluate_projected_design(
            weather, load, config, n_modules=n, battery_kwh=b, tilt=30.0, azimuth=180.0
        )
        for n, b in designs
    ]
    scaled = [
        optimization.evaluate_projected_design(
            weather, load, scaled_config, n_modules=n, battery_kwh=b, tilt=30.0, azimuth=180.0
        )
        for n, b in designs
    ]
    for left, right in zip(base, scaled):
        assert right.provenance["currency"] == "JPY"
        assert right.metrics["Projected_NPV"] == K * left.metrics["Projected_NPV"]
        assert right.metrics["Projected_Initial_Cost"] == K * left.metrics["Projected_Initial_Cost"]
        assert right.metrics["Projected_Payback_Year_Interpolated"] == left.metrics[
            "Projected_Payback_Year_Interpolated"
        ] or (
            np.isnan(left.metrics["Projected_Payback_Year_Interpolated"])
            and np.isnan(right.metrics["Projected_Payback_Year_Interpolated"])
        )

    def ranking(results, budget):
        return [
            (index, result.metrics["Projected_Initial_Cost"] <= budget)
            for index, result in sorted(enumerate(results), key=lambda item: -item[1].metrics["Projected_NPV"])
        ]

    assert ranking(base, config["constraints"]["budget"]) == ranking(scaled, scaled_config["constraints"]["budget"])
    assert {feasible for _, feasible in ranking(base, config["constraints"]["budget"])} == {True, False}

    run = {"pop_size": 8, "n_gen": 2, "seed": 5}
    left = optimization.optimize_system_multi_objective(weather, load, config, **run).details["pareto"]
    right = optimization.optimize_system_multi_objective(weather, load, scaled_config, **run).details["pareto"]
    assert right.attrs["currency"] == "JPY"
    _assert_frames_scale(left, right, money=["NPV"])


# --- Selection and consistency -----------------------------------------------


def test_the_run_currency_is_the_key_else_the_tariff_else_eur():
    assert resolve_app_config(BASE).currency == "EUR"
    assert resolve_app_config({**BASE, "currency": "usd", "costs": USD_COSTS}).currency == "USD"
    usd_tariff = {**TOU, "currency": "USD"}
    assert resolve_app_config({**TARIFF_RUN, "tariff": usd_tariff, "reference_tariff": None}).currency == "USD"
    with pytest.raises(ValueError, match=r"'currency' must be an ISO 4217 currency code"):
        resolve_app_config({**BASE, "currency": "EURO"})


def test_a_tariff_in_another_currency_is_refused():
    with pytest.raises(ValueError, match=r"'tariff\.currency' is EUR, but 'currency' is USD\. BREOS does not convert"):
        resolve_app_config({**TARIFF_RUN, "currency": "USD"})
    with pytest.raises(ValueError, match=r"'reference_tariff\.currency' is EUR, but the run is in USD"):
        resolve_app_config({**BASE, "currency": "USD", "costs": USD_COSTS, "reference_tariff": REFERENCE})


def test_a_eur_preset_is_never_relabelled():
    with pytest.raises(ValueError, match=r"cost_preset 'residential_pt' is priced in EUR, but the run is in USD"):
        resolve_app_config({**BASE, "currency": "USD", "costs": USD_COSTS, "cost_preset": "residential_pt"})


def _named_defaults(message, currency):
    match = re.fullmatch(
        rf"The run is in {currency}, but (.+) would come from the built-in EUR defaults\. BREOS does not convert "
        rf"currencies: set them in {currency} under \[costs\]\.",
        message,
    )
    assert match, message
    return match.group(1).split(", ")


def test_a_non_eur_run_names_every_default_it_would_inherit():
    pv_only = {**BASE, "battery_kwh": 0, "currency": "GBP", "costs": {}}
    with pytest.raises(ValueError) as error:
        resolve_app_config(pv_only)
    named = _named_defaults(str(error.value), "GBP")
    # A PV-only run prices no battery, and a zero default stays zero.
    assert named == [
        "electricity_cost",
        "electricity_sold_cost",
        "daily_power_cost",
        "module_cost_per_w",
        "inverter_cost_per_kw_simple",
        "installation_cost_per_module",
        "other_cost_per_module",
        "maintenance_cost_per_panel",
    ]
    tariff_run = {
        **TARIFF_RUN,
        "currency": "GBP",
        "costs": {},
        "reference_tariff": None,
        "tariff": {**TOU, "currency": "GBP"},
    }
    with pytest.raises(ValueError) as error:
        resolve_app_config(tariff_run)
    named = _named_defaults(str(error.value), "GBP")
    assert "electricity_cost" not in named and "inverter_cost_per_kw_simple" not in named
    assert {"storage_cost_per_kwh", "installation_cost_battery", "inverter_cost_per_kw_hybrid"} <= set(named)
    # Every money default given, the run resolves.
    given = {key: 1.0 for key in default_amount_cost_keys(tariff=True, battery=True)}
    assert resolve_app_config({**tariff_run, "costs": given}).currency == "GBP"


def test_eur_runs_keep_inheriting_the_defaults_and_presets():
    assert resolve_app_config({**BASE, "costs": {}}).currency == "EUR"
    assert (
        resolve_app_config({**BASE, "costs": {}, "cost_preset": "residential_de", "currency": "EUR"}).currency == "EUR"
    )


def test_optimizer_currency_is_checked_like_the_apps(monkeypatch):
    _, _, config = _optimizer_case(monkeypatch)
    assert resolve_optimization_config(config)["currency"] == "EUR"
    usd = {**_scaled_optimizer_config(config, 2.0, "USD")}
    resolved = resolve_optimization_config(usd)
    assert resolved["currency"] == "USD"
    assert resolve_optimization_config(resolved) == resolved

    no_budget = deepcopy(usd)
    del no_budget["constraints"]["budget"]
    with pytest.raises(ValueError, match=r"constraints\.budget would default to 10000 EUR"):
        resolve_optimization_config(no_budget)
    with pytest.raises(ValueError, match=r"'tariff\.currency' is EUR, but 'currency' is USD"):
        resolve_optimization_config({**usd, "tariff": config["tariff"]})
    with pytest.raises(ValueError, match=r"The run is in USD, but module_cost_per_w"):
        resolve_optimization_config({**usd, "costs": {}})
    # Without a battery in the search, the battery prices are not needed;
    # flat prices given under [financials] count.
    pv_only = {
        **usd,
        "tariff": None,
        "reference_tariff": None,
        "costs": {key: 1.0 for key in default_amount_cost_keys(tariff=True, battery=False)} | {"daily_power_cost": 0.3},
        "financials": {"electricity_cost": 0.3, "electricity_sold_cost": 0.05},
        "constraints": {**usd["constraints"], "max_battery_kwh": 0},
    }
    assert resolve_optimization_config(pv_only)["currency"] == "USD"


# --- Outputs ------------------------------------------------------------------


def test_cost_presets_list_their_own_currency(capsys):
    assert cli.main(["list", "cost-presets", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert {row["currency"] for row in rows} == {"EUR"}


def test_sweep_and_monte_carlo_csvs_name_the_currency(tmp_path, write_multiyear_weather):
    config = tmp_path / "sweep.toml"
    costs = "\n".join(f"{key} = {value * K}" for key, value in EUR_COSTS.items())
    config.write_text(
        'location = "porto"\nn_modules = 8\nannual_consumption_kwh = 4000\nbattery_kwh = 0\nprojection_years = 2\n'
        f'currency = "JPY"\n\n[costs]\n{costs}\n\n[sweep]\nn_modules = [6, 8]\n',
        encoding="utf-8",
    )
    output = tmp_path / "sweep.csv"
    weather_patch, load_patch = _offline()
    with weather_patch, load_patch:
        assert cli.main(["sweep", "--config", str(config), "--output", str(output)]) == 0
    with output.open(encoding="utf-8") as handle:
        assert {row["currency"] for row in csv.DictReader(handle)} == {"JPY"}

    weather = write_multiyear_weather(tmp_path / "multi.csv")
    mc_config = tmp_path / "mc.toml"
    mc_config.write_text(
        config.read_text(encoding="utf-8").split("[sweep]")[0]
        + f'[montecarlo]\nweather_file = "{weather}"\nn_runs = 2\nyears_per_run = 2\n',
        encoding="utf-8",
    )
    runs = tmp_path / "mc.csv"
    assert cli.main(["montecarlo", "--config", str(mc_config), "--output", str(runs)]) == 0
    assert set(pd.read_csv(runs)["currency"]) == {"JPY"}
    assert json.loads((tmp_path / "mc.provenance.json").read_text(encoding="utf-8"))["currency"] == "JPY"


def test_a_written_projection_keeps_the_currency_it_records(tmp_path):
    columns = {"Year": [1, 2], "Cost_Import": [12800.0, 13056.0]}
    projection = pd.DataFrame(columns)
    projection.attrs["currency"] = "JPY"
    first = pd.read_csv(write_cost_projection(projection, str(tmp_path), "first"))
    assert first.attrs == {} and set(first["currency"]) == {"JPY"}
    # Read back, the column is the only record, and writing again keeps it.
    second = pd.read_csv(write_cost_projection(first, str(tmp_path), "second"))
    pd.testing.assert_frame_equal(second, first)
    # A frame that records no currency is not labelled with the default.
    unlabelled = pd.read_csv(write_cost_projection(pd.DataFrame(columns), str(tmp_path), "none"))
    assert "currency" not in unlabelled.columns


def test_plots_read_the_currency_column_of_a_breos_csv(tmp_path):
    from breos import plotting

    frame = pd.DataFrame({"npv_savings": [1.0, 2.0], "currency": ["JPY", "JPY"]})
    assert plotting._label_currency([frame], None) == "JPY"
    with pytest.raises(ValueError, match="records its currency as 'JPY'"):
        plotting._label_currency([frame], "EUR")
    mixed = frame.assign(currency=["JPY", "USD"])
    with pytest.raises(ValueError, match="several currencies"):
        plotting._label_currency([mixed], None)
