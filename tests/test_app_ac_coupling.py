"""AC-coupled batteries through the App (ADR 0004 C2, C8, C9, C14)."""

from unittest import mock

import pytest

from breos import App
from breos.app_config import build_costs_dict, resolve_app_config
from breos.economics import default_amount_cost_keys, replacement_event_cost
from breos.projection import build_battery_config
from tools.generate_app_golden import _fake_fetch

BASE = {
    "location": "porto",
    "n_modules": 8,
    "annual_consumption_kwh": 4000,
    "start_date": "2025-01-01",
    "projection_years": 2,
    "cost_preset": "residential_pt",
    "emissions_country": "PT",
    "execution_backend": "python",
}
AC = {
    "battery_coupling": "ac",
    "battery_inverter_kw_per_kwh": 0.6,
    "battery_inverter_efficiency": 0.95,
    "costs": {"battery_inverter_cost_per_kw": 120.0},
}


def _resolve(**overrides):
    return resolve_app_config({**BASE, **overrides})


def _result(config):
    with (
        mock.patch("breos.app.fetch_tmy_weather_data", _fake_fetch),
        mock.patch("breos.app.load_weather", lambda **_kwargs: None),
    ):
        app = App({**BASE, **config})
        app.simulate()
        return app.result()


# --- configuration -------------------------------------------------------------


def test_dc_is_the_default_and_resolves_no_battery_inverter():
    resolved = _resolve(battery_kwh=5)
    assert resolved.cfg["battery_coupling"] == "dc"
    assert resolved.battery_inverter_ac_capacity_w is None
    battery = build_battery_config(resolved.cfg, resolved, initial_soh=100.0)
    assert battery.coupling == "dc"
    assert battery.battery_inverter_ac_capacity_w is None


@pytest.mark.parametrize(
    "extra",
    [
        {"battery_inverter_ac_rating_kw": 3.0},
        {"battery_inverter_kw_per_kwh": 0.5},
        {"battery_inverter_efficiency": 0.96},
        {"costs": {"battery_inverter_cost_per_kw": 120.0}},
    ],
)
def test_dc_refuses_every_battery_inverter_setting(extra):
    with pytest.raises(ValueError, match="set battery_coupling = 'ac' or remove them"):
        _resolve(battery_kwh=5, **extra)


def test_the_rating_resolves_against_the_nominal_capacity():
    assert _resolve(battery_kwh=5, **AC).battery_inverter_ac_capacity_w == pytest.approx(3000.0, rel=1e-15)
    absolute = {**AC, "battery_inverter_kw_per_kwh": None, "battery_inverter_ac_rating_kw": 2.5}
    assert _resolve(battery_kwh=5, **absolute).battery_inverter_ac_capacity_w == 2500.0


def test_ac_builds_the_battery_inverter_into_the_battery_config():
    resolved = _resolve(battery_kwh=5, **AC)
    battery = build_battery_config(resolved.cfg, resolved, initial_soh=100.0)
    assert battery.coupling == "ac"
    assert battery.battery_inverter_ac_capacity_w == resolved.battery_inverter_ac_capacity_w
    assert battery.battery_inverter_efficiency == 0.95


@pytest.mark.parametrize(
    ("drop", "message"),
    [
        ("battery_inverter_kw_per_kwh", "needs its inverter's rating"),
        ("battery_inverter_efficiency", "needs 'battery_inverter_efficiency'"),
        ("costs", r"needs '\[costs\] battery_inverter_cost_per_kw'"),
    ],
)
def test_an_ac_coupled_battery_needs_its_rating_efficiency_and_price(drop, message):
    config = {key: value for key, value in AC.items() if key != drop}
    with pytest.raises(ValueError, match=message):
        _resolve(battery_kwh=5, **config)


def test_the_two_rating_keys_are_exclusive():
    with pytest.raises(ValueError, match="battery_inverter_ac_rating_kw"):
        _resolve(battery_kwh=5, **AC, battery_inverter_ac_rating_kw=3.0)


@pytest.mark.parametrize(
    "extra",
    [
        {"battery_inverter_kw_per_kwh": 0.0},
        {"battery_inverter_kw_per_kwh": float("nan")},
        {"battery_inverter_efficiency": 0.0},
        {"battery_inverter_efficiency": 1.01},
        {"battery_inverter_efficiency": True},
        {"battery_coupling": "hybrid"},
    ],
)
def test_ac_settings_out_of_range_are_refused(extra):
    with pytest.raises((TypeError, ValueError)):
        _resolve(battery_kwh=5, **{**AC, **extra})


def test_an_ac_pv_only_config_needs_nothing_and_prices_no_battery_inverter():
    # A capacity sweep from 0 keeps one config (C2).
    for config in ({"battery_coupling": "ac"}, AC):
        resolved = _resolve(battery_kwh=0, **config)
        assert resolved.battery_inverter_ac_capacity_w is None
        assert build_costs_dict(resolved.cfg, resolved)["battery_inverter_cost"] == 0.0


# --- costs -----------------------------------------------------------------------


def test_ac_prices_a_simple_pv_inverter_and_the_battery_inverter():
    dc = _resolve(battery_kwh=5)
    ac = _resolve(battery_kwh=5, **AC)
    dc_costs, ac_costs = build_costs_dict(dc.cfg, dc), build_costs_dict(ac.cfg, ac)
    params = ac.cost_params
    assert ac_costs["inverter_cost"] == params.inverter_cost_per_kw_nobatt * ac.inverter_ac_capacity_w / 1000
    assert ac_costs["battery_inverter_cost"] == pytest.approx(120.0 * 3.0, rel=1e-15)
    assert ac_costs["battery_cost"] == dc_costs["battery_cost"]
    assert ac_costs["replacement_cost_each"] == dc_costs["replacement_cost_each"]


@pytest.mark.parametrize("coupling", [{}, AC], ids=["dc", "ac"])
def test_battery_replacement_cost_prices_one_replacement(coupling):
    resolved = _resolve(battery_kwh=5, **coupling)
    pack = replacement_event_cost(5, resolved.cost_params.battery_cost_per_kwh)
    assert build_costs_dict(resolved.cfg, resolved)["replacement_cost_each"] == pack
    priced = _resolve(battery_kwh=5, battery_replacement_cost=4321.0, **coupling)
    assert build_costs_dict(priced.cfg, priced)["replacement_cost_each"] == 4321.0


@pytest.mark.parametrize("value", [-1.0, float("inf"), True, "calculate"])
def test_battery_replacement_cost_must_be_a_finite_non_negative_number(value):
    with pytest.raises((TypeError, ValueError), match="battery_replacement_cost"):
        _resolve(battery_kwh=5, battery_replacement_cost=value)


# --- results -----------------------------------------------------------------


def _without_coupling(result):
    """``result`` without the fields that name the coupling, which P2 lets differ."""
    resolved = {
        key: value
        for key, value in result["provenance"]["resolved_config"].items()
        if key not in ("battery_coupling", "costs")
    }
    return {**result, "provenance": {**result["provenance"], "resolved_config": resolved}}


def test_p2_an_ac_pv_only_app_run_is_the_dc_pv_only_run():
    dc = _result({"battery_kwh": 0})
    ac = _result({"battery_kwh": 0, **AC})
    for key in ("npv_savings", "pv_loss_waterfall", "yearly", "financial", "monthly"):
        assert _without_coupling(ac)[key] == _without_coupling(dc)[key], key
    assert ac["provenance"]["resolved_config"]["battery_coupling"] == "ac"


def test_an_ac_battery_reports_its_inverter_apart_from_the_pv_inverter():
    dc = _result({"battery_kwh": 5})
    ac = _result({"battery_kwh": 5, **AC})
    assert "battery_inverter" not in dc["pv_loss_waterfall"]
    assert "pv_to_battery_kwh" not in dc["pv_loss_waterfall"]["energy_balance"]["ac_delivery"]

    waterfall = ac["pv_loss_waterfall"]
    block = waterfall["battery_inverter"]
    assert block["ac_capacity_kw"] == 3.0
    assert block["efficiency_pct"] == 95.0
    assert block["charge_conversion_loss_kwh"] > 0.0
    assert block["discharge_conversion_loss_kwh"] > 0.0
    assert block["conversion_loss_kwh"] == pytest.approx(
        block["charge_conversion_loss_kwh"] + block["discharge_conversion_loss_kwh"], abs=0.011
    )
    # The inverter block is the PV inverter's alone.
    inverter = waterfall["inverter"]
    assert inverter["conversion_loss_kwh"] == inverter["direct_pv_conversion_loss_kwh"]
    assert inverter["battery_discharge_conversion_loss_kwh"] == 0.0
    # No PV DC reaches an AC-coupled battery; its PV arrives as AC.
    balance = waterfall["energy_balance"]
    assert balance["pv_dc"]["to_battery_kwh"] == 0.0
    assert balance["ac_delivery"]["pv_to_battery_kwh"] > 0.0
    assert ac["provenance"]["resolved_config"]["battery_inverter_kw_per_kwh"] == 0.6


# --- currency and revaluation ------------------------------------------------


def test_an_ac_run_needs_the_simple_pv_inverter_price_never_the_hybrid_one():
    dc = default_amount_cost_keys(tariff=False, battery=True)
    ac = default_amount_cost_keys(tariff=False, battery=True, coupling="ac")
    assert "inverter_cost_per_kw_hybrid" in dc and "inverter_cost_per_kw_simple" not in dc
    assert "inverter_cost_per_kw_simple" in ac and "inverter_cost_per_kw_hybrid" not in ac
    either = default_amount_cost_keys(tariff=False, battery=None, coupling="ac")
    assert "inverter_cost_per_kw_hybrid" not in either
    assert default_amount_cost_keys(tariff=False, battery=False, coupling="ac") == default_amount_cost_keys(
        tariff=False, battery=False
    )


def _usd(keys):
    return {key: 1.0 for key in keys}


def test_a_non_eur_ac_run_refuses_a_eur_default_pv_inverter_price():
    needed = default_amount_cost_keys(tariff=False, battery=True, coupling="ac")
    base = {**BASE, "cost_preset": None, "currency": "USD", "battery_kwh": 5, **AC}
    costs = {**_usd(needed), "battery_inverter_cost_per_kw": 120.0}
    resolve_app_config({**base, "costs": costs})
    missing = {key: value for key, value in costs.items() if key != "inverter_cost_per_kw_simple"}
    with pytest.raises(ValueError, match="inverter_cost_per_kw_simple would come from the built-in EUR defaults"):
        resolve_app_config({**base, "costs": missing})


def test_revalue_changes_the_replacement_price_and_refuses_it_across_currencies():
    with (
        mock.patch("breos.app.fetch_tmy_weather_data", _fake_fetch),
        mock.patch("breos.app.load_weather", lambda **_kwargs: None),
    ):
        app = App({**BASE, "battery_kwh": 5, "battery_eol_percentage": 0.97, "battery_replacement_cost": 4321.0})
        app.simulate()
        result = app.result()
        assert result["battery_replacements"] > 0
        revalued = app.revalue({"battery_replacement_cost": 1000.0})
        assert revalued["battery_replacements"] == result["battery_replacements"]
        assert revalued["battery_replacement_cost_npv"] == pytest.approx(
            result["battery_replacement_cost_npv"] * 1000.0 / 4321.0, abs=0.02
        )
        usd = {
            "currency": "USD",
            "cost_preset": None,
            "costs": _usd(default_amount_cost_keys(tariff=False, battery=True)),
        }
        with pytest.raises(ValueError, match="battery_replacement_cost is an amount in EUR"):
            app.revalue(usd)
        restated = app.revalue({**usd, "battery_replacement_cost": 5000.0})
        assert restated["battery_replacements"] == result["battery_replacements"]


def test_the_ac_cable_carries_the_pv_inverter_output_alone():
    # Beside an AC-coupled battery the PV inverter and its cable convert as
    # in a PV-only run (ADR 0004 C3), so the cable loss is the PV-only one.
    cable = {"ac_cable": {"length_m": 30, "cross_section_mm2": 2.5, "phases": 1, "voltage_v": 230}}
    pv_only = _result({"battery_kwh": 0, **cable})["pv_loss_waterfall"]["inverter"]["ac_cable"]
    ac = _result({"battery_kwh": 5, **AC, **cable})["pv_loss_waterfall"]
    assert ac["energy_balance"]["ac_delivery"]["pv_to_battery_kwh"] > 0.0
    assert ac["inverter"]["ac_cable"]["loss_kwh"] == pv_only["loss_kwh"] > 0.0
