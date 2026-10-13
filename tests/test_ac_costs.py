"""Pricing an AC-coupled battery's inverter (ADR 0004 C9)."""

import math

import pytest

from breos.economics import CostParams, calculate_costs

SYSTEM = dict(n_modules=10, module_power_w=400.0, inverter_ac_capacity_w=3200.0)
PRICES = CostParams(battery_inverter_cost_per_kw=120.0)


def test_dc_coupled_pricing_is_unchanged():
    costs = calculate_costs(battery_capacity_wh=10000.0, cost_params=PRICES, **SYSTEM)
    assert costs["inverter_cost"] == PRICES.inverter_cost_per_kw * 3.2
    assert costs["battery_inverter_cost"] == 0.0
    assert costs["total_initial_cost"] == (
        costs["pv_cost"]
        + costs["inverter_cost"]
        + costs["battery_cost"]
        + costs["installation_cost"]
        + PRICES.land_cost
        + costs["other_costs"]
    )


def test_ac_coupling_prices_a_simple_pv_inverter_and_the_battery_inverter():
    dc = calculate_costs(battery_capacity_wh=10000.0, cost_params=PRICES, **SYSTEM)
    ac = calculate_costs(
        battery_capacity_wh=10000.0,
        cost_params=PRICES,
        battery_coupling="ac",
        battery_inverter_ac_capacity_w=5000.0,
        **SYSTEM,
    )
    assert ac["inverter_cost"] == PRICES.inverter_cost_per_kw_nobatt * 3.2
    assert ac["battery_inverter_cost"] == 120.0 * 5.0
    assert ac["battery_cost"] == dc["battery_cost"]
    assert ac["installation_cost"] == dc["installation_cost"]
    assert ac["total_initial_cost"] == pytest.approx(
        dc["total_initial_cost"] - dc["inverter_cost"] + ac["inverter_cost"] + 600.0, rel=1e-15
    )
    assert ac["replacement_cost_each"] == dc["replacement_cost_each"]


def test_a_missing_battery_inverter_price_is_refused_never_zero():
    with pytest.raises(ValueError, match="needs battery_inverter_cost_per_kw"):
        calculate_costs(
            battery_capacity_wh=10000.0, battery_coupling="ac", battery_inverter_ac_capacity_w=5000.0, **SYSTEM
        )


def test_an_explicit_zero_price_is_a_bundled_inverter():
    costs = calculate_costs(
        battery_capacity_wh=10000.0,
        cost_params=CostParams(battery_inverter_cost_per_kw=0.0),
        battery_coupling="ac",
        battery_inverter_ac_capacity_w=5000.0,
        **SYSTEM,
    )
    assert costs["battery_inverter_cost"] == 0.0


def test_an_ac_coupled_battery_needs_its_rating_to_be_priced():
    with pytest.raises(ValueError, match="needs battery_inverter_ac_capacity_w"):
        calculate_costs(battery_capacity_wh=10000.0, cost_params=PRICES, battery_coupling="ac", **SYSTEM)


def test_ac_coupling_without_a_battery_prices_no_battery_inverter():
    # A capacity sweep from 0 keeps one config: without a battery the price
    # and rating are not needed and price nothing.
    pv_only = calculate_costs(**SYSTEM)
    for params in (CostParams(), PRICES):
        costs = calculate_costs(
            cost_params=params, battery_coupling="ac", battery_inverter_ac_capacity_w=5000.0, **SYSTEM
        )
        assert costs["battery_inverter_cost"] == 0.0
        assert costs["total_initial_cost"] == pv_only["total_initial_cost"]


def test_dc_coupling_refuses_a_battery_inverter_rating():
    with pytest.raises(ValueError, match="this one is DC-coupled"):
        calculate_costs(battery_capacity_wh=10000.0, battery_inverter_ac_capacity_w=5000.0, **SYSTEM)


@pytest.mark.parametrize(
    ("price", "rating"), [(-1.0, 5000.0), (math.nan, 5000.0), (math.inf, 5000.0), (True, 5000.0), (120.0, -1.0)]
)
def test_battery_inverter_price_and_rating_must_be_finite_and_non_negative(price, rating):
    with pytest.raises(ValueError, match="must be a finite non-negative number"):
        calculate_costs(
            battery_capacity_wh=10000.0,
            cost_params=CostParams(battery_inverter_cost_per_kw=price),
            battery_coupling="ac",
            battery_inverter_ac_capacity_w=rating,
            **SYSTEM,
        )


def test_unknown_coupling_is_refused():
    with pytest.raises(ValueError, match="battery_coupling must be 'dc' or 'ac'"):
        calculate_costs(battery_coupling="hybrid", **SYSTEM)
