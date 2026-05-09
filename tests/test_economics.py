"""Tests for the economics module."""

import pytest

from breos.economics import CostParams, calculate_costs, calculate_lcoe, find_payback_year


class TestCalculateCosts:
    def test_pv_only(self, cost_params):
        costs = calculate_costs(
            n_modules=10,
            module_power_w=550,
            battery_capacity_wh=0,
            cost_params=cost_params,
        )
        assert costs["battery_cost"] == 0.0
        assert costs["pv_cost"] == pytest.approx(10 * 550 * 0.125)
        assert costs["total_initial_cost"] > 0
        # No battery → simple inverter
        assert costs["inverter_cost"] == pytest.approx(48.37 * (10 * 550 / 1000) / 1.25, rel=0.01)

    def test_with_battery(self, cost_params):
        costs = calculate_costs(
            n_modules=10,
            module_power_w=550,
            battery_capacity_wh=5000,
            cost_params=cost_params,
        )
        assert costs["battery_cost"] == pytest.approx(5 * 500.0)
        # With battery → hybrid inverter (more expensive)
        assert costs["inverter_cost"] == pytest.approx(102.58 * (10 * 550 / 1000) / 1.25, rel=0.01)
        assert costs["total_initial_cost"] > costs["pv_cost"] + costs["battery_cost"]

    def test_cost_breakdown_sums_to_total(self, cost_params):
        costs = calculate_costs(
            n_modules=6,
            module_power_w=550,
            battery_capacity_wh=5000,
            cost_params=cost_params,
        )
        parts = (
            costs["pv_cost"]
            + costs["inverter_cost"]
            + costs["battery_cost"]
            + costs["installation_cost"]
            + costs["other_costs"]
            + costs.get("tes_cost", 0)
            + costs.get("hp_cost", 0)
            + costs.get("tes_installation_cost", 0)
        )
        assert costs["total_initial_cost"] == pytest.approx(parts, rel=0.001)


class TestFindPaybackYear:
    def test_known_payback(self):
        import pandas as pd

        # Savings turn positive at year 8
        proj = pd.DataFrame(
            {
                "Year": range(1, 11),
                "Savings_Cumulative_NPV": [-500, -400, -300, -200, -100, -50, -10, 30, 100, 200],
            }
        )
        assert find_payback_year(proj) == 8

    def test_no_payback(self):
        import pandas as pd

        proj = pd.DataFrame(
            {
                "Year": range(1, 6),
                "Savings_Cumulative_NPV": [-500, -400, -300, -200, -100],
            }
        )
        assert find_payback_year(proj) is None


class TestLCOE:
    def test_basic(self):
        lcoe = calculate_lcoe(
            total_investment=5000,
            annual_production_kwh=5000,
            annual_operation_cost=50,
            lifetime_years=20,
            discount_rate=0.0,
            degradation_rate=0.0,
        )
        # (5000 + 50*20) / (5000*20) = 6000/100000 = 0.06
        assert lcoe == pytest.approx(0.06, rel=0.01)

    def test_degradation_increases_lcoe(self):
        lcoe_no_deg = calculate_lcoe(
            total_investment=5000,
            annual_production_kwh=5000,
            annual_operation_cost=50,
            lifetime_years=20,
            discount_rate=0.0,
            degradation_rate=0.0,
        )
        lcoe_with_deg = calculate_lcoe(
            total_investment=5000,
            annual_production_kwh=5000,
            annual_operation_cost=50,
            lifetime_years=20,
            discount_rate=0.0,
            degradation_rate=0.01,
        )
        assert lcoe_with_deg > lcoe_no_deg

    def test_zero_production(self):
        lcoe = calculate_lcoe(
            total_investment=5000,
            annual_production_kwh=0,
            annual_operation_cost=50,
            lifetime_years=20,
        )
        assert lcoe == float("inf")
