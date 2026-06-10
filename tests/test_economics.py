"""Tests for the economics module."""

import pytest

from breos.economics import CostParams, calculate_costs, calculate_lcoe, cost_params_from_config, find_payback_year
from breos.optimization import calculate_financials


class TestCostDefaultsSingleSource:
    def test_cost_params_from_config_empty_matches_dataclass_defaults(self):
        # Missing config keys must fall back to the CostParams defaults, so
        # config-driven and direct-construction paths cannot diverge.
        assert cost_params_from_config({}, {}) == CostParams()

    def test_resolve_costs_preset_fallbacks_match_dataclass_defaults(self, monkeypatch):
        from breos import app_config

        monkeypatch.setattr(app_config, "load_json", lambda name: {"minimal": {}})
        cfg = {
            "cost_preset": "minimal",
            "inverter_loading_ratio": 1.25,
            "inflation_rate": 0.02,
            "discount_rate": 0.0,
            "pv_degradation_rate": 0.005,
        }

        params = app_config.resolve_costs(cfg)

        # A preset that omits every key behaves exactly like no preset
        assert params == CostParams(
            dc_ac_ratio=1.25,
            inflation_rate=0.02,
            discount_rate=0.0,
            pv_degradation_rate=0.005,
        )


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

    def test_optimizer_financials_honor_modern_cost_keys(self):
        base = dict(
            n_modules=10,
            battery_kwh=5.0,
            annual_import_kwh=2000.0,
            annual_export_kwh=1000.0,
            annual_load_kwh=4000.0,
            costs_config={
                "module_cost_per_w": 0.10,
                "storage_cost_per_kwh": 400.0,
                "installation_cost_per_module": 200.0,
                "installation_cost_battery": 500.0,
                "other_costs": 100.0,
            },
            financials_config={
                "electricity_cost": 0.30,
                "electricity_sold_cost": 0.05,
                "inflation_rate": 0.01,
                "discount_rate": 0.0,
                "project_lifespan": 5,
            },
        )

        capex_a, npv_a = calculate_financials(**base)

        modified = dict(base)
        modified["costs_config"] = dict(base["costs_config"], module_cost_per_w=0.30)
        modified["financials_config"] = dict(base["financials_config"], electricity_cost=0.45)
        capex_b, npv_b = calculate_financials(**modified)

        assert capex_b > capex_a
        assert npv_b != npv_a


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
